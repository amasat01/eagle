// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Pure-kernel injection demo — deployability is an inherited property.
//
// The pure analogue of graph_inject/inject_demo.cu: a runtime-loaded *pure*
// plugin set is injected into a CUDA graph via the SAME independent PluginRegistry.
// A pure kernel has no outVec accumulator; its output is its writable `Mutable`
// buffer(s), read-modify-written IN PLACE. So unlike a force (reset-then-accumulate,
// idempotent on replay), a pure kernel carries state across replays — the host seeds
// the Mutable once BEFORE capture and never resets it inside the graph. This proves the
// registry binds a `mutable` role by name and injects it exactly like a force node.
//
// The kernel here is `advance(step, age: Mutable[float])` -> age = age + step. The
// captured graph is a single pure node; replaying it R times gives age += R*step.
//
// Usage:  pure_inject <artifact_dir> <N> [R]
//   <artifact_dir> holds manifest.json + <id>.ptx/<id>.json, plus age.bin (N f64, the
//   initial Mutable state), uniforms.bin (the scalar step), and golden.bin (N f64, the
//   expected age after R replays). R defaults to 2.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>            // Driver API — loads + launches the JIT plugins
#include <cuda_runtime.h>   // Runtime API — capture, graph

#include "../plugin_registry/registry.h"

namespace {

constexpr double TOL = 1e-12;

void cu_check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* m = nullptr; cuGetErrorString(r, &m);
        std::fprintf(stderr, "Driver error %d (%s) at %s\n", int(r), m ? m : "?", what);
        std::exit(2);
    }
}
void rt_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "Runtime error %d (%s) at %s\n", int(e),
                     cudaGetErrorString(e), what);
        std::exit(2);
    }
}
#define CU_CHECK(c) cu_check((c), #c)
#define RT_CHECK(c) rt_check((c), #c)

std::vector<double> read_doubles(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "missing %s\n", p.c_str()); std::exit(2); }
    std::vector<double> v(n);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(double)));
    return v;
}
double max_abs(const std::vector<double>& g, const std::vector<double>& r) {
    double w = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i)
        w = std::max(w, std::abs(g[i] - r[i]));
    return w;
}

}  // namespace

int run(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <dir> <N> [R]\n", argv[0]);
        return 2;
    }
    const std::string dir = std::string(argv[1]) + "/";
    const int N = std::atoi(argv[2]);
    const int R = (argc >= 4) ? std::atoi(argv[3]) : 2;

    // --- init: Runtime creates the primary context; Driver API shares it -------
    RT_CHECK(cudaSetDevice(0));
    RT_CHECK(cudaFree(0));
    CU_CHECK(cuInit(0));
    CUcontext ctx = nullptr; CU_CHECK(cuCtxGetCurrent(&ctx));
    if (ctx == nullptr) {
        CUdevice d; CU_CHECK(cuDeviceGet(&d, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, d));
        CU_CHECK(cuCtxSetCurrent(ctx));
    }

    // --- load the pure plugin set (manifest.json must carry `pattern: "pure"` or
    // omit it — from_manifest rejects an unrecognized value) ------------
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(dir + "manifest.json");

    // --- device buffers (all PRE-ALLOCATED; the registry only references them) --
    double* d_age; char* d_term;
    RT_CHECK(cudaMalloc(&d_age, N * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_term, N));
    RT_CHECK(cudaMemset(d_term, 0, N));

    // Seed the Mutable ONCE, before capture: a pure kernel reads its Mutable at entry
    // (RMW), so zeroing it inside the graph would destroy the input. This is the
    // pure counterpart to a force's reset-then-accumulate.
    const std::vector<double> age0 = read_doubles(dir + "age.bin", N);
    RT_CHECK(cudaMemcpy(d_age, age0.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    double step = 0.0;
    if (std::ifstream(dir + "uniforms.bin", std::ios::binary))
        step = read_doubles(dir + "uniforms.bin", 1)[0];

    // Bind the Mutable + mask + uniform by name; the registry references them.
    registry.bind_handle("age", d_age);          // scalar Mutable -> flat handle
    registry.bind_handle("terminated", d_term);
    registry.bind_uniform("step", step);

    const int block = 256;

    // === capture the pure plugin node, then replay it R times ==================
    cudaStream_t stream; RT_CHECK(cudaStreamCreate(&stream));
    RT_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const int injected = registry.inject(reinterpret_cast<CUstream>(stream), N, block);
    cudaGraph_t graph; RT_CHECK(cudaStreamEndCapture(stream, &graph));

    std::size_t n_nodes = 0;
    RT_CHECK(cudaGraphGetNodes(graph, nullptr, &n_nodes));

    cudaGraphExec_t exec; RT_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    for (int r = 0; r < R; ++r)
        RT_CHECK(cudaGraphLaunch(exec, stream));   // RMW in place: age += step each time
    RT_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> got(N);
    RT_CHECK(cudaMemcpy(got.data(), d_age, N * sizeof(double), cudaMemcpyDeviceToHost));
    const std::vector<double> golden = read_doubles(dir + "golden.bin", N);
    const double err = max_abs(got, golden);

    std::printf("N_PLUGINS=%zu\n", registry.size());
    std::printf("ACTIVE=%d\n", registry.active());
    std::printf("INJECTED=%d\n", injected);
    std::printf("REPLAYS=%d\n", R);
    std::printf("TOTAL_NODES=%zu\n", n_nodes);
    std::printf("MAX_ABS=%.3e\n", err);

    // one pure node per active plugin.
    const bool ok = (n_nodes == std::size_t(registry.active())) && (err < TOL);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
