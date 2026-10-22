// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The C++ embedding reference demo.
//
// One driver-loaded generated "pure" plugin SET (three artifacts, one manifest),
// launched through the SAME independent `eagle::cuda::PluginRegistry` the other
// three plugin/ demos use (plugin/plugin_host.cpp, plugin/graph_inject/inject_demo.cu,
// plugin/pure_inject/pure_inject_demo.cu -- see README.rst for how this relates to
// them and why it lives in a separate directory rather than a fourth copy under
// plugin/). It proves the two capabilities this example exists to close:
//
//   1. a matrix (`mat_in`) input, bound through the device registry's new
//      `bind_matrix` (plugin/plugin_registry/registry.h) -- fixtures/mattrace.cu,
//      `out[i] = trace(M[i])` for a batch of 3x3 matrices;
//   2. a derivative (VJP) artifact -- fixtures/toy_energy.cu (the primal,
//      `e = p * exp(-0.5*|x|^2)`) and fixtures/toy_energy_vjp.cu (its custom VJP,
//      carrying the optional `derivative` sidecar block, Phase A recompute-only),
//      checked against a host CENTRAL-DIFFERENCE reference of the primal (an
//      reference independent of the analytic formula the VJP kernel itself uses) --
//      the same shape of proof as the code generator's downstream-toy-extension deploy test
//      reimplemented by
//      hand here so this example never imports it.
//
// All three plugins are captured as nodes in ONE CUDA graph opening (Runtime-API
// capture, Driver-API launch -- the same idiom pure_inject_demo.cu uses) and
// replayed twice, proving the matrix + derivative artifacts both survive replay
// like any other pure kernel (each is idempotent -- a fresh recompute every
// launch, no RMW state -- so both replays must agree with the same golden).
//
// Usage: embedding_demo [artifact_dir] [N]
//   artifact_dir defaults to the fixtures this target compiles to PTX at build
//   time (baked in via EMBED_DEMO_ART_DIR); N defaults to 4096.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda.h>           // Driver API -- loads + launches the JIT plugins
#include <cuda_runtime.h>  // Runtime API -- capture, graph

#include "demo_common.h"
#include "plugin/plugin_registry/registry.h"

#ifndef EMBED_DEMO_ART_DIR
#define EMBED_DEMO_ART_DIR "art"   // fallback if built outside this target's CMakeLists.txt
#endif

namespace {

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

}  // namespace

int run(int argc, char** argv) {
    const std::string dir = (argc >= 2) ? (std::string(argv[1]) + "/")
                                         : (std::string(EMBED_DEMO_ART_DIR) + "/");
    const std::uint32_t N = (argc >= 3)
        ? std::uint32_t(std::strtoul(argv[2], nullptr, 10)) : 4096;

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

    // --- load the 3-plugin pure set (mattrace, toy_energy, toy_energy_vjp) ------
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(dir + "manifest.json");

    // --- host inputs (shared with the CPU demo) -----------------------------
    const demo::Inputs in(N);
    const std::vector<double>& M = in.M;
    const std::vector<double>& x = in.x;
    const std::vector<double>& e_bar = in.e_bar;
    const double p = in.p;

    // --- device buffers (all PRE-ALLOCATED; the registry only references them) --
    double *d_M, *d_out, *d_x, *d_e, *d_e_bar, *d_x_bar, *d_p_bar;
    RT_CHECK(cudaMalloc(&d_M, M.size() * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_out, N * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_x, x.size() * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_e, N * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_e_bar, N * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_x_bar, x.size() * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_p_bar, N * sizeof(double)));
    RT_CHECK(cudaMemcpy(d_M, M.data(), M.size() * sizeof(double), cudaMemcpyHostToDevice));
    RT_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    RT_CHECK(cudaMemcpy(d_e_bar, e_bar.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // --- bind by name (gap 1: bind_matrix is the new device entry point) -------
    registry.bind_matrix("M", d_M, N, 3, 3);
    registry.bind_handle("out", d_out);
    registry.bind_vector("x", d_x, N);
    registry.bind_handle("e", d_e);
    registry.bind_uniform("p", p);
    registry.bind_handle("e_bar", d_e_bar);
    registry.bind_vector("x_bar", d_x_bar, N);
    registry.bind_handle("p_bar", d_p_bar);

    const int block = 256;

    // === capture all 3 pure nodes in one opening, replay twice ==================
    cudaStream_t stream; RT_CHECK(cudaStreamCreate(&stream));
    RT_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const int injected = registry.inject(reinterpret_cast<CUstream>(stream), int(N), block);
    cudaGraph_t graph; RT_CHECK(cudaStreamEndCapture(stream, &graph));

    std::size_t n_nodes = 0;
    RT_CHECK(cudaGraphGetNodes(graph, nullptr, &n_nodes));

    cudaGraphExec_t exec; RT_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    RT_CHECK(cudaGraphLaunch(exec, stream));
    RT_CHECK(cudaGraphLaunch(exec, stream));   // replay: every plugin here is idempotent
    RT_CHECK(cudaStreamSynchronize(stream));

    demo::Outputs o(N);
    RT_CHECK(cudaMemcpy(o.out.data(), d_out, N * sizeof(double), cudaMemcpyDeviceToHost));
    RT_CHECK(cudaMemcpy(o.e.data(), d_e, N * sizeof(double), cudaMemcpyDeviceToHost));
    RT_CHECK(cudaMemcpy(o.x_bar.data(), d_x_bar, o.x_bar.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
    RT_CHECK(cudaMemcpy(o.p_bar.data(), d_p_bar, N * sizeof(double), cudaMemcpyDeviceToHost));

    std::printf("N_PLUGINS=%zu\n", registry.size());
    std::printf("ACTIVE=%d\n", registry.active());
    std::printf("INJECTED=%d\n", injected);
    std::printf("TOTAL_NODES=%zu\n", n_nodes);
    const bool checks_ok = demo::check(in, o);

    const bool ok = (injected == 3) && (n_nodes == 3) && checks_ok;
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
