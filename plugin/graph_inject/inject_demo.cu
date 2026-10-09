// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Multi-plugin mid-pipeline injection demo (a downstream-consumer seam, in miniature).
//
// A C++/CUDA host owns its OWN compiled-in `__global__` kernels and builds a CUDA
// graph from them, with a deliberate *opening in the middle* into which a whole
// SET of runtime-loaded generated JIT plugins is injected via the independent
// PluginRegistry. The captured graph is:
//
//   backend_pre -> [reset outVec] -> *** registry.inject(): N plugins *** -> backend_post
//   (compiled-in)     (memset)         (each PTX, loaded at run-time, += outVec)   (compiled-in)
//
// What it proves, generalizing the single-plugin case:
//   * MANY plugins (each its own CUDA module, all under the fixed `raptor_kernel`
//     symbol) inject as nodes between two compiled-in Runtime-API kernels;
//   * the injection is gated by a runtime flag + per-plugin enable — flag-off (or
//     all-disabled) yields the baseline graph with no plugin nodes;
//   * the registry binds only host-provided PRE-ALLOCATED buffers by name and
//     never owns the capture -- exactly the drop-in shape a downstream consumer needs.
//
// Usage:  inject_demo <artifact_dir> <N> [off]
//   <artifact_dir> holds manifest.json + each plugin's <id>.ptx/<id>.json, plus
//   the input bins (SoA, f64): position.bin (3xN, required), velocity.bin (3xN),
//   mass.bin / area.bin / cd.bin (N), uniforms.bin (the scalar mu), and
//   golden.bin (3xN, the expected post-pipeline outVec).
//   "off" flips the registry's global flag, exercising the no-op opening.
//   A lookup-table plugin reads its data from <name>.bin (consolidated pre-capture);
//   "noconsolidate" skips that upload to exercise the clear "not supplied" error.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>            // Driver API — loads + launches the JIT plugins
#include <cuda_runtime.h>   // Runtime API — backend kernels, capture, graph

#include "../plugin_registry/registry.h"

namespace {

constexpr double TOL = 1e-12;
// Backend transform constants (the Python golden uses the same values).
constexpr double SCALE = 2.5;   // backend_pre: scaled = SCALE * input
constexpr double POST  = 0.5;   // backend_post: outVec *= POST

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

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return bool(f);
}
std::vector<double> read_doubles(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "missing %s\n", p.c_str()); std::exit(2); }
    std::vector<double> v(n);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(double)));
    return v;
}
double max_rel(const std::vector<double>& g, const std::vector<double>& r, std::size_t n) {
    double w = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double dn = 0, rn = 0;
        for (int d = 0; d < 3; ++d) {
            double e = g[d * n + i] - r[d * n + i], rr = r[d * n + i];
            dn += e * e; rn += rr * rr;
        }
        double den = std::sqrt(rn);
        double rel = den > 0.0 ? std::sqrt(dn) / den : std::sqrt(dn);
        if (rel > w) w = rel;
    }
    return w;
}

}  // namespace

// --- compiled-in C++/CUDA backend kernels (stand in for a downstream consumer's own) ----
// Plain SoA (3, N) indexing: component d of sample i at arr[d*N + i]. A single
// pre-kernel scales BOTH the position and velocity inputs (one graph node), so
// every plugin (position- or velocity-based) reads from a pre-scaled buffer.
__global__ void backend_pre(const double* pos, const double* vel, double scale,
                            double* spos, double* svel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int d = 0; d < 3; ++d) {
        spos[d * n + i] = scale * pos[d * n + i];
        svel[d * n + i] = scale * vel[d * n + i];
    }
}
__global__ void backend_post(double* outVec, double factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int d = 0; d < 3; ++d) outVec[d * n + i] *= factor;
}

int run(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <dir> <N> [off|noconsolidate]\n", argv[0]);
        return 2;
    }
    const std::string dir = std::string(argv[1]) + "/";
    const int N = std::atoi(argv[2]);
    const std::string mode = (argc >= 4) ? std::string(argv[3]) : std::string();
    const bool flag_off = (mode == "off");
    const bool no_consolidate = (mode == "noconsolidate");  // skip table upload
    const std::size_t vlen = std::size_t(N) * 3;

    // --- init: Runtime creates the primary context; Driver API shares it -------
    RT_CHECK(cudaSetDevice(0));
    RT_CHECK(cudaFree(0));                 // force primary-context creation + bind
    CU_CHECK(cuInit(0));
    CUcontext ctx = nullptr; CU_CHECK(cuCtxGetCurrent(&ctx));
    if (ctx == nullptr) {                  // be defensive: retain + bind the primary ctx
        CUdevice d; CU_CHECK(cuDeviceGet(&d, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, d));
        CU_CHECK(cuCtxSetCurrent(ctx));
    }

    // --- load the whole plugin set via the independent registry (no nvcc) ------
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(dir + "manifest.json");
    if (flag_off) registry.set_enabled(false);

    // --- device buffers (all PRE-ALLOCATED; the registry only references them) --
    double *d_pos, *d_vel, *d_spos, *d_svel, *d_outVec, *d_mass, *d_area, *d_cd;
    char* d_term;
    RT_CHECK(cudaMalloc(&d_pos, vlen * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_vel, vlen * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_spos, vlen * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_svel, vlen * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_outVec, vlen * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_mass, N * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_area, N * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_cd, N * sizeof(double)));
    RT_CHECK(cudaMalloc(&d_term, N));
    RT_CHECK(cudaMemset(d_term, 0, N));    // nothing terminated

    // Upload whichever inputs the manifest's plugins actually use; the rest stay
    // zeroed (a plugin never reads a buffer outside its own arg_spec).
    auto upload = [&](double* dptr, const std::string& fname, std::size_t count,
                      bool required) {
        const std::string p = dir + fname;
        if (!file_exists(p)) {
            if (required) { std::fprintf(stderr, "missing required %s\n", p.c_str()); std::exit(2); }
            RT_CHECK(cudaMemset(dptr, 0, count * sizeof(double)));
            return;
        }
        const std::vector<double> h = read_doubles(p, count);
        RT_CHECK(cudaMemcpy(dptr, h.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    };
    upload(d_pos, "position.bin", vlen, /*required=*/true);
    upload(d_vel, "velocity.bin", vlen, /*required=*/false);
    upload(d_mass, "mass.bin", N, /*required=*/false);
    upload(d_area, "area.bin", N, /*required=*/false);
    upload(d_cd, "cd.bin", N, /*required=*/false);

    double mu = 0.0;
    if (file_exists(dir + "uniforms.bin")) mu = read_doubles(dir + "uniforms.bin", 1)[0];

    // Bind everything by name; the registry hands each plugin only the bindings
    // its sidecar arg_spec names. Vectors point at the *scaled* buffers.
    registry.bind_vector("out", d_outVec, N);
    registry.bind_vector("position", d_spos, N);
    registry.bind_vector("velocity", d_svel, N);
    registry.bind_handle("mass", d_mass);
    registry.bind_handle("area", d_area);
    registry.bind_handle("cd", d_cd);
    registry.bind_handle("terminated", d_term);
    registry.bind_uniform("mu", mu);

    // --- consolidation: allocate + upload each declared lookup table -----------
    // The registry-owned "bring your own data" phase, run BEFORE capture. The host
    // supplies each declared table's data (here from <name>.bin); the registry
    // allocates device memory, uploads it once, and holds it for the run so
    // inject() can pack it by value. The `noconsolidate` mode skips this on
    // purpose, so a table plugin then hits the clear "not supplied" error.
    if (!no_consolidate) {
        for (const auto& t : registry.declared_tables()) {
            const std::vector<double> h = read_doubles(dir + t.name + ".bin", t.count);
            registry.consolidate(t.name, h.data(), t.count);
        }
    }

    const int block = 256, grid = (N + block - 1) / block;

    // === capture the pipeline, injecting the plugin SET in the middle ==========
    cudaStream_t stream; RT_CHECK(cudaStreamCreate(&stream));
    RT_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    backend_pre<<<grid, block, 0, stream>>>(d_pos, d_vel, SCALE, d_spos, d_svel, N);
    RT_CHECK(cudaGetLastError());
    RT_CHECK(cudaMemsetAsync(d_outVec, 0, vlen * sizeof(double), stream)); // reset accumulator

    // ---------------- THE FLAG-GATED INJECTION OPENING -------------------------
    // The host owns the capture; the registry just launches the enabled plugins
    // on the (capturing) stream — each becomes a node accumulating into outVec.
    const int injected = registry.inject(reinterpret_cast<CUstream>(stream), N, block);
    // ---------------------------------------------------------------------------

    backend_post<<<grid, block, 0, stream>>>(d_outVec, POST, N);
    RT_CHECK(cudaGetLastError());

    cudaGraph_t graph; RT_CHECK(cudaStreamEndCapture(stream, &graph));

    std::size_t n_nodes = 0;
    RT_CHECK(cudaGraphGetNodes(graph, nullptr, &n_nodes));

    cudaGraphExec_t exec; RT_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    // Replay the captured graph twice (plain graph reuse): a force is stateless, so
    // each replay recomputes the same outVec from the same inputs.
    RT_CHECK(cudaGraphLaunch(exec, stream));
    RT_CHECK(cudaGraphLaunch(exec, stream));     // replay: full pipeline again
    RT_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> got(vlen);
    RT_CHECK(cudaMemcpy(got.data(), d_outVec, vlen * sizeof(double), cudaMemcpyDeviceToHost));
    const std::vector<double> golden = read_doubles(dir + "golden.bin", vlen);
    const double rel = max_rel(got, golden, N);

    // Node count = backend_pre (1) + outVec memset (1) + active plugins +
    // backend_post (1).
    const std::size_t expected = std::size_t(3 + registry.active());

    std::printf("N_PLUGINS=%zu\n", registry.size());
    std::printf("ACTIVE=%d\n", registry.active());
    std::printf("INJECTED=%d\n", injected);
    std::printf("TOTAL_NODES=%zu\n", n_nodes);
    std::printf("INJECT_MAX_REL=%.3e\n", rel);

    const bool ok = (n_nodes == expected) && (rel < TOL);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Thin wrapper: surface a consolidation/registry error (e.g. a lookup table not
// supplied) as a clean stderr message + nonzero exit, rather than a terminate.
int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
