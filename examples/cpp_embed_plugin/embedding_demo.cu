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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>           // Driver API -- loads + launches the JIT plugins
#include <cuda_runtime.h>  // Runtime API -- capture, graph

#include "plugin/plugin_registry/registry.h"

#ifndef EMBED_DEMO_ART_DIR
#define EMBED_DEMO_ART_DIR "art"   // fallback if built outside this target's CMakeLists.txt
#endif

namespace {

constexpr double TOL     = 1e-12;  // mattrace / primal: exact closed-form agreement
constexpr double FD_H    = 1e-6;   // central-difference step
constexpr double FD_RTOL = 1e-5;   // VJP-vs-FD relative tolerance
constexpr double FD_ATOL = 1e-8;

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

// The SAME closed form fixtures/toy_energy.cu computes on device -- an
// independent (different language, different hardware) host implementation.
double host_energy(double x0, double x1, double x2, double p) {
    const double r2 = x0 * x0 + x1 * x1 + x2 * x2;
    return p * std::exp(-0.5 * r2);
}

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

    // --- host inputs (deterministic RNG; no external data files) ---------------
    std::mt19937_64 rng(20260719);
    std::uniform_real_distribution<double> ux(-1.3, 1.3);
    std::uniform_real_distribution<double> ueb(0.3, 2.0);
    const double p = 1.7;

    std::vector<double> M(9 * std::size_t(N));   // flat (9, N) SoA, mattrace input
    for (int d = 0; d < 9; ++d)
        for (std::uint32_t i = 0; i < N; ++i)
            M[std::size_t(d) * N + i] = double(d) + 0.25 * double(i);

    std::vector<double> x(3 * std::size_t(N));   // flat (3, N) SoA, shared primal input
    for (int k = 0; k < 3; ++k)
        for (std::uint32_t i = 0; i < N; ++i)
            x[std::size_t(k) * N + i] = ux(rng);

    std::vector<double> e_bar(N);
    for (std::uint32_t i = 0; i < N; ++i) e_bar[i] = ueb(rng);

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
    RT_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    const int injected = registry.inject(reinterpret_cast<CUstream>(stream), int(N), block);
    cudaGraph_t graph; RT_CHECK(cudaStreamEndCapture(stream, &graph));

    std::size_t n_nodes = 0;
    RT_CHECK(cudaGraphGetNodes(graph, nullptr, &n_nodes));

    cudaGraphExec_t exec; RT_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    RT_CHECK(cudaGraphLaunch(exec, stream));
    RT_CHECK(cudaGraphLaunch(exec, stream));   // replay: every plugin here is idempotent
    RT_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> out(N), e(N), x_bar(3 * std::size_t(N)), p_bar(N);
    RT_CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(double), cudaMemcpyDeviceToHost));
    RT_CHECK(cudaMemcpy(e.data(), d_e, N * sizeof(double), cudaMemcpyDeviceToHost));
    RT_CHECK(cudaMemcpy(x_bar.data(), d_x_bar, x_bar.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
    RT_CHECK(cudaMemcpy(p_bar.data(), d_p_bar, N * sizeof(double), cudaMemcpyDeviceToHost));

    // --- gap 1 check: mattrace vs the direct host sum ---------------------------
    double mat_max_rel = 0.0;
    for (std::uint32_t i = 0; i < N; ++i) {
        const double ref = M[0 * N + i] + M[4 * N + i] + M[8 * N + i];
        const double rel = std::abs(out[i] - ref) / std::max(std::abs(ref), 1.0);
        mat_max_rel = std::max(mat_max_rel, rel);
    }

    // --- gap 2 check, primal: e vs the closed-form host reference ---------------
    double primal_max_rel = 0.0;
    for (std::uint32_t i = 0; i < N; ++i) {
        const double ref = host_energy(x[0 * N + i], x[1 * N + i], x[2 * N + i], p);
        const double rel = std::abs(e[i] - ref) / std::max(std::abs(ref), 1e-12);
        primal_max_rel = std::max(primal_max_rel, rel);
    }

    // --- gap 2 check, VJP: x_bar/p_bar vs an INDEPENDENT central-difference of the
    // same host closed form (not the analytic formula the device kernel uses) ----
    double vjp_x_max_rel = 0.0, vjp_p_max_rel = 0.0;
    for (std::uint32_t i = 0; i < N; ++i) {
        const double x0 = x[0 * N + i], x1 = x[1 * N + i], x2 = x[2 * N + i];
        double want_x[3];
        want_x[0] = e_bar[i] *
            (host_energy(x0 + FD_H, x1, x2, p) - host_energy(x0 - FD_H, x1, x2, p)) / (2 * FD_H);
        want_x[1] = e_bar[i] *
            (host_energy(x0, x1 + FD_H, x2, p) - host_energy(x0, x1 - FD_H, x2, p)) / (2 * FD_H);
        want_x[2] = e_bar[i] *
            (host_energy(x0, x1, x2 + FD_H, p) - host_energy(x0, x1, x2 - FD_H, p)) / (2 * FD_H);
        const double want_p = e_bar[i] *
            (host_energy(x0, x1, x2, p + FD_H) - host_energy(x0, x1, x2, p - FD_H)) / (2 * FD_H);
        for (int k = 0; k < 3; ++k) {
            const double got = x_bar[std::size_t(k) * N + i];
            const double rel = std::abs(got - want_x[k]) / std::max(std::abs(want_x[k]), FD_ATOL);
            vjp_x_max_rel = std::max(vjp_x_max_rel, rel);
        }
        const double relp = std::abs(p_bar[i] - want_p) / std::max(std::abs(want_p), FD_ATOL);
        vjp_p_max_rel = std::max(vjp_p_max_rel, relp);
    }

    std::printf("N_PLUGINS=%zu\n", registry.size());
    std::printf("ACTIVE=%d\n", registry.active());
    std::printf("INJECTED=%d\n", injected);
    std::printf("TOTAL_NODES=%zu\n", n_nodes);
    std::printf("MATTRACE_MAX_REL=%.3e\n", mat_max_rel);
    std::printf("PRIMAL_MAX_REL=%.3e\n", primal_max_rel);
    std::printf("VJP_X_MAX_REL_VS_FD=%.3e\n", vjp_x_max_rel);
    std::printf("VJP_P_MAX_REL_VS_FD=%.3e\n", vjp_p_max_rel);

    const bool ok = (injected == 3) && (n_nodes == 3)
        && (mat_max_rel < TOL) && (primal_max_rel < TOL)
        && (vjp_x_max_rel < FD_RTOL) && (vjp_p_max_rel < FD_RTOL);
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
