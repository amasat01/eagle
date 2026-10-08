// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Host-graph performance assessment (standalone, CPP mode only).
//
// NOT a gated gtest: timings are noisy, so this is an *assessment* that prints a
// table. Run it directly (build_cpp/tests/eagle_bench_host_graph). It answers the
// two questions behind the host-graph + plugin design:
//   (1) does the executor add a penalty vs calling the kernels directly?  -> no
//   (2) does the graph-level arena keep memory flat as chains deepen?      -> yes
//   (3) does the dlopen'd CPU plugin match a native OpenMP loop and scale?  -> yes
#include "eagle/filtering/ScanNode.h"
#include "eagle/cpu/Graph.h"
#include "eagle/cpu/Host.h"
#include "eagle/reduce/ReductionNode.h"
#include "eagle/cuda/Reduction.h"
#include "eagle/cpu/Reduction.h"
#include "plugin/host_registry.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <omp.h>
#include <vector>

using namespace eagle::plugin;  // the ABI protocol PODs (Sidecar / ArgEntry / handles)

using eagle::idx_t;
using Arr     = aether::Array<double>;
using IArr    = aether::Array<idx_t>;
using SumOp   = aether::SumOp<double>;
using RN      = eagle::reduce::ReductionNode<double, SumOp>;
using IScanOp = aether::SumOp<idx_t>;
using SN      = eagle::filtering::ScanNode<idx_t, IScanOp, false>;

template<typename F>
static double time_ms(int iters, F&& f)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static Sidecar makeAddvecSidecar()
{
    Sidecar sc;
    sc.kernel         = "addvec";
    sc.aether_abi       = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float64";
    sc.arg_spec       = { ArgEntry{ "mutable", "out" }, ArgEntry{ "per_sample", "a" },
        ArgEntry{ "per_sample", "b" }, ArgEntry{ "nsamples", "n" } };
    sc.mutables       = { MutableInfo{ "out", "float", 1 } };
    return sc;
}

int main()
{
    std::printf("=== Host-graph performance assessment ===\n");
    std::printf("omp_max_threads=%d\n\n", omp_get_max_threads());

    // [1] Executor overhead: cpu::Graph.run() over a K-chain of reductions vs the
    //     same K cpu::Reduction calls inline. The delta is the pure per-node
    //     closure-dispatch cost of the executor (the nodes do identical work).
    //     Pinned to ONE thread so the measurement isolates the executor's
    //     structural cost (a std::function call + arena pointer resolve) from
    //     OpenMP parallel-region wake-up jitter, which is variance inherent to
    //     OMP and identical on both paths (it dominates for tiny multi-thread
    //     kernels — an OMP property, orthogonal to the executor).
    {
        omp_set_num_threads(1);
        const idx_t N = idx_t(1) << 16;
        const int K   = 16;
        std::vector<std::unique_ptr<Arr>> arrs;
        for (int k = 0; k < K; ++k) {
            arrs.push_back(std::make_unique<Arr>(eagle::makeArray<double>(N)));
            auto h = arrs[k]->hostView();
            for (idx_t i = 0; i < N; ++i)
                h(i) = 1.0 + double((i + k) % 7);
        }
        std::vector<double> r(K, 0.0), r2(K, 0.0);

        eagle::cpu::Graph g;
        idx_t prev = 0;
        for (int k = 0; k < K; ++k) {
            std::vector<idx_t> deps;
            if (k > 0)
                deps = { prev };
            prev = g.addNative(
                RN(&r[k], arrs[k]->hostView().as_const(), N, 0.0), deps);
        }
        g.finalize();

        const int iters = 200;
        const double a  = time_ms(iters, [&] { g.run(); });
        const double b  = time_ms(iters, [&] {
            for (int k = 0; k < K; ++k)
                r2[k] = eagle::cpu::Reduction<double, SumOp>::reduce(
                    arrs[k]->hostView().as_const(), 0.0);
        });
        std::printf("[1] executor overhead: graph=%.4f ms  inline=%.4f ms  "
                    "per-node=%.1f ns  (K=%d, N=%d)\n",
            a, b, (a - b) * 1e6 / K, K, int(N));
    }

    // [2] Arena footprint: a K-deep chain of ScanNode, each reserving an N-element
    //     BLOCKSUMS slot. Ancestor-liveness reuse collapses the whole chain to ONE
    //     slot -> peak stays flat as K grows (the graph-level-arena rationale).
    {
        std::printf("\n[2] arena footprint (chained ScanNode, each reserves N idx):\n");
        const idx_t N = idx_t(1) << 16;
        IArr in  = eagle::makeArray<idx_t>(N);
        IArr out = eagle::makeArray<idx_t>(N);
        {
            auto h = in.hostView();
            for (idx_t i = 0; i < N; ++i)
                h(i) = idx_t(i % 2);
        }
        for (int K : { 1, 2, 4, 8, 16, 32, 64 }) {
            eagle::cpu::Graph g;
            idx_t prev = 0;
            for (int k = 0; k < K; ++k) {
                std::vector<idx_t> deps;
                if (k > 0)
                    deps = { prev };
                prev = g.addNative(
                    SN(in.hostView().as_const(), out.hostView()), deps);
            }
            g.finalize();
            std::printf("    K=%2d  slots=%d  peak=%zu B   (per-node-ownership would "
                        "be %zu B)\n",
                K, int(g.scratchArena().slotCount()),
                std::size_t(g.scratchArena().peakBytes()),
                std::size_t(K) * std::size_t(N) * sizeof(idx_t));
        }
    }

    // [3+5] CPU plugin vs native OpenMP addvec, thread scaling. The dlopen'd plugin
    //       runs its OWN #pragma omp parallel for; the native loop is the same op
    //       inline. Ratio ~1 == no plugin-dispatch penalty; both scale with threads.
    {
        std::printf("\n[3+5] plugin (dlopen'd) vs native OpenMP addvec, thread scaling:\n");
        const idx_t N = idx_t(1) << 22;
        std::vector<double> a(N), b(N), op(N, 0.0), on(N, 0.0);
        for (idx_t i = 0; i < N; ++i) {
            a[i] = double(i);
            b[i] = 2.0 * double(i);
        }

        eagle::cpu::PluginRegistry reg;
        reg.add_plugin(makeAddvecSidecar(), HOST_PLUGIN_ADDVEC_SO);
        reg.bind_handle("out", op.data());
        reg.bind_handle("a", a.data());
        reg.bind_handle("b", b.data());

        const int iters    = 100;
        const double bytes = 3.0 * double(N) * sizeof(double);  // 2 read + 1 write
        double tp1         = 0.0;
        for (int t : { 1, 2, 4, 8 }) {
            omp_set_num_threads(t);
            const double tp = time_ms(iters, [&] { reg.run(std::int32_t(N)); });
            const double tn = time_ms(iters, [&] {
#pragma omp parallel for
                for (idx_t i = 0; i < N; ++i)
                    on[i] = a[i] + b[i];
            });
            if (t == 1)
                tp1 = tp;
            std::printf("    threads=%d  plugin=%.3f ms (%.1f GB/s)  native=%.3f ms "
                        "(%.1f GB/s)  plugin/native=%.2f  scaling=%.2fx\n",
                t, tp, bytes / (tp * 1e6), tn, bytes / (tn * 1e6), tp / tn, tp1 / tp);
        }
        bool ok = true;
        for (idx_t i = 0; i < N; ++i)
            if (op[i] != on[i]) {
                ok = false;
                break;
            }
        std::printf("    plugin result == native result: %s\n", ok ? "yes" : "NO");
    }

    std::printf("\n=== done ===\n");
    return 0;
}
