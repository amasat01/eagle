// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `eagle::exec::DeviceKernel` — the DEVICE execution structure of the heterogeneous
// execution contract (aether-abi/2).
//
// ONE `cuLaunchKernel` over one partition, with the int64 triple appended after the
// role args. The GRID IS EAGLE'S ("plugins contain no launch geometry"): the
// kernel still early-outs on a FLAT index, exactly as a generated kernel does
// today, and the global sample index it computes is `base + flat`.
//
// Deliberately parallel in shape to `plugin/plugin_registry/registry.h`: Driver API
// only, no aether. It takes an already-packed `params[]` (the registry owns
// the role -> wire packing and the bindings) and adds only what a PARTITION means.
#pragma once

#include "eagle/exec/Partition.h"

#include "eagle/cuda/detail/DriverEntry.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace eagle {
namespace exec {

namespace detail {

using cuda::detail::driver_entry;
using cuda::detail::driver_error_text;

}  // namespace detail

/**
 * @brief The device execution structure: one kernel launch per partition.
 */
struct DeviceKernel {

    /** @brief The launch geometry eagle derives from a partition — one thread per
     *  sample OF THIS PARTITION (`count`, never `nSamples`). */
    static unsigned grid(std::int64_t count, unsigned block) {
        return launch_grid(count, block);
    }

    /**
     * @brief Launch @p fn once over @p part on @p stream.
     *
     * @param fn     the plugin's `CUfunction` (the registry resolved the symbol).
     * @param args   the packed role args, in `arg_spec` order. Passed WHOLE — eagle
     *               does no pointer arithmetic on them (L2).
     * @param part   the partition; its triple is appended after the role args, so
     *               the kernel signature is `{name}({sig}, int64 base, int64 count,
     *               int64 nSamples)`.
     * @param stream the stream to issue on. Under capture the launch is recorded as
     *               a graph node, exactly like `PluginRegistry::inject`'s.
     * @param block  threads per block (eagle's choice; the plugin never sees it).
     * @return 1 if a kernel was launched, 0 if the partition was empty.
     */
    static int run(CUfunction fn, const std::vector<void*>& args,
                   const Partition& part, CUstream stream, unsigned block = 256) {
        if (fn == nullptr)
            throw std::runtime_error("DeviceKernel::run: null CUfunction");
        if (part.count <= 0) return 0;
        // The triple is a launch ARGUMENT, so its storage must outlive the
        // cuLaunchKernel call — locals here, by value into the params array.
        std::int64_t base = part.base, count = part.count, n = part.nSamples;
        std::vector<void*> params = args;
        params.push_back(&base);
        params.push_back(&count);
        params.push_back(&n);
        static const auto launch =
            detail::driver_entry<decltype(&::cuLaunchKernel)>("cuLaunchKernel");
        const CUresult r = launch(fn, grid(part.count, block), 1, 1,
                                  block, 1, 1, 0, stream, params.data(), nullptr);
        if (r != CUDA_SUCCESS)
            throw std::runtime_error(std::string("DeviceKernel::run: cuLaunchKernel "
                "failed: ") + detail::driver_error_text(r));
        return 1;
    }

    /**
     * @brief :func:`run` over packed args that may carry the registry's finish-counter
     *        cell: zeroes it on @p stream (async, no sync) before the launch, so each
     *        run through ``pack()`` + this call counts from zero with no caller step.
     *        @p packed needs ``params`` (the role args) and ``finished`` (a device
     *        address, 0 when the kernel has no counter).
     */
    template <class Packed>
    static int run(CUfunction fn, const Packed& packed, const Partition& part,
                   CUstream stream, unsigned block = 256) {
        if (packed.finished != 0 && part.count > 0) {
            static const auto zero =
                detail::driver_entry<decltype(&::cuMemsetD32Async)>("cuMemsetD32Async");
            const CUresult r = zero(packed.finished, 0, 1, stream);
            if (r != CUDA_SUCCESS)
                throw std::runtime_error(std::string("DeviceKernel::run: zeroing the "
                    "finished counter failed: ") + detail::driver_error_text(r));
        }
        return run(fn, packed.params, part, stream, block);
    }

    /**
     * @brief Combine @p n device-resident partials under @p op, in eagle's FIXED
     *        ascending order (never atomics for the deterministic arm).
     *
     * The partial plane is a handful of doubles (one slot per partition/tile), so
     * it is staged to the host and folded with the SAME `exec::fold` the host team
     * uses — one combine order across both arms, which is what makes the
     * host/device twin band meaningful. A device-side tree reduction
     * (`eagle/cuda/Reduction.h`) is the path for a plane large enough to matter;
     * it is aether-typed (`aether::Array<T>`) and would drag aether into this
     * deliberately ABI-only header, so it is not used here.
     */
    static double combine_partials(ReduceOp op, CUdeviceptr partials,
                                   std::size_t n) {
        std::vector<double> host(n, 0.0);
        if (n > 0) {
            static const auto copy =
                detail::driver_entry<decltype(&::cuMemcpyDtoH)>("cuMemcpyDtoH");
            const CUresult r = copy(host.data(), partials, n * sizeof(double));
            if (r != CUDA_SUCCESS)
                throw std::runtime_error(std::string("DeviceKernel::combine_partials:"
                    " cuMemcpyDtoH failed: ") + detail::driver_error_text(r));
        }
        return fold(op, host.data(), n);
    }
};

}  // namespace exec
}  // namespace eagle
