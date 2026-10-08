// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file DeviceKernel.h
 * @brief ``eagle_seam::DeviceKernel``: ``eagle::exec::DeviceKernel::run``'s shape,
 *        forwarded to the CUDA backend's ``eagle_backend_run_device``.
 *
 * For a pure C++ TU that drives a device launch through an eagle structure
 * template (``RankPartition<Inner>`` only needs ``Inner::run``). Handles cross as
 * integers: ``fn`` is a ``CUfunction``, ``stream`` a ``cudaStream_t``.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "Loader.h"
#include "Status.h"
#include "eagle/exec/Partition.h"

namespace eagle_seam {

struct DeviceKernel {
    /** @brief One launch of @p fn over @p part on @p stream; 1, or 0 when empty. */
    static int run(std::uintptr_t fn, const std::vector<void*>& args, const eagle::exec::Partition& part,
        std::uintptr_t stream, unsigned block = 256)
    {
        std::vector<std::uint64_t> params;
        params.reserve(args.size());
        for (void* a : args)
            params.push_back(reinterpret_cast<std::uint64_t>(a));
        struct eagle_backend_launch_desc d {};
        d.struct_size = sizeof d;
        d.device = -1;
        d.block = block;
        d.function = fn;
        d.params = params.data();
        d.nparams = static_cast<std::int64_t>(params.size());
        d.base = part.base;
        d.count = part.count;
        d.n_samples = part.nSamples;
        d.stream = stream;
        std::int32_t launched = 0;
        check(backend("cuda"), EAGLE_SEAM_FN("exec", eagle_backend_run_device)(&d, &launched));
        return static_cast<int>(launched);
    }
};

} // namespace eagle_seam
