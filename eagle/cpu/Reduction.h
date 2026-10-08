// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

#include <omp.h>
#include <type_traits>
#include <vector>

namespace eagle {
namespace cpu {

/**
 * @brief CPU parallel reduction over aether scalar arrays using OpenMP.
 *
 * Cache-line padding (``alignas(64)`` on ``PaddedT``) prevents false sharing
 * across threads in the per-thread accumulator buffer.
 *
 * @tparam T   Element type of the array being reduced.
 * @tparam OP  Reduction operator; must be CALLABLE as ``OP{}(a, b)``
 *             (aether's functor shape — there is no static
 *             ``OP::eval``), or be one of ``aether::SumOp<T>``,
 *             ``aether::MaxOp<T>`` or ``aether::LogicalAndOp`` (which use
 *             native OpenMP reduction clauses for better performance).
 */
template<typename T, typename OP>
struct Reduction {
    using Self = Reduction;

    /* Avoid false sharing by padding each thread's local value */
    struct alignas(64) PaddedT {
        T value;
    };

    /** @brief Run the OpenMP based reduction */
    static T reduce(const CRefArrT<T>& in,
        std::vector<PaddedT>& buffer, const T& init = 0)
    {
        const idx_t sz = in.samples();
        EAGLE_ASSERT(buffer.size() == (idx_t)omp_get_max_threads(),
            "Number of threads is not compatible with OpenMP buffer size");
        const idx_t numThreads = buffer.size();

        T final = init;

        if constexpr (std::is_same_v<OP, aether::SumOp<T>>) {
#pragma omp parallel for reduction(+ : final)
            for (idx_t i = 0; i < sz; i++) {
                final += in(i);
            }
            return final;
        }

        if constexpr (std::is_same_v<OP, aether::MaxOp<T>>) {
#pragma omp parallel for reduction(max : final)
            for (idx_t i = 0; i < sz; i++) {
                const T v = in(i);
                if (v > final)
                    final = v;
            }
            return final;
        }

        if constexpr (std::is_same_v<OP, aether::LogicalAndOp>) {
            static_assert(std::is_same_v<T, bool>,
                "logicalAnd reduction requires bool type");
#pragma omp parallel for reduction(&& : final)
            for (idx_t i = 0; i < sz; i++) {
                final = final && in(i);
            }
            return final;
        }

#pragma omp parallel
        {
            /* Create a copy per thread */
            idx_t tid = omp_get_thread_num();
            T local   = init;

            /* Run the reduction on thread copies */
#pragma omp for simd
            for (idx_t i = 0; i < sz; i++) {
                local = OP{}(local, in(i));
            }

            buffer[tid].value = local;
        }

        /* Final reduction */
#pragma omp simd
        for (idx_t i = 0; i < numThreads; ++i) {
            final = OP{}(final, buffer[i].value);
        }

        /* Copy the results back where it should be */
        return final;
    }

    /** @brief Allocate buffer on the fly and run the OpenMP based reduction */
    static T reduce(const CRefArrT<T>& in, const T& init = 0)
    {
        std::vector<PaddedT> buffer(omp_get_max_threads(), { init });
        return Self::reduce(in, buffer, init);
    }
};

} // namespace cpu
} // namespace eagle
