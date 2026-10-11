// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/reduce/canonical.h"
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

    /** @brief Run the OpenMP based reduction.
     *
     * Integer sums, ``MaxOp`` and ``LogicalAndOp`` use native OpenMP reduction
     * clauses (exactly associative, so any schedule gives the same value).
     * A floating-point sum and any other operator use the canonical tree
     * (``eagle/reduce/canonical.h``): the result is bitwise identical for
     * every OpenMP thread count. */
    static T reduce(const CRefArrT<T>& in,
        std::vector<PaddedT>& buffer, const T& init = 0)
    {
        const idx_t sz = in.samples();
        EAGLE_ASSERT(buffer.size() == (idx_t)omp_get_max_threads(),
            "Number of threads is not compatible with OpenMP buffer size");
        const idx_t numThreads = buffer.size();

        T final = init;

        if constexpr (std::is_same_v<OP, aether::SumOp<T>>
            && !std::is_floating_point_v<T>) {
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

        /* Floating-point sum and every other operator: the canonical tree
         * (eagle/reduce/canonical.h) — a function of the values in sample
         * order only, bitwise independent of the OpenMP thread count. */
        (void)numThreads;
        if constexpr (!std::is_same_v<T, bool>) {
            if (sz == 0)
                return init;
            return OP{}(init,
                eagle::reduce::canonical::reduce<T, OP>(in, sz));
        } else {
            return final; // bool: only LogicalAndOp, handled above
        }
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
