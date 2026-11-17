// SPDX-License-Identifier: Apache-2.0

// The host matrix product: a plain ordered triple loop, one thread, accumulating in the planes' own width, so the
// result is the same bits on every run and every thread count. Reference speed: no blocking, no vectorised library.
#include "eagle/door/host/Ops.h"

namespace eagle {
namespace door {
namespace host {

namespace {

/** @brief The first float of the 2-D plane @p p. */
template <class T>
T* first(const eg_plane_s& p) {
    const DLTensor& t = p.tensor->dl_tensor;
    return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

/** @brief C (+)= op(A) op(B) over row-major planes of row strides lda, ldb, ldc (each plane's second extent). */
template <class T>
void exec(const GemmCall& g, const eg_plane_s& a, const eg_plane_s& b, const eg_plane_s& c) {
    const bool ta = g.flags & EG_GEMM_TRANS_A, tb = g.flags & EG_GEMM_TRANS_B, tc = g.flags & EG_GEMM_TRANS_C, accumulate = g.flags & EG_GEMM_ACCUMULATE;
    const std::int64_t lda = a.tensor->dl_tensor.shape[1], ldb = b.tensor->dl_tensor.shape[1], ldc = c.tensor->dl_tensor.shape[1];
    const T *pa = first<T>(a), *pb = first<T>(b);
    T* pc = first<T>(c);
    for (std::int64_t i = 0; i < g.m; ++i)
        for (std::int64_t j = 0; j < g.n; ++j) {
            T sum = 0;
            for (std::int64_t p = 0; p < g.k; ++p)
                sum += (ta ? pa[p * lda + i] : pa[i * lda + p]) * (tb ? pb[j * ldb + p] : pb[p * ldb + j]);
            T& out = pc[tc ? j * ldc + i : i * ldc + j];
            out = accumulate ? out + sum : sum;
        }
}

}  // namespace

void gemm_exec(const eg_gemm_s& gemm, const eg_plane_s& a, const eg_plane_s& b, const eg_plane_s& c) {
    if (gemm.f32) exec<float>(gemm.call, a, b, c);
    else exec<double>(gemm.call, a, b, c);
}

}  // namespace host
}  // namespace door
}  // namespace eagle
