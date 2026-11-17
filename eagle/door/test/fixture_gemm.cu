// SPDX-License-Identifier: Apache-2.0

// The device functions of fixture_gemm.c (built to a fatbin for sm_61 and embedded in libfixture_gemm.so): the same
// products as the unit's host loop, so a DEVICE plan the door cannot route to cuBLAS has a kernel to launch. The whole
// product is one piece of work: the thread of the partition that starts at sample 0 does it, as the host entry does.
#include "plugin/gref_layout.h"

using eagle::plugin::ScalarHandle;

namespace {

constexpr long long kLane = -1;   // an extent of n_samples / lane_fanin; the fixture's fan-in is 3
constexpr unsigned kTa = 1u, kTb = 2u, kTc = 4u, kAcc = 8u;   // EG_GEMM_TRANS_A, _B, _C, _ACCUMULATE

__device__ long long extent(long long e, long long n) { return e == kLane ? n / 3 : e; }

/** C (+)= op(A) op(B) over row-major planes of (extent + slack) columns; sums in double, stored in T. */
template <class T>
__device__ void product(const ScalarHandle& ha, const ScalarHandle& hb, const ScalarHandle& hc, unsigned flags, long long ar, long long ac,
                        long long br, long long bc, long long cr, long long cc, long long slack, long long n) {
    ar = extent(ar, n), ac = extent(ac, n), br = extent(br, n), bc = extent(bc, n), cr = extent(cr, n), cc = extent(cc, n);
    const T* a = static_cast<const T*>(ha.data);
    const T* b = static_cast<const T*>(hb.data);
    T* c = static_cast<T*>(hc.data);
    const bool ta = flags & kTa, tb = flags & kTb, tc = flags & kTc;
    const long long lda = ac + slack, ldb = bc + slack, ldc = cc + slack;
    const long long m = ta ? ac : ar, k = ta ? ar : ac, cols = tb ? br : bc;
    for (long long r = 0; r < m; ++r)
        for (long long q = 0; q < cols; ++q) {
            double sum = 0.0;
            for (long long p = 0; p < k; ++p)
                sum += static_cast<double>(ta ? a[p * lda + r] : a[r * lda + p]) * static_cast<double>(tb ? b[q * ldb + p] : b[p * ldb + q]);
            T& out = c[tc ? q * ldc + r : r * ldc + q];
            out = static_cast<T>((flags & kAcc) ? static_cast<double>(out) + sum : sum);
        }
}

__device__ bool leads(long long base) { return base == 0 && blockIdx.x * blockDim.x + threadIdx.x == 0; }

}  // namespace

// <width>_<shape>: one forward product (the shapes of fixture_gemm.c).
#define FORWARD(name, T, flags, ar, ac, br, bc, cr, cc, slack)                                                                  \
    extern "C" __global__ void name(ScalarHandle a, ScalarHandle b, ScalarHandle c, long long base, long long count,           \
                                    long long n) {                                                                              \
        (void)count;                                                                                                            \
        if (leads(base)) product<T>(a, b, c, flags, ar, ac, br, bc, cr, cc, slack, n);                                          \
    }
#define WIDTH(w, T)                                                                                                             \
    FORWARD(w##_sq_dev, T, 0, 4, 4, 4, 4, 4, 4, 0) FORWARD(w##_rect_dev, T, 0, 3, 5, 5, 4, 3, 4, 0)                              \
    FORWARD(w##_ta_dev, T, kTa, 5, 3, 5, 4, 3, 4, 0) FORWARD(w##_tb_dev, T, kTb, 3, 5, 4, 5, 3, 4, 0)                            \
    FORWARD(w##_tc_dev, T, kTc, 3, 5, 5, 4, 4, 3, 0) FORWARD(w##_all_dev, T, kTa | kTb | kTc, 5, 3, 4, 5, 4, 3, 0)              \
    FORWARD(w##_lane_dev, T, 0, 3, 5, 5, kLane, 3, kLane, 0) FORWARD(w##_wide_dev, T, 0, 3, 5, 5, 4, 3, 4, 2)                    \
    extern "C" __global__ void w##_vjp_dev(ScalarHandle wt, ScalarHandle x, ScalarHandle dy, ScalarHandle dw, ScalarHandle dx,  \
                                           long long base, long long count, long long n) {                                      \
        (void)count;                                                                                                            \
        if (!leads(base)) return;                                                                                               \
        product<T>(dy, x, dw, kTb, 3, 4, 5, 4, 3, 5, 0, n);                                                                     \
        product<T>(wt, dy, dx, kTa | kAcc, 3, 5, 3, 4, 5, 4, 0, n);                                                             \
    }
WIDTH(f64, double)
WIDTH(f32, float)

// gemm_index: the f64 rect product beside an index count plane; gemm_finish: the same beside the slots of a finishing unit.
extern "C" __global__ void gemm_index_dev(ScalarHandle a, ScalarHandle b, ScalarHandle c, ScalarHandle count, long long base,
                                          long long n_count, long long n) {
    (void)count, (void)n_count;
    if (leads(base)) product<double>(a, b, c, 0, 3, 5, 5, 4, 3, 4, 0, n);
}

extern "C" __global__ void gemm_finish_dev(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle a,
                                           ScalarHandle b, ScalarHandle c, long long base, long long count, long long n) {
    (void)k, (void)limit, (void)mask, (void)finished, (void)count;
    if (leads(base)) product<double>(a, b, c, 0, 3, 5, 5, 4, 3, 4, 0, n);
}
