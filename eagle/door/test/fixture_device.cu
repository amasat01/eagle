// SPDX-License-Identifier: Apache-2.0

// The device fixture's kernel (built to a fatbin for sm_61 and embedded in libfixture_device.so):
// y = fma(a, x, v[0]) + n * v[1], every operation rounded on its own (the test's plain-C oracle
// computes the same), over the partition triple a DEVICE launch appends.
#include "plugin/gref_layout.h"

using eagle::plugin::GRefMirror;
using eagle::plugin::ScalarHandle;

extern "C" __global__ void axpy(ScalarHandle y, ScalarHandle x, GRefMirror v, double a, unsigned n,
                                long long base, long long count, long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)nSamples;
    const double* xs = static_cast<const double*>(x.data);
    const double* vs = v.data_;
    static_cast<double*>(y.data)[i] =
        __dadd_rn(__fma_rn(a, xs[i], vs[i]), __dmul_rn(static_cast<double>(n), vs[v.compStride_ + i]));
}

// dcells: the rows of 4 samples hold the min (tot) and the max (mx) of x (the row's first sample resets them).
extern "C" __global__ void dcells(ScalarHandle tot, ScalarHandle mx, ScalarHandle x, long long base, long long count,
                                  long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)nSamples;
    if (i % 4 != 0) return;
    double* t = static_cast<double*>(tot.data);
    double* m = static_cast<double*>(mx.data);
    const double* xs = static_cast<const double*>(x.data);
    t[i / 4] = 1e300;
    m[i / 4] = -1e300;
    for (long long j = i; j < i + 4 && j < nSamples; ++j) {
        if (xs[j] < t[i / 4]) t[i / 4] = xs[j];
        if (xs[j] > m[i / 4]) m[i / 4] = xs[j];
    }
}

// dkeyed: acc[i / 4] += x[i] and comp[i / 4] += 0.5 for the 4 samples of the key the lane leads; the fold:
// acc += comp, comp = 0.125 over the target's slots.
extern "C" __global__ void dkeyed(ScalarHandle acc, ScalarHandle comp, ScalarHandle x, long long base, long long count,
                                  long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    if (i % 4 != 0) return;
    const double* xs = static_cast<const double*>(x.data);
    for (long long j = i; j < i + 4 && j < nSamples; ++j) {
        static_cast<double*>(acc.data)[i / 4] += xs[j];
        static_cast<double*>(comp.data)[i / 4] += 0.5;
    }
}

extern "C" __global__ void dkeyed__fold_acc(ScalarHandle acc, ScalarHandle comp, ScalarHandle x, long long base,
                                            long long count, long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)x;
    (void)nSamples;
    static_cast<double*>(acc.data)[i] += static_cast<double*>(comp.data)[i];
    static_cast<double*>(comp.data)[i] = 0.125;
}

// dcount: a finishing unit stepping k += 1 until k >= limit, then marking the sample and counting it; the first thread
// states the launch it runs in (geom[0] = threads per block, geom[1] = blocks).
extern "C" __global__ void dcount(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished,
                                  ScalarHandle geom, long long base, long long count, long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    (void)nSamples;
    if (flat == 0) {
        static_cast<double*>(geom.data)[0] = blockDim.x;
        static_cast<double*>(geom.data)[1] = gridDim.x;
    }
    if (flat >= count) return;
    const long long i = base + flat;
    char* done = static_cast<char*>(mask.data);
    if (done[i] != 0) return;
    double* ks = static_cast<double*>(k.data);
    ks[i] += 1.0;
    if (ks[i] >= static_cast<const double*>(limit.data)[i]) {
        done[i] = 1;
        atomicAdd(static_cast<unsigned*>(finished.data), 1u);
    }
}

// dprefix: the first *cnt lanes (the live prefix count, a uint32 cell) hold y = 2 x + 1.
extern "C" __global__ void dprefix(ScalarHandle y, ScalarHandle x, ScalarHandle cnt, long long base, long long count,
                                   long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)nSamples;
    if (i >= static_cast<long long>(*static_cast<const unsigned*>(cnt.data))) return;
    static_cast<double*>(y.data)[i] = 2.0 * static_cast<const double*>(x.data)[i] + 1.0;
}

// dcount1 / dcount1a: one step k += 1 of every live sample until k >= limit, which sets the mask and counts it in
// `finished`; the active variant serves the sample map[t] of each thread t below the live count.
extern "C" __global__ void dcount1(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished,
                                   long long base, long long count, long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)nSamples;
    unsigned char* m = static_cast<unsigned char*>(mask.data);
    if (m[i]) return;
    double* kk = static_cast<double*>(k.data);
    kk[i] += 1.0;
    if (kk[i] >= static_cast<const double*>(limit.data)[i]) {
        m[i] = 1;
        atomicAdd(static_cast<unsigned*>(finished.data), 1u);
    }
}

extern "C" __global__ void dcount1a(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished,
                                    ScalarHandle map, ScalarHandle live, long long base, long long count,
                                    long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long t = base + flat;
    (void)nSamples;
    if (t >= static_cast<long long>(*static_cast<const unsigned*>(live.data))) return;
    const long long i = static_cast<const int*>(map.data)[t];
    unsigned char* m = static_cast<unsigned char*>(mask.data);
    if (m[i]) return;
    double* kk = static_cast<double*>(k.data);
    kk[i] += 1.0;
    if (kk[i] >= static_cast<const double*>(limit.data)[i]) {
        m[i] = 1;
        atomicAdd(static_cast<unsigned*>(finished.data), 1u);
    }
}

// dauto: an automatic unit (finish steps AUTO) with the fast entries. A launch visits each sample once: hits[i] += 1,
// done[i] = 1, count += 1 for a sample not yet done. The persist entry claims chunks of `chunk` samples from the
// uint32 cell `next` (one fetch per chunk) and writes its geometry into geo[0..2] = grid, block, chunk.
__device__ void dautoVisit(unsigned* hits, unsigned char* done, unsigned* finished, long long i) {
    atomicAdd(&hits[i], 1u);
    if (done[i] == 0) {
        done[i] = 1;
        atomicAdd(finished, 1u);
    }
}

#define DAUTO_SLOTS ScalarHandle hits, ScalarHandle geo, ScalarHandle done, ScalarHandle finished, ScalarHandle stepWord

#define DAUTO_FUSED(name)                                                                                        \
    extern "C" __global__ void name(DAUTO_SLOTS, long long base, long long count, long long nSamples) {          \
        const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;                    \
        (void)geo; (void)stepWord; (void)nSamples;                                                               \
        if (flat >= count) return;                                                                               \
        dautoVisit(static_cast<unsigned*>(hits.data), static_cast<unsigned char*>(done.data),                    \
                   static_cast<unsigned*>(finished.data), base + flat);                                          \
    }

DAUTO_FUSED(dauto)
DAUTO_FUSED(dautorange)
DAUTO_FUSED(dautorange_range)

extern "C" __global__ void dauto_persist(DAUTO_SLOTS, long long base, long long count, long long nSamples,
                                         unsigned* next, unsigned* util, unsigned* steps, unsigned* stepsum,
                                         unsigned chunk) {
    (void)stepWord; (void)util; (void)stepsum;
    if (blockIdx.x == 0 && threadIdx.x == 0 && nSamples >= 3) {
        long long* g = static_cast<long long*>(geo.data);
        g[0] = gridDim.x;
        g[1] = blockDim.x;
        g[2] = chunk;
    }
    for (;;) {
        const unsigned long long head = static_cast<unsigned long long>(atomicAdd(next, 1u)) * chunk;
        if (head >= static_cast<unsigned long long>(count)) break;
        const long long end = static_cast<long long>(head + chunk < static_cast<unsigned long long>(count) ? head + chunk : count);
        for (long long i = static_cast<long long>(head); i < end; ++i)
            dautoVisit(static_cast<unsigned*>(hits.data), static_cast<unsigned char*>(done.data),
                       static_cast<unsigned*>(finished.data), base + i);
    }
    atomicMax(steps, 1u);
}

// dlookup: the first *cnt lanes (a uint32 cell the caller owns) write y at the sample map[lane]: y[s] = 2 x[s] + 1.
extern "C" __global__ void dlookup(ScalarHandle y, ScalarHandle x, ScalarHandle map, ScalarHandle cnt, long long base,
                                   long long count, long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)nSamples;
    if (i >= static_cast<long long>(*static_cast<const unsigned*>(cnt.data))) return;
    const int s = static_cast<const int*>(map.data)[i];
    static_cast<double*>(y.data)[s] = 2.0 * static_cast<const double*>(x.data)[s] + 1.0;
}

// SEGMENTED groups (the host fixture's "sg" and "dg", test/fixture_unit.c): slots k, limit, mask, finished, then the group's own;
// thread t of branch j serves the slot offsets[j] + t. The sweep reads the live prefix of its active map.
#define SEG_HEAD(first)                                                                                      \
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;                    \
    if (flat >= count) return;                                                                               \
    const long long i = first;
__device__ static void segStep(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, long long i, double step) {
    double* ks = static_cast<double*>(k.data);
    ks[i] += step;
    if (ks[i] >= static_cast<const double*>(limit.data)[i]) {
        static_cast<char*>(mask.data)[i] = 1;
        atomicAdd(static_cast<unsigned*>(finished.data), 1u);
    }
}

extern "C" __global__ void sgCls(ScalarHandle limit, ScalarHandle cls, long long base, long long count, long long nSamples) {
    SEG_HEAD(base + flat)
    (void)nSamples;
    static_cast<long long*>(cls.data)[i] = static_cast<const double*>(limit.data)[i] >= 6.0 ? 1 : 0;
}

__device__ static void sgBranch(int j, ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                long long count) {
    const long long* off = static_cast<const long long*>(offsets.data);
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = off[j] + flat;
    if (i >= off[j + 1] || static_cast<char*>(mask.data)[i] != 0) return;
    segStep(k, limit, mask, finished, i, j + 1.0);
}

extern "C" __global__ void sgK0(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                long long base, long long count, long long nSamples) { (void)base; (void)nSamples; sgBranch(0, k, limit, mask, finished, offsets, count); }
extern "C" __global__ void sgK1(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                long long base, long long count, long long nSamples) { (void)base; (void)nSamples; sgBranch(1, k, limit, mask, finished, offsets, count); }

extern "C" __global__ void sgf(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, long long base,
                               long long count, long long nSamples) {
    SEG_HEAD(base + flat)
    (void)nSamples;
    if (static_cast<char*>(mask.data)[i] != 0) return;
    segStep(k, limit, mask, finished, i, static_cast<const double*>(limit.data)[i] >= 6.0 ? 2.0 : 1.0);
}

// dg: slots k, limit, mask, finished, offsets (4), class (5), stale (6), pos (7); the class of a sample is k >= 4.
__device__ static void dgBranch(int j, ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                ScalarHandle cls, ScalarHandle stale, ScalarHandle pos, long long count) {
    const long long* off = static_cast<const long long*>(offsets.data);
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = off[j] + flat;
    if (i >= off[j + 1] || static_cast<char*>(mask.data)[i] != 0) return;
    const int c = static_cast<const double*>(k.data)[i] >= 4.0 ? 1 : 0;
    static_cast<long long*>(cls.data)[i] = c;
    static_cast<char*>(stale.data)[i] = c != j;
    static_cast<double*>(pos.data)[i] = static_cast<double>(i);
    if (c == j) segStep(k, limit, mask, finished, i, 0.125 * (j + 1));
}

extern "C" __global__ void dgK0(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                ScalarHandle cls, ScalarHandle stale, ScalarHandle pos, long long base, long long count, long long nSamples) {
    (void)base; (void)nSamples;
    dgBranch(0, k, limit, mask, finished, offsets, cls, stale, pos, count);
}
extern "C" __global__ void dgK1(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                ScalarHandle cls, ScalarHandle stale, ScalarHandle pos, long long base, long long count, long long nSamples) {
    (void)base; (void)nSamples;
    dgBranch(1, k, limit, mask, finished, offsets, cls, stale, pos, count);
}

extern "C" __global__ void dgSweep(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle map,
                                   ScalarHandle live, long long base, long long count, long long nSamples) {
    SEG_HEAD(base + flat)
    (void)nSamples;
    if (i >= static_cast<long long>(static_cast<const unsigned*>(live.data)[0])) return;
    const long long s = static_cast<const int*>(map.data)[i];
    if (static_cast<char*>(mask.data)[s] != 0) return;
    segStep(k, limit, mask, finished, s, 0.125 * ((static_cast<const double*>(k.data)[s] >= 4.0 ? 1 : 0) + 1));
}

extern "C" __global__ void dgf(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, long long base,
                               long long count, long long nSamples) {
    SEG_HEAD(base + flat)
    (void)nSamples;
    if (static_cast<char*>(mask.data)[i] != 0) return;
    segStep(k, limit, mask, finished, i, 0.125 * ((static_cast<const double*>(k.data)[i] >= 4.0 ? 1 : 0) + 1));
}

// The AUTO group "dsga" (test/fixture_unit.c's "sga"): slots k, limit, mask, finished, then offsets or mix (4) and how (5); a step adds 1 to how
// in a branch unit and 1000 in the fused unit. Its classify unit is sgCls.
__device__ static void sgaBranch(int j, ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                 ScalarHandle how, long long count) {
    const long long* off = static_cast<const long long*>(offsets.data);
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = off[j] + flat;
    if (i >= off[j + 1] || static_cast<char*>(mask.data)[i] != 0) return;
    static_cast<double*>(how.data)[i] += 1.0;
    segStep(k, limit, mask, finished, i, j + 1.0);
}

extern "C" __global__ void sgaK0(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                 ScalarHandle how, long long base, long long count, long long nSamples) { (void)base; (void)nSamples; sgaBranch(0, k, limit, mask, finished, offsets, how, count); }
extern "C" __global__ void sgaK1(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle offsets,
                                 ScalarHandle how, long long base, long long count, long long nSamples) { (void)base; (void)nSamples; sgaBranch(1, k, limit, mask, finished, offsets, how, count); }

extern "C" __global__ void sga(ScalarHandle k, ScalarHandle limit, ScalarHandle mask, ScalarHandle finished, ScalarHandle mix, ScalarHandle how,
                               long long base, long long count, long long nSamples) {
    SEG_HEAD(base + flat)
    (void)nSamples;
    if (static_cast<char*>(mask.data)[i] != 0) return;
    const double l = static_cast<const double*>(limit.data)[i];
    atomicAdd(static_cast<unsigned*>(mix.data) + (l >= 6.0 ? 1 : 0), 1u);
    static_cast<double*>(how.data)[i] += 1000.0;
    segStep(k, limit, mask, finished, i, l >= 6.0 ? 2.0 : 1.0);
}
