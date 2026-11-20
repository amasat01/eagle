/* SPDX-License-Identifier: Apache-2.0 */

/*
 * A hand-written unit library for the door test: three units and their meta
 * as constant data, reached through eg_unit_library_meta (declared in
 * eg_door.h, so this definition is checked against it).
 *
 *  - "axpb": y = a*x + b, sample-local, the serial v2 host entry eagle's
 *    HostTeam drives (params: y and x as 32-byte scalar handles, a and b by
 *    address).
 *  - "wait": the same slots; its entry marks y[1] = 1 on entry and returns
 *    once the test sets y[0] (or after a few seconds): it holds a context for a call to queue behind.
 *  - "tick": adds 1 to its one LOOKUP cell (a plain, non-atomic increment): exact only when launches never overlap.
 *  - "spans": logs the (base, count) of each non-empty serial call into fixture_spans, counted by fixture_span_n.
 *  - "mix2": count1 over two AUTO dispatch groups (k 2 and 3) whose mix counters the entry fills: group a counts the live samples
 *    with limit >= 6 (cell 1) or below (cell 0), group b the live samples by limit mod 3.
 *  - "widesum": y[e] = theta[3e] + 10 theta[3e + 1] + 100 theta[3e + 2], each read clamped to the WIDE_IN handle's extent.
 *  - "prefixsum": a prefix set (the live count is its last slot) with a SUM cell of rows of 4 samples: row r holds the sum of the
 *    lanes 4r .. 4r + 3 it ran.
 *  - "axpb_split": the "axpb" body named as a one-stage split unit.
 *  - "count1a", "count3a": the same finishing units reading an active set: the
 *    two LOOKUP slots after the counter hold the live sample slots (int32)
 *    and the live prefix length (uint32); thread t serves sample map[t].
 *  - "countA", "countAa": finishing units with steps AUTO: each launch steps every live sample up to
 *    the int64 word eagle sets in slot 4 (countAa also reads the active set, slots 5 and 6).
 *  - "vecs": every plane role the door binds beyond the scalar ones: a rank-1 MUTABLE and VEC_IN, a rank-2
 *    MAT_IN, an int32 PER_SAMPLE, a LOOKUP table, the NSAMPLES count, a rank-0 OUT and a uniform.
 *  - "count1g": count1 with an active set over groups of 5 lanes (the map holds group indices).
 *  - "widemul": a WIDE_OUT and a WIDE_IN plane of three rows each beside a MUTABLE plane.
 *  - "cells": two cells of rows of 4 samples (a MIN and a MAX); "cellsum": a SUM cell, which the meta cannot complete; "keyed": a keyed
 *    accumulate whose companion eagle makes and whose fold entry runs after each launch; "keyedbad": the same
 *    with no fold entry.
 *  - "vecsplit": a split unit of two stages, "vecw" and "vecr", that pass a width-4 vector through an eagle-made plane "cross":
 *    stage 1 writes cross[j, i] = (j + 1) * x[i] + j, stage 2 reads it as y[i] = sum_j (j + 1) * cross[j, i]; both read the
 *    plane through its 40-byte mirror (the component and sample pitches it states).
 *  - "actrow": an active set (slots 1 and 2) and no finish; each launch adds 1 to w[map[t]] for the live positions, the plain
 *    plan that shares a step with a finishing unit.
 *  - "count1", "count3": a finishing unit that steps k += 1 until k >= limit,
 *    then marks the sample and counts it into the finished counter; one or
 *    three steps per launch (slots: k, limit, mask, the counter).
 *
 *  - "sg": a SEGMENTED group of two branches with a static selector (the classify unit "sg__cls" reads only limit; branch j of
 *    "sg__k0" / "sg__k1" steps k by j + 1 until k >= limit) and "sgf", the fused unit computing the same steps per sample.
 *  - "sga_rare", "sga_even", "sga_scalar": AUTO units over "sg"'s static selector (the fused kernel counts its mix into sga_mix), each naming the
 *    segmented set "sga__cls", "sga__k0", "sga__k1" and stating the branch costs of test/auto_costs.h. Every step also adds to the plane "how": 1 in a
 *    branch unit, 1000 in the fused unit, so a run's path shows in it.
 *  - "dg": the same with a dynamic selector: the class is k >= 4, a state the step writes, so a branch unit marks the samples
 *    of its run whose class is not its own, the sweep unit "dg__sweep" steps those by their class, and "dgf" is the fused unit.
 *    Branch j steps k by 0.125 * (j + 1); a branch also writes the slot of its sample into the plane "pos".
 *
 * The size block beside the host entries (``eagle_abi_tag``,
 * ``eagle_layout_sizes``) follows FIXTURE_BLOCK: 0 (default) the real one,
 * 1 a wrong sizeof(ScalarHandle), 2 none exported, 3 the v1 tag. 4 is the real block with "axpb"
 * rebuilt as another library of the same unit: digest 0xA2 and y = a*x + 2b.
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"
#include "auto_costs.h"

#include <stdint.h>
#include <time.h>

#define EG_FIXTURE_EXPORT __attribute__((visibility("default")))

#ifndef FIXTURE_BLOCK
#define FIXTURE_BLOCK 0
#endif

/* "axpb" of the alternate library (FIXTURE_BLOCK 4) differs in digest and in value. */
#if FIXTURE_BLOCK == 4
#define AXPB_DIGEST_BYTE 0xA2
#define AXPB_B_FACTOR 2.0
#else
#define AXPB_DIGEST_BYTE 0xA1
#define AXPB_B_FACTOR 1.0
#endif

/* The scalar handle of the v2 host ABI (plugin/gref_layout.h, 32 bytes). */
struct fixture_handle {
    void* data;
    uint64_t samples;
    uint64_t stride;
    int32_t device_type;
    int32_t device_id;
};

/* The 40-byte mirror of a (components, n) plane (plugin/gref_layout.h's GRefMirror). */
struct fixture_mirror {
    double* data;
    uint64_t samples, comp_stride, sample_stride;
    int32_t device_type, device_id;
};

EG_FIXTURE_EXPORT void axpb_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void wait_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void tick_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void spans_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void mix2_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void widesum_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void prefixsum_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void vecw_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void vecr_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void actrow_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void count1_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void count3_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void count1a_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void count3a_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void countA_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void countAa_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void count1i_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void count1ia_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void lanes_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void prefix_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void vecs_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void count1g_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void widemul_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void gath_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void boxed_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void shr_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void cells_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void keyed_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void keyed__fold_acc_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sg_cls_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sg_k0_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sg_k1_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sgf_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void dg_k0_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void dg_k1_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void dg_sweep_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void dgf_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sga_cls_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sga_k0_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sga_k1_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void sga_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT const struct eg_library_meta* eg_unit_library_meta(void);

/* Lanes of a partial last group a count1g launch was handed (the entry skips them). */
EG_FIXTURE_EXPORT int64_t fixture_past_n;
EG_FIXTURE_EXPORT int64_t fixture_box_threads;
int64_t fixture_past_n;

void axpb_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[1])->data;
    const double a = *(const double*)params[2];
    const double b = *(const double*)params[3];
    (void)n;
    for (int64_t i = base; i < base + count; ++i) y[i] = a * x[i] + AXPB_B_FACTOR * b;
}

/* The width of the "cross" vector the "vecsplit" stages pass. */
#define VEC_WIDTH 4

/* Element (component j, sample i) of a 40-byte mirror: the pitches are the plane's own. */
static double* mirror_at(const struct fixture_mirror* m, int64_t j, int64_t i) {
    return m->data + j * (int64_t)m->comp_stride + i * (int64_t)m->sample_stride;
}

void vecw_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const struct fixture_mirror* cross = (const struct fixture_mirror*)params[0];
    const double* x = (const double*)((const struct fixture_handle*)params[1])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i)
        for (int64_t j = 0; j < VEC_WIDTH; ++j) *mirror_at(cross, j, i) = (double)(j + 1) * x[i] + (double)j;
}

void vecr_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const struct fixture_mirror* cross = (const struct fixture_mirror*)params[1];
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        double sum = 0.0;
        for (int64_t j = 0; j < VEC_WIDTH; ++j) sum += (double)(j + 1) * *mirror_at(cross, j, i);
        y[i] = sum;
    }
}

/* "actrow": adds 1 to w[map[t]] for each position t it is given (an active set, no finish). */
void actrow_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* w = (double*)((const struct fixture_handle*)params[0])->data;
    const int32_t* map = (const int32_t*)((const struct fixture_handle*)params[1])->data;
    (void)n;
    for (int64_t t = base; t < base + count; ++t) w[map[t]] += 1.0;
}

void wait_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    volatile double* y = (volatile double*)((const struct fixture_handle*)params[0])->data;
    struct timespec t0, t;
    (void)base;
    (void)count;
    (void)n;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    y[1] = 1.0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (y[0] != 0.0) return;
        clock_gettime(CLOCK_MONOTONIC, &t);
        if (t.tv_sec - t0.tv_sec > 10) return;
    }
}

void tick_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)base;
    (void)count;
    (void)n;
    ++*(int64_t*)((const struct fixture_handle*)params[0])->data;
}

EG_FIXTURE_EXPORT int64_t fixture_spans[64];
EG_FIXTURE_EXPORT int64_t fixture_span_n;
int64_t fixture_spans[64];
int64_t fixture_span_n;

void spans_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)params;
    (void)n;
    fixture_spans[2 * fixture_span_n] = base;
    fixture_spans[2 * fixture_span_n + 1] = count;
    ++fixture_span_n;
}

/* Up to `steps` steps of every live sample served by threads [base, base + count); `map` is the
 * sample of each thread, or NULL: thread t serves sample t. */
static void count_steps_w(void* const* params, int64_t base, int64_t count, int steps, const int32_t* map, int mask_bytes) {
    double* k = (double*)((const struct fixture_handle*)params[0])->data;
    const double* limit = (const double*)((const struct fixture_handle*)params[1])->data;
    char* mask = (char*)((const struct fixture_handle*)params[2])->data;
    uint32_t* finished = (uint32_t*)((const struct fixture_handle*)params[3])->data;
    for (int64_t t = base; t < base + count; ++t) {
        const int64_t i = map ? map[t] : t;
        for (int s = 0; s < steps && !(mask_bytes == 4 ? ((int32_t*)mask)[i] != 0 : mask[i] != 0); ++s) {
            k[i] += 1.0;
            if (k[i] >= limit[i]) {
                if (mask_bytes == 4) ((int32_t*)mask)[i] = 1; else mask[i] = 1;
                __atomic_fetch_add(finished, 1u, __ATOMIC_RELAXED);
            }
        }
    }
}

static void count_steps(void* const* params, int64_t base, int64_t count, int steps, const int32_t* map) {
    count_steps_w(params, base, count, steps, map, 1);
}

void count1_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps(params, base, count, 1, NULL);
}

void mix2_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const double* limit = (const double*)((const struct fixture_handle*)params[1])->data;
    const char* mask = (const char*)((const struct fixture_handle*)params[2])->data;
    uint32_t* mix_a = (uint32_t*)((const struct fixture_handle*)params[4])->data;
    uint32_t* mix_b = (uint32_t*)((const struct fixture_handle*)params[5])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        if (mask[i] != 0) continue;
        __atomic_fetch_add(&mix_a[limit[i] >= 6.0 ? 1 : 0], 1u, __ATOMIC_RELAXED);
        __atomic_fetch_add(&mix_b[(int64_t)limit[i] % 3], 1u, __ATOMIC_RELAXED);
    }
    count_steps(params, base, count, 1, NULL);
}

void count3_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps(params, base, count, 3, NULL);
}

void count1a_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps(params, base, count, 1, (const int32_t*)((const struct fixture_handle*)params[4])->data);
}

void count3a_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps(params, base, count, 3, (const int32_t*)((const struct fixture_handle*)params[4])->data);
}

void countA_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps(params, base, count, (int)*(const int64_t*)((const struct fixture_handle*)params[4])->data, NULL);
}

void countAa_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps(params, base, count, (int)*(const int64_t*)((const struct fixture_handle*)params[4])->data,
                (const int32_t*)((const struct fixture_handle*)params[5])->data);
}

/* The int32-mask variants of count1 and count1a. */
void count1i_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps_w(params, base, count, 1, NULL, 4);
}

void count1ia_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    count_steps_w(params, base, count, 1, (const int32_t*)((const struct fixture_handle*)params[4])->data, 4);
}

/* An active set over groups of 5 lanes: thread t serves lane map[t / 5] * 5 + t % 5; a lane past n is skipped. */
enum { GROUP = 5 };
void count1g_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* k = (double*)((const struct fixture_handle*)params[0])->data;
    const double* limit = (const double*)((const struct fixture_handle*)params[1])->data;
    char* mask = (char*)((const struct fixture_handle*)params[2])->data;
    uint32_t* finished = (uint32_t*)((const struct fixture_handle*)params[3])->data;
    const int32_t* map = (const int32_t*)((const struct fixture_handle*)params[4])->data;
    for (int64_t t = base; t < base + count; ++t) {
        const int64_t i = (int64_t)map[t / GROUP] * GROUP + t % GROUP;
        if (i >= n) {
            __atomic_fetch_add(&fixture_past_n, 1, __ATOMIC_RELAXED);
            continue;
        }
        if (mask[i] != 0) continue;
        k[i] += 1.0;
        if (k[i] >= limit[i]) {
            mask[i] = 1;
            __atomic_fetch_add(finished, 1u, __ATOMIC_RELAXED);
        }
    }
}

void widesum_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const struct fixture_handle* theta = (const struct fixture_handle*)params[1];
    (void)n;
    for (int64_t e = base; e < base + count; ++e) {
        double v = 0.0, w = 1.0;
        for (int64_t c = 0; c < 3; ++c, w *= 10.0) {
            const int64_t at = 3 * e + c;
            v += w * ((const double*)theta->data)[at < (int64_t)theta->samples ? at : (int64_t)theta->samples - 1];
        }
        y[e] = v;
    }
}

/* Wide planes of 3 rows: wo[r][i] = 2 * wi[r][i] + r; y[i] records both handles' extents. */
void widemul_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const struct fixture_handle* wo = (const struct fixture_handle*)params[0];
    const struct fixture_handle* wi = (const struct fixture_handle*)params[1];
    double* y = (double*)((const struct fixture_handle*)params[2])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        for (int r = 0; r < 3; ++r)
            ((double*)wo->data)[r * wo->samples + i] = 2.0 * ((const double*)wi->data)[r * wi->samples + i] + r;
        y[i] = (double)wi->samples + 1000.0 * (double)wo->samples;
    }
}

/* A gathered 2x2 matrix of m columns: y[i] = M00 + 10 M01 + 100 M10 + 1000 M11 at column i % m. */
void gath_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const struct fixture_mirror* m = (const struct fixture_mirror*)params[1];
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        const uint64_t col = (uint64_t)i % m->samples;
        double v = 0.0, w = 1.0;
        for (int c = 0; c < 4; ++c, w *= 10.0) v += w * m->data[c * m->comp_stride + col * m->sample_stride];
        y[i] = v;
    }
}

/* A thread box (4 lanes 3 apart): thread th serves lanes (th / 3) * 12 + th % 3 + 3 j, j < 4, below n: y = 2x + 1. */
int64_t fixture_box_threads;
void boxed_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[1])->data;
    __atomic_fetch_add(&fixture_box_threads, count, __ATOMIC_RELAXED);
    for (int64_t th = base; th < base + count; ++th)
        for (int64_t j = 0; j < 4; ++j) {
            const int64_t lane = (th / 3) * 12 + th % 3 + 3 * j;
            if (lane < n) y[lane] = 2.0 * x[lane] + 1.0;
        }
}

/* A shared parameter plane read flat: y = x * A[0] + A[3]. */
void shr_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[1])->data;
    const double* a = (const double*)((const struct fixture_handle*)params[2])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i) y[i] = x[i] * a[0] + a[3];
}

/* Index sets: thread t serves lane map[t] (lanes) or lane t (prefix): y[lane] = 2 * x[lane] + 1. */
/* thread t serves lane t: tot[t / 4] sums the lanes of its row that ran; params tot, tot_c, x, count */
void prefixsum_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* tot = (double*)((const struct fixture_handle*)params[0])->data;
    double* comp = (double*)((const struct fixture_handle*)params[1])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[2])->data;
    (void)n;
    for (int64_t t = base; t < base + count; ++t) {
        if (t % 4 == 0) tot[t / 4] = 0.0, comp[t / 4] = 0.0;
        tot[t / 4] += x[t];
    }
}

/* rows of 4 samples: tot[r] = min of x, mx[r] = max of x; params tot, mx, x */
void cells_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* tot = (double*)((const struct fixture_handle*)params[0])->data;
    double* mx = (double*)((const struct fixture_handle*)params[1])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[2])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        const int64_t r = i / 4;
        if (i % 4 == 0) tot[r] = 1e300, mx[r] = -1e300;
        if (x[i] < tot[r]) tot[r] = x[i];
        if (x[i] > mx[r]) mx[r] = x[i];
    }
}

/* acc[i / 4] += x[i], acc_c[i / 4] += 0.5; the fold: acc += acc_c, acc_c = 0.125 over the target's slots; params acc, acc_c, x */
void keyed_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* acc = (double*)((const struct fixture_handle*)params[0])->data;
    double* comp = (double*)((const struct fixture_handle*)params[1])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[2])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i) acc[i / 4] += x[i], comp[i / 4] += 0.5;
}

void keyed__fold_acc_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* acc = (double*)((const struct fixture_handle*)params[0])->data;
    double* comp = (double*)((const struct fixture_handle*)params[1])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i) acc[i] += comp[i], comp[i] = 0.125;
}

void lanes_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[1])->data;
    const int32_t* map = (const int32_t*)((const struct fixture_handle*)params[2])->data;
    (void)n;
    for (int64_t t = base; t < base + count; ++t) y[map[t]] = 2.0 * x[map[t]] + 1.0;
}

void prefix_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[1])->data;
    (void)n;
    for (int64_t t = base; t < base + count; ++t) y[t] = 2.0 * x[t] + 1.0;
}

/* v[c] += w[c]*s + table[idx % tlen] (c < 3); o = v0 + v1 + v2 + (mm0 + mm1 + mm2 + mm3) + n. */
void vecs_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const struct fixture_mirror* v = (const struct fixture_mirror*)params[0];
    const struct fixture_mirror* w = (const struct fixture_mirror*)params[1];
    const struct fixture_mirror* mm = (const struct fixture_mirror*)params[2];
    const int32_t* idx = (const int32_t*)((const struct fixture_handle*)params[3])->data;
    const struct fixture_handle* table = (const struct fixture_handle*)params[4];
    const uint32_t nn = *(const uint32_t*)params[5];
    const struct fixture_mirror* o = (const struct fixture_mirror*)params[6];
    const double s = *(const double*)params[7];
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        const double add = ((const double*)table->data)[(uint64_t)idx[i] % table->samples];
        for (int c = 0; c < 3; ++c)
            v->data[c * v->comp_stride + i * v->sample_stride] += w->data[c * w->comp_stride + i * w->sample_stride] * s + add;
        double m = 0.0;
        for (int c = 0; c < 4; ++c) m += mm->data[c * mm->comp_stride + i * mm->sample_stride];
        o->data[i * o->sample_stride] = (v->data[i * v->sample_stride] + v->data[v->comp_stride + i * v->sample_stride] +
                                         v->data[2 * v->comp_stride + i * v->sample_stride]) + m + (double)nn;
    }
}

/* ---- the size block (plugin/gref_layout.h's field order) ------------------ */

/* ---- SEGMENTED groups: slots k, limit, mask, finished, then the group's own; thread t of branch j serves the slot offsets[j] + t ---- */
static void seg_step(void* const* params, int j, int64_t i, double step) {
    double* k = (double*)((const struct fixture_handle*)params[0])->data;
    const double* limit = (const double*)((const struct fixture_handle*)params[1])->data;
    char* mask = (char*)((const struct fixture_handle*)params[2])->data;
    uint32_t* finished = (uint32_t*)((const struct fixture_handle*)params[3])->data;
    (void)j;
    k[i] += step;
    if (k[i] >= limit[i]) {
        mask[i] = 1;
        __atomic_fetch_add(finished, 1u, __ATOMIC_RELAXED);
    }
}

static int is_dead(void* const* params, int64_t i) { return ((const char*)((const struct fixture_handle*)params[2])->data)[i] != 0; }

static double sg_step_of(const double* limit, int64_t i) { return limit[i] >= 6.0 ? 2.0 : 1.0; }

void sg_cls_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const double* limit = (const double*)((const struct fixture_handle*)params[0])->data;
    int64_t* cls = (int64_t*)((const struct fixture_handle*)params[1])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i) cls[i] = limit[i] >= 6.0 ? 1 : 0;
}

static void sg_branch(void* const* params, int j, int64_t base, int64_t count) {
    const int64_t* offsets = (const int64_t*)((const struct fixture_handle*)params[4])->data;
    for (int64_t t = base; t < base + count; ++t) {
        const int64_t i = offsets[j] + t;
        if (i >= offsets[j + 1]) break;
        if (!is_dead(params, i)) seg_step(params, j, i, j + 1.0);
    }
}

void sg_k0_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; sg_branch(params, 0, base, count); }
void sg_k1_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; sg_branch(params, 1, base, count); }

void sgf_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const double* limit = (const double*)((const struct fixture_handle*)params[1])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i)
        if (!is_dead(params, i)) seg_step(params, 0, i, sg_step_of(limit, i));
}


/* sga: slots k, limit, mask, finished, then sga_offsets / sga_mix (4) and how (5); a step adds `path` to how */
static void sga_step(void* const* params, int64_t i, double step, double path) {
    ((double*)((const struct fixture_handle*)params[5])->data)[i] += path;
    seg_step(params, 0, i, step);
}

void sga_cls_host(void* const* params, int64_t base, int64_t count, int64_t n) { sg_cls_host(params, base, count, n); }

static void sga_branch(void* const* params, int j, int64_t base, int64_t count) {
    const int64_t* offsets = (const int64_t*)((const struct fixture_handle*)params[4])->data;
    for (int64_t t = base; t < base + count; ++t) {
        const int64_t i = offsets[j] + t;
        if (i >= offsets[j + 1]) break;
        if (!is_dead(params, i)) sga_step(params, i, j + 1.0, 1.0);
    }
}

void sga_k0_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; sga_branch(params, 0, base, count); }
void sga_k1_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; sga_branch(params, 1, base, count); }

void sga_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const double* limit = (const double*)((const struct fixture_handle*)params[1])->data;
    uint32_t* mix = (uint32_t*)((const struct fixture_handle*)params[4])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i)
        if (!is_dead(params, i)) {
            __atomic_fetch_add(&mix[limit[i] >= 6.0 ? 1 : 0], 1u, __ATOMIC_RELAXED);
            sga_step(params, i, sg_step_of(limit, i), 1000.0);
        }
}

/* dg: slots k, limit, mask, finished, offsets (4), class (5), stale (6), pos (7); the class of a sample is k >= 4 */
static int dg_class(const double* k, int64_t i) { return k[i] >= 4.0 ? 1 : 0; }

static void dg_branch(void* const* params, int j, int64_t base, int64_t count) {
    const double* k = (const double*)((const struct fixture_handle*)params[0])->data;
    const int64_t* offsets = (const int64_t*)((const struct fixture_handle*)params[4])->data;
    int64_t* cls = (int64_t*)((const struct fixture_handle*)params[5])->data;
    char* stale = (char*)((const struct fixture_handle*)params[6])->data;
    double* pos = (double*)((const struct fixture_handle*)params[7])->data;
    for (int64_t t = base; t < base + count; ++t) {
        const int64_t i = offsets[j] + t;
        if (i >= offsets[j + 1]) break;
        if (is_dead(params, i)) continue;
        const int c = dg_class(k, i);
        cls[i] = c;
        stale[i] = c != j;
        pos[i] = (double)i;
        if (c == j) seg_step(params, j, i, 0.125 * (j + 1));
    }
}

void dg_k0_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; dg_branch(params, 0, base, count); }
void dg_k1_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; dg_branch(params, 1, base, count); }

/* the sweep: slots k, limit, mask, finished, active map (4), active count (5); it steps each stale slot by its class */
void dg_sweep_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const double* k = (const double*)((const struct fixture_handle*)params[0])->data;
    const int32_t* map = (const int32_t*)((const struct fixture_handle*)params[4])->data;
    const int64_t live = ((const uint32_t*)((const struct fixture_handle*)params[5])->data)[0];
    const int64_t stop = base + count < live ? base + count : live;
    (void)n;
    for (int64_t t = base; t < stop; ++t) {
        const int64_t i = map[t];
        if (!is_dead(params, i)) seg_step(params, 0, i, 0.125 * (dg_class(k, i) + 1));
    }
}

void dgf_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    const double* k = (const double*)((const struct fixture_handle*)params[0])->data;
    (void)n;
    for (int64_t i = base; i < base + count; ++i)
        if (!is_dead(params, i)) seg_step(params, 0, i, 0.125 * (dg_class(k, i) + 1));
}

#if FIXTURE_BLOCK != 2
EG_FIXTURE_EXPORT extern const char eagle_abi_tag[];
EG_FIXTURE_EXPORT extern const uint64_t eagle_layout_sizes[5];
#if FIXTURE_BLOCK == 3
const char eagle_abi_tag[] = "aether-abi/1";
#else
const char eagle_abi_tag[] = "aether-abi/2";
#endif
/* GRefMirror, ScalarHandle, IntHandle, the index type, the partition triple */
const uint64_t eagle_layout_sizes[5] = {
    40, FIXTURE_BLOCK == 1 ? 48 : sizeof(struct fixture_handle), 32, 4, 3 * sizeof(int64_t) };
#endif

/* ---- the meta ------------------------------------------------------------ */

#define SLOT_OF(var, role_, dtype_, name_)                                     \
    static const struct eg_unit_slot var = {                                   \
        .struct_size = sizeof(struct eg_unit_slot), .flags = 0, .role = role_, \
        .dtype = dtype_, .rank = 0, .like_slot = -1, .n_gather = 0,            \
        .shape = NULL, .gather = NULL, .name = name_ }
#define SLOT(var, role_, name_) SLOT_OF(var, role_, EG_DTYPE_REAL, name_)

SLOT(slot_y, EG_ROLE_MUTABLE, "y");
SLOT(slot_x, EG_ROLE_PER_SAMPLE, "x");
SLOT(slot_a, EG_ROLE_UNIFORM, "a");
SLOT(slot_b, EG_ROLE_UNIFORM, "b");

static const struct eg_unit_slot* const slots[] = { &slot_y, &slot_x, &slot_a, &slot_b };

static const struct eg_unit_body body = {
    .struct_size = sizeof(struct eg_unit_body),
    .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 4,
    .slots = slots,
};

/* Every slot or table index is written: -1 is absent, the zero a C
 * initializer leaves would name slot 0. */
static const struct eg_unit_launch launch = {
    .struct_size = sizeof(struct eg_unit_launch),
    .n_lane_axes = 0,
    .block = 0,
    .active_map_slot = -1,
    .active_count_slot = -1,
    .index_map_slot = -1,
    .index_count_slot = -1,
    .n_plane_axes = 0,
    .group = 1,
};

#define DIGEST(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, \
                    c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c }

static const struct eg_unit_meta unit_axpb = {
    .struct_size = sizeof(struct eg_unit_meta),
    .name = "axpb",
    .digest = DIGEST(AXPB_DIGEST_BYTE),
    .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64,
    .host_entry = "axpb_host",
    .device_image_format = EG_IMAGE_NONE,
    .body = &body,
    .launch = &launch,
};

/* axpb lowered under caller-stated cards (loads as axpb does), and axpb with a flag bit eagle does not know (refused). */
static const struct eg_unit_meta unit_axpb_cards = {
    .struct_size = sizeof(struct eg_unit_meta),
    .flags = EG_UNIT_CALLER_CARDS,
    .name = "axpb_cards",
    .digest = DIGEST(0xD4),
    .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64,
    .host_entry = "axpb_host",
    .device_image_format = EG_IMAGE_NONE,
    .body = &body,
    .launch = &launch,
};

static const struct eg_unit_meta unit_axpb_badflag = {
    .struct_size = sizeof(struct eg_unit_meta),
    .flags = 1u << 1,
    .name = "axpb_badflag",
    .digest = DIGEST(0xD5),
    .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64,
    .host_entry = "axpb_host",
    .device_image_format = EG_IMAGE_NONE,
    .body = &body,
    .launch = &launch,
};

static const struct eg_unit_meta unit_wait = {
    .struct_size = sizeof(struct eg_unit_meta),
    .name = "wait",
    .digest = DIGEST(0xB2),
    .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64,
    .host_entry = "wait_host",
    .device_image_format = EG_IMAGE_NONE,
    .body = &body,
    .launch = &launch,
};

SLOT_OF(slot_cell, EG_ROLE_LOOKUP, EG_DTYPE_INT64, "cell");
static const struct eg_unit_slot* const tick_slots[] = { &slot_cell };
static const struct eg_unit_body tick_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 1, .slots = tick_slots };
static const struct eg_unit_meta unit_tick = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "tick", .digest = DIGEST(0xB8), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "tick_host", .device_image_format = EG_IMAGE_NONE,
    .body = &tick_body, .launch = &launch };

static const struct eg_unit_meta unit_spans = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "spans", .digest = DIGEST(0xB9), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "spans_host", .device_image_format = EG_IMAGE_NONE,
    .body = &tick_body, .launch = &launch };

static const struct eg_unit_ref stage_ref = {
    .struct_size = sizeof(struct eg_unit_ref), .name = "axpb", .digest = DIGEST(0xA1) };
static const struct eg_unit_stage stage0 = {
    .struct_size = sizeof(struct eg_unit_stage), .unit = &stage_ref };
static const struct eg_unit_stage* const stages[] = { &stage0 };

static const struct eg_unit_meta unit_split = {
    .struct_size = sizeof(struct eg_unit_meta),
    .name = "axpb_split",
    .digest = DIGEST(0xC3),
    .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64,
    .host_entry = NULL,
    .device_image_format = EG_IMAGE_NONE,
    .n_stages = 1,
    .stages = stages,
    .body = &body,
    .launch = &launch,
};

SLOT(slot_k, EG_ROLE_MUTABLE, "k");
SLOT(slot_limit, EG_ROLE_PER_SAMPLE, "limit");
SLOT_OF(slot_mask, EG_ROLE_TERMINATED, EG_DTYPE_BOOL, "mask");
SLOT_OF(slot_finished, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "finished");

static const struct eg_unit_slot* const count_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished };

static const struct eg_unit_body count_body = {
    .struct_size = sizeof(struct eg_unit_body),
    .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 4,
    .slots = count_slots,
};

#define FINISH(var, steps_)                                                         \
    static const struct eg_unit_finish var = {                                      \
        .struct_size = sizeof(struct eg_unit_finish), .flags = 0, .mask_slot = 2,   \
        .counter_slot = 3, .steps_slot = -1, .steps = steps_, .steps_max = 0,       \
        .step_ops = -1, .fp64_ops_ir = -1, .host_steps_reverse = NULL,              \
        .device_steps_reverse = NULL }
#define COUNT_UNIT(var, name_, byte, entry, finish_)                     \
    static const struct eg_unit_meta var = {                             \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_,       \
        .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE,              \
        .scalar_type = EG_SCALAR_FLOAT64, .host_entry = entry,           \
        .device_image_format = EG_IMAGE_NONE, .body = &count_body,       \
        .launch = &launch, .finish = &finish_ }

FINISH(finish1, 1);
FINISH(finish3, 3);
COUNT_UNIT(unit_count1, "count1", 0xD1, "count1_host", finish1);
COUNT_UNIT(unit_count3, "count3", 0xD3, "count3_host", finish3);

/* The active-set variants: the map and the live count are LOOKUP slots 4 and 5. */
SLOT_OF(slot_map, EG_ROLE_LOOKUP, EG_DTYPE_INT32, "active_map");
SLOT_OF(slot_live, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "active_count");
static const struct eg_unit_slot* const active_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished,
                                                           &slot_map, &slot_live };
static const struct eg_unit_body active_body = {
    .struct_size = sizeof(struct eg_unit_body),
    .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 6,
    .slots = active_slots,
};
static const struct eg_unit_launch launch_active = {
    .struct_size = sizeof(struct eg_unit_launch),
    .n_lane_axes = 0,
    .block = 0,
    .active_map_slot = 4,
    .active_count_slot = 5,
    .index_map_slot = -1,
    .index_count_slot = -1,
    .n_plane_axes = 0,
    .group = 1,
};
#define ACTIVE_UNIT(var, name_, byte, entry, finish_)                    \
    static const struct eg_unit_meta var = {                             \
        .struct_size = sizeof(struct eg_unit_meta), .name = name_,       \
        .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE,              \
        .scalar_type = EG_SCALAR_FLOAT64, .host_entry = entry,           \
        .device_image_format = EG_IMAGE_NONE, .body = &active_body,      \
        .launch = &launch_active, .finish = &finish_ }
ACTIVE_UNIT(unit_count1a, "count1a", 0xE1, "count1a_host", finish1);
ACTIVE_UNIT(unit_count3a, "count3a", 0xE3, "count3a_host", finish3);

/* steps AUTO: the int64 steps word is LOOKUP slot 4 (up to 64 steps a launch) */
SLOT_OF(slot_steps, EG_ROLE_LOOKUP, EG_DTYPE_INT64, "steps");
static const struct eg_unit_slot* const auto_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_steps };
static const struct eg_unit_slot* const auto_active_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished,
                                                                &slot_steps, &slot_map, &slot_live };
static const struct eg_unit_body auto_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 5, .slots = auto_slots };
static const struct eg_unit_body auto_active_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 7, .slots = auto_active_slots };
static const struct eg_unit_launch launch_auto_active = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, .block = 0, .active_map_slot = 5,
    .active_count_slot = 6, .index_map_slot = -1, .index_count_slot = -1, .n_plane_axes = 0, .group = 1 };
static const struct eg_unit_finish finish_auto = {
    .struct_size = sizeof(struct eg_unit_finish), .flags = 0, .mask_slot = 2, .counter_slot = 3, .steps_slot = 4,
    .steps = EG_STEPS_AUTO, .steps_max = 64, .step_ops = -1, .fp64_ops_ir = -1, .host_steps_reverse = NULL,
    .device_steps_reverse = NULL };
static const struct eg_unit_meta unit_countA = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "countA", .digest = DIGEST(0xF1), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "countA_host", .device_image_format = EG_IMAGE_NONE,
    .body = &auto_body, .launch = &launch, .finish = &finish_auto };
static const struct eg_unit_meta unit_countAa = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "countAa", .digest = DIGEST(0xF2), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "countAa_host", .device_image_format = EG_IMAGE_NONE,
    .body = &auto_active_body, .launch = &launch_auto_active, .finish = &finish_auto };

/* "vecs": rank-1 and rank-2 slots */
static const int64_t shape3[1] = { 3 };
static const int64_t shape22[2] = { 2, 2 };
#define SLOT_R(var, role_, dtype_, rank_, shape_, name_)                       \
    static const struct eg_unit_slot var = {                                   \
        .struct_size = sizeof(struct eg_unit_slot), .flags = 0, .role = role_, \
        .dtype = dtype_, .rank = rank_, .like_slot = -1, .n_gather = 0,        \
        .shape = shape_, .gather = NULL, .name = name_ }
SLOT_R(slot_v, EG_ROLE_MUTABLE, EG_DTYPE_REAL, 1, shape3, "v");
SLOT_R(slot_w, EG_ROLE_VEC_IN, EG_DTYPE_REAL, 1, shape3, "w");
SLOT_R(slot_mm, EG_ROLE_MAT_IN, EG_DTYPE_REAL, 2, shape22, "mm");
SLOT_R(slot_idx, EG_ROLE_PER_SAMPLE, EG_DTYPE_INT32, 0, NULL, "idx");
SLOT_R(slot_table, EG_ROLE_LOOKUP, EG_DTYPE_REAL, 0, NULL, "table");
SLOT_R(slot_nn, EG_ROLE_NSAMPLES, EG_DTYPE_UINT32, 0, NULL, "nn");
SLOT_R(slot_o, EG_ROLE_OUT, EG_DTYPE_REAL, 0, NULL, "o");
static const struct eg_unit_slot* const vecs_slots[] = { &slot_v, &slot_w, &slot_mm, &slot_idx, &slot_table,
                                                         &slot_nn, &slot_o, &slot_a };
static const struct eg_unit_body vecs_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 8, .slots = vecs_slots };
static const struct eg_unit_meta unit_vecs = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "vecs", .digest = DIGEST(0x71), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "vecs_host", .device_image_format = EG_IMAGE_NONE,
    .body = &vecs_body, .launch = &launch };

/* count1 with an int32 mask (plain and with the active set), and count1 carrying EG_FINISH_ENTRY_COUNTS */
SLOT_OF(slot_mask32, EG_ROLE_TERMINATED, EG_DTYPE_INT32, "mask");
static const struct eg_unit_slot* const i_slots[] = { &slot_k, &slot_limit, &slot_mask32, &slot_finished };
static const struct eg_unit_slot* const ia_slots[] = { &slot_k, &slot_limit, &slot_mask32, &slot_finished, &slot_map, &slot_live };
static const struct eg_unit_body i_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 4, .slots = i_slots };
static const struct eg_unit_body ia_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 6, .slots = ia_slots };
static const struct eg_unit_meta unit_count1i = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "count1i", .digest = DIGEST(0x81), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "count1i_host", .device_image_format = EG_IMAGE_NONE,
    .body = &i_body, .launch = &launch, .finish = &finish1 };
static const struct eg_unit_meta unit_count1ia = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "count1ia", .digest = DIGEST(0x82), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "count1ia_host", .device_image_format = EG_IMAGE_NONE,
    .body = &ia_body, .launch = &launch_active, .finish = &finish1 };
static const struct eg_unit_finish finish1e = {
    .struct_size = sizeof(struct eg_unit_finish), .flags = EG_FINISH_ENTRY_COUNTS, .mask_slot = 2, .counter_slot = 3,
    .steps_slot = -1, .steps = 1, .steps_max = 0, .step_ops = -1, .fp64_ops_ir = -1, .host_steps_reverse = NULL,
    .device_steps_reverse = NULL };
static const struct eg_unit_meta unit_count1e = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "count1e", .digest = DIGEST(0x83), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "count1_host", .device_image_format = EG_IMAGE_NONE,
    .body = &count_body, .launch = &launch, .finish = &finish1e };

/* an index set (slots y, x, map, count) and a prefix set (slots y, x, count) */
SLOT_OF(slot_imap, EG_ROLE_LOOKUP, EG_DTYPE_INT32, "index_map");
SLOT_OF(slot_icount, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "index_count");
static const struct eg_unit_slot* const lane_slots[] = { &slot_y, &slot_x, &slot_imap, &slot_icount };
static const struct eg_unit_slot* const prefix_slots[] = { &slot_y, &slot_x, &slot_icount };
static const struct eg_unit_body lane_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 4, .slots = lane_slots };
static const struct eg_unit_body prefix_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 3, .slots = prefix_slots };
static const struct eg_unit_launch launch_lanes = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, .block = 0, .active_map_slot = -1,
    .active_count_slot = -1, .index_map_slot = 2, .index_count_slot = 3, .n_plane_axes = 0, .group = 1 };
static const struct eg_unit_launch launch_prefix = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, .block = 0, .active_map_slot = -1,
    .active_count_slot = -1, .index_map_slot = -1, .index_count_slot = 2, .n_plane_axes = 0, .group = 1 };
static const struct eg_unit_meta unit_lanes = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "lanes", .digest = DIGEST(0x91), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "lanes_host", .device_image_format = EG_IMAGE_NONE,
    .body = &lane_body, .launch = &launch_lanes };
static const struct eg_unit_meta unit_prefix = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "prefix", .digest = DIGEST(0x92), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "prefix_host", .device_image_format = EG_IMAGE_NONE,
    .body = &prefix_body, .launch = &launch_prefix };

/* count1 over groups of 5 lanes with an active set (map and count: LOOKUP slots 4 and 5) */
static const struct eg_unit_launch launch_group = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, .block = 0, .active_map_slot = 4,
    .active_count_slot = 5, .index_map_slot = -1, .index_count_slot = -1, .n_plane_axes = 0, .group = GROUP };
static const struct eg_unit_meta unit_count1g = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "count1g", .digest = DIGEST(0x84), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "count1g_host", .device_image_format = EG_IMAGE_NONE,
    .body = &active_body, .launch = &launch_group, .finish = &finish1 };

/* wide roles: slots wo (WIDE_OUT), wi (WIDE_IN), y (MUTABLE) */
SLOT(slot_wo, EG_ROLE_WIDE_OUT, "wo");
SLOT(slot_wi, EG_ROLE_WIDE_IN, "wi");
static const struct eg_unit_slot* const wide_slots[] = { &slot_wo, &slot_wi, &slot_y };
static const struct eg_unit_body wide_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 3, .slots = wide_slots };
static const struct eg_unit_meta unit_widemul = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "widemul", .digest = DIGEST(0x72), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "widemul_host", .device_image_format = EG_IMAGE_NONE,
    .body = &wide_body, .launch = &launch };

static const struct eg_unit_slot* const widesum_slots[] = { &slot_y, &slot_wi };
static const struct eg_unit_body widesum_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 2, .slots = widesum_slots };
static const struct eg_unit_meta unit_widesum = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "widesum", .digest = DIGEST(0xBB), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "widesum_host", .device_image_format = EG_IMAGE_NONE,
    .body = &widesum_body, .launch = &launch };

/* gather: y (MUTABLE) and a GATHER 2x2 MAT_IN over a two-axis launch (runtime a, static b = 4) */
static const uint8_t keep_b[2] = { 0, 1 }, keep_a[2] = { 1, 0 };
static const int64_t gath_shape[2] = { 2, 2 }, gath_extents[2] = { -1, 4 };
#define GATH_SLOT(var, flags_, n_, pattern_)                                                                 \
    static const struct eg_unit_slot var = {                                                                 \
        .struct_size = sizeof(struct eg_unit_slot), .flags = flags_, .role = EG_ROLE_MAT_IN,                 \
        .dtype = EG_DTYPE_REAL, .rank = 2, .like_slot = -1, .n_gather = n_, .shape = gath_shape,            \
        .gather = pattern_, .name = "M" }
GATH_SLOT(slot_gb, EG_SLOT_GATHER, 2, keep_b);
GATH_SLOT(slot_ga, EG_SLOT_GATHER, 2, keep_a);
GATH_SLOT(slot_gn, EG_SLOT_GATHER, 0, NULL);
static const struct eg_unit_slot* const gb_slots[] = { &slot_y, &slot_gb };
static const struct eg_unit_slot* const ga_slots[] = { &slot_y, &slot_ga };
static const struct eg_unit_slot* const gn_slots[] = { &slot_y, &slot_gn };
#define GATH_BODY(var, slots_) static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body), \
    .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 2, .slots = slots_ }
GATH_BODY(gb_body, gb_slots);
GATH_BODY(ga_body, ga_slots);
GATH_BODY(gn_body, gn_slots);
static const struct eg_unit_launch launch_two_axes = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 2, .block = 0, .active_map_slot = -1,
    .active_count_slot = -1, .index_map_slot = -1, .index_count_slot = -1, .n_plane_axes = 0, .group = 1,
    .lane_extents = gath_extents };
#define GATH_UNIT(var, name_, byte, body_) static const struct eg_unit_meta var = {                            \
    .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE, \
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "gath_host", .device_image_format = EG_IMAGE_NONE,         \
    .body = &body_, .launch = &launch_two_axes }
GATH_UNIT(unit_gathb, "gathb", 0x61, gb_body);
GATH_UNIT(unit_gatha, "gatha", 0x62, ga_body);
GATH_UNIT(unit_gathn, "gathn", 0x63, gn_body);

/* a thread box and a shared parameter */
static const struct eg_unit_box box_4_3 = { .struct_size = sizeof(struct eg_unit_box), .axis = 0, .t = 4, .stride = 3 };
static const struct eg_unit_box box_bad = { .struct_size = sizeof(struct eg_unit_box), .axis = 0, .t = 1, .stride = 3 };
static const struct eg_unit_slot* const xy_slots[] = { &slot_y, &slot_x };
static const struct eg_unit_body xy_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 2, .slots = xy_slots };
#define BOX_LAUNCH(var, box_) static const struct eg_unit_launch var = { .struct_size = sizeof(struct eg_unit_launch),   \
    .n_lane_axes = 0, .block = 0, .active_map_slot = -1, .active_count_slot = -1, .index_map_slot = -1,                \
    .index_count_slot = -1, .n_plane_axes = 0, .group = 1, .host_box = box_ }
BOX_LAUNCH(launch_box, &box_4_3);
BOX_LAUNCH(launch_box_bad, &box_bad);
#define BOX_UNIT(var, name_, byte, launch_) static const struct eg_unit_meta var = {                         \
    .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE, \
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "boxed_host", .device_image_format = EG_IMAGE_NONE,     \
    .body = &xy_body, .launch = &launch_ }
BOX_UNIT(unit_boxed, "boxed", 0x64, launch_box);
BOX_UNIT(unit_boxbad, "boxbad", 0x65, launch_box_bad);
static const struct eg_unit_slot slot_shared = {
    .struct_size = sizeof(struct eg_unit_slot), .flags = EG_SLOT_SHARED, .role = EG_ROLE_LOOKUP, .dtype = EG_DTYPE_REAL,
    .rank = 2, .like_slot = -1, .n_gather = 0, .shape = gath_shape, .gather = NULL, .name = "A" };
static const struct eg_unit_slot* const shr_slots[] = { &slot_y, &slot_x, &slot_shared };
static const struct eg_unit_body shr_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 3, .slots = shr_slots };
static const struct eg_unit_meta unit_shr = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "shr", .digest = DIGEST(0x66), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "shr_host", .device_image_format = EG_IMAGE_NONE,
    .body = &shr_body, .launch = &launch };

/* cells and companions: ACCUM_OUT slots, cell rows of 4 samples, a keyed accumulate over a lane-axis plane axis */
SLOT_OF(slot_tot, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "tot");
SLOT_OF(slot_tot_c, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "tot_comp");
SLOT_OF(slot_mx, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "mx");
SLOT_OF(slot_icount_ps, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "index_count");
SLOT_OF(slot_acc, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "acc");
SLOT_OF(slot_acc_c, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "acc_c");
static const int32_t cell_tot_slots[] = { 0 }, cell_mx_slots[] = { 1 };
static const struct eg_unit_cell cell_tot = { .struct_size = sizeof(struct eg_unit_cell), .op = EG_REDUCE_MIN, .internal = 0,
                                              .columns = 1, .slots = cell_tot_slots, .name = "tot" };
static const struct eg_unit_cell cell_sum = { .struct_size = sizeof(struct eg_unit_cell), .op = EG_REDUCE_SUM, .internal = 0,
                                              .columns = 1, .slots = cell_tot_slots, .name = "tot" };
static const struct eg_unit_cell cell_mx = { .struct_size = sizeof(struct eg_unit_cell), .op = EG_REDUCE_MAX, .internal = 0,
                                             .columns = 1, .slots = cell_mx_slots, .name = "mx" };
static const struct eg_unit_cell* const cell_list[] = { &cell_tot, &cell_mx };
static const struct eg_unit_cell* const cell_sum_list[] = { &cell_sum };
static const struct eg_unit_slot* const cells_slots[] = { &slot_tot, &slot_mx, &slot_x };
static const struct eg_unit_slot* const cellsum_slots[] = { &slot_tot, &slot_tot_c, &slot_mx, &slot_x };
static const struct eg_unit_body cells_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 3, .n_cells = 2, .cell_row = 4, .slots = cells_slots, .cells = cell_list };
static const struct eg_unit_body cellsum_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 4, .n_cells = 1, .cell_row = 4, .slots = cellsum_slots, .cells = cell_sum_list };
static const struct eg_unit_meta unit_cells = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "cells", .digest = DIGEST(0x67), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "cells_host", .device_image_format = EG_IMAGE_NONE,
    .body = &cells_body, .launch = &launch };
static const struct eg_unit_meta unit_cellsum = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "cellsum", .digest = DIGEST(0x6a), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "cells_host", .device_image_format = EG_IMAGE_NONE,
    .body = &cellsum_body, .launch = &launch };
static const int32_t cell_prefix_comp[] = { 1 };
static const struct eg_unit_cell cell_prefix_sum = { .struct_size = sizeof(struct eg_unit_cell), .op = EG_REDUCE_SUM, .internal = 0,
                                                     .columns = 1, .slots = cell_tot_slots, .name = "tot", .comp_slots = cell_prefix_comp };
static const struct eg_unit_cell* const cell_prefix_list[] = { &cell_prefix_sum };
static const struct eg_unit_slot* const prefixsum_slots[] = { &slot_tot, &slot_tot_c, &slot_x, &slot_icount_ps };
static const struct eg_unit_body prefixsum_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL,
    .n_slots = 4, .n_cells = 1, .cell_row = 4, .slots = prefixsum_slots, .cells = cell_prefix_list };
static const struct eg_unit_launch launch_prefixsum = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, .block = 0, .active_map_slot = -1,
    .active_count_slot = -1, .index_map_slot = -1, .index_count_slot = 3, .n_plane_axes = 0, .group = 1 };
static const struct eg_unit_meta unit_prefixsum = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "prefixsum", .digest = DIGEST(0xBC), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "prefixsum_host", .device_image_format = EG_IMAGE_NONE,
    .body = &prefixsum_body, .launch = &launch_prefixsum };
static const struct eg_unit_companion comp_acc = { .struct_size = sizeof(struct eg_unit_companion), .slot = 1, .target_slot = 0,
                                                   .host_fold_entry = "keyed__fold_acc_host", .device_fold_function = NULL };
static const struct eg_unit_companion comp_none = { .struct_size = sizeof(struct eg_unit_companion), .slot = 1, .target_slot = 0,
                                                    .host_fold_entry = NULL, .device_fold_function = NULL };
static const struct eg_unit_companion* const comp_list[] = { &comp_acc };
static const struct eg_unit_companion* const comp_none_list[] = { &comp_none };
static const struct eg_unit_slot* const keyed_slots[] = { &slot_acc, &slot_acc_c, &slot_x };
#define KEYED_BODY(var, list_) static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body),   \
    .access = EG_ACCESS_CROSS_SAMPLE_WRITE, .n_slots = 3, .n_companions = 1, .slots = keyed_slots, .companions = list_ }
KEYED_BODY(keyed_body, comp_list);
KEYED_BODY(keyedbad_body, comp_none_list);
static const char* const key_lane_labels[] = { "key", NULL };
static const int64_t key_lane_extents[] = { -1, 4 };
static const char* const key_axis_labels[] = { "key" };
static const int64_t key_axis_extents[] = { -1 };
static const struct eg_unit_plane_axes key_axes = { .struct_size = sizeof(struct eg_unit_plane_axes), .slot = 0, .n_axes = 1,
                                                    .labels = key_axis_labels, .extents = key_axis_extents };
static const struct eg_unit_plane_axes* const key_axes_list[] = { &key_axes };
static const struct eg_unit_launch launch_keyed = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 2, .block = 0, .active_map_slot = -1, .active_count_slot = -1,
    .index_map_slot = -1, .index_count_slot = -1, .n_plane_axes = 1, .group = 1, .lane_labels = key_lane_labels,
    .lane_extents = key_lane_extents, .plane_axes = key_axes_list };
#define KEYED_UNIT(var, name_, byte, body_) static const struct eg_unit_meta var = {                                \
    .struct_size = sizeof(struct eg_unit_meta), .name = name_, .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE,    \
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "keyed_host", .device_image_format = EG_IMAGE_NONE,             \
    .body = &body_, .launch = &launch_keyed }
KEYED_UNIT(unit_keyed, "keyed", 0x68, keyed_body);
KEYED_UNIT(unit_keyedbad, "keyedbad", 0x69, keyedbad_body);

/* ---- SEGMENTED group units ---- */
SLOT_OF(slot_sg_offsets, EG_ROLE_LOOKUP, EG_DTYPE_INT64, "sg_offsets");
SLOT_OF(slot_sg_class, EG_ROLE_MUTABLE, EG_DTYPE_INT64, "sg_class");
SLOT_OF(slot_dg_offsets, EG_ROLE_LOOKUP, EG_DTYPE_INT64, "dg_offsets");
SLOT_OF(slot_dg_class, EG_ROLE_MUTABLE, EG_DTYPE_INT64, "dg_class");
SLOT_OF(slot_dg_stale, EG_ROLE_MUTABLE, EG_DTYPE_BOOL, "dg_stale");
SLOT(slot_pos, EG_ROLE_MUTABLE, "pos");
static const struct eg_unit_slot* const sg_k_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_sg_offsets };
static const struct eg_unit_slot* const sg_cls_slots[] = { &slot_limit, &slot_sg_class };
static const struct eg_unit_slot* const dg_k_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_dg_offsets,
                                                         &slot_dg_class, &slot_dg_stale, &slot_pos };
static const struct eg_unit_slot* const dg_sweep_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_map, &slot_live };

#define SEG_REF(var, name_, byte) static const struct eg_unit_ref var = { .struct_size = sizeof(struct eg_unit_ref), .name = name_, .digest = DIGEST(byte) }
SEG_REF(ref_sg_cls, "sg__cls", 0xE1);
SEG_REF(ref_sg_k0, "sg__k0", 0xE2);
SEG_REF(ref_sg_k1, "sg__k1", 0xE3);
SEG_REF(ref_dg_k0, "dg__k0", 0xE5);
SEG_REF(ref_dg_k1, "dg__k1", 0xE6);
SEG_REF(ref_dg_sweep, "dg__sweep", 0xE7);
static const struct eg_unit_ref* const sg_branches[] = { &ref_sg_k0, &ref_sg_k1 };
static const struct eg_unit_ref* const dg_branches[] = { &ref_dg_k0, &ref_dg_k1 };

#define DISPATCH(var, strategy_, class_, stale_, offsets_, classify_, name_, branches_, sweep_)                     \
    static const struct eg_unit_dispatch var = { .struct_size = sizeof(struct eg_unit_dispatch), .strategy = strategy_, \
        .mix_slot = -1, .class_slot = class_, .stale_slot = stale_, .offsets_slot = offsets_, .classify = classify_,   \
        .name = name_, .k = 2, .branches = branches_, .sweep = sweep_ }
DISPATCH(disp_sg_cls, EG_DISPATCH_SEGMENTED, 1, -1, -1, NULL, "sg", NULL, NULL);
DISPATCH(disp_sg_branch, EG_DISPATCH_SEGMENTED, -1, -1, 4, &ref_sg_cls, "sg", sg_branches, NULL);
DISPATCH(disp_dg_branch, EG_DISPATCH_SEGMENTED, 5, 6, 4, NULL, "dg", dg_branches, &ref_dg_sweep);
DISPATCH(disp_dg_sweep, EG_DISPATCH_SWITCH, -1, -1, -1, NULL, "dg", NULL, NULL);
static const struct eg_unit_dispatch* const sg_cls_list[] = { &disp_sg_cls };
static const struct eg_unit_dispatch* const sg_branch_list[] = { &disp_sg_branch };
static const struct eg_unit_dispatch* const dg_branch_list[] = { &disp_dg_branch };
static const struct eg_unit_dispatch* const dg_sweep_list[] = { &disp_dg_sweep };

#define SEG_BODY(var, slots_, n_, list_) static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body), \
    .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = n_, .slots = slots_, .n_dispatches = 1, .dispatches = list_ }
SEG_BODY(sg_cls_body, sg_cls_slots, 2, sg_cls_list);
SEG_BODY(sg_branch_body, sg_k_slots, 5, sg_branch_list);
SEG_BODY(dg_branch_body, dg_k_slots, 8, dg_branch_list);
SEG_BODY(dg_sweep_body, dg_sweep_slots, 6, dg_sweep_list);
static const struct eg_unit_launch launch_sweep = {
    .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, .block = 0, .active_map_slot = 4, .active_count_slot = 5,
    .index_map_slot = -1, .index_count_slot = -1, .n_plane_axes = 0, .group = 1 };

#define SEG_UNIT(var, name_, byte, entry, body_, launch_, finish_)                                                    \
    static const struct eg_unit_meta var = { .struct_size = sizeof(struct eg_unit_meta), .name = name_,              \
        .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE, .scalar_type = EG_SCALAR_FLOAT64, .host_entry = entry,   \
        .device_image_format = EG_IMAGE_NONE, .body = &body_, .launch = &launch_, .finish = finish_ }
SEG_UNIT(unit_sg_cls, "sg__cls", 0xE1, "sg_cls_host", sg_cls_body, launch, NULL);
SEG_UNIT(unit_sg_k0, "sg__k0", 0xE2, "sg_k0_host", sg_branch_body, launch, &finish1);
SEG_UNIT(unit_sg_k1, "sg__k1", 0xE3, "sg_k1_host", sg_branch_body, launch, &finish1);
COUNT_UNIT(unit_sgf, "sgf", 0xE4, "sgf_host", finish1);
SEG_UNIT(unit_dg_k0, "dg__k0", 0xE5, "dg_k0_host", dg_branch_body, launch, &finish1);
SEG_UNIT(unit_dg_k1, "dg__k1", 0xE6, "dg_k1_host", dg_branch_body, launch, &finish1);
SEG_UNIT(unit_dg_sweep, "dg__sweep", 0xE7, "dg_sweep_host", dg_sweep_body, launch_sweep, &finish1);
COUNT_UNIT(unit_dgf, "dgf", 0xE8, "dgf_host", finish1);


/* ---- the AUTO group "sga" ---- */
SLOT_OF(slot_sga_offsets, EG_ROLE_LOOKUP, EG_DTYPE_INT64, "sga_offsets");
SLOT_OF(slot_sga_class, EG_ROLE_MUTABLE, EG_DTYPE_INT64, "sga_class");
SLOT_OF(slot_sga_mix, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "sga_mix");
SLOT(slot_how, EG_ROLE_MUTABLE, "how");
static const struct eg_unit_slot* const sga_k_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_sga_offsets, &slot_how };
static const struct eg_unit_slot* const sga_cls_slots[] = { &slot_limit, &slot_sga_class };
static const struct eg_unit_slot* const sga_auto_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_sga_mix, &slot_how };
SEG_REF(ref_sga_cls, "sga__cls", 0xC1);
SEG_REF(ref_sga_k0, "sga__k0", 0xC2);
SEG_REF(ref_sga_k1, "sga__k1", 0xC3);
static const struct eg_unit_ref* const sga_branches[] = { &ref_sga_k0, &ref_sga_k1 };
DISPATCH(disp_sga_cls, EG_DISPATCH_SEGMENTED, 1, -1, -1, NULL, "sga", NULL, NULL);
DISPATCH(disp_sga_branch, EG_DISPATCH_SEGMENTED, -1, -1, 4, &ref_sga_cls, "sga", sga_branches, NULL);
static const struct eg_unit_dispatch* const sga_cls_list[] = { &disp_sga_cls };
static const struct eg_unit_dispatch* const sga_branch_list[] = { &disp_sga_branch };
SEG_BODY(sga_cls_body, sga_cls_slots, 2, sga_cls_list);
SEG_BODY(sga_branch_body, sga_k_slots, 6, sga_branch_list);
SEG_UNIT(unit_sga_cls, "sga__cls", 0xC1, "sga_cls_host", sga_cls_body, launch, NULL);
SEG_UNIT(unit_sga_k0, "sga__k0", 0xC2, "sga_k0_host", sga_branch_body, launch, &finish1);
SEG_UNIT(unit_sga_k1, "sga__k1", 0xC3, "sga_k1_host", sga_branch_body, launch, &finish1);

#define COST(var, ...) static const struct eg_branch_cost var = { sizeof(struct eg_branch_cost), __VA_ARGS__ }
#define AUTO_SET(name_, byte)                                                                                            \
    COST(cost_##name_##_0, AUTO_COST_##name_##_0);                                                                       \
    COST(cost_##name_##_1, AUTO_COST_##name_##_1);                                                                       \
    static const struct eg_branch_cost* const costs_##name_[] = { &cost_##name_##_0, &cost_##name_##_1 };                 \
    static const struct eg_unit_dispatch disp_sga_##name_ = { .struct_size = sizeof(struct eg_unit_dispatch),           \
        .strategy = EG_DISPATCH_AUTO, .mix_slot = 4, .class_slot = -1, .stale_slot = -1, .offsets_slot = -1,           \
        .classify = &ref_sga_cls, .name = "sga", .k = 2, .branches = sga_branches, .branch_costs = costs_##name_,       \
        .cost_flags = AUTO_FLAGS_##name_ };                                                                              \
    static const struct eg_unit_dispatch* const sga_##name_##_list[] = { &disp_sga_##name_ };                            \
    SEG_BODY(sga_##name_##_body, sga_auto_slots, 6, sga_##name_##_list);                                                 \
    SEG_UNIT(unit_sga_##name_, "sga_" #name_, byte, "sga_host", sga_##name_##_body, launch, &finish1)
AUTO_SET(rare, 0xC4);
AUTO_SET(even, 0xC5);
AUTO_SET(scalar, 0xC6);
AUTO_SET(heavy, 0xC7);


/* AUTO records that contradict themselves (eg_unit_load refuses each by name): branches without a classify unit, a classify unit without branches, a sweep,
 * branches without costs, and a segmented set of another group. */
#define BAD_AUTO(var, name_, byte, group_, classify_, branches_, costs_, sweep_)                                          \
    static const struct eg_unit_dispatch disp_##var = { .struct_size = sizeof(struct eg_unit_dispatch),                \
        .strategy = EG_DISPATCH_AUTO, .mix_slot = 4, .class_slot = -1, .stale_slot = -1, .offsets_slot = -1,           \
        .classify = classify_, .name = group_, .k = 2, .branches = branches_, .sweep = sweep_, .branch_costs = costs_ }; \
    static const struct eg_unit_dispatch* const var##_list[] = { &disp_##var };                                           \
    SEG_BODY(var##_body, sga_auto_slots, 6, var##_list);                                                                  \
    SEG_UNIT(unit_##var, name_, byte, "sga_host", var##_body, launch, &finish1)
SEG_REF(ref_sg_cls_a, "sg__cls", 0xE1);
SEG_REF(ref_sg_k0_a, "sg__k0", 0xE2);
SEG_REF(ref_sg_k1_a, "sg__k1", 0xE3);
static const struct eg_unit_ref* const sg_branches_a[] = { &ref_sg_k0_a, &ref_sg_k1_a };
BAD_AUTO(auto_nocls, "auto_nocls", 0xB1, "sga", NULL, sga_branches, costs_rare, NULL);
BAD_AUTO(auto_nobr, "auto_nobr", 0xB2, "sga", &ref_sga_cls, NULL, NULL, NULL);
BAD_AUTO(auto_sweep, "auto_sweep", 0xB3, "sga", &ref_sga_cls, sga_branches, costs_rare, &ref_dg_sweep);
BAD_AUTO(auto_nocost, "auto_nocost", 0xB4, "sga", &ref_sga_cls, sga_branches, NULL, NULL);
BAD_AUTO(auto_group, "auto_group", 0xB5, "sga", &ref_sg_cls_a, sg_branches_a, costs_rare, NULL);


/* AUTO units the runner refuses: one with steps AUTO, one that steps 3 a launch where its segmented set steps 1 (both name the set of "sga"). */
static const struct eg_unit_slot* const sga_steps_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_steps, &slot_sga_mix, &slot_how };
#define OTHER_AUTO(var, name_, byte, slots_, n_, mix_, finish_)                                                           \
    static const struct eg_unit_dispatch disp_##var = { .struct_size = sizeof(struct eg_unit_dispatch),                \
        .strategy = EG_DISPATCH_AUTO, .mix_slot = mix_, .class_slot = -1, .stale_slot = -1, .offsets_slot = -1,         \
        .classify = &ref_sga_cls, .name = "sga", .k = 2, .branches = sga_branches, .branch_costs = costs_rare,           \
        .cost_flags = AUTO_FLAGS_rare };                                                                                 \
    static const struct eg_unit_dispatch* const var##_list[] = { &disp_##var };                                           \
    SEG_BODY(var##_body, slots_, n_, var##_list);                                                                         \
    SEG_UNIT(unit_##var, name_, byte, "sga_host", var##_body, launch_auto_active_none, &finish_)
static const struct eg_unit_launch launch_auto_active_none = { .struct_size = sizeof(struct eg_unit_launch), .group = 1, .active_map_slot = -1,
    .active_count_slot = -1, .index_map_slot = -1, .index_count_slot = -1 };
OTHER_AUTO(auto_steps, "auto_steps", 0xB6, sga_steps_slots, 7, 5, finish_auto);
OTHER_AUTO(auto_k3, "auto_k3", 0xB7, sga_auto_slots, 6, 4, finish3);

/* "mix2": two AUTO groups of no segmented set; their mix slots are 4 and 5 */
SLOT_OF(slot_mix_b, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "mix_b");
static const struct eg_unit_slot* const mix2_slots[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_sga_mix, &slot_mix_b };
#define MIX_GROUP(var, name_, mix_, k_)                                                                                  \
    static const struct eg_unit_dispatch var = { .struct_size = sizeof(struct eg_unit_dispatch), .strategy = EG_DISPATCH_AUTO, \
        .mix_slot = mix_, .class_slot = -1, .stale_slot = -1, .offsets_slot = -1, .name = name_, .k = k_ }
MIX_GROUP(disp_mix_a, "a", 4, 2);
MIX_GROUP(disp_mix_b, "b", 5, 3);
static const struct eg_unit_dispatch* const mix2_list[] = { &disp_mix_a, &disp_mix_b };
static const struct eg_unit_body mix2_body = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 6,
    .slots = mix2_slots, .n_dispatches = 2, .dispatches = mix2_list };
SEG_UNIT(unit_mix2, "mix2", 0xBA, "mix2_host", mix2_body, launch, &finish1);

/* "vecsplit": stage "vecw" writes the width-4 plane "cross", stage "vecr" reads it. */
static const int64_t vec_shape[1] = { VEC_WIDTH };
static const struct eg_unit_slot slot_cross_out = {
    .struct_size = sizeof(struct eg_unit_slot), .role = EG_ROLE_MUTABLE, .dtype = EG_DTYPE_REAL, .rank = 1,
    .like_slot = -1, .shape = vec_shape, .name = "cross" };
static const struct eg_unit_slot slot_cross_in = {
    .struct_size = sizeof(struct eg_unit_slot), .role = EG_ROLE_VEC_IN, .dtype = EG_DTYPE_REAL, .rank = 1,
    .like_slot = -1, .shape = vec_shape, .name = "cross" };
static const struct eg_unit_slot* const vecw_slots[] = { &slot_cross_out, &slot_x };
static const struct eg_unit_slot* const vecr_slots[] = { &slot_y, &slot_cross_in };
static const struct eg_unit_body vecw_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 2, .slots = vecw_slots };
static const struct eg_unit_body vecr_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 2, .slots = vecr_slots };
static const struct eg_unit_meta unit_vecw = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "vecw", .digest = DIGEST(0xE1), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "vecw_host", .device_image_format = EG_IMAGE_NONE,
    .body = &vecw_body, .launch = &launch };
static const struct eg_unit_meta unit_vecr = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "vecr", .digest = DIGEST(0xE2), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "vecr_host", .device_image_format = EG_IMAGE_NONE,
    .body = &vecr_body, .launch = &launch };
static const struct eg_unit_ref vecw_ref = { .struct_size = sizeof(struct eg_unit_ref), .name = "vecw", .digest = DIGEST(0xE1) };
static const struct eg_unit_ref vecr_ref = { .struct_size = sizeof(struct eg_unit_ref), .name = "vecr", .digest = DIGEST(0xE2) };
static const struct eg_unit_stage vec_stage0 = { .struct_size = sizeof(struct eg_unit_stage), .unit = &vecw_ref };
static const struct eg_unit_stage vec_stage1 = { .struct_size = sizeof(struct eg_unit_stage), .unit = &vecr_ref };
static const struct eg_unit_stage* const vec_stages[] = { &vec_stage0, &vec_stage1 };
static const struct eg_unit_team_plane vec_cross = {
    .struct_size = sizeof(struct eg_unit_team_plane), .dtype = EG_DTYPE_REAL, .unit = EG_UNIT_LANE,
    .layout = EG_LAYOUT_PLANE_MAJOR, .wide = EG_WIDE_NONE, .keyed = -1, .target_slot = -1, .per_lane = VEC_WIDTH,
    .name = "cross", .target_stage = -1 };
static const struct eg_unit_team_plane* const vec_planes[] = { &vec_cross };
static const struct eg_unit_slot* const vecsplit_slots[] = { &slot_y, &slot_x };
static const struct eg_unit_body vecsplit_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 2, .n_team_planes = 1,
    .slots = vecsplit_slots, .team_planes = vec_planes };
static const struct eg_unit_meta unit_vecsplit = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "vecsplit", .digest = DIGEST(0xE3), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = NULL, .device_image_format = EG_IMAGE_NONE,
    .n_stages = 2, .stages = vec_stages, .body = &vecsplit_body, .launch = &launch };

static const struct eg_unit_slot* const actrow_slots[] = { &slot_k, &slot_map, &slot_live };
static const struct eg_unit_body actrow_body = {
    .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = 3, .slots = actrow_slots };
static const struct eg_unit_launch launch_actrow = {
    .struct_size = sizeof(struct eg_unit_launch), .active_map_slot = 1, .active_count_slot = 2, .index_map_slot = -1,
    .index_count_slot = -1, .group = 1 };
static const struct eg_unit_meta unit_actrow = {
    .struct_size = sizeof(struct eg_unit_meta), .name = "actrow", .digest = DIGEST(0xE4), .pattern = EG_PATTERN_PURE,
    .scalar_type = EG_SCALAR_FLOAT64, .host_entry = "actrow_host", .device_image_format = EG_IMAGE_NONE,
    .body = &actrow_body, .launch = &launch_actrow };

static const struct eg_unit_meta* const units[] = { &unit_axpb, &unit_wait, &unit_tick, &unit_spans, &unit_prefixsum, &unit_widesum, &unit_mix2, &unit_split, &unit_count1,
                                                    &unit_count3, &unit_count1a, &unit_count3a, &unit_countA, &unit_countAa, &unit_vecs, &unit_count1i,
                                                    &unit_count1ia, &unit_count1e, &unit_lanes, &unit_prefix, &unit_count1g, &unit_widemul,
                                                    &unit_gathb, &unit_gatha, &unit_gathn, &unit_boxed, &unit_boxbad, &unit_shr, &unit_cells, &unit_keyed,
                                                    &unit_keyedbad, &unit_cellsum, &unit_axpb_cards, &unit_axpb_badflag,
                                                    &unit_sg_cls, &unit_sg_k0, &unit_sg_k1, &unit_sgf, &unit_dg_k0, &unit_dg_k1, &unit_dg_sweep, &unit_dgf,
                                                    &unit_sga_cls, &unit_sga_k0, &unit_sga_k1, &unit_sga_rare, &unit_sga_even, &unit_sga_scalar, &unit_sga_heavy,
                                                    &unit_auto_nocls, &unit_auto_nobr, &unit_auto_sweep, &unit_auto_nocost, &unit_auto_group, &unit_auto_steps, &unit_auto_k3,
                                                    &unit_vecw, &unit_vecr, &unit_vecsplit, &unit_actrow };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta),
    .schema_major = EG_META_SCHEMA_MAJOR,
    .schema_minor = 0,
    .n_units = 60,
    .units = units,
};

const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
