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
 *    once the test sets y[0] (or after a few seconds): it holds a context busy.
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
 *  - "count1", "count3": a finishing unit that steps k += 1 until k >= limit,
 *    then marks the sample and counts it into the finished counter; one or
 *    three steps per launch (slots: k, limit, mask, the counter).
 *
 * The size block beside the host entries (``eagle_abi_tag``,
 * ``eagle_layout_sizes``) follows FIXTURE_BLOCK: 0 (default) the real one,
 * 1 a wrong sizeof(ScalarHandle), 2 none exported, 3 the v1 tag.
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"

#include <stdint.h>
#include <time.h>

#define EG_FIXTURE_EXPORT __attribute__((visibility("default")))

/* The scalar handle of the v2 host ABI (plugin/gref_layout.h, 32 bytes). */
struct fixture_handle {
    void* data;
    uint64_t samples;
    uint64_t stride;
    int32_t device_type;
    int32_t device_id;
};

EG_FIXTURE_EXPORT void axpb_host(void* const* params, int64_t base, int64_t count, int64_t n);
EG_FIXTURE_EXPORT void wait_host(void* const* params, int64_t base, int64_t count, int64_t n);
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
EG_FIXTURE_EXPORT const struct eg_library_meta* eg_unit_library_meta(void);

/* Lanes of a partial last group a count1g launch was handed (the entry skips them). */
EG_FIXTURE_EXPORT int64_t fixture_past_n;
int64_t fixture_past_n;

void axpb_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* y = (double*)((const struct fixture_handle*)params[0])->data;
    const double* x = (const double*)((const struct fixture_handle*)params[1])->data;
    const double a = *(const double*)params[2];
    const double b = *(const double*)params[3];
    (void)n;
    for (int64_t i = base; i < base + count; ++i) y[i] = a * x[i] + b;
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

/* Index sets: thread t serves lane map[t] (lanes) or lane t (prefix): y[lane] = 2 * x[lane] + 1. */
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

/* The 40-byte mirror of a (components, n) plane (plugin/gref_layout.h's GRefMirror). */
struct fixture_mirror {
    double* data;
    uint64_t samples, comp_stride, sample_stride;
    int32_t device_type, device_id;
};

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

#ifndef FIXTURE_BLOCK
#define FIXTURE_BLOCK 0
#endif

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
    .digest = DIGEST(0xA1),
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

static const struct eg_unit_meta* const units[] = { &unit_axpb, &unit_wait, &unit_split, &unit_count1,
                                                    &unit_count3, &unit_count1a, &unit_count3a, &unit_countA, &unit_countAa, &unit_vecs, &unit_count1i,
                                                    &unit_count1ia, &unit_count1e, &unit_lanes, &unit_prefix, &unit_count1g, &unit_widemul };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta),
    .schema_major = EG_META_SCHEMA_MAJOR,
    .schema_minor = 0,
    .n_units = 17,
    .units = units,
};

const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
