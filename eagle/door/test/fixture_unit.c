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
EG_FIXTURE_EXPORT const struct eg_library_meta* eg_unit_library_meta(void);

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
static void count_steps(void* const* params, int64_t base, int64_t count, int steps, const int32_t* map) {
    double* k = (double*)((const struct fixture_handle*)params[0])->data;
    const double* limit = (const double*)((const struct fixture_handle*)params[1])->data;
    unsigned char* mask = (unsigned char*)((const struct fixture_handle*)params[2])->data;
    uint32_t* finished = (uint32_t*)((const struct fixture_handle*)params[3])->data;
    for (int64_t t = base; t < base + count; ++t) {
        const int64_t i = map ? map[t] : t;
        for (int s = 0; s < steps && !mask[i]; ++s) {
            k[i] += 1.0;
            if (k[i] >= limit[i]) {
                mask[i] = 1;
                __atomic_fetch_add(finished, 1u, __ATOMIC_RELAXED);
            }
        }
    }
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

static const struct eg_unit_meta* const units[] = { &unit_axpb, &unit_wait, &unit_split, &unit_count1,
                                                    &unit_count3, &unit_count1a, &unit_count3a };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta),
    .schema_major = EG_META_SCHEMA_MAJOR,
    .schema_minor = 0,
    .n_units = 7,
    .units = units,
};

const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
