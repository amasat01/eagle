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

/* ---- the meta ------------------------------------------------------------ */

#define SLOT(var, role_, name_)                                                \
    static const struct eg_unit_slot var = {                                   \
        .struct_size = sizeof(struct eg_unit_slot), .flags = 0, .role = role_, \
        .dtype = EG_DTYPE_REAL, .rank = 0, .like_slot = -1, .n_gather = 0,     \
        .shape = NULL, .gather = NULL, .name = name_ }

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

static const struct eg_unit_meta* const units[] = { &unit_axpb, &unit_wait, &unit_split };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta),
    .schema_major = EG_META_SCHEMA_MAJOR,
    .schema_minor = 0,
    .n_units = 3,
    .units = units,
};

const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
