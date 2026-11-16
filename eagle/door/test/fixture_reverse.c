// SPDX-License-Identifier: Apache-2.0
/*
 * A unit library for eg_reverse with gradients known in closed form: the step
 * x <- a*x + b, t <- t + 1, each sample finishing once t reaches its own T;
 * its one-step taping twins (windows 1, 3 and 16, each built into the twin,
 * taping t and x) and its step vjp. The derived records are written by hand,
 * as the library generator will emit them; the broken ones below each break
 * one fact eg_runner_new checks.
 *
 * Table: 0 lin, 1-3 lin twins (W 1, 3, 16), 4 lin_vjp, 5 lina (lin reading an
 * active set), then the broken records.
 */
#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"

#include <stdint.h>

#define EXPORT __attribute__((visibility("default")))

struct handle {
    void* data;
    uint64_t samples;
    uint64_t stride;
    int32_t device_type;
    int32_t device_id;
};
struct mirror {
    double* data;
    uint64_t samples, comp_stride, sample_stride;
    int32_t device_type, device_id;
};

#define H(i, type) ((type*)((const struct handle*)params[i])->data)
#define U(i, type) (*(const type*)params[i])

EXPORT void lin_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void lin_tape1_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void lin_tape3_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void lin_tape16_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void lin_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT const struct eg_library_meta* eg_unit_library_meta(void);

/* slots: x, t, terminated, finished_count, T, a, b */
void lin_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* x = H(0, double);
    double* t = H(1, double);
    uint8_t* done = H(2, uint8_t);
    uint32_t* finished = H(3, uint32_t);
    const double* T = H(4, double);
    const double a = U(5, double), b = U(6, double);
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        if (done[i]) continue;
        x[i] = a * x[i] + b;
        t[i] = t[i] + 1.0;
        if (t[i] >= T[i]) {
            done[i] = 1;
            __atomic_fetch_add(finished, 1u, __ATOMIC_RELAXED);
        }
    }
}

/* slots: applied_steps, x, t, tape (window * 2 rows: t, x), T, finished_count, terminated, a, b, tape_on */
static void lin_tape(void* const* params, int64_t base, int64_t count, int64_t window) {
    int64_t* steps = H(0, int64_t);
    double* x = H(1, double);
    double* t = H(2, double);
    const struct mirror* tape = (const struct mirror*)params[3];
    const double* T = H(4, double);
    uint32_t* finished = H(5, uint32_t);
    uint8_t* done = H(6, uint8_t);
    const double a = U(7, double), b = U(8, double);
    const int64_t on = U(9, int64_t);
    for (int64_t i = base; i < base + count; ++i) {
        if (done[i]) continue;
        if (on != 0) {
            const int64_t row = steps[i] % window;
            tape->data[(uint64_t)(row * 2 + 0) * tape->comp_stride + (uint64_t)i] = t[i];
            tape->data[(uint64_t)(row * 2 + 1) * tape->comp_stride + (uint64_t)i] = x[i];
        }
        x[i] = a * x[i] + b;
        t[i] = t[i] + 1.0;
        steps[i] = steps[i] + 1;
        if (t[i] >= T[i]) {
            done[i] = 1;
            __atomic_fetch_add(finished, 1u, __ATOMIC_RELAXED);
        }
    }
}
void lin_tape1_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; lin_tape(params, base, count, 1); }
void lin_tape3_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; lin_tape(params, base, count, 3); }
void lin_tape16_host(void* const* params, int64_t base, int64_t count, int64_t n) { (void)n; lin_tape(params, base, count, 16); }

/* slots: bar_a, bar_b, bar_x, bar_t, bar_x_next, bar_t_next, x (taped), a, b */
void lin_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    double* ga = H(0, double);
    double* gb = H(1, double);
    double* gx = H(2, double);
    double* gt = H(3, double);
    const double* gxn = H(4, double);
    const double* gtn = H(5, double);
    const double* x = H(6, double);
    const double a = U(7, double);
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        ga[i] = x[i] * gxn[i];
        gb[i] = gxn[i];
        gx[i] = a * gxn[i];
        gt[i] = gtn[i];
    }
}

/* ---- the size block ---- */
EXPORT extern const char eagle_abi_tag[];
EXPORT extern const uint64_t eagle_layout_sizes[5];
const char eagle_abi_tag[] = "aether-abi/2";
const uint64_t eagle_layout_sizes[5] = { 40, sizeof(struct handle), 32, 4, 3 * sizeof(int64_t) };

/* ---- the meta ---- */
#define SLOT_R(var, role_, dtype_, name_, rank_, shape_)                                                       \
    static const struct eg_unit_slot var = { .struct_size = sizeof(struct eg_unit_slot), .flags = 0,         \
                                             .role = role_, .dtype = dtype_, .rank = rank_, .like_slot = -1, \
                                             .n_gather = 0, .shape = shape_, .gather = NULL, .name = name_ }
#define SLOT(var, role_, dtype_, name_) SLOT_R(var, role_, dtype_, name_, 0, NULL)

SLOT(s_x, EG_ROLE_MUTABLE, EG_DTYPE_REAL, "x");
SLOT(s_t, EG_ROLE_MUTABLE, EG_DTYPE_REAL, "t");
SLOT(s_done, EG_ROLE_TERMINATED, EG_DTYPE_BOOL, "terminated");
SLOT(s_fin, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "finished_count");
SLOT(s_T, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, "T");
SLOT(s_a, EG_ROLE_UNIFORM, EG_DTYPE_REAL, "a");
SLOT(s_b, EG_ROLE_UNIFORM, EG_DTYPE_REAL, "b");
SLOT(s_steps, EG_ROLE_MUTABLE, EG_DTYPE_INT64, "applied_steps");
SLOT(s_on, EG_ROLE_UNIFORM, EG_DTYPE_INT64, "tape_on");
SLOT(s_map, EG_ROLE_LOOKUP, EG_DTYPE_INT32, "active_map");
SLOT(s_live, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "active_count");
static const int64_t tape1_shape[] = { 2 }, tape3_shape[] = { 6 }, tape16_shape[] = { 32 };
SLOT_R(s_tape1, EG_ROLE_OUT, EG_DTYPE_REAL, "tape", 1, tape1_shape);
SLOT_R(s_tape3, EG_ROLE_OUT, EG_DTYPE_REAL, "tape", 1, tape3_shape);
SLOT_R(s_tape16, EG_ROLE_OUT, EG_DTYPE_REAL, "tape", 1, tape16_shape);
SLOT(s_ga, EG_ROLE_MUTABLE, EG_DTYPE_REAL, "bar_a");
SLOT(s_gb, EG_ROLE_MUTABLE, EG_DTYPE_REAL, "bar_b");
SLOT(s_gx, EG_ROLE_MUTABLE, EG_DTYPE_REAL, "bar_x");
SLOT(s_gt, EG_ROLE_MUTABLE, EG_DTYPE_REAL, "bar_t");
SLOT(s_gxn, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, "bar_x_next");
SLOT(s_gtn, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, "bar_t_next");
SLOT(s_xr, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, "x");

static const struct eg_unit_slot* const lin_slots[] = { &s_x, &s_t, &s_done, &s_fin, &s_T, &s_a, &s_b };
static const struct eg_unit_slot* const lina_slots[] = { &s_x, &s_t, &s_done, &s_fin, &s_T, &s_a, &s_b, &s_map, &s_live };
#define TWIN_SLOTS(var, tape_) \
    static const struct eg_unit_slot* const var[] = { &s_steps, &s_x, &s_t, &tape_, &s_T, &s_fin, &s_done, &s_a, &s_b, &s_on }
TWIN_SLOTS(tw1_slots, s_tape1);
TWIN_SLOTS(tw3_slots, s_tape3);
TWIN_SLOTS(tw16_slots, s_tape16);
static const struct eg_unit_slot* const vjp_slots[] = { &s_ga, &s_gb, &s_gx, &s_gt, &s_gxn, &s_gtn, &s_xr, &s_a, &s_b };

#define BODY(var, slots_, n_) \
    static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = n_, .slots = slots_ }
BODY(lin_body, lin_slots, 7);
BODY(lina_body, lina_slots, 9);
BODY(tw1_body, tw1_slots, 10);
BODY(tw3_body, tw3_slots, 10);
BODY(tw16_body, tw16_slots, 10);
BODY(vjp_body, vjp_slots, 9);

#define LAUNCH(var, map_, live_)                                                                               \
    static const struct eg_unit_launch var = { .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, \
                                               .block = 0, .active_map_slot = map_, .active_count_slot = live_, \
                                               .index_map_slot = -1, .index_count_slot = -1, .n_plane_axes = 0, .group = 1 }
LAUNCH(launch, -1, -1);
LAUNCH(launch_active, 7, 8);

#define FINISH(var, mask_, counter_)                                                                            \
    static const struct eg_unit_finish var = { .struct_size = sizeof(struct eg_unit_finish), .flags = 0,        \
                                               .mask_slot = mask_, .counter_slot = counter_, .steps_slot = -1,  \
                                               .steps = 1, .steps_max = 0, .step_ops = -1, .fp64_ops_ir = -1,   \
                                               .host_steps_reverse = NULL, .device_steps_reverse = NULL }
FINISH(lin_finish, 2, 3);
FINISH(twin_finish, 6, 5);

/* The twin's links: each ordinary slot to the primal slot of the same name. */
#define LINK(var, slot_, source_, of_, reduced_) \
    static const struct eg_unit_link var = { .struct_size = sizeof(struct eg_unit_link), .slot = slot_, .source = source_, .of_slot = of_, .reduced = reduced_ }
LINK(tl_x, 1, EG_SRC_PRIMAL, 0, 0);
LINK(tl_t, 2, EG_SRC_PRIMAL, 1, 0);
LINK(tl_T, 4, EG_SRC_PRIMAL, 4, 0);
LINK(tl_done, 6, EG_SRC_PRIMAL, 2, 0);
LINK(tl_a, 7, EG_SRC_PRIMAL, 5, 0);
LINK(tl_b, 8, EG_SRC_PRIMAL, 6, 0);
LINK(tl_bad_slot, 12, EG_SRC_PRIMAL, 0, 0);
LINK(tl_time0, 4, EG_SRC_TIME0, 1, 0);
static const struct eg_unit_link* const twin_links[] = { &tl_x, &tl_t, &tl_T, &tl_done, &tl_a, &tl_b };
static const struct eg_unit_link* const twin_links_bad_slot[] = { &tl_x, &tl_t, &tl_T, &tl_done, &tl_a, &tl_bad_slot };
static const struct eg_unit_link* const twin_links_time0[] = { &tl_x, &tl_t, &tl_time0, &tl_done, &tl_a, &tl_b };

/* The tape rows of one step: t in row 0, x in row 1. */
static const struct eg_unit_taped tp_t = { .struct_size = sizeof(struct eg_unit_taped), .slot = 1, .row0 = 0, .rows = 1 };
static const struct eg_unit_taped tp_x = { .struct_size = sizeof(struct eg_unit_taped), .slot = 0, .row0 = 1, .rows = 1 };
static const struct eg_unit_taped* const taped[] = { &tp_t, &tp_x };

/* The vjp's links. */
LINK(vl_ga, 0, EG_SRC_PARAM_GRAD, 5, 1);
LINK(vl_gb, 1, EG_SRC_PARAM_GRAD, 6, 1);
LINK(vl_gx, 2, EG_SRC_STATE_ADJOINT_OUT, 0, 0);
LINK(vl_gt, 3, EG_SRC_STATE_ADJOINT_OUT, 1, 0);
LINK(vl_gxn, 4, EG_SRC_STATE_ADJOINT_IN, 0, 0);
LINK(vl_gtn, 5, EG_SRC_STATE_ADJOINT_IN, 1, 0);
LINK(vl_x, 6, EG_SRC_TAPED, 0, 0);
LINK(vl_a, 7, EG_SRC_PRIMAL, 5, 0);
LINK(vl_b, 8, EG_SRC_PRIMAL, 6, 0);
LINK(vl_x_reduced, 6, EG_SRC_TAPED, 0, 1);
LINK(vl_x_steps, 6, EG_SRC_STEPS, -1, 0);
LINK(vl_ga_unreduced, 0, EG_SRC_PARAM_GRAD, 5, 0);
static const struct eg_unit_link* const vjp_links[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a, &vl_b };
static const struct eg_unit_link* const vjp_links_missing[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a };
static const struct eg_unit_link* const vjp_links_twice[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a, &vl_a };
static const struct eg_unit_link* const vjp_links_reduced[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x_reduced, &vl_a, &vl_b };
static const struct eg_unit_link* const vjp_links_steps[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x_steps, &vl_a, &vl_b };
static const struct eg_unit_link* const vjp_links_unreduced[] = { &vl_ga_unreduced, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a, &vl_b };

#define TWIN_DERIVED(var, of_, rows_, links_)                                                                          \
    static const struct eg_unit_derived var = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_TAPE, \
                                                .of_unit = of_, .tape_slot = 3, .n_taped = 2, .tape_rows = rows_,     \
                                                .steps_slot = 0, .tape_on_slot = 9, .time0_slot = -1, .tau_slot = -1, \
                                                .fused_steps_slot = -1, .n_links = 6, .n_refined = 0, .taped = taped, \
                                                .links = links_, .refined = NULL }
TWIN_DERIVED(d_twin, 0, 2, twin_links);
TWIN_DERIVED(d_twin_of, 4, 2, twin_links);
TWIN_DERIVED(d_twin_rows, 0, 1, twin_links);
TWIN_DERIVED(d_twin_slot, 0, 2, twin_links_bad_slot);
TWIN_DERIVED(d_twin_time0, 0, 2, twin_links_time0);
#define VJP_DERIVED(var, n_, links_)                                                                                       \
    static const struct eg_unit_derived var = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_STEP_VJP, \
                                                .of_unit = 0, .tape_slot = -1, .n_taped = 0, .tape_rows = 0,              \
                                                .steps_slot = -1, .tape_on_slot = -1, .time0_slot = -1, .tau_slot = -1,   \
                                                .fused_steps_slot = -1, .n_links = n_, .n_refined = 0, .taped = NULL,     \
                                                .links = links_, .refined = NULL }
VJP_DERIVED(d_vjp, 9, vjp_links);
VJP_DERIVED(d_vjp_missing, 8, vjp_links_missing);
VJP_DERIVED(d_vjp_twice, 9, vjp_links_twice);
VJP_DERIVED(d_vjp_reduced, 9, vjp_links_reduced);
VJP_DERIVED(d_vjp_steps, 9, vjp_links_steps);
VJP_DERIVED(d_vjp_unreduced, 9, vjp_links_unreduced);

#define DIGEST(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c }
#define UNIT(var, name_, byte, entry, body_, launch_, finish_, derived_)                                              \
    static const struct eg_unit_meta var = { .struct_size = sizeof(struct eg_unit_meta), .name = name_,             \
                                             .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE,                    \
                                             .scalar_type = EG_SCALAR_FLOAT64, .host_entry = entry,                 \
                                             .device_image_format = EG_IMAGE_NONE, .body = &body_, .launch = &launch_, \
                                             .finish = finish_, .derived = derived_ }
UNIT(u_lin, "lin", 0xE0, "lin_host", lin_body, launch, &lin_finish, NULL);
UNIT(u_tw1, "lin", 0xE1, "lin_tape1_host", tw1_body, launch, &twin_finish, &d_twin);
UNIT(u_tw3, "lin", 0xE3, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin);
UNIT(u_tw16, "lin", 0xE6, "lin_tape16_host", tw16_body, launch, &twin_finish, &d_twin);
UNIT(u_vjp, "lin_vjp", 0xE4, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp);
UNIT(u_lina, "lina", 0xE5, "lin_host", lina_body, launch_active, &lin_finish, NULL);
UNIT(u_tw_of, "lin", 0xF0, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_of);
UNIT(u_tw_rows, "lin", 0xF1, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_rows);
UNIT(u_tw_slot, "lin", 0xF2, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_slot);
UNIT(u_tw_time0, "lin", 0xF3, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_time0);
UNIT(u_vjp_missing, "lin_vjp", 0xF4, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_missing);
UNIT(u_vjp_twice, "lin_vjp", 0xF5, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_twice);
UNIT(u_vjp_reduced, "lin_vjp", 0xF6, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_reduced);
UNIT(u_vjp_steps, "lin_vjp", 0xF7, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_steps);
UNIT(u_vjp_unreduced, "lin_vjp", 0xF8, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_unreduced);

static const struct eg_unit_meta* const units[] = { &u_lin, &u_tw1, &u_tw3, &u_tw16, &u_vjp, &u_lina, &u_tw_of, &u_tw_rows,
                                                    &u_tw_slot, &u_tw_time0, &u_vjp_missing, &u_vjp_twice, &u_vjp_reduced,
                                                    &u_vjp_steps, &u_vjp_unreduced };
static const struct eg_library_meta library = { .struct_size = sizeof(struct eg_library_meta), .schema_major = EG_META_SCHEMA_MAJOR,
                                                .schema_minor = 0, .n_units = 15, .units = units };
const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
