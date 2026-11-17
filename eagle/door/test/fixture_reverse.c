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
EXPORT void lin3_host(void* const* params, int64_t base, int64_t count, int64_t n);
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

/* The primal of the K = 3 units: three steps of lin a launch. */
void lin3_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    for (int step = 0; step < 3; ++step) lin_host(params, base, count, n);
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
#define LINK(var, slot_, what_, form_, of_, reduced_) \
    static const struct eg_unit_link var = { .struct_size = sizeof(struct eg_unit_link), .slot = slot_, .what = EG_WHAT_##what_, \
                                             .of_slot = of_, .reduced = reduced_, .form = EG_FORM_##form_ }
LINK(tl_x, 1, PRIMAL_SLOT, VALUE, 0, 0);
LINK(tl_t, 2, PRIMAL_SLOT, VALUE, 1, 0);
LINK(tl_T, 4, PRIMAL_SLOT, VALUE, 4, 0);
LINK(tl_done, 6, PRIMAL_SLOT, VALUE, 2, 0);
LINK(tl_a, 7, PRIMAL_SLOT, VALUE, 5, 0);
LINK(tl_b, 8, PRIMAL_SLOT, VALUE, 6, 0);
LINK(tl_bad_slot, 12, PRIMAL_SLOT, VALUE, 0, 0);
LINK(tl_time0, 4, TIME0, VALUE, 1, 0);
static const struct eg_unit_link* const twin_links[] = { &tl_x, &tl_t, &tl_T, &tl_done, &tl_a, &tl_b };
static const struct eg_unit_link* const twin_links_bad_slot[] = { &tl_x, &tl_t, &tl_T, &tl_done, &tl_a, &tl_bad_slot };
static const struct eg_unit_link* const twin_links_time0[] = { &tl_x, &tl_t, &tl_time0, &tl_done, &tl_a, &tl_b };

/* The tape rows of one step: t in row 0, x in row 1. */
static const struct eg_unit_taped tp_t = { .struct_size = sizeof(struct eg_unit_taped), .slot = 1, .row0 = 0, .rows = 1 };
static const struct eg_unit_taped tp_x = { .struct_size = sizeof(struct eg_unit_taped), .slot = 0, .row0 = 1, .rows = 1 };
static const struct eg_unit_taped* const taped[] = { &tp_t, &tp_x };

/* The vjp's links. */
LINK(vl_ga, 0, PRIMAL_SLOT, GRAD, 5, 1);
LINK(vl_gb, 1, PRIMAL_SLOT, GRAD, 6, 1);
LINK(vl_gx, 2, PRIMAL_SLOT, ADJOINT_OUT, 0, 0);
LINK(vl_gt, 3, PRIMAL_SLOT, ADJOINT_OUT, 1, 0);
LINK(vl_gxn, 4, PRIMAL_SLOT, ADJOINT_IN, 0, 0);
LINK(vl_gtn, 5, PRIMAL_SLOT, ADJOINT_IN, 1, 0);
LINK(vl_x, 6, TAPE_ROW, VALUE, 0, 0);
LINK(vl_a, 7, PRIMAL_SLOT, VALUE, 5, 0);
LINK(vl_b, 8, PRIMAL_SLOT, VALUE, 6, 0);
LINK(vl_x_reduced, 6, TAPE_ROW, VALUE, 0, 1);
LINK(vl_x_steps, 6, STEPS, VALUE, -1, 0);
LINK(vl_ga_unreduced, 0, PRIMAL_SLOT, GRAD, 5, 0);
LINK(vl_x_badform, 6, TAPE_ROW, GRAD, 0, 0);   /* (TAPE_ROW, GRAD) is not a row of EG_LINK_RULES */
LINK(nl_h1, 7, TIME_STEP, VALUE, 5, 0);
LINK(nl_h2, 8, TIME_STEP, VALUE, 6, 0);
static const struct eg_unit_link* const vjp_links[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a, &vl_b };
static const struct eg_unit_link* const vjp_links_missing[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a };
static const struct eg_unit_link* const vjp_links_twice[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a, &vl_a };
static const struct eg_unit_link* const vjp_links_reduced[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x_reduced, &vl_a, &vl_b };
static const struct eg_unit_link* const vjp_links_steps[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x_steps, &vl_a, &vl_b };
static const struct eg_unit_link* const vjp_links_badform[] = { &vl_ga, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x_badform, &vl_a, &vl_b };
static const struct eg_unit_link* const newton_links_two_steps[] = { &nl_h1, &nl_h2 };
static const struct eg_unit_link* const vjp_links_unreduced[] = { &vl_ga_unreduced, &vl_gb, &vl_gx, &vl_gt, &vl_gxn, &vl_gtn, &vl_x, &vl_a, &vl_b };

#define TWIN_DERIVED(var, of_, rows_, links_)                                                                          \
    static const struct eg_unit_derived var = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_TAPE, \
                                                .of_unit = of_, .tape_slot = 3, .n_taped = 2, .tape_rows = rows_,     \
                                                .steps_slot = 0, .tape_on_slot = 9, .time0_slot = -1, \
                                                .fused_steps_slot = -1, .n_links = 6, .taped = taped, \
                                                .links = links_ }
TWIN_DERIVED(d_twin, 0, 2, twin_links);
TWIN_DERIVED(d_twin_of, 4, 2, twin_links);
TWIN_DERIVED(d_twin_rows, 0, 1, twin_links);
TWIN_DERIVED(d_twin_slot, 0, 2, twin_links_bad_slot);
TWIN_DERIVED(d_twin_time0, 0, 2, twin_links_time0);
#define VJP_DERIVED(var, n_, links_)                                                                                       \
    static const struct eg_unit_derived var = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_STEP_VJP, \
                                                .of_unit = 0, .tape_slot = -1, .n_taped = 0, .tape_rows = 0,              \
                                                .steps_slot = -1, .tape_on_slot = -1, .time0_slot = -1,   \
                                                .fused_steps_slot = -1, .n_links = n_, .taped = NULL, \
                                                .links = links_ }
VJP_DERIVED(d_vjp, 9, vjp_links);
VJP_DERIVED(d_vjp_missing, 8, vjp_links_missing);
VJP_DERIVED(d_vjp_twice, 9, vjp_links_twice);
VJP_DERIVED(d_vjp_reduced, 9, vjp_links_reduced);
VJP_DERIVED(d_vjp_steps, 9, vjp_links_steps);
VJP_DERIVED(d_vjp_unreduced, 9, vjp_links_unreduced);
VJP_DERIVED(d_vjp_badform, 9, vjp_links_badform);
/* A Newton record with two (TIME_STEP, VALUE) links: refused at load (the cardinality is exactly one). */
static const struct eg_unit_derived d_newton_two_steps = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_REFINE_NEWTON,
                                                           .of_unit = 0, .tape_slot = -1, .n_taped = 0, .tape_rows = 0,
                                                           .steps_slot = -1, .tape_on_slot = -1, .time0_slot = -1,
                                                           .fused_steps_slot = -1, .n_links = 2, .taped = NULL,
                                                           .links = newton_links_two_steps };

/* Measured tables of steps_reverse AUTO (the K = 3 primals below have state width 2): the nearest row to (2, 3) is the first of
 * each pair. The near row of "in" holds regime 37 and its far row does not; the near row of "out" does not and its far row does. */
static const int64_t row_widths[] = { 2, 2 }, row_ks[] = { 3, 12 };
static const double in_lo[] = { 0.0, 1000.0 }, in_hi[] = { 100.0, 2000.0 };
static const double out_lo[] = { 1000.0, 0.0 }, out_hi[] = { 2000.0, 100.0 };
static const double res_lo[] = { 0.0 }, res_hi[] = { 1.0 };
#define STEPS_TABLE(var, regime_, n_, lo_, hi_)                                                                     \
    static const struct eg_unit_steps_table var = { .struct_size = sizeof(struct eg_unit_steps_table), .regime = regime_, \
                                                    .n_rows = n_, .widths = row_widths, .ks = row_ks, .lo = lo_, .hi = hi_ }
STEPS_TABLE(table_in, EG_REGIME_LANES, 2, in_lo, in_hi);
STEPS_TABLE(table_out, EG_REGIME_LANES, 2, out_lo, out_hi);
STEPS_TABLE(table_res, EG_REGIME_PER_RESIDENT, 1, res_lo, res_hi);
#define FINISH3(var, table_)                                                                                    \
    static const struct eg_unit_finish var = { .struct_size = sizeof(struct eg_unit_finish), .flags = 0,        \
                                               .mask_slot = 2, .counter_slot = 3, .steps_slot = -1,             \
                                               .steps = 3, .steps_max = 0, .step_ops = -1, .fp64_ops_ir = -1,   \
                                               .host_steps_reverse = table_, .device_steps_reverse = NULL }
FINISH3(finish3_in, &table_in);
FINISH3(finish3_out, &table_out);
FINISH3(finish3_res, &table_res);
FINISH3(finish3_none, NULL);

#define DIGEST(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c }
/* The K = 3 primals (library indices 17, 20, 23, 26), each with the twin and the step vjp that state it as their primal. */
#define VJP_OF(var, of_)                                                                                               \
    static const struct eg_unit_derived var = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_STEP_VJP, \
                                                .of_unit = of_, .tape_slot = -1, .n_taped = 0, .tape_rows = 0,            \
                                                .steps_slot = -1, .tape_on_slot = -1, .time0_slot = -1,                  \
                                                .fused_steps_slot = -1, .n_links = 9, .taped = NULL, .links = vjp_links }
TWIN_DERIVED(d_twin_17, 17, 2, twin_links);
TWIN_DERIVED(d_twin_20, 20, 2, twin_links);
TWIN_DERIVED(d_twin_23, 23, 2, twin_links);
TWIN_DERIVED(d_twin_26, 26, 2, twin_links);
VJP_OF(d_vjp_17, 17);
VJP_OF(d_vjp_20, 20);
VJP_OF(d_vjp_23, 23);
VJP_OF(d_vjp_26, 26);
/* A K-step vjp that no link describes: steps_reverse AUTO drops it unread when it chooses PER_STEP. */
static const struct eg_unit_derived d_fused_vjp = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_FUSED_VJP,
                                                    .of_unit = 17, .tape_slot = -1, .n_taped = 0, .tape_rows = 0,
                                                    .steps_slot = -1, .tape_on_slot = -1, .time0_slot = -1,
                                                    .fused_steps_slot = -1, .n_links = 0, .taped = NULL, .links = NULL };

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
UNIT(u_vjp_badform, "lin_vjp", 0xF9, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_badform);
UNIT(u_newton_two_steps, "lin_newton", 0xFA, "lin_vjp_host", vjp_body, launch, NULL, &d_newton_two_steps);
UNIT(u_lin3_in, "lin3_in", 0xA0, "lin3_host", lin_body, launch, &finish3_in, NULL);
UNIT(u_tw_in, "lin3_in", 0xA1, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_17);
UNIT(u_vjp_in, "lin3_in_vjp", 0xA2, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_17);
UNIT(u_lin3_out, "lin3_out", 0xA3, "lin3_host", lin_body, launch, &finish3_out, NULL);
UNIT(u_tw_out, "lin3_out", 0xA4, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_20);
UNIT(u_vjp_out, "lin3_out_vjp", 0xA5, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_20);
UNIT(u_lin3_res, "lin3_res", 0xA6, "lin3_host", lin_body, launch, &finish3_res, NULL);
UNIT(u_tw_res, "lin3_res", 0xA7, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_23);
UNIT(u_vjp_res, "lin3_res_vjp", 0xA8, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_23);
UNIT(u_lin3_none, "lin3_none", 0xA9, "lin3_host", lin_body, launch, &finish3_none, NULL);
UNIT(u_tw_none, "lin3_none", 0xAA, "lin_tape3_host", tw3_body, launch, &twin_finish, &d_twin_26);
UNIT(u_vjp_none, "lin3_none_vjp", 0xAB, "lin_vjp_host", vjp_body, launch, NULL, &d_vjp_26);
UNIT(u_fused_vjp, "lin3_fused_vjp", 0xAC, "lin_vjp_host", vjp_body, launch, NULL, &d_fused_vjp);

static const struct eg_unit_meta* const units[] = { &u_lin, &u_tw1, &u_tw3, &u_tw16, &u_vjp, &u_lina, &u_tw_of, &u_tw_rows,
                                                    &u_tw_slot, &u_tw_time0, &u_vjp_missing, &u_vjp_twice, &u_vjp_reduced,
                                                    &u_vjp_steps, &u_vjp_unreduced, &u_vjp_badform, &u_newton_two_steps,
                                                    &u_lin3_in, &u_tw_in, &u_vjp_in, &u_lin3_out, &u_tw_out, &u_vjp_out, &u_lin3_res,
                                                    &u_tw_res, &u_vjp_res, &u_lin3_none, &u_tw_none, &u_vjp_none, &u_fused_vjp };
static const struct eg_library_meta library = { .struct_size = sizeof(struct eg_library_meta), .schema_major = EG_META_SCHEMA_MAJOR,
                                                .schema_minor = 0, .n_units = 30, .units = units };
const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
