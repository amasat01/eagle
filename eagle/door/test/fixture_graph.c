// SPDX-License-Identifier: Apache-2.0
/*
 * A unit library for eg_graph_reverse with gradients known in closed form. Each primal has a vjp unit whose derived record
 * links its slots to the primal's:
 *
 *   0 scale   y = w * x        (x per sample, w a one-cell shared parameter plane)   1 scale_vjp
 *   2 square  o = a * a                                                              3 square_vjp
 *   4 mul2    o = a * b                                                              5 mul2_vjp
 *   6 uscale  y = k * x        (k a by-value uniform)                                7 uscale_vjp (reduced GRAD of the uniform)
 *  10 fan     v[c] = (c + 1) x  (a width-3 vector plane out)                         11 fan_vjp
 *  12 norm2   o = sum_c v[c]^2  (the vector plane in)                                13 norm2_vjp
 *  14 tick    c[i] -= 1          (an int32 per-sample counter plane in place; a loop's stop, no gradient)
 *   8 scale_tape_vjp   scale's vjp reading a tape row (a link a graph vjp does not take)
 *   9 scale_open_vjp   scale's vjp with a slot no link names
 *
 * Slots of scale: x, w, y. vjp of scale: x, w, bar_y, bar_x, bar_w.
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

#define H(i) ((double*)((const struct handle*)params[i])->data)

/* A (components, n) plane through its 40-byte mirror. */
struct mirror {
    double* data;
    uint64_t samples, comp_stride, sample_stride;
    int32_t device_type, device_id;
};
#define V(i, c, s) (((const struct mirror*)params[i])->data[(c) * ((const struct mirror*)params[i])->comp_stride + (s)])
#define WIDTH 3

EXPORT void scale_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void scale_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void square_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void square_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void mul2_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void mul2_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void uscale_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void uscale_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void fan_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void fan_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void norm2_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void norm2_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT void tick_host(void* const* params, int64_t base, int64_t count, int64_t n);
EXPORT const struct eg_library_meta* eg_unit_library_meta(void);

/* slots: x, w, y */
void scale_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) H(2)[i] = H(1)[0] * H(0)[i];
}
/* slots: x, w, bar_y, bar_x, bar_w */
void scale_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        H(3)[i] = H(1)[0] * H(2)[i];
        H(4)[i] = H(0)[i] * H(2)[i];
    }
}
/* slots: a, o */
void square_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) H(1)[i] = H(0)[i] * H(0)[i];
}
/* slots: a, bar_o, bar_a */
void square_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) H(2)[i] = 2.0 * H(0)[i] * H(1)[i];
}
/* slots: a, b, o */
void mul2_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) H(2)[i] = H(0)[i] * H(1)[i];
}
/* slots: a, b, bar_o, bar_a, bar_b */
void mul2_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        H(3)[i] = H(1)[i] * H(2)[i];
        H(4)[i] = H(0)[i] * H(2)[i];
    }
}
/* slots: x, k (uniform), y */
void uscale_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) H(2)[i] = *(const double*)params[1] * H(0)[i];
}
/* slots: x, k (uniform), bar_y, bar_x, bar_k */
void uscale_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        H(3)[i] = *(const double*)params[1] * H(2)[i];
        H(4)[i] = H(0)[i] * H(2)[i];
    }
}

/* slots: x, v */
void fan_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i)
        for (int c = 0; c < WIDTH; ++c) V(1, c, i) = (c + 1) * H(0)[i];
}
/* slots: bar_v, bar_x */
void fan_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        double g = 0.0;
        for (int c = 0; c < WIDTH; ++c) g += (c + 1) * V(0, c, i);
        H(1)[i] = g;
    }
}
/* slots: v, o */
void norm2_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i) {
        double t = 0.0;
        for (int c = 0; c < WIDTH; ++c) t += V(0, c, i) * V(0, c, i);
        H(1)[i] = t;
    }
}
/* slots: v, bar_o, bar_v */
void norm2_vjp_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    for (int64_t i = base; i < base + count; ++i)
        for (int c = 0; c < WIDTH; ++c) V(2, c, i) = 2.0 * V(0, c, i) * H(1)[i];
}

/* slots: c (int32) */
void tick_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)n;
    int32_t* c = (int32_t*)((const struct handle*)params[0])->data;
    for (int64_t i = base; i < base + count; ++i) c[i] -= 1;
}

EXPORT extern const char eagle_abi_tag[];
EXPORT extern const uint64_t eagle_layout_sizes[5];
const char eagle_abi_tag[] = "aether-abi/2";
const uint64_t eagle_layout_sizes[5] = { 40, sizeof(struct handle), 32, 4, 3 * sizeof(int64_t) };

#define SLOT_F(var, role_, name_, flags_)                                                                         \
    static const struct eg_unit_slot var = { .struct_size = sizeof(struct eg_unit_slot), .flags = flags_,        \
                                             .role = role_, .dtype = EG_DTYPE_REAL, .rank = 0, .like_slot = -1,  \
                                             .n_gather = 0, .shape = NULL, .gather = NULL, .name = name_ }
#define SLOT(var, role_, name_) SLOT_F(var, role_, name_, 0)
SLOT(s_x, EG_ROLE_PER_SAMPLE, "x");
SLOT(s_a, EG_ROLE_PER_SAMPLE, "a");
SLOT(s_b, EG_ROLE_PER_SAMPLE, "b");
SLOT_F(s_w, EG_ROLE_LOOKUP, "w", EG_SLOT_PARAM_PLANE | EG_SLOT_SHARED);
SLOT(s_k, EG_ROLE_UNIFORM, "k");
SLOT(s_y, EG_ROLE_OUT, "y");
SLOT(s_o, EG_ROLE_OUT, "o");
SLOT(s_bar_y, EG_ROLE_PER_SAMPLE, "bar_y");
SLOT(s_bar_o, EG_ROLE_PER_SAMPLE, "bar_o");
SLOT(s_bar_x, EG_ROLE_OUT, "bar_x");
SLOT(s_bar_a, EG_ROLE_OUT, "bar_a");
SLOT(s_bar_b, EG_ROLE_OUT, "bar_b");
SLOT(s_bar_w, EG_ROLE_OUT, "bar_w");
SLOT(s_bar_k, EG_ROLE_OUT, "bar_k");
static const struct eg_unit_slot s_c = { .struct_size = sizeof(struct eg_unit_slot), .flags = 0, .role = EG_ROLE_MUTABLE, .dtype = EG_DTYPE_INT32,
                                         .rank = 0, .like_slot = -1, .n_gather = 0, .shape = NULL, .gather = NULL, .name = "c" };
static const int64_t width_shape[] = { 3 };
#define SLOT_V(var, role_, name_)                                                                                 \
    static const struct eg_unit_slot var = { .struct_size = sizeof(struct eg_unit_slot), .flags = 0,             \
                                             .role = role_, .dtype = EG_DTYPE_REAL, .rank = 1, .like_slot = -1,  \
                                             .n_gather = 0, .shape = width_shape, .gather = NULL, .name = name_ }
SLOT_V(s_v_out, EG_ROLE_OUT, "v");
SLOT_V(s_v_in, EG_ROLE_VEC_IN, "v");
SLOT_V(s_bar_v_in, EG_ROLE_VEC_IN, "bar_v");
SLOT_V(s_bar_v_out, EG_ROLE_OUT, "bar_v");

static const struct eg_unit_slot* const scale_slots[] = { &s_x, &s_w, &s_y };
static const struct eg_unit_slot* const scale_vjp_slots[] = { &s_x, &s_w, &s_bar_y, &s_bar_x, &s_bar_w };
static const struct eg_unit_slot* const square_slots[] = { &s_a, &s_o };
static const struct eg_unit_slot* const square_vjp_slots[] = { &s_a, &s_bar_o, &s_bar_a };
static const struct eg_unit_slot* const mul2_slots[] = { &s_a, &s_b, &s_o };
static const struct eg_unit_slot* const mul2_vjp_slots[] = { &s_a, &s_b, &s_bar_o, &s_bar_a, &s_bar_b };
static const struct eg_unit_slot* const uscale_slots[] = { &s_x, &s_k, &s_y };
static const struct eg_unit_slot* const uscale_vjp_slots[] = { &s_x, &s_k, &s_bar_y, &s_bar_x, &s_bar_k };

static const struct eg_unit_slot* const fan_slots[] = { &s_x, &s_v_out };
static const struct eg_unit_slot* const fan_vjp_slots[] = { &s_bar_v_in, &s_bar_x };
static const struct eg_unit_slot* const norm2_slots[] = { &s_v_in, &s_o };
static const struct eg_unit_slot* const norm2_vjp_slots[] = { &s_v_in, &s_bar_o, &s_bar_v_out };
static const struct eg_unit_slot* const tick_slots[] = { &s_c };

#define BODY(var, slots_, n_) \
    static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, .n_slots = n_, .slots = slots_ }
BODY(scale_body, scale_slots, 3);
BODY(scale_vjp_body, scale_vjp_slots, 5);
BODY(square_body, square_slots, 2);
BODY(square_vjp_body, square_vjp_slots, 3);
BODY(mul2_body, mul2_slots, 3);
BODY(mul2_vjp_body, mul2_vjp_slots, 5);
BODY(uscale_body, uscale_slots, 3);
BODY(uscale_vjp_body, uscale_vjp_slots, 5);

BODY(fan_body, fan_slots, 2);
BODY(fan_vjp_body, fan_vjp_slots, 2);
BODY(norm2_body, norm2_slots, 2);
BODY(norm2_vjp_body, norm2_vjp_slots, 3);
BODY(tick_body, tick_slots, 1);

static const struct eg_unit_launch launch = { .struct_size = sizeof(struct eg_unit_launch), .n_lane_axes = 0, .block = 0,
                                              .active_map_slot = -1, .active_count_slot = -1, .index_map_slot = -1,
                                              .index_count_slot = -1, .n_plane_axes = 0, .group = 1 };

#define LINK(var, slot_, what_, form_, of_, reduced_) \
    static const struct eg_unit_link var = { .struct_size = sizeof(struct eg_unit_link), .slot = slot_, .what = EG_WHAT_##what_, \
                                             .of_slot = of_, .reduced = reduced_, .form = EG_FORM_##form_ }
/* scale_vjp: x, w, bar_y, bar_x, bar_w */
LINK(sv_x, 0, PRIMAL_SLOT, VALUE, 0, 0);
LINK(sv_w, 1, PRIMAL_SLOT, VALUE, 1, 0);
LINK(sv_by, 2, PRIMAL_SLOT, ADJOINT_IN, 2, 0);
LINK(sv_bx, 3, PRIMAL_SLOT, GRAD, 0, 0);
LINK(sv_bw, 4, PRIMAL_SLOT, GRAD, 1, 1);
LINK(sv_x_tape, 0, TAPE_ROW, VALUE, 0, 0);
static const struct eg_unit_link* const scale_vjp_links[] = { &sv_x, &sv_w, &sv_by, &sv_bx, &sv_bw };
static const struct eg_unit_link* const scale_tape_links[] = { &sv_x_tape, &sv_w, &sv_by, &sv_bx, &sv_bw };
static const struct eg_unit_link* const scale_open_links[] = { &sv_x, &sv_w, &sv_by, &sv_bx };
/* square_vjp: a, bar_o, bar_a */
LINK(qv_a, 0, PRIMAL_SLOT, VALUE, 0, 0);
LINK(qv_bo, 1, PRIMAL_SLOT, ADJOINT_IN, 1, 0);
LINK(qv_ba, 2, PRIMAL_SLOT, GRAD, 0, 0);
static const struct eg_unit_link* const square_vjp_links[] = { &qv_a, &qv_bo, &qv_ba };
/* mul2_vjp: a, b, bar_o, bar_a, bar_b */
LINK(mv_a, 0, PRIMAL_SLOT, VALUE, 0, 0);
LINK(mv_b, 1, PRIMAL_SLOT, VALUE, 1, 0);
LINK(mv_bo, 2, PRIMAL_SLOT, ADJOINT_IN, 2, 0);
LINK(mv_ba, 3, PRIMAL_SLOT, GRAD, 0, 0);
LINK(mv_bb, 4, PRIMAL_SLOT, GRAD, 1, 0);
static const struct eg_unit_link* const mul2_vjp_links[] = { &mv_a, &mv_b, &mv_bo, &mv_ba, &mv_bb };
/* uscale_vjp: x, k, bar_y, bar_x, bar_k */
LINK(uv_k, 1, PRIMAL_SLOT, VALUE, 1, 0);
LINK(uv_bk, 4, PRIMAL_SLOT, GRAD, 1, 1);
static const struct eg_unit_link* const uscale_vjp_links[] = { &sv_x, &uv_k, &sv_by, &sv_bx, &uv_bk };

/* fan_vjp: bar_v, bar_x. norm2_vjp: v, bar_o, bar_v. */
LINK(fv_bv, 0, PRIMAL_SLOT, ADJOINT_IN, 1, 0);
LINK(fv_bx, 1, PRIMAL_SLOT, GRAD, 0, 0);
static const struct eg_unit_link* const fan_vjp_links[] = { &fv_bv, &fv_bx };
LINK(nv_v, 0, PRIMAL_SLOT, VALUE, 0, 0);
LINK(nv_bo, 1, PRIMAL_SLOT, ADJOINT_IN, 1, 0);
LINK(nv_bv, 2, PRIMAL_SLOT, GRAD, 0, 0);
static const struct eg_unit_link* const norm2_vjp_links[] = { &nv_v, &nv_bo, &nv_bv };

#define VJP_OF(var, of_, n_, links_)                                                                                    \
    static const struct eg_unit_derived var = { .struct_size = sizeof(struct eg_unit_derived), .role = EG_DERIVED_STEP_VJP, \
                                                .of_unit = of_, .tape_slot = -1, .n_taped = 0, .tape_rows = 0,            \
                                                .steps_slot = -1, .tape_on_slot = -1, .time0_slot = -1,                  \
                                                .fused_steps_slot = -1, .n_links = n_, .taped = NULL, .links = links_ }
VJP_OF(d_scale_vjp, 0, 5, scale_vjp_links);
VJP_OF(d_square_vjp, 2, 3, square_vjp_links);
VJP_OF(d_mul2_vjp, 4, 5, mul2_vjp_links);
VJP_OF(d_uscale_vjp, 6, 5, uscale_vjp_links);
VJP_OF(d_fan_vjp, 10, 2, fan_vjp_links);
VJP_OF(d_norm2_vjp, 12, 3, norm2_vjp_links);
VJP_OF(d_scale_tape_vjp, 0, 5, scale_tape_links);
VJP_OF(d_scale_open_vjp, 0, 4, scale_open_links);

#define DIGEST(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c }
#ifdef EG_GRAPH_IMAGE
#define IMAGE(function) .device_image = "graph__image", .device_image_size = "graph__image_size", .device_function = function, \
    .device_image_format = EG_IMAGE_FATBIN
#else
#define IMAGE(function) .device_image_format = EG_IMAGE_NONE
#endif
#define NO_IMAGE .device_image_format = EG_IMAGE_NONE
#define UNIT(var, name_, byte, entry, body_, derived_, image_)                                                        \
    static const struct eg_unit_meta var = { .struct_size = sizeof(struct eg_unit_meta), .name = name_,             \
                                             .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE,                    \
                                             .scalar_type = EG_SCALAR_FLOAT64, .host_entry = entry, image_,         \
                                             .body = &body_, .launch = &launch, .finish = NULL, .derived = derived_ }
UNIT(u_scale, "scale", 0x10, "scale_host", scale_body, NULL, IMAGE("scale"));
UNIT(u_scale_vjp, "scale_vjp", 0x11, "scale_vjp_host", scale_vjp_body, &d_scale_vjp, IMAGE("scale_vjp"));
UNIT(u_square, "square", 0x12, "square_host", square_body, NULL, IMAGE("square"));
UNIT(u_square_vjp, "square_vjp", 0x13, "square_vjp_host", square_vjp_body, &d_square_vjp, IMAGE("square_vjp"));
UNIT(u_mul2, "mul2", 0x14, "mul2_host", mul2_body, NULL, IMAGE("mul2"));
UNIT(u_mul2_vjp, "mul2_vjp", 0x15, "mul2_vjp_host", mul2_vjp_body, &d_mul2_vjp, IMAGE("mul2_vjp"));
UNIT(u_uscale, "uscale", 0x16, "uscale_host", uscale_body, NULL, NO_IMAGE);
UNIT(u_uscale_vjp, "uscale_vjp", 0x17, "uscale_vjp_host", uscale_vjp_body, &d_uscale_vjp, NO_IMAGE);
UNIT(u_scale_tape_vjp, "scale_tape_vjp", 0x18, "scale_vjp_host", scale_vjp_body, &d_scale_tape_vjp, NO_IMAGE);
UNIT(u_scale_open_vjp, "scale_open_vjp", 0x19, "scale_vjp_host", scale_vjp_body, &d_scale_open_vjp, NO_IMAGE);

UNIT(u_fan, "fan", 0x1a, "fan_host", fan_body, NULL, IMAGE("fan"));
UNIT(u_fan_vjp, "fan_vjp", 0x1b, "fan_vjp_host", fan_vjp_body, &d_fan_vjp, IMAGE("fan_vjp"));
UNIT(u_norm2, "norm2", 0x1c, "norm2_host", norm2_body, NULL, IMAGE("norm2"));
UNIT(u_norm2_vjp, "norm2_vjp", 0x1d, "norm2_vjp_host", norm2_vjp_body, &d_norm2_vjp, IMAGE("norm2_vjp"));
UNIT(u_tick, "tick", 0x1e, "tick_host", tick_body, NULL, IMAGE("tick"));

static const struct eg_unit_meta* const units[] = { &u_scale, &u_scale_vjp, &u_square, &u_square_vjp, &u_mul2, &u_mul2_vjp,
                                                    &u_uscale, &u_uscale_vjp, &u_scale_tape_vjp, &u_scale_open_vjp, &u_fan, &u_fan_vjp, &u_norm2, &u_norm2_vjp, &u_tick };
static const struct eg_library_meta library = { .struct_size = sizeof(struct eg_library_meta), .schema_major = EG_META_SCHEMA_MAJOR,
                                                .schema_minor = 0, .n_units = 15, .units = units };
const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }
