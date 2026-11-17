/* SPDX-License-Identifier: Apache-2.0 */

/*
 * A hand-written unit library for the gemm record: units whose meta states matrix products (eg_unit_gemm) and whose
 * host entry computes the SAME products with plain loops over the same record, so a host plan is the oracle of a
 * DEVICE plan. Planes are LOOKUP (inputs) and ACCUM_OUT (outputs) slots, each a dense row-major (rows, cols + slack)
 * view; the loop works out its row stride the way the door states it.
 *
 *  - "<w>_<shape>" (w: f32, f64): one forward product C = op(A) op(B) of (3, 5) x (5, 4): sq (4x4x4), rect, ta, tb,
 *    tc, all (the three transposes), lane (columns are n_samples / 3), wide (planes 2 columns wider than stated).
 *  - "<w>_vjp": the adjoint of a forward unit, two calls: dW = dY X^T, then dX += W^T dY (ACCUMULATE).
 *  - "bad_*": records the door refuses, one defect each (the test names the status and the reason).
 *  - "gemm_index", "gemm_finish": a gemm unit with an index set / a finishing loop.
 */
#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"

#include <stdint.h>

#define EG_FIXTURE_EXPORT __attribute__((visibility("default")))

/* The scalar handle of the v2 host ABI (plugin/gref_layout.h, 32 bytes). */
struct fixture_handle {
    void* data;
    uint64_t samples;
    uint64_t stride;
    int32_t device_type;
    int32_t device_id;
};

EG_FIXTURE_EXPORT const struct eg_library_meta* eg_unit_library_meta(void);
EG_FIXTURE_EXPORT void noop_host(void* const* params, int64_t base, int64_t count, int64_t n);
void noop_host(void* const* params, int64_t base, int64_t count, int64_t n) {
    (void)params, (void)base, (void)count, (void)n;
}

/* The reference loop: every call of the record, C (+)= op(A) op(B). The whole product is one piece of work: the tile
 * that starts at sample 0 does it. */
#define REFERENCE(T, name)                                                                                     \
    static void name(const struct eg_unit_gemm* g, void* const* params, int64_t n, int64_t slack) {            \
        for (int32_t i = 0; i < g->n_calls; ++i) {                                                             \
            const struct eg_unit_gemm_call* c = g->calls[i];                                                   \
            const int64_t ext[6] = { c->a_rows, c->a_cols, c->b_rows, c->b_cols, c->c_rows, c->c_cols };       \
            int64_t e[6];                                                                                      \
            for (int j = 0; j < 6; ++j) e[j] = ext[j] == EG_GEMM_PER_LANE ? n / g->lane_fanin : ext[j];        \
            const T* a = (const T*)((const struct fixture_handle*)params[c->a_slot])->data;                    \
            const T* b = (const T*)((const struct fixture_handle*)params[c->b_slot])->data;                    \
            T* out = (T*)((const struct fixture_handle*)params[c->c_slot])->data;                              \
            const int ta = (c->flags & EG_GEMM_TRANS_A) != 0, tb = (c->flags & EG_GEMM_TRANS_B) != 0;          \
            const int tc = (c->flags & EG_GEMM_TRANS_C) != 0, acc = (c->flags & EG_GEMM_ACCUMULATE) != 0;      \
            const int64_t lda = e[1] + slack, ldb = e[3] + slack, ldc = e[5] + slack;                          \
            const int64_t m = ta ? e[1] : e[0], k = ta ? e[0] : e[1], cols = tb ? e[2] : e[3];                 \
            for (int64_t r = 0; r < m; ++r)                                                                    \
                for (int64_t q = 0; q < cols; ++q) {                                                           \
                    double s = 0.0;                                                                            \
                    for (int64_t p = 0; p < k; ++p)                                                            \
                        s += (double)(ta ? a[p * lda + r] : a[r * lda + p]) *                                  \
                             (double)(tb ? b[q * ldb + p] : b[p * ldb + q]);                                   \
                    T* at = &out[tc ? q * ldc + r : r * ldc + q];                                              \
                    *at = (T)(acc ? (double)*at + s : s);                                                      \
                }                                                                                              \
        }                                                                                                      \
    }
REFERENCE(double, reference_f64)
REFERENCE(float, reference_f32)

#define ENTRY(tag, reference, slack)                                                                           \
    EG_FIXTURE_EXPORT void tag##_host(void* const* params, int64_t base, int64_t count, int64_t n);            \
    void tag##_host(void* const* params, int64_t base, int64_t count, int64_t n) {                             \
        (void)count;                                                                                           \
        if (base == 0) reference(&gemm_##tag, params, n, slack);                                               \
    }

#define DIGEST(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, \
                    c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c }

#define SLOT_OF(var, role_, dtype_, name_)                                                                     \
    static const struct eg_unit_slot var = { .struct_size = sizeof(struct eg_unit_slot), .role = role_,        \
                                             .dtype = dtype_, .like_slot = -1, .name = name_ }
SLOT_OF(slot_a, EG_ROLE_LOOKUP, EG_DTYPE_REAL, "a");
SLOT_OF(slot_b, EG_ROLE_LOOKUP, EG_DTYPE_REAL, "b");
SLOT_OF(slot_c, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "c");
SLOT_OF(slot_ci, EG_ROLE_ACCUM_OUT, EG_DTYPE_INT32, "c_int");
SLOT_OF(slot_w, EG_ROLE_LOOKUP, EG_DTYPE_REAL, "w");
SLOT_OF(slot_x, EG_ROLE_LOOKUP, EG_DTYPE_REAL, "x");
SLOT_OF(slot_dy, EG_ROLE_LOOKUP, EG_DTYPE_REAL, "dy");
SLOT_OF(slot_dw, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "dw");
SLOT_OF(slot_dx, EG_ROLE_ACCUM_OUT, EG_DTYPE_REAL, "dx");
SLOT_OF(slot_count, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "index_count");
SLOT_OF(slot_k, EG_ROLE_MUTABLE, EG_DTYPE_REAL, "k");
SLOT_OF(slot_limit, EG_ROLE_PER_SAMPLE, EG_DTYPE_REAL, "limit");
SLOT_OF(slot_mask, EG_ROLE_TERMINATED, EG_DTYPE_BOOL, "mask");
SLOT_OF(slot_finished, EG_ROLE_LOOKUP, EG_DTYPE_UINT32, "finished");

static const struct eg_unit_slot* const slots_abc[] = { &slot_a, &slot_b, &slot_c };
static const struct eg_unit_slot* const slots_abci[] = { &slot_a, &slot_b, &slot_ci };
static const struct eg_unit_slot* const slots_vjp[] = { &slot_w, &slot_x, &slot_dy, &slot_dw, &slot_dx };
static const struct eg_unit_slot* const slots_index[] = { &slot_a, &slot_b, &slot_c, &slot_count };
static const struct eg_unit_slot* const slots_finish[] = { &slot_k, &slot_limit, &slot_mask, &slot_finished, &slot_a, &slot_b, &slot_c };

#define BODY(var, slots_, count_)                                                                              \
    static const struct eg_unit_body var = { .struct_size = sizeof(struct eg_unit_body), .access = EG_ACCESS_SAMPLE_LOCAL, \
                                             .n_slots = count_, .slots = slots_ }
BODY(body_abc, slots_abc, 3);
BODY(body_abci, slots_abci, 3);
BODY(body_vjp, slots_vjp, 5);
BODY(body_index, slots_index, 4);
BODY(body_finish, slots_finish, 7);

/* Every slot or table index is written: -1 is absent. */
#define LAUNCH(var, index_count_)                                                                              \
    static const struct eg_unit_launch var = { .struct_size = sizeof(struct eg_unit_launch), .active_map_slot = -1, \
        .active_count_slot = -1, .index_map_slot = -1, .index_count_slot = index_count_, .group = 1 }
LAUNCH(launch_plain, -1);
LAUNCH(launch_index, 3);
static const struct eg_unit_finish finish_one = {
    .struct_size = sizeof(struct eg_unit_finish), .mask_slot = 2, .counter_slot = 3, .steps_slot = -1, .steps = 1,
    .step_ops = -1, .fp64_ops_ir = -1 };

#define CALL(var, flags_, a_slot_, b_slot_, c_slot_, ar, ac, br, bc, cr, cc)                                  \
    static const struct eg_unit_gemm_call var = { .struct_size = sizeof(struct eg_unit_gemm_call), .flags = flags_, \
        .a_slot = a_slot_, .b_slot = b_slot_, .c_slot = c_slot_, .a_rows = ar, .a_cols = ac, .b_rows = br,     \
        .b_cols = bc, .c_rows = cr, .c_cols = cc }

/* One unit with one call: the record, its meta (body_, launch_, finish_ by pointer) and, with REF, its loop. */
#define UNIT(tag, byte, scalar, entry, body_, launch_, finish_, ncalls, fanin, flags, as, bs, cs, ar, ac, br, bc, cr, cc) \
    CALL(call_##tag, flags, as, bs, cs, ar, ac, br, bc, cr, cc);                                               \
    static const struct eg_unit_gemm_call* const calls_##tag[] = { &call_##tag };                              \
    static const struct eg_unit_gemm gemm_##tag = { .struct_size = sizeof(struct eg_unit_gemm), .n_calls = ncalls, \
        .lane_fanin = fanin, .calls = calls_##tag };                                                           \
    static const struct eg_unit_meta unit_##tag = { .struct_size = sizeof(struct eg_unit_meta), .name = #tag, \
        .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE, .scalar_type = scalar, .host_entry = entry,       \
        .device_image_format = EG_IMAGE_NONE, .body = &body_, .launch = &launch_, .finish = finish_,           \
        .gemm = &gemm_##tag }

#define TA EG_GEMM_TRANS_A
#define TB EG_GEMM_TRANS_B
#define TC EG_GEMM_TRANS_C
#define LANE EG_GEMM_PER_LANE

/* The forward units of one width: (tag, digest, flags, A, B, C stored shapes, fanin, slack). */
#define FORWARD(W, byte, scalar, reference)                                                                    \
    UNIT(W##_sq, byte + 0, scalar, #W "_sq_host", body_abc, launch_plain, NULL, 1, 0, 0, 0, 1, 2, 4, 4, 4, 4, 4, 4); \
    UNIT(W##_rect, byte + 1, scalar, #W "_rect_host", body_abc, launch_plain, NULL, 1, 0, 0, 0, 1, 2, 3, 5, 5, 4, 3, 4); \
    UNIT(W##_ta, byte + 2, scalar, #W "_ta_host", body_abc, launch_plain, NULL, 1, 0, TA, 0, 1, 2, 5, 3, 5, 4, 3, 4); \
    UNIT(W##_tb, byte + 3, scalar, #W "_tb_host", body_abc, launch_plain, NULL, 1, 0, TB, 0, 1, 2, 3, 5, 4, 5, 3, 4); \
    UNIT(W##_tc, byte + 4, scalar, #W "_tc_host", body_abc, launch_plain, NULL, 1, 0, TC, 0, 1, 2, 3, 5, 5, 4, 4, 3); \
    UNIT(W##_all, byte + 5, scalar, #W "_all_host", body_abc, launch_plain, NULL, 1, 0, TA | TB | TC, 0, 1, 2, 5, 3, 4, 5, 4, 3); \
    UNIT(W##_lane, byte + 6, scalar, #W "_lane_host", body_abc, launch_plain, NULL, 1, 3, 0, 0, 1, 2, 3, 5, 5, LANE, 3, LANE); \
    UNIT(W##_wide, byte + 7, scalar, #W "_wide_host", body_abc, launch_plain, NULL, 1, 0, 0, 0, 1, 2, 3, 5, 5, 4, 3, 4); \
    ENTRY(W##_sq, reference, 0) ENTRY(W##_rect, reference, 0) ENTRY(W##_ta, reference, 0) ENTRY(W##_tb, reference, 0) \
    ENTRY(W##_tc, reference, 0) ENTRY(W##_all, reference, 0) ENTRY(W##_lane, reference, 0) ENTRY(W##_wide, reference, 2)

FORWARD(f64, 0x30, EG_SCALAR_FLOAT64, reference_f64)
FORWARD(f32, 0x40, EG_SCALAR_FLOAT32, reference_f32)

/* The adjoint of Y = W X (W (3, 5), X (5, 4), Y (3, 4)): dW = dY X^T, then dX += W^T dY. */
#define ADJOINT(W, byte, scalar, reference)                                                                    \
    CALL(call_##W##_vjp0, TB, 2, 1, 3, 3, 4, 5, 4, 3, 5);                                                      \
    CALL(call_##W##_vjp1, TA | EG_GEMM_ACCUMULATE, 0, 2, 4, 3, 5, 3, 4, 5, 4);                                  \
    static const struct eg_unit_gemm_call* const calls_##W##_vjp[] = { &call_##W##_vjp0, &call_##W##_vjp1 };   \
    static const struct eg_unit_gemm gemm_##W##_vjp = { .struct_size = sizeof(struct eg_unit_gemm), .n_calls = 2, \
        .calls = calls_##W##_vjp };                                                                            \
    static const struct eg_unit_meta unit_##W##_vjp = { .struct_size = sizeof(struct eg_unit_meta), .name = #W "_vjp", \
        .digest = DIGEST(byte), .pattern = EG_PATTERN_PURE, .scalar_type = scalar, .host_entry = #W "_vjp_host", \
        .device_image_format = EG_IMAGE_NONE, .body = &body_vjp, .launch = &launch_plain, .gemm = &gemm_##W##_vjp }; \
    ENTRY(W##_vjp, reference, 0)
ADJOINT(f64, 0x50, EG_SCALAR_FLOAT64, reference_f64)
ADJOINT(f32, 0x51, EG_SCALAR_FLOAT32, reference_f32)

/* Records the door refuses (digest 0x60 + n). The shapes are the forward (3, 5) x (5, 4) unless the defect is in them. */
#define BAD(tag, n, scalar, body_, flags, as, bs, cs, ncalls, fanin, ar, ac, br, bc, cr, cc) \
    UNIT(tag, 0x60 + n, scalar, "noop_host", body_, launch_plain, NULL, ncalls, fanin, flags, as, bs, cs, ar, ac, br, bc, cr, cc)
BAD(bad_slot, 0, EG_SCALAR_FLOAT64, body_abc, 0, 0, 1, 9, 1, 0, 3, 5, 5, 4, 3, 4);
BAD(bad_role, 1, EG_SCALAR_FLOAT64, body_abc, 0, 0, 1, 0, 1, 0, 3, 5, 5, 4, 3, 4);
BAD(bad_dtype, 2, EG_SCALAR_FLOAT64, body_abci, 0, 0, 1, 2, 1, 0, 3, 5, 5, 4, 3, 4);
BAD(bad_ncalls, 3, EG_SCALAR_FLOAT64, body_abc, 0, 0, 1, 2, 0, 0, 3, 5, 5, 4, 3, 4);
BAD(bad_extent, 4, EG_SCALAR_FLOAT64, body_abc, 0, 0, 1, 2, 1, 0, 0, 5, 5, 4, 3, 4);
BAD(bad_lane, 5, EG_SCALAR_FLOAT64, body_abc, 0, 0, 1, 2, 1, 0, 3, 5, 5, LANE, 3, LANE);
BAD(bad_flag, 6, EG_SCALAR_FLOAT64, body_abc, 1u << 7, 0, 1, 2, 1, 0, 3, 5, 5, 4, 3, 4);
BAD(bad_banded, 7, EG_SCALAR_BANDED, body_abc, 0, 0, 1, 2, 1, 0, 3, 5, 5, 4, 3, 4);
BAD(bad_k, 8, EG_SCALAR_FLOAT64, body_abc, 0, 0, 1, 2, 1, 0, 3, 5, 4, 4, 3, 4);
BAD(bad_c, 9, EG_SCALAR_FLOAT64, body_abc, 0, 0, 1, 2, 1, 0, 3, 5, 5, 4, 3, 3);
UNIT(gemm_index, 0x70, EG_SCALAR_FLOAT64, "noop_host", body_index, launch_index, NULL, 1, 0, 0, 0, 1, 2, 3, 5, 5, 4, 3, 4);
UNIT(gemm_finish, 0x71, EG_SCALAR_FLOAT64, "noop_host", body_finish, launch_plain, &finish_one, 1, 0, 0, 4, 5, 6, 3, 5, 5, 4, 3, 4);

#define LISTED(X)                                                                                              \
    X(f64_sq) X(f64_rect) X(f64_ta) X(f64_tb) X(f64_tc) X(f64_all) X(f64_lane) X(f64_wide) X(f64_vjp)         \
    X(f32_sq) X(f32_rect) X(f32_ta) X(f32_tb) X(f32_tc) X(f32_all) X(f32_lane) X(f32_wide) X(f32_vjp)         \
    X(bad_slot) X(bad_role) X(bad_dtype) X(bad_ncalls) X(bad_extent) X(bad_lane) X(bad_flag) X(bad_banded)    \
    X(bad_k) X(bad_c) X(gemm_index) X(gemm_finish)
#define POINTER(tag) &unit_##tag,
static const struct eg_unit_meta* const units[] = { LISTED(POINTER) };

static const struct eg_library_meta library = {
    .struct_size = sizeof(struct eg_library_meta), .schema_major = EG_META_SCHEMA_MAJOR, .schema_minor = 0,
    .n_units = sizeof(units) / sizeof(units[0]), .units = units };

const struct eg_library_meta* eg_unit_library_meta(void) { return &library; }

/* The size block beside the host entries (plugin/gref_abi.h): the v2 tag and eagle's own layout sizes. */
EG_FIXTURE_EXPORT extern const char eagle_abi_tag[];
EG_FIXTURE_EXPORT extern const uint64_t eagle_layout_sizes[5];
const char eagle_abi_tag[] = "aether-abi/2";
const uint64_t eagle_layout_sizes[5] = { 40, sizeof(struct fixture_handle), 32, 4, 3 * sizeof(int64_t) };
