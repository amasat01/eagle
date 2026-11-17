/*
 * eg_reduce and eg_copy over DEVICE planes, run on the context's stream. Reduce is eagle's canonical ordered tree (the order
 * the host runs, whatever the policy): each case is compared BITWISE with the old device reduction (the seam's
 * reduce_canonical_device, policy ordered) on the same planes, and with a plain loop for the exact ops and a long-double
 * sum for the float ones. Order rows (order_rows) pin the sum at sizes around the leaf and warp boundaries to committed bits
 * of that old reduction. Copy is checked against the source's bytes. Refusals by name: planes on both sides, overlapping
 * device planes, a shape mismatch, either call while recording on a device context.
 *
 * usage: device_ops_test <oracle dir>   (in_<case>.bin, comp_<case>.bin, out_<case>.bin)
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdlib.h>

#include "reduce_order_expected.h"

struct tensor2 {
    struct DLManagedTensorVersioned managed;
    int64_t shape[2];
};

struct dplane {
    void* d;
    struct tensor2 t;
    eg_plane* p;
};

static int dplane_new(eg_ctx* ctx, struct dplane* s, int ndim, int64_t rows, int64_t cols, uint8_t code, uint8_t bits, const void* init) {
    const size_t bytes = (size_t)rows * cols * bits / 8;
    memset(s, 0, sizeof(*s));
    if (cudaMalloc(&s->d, bytes ? bytes : 1) != cudaSuccess) return 0;
    if (init != NULL ? cudaMemcpy(s->d, init, bytes, cudaMemcpyHostToDevice) != cudaSuccess : cudaMemset(s->d, 0, bytes) != cudaSuccess) return 0;
    s->t.shape[0] = rows;
    s->t.shape[1] = cols;
    s->t.managed.version.major = DLPACK_MAJOR_VERSION;
    s->t.managed.dl_tensor.data = s->d;
    s->t.managed.dl_tensor.device.device_type = kDLCUDA;
    s->t.managed.dl_tensor.ndim = ndim;
    s->t.managed.dl_tensor.dtype.code = code;
    s->t.managed.dl_tensor.dtype.bits = bits;
    s->t.managed.dl_tensor.dtype.lanes = 1;
    s->t.managed.dl_tensor.shape = s->t.shape;
    return eg_plane_wrap(ctx, &s->t.managed, &s->p) == EG_OK;
}

static void dplane_free(eg_ctx* ctx, struct dplane* s) {
    if (s->p != NULL) eg_plane_free(ctx, s->p);
    cudaFree(s->d);
}

struct rcase {
    const char* name;
    uint8_t code, bits;
    eg_reduce_op op;
    int64_t n, cols;
    int comp, pair;
};

static const struct rcase CASES[] = {
    { "f64_sum_2x5000_pair", kDLFloat, 64, EG_REDUCE_SUM, 5000, 2, 0, 1 }, { "f64_sum_1x10000", kDLFloat, 64, EG_REDUCE_SUM, 10000, 1, 0, 0 },
    { "f32_sum_3x4097", kDLFloat, 32, EG_REDUCE_SUM, 4097, 3, 0, 0 }, { "f64_max_2x5000", kDLFloat, 64, EG_REDUCE_MAX, 5000, 2, 0, 0 },
    { "f32_min_1x777", kDLFloat, 32, EG_REDUCE_MIN, 777, 1, 0, 0 }, { "i32_sum_2x3000", kDLInt, 32, EG_REDUCE_SUM, 3000, 2, 0, 0 },
    { "i64_sum_1x9000", kDLInt, 64, EG_REDUCE_SUM, 9000, 1, 0, 0 }, { "u8_all_2x4500", kDLBool, 8, EG_REDUCE_ALL, 4500, 2, 0, 0 },
    { "f64_sum_comp_1x5000", kDLFloat, 64, EG_REDUCE_SUM, 5000, 1, 1, 0 }, { "f64_sum_1x1", kDLFloat, 64, EG_REDUCE_SUM, 1, 1, 0, 0 },
    { "f32_sum_pair_2x4097", kDLFloat, 32, EG_REDUCE_SUM, 4097, 2, 0, 1 },
};

/* Column j of the input as a long double. */
static long double at(const struct rcase* c, const void* in, int64_t j, int64_t i) {
    const int64_t k = j * c->n + i;
    if (c->code == kDLFloat) return c->bits == 32 ? ((const float*)in)[k] : ((const double*)in)[k];
    if (c->code == kDLInt) return c->bits == 32 ? ((const int32_t*)in)[k] : ((const int64_t*)in)[k];
    return ((const uint8_t*)in)[k];
}

/* The closed form of column j: exact for integer sums, max, min and all; a long-double sum (to 1e-12 of sum |x|) for floats. */
static int closed(const struct rcase* c, const void* in, const void* out, int64_t j) {
    long double acc = c->op == EG_REDUCE_ALL ? 1 : c->op == EG_REDUCE_SUM ? 0 : at(c, in, j, 0), mag = 0;
    for (int64_t i = 0; i < c->n; ++i) {
        const long double v = at(c, in, j, i);
        if (c->op == EG_REDUCE_SUM) acc += v;
        else if (c->op == EG_REDUCE_MAX) acc = v > acc ? v : acc;
        else if (c->op == EG_REDUCE_MIN) acc = v < acc ? v : acc;
        else acc = acc != 0 && v != 0;
        mag += fabsl(v);
    }
    long double got = 0;
    if (c->code == kDLFloat) got = c->bits == 32 ? ((const float*)out)[j] : ((const double*)out)[j];
    else if (c->code == kDLInt) got = c->bits == 32 ? ((const int32_t*)out)[j] : ((const int64_t*)out)[j];
    else got = ((const uint8_t*)out)[j];
    if (c->code == kDLFloat && c->op == EG_REDUCE_SUM) return fabsl(got - acc) <= mag * (c->bits == 32 ? 1e-6L : 1e-12L);
    return got == acc;
}

static void reduce_case(eg_ctx* ctx, const char* dir, const struct rcase* c, eg_reduce_policy policy) {
    const size_t item = c->bits / 8, in_bytes = (size_t)c->n * c->cols * item, out_n = (size_t)c->cols * (c->pair ? 2 : 1);
    char file[160], what[240];
    snprintf(file, sizeof(file), "in_%s.bin", c->name);
    void* in = malloc(in_bytes);
    void* comp = malloc(in_bytes);
    void* want = malloc(out_n * item);
    void* got = malloc(out_n * item);
    snprintf(what, sizeof(what), "%s (policy %d): old path inputs read", c->name, policy);
    int ok = in && comp && want && got && read_file(dir, file, in, in_bytes);
    if (ok && c->comp) {
        snprintf(file, sizeof(file), "comp_%s.bin", c->name);
        ok = read_file(dir, file, comp, in_bytes);
    }
    snprintf(file, sizeof(file), "out_%s.bin", c->name);
    ok = ok && read_file(dir, file, want, out_n * item);
    struct dplane pin = { 0 }, pout = { 0 }, pcomp = { 0 };
    ok = ok && dplane_new(ctx, &pin, 2, c->cols, c->n, c->code, c->bits, in) && dplane_new(ctx, &pout, 1, c->cols, 1, c->code, c->bits, NULL) &&
         (!(c->comp || c->pair) || dplane_new(ctx, &pcomp, c->pair ? 1 : 2, c->pair ? c->cols : c->cols, c->pair ? 1 : c->n, c->code, c->bits, c->pair ? NULL : comp));
    struct eg_reduce_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = c->pair ? EG_REDUCE_PAIR : 0;
    d.in = pin.p;
    d.out = pout.p;
    d.comp = (c->comp || c->pair) ? pcomp.p : NULL;
    d.op = c->op;
    d.policy = policy;
    ok = ok && eg_reduce(ctx, &d) == EG_OK && eg_sync(ctx) == EG_OK &&
         cudaMemcpy(got, pout.d, c->cols * item, cudaMemcpyDeviceToHost) == cudaSuccess &&
         (!c->pair || cudaMemcpy((char*)got + c->cols * item, pcomp.d, c->cols * item, cudaMemcpyDeviceToHost) == cudaSuccess);
    snprintf(what, sizeof(what), "eg_reduce %s (policy %d) over DEVICE planes: bitwise vs the old device reduction", c->name, policy);
    check(ok && memcmp(got, want, out_n * item) == 0, what);
    int exact = ok;
    for (int64_t j = 0; exact && j < c->cols && !c->comp; ++j) exact = closed(c, in, got, j);
    snprintf(what, sizeof(what), "eg_reduce %s: closed form (exact op, or a long-double sum)", c->name);
    check(exact, what);
    if (ok) {
        dplane_free(ctx, &pin);
        dplane_free(ctx, &pout);
        if (c->comp || c->pair) dplane_free(ctx, &pcomp);
    }
    free(in); free(comp); free(want); free(got);
}

/* Sample i of column j of the order-exposing input (reduce_order_oracle.py makes the same values): a splitmix64 hash of the flat
 * index picks sign and a 24-bit mantissa, even samples are scaled by 1e8 and odd ones by 1, so any change of summation order shows. */
static double order_input(int64_t n, int64_t j, int64_t i) {
    uint64_t z = (uint64_t)(j * n + i) * 0x9E3779B97F4A7C15ULL + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double v = (1.0 + (double)(z >> 40) / 16777216.0) * (i % 2 == 0 ? 1e8 : 1.0);
    return (z >> 39) & 1 ? -v : v;
}

/* eg_reduce SUM of two columns at each size, float32 and float64, under both policies: bitwise the committed bits of the old
 * device reduction. */
static void order_rows(eg_ctx* ctx) {
    enum { COLS = 2 };
    for (size_t r = 0; r < sizeof(ORDER_ROWS) / sizeof(ORDER_ROWS[0]); ++r) {
        const struct order_row* row = &ORDER_ROWS[r];
        const size_t item = (size_t)row->bits / 8;
        char what[160];
        for (eg_reduce_policy policy = EG_REDUCE_COMPENSATED; policy <= EG_REDUCE_FIXED; ++policy) {
            void* in = malloc((size_t)(COLS * row->n) * item);
            uint64_t got[COLS] = { 0 };
            int ok = in != NULL;
            for (int64_t k = 0; ok && k < COLS * row->n; ++k) {
                const double v = order_input(row->n, k / row->n, k % row->n);
                if (row->bits == 32) ((float*)in)[k] = (float)v;
                else ((double*)in)[k] = v;
            }
            const uint8_t code = kDLFloat;
            struct dplane pin = { 0 }, pout = { 0 };
            ok = ok && dplane_new(ctx, &pin, 2, COLS, row->n, code, (uint8_t)row->bits, in) &&
                 dplane_new(ctx, &pout, 1, COLS, 1, code, (uint8_t)row->bits, NULL);
            struct eg_reduce_desc d;
            memset(&d, 0, sizeof(d));
            d.struct_size = sizeof(d);
            d.in = pin.p;
            d.out = pout.p;
            d.op = EG_REDUCE_SUM;
            d.policy = policy;
            ok = ok && eg_reduce(ctx, &d) == EG_OK && eg_sync(ctx) == EG_OK;
            for (int j = 0; ok && j < COLS; ++j) {    /* the sum's bits, zero-extended to 64 */
                ok = cudaMemcpy(&got[j], (char*)pout.d + (size_t)j * item, item, cudaMemcpyDeviceToHost) == cudaSuccess;
            }
            snprintf(what, sizeof(what), "eg_reduce SUM f%d n=%lld policy %d: bitwise the old device reduction (order proof)", row->bits,
                     (long long)row->n, policy);
            check(ok && got[0] == row->sum[0] && got[1] == row->sum[1], what);
            if (pin.p != NULL) dplane_free(ctx, &pin);
            if (pout.p != NULL) dplane_free(ctx, &pout);
            free(in);
        }
    }
}

static void copy_rows(eg_ctx* ctx, const char* dir) {
    enum { N = 10000 };
    double* x = malloc(N * sizeof(double));
    double* back = malloc(N * sizeof(double));
    struct dplane a = { 0 }, b = { 0 }, c = { 0 };
    const int ok = x && back && read_file(dir, "in_f64_sum_1x10000.bin", x, N * sizeof(double)) && dplane_new(ctx, &a, 1, N, 1, kDLFloat, 64, x) &&
                   dplane_new(ctx, &b, 1, N, 1, kDLFloat, 64, NULL) && dplane_new(ctx, &c, 1, N - 1, 1, kDLFloat, 64, NULL);
    check(ok && eg_copy(ctx, a.p, b.p) == EG_OK && eg_sync(ctx) == EG_OK && cudaMemcpy(back, b.d, N * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess &&
              memcmp(back, x, N * sizeof(double)) == 0,
          "eg_copy over DEVICE planes: dst equals src bytes");
    check(ok && eg_copy(ctx, a.p, b.p) == EG_OK && eg_copy(ctx, a.p, a.p) == EG_OK, "eg_copy of a plane onto itself copies nothing");
    check(ok && eg_copy(ctx, a.p, c.p) == EG_BAD_REQUEST && reason_has(ctx, "one shape and dtype"), "eg_copy: a shape mismatch refused BAD_REQUEST");
    static double host_x[N];
    struct tensor ht;
    tensor_fill(&ht, N, host_x, kDLFloat, 64);
    ht.managed.dl_tensor.device.device_type = kDLCPU;
    eg_plane* hp = NULL;
    check(ok && eg_plane_wrap(ctx, &ht.managed, &hp) == EG_OK && eg_copy(ctx, a.p, hp) == EG_BAD_REQUEST && reason_has(ctx, "both sides"),
          "eg_copy: one device plane and one host plane refused BAD_REQUEST (one side per operation)");
    struct dplane o1 = { 0 }, o2 = { 0 };    /* two planes over one allocation, 8 bytes apart: partly overlapping */
    int ov = ok && dplane_new(ctx, &o1, 1, N - 1, 1, kDLFloat, 64, x) && dplane_new(ctx, &o2, 1, N - 1, 1, kDLFloat, 64, NULL);
    if (ov) {
        eg_plane_free(ctx, o2.p);
        cudaFree(o2.d);
        o2.d = NULL;
        o2.t.managed.dl_tensor.data = o1.d;
        o2.t.managed.dl_tensor.byte_offset = 8;
        ov = eg_plane_wrap(ctx, &o2.t.managed, &o2.p) == EG_OK;
    }
    check(ov && eg_copy(ctx, o1.p, o2.p) == EG_BAD_REQUEST && reason_has(ctx, "overlap"), "eg_copy: partly overlapping device planes refused BAD_REQUEST");
    /* recording on a device context: the host-side operations are refused by name */
    struct eg_reduce_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.in = a.p;
    d.out = b.p;
    eg_graph* g = NULL;
    check(ok && eg_record_begin(ctx) == EG_OK && eg_copy(ctx, a.p, b.p) == EG_NOT_WRITTEN && reason_has(ctx, "while recording") &&
              eg_reduce(ctx, &d) == EG_NOT_WRITTEN && reason_has(ctx, "while recording") && eg_record_end(ctx, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK,
          "eg_copy and eg_reduce while recording on a device context: NOT_WRITTEN by name");
    if (hp) eg_plane_free(ctx, hp);
    if (ov) {
        eg_plane_free(ctx, o2.p);
        eg_plane_free(ctx, o1.p);
        cudaFree(o1.d);
    }
    if (ok) {
        dplane_free(ctx, &a);
        dplane_free(ctx, &b);
        dplane_free(ctx, &c);
    }
    free(x); free(back);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <oracle dir>\n", argv[0]);
        return 2;
    }
    (void)parse_digest;    /* shared helper this test does not need */
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.flags = EG_CTX_DEVICE;
    st.device = 0;
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); ++i) reduce_case(ctx, argv[1], &CASES[i], EG_REDUCE_COMPENSATED);
    reduce_case(ctx, argv[1], &CASES[1], EG_REDUCE_FIXED);
    order_rows(ctx);
    copy_rows(ctx, argv[1]);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
