/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The gemm unit record through the door. HOST rows (always): every load, plan and bind refusal of the record by status
 * and reason, and a host plan of each gemm unit running its own loop (the oracle) without cuBLAS ever being opened.
 * DEVICE rows (-DEG_GEMM_DEVICE, run through the GPU pool): each unit's DEVICE launch against that loop within the
 * width's tolerance (routed to cuBLAS), the replay of a recorded launch against the eager one bit for bit, a launch on a
 * stream the caller gave, and each shape the door cannot route (two partitions, an index set, a runner's loop) launching
 * the unit's own kernel to the host plan's values. With "--no-cublas" (run with libno_cublas.so preloaded) only the missing
 * cuBLAS row runs.
 *
 * Usage: gemm_test <libfixture_gemm.so> [--no-cublas]
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"
#include "plugin/dlpack.h"

#ifdef EG_GEMM_DEVICE
#include <cuda_runtime_api.h>
#endif

#include <dlfcn.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static const struct eg_library_meta* library;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

/* True when the context's reason contains @p part (printed when it does not). */
static int reason_has(eg_ctx* ctx, const char* part) {
    char buf[1024];
    size_t len = 0;
    if (eg_error(ctx, buf, sizeof(buf), &len) != EG_OK) return 0;
    if (strstr(buf, part) != NULL) return 1;
    printf("      reason: %s\n", buf);
    return 0;
}

static int refused(eg_status got, eg_status want, eg_ctx* ctx, const char* part) {
    return got == want && reason_has(ctx, part);
}

static eg_ctx* context(int device) {
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.flags = device ? EG_CTX_DEVICE : 0;
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    return eg_ctx_new(&s, &ctx) == EG_OK ? ctx : NULL;
}

static const struct eg_unit_meta* unit_meta(const char* name) {
    for (uint32_t i = 0; i < library->n_units; ++i)
        if (strcmp(library->units[i]->name, name) == 0) return library->units[i];
    return NULL;
}

static eg_status load(eg_ctx* ctx, const char* path, const char* name, eg_unit** out) {
    const struct eg_unit_meta* m = unit_meta(name);
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memcpy(ref.digest, m->digest, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = path;
    d.unit = &ref;
    *out = NULL;
    return eg_unit_load(ctx, &d, out);
}

static eg_status plan_new(eg_ctx* ctx, eg_unit* unit, eg_structure s, int64_t n, int64_t parts, const int64_t* base,
                          const int64_t* count, eg_plan** out) {
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = s;
    d.n_samples = n;
    d.n_partitions = parts;
    d.part_base = base;
    d.part_count = count;
    *out = NULL;
    return eg_plan_new(ctx, &d, out);
}

/* ---- planes: row-major (rows, cols + slack) matrices of one width, on the host or the device ---------------------- */

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[2];
};

struct matrix {
    int slot, bits, device;
    uint32_t cell;  /* the value of a uint32 slot: the index count, every lane */
    int kind;       /* 0: a product plane (2-D float of ``bits``); else a slot no product names, 1-D, zero-filled, never compared: 1 uint32, 2 bool, 3 float64 */
    int64_t rows, cols, elements;
    void* host;     /* the test's own copy */
    void* data;     /* where the plane lives */
    struct tensor tensor;
    eg_plane* plane;
};

static eg_plane* wrap(eg_ctx* ctx, struct matrix* m, int64_t rows, int64_t cols) {
    memset(&m->tensor, 0, sizeof(m->tensor));
    m->tensor.shape[0] = rows;
    m->tensor.shape[1] = cols;
    m->tensor.managed.version.major = DLPACK_MAJOR_VERSION;
    m->tensor.managed.dl_tensor.data = m->data;
    m->tensor.managed.dl_tensor.device.device_type = m->device ? kDLCUDA : kDLCPU;
    m->tensor.managed.dl_tensor.ndim = m->kind == 0 ? 2 : 1;
    m->tensor.managed.dl_tensor.dtype.code = m->kind == 1 ? kDLUInt : m->kind == 2 ? kDLBool : kDLFloat;
    m->tensor.managed.dl_tensor.dtype.bits = (uint8_t)m->bits;
    m->tensor.managed.dl_tensor.dtype.lanes = 1;
    m->tensor.managed.dl_tensor.shape = m->tensor.shape;
    return eg_plane_wrap(ctx, &m->tensor.managed, &m->plane) == EG_OK ? m->plane : NULL;
}

static double sample(unsigned* state) {   /* a deterministic value in [-1, 1) */
    *state = *state * 1664525u + 1013904223u;
    return (double)(*state >> 8) / (double)(1u << 23) - 1.0;
}

/* The stored shape of every plane slot of @p u at @p n lanes, from its gemm record (the same extent resolved as the door does). */
static void shapes_of(const struct eg_unit_meta* u, int64_t n, int slack, struct matrix* out, int* count) {
    const struct eg_unit_gemm* g = u->gemm;
    *count = 0;
    for (int32_t i = 0; i < g->n_calls; ++i) {
        const struct eg_unit_gemm_call* c = g->calls[i];
        const int32_t slots[3] = { c->a_slot, c->b_slot, c->c_slot };
        const int64_t raw[3][2] = { { c->a_rows, c->a_cols }, { c->b_rows, c->b_cols }, { c->c_rows, c->c_cols } };
        for (int j = 0; j < 3; ++j) {
            int known = 0;
            for (int q = 0; q < *count; ++q) known = known || out[q].slot == slots[j];
            if (known) continue;
            struct matrix* m = &out[(*count)++];
            memset(m, 0, sizeof(*m));
            m->slot = slots[j];
            m->bits = u->scalar_type == EG_SCALAR_FLOAT32 ? 32 : 64;
            m->rows = raw[j][0] == EG_GEMM_PER_LANE ? n / g->lane_fanin : raw[j][0];
            m->cols = (raw[j][1] == EG_GEMM_PER_LANE ? n / g->lane_fanin : raw[j][1]) + slack;
            m->elements = m->rows * m->cols;
        }
    }
    for (int32_t i = 0; u->body != NULL && i < u->body->n_slots; ++i) {   /* the unit's other slots: one-element planes, but the door's own counter */
        int known = u->finish != NULL && u->finish->counter_slot == i;
        for (int q = 0; q < *count; ++q) known = known || out[q].slot == i;
        if (known) continue;
        const int32_t dtype = u->body->slots[i]->dtype;
        struct matrix* m = &out[(*count)++];
        memset(m, 0, sizeof(*m));
        m->slot = i, m->rows = m->cols = m->elements = u->body->slots[i]->role == EG_ROLE_LOOKUP ? 1 : n;   /* the index count is one cell, the rest per-sample */
        m->kind = dtype == EG_DTYPE_UINT32 ? 1 : dtype == EG_DTYPE_BOOL ? 2 : 3;
        m->bits = m->kind == 3 ? 64 : m->kind == 1 ? 32 : 8;
        m->cell = (uint32_t)n;
    }
}

static void fill(struct matrix* m, unsigned seed) {
    unsigned state = seed;
    m->host = calloc((size_t)m->elements, (size_t)m->bits / 8);
    if (m->kind != 0) {
        if (m->kind == 1) *(uint32_t*)m->host = m->cell;
        return;
    }
    for (int64_t i = 0; i < m->elements; ++i) {
        const double v = sample(&state);
        if (m->bits == 32) ((float*)m->host)[i] = (float)v;
        else ((double*)m->host)[i] = v;
    }
}

/* How a unit is launched: recorded and replayed, as the loop of a runner, over @p parts partitions (base/count). ``entry``
 * receives the plan's EG_ENTRY_* flags. */
struct variant {
    int record, runner;
    int64_t parts;
    const int64_t *base, *count;
    uint32_t entry;
};

/* Run @p u over @p n lanes on a context: host planes of the test's own data, or device copies of them. Returns the
 * planes (outputs hold the result after eg_sync and a read back); @p matrices hold the initial data in ``host``. */
static eg_status run(eg_ctx* ctx, eg_unit* unit, eg_structure s, struct matrix* matrices, int count, int64_t n, struct variant* v,
                     eg_plan** plan_out) {
    int32_t slots[8];
    eg_plane* planes[8];
    for (int i = 0; i < count; ++i) {
        struct matrix* m = &matrices[i];
        m->device = s == EG_STRUCTURE_DEVICE;
        m->data = m->host;
#ifdef EG_GEMM_DEVICE
        if (m->device) {
            const size_t bytes = (size_t)m->elements * (size_t)m->bits / 8;
            if (cudaMalloc(&m->data, bytes) != cudaSuccess) return EG_FAULT;
            cudaMemcpy(m->data, m->host, bytes, cudaMemcpyHostToDevice);
        }
#endif
        slots[i] = m->slot;
        planes[i] = wrap(ctx, m, m->rows, m->cols);
        if (planes[i] == NULL) return EG_BAD_REQUEST;
    }
    eg_status st = plan_new(ctx, unit, s, n, v->parts, v->base, v->count, plan_out);
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = count;
    b.plane_slots = slots;
    b.planes = planes;
    if (st == EG_OK) st = eg_bind(ctx, *plan_out, &b);
    struct eg_entry_figures figures;
    memset(&figures, 0, sizeof(figures));
    figures.struct_size = sizeof(figures);
    if (st == EG_OK) st = eg_plan_entry_figures(ctx, *plan_out, &figures);
    v->entry = figures.flags;
    eg_graph* graph = NULL;
    if (st == EG_OK && v->runner) {
        struct eg_runner_desc r;
        eg_runner* runner = NULL;
        struct eg_run_report report;
        memset(&r, 0, sizeof(r));
        memset(&report, 0, sizeof(report));
        report.struct_size = sizeof(report);
        r.struct_size = sizeof(r);
        r.n_plans = 1;
        r.plans = plan_out;
        r.max_steps = 1;
        st = eg_runner_new(ctx, &r, &runner);
        if (st == EG_OK) st = eg_run(ctx, runner, &report);
        if (runner != NULL) eg_runner_free(ctx, runner);
    } else {
        if (st == EG_OK && v->record) st = eg_record_begin(ctx);
        if (st == EG_OK) st = eg_launch(ctx, *plan_out, NULL);
        if (st == EG_OK && v->record) st = eg_record_end(ctx, &graph);
        if (st == EG_OK && v->record) st = eg_graph_launch(ctx, graph);
    }
    if (st == EG_OK) st = eg_sync(ctx);
    if (graph != NULL) eg_graph_free(ctx, graph);
    return st;
}

/* The device planes read back into ``host`` copies; returns them in a fresh array of buffers. */
static void read_back(struct matrix* matrices, int count, void** results) {
    for (int i = 0; i < count; ++i) {
        const size_t bytes = (size_t)matrices[i].elements * (size_t)matrices[i].bits / 8;
        results[i] = malloc(bytes);
        if (!matrices[i].device) memcpy(results[i], matrices[i].data, bytes);
#ifdef EG_GEMM_DEVICE
        else cudaMemcpy(results[i], matrices[i].data, bytes, cudaMemcpyDeviceToHost);
#endif
    }
}

static void release(eg_ctx* ctx, eg_plan* plan, struct matrix* matrices, int count) {
    if (plan != NULL) eg_plan_free(ctx, plan);
    for (int i = 0; i < count; ++i) {
        if (matrices[i].plane != NULL) eg_plane_free(ctx, matrices[i].plane);
#ifdef EG_GEMM_DEVICE
        if (matrices[i].device) cudaFree(matrices[i].data);
#endif
        free(matrices[i].host);
    }
}

/* The largest |got - want| over every element of every plane, or -1 when a value is not a number. */
static double worst_difference(struct matrix* matrices, int count, void* const* got, void* const* want) {
    double worst = 0.0;
    for (int i = 0; i < count; ++i)
        for (int64_t j = 0; matrices[i].kind == 0 && j < matrices[i].elements; ++j) {
            const double g = matrices[i].bits == 32 ? ((float*)got[i])[j] : ((double*)got[i])[j];
            const double w = matrices[i].bits == 32 ? ((float*)want[i])[j] : ((double*)want[i])[j];
            if (isnan(g) || isnan(w)) return -1.0;
            worst = fabs(g - w) > worst ? fabs(g - w) : worst;
        }
    return worst;
}

static int process_loaded(const char* needle) {
    FILE* f = fopen("/proc/self/maps", "r");
    char line[1024];
    int found = 0;
    while (f != NULL && fgets(line, sizeof(line), f) != NULL) found = found || strstr(line, needle) != NULL;
    if (f != NULL) fclose(f);
    return found;
}


/* ---- direct products (eg_gemm_new / eg_gemm_run / eg_gemm_free) -------------------------------------------------- */

struct direct {
    int bits;
    uint32_t flags;
    int64_t m, n, k;
    int slack;      /* columns of slack on every plane: a column slice */
};

/* The stored (rows, cols) of A, B, C of a case, from its flags. */
static void direct_shapes(const struct direct* d, int64_t shape[3][2]) {
    const int ta = (d->flags & EG_GEMM_TRANS_A) != 0, tb = (d->flags & EG_GEMM_TRANS_B) != 0, tc = (d->flags & EG_GEMM_TRANS_C) != 0;
    const int64_t a[2] = { ta ? d->k : d->m, ta ? d->m : d->k }, b[2] = { tb ? d->n : d->k, tb ? d->k : d->n }, c[2] = { tc ? d->n : d->m, tc ? d->m : d->n };
    memcpy(shape[0], a, sizeof(a)), memcpy(shape[1], b, sizeof(b)), memcpy(shape[2], c, sizeof(c));
}

static eg_status direct_desc(eg_ctx* ctx, const struct direct* d, eg_gemm** out) {
    int64_t shape[3][2];
    direct_shapes(d, shape);
    struct eg_gemm_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    desc.flags = d->flags;
    desc.dtype_bits = (uint32_t)d->bits;
    desc.a_rows = shape[0][0], desc.a_cols = shape[0][1], desc.b_rows = shape[1][0], desc.b_cols = shape[1][1];
    desc.c_rows = shape[2][0], desc.c_cols = shape[2][1];
    return eg_gemm_new(ctx, &desc, out);
}

/* Run a case on @p ctx (host or device planes) from the data of @p seed: C is returned (m*n-stored elements incl. slack) in
 * @p c_out. @p record: through eg_record_begin/end and a replay; @p free_first: eg_gemm_free before the replay. */
static eg_status direct_run(eg_ctx* ctx, int device, const struct direct* d, int record, int free_first, void** c_out) {
    int64_t shape[3][2];
    direct_shapes(d, shape);
    struct matrix mat[3];
    eg_plane* plane[3];
    for (int i = 0; i < 3; ++i) {
        memset(&mat[i], 0, sizeof(mat[i]));
        mat[i].bits = d->bits, mat[i].device = device, mat[i].rows = shape[i][0], mat[i].cols = shape[i][1] + d->slack;
        mat[i].elements = mat[i].rows * mat[i].cols;
        fill(&mat[i], 21u + (unsigned)i);
        mat[i].data = mat[i].host;
#ifdef EG_GEMM_DEVICE
        if (device) {
            const size_t bytes = (size_t)mat[i].elements * (size_t)d->bits / 8;
            if (cudaMalloc(&mat[i].data, bytes) != cudaSuccess) return EG_FAULT;
            cudaMemcpy(mat[i].data, mat[i].host, bytes, cudaMemcpyHostToDevice);
        }
#endif
        plane[i] = wrap(ctx, &mat[i], mat[i].rows, mat[i].cols);
    }
    eg_gemm* gemm = NULL;
    eg_graph* graph = NULL;
    eg_status st = direct_desc(ctx, d, &gemm);
    if (st == EG_OK && record) st = eg_record_begin(ctx);
    if (st == EG_OK) st = eg_gemm_run(ctx, gemm, plane[0], plane[1], plane[2]);
    if (st == EG_OK && record) st = eg_record_end(ctx, &graph);
    if (st == EG_OK && record && free_first) st = eg_gemm_free(ctx, gemm), gemm = NULL;
    if (st == EG_OK && record) st = eg_graph_launch(ctx, graph);
    if (st == EG_OK) st = eg_sync(ctx);
    read_back(&mat[2], 1, c_out);
    if (graph != NULL) eg_graph_free(ctx, graph);
    if (gemm != NULL) eg_gemm_free(ctx, gemm);
    for (int i = 0; i < 3; ++i) {
        eg_plane_free(ctx, plane[i]);
#ifdef EG_GEMM_DEVICE
        if (device) cudaFree(mat[i].data);
#endif
        free(mat[i].host);
    }
    return st;
}

/* The independent oracle: the same inputs as direct_run, the product and the accumulate in long double, in plain C
 * with its own index arithmetic. @p want gets C, @p scale the per-element sum |a||b| + |c0| that the rounding bound scales. */
static void direct_oracle(const struct direct* d, long double* want, long double* scale) {
    int64_t shape[3][2];
    direct_shapes(d, shape);
    struct matrix mat[3];
    for (int i = 0; i < 3; ++i) {
        memset(&mat[i], 0, sizeof(mat[i]));
        mat[i].bits = d->bits, mat[i].elements = shape[i][0] * (shape[i][1] + d->slack);
        fill(&mat[i], 21u + (unsigned)i);
    }
    const int64_t lda = shape[0][1] + d->slack, ldb = shape[1][1] + d->slack, ldc = shape[2][1] + d->slack;
#define AT(i, r, c, ld) (d->bits == 32 ? (long double)((float*)mat[i].host)[(r) * (ld) + (c)] : (long double)((double*)mat[i].host)[(r) * (ld) + (c)])
    for (int64_t i = 0; i < d->m; ++i)
        for (int64_t j = 0; j < d->n; ++j) {
            long double sum = 0.0L, mag = 0.0L;
            for (int64_t p = 0; p < d->k; ++p) {
                const long double x = (d->flags & EG_GEMM_TRANS_A) ? AT(0, p, i, lda) : AT(0, i, p, lda);
                const long double y = (d->flags & EG_GEMM_TRANS_B) ? AT(1, j, p, ldb) : AT(1, p, j, ldb);
                sum += x * y;
                mag += fabsl(x * y);
            }
            const long double before = (d->flags & EG_GEMM_TRANS_C) ? AT(2, j, i, ldc) : AT(2, i, j, ldc);
            const int64_t at = (d->flags & EG_GEMM_TRANS_C) ? j * ldc + i : i * ldc + j;
            want[at] = (d->flags & EG_GEMM_ACCUMULATE) ? before + sum : sum;
            scale[at] = mag + fabsl(before);
        }
#undef AT
    for (int i = 0; i < 3; ++i) free(mat[i].host);
}

/* Whether @p got is within @p factor * (k + 2) * u * scale of the oracle everywhere C is written (u: the width's unit round-off).
 * Slack columns are never written: they must equal the initial data, which the oracle holds as ``want`` = 0 there, so they are skipped. */
static int within(const struct direct* d, const void* got, const long double* want, const long double* scale, double factor, double* worst_ratio) {
    int64_t shape[3][2];
    direct_shapes(d, shape);
    const int64_t ldc = shape[2][1] + d->slack;
    const long double u = d->bits == 32 ? 5.9604644775390625e-8L : 1.1102230246251565e-16L;
    int ok = 1;
    *worst_ratio = 0.0;
    for (int64_t r = 0; r < shape[2][0]; ++r)
        for (int64_t q = 0; q < shape[2][1]; ++q) {
            const long double g = d->bits == 32 ? ((const float*)got)[r * ldc + q] : ((const double*)got)[r * ldc + q];
            const long double bound = factor * (long double)(d->k + 2) * u * scale[r * ldc + q];
            const long double err = fabsl(g - want[r * ldc + q]);
            if (!(err <= bound)) ok = 0;
            if (bound > 0 && (double)(err / bound) > *worst_ratio) *worst_ratio = (double)(err / bound);
        }
    return ok;
}

static void direct_cases(struct direct* out, int* n) {
    *n = 0;
    for (int bits = 32; bits <= 64; bits += 32)
        for (uint32_t t = 0; t < 8; ++t)   /* every TRANS_A / TRANS_B / TRANS_C combination; odd t accumulates, t divisible by 3 slices columns */
            out[(*n)++] = (struct direct){ bits, t | ((t & 1u) ? EG_GEMM_ACCUMULATE : 0u), 5, 4, 3, t % 3 == 0 ? 2 : 0 };
    out[(*n)++] = (struct direct){ 64, 0, 6, 1, 7, 0 };   /* a reduce against a ones plane: n = 1 */
    out[(*n)++] = (struct direct){ 32, EG_GEMM_TRANS_A, 6, 1, 7, 0 };
}

static void direct_rows(eg_ctx* host, eg_ctx* dev) {
    struct direct cases[24];
    int count = 0;
    direct_cases(cases, &count);
    int host_ok = 1, rec_ok = 1;
    double worst_host = 0.0;
#ifdef EG_GEMM_DEVICE
    int dev_ok = 1;
    double worst_dev = 0.0;
#endif
    for (int i = 0; i < count; ++i) {
        const struct direct* d = &cases[i];
        long double want[64], scale[64];
        direct_oracle(d, want, scale);
        void* h[1];
        double ratio = 0.0;
        const eg_status st = direct_run(host, 0, d, 0, 0, h);
        /* the host loop sums in the plane's own width in order: within (k + 2) u of exact per element (the standard bound of a recursive sum of k products and one add) */
        host_ok = host_ok && st == EG_OK && within(d, h[0], want, scale, 1.0, &ratio);
        worst_host = ratio > worst_host ? ratio : worst_host;
        int64_t shape[3][2];
        direct_shapes(d, shape);
        const size_t bytes = (size_t)(shape[2][0] * (shape[2][1] + d->slack)) * (size_t)d->bits / 8;
        for (int free_first = 0; free_first < 2; ++free_first) {   /* the loop is deterministic: a recorded replay gives the eager run's bits */
            void* again[1];
            rec_ok = rec_ok && direct_run(host, 0, d, 1, free_first, again) == EG_OK && memcmp(h[0], again[0], bytes) == 0;
            free(again[0]);
        }
#ifdef EG_GEMM_DEVICE
        void* g[1];
        /* device == host: each is within (k + 2) u of exact, so they differ by at most 2 (k + 2) u of the scale */
        const eg_status sd = direct_run(dev, 1, d, 0, 0, g);
        dev_ok = dev_ok && sd == EG_OK && within(d, g[0], want, scale, 1.0, &ratio);
        worst_dev = ratio > worst_dev ? ratio : worst_dev;
        long double device_vs_host[64];   /* the device result against the host loop's, 2 (k + 2) u of the scale apart at most */
        for (size_t e = 0; e < bytes / (size_t)(d->bits / 8); ++e) device_vs_host[e] = d->bits == 32 ? ((float*)h[0])[e] : ((double*)h[0])[e];
        dev_ok = dev_ok && within(d, g[0], device_vs_host, scale, 2.0, &ratio);
        void* rg[1];
        void* fg[1];
        const int replay = direct_run(dev, 1, d, 1, 0, rg) == EG_OK && direct_run(dev, 1, d, 1, 1, fg) == EG_OK;
        rec_ok = rec_ok && replay && memcmp(g[0], rg[0], bytes) == 0 && memcmp(g[0], fg[0], bytes) == 0;
        free(g[0]), free(rg[0]), free(fg[0]);
#endif
        free(h[0]);
    }
    (void)dev;
    char what[200];
    snprintf(what, sizeof(what), "eg_gemm_run on the host: %d cases (all 8 TRANS combinations at f32 and f64, ACCUMULATE, a column slice, a reduce with n = 1) match a long-double C loop within (k + 2) u (worst %.3g of the bound)", count, worst_host);
    check(host_ok, what);
#ifdef EG_GEMM_DEVICE
    snprintf(what, sizeof(what), "eg_gemm_run on the device: the same cases match the long-double loop within (k + 2) u, so device and host differ by at most 2 (k + 2) u (worst %.3g of one bound)", worst_dev);
    check(dev_ok, what);
#endif
    check(rec_ok, "a recorded eg_gemm_run replays to the bits of the eager run (host and device), and eg_gemm_free before the replay is safe");
}


/* One refused descriptor: the status and the text. */
static void desc_row(eg_ctx* ctx, const char* what, uint32_t flags, uint32_t bits, int64_t ar, int64_t ac, int64_t br, int64_t bc, int64_t cr, int64_t cc,
                     const char* reason) {
    struct eg_gemm_desc d = { sizeof(d), flags, bits, 0, ar, ac, br, bc, cr, cc };
    eg_gemm* g = NULL;
    char text[160];
    snprintf(text, sizeof(text), "eg_gemm_new refuses %s: BAD_REQUEST naming it", what);
    check(refused(eg_gemm_new(ctx, &d, &g), EG_BAD_REQUEST, ctx, reason) && g == NULL, text);
}

static struct matrix* plain(eg_ctx* ctx, struct matrix* m, int bits, int device, int64_t rows, int64_t cols) {
    memset(m, 0, sizeof(*m));
    m->bits = bits, m->device = device, m->rows = rows, m->cols = cols, m->elements = rows * cols;
    m->data = m->host = calloc((size_t)m->elements, (size_t)bits / 8);
    return wrap(ctx, m, rows, cols) == NULL ? NULL : m;
}

static void direct_refusals(eg_ctx* ctx) {
    desc_row(ctx, "an unknown flag bit", 1u << 7, 64, 3, 5, 5, 4, 3, 4, "eg_gemm_desc.flags: unknown bits 128");
    desc_row(ctx, "a float width of 16", 0, 16, 3, 5, 5, 4, 3, 4, "eg_gemm_desc.dtype_bits 16 is not a float width (32 or 64)");
    desc_row(ctx, "an empty extent", 0, 64, 0, 5, 5, 4, 3, 4, "eg_gemm_desc.a_rows 0 is outside 1 .. 2147483647");
    desc_row(ctx, "an extent past 2^31 - 1", 0, 64, 3, 5, 5, 2147483648LL, 3, 4, "eg_gemm_desc.b_cols 2147483648 is outside 1 .. 2147483647");
    desc_row(ctx, "contracted extents that disagree", 0, 64, 3, 5, 4, 4, 3, 4, "eg_gemm_desc: the contracted extents disagree: A stored (3, 5) gives 5, B stored (4, 4) gives 4");
    desc_row(ctx, "a C stored other than the product's", 0, 64, 3, 5, 5, 4, 3, 3, "eg_gemm_desc: C is stored (3, 3) but the product of (3, 5) and (5, 4) is stored (3, 4)");
    struct eg_gemm_desc ok = { sizeof(ok), 0, 64, 0, 3, 5, 5, 4, 3, 4 };
    eg_gemm* g = NULL;
    check(refused(eg_gemm_new(ctx, &ok, NULL), EG_BAD_REQUEST, ctx, "eg_gemm_new: out is NULL") &&
              refused(eg_gemm_new(ctx, NULL, &g), EG_BAD_REQUEST, ctx, "eg_gemm_desc is NULL"),
          "eg_gemm_new refuses a NULL out and a NULL descriptor: BAD_REQUEST naming it");
    check(eg_record_begin(ctx) == EG_OK && refused(eg_gemm_new(ctx, &ok, &g), EG_BAD_REQUEST, ctx, "eg_gemm_new while recording") && eg_record_end(ctx, (eg_graph**)&g) == EG_OK,
          "eg_gemm_new while recording: BAD_REQUEST naming it, and the recording still ends");
    eg_graph_free(ctx, (eg_graph*)g);
    g = NULL;
    struct matrix a, b, c, wrong_width, too_few_rows, narrow, overlap;
    eg_plane *pa = plain(ctx, &a, 64, 0, 3, 5) ? a.plane : NULL, *pb = plain(ctx, &b, 64, 0, 5, 4) ? b.plane : NULL, *pc = plain(ctx, &c, 64, 0, 3, 4) ? c.plane : NULL;
    eg_plane *p32 = plain(ctx, &wrong_width, 32, 0, 3, 5) ? wrong_width.plane : NULL, *p4 = plain(ctx, &too_few_rows, 64, 0, 4, 5) ? too_few_rows.plane : NULL;
    eg_plane *pn = plain(ctx, &narrow, 64, 0, 3, 4) ? narrow.plane : NULL;
    memset(&overlap, 0, sizeof(overlap));
    overlap.bits = 64, overlap.data = a.data;   /* C over A's memory */
    eg_plane* po = wrap(ctx, &overlap, 3, 4);
    check(eg_gemm_new(ctx, &ok, &g) == EG_OK && pa != NULL && pb != NULL && pc != NULL && p32 != NULL && p4 != NULL && pn != NULL && po != NULL, "a product and planes to refuse runs of");
    struct { eg_plane *a, *b, *c; const char* reason; } runs[] = {
        { p32, pb, pc, "eg_gemm_run a: the plane must be float64 (the product's width)" },
        { p4, pb, pc, "eg_gemm_run a: the plane must be 2-D with 3 rows" },
        { pn, pb, pc, "eg_gemm_run a: the row stride 4 is smaller than the 5 stated columns" },
        { pa, pb, po, "eg_gemm_run: c overlaps a or b" },
        { NULL, pb, pc, "a is NULL" } };
    for (size_t i = 0; g != NULL && i < sizeof(runs) / sizeof(runs[0]); ++i)
        check(refused(eg_gemm_run(ctx, g, runs[i].a, runs[i].b, runs[i].c), EG_BAD_REQUEST, ctx, runs[i].reason), runs[i].reason);
    check(refused(eg_gemm_run(ctx, NULL, pa, pb, pc), EG_BAD_REQUEST, ctx, "gemm is NULL"), "eg_gemm_run of a NULL handle: BAD_REQUEST naming it");
    check(g != NULL && eg_gemm_free(ctx, g) == EG_OK && refused(eg_gemm_run(ctx, g, pa, pb, pc), EG_BAD_REQUEST, ctx, "gemm is not a live handle") &&
              refused(eg_gemm_free(ctx, g), EG_BAD_REQUEST, ctx, "gemm is not a live handle") && eg_gemm_free(ctx, NULL) == EG_OK,
          "eg_gemm_run and a second eg_gemm_free of a freed handle: BAD_REQUEST; eg_gemm_free(NULL) is OK");
    struct matrix* all[] = { &a, &b, &c, &wrong_width, &too_few_rows, &narrow };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) eg_plane_free(ctx, all[i]->plane), free(all[i]->host);
    eg_plane_free(ctx, po);
}

/* ---- HOST rows ------------------------------------------------------------------------------------------------- */

static void host_loop_rows(eg_ctx* ctx, const char* path) {
    struct variant eager = { 0 };
    static const char* const names[] = { "f64_sq", "f64_rect", "f64_ta", "f64_tb", "f64_tc", "f64_all", "f64_lane", "f64_wide", "f64_vjp",
                                         "f32_sq", "f32_rect", "f32_ta", "f32_tb", "f32_tc", "f32_all", "f32_lane", "f32_wide", "f32_vjp" };
    int ok = 1;
    for (size_t u = 0; u < sizeof(names) / sizeof(names[0]); ++u) {
        eg_unit* unit = NULL;
        struct matrix matrices[8];
        int count = 0;
        const struct eg_unit_meta* meta = unit_meta(names[u]);
        const int64_t n = strstr(names[u], "lane") != NULL ? 12 : 7;
        shapes_of(meta, n, strstr(names[u], "wide") != NULL ? 2 : 0, matrices, &count);
        for (int i = 0; i < count; ++i) fill(&matrices[i], 11u + (unsigned)i);
        void* before[8];
        for (int i = 0; i < count; ++i) {
            before[i] = malloc((size_t)matrices[i].elements * (size_t)matrices[i].bits / 8);
            memcpy(before[i], matrices[i].host, (size_t)matrices[i].elements * (size_t)matrices[i].bits / 8);
        }
        eg_plan* plan = NULL;
        const eg_status st = load(ctx, path, names[u], &unit) == EG_OK ? run(ctx, unit, EG_STRUCTURE_HOST_TEAM, matrices, count, n, &eager, &plan) : EG_FAULT;
        void* after[8];
        read_back(matrices, count, after);
        const int changed = worst_difference(matrices, count, after, before) > 1e-3;   /* the loop wrote an output */
        if (st != EG_OK || !changed) printf("      %s: status %d, wrote %d\n", names[u], (int)st, changed);
        ok = ok && st == EG_OK && changed;
        for (int i = 0; i < count; ++i) free(before[i]), free(after[i]);
        release(ctx, plan, matrices, count);
        if (unit != NULL) eg_unit_free(ctx, unit);
    }
    check(ok, "a HOST plan of each gemm unit runs the unit's own loop and writes its outputs");
    check(!process_loaded("libcublas"), "a host plan never opens cuBLAS");
}

static void refusal_rows(eg_ctx* ctx, const char* path) {
    struct { const char* unit; eg_status status; const char* reason; } loads[] = {
        { "bad_slot", EG_BAD_REQUEST, "c_slot 9 is outside the 3 slots" },
        { "bad_role", EG_BAD_REQUEST, "is not a written role" },
        { "bad_dtype", EG_BAD_REQUEST, "is not of the unit's scalar_type" },
        { "bad_ncalls", EG_BAD_REQUEST, "n_calls 0 without calls" },
        { "bad_extent", EG_BAD_REQUEST, "has the extent 0" },
        { "bad_lane", EG_BAD_REQUEST, "an EG_GEMM_PER_LANE extent and lane_fanin 0" },
        { "bad_flag", EG_NOT_WRITTEN, "unknown flag bits 128" },
        { "bad_banded", EG_NOT_WRITTEN, "a banded unit with a gemm record" },
        { "bad_param_plane", EG_BAD_REQUEST, "is PARAM_PLANE but not a LOOKUP" } };
    for (size_t i = 0; i < sizeof(loads) / sizeof(loads[0]); ++i) {
        eg_unit* unit = NULL;
        char what[96];
        snprintf(what, sizeof(what), "eg_unit_load of '%s' refused %d naming the defect", loads[i].unit, (int)loads[i].status);
        check(refused(load(ctx, path, loads[i].unit, &unit), loads[i].status, ctx, loads[i].reason), what);
    }
    struct { const char* unit; int64_t n; const char* reason; } plans[] = {
        { "bad_k", 5, "the contracted extents disagree: A stored (3, 5) gives 5, B stored (4, 4) gives 4" },
        { "bad_c", 5, "C is stored (3, 3) but the product" },
        { "f64_lane", 10, "n_samples 10 is not divisible by lane_fanin 3" },
        { "f64_lane", 0, "the extent 0 (n_samples 0) is outside 1 .." } };
    for (size_t i = 0; i < sizeof(plans) / sizeof(plans[0]); ++i) {
        eg_unit* unit = NULL;
        eg_plan* plan = NULL;
        char what[96];
        snprintf(what, sizeof(what), "eg_plan_new of '%s' at %d lanes refused BAD_REQUEST naming the defect", plans[i].unit, (int)plans[i].n);
        check(load(ctx, path, plans[i].unit, &unit) == EG_OK &&
                  refused(plan_new(ctx, unit, EG_STRUCTURE_HOST_TEAM, plans[i].n, 0, NULL, NULL, &plan), EG_BAD_REQUEST, ctx, plans[i].reason),
              what);
        if (unit != NULL) eg_unit_free(ctx, unit);
    }
    /* eg_bind: a plane that is not a whole number of rows, one whose row stride is under the stated columns, one of another width */
    eg_unit* unit = NULL;
    eg_plan* plan = NULL;
    check(load(ctx, path, "f64_rect", &unit) == EG_OK && plan_new(ctx, unit, EG_STRUCTURE_HOST_TEAM, 1, 0, NULL, NULL, &plan) == EG_OK, "a plan of f64_rect to bind");
    struct { int64_t rows, cols; int bits; const char* reason; } binds[] = {
        { 2, 7, 64, "a plane of 14 elements is not a whole number of rows of its 3 stated rows" },
        { 3, 4, 64, "the row stride 4 is smaller than the 5 stated columns" },
        { 3, 5, 32, "dtype (code 2, bits 32) is not the slot's" } };
    for (size_t i = 0; plan != NULL && i < sizeof(binds) / sizeof(binds[0]); ++i) {
        struct matrix m;
        memset(&m, 0, sizeof(m));
        m.bits = binds[i].bits;
        m.data = m.host = calloc((size_t)(binds[i].rows * binds[i].cols), 8);
        eg_plane* p = wrap(ctx, &m, binds[i].rows, binds[i].cols);
        struct eg_bind_desc b;
        const int32_t slot = 0;
        memset(&b, 0, sizeof(b));
        b.struct_size = sizeof(b);
        b.n_planes = 1;
        b.plane_slots = &slot;
        b.planes = &p;
        check(p != NULL && refused(eg_bind(ctx, plan, &b), EG_BAD_REQUEST, ctx, binds[i].reason), binds[i].reason);
        if (p != NULL) eg_plane_free(ctx, p);
        free(m.host);
    }
    if (plan != NULL) eg_plan_free(ctx, plan);
    if (unit != NULL) eg_unit_free(ctx, unit);
}

#ifdef EG_GEMM_DEVICE

/* ---- DEVICE rows ----------------------------------------------------------------------------------------------- */

/* Run @p name on the host (the oracle) and on the device from the same data; the worst difference, or -1 on failure. */
static double device_against_host(eg_ctx* host, eg_ctx* dev, const char* path, const char* name, int64_t n, int slack, struct variant* v,
                                  void** device_bits, int* count_out) {
    struct variant eager = { 0 };
    struct matrix hm[8], dm[8];
    int count = 0;
    const struct eg_unit_meta* meta = unit_meta(name);
    shapes_of(meta, n, slack, hm, &count);
    shapes_of(meta, n, slack, dm, &count);
    for (int i = 0; i < count; ++i) {
        fill(&hm[i], 11u + (unsigned)i);
        dm[i].host = malloc((size_t)dm[i].elements * (size_t)dm[i].bits / 8);
        memcpy(dm[i].host, hm[i].host, (size_t)dm[i].elements * (size_t)dm[i].bits / 8);
    }
    eg_unit *hu = NULL, *du = NULL;
    eg_plan *hp = NULL, *dp = NULL;
    double worst = -1.0;
    if (load(host, path, name, &hu) == EG_OK && load(dev, path, name, &du) == EG_OK &&
        run(host, hu, EG_STRUCTURE_HOST_TEAM, hm, count, n, &eager, &hp) == EG_OK &&
        run(dev, du, EG_STRUCTURE_DEVICE, dm, count, n, v, &dp) == EG_OK) {
        void *want[8], *got[8];
        read_back(hm, count, want);
        read_back(dm, count, got);
        worst = worst_difference(hm, count, got, want);
        if (device_bits != NULL) memcpy(device_bits, got, sizeof(void*) * (size_t)count), *count_out = count;
        else for (int i = 0; i < count; ++i) free(got[i]);
        for (int i = 0; i < count; ++i) free(want[i]);
    } else {
        reason_has(dev, "(device run failed)"), reason_has(host, "(host run failed)");
    }
    release(host, hp, hm, count);
    release(dev, dp, dm, count);
    if (hu != NULL) eg_unit_free(host, hu);
    if (du != NULL) eg_unit_free(dev, du);
    return worst;
}

static void device_rows(eg_ctx* host, eg_ctx* dev, const char* path) {
    static const char* const shapes[] = { "sq", "rect", "ta", "tb", "tc", "all", "lane", "wide", "vjp" };
    for (int w = 0; w < 2; ++w) {
        int ok = 1;
        double worst = 0.0;
        struct variant routed = { 0 };
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
            char name[16];
            snprintf(name, sizeof(name), "%s_%s", w ? "f32" : "f64", shapes[s]);
            const double d = device_against_host(host, dev, path, name, strcmp(shapes[s], "lane") == 0 ? 12 : 7, strcmp(shapes[s], "wide") == 0 ? 2 : 0, &routed, NULL, NULL);
            ok = ok && (routed.entry & EG_ENTRY_GEMM) != 0;
            if (d < 0 || d > (w ? 1e-5 : 1e-12)) printf("      %s: worst difference %g\n", name, d);
            ok = ok && d >= 0 && d <= (w ? 1e-5 : 1e-12);
            worst = d > worst ? d : worst;
        }
        char what[256];
        snprintf(what, sizeof(what), "%s: every gemm unit (square, non-square, TRANS_A/B/C, PER_LANE, column slice, 2-call vjp with ACCUMULATE) on the device == the host loop (worst %g), each routed to cuBLAS (EG_ENTRY_GEMM set)",
                 w ? "float32" : "float64", worst);
        check(ok, what);
    }
    /* a recording replays to the bits of the eager launch, for a forward unit and for the accumulating vjp */
    for (int v = 0; v < 2; ++v) {
        const char* name = v ? "f64_vjp" : "f32_all";
        void *eager[8], *replay[8];
        int ce = 0, cr = 0;
        struct variant once = { 0 }, recorded = { .record = 1 };
        const double a = device_against_host(host, dev, path, name, 7, 0, &once, eager, &ce);
        const double b = device_against_host(host, dev, path, name, 7, 0, &recorded, replay, &cr);
        int same = a >= 0 && b >= 0 && ce == cr;
        struct matrix m[8];
        int count = 0;
        shapes_of(unit_meta(name), 7, 0, m, &count);
        for (int i = 0; same && i < count; ++i) same = memcmp(eager[i], replay[i], (size_t)m[i].elements * (size_t)m[i].bits / 8) == 0;
        for (int i = 0; i < ce; ++i) free(eager[i]);
        for (int i = 0; i < cr; ++i) free(replay[i]);
        char what[96];
        snprintf(what, sizeof(what), "%s recorded with eg_record_begin/end replays to the same bits as the eager launch", name);
        check(same, what);
    }
    /* a stream the caller gave: the handle is made at eg_stream_use, the launch runs on it */
    cudaStream_t stream;
    check(cudaStreamCreate(&stream) == cudaSuccess && eg_stream_use(dev, (uint64_t)(uintptr_t)stream) == EG_OK, "eg_stream_use of a caller's stream after a gemm plan");
    struct variant once = { 0 };
    const double on_stream = device_against_host(host, dev, path, "f32_rect", 7, 0, &once, NULL, NULL);
    check(on_stream >= 0 && on_stream <= 1e-5, "a gemm launch on the caller's stream == the host loop");
    eg_stream_use(dev, 0);
    cudaStreamDestroy(stream);
    /* a DEVICE plan the door cannot route launches the unit's own kernel: EG_ENTRY_GEMM clear, the host plan's values */
    const int64_t base[2] = { 0, 4 }, count[2] = { 4, 3 };
    struct { const char* what; const char* unit; struct variant v; } kernels[] = {
        { "two partitions", "f64_rect", { 0, 0, 2, base, count, 0 } },
        { "an index set", "gemm_index", { 0, 0, 0, NULL, NULL, 0 } },
        { "a device runner's loop (a finishing unit)", "gemm_finish", { 0, 1, 0, NULL, NULL, 0 } } };
    for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
        const double d = device_against_host(host, dev, path, kernels[i].unit, 7, 0, &kernels[i].v, NULL, NULL);
        const int routed_entry = (kernels[i].v.entry & EG_ENTRY_GEMM) != 0;
        char what[160];
        snprintf(what, sizeof(what), "%s on %s runs the unit's kernel == the host plan (worst %g)%s", kernels[i].unit, kernels[i].what, d,
                 kernels[i].v.runner ? "" : ", EG_ENTRY_GEMM clear");
        check(d >= 0 && d <= 1e-12 && (kernels[i].v.runner || !routed_entry), what);
    }
}

#ifdef EG_GEMM_ROUTE

/* ---- ROUTE rows: which path a routable device plan takes, by the context's route table ----------------------------- */

/* The test seam (test/gemm_route_seam.cpp): rows of (target, family, model, dtype_bits, route) read into a fresh device context. */
eg_status gemm_route_rows_set(eg_ctx* ctx, int (*rows)[5], int count);

enum { ROUTE_M = 3, ROUTE_K = 5, ROUTE_N = 4 };   /* the shapes of the "rect" units: C (3, 4) = A (3, 5) B (5, 4) */

/* Run @p name (a rect unit, one DEVICE plan) on a fresh device context whose route table is @p rows; ``*entry`` gets the plan's EG_ENTRY_* flags, ``got`` the
 * C plane and ``scale`` the sum over k of |a b| of each element of C. Returns 0, or a nonzero status. */
static int route_run(const char* path, const char* name, int (*rows)[5], int count, uint32_t* entry, double* got, double* scale) {
    eg_ctx* ctx = context(1);
    if (ctx == NULL) return 1;
    struct variant v = { 0 };
    struct matrix m[8];
    int n_planes = 0;
    shapes_of(unit_meta(name), 7, 0, m, &n_planes);
    for (int i = 0; i < n_planes; ++i) fill(&m[i], 11u + (unsigned)i);
    const int bits = m[0].bits;
    eg_unit* unit = NULL;
    eg_plan* plan = NULL;
    int bad = gemm_route_rows_set(ctx, rows, count) != EG_OK || load(ctx, path, name, &unit) != EG_OK ||
              run(ctx, unit, EG_STRUCTURE_DEVICE, m, n_planes, 7, &v, &plan) != EG_OK;
    void* back[8];
    if (!bad) read_back(m, n_planes, back);
    for (int r = 0; !bad && r < ROUTE_M; ++r)
        for (int c = 0; c < ROUTE_N; ++c) {
            double sum = 0.0;
            for (int p = 0; p < ROUTE_K; ++p) {
                const double a = bits == 32 ? ((float*)m[0].host)[r * ROUTE_K + p] : ((double*)m[0].host)[r * ROUTE_K + p];
                const double b = bits == 32 ? ((float*)m[1].host)[p * ROUTE_N + c] : ((double*)m[1].host)[p * ROUTE_N + c];
                sum += fabs(a * b);
            }
            scale[r * ROUTE_N + c] = sum;
            got[r * ROUTE_N + c] = bits == 32 ? ((float*)back[2])[r * ROUTE_N + c] : ((double*)back[2])[r * ROUTE_N + c];
        }
    for (int i = 0; !bad && i < n_planes; ++i) free(back[i]);
    if (!bad) *entry = v.entry;
    release(ctx, plan, m, n_planes);
    if (unit != NULL) eg_unit_free(ctx, unit);
    eg_ctx_free(ctx);
    return bad;
}

/* The device class of this machine, as the context states the key it looked its rows up by. */
static void route_class(int* family, int* model) {
    eg_ctx* ctx = context(1);
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    *family = *model = -1;
    if (ctx != NULL && eg_ctx_info(ctx, &info) == EG_OK) *family = info.weight_family, *model = info.weight_model;
    if (ctx != NULL) eg_ctx_free(ctx);
}

static void route_rows(const char* path) {
    int fam, mod;
    route_class(&fam, &mod);
    check(fam > 0, "the device context states its class key (the compute capability) for the route rows");
    static const char* const names[2] = { "f32_rect", "f64_rect" };
    /* Rows: each case sets, for the width under test, one class row over built-in rows of the case's route; the other width keeps the built-in route. */
    struct { const char* what; int builtin; int class_route; int class_family, class_model; int expect_gemm; } cases[] = {
        { "a direct row of this device's class runs the unit's own kernel", 1, 0, fam, mod, 0 },
        { "a gemm row of this device's class over direct built-in rows routes to cuBLAS", 0, 1, fam, mod, 1 },
        { "a direct row of another family takes the built-in gemm route", 1, 0, fam + 1, mod, 1 },
        { "a direct row of another model takes the built-in gemm route", 1, 0, fam, mod + 1, 1 } };
    for (int w = 0; w < 2; ++w) {
        const int bits = w ? 64 : 32;
        const double eps = w ? DBL_EPSILON : FLT_EPSILON;
        double want[ROUTE_M * ROUTE_N], scale[ROUTE_M * ROUTE_N];
        uint32_t entry = 0;
        int base[2][5] = { { 1, 0, 0, 32, 1 }, { 1, 0, 0, 64, 1 } };
        check(route_run(path, names[w], base, 2, &entry, want, scale) == 0 && (entry & EG_ENTRY_GEMM) != 0, "the built-in rows route a rect unit to cuBLAS (reference result)");
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
            int rows[3][5] = { { 1, 0, 0, 32, cases[c].builtin }, { 1, 0, 0, 64, cases[c].builtin }, { 1, cases[c].class_family, cases[c].class_model, bits, cases[c].class_route } };
            double got[ROUTE_M * ROUTE_N], other[ROUTE_M * ROUTE_N], other_scale[ROUTE_M * ROUTE_N];
            uint32_t other_entry = 0;
            entry = 0;
            /* dtype separation: the other width's unit sees only the built-in row, whatever this width's class row says */
            const int other_route = route_run(path, names[1 - w], rows, 3, &other_entry, other, other_scale) == 0 && ((other_entry & EG_ENTRY_GEMM) != 0) == (cases[c].builtin == 1);
            int ok = route_run(path, names[w], rows, 3, &entry, got, scale) == 0 && other_route && ((entry & EG_ENTRY_GEMM) != 0) == cases[c].expect_gemm &&
                     ((entry & EG_ENTRY_DEVICE) != 0) == !cases[c].expect_gemm;
            /* two different computations (cuBLAS and the unit's kernel): within (k - 1) eps sum |a b| per element, not bitwise */
            double worst = 0.0;
            for (int i = 0; ok && i < ROUTE_M * ROUTE_N; ++i) {
                const double bound = (ROUTE_K - 1) * eps * scale[i];
                const double ratio = bound > 0 ? fabs(got[i] - want[i]) / bound : 0.0;
                worst = ratio > worst ? ratio : worst;
                ok = fabs(got[i] - want[i]) <= bound;
            }
            char what[256];
            snprintf(what, sizeof(what), "float%d: %s (EG_ENTRY %s, the other width unmoved, result within (k-1) eps sum|ab| of the cuBLAS one, worst %.3f of the bound)", bits,
                     cases[c].what, cases[c].expect_gemm ? "GEMM" : "DEVICE", worst);
            check(ok, what);
        }
    }
    eg_ctx* ctx = context(1);
    int no32[1][5] = { { 1, 0, 0, 64, 1 } };
    int twice[4][5] = { { 1, 0, 0, 32, 1 }, { 1, 0, 0, 64, 1 }, { 1, 7, 5, 32, 0 }, { 1, 7, 5, 32, 1 } };
    check(ctx != NULL && refused(gemm_route_rows_set(ctx, no32, 1), EG_FAULT, ctx, "a float width has no built-in row (family 0, model 0)"),
          "a route table with no built-in row for a width is refused EG_FAULT naming it, never routed silently");
    check(ctx != NULL && refused(gemm_route_rows_set(ctx, twice, 4), EG_FAULT, ctx, "two rows have one class and width"),
          "a route table with two rows of one class and width is refused EG_FAULT naming it");
    if (ctx != NULL) eg_ctx_free(ctx);
}

#endif /* EG_GEMM_ROUTE */

/* A host plane handed to a device product is refused by name. */
static void direct_device_refusal(eg_ctx* dev) {
    struct eg_gemm_desc d = { sizeof(d), 0, 64, 0, 3, 5, 5, 4, 3, 4 };
    eg_gemm* g = NULL;
    struct matrix a, b, c;
    eg_plane *pa = plain(dev, &a, 64, 0, 3, 5) ? a.plane : NULL, *pb = plain(dev, &b, 64, 0, 5, 4) ? b.plane : NULL, *pc = plain(dev, &c, 64, 0, 3, 4) ? c.plane : NULL;
    check(eg_gemm_new(dev, &d, &g) == EG_OK && refused(eg_gemm_run(dev, g, pa, pb, pc), EG_BAD_REQUEST, dev, "eg_gemm_run a: the product runs on the device and this plane is on the host"),
          "eg_gemm_run of a device product over a host plane: BAD_REQUEST naming the target");
    eg_gemm_free(dev, g);
    eg_plane_free(dev, pa), eg_plane_free(dev, pb), eg_plane_free(dev, pc);
    free(a.host), free(b.host), free(c.host);
}

#endif /* EG_GEMM_DEVICE */

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: gemm_test <libfixture_gemm.so> [--no-cublas]\n");
        return 2;
    }
    void* fixture = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
    eg_unit_library_meta_fn meta = NULL;
    if (fixture != NULL) *(void**)&meta = dlsym(fixture, EG_UNIT_LIBRARY_META_SYMBOL);
    if (meta == NULL) {
        fprintf(stderr, "cannot read the fixture's meta\n");
        return 2;
    }
    library = meta();
    eg_ctx* host = context(0);
    if (host == NULL) return 2;
#ifdef EG_GEMM_DEVICE
    eg_ctx* dev = context(1);
    if (dev == NULL) {
        fprintf(stderr, "no device context\n");
        return 2;
    }
    if (argc > 2 && strcmp(argv[2], "--no-cublas") == 0) {
        eg_unit* unit = NULL;
        eg_plan* plan = NULL;
        check(load(dev, argv[1], "f32_rect", &unit) == EG_OK &&
                  refused(plan_new(dev, unit, EG_STRUCTURE_DEVICE, 7, 0, NULL, NULL, &plan), EG_FAULT, dev, "tried libcublas.so.13: does not load; libcublas.so.12: does not load"),
              "a DEVICE plan of a gemm unit with no cuBLAS is EG_FAULT naming both libraries tried");
        eg_unit_free(dev, unit);
        struct eg_gemm_desc d = { sizeof(d), 0, 64, 0, 3, 5, 5, 4, 3, 4 };
        eg_gemm* g = NULL;
        check(refused(eg_gemm_new(dev, &d, &g), EG_FAULT, dev, "tried libcublas.so.13: does not load; libcublas.so.12: does not load") && g == NULL,
              "eg_gemm_new with no cuBLAS is EG_FAULT naming both libraries tried");
        const int64_t base[2] = { 0, 4 }, count[2] = { 4, 3 };
        struct variant two = { 0, 0, 2, base, count, 0 };
        check(device_against_host(host, dev, argv[1], "f64_rect", 7, 0, &two, NULL, NULL) >= 0, "a DEVICE plan the door cannot route runs its kernel with no cuBLAS");
        eg_ctx_free(dev);
        eg_ctx_free(host);
        printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
        return failures != 0;
    }
#endif
    host_loop_rows(host, argv[1]);
    refusal_rows(host, argv[1]);
    direct_refusals(host);
#ifndef EG_GEMM_DEVICE
    direct_rows(host, NULL);
#endif
#ifdef EG_GEMM_DEVICE
    device_rows(host, dev, argv[1]);
    direct_rows(host, dev);
    direct_device_refusal(dev);
#ifdef EG_GEMM_ROUTE
    route_rows(argv[1]);
#endif
    eg_ctx_free(dev);
#endif
    eg_ctx_free(host);
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures != 0;
}
