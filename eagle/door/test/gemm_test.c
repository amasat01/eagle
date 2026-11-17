/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The gemm unit record through the door. HOST rows (always): every load, plan and bind refusal of the record by status
 * and reason, and a host plan of each gemm unit running its own loop (the oracle) without cuBLAS ever being opened.
 * DEVICE rows (-DEG_GEMM_DEVICE, run through the GPU pool): each unit's DEVICE launch against that loop within the
 * width's tolerance, the replay of a recorded launch against the eager one bit for bit, a launch on a stream the
 * caller gave, and the DEVICE-only refusals. With "--no-cublas" (run with libno_cublas.so preloaded) only the missing
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
    m->tensor.managed.dl_tensor.ndim = 2;
    m->tensor.managed.dl_tensor.dtype.code = kDLFloat;
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
}

static void fill(struct matrix* m, unsigned seed) {
    unsigned state = seed;
    m->host = malloc((size_t)m->elements * (size_t)m->bits / 8);
    for (int64_t i = 0; i < m->elements; ++i) {
        const double v = sample(&state);
        if (m->bits == 32) ((float*)m->host)[i] = (float)v;
        else ((double*)m->host)[i] = v;
    }
}

/* Run @p u over @p n lanes on a context: host planes of the test's own data, or device copies of them. Returns the
 * planes (outputs hold the result after eg_sync and a read back); @p matrices hold the initial data in ``host``. */
static eg_status run(eg_ctx* ctx, eg_unit* unit, eg_structure s, struct matrix* matrices, int count, int64_t n, int record,
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
    eg_status st = plan_new(ctx, unit, s, n, 0, NULL, NULL, plan_out);
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = count;
    b.plane_slots = slots;
    b.planes = planes;
    if (st == EG_OK) st = eg_bind(ctx, *plan_out, &b);
    eg_graph* graph = NULL;
    if (st == EG_OK && record) st = eg_record_begin(ctx);
    if (st == EG_OK) st = eg_launch(ctx, *plan_out, NULL);
    if (st == EG_OK && record) st = eg_record_end(ctx, &graph);
    if (st == EG_OK && record) st = eg_graph_launch(ctx, graph);
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
        for (int64_t j = 0; j < matrices[i].elements; ++j) {
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

/* ---- HOST rows ------------------------------------------------------------------------------------------------- */

static void host_loop_rows(eg_ctx* ctx, const char* path) {
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
        const eg_status st = load(ctx, path, names[u], &unit) == EG_OK ? run(ctx, unit, EG_STRUCTURE_HOST_TEAM, matrices, count, n, 0, &plan) : EG_FAULT;
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
        { "bad_banded", EG_NOT_WRITTEN, "a banded unit with a gemm record" } };
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
static double device_against_host(eg_ctx* host, eg_ctx* dev, const char* path, const char* name, int64_t n, int slack, int record,
                                  void** device_bits, int* count_out) {
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
        run(host, hu, EG_STRUCTURE_HOST_TEAM, hm, count, n, 0, &hp) == EG_OK &&
        run(dev, du, EG_STRUCTURE_DEVICE, dm, count, n, record, &dp) == EG_OK) {
        void *want[8], *got[8];
        read_back(hm, count, want);
        read_back(dm, count, got);
        worst = worst_difference(hm, count, got, want);
        if (device_bits != NULL) memcpy(device_bits, got, sizeof(void*) * (size_t)count), *count_out = count;
        else for (int i = 0; i < count; ++i) free(got[i]);
        for (int i = 0; i < count; ++i) free(want[i]);
    } else {
        reason_has(dev, "(device run failed)");
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
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
            char name[16];
            snprintf(name, sizeof(name), "%s_%s", w ? "f32" : "f64", shapes[s]);
            const double d = device_against_host(host, dev, path, name, strcmp(shapes[s], "lane") == 0 ? 12 : 7, strcmp(shapes[s], "wide") == 0 ? 2 : 0, 0, NULL, NULL);
            if (d < 0 || d > (w ? 1e-5 : 1e-12)) printf("      %s: worst difference %g\n", name, d);
            ok = ok && d >= 0 && d <= (w ? 1e-5 : 1e-12);
            worst = d > worst ? d : worst;
        }
        char what[256];
        snprintf(what, sizeof(what), "%s: every gemm unit (square, non-square, TRANS_A/B/C, PER_LANE, column slice, 2-call vjp with ACCUMULATE) on the device == the host loop (worst %g)",
                 w ? "float32" : "float64", worst);
        check(ok, what);
    }
    /* a recording replays to the bits of the eager launch, for a forward unit and for the accumulating vjp */
    for (int v = 0; v < 2; ++v) {
        const char* name = v ? "f64_vjp" : "f32_all";
        void *eager[8], *replay[8];
        int ce = 0, cr = 0;
        const double a = device_against_host(host, dev, path, name, 7, 0, 0, eager, &ce);
        const double b = device_against_host(host, dev, path, name, 7, 0, 1, replay, &cr);
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
    const double on_stream = device_against_host(host, dev, path, "f32_rect", 7, 0, 0, NULL, NULL);
    check(on_stream >= 0 && on_stream <= 1e-5, "a gemm launch on the caller's stream == the host loop");
    eg_stream_use(dev, 0);
    cudaStreamDestroy(stream);
    /* DEVICE-only refusals */
    eg_unit *unit = NULL, *idx = NULL, *fin = NULL;
    eg_plan* plan = NULL;
    const int64_t base[2] = { 0, 4 }, count[2] = { 4, 3 };
    check(load(dev, path, "f64_rect", &unit) == EG_OK &&
              refused(plan_new(dev, unit, EG_STRUCTURE_DEVICE, 7, 2, base, count, &plan), EG_NOT_WRITTEN, dev, "a gemm unit is one product per launch; several partitions is not written"),
          "a DEVICE plan of a gemm unit over two partitions refused NOT_WRITTEN by name");
    check(load(dev, path, "gemm_index", &idx) == EG_OK &&
              refused(plan_new(dev, idx, EG_STRUCTURE_DEVICE, 7, 0, NULL, NULL, &plan), EG_NOT_WRITTEN, dev, "an index set is not written"),
          "a DEVICE plan of a gemm unit with an index set refused NOT_WRITTEN by name");
    struct eg_runner_desc r;
    eg_runner* runner = NULL;
    memset(&r, 0, sizeof(r));
    check(load(dev, path, "gemm_finish", &fin) == EG_OK && plan_new(dev, fin, EG_STRUCTURE_DEVICE, 7, 0, NULL, NULL, &plan) == EG_OK, "a DEVICE plan of a gemm unit with a finish loop");
    r.struct_size = sizeof(r);
    r.n_plans = 1;
    r.plans = &plan;
    r.max_steps = 1;
    check(plan != NULL && refused(eg_runner_new(dev, &r, &runner), EG_NOT_WRITTEN, dev, "plans[0]: a gemm unit inside a device runner loop"),
          "eg_runner_new over a gemm unit refused NOT_WRITTEN by name");
    if (plan != NULL) eg_plan_free(dev, plan);
    eg_unit_free(dev, unit);
    eg_unit_free(dev, idx);
    eg_unit_free(dev, fin);
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
        eg_ctx_free(dev);
        eg_ctx_free(host);
        printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
        return failures != 0;
    }
#endif
    host_loop_rows(host, argv[1]);
    refusal_rows(host, argv[1]);
#ifdef EG_GEMM_DEVICE
    device_rows(host, dev, argv[1]);
    eg_ctx_free(dev);
#endif
    eg_ctx_free(host);
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures != 0;
}
