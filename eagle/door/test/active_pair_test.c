/*
 * The device runner over an active set: a recorded finishing step unit read through the active-set map, at one
 * step per launch (_osc) and at four (_osc_x4), run until done by eg_runner_new / eg_run over DEVICE plans
 * (launches over the live prefix, a compaction every ``every`` steps when a sample finished). Planes, mask and
 * report counts are compared BITWISE with the old device runner's on the same inputs, over five cases (48 and 1000
 * samples, a cap that finishes every sample and one that does not, the default cadence, every 4 and every 32).
 * Beside the oracle: the closed form of the run (iterations, steps, mapped, finished, compactions and each
 * sample's final time), a second run of a finished batch, and what a device active set refuses by name.
 *
 * usage: active_pair_test <_osc library> <_osc digest hex> <_osc_x4 library> <_osc_x4 digest hex> <oracle dir>
 *                          <_gosc library> <_gosc digest hex>
 * oracle dir: in_{t_final,x,v}_<n>.f64 and <unit>_<case>_{x,v,t}.f64, _term.u8, _report.i64 (8 counts).
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DT 0.0625
enum { S_T, S_V, S_X, S_TFINAL, S_COUNT, S_MAP, S_COUNTER, S_TERM, S_DT };

static eg_plane* wrap(eg_ctx* ctx, struct tensor* t, int64_t n, void* data, uint8_t code, uint8_t bits) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = n;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCUDA;
    t->managed.dl_tensor.ndim = 1;
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

/* One case: its input arrays (host) and the device planes t, v, x, t_final (f64) and the mask (bool). */
struct run {
    int64_t n;
    double *tf, *x0, *v0;
    void* d[5];
    struct tensor t[5];
    eg_plane* p[5];
};

static int run_new(eg_ctx* ctx, struct run* s, const char* dir, int64_t n) {
    char file[64];
    s->n = n;
    s->tf = malloc(n * sizeof(double));
    s->x0 = malloc(n * sizeof(double));
    s->v0 = malloc(n * sizeof(double));
    double* zeros = calloc(n, sizeof(double));
    uint8_t* none = calloc(n, 1);
    int ok = s->tf && s->x0 && s->v0 && zeros && none;
    snprintf(file, sizeof(file), "in_t_final_%lld.f64", (long long)n);
    ok = ok && read_file(dir, file, s->tf, n * sizeof(double));
    snprintf(file, sizeof(file), "in_x_%lld.f64", (long long)n);
    ok = ok && read_file(dir, file, s->x0, n * sizeof(double));
    snprintf(file, sizeof(file), "in_v_%lld.f64", (long long)n);
    ok = ok && read_file(dir, file, s->v0, n * sizeof(double));
    const void* init[5] = { zeros, s->v0, s->x0, s->tf, none };
    for (int i = 0; ok && i < 5; ++i) {
        const size_t bytes = i < 4 ? n * sizeof(double) : (size_t)n;
        ok = cudaMalloc(&s->d[i], bytes) == cudaSuccess && cudaMemcpy(s->d[i], init[i], bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        s->p[i] = ok ? wrap(ctx, &s->t[i], n, s->d[i], i < 4 ? kDLFloat : kDLBool, i < 4 ? 64 : 8) : NULL;
        ok = ok && s->p[i] != NULL;
    }
    free(zeros);
    free(none);
    return ok;
}

static void run_free(eg_ctx* ctx, struct run* s) {
    for (int i = 0; i < 5; ++i) {
        eg_plane_free(ctx, s->p[i]);
        cudaFree(s->d[i]);
    }
    free(s->tf);
    free(s->x0);
    free(s->v0);
}

static eg_status bind(eg_ctx* ctx, eg_plan* plan, const struct run* s) {
    const int32_t slots[5] = { S_T, S_V, S_X, S_TFINAL, S_TERM }, uslot[1] = { S_DT };
    const double dt = DT;
    const void* values[1] = { &dt };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 5;
    b.plane_slots = slots;
    b.planes = (eg_plane* const*)s->p;
    b.n_uniforms = 1;
    b.uniform_slots = uslot;
    b.uniform_values = values;
    return eg_bind(ctx, plan, &b);
}

static struct eg_runner_desc runner_desc(eg_plan* const* plan, int64_t max_steps, int64_t every, double theta) {
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_plans = 1;
    d.plans = plan;
    d.max_steps = max_steps;
    d.every = every;
    if (theta > 0.0) {
        d.flags = EG_RUN_REORDER;
        d.reorder_theta = theta;
    }
    d.steps_reverse = EG_STEPS_REVERSE_PER_STEP;
    d.events = EG_EVENTS_STEP;
    d.on_no_event = EG_NO_EVENT_REFUSE;
    return d;
}

static int64_t roundup(int64_t v, int64_t k) { return (v + k - 1) / k * k; }

/* The closed form of the run: the iterations, the finished count after each, the compactions. */
struct closed {
    int64_t iterations, finished, compactions;
};

static struct closed closed_form(const struct run* s, int64_t k, int64_t every, int64_t max_steps) {
    struct closed c = { 0, 0, 0 };
    const int64_t cap = (max_steps + every - 1) / every;
    int64_t last = 0;
    while (c.iterations < cap && last != s->n) {
        ++c.iterations;
        int64_t fin = 0;
        for (int64_t i = 0; i < s->n; ++i) fin += roundup((int64_t)llround(s->tf[i] / DT), k) <= c.iterations * every;
        c.compactions += fin != last;
        last = fin;
    }
    c.finished = last;
    return c;
}

static void one_case(eg_ctx* ctx, eg_unit* unit, const char* name, int64_t k, const char* dir, const char* cname, int64_t n,
                     int64_t max_steps, int64_t every, double theta, int first) {
    char what[320], file[160];
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = unit;
    pd.structure = EG_STRUCTURE_DEVICE;
    pd.n_samples = n;
    eg_plan* plan = NULL;
    struct run s;
    memset(&s, 0, sizeof(s));
    int ok = eg_plan_new(ctx, &pd, &plan) == EG_OK && run_new(ctx, &s, dir, n) && bind(ctx, plan, &s) == EG_OK;
    eg_runner* runner = NULL;
    const struct eg_runner_desc rd = runner_desc(&plan, max_steps, every, theta);
    if (first && ok) {
        eg_runner* other = NULL;
        struct eg_runner_desc bad = rd;
        bad.flags = EG_RUN_REORDER;
        bad.reorder_theta = 0.9;
        check(eg_runner_new(ctx, &bad, &other) == EG_BAD_REQUEST && reason_has(ctx, "reorder_theta") && other == NULL,
              "device active set: EG_RUN_REORDER with a theta outside [0.25, 0.75] refused BAD_REQUEST");
        bad = rd;
        bad.every = 2;
        check(eg_runner_new(ctx, &bad, &other) == EG_BAD_REQUEST && reason_has(ctx, "every 2"),
              "device active set: every 2 refused BAD_REQUEST (not a cadence)");
        bad.every = 4 * k + 2;
        if (k > 1)
            check(eg_runner_new(ctx, &bad, &other) == EG_BAD_REQUEST && reason_has(ctx, "whole number of launches"),
                  "device active set: every that is not a whole number of launches refused BAD_REQUEST");
    }
    ok = ok && eg_runner_new(ctx, &rd, &runner) == EG_OK;
    struct eg_run_report r;
    memset(&r, 0, sizeof(r));
    r.struct_size = sizeof(r);
    ok = ok && eg_run(ctx, runner, &r) == EG_OK;
    snprintf(what, sizeof(what), "%s %s: DEVICE plan over an active set, eg_runner_new and eg_run OK", name, cname);
    check(ok, what);
    if (!ok) {
        reason_has(ctx, "\x01");
        return;
    }
    static const char* names[3] = { "t", "v", "x" };
    for (int i = 0; i < 3; ++i) {
        double* got = malloc(n * sizeof(double));
        double* want = malloc(n * sizeof(double));
        snprintf(file, sizeof(file), "%s_%s_%s.f64", name, cname, names[i]);
        const int good = read_file(dir, file, want, n * sizeof(double)) &&
                         cudaMemcpy(got, s.d[i], n * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
        snprintf(what, sizeof(what), "%s %s: %s bitwise vs the old device runner", name, cname, names[i]);
        check(good && memcmp(got, want, n * sizeof(double)) == 0, what);
        free(got);
        free(want);
    }
    uint8_t* got = malloc(n);
    uint8_t* want = malloc(n);
    snprintf(file, sizeof(file), "%s_%s_term.u8", name, cname);
    int good = read_file(dir, file, want, n) && cudaMemcpy(got, s.d[4], n, cudaMemcpyDeviceToHost) == cudaSuccess;
    for (int64_t i = 0; good && i < n; ++i) good = (got[i] != 0) == (want[i] != 0);
    snprintf(what, sizeof(what), "%s %s: the mask equals the old device runner's", name, cname);
    check(good, what);
    free(got);
    free(want);
    int64_t rep[9];
    snprintf(file, sizeof(file), "%s_%s_report.i64", name, cname);
    good = read_file(dir, file, rep, sizeof(rep));
    const int64_t have[9] = { r.n, r.finished, r.launches, r.steps, r.mapped, (r.flags & EG_REPORT_DONE) != 0,
                              (r.flags & EG_REPORT_EXACT_STEPS) != 0, r.compactions, r.reorders };
    snprintf(what, sizeof(what), "%s %s: report n %lld finished %lld launches %lld steps %lld mapped %lld done %lld exact %lld "
             "compactions %lld reorders %lld equals the old device runner's, mode BAND", name, cname, (long long)have[0], (long long)have[1],
             (long long)have[2], (long long)have[3], (long long)have[4], (long long)have[5], (long long)have[6], (long long)have[7], (long long)have[8]);
    check(good && memcmp(rep, have, sizeof(rep)) == 0 && r.mode == EG_MODE_BAND && r.build_s > 0.0, what);
    snprintf(what, sizeof(what), "%s %s: a reorder fires over at least 4096 samples and never below (reorders %lld, compactions %lld)", name, cname,
             (long long)r.reorders, (long long)r.compactions);
    check(theta > 0.0 && n >= 4096 ? r.reorders > 0 && r.reorders <= r.compactions : r.reorders == 0, what);
    /* the closed form: iterations, steps, mapped, finished, compactions and each sample's final time */
    const int64_t e = every != 0 ? every : k * ((16 + k - 1) / k);
    const struct closed c = closed_form(&s, k, e, max_steps);
    double* t = malloc(n * sizeof(double));
    int t_ok = cudaMemcpy(t, s.d[0], n * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
    for (int64_t i = 0; t_ok && i < n; ++i) {
        const int64_t steps = (int64_t)llround(s.tf[i] / DT);
        t_ok = t[i] == (double)(steps < c.iterations * e ? steps : c.iterations * e) * DT;
    }
    free(t);
    snprintf(what, sizeof(what), "%s %s: closed form, iterations %lld finished %lld compactions %lld, t = min(stop, iterations*every)*dt",
             name, cname, (long long)c.iterations, (long long)c.finished, (long long)c.compactions);
    check(r.launches == c.iterations && r.finished == c.finished && r.steps == c.iterations * e && r.mapped == c.iterations * n &&
              r.compactions == c.compactions && t_ok, what);
    if (first) {
        struct eg_run_report again;
        memset(&again, 0, sizeof(again));
        again.struct_size = sizeof(again);
        snprintf(what, sizeof(what), "%s: a second run of a finished batch launches and compacts nothing", name);
        check(eg_run(ctx, runner, &again) == EG_OK && again.launches == 0 && again.compactions == 0 && again.finished == n &&
                  (again.flags & EG_REPORT_DONE), what);
        eg_runner* other = NULL;
        check(eg_runner_new(ctx, &rd, &other) == EG_BAD_REQUEST && reason_has(ctx, "already reads the active set") && other == NULL,
              "a second runner over plans that read an active set: BAD_REQUEST");
    }
    check(eg_runner_free(ctx, runner) == EG_OK && eg_plan_free(ctx, plan) == EG_OK, "eg_runner_free and eg_plan_free");
    run_free(ctx, &s);
}

static void run_unit(eg_ctx* ctx, const char* lib, const char* name, const char* hex, int64_t k, const char* dir) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    struct eg_unit_load_desc ld;
    memset(&ld, 0, sizeof(ld));
    ld.struct_size = sizeof(ld);
    ld.path = lib;
    ld.unit = &ref;
    eg_unit* unit = NULL;
    char what[256];
    snprintf(what, sizeof(what), "%s: eg_unit_load of the recorded active-set unit (PTX image) on the device context", name);
    const eg_status ls = parse_digest(hex, ref.digest) ? eg_unit_load(ctx, &ld, &unit) : EG_BAD_REQUEST;
    if (ls != EG_OK) reason_has(ctx, "\x01");
    check(ls == EG_OK, what);
    if (unit == NULL) return;
    one_case(ctx, unit, name, k, dir, "48_400_d", 48, 400, 0, 0.0, 1);
    one_case(ctx, unit, name, k, dir, "48_20_d", 48, 20, 0, 0.0, 0);
    one_case(ctx, unit, name, k, dir, "1000_400_d", 1000, 400, 0, 0.0, 0);
    one_case(ctx, unit, name, k, dir, "1000_400_e4", 1000, 400, 4, 0.0, 0);
    one_case(ctx, unit, name, k, dir, "1000_400_e32", 1000, 400, 32, 0.0, 0);
    one_case(ctx, unit, name, k, dir, "1000_400_d_t50", 1000, 400, 0, 0.5, 0);
    one_case(ctx, unit, name, k, dir, "20000_400_d_t50", 20000, 400, 0, 0.5, 0);
    one_case(ctx, unit, name, k, dir, "20000_40_d_t50", 20000, 40, 0, 0.5, 0);
    one_case(ctx, unit, name, k, dir, "5000_400_e4_t25", 5000, 400, 4, 0.25, 0);
    one_case(ctx, unit, name, k, dir, "5000_400_e4_t75", 5000, 400, 4, 0.75, 0);
    eg_unit_free(ctx, unit);
}

/* A unit over groups of 32 lanes: a runner over its plan is refused by name. */
static void group_refused(eg_ctx* ctx, const char* lib, const char* hex) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = "_gosc";
    struct eg_unit_load_desc ld;
    memset(&ld, 0, sizeof(ld));
    ld.struct_size = sizeof(ld);
    ld.path = lib;
    ld.unit = &ref;
    eg_unit* unit = NULL;
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.structure = EG_STRUCTURE_DEVICE;
    pd.n_samples = 100;
    eg_plan* plan = NULL;
    const int ok = parse_digest(hex, ref.digest) && eg_unit_load(ctx, &ld, &unit) == EG_OK && (pd.unit = unit) != NULL &&
                   eg_plan_new(ctx, &pd, &plan) == EG_OK;
    const struct eg_runner_desc rd = runner_desc(&plan, 400, 0, 0.0);
    eg_runner* runner = NULL;
    check(ok && eg_runner_new(ctx, &rd, &runner) == EG_NOT_WRITTEN && reason_has(ctx, "groups of 32 lanes") && runner == NULL,
          "device active set over groups of 32 lanes: refused NOT_WRITTEN by name");
    if (plan != NULL) eg_plan_free(ctx, plan);
    if (unit != NULL) eg_unit_free(ctx, unit);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <_osc lib> <_osc digest> <_osc_x4 lib> <_osc_x4 digest> <oracle dir> <_gosc lib> <_gosc digest>\n", argv[0]);
        return 2;
    }
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.flags = EG_CTX_DEVICE;
    st.device = 0;
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    run_unit(ctx, argv[1], "_osc", argv[2], 1, argv[5]);
    run_unit(ctx, argv[3], "_osc_x4", argv[4], 4, argv[5]);
    group_refused(ctx, argv[6], argv[7]);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
