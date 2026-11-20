/*
 * The device runner pair: a recorded finishing step unit at one step per launch (osc) and at four (osc_x4), run
 * until done by eg_runner_new / eg_run over DEVICE plans, planes and report counts compared BITWISE with the
 * old device runner's (the band WHILE-graph loop) on the same inputs, at a cap that finishes every sample and
 * one that does not. Beside the oracle: the closed form of the run (t ends at t_final, the launches are
 * ceil(stop / K) capped, finished counts the stops inside the cap, a second run launches nothing), a rebind
 * between runs that the loop follows, and the regimes a device runner refuses by name.
 *
 * usage: runner_pair_test <osc library> <osc digest hex> <osc_x4 library> <osc_x4 digest hex> <oracle dir>
 * oracle dir: {t_final,x,v}_in.f64 and <unit>_<cap>_{x,v,t}.f64, _term.u8, _report.i64.
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 48
#define DT 0.0625
enum { S_T, S_V, S_X, S_TFINAL, S_COUNTER, S_TERM, S_DT };

static double t_final[N], x_in[N], v_in[N];

/* Device planes t, v, x, t_final (f64) and the mask (bool), at the recorded inputs. */
struct planes {
    void* d[5];
    struct tensor t[5];
    eg_plane* p[5];
};

static int planes_new(eg_ctx* ctx, struct planes* s) {
    static const double zeros[N];
    static const uint8_t none[N];
    const void* init[5] = { zeros, v_in, x_in, t_final, none };
    int ok = 1;
    for (int i = 0; i < 5; ++i) {
        const size_t bytes = i < 4 ? N * sizeof(double) : N;
        ok &= cudaMalloc(&s->d[i], bytes) == cudaSuccess && cudaMemcpy(s->d[i], init[i], bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        s->p[i] = ok ? wrap(ctx, &s->t[i], N, s->d[i], i < 4 ? kDLFloat : kDLBool, i < 4 ? 64 : 8) : NULL;
        ok &= s->p[i] != NULL;
    }
    return ok;
}

static void planes_free(eg_ctx* ctx, struct planes* s) {
    for (int i = 0; i < 5; ++i) {
        eg_plane_free(ctx, s->p[i]);
        cudaFree(s->d[i]);
    }
}

static eg_status bind(eg_ctx* ctx, eg_plan* plan, const struct planes* s) {
    const int32_t slots[5] = { S_T, S_V, S_X, S_TFINAL, S_TERM }, uslot[1] = { S_DT };
    const double dt = DT;
    const void* values[1] = { &dt };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 5;
    b.plane_slots = slots;
    b.planes = s->p;
    b.n_uniforms = 1;
    b.uniform_slots = uslot;
    b.uniform_values = values;
    return eg_bind(ctx, plan, &b);
}

/* The planes and report of one run against the old runner's files <unit>_<cap>_*. */
static void compare(const char* dir, const char* unit, int64_t cap, const struct planes* s, const struct eg_run_report* r,
                    const char* tag) {
    static const char* names[3] = { "t", "v", "x" };
    char file[256], what[256];
    for (int i = 0; i < 3; ++i) {
        double got[N], want[N];
        snprintf(file, sizeof(file), "%s_%lld_%s.f64", unit, (long long)cap, names[i]);
        const int ok = read_file(dir, file, want, sizeof(want)) && cudaMemcpy(got, s->d[i], sizeof(got), cudaMemcpyDeviceToHost) == cudaSuccess;
        snprintf(what, sizeof(what), "%s cap %lld%s: %s bitwise vs the old device runner", unit, (long long)cap, tag, names[i]);
        check(ok && memcmp(got, want, sizeof(got)) == 0, what);
    }
    uint8_t got[N], want[N];
    snprintf(file, sizeof(file), "%s_%lld_term.u8", unit, (long long)cap);
    int ok = read_file(dir, file, want, sizeof(want)) && cudaMemcpy(got, s->d[4], sizeof(got), cudaMemcpyDeviceToHost) == cudaSuccess;
    for (int i = 0; ok && i < N; ++i) ok = (got[i] != 0) == (want[i] != 0);
    snprintf(what, sizeof(what), "%s cap %lld%s: the mask equals the old device runner's", unit, (long long)cap, tag);
    check(ok, what);
    int64_t rep[7];
    snprintf(file, sizeof(file), "%s_%lld_report.i64", unit, (long long)cap);
    ok = read_file(dir, file, rep, sizeof(rep));
    const int64_t have[7] = { r->n, r->finished, r->launches, r->steps, r->mapped, (r->flags & EG_REPORT_DONE) != 0,
                              (r->flags & EG_REPORT_EXACT_STEPS) != 0 };
    snprintf(what, sizeof(what), "%s cap %lld%s: report n %lld finished %lld launches %lld steps %lld mapped %lld done %lld "
             "exact %lld equals the old device runner's, mode BAND", unit, (long long)cap, tag, (long long)have[0],
             (long long)have[1], (long long)have[2], (long long)have[3], (long long)have[4], (long long)have[5], (long long)have[6]);
    check(ok && memcmp(rep, have, sizeof(rep)) == 0 && r->mode == EG_MODE_BAND && r->compactions == 0 && r->build_s > 0.0, what);
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
    snprintf(what, sizeof(what), "%s: eg_unit_load of the recorded unit (PTX image) on the device context", name);
    const eg_status ls = parse_digest(hex, ref.digest) ? eg_unit_load(ctx, &ld, &unit) : EG_BAD_REQUEST;
    if (ls != EG_OK) reason_has(ctx, "\x01");
    check(ls == EG_OK, what);
    if (unit == NULL) return;
    int64_t max_stop = 0;
    for (int i = 0; i < N; ++i) max_stop = fmax((double)max_stop, round(t_final[i] / DT));
    const int64_t caps[2] = { 400, 20 };
    for (int c = 0; c < 2; ++c) {
        struct eg_plan_desc pd;
        memset(&pd, 0, sizeof(pd));
        pd.struct_size = sizeof(pd);
        pd.unit = unit;
        pd.structure = EG_STRUCTURE_DEVICE;
        pd.n_samples = N;
        eg_plan* plan = NULL;
        struct planes s;
        int ok = eg_plan_new(ctx, &pd, &plan) == EG_OK && planes_new(ctx, &s) && bind(ctx, plan, &s) == EG_OK;
        eg_runner* runner = NULL;
        const struct eg_runner_desc rd = runner_desc(&plan, caps[c]);
        ok = ok && eg_runner_new(ctx, &rd, &runner) == EG_OK;
        struct eg_run_report r;
        memset(&r, 0, sizeof(r));
        r.struct_size = sizeof(r);
        ok = ok && eg_run(ctx, runner, &r) == EG_OK;
        snprintf(what, sizeof(what), "%s cap %lld: DEVICE plan, eg_runner_new and eg_run OK", name, (long long)caps[c]);
        check(ok, what);
        if (!ok) return;
        compare(dir, name, caps[c], &s, &r, "");
        /* the closed form: launches ceil(stop / K) capped, finished = stops inside the cap, t ends at t_final */
        const int64_t cap_launches = caps[c] / k + (caps[c] % k != 0);
        const int64_t launches = (max_stop + k - 1) / k < cap_launches ? (max_stop + k - 1) / k : cap_launches;
        int64_t inside = 0;
        for (int i = 0; i < N; ++i) inside += round(t_final[i] / DT) <= launches * k;
        double t[N];
        int t_ok = cudaMemcpy(t, s.d[0], sizeof(t), cudaMemcpyDeviceToHost) == cudaSuccess;
        for (int i = 0; t_ok && i < N; ++i)
            t_ok = round(t_final[i] / DT) <= launches * k ? t[i] == t_final[i] : t[i] == (double)(launches * k) * DT;
        snprintf(what, sizeof(what), "%s cap %lld: closed form, launches %lld finished %lld, t = min(stop, launches*K)*dt",
                 name, (long long)caps[c], (long long)launches, (long long)inside);
        check(r.launches == launches && r.finished == inside && r.steps == launches * k && r.mapped == launches * N && t_ok, what);
        if (c == 0) {
            struct eg_run_report again;
            memset(&again, 0, sizeof(again));
            again.struct_size = sizeof(again);
            snprintf(what, sizeof(what), "%s: a second run of a finished batch launches nothing (finished %d, done)", name, N);
            check(eg_run(ctx, runner, &again) == EG_OK && again.launches == 0 && again.finished == N &&
                  (again.flags & EG_REPORT_DONE) && again.build_s == r.build_s, what);
            /* fresh planes bound to the same plan: the loop is captured again over them */
            struct planes s2;
            ok = planes_new(ctx, &s2) && bind(ctx, plan, &s2) == EG_OK && eg_run(ctx, runner, &again) == EG_OK;
            snprintf(what, sizeof(what), "%s: rebinding fresh planes between runs, eg_run OK", name);
            check(ok, what);
            if (ok) compare(dir, name, caps[c], &s2, &again, " after a rebind");
            if (ok) planes_free(ctx, &s2);
            /* refusals by name */
            struct eg_runner_desc bad = rd;
            eg_runner* none = NULL;
            bad.flags = EG_RUN_REORDER;
            check(eg_runner_new(ctx, &bad, &none) == EG_BAD_REQUEST && reason_has(ctx, "EG_RUN_REORDER") && none == NULL,
                  "device runner: EG_RUN_REORDER refused BAD_REQUEST (no active set)");
            bad.flags = EG_RUN_RECORD;
            check(eg_runner_new(ctx, &bad, &none) == EG_BAD_REQUEST && reason_has(ctx, "EG_RUN_RECORD needs its derived units"),
                  "device runner: EG_RUN_RECORD without its derived units refused BAD_REQUEST by name");
            bad = rd;
            bad.events = EG_EVENTS_REFINE;
            check(eg_runner_new(ctx, &bad, &none) == EG_BAD_REQUEST && reason_has(ctx, "events refine needs the"),
                  "device runner: events REFINE without its derived units refused BAD_REQUEST by name");
            bad = rd;
            bad.every = k + 1;
            check(eg_runner_new(ctx, &bad, &none) == EG_BAD_REQUEST && reason_has(ctx, "without an active set"),
                  "device runner: every other than K refused BAD_REQUEST (no active set)");
            bad = rd;
            bad.step_graph = (eg_graph*)&bad;
            check(eg_runner_new(ctx, &bad, &none) == EG_BAD_REQUEST && reason_has(ctx, "step_graph"),
                  "device runner: a step_graph that is not a graph of the context refused BAD_REQUEST by name");
        }
        check(eg_runner_free(ctx, runner) == EG_OK && eg_plan_free(ctx, plan) == EG_OK, "eg_runner_free and eg_plan_free");
        planes_free(ctx, &s);
    }
    eg_unit_free(ctx, unit);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <osc lib> <osc digest> <osc_x4 lib> <osc_x4 digest> <oracle dir>\n", argv[0]);
        return 2;
    }
    const char* dir = argv[5];
    if (!read_file(dir, "t_final_in.f64", t_final, sizeof(t_final)) || !read_file(dir, "x_in.f64", x_in, sizeof(x_in)) ||
        !read_file(dir, "v_in.f64", v_in, sizeof(v_in))) {
        fprintf(stderr, "oracle inputs missing in %s\n", dir);
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
    run_unit(ctx, argv[1], "osc", argv[2], 1, dir);
    run_unit(ctx, argv[3], "osc_x4", argv[4], 4, dir);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
