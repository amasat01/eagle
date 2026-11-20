// SPDX-License-Identifier: Apache-2.0
/*
 * events refine on the device (EG_EVENTS_REFINE over DEVICE plans, a plain run) and eg_run_events: the twin, the Newton and the state
 * guard vjp of a unit with a stop rule run through the old device path's own images; the end states, stop times and statuses, and the
 * report's steps and launches, equal the old device path's bit for bit (osc, pend, grid). tangent and two are refused by name.
 *
 * usage: device_refine_test <library> osc|pend|grid|tangent|two <oracle dir> <primal> <twin> <newton> <guard digest> <max_steps> <on_no_event 0|1> <compare 0|1>
 * oracle dir: x0 (osc, pend) or h and bigt (grid), end_{x,v,t}, t_stop and status .bin.
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 480

/* Device planes made from host values; freed together. */
struct dplanes {
    void* d[8];
    struct tensor t[8];
    eg_plane* p[8];
    int k;
};

static eg_plane* dplane(eg_ctx* ctx, struct dplanes* s, const void* init, int64_t n, uint8_t code, uint8_t bits) {
    const int i = s->k++;
    const size_t bytes = (size_t)n * bits / 8;
    s->d[i] = NULL;
    s->p[i] = NULL;
    if (cudaMalloc(&s->d[i], bytes) != cudaSuccess || cudaMemcpy(s->d[i], init, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    s->p[i] = wrap(ctx, &s->t[i], n, s->d[i], code, bits);
    return s->p[i];
}

static void dfree(eg_ctx* ctx, struct dplanes* s) {
    for (int i = 0; i < s->k; ++i) {
        if (s->p[i]) eg_plane_free(ctx, s->p[i]);
        if (s->d[i]) cudaFree(s->d[i]);
    }
    s->k = 0;
}

static int dget(const struct dplanes* s, int i, void* host, size_t bytes) {
    return s->d[i] && cudaMemcpy(host, s->d[i], bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

static eg_unit* load(eg_ctx* ctx, const char* path, const char* name, const uint8_t* digest) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memcpy(ref.digest, digest, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = path;
    d.unit = &ref;
    eg_unit* u = NULL;
    return eg_unit_load(ctx, &d, &u) == EG_OK ? u : NULL;
}

int main(int argc, char** argv) {
    if (argc != 11) {
        fprintf(stderr, "usage: %s <library> osc|pend|grid|tangent|two <oracle dir> <4 digests> <max_steps> <on_no_event> <compare>\n", argv[0]);
        return 2;
    }
    const char *lib = argv[1], *mode = argv[2], *dir = argv[3];
    const int64_t max_steps = atol(argv[8]);
    const int zero = atoi(argv[9]), compare = atoi(argv[10]);
    const int grid = strcmp(mode, "grid") == 0, tangent = strcmp(mode, "tangent") == 0, two = strcmp(mode, "two") == 0;
    const int pend = strcmp(mode, "pend") == 0;
    const int64_t n = grid ? 480 : (tangent || two) ? 4 : 64;
    char nm[4][48];
    snprintf(nm[0], sizeof(nm[0]), "ev_%s", mode);
    snprintf(nm[1], sizeof(nm[1]), "ev_%s_tape", mode);
    snprintf(nm[2], sizeof(nm[2]), "ev_%s_newton", mode);
    snprintf(nm[3], sizeof(nm[3]), "ev_%s_state_guard_vjp", mode);
    uint8_t dg[4][EG_DIGEST_BYTES];
    for (int i = 0; i < 4; ++i)
        if (!parse_digest(argv[4 + i], dg[i])) return 2;
    static double x0[MAXN], h[MAXN], bigt[MAXN], ex[MAXN], ev[MAXN], et[MAXN], ts[MAXN];
    static int8_t st[MAXN];
    int have = grid || read_file(dir, "x0.bin", x0, (size_t)n * 8) || tangent || two;   /* the start states, read for the refusal too */
    if (compare) {
        have = read_file(dir, "end_x.bin", ex, (size_t)n * 8) && read_file(dir, "end_t.bin", et, (size_t)n * 8) &&
               read_file(dir, "t_stop.bin", ts, (size_t)n * 8) && read_file(dir, "status.bin", st, (size_t)n);
        have = have && (grid ? read_file(dir, "h.bin", h, (size_t)n * 8) && read_file(dir, "bigt.bin", bigt, (size_t)n * 8)
                             : read_file(dir, "end_v.bin", ev, (size_t)n * 8));
    }
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.flags = EG_CTX_DEVICE;
    s.device = 0;
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    eg_unit* p = load(ctx, lib, nm[0], dg[0]);
    eg_unit* units[3] = { load(ctx, lib, nm[1], dg[1]), load(ctx, lib, nm[2], dg[2]), load(ctx, lib, nm[3], dg[3]) };
    char what[300], msg[600];
    snprintf(what, sizeof(what), "%s: the primal, its twin, its Newton and its state guard vjp load (PTX images) on the device context", nm[0]);
    check(p && units[0] && units[1] && units[2] && have, what);
    if (p && units[0] && units[1] && units[2] && have) {
        const eg_derived_role roles[3] = { EG_DERIVED_TAPE, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP };
        static double t[MAXN], x[MAXN], v[MAXN], tstop[MAXN];
        static int8_t status[MAXN];
        static uint8_t done[MAXN];
        for (int i = 0; i < n; ++i) x[i] = grid || tangent ? 0.0 : two ? 1.0 : x0[i];
        struct dplanes s_ = { .k = 0 };
        eg_plane* planes[5];
        int32_t slots[5];
        int np_ = 0;
        planes[np_] = dplane(ctx, &s_, t, n, kDLFloat, 64); slots[np_++] = 0;
        if (grid) {
            planes[np_] = dplane(ctx, &s_, x, n, kDLFloat, 64); slots[np_++] = 1;
            planes[np_] = dplane(ctx, &s_, bigt, n, kDLFloat, 64); slots[np_++] = 2;
            planes[np_] = dplane(ctx, &s_, h, n, kDLFloat, 64); slots[np_++] = 3;
            planes[np_] = dplane(ctx, &s_, done, n, kDLBool, 8); slots[np_++] = 5;
        } else if (tangent) {
            planes[np_] = dplane(ctx, &s_, x, n, kDLFloat, 64); slots[np_++] = 1;
            planes[np_] = dplane(ctx, &s_, done, n, kDLBool, 8); slots[np_++] = 3;
        } else {
            planes[np_] = dplane(ctx, &s_, v, n, kDLFloat, 64); slots[np_++] = 1;
            planes[np_] = dplane(ctx, &s_, x, n, kDLFloat, 64); slots[np_++] = 2;
            planes[np_] = dplane(ctx, &s_, done, n, kDLBool, 8); slots[np_++] = 4;
        }
        /* the uniforms: the oscillator a, dt, w in slots 5, 6, 7; the pendulum a, c, dt; tangent's in slot 4; two's in 5, 6 */
        const double uvals[3] = { pend ? 0.05 : 0.0, pend ? 0.2 : 0.05, pend ? 0.05 : 2.0 }, u_two[2] = { 0.8, 0.1 }, u_tan = 0.125;
        const int32_t us3[3] = { 5, 6, 7 }, us_two[2] = { 5, 6 }, us_tan[1] = { 4 };
        const void *const uv3[3] = { &uvals[0], &uvals[1], &uvals[2] }, *const uv2[2] = { &u_two[0], &u_two[1] }, *const uv1[1] = { &u_tan };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = np_;
        bd.plane_slots = slots;
        bd.planes = planes;
        bd.n_uniforms = grid ? 0 : tangent ? 1 : two ? 2 : 3;
        bd.uniform_slots = tangent ? us_tan : two ? us_two : us3;
        bd.uniform_values = tangent ? uv1 : two ? uv2 : uv3;
        struct eg_plan_desc pd;
        memset(&pd, 0, sizeof(pd));
        pd.struct_size = sizeof(pd);
        pd.unit = p;
        pd.structure = EG_STRUCTURE_DEVICE;
        pd.policy = EG_REDUCE_COMPENSATED;
        pd.n_samples = n;
        eg_plan* plan = NULL;
        const int bound = eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK;
        struct eg_runner_desc d = runner_desc(&plan, max_steps);
        d.events = EG_EVENTS_REFINE;
        d.on_no_event = zero ? EG_NO_EVENT_ZERO : EG_NO_EVENT_REFUSE;
        d.n_derived = 3;
        d.derived_roles = roles;
        d.derived_units = units;
        eg_runner* r = NULL;
        struct eg_run_report rep;
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        const int ok = bound && eg_runner_new(ctx, &d, &r) == EG_OK;
        if (!ok) reason_has(ctx, "(the refine runner builds)");
        snprintf(what, sizeof(what), "%s: eg_runner_new with events REFINE over a DEVICE plan", nm[0]);
        check(ok, what);
        double tsd[MAXN];
        int8_t std_[MAXN];
        struct dplanes o = { .k = 0 };
        eg_plane* tp = dplane(ctx, &o, tsd, n, kDLFloat, 64);
        eg_plane* sp = dplane(ctx, &o, std_, n, kDLInt, 8);
        check(ok && tp && sp && eg_run_events(ctx, r, tp, sp) == EG_BAD_REQUEST && reason_has(ctx, "no refine run"),
              "eg_run_events before any run refused BAD_REQUEST 'no refine run'");
        if (tangent || two) {
            const char* why = tangent ? "graze the surface" : "two stop rules cross in the same step";
            snprintf(msg, sizeof(msg), "%s: refused BAD_REQUEST naming '%s'", nm[0], why);
            check(ok && eg_run(ctx, r, &rep) == EG_BAD_REQUEST && reason_has(ctx, why), msg);
        } else if (!compare) {
            snprintf(msg, sizeof(msg), "%s: no crossing in the stop step refused BAD_REQUEST by name (EG_NO_EVENT_REFUSE)", nm[0]);
            check(ok && eg_run(ctx, r, &rep) == EG_BAD_REQUEST && reason_has(ctx, "no crossing in their stop step"), msg);
        } else {
            const int ran = ok && eg_run(ctx, r, &rep) == EG_OK && eg_run_events(ctx, r, tp, sp) == EG_OK;
            /* the planes: t, then x (grid) or v and x */
            int e_ok = ran && dget(&s_, 0, t, (size_t)n * 8) && dget(&s_, grid ? 1 : 2, x, (size_t)n * 8) && (grid || dget(&s_, 1, v, (size_t)n * 8)) &&
                       dget(&o, 0, tstop, (size_t)n * 8) && dget(&o, 1, status, (size_t)n);
            for (int i = 0; e_ok && i < n; ++i) {
                e_ok = memcmp(&x[i], &ex[i], 8) == 0 && memcmp(&t[i], &et[i], 8) == 0 && memcmp(&tstop[i], &ts[i], 8) == 0 && status[i] == st[i] &&
                       (grid || memcmp(&v[i], &ev[i], 8) == 0);
                if (!e_ok)
                    printf("      sample %d: x %.17g vs %.17g, t %.17g vs %.17g, t_stop %.17g vs %.17g, status %d vs %d\n", i, x[i], ex[i], t[i], et[i], tstop[i], ts[i], status[i], st[i]);
            }
            snprintf(msg, sizeof(msg), "%s: the refined states, stop times and statuses equal the old device path's bitwise (status %d, t_stop %.17g)", nm[0],
                     ran ? status[0] : -1, tstop[0]);
            check(e_ok, msg);
            snprintf(msg, sizeof(msg), "%s: the report's steps and launches equal the old device path's (%lld, %lld)", nm[0], (long long)rep.steps, (long long)rep.launches);
            check(ran && rep.steps == max_steps && rep.launches == max_steps && rep.mode == EG_MODE_REFINE, msg);
            struct dplanes b = { .k = 0 };
            float f32[MAXN];
            eg_plane* bp = dplane(ctx, &b, f32, n, kDLFloat, 32);
            check(ran && bp && eg_run_events(ctx, r, bp, NULL) == EG_BAD_REQUEST && reason_has(ctx, "plane shape or dtype"),
                  "eg_run_events with a float32 time plane refused BAD_REQUEST 'plane shape or dtype'");
            dfree(ctx, &b);
            check(ran && eg_run_events(ctx, r, NULL, NULL) == EG_OK, "eg_run_events with both planes NULL is OK");
        }
        dfree(ctx, &o);
        dfree(ctx, &s_);
        if (r) eg_runner_free(ctx, r);
        if (plan) eg_plan_free(ctx, plan);
    }
    if (p) eg_unit_free(ctx, p);
    for (int i = 0; i < 3; ++i)
        if (units[i]) eg_unit_free(ctx, units[i]);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
