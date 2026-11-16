// SPDX-License-Identifier: Apache-2.0
/*
 * A recorded events refine run on the device (EG_EVENTS_REFINE with EG_RUN_RECORD over DEVICE plans, window 1) and eg_reverse through the stop.
 *
 *   device_refine_reverse_test <library> <oracle dir> <primal> <twin> <newton> <guard> <step vjp> <refine vjp> digests [pend|grid]
 *
 * ev_osc (or ev_pend) recorded with its taping twin, step vjp, Newton, state guard vjp and refine vjp, run through the old device path's own images;
 * the end states, the report's steps and launches, and the six gradients of the reverse seeded with ones on t, v and x in turn equal the old
 * device path's (the same run through eagle.diff.reverse on cuda) bit for bit.
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PN 64

/* Device planes made from host values; freed together. */
struct dplanes {
    void* d[16];
    struct tensor t[16];
    eg_plane* p[16];
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

static int dput(const struct dplanes* s, int i, const void* host, size_t bytes) {
    return s->d[i] && cudaMemcpy(s->d[i], host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
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

static struct eg_bind_desc binds(int n_planes, const int32_t* slots, eg_plane* const* planes) {
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = n_planes;
    b.plane_slots = slots;
    b.planes = planes;
    return b;
}

#define GN 480

/* ev_grid on the device: per-sample dt and big_t planes, every sample stops AT_STEP; seeds t and x; the gradients of t, x, dt and big_t, one
 * per sample (unsummed), equal the old device path's bit for bit. */
static int grid_rows(char** argv) {
    const char *lib = argv[1], *dir = argv[2];
    uint8_t dg[6][EG_DIGEST_BYTES];
    for (int i = 0; i < 6; ++i)
        if (!parse_digest(argv[3 + i], dg[i])) return 2;
    static double h[GN], bigt[GN], ex[GN], et[GN], ts[GN];
    static int8_t st[GN];
    if (!(read_file(dir, "h.bin", h, sizeof(h)) && read_file(dir, "bigt.bin", bigt, sizeof(bigt)) && read_file(dir, "end_x.bin", ex, sizeof(ex)) &&
          read_file(dir, "end_t.bin", et, sizeof(et)) && read_file(dir, "t_stop.bin", ts, sizeof(ts)) && read_file(dir, "status.bin", st, sizeof(st)))) {
        fprintf(stderr, "oracle files missing in %s\n", dir);
        return 2;
    }
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.flags = EG_CTX_DEVICE;
    s.device = 0;
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    eg_unit* p = load(ctx, lib, "ev_grid", dg[0]);
    eg_unit* units[5] = { load(ctx, lib, "ev_grid_tape", dg[1]), load(ctx, lib, "ev_grid_step_vjp", dg[4]), load(ctx, lib, "ev_grid_newton", dg[2]),
                          load(ctx, lib, "ev_grid_state_guard_vjp", dg[3]), load(ctx, lib, "ev_grid_refine_vjp", dg[5]) };
    const int loaded = p && units[0] && units[1] && units[2] && units[3] && units[4];
    check(loaded, "ev_grid: the primal, its twin, step vjp, Newton, state guard vjp and refine vjp load (PTX images) on the device context");
    if (loaded) {
        const eg_derived_role roles[5] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP,
                                           EG_DERIVED_REFINE_VJP };
        const char* seeds_of[2] = { "t", "x" };
        const int32_t seed_slot[2] = { 0, 1 };
        const char* grads_of[4] = { "t", "x", "dt", "big_t" };
        const int32_t grad_slot[4] = { 0, 1, 3, 2 };
        static const double zeros[GN];
        static const uint8_t none[GN];
        static double ones[GN];
        for (int i = 0; i < GN; ++i) ones[i] = 1.0;
        struct dplanes sp_ = { .k = 0 };
        eg_plane* planes[5] = { dplane(ctx, &sp_, zeros, GN, kDLFloat, 64), dplane(ctx, &sp_, zeros, GN, kDLFloat, 64), dplane(ctx, &sp_, bigt, GN, kDLFloat, 64),
                                dplane(ctx, &sp_, h, GN, kDLFloat, 64), dplane(ctx, &sp_, none, GN, kDLBool, 8) };
        const int32_t slots[5] = { 0, 1, 2, 3, 5 };
        const struct eg_bind_desc bd = binds(5, slots, planes);
        struct eg_plan_desc pd;
        memset(&pd, 0, sizeof(pd));
        pd.struct_size = sizeof(pd);
        pd.unit = p;
        pd.structure = EG_STRUCTURE_DEVICE;
        pd.policy = EG_REDUCE_COMPENSATED;
        pd.n_samples = GN;
        eg_plan* plan = NULL;
        const int planned = planes[4] && eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK;
        struct eg_runner_desc d = runner_desc(&plan, 80);
        d.events = EG_EVENTS_REFINE;
        d.on_no_event = EG_NO_EVENT_REFUSE;
        d.flags = EG_RUN_RECORD;
        d.window = 1;
        d.n_derived = 5;
        d.derived_roles = roles;
        d.derived_units = units;
        eg_runner* r = NULL;
        const int ok = planned && eg_runner_new(ctx, &d, &r) == EG_OK;
        if (!ok) reason_has(ctx, "(the runner builds)");
        check(ok, "ev_grid: the recorded refine runner builds over a DEVICE plan");
        for (int sd_i = 0; ok && sd_i < 2; ++sd_i) {
            int ready = dput(&sp_, 0, zeros, sizeof(zeros)) && dput(&sp_, 1, zeros, sizeof(zeros)) && dput(&sp_, 4, none, sizeof(none));
            struct eg_run_report rep;
            memset(&rep, 0, sizeof(rep));
            rep.struct_size = sizeof(rep);
            struct dplanes o = { .k = 0 };
            static double tstop[GN];
            static int8_t status[GN];
            eg_plane* tp = dplane(ctx, &o, tstop, GN, kDLFloat, 64);
            eg_plane* sp = dplane(ctx, &o, status, GN, kDLInt, 8);
            const int ran = ready && eg_run(ctx, r, &rep) == EG_OK && eg_run_events(ctx, r, tp, sp) == EG_OK;
            static double x[GN], t[GN];
            int s_ok = ran && dget(&sp_, 0, t, sizeof(t)) && dget(&sp_, 1, x, sizeof(x)) && dget(&o, 0, tstop, sizeof(tstop)) && dget(&o, 1, status, sizeof(status));
            for (int i = 0; s_ok && i < GN; ++i)
                s_ok = memcmp(&x[i], &ex[i], 8) == 0 && memcmp(&t[i], &et[i], 8) == 0 && memcmp(&tstop[i], &ts[i], 8) == 0 && status[i] == st[i];
            char msg[300];
            snprintf(msg, sizeof(msg), "ev_grid, seed %s: the recorded run's states, stop times and statuses equal the old device path's bitwise", seeds_of[sd_i]);
            check(s_ok, msg);
            struct dplanes g = { .k = 0 };
            static double gv[4][GN];
            eg_plane* gpl[4];
            for (int k = 0; k < 4; ++k) gpl[k] = dplane(ctx, &g, gv[k], GN, kDLFloat, 64);
            eg_plane* sdp = dplane(ctx, &g, ones, GN, kDLFloat, 64);
            const struct eg_bind_desc sd = binds(1, &seed_slot[sd_i], &sdp), gd = binds(4, grad_slot, gpl);
            const int rev = ran && eg_reverse(ctx, r, &sd, &gd) == EG_OK;
            snprintf(msg, sizeof(msg), "ev_grid, seed %s: eg_run and eg_reverse are OK", seeds_of[sd_i]);
            check(rev, msg);
            for (int k = 0; rev && k < 4; ++k) {
                char file[64];
                static double want[GN], got[GN];
                snprintf(file, sizeof(file), "g_%s_%s.bin", seeds_of[sd_i], grads_of[k]);
                int eq = read_file(dir, file, want, sizeof(want)) && dget(&g, k, got, sizeof(got));
                for (int i = 0; eq && i < GN; ++i) {
                    eq = memcmp(&got[i], &want[i], 8) == 0;
                    if (!eq) printf("      %s[%d]: %.17g vs %.17g\n", file, i, got[i], want[i]);
                }
                snprintf(msg, sizeof(msg), "ev_grid, seed %s: the per-sample gradient of %s equals the old device path's bitwise", seeds_of[sd_i], grads_of[k]);
                check(eq, msg);
            }
            dfree(ctx, &g);
            dfree(ctx, &o);
        }
        if (r) eg_runner_free(ctx, r);
        if (plan) eg_plan_free(ctx, plan);
        dfree(ctx, &sp_);
    }
    if (p) eg_unit_free(ctx, p);
    for (int i = 0; i < 5; ++i)
        if (units[i]) eg_unit_free(ctx, units[i]);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}

int main(int argc, char** argv) {
    if (argc == 10 && strcmp(argv[9], "grid") == 0) return grid_rows(argv);
    if (argc != 9 && argc != 10) {   /* 10: the pendulum (ev_pend) instead of the oscillator */
        fprintf(stderr, "usage: %s <library> <oracle dir> <primal> <twin> <newton> <guard> <step vjp> <refine vjp> [pend]\n", argv[0]);
        return 2;
    }
    const char *lib = argv[1], *dir = argv[2];
    const int pend = argc == 10;
    char n_p[16], n_t[64], n_s[64], n_n[64], n_g[64], n_r[64];
    snprintf(n_p, sizeof(n_p), pend ? "ev_pend" : "ev_osc");
    snprintf(n_t, sizeof(n_t), "%s_tape", n_p);
    snprintf(n_s, sizeof(n_s), "%s_step_vjp", n_p);
    snprintf(n_n, sizeof(n_n), "%s_newton", n_p);
    snprintf(n_g, sizeof(n_g), "%s_state_guard_vjp", n_p);
    snprintf(n_r, sizeof(n_r), "%s_refine_vjp", n_p);
    uint8_t dg[6][EG_DIGEST_BYTES];
    for (int i = 0; i < 6; ++i)
        if (!parse_digest(argv[3 + i], dg[i])) return 2;
    static double x0[PN], ex[PN], ev[PN], et[PN], ts[PN];
    static int8_t st[PN];
    int64_t run_ref[2];
    if (!read_file(dir, "x0.bin", x0, sizeof(x0)) || !read_file(dir, "end_x.bin", ex, sizeof(ex)) || !read_file(dir, "end_v.bin", ev, sizeof(ev)) ||
        !read_file(dir, "end_t.bin", et, sizeof(et)) || !read_file(dir, "t_stop.bin", ts, sizeof(ts)) || !read_file(dir, "status.bin", st, sizeof(st)) ||
        !read_file(dir, "run.bin", run_ref, sizeof(run_ref))) {
        fprintf(stderr, "oracle files missing in %s\n", dir);
        return 2;
    }
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.flags = EG_CTX_DEVICE;
    s.device = 0;
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    eg_unit* p = load(ctx, lib, n_p, dg[0]);
    eg_unit* units[5] = { load(ctx, lib, n_t, dg[1]), load(ctx, lib, n_s, dg[4]), load(ctx, lib, n_n, dg[2]), load(ctx, lib, n_g, dg[3]),
                          load(ctx, lib, n_r, dg[5]) };
    const int loaded = p && units[0] && units[1] && units[2] && units[3] && units[4];
    char what[300], msg[640];
    snprintf(what, sizeof(what), "%s: the primal, its twin, step vjp, Newton, state guard vjp and refine vjp load (PTX images) on the device context", n_p);
    check(loaded, what);
    if (loaded) {
        const eg_derived_role roles[5] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP,
                                           EG_DERIVED_REFINE_VJP };
        /* the uniforms by slot 5, 6, 7: the oscillator a, dt, w; the pendulum a, c, dt */
        const double uvals[3] = { pend ? 0.05 : 0.0, pend ? 0.2 : 0.05, pend ? 0.05 : 2.0 };
        const char* seeds_of[3] = { "t", "v", "x" };
        const int32_t seed_slot[3] = { 0, 1, 2 };
        const char* grads_osc[6] = { "t", "v", "x", "dt", "w", "a" }, *grads_pend[6] = { "t", "v", "x", "dt", "c", "a" };
        const char* const* grads_of = pend ? grads_pend : grads_osc;
        const int32_t grad_slot_osc[6] = { 0, 1, 2, 6, 7, 5 }, grad_slot_pend[6] = { 0, 1, 2, 7, 6, 5 };
        const int32_t* grad_slot = pend ? grad_slot_pend : grad_slot_osc;
        static const double zeros[PN];
        static const uint8_t none[PN];
        static double ones[PN];
        for (int i = 0; i < PN; ++i) ones[i] = 1.0;
        struct dplanes sp_ = { .k = 0 };
        eg_plane* planes[4] = { dplane(ctx, &sp_, zeros, PN, kDLFloat, 64), dplane(ctx, &sp_, zeros, PN, kDLFloat, 64), dplane(ctx, &sp_, x0, PN, kDLFloat, 64),
                                dplane(ctx, &sp_, none, PN, kDLBool, 8) };
        const int32_t slots[4] = { 0, 1, 2, 4 }, us[3] = { 5, 6, 7 };
        const void* const uv[3] = { &uvals[0], &uvals[1], &uvals[2] };
        struct eg_bind_desc bd = binds(4, slots, planes);
        bd.n_uniforms = 3;
        bd.uniform_slots = us;
        bd.uniform_values = uv;
        struct eg_plan_desc pd;
        memset(&pd, 0, sizeof(pd));
        pd.struct_size = sizeof(pd);
        pd.unit = p;
        pd.structure = EG_STRUCTURE_DEVICE;
        pd.policy = EG_REDUCE_COMPENSATED;
        pd.n_samples = PN;
        eg_plan* plan = NULL;
        const int planned = planes[3] && eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK;
        struct eg_runner_desc d = runner_desc(&plan, 400);
        d.events = EG_EVENTS_REFINE;
        d.flags = EG_RUN_RECORD;
        d.window = 2;
        d.n_derived = 5;
        d.derived_roles = roles;
        d.derived_units = units;
        eg_runner* r = NULL;
        snprintf(msg, sizeof(msg), "%s: window 2 refused BAD_REQUEST naming 'window'", n_p);
        check(planned && eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "window"), msg);
        d.window = 1;
        const int ok = planned && eg_runner_new(ctx, &d, &r) == EG_OK;
        snprintf(msg, sizeof(msg), "%s: the recorded refine runner builds over a DEVICE plan", n_p);
        check(ok, msg);
        for (int sd_i = 0; ok && sd_i < 3; ++sd_i) {
            int ready = dput(&sp_, 0, zeros, sizeof(zeros)) && dput(&sp_, 1, zeros, sizeof(zeros)) && dput(&sp_, 2, x0, sizeof(x0)) &&
                        dput(&sp_, 3, none, sizeof(none));
            struct eg_run_report rep;
            memset(&rep, 0, sizeof(rep));
            rep.struct_size = sizeof(rep);
            const int ran = ready && eg_run(ctx, r, &rep) == EG_OK;
            static double x[PN], v[PN], t[PN];
            int s_ok = ran && dget(&sp_, 0, t, sizeof(t)) && dget(&sp_, 1, v, sizeof(v)) && dget(&sp_, 2, x, sizeof(x)) && rep.steps == run_ref[0] &&
                       rep.launches == run_ref[1] && rep.finished == PN;
            for (int i = 0; s_ok && i < PN; ++i) s_ok = memcmp(&x[i], &ex[i], 8) == 0 && memcmp(&v[i], &ev[i], 8) == 0 && memcmp(&t[i], &et[i], 8) == 0;
            snprintf(msg, sizeof(msg), "%s, seed %s: the recorded run's states equal the old device path's bitwise (%lld steps, %lld launches)", n_p,
                     seeds_of[sd_i], (long long)rep.steps, (long long)rep.launches);
            check(s_ok, msg);
            struct dplanes g = { .k = 0 };
            static double gv[6][PN];
            eg_plane* gpl[6];
            for (int k = 0; k < 6; ++k) gpl[k] = dplane(ctx, &g, gv[k], k < 3 ? PN : 1, kDLFloat, 64);
            eg_plane* sp = dplane(ctx, &g, ones, PN, kDLFloat, 64);
            const struct eg_bind_desc sd = binds(1, &seed_slot[sd_i], &sp), gd = binds(6, grad_slot, gpl);
            const int rev = ran && eg_reverse(ctx, r, &sd, &gd) == EG_OK;
            snprintf(msg, sizeof(msg), "%s, seed %s: eg_run and eg_reverse are OK", n_p, seeds_of[sd_i]);
            check(rev, msg);
            for (int k = 0; rev && k < 6; ++k) {
                char file[64];
                static double want[PN], got[PN];
                const size_t cnt = k < 3 ? PN : 1;
                snprintf(file, sizeof(file), "g_%s_%s.bin", seeds_of[sd_i], grads_of[k]);
                int eq = read_file(dir, file, want, cnt * sizeof(double)) && dget(&g, k, got, cnt * sizeof(double));
                for (size_t i = 0; eq && i < cnt; ++i) {
                    eq = memcmp(&got[i], &want[i], 8) == 0;
                    if (!eq) printf("      %s[%zu]: %.17g vs %.17g\n", file, i, got[i], want[i]);
                }
                snprintf(msg, sizeof(msg), "%s, seed %s: the gradient of %s equals the old device path's bitwise", n_p, seeds_of[sd_i], grads_of[k]);
                check(eq, msg);
            }
            dfree(ctx, &g);
        }
        {   /* eg_run_events after a recorded run: the old device run's stop times and statuses */
            static double tstop[PN];
            static int8_t status[PN];
            struct dplanes o = { .k = 0 };
            eg_plane* tp = dplane(ctx, &o, tstop, PN, kDLFloat, 64);
            eg_plane* sp = dplane(ctx, &o, status, PN, kDLInt, 8);
            struct eg_run_report rep;
            memset(&rep, 0, sizeof(rep));
            rep.struct_size = sizeof(rep);
            int e_ok = ok && dput(&sp_, 0, zeros, sizeof(zeros)) && dput(&sp_, 1, zeros, sizeof(zeros)) && dput(&sp_, 2, x0, sizeof(x0)) &&
                       dput(&sp_, 3, none, sizeof(none)) && eg_run(ctx, r, &rep) == EG_OK && eg_run_events(ctx, r, tp, sp) == EG_OK &&
                       dget(&o, 0, tstop, sizeof(tstop)) && dget(&o, 1, status, sizeof(status));
            for (int i = 0; e_ok && i < PN; ++i) e_ok = memcmp(&tstop[i], &ts[i], 8) == 0 && status[i] == st[i];
            snprintf(msg, sizeof(msg), "%s: eg_run_events of a recorded run equals the old device path's stop times and statuses bitwise", n_p);
            check(e_ok, msg);
            dfree(ctx, &o);
        }
        if (r) eg_runner_free(ctx, r);
        if (plan) eg_plan_free(ctx, plan);
        dfree(ctx, &sp_);
    }
    if (p) eg_unit_free(ctx, p);
    for (int i = 0; i < 5; ++i)
        if (units[i]) eg_unit_free(ctx, units[i]);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
