// SPDX-License-Identifier: Apache-2.0
/*
 * A recorded events refine run and its reverse (EG_EVENTS_REFINE with EG_RUN_RECORD, window 1; eg_reverse).
 *
 *   refine_reverse_test <pair lib> <oracle dir> <primal> <twin> <newton> <guard> <step vjp> <refine vjp> digests
 *
 * ev_osc (the old path's oscillator with a stop rule) recorded with its taping twin, step vjp, Newton, state guard vjp and
 * refine vjp; the reverse seeded with ones on t, v and x in turn must give the old path's gradients (the same run through
 * eagle.diff.reverse on cpu, central differences having checked them there) bit for bit, on SERIAL and TEAM with 1 and 3 threads.
 *
 *   refine_reverse_test <pair lib> <oracle dir> <primal> <twin> <newton> <guard> <step vjp> <refine vjp> grid
 *
 * ev_grid (a horizon on the step grid: per-sample dt and big_t planes, 480 samples, every sample stops AT_STEP) the same way: seeds t
 * and x, the gradients of t, x, dt and big_t, the end states, stop times and statuses, all equal to the old path's bit for bit.
 */
#include "pair_util.h"

#include <stdlib.h>

#define PN 64
#define GN 480

static eg_plane* host_plane(eg_ctx* ctx, struct tensor* t, int64_t n, void* data, uint8_t code, uint8_t bits) {
    tensor_fill(t, n, data, code, bits);
    t->managed.dl_tensor.device.device_type = kDLCPU;
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
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

/* The recorded refine run of ev_grid: its states, stop times and statuses and the reverse's gradients (seeds t, x) equal the old path's. */
static int grid_rows(char** argv) {
    const char *lib = argv[1], *dir = argv[2];
    uint8_t dg[6][EG_DIGEST_BYTES];
    for (int i = 0; i < 6; ++i)
        if (!parse_digest(argv[3 + i], dg[i])) return 2;
    static double h[GN], bigt[GN], ex[GN], et[GN], ts[GN];
    static int8_t st[GN];
    if (!(read_file(dir, "h.bin", h, sizeof(h)) && read_file(dir, "bigt.bin", bigt, sizeof(bigt)) && read_file(dir, "end_x.bin", ex, sizeof(ex)) &&
          read_file(dir, "end_t.bin", et, sizeof(et)) && read_file(dir, "t_stop.bin", ts, sizeof(ts)) && read_file(dir, "status.bin", st, sizeof(st)))) {
        check(0, "grid recorded refine: the oracle files read");
        return 1;
    }
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new host context");
    eg_unit* p = load(ctx, lib, "ev_grid", dg[0]);
    eg_unit* units[5] = { load(ctx, lib, "ev_grid_tape", dg[1]), load(ctx, lib, "ev_grid_step_vjp", dg[4]), load(ctx, lib, "ev_grid_newton", dg[2]),
                          load(ctx, lib, "ev_grid_state_guard_vjp", dg[3]), load(ctx, lib, "ev_grid_refine_vjp", dg[5]) };
    const int loaded = p && units[0] && units[1] && units[2] && units[3] && units[4];
    check(loaded, "recorded refine: ev_grid, its twin, step vjp, Newton, state guard vjp and refine vjp load from one library");
    if (!loaded) {
        eg_ctx_free(ctx);
        return 1;
    }
    const eg_derived_role roles[5] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP,
                                       EG_DERIVED_REFINE_VJP };
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int threads[2] = { 1, 3 };
    const char* seeds_of[2] = { "t", "x" };
    const int32_t seed_slot[2] = { 0, 1 };
    const char* grads_of[4] = { "t", "x", "dt", "big_t" };
    const int32_t grad_slot[4] = { 0, 1, 3, 2 };
    char what[256], msg[640];
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            eg_host_threads(ctx, threads[ti]);
            static double x[GN], t[GN], bt[GN], dtp[GN], ones[GN], tstop[GN];
            static int8_t status[GN];
            uint8_t done[GN];
            struct tensor tt, tx, tbt, tdt, tdn;
            eg_plane* planes[5] = { host_plane(ctx, &tt, GN, t, kDLFloat, 64), host_plane(ctx, &tx, GN, x, kDLFloat, 64),
                                    host_plane(ctx, &tbt, GN, bt, kDLFloat, 64), host_plane(ctx, &tdt, GN, dtp, kDLFloat, 64),
                                    host_plane(ctx, &tdn, GN, done, kDLBool, 8) };
            const int32_t slots[5] = { 0, 1, 2, 3, 5 };
            memcpy(bt, bigt, sizeof(bt));
            memcpy(dtp, h, sizeof(dtp));
            const struct eg_bind_desc bd = binds(5, slots, planes);
            eg_plan* plan = NULL;
            struct eg_plan_desc pd;
            memset(&pd, 0, sizeof(pd));
            pd.struct_size = sizeof(pd);
            pd.unit = p;
            pd.structure = structures[si];
            pd.policy = EG_REDUCE_COMPENSATED;
            pd.n_samples = GN;
            const int planned = eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK;
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
            snprintf(what, sizeof(what), "recorded refine grid %s %d threads", si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti]);
            snprintf(msg, sizeof(msg), "%s: the recorded refine runner builds", what);
            check(ok, msg);
            for (int sd_i = 0; ok && sd_i < 2; ++sd_i) {
                memset(x, 0, sizeof(x));
                memset(t, 0, sizeof(t));
                memset(done, 0, sizeof(done));
                for (int i = 0; i < GN; ++i) ones[i] = 1.0;
                struct eg_run_report rep;
                memset(&rep, 0, sizeof(rep));
                rep.struct_size = sizeof(rep);
                struct tensor a1, a2;
                eg_plane* tp = host_plane(ctx, &a1, GN, tstop, kDLFloat, 64);
                eg_plane* sp = host_plane(ctx, &a2, GN, status, kDLInt, 8);
                const int ran = eg_run(ctx, r, &rep) == EG_OK && eg_run_events(ctx, r, tp, sp) == EG_OK;
                int e_ok = ran;
                for (int i = 0; e_ok && i < GN; ++i) {
                    e_ok = memcmp(&x[i], &ex[i], 8) == 0 && memcmp(&t[i], &et[i], 8) == 0 && memcmp(&tstop[i], &ts[i], 8) == 0 && status[i] == st[i];
                    if (!e_ok) printf("      sample %d: x %.17g vs %.17g, t %.17g vs %.17g, t_stop %.17g vs %.17g, status %d vs %d\n", i, x[i], ex[i], t[i], et[i], tstop[i], ts[i], status[i], st[i]);
                }
                snprintf(msg, sizeof(msg), "%s, seed %s: the recorded run's states, stop times and statuses equal the old path's bitwise", what, seeds_of[sd_i]);
                check(e_ok, msg);
                struct tensor tg[4];
                static double gv[4][GN];
                eg_plane* gpl[4];
                for (int k = 0; k < 4; ++k) gpl[k] = host_plane(ctx, &tg[k], GN, gv[k], kDLFloat, 64);
                struct tensor tone;
                eg_plane* sdp = host_plane(ctx, &tone, GN, ones, kDLFloat, 64);
                const struct eg_bind_desc sd = binds(1, &seed_slot[sd_i], &sdp), gd = binds(4, grad_slot, gpl);
                const int rev = ran && eg_reverse(ctx, r, &sd, &gd) == EG_OK;
                snprintf(msg, sizeof(msg), "%s, seed %s: eg_run and eg_reverse are OK", what, seeds_of[sd_i]);
                check(rev, msg);
                for (int k = 0; rev && k < 4; ++k) {
                    char file[64];
                    snprintf(file, sizeof(file), "g_%s_%s.bin", seeds_of[sd_i], grads_of[k]);
                    static double want[GN];
                    int eq = read_file(dir, file, want, sizeof(want));
                    for (int i = 0; eq && i < GN; ++i) {
                        eq = memcmp(&gv[k][i], &want[i], 8) == 0;
                        if (!eq) printf("      %s[%d]: %.17g vs %.17g\n", file, i, gv[k][i], want[i]);
                    }
                    snprintf(msg, sizeof(msg), "%s, seed %s: the gradient of %s equals the old path's bitwise", what, seeds_of[sd_i], grads_of[k]);
                    check(eq, msg);
                }
                for (int k = 0; k < 4; ++k) eg_plane_free(ctx, gpl[k]);
                eg_plane_free(ctx, sdp);
                eg_plane_free(ctx, tp);
                eg_plane_free(ctx, sp);
            }
            for (int i = 0; i < 5; ++i) eg_plane_free(ctx, planes[i]);
            if (r) eg_runner_free(ctx, r);
            if (plan) eg_plan_free(ctx, plan);
        }
    eg_host_threads(ctx, 0);
    eg_unit_free(ctx, p);
    for (int i = 0; i < 5; ++i) eg_unit_free(ctx, units[i]);
    eg_ctx_free(ctx);
    return failures == 0 ? 0 : 1;
}

/* A per-sample read's gradient declared reduced 1 (on the step vjp or on the refine vjp): the recorded runner is refused by name. */
static int grid_refused(char** argv) {
    uint8_t dg[6][EG_DIGEST_BYTES];
    for (int i = 0; i < 6; ++i)
        if (!parse_digest(argv[3 + i], dg[i])) return 2;
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new host context");
    eg_unit* p = load(ctx, argv[1], "ev_grid", dg[0]);
    eg_unit* units[5] = { load(ctx, argv[1], "ev_grid_tape", dg[1]), load(ctx, argv[1], "ev_grid_step_vjp", dg[4]), load(ctx, argv[1], "ev_grid_newton", dg[2]),
                          load(ctx, argv[1], "ev_grid_state_guard_vjp", dg[3]), load(ctx, argv[1], "ev_grid_refine_vjp", dg[5]) };
    const int loaded = p && units[0] && units[1] && units[2] && units[3] && units[4];
    check(loaded, "refused grid: the units load");
    if (!loaded) {
        eg_ctx_free(ctx);
        return 1;
    }
    const eg_derived_role roles[5] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP, EG_DERIVED_REFINE_VJP };
    static double x[GN], t[GN], bt[GN], dtp[GN];
    uint8_t done[GN];
    struct tensor tt, tx, tbt, tdt, tdn;
    eg_plane* planes[5] = { host_plane(ctx, &tt, GN, t, kDLFloat, 64), host_plane(ctx, &tx, GN, x, kDLFloat, 64), host_plane(ctx, &tbt, GN, bt, kDLFloat, 64),
                            host_plane(ctx, &tdt, GN, dtp, kDLFloat, 64), host_plane(ctx, &tdn, GN, done, kDLBool, 8) };
    const int32_t slots[5] = { 0, 1, 2, 3, 5 };
    const struct eg_bind_desc bd = binds(5, slots, planes);
    eg_plan* plan = NULL;
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = p;
    pd.structure = EG_STRUCTURE_HOST_SERIAL;
    pd.policy = EG_REDUCE_COMPENSATED;
    pd.n_samples = GN;
    check(eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK, "refused grid: the plan binds");
    struct eg_runner_desc d = runner_desc(&plan, 80);
    d.events = EG_EVENTS_REFINE;
    d.on_no_event = EG_NO_EVENT_REFUSE;
    d.flags = EG_RUN_RECORD;
    d.window = 1;
    d.n_derived = 5;
    d.derived_roles = roles;
    d.derived_units = units;
    eg_runner* r = NULL;
    check(eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "reduced") && reason_has(ctx, "per-sample read"),
          "a per-sample read's gradient declared reduced 1 refused BAD_REQUEST naming 'reduced' and the per-sample read");
    if (r) eg_runner_free(ctx, r);
    for (int i = 0; i < 5; ++i) eg_plane_free(ctx, planes[i]);
    if (plan) eg_plan_free(ctx, plan);
    eg_unit_free(ctx, p);
    for (int i = 0; i < 5; ++i) eg_unit_free(ctx, units[i]);
    eg_ctx_free(ctx);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc != 9 && argc != 10) {   /* 10: the pendulum (ev_pend) or "grid" instead of the oscillator */
        fprintf(stderr, "usage: %s <pair lib> <oracle dir> <primal> <twin> <newton> <guard> <step vjp> <refine vjp>\n", argv[0]);
        return 2;
    }
    if (argc == 10 && (strcmp(argv[9], "grid") == 0 || strcmp(argv[9], "grid-refused") == 0)) {
        const int rc = argv[9][4] == '-' ? grid_refused(argv) : grid_rows(argv);
        printf("%s: %d failure(s)\n", argv[0], failures);
        return rc;
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
    static double x0[PN];
    if (!read_file(dir, "x0.bin", x0, sizeof(x0))) { fprintf(stderr, "no x0.bin\n"); return 2; }
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new host context");
    eg_unit* p = load(ctx, lib, n_p, dg[0]);
    eg_unit* units[5] = { load(ctx, lib, n_t, dg[1]), load(ctx, lib, n_s, dg[4]), load(ctx, lib, n_n, dg[2]),
                          load(ctx, lib, n_g, dg[3]), load(ctx, lib, n_r, dg[5]) };
    const int loaded = p && units[0] && units[1] && units[2] && units[3] && units[4];
    check(loaded, "recorded refine: ev_osc, its twin, step vjp, Newton, state guard vjp and refine vjp load from one library");
    if (!loaded) {
        eg_ctx_free(ctx);
        return 1;
    }
    const eg_derived_role roles[5] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP,
                                       EG_DERIVED_REFINE_VJP };
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int threads[2] = { 1, 3 };
    /* the uniforms by slot 5, 6, 7: the oscillator a, dt, w; the pendulum a, c, dt */
    const double uvals[3] = { pend ? 0.05 : 0.0, pend ? 0.2 : 0.05, pend ? 0.05 : 2.0 };
    const char* seeds_of[3] = { "t", "v", "x" };
    const int32_t seed_slot[3] = { 0, 1, 2 };
    const char* grads_osc[6] = { "t", "v", "x", "dt", "w", "a" }, *grads_pend[6] = { "t", "v", "x", "dt", "c", "a" };
    const char* const* grads_of = pend ? grads_pend : grads_osc;
    const int32_t grad_slot_osc[6] = { 0, 1, 2, 6, 7, 5 }, grad_slot_pend[6] = { 0, 1, 2, 7, 6, 5 };
    const int32_t* grad_slot = pend ? grad_slot_pend : grad_slot_osc;
    char what[256];
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            eg_host_threads(ctx, threads[ti]);
            static double x[PN], v[PN], t[PN], ones[PN];
            uint8_t done[PN];
            struct tensor tx, tv, tt, tdn, tone;
            eg_plane* planes[4] = { host_plane(ctx, &tt, PN, t, kDLFloat, 64), host_plane(ctx, &tv, PN, v, kDLFloat, 64),
                                    host_plane(ctx, &tx, PN, x, kDLFloat, 64), host_plane(ctx, &tdn, PN, done, kDLBool, 8) };
            const int32_t slots[4] = { 0, 1, 2, 4 }, us[3] = { 5, 6, 7 };
            const void* const uv[3] = { &uvals[0], &uvals[1], &uvals[2] };
            struct eg_bind_desc bd = binds(4, slots, planes);
            bd.n_uniforms = 3;
            bd.uniform_slots = us;
            bd.uniform_values = uv;
            eg_plan* plan = NULL;
            struct eg_plan_desc pd;
            memset(&pd, 0, sizeof(pd));
            pd.struct_size = sizeof(pd);
            pd.unit = p;
            pd.structure = structures[si];
            pd.policy = EG_REDUCE_COMPENSATED;
            pd.n_samples = PN;
            const int planned = eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK;
            struct eg_runner_desc d = runner_desc(&plan, 400);
            d.events = EG_EVENTS_REFINE;
            d.flags = EG_RUN_RECORD;
            d.window = 2;
            d.n_derived = 5;
            d.derived_roles = roles;
            d.derived_units = units;
            eg_runner* r = NULL;
            snprintf(what, sizeof(what), "recorded refine %s %d threads", si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti]);
            char msg[640];
            snprintf(msg, sizeof(msg), "%s: window 2 refused BAD_REQUEST naming 'window'", what);
            check(planned && eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "window"), msg);
            d.window = 1;
            const int ok = planned && eg_runner_new(ctx, &d, &r) == EG_OK;
            snprintf(msg, sizeof(msg), "%s: the recorded refine runner builds", what);
            check(ok, msg);
            for (int sd_i = 0; ok && sd_i < 3; ++sd_i) {
                memcpy(x, x0, sizeof(x));
                memset(v, 0, sizeof(v));
                memset(t, 0, sizeof(t));
                memset(done, 0, sizeof(done));
                for (int i = 0; i < PN; ++i) ones[i] = 1.0;
                struct eg_run_report rep;
                memset(&rep, 0, sizeof(rep));
                rep.struct_size = sizeof(rep);
                const int ran = eg_run(ctx, r, &rep) == EG_OK;
                struct tensor tg[6];
                static double gv[6][PN];
                eg_plane* gpl[6];
                for (int k = 0; k < 6; ++k) gpl[k] = host_plane(ctx, &tg[k], k < 3 ? PN : 1, gv[k], kDLFloat, 64);
                eg_plane* sp = host_plane(ctx, &tone, PN, ones, kDLFloat, 64);
                const struct eg_bind_desc sd = binds(1, &seed_slot[sd_i], &sp), gd = binds(6, grad_slot, gpl);
                const int rev = ran && eg_reverse(ctx, r, &sd, &gd) == EG_OK;
                snprintf(msg, sizeof(msg), "%s, seed %s: eg_run and eg_reverse are OK", what, seeds_of[sd_i]);
                check(rev, msg);
                for (int k = 0; rev && k < 6; ++k) {
                    char file[64];
                    snprintf(file, sizeof(file), "g_%s_%s.bin", seeds_of[sd_i], grads_of[k]);
                    static double want[PN];
                    const size_t cnt = k < 3 ? PN : 1;
                    const int have = read_file(dir, file, want, cnt * sizeof(double));
                    int eq = have;
                    for (size_t i = 0; eq && i < cnt; ++i) {
                        eq = memcmp(&gv[k][i], &want[i], 8) == 0;
                        if (!eq) printf("      %s[%zu]: %.17g vs %.17g\n", file, i, gv[k][i], want[i]);
                    }
                    snprintf(msg, sizeof(msg), "%s, seed %s: the gradient of %s equals the old path's bitwise", what, seeds_of[sd_i], grads_of[k]);
                    check(eq, msg);
                }
                for (int k = 0; k < 6; ++k) eg_plane_free(ctx, gpl[k]);
                eg_plane_free(ctx, sp);
            }
            for (int i = 0; i < 4; ++i) eg_plane_free(ctx, planes[i]);
            if (r) eg_runner_free(ctx, r);
            if (plan) eg_plan_free(ctx, plan);
        }
    eg_host_threads(ctx, 0);
    eg_unit_free(ctx, p);
    for (int i = 0; i < 5; ++i) eg_unit_free(ctx, units[i]);
    eg_ctx_free(ctx);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures == 0 ? 0 : 1;
}
