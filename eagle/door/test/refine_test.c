// SPDX-License-Identifier: Apache-2.0
/*
 * events refine on the host (EG_EVENTS_REFINE, a plain run) and eg_run_events.
 *
 *   refine_test <pair lib> <oracle dir> <primal digest> <twin digest> <newton digest> <guard digest> <max_steps> <on_no_event 0|1> <compare 0|1>
 *
 * ev_osc (the old path's oscillator with a stop rule) run with the taping twin, the Newton and the state guard vjp;
 * its end states, stop times and statuses must equal the old path's bit for bit. A run whose samples never reach
 * their stop (max_steps 3) is refused by name under EG_NO_EVENT_REFUSE (on_no_event 0) and, under EG_NO_EVENT_ZERO
 * (1), gives the old path's stepped stop times and status 3.
 */
#include "pair_util.h"

#include <stdlib.h>

#define PN 64

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

int main(int argc, char** argv) {
    if (argc != 10 && argc != 11) {   /* 11: the pendulum (ev_pend: a 0.05, c 0.2, dt 0.05) instead of the oscillator */
        fprintf(stderr, "usage: %s <pair lib> <oracle dir> <primal digest> <twin digest> <newton digest> <guard digest> <max_steps> <on_no_event> <compare>\n", argv[0]);
        return 2;
    }
    const char *lib = argv[1], *dir = argv[2];
    const int64_t max_steps = atol(argv[7]);
    const int zero = atoi(argv[8]), compare = atoi(argv[9]);
    const int pend = argc == 11;
    char n_p[16], n_t[64], n_n[64], n_g[64];
    snprintf(n_p, sizeof(n_p), pend ? "ev_pend" : "ev_osc");
    snprintf(n_t, sizeof(n_t), "%s_tape", n_p);
    snprintf(n_n, sizeof(n_n), "%s_newton", n_p);
    snprintf(n_g, sizeof(n_g), "%s_state_guard_vjp", n_p);
    uint8_t dg[4][EG_DIGEST_BYTES];
    static double x0[PN], ex[PN], ev[PN], et[PN], ts[PN];
    static int8_t st[PN];
    for (int i = 0; i < 4; ++i)
        if (!parse_digest(argv[3 + i], dg[i])) return 2;
    const int have = read_file(dir, "x0.bin", x0, sizeof(x0)) && read_file(dir, "end_x.bin", ex, sizeof(ex)) &&
                     read_file(dir, "end_v.bin", ev, sizeof(ev)) && read_file(dir, "end_t.bin", et, sizeof(et)) &&
                     read_file(dir, "t_stop.bin", ts, sizeof(ts)) && read_file(dir, "status.bin", st, sizeof(st));
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new host context");
    eg_unit* p = load(ctx, lib, n_p, dg[0]);
    eg_unit* units[3] = { load(ctx, lib, n_t, dg[1]), load(ctx, lib, n_n, dg[2]), load(ctx, lib, n_g, dg[3]) };
    check(p && units[0] && units[1] && units[2], "refine: ev_osc, its twin, its Newton and its state guard vjp load from one library");
    if (!(p && units[0] && units[1] && units[2]) || !(have || !compare)) {
        check(have || !compare, "refine: the oracle files read");
        eg_ctx_free(ctx);
        return 1;
    }
    const eg_derived_role roles[3] = { EG_DERIVED_TAPE, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP };
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int threads[2] = { 1, 3 };
    /* the uniforms by slot 5, 6, 7: the oscillator a, dt, w; the pendulum a, c, dt */
    const double uvals[3] = { pend ? 0.05 : 0.0, pend ? 0.2 : 0.05, pend ? 0.05 : 2.0 };
    char what[256];
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            eg_host_threads(ctx, threads[ti]);
            static double x[PN], v[PN], t[PN], tstop[PN];
            static int8_t status[PN];
            uint8_t done[PN];
            memcpy(x, x0, sizeof(x));
            memset(v, 0, sizeof(v));
            memset(t, 0, sizeof(t));
            memset(done, 0, sizeof(done));
            struct tensor tx, tv, tt, tdn;
            eg_plane* planes[4] = { host_plane(ctx, &tt, PN, t, kDLFloat, 64), host_plane(ctx, &tv, PN, v, kDLFloat, 64),
                                    host_plane(ctx, &tx, PN, x, kDLFloat, 64), host_plane(ctx, &tdn, PN, done, kDLBool, 8) };
            const int32_t slots[4] = { 0, 1, 2, 4 }, us[3] = { 5, 6, 7 };
            const void* const uv[3] = { &uvals[0], &uvals[1], &uvals[2] };
            struct eg_bind_desc bd;
            memset(&bd, 0, sizeof(bd));
            bd.struct_size = sizeof(bd);
            bd.n_planes = 4;
            bd.plane_slots = slots;
            bd.planes = planes;
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
            const int planned = eg_plan_new(ctx, &pd, &plan) == EG_OK;
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
            const int bound = planned && eg_bind(ctx, plan, &bd) == EG_OK;
            if (zero == 9) {    /* the library states n_events 2 for one guard */
                check(bound && eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "n_events"),
                      "refine: guards that do not cover the primal's finish.n_events refused BAD_REQUEST naming n_events");
                for (int i = 0; i < 4; ++i) eg_plane_free(ctx, planes[i]);
                if (plan) eg_plan_free(ctx, plan);
                si = ti = 2;
                continue;
            }
            const int ok = bound && eg_runner_new(ctx, &d, &r) == EG_OK;
            if (!ok) reason_has(ctx, "(the refine runner builds)");
            struct tensor a1, a2;
            eg_plane* tp = host_plane(ctx, &a1, PN, tstop, kDLFloat, 64);
            eg_plane* sp = host_plane(ctx, &a2, PN, status, kDLInt, 8);
            snprintf(what, 200, "refine %s %d threads", si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti]);
            char msg[640];
            check(ok && eg_run_events(ctx, r, tp, sp) == EG_BAD_REQUEST && reason_has(ctx, "no refine run"),
                  "eg_run_events before any run refused BAD_REQUEST 'no refine run'");
            if (!compare) {
                snprintf(msg, sizeof(msg), "%s: no crossing in the stop step refused BAD_REQUEST by name (EG_NO_EVENT_REFUSE)", what);
                check(ok && eg_run(ctx, r, &rep) == EG_BAD_REQUEST && reason_has(ctx, "no crossing in their stop step"), msg);
            } else {
                const int ran = ok && eg_run(ctx, r, &rep) == EG_OK && eg_run_events(ctx, r, tp, sp) == EG_OK;
                int e_ok = ran;
                for (int i = 0; e_ok && i < PN; ++i) {
                    const int cx = memcmp(&x[i], &ex[i], 8) == 0, cv = memcmp(&v[i], &ev[i], 8) == 0, ct = memcmp(&t[i], &et[i], 8) == 0,
                              cs = memcmp(&tstop[i], &ts[i], 8) == 0;
                    e_ok = cx && cv && ct && cs && status[i] == st[i];
                    if (!e_ok)
                        printf("      sample %d: x %d v %d t %d t_stop %d (%.17g vs %.17g) status %d vs %d\n", i, cx, cv, ct, cs, tstop[i],
                               ts[i], status[i], st[i]);
                }
                snprintf(msg, sizeof(msg), "%s: the refined states, stop times and statuses equal the old path's bitwise (status %d, t_stop %.17g)",
                         what, ran ? status[0] : -1, tstop[0]);
                check(e_ok, msg);
                int64_t binned = 0;
                for (int b = 0; b < EG_K_BINS; ++b) binned += rep.launches_by_k[b];
                check(ran && rep.mode == EG_MODE_REFINE && binned == rep.launches,
                      "a refine run reports EG_MODE_REFINE and its launches_by_k sum to its launches");
                struct tensor bad;
                float f32[PN];
                eg_plane* bp = host_plane(ctx, &bad, PN, f32, kDLFloat, 32);
                check(ran && eg_run_events(ctx, r, bp, NULL) == EG_BAD_REQUEST && reason_has(ctx, "plane shape or dtype"),
                      "eg_run_events with a float32 time plane refused BAD_REQUEST 'plane shape or dtype'");
                eg_plane_free(ctx, bp);
                check(ran && eg_run_events(ctx, r, NULL, NULL) == EG_OK, "eg_run_events with both planes NULL is OK");
            }
            eg_plane_free(ctx, tp);
            eg_plane_free(ctx, sp);
            for (int i = 0; i < 4; ++i) eg_plane_free(ctx, planes[i]);
            if (r) eg_runner_free(ctx, r);
            if (plan) eg_plan_free(ctx, plan);
        }
    eg_host_threads(ctx, 0);
    eg_unit_free(ctx, p);
    for (int i = 0; i < 3; ++i) eg_unit_free(ctx, units[i]);
    eg_ctx_free(ctx);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures == 0 ? 0 : 1;
}
