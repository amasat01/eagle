// SPDX-License-Identifier: Apache-2.0
/*
 * events refine statuses beyond REFINED, on the host (a plain run).
 *
 *   refine_status_test <lib> grid|tangent|two <oracle dir> <primal> <twin> <newton> <guard>
 *
 * grid:    ev_grid, a horizon on the step grid (per-sample dt and big_t planes, 480 samples): every sample stops AT_STEP (2);
 *          the states, stop times and statuses equal the old path's bit for bit.
 * tangent: ev_tangent, a cubic surface crossing: refused BAD_REQUEST, "graze the surface" (the old path's ValueError class).
 * two:     ev_two, two stop rules crossing in one step: refused BAD_REQUEST, "two stop rules cross in the same step".
 * Each on SERIAL and TEAM with 1 and 3 threads.
 */
#include "pair_util.h"

#include <stdlib.h>

#define GN 480
#define SN 4

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
    if (argc != 8) {
        fprintf(stderr, "usage: %s <lib> grid|tangent|two <oracle dir> <primal> <twin> <newton> <guard>\n", argv[0]);
        return 2;
    }
    const char *lib = argv[1], *mode = argv[2], *dir = argv[3];
    const int grid = strcmp(mode, "grid") == 0, tangent = strcmp(mode, "tangent") == 0;
    const int n = grid ? GN : SN;
    char nm[4][48];
    snprintf(nm[0], sizeof(nm[0]), "ev_%s", mode);
    snprintf(nm[1], sizeof(nm[1]), "ev_%s_tape", mode);
    snprintf(nm[2], sizeof(nm[2]), "ev_%s_newton", mode);
    snprintf(nm[3], sizeof(nm[3]), "ev_%s_state_guard_vjp", mode);
    uint8_t dg[4][EG_DIGEST_BYTES];
    for (int i = 0; i < 4; ++i)
        if (!parse_digest(argv[4 + i], dg[i])) return 2;
    static double h[GN], bigt[GN], ex[GN], et[GN], ts[GN];
    static int8_t st[GN];
    if (grid && !(read_file(dir, "h.bin", h, sizeof(h)) && read_file(dir, "bigt.bin", bigt, sizeof(bigt)) && read_file(dir, "end_x.bin", ex, sizeof(ex)) &&
                  read_file(dir, "end_t.bin", et, sizeof(et)) && read_file(dir, "t_stop.bin", ts, sizeof(ts)) && read_file(dir, "status.bin", st, sizeof(st)))) {
        check(0, "status: the oracle files read");
        return 1;
    }
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&s, &ctx) == EG_OK, "eg_ctx_new host context");
    eg_unit* p = load(ctx, lib, nm[0], dg[0]);
    eg_unit* units[3] = { load(ctx, lib, nm[1], dg[1]), load(ctx, lib, nm[2], dg[2]), load(ctx, lib, nm[3], dg[3]) };
    const int loaded = p && units[0] && units[1] && units[2];
    check(loaded, "status: the primal, its twin, its Newton and its state guard vjp load from one library");
    if (!loaded) {
        eg_ctx_free(ctx);
        return 1;
    }
    const eg_derived_role roles[3] = { EG_DERIVED_TAPE, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP };
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int threads[2] = { 1, 3 };
    char what[256], msg[512];
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            eg_host_threads(ctx, threads[ti]);
            static double t[GN], x[GN], v[GN], bt[GN], dtp[GN], tstop[GN];
            static int8_t status[GN];
            uint8_t done[GN];
            memset(t, 0, sizeof(t));
            memset(v, 0, sizeof(v));
            memset(done, 0, sizeof(done));
            for (int i = 0; i < n; ++i) x[i] = (grid || tangent) ? 0.0 : 1.0;
            memcpy(bt, bigt, sizeof(bt));
            memcpy(dtp, h, sizeof(dtp));
            const double u_two[2] = { 0.8, 0.1 }, u_tan = 0.125;
            struct tensor tt, tx, tv, tbt, tdt, tdn;
            eg_plane* planes[6];
            int32_t slots[6];
            int np_ = 0;
            planes[np_] = host_plane(ctx, &tt, n, t, kDLFloat, 64); slots[np_++] = 0;
            if (grid) {
                planes[np_] = host_plane(ctx, &tx, n, x, kDLFloat, 64); slots[np_++] = 1;
                planes[np_] = host_plane(ctx, &tbt, n, bt, kDLFloat, 64); slots[np_++] = 2;
                planes[np_] = host_plane(ctx, &tdt, n, dtp, kDLFloat, 64); slots[np_++] = 3;
                planes[np_] = host_plane(ctx, &tdn, n, done, kDLBool, 8); slots[np_++] = 5;
            } else if (tangent) {
                planes[np_] = host_plane(ctx, &tx, n, x, kDLFloat, 64); slots[np_++] = 1;
                planes[np_] = host_plane(ctx, &tdn, n, done, kDLBool, 8); slots[np_++] = 3;
            } else {
                planes[np_] = host_plane(ctx, &tv, n, v, kDLFloat, 64); slots[np_++] = 1;
                planes[np_] = host_plane(ctx, &tx, n, x, kDLFloat, 64); slots[np_++] = 2;
                planes[np_] = host_plane(ctx, &tdn, n, done, kDLBool, 8); slots[np_++] = 4;
            }
            const int32_t us_tan[1] = { 4 }, us_two[2] = { 5, 6 };
            const void* const uv_tan[1] = { &u_tan }, *uv_two[2] = { &u_two[0], &u_two[1] };
            struct eg_bind_desc bd;
            memset(&bd, 0, sizeof(bd));
            bd.struct_size = sizeof(bd);
            bd.n_planes = np_;
            bd.plane_slots = slots;
            bd.planes = planes;
            bd.n_uniforms = grid ? 0 : tangent ? 1 : 2;
            bd.uniform_slots = tangent ? us_tan : us_two;
            bd.uniform_values = tangent ? uv_tan : uv_two;
            eg_plan* plan = NULL;
            struct eg_plan_desc pd;
            memset(&pd, 0, sizeof(pd));
            pd.struct_size = sizeof(pd);
            pd.unit = p;
            pd.structure = structures[si];
            pd.policy = EG_REDUCE_COMPENSATED;
            pd.n_samples = n;
            const int planned = eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK;
            struct eg_runner_desc d = runner_desc(&plan, grid ? 80 : 50);
            d.events = EG_EVENTS_REFINE;
            d.on_no_event = EG_NO_EVENT_REFUSE;
            d.n_derived = 3;
            d.derived_roles = roles;
            d.derived_units = units;
            eg_runner* r = NULL;
            struct eg_run_report rep;
            memset(&rep, 0, sizeof(rep));
            rep.struct_size = sizeof(rep);
            const int ok = planned && eg_runner_new(ctx, &d, &r) == EG_OK;
            snprintf(what, sizeof(what), "%s %s %d threads", mode, si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti]);
            if (!ok) reason_has(ctx, "(the runner builds)");
            if (grid) {
                struct tensor a1, a2;
                eg_plane* tp = host_plane(ctx, &a1, n, tstop, kDLFloat, 64);
                eg_plane* sp = host_plane(ctx, &a2, n, status, kDLInt, 8);
                const int ran = ok && eg_run(ctx, r, &rep) == EG_OK && eg_run_events(ctx, r, tp, sp) == EG_OK;
                int e_ok = ran;
                for (int i = 0; e_ok && i < n; ++i) {
                    e_ok = memcmp(&x[i], &ex[i], 8) == 0 && memcmp(&t[i], &et[i], 8) == 0 && memcmp(&tstop[i], &ts[i], 8) == 0 && status[i] == st[i];
                    if (!e_ok) printf("      sample %d: x %.17g vs %.17g, t %.17g vs %.17g, t_stop %.17g vs %.17g, status %d vs %d\n", i, x[i], ex[i], t[i], et[i], tstop[i], ts[i], status[i], st[i]);
                }
                int all2 = ran;
                for (int i = 0; i < n; ++i) all2 = all2 && status[i] == 2;
                snprintf(msg, sizeof(msg), "%s: the states, stop times and statuses equal the old path's bitwise", what);
                check(e_ok, msg);
                snprintf(msg, sizeof(msg), "%s: every status is AT_STEP (2)", what);
                check(all2, msg);
                eg_plane_free(ctx, tp);
                eg_plane_free(ctx, sp);
            } else {
                snprintf(msg, sizeof(msg), "%s: refused BAD_REQUEST naming '%s'", what, tangent ? "graze the surface" : "two stop rules cross in the same step");
                check(ok && eg_run(ctx, r, &rep) == EG_BAD_REQUEST && reason_has(ctx, tangent ? "graze the surface" : "two stop rules cross in the same step"), msg);
            }
            for (int i = 0; i < np_; ++i) eg_plane_free(ctx, planes[i]);
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
