// SPDX-License-Identifier: Apache-2.0
/*
 * A recorded events refine run and its reverse (EG_EVENTS_REFINE with EG_RUN_RECORD, window 1; eg_reverse).
 *
 *   refine_reverse_test <pair lib> <oracle dir> <primal> <twin> <newton> <guard> <step vjp> <refine vjp> digests
 *
 * ev_osc (the old path's oscillator with a stop rule) recorded with its taping twin, step vjp, Newton, state guard vjp and
 * refine vjp; the reverse seeded with ones on t, v and x in turn must give the old path's gradients (the same run through
 * eagle.diff.reverse on cpu, central differences having checked them there) bit for bit, on SERIAL and TEAM with 1 and 3 threads.
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

static struct eg_bind_desc binds(int n_planes, const int32_t* slots, eg_plane* const* planes) {
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = n_planes;
    b.plane_slots = slots;
    b.planes = planes;
    return b;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s <pair lib> <oracle dir> <primal> <twin> <newton> <guard> <step vjp> <refine vjp>\n", argv[0]);
        return 2;
    }
    const char *lib = argv[1], *dir = argv[2];
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
    eg_unit* p = load(ctx, lib, "ev_osc", dg[0]);
    eg_unit* units[5] = { load(ctx, lib, "ev_osc_tape", dg[1]), load(ctx, lib, "ev_osc_step_vjp", dg[4]), load(ctx, lib, "ev_osc_newton", dg[2]),
                          load(ctx, lib, "ev_osc_state_guard_vjp", dg[3]), load(ctx, lib, "ev_osc_refine_vjp", dg[5]) };
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
    const double a = 0.0, dt = 0.05, w = 2.0;
    const char* seeds_of[3] = { "t", "v", "x" };
    const int32_t seed_slot[3] = { 0, 1, 2 };
    const char* grads_of[6] = { "t", "v", "x", "dt", "w", "a" };
    const int32_t grad_slot[6] = { 0, 1, 2, 6, 7, 5 };
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
            const void* const uv[3] = { &a, &dt, &w };
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
