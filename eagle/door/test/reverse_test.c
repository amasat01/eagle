// SPDX-License-Identifier: Apache-2.0
/*
 * eg_reverse of an EG_RUN_RECORD run on the host.
 *
 *   reverse_test <fixture_reverse lib> [<pair lib> <oracle dir> <primal digest> <twin digest> <vjp digest>]
 *
 * The fixture rows: the step x <- a*x + b (t <- t + 1, each sample stopping at
 * its own T) recorded with windows 1, 3 and 16, one pass and several passes,
 * on HOST_SERIAL and HOST_TEAM with 1 and 3 threads, against its closed-form
 * end state and gradients; then every refusal by status and reason. The pair
 * rows: rr_pend (the damped pendulum of the old path's recorded-run tests),
 * whose states and gradients must equal the old path's bit for bit.
 */
#include "pair_util.h"

#include <math.h>
#include <stdlib.h>

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

static eg_unit* load_byte(eg_ctx* ctx, const char* path, const char* name, uint8_t byte) {
    uint8_t digest[EG_DIGEST_BYTES];
    memset(digest, byte, sizeof(digest));
    return load(ctx, path, name, digest);
}

static eg_plan* plan_of(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t n) {
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = structure;
    d.policy = EG_REDUCE_COMPENSATED;
    d.n_samples = n;
    eg_plan* p = NULL;
    return eg_plan_new(ctx, &d, &p) == EG_OK ? p : NULL;
}

static struct eg_bind_desc binds(int32_t n_planes, const int32_t* slots, eg_plane* const* planes, int32_t n_uniforms,
                                 const int32_t* uslots, const void* const* values) {
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = n_planes;
    d.plane_slots = slots;
    d.planes = planes;
    d.n_uniforms = n_uniforms;
    d.uniform_slots = uslots;
    d.uniform_values = values;
    return d;
}

static struct eg_runner_desc recorded(eg_plan* const* plan, int64_t max_steps, int64_t window, uint64_t budget,
                                      const eg_derived_role* roles, eg_unit* const* units) {
    struct eg_runner_desc d = runner_desc(plan, max_steps);
    d.flags = EG_RUN_RECORD;
    d.n_derived = 2;
    d.derived_roles = roles;
    d.derived_units = units;
    d.window = window;
    d.memory_budget = budget;
    return d;
}

/* ---- the closed-form fixture ---- */

#define LN 37
#define LSTEPS 10
static const double LA = 0.5, LB = 0.25;

struct lin {
    double x[LN], t[LN], T[LN];
    uint8_t done[LN];
    struct tensor tx, tt, tT, tdone;
    eg_plane *px, *pt, *pT, *pdone;
};

static int lin_bind(eg_ctx* ctx, eg_plan* plan, struct lin* s) {
    for (int i = 0; i < LN; ++i) {
        s->x[i] = 1.0 + i / 8.0;
        s->t[i] = 0.0;
        s->T[i] = 1.0 + (double)((i * 7) % LSTEPS);
        s->done[i] = 0;
    }
    s->px = host_plane(ctx, &s->tx, LN, s->x, kDLFloat, 64);
    s->pt = host_plane(ctx, &s->tt, LN, s->t, kDLFloat, 64);
    s->pT = host_plane(ctx, &s->tT, LN, s->T, kDLFloat, 64);
    s->pdone = host_plane(ctx, &s->tdone, LN, s->done, kDLBool, 8);
    const int32_t slots[] = { 0, 1, 2, 4 };
    eg_plane* const planes[] = { s->px, s->pt, s->pdone, s->pT };
    const int32_t us[] = { 5, 6 };
    const void* const uv[] = { &LA, &LB };
    const struct eg_bind_desc d = binds(4, slots, planes, 2, us, uv);
    return s->px && s->pt && s->pT && s->pdone && eg_bind(ctx, plan, &d) == EG_OK;
}
static void lin_free(eg_ctx* ctx, struct lin* s) {
    eg_plane_free(ctx, s->px);
    eg_plane_free(ctx, s->pt);
    eg_plane_free(ctx, s->pT);
    eg_plane_free(ctx, s->pdone);
}

/* Closed form for a sample of k steps from x0: the end state and the gradients of x_k. */
static void lin_closed(double x0, int k, double* xk, double* dx0, double* da, double* db) {
    double x = x0, p = 1.0, sa = 0.0, sb = 0.0;
    double xs[LSTEPS];
    for (int j = 0; j < k; ++j) {
        xs[j] = x;
        x = LA * x + LB;
    }
    for (int j = k - 1; j >= 0; --j) {    /* d x_k / d x_j = a^(k-1-j) */
        sa += xs[j] * p;
        sb += p;
        p *= LA;
    }
    *xk = x;
    *dx0 = p;
    *da = sa;
    *db = sb;
}

static int close_to(double got, double want) { return fabs(got - want) <= 1e-13 * fmax(1.0, fabs(want)); }

static void lin_rows(eg_ctx* ctx, const char* lib) {
    eg_unit* lin = load_byte(ctx, lib, "lin", 0xE0);
    eg_unit* twins[3] = { load_byte(ctx, lib, "lin", 0xE1), load_byte(ctx, lib, "lin", 0xE3), load_byte(ctx, lib, "lin", 0xE6) };
    const int64_t windows[3] = { 1, 3, 16 };
    eg_unit* vjp = load_byte(ctx, lib, "lin_vjp", 0xE4);
    check(lin && twins[0] && twins[1] && twins[2] && vjp, "fixture: the primal, three taping twins and the step vjp load");
    if (!(lin && twins[0] && twins[1] && twins[2] && vjp)) return;
    double want_x[LN], want_dx[LN], want_da = 0.0, want_db = 0.0;
    for (int i = 0; i < LN; ++i) {
        double da, db;
        lin_closed(1.0 + i / 8.0, 1 + (i * 7) % LSTEPS, &want_x[i], &want_dx[i], &da, &db);
        want_da += da;
        want_db += db;
    }
    const eg_derived_role roles[2] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP };
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int threads[2] = { 1, 3 };
    char what[256];
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti)
            for (int w = 0; w < 3; ++w)
                for (int pass = 0; pass < (w == 1 ? 2 : 1); ++pass) {
                    /* several passes: ten samples' checkpoints and tape (25 bytes a checkpoint, 16 a tape row) */
                    const uint64_t budget = pass ? 10u * (4u * 25u + 3u * 16u) : 0u;
                    eg_host_threads(ctx, threads[ti]);
                    eg_plan* plan = plan_of(ctx, lin, structures[si], LN);
                    struct lin s, s2;
                    int rebound = 0;
                    eg_unit* const units[2] = { twins[w], vjp };
                    eg_runner* r = NULL;
                    const struct eg_runner_desc d = recorded(&plan, LSTEPS, windows[w], budget, roles, units);
                    struct eg_run_report rep;
                    memset(&rep, 0, sizeof(rep));
                    rep.struct_size = sizeof(rep);
                    const int ok = plan && lin_bind(ctx, plan, &s) && eg_runner_new(ctx, &d, &r) == EG_OK &&
                                   eg_run(ctx, r, &rep) == EG_OK;
                    int x_ok = ok && rep.steps == LSTEPS && rep.finished == LN && (rep.flags & EG_REPORT_DONE);
                    for (int i = 0; x_ok && i < LN; ++i) x_ok = s.x[i] == want_x[i] && s.done[i];
                    snprintf(what, sizeof(what), "fixture %s %d threads, window %lld%s: the recorded run ends on the closed form "
                             "(%lld steps, all done)", si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti], (long long)windows[w],
                             pass ? ", several passes" : "", (long long)rep.steps);
                    check(x_ok, what);
                    double seed[LN], gx[LN], gt[LN], ga = -1.0, gb = -1.0;
                    for (int i = 0; i < LN; ++i) seed[i] = 1.0;
                    struct tensor ts, tgx, tgt, tga, tgb;
                    eg_plane* ps = host_plane(ctx, &ts, LN, seed, kDLFloat, 64);
                    eg_plane* pgx = host_plane(ctx, &tgx, LN, gx, kDLFloat, 64);
                    eg_plane* pgt = host_plane(ctx, &tgt, LN, gt, kDLFloat, 64);
                    eg_plane* pga = host_plane(ctx, &tga, 1, &ga, kDLFloat, 64);
                    eg_plane* pgb = host_plane(ctx, &tgb, 1, &gb, kDLFloat, 64);
                    const int32_t sslot[] = { 0 }, gslot[] = { 0, 1, 5, 6 };
                    eg_plane* const sp[] = { ps };
                    eg_plane* const gp[] = { pgx, pgt, pga, pgb };
                    const struct eg_bind_desc sd = binds(1, sslot, sp, 0, NULL, NULL), gd = binds(4, gslot, gp, 0, NULL, NULL);
                    const int rok = ok && eg_reverse(ctx, r, &sd, &gd) == EG_OK;
                    int g_ok = rok && close_to(ga, want_da) && close_to(gb, want_db);
                    for (int i = 0; g_ok && i < LN; ++i) g_ok = close_to(gx[i], want_dx[i]) && gt[i] == 0.0;
                    snprintf(what, sizeof(what), "fixture %s %d threads, window %lld%s: eg_reverse gives the closed-form "
                             "gradients (x0 per sample, a and b summed: %.17g, %.17g)", si ? "HOST_TEAM" : "HOST_SERIAL",
                             threads[ti], (long long)windows[w], pass ? ", several passes" : "", ga, gb);
                    check(g_ok, what);
                    if (si == 0 && ti == 0 && w == 1 && pass == 0) {
                        check(eg_reverse(ctx, r, &sd, &gd) == EG_BAD_REQUEST && reason_has(ctx, "reversed once"),
                              "a record is reversed once: a second eg_reverse refused BAD_REQUEST");
                        const int32_t unknown[] = { 99 }, unlinked[] = { 4 };
                        struct eg_run_report again;
                        memset(&again, 0, sizeof(again));
                        again.struct_size = sizeof(again);
                        rebound = lin_bind(ctx, plan, &s2);
                        check(rebound && eg_run(ctx, r, &again) == EG_OK && eg_run(ctx, r, &again) == EG_BAD_REQUEST &&
                                  reason_has(ctx, "not reversed yet"),
                              "a second eg_run before eg_reverse refused BAD_REQUEST (the record is still live)");
                        struct eg_bind_desc bad = binds(1, unknown, sp, 0, NULL, NULL);
                        check(eg_reverse(ctx, r, &bad, &gd) == EG_BAD_REQUEST && reason_has(ctx, "unknown slot"),
                              "a seed on an unknown slot refused BAD_REQUEST");
                        bad = binds(1, unlinked, sp, 0, NULL, NULL);
                        check(eg_reverse(ctx, r, &bad, &gd) == EG_BAD_REQUEST && reason_has(ctx, "no link"),
                              "a seed on a slot that is not a taped state refused BAD_REQUEST (no link)");
                        bad = binds(1, unlinked, gp, 0, NULL, NULL);
                        check(eg_reverse(ctx, r, &sd, &bad) == EG_BAD_REQUEST && reason_has(ctx, "no link"),
                              "a gradient on a slot with no link refused BAD_REQUEST");
                        float f32[LN];
                        struct tensor tf;
                        eg_plane* pf = host_plane(ctx, &tf, LN, f32, kDLFloat, 32);
                        eg_plane* const fp[] = { pf };
                        bad = binds(1, sslot, fp, 0, NULL, NULL);
                        check(eg_reverse(ctx, r, &bad, &gd) == EG_BAD_REQUEST && reason_has(ctx, "dtype mismatch"),
                              "a float32 seed for a float64 state refused BAD_REQUEST (a dtype mismatch)");
                        eg_plane_free(ctx, pf);
                        check(eg_record_begin(ctx) == EG_OK && eg_reverse(ctx, r, &sd, &gd) == EG_NOT_WRITTEN &&
                                  reason_has(ctx, "inside a recording") && eg_run(ctx, r, &again) == EG_NOT_WRITTEN &&
                                  reason_has(ctx, "inside a recording"),
                              "eg_reverse and eg_run of a recorded runner inside a recording refused NOT_WRITTEN");
                        eg_graph* g = NULL;
                        eg_record_end(ctx, &g);
                        if (g) eg_graph_free(ctx, g);
                    }
                    eg_plane_free(ctx, ps);
                    eg_plane_free(ctx, pgx);
                    eg_plane_free(ctx, pgt);
                    eg_plane_free(ctx, pga);
                    eg_plane_free(ctx, pgb);
                    eg_runner_free(ctx, r);
                    eg_plan_free(ctx, plan);
                    lin_free(ctx, &s);
                    if (rebound) lin_free(ctx, &s2);
                }
    eg_host_threads(ctx, 0);
    /* the end state of a plain run equals the recorded run's */
    {
        eg_plan* plan = plan_of(ctx, lin, EG_STRUCTURE_HOST_TEAM, LN);
        struct lin s;
        eg_runner* r = NULL;
        const struct eg_runner_desc d = runner_desc(&plan, LSTEPS);
        struct eg_run_report rep;
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        int ok = plan && lin_bind(ctx, plan, &s) && eg_runner_new(ctx, &d, &r) == EG_OK && eg_run(ctx, r, &rep) == EG_OK;
        for (int i = 0; ok && i < LN; ++i) ok = s.x[i] == want_x[i];
        check(ok, "fixture: the plain run of the primal ends on the same state as the recorded run");
        eg_runner_free(ctx, r);
        eg_plan_free(ctx, plan);
        lin_free(ctx, &s);
    }
    /* refusals of eg_runner_new, each by status and reason */
    eg_plan* plan = plan_of(ctx, lin, EG_STRUCTURE_HOST_TEAM, LN);
    struct lin s;
    lin_bind(ctx, plan, &s);
    eg_runner* none = NULL;
    eg_unit* const good[2] = { twins[1], vjp };
    struct eg_runner_desc d = recorded(&plan, LSTEPS, 3, 0, roles, good);
    d.n_derived = 0;
    check(eg_runner_new(ctx, &d, &none) == EG_BAD_REQUEST && reason_has(ctx, "needs its derived units"),
          "EG_RUN_RECORD with no derived units refused BAD_REQUEST");
    const eg_derived_role swapped[2] = { EG_DERIVED_STEP_VJP, EG_DERIVED_TAPE };
    d = recorded(&plan, LSTEPS, 3, 0, swapped, good);
    check(eg_runner_new(ctx, &d, &none) == EG_BAD_REQUEST && reason_has(ctx, "derived.role"),
          "a role that is not the unit's derived.role refused BAD_REQUEST");
    const eg_derived_role fused[2] = { EG_DERIVED_TAPE_FUSED, EG_DERIVED_STEP_VJP };
    d = recorded(&plan, LSTEPS, 3, 0, fused, good);
    check(eg_runner_new(ctx, &d, &none) == EG_NOT_WRITTEN && reason_has(ctx, "steps_reverse fused"),
          "the K-step taping twin refused NOT_WRITTEN by name");
    const eg_derived_role fvjp[2] = { EG_DERIVED_TAPE, EG_DERIVED_FUSED_VJP };
    d = recorded(&plan, LSTEPS, 3, 0, fvjp, good);
    check(eg_runner_new(ctx, &d, &none) == EG_NOT_WRITTEN && reason_has(ctx, "steps_reverse fused"),
          "the K-step vjp refused NOT_WRITTEN by name");
    const eg_derived_role refine[2] = { EG_DERIVED_TAPE, EG_DERIVED_REFINE_VJP };
    d = recorded(&plan, LSTEPS, 3, 0, refine, good);
    check(eg_runner_new(ctx, &d, &none) == EG_NOT_WRITTEN && reason_has(ctx, "events refine"),
          "a refine role under record refused NOT_WRITTEN by name");
    d = recorded(&plan, LSTEPS, 3, 0, roles, good);
    d.steps_reverse = EG_STEPS_REVERSE_FUSED;
    check(eg_runner_new(ctx, &d, &none) == EG_NOT_WRITTEN && reason_has(ctx, "steps_reverse"),
          "steps_reverse FUSED refused NOT_WRITTEN by name");
    d = recorded(&plan, LSTEPS, 3, 0, roles, good);
    d.events = EG_EVENTS_REFINE;
    check(eg_runner_new(ctx, &d, &none) == EG_NOT_WRITTEN && reason_has(ctx, "events REFINE"),
          "events refine under record refused NOT_WRITTEN by name");
    d = recorded(&plan, LSTEPS, 4, 0, roles, good);
    check(eg_runner_new(ctx, &d, &none) == EG_BAD_REQUEST && reason_has(ctx, "window"),
          "a window other than the twin's built-in one refused BAD_REQUEST");
    d = recorded(&plan, LSTEPS, 0, 0, roles, good);
    check(eg_runner_new(ctx, &d, &none) == EG_BAD_REQUEST && reason_has(ctx, "window"),
          "window 0 (the default) refused BAD_REQUEST: the twin builds its window in");
    eg_plan* two[2] = { plan, plan };
    d = recorded(two, LSTEPS, 3, 0, roles, good);
    d.n_plans = 2;
    check(eg_runner_new(ctx, &d, &none) != EG_OK, "a several-plan step under record refused");
    const struct { uint8_t twin, vjp; eg_status status; const char* reason, *what; } broken[] = {
        { 0xF0, 0xE4, EG_BAD_REQUEST, "of_unit", "a twin whose of_unit is not the runner's primal" },
        { 0xF1, 0xE4, EG_BAD_REQUEST, "tape_rows", "tape_rows other than the sum of the taped rows" },
        { 0xF2, 0xE4, EG_BAD_REQUEST, "links[5].slot", "a twin link to a slot out of range" },
        { 0xF3, 0xE4, EG_BAD_REQUEST, "(TIME0, VALUE)", "a start-copy link on a slot that is not time0_slot (TIME0, VALUE)" },
        { 0xE3, 0xF4, EG_BAD_REQUEST, "has no link", "a vjp slot with no link" },
        { 0xE3, 0xF5, EG_BAD_REQUEST, "second link", "two links on one vjp slot" },
        { 0xE3, 0xF7, EG_BAD_REQUEST, "step plane", "a step-plane link on a real slot (STEPS, VALUE)" },
        { 0xE3, 0xF8, EG_BAD_REQUEST, "reduced", "a parameter gradient not summed over samples" },
    };
    for (size_t i = 0; i < sizeof(broken) / sizeof(broken[0]); ++i) {
        eg_unit* t = load_byte(ctx, lib, "lin", broken[i].twin);
        eg_unit* v = load_byte(ctx, lib, "lin_vjp", broken[i].vjp);
        eg_unit* const us[2] = { t, v };
        d = recorded(&plan, LSTEPS, 3, 0, roles, us);
        snprintf(what, sizeof(what), "%s refused %s naming '%s'", broken[i].what,
                 broken[i].status == EG_BAD_REQUEST ? "BAD_REQUEST" : "NOT_WRITTEN", broken[i].reason);
        check(t && v && eg_runner_new(ctx, &d, &none) == broken[i].status && reason_has(ctx, broken[i].reason), what);
        eg_unit_free(ctx, t);
        eg_unit_free(ctx, v);
    }
    /* records refused at eg_unit_load: a triple off the table, reduced off GRAD, two time steps on a Newton */
    const struct { const char* name; uint8_t byte; const char* reason, *what; } at_load[] = {
        { "lin_vjp", 0xF9, "EG_LINK_RULES", "a (TAPE_ROW, GRAD) link, a triple not in the table" },
        { "lin_vjp", 0xF6, "only a GRAD link is reduced", "reduced on a (TAPE_ROW, VALUE) link" },
        { "lin_newton", 0xFA, "second (TIME_STEP, VALUE)", "two TIME_STEP links on a Newton" },
    };
    for (size_t i = 0; i < sizeof(at_load) / sizeof(at_load[0]); ++i) {
        eg_unit* u = load_byte(ctx, lib, at_load[i].name, at_load[i].byte);
        snprintf(what, sizeof(what), "%s refused at load naming '%s'", at_load[i].what, at_load[i].reason);
        check(u == NULL && reason_has(ctx, at_load[i].reason), what);
        if (u) eg_unit_free(ctx, u);
    }
    /* a record too large for its budget, and a runner that never recorded */
    eg_runner* r = NULL;
    d = recorded(&plan, LSTEPS, 3, 1, roles, good);
    check(eg_runner_new(ctx, &d, &r) == EG_OK && eg_reverse(ctx, r, NULL, NULL) == EG_BAD_REQUEST &&
              reason_has(ctx, "no recorded run"),
          "eg_reverse before any eg_run refused BAD_REQUEST");
    double seed[LN];
    for (int i = 0; i < LN; ++i) seed[i] = 1.0;
    struct tensor ts;
    eg_plane* ps = host_plane(ctx, &ts, LN, seed, kDLFloat, 64);
    const int32_t sslot[] = { 0 };
    eg_plane* const sp[] = { ps };
    const struct eg_bind_desc sd = binds(1, sslot, sp, 0, NULL, NULL), gd = binds(0, NULL, NULL, 0, NULL, NULL);
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    check(eg_run(ctx, r, &rep) == EG_OK && eg_reverse(ctx, r, &sd, &gd) == EG_BAD_REQUEST && reason_has(ctx, "memory_budget"),
          "a memory_budget below one sample's checkpoints and tape refused BAD_REQUEST");
    eg_runner_free(ctx, r);
    eg_plane_free(ctx, ps);
    d = runner_desc(&plan, LSTEPS);
    check(eg_runner_new(ctx, &d, &r) == EG_OK && eg_reverse(ctx, r, NULL, NULL) == EG_BAD_REQUEST &&
              reason_has(ctx, "not recorded"),
          "eg_reverse of a runner without EG_RUN_RECORD refused BAD_REQUEST");
    eg_runner_free(ctx, r);
    eg_plan_free(ctx, plan);
    lin_free(ctx, &s);
    eg_unit* lina = load_byte(ctx, lib, "lina", 0xE5);
    eg_plan* aplan = lina ? plan_of(ctx, lina, EG_STRUCTURE_HOST_TEAM, LN) : NULL;
    struct lin sa;
    if (aplan) lin_bind(ctx, aplan, &sa);
    d = recorded(&aplan, LSTEPS, 3, 0, roles, good);
    check(aplan && eg_runner_new(ctx, &d, &none) == EG_BAD_REQUEST && reason_has(ctx, "reads an active set"),
          "EG_RUN_RECORD of an active-set step whose twin reads no active set refused BAD_REQUEST by name");
    d = runner_desc(&aplan, LSTEPS);
    d.events = EG_EVENTS_REFINE;
    check(aplan && eg_runner_new(ctx, &d, &none) == EG_NOT_WRITTEN && reason_has(ctx, "events REFINE with an active set"),
          "events REFINE over an active-set step refused NOT_WRITTEN by name");
    if (aplan) {
        eg_plan_free(ctx, aplan);
        lin_free(ctx, &sa);
    }
    eg_unit_free(ctx, lina);
    eg_unit_free(ctx, lin);
    for (int w = 0; w < 3; ++w) eg_unit_free(ctx, twins[w]);
    eg_unit_free(ctx, vjp);
}

/* ---- the old-path pair: rr_pend ---- */

#define PN 64
#define PSTEPS 400

#define PMAX 4096
static void pair_rows(eg_ctx* ctx, int argc, char** argv) {
    const char *lib = argv[2], *dir = argv[3];
    /* with an active set (argc 9): the cadence in steps and the primal's name (_pend_body: the mask is slot 7, c and dt 8 and 9) */
    const int named = argc >= 9;
    const long every = named ? atol(argv[7]) : 0;
    const int act = every > 0;
    const int64_t pn = argc >= 10 ? atol(argv[9]) : 64;
    const int reorder = argc >= 10;
    const char* base = named ? argv[8] : "rr_pend";
    char tape_name[64], vjp_name[64];
    snprintf(tape_name, sizeof(tape_name), "%s_tape", base);
    snprintf(vjp_name, sizeof(vjp_name), "%s_step_vjp", base);
    const int32_t mask_slot = act ? 7 : 5, c_slot = act ? 8 : 6, dt_slot = act ? 9 : 7;
    int64_t smax = 0;
    uint8_t dg[3][EG_DIGEST_BYTES];
    static double x0[PMAX], v0[PMAX], tf[PMAX], ex[PMAX], ev[PMAX], et[PMAX], gx[PMAX], gv[PMAX], gt[PMAX];
    double gc = 0.0, gdt = 0.0;
    int64_t steps[PMAX];
    if (!parse_digest(argv[4], dg[0]) || !parse_digest(argv[5], dg[1]) || !parse_digest(argv[6], dg[2]) ||
        !read_file(dir, "x0.bin", x0, (size_t)pn * sizeof(*x0)) || !read_file(dir, "v0.bin", v0, (size_t)pn * sizeof(*v0)) ||
        !read_file(dir, "tf.bin", tf, (size_t)pn * sizeof(*tf)) || !read_file(dir, "end_x.bin", ex, (size_t)pn * sizeof(*ex)) ||
        !read_file(dir, "end_v.bin", ev, (size_t)pn * sizeof(*ev)) || !read_file(dir, "end_t.bin", et, (size_t)pn * sizeof(*et)) ||
        !read_file(dir, "grad_x.bin", gx, (size_t)pn * sizeof(*gx)) || !read_file(dir, "grad_v.bin", gv, (size_t)pn * sizeof(*gv)) ||
        !read_file(dir, "grad_t.bin", gt, (size_t)pn * sizeof(*gt)) || !read_file(dir, "grad_c.bin", &gc, sizeof(gc)) ||
        !read_file(dir, "grad_dt.bin", &gdt, sizeof(gdt)) || !read_file(dir, "steps.bin", steps, (size_t)pn * sizeof(*steps))) {
        check(0, "pair: the digests and the old path's oracle files read");
        return;
    }
    for (int i = 0; i < pn; ++i) smax = steps[i] > smax ? steps[i] : smax;
    eg_unit* p = load(ctx, lib, base, dg[0]);
    eg_unit* units[2] = { load(ctx, lib, tape_name, dg[1]), load(ctx, lib, vjp_name, dg[2]) };
    check(p && units[0] && units[1], "pair: rr_pend, its taping twin and its step vjp load from one library");
    if (!(p && units[0] && units[1])) return;
    const eg_derived_role roles[2] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP };
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int threads[2] = { 1, 3 };
    const double c = 0.3, dt = 0.01;
    char what[256];
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            eg_host_threads(ctx, threads[ti]);
            static double x[PMAX], v[PMAX], t[PMAX];
            uint8_t done[PMAX];
            double tfin[PMAX];
            memcpy(x, x0, (size_t)pn * sizeof(*x));
            memcpy(v, v0, (size_t)pn * sizeof(*v));
            memcpy(tfin, tf, (size_t)pn * sizeof(*tfin));
            memset(t, 0, (size_t)pn * sizeof(*t));
            memset(done, 0, (size_t)pn * sizeof(*done));
            struct tensor tx, tv, tt, ttf, tdn;
            eg_plane* planes[5] = { host_plane(ctx, &tt, pn, t, kDLFloat, 64), host_plane(ctx, &tv, pn, v, kDLFloat, 64),
                                    host_plane(ctx, &tx, pn, x, kDLFloat, 64), host_plane(ctx, &ttf, pn, tfin, kDLFloat, 64),
                                    host_plane(ctx, &tdn, pn, done, kDLBool, 8) };
            const int32_t slots[5] = { 0, 1, 2, 3, mask_slot }, us[2] = { c_slot, dt_slot };
            const void* const uv[2] = { &c, &dt };
            const struct eg_bind_desc bd = binds(5, slots, planes, 2, us, uv);
            eg_plan* plan = plan_of(ctx, p, structures[si], pn);
            eg_runner* r = NULL;
            struct eg_runner_desc d = recorded(&plan, PSTEPS, named ? 37 : si ? PSTEPS : -1, 0, roles, units);
            d.every = every;
            if (reorder) {
                d.flags |= EG_RUN_REORDER;
                d.reorder_theta = 0.5;
            }
            struct eg_run_report rep;
            memset(&rep, 0, sizeof(rep));
            rep.struct_size = sizeof(rep);
            if (argc == 11) {   // EG_RUN_REORDER with a start copy: refused by name
                const int counter = strcmp(argv[10], "counter") == 0;
                snprintf(what, sizeof(what), counter ? "pair %s %s %d threads: a link on the twin's finish counter refused BAD_REQUEST naming eagle's own"
                                                     : "pair %s %s %d threads: EG_RUN_REORDER with a start copy refused NOT_WRITTEN by name",
                         base, si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti]);
                check(plan && eg_bind(ctx, plan, &bd) == EG_OK &&
                          (counter ? eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "links[7].slot 8 is eagle's own")
                                   : eg_runner_new(ctx, &d, &r) == EG_NOT_WRITTEN &&
                                         reason_has(ctx, "EG_RUN_REORDER with a start copy (time0_slot)") && reason_has(ctx, "no old-path oracle")),
                      what);
                for (int i = 0; i < 5; ++i) eg_plane_free(ctx, planes[i]);
                eg_plan_free(ctx, plan);
                continue;
            }
            const int ok = plan && eg_bind(ctx, plan, &bd) == EG_OK && eg_runner_new(ctx, &d, &r) == EG_OK &&
                           eg_run(ctx, r, &rep) == EG_OK;
            int s_ok = ok && rep.steps == smax && rep.finished == pn && (!reorder || rep.reorders > 0);
            for (int i = 0; s_ok && i < pn; ++i) s_ok = x[i] == ex[i] && v[i] == ev[i] && t[i] == et[i];
            snprintf(what, sizeof(what), "pair %s%s %s %d threads (window %s): the recorded run's states equal the old "
                     "path's bitwise (%lld steps)", base, reorder ? " (reordering active set)" : act ? " (active set)" : "", si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti], named ? "37" : si ? "400" : "-1, full",
                     (long long)rep.steps);
            check(s_ok, what);
            static double sx[PMAX], sv[PMAX], ox[PMAX], ov[PMAX], ot[PMAX];
            double oc = 0.0, odt = 0.0;
            for (int i = 0; i < pn; ++i) {
                sx[i] = 1.0;
                sv[i] = 0.5;
            }
            struct tensor a1, a2, b1, b2, b3, b4, b5;
            eg_plane* const sp[2] = { host_plane(ctx, &a1, pn, sx, kDLFloat, 64), host_plane(ctx, &a2, pn, sv, kDLFloat, 64) };
            eg_plane* const gp[5] = { host_plane(ctx, &b1, pn, ox, kDLFloat, 64), host_plane(ctx, &b2, pn, ov, kDLFloat, 64),
                                      host_plane(ctx, &b3, pn, ot, kDLFloat, 64), host_plane(ctx, &b4, 1, &oc, kDLFloat, 64),
                                      host_plane(ctx, &b5, 1, &odt, kDLFloat, 64) };
            const int32_t sslot[2] = { 2, 1 }, gslot[5] = { 2, 1, 0, c_slot, dt_slot };
            const struct eg_bind_desc sd = binds(2, sslot, sp, 0, NULL, NULL), gd = binds(5, gslot, gp, 0, NULL, NULL);
            const int rok = ok && eg_reverse(ctx, r, &sd, &gd) == EG_OK;
            int g_ok = rok && memcmp(&oc, &gc, sizeof(gc)) == 0 && memcmp(&odt, &gdt, sizeof(gdt)) == 0;
            for (int i = 0; g_ok && i < pn; ++i)
                g_ok = memcmp(&ox[i], &gx[i], 8) == 0 && memcmp(&ov[i], &gv[i], 8) == 0 && memcmp(&ot[i], &gt[i], 8) == 0;
            snprintf(what, sizeof(what), "pair %s%s %s %d threads: eg_reverse's gradients (x, v, t per sample; c, dt summed) "
                     "equal the old path's bitwise (c %.17g, dt %.17g)", base, reorder ? " (reordering active set)" : act ? " (active set)" : "", si ? "HOST_TEAM" : "HOST_SERIAL", threads[ti], oc, odt);
            check(g_ok, what);
            for (int i = 0; i < 2; ++i) eg_plane_free(ctx, sp[i]);
            for (int i = 0; i < 5; ++i) eg_plane_free(ctx, gp[i]);
            for (int i = 0; i < 5; ++i) eg_plane_free(ctx, planes[i]);
            eg_runner_free(ctx, r);
            eg_plan_free(ctx, plan);
        }
    eg_host_threads(ctx, 0);
    eg_unit_free(ctx, p);
    eg_unit_free(ctx, units[0]);
    eg_unit_free(ctx, units[1]);
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 7 && argc != 9 && argc != 10 && argc != 11) {
        fprintf(stderr, "usage: %s <fixture_reverse lib> [<pair lib> <oracle dir> <primal digest> <twin digest> <vjp digest> [<every> <primal name> [<n>: reorder]]]\n",
                argv[0]);
        return 2;
    }
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new host context");
    if (argc == 2) lin_rows(ctx, argv[1]);
    else pair_rows(ctx, argc, argv);
    eg_ctx_free(ctx);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures == 0 ? 0 : 1;
}
