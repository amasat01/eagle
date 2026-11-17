// SPDX-License-Identifier: Apache-2.0
/*
 * The gradient of a finishing rollout with respect to a parameter, through eg_reverse.
 *
 *   param_reverse_test <library> <6 digests: rr_pend_c, its tape, its step vjp, then the three of rr_pend_c_params>
 *                      [<device library>]   (built with EG_PARAM_DEVICE)
 *
 * The damped pendulum rr_pend_c reads its damping c by value (a UNIFORM): the record differentiates the unpromoted step and the
 * door reduces the per-sample gradient of c itself. The samples stop at different steps, so the live prefix shrinks through counts
 * that are not multiples of 32. Rows: the gradient of the sum of the end positions equals the central finite difference of two
 * forward runs at c +- h, in one pass and in several (a small memory budget), on HOST_SERIAL, HOST_TEAM and, with a device
 * library, DEVICE (which must equal the host's); and the promoted units (c in a one-cell plane, the vjp owning cell rows) are
 * refused at eg_runner_new, on the host and the device, because the reverse launches them over the live prefix.
 */
#include "pair_util.h"

#include <math.h>
#include <stdlib.h>

#ifdef EG_PARAM_DEVICE
#include <cuda_runtime_api.h>
#endif

#define N 100
#define STEPS 400
#define BUDGET_SEVERAL_PASSES 70000   // about 40 samples' checkpoints and tape: three passes over N samples
#define C0 0.3
#define DT 0.01
#define H 1e-6
#define FD_REL_TOL 1e-8

/* A plane over host memory @p host, mirrored in device memory when @p dev. */
struct plane {
    struct tensor t;
    eg_plane* p;
    void* d;
    void* host;
    size_t bytes;
};

static int plane_make(eg_ctx* ctx, struct plane* s, int dev, void* host, int64_t n, uint8_t code, uint8_t bits) {
    s->host = host;
    s->bytes = (size_t)n * bits / 8;
    s->d = NULL;
    void* data = host;
#ifdef EG_PARAM_DEVICE
    if (dev) {
        if (cudaMalloc(&s->d, s->bytes) != cudaSuccess || cudaMemcpy(s->d, host, s->bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
        data = s->d;
    }
#else
    (void)dev;
#endif
    tensor_fill(&s->t, n, data, code, bits);
    if (!dev) s->t.managed.dl_tensor.device.device_type = kDLCPU;
    s->p = NULL;
    return eg_plane_wrap(ctx, &s->t.managed, &s->p) == EG_OK;
}

/* Copy a device plane back into its host memory (a host plane is the host memory). */
static int plane_pull(const struct plane* s) {
#ifdef EG_PARAM_DEVICE
    if (s->d) return cudaMemcpy(s->host, s->d, s->bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
#endif
    (void)s;
    return 1;
}

static void plane_drop(eg_ctx* ctx, struct plane* s) {
    if (s->p) eg_plane_free(ctx, s->p);
#ifdef EG_PARAM_DEVICE
    if (s->d) cudaFree(s->d);
#endif
}

static eg_unit* load(eg_ctx* ctx, const char* lib, const char* name, const char* hex) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    if (!parse_digest(hex, ref.digest)) return NULL;
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = lib;
    d.unit = &ref;
    eg_unit* u = NULL;
    if (eg_unit_load(ctx, &d, &u) == EG_OK) return u;
    char reason[512];
    size_t len = 0;
    if (eg_error(ctx, reason, sizeof(reason), &len) == EG_OK) printf("      reason: %s\n", reason);
    return NULL;
}

static struct eg_bind_desc binds(int32_t n_planes, const int32_t* slots, eg_plane* const* planes, int32_t n_uniforms, const int32_t* uslots,
                                 const void* const* values) {
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

struct inputs {
    double x0[N], v0[N], tf[N];
};

struct result {
    double x[N], v[N], t[N], gx[N], gv[N], gt[N], gc, gdt;
    int ok;
};

enum mode { FORWARD, REVERSED, PROMOTED_REFUSED };

/* One run of a primal unit over the inputs at damping @p c: forward only, recorded and reversed (the seeds are 1 for x and 0 for v: the
 * gradient of the sum of the end positions), or, for the promoted units, eg_runner_new alone (refused). The promoted c is a one-cell plane
 * (slots t 0, v 1, x 2, t_final 3, c 4, mask 6, dt 7); the plain c a uniform (c 6, mask 5, dt 7). */
static void run(eg_ctx* ctx, int dev, eg_unit* primal, eg_unit* const* derived, enum mode mode, eg_structure structure, uint64_t budget,
                double c, const struct inputs* in, struct result* out) {
    uint8_t done[N] = { 0 };
    double tf[N], cell = c, sx[N], sv[N];
    const int plane_c = mode == PROMOTED_REFUSED, mask = plane_c ? 6 : 5, c_slot = plane_c ? 4 : 6;
    memset(out, 0, sizeof(*out));
    memcpy(out->x, in->x0, sizeof(in->x0));
    memcpy(out->v, in->v0, sizeof(in->v0));
    memcpy(tf, in->tf, sizeof(tf));
    for (int i = 0; i < N; ++i) {
        sx[i] = 1.0;
        sv[i] = 0.0;
    }
    struct plane pt, pv, px, ptf, pdone, pcell;
    int ok = plane_make(ctx, &pt, dev, out->t, N, kDLFloat, 64) && plane_make(ctx, &pv, dev, out->v, N, kDLFloat, 64) &&
             plane_make(ctx, &px, dev, out->x, N, kDLFloat, 64) && plane_make(ctx, &ptf, dev, tf, N, kDLFloat, 64) &&
             plane_make(ctx, &pdone, dev, done, N, kDLBool, 8) && (!plane_c || plane_make(ctx, &pcell, dev, &cell, 1, kDLFloat, 64));
    eg_plane* planes[6] = { pt.p, pv.p, px.p, ptf.p, pdone.p, plane_c ? pcell.p : NULL };
    const int32_t slots[6] = { 0, 1, 2, 3, mask, 4 }, uslots[2] = { 6, 7 }, dt_only[1] = { 7 };
    const double dt = DT;
    const void* const uvalues[2] = { &c, &dt }, *const dt_value[1] = { &dt };
    const struct eg_bind_desc bd = plane_c ? binds(6, slots, planes, 1, dt_only, dt_value) : binds(5, slots, planes, 2, uslots, uvalues);
    eg_plan* plan = NULL;
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = primal;
    pd.structure = structure;
    pd.policy = EG_REDUCE_COMPENSATED;
    pd.n_samples = N;
    ok = ok && eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK;
    eg_runner* r = NULL;
    struct eg_runner_desc rd = runner_desc(&plan, STEPS);
    const eg_derived_role roles[2] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP };
    if (mode != FORWARD) {
        rd.flags = EG_RUN_RECORD;
        rd.n_derived = 2;
        rd.derived_roles = roles;
        rd.derived_units = derived;
        rd.window = 1;   // the twin records one step per launch (built in)
        rd.memory_budget = budget;
    }
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    if (mode == PROMOTED_REFUSED) {
        out->ok = ok && eg_runner_new(ctx, &rd, &r) == EG_BAD_REQUEST;
        out->gc = (double)reason_has(ctx, "owns cell rows and is launched over the live prefix by the reverse: it must be emitted in the active-set form");
    } else {
        ok = ok && eg_runner_new(ctx, &rd, &r) == EG_OK && eg_run(ctx, r, &rep) == EG_OK && rep.finished == N;
        ok = ok && plane_pull(&pt) && plane_pull(&pv) && plane_pull(&px);
        if (ok && mode == REVERSED) {
            struct plane s1, s2, g1, g2, g3, g4, g5;
            ok = plane_make(ctx, &s1, dev, sx, N, kDLFloat, 64) && plane_make(ctx, &s2, dev, sv, N, kDLFloat, 64) &&
                 plane_make(ctx, &g1, dev, out->gx, N, kDLFloat, 64) && plane_make(ctx, &g2, dev, out->gv, N, kDLFloat, 64) &&
                 plane_make(ctx, &g3, dev, out->gt, N, kDLFloat, 64) && plane_make(ctx, &g4, dev, &out->gc, 1, kDLFloat, 64) &&
                 plane_make(ctx, &g5, dev, &out->gdt, 1, kDLFloat, 64);
            eg_plane* sp[2] = { s1.p, s2.p };
            eg_plane* gp[5] = { g1.p, g2.p, g3.p, g4.p, g5.p };
            const int32_t sslot[2] = { 2, 1 }, gslot[5] = { 2, 1, 0, c_slot, 7 };
            const struct eg_bind_desc sd = binds(2, sslot, sp, 0, NULL, NULL), gd = binds(5, gslot, gp, 0, NULL, NULL);
            ok = ok && eg_reverse(ctx, r, &sd, &gd) == EG_OK && plane_pull(&g4) && plane_pull(&g5);
            plane_drop(ctx, &s1);
            plane_drop(ctx, &s2);
            plane_drop(ctx, &g1);
            plane_drop(ctx, &g2);
            plane_drop(ctx, &g3);
            plane_drop(ctx, &g4);
            plane_drop(ctx, &g5);
        }
        out->ok = ok;
    }
    if (!out->ok) {
        char reason[512];
        size_t len = 0;
        if (eg_error(ctx, reason, sizeof(reason), &len) == EG_OK) printf("      reason: %s\n", reason);
    }
    plane_drop(ctx, &pt);
    plane_drop(ctx, &pv);
    plane_drop(ctx, &px);
    plane_drop(ctx, &ptf);
    plane_drop(ctx, &pdone);
    if (plane_c) plane_drop(ctx, &pcell);
    if (r) eg_runner_free(ctx, r);
    if (plan) eg_plan_free(ctx, plan);
}

static double objective(const struct result* r) {
    double j = 0.0;
    for (int i = 0; i < N; ++i) j += r->x[i];
    return j;
}

/* The steps whose live count is not a multiple of 32, from the stopping times in @p end_t. */
static int ragged_steps(const struct result* r, int* total) {
    int ragged = 0;
    *total = 0;
    for (int j = 0; j < STEPS; ++j) {
        int live = 0;
        for (int i = 0; i < N; ++i) live += lround(r->t[i] / DT) > j;
        if (live == 0) break;
        ++*total;
        ragged += live % 32 != 0;
    }
    return ragged;
}

/* The finite-difference row and the reversed rows of one library on one context; the gradients of c in @p gc (one, several passes). */
static void rows(eg_ctx* ctx, int dev, const char* lib, char** hex, const char* where, const struct inputs* in, double* fd, double gc[2]) {
    eg_unit* plain = load(ctx, lib, "rr_pend_c", hex[0]);
    eg_unit* plain_d[2] = { load(ctx, lib, "rr_pend_c_tape", hex[1]), load(ctx, lib, "rr_pend_c_step_vjp", hex[2]) };
    eg_unit* promoted = load(ctx, lib, "rr_pend_c_params", hex[3]);
    eg_unit* promoted_d[2] = { load(ctx, lib, "rr_pend_c_params_tape", hex[4]), load(ctx, lib, "rr_pend_c_params_step_vjp", hex[5]) };
    char what[256];
    snprintf(what, sizeof(what), "%s: the plain and the promoted units load", where);
    check(plain && plain_d[0] && plain_d[1] && promoted && promoted_d[0] && promoted_d[1], what);
    if (!(plain && plain_d[0] && plain_d[1] && promoted && promoted_d[0] && promoted_d[1])) return;
    static struct result up, down, got;
    const eg_structure structure = dev ? EG_STRUCTURE_DEVICE : EG_STRUCTURE_HOST_SERIAL;
    run(ctx, dev, plain, NULL, FORWARD, structure, 0, C0 + H, in, &up);
    run(ctx, dev, plain, NULL, FORWARD, structure, 0, C0 - H, in, &down);
    *fd = (objective(&up) - objective(&down)) / (2 * H);
    int total = 0;
    const int ragged = ragged_steps(&up, &total);
    snprintf(what, sizeof(what), "%s: the finite-difference oracle (forward runs at c +- h) gives %.15g; %d of %d steps have a live count that is not a multiple of 32",
             where, *fd, ragged, total);
    check(up.ok && down.ok && isfinite(*fd) && fabs(*fd) > 1e-3 && ragged > 0, what);
    const uint64_t budgets[2] = { 0, BUDGET_SEVERAL_PASSES };
    for (int bi = 0; bi < 2; ++bi) {
        run(ctx, dev, plain, plain_d, REVERSED, structure, budgets[bi], C0, in, &got);
        gc[bi] = got.gc;
        snprintf(what, sizeof(what), "%s %s: the gradient of c (%.15g) equals the finite difference (relative %.2e)", where, bi ? "several passes" : "one pass",
                 got.gc, fabs(got.gc - *fd) / fabs(*fd));
        check(got.ok && fabs(got.gc - *fd) <= FD_REL_TOL * fabs(*fd), what);
    }
    run(ctx, dev, promoted, promoted_d, PROMOTED_REFUSED, structure, 0, C0, in, &got);
    snprintf(what, sizeof(what), "%s: the promoted units (the vjp owns cell rows) are refused BAD_REQUEST at eg_runner_new, naming the active-set form", where);
    check(got.ok && got.gc == 1.0, what);
    eg_unit_free(ctx, plain);
    eg_unit_free(ctx, plain_d[0]);
    eg_unit_free(ctx, plain_d[1]);
    eg_unit_free(ctx, promoted);
    eg_unit_free(ctx, promoted_d[0]);
    eg_unit_free(ctx, promoted_d[1]);
}

int main(int argc, char** argv) {
    if (argc != 8 && argc != 9) {
        fprintf(stderr, "usage: %s <library> <6 digests> [<device library>]\n", argv[0]);
        return 2;
    }
    static struct inputs in;
    for (int i = 0; i < N; ++i) {
        in.x0[i] = 0.5 + i * (1.0 / (N - 1));
        in.v0[i] = -0.2 + i * (0.5 / (N - 1));
        in.tf[i] = 0.5 + i * (2.5 / (N - 1));
    }
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* host = NULL;
    check(eg_ctx_new(&st, &host) == EG_OK, "eg_ctx_new host context");
    double fd = 0.0, host_gc[2] = { 0.0, 0.0 };
    rows(host, 0, argv[1], argv + 2, "host", &in, &fd, host_gc);
    char what[256];
    snprintf(what, sizeof(what), "host: one pass and several passes give the same gradient (%.17g, %.17g)", host_gc[0], host_gc[1]);
    check(host_gc[0] == host_gc[1], what);
#ifdef EG_PARAM_DEVICE
    if (argc == 9) {
        st.flags = EG_CTX_DEVICE;
        st.device = 0;
        eg_ctx* device = NULL;
        check(eg_ctx_new(&st, &device) == EG_OK, "eg_ctx_new device context (device 0)");
        double dfd = 0.0, dev_gc[2] = { 0.0, 0.0 };
        rows(device, 1, argv[8], argv + 2, "device", &in, &dfd, dev_gc);
        for (int bi = 0; bi < 2; ++bi) {
            snprintf(what, sizeof(what), "device %s: the gradient of c (%.17g) equals the host's (%.17g) to 1e-12 relative", bi ? "several passes" : "one pass",
                     dev_gc[bi], host_gc[bi]);
            check(fabs(dev_gc[bi] - host_gc[bi]) <= 1e-12 * fabs(host_gc[bi]), what);
        }
        eg_ctx_free(device);
    }
#endif
    eg_ctx_free(host);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures != 0;
}
