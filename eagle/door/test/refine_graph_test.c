// SPDX-License-Identifier: Apache-2.0
/*
 * Two recorded events-refine runs of the oscillator ev_osc in one graph, a reset launch between them, and eg_graph_reverse through all three
 * (the run leaf of a DIFF recording, host and device). Phase A runs the oscillator to its stop from x0 and v0; the reset launch hands the
 * stop velocity on scaled by the restitution w (phase B's v is y = w v of phase A); phase B runs from its own x to its stop. J = sum of
 * phase B's end velocity. The reference is central differences of the whole graph: the graph is launched again with x0 (or v0) moved by
 * +-h, the planes of both phases rewritten as at the first launch, and dJ/dx0, dJ/dv0 are compared with the reverse. The differences have a
 * truncation error of h^2/6 times a third derivative (h = 1e-4: about 1e-9) and the Newton stop's tolerance over h; 1e-7 holds both with margin, and
 * the differences at h and 2h agreeing to it shows no sample's stop moved to another step between the runs.
 *
 *   refine_graph_test <pair lib> <graph fixture lib> <primal> <twin> <newton> <guard> <step vjp> <refine vjp> [device]
 * "device": a DEVICE context over device planes (the pair lib is then the device library of ev_osc).
 */
#define _POSIX_C_SOURCE 200809L
#include "pair_util.h"

#include <math.h>

#define PN 8
#define MAX_PLANES 16

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

/* The planes of a row: a host array each and, on a DEVICE context, a device plane beside it that the row copies up before a launch and down after it. */
struct side {
    eg_ctx* ctx;
    int device;
    int n;
    struct tensor tensors[MAX_PLANES];
    eg_plane* handle[MAX_PLANES];
    eg_plane* host[MAX_PLANES];
};

static eg_plane* make_n(struct side* s, int64_t n, void* data, uint8_t code, uint8_t bits) {
    struct tensor* t = &s->tensors[s->n];
    tensor_fill(t, n, data, code, bits);
    t->managed.dl_tensor.device.device_type = kDLCPU;
    eg_plane* host = NULL;
    if (eg_plane_wrap(s->ctx, &t->managed, &host) != EG_OK) return NULL;
    eg_plane* handle = host;
    if (s->device) {
        const int64_t shape[1] = { n };
        struct eg_plane_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.kind = EG_MEM_DEVICE;
        d.ndim = 1;
        d.dtype_code = code;
        d.dtype_bits = bits;
        d.shape = shape;
        handle = NULL;
        if (eg_plane_new(s->ctx, &d, &handle) != EG_OK) return NULL;
    }
    s->handle[s->n] = handle;
    s->host[s->n++] = s->device ? host : NULL;
    return handle;
}

static eg_plane* make(struct side* s, void* data, uint8_t code, uint8_t bits) { return make_n(s, PN, data, code, bits); }

static void up(struct side* s) {
    for (int i = 0; s->device && i < s->n; ++i) eg_copy(s->ctx, s->host[i], s->handle[i]);
}

static void down(struct side* s) {
    for (int i = 0; s->device && i < s->n; ++i) eg_copy(s->ctx, s->handle[i], s->host[i]);
    if (s->device) eg_sync(s->ctx);
}

static void release(struct side* s) {
    for (int i = 0; i < s->n; ++i) {
        eg_plane_free(s->ctx, s->handle[i]);
        if (s->host[i] != NULL) eg_plane_free(s->ctx, s->host[i]);
    }
    s->n = 0;
}

/* One phase: the state planes t, v, x and the finished mask of an ev_osc batch and the recorded refine runner over them. */
struct phase {
    double t[PN], v[PN], x[PN];
    uint8_t done[PN];
    eg_plane *pt, *pv, *px, *pd;
    eg_plan* plan;
    eg_runner* runner;
};

static int phase_make(struct side* s, struct phase* f, eg_unit* osc, eg_unit* const derived[5]) {
    f->pt = make(s, f->t, kDLFloat, 64);
    f->pv = make(s, f->v, kDLFloat, 64);
    f->px = make(s, f->x, kDLFloat, 64);
    f->pd = make(s, f->done, kDLBool, 8);
    if (f->pt == NULL || f->pv == NULL || f->px == NULL || f->pd == NULL) return 0;
    const double uvals[3] = { 0.0, 0.05, 2.0 };    /* the oscillator's a, dt, w */
    const int32_t slots[4] = { 0, 1, 2, 4 }, us[3] = { 5, 6, 7 };
    eg_plane* const planes[4] = { f->pt, f->pv, f->px, f->pd };
    const void* const uv[3] = { &uvals[0], &uvals[1], &uvals[2] };
    struct eg_bind_desc bd = binds(4, slots, planes);
    bd.n_uniforms = 3;
    bd.uniform_slots = us;
    bd.uniform_values = uv;
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = osc;
    pd.structure = s->device ? EG_STRUCTURE_DEVICE : EG_STRUCTURE_HOST_SERIAL;
    pd.policy = EG_REDUCE_COMPENSATED;
    pd.n_samples = PN;
    if (eg_plan_new(s->ctx, &pd, &f->plan) != EG_OK || eg_bind(s->ctx, f->plan, &bd) != EG_OK) return 0;
    const eg_derived_role roles[5] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP, EG_DERIVED_REFINE_NEWTON, EG_DERIVED_STATE_GUARD_VJP, EG_DERIVED_REFINE_VJP };
    struct eg_runner_desc d = runner_desc(&f->plan, 400);
    d.events = EG_EVENTS_REFINE;
    d.flags = EG_RUN_RECORD;
    d.window = 1;
    d.n_derived = 5;
    d.derived_roles = roles;
    d.derived_units = derived;
    return eg_runner_new(s->ctx, &d, &f->runner) == EG_OK;
}

static void phase_free(struct side* s, struct phase* f) {
    if (f->runner) eg_runner_free(s->ctx, f->runner);
    if (f->plan) eg_plan_free(s->ctx, f->plan);
}

/* The state both phases start from at every launch of the graph: phase A at (x0, v0), phase B at its own x (its v is the reset's). */
static void start(struct phase* a, struct phase* b, const double* x0, const double* v0, double dx, double dv) {
    for (int i = 0; i < PN; ++i) {
        a->t[i] = b->t[i] = 0.0;
        a->x[i] = x0[i] + dx;
        a->v[i] = v0[i] + dv;
        b->x[i] = 0.5 + 0.1 * i;
        b->v[i] = 0.0;
        a->done[i] = b->done[i] = 0;
    }
}

/* J_i = the end velocity of phase B of sample i after the graph runs from the state @p start gives. */
static int jvalues(struct side* s, eg_graph* g, struct phase* a, struct phase* b, const double* x0, const double* v0, double dx, double dv, double* j) {
    start(a, b, x0, v0, dx, dv);
    up(s);
    const int ok = eg_graph_launch(s->ctx, g) == EG_OK;
    down(s);
    for (int i = 0; ok && i < PN; ++i) j[i] = b->v[i];
    return ok;
}

static eg_status end_diff(eg_ctx* ctx, int32_t n, const struct eg_node_vjp* v, eg_graph** g) {
    struct eg_record_end_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = EG_RECORD_DIFF;
    d.n_vjps = n;
    d.vjps = v;
    return eg_record_end(ctx, &d, g);
}

static void rows(struct side* s, const struct eg_node_vjp* reset_vjp, eg_unit* osc, eg_unit* const derived[5], eg_unit* scale) {
    eg_ctx* ctx = s->ctx;
    struct phase a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    double x0[PN], v0[PN], w[1] = { -0.8 }, ones[PN], gx[PN], gv[PN];
    for (int i = 0; i < PN; ++i) {
        x0[i] = 1.0 + 0.15 * i;
        v0[i] = 0.3 - 0.05 * i;
        ones[i] = 1.0;
        gx[i] = gv[i] = 0.0;
    }
    start(&a, &b, x0, v0, 0.0, 0.0);
    const int phases = phase_make(s, &a, osc, derived) && phase_make(s, &b, osc, derived);
    check(phases, s->device ? "on a DEVICE context: two recorded refine runners build" : "two recorded refine runners build");
    if (!phases) return;
    struct eg_runner_figures figures;
    memset(&figures, 0, sizeof(figures));
    figures.struct_size = sizeof(figures);
    check(eg_runner_figures(ctx, a.runner, &figures) == EG_OK && (figures.mode == EG_MODE_HOST) == !s->device, "the runners run where the context says (the device mode is not the host's)");
    eg_plane *pw = make_n(s, 1, w, kDLFloat, 64), *pone = make(s, ones, kDLFloat, 64), *pgx = make(s, gx, kDLFloat, 64), *pgv = make(s, gv, kDLFloat, 64);
    eg_plane* reset_planes[] = { a.pv, pw, b.pv };
    int32_t slots[3] = { 0, 1, 2 };
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = scale;
    pd.structure = s->device ? EG_STRUCTURE_DEVICE : EG_STRUCTURE_HOST_SERIAL;
    pd.policy = EG_REDUCE_COMPENSATED;
    pd.n_samples = PN;
    eg_plan* reset = NULL;
    struct eg_bind_desc rb = binds(3, slots, reset_planes);
    eg_graph* g = NULL;
    int ok = eg_plan_new(ctx, &pd, &reset) == EG_OK && eg_bind(ctx, reset, &rb) == EG_OK && eg_record_begin(ctx) == EG_OK &&
             eg_run(ctx, a.runner, NULL) == EG_OK && eg_launch(ctx, reset, NULL) == EG_OK && eg_run(ctx, b.runner, NULL) == EG_OK &&
             end_diff(ctx, 1, reset_vjp, &g) == EG_OK;
    check(ok, "the graph run(A), reset launch, run(B) is recorded with EG_RECORD_DIFF");
    if (!ok) reason_has(ctx, "\x01");    /* the door's last reason, for the row that failed */
    if (ok) {
        double j[PN], jp[PN], jm[PN], jp2[PN], jm2[PN], want_x[PN], want_v[PN], want_x2[PN], want_v2[PN];
        const double h = 1e-4;
        ok = jvalues(s, g, &a, &b, x0, v0, 0.0, 0.0, j);
        eg_plane *in[] = { a.px, a.pv }, *gr[] = { pgx, pgv };
        struct eg_graph_reverse_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.n_seeds = 1;
        eg_plane* out = b.pv;
        eg_plane* seed = pone;
        d.outputs = &out;
        d.seeds = &seed;
        d.n_grads = 2;
        d.inputs = in;
        d.grads = gr;
        ok = ok && eg_graph_reverse(ctx, g, &d) == EG_OK;
        down(s);
        /* central differences of the whole graph, at h and 2h: they agree only if no sample crossed the step of its stop between the runs */
        ok = ok && jvalues(s, g, &a, &b, x0, v0, h, 0.0, jp) && jvalues(s, g, &a, &b, x0, v0, -h, 0.0, jm) && jvalues(s, g, &a, &b, x0, v0, 2 * h, 0.0, jp2) &&
             jvalues(s, g, &a, &b, x0, v0, -2 * h, 0.0, jm2);
        for (int i = 0; ok && i < PN; ++i) {
            want_x[i] = (jp[i] - jm[i]) / (2 * h);
            want_x2[i] = (jp2[i] - jm2[i]) / (4 * h);
        }
        ok = ok && jvalues(s, g, &a, &b, x0, v0, 0.0, h, jp) && jvalues(s, g, &a, &b, x0, v0, 0.0, -h, jm) && jvalues(s, g, &a, &b, x0, v0, 0.0, 2 * h, jp2) &&
             jvalues(s, g, &a, &b, x0, v0, 0.0, -2 * h, jm2);
        for (int i = 0; ok && i < PN; ++i) {
            want_v[i] = (jp[i] - jm[i]) / (2 * h);
            want_v2[i] = (jp2[i] - jm2[i]) / (4 * h);
        }
        double spread = 0.0, err_x = 0.0, err_v = 0.0;
        for (int i = 0; ok && i < PN; ++i) {
            spread = fmax(spread, fmax(fabs(want_x[i] - want_x2[i]), fabs(want_v[i] - want_v2[i])));
            err_x = fmax(err_x, fabs(gx[i] - want_x[i]) / (1.0 + fabs(want_x[i])));
            err_v = fmax(err_v, fabs(gv[i] - want_v[i]) / (1.0 + fabs(want_v[i])));
        }
        printf("      central differences at h and 2h differ by at most %.3g; reverse vs differences: dx %.3g, dv %.3g\n", spread, err_x, err_v);
        check(ok && spread < 1e-7, "no sample crosses the step of its stop between the perturbed runs (the differences at h and 2h agree to 1e-7)");
        check(ok && err_x < 1e-7 && err_v < 1e-7,
              "eg_graph_reverse through run, reset launch, run: dJ/dx0 and dJ/dv0 equal the central differences of the whole graph to 1e-7");
    }
    if (g) eg_graph_free(ctx, g);
    if (reset) eg_plan_free(ctx, reset);
    phase_free(s, &a);
    phase_free(s, &b);
    release(s);
}

static eg_unit* digest_load(eg_ctx* ctx, const char* lib, const char* name, const char* hex) {
    uint8_t digest[EG_DIGEST_BYTES];
    return parse_digest(hex, digest) ? load(ctx, lib, name, digest) : NULL;
}

int main(int argc, char** argv) {
    if (argc != 9 && argc != 10) {
        fprintf(stderr, "usage: %s <pair lib> <graph fixture lib> <primal> <twin> <newton> <guard> <step vjp> <refine vjp> [device]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    const int device = argc == 10 && strcmp(argv[9], "device") == 0;
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    st.flags = device ? EG_CTX_DEVICE : 0;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, device ? "eg_ctx_new DEVICE context" : "eg_ctx_new host context");
    eg_unit* osc = digest_load(ctx, argv[1], "ev_osc", argv[3]);
    eg_unit* derived[5] = { digest_load(ctx, argv[1], "ev_osc_tape", argv[4]), digest_load(ctx, argv[1], "ev_osc_step_vjp", argv[7]),
                            digest_load(ctx, argv[1], "ev_osc_newton", argv[5]), digest_load(ctx, argv[1], "ev_osc_state_guard_vjp", argv[6]),
                            digest_load(ctx, argv[1], "ev_osc_refine_vjp", argv[8]) };
    uint8_t scale_digest[EG_DIGEST_BYTES], vjp_digest[EG_DIGEST_BYTES];
    memset(scale_digest, 0x10, sizeof(scale_digest));
    memset(vjp_digest, 0x11, sizeof(vjp_digest));
    eg_unit* scale = load(ctx, argv[2], "scale", scale_digest);
    eg_unit* scale_vjp = load(ctx, argv[2], "scale_vjp", vjp_digest);
    const int loaded = osc && derived[0] && derived[1] && derived[2] && derived[3] && derived[4] && scale && scale_vjp;
    check(loaded, "ev_osc with its twin, Newton, guard vjp, step vjp and refine vjp, and the reset unit scale with its vjp, load");
    if (loaded) {
        struct side s = { .ctx = ctx, .device = device };
        const struct eg_node_vjp reset_vjp[1] = { { 0, scale_vjp } };
        rows(&s, reset_vjp, osc, derived, scale);
    }
    eg_ctx_free(ctx);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures == 0 && loaded ? 0 : 1;
}
