// SPDX-License-Identifier: Apache-2.0
/*
 * eg_graph_reverse on the host.
 *
 *   graph_reverse_test <fixture_graph lib>
 *
 * Gradients of recorded graphs of plain launches against closed forms: a chain, an in-place overwrite (the snapshot plan), a
 * plane read by two nodes, a second launch with other inputs; then every refusal by status and reason.
 */
#include "pair_util.h"

#include <math.h>

#define N 7

/* A context, the fixture library, and (on a DEVICE context) the planes the rows made: each row builds host arrays, keeps device
 * planes beside them, and copies the arrays up before a launch and the results down after a reverse. */
struct rig {
    eg_ctx* ctx;
    const char* lib;
    int device;
    int n_planes;
    eg_plane* handle[16];    /* the plane given to the door */
    eg_plane* host[16];      /* DEVICE: the wrapper of the host array, else NULL */
};

static eg_plane* host_plane(eg_ctx* ctx, struct tensor* t, int64_t n, double* data) {
    tensor_fill(t, n, data, kDLFloat, 64);
    t->managed.dl_tensor.device.device_type = kDLCPU;
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

/* A (rows, N) host plane over @p data. */
struct tensor2 {
    struct DLManagedTensorVersioned managed;
    int64_t shape[2];
};
static eg_plane* host_plane2(eg_ctx* ctx, struct tensor2* t, int64_t rows, double* data) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = rows;
    t->shape[1] = N;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCPU;
    t->managed.dl_tensor.ndim = 2;
    t->managed.dl_tensor.dtype.code = kDLFloat;
    t->managed.dl_tensor.dtype.bits = 64;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

/* Make the plane of a row over @p data: a host plane, or on a DEVICE context a device plane of the same shape beside the host wrapper. */
static eg_plane* mk_shaped(struct rig* r, eg_plane* host, int ndim, const int64_t* shape) {
    eg_plane* handle = host;
    if (r->device) {
        struct eg_plane_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.kind = EG_MEM_DEVICE;
        d.ndim = ndim;
        d.dtype_code = kDLFloat;
        d.dtype_bits = 64;
        d.shape = shape;
        handle = NULL;
        if (eg_plane_new(r->ctx, &d, &handle) != EG_OK) return NULL;
    }
    r->handle[r->n_planes] = handle;
    r->host[r->n_planes++] = r->device ? host : NULL;
    return handle;
}
static eg_plane* mk(struct rig* r, struct tensor* t, int64_t n, double* data) {
    const int64_t shape[1] = { n };
    return mk_shaped(r, host_plane(r->ctx, t, n, data), 1, shape);
}
static eg_plane* mk2(struct rig* r, struct tensor2* t, int64_t rows, double* data) {
    const int64_t shape[2] = { rows, N };
    return mk_shaped(r, host_plane2(r->ctx, t, rows, data), 2, shape);
}
/* Copy the row's host arrays to its device planes (up) or back (down); a host context shares them. */
static void up(struct rig* r) {
    for (int i = 0; r->device && i < r->n_planes; ++i) eg_copy(r->ctx, r->host[i], r->handle[i]);
}
static void down(struct rig* r) {
    for (int i = 0; r->device && i < r->n_planes; ++i) eg_copy(r->ctx, r->handle[i], r->host[i]);
    if (r->device) eg_sync(r->ctx);
}
static void release(struct rig* r) {
    for (int i = 0; i < r->n_planes; ++i) {
        eg_plane_free(r->ctx, r->handle[i]);
        if (r->host[i] != NULL) eg_plane_free(r->ctx, r->host[i]);
    }
    r->n_planes = 0;
}
/* A check whose name says which context it ran on. */
static void verdict(struct rig* r, int ok, const char* what) {
    char name[600];
    snprintf(name, sizeof(name), "%s%s", r->device ? "on a DEVICE context: " : "", what);
    check(ok, name);
    if (!ok) reason_has(r->ctx, "\x01");    /* the door's last reason, for the row that failed */
}

static eg_unit* load(struct rig* r, const char* name, uint8_t byte) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memset(ref.digest, byte, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = r->lib;
    d.unit = &ref;
    eg_unit* u = NULL;
    return eg_unit_load(r->ctx, &d, &u) == EG_OK ? u : NULL;
}

/* A plan of @p unit with the planes @p p bound at the slots 0 .. n_planes - 1 (a slot given NULL is skipped). */
static eg_plan* plan_with(struct rig* r, eg_unit* unit, int32_t n_slots, eg_plane* const* p, const double* uniform_k) {
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = r->device ? EG_STRUCTURE_DEVICE : EG_STRUCTURE_HOST_SERIAL;
    d.policy = EG_REDUCE_COMPENSATED;
    d.n_samples = N;
    eg_plan* plan = NULL;
    if (eg_plan_new(r->ctx, &d, &plan) != EG_OK) return NULL;
    int32_t slots[4];
    eg_plane* planes[4];
    int32_t k = 0;
    for (int32_t s = 0; s < n_slots; ++s)
        if (p[s] != NULL) {
            slots[k] = s;
            planes[k++] = p[s];
        }
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = k;
    b.plane_slots = slots;
    b.planes = planes;
    const int32_t uslot = 1;
    const void* uvalue = uniform_k;
    if (uniform_k != NULL) {
        b.n_uniforms = 1;
        b.uniform_slots = &uslot;
        b.uniform_values = &uvalue;
    }
    return eg_bind(r->ctx, plan, &b) == EG_OK ? plan : NULL;
}

static int refused(eg_status got, eg_status want, eg_ctx* ctx, const char* part) { return got == want && reason_has(ctx, part); }

static int close_to(const double* got, const double* want, int n) {
    for (int i = 0; i < n; ++i)
        if (fabs(got[i] - want[i]) > 1e-12 * (1.0 + fabs(want[i]))) return 0;
    return 1;
}

/* The values every row starts from. */
static void fill(double* x, double* w, double base) {
    for (int i = 0; i < N; ++i) x[i] = base + 0.3 * i;
    w[0] = 1.7;
}

/* Free the caller's references to @p n planes: the tensors behind them live on the stack of the row. */
static void drop(eg_ctx* ctx, int n, eg_plane* const* p) {
    for (int i = 0; i < n; ++i) eg_plane_free(ctx, p[i]);
}

struct units {
    eg_unit *scale, *scale_vjp, *square, *square_vjp, *mul2, *mul2_vjp, *uscale, *uscale_vjp, *tape_vjp, *open_vjp;
    eg_unit *fan, *fan_vjp, *norm2, *norm2_vjp;
    eg_unit *tick, *scatter_vjp;
};

static eg_status end_diff(eg_ctx* ctx, int32_t n, const struct eg_node_vjp* v, eg_graph** g) {
    struct eg_record_end_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = EG_RECORD_DIFF;
    d.n_vjps = n;
    d.vjps = v;
    return eg_record_end(ctx, &d, g);
}

static eg_status reverse(eg_ctx* ctx, eg_graph* g, eg_plane* out, eg_plane* seed, int32_t n_in, eg_plane* const* in, eg_plane* const* grads) {
    struct eg_graph_reverse_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_seeds = out != NULL;
    d.outputs = &out;
    d.seeds = &seed;
    d.n_grads = n_in;
    d.inputs = in;
    d.grads = grads;
    return eg_graph_reverse(ctx, g, &d);
}

/* y = w x, z = y^2 with J = sum z: dJ/dx = 2 w^2 x, dJ/dw = sum 2 w x^2. */
static void chain_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], z[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1];
    struct tensor tx, ty, tz, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane* px = mk(r, &tx, N, x);
    eg_plane* py = mk(r, &ty, N, y);
    eg_plane* pz = mk(r, &tz, N, z);
    eg_plane* pw = mk(r, &tw, 1, w);
    eg_plane* p1 = mk(r, &t1, N, one);
    eg_plane* pgx = mk(r, &tgx, N, gx);
    eg_plane* pgw = mk(r, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, pz };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->square, 2, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->square_vjp } };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             end_diff(ctx, 2, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, pz, p1, 2, in, gr) == EG_OK;
    down(r);
    ww[0] = 0.0;
    for (int i = 0; i < N; ++i) {
        wx[i] = 2.0 * w[0] * w[0] * x[i];
        ww[0] += 2.0 * w[0] * x[i] * x[i];
    }
    verdict(r, ok && close_to(gx, wx, N) && close_to(gw, ww, 1), "eg_graph_reverse of y = w x, z = y^2: dJ/dx per sample and dJ/dw (a shared parameter plane) match the closed form");
    struct eg_graph_description desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    verdict(r, eg_graph_describe(ctx, graph, &desc) == EG_OK && desc.flags == EG_RECORD_DIFF && desc.n_nodes == 2 &&
                   desc.nodes[0].kind == EG_GNODE_LAUNCH && desc.n_levels == 2,
            "eg_graph_describe of a graph ended with EG_RECORD_DIFF: the flag is set, two LAUNCH nodes on two levels");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    release(r);
}

/* y = w x, then x <- y^2 in place: node 0 reads x, node 1 overwrites it. J = sum x_new = sum y^2. */
static void overwrite_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1];
    struct tensor tx, ty, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane* px = mk(r, &tx, N, x);
    eg_plane* py = mk(r, &ty, N, y);
    eg_plane* pw = mk(r, &tw, 1, w);
    eg_plane* p1 = mk(r, &t1, N, one);
    eg_plane* pgx = mk(r, &tgx, N, gx);
    eg_plane* pgw = mk(r, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, px };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->square, 2, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->square_vjp } };
    eg_graph* graph = NULL;
    double x0[N];
    memcpy(x0, x, sizeof(x0));
    up(r);
    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             end_diff(ctx, 2, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, px, p1, 2, in, gr) == EG_OK;
    down(r);
    ww[0] = 0.0;
    for (int i = 0; i < N; ++i) {
        wx[i] = 2.0 * w[0] * w[0] * x0[i];
        ww[0] += 2.0 * w[0] * x0[i] * x0[i];
    }
    verdict(r, ok && close_to(gx, wx, N) && close_to(gw, ww, 1), "eg_graph_reverse with the second launch overwriting the plane the first read: the snapshot gives the gradient at the original x");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    release(r);
}

/* y = w x, z = y x: x is read by both nodes. dJ/dx = 2 w x, dJ/dw = sum x^2. A second launch with other x gives its own gradient. */
static void shared_read_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], z[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1];
    struct tensor tx, ty, tz, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane* px = mk(r, &tx, N, x);
    eg_plane* py = mk(r, &ty, N, y);
    eg_plane* pz = mk(r, &tz, N, z);
    eg_plane* pw = mk(r, &tw, 1, w);
    eg_plane* p1 = mk(r, &t1, N, one);
    eg_plane* pgx = mk(r, &tgx, N, gx);
    eg_plane* pgw = mk(r, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, px, pz };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->mul2, 3, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->mul2_vjp } };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             end_diff(ctx, 2, vjps, &graph) == EG_OK;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    for (int round = 0; round < 2; ++round) {
        fill(x, w, round == 0 ? 0.5 : 1.25);
        up(r);
        ok = ok && eg_graph_launch(ctx, graph) == EG_OK && reverse(ctx, graph, pz, p1, 2, in, gr) == EG_OK;
        down(r);
        ww[0] = 0.0;
        for (int i = 0; i < N; ++i) {
            wx[i] = 2.0 * w[0] * x[i];
            ww[0] += x[i] * x[i];
        }
        verdict(r, ok && close_to(gx, wx, N) && close_to(gw, ww, 1),
              round == 0 ? "eg_graph_reverse of a plane read by two nodes: the two adjoints are summed"
                         : "a second eg_graph_launch with other inputs, then eg_graph_reverse: the gradient of the second launch");
    }
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    release(r);
}

/* v = fan(x) = (x, 2x, 3x), o = |v|^2, J = sum o: dJ/dx = 28 x. With @p overwrite, a third launch rewrites v from z = 5x after
 * the norm read it, so the norm's vjp needs the snapshot of v. */
static void vector_row(struct rig* r, const struct units* u, int overwrite) {
    eg_ctx* ctx = r->ctx;
    double x[N], z[N], o[N], v[3 * N], one[N], gx[N], want[N];
    struct tensor tx, tz, to, t1, tgx;
    struct tensor2 tv;
    fill(x, o, 0.5);
    for (int i = 0; i < N; ++i) {
        one[i] = 1.0;
        z[i] = 5.0 * x[i];
    }
    eg_plane* px = mk(r, &tx, N, x);
    eg_plane* pz = mk(r, &tz, N, z);
    eg_plane* po = mk(r, &to, N, o);
    eg_plane* p1 = mk(r, &t1, N, one);
    eg_plane* pgx = mk(r, &tgx, N, gx);
    eg_plane* pv = mk2(r, &tv, 3, v);
    eg_plane* a[] = { px, pv };
    eg_plane* b[] = { pv, po };
    eg_plane* c[] = { pz, pv };
    eg_plan* f = plan_with(r, u->fan, 2, a, NULL);
    eg_plan* g = plan_with(r, u->norm2, 2, b, NULL);
    eg_plan* h = overwrite ? plan_with(r, u->fan, 2, c, NULL) : NULL;
    const struct eg_node_vjp vjps[] = { { 0, u->fan_vjp }, { 1, u->norm2_vjp } };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && g && (h || !overwrite) && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             (!overwrite || eg_launch(ctx, h, NULL) == EG_OK) && end_diff(ctx, 2, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane* in[] = { px };
    eg_plane* gr[] = { pgx };
    ok = ok && reverse(ctx, graph, po, p1, 1, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) want[i] = 28.0 * x[i];
    verdict(r, ok && close_to(gx, want, N),
          overwrite ? "eg_graph_reverse through a width-3 vector plane that a later launch overwrites: the snapshot keeps the plane's shape and the gradient is right"
                    : "eg_graph_reverse through a width-3 vector plane between two launches: dJ/dx matches the closed form");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    if (h) eg_plan_free(ctx, h);
    release(r);
}

static void refusal_rows(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], z[N], w[1], one[N], gx[N], k = 2.0;
    struct tensor tx, ty, tz, tw, t1, tgx;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane* px = host_plane(ctx, &tx, N, x);
    eg_plane* py = host_plane(ctx, &ty, N, y);
    eg_plane* pz = host_plane(ctx, &tz, N, z);
    eg_plane* pw = host_plane(ctx, &tw, 1, w);
    eg_plane* p1 = host_plane(ctx, &t1, N, one);
    eg_plane* pgx = host_plane(ctx, &tgx, N, gx);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, pz };
    eg_plane* in[] = { px };
    eg_plane* gr[] = { pgx };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->square, 2, b, NULL);
    const struct eg_node_vjp both[] = { { 0, u->scale_vjp }, { 1, u->square_vjp } };
    eg_graph *plain = NULL, *graph = NULL;

    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_record_end(ctx, NULL, &plain) == EG_OK &&
             eg_graph_launch(ctx, plain) == EG_OK;
    check(ok && refused(reverse(ctx, plain, pz, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "not ended with EG_RECORD_DIFF"),
          "eg_graph_reverse of a graph ended with no description: refused BAD_REQUEST by name");

    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK && end_diff(ctx, 2, both, &graph) == EG_OK;
    check(ok && refused(reverse(ctx, graph, pz, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "never launched"),
          "eg_graph_reverse of a graph never launched: refused BAD_REQUEST by name");
    ok = eg_graph_launch(ctx, graph) == EG_OK;
    check(ok && eg_record_begin(ctx) == EG_OK &&
              refused(reverse(ctx, graph, pz, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "while recording"),
          "eg_graph_reverse while recording: refused BAD_REQUEST by name");
    eg_graph* open = NULL;
    eg_record_end(ctx, NULL, &open);
    eg_graph_free(ctx, open);

    eg_graph* none = NULL;
    const struct eg_node_vjp unseen[] = { { 0, u->scale_vjp }, { 1, u->square_vjp }, { 2, u->square_vjp } };
    const struct eg_node_vjp twice[] = { { 0, u->scale_vjp }, { 0, u->scale_vjp } }, wrong[] = { { 0, u->square_vjp } };
    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK;
    check(ok && refused(end_diff(ctx, 3, unseen, &none), EG_BAD_REQUEST, ctx, "out of range") && none == NULL &&
              refused(end_diff(ctx, 2, twice, &none), EG_BAD_REQUEST, ctx, "given twice") &&
              refused(end_diff(ctx, 1, wrong, &none), EG_BAD_REQUEST, ctx, "not a derived unit"),
          "eg_record_end with a launch ordinal out of range, given twice, or a vjp of another unit: refused BAD_REQUEST by name, the recording stays open");
    const struct eg_node_vjp tape[] = { { 0, u->tape_vjp } }, open_slot[] = { { 0, u->open_vjp } };
    check(refused(end_diff(ctx, 1, tape, &none), EG_BAD_REQUEST, ctx, "no tape slot") &&
              refused(end_diff(ctx, 1, open_slot, &none), EG_BAD_REQUEST, ctx, "has no link"),
          "eg_record_end with a vjp that links a tape row, or leaves a slot unlinked: refused BAD_REQUEST by name");
    const struct eg_node_vjp one_vjp[] = { { 0, u->scale_vjp } };
    eg_graph* partial = NULL;
    ok = end_diff(ctx, 1, one_vjp, &partial) == EG_OK && eg_graph_launch(ctx, partial) == EG_OK;
    check(ok && refused(reverse(ctx, partial, pz, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "has no vjp"),
          "eg_graph_reverse through a launch with no vjp: refused BAD_REQUEST by name");
    check(ok && refused(reverse(ctx, partial, px, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "not written by the graph"),
          "eg_graph_reverse seeding a plane the graph does not write: refused BAD_REQUEST by name");

    ok = eg_graph_launch(ctx, partial) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK;
    check(ok && refused(reverse(ctx, partial, py, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "launched outside the graph"),
          "eg_graph_reverse after a plan of the graph was launched outside it: refused BAD_REQUEST by name");
    ok = eg_graph_launch(ctx, partial) == EG_OK;
    struct eg_bind_desc rebind;
    memset(&rebind, 0, sizeof(rebind));
    rebind.struct_size = sizeof(rebind);
    const int32_t slot_x = 0;
    rebind.n_planes = 1;
    rebind.plane_slots = &slot_x;
    rebind.planes = &px;
    check(ok && eg_bind(ctx, f, &rebind) == EG_OK && refused(reverse(ctx, partial, py, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "rebound") &&
              refused(eg_graph_launch(ctx, partial), EG_BAD_REQUEST, ctx, "rebound"),
          "eg_graph_reverse and eg_graph_launch of a graph whose plan was rebound: refused BAD_REQUEST by name");
    eg_graph_free(ctx, partial);

    /* a node that is not a launch, and the units no vjp may cover */
    ok = eg_record_begin(ctx) == EG_OK && eg_copy(ctx, px, pz) == EG_OK;
    check(ok && refused(end_diff(ctx, 0, NULL, &none), EG_NOT_WRITTEN, ctx, "not an eg_launch"),
          "eg_record_end with EG_RECORD_DIFF over a recorded copy: refused NOT_WRITTEN by name");
    eg_graph* copy_only = NULL;
    eg_record_end(ctx, NULL, &copy_only);
    eg_graph_free(ctx, copy_only);
    eg_plane* ua[] = { px, NULL, py };
    eg_plan* uf = plan_with(r, u->uscale, 3, ua, &k);
    const struct eg_node_vjp uv[] = { { 0, u->uscale_vjp } };
    ok = uf && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, uf, NULL) == EG_OK;
    check(ok && refused(end_diff(ctx, 1, uv, &none), EG_NOT_WRITTEN, ctx, "by-value uniform"),
          "eg_record_end with a vjp that takes the gradient of a by-value uniform: refused NOT_WRITTEN by name");
    eg_record_end(ctx, NULL, &none);
    eg_graph_free(ctx, none);
    eg_plan_free(ctx, uf);
    eg_graph_free(ctx, plain);
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    eg_plane* all[] = { px, py, pz, pw, p1, pgx };
    drop(ctx, 6, all);
}

/* The refusals a DEVICE recording adds: a node that is not a launch, and a tape plan that is refused (it ends the recording). */
static void device_refusal_rows(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], z[N], w[1];
    struct tensor tx, ty, tz, tw;
    fill(x, w, 0.5);
    memset(y, 0, sizeof(y));
    memset(z, 0, sizeof(z));
    eg_plane* px = mk(r, &tx, N, x);
    eg_plane* py = mk(r, &ty, N, y);
    eg_plane* pz = mk(r, &tz, N, z);
    eg_plane* pw = mk(r, &tw, 1, w);
    up(r);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, pz };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->square, 2, b, NULL);
    const struct eg_node_vjp both[] = { { 0, u->scale_vjp }, { 1, u->square_vjp } }, wrong[] = { { 0, u->square_vjp } };
    eg_graph *none = NULL, *plain = NULL;

    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_copy(ctx, py, pz) == EG_OK &&
             eg_launch(ctx, g, NULL) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 2, both, &none), EG_NOT_WRITTEN, ctx, "not an eg_launch") && none == NULL,
            "eg_record_end with EG_RECORD_DIFF over a recording that holds a copy between two launches: refused NOT_WRITTEN by name");
    ok = ok && eg_record_end(ctx, NULL, &plain) == EG_OK;
    verdict(r, ok, "the same recording, still open after the refusal, ends as a plain graph");
    eg_graph_free(ctx, plain);

    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 1, wrong, &none), EG_BAD_REQUEST, ctx, "not a derived unit") &&
                   refused(eg_record_end(ctx, NULL, &none), EG_BAD_REQUEST, ctx, "not recording"),
            "eg_record_end with a vjp of another unit: refused BAD_REQUEST by name, and the refusal ends the device recording");
    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_record_end(ctx, NULL, &plain) == EG_OK;
    verdict(r, ok, "a new eg_record_begin after the refused device tape plan records and ends");
    eg_graph_free(ctx, plain);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    release(r);
}

/* ---- slice 2: forks, conditionals and loops (host) ---------------------------------------------------------------------- */

static eg_plane* mk_i32(struct rig* r, struct tensor* t, int64_t n, int32_t* data) {
    tensor_fill(t, n, data, kDLInt, 32);
    t->managed.dl_tensor.device.device_type = kDLCPU;
    eg_plane* host = NULL;
    if (eg_plane_wrap(r->ctx, &t->managed, &host) != EG_OK) return NULL;
    eg_plane* handle = host;
    if (r->device) {                            /* the test cell of a device recording is a device plane, copied up like the others */
        const int64_t shape[1] = { n };
        struct eg_plane_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.kind = EG_MEM_DEVICE;
        d.ndim = 1;
        d.dtype_code = kDLInt;
        d.dtype_bits = 32;
        d.shape = shape;
        handle = NULL;
        if (eg_plane_new(r->ctx, &d, &handle) != EG_OK) return NULL;
    }
    r->handle[r->n_planes] = handle;            /* released with the row, before its tensor goes out of scope */
    r->host[r->n_planes++] = r->device ? host : NULL;
    return handle;
}

static eg_status cond_open(eg_ctx* ctx, eg_cond_kind kind, uint32_t flags, eg_plane* cell, int64_t index, int64_t max_iters) {
    struct eg_cond_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.kind = kind;
    d.flags = flags;
    d.cell = cell;
    d.cell_index = index;
    d.max_iters = max_iters;
    return eg_record_cond_begin(ctx, &d);
}

/* y1 = w1 x and y2 = w2 x in two branches, z = y1 y2: dJ/dx = 2 w1 w2 x from both paths. */
static void fork_reverse_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y1[N], y2[N], z[N], w1[1] = { 1.7 }, w2[1] = { -0.6 }, one[N], gx[N], want[N];
    struct tensor tx, t1, t2, tz, tw1, tw2, tone, tgx;
    fill(x, w1, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *py1 = mk(r, &t1, N, y1), *py2 = mk(r, &t2, N, y2), *pz = mk(r, &tz, N, z);
    eg_plane *pw1 = mk(r, &tw1, 1, w1), *pw2 = mk(r, &tw2, 1, w2), *p1 = mk(r, &tone, N, one), *pgx = mk(r, &tgx, N, gx);
    eg_plane *a1[] = { px, pw1, py1 }, *a2[] = { px, pw2, py2 }, *b[] = { py1, py2, pz };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL), *m = plan_with(r, u->mul2, 3, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->scale_vjp }, { 2, u->mul2_vjp } };
    eg_graph* g = NULL;
    up(r);
    int ok = f1 && f2 && m && eg_record_begin(ctx) == EG_OK && eg_record_fork_begin(ctx) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK &&
             eg_record_fork_branch(ctx) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK && eg_record_fork_end(ctx) == EG_OK &&
             eg_launch(ctx, m, NULL) == EG_OK && end_diff(ctx, 3, vjps, &g) == EG_OK && eg_graph_launch(ctx, g) == EG_OK;
    eg_plane* in[] = { px };
    eg_plane* gr[] = { pgx };
    ok = ok && reverse(ctx, g, pz, p1, 1, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) want[i] = 2.0 * w1[0] * w2[0] * x[i];
    verdict(r, ok && close_to(gx, want, N), "a fork of two branches y1 = w1 x, y2 = w2 x then z = y1 y2: dJ/dx = 2 w1 w2 x, the sum of both paths");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    eg_plan_free(ctx, m);
    release(r);
}

/* y = w1 x always, then under a test on lane 2 of a flag array y = w2 x2 (IF), or either one (IF_ELSE): the gradient follows the arm
 * that ran, whatever the flag is when the reverse starts. */
static void if_row(struct rig* r, const struct units* u, eg_cond_kind kind, int negate, int rewrite_before_reverse) {
    eg_ctx* ctx = r->ctx;
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    if (kind == EG_COND_IF_ELSE && eg_ctx_info(ctx, &info) == EG_OK && (info.capabilities & EG_CAP_COND_IF_ELSE) == 0) {
        verdict(r, 1, "an IF_ELSE row skipped: the context has no EG_CAP_COND_IF_ELSE (the driver has no conditional arms)");
        return;
    }
    double x[N], x2[N], y[N], w1[1] = { 1.7 }, w2[1] = { -0.6 }, one[N], gx[N], gx2[N], want[N], want2[N];
    int32_t flags[4] = { 7, 7, 0, 7 };
    struct tensor tx, tx2, ty, tw1, tw2, tone, tgx, tgx2, tf;
    fill(x, w1, 0.5);
    fill(x2, w1, -1.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *px2 = mk(r, &tx2, N, x2), *py = mk(r, &ty, N, y), *pw1 = mk(r, &tw1, 1, w1), *pw2 = mk(r, &tw2, 1, w2);
    eg_plane *p1 = mk(r, &tone, N, one), *pgx = mk(r, &tgx, N, gx), *pgx2 = mk(r, &tgx2, N, gx2), *pf = mk_i32(r, &tf, 4, flags);
    eg_plane *a1[] = { px, pw1, py }, *a2[] = { px2, pw2, py };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->scale_vjp } };
    const uint32_t flag = negate ? EG_COND_NEGATE : 0;
    eg_graph* g = NULL;
    up(r);
    int ok = f1 && f2 && pf && eg_record_begin(ctx) == EG_OK;
    if (kind == EG_COND_IF) {
        ok = ok && eg_launch(ctx, f1, NULL) == EG_OK && cond_open(ctx, kind, flag, pf, 2, 0) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK;
    } else {
        ok = ok && cond_open(ctx, kind, flag, pf, 2, 0) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK && eg_record_cond_arm(ctx) == EG_OK &&
             eg_launch(ctx, f2, NULL) == EG_OK;
    }
    ok = ok && eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 2, vjps, &g) == EG_OK;
    eg_plane* in[] = { px, px2 };
    eg_plane* gr[] = { pgx, pgx2 };
    for (int lane = 1; ok && lane >= 0; --lane) {
        flags[2] = lane;
        up(r);
        ok = eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
        if (rewrite_before_reverse) {
            flags[2] = lane ? 0 : 1;
            up(r);
            eg_sync(ctx);
        }
        ok = ok && reverse(ctx, g, py, p1, 2, in, gr) == EG_OK;
        down(r);
        const int truth = (lane != 0) != (negate != 0);
        const int second = kind == EG_COND_IF ? truth : !truth;     /* the arm of f2 ran */
        const int first = kind == EG_COND_IF ? 1 : !second;          /* f1 ran: always for IF, else the other arm */
        const int overwritten = kind == EG_COND_IF && second;        /* f2 replaced what f1 wrote */
        for (int i = 0; i < N; ++i) {
            want[i] = first && !overwritten ? w1[0] : 0.0;
            want2[i] = second ? w2[0] : 0.0;
        }
        ok = ok && close_to(gx, want, N) && close_to(gx2, want2, N);
    }
    char what[200];
    snprintf(what, sizeof(what), "%s%s, flag lane 2 %s: the gradient follows the arm that ran, lane 1 and 0",
             kind == EG_COND_IF ? "an IF" : "an IF_ELSE", negate ? " negated" : "", rewrite_before_reverse ? "rewritten after the replay" : "set for the replay");
    verdict(r, ok, what);
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    release(r);
}

/* x <- a x per pass (t = a x, x = 1 t), a counter plane ticking down: the loop stops at zero or at its cap. With @p guarded the two
 * launches sit in an IF on lane 2 of a flag array, inside the loop. */
static void while_row(struct rig* r, const struct units* u, int32_t start, int guarded, int lane) {
    eg_ctx* ctx = r->ctx;
    double x[N], x0[N], t[N], a[1] = { 0.9 }, one1[1] = { 1.0 }, seed[N], gx[N], ga[1], want[N], want_a = 0.0;
    int32_t c[N], flags[4] = { 7, 7, 7, 7 };
    struct tensor tx, tt, ta, to, ts, tgx, tga, tc, tf;
    fill(x, a, 0.5);
    for (int i = 0; i < N; ++i) {
        x0[i] = x[i];
        seed[i] = 1.0;
    }
    a[0] = 0.9;
    eg_plane *px = mk(r, &tx, N, x), *pt = mk(r, &tt, N, t), *pa = mk(r, &ta, 1, a), *po = mk(r, &to, 1, one1), *ps = mk(r, &ts, N, seed);
    eg_plane *pgx = mk(r, &tgx, N, gx), *pga = mk(r, &tga, 1, ga), *pc = mk_i32(r, &tc, N, c), *pf = mk_i32(r, &tf, 4, flags);
    eg_plane *a1[] = { px, pa, pt }, *a2[] = { pt, po, px }, *a3[] = { pc };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL), *tk = plan_with(r, u->tick, 1, a3, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->scale_vjp } };
    eg_graph* g = NULL;
    up(r);
    int ok = f1 && f2 && tk && pf && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, 4) == EG_OK;
    if (guarded) ok = ok && cond_open(ctx, EG_COND_IF, 0, pf, 2, 0) == EG_OK;
    ok = ok && eg_launch(ctx, f1, NULL) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK;
    if (guarded) ok = ok && eg_record_cond_end(ctx) == EG_OK;
    ok = ok && eg_launch(ctx, tk, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 2, vjps, &g) == EG_OK;
    for (int i = 0; i < N; ++i) c[i] = start;
    flags[2] = lane;
    up(r);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK;
    const int passes = start > 4 ? 4 : start;              /* the cap is 4 */
    const int ran = guarded && lane == 0 ? 0 : passes;     /* passes that multiplied */
    eg_plane* in[] = { px, pa };
    eg_plane* gr[] = { pgx, pga };
    ok = ok && reverse(ctx, g, px, ps, 2, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) {
        want[i] = pow(a[0], ran);
        want_a += ran * pow(a[0], ran - 1) * x0[i];
    }
    ok = ok && fabs(x[0] - pow(a[0], ran) * x0[0]) < 1e-12 && close_to(gx, want, N) && close_to(ga, &want_a, 1);
    char what[200];
    snprintf(what, sizeof(what), "a WHILE with cap 4 whose counter starts at %d%s%s: %d pass(es), dJ/dx0 = a^%d, dJ/da the closed form", (int)start,
             guarded ? ", an IF around the body, lane " : "", guarded ? (lane ? "set" : "clear") : "", ran, ran);
    verdict(r, ok, what);
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    eg_plan_free(ctx, tk);
    release(r);
}

/* Under a test on lane 2, a fork of y1 = w1 x and y2 = w2 x, then z = y1 y2 (y1, y2 preset): the fork's gradient flows when the arm ran. */
static void fork_in_arm_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y1[N], y2[N], z[N], w1[1] = { 1.7 }, w2[1] = { -0.6 }, one[N], gx[N], want[N];
    int32_t flags[4] = { 7, 7, 0, 7 };
    struct tensor tx, t1, t2, tz, tw1, tw2, tone, tgx, tf;
    fill(x, w1, 0.5);
    for (int i = 0; i < N; ++i) {
        one[i] = 1.0;
        y1[i] = 2.0;
        y2[i] = 3.0;
    }
    eg_plane *px = mk(r, &tx, N, x), *py1 = mk(r, &t1, N, y1), *py2 = mk(r, &t2, N, y2), *pz = mk(r, &tz, N, z), *pw1 = mk(r, &tw1, 1, w1);
    eg_plane *pw2 = mk(r, &tw2, 1, w2), *p1 = mk(r, &tone, N, one), *pgx = mk(r, &tgx, N, gx), *pf = mk_i32(r, &tf, 4, flags);
    eg_plane *a1[] = { px, pw1, py1 }, *a2[] = { px, pw2, py2 }, *b[] = { py1, py2, pz };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL), *m = plan_with(r, u->mul2, 3, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->scale_vjp }, { 2, u->mul2_vjp } };
    eg_graph* g = NULL;
    up(r);
    int ok = f1 && f2 && m && pf && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_IF, 0, pf, 2, 0) == EG_OK &&
             eg_record_fork_begin(ctx) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK && eg_record_fork_branch(ctx) == EG_OK &&
             eg_launch(ctx, f2, NULL) == EG_OK && eg_record_fork_end(ctx) == EG_OK && eg_record_cond_end(ctx) == EG_OK &&
             eg_launch(ctx, m, NULL) == EG_OK && end_diff(ctx, 3, vjps, &g) == EG_OK;
    eg_plane* in[] = { px };
    eg_plane* gr[] = { pgx };
    for (int lane = 1; ok && lane >= 0; --lane) {
        flags[2] = lane;
        up(r);
        ok = eg_graph_launch(ctx, g) == EG_OK && reverse(ctx, g, pz, p1, 1, in, gr) == EG_OK;
        down(r);
        for (int i = 0; i < N; ++i) want[i] = lane ? 2.0 * w1[0] * w2[0] * x[i] : 0.0;
        ok = ok && close_to(gx, want, N);
    }
    verdict(r, ok, "a fork inside an IF arm: dJ/dx = 2 w1 w2 x when the arm ran, 0 when it did not (y1, y2 keep their preset values)");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    eg_plan_free(ctx, m);
    release(r);
}

/* A WHILE with no cap has no bound for the tape. */
static void while_refusal_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], t[N], a[1] = { 0.9 };
    int32_t c[N];
    struct tensor tx, tt, ta, tc;
    fill(x, a, 0.5);
    for (int i = 0; i < N; ++i) c[i] = 1;
    eg_plane *px = mk(r, &tx, N, x), *pt = mk(r, &tt, N, t), *pa = mk(r, &ta, 1, a), *pc = mk_i32(r, &tc, N, c);
    eg_plane* a1[] = { px, pa, pt };
    eg_plan* f1 = plan_with(r, u->scale, 3, a1, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    eg_graph *g = NULL, *plain = NULL;
    int ok = f1 && pc && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, 0) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK &&
             eg_record_cond_end(ctx) == EG_OK;
    ok = ok && refused(end_diff(ctx, 1, vjps, &g), EG_BAD_REQUEST, ctx, "max_iters 0") && g == NULL && eg_record_end(ctx, NULL, &plain) == EG_OK;
    verdict(r, ok, "EG_RECORD_DIFF over a WHILE with max_iters 0 is refused by name; the recording stays open and ends plain");
    eg_graph_free(ctx, plain);
    eg_plan_free(ctx, f1);
    release(r);
}

static void conditional_rows(struct rig* r, const struct units* u) {
    fork_reverse_row(r, u);
    if_row(r, u, EG_COND_IF, 0, 0);
    if_row(r, u, EG_COND_IF, 1, 0);
    if_row(r, u, EG_COND_IF_ELSE, 0, 0);
    if_row(r, u, EG_COND_IF, 0, 1);
    while_row(r, u, 3, 0, 1);
    while_row(r, u, 100, 0, 1);
    while_row(r, u, 3, 1, 1);
    while_row(r, u, 3, 1, 0);
    fork_in_arm_row(r, u);
    while_refusal_row(r, u);
}

/* ---- reverse options: bitwise report, strict, passes, a non-float input (host) ------------------------------------------------ */

static eg_status reverse_with(eg_ctx* ctx, eg_graph* g, eg_plane* out, eg_plane* seed, int32_t n_in, eg_plane* const* in, eg_plane* const* grads,
                              uint32_t flags, uint8_t* bitwise, int32_t* passes) {
    struct eg_graph_reverse_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = flags;
    d.n_seeds = 1;
    d.outputs = &out;
    d.seeds = &seed;
    d.n_grads = n_in;
    d.inputs = in;
    d.grads = grads;
    d.bitwise = bitwise;
    d.passes = passes;
    return eg_graph_reverse(ctx, g, &d);
}

/* u = w1 x, y = w2 u: the vjp of the second launch is @p vjp1. Oracle: the closed forms dJ/dx = w1 w2, dJ/dw1 = sum w2 x,
 * dJ/dw2 = sum w1 x, and which gradients a scatter-add reaches written out by hand: the scatter vjp of launch 1 feeds the adjoint of
 * u, which launch 0 passes on to x and w1; w2 is fed by launch 1's plain reduced slot only. */
static void scatter_row(struct rig* r, const struct units* u, eg_unit* vjp1, int scatter) {
    eg_ctx* ctx = r->ctx;
    double x[N], uu[N], y[N], w1[1] = { 1.7 }, w2[1] = { -0.6 }, one[N], gx[N], gw1[1], gw2[1], want[N], sx = 0.0;
    struct tensor tx, tu, ty, tw1, tw2, tone, tgx, tgw1, tgw2;
    fill(x, w1, 0.5);
    for (int i = 0; i < N; ++i) {
        one[i] = 1.0;
        gx[i] = -999.0;
        sx += x[i];
    }
    eg_plane *px = mk(r, &tx, N, x), *pu = mk(r, &tu, N, uu), *py = mk(r, &ty, N, y), *pw1 = mk(r, &tw1, 1, w1), *pw2 = mk(r, &tw2, 1, w2);
    eg_plane *p1 = mk(r, &tone, N, one), *pgx = mk(r, &tgx, N, gx), *pgw1 = mk(r, &tgw1, 1, gw1), *pgw2 = mk(r, &tgw2, 1, gw2);
    eg_plane *a1[] = { px, pw1, pu }, *a2[] = { pu, pw2, py };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, vjp1 } };
    eg_graph* g = NULL;
    int ok = f1 && f2 && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK &&
             end_diff(ctx, 2, vjps, &g) == EG_OK && eg_graph_launch(ctx, g) == EG_OK;
    eg_plane* in[] = { px, pw1, pw2 };
    eg_plane* gr[] = { pgx, pgw1, pgw2 };
    uint8_t bitwise[3] = { 9, 9, 9 };
    int32_t passes = 0;
    ok = ok && reverse_with(ctx, g, py, p1, 3, in, gr, 0, bitwise, &passes) == EG_OK;
    for (int i = 0; i < N; ++i) want[i] = w1[0] * w2[0];
    const double want_w1 = w2[0] * sx, want_w2 = w1[0] * sx;
    ok = ok && close_to(gx, want, N) && close_to(gw1, &want_w1, 1) && close_to(gw2, &want_w2, 1) && passes == 1;
    ok = ok && bitwise[0] == !scatter && bitwise[1] == !scatter && bitwise[2] == 1;
    verdict(r, ok, scatter ? "a scatter-add vjp taints the gradients its adjoint reaches: bitwise 0 for x and w1, 1 for w2; values are the closed forms; passes 1"
                           : "without a scatter-add every gradient is bitwise; values are the closed forms; passes 1");
    /* STRICT refuses a tainted gradient by name before anything is written, and accepts a clean graph */
    for (int i = 0; i < N; ++i) gx[i] = -999.0;
    const eg_status st = reverse_with(ctx, g, py, p1, 3, in, gr, EG_REVERSE_STRICT, NULL, NULL);
    ok = scatter ? refused(st, EG_BAD_REQUEST, ctx, "inputs[0]") && reason_has(ctx, "launch 1") && gx[0] == -999.0 : st == EG_OK && close_to(gx, want, N);
    verdict(r, ok, scatter ? "EG_REVERSE_STRICT over a scatter-add gradient is refused by name (the input and the launch) and writes no gradient"
                           : "EG_REVERSE_STRICT over a graph with no scatter-add succeeds");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    release(r);
}

/* A scatter-add inside an IF arm taints only when the arm ran. */
static void scatter_if_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], x2[N], y[N], w1[1] = { 1.7 }, w2[1] = { -0.6 }, one[N], gx[N], gx2[N];
    int32_t flags[4] = { 7, 7, 0, 7 };
    struct tensor tx, tx2, ty, tw1, tw2, tone, tgx, tgx2, tf;
    fill(x, w1, 0.5);
    fill(x2, w1, -1.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *px2 = mk(r, &tx2, N, x2), *py = mk(r, &ty, N, y), *pw1 = mk(r, &tw1, 1, w1), *pw2 = mk(r, &tw2, 1, w2);
    eg_plane *p1 = mk(r, &tone, N, one), *pgx = mk(r, &tgx, N, gx), *pgx2 = mk(r, &tgx2, N, gx2), *pf = mk_i32(r, &tf, 4, flags);
    eg_plane *a1[] = { px, pw1, py }, *a2[] = { px2, pw2, py };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->scatter_vjp } };
    eg_graph* g = NULL;
    int ok = f1 && f2 && pf && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK && cond_open(ctx, EG_COND_IF, 0, pf, 2, 0) == EG_OK &&
             eg_launch(ctx, f2, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 2, vjps, &g) == EG_OK;
    eg_plane* in[] = { px, px2 };
    eg_plane* gr[] = { pgx, pgx2 };
    for (int lane = 1; ok && lane >= 0; --lane) {
        uint8_t bitwise[2] = { 9, 9 };
        flags[2] = lane;
        ok = eg_graph_launch(ctx, g) == EG_OK && reverse_with(ctx, g, py, p1, 2, in, gr, 0, bitwise, NULL) == EG_OK && bitwise[0] == 1 && bitwise[1] == !lane;
    }
    verdict(r, ok, "a scatter-add vjp under an IF arm taints x2 only when the arm ran; x, overwritten or alone, stays bitwise");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    release(r);
}

/* A non-float input has no gradient: refused by name. */
static void int_input_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], w[1] = { 1.7 }, one[N], gx[N];
    int32_t flags[4] = { 7, 7, 7, 7 };
    struct tensor tx, ty, tw, tone, tgx, tf;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *pw = mk(r, &tw, 1, w), *p1 = mk(r, &tone, N, one), *pgx = mk(r, &tgx, N, gx);
    eg_plane* pf = mk_i32(r, &tf, 4, flags);
    eg_plane* a1[] = { px, pw, py };
    eg_plan* f1 = plan_with(r, u->scale, 3, a1, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    eg_graph* g = NULL;
    int ok = f1 && pf && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_IF, 0, pf, 0, 0) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK &&
             eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 1, vjps, &g) == EG_OK && eg_graph_launch(ctx, g) == EG_OK;
    eg_plane* in[] = { pf };
    eg_plane* gr[] = { pgx };
    ok = ok && refused(reverse_with(ctx, g, py, p1, 1, in, gr, 0, NULL, NULL), EG_BAD_REQUEST, ctx, "is not a float plane");
    verdict(r, ok, "an int32 plane listed in inputs is refused by name: only a float input has a gradient");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    release(r);
}

static void option_rows(struct rig* r, const struct units* u) {
    scatter_row(r, u, u->scatter_vjp, 1);
    scatter_row(r, u, u->scale_vjp, 0);
    scatter_if_row(r, u);
    int_input_row(r, u);
}

static struct units load_units(struct rig* r) {
    const struct units u = { load(r, "scale", 0x10), load(r, "scale_vjp", 0x11), load(r, "square", 0x12), load(r, "square_vjp", 0x13),
                             load(r, "mul2", 0x14), load(r, "mul2_vjp", 0x15), load(r, "uscale", 0x16), load(r, "uscale_vjp", 0x17),
                             load(r, "scale_tape_vjp", 0x18), load(r, "scale_open_vjp", 0x19),
                             load(r, "fan", 0x1a), load(r, "fan_vjp", 0x1b), load(r, "norm2", 0x1c), load(r, "norm2_vjp", 0x1d), load(r, "tick", 0x1e), load(r, "scale_scatter_vjp", 0x1f) };
    verdict(r, u.scale && u.scale_vjp && u.square && u.square_vjp && u.mul2 && u.mul2_vjp && u.uscale && u.uscale_vjp && u.tape_vjp &&
                   u.open_vjp && u.fan && u.fan_vjp && u.norm2 && u.norm2_vjp && u.tick && u.scatter_vjp,
            "the graph fixture's sixteen units load");
    return u;
}

static void gradient_rows(struct rig* r, const struct units* u) {
    chain_row(r, u);
    overwrite_row(r, u);
    shared_read_row(r, u);
    vector_row(r, u, 0);
    vector_row(r, u, 1);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_graph lib>\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);    /* a crash shows the last row that ran */
    (void)parse_digest;    /* shared helper this test does not need */
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new host context");
    struct rig host = { .ctx = ctx, .lib = argv[1] };
    const struct units u = load_units(&host);
    gradient_rows(&host, &u);
    refusal_rows(&host, &u);
    conditional_rows(&host, &u);
    option_rows(&host, &u);

    int32_t devices = 0;
    if (eg_device_count(ctx, &devices) != EG_OK || devices < 1) {
        check(1, "graph reverse rows on a DEVICE context: skipped, no device present");
    } else {
        st.flags = EG_CTX_DEVICE;
        eg_ctx* dctx = NULL;
        check(eg_ctx_new(&st, &dctx) == EG_OK, "eg_ctx_new DEVICE context");
        struct rig dev = { .ctx = dctx, .lib = argv[1], .device = 1 };
        const struct units du = load_units(&dev);
        gradient_rows(&dev, &du);
        device_refusal_rows(&dev, &du);
        conditional_rows(&dev, &du);
        eg_ctx_free(dctx);
    }
    eg_ctx_free(ctx);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures == 0 ? 0 : 1;
}
