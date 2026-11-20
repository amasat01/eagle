// SPDX-License-Identifier: Apache-2.0
/*
 * eg_graph_reverse on the host.
 *
 *   graph_reverse_test <fixture_graph lib>
 *
 * Gradients of recorded graphs against closed forms: launches (a chain, an in-place overwrite, a plane read by two nodes), copies, sum
 * reduces, transforms and products, in a WHILE too; then every refusal by status and reason.
 */
#include "pair_util.h"

#include <math.h>
#include <stdlib.h>

#ifdef EG_TEST_CUDA
#include <cuda_runtime_api.h>
#endif

#define N 7

/* A context, the fixture library, and (on a DEVICE context) the planes the rows made: each row builds host arrays, keeps device
 * planes beside them, and copies the arrays up before a launch and the results down after a reverse. */
struct rig {
    eg_ctx* ctx;
    const char* lib;
    const char* run_lib;     /* the library of the recorded runner's units (fixture_reverse) */
    int device;
    int64_t n_samples;       /* 0: N */
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
    d.n_samples = r->n_samples != 0 ? r->n_samples : N;
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
    eg_unit *iscale, *iscale_vjp, *scale_vjp_comp;
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

/* Record one eg_reduce of @p in into @p out (one column) with @p op. */
static eg_status reduce_op(eg_ctx* ctx, eg_plane* in, eg_plane* out, eg_reduce_op op) {
    struct eg_reduce_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.in = in;
    d.out = out;
    d.op = op;
    d.policy = EG_REDUCE_COMPENSATED;
    return eg_reduce(ctx, &d);
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
    ok = eg_record_begin(ctx) == EG_OK && reduce_op(ctx, px, pw, EG_REDUCE_MAX) == EG_OK;
    check(ok && refused(end_diff(ctx, 0, NULL, &none), EG_BAD_REQUEST, ctx, "only op='sum' reverses"),
          "eg_record_end with EG_RECORD_DIFF over a recorded reduce MAX: refused BAD_REQUEST by name");
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

/* The refusals a DEVICE recording adds: a reduce with no gradient, and a tape plan that is refused (it ends the recording). */
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

    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && reduce_op(ctx, py, pw, EG_REDUCE_MAX) == EG_OK &&
             eg_launch(ctx, g, NULL) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 2, both, &none), EG_BAD_REQUEST, ctx, "only op='sum' reverses") && none == NULL,
            "eg_record_end with EG_RECORD_DIFF over a recording that holds a reduce MAX between two launches: refused BAD_REQUEST by name");
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

/* A one-dimensional plane of @p bits wide elements of type @p code over @p data: a host plane, or on a DEVICE context a device plane beside the host wrapper. */
static eg_plane* mk_typed(struct rig* r, struct tensor* t, int64_t n, void* data, uint8_t code, uint8_t bits) {
    tensor_fill(t, n, data, code, bits);
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
        d.dtype_code = code;
        d.dtype_bits = bits;
        d.shape = shape;
        handle = NULL;
        if (eg_plane_new(r->ctx, &d, &handle) != EG_OK) return NULL;
    }
    r->handle[r->n_planes] = handle;            /* released with the row, before its tensor goes out of scope */
    r->host[r->n_planes++] = r->device ? host : NULL;
    return handle;
}
static eg_plane* mk_i32(struct rig* r, struct tensor* t, int64_t n, int32_t* data) { return mk_typed(r, t, n, data, kDLInt, 32); }

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

static eg_status reverse_with(eg_ctx* ctx, eg_graph* g, eg_plane* out, eg_plane* seed, int32_t n_in, eg_plane* const* in, eg_plane* const* grads,
                              uint32_t flags, uint8_t* bitwise, int32_t* passes);

/* x <- a x per pass (t = a x, x = 1 t), a counter plane ticking down: the loop stops at zero or at its cap. With @p guarded the two
 * launches sit in an IF on lane 2 of a flag array, inside the loop. */
static void while_row(struct rig* r, const struct units* u, int32_t start, int guarded, int lane, int cap, eg_unit* vjp0) {
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
    const struct eg_node_vjp vjps[] = { { 0, vjp0 }, { 1, u->scale_vjp } };
    eg_graph* g = NULL;
    up(r);
    int ok = f1 && f2 && tk && pf && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, cap) == EG_OK;
    if (guarded) ok = ok && cond_open(ctx, EG_COND_IF, 0, pf, 2, 0) == EG_OK;
    ok = ok && eg_launch(ctx, f1, NULL) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK;
    if (guarded) ok = ok && eg_record_cond_end(ctx) == EG_OK;
    ok = ok && eg_launch(ctx, tk, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 2, vjps, &g) == EG_OK;
    for (int i = 0; i < N; ++i) c[i] = start;
    flags[2] = lane;
    up(r);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK;
    const int passes = start > cap ? cap : start;
    const int ran = guarded && lane == 0 ? 0 : passes;     /* passes that multiplied */
    eg_plane* in[] = { px, pa };
    eg_plane* gr[] = { pgx, pga };
    uint8_t bitwise[2] = { 9, 9 };
    ok = ok && reverse_with(ctx, g, px, ps, 2, in, gr, 0, bitwise, NULL) == EG_OK && bitwise[0] == (vjp0 == u->scale_vjp) && bitwise[1] == (vjp0 == u->scale_vjp);
    down(r);
    for (int i = 0; i < N; ++i) {
        want[i] = pow(a[0], ran);
        want_a += ran * pow(a[0], ran - 1) * x0[i];
    }
    ok = ok && fabs(x[0] - pow(a[0], ran) * x0[0]) < 1e-12 && close_to(gx, want, N) && close_to(ga, &want_a, 1);
    char what[200];
    snprintf(what, sizeof(what), "a WHILE with cap %d whose counter starts at %d%s%s%s: %d pass(es), dJ/dx0 = a^%d, dJ/da the closed form", cap, (int)start,
             vjp0 == u->scale_vjp ? "" : ", a scatter-add vjp (bitwise 0)",
             guarded ? ", an IF around the body, lane " : "", guarded ? (lane ? "set" : "clear") : "", ran, ran);
    verdict(r, ok, what);
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    eg_plan_free(ctx, tk);
    release(r);
}

/* The WHILE of cap 40 (windows of 16, stopping after 37 passes) against the same 37 passes written out as 74 launches: the window gives
 * the unrolled chain's gradients bit for bit. */
static void window_chain_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    enum { PASSES = 37, CAP = 40 };
    double x0[N], x[N], t[N], a[1] = { 0.9 }, one1[1] = { 1.0 }, seed[N], gx[2][N], ga[2][1];
    int32_t c[N];
    struct tensor tx, tt, ta, to, ts, tgx, tga, tc;
    fill(x, a, 0.5);
    a[0] = 0.9;
    for (int i = 0; i < N; ++i) {
        x0[i] = x[i];
        seed[i] = 1.0;
    }
    eg_plane *px = mk(r, &tx, N, x), *pt = mk(r, &tt, N, t), *pa = mk(r, &ta, 1, a), *po = mk(r, &to, 1, one1), *ps = mk(r, &ts, N, seed);
    eg_plane *pgx = mk(r, &tgx, N, gx[0]), *pga = mk(r, &tga, 1, ga[0]), *pc = mk_i32(r, &tc, N, c);
    eg_plane *a1[] = { px, pa, pt }, *a2[] = { pt, po, px }, *a3[] = { pc };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL), *tk = plan_with(r, u->tick, 1, a3, NULL);
    struct eg_node_vjp vjps[2 * PASSES];
    for (int i = 0; i < 2 * PASSES; ++i) {
        vjps[i].launch = i;
        vjps[i].vjp = u->scale_vjp;
    }
    eg_graph *loop = NULL, *chain = NULL;
    int ok = f1 && f2 && tk && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, CAP) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK &&
             eg_launch(ctx, f2, NULL) == EG_OK && eg_launch(ctx, tk, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 2, vjps, &loop) == EG_OK;
    if (!ok) printf("      stage: the loop recording\n");
    ok = ok && eg_record_begin(ctx) == EG_OK;
    for (int k = 0; ok && k < PASSES; ++k) ok = eg_launch(ctx, f1, NULL) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK;
    if (!ok) printf("      stage: the chain launches\n");
    ok = ok && end_diff(ctx, 2 * PASSES, vjps, &chain) == EG_OK;
    if (!ok) printf("      stage: the chain end\n");
    eg_plane* in[] = { px, pa };
    for (int which = 0; ok && which < 2; ++which) {
        eg_graph* g = which ? chain : loop;
        for (int i = 0; i < N; ++i) {
            x[i] = x0[i];
            c[i] = PASSES;
        }
        up(r);
        ok = eg_graph_launch(ctx, g) == EG_OK;
        eg_plane* gr[] = { which ? NULL : pgx, pga };
        double* gxw = gx[which];
        struct tensor tgx2, tga2;
        eg_plane* qgx = mk(r, &tgx2, N, gxw);
        eg_plane* qga = mk(r, &tga2, 1, ga[which]);
        gr[0] = qgx;
        gr[1] = qga;
        if (!ok) printf("      stage: launch of graph %d\n", which);
        eg_status rs = ok && qgx && qga ? reverse(ctx, g, px, ps, 2, in, gr) : EG_BAD_REQUEST;
        if (ok && rs != EG_OK) {
            char buf[400];
            size_t len = 0;
            eg_error(ctx, buf, sizeof(buf), &len);
            printf("      reverse of graph %d: status %d, %s\n", which, (int)rs, buf);
        }
        ok = ok && rs == EG_OK;
        down(r);
    }
    verdict(r, ok && memcmp(gx[0], gx[1], sizeof(gx[0])) == 0 && memcmp(ga[0], ga[1], sizeof(ga[0])) == 0 && fabs(gx[0][0] - pow(a[0], PASSES)) < 1e-12,
            "a WHILE of cap 40 (windows of 16) over 37 passes gives bit for bit the gradients of the same 37 passes written out as launches");
    eg_graph_free(ctx, loop);
    eg_graph_free(ctx, chain);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    eg_plan_free(ctx, tk);
    release(r);
}

/* y = w x reads x before a WHILE (cap 40, 37 passes) overwrites it: the gradient of y needs x as it was, the checkpoint at loop entry. */
static void window_reader_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], x0[N], t[N], y[N], w[1] = { 1.7 }, a[1] = { 0.9 }, one1[1] = { 1.0 }, seed[N], gx[N], gw[1], want_w = 0.0;
    int32_t c[N];
    struct tensor tx, tt, ty, tw, ta, to, ts, tgx, tgw, tc;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) {
        x0[i] = x[i];
        seed[i] = 1.0;
        want_w += x0[i];
        c[i] = 37;
    }
    eg_plane *px = mk(r, &tx, N, x), *pt = mk(r, &tt, N, t), *py = mk(r, &ty, N, y), *pw = mk(r, &tw, 1, w), *pa = mk(r, &ta, 1, a);
    eg_plane *po = mk(r, &to, 1, one1), *ps = mk(r, &ts, N, seed), *pgx = mk(r, &tgx, N, gx), *pgw = mk(r, &tgw, 1, gw), *pc = mk_i32(r, &tc, N, c);
    eg_plane *a0[] = { px, pw, py }, *a1[] = { px, pa, pt }, *a2[] = { pt, po, px }, *a3[] = { pc };
    eg_plan *f0 = plan_with(r, u->scale, 3, a0, NULL), *f1 = plan_with(r, u->scale, 3, a1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL);
    eg_plan* tk = plan_with(r, u->tick, 1, a3, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->scale_vjp }, { 2, u->scale_vjp } };
    eg_graph* g = NULL;
    int ok = f0 && f1 && f2 && tk && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f0, NULL) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, 40) == EG_OK &&
             eg_launch(ctx, f1, NULL) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK && eg_launch(ctx, tk, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK &&
             end_diff(ctx, 3, vjps, &g) == EG_OK && (up(r), eg_graph_launch(ctx, g)) == EG_OK;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, g, py, ps, 2, in, gr) == EG_OK;
    down(r);
    double want[N];
    for (int i = 0; i < N; ++i) want[i] = w[0];
    verdict(r, ok && close_to(gx, want, N) && close_to(gw, &want_w, 1) && fabs(x[0] - pow(a[0], 37) * x0[0]) < 1e-12,
            "a launch before a WHILE reads a plane the loop overwrites: its gradient uses the plane as it was at loop entry (dJ/dw = sum x0)");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f0);
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
    while_row(r, u, 3, 0, 1, 4, u->scale_vjp);
    while_row(r, u, 100, 0, 1, 4, u->scale_vjp);
    while_row(r, u, 3, 1, 1, 4, u->scale_vjp);
    while_row(r, u, 3, 1, 0, 4, u->scale_vjp);
    while_row(r, u, 37, 0, 1, 40, u->scale_vjp);      /* 3 windows of 16: the loop stops inside the last */
    while_row(r, u, 100, 0, 1, 40, u->scale_vjp);     /* the cap is reached */
    while_row(r, u, 37, 1, 1, 40, u->scale_vjp);
    while_row(r, u, 3, 0, 1, 4, u->scatter_vjp);
    while_row(r, u, 37, 0, 1, 40, u->scatter_vjp);
    window_chain_row(r, u);
    window_reader_row(r, u);
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
             end_diff(ctx, 2, vjps, &g) == EG_OK && (up(r), eg_graph_launch(ctx, g)) == EG_OK;
    eg_plane* in[] = { px, pw1, pw2 };
    eg_plane* gr[] = { pgx, pgw1, pgw2 };
    uint8_t bitwise[3] = { 9, 9, 9 };
    int32_t passes = 0;
    ok = ok && reverse_with(ctx, g, py, p1, 3, in, gr, 0, bitwise, &passes) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) want[i] = w1[0] * w2[0];
    const double want_w1 = w2[0] * sx, want_w2 = w1[0] * sx;
    ok = ok && close_to(gx, want, N) && close_to(gw1, &want_w1, 1) && close_to(gw2, &want_w2, 1) && passes == 1;
    ok = ok && bitwise[0] == !scatter && bitwise[1] == !scatter && bitwise[2] == 1;
    if (!ok) printf("      bitwise %d %d %d, passes %d, gx %g (want %g), gw1 %g (want %g), gw2 %g (want %g)\n", bitwise[0], bitwise[1], bitwise[2], (int)passes, gx[0], want[0], gw1[0], want_w1, gw2[0], want_w2);
    verdict(r, ok, scatter ? "a scatter-add vjp taints the gradients its adjoint reaches: bitwise 0 for x and w1, 1 for w2; values are the closed forms; passes 1"
                           : "without a scatter-add every gradient is bitwise; values are the closed forms; passes 1");
    /* STRICT refuses a tainted gradient by name before anything is written, and accepts a clean graph */
    for (int i = 0; i < N; ++i) gx[i] = -999.0;
    up(r);
    const eg_status st = reverse_with(ctx, g, py, p1, 3, in, gr, EG_REVERSE_STRICT, NULL, NULL);
    down(r);
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
        up(r);
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


/* ---- slice 3b: copies, sum reduces, transforms and products in a recording ---------------------------------------------- */

struct tensor3 {
    struct DLManagedTensorVersioned managed;
    int64_t shape[3];
};

/* A float64 plane of @p ndim axes over @p data (the row's host array): a host plane, or on a DEVICE context a device plane beside it. */
static eg_plane* mkn(struct rig* r, struct tensor3* t, int ndim, const int64_t* shape, double* data) {
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < ndim; ++i) t->shape[i] = shape[i];
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCPU;
    t->managed.dl_tensor.ndim = ndim;
    t->managed.dl_tensor.dtype.code = kDLFloat;
    t->managed.dl_tensor.dtype.bits = 64;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    eg_plane* host = NULL;
    if (eg_plane_wrap(r->ctx, &t->managed, &host) != EG_OK) return NULL;
    return mk_shaped(r, host, ndim, shape);
}

static int close_rel(const double* got, const double* want, int n, double tol) {
    for (int i = 0; i < n; ++i)
        if (fabs(got[i] - want[i]) > tol * (1.0 + fabs(want[i]))) return 0;
    return 1;
}

/* A copy before a launch (@p after 0: z = (copy x)^2, dJ/dx = 2 x) or between two (@p after 1: y = w x, t = copy y, z = t^2). J = sum z. */
static void copy_row(struct rig* r, const struct units* u, int after) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], t[N], z[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1] = { 0.0 };
    struct tensor tx, ty, tt, tz, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    memset(y, 0, sizeof(y));
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *pt = mk(r, &tt, N, t), *pz = mk(r, &tz, N, z), *pw = mk(r, &tw, 1, w);
    eg_plane *p1 = mk(r, &t1, N, one), *pgx = mk(r, &tgx, N, gx), *pgw = mk(r, &tgw, 1, gw);
    eg_plane *a[] = { px, pw, py }, *b[] = { pt, pz };
    eg_plan *f = plan_with(r, u->scale, 3, a, NULL), *g = plan_with(r, u->square, 2, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->square_vjp } }, only_square[] = { { 0, u->square_vjp } };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && g && eg_record_begin(ctx) == EG_OK;
    if (after) ok = ok && eg_launch(ctx, f, NULL) == EG_OK && eg_copy(ctx, py, pt) == EG_OK;
    else ok = ok && eg_copy(ctx, px, pt) == EG_OK;
    ok = ok && eg_launch(ctx, g, NULL) == EG_OK && end_diff(ctx, after ? 2 : 1, after ? vjps : only_square, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane *in[] = { px, pw }, *gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, pz, p1, after ? 2 : 1, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) {
        wx[i] = after ? 2.0 * w[0] * w[0] * x[i] : 2.0 * x[i];
        ww[0] += after ? 2.0 * w[0] * x[i] * x[i] : 0.0;
    }
    verdict(r, ok && close_to(gx, wx, N) && (!after || close_to(gw, ww, 1)),
            after ? "eg_graph_reverse through a copy between two launches (y = w x, t = copy y, z = t^2): dJ/dx and dJ/dw match the closed form"
                  : "eg_graph_reverse through a copy before a launch (z = (copy x)^2): dJ/dx = 2 x, the launch's own gradient");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    release(r);
}

/* y = w x, s = sum y, J = c s: every sample's adjoint is the seed c, so dJ/dx_i = c w and dJ/dw = c sum x. Then the reduces with no gradient. */
static void reduce_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], s[1], w[1], seed[1] = { 1.3 }, gx[N], gw[1], wx[N], ww[1] = { 0.0 };
    struct tensor tx, ty, ts, tw, tseed, tgx, tgw;
    fill(x, w, 0.5);
    memset(y, 0, sizeof(y));
    s[0] = 0.0;
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *ps = mk(r, &ts, 1, s), *pw = mk(r, &tw, 1, w), *pseed = mk(r, &tseed, 1, seed);
    eg_plane *pgx = mk(r, &tgx, N, gx), *pgw = mk(r, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && reduce_op(ctx, py, ps, EG_REDUCE_SUM) == EG_OK &&
             end_diff(ctx, 1, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane *in[] = { px, pw }, *gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, ps, pseed, 2, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) {
        wx[i] = seed[0] * w[0];
        ww[0] += seed[0] * x[i];
    }
    verdict(r, ok && close_to(gx, wx, N) && close_to(gw, ww, 1),
            "eg_graph_reverse through a sum reduce (y = w x, s = sum y, J = 1.3 s): every sample's adjoint is the seed, dJ/dx_i = 1.3 w and dJ/dw = 1.3 sum x");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);

    double q[2 * N], two[2];
    struct tensor2 tq;
    struct tensor tt2;
    memset(q, 0, sizeof(q));
    memset(two, 0, sizeof(two));
    eg_plane *pq = mk2(r, &tq, 2, q), *p2 = mk(r, &tt2, 2, two);
    eg_graph* none = NULL;
    ok = pq && p2 && eg_record_begin(ctx) == EG_OK && reduce_op(ctx, pq, p2, EG_REDUCE_SUM) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 0, NULL, &none), EG_NOT_WRITTEN, ctx, "only a 1-D op='sum' reduce reverses") && none == NULL,
            "eg_record_end with EG_RECORD_DIFF over a 2-D sum reduce: refused NOT_WRITTEN with the 1-D text");
    ok = eg_record_end(ctx, NULL, &none) == EG_OK;
    verdict(r, ok, "the recording left open by that refusal ends as a plain graph");
    eg_graph_free(ctx, none);
    release(r);
}

/* The closed forms of J = <w, T x> over the interleaved floats: dJ/dx for the transform @p kind (sign @p sigma: +1 forward, -1 inverse) of length n. */
static void fft_reference(eg_fft_kind kind, double sigma, double scale, int n, const double* w, double* gx) {
    const int m = n / 2 + 1;
    const double two_pi = 8.0 * atan(1.0);
    if (kind == EG_FFT_C2C) {
        for (int j = 0; j < n; ++j) {
            gx[2 * j] = gx[2 * j + 1] = 0.0;
            for (int k = 0; k < n; ++k) {
                const double th = two_pi * ((j * k) % n) / n;
                gx[2 * j] += scale * (w[2 * k] * cos(th) - sigma * w[2 * k + 1] * sin(th));
                gx[2 * j + 1] += scale * (sigma * w[2 * k] * sin(th) + w[2 * k + 1] * cos(th));
            }
        }
    } else if (kind == EG_FFT_R2C) {
        for (int j = 0; j < n; ++j) {
            gx[j] = 0.0;
            for (int k = 0; k < m; ++k) {
                const double th = two_pi * ((j * k) % n) / n;
                gx[j] += scale * (w[2 * k] * cos(th) - w[2 * k + 1] * sin(th));
            }
        }
    } else {
        for (int k = 0; k < m; ++k) {
            const double c = k == 0 || (n % 2 == 0 && k == n / 2) ? 1.0 : 2.0;
            gx[2 * k] = gx[2 * k + 1] = 0.0;
            for (int j = 0; j < n; ++j) {
                const double th = two_pi * ((j * k) % n) / n;
                gx[2 * k] += scale * c * w[j] * cos(th);
                gx[2 * k + 1] -= scale * c * w[j] * sin(th);
            }
        }
    }
}

/* J = <w, T x> for one transform: dJ/dx from the door against the closed form and, with @p check_fd, against central differences of J through the graph. */
static void fft_row(struct rig* r, eg_fft_kind kind, int inverse, eg_fft_norm norm, int n, int check_fd, const char* name) {
    eg_ctx* ctx = r->ctx;
    const int m = n / 2 + 1;
    const int64_t nx_shape[2] = { kind == EG_FFT_C2R ? m : n, 2 }, ny_shape[2] = { kind == EG_FFT_R2C ? m : n, 2 };
    const int x_ndim = kind == EG_FFT_R2C ? 1 : 2, y_ndim = kind == EG_FFT_C2R ? 1 : 2;
    const int nx = x_ndim == 1 ? n : 2 * (int)nx_shape[0], ny = y_ndim == 1 ? n : 2 * (int)ny_shape[0];
    double x[32], y[32], w[32], gx[32], want[32], saved[32], x0[32];
    for (int i = 0; i < nx; ++i) x[i] = x0[i] = sin(0.7 * i + 0.3);
    for (int i = 0; i < ny; ++i) w[i] = cos(1.1 * i + 0.2);
    memset(y, 0, sizeof(y));
    memset(gx, 0, sizeof(gx));
    const double sigma = (kind == EG_FFT_C2C && !inverse) || kind == EG_FFT_R2C ? 1.0 : -1.0;
    const int forward = sigma > 0;
    const double scale = norm == EG_FFT_NORM_ORTHO ? 1.0 / sqrt((double)n) : (norm == EG_FFT_NORM_FORWARD) == forward ? 1.0 / n : 1.0;
    struct tensor3 tx, ty, tw, tg;
    eg_plane *px = mkn(r, &tx, x_ndim, nx_shape, x), *py = mkn(r, &ty, y_ndim, ny_shape, y), *pw = mkn(r, &tw, y_ndim, ny_shape, w);
    eg_plane* pg = mkn(r, &tg, x_ndim, nx_shape, gx);
    const int64_t shape_n = n;
    const int32_t axis = 0;
    struct eg_fft_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = inverse ? EG_FFT_INVERSE : 0;
    d.kind = kind;
    d.norm = norm;
    d.dtype_bits = 64;
    d.ndim = 1;
    d.shape = &shape_n;
    d.n_axes = 1;
    d.axes = &axis;
    eg_fft* plan = NULL;
    eg_graph* graph = NULL;
    up(r);
    int ok = px && py && pw && pg && eg_fft_new(ctx, &d, &plan) == EG_OK && eg_record_begin(ctx) == EG_OK && eg_fft_run(ctx, plan, px, py) == EG_OK &&
             end_diff(ctx, 0, NULL, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane *in[] = { px }, *gr[] = { pg };
    ok = ok && reverse(ctx, graph, py, pw, 1, in, gr) == EG_OK;
    down(r);
    fft_reference(kind, sigma, scale, n, w, want);
    memcpy(saved, gx, sizeof(gx));
    char what[300];
    snprintf(what, sizeof(what), "eg_graph_reverse through %s: dJ/dx of J = <w, T x> matches the closed form", name);
    verdict(r, ok && close_rel(saved, want, nx, 1e-11), what);
    if (check_fd) {
        const double h = 0.5;
        double fd[32];
        for (int j = 0; ok && j < nx; ++j) {
            double j_two[2];
            for (int side = 0; side < 2; ++side) {
                x[j] = x0[j] + (side == 0 ? h : -h);
                up(r);
                ok = eg_graph_launch(ctx, graph) == EG_OK;
                down(r);
                j_two[side] = 0.0;
                for (int i = 0; i < ny; ++i) j_two[side] += w[i] * y[i];
            }
            x[j] = x0[j];
            fd[j] = (j_two[0] - j_two[1]) / (2.0 * h);
        }
        snprintf(what, sizeof(what), "the same gradient of %s against central differences of J through the graph", name);
        verdict(r, ok && close_rel(saved, fd, nx, 1e-9), what);
    }
    eg_graph_free(ctx, graph);
    eg_fft_free(ctx, plan);
    release(r);
}

/* The logical matrix entries of a stored (rows, cols) matrix that is transposed when @p trans. */
static double entry(const double* p, int cols, int trans, int i, int j) { return trans ? p[j * cols + i] : p[i * cols + j]; }

/* J = <W, C> for C = op(A) op(B) (+ C0 with ACCUMULATE): dJ/dA = W op(B)^T and dJ/dB = op(A)^T W in the stored layout, dJ/dC0 = W. */
static void gemm_row(struct rig* r, uint32_t flags, const char* name) {
    eg_ctx* ctx = r->ctx;
    enum { M = 3, K = 4, NC = 5 };
    const int ta = (flags & EG_GEMM_TRANS_A) != 0, tb = (flags & EG_GEMM_TRANS_B) != 0, tc = (flags & EG_GEMM_TRANS_C) != 0, acc = (flags & EG_GEMM_ACCUMULATE) != 0;
    const int64_t as[2] = { ta ? K : M, ta ? M : K }, bs[2] = { tb ? NC : K, tb ? K : NC }, cs[2] = { tc ? NC : M, tc ? M : NC };
    double a[M * K], b[K * NC], c[M * NC], seed[M * NC], ga[M * K], gb[K * NC], gc[M * NC], want_a[M * K], want_b[K * NC];
    for (int i = 0; i < M * K; ++i) a[i] = sin(0.9 * i + 0.1);
    for (int i = 0; i < K * NC; ++i) b[i] = cos(0.7 * i + 0.4);
    for (int i = 0; i < M * NC; ++i) {
        c[i] = 0.25 * i - 1.0;
        seed[i] = sin(1.3 * i + 0.6);
    }
    memset(ga, 0, sizeof(ga));
    memset(gb, 0, sizeof(gb));
    memset(gc, 0, sizeof(gc));
    for (int i = 0; i < M; ++i)
        for (int p = 0; p < K; ++p) {
            double sum = 0.0;
            for (int j = 0; j < NC; ++j) sum += entry(seed, tc ? M : NC, tc, i, j) * entry(b, tb ? K : NC, tb, p, j);
            want_a[ta ? p * M + i : i * K + p] = sum;
        }
    for (int p = 0; p < K; ++p)
        for (int j = 0; j < NC; ++j) {
            double sum = 0.0;
            for (int i = 0; i < M; ++i) sum += entry(a, ta ? M : K, ta, i, p) * entry(seed, tc ? M : NC, tc, i, j);
            want_b[tb ? j * K + p : p * NC + j] = sum;
        }
    struct tensor3 ta3, tb3, tc3, ts3, tga, tgb, tgc;
    eg_plane *pa = mkn(r, &ta3, 2, as, a), *pb = mkn(r, &tb3, 2, bs, b), *pc = mkn(r, &tc3, 2, cs, c), *ps = mkn(r, &ts3, 2, cs, seed);
    eg_plane *pga = mkn(r, &tga, 2, as, ga), *pgb = mkn(r, &tgb, 2, bs, gb), *pgc = mkn(r, &tgc, 2, cs, gc);
    struct eg_gemm_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = flags;
    d.dtype_bits = 64;
    d.a_rows = as[0], d.a_cols = as[1], d.b_rows = bs[0], d.b_cols = bs[1], d.c_rows = cs[0], d.c_cols = cs[1];
    eg_gemm* gemm = NULL;
    eg_graph* graph = NULL;
    up(r);
    int ok = pa && pb && pc && ps && pga && pgb && pgc && eg_gemm_new(ctx, &d, &gemm) == EG_OK && eg_record_begin(ctx) == EG_OK &&
             eg_gemm_run(ctx, gemm, pa, pb, pc) == EG_OK && end_diff(ctx, 0, NULL, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane *in[] = { pa, pb, pc }, *gr[] = { pga, pgb, pgc };
    ok = ok && reverse(ctx, graph, pc, ps, acc ? 3 : 2, in, gr) == EG_OK;
    down(r);
    char what[300];
    snprintf(what, sizeof(what), "eg_graph_reverse through a product%s: dJ/dA = W op(B)^T and dJ/dB = op(A)^T W match the closed form%s", name,
             acc ? ", and dJ/dC0 = W" : "");
    verdict(r, ok && close_rel(ga, want_a, M * K, 1e-12) && close_rel(gb, want_b, K * NC, 1e-12) && (!acc || close_rel(gc, seed, M * NC, 1e-12)), what);
    eg_graph_free(ctx, graph);
    eg_gemm_free(ctx, gemm);
    release(r);
}

/* X <- X M by a product and a copy, in a WHILE of cap 40 (windows of 16) that runs 37 times, against the same 37 pairs written out: the
 * window rebuild re-runs the product and the copy, and the gradients are the written-out sequence's bit for bit. */
static void window_gemm_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    if (r->device) {    /* the body of a WHILE runs on an arm stream of the capture, and a product there needs a cuBLAS handle of that stream, which the door does not make */
        check(1, "on a DEVICE context: a WHILE with a product in its body: skipped, a device product on an arm stream has no cuBLAS handle (the copy rebuild is the next row)");
        return;
    }
    enum { PASSES = 37, CAP = 40, S = 3 };
    const int64_t sq[2] = { S, S };
    double x0[S * S], x[S * S], mm[S * S], t[S * S], seed[S * S], gx[2][S * S], gm[2][S * S], want[S * S];
    int32_t c[N];
    for (int i = 0; i < S * S; ++i) {
        x0[i] = x[i] = sin(0.8 * i + 0.2);
        mm[i] = 0.2 + 0.05 * (i % 4);
        seed[i] = cos(0.6 * i + 0.1);
        t[i] = 0.0;
    }
    struct tensor3 tx, tm, tt, ts, tgx[2], tgm[2];
    struct tensor tc;
    eg_plane *px = mkn(r, &tx, 2, sq, x), *pm = mkn(r, &tm, 2, sq, mm), *pt = mkn(r, &tt, 2, sq, t), *ps = mkn(r, &ts, 2, sq, seed);
    eg_plane *pgx[2] = { mkn(r, &tgx[0], 2, sq, gx[0]), mkn(r, &tgx[1], 2, sq, gx[1]) }, *pgm[2] = { mkn(r, &tgm[0], 2, sq, gm[0]), mkn(r, &tgm[1], 2, sq, gm[1]) };
    eg_plane* pc = mk_i32(r, &tc, N, c);
    eg_plane* a3[] = { pc };
    eg_plan* tk = plan_with(r, u->tick, 1, a3, NULL);
    struct eg_gemm_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.dtype_bits = 64;
    d.a_rows = d.a_cols = d.b_rows = d.b_cols = d.c_rows = d.c_cols = S;
    eg_gemm* gemm = NULL;
    eg_graph *loop = NULL, *chain = NULL;
    int ok = tk && eg_gemm_new(ctx, &d, &gemm) == EG_OK && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, CAP) == EG_OK &&
             eg_gemm_run(ctx, gemm, px, pm, pt) == EG_OK && eg_copy(ctx, pt, px) == EG_OK && eg_launch(ctx, tk, NULL) == EG_OK &&
             eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 0, NULL, &loop) == EG_OK;
    ok = ok && eg_record_begin(ctx) == EG_OK;
    for (int k = 0; ok && k < PASSES; ++k) ok = eg_gemm_run(ctx, gemm, px, pm, pt) == EG_OK && eg_copy(ctx, pt, px) == EG_OK;
    ok = ok && end_diff(ctx, 0, NULL, &chain) == EG_OK;
    eg_plane* in[] = { px, pm };
    for (int which = 0; ok && which < 2; ++which) {
        for (int i = 0; i < S * S; ++i) x[i] = x0[i];
        for (int i = 0; i < N; ++i) c[i] = PASSES;
        up(r);
        eg_plane* gr[] = { pgx[which], pgm[which] };
            ok = eg_graph_launch(ctx, which ? chain : loop) == EG_OK && reverse(ctx, which ? chain : loop, px, ps, 2, in, gr) == EG_OK;
        down(r);
    }
    double power[S * S], next[S * S];
    for (int i = 0; i < S * S; ++i) power[i] = i / S == i % S;    /* M^0 */
    for (int k = 0; k < PASSES; ++k) {
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) {
                next[i * S + j] = 0.0;
                for (int p = 0; p < S; ++p) next[i * S + j] += power[i * S + p] * mm[p * S + j];
            }
        memcpy(power, next, sizeof(power));
    }
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) {    /* X_37 = X_0 M^37, J = <W, X_37>: dJ/dX_0 = W (M^37)^T */
            want[i * S + j] = 0.0;
            for (int p = 0; p < S; ++p) want[i * S + j] += seed[i * S + p] * power[j * S + p];
        }
    verdict(r, ok && memcmp(gx[0], gx[1], sizeof(gx[0])) == 0 && memcmp(gm[0], gm[1], sizeof(gm[0])) == 0 && close_rel(gx[0], want, S * S, 1e-11),
            "a WHILE of cap 40 (windows of 16) with a product and a copy in its body, 37 passes: bit for bit the gradients of the same 37 pairs written out, dJ/dX_0 the closed form");
    eg_graph_free(ctx, loop);
    eg_graph_free(ctx, chain);
    eg_gemm_free(ctx, gemm);
    eg_plan_free(ctx, tk);
    release(r);
}

/* x <- a x by a launch and a copy per pass, in a WHILE of cap 40 (windows of 16) that runs 37 times, against the same 37 pairs written out: the
 * window rebuild re-runs the copy, and the gradients are the written-out sequence's bit for bit, dJ/dx_0 = a^37. */
static void window_copy_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    enum { PASSES = 37, CAP = 40 };
    double x0[N], x[N], t[N], a[1] = { 0.9 }, seed[N], gx[2][N], ga[2][1];
    int32_t c[N];
    struct tensor tx, tt, ta, ts, tgx[2], tga[2], tc;
    fill(x, a, 0.5);
    a[0] = 0.9;
    for (int i = 0; i < N; ++i) {
        x0[i] = x[i];
        seed[i] = 1.0;
        t[i] = 0.0;
    }
    eg_plane *px = mk(r, &tx, N, x), *pt = mk(r, &tt, N, t), *pa = mk(r, &ta, 1, a), *ps = mk(r, &ts, N, seed);
    eg_plane *pgx[2] = { mk(r, &tgx[0], N, gx[0]), mk(r, &tgx[1], N, gx[1]) }, *pga[2] = { mk(r, &tga[0], 1, ga[0]), mk(r, &tga[1], 1, ga[1]) };
    eg_plane* pc = mk_i32(r, &tc, N, c);
    eg_plane *a1[] = { px, pa, pt }, *a3[] = { pc };
    eg_plan *f = plan_with(r, u->scale, 3, a1, NULL), *tk = plan_with(r, u->tick, 1, a3, NULL);
    struct eg_node_vjp vjps[PASSES];
    for (int i = 0; i < PASSES; ++i) {
        vjps[i].launch = i;
        vjps[i].vjp = u->scale_vjp;
    }
    eg_graph *loop = NULL, *chain = NULL;
    int ok = f && tk && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, CAP) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK &&
             eg_copy(ctx, pt, px) == EG_OK && eg_launch(ctx, tk, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK && end_diff(ctx, 1, vjps, &loop) == EG_OK;
    ok = ok && eg_record_begin(ctx) == EG_OK;
    for (int k = 0; ok && k < PASSES; ++k) ok = eg_launch(ctx, f, NULL) == EG_OK && eg_copy(ctx, pt, px) == EG_OK;
    ok = ok && end_diff(ctx, PASSES, vjps, &chain) == EG_OK;
    eg_plane* in[] = { px, pa };
    for (int which = 0; ok && which < 2; ++which) {
        for (int i = 0; i < N; ++i) {
            x[i] = x0[i];
            c[i] = PASSES;
        }
        up(r);
        eg_plane* gr[] = { pgx[which], pga[which] };
        ok = eg_graph_launch(ctx, which ? chain : loop) == EG_OK && reverse(ctx, which ? chain : loop, px, ps, 2, in, gr) == EG_OK;
        down(r);
    }
    verdict(r, ok && memcmp(gx[0], gx[1], sizeof(gx[0])) == 0 && memcmp(ga[0], ga[1], sizeof(ga[0])) == 0 && fabs(gx[0][0] - pow(a[0], PASSES)) < 1e-12,
            "a WHILE of cap 40 (windows of 16) with a launch and a copy in its body, 37 passes: bit for bit the gradients of the same 37 pairs written out, dJ/dx_0 = a^37");
    eg_graph_free(ctx, loop);
    eg_graph_free(ctx, chain);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, tk);
    release(r);
}

/* A WHILE of cap 40 (windows of 16), 37 passes: the body ticks the int plane n down, launches t = n x and copies t into x. The vjp of the launch reads
 * n by a Value link, so n is loop state the window checkpoints keep: dJ/dx_0 is the product of the n each pass read, n_0 - k for pass k = 1 .. 37. */
static void window_int_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    enum { PASSES = 37, CAP = 40 };
    double x[N], t[N], seed[N], gx[N], want[N];
    int32_t n[N], c[N];
    struct tensor tx, tt, ts, tg, tn, tc;
    for (int i = 0; i < N; ++i) {
        x[i] = 1.0 + 0.01 * i;
        t[i] = gx[i] = 0.0;
        seed[i] = 1.0;
        want[i] = 1.0;
        for (int k = 1; k <= PASSES; ++k) want[i] *= (double)(60 + i % 3 - k);
    }
    eg_plane *px = mk(r, &tx, N, x), *pt = mk(r, &tt, N, t), *ps = mk(r, &ts, N, seed), *pg = mk(r, &tg, N, gx);
    eg_plane *pn = mk_typed(r, &tn, N, n, kDLInt, 32), *pc = mk_i32(r, &tc, N, c);
    eg_plane *a1[] = { px, pn, pt }, *a2[] = { pn }, *a3[] = { pc };
    eg_plan *f = plan_with(r, u->iscale, 3, a1, NULL), *tn_plan = plan_with(r, u->tick, 1, a2, NULL), *tk = plan_with(r, u->tick, 1, a3, NULL);
    const struct eg_node_vjp vjps[] = { { 1, u->iscale_vjp } };
    eg_graph* g = NULL;
    int ok = f && tn_plan && tk && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, CAP) == EG_OK && eg_launch(ctx, tn_plan, NULL) == EG_OK &&
             eg_launch(ctx, f, NULL) == EG_OK && eg_copy(ctx, pt, px) == EG_OK && eg_launch(ctx, tk, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK &&
             end_diff(ctx, 1, vjps, &g) == EG_OK;
    for (int i = 0; i < N; ++i) {
        n[i] = 60 + i % 3;
        c[i] = PASSES;
    }
    up(r);
    eg_plane *in[] = { px }, *gr[] = { pg };
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && reverse(ctx, g, px, ps, 1, in, gr) == EG_OK;
    down(r);
    verdict(r, ok && close_rel(gx, want, N, 1e-12),
            "a WHILE of cap 40 (windows of 16), 37 passes, whose body ticks an int plane n and launches t = n x: dJ/dx_0 = the product of the n each pass read");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, tn_plan);
    eg_plan_free(ctx, tk);
    release(r);
}

static void leaf_rows(struct rig* r, const struct units* u) {
    copy_row(r, u, 0);
    copy_row(r, u, 1);
    reduce_row(r, u);
    fft_row(r, EG_FFT_C2C, 0, EG_FFT_NORM_BACKWARD, 8, 1, "a C2C forward transform (backward norm, n = 8)");
    fft_row(r, EG_FFT_C2C, 1, EG_FFT_NORM_BACKWARD, 8, 0, "a C2C inverse transform (backward norm, 1/n)");
    fft_row(r, EG_FFT_C2C, 0, EG_FFT_NORM_FORWARD, 8, 0, "a C2C forward transform (forward norm, 1/n)");
    fft_row(r, EG_FFT_C2C, 0, EG_FFT_NORM_ORTHO, 7, 0, "a C2C forward transform (ortho norm, n = 7)");
    fft_row(r, EG_FFT_R2C, 0, EG_FFT_NORM_BACKWARD, 8, 1, "an R2C transform (n = 8, half spectrum of 5 bins)");
    fft_row(r, EG_FFT_R2C, 0, EG_FFT_NORM_ORTHO, 7, 0, "an R2C transform (ortho norm, n = 7)");
    fft_row(r, EG_FFT_C2R, 0, EG_FFT_NORM_BACKWARD, 8, 1, "a C2R transform (n = 8)");
    fft_row(r, EG_FFT_C2R, 0, EG_FFT_NORM_FORWARD, 7, 0, "a C2R transform (forward norm, n = 7)");
    gemm_row(r, 0, "");
    gemm_row(r, EG_GEMM_TRANS_A | EG_GEMM_TRANS_B | EG_GEMM_TRANS_C, " with every operand transposed");
    gemm_row(r, EG_GEMM_TRANS_A, " with A transposed");
    gemm_row(r, EG_GEMM_TRANS_B | EG_GEMM_TRANS_C, " with B and C transposed");
    gemm_row(r, EG_GEMM_ACCUMULATE | EG_GEMM_TRANS_A, " that accumulates, with A transposed");
    window_gemm_row(r, u);
    window_copy_row(r, u);
}


/* ---- an int plane a vjp reads by a VALUE link: kept as a primal, never given an adjoint ----------------------------------------------- */

/* y = n x with n an int32 per-sample plane the kernel and its vjp's Value link read: dJ/dx = n, the closed form. With @p overwritten an eg_copy after the
 * launch puts other values in n: the reverse still reads the n the launch read (the graph copied it before the overwrite). An int plane in inputs is refused. */
static void int_read_row(struct rig* r, const struct units* u, int overwritten) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], seed[N], gx[N], want[N];
    int32_t n[N], later[N];
    struct tensor tx, ty, ts, tg, tn, tl;
    for (int i = 0; i < N; ++i) {
        x[i] = 0.5 + 0.25 * i;
        seed[i] = 1.0;
        n[i] = 2 + i % 5;
        later[i] = 100 + i;
        want[i] = (double)n[i];
        gx[i] = y[i] = 0.0;
    }
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *ps = mk(r, &ts, N, seed), *pg = mk(r, &tg, N, gx);
    eg_plane *pn = mk_typed(r, &tn, N, n, kDLInt, 32), *pl = mk_typed(r, &tl, N, later, kDLInt, 32);
    eg_plane* a[] = { px, pn, py };
    eg_plan* f = plan_with(r, u->iscale, 3, a, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->iscale_vjp } };
    eg_graph *g = NULL, *none = NULL;
    up(r);
    int ok = f && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && (!overwritten || eg_copy(ctx, pl, pn) == EG_OK) && end_diff(ctx, 1, vjps, &g) == EG_OK;
    eg_plane *in[] = { px }, *gr[] = { pg }, *int_in[] = { pn };
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && reverse(ctx, g, py, ps, 1, in, gr) == EG_OK;
    down(r);
    verdict(r, ok && close_to(gx, want, N) && (!overwritten || n[N - 1] == later[N - 1]),
            overwritten ? "eg_graph_reverse of y = n x with n overwritten by a later eg_copy: dJ/dx = the n the launch read (the copy kept before the overwrite)"
                        : "eg_graph_reverse of y = n x with n an int32 plane read by the kernel and by its vjp's Value link: dJ/dx = n");
    if (!overwritten)
        verdict(r, ok && refused(reverse(ctx, g, py, ps, 1, int_in, gr), EG_BAD_REQUEST, ctx, "not a float plane"),
                "an int32 plane listed in inputs is still refused BAD_REQUEST by name");
    (void)none;
    if (g) eg_graph_free(ctx, g);
    if (f) eg_plan_free(ctx, f);
    release(r);
}

/* y = w x with the vjp whose gradient of the shared parameter w is a Kahan sum over samples kept by a companion plane eagle makes (scale_vjp_comp). Through a recorded
 * graph the gradient of w is bit for bit what the same vjp unit gives launched outside any graph (same data, same code), and the closed form sum x_i within rounding. */
static void companion_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], w[1] = { 1.7 }, y[N], seed[N], gx[N], gw[1] = { 0.0 }, ox[N], ow[1] = { 0.0 }, want_x[N], want_w[1] = { 0.0 };
    struct tensor tx, tw, ty, ts, tgx, tgw, tox, tow;
    for (int i = 0; i < N; ++i) {
        x[i] = 0.1 * (i + 1) + 1e-9 * i;
        seed[i] = 1.0 + 0.5 * (i % 3);
        want_x[i] = w[0] * seed[i];
        want_w[0] += x[i] * seed[i];
        y[i] = gx[i] = ox[i] = 0.0;
    }
    eg_plane *px = mk(r, &tx, N, x), *pw = mk(r, &tw, 1, w), *py = mk(r, &ty, N, y), *ps = mk(r, &ts, N, seed), *pgx = mk(r, &tgx, N, gx), *pgw = mk(r, &tgw, 1, gw);
    eg_plane *pox = mk(r, &tox, N, ox), *pow_ = mk(r, &tow, 1, ow);
    eg_plane* a[] = { px, pw, py };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    /* the same vjp unit launched outside any graph: x, w, bar_y, bar_x, bar_w (the companion is eagle's) */
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = u->scale_vjp_comp;
    pd.structure = r->device ? EG_STRUCTURE_DEVICE : EG_STRUCTURE_HOST_SERIAL;
    pd.policy = EG_REDUCE_COMPENSATED;
    pd.n_samples = N;
    eg_plan* vp = NULL;
    eg_plane* vplanes[] = { px, pw, ps, pox, pow_ };
    const int32_t vslots[] = { 0, 1, 2, 3, 4 };
    struct eg_bind_desc vb;
    memset(&vb, 0, sizeof(vb));
    vb.struct_size = sizeof(vb);
    vb.n_planes = 5;
    vb.plane_slots = vslots;
    vb.planes = vplanes;
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp_comp } };
    eg_graph* g = NULL;
    eg_plane *in[] = { px, pw }, *gr[] = { pgx, pgw };
    up(r);
    int ok = f && eg_plan_new(ctx, &pd, &vp) == EG_OK && eg_bind(ctx, vp, &vb) == EG_OK && eg_launch(ctx, vp, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
    ok = ok && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && end_diff(ctx, 1, vjps, &g) == EG_OK && eg_graph_launch(ctx, g) == EG_OK &&
         reverse(ctx, g, py, ps, 2, in, gr) == EG_OK;
    down(r);
    verdict(r, ok && memcmp(gw, ow, sizeof(gw)) == 0 && memcmp(gx, ox, sizeof(gx)) == 0,
            "a vjp with a companion plane (a Kahan-summed gradient of a shared parameter) through a recorded DIFF graph: bit for bit the same vjp launched outside any graph");
    verdict(r, ok && close_rel(gw, want_w, 1, 1e-12) && close_rel(gx, want_x, N, 1e-14), "the same gradients equal the closed form: dJ/dw = sum x_i s_i, dJ/dx_i = w s_i");
    if (g) eg_graph_free(ctx, g);
    if (vp) eg_plan_free(ctx, vp);
    if (f) eg_plan_free(ctx, f);
    release(r);
}

static void int_read_rows(struct rig* r, const struct units* u) {
    companion_row(r, u);
    int_read_row(r, u, 0);
    int_read_row(r, u, 1);
    window_int_row(r, u);
}

/* ---- slice 3c: runs of recorded runners in a recording --------------------------------------------------------------------- */

#define LN 37
#define LSTEPS 10
static const double LA = 0.5, LB = 0.25;

/* The runner fixture's step x <- a x + b, t <- t + 1, a sample finishing when t reaches its T, over LN samples, with its state planes. */
struct lin {
    double x[LN], x0[LN], t[LN], T[LN];
    uint8_t done[LN];
    struct tensor tx, tt, tT, td;
    eg_plane *px, *pt, *pT, *pd;
    eg_plan* plan;
    eg_runner* runner;
};

struct run_units {
    eg_unit *lin, *twin, *vjp;
};

static eg_unit* load_run_unit(struct rig* r, const char* name, uint8_t byte) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memset(ref.digest, byte, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = r->run_lib;
    d.unit = &ref;
    eg_unit* u = NULL;
    return eg_unit_load(r->ctx, &d, &u) == EG_OK ? u : NULL;
}

/* Put the samples back at the start of a batch: x at its first value, t at zero, none finished. */
static void lin_restart(struct lin* l, int stride_for_T) {
    for (int i = 0; i < LN; ++i) {
        l->x[i] = l->x0[i];
        l->t[i] = 0.0;
        l->done[i] = 0;
        l->T[i] = 1.0 + (double)((i * stride_for_T) % LSTEPS);
    }
}

/* Put x and t back as they were and leave the finished marks to the run: a replay of the graph clears them itself. */
static void lin_rewind(struct lin* l) {
    for (int i = 0; i < LN; ++i) {
        l->x[i] = l->x0[i];
        l->t[i] = 0.0;
    }
}

/* A lin batch with a runner: recorded (the twin and the step vjp derived, window 1) or plain. Its planes belong to the row's rig. */
static int lin_planes(struct rig* r, const struct run_units* ru, struct lin* l, int stride_for_T);
static int lin_runner(struct rig* r, const struct run_units* ru, struct lin* l, int recorded, eg_unit* twin, int64_t window, eg_graph* step);
static int lin_make(struct rig* r, const struct run_units* ru, struct lin* l, int stride_for_T, int recorded) {
    return lin_planes(r, ru, l, stride_for_T) && lin_runner(r, ru, l, recorded, ru->twin, 1, NULL);
}

/* The lin batch's planes and plan, with no runner yet. */
static int lin_planes(struct rig* r, const struct run_units* ru, struct lin* l, int stride_for_T) {
    for (int i = 0; i < LN; ++i) l->x0[i] = 1.0 + i / 8.0;
    lin_restart(l, stride_for_T);
    tensor_fill(&l->tx, LN, l->x, kDLFloat, 64);
    tensor_fill(&l->tt, LN, l->t, kDLFloat, 64);
    tensor_fill(&l->tT, LN, l->T, kDLFloat, 64);
    tensor_fill(&l->td, LN, l->done, kDLBool, 8);
    if (r->device) {    /* device planes beside their host arrays: the row copies up before a launch and down before it reads */
        l->px = mk(r, &l->tx, LN, l->x);
        l->pt = mk(r, &l->tt, LN, l->t);
        l->pT = mk(r, &l->tT, LN, l->T);
        l->pd = mk_typed(r, &l->td, LN, l->done, kDLBool, 8);
        if (l->px == NULL || l->pt == NULL || l->pT == NULL || l->pd == NULL) return 0;
    } else {
        struct tensor* ts[] = { &l->tx, &l->tt, &l->tT, &l->td };
        eg_plane** ps[] = { &l->px, &l->pt, &l->pT, &l->pd };
        for (int k = 0; k < 4; ++k) {
            ts[k]->managed.dl_tensor.device.device_type = kDLCPU;
            *ps[k] = NULL;
            if (eg_plane_wrap(r->ctx, &ts[k]->managed, ps[k]) != EG_OK) return 0;
            r->handle[r->n_planes] = *ps[k];
            r->host[r->n_planes++] = NULL;
        }
    }
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = ru->lin;
    pd.structure = r->device ? EG_STRUCTURE_DEVICE : EG_STRUCTURE_HOST_SERIAL;
    pd.policy = EG_REDUCE_COMPENSATED;
    pd.n_samples = LN;
    if (eg_plan_new(r->ctx, &pd, &l->plan) != EG_OK) return 0;
    const int32_t slots[] = { 0, 1, 2, 4 }, us[] = { 5, 6 };
    eg_plane* const planes[] = { l->px, l->pt, l->pd, l->pT };
    const void* const uv[] = { &LA, &LB };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 4;
    b.plane_slots = slots;
    b.planes = planes;
    b.n_uniforms = 2;
    b.uniform_slots = us;
    b.uniform_values = uv;
    return eg_bind(r->ctx, l->plan, &b) == EG_OK;
}

/* The runner over the lin plan: recorded with @p twin (its tape window @p window) or plain, with @p step as its step graph when given. */
static int lin_runner(struct rig* r, const struct run_units* ru, struct lin* l, int recorded, eg_unit* twin, int64_t window, eg_graph* step) {
    struct eg_runner_desc d = runner_desc(&l->plan, LSTEPS);
    const eg_derived_role roles[2] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP };
    eg_unit* const units[2] = { twin, ru->vjp };
    if (recorded) {
        d.flags = EG_RUN_RECORD;
        d.n_derived = 2;
        d.derived_roles = roles;
        d.derived_units = units;
        d.window = window;
    }
    d.step_graph = step;
    return eg_runner_new(r->ctx, &d, &l->runner) == EG_OK;
}

static void lin_free(struct rig* r, struct lin* l) {
    eg_runner_free(r->ctx, l->runner);
    eg_plan_free(r->ctx, l->plan);
}

/* After k steps from x0: x_k and dx_k/dx_0 = a^k. */
static void lin_closed(double x0, int steps, double* xk, double* dx0) {
    double x = x0, p = 1.0;
    for (int j = 0; j < steps; ++j) {
        x = LA * x + LB;
        p *= LA;
    }
    *xk = x;
    *dx0 = p;
}
static int lin_steps(int i, int stride_for_T) { return 1 + (i * stride_for_T) % LSTEPS; }

/* y = w x_k after a recorded run of x: the gradients of J = sum y through the run (eg_reverse of its runner) and the launch's vjp against the
 * composed closed form (dJ/dx_0 = w a^k per sample, dJ/dw = sum x_k); then a second launch of the graph from the start gives the same gradients
 * (each replay restarts the run), and the same recording ended plain also runs the run again at its second launch. */
static void run_launch_row(struct rig* r, const struct units* u, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    struct lin l;
    double y[LN], w[1] = { 1.7 }, seed[LN], gx[2][LN], gw[2][1], want_gx[LN], want_y[LN], want_xk[LN], want_gw = 0.0;
    struct tensor ty, tw, tseed, tgx[2], tgw[2];
    for (int i = 0; i < LN; ++i) seed[i] = 1.0;
    memset(y, 0, sizeof(y));
    r->n_samples = LN;
    int ok = lin_make(r, ru, &l, 7, 1);
    eg_plane *py = mk(r, &ty, LN, y), *pw = mk(r, &tw, 1, w), *ps = mk(r, &tseed, LN, seed);
    eg_plane *pgx[2] = { mk(r, &tgx[0], LN, gx[0]), mk(r, &tgx[1], LN, gx[1]) }, *pgw[2] = { mk(r, &tgw[0], 1, gw[0]), mk(r, &tgw[1], 1, gw[1]) };
    eg_plane* a[] = { l.px, pw, py };
    eg_plan* f = ok ? plan_with(r, u->scale, 3, a, NULL) : NULL;
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    eg_graph* g = NULL;
    ok = ok && f && eg_record_begin(ctx) == EG_OK && eg_run(ctx, l.runner, NULL) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && end_diff(ctx, 1, vjps, &g) == EG_OK;
    for (int i = 0; i < LN; ++i) {
        double xk, dx;
        lin_closed(l.x0[i], lin_steps(i, 7), &xk, &dx);
        want_gx[i] = w[0] * dx;
        want_y[i] = w[0] * xk;
        want_xk[i] = xk;
        want_gw += xk;
    }
    eg_plane *in[] = { l.px, pw }, *gr0[] = { pgx[0], pgw[0] }, *gr1[] = { pgx[1], pgw[1] };
    up(r);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK;
    down(r);
    ok = ok && close_to(y, want_y, LN) && reverse(ctx, g, py, ps, 2, in, gr0) == EG_OK;
    down(r);
    verdict(r, ok && close_rel(gx[0], want_gx, LN, 1e-12) && close_rel(gw[0], &want_gw, 1, 1e-12),
            "eg_graph_reverse through a recorded run then a launch (y = w x_k): dJ/dx_0 = w a^k per sample and dJ/dw = sum x_k, the composed closed form");
    lin_rewind(&l);
    up(r);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK;
    down(r);
    ok = ok && close_to(y, want_y, LN) && reverse(ctx, g, py, ps, 2, in, gr1) == EG_OK;
    down(r);
    verdict(r, ok && memcmp(gx[0], gx[1], sizeof(gx[0])) == 0 && memcmp(gw[0], gw[1], sizeof(gw[0])) == 0,
            "a second eg_graph_launch restarts the run from the start of its batch: the second eg_graph_reverse gives the first one's gradients, bit for bit");
    eg_graph_free(ctx, g);
    eg_graph* plain = NULL;
    const double xtol = r->device ? 1e-12 : 0.0;    /* the host's run is the closed form bit for bit; the device's step may fuse its multiply-add */
    lin_restart(&l, 7);
    up(r);
    ok = ok && l.runner && eg_record_begin(ctx) == EG_OK && eg_run(ctx, l.runner, NULL) == EG_OK && eg_record_end(ctx, NULL, &plain) == EG_OK &&
         eg_graph_launch(ctx, plain) == EG_OK;
    down(r);
    int ran_twice = ok;
    for (int i = 0; ran_twice && i < LN; ++i) ran_twice = fabs(l.x[i] - want_xk[i]) <= xtol * (1.0 + fabs(want_xk[i]));
    lin_rewind(&l);
    up(r);
    ran_twice = ran_twice && eg_graph_launch(ctx, plain) == EG_OK;
    down(r);
    for (int i = 0; ran_twice && i < LN; ++i) ran_twice = fabs(l.x[i] - want_xk[i]) <= xtol * (1.0 + fabs(want_xk[i])) && l.done[i];
    verdict(r, ran_twice, "a plain graph with a recorded run runs the run again at its second launch (a replay is a restart)");
    eg_graph_free(ctx, plain);
    if (f) eg_plan_free(ctx, f);
    lin_free(r, &l);
    release(r);
    r->n_samples = 0;
}

/* Two runs of two runners in one graph with a launch between: x1 -run-> scale -> x2 -run-> J = sum x2, dJ/dx1_i against central differences of J through the graph. */
static void phase_row(struct rig* r, const struct units* u, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    struct lin a, b;
    double w[1] = { 1.3 }, seed[LN], gx[LN], gw[1], want_fd[LN], fd_w = 0.0;
    struct tensor tw, tseed, tgx, tgw;
    for (int i = 0; i < LN; ++i) seed[i] = 1.0;
    r->n_samples = LN;
    int ok = lin_make(r, ru, &a, 7, 1) && lin_make(r, ru, &b, 3, 1);
    eg_plane *pw = mk(r, &tw, 1, w), *ps = mk(r, &tseed, LN, seed), *pgx = mk(r, &tgx, LN, gx), *pgw = mk(r, &tgw, 1, gw);
    eg_plane* sa[] = { a.px, pw, b.px };
    eg_plan* f = ok ? plan_with(r, u->scale, 3, sa, NULL) : NULL;
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    eg_graph* g = NULL;
    ok = ok && f && eg_record_begin(ctx) == EG_OK && eg_run(ctx, a.runner, NULL) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK &&
         eg_run(ctx, b.runner, NULL) == EG_OK && end_diff(ctx, 1, vjps, &g) == EG_OK;
    eg_plane *in[] = { a.px, pw }, *gr[] = { pgx, pgw };
    up(r);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && reverse(ctx, g, b.px, ps, 2, in, gr) == EG_OK;
    down(r);
    /* J(x1_i, w) through the whole graph, from the start of both batches */
    const double h = 0.5;
    for (int pass = 0; ok && pass <= LN; ++pass) {         /* passes 0 .. LN - 1: sample pass of x1; pass LN: w */
        double jp[2] = { 0.0, 0.0 };
        for (int side = 0; ok && side < 2; ++side) {
            lin_rewind(&a);
            lin_rewind(&b);
            double wv = w[0];
            if (pass < LN) a.x[pass] += side == 0 ? h : -h;
            else wv += side == 0 ? 0.25 : -0.25;
            w[0] = wv;
            up(r);
            ok = eg_graph_launch(ctx, g) == EG_OK;
            down(r);
            for (int i = 0; i < LN; ++i) jp[side] += b.x[i];
            w[0] = 1.3;
        }
        if (pass < LN) want_fd[pass] = (jp[0] - jp[1]) / (2.0 * h);
        else fd_w = (jp[0] - jp[1]) / 0.5;
    }
    verdict(r, ok && close_rel(gx, want_fd, LN, 1e-9) && close_rel(gw, &fd_w, 1, 1e-9),
            "eg_graph_reverse through run, launch, run (two recorded runners): dJ/dx_0 and dJ/dw against central differences of the whole graph");
    eg_graph_free(ctx, g);
    if (f) eg_plan_free(ctx, f);
    lin_free(r, &a);
    lin_free(r, &b);
    release(r);
    r->n_samples = 0;
}

/* A run in an IF arm: when the arm ran the gradients are the run's, when it did not they pass through the launch alone (dJ/dx = w). */
static void run_if_row(struct rig* r, const struct units* u, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    for (int taken = 0; taken < 2; ++taken) {
        struct lin l;
        double y[LN], w[1] = { 1.7 }, seed[LN], gx[LN], want[LN];
        int32_t flag[1] = { taken };
        struct tensor ty, tw, tseed, tgx, tf;
        for (int i = 0; i < LN; ++i) seed[i] = 1.0;
        r->n_samples = LN;
        int ok = lin_make(r, ru, &l, 7, 1);
        eg_plane *py = mk(r, &ty, LN, y), *pw = mk(r, &tw, 1, w), *ps = mk(r, &tseed, LN, seed), *pgx = mk(r, &tgx, LN, gx), *pf = mk_i32(r, &tf, 1, flag);
        eg_plane* a[] = { l.px, pw, py };
        eg_plan* f = ok ? plan_with(r, u->scale, 3, a, NULL) : NULL;
        const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
        eg_graph* g = NULL;
        ok = ok && f && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_IF, 0, pf, 0, 0) == EG_OK && eg_run(ctx, l.runner, NULL) == EG_OK &&
             eg_record_cond_end(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && end_diff(ctx, 1, vjps, &g) == EG_OK;
        for (int i = 0; i < LN; ++i) {
            double xk, dx;
            lin_closed(l.x0[i], lin_steps(i, 7), &xk, &dx);
            want[i] = w[0] * (taken ? dx : 1.0);
        }
        eg_plane *in[] = { l.px }, *gr[] = { pgx };
        ok = ok && eg_graph_launch(ctx, g) == EG_OK && reverse(ctx, g, py, ps, 1, in, gr) == EG_OK;
        verdict(r, ok && close_rel(gx, want, LN, 1e-12),
                taken ? "a recorded run in an IF arm that ran: dJ/dx_0 = w a^k, the run's reverse"
                      : "a recorded run in an IF arm that did not run: its read planes' adjoints pass through, dJ/dx = w");
        eg_graph_free(ctx, g);
        if (f) eg_plan_free(ctx, f);
        lin_free(r, &l);
        release(r);
        r->n_samples = 0;
    }
}

/* The refusals of a run leaf, each by name; every one leaves the recording open. */
static void run_refusal_row(struct rig* r, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    struct lin plain, rec;
    r->n_samples = LN;
    int ok = lin_make(r, ru, &plain, 7, 0) && lin_make(r, ru, &rec, 7, 1);
    eg_graph *none = NULL, *open = NULL;
    ok = ok && eg_record_begin(ctx) == EG_OK && eg_run(ctx, plain.runner, NULL) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 0, NULL, &none), EG_NOT_WRITTEN, ctx, "needs EG_RUN_RECORD to be reversed") && eg_record_end(ctx, NULL, &open) == EG_OK,
            "eg_record_end with EG_RECORD_DIFF over the run of a runner made without EG_RUN_RECORD: refused NOT_WRITTEN by name");
    eg_graph_free(ctx, open);
    ok = ok && eg_record_begin(ctx) == EG_OK && eg_run(ctx, rec.runner, NULL) == EG_OK && eg_run(ctx, rec.runner, NULL) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 0, NULL, &none), EG_BAD_REQUEST, ctx, "record the second run with a second runner") && eg_record_end(ctx, NULL, &open) == EG_OK,
            "eg_record_end with EG_RECORD_DIFF over one runner run twice: refused BAD_REQUEST, naming the way out");
    eg_graph_free(ctx, open);
    lin_free(r, &plain);
    lin_free(r, &rec);
    release(r);
    r->n_samples = 0;
}

/* A host recording keeps a run inside a WHILE and refuses it at the end of a DIFF recording. */
static void run_while_row(struct rig* r, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    struct lin rec;
    int32_t cell[1] = { 1 };
    struct tensor tc;
    r->n_samples = LN;
    int ok = lin_make(r, ru, &rec, 7, 1);
    eg_plane* pc = mk_i32(r, &tc, 1, cell);
    eg_graph *none = NULL, *open = NULL;
    ok = ok && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, 3) == EG_OK && eg_run(ctx, rec.runner, NULL) == EG_OK &&
         eg_record_cond_end(ctx) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 0, NULL, &none), EG_NOT_WRITTEN, ctx, "a phase count must be static") && eg_record_end(ctx, NULL, &open) == EG_OK,
            "eg_record_end with EG_RECORD_DIFF over a run inside a WHILE: refused NOT_WRITTEN with the static-count text");
    eg_graph_free(ctx, open);
    lin_free(r, &rec);
    release(r);
    r->n_samples = 0;
}

/* A recorded runner of host planes in a device recording is a host node of the capture, which the device tape does not cover: refused by name, the recording left open. */
static void device_run_refusal_row(struct rig* r, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    struct lin l;
    struct rig on_host = *r;    /* the runner of host planes on the device context */
    on_host.device = 0;
    on_host.n_planes = 0;
    r->n_samples = LN;
    on_host.n_samples = LN;
    eg_graph *none = NULL, *open = NULL;
    int ok = lin_make(&on_host, ru, &l, 7, 1) && eg_record_begin(ctx) == EG_OK && eg_run(ctx, l.runner, NULL) == EG_OK;
    verdict(r, ok && refused(end_diff(ctx, 0, NULL, &none), EG_NOT_WRITTEN, ctx, "runs on the host in a device recording") && eg_record_end(ctx, NULL, &open) == EG_OK,
            "eg_record_end with EG_RECORD_DIFF over the run of a host runner in a device recording: refused NOT_WRITTEN by name");
    eg_graph_free(ctx, open);
    lin_free(&on_host, &l);
    release(&on_host);
    r->n_samples = 0;
}

/* A run of a DEVICE runner is a step of the graph between two captures, so it cannot be a node inside a conditional arm, a fork branch or a WHILE:
 * each is refused NOT_WRITTEN by name at the call, the recording left open and usable (every arm still needs a launch of its own). */
static void device_run_scope_row(struct rig* r, const struct units* u, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    struct lin l;
    double x[LN], w[1] = { 1.7 }, y1[LN], y2[LN];
    int32_t flag[1] = { 1 };
    struct tensor tx, tw, ty1, ty2, tf;
    for (int i = 0; i < LN; ++i) x[i] = 1.0 + i;
    r->n_samples = LN;
    int ok = lin_make(r, ru, &l, 7, 1);
    eg_plane *px = mk(r, &tx, LN, x), *pw = mk(r, &tw, 1, w), *py1 = mk(r, &ty1, LN, y1), *py2 = mk(r, &ty2, LN, y2), *pf = mk_i32(r, &tf, 1, flag);
    eg_plane *s1[] = { px, pw, py1 }, *s2[] = { px, pw, py2 };
    eg_plan *f1 = ok ? plan_with(r, u->scale, 3, s1, NULL) : NULL, *f2 = ok ? plan_with(r, u->scale, 3, s2, NULL) : NULL;
    eg_graph* open = NULL;
    static const char scope[] = "inside a conditional arm, a fork branch or a WHILE";
    ok = ok && f1 && f2 && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_IF, 0, pf, 0, 0) == EG_OK;
    verdict(r, ok && refused(eg_run(ctx, l.runner, NULL), EG_NOT_WRITTEN, ctx, scope) && eg_launch(ctx, f1, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK &&
                   eg_record_end(ctx, NULL, &open) == EG_OK,
            "the run of a device runner inside an IF arm: refused NOT_WRITTEN by name, the recording left open");
    eg_graph_free(ctx, open);
    ok = ok && eg_record_begin(ctx) == EG_OK && eg_record_fork_begin(ctx) == EG_OK;
    verdict(r, ok && refused(eg_run(ctx, l.runner, NULL), EG_NOT_WRITTEN, ctx, scope) && eg_launch(ctx, f1, NULL) == EG_OK && eg_record_fork_branch(ctx) == EG_OK &&
                   eg_launch(ctx, f2, NULL) == EG_OK && eg_record_fork_end(ctx) == EG_OK && eg_record_end(ctx, NULL, &open) == EG_OK,
            "the run of a device runner inside a fork branch: refused NOT_WRITTEN by name, the recording left open");
    eg_graph_free(ctx, open);
    ok = ok && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pf, 0, 3) == EG_OK;
    verdict(r, ok && refused(eg_run(ctx, l.runner, NULL), EG_NOT_WRITTEN, ctx, scope) && eg_launch(ctx, f1, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK &&
                   eg_record_end(ctx, NULL, &open) == EG_OK,
            "the run of a device runner inside a WHILE: refused NOT_WRITTEN by name, the recording left open");
    eg_graph_free(ctx, open);
    if (f1) eg_plan_free(ctx, f1);
    if (f2) eg_plan_free(ctx, f2);
    lin_free(r, &l);
    release(r);
    r->n_samples = 0;
}

/* Run a, launch y1 = w a, run b, launch y2 = w a: a graph of three captures and two runs. Its description is the recording order on two levels (the
 * launches wait for run a, run b is free), and a plane rebound in the plans of both launches is re-applied in each capture. */
static void device_run_segments_row(struct rig* r, const struct units* u, const struct run_units* ru) {
    eg_ctx* ctx = r->ctx;
    struct lin a, b;
    double w[1] = { 1.7 }, x2[LN], y1[LN], y2[LN];
    struct tensor tw, tx2, ty1, ty2;
    r->n_samples = LN;
    int ok = lin_make(r, ru, &a, 7, 0) && lin_make(r, ru, &b, 3, 0);
    for (int i = 0; i < LN; ++i) x2[i] = 100.0 + i;
    eg_plane *pw = mk(r, &tw, 1, w), *px2 = mk(r, &tx2, LN, x2), *py1 = mk(r, &ty1, LN, y1), *py2 = mk(r, &ty2, LN, y2);
    eg_plane *s1[] = { a.px, pw, py1 }, *s2[] = { a.px, pw, py2 };
    eg_plan *f1 = ok ? plan_with(r, u->scale, 3, s1, NULL) : NULL, *f2 = ok ? plan_with(r, u->scale, 3, s2, NULL) : NULL;
    eg_graph* g = NULL;
    ok = ok && f1 && f2 && eg_record_begin(ctx) == EG_OK && eg_run(ctx, a.runner, NULL) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK &&
         eg_run(ctx, b.runner, NULL) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK && eg_record_end(ctx, NULL, &g) == EG_OK;
    struct eg_graph_description desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    const eg_gnode_kind kinds[] = { EG_GNODE_RUN, EG_GNODE_LAUNCH, EG_GNODE_RUN, EG_GNODE_LAUNCH };
    int described = ok && eg_graph_describe(ctx, g, &desc) == EG_OK && desc.n_nodes == 4 && desc.n_levels == 2;
    for (int i = 0; described && i < 4; ++i) {
        const struct eg_graph_node* n = (const struct eg_graph_node*)((const char*)desc.nodes + (size_t)i * desc.node_size);
        described = n->kind == kinds[i] && n->parent == -1;
    }
    verdict(r, described, "eg_graph_describe of a device graph with two runs between launches: the four nodes in recording order (run, launch, run, launch), as for one capture");
    up(r);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK;
    down(r);
    double want1[LN], want2[LN];
    for (int i = 0; i < LN; ++i) {
        double xk, dx;
        lin_closed(a.x0[i], lin_steps(i, 7), &xk, &dx);
        want1[i] = w[0] * xk;
        want2[i] = w[0] * xk;
    }
    verdict(r, ok && close_rel(y1, want1, LN, 1e-12) && close_rel(y2, want2, LN, 1e-12),
            "a device graph of run, launch, run, launch replays in order: both launches read the state the first run left");
    struct eg_bind_desc rebind;
    memset(&rebind, 0, sizeof(rebind));
    rebind.struct_size = sizeof(rebind);
    const int32_t slot_x = 0;
    rebind.n_planes = 1;
    rebind.plane_slots = &slot_x;
    rebind.planes = &px2;
    ok = ok && eg_bind(ctx, f1, &rebind) == EG_OK && eg_bind(ctx, f2, &rebind) == EG_OK;
    lin_rewind(&a);
    lin_rewind(&b);
    up(r);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK;
    down(r);
    double wantx2[LN];
    for (int i = 0; i < LN; ++i) wantx2[i] = w[0] * x2[i];
    verdict(r, ok && close_rel(y1, wantx2, LN, 1e-12) && close_rel(y2, wantx2, LN, 1e-12),
            "a plane rebound in the plans of two captures of one device graph: the next launch reads the new plane in both");
    eg_graph_free(ctx, g);
    if (f1) eg_plan_free(ctx, f1);
    if (f2) eg_plan_free(ctx, f2);
    lin_free(r, &a);
    lin_free(r, &b);
    release(r);
    r->n_samples = 0;
}

/* ---- a runner's step graph in a recorded run, reversed per step ------------------------------------------------------------------------------ */

/* The whole run as the model: per global step the graph (x <- w x, or with @p mixing x <- (sum x) x: a graph that mixes samples), then the lin step
 * x <- a x + b for the samples still live (sample i takes T_i steps), the graph running over every sample at every step, stopped or not. */
static void step_model(const double* x0, int mixing, double w, double a, double b, double* x) {
    const int steps = LSTEPS;
    for (int i = 0; i < LN; ++i) x[i] = x0[i];
    for (int j = 0; j < steps; ++j) {
        double s = 0.0;
        for (int i = 0; i < LN; ++i) s += (1.0 / 80.0) * x[i];
        for (int i = 0; i < LN; ++i) {
            x[i] = (mixing ? s : w) * x[i];
            if (j < lin_steps(i, 7)) x[i] = a * x[i] + b;
        }
    }
}
static double step_model_j(const double* x0, int mixing, double w, double a, double b) {
    double x[LN], j = 0.0;
    step_model(x0, mixing, w, a, b, x);
    for (int i = 0; i < LN; ++i) j += x[i];
    return j;
}

/* One recorded run with a step graph (a launch y = w x then y copied into x; with @p mixing a sum of x into w first), its reverse, and the gradients against
 * central differences of the model, for the checkpoint window of @p twin; the gradients are returned for the cross-window comparison. */
static int step_case(struct rig* r, const struct units* u, const struct run_units* ru, eg_unit* twin, int64_t window, int mixing, uint64_t budget, double* gx_out, double* ga_out,
                     double* gb_out, int32_t* passes, const char* what) {
    eg_ctx* ctx = r->ctx;
    struct lin l;
    double y[LN], z[LN], w[1] = { 1.3 }, wc[1] = { 1.0 / 80.0 }, seed[LN], gx[LN], gt[LN], ga = 0.0, gb = 0.0;
    struct tensor ty, tw, tseed, tgx, tgt, tga, tgb, tz, twc;
    for (int i = 0; i < LN; ++i) seed[i] = 1.0, y[i] = 0.0;
    r->n_samples = LN;
    int ok = lin_planes(r, ru, &l, 7);
    for (int i = 0; i < LN; ++i) l.x0[i] = l.x[i] = 1.0 + i / 8.0;
    eg_plane *py = mk(r, &ty, LN, y), *pw = mk(r, &tw, 1, w), *ps = mk(r, &tseed, LN, seed);
    eg_plane *pgx = mk(r, &tgx, LN, gx), *pgt = mk(r, &tgt, LN, gt), *pga = mk(r, &tga, 1, &ga), *pgb = mk(r, &tgb, 1, &gb);
    eg_plane* a3[] = { l.px, pw, py };
    eg_plan* f = ok ? plan_with(r, u->scale, 3, a3, NULL) : NULL;
    /* a graph that mixes samples: z = x / 80, w = sum z, then y = w x */
    eg_plane *pz = mixing ? mk(r, &tz, LN, z) : NULL, *pwc = mixing ? mk(r, &twc, 1, wc) : NULL;
    eg_plane* a2[] = { l.px, pwc, pz };
    eg_plan* f2 = mixing && ok ? plan_with(r, u->scale, 3, a2, NULL) : NULL;
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->scale_vjp } };
    const struct eg_node_vjp vjps_plain[] = { { 0, u->scale_vjp } };
    eg_graph* g = NULL;
    ok = ok && f && (!mixing || f2) && eg_record_begin(ctx) == EG_OK && (!mixing || (eg_launch(ctx, f2, NULL) == EG_OK && reduce_op(ctx, pz, pw, EG_REDUCE_SUM) == EG_OK)) &&
         eg_launch(ctx, f, NULL) == EG_OK && eg_copy(ctx, py, l.px) == EG_OK && end_diff(ctx, 1 + (mixing != 0), mixing ? vjps : vjps_plain, &g) == EG_OK &&
         lin_runner(r, ru, &l, 1, twin, window, g);
    struct eg_runner_desc d;    /* the budget is a field of the descriptor: remake the runner when one is asked for */
    if (ok && budget != 0) {
        eg_runner_free(ctx, l.runner);
        d = runner_desc(&l.plan, LSTEPS);
        const eg_derived_role roles[2] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP };
        eg_unit* const units[2] = { twin, ru->vjp };
        d.flags = EG_RUN_RECORD, d.n_derived = 2, d.derived_roles = roles, d.derived_units = units, d.window = window, d.memory_budget = budget, d.step_graph = g;
        ok = eg_runner_new(ctx, &d, &l.runner) == EG_OK;
    }
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    up(r);
    ok = ok && eg_run(ctx, l.runner, &rep) == EG_OK;
    down(r);
    double want_x[LN];
    step_model(l.x0, mixing, w[0], LA, LB, want_x);
    for (int i = 0; ok && i < LN; ++i) ok = fabs(l.x[i] - want_x[i]) <= 1e-11 * (1.0 + fabs(want_x[i]));
    const int32_t sslot[] = { 0 }, gslot[] = { 0, 1, 5, 6 };
    eg_plane* const sp[] = { ps };
    eg_plane* const gp[] = { pgx, pgt, pga, pgb };
    struct eg_bind_desc sd, gd;
    memset(&sd, 0, sizeof(sd)), memset(&gd, 0, sizeof(gd));
    sd.struct_size = gd.struct_size = sizeof(sd);
    sd.n_planes = 1, sd.plane_slots = sslot, sd.planes = sp;
    gd.n_planes = 4, gd.plane_slots = gslot, gd.planes = gp;
    ok = ok && eg_reverse(ctx, l.runner, &sd, &gd) == EG_OK;
    down(r);
    struct eg_runner_figures fg;
    memset(&fg, 0, sizeof(fg));
    fg.struct_size = sizeof(fg);
    ok = ok && eg_runner_figures(ctx, l.runner, &fg) == EG_OK;
    *passes = fg.reverse_passes;
    int fd_ok = ok;    /* central differences of the model: dJ/dx0 of every fourth sample, dJ/da and dJ/db */
    const double h = 1e-6;
    for (int i = 0; fd_ok && i < LN; i += 4) {
        double xp[LN], xm[LN];
        memcpy(xp, l.x0, sizeof(xp)), memcpy(xm, l.x0, sizeof(xm));
        xp[i] += h * (1.0 + fabs(xp[i])), xm[i] -= h * (1.0 + fabs(xm[i]));
        const double fd = (step_model_j(xp, mixing, w[0], LA, LB) - step_model_j(xm, mixing, w[0], LA, LB)) / (xp[i] - xm[i]);
        fd_ok = fabs(gx[i] - fd) <= 1e-6 * (1.0 + fabs(fd));
        if (!fd_ok) printf("      dJ/dx0[%d] %.10g (central %.10g)\n", i, gx[i], fd);
    }
    const double fda = (step_model_j(l.x0, mixing, w[0], LA * (1 + h), LB) - step_model_j(l.x0, mixing, w[0], LA * (1 - h), LB)) / (2.0 * h * LA);
    const double fdb = (step_model_j(l.x0, mixing, w[0], LA, LB * (1 + h)) - step_model_j(l.x0, mixing, w[0], LA, LB * (1 - h))) / (2.0 * h * LB);
    fd_ok = fd_ok && fabs(ga - fda) <= 1e-6 * (1.0 + fabs(fda)) && fabs(gb - fdb) <= 1e-6 * (1.0 + fabs(fdb));
    if (!fd_ok) printf("      dJ/da %.10g (central %.10g), dJ/db %.10g (central %.10g)\n", ga, fda, gb, fdb);
    verdict(r, fd_ok, what);
    memcpy(gx_out, gx, sizeof(gx)), *ga_out = ga, *gb_out = gb;
    eg_runner_free(ctx, l.runner);
    if (g) eg_graph_free(ctx, g);
    if (f) eg_plan_free(ctx, f);
    if (f2) eg_plan_free(ctx, f2);
    eg_plan_free(ctx, l.plan);
    release(r);
    r->n_samples = 0;
    return fd_ok;
}

static void step_reverse_rows(struct rig* r, const struct units* u, const struct run_units* ru) {
    eg_unit* twin3 = r->device ? NULL : load_run_unit(r, "lin", 0xE3);    /* the three-row twin has no device image: a device run keeps one-step windows */
    eg_unit* const wide = twin3 != NULL ? twin3 : ru->twin;
    const int64_t wide_window = twin3 != NULL ? 3 : 1;
    verdict(r, r->device || twin3 != NULL, "the three-row taping twin loads");
    if (!r->device && twin3 == NULL) return;
    double g1[LN], g3[LN], gb3[LN], a1, b1, a3, b3, ab, bb;
    int32_t p1, p3, pb;
    step_case(r, u, ru, ru->twin, 1, 0, 0, g1, &a1, &b1, &p1,
              "a recorded run with a step graph (a launch scaling the state, its output copied back) over stopping samples: eg_reverse's dJ/dx0, dJ/da and dJ/db equal central differences of the whole run, window 1");
    if (twin3 != NULL) {
        step_case(r, u, ru, twin3, 3, 0, 0, g3, &a3, &b3, &p3, "the same run with checkpoint windows of 3 steps: the gradients equal central differences too");
        verdict(r, memcmp(g1, g3, sizeof(g1)) == 0 && memcmp(&a1, &a3, sizeof(a1)) == 0 && memcmp(&b1, &b3, sizeof(b1)) == 0,
                "the step graph's gradients are bitwise the same for two checkpoint window lengths");
    }
    step_case(r, u, ru, wide, wide_window, 1, 0, gb3, &ab, &bb, &pb,
              "a step graph that mixes samples (a sum of the state scales it): the gradients equal central differences of the whole run");
    step_case(r, u, ru, wide, wide_window, 0, 300, g3, &a3, &b3, &p3,
              "a memory budget that would split the samples into passes: a step graph's reverse is one pass over all samples, with the same gradients");
    verdict(r, p3 == 1 && memcmp(g1, g3, sizeof(g1)) == 0 && memcmp(&a1, &a3, sizeof(a1)) == 0, "the budgeted run's gradients are the unbudgeted ones, bit for bit, in one pass");
    if (twin3 != NULL) eg_unit_free(r->ctx, twin3);
}

static void run_rows(struct rig* r, const struct units* u) {
    struct run_units ru = { load_run_unit(r, "lin", 0xE0), load_run_unit(r, "lin", 0xE1), load_run_unit(r, "lin_vjp", 0xE4) };
    verdict(r, ru.lin && ru.twin && ru.vjp, "the runner fixture's step, taping twin and step vjp load");
    if (!(ru.lin && ru.twin && ru.vjp)) return;
    if (r->device) {
        device_run_refusal_row(r, &ru);
        device_run_scope_row(r, u, &ru);
        device_run_segments_row(r, u, &ru);
    } else {
        run_if_row(r, u, &ru);
        run_while_row(r, &ru);
    }
    run_launch_row(r, u, &ru);
    phase_row(r, u, &ru);
    run_refusal_row(r, &ru);
    step_reverse_rows(r, u, &ru);
}

/* ---- the policy row: the reduced gradient of a shared parameter is the same bits on the host and on the device, run to run ---------- */

enum { M = 100000 };

/* dJ/dw of y = w x over M samples with a seed of ones: the reduced GRAD of w. Returns the sum, or NaN when a call failed. */
static double reduced_grad(struct rig* r, const struct units* u, const double* x0) {
    eg_ctx* ctx = r->ctx;
    double *x = malloc(M * sizeof(double)), *y = malloc(M * sizeof(double)), *one = malloc(M * sizeof(double)), *gx = malloc(M * sizeof(double));
    double w[1] = { 1.3 }, gw[1] = { 0.0 };
    struct tensor tx, ty, t1, tgx, tw, tgw;
    for (int i = 0; i < M; ++i) {
        x[i] = x0[i];
        one[i] = 1.0;
    }
    r->n_samples = M;
    eg_plane *px = mk(r, &tx, M, x), *py = mk(r, &ty, M, y), *p1 = mk(r, &t1, M, one), *pgx = mk(r, &tgx, M, gx), *pw = mk(r, &tw, 1, w), *pgw = mk(r, &tgw, 1, gw);
    eg_plane* a1[] = { px, pw, py };
    eg_plan* f = plan_with(r, u->scale, 3, a1, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    eg_graph* g = NULL;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    up(r);
    int ok = f && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && end_diff(ctx, 1, vjps, &g) == EG_OK && eg_graph_launch(ctx, g) == EG_OK &&
             reverse(ctx, g, py, p1, 2, in, gr) == EG_OK;
    down(r);
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, f);
    release(r);
    r->n_samples = 0;
    free(x), free(y), free(one), free(gx);
    return ok ? gw[0] : NAN;
}

static void policy_row(struct rig* host, const struct units* uh, struct rig* dev, const struct units* ud) {
    double* x = malloc(M * sizeof(double));
    uint64_t s = 88172645463325252ull;    /* varied magnitudes and signs: an order of summation that differs shows in the bits */
    for (int i = 0; i < M; ++i) {
        s ^= s << 13, s ^= s >> 7, s ^= s << 17;
        x[i] = ((double)(s >> 11) / 9007199254740992.0 - 0.5) * pow(10.0, (double)(int)(s % 7) - 3.0);
    }
    const double h = reduced_grad(host, uh, x), d0 = reduced_grad(dev, ud, x), d1 = reduced_grad(dev, ud, x), d2 = reduced_grad(dev, ud, x);
    const int same_host = memcmp(&h, &d0, sizeof(h)) == 0, same_runs = memcmp(&d0, &d1, sizeof(h)) == 0 && memcmp(&d0, &d2, sizeof(h)) == 0;
    if (!same_host || !same_runs) printf("      host %.17g, device %.17g %.17g %.17g\n", h, d0, d1, d2);
    verdict(dev, same_host, "a reduced GRAD over 100000 samples: the host and the device give the same bits");
    verdict(dev, same_runs, "the same reduced GRAD on the device, 3 runs: the same bits");
    free(x);
}

/* ---- host nodes: the caller's callback and its vjp as leaves of the graph and of the reverse -------------------------------------------- */

/* Read or write @p n doubles (or int32) of a view the door hands a callback: host memory, or a device plane through the CUDA runtime. */
static void view_copy(const void* view, void* host, size_t bytes, int to_view) {
    const DLTensor* t = &((const struct DLManagedTensorVersioned*)view)->dl_tensor;
    char* data = (char*)t->data + t->byte_offset;
#ifdef EG_TEST_CUDA
    if (t->device.device_type == kDLCUDA) {
        if (to_view) cudaMemcpy(data, host, bytes, cudaMemcpyHostToDevice);
        else cudaMemcpy(host, data, bytes, cudaMemcpyDeviceToHost);
        return;
    }
#endif
    if (to_view) memcpy(data, host, bytes);
    else memcpy(host, data, bytes);
}
static void view_get(const void* view, double* out) { view_copy(view, out, N * sizeof(double), 0); }
static void view_put(const void* view, const double* in) { view_copy(view, (void*)in, N * sizeof(double), 1); }
static void view_get_i32(const void* view, int32_t* out) { view_copy(view, out, N * sizeof(int32_t), 0); }

/* What a callback records about the calls it saw. */
struct host_state {
    int calls, vjp_calls;
    int layout_ok;      /* the vjp saw NULL entries exactly where the declaration says and zeroed adjoints_out */
    int fail_code;      /* nonzero: the callback returns it */
};

/* z = y^2 over planes {y READ, z WRITE}. */
static int32_t hn_square(void* user, int32_t n, struct DLManagedTensorVersioned* const* planes) {
    struct host_state* st = (struct host_state*)user;
    double y[N], z[N];
    ++st->calls;
    if (n != 2) return 1;
    view_get(planes[0], y);
    for (int i = 0; i < N; ++i) z[i] = y[i] * y[i];
    view_put(planes[1], z);
    return st->fail_code;
}
/* The vjp of hn_square: dy = 2 y dz, written into the zeroed adjoints_out[0]. */
static int32_t hn_square_vjp(void* user, int32_t n, struct DLManagedTensorVersioned* const* primals, struct DLManagedTensorVersioned* const* adjoints_in, struct DLManagedTensorVersioned* const* adjoints_out) {
    struct host_state* st = (struct host_state*)user;
    double y[N], dz[N], dy[N], zero[N];
    ++st->vjp_calls;
    st->layout_ok = n == 2 && primals[0] && !primals[1] && !adjoints_in[0] && adjoints_in[1] && adjoints_out[0] && !adjoints_out[1];
    if (!st->layout_ok) return 1;
    view_get(primals[0], y);
    view_get(adjoints_in[1], dz);
    view_get(adjoints_out[0], zero);
    for (int i = 0; i < N; ++i) {
        st->layout_ok = st->layout_ok && zero[i] == 0.0;
        dy[i] = 2.0 * y[i] * dz[i];
    }
    view_put(adjoints_out[0], dy);
    return 0;
}

/* y = y^2 in place over {y READ|WRITE}. */
static int32_t hn_square_in_place(void* user, int32_t n, struct DLManagedTensorVersioned* const* planes) {
    struct host_state* st = (struct host_state*)user;
    double y[N];
    ++st->calls;
    if (n != 1) return 1;
    view_get(planes[0], y);
    for (int i = 0; i < N; ++i) y[i] *= y[i];
    view_put(planes[0], y);
    return 0;
}
/* Its vjp: the primal is the y the node read (before it overwrote it), dy = 2 y dy_out; adjoints_in and adjoints_out both name the one plane. */
static int32_t hn_square_in_place_vjp(void* user, int32_t n, struct DLManagedTensorVersioned* const* primals, struct DLManagedTensorVersioned* const* adjoints_in, struct DLManagedTensorVersioned* const* adjoints_out) {
    struct host_state* st = (struct host_state*)user;
    double y[N], dy[N];
    ++st->vjp_calls;
    st->layout_ok = n == 1 && primals[0] && adjoints_in[0] && adjoints_out[0];
    if (!st->layout_ok) return 1;
    view_get(primals[0], y);
    view_get(adjoints_in[0], dy);
    for (int i = 0; i < N; ++i) dy[i] *= 2.0 * y[i];
    view_put(adjoints_out[0], dy);
    return 0;
}

/* y = n x over {x READ, n READ (int32), y WRITE}, and its vjp dx = n dy: the int plane has no adjoint view. */
static int32_t hn_int_scale(void* user, int32_t n, struct DLManagedTensorVersioned* const* planes) {
    struct host_state* st = (struct host_state*)user;
    double x[N], y[N];
    int32_t k[N];
    ++st->calls;
    if (n != 3) return 1;
    view_get(planes[0], x);
    view_get_i32(planes[1], k);
    for (int i = 0; i < N; ++i) y[i] = k[i] * x[i];
    view_put(planes[2], y);
    return 0;
}
static int32_t hn_int_scale_vjp(void* user, int32_t n, struct DLManagedTensorVersioned* const* primals, struct DLManagedTensorVersioned* const* adjoints_in, struct DLManagedTensorVersioned* const* adjoints_out) {
    struct host_state* st = (struct host_state*)user;
    double dy[N], dx[N];
    int32_t k[N];
    ++st->vjp_calls;
    st->layout_ok = n == 3 && primals[0] && primals[1] && !primals[2] && !adjoints_in[0] && !adjoints_in[1] && adjoints_in[2] && adjoints_out[0] && !adjoints_out[1] && !adjoints_out[2];
    if (!st->layout_ok) return 1;
    view_get_i32(primals[1], k);
    view_get(adjoints_in[2], dy);
    for (int i = 0; i < N; ++i) dx[i] = k[i] * dy[i];
    view_put(adjoints_out[0], dx);
    return 0;
}

/* s = 3 y over {y READ, s WRITE}, with no vjp. */
static int32_t hn_triple(void* user, int32_t n, struct DLManagedTensorVersioned* const* planes) {
    struct host_state* st = (struct host_state*)user;
    double y[N], s[N];
    ++st->calls;
    if (n != 2) return 1;
    view_get(planes[0], y);
    for (int i = 0; i < N; ++i) s[i] = 3.0 * y[i];
    view_put(planes[1], s);
    return 0;
}

/* y *= 3 in place over {y READ|WRITE}. */
static int32_t hn_triple_in_place(void* user, int32_t n, struct DLManagedTensorVersioned* const* planes) {
    struct host_state* st = (struct host_state*)user;
    double y[N];
    ++st->calls;
    if (n != 1) return 1;
    view_get(planes[0], y);
    for (int i = 0; i < N; ++i) y[i] *= 3.0;
    view_put(planes[0], y);
    return 0;
}

static eg_host_node* host_node(eg_ctx* ctx, const char* name, eg_host_fn fn, eg_host_vjp_fn vjp, struct host_state* st) {
    struct eg_host_node_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.name = name;
    d.fn = fn;
    d.vjp = vjp;
    d.user = st;
    eg_host_node* node = NULL;
    return eg_host_node_new(ctx, &d, &node) == EG_OK ? node : NULL;
}

static eg_status host_call(eg_ctx* ctx, eg_host_node* node, int32_t n, eg_plane* const* planes, const uint32_t* access) { return eg_host_call(ctx, node, n, planes, access); }

#define R_ EG_ACCESS_READ
#define W_ EG_ACCESS_WRITE
#define RW_ (EG_ACCESS_READ | EG_ACCESS_WRITE)

/* y = w x, then a host node z = y^2 (with a vjp), J = sum z: the node is a leaf of the reverse, and dJ/dx = 2 w^2 x, dJ/dw = sum 2 w x^2, the closed form of chain_row
 * with the second launch replaced by the caller's function. */
static void host_leaf_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], z[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1] = { 0.0 };
    struct tensor tx, ty, tz, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *pz = mk(r, &tz, N, z), *pw = mk(r, &tw, 1, w), *p1 = mk(r, &t1, N, one);
    eg_plane *pgx = mk(r, &tgx, N, gx), *pgw = mk(r, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    struct host_state st = { 0 };
    eg_host_node* node = host_node(ctx, "square", hn_square, hn_square_vjp, &st);
    eg_plane* decl[] = { py, pz };
    const uint32_t access[] = { R_, W_ };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && node && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && host_call(ctx, node, 2, decl, access) == EG_OK && st.calls == 0 &&
             end_diff(ctx, 1, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane *in[] = { px, pw }, *gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, pz, p1, 2, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) {
        wx[i] = 2.0 * w[0] * w[0] * x[i];
        ww[0] += 2.0 * w[0] * x[i] * x[i];
    }
    verdict(r, ok && st.calls == 1 && st.vjp_calls == 1 && st.layout_ok && close_to(gx, wx, N) && close_to(gw, ww, 1),
            "eg_graph_reverse through a host node with a vjp (y = w x, z = host y^2): the vjp saw the primal, the adjoint of z and a zeroed adjoint_out, and dJ/dx, dJ/dw match the closed form");
    struct eg_graph_description desc;
    memset(&desc, 0, sizeof(desc));
    desc.struct_size = sizeof(desc);
    verdict(r, eg_graph_describe(ctx, graph, &desc) == EG_OK && desc.n_nodes == 2 && desc.nodes[0].kind == EG_GNODE_LAUNCH && desc.nodes[1].kind == EG_GNODE_HOST &&
                   strcmp(desc.nodes[1].unit, "square") == 0 && desc.nodes[1].n_planes == 2 && desc.nodes[1].access[0] == 1 && desc.nodes[1].access[1] == 2 && desc.n_levels == 2,
            "eg_graph_describe of a DIFF graph with a host node: EG_GNODE_HOST named 'square', a read and a written plane, two levels");
    eg_graph_free(ctx, graph);
    eg_host_node_free(ctx, node);
    eg_plan_free(ctx, f);
    release(r);
}

/* y = w x, then a host node y <- y^2 in place (read, then overwritten), J = sum y: its vjp needs the y it read, which the graph copied before the node
 * overwrote it. dJ/dx = 2 w^2 x, dJ/dw = sum 2 w x^2. */
static void host_in_place_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1] = { 0.0 };
    struct tensor tx, ty, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *pw = mk(r, &tw, 1, w), *p1 = mk(r, &t1, N, one), *pgx = mk(r, &tgx, N, gx), *pgw = mk(r, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    struct host_state st = { 0 };
    eg_host_node* node = host_node(ctx, "square_in_place", hn_square_in_place, hn_square_in_place_vjp, &st);
    eg_plane* decl[] = { py };
    const uint32_t access[] = { RW_ };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && node && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && host_call(ctx, node, 1, decl, access) == EG_OK &&
             end_diff(ctx, 1, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane *in[] = { px, pw }, *gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, py, p1, 2, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) {
        wx[i] = 2.0 * w[0] * w[0] * x[i];
        ww[0] += 2.0 * w[0] * x[i] * x[i];
    }
    verdict(r, ok && st.layout_ok && close_to(gx, wx, N) && close_to(gw, ww, 1),
            "eg_graph_reverse through a host node that overwrites its input in place (y <- y^2): the vjp read the y the node read, and dJ/dx, dJ/dw match the closed form");
    eg_graph_free(ctx, graph);
    eg_host_node_free(ctx, node);
    eg_plan_free(ctx, f);
    release(r);
}

/* y = n x in a host node over {x, n (int32), y}, with n overwritten by a later eg_copy: the vjp reads the n the node read (kept before the overwrite),
 * gets NULL for the int plane's adjoints, and dJ/dx = n. */
static void host_int_row(struct rig* r) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], seed[N], gx[N], want[N];
    int32_t n[N], later[N];
    struct tensor tx, ty, ts, tg, tn, tl;
    for (int i = 0; i < N; ++i) {
        x[i] = 0.5 + 0.25 * i;
        seed[i] = 1.0;
        n[i] = 2 + i % 5;
        later[i] = 100 + i;
        want[i] = (double)n[i];
        gx[i] = y[i] = 0.0;
    }
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *ps = mk(r, &ts, N, seed), *pg = mk(r, &tg, N, gx);
    eg_plane *pn = mk_i32(r, &tn, N, n), *pl = mk_i32(r, &tl, N, later);
    struct host_state st = { 0 };
    eg_host_node* node = host_node(ctx, "int_scale", hn_int_scale, hn_int_scale_vjp, &st);
    eg_plane* decl[] = { px, pn, py };
    const uint32_t access[] = { R_, R_, W_ };
    eg_graph* g = NULL;
    up(r);
    int ok = node && eg_record_begin(ctx) == EG_OK && host_call(ctx, node, 3, decl, access) == EG_OK && eg_copy(ctx, pl, pn) == EG_OK && end_diff(ctx, 0, NULL, &g) == EG_OK;
    eg_plane *in[] = { px }, *gr[] = { pg };
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && reverse(ctx, g, py, ps, 1, in, gr) == EG_OK;
    down(r);
    verdict(r, ok && st.layout_ok && close_to(gx, want, N) && n[N - 1] == later[N - 1],
            "eg_graph_reverse through a host node reading an int32 plane that a later eg_copy overwrites: the vjp read the n the node read, the int plane has NULL adjoint entries, dJ/dx = n");
    eg_graph_free(ctx, g);
    eg_host_node_free(ctx, node);
    release(r);
}

/* A host node with no vjp: refused by name when a gradient flows through it; skipped when no seeded output depends on what it writes (y = w x, side plane s = 3 y
 * by the node: seeding y gives dJ/dx = w and the node's callback ran in the forward only; seeding s is refused). */
static void host_no_vjp_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], s[N], w[1], one[N], gx[N], gw[1], wx[N];
    struct tensor tx, ty, ts, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *ps = mk(r, &ts, N, s), *pw = mk(r, &tw, 1, w), *p1 = mk(r, &t1, N, one), *pgx = mk(r, &tgx, N, gx);
    eg_plane* pgw = mk(r, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp } };
    struct host_state st = { 0 };
    eg_host_node* node = host_node(ctx, "triple", hn_triple, NULL, &st);
    eg_plane* decl[] = { py, ps };
    const uint32_t access[] = { R_, W_ };
    eg_graph* graph = NULL;
    up(r);
    int ok = f && node && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && host_call(ctx, node, 2, decl, access) == EG_OK &&
             end_diff(ctx, 1, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane *in[] = { px, pw }, *gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, py, p1, 1, in, gr) == EG_OK;
    down(r);
    for (int i = 0; i < N; ++i) wx[i] = w[0];
    verdict(r, ok && st.calls == 1 && close_to(gx, wx, N),
            "eg_graph_reverse with a host node that has no vjp and writes what no seeded output depends on: the node is skipped and dJ/dx = w");
    verdict(r, ok && refused(reverse(ctx, graph, ps, p1, 1, in, gr), EG_BAD_REQUEST, ctx, "host node 'triple'") && reason_has(ctx, "has no vjp"),
            "eg_graph_reverse seeded on what a host node with no vjp writes: refused BAD_REQUEST naming the node");
    eg_graph_free(ctx, graph);
    eg_host_node_free(ctx, node);
    eg_plan_free(ctx, f);
    release(r);
}

/* On the host, a host node that writes planes inside a WHILE body of a DIFF recording is refused NOT_WRITTEN by name at eg_record_end, the recording left open
 * (a device recording refuses the node at the call, see device_host_scope_row). */
static void host_while_refusal_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], a[1] = { 0.9 };
    int32_t c[N];
    struct tensor tx, ty, tc;
    fill(x, a, 0.5);
    for (int i = 0; i < N; ++i) c[i] = 1;
    eg_plane *px = mk(r, &tx, N, x), *py = mk(r, &ty, N, y), *pc = mk_i32(r, &tc, N, c);
    (void)u;
    struct host_state st = { 0 };
    eg_host_node* node = host_node(ctx, "square_in_place", hn_square_in_place, hn_square_in_place_vjp, &st);
    eg_plane* decl[] = { py };
    const uint32_t access[] = { RW_ };
    eg_graph *g = NULL, *plain = NULL;
    int ok = px && node && pc && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pc, 0, 3) == EG_OK && host_call(ctx, node, 1, decl, access) == EG_OK &&
             eg_record_cond_end(ctx) == EG_OK;
    ok = ok && refused(end_diff(ctx, 0, NULL, &g), EG_NOT_WRITTEN, ctx, "host node 'square_in_place'") && g == NULL && eg_record_end(ctx, NULL, &plain) == EG_OK;
    verdict(r, ok, "EG_RECORD_DIFF over a host node that writes inside a WHILE body: refused NOT_WRITTEN naming the node; the recording stays open and ends plain");
    eg_graph_free(ctx, plain);
    eg_host_node_free(ctx, node);
    release(r);
}

/* launch (y = w x) -> host node (y *= 3) -> launch (z = y^2), recorded plain and replayed twice, against the same work done eagerly: bitwise equal. On a DEVICE
 * context the host node is a segment between two captures. */
static void host_segment_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], w[1], yr[N], zr[N], yd[N], zd[N];
    struct tensor tx, tw, tyr, tzr, tyd, tzd;
    fill(x, w, 0.5);
    memset(yr, 0, sizeof(yr));
    memset(yd, 0, sizeof(yd));
    memset(zr, 0, sizeof(zr));
    memset(zd, 0, sizeof(zd));
    eg_plane *px = mk(r, &tx, N, x), *pw = mk(r, &tw, 1, w), *pyr = mk(r, &tyr, N, yr), *pzr = mk(r, &tzr, N, zr), *pyd = mk(r, &tyd, N, yd), *pzd = mk(r, &tzd, N, zd);
    eg_plane *a1[] = { px, pw, pyr }, *b1[] = { pyr, pzr }, *a2[] = { px, pw, pyd }, *b2[] = { pyd, pzd };
    eg_plan *f1 = plan_with(r, u->scale, 3, a1, NULL), *g1 = plan_with(r, u->square, 2, b1, NULL), *f2 = plan_with(r, u->scale, 3, a2, NULL), *g2 = plan_with(r, u->square, 2, b2, NULL);
    struct host_state sr = { 0 }, sd = { 0 };
    eg_host_node *nr = host_node(ctx, "triple_in_place", hn_triple_in_place, NULL, &sr), *nd = host_node(ctx, "triple_in_place", hn_triple_in_place, NULL, &sd);
    eg_plane *dr[] = { pyr }, *dd[] = { pyd };
    const uint32_t access[] = { RW_ };
    eg_graph* graph = NULL;
    up(r);
    int ok = f1 && g1 && f2 && g2 && nr && nd && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f1, NULL) == EG_OK && host_call(ctx, nr, 1, dr, access) == EG_OK &&
             eg_launch(ctx, g1, NULL) == EG_OK && eg_record_end(ctx, NULL, &graph) == EG_OK && sr.calls == 0;
    for (int rep = 0; ok && rep < 2; ++rep)
        ok = eg_graph_launch(ctx, graph) == EG_OK && eg_launch(ctx, f2, NULL) == EG_OK && host_call(ctx, nd, 1, dd, access) == EG_OK && eg_launch(ctx, g2, NULL) == EG_OK;
    down(r);
    verdict(r, ok && sr.calls == 2 && sd.calls == 2 && memcmp(yr, yd, sizeof(yr)) == 0 && memcmp(zr, zd, sizeof(zr)) == 0 && yr[1] != 0.0,
            "launch -> host node -> launch recorded and replayed twice equals the same work done eagerly, bitwise");
    eg_graph_free(ctx, graph);
    eg_host_node_free(ctx, nr);
    eg_host_node_free(ctx, nd);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, g1);
    eg_plan_free(ctx, f2);
    eg_plan_free(ctx, g2);
    release(r);
}

/* On a DEVICE context a host node is a step between two captures: refused NOT_WRITTEN by name inside an IF arm, a fork branch and a WHILE, the recording left open. */
static void device_host_scope_row(struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], w[1], y1[N], y2[N];
    int32_t flag[1] = { 1 };
    struct tensor tx, tw, ty1, ty2, tf;
    fill(x, w, 0.5);
    eg_plane *px = mk(r, &tx, N, x), *pw = mk(r, &tw, 1, w), *py1 = mk(r, &ty1, N, y1), *py2 = mk(r, &ty2, N, y2), *pf = mk_i32(r, &tf, 1, flag);
    eg_plane *s1[] = { px, pw, py1 }, *s2[] = { px, pw, py2 };
    eg_plan *f1 = plan_with(r, u->scale, 3, s1, NULL), *f2 = plan_with(r, u->scale, 3, s2, NULL);
    struct host_state st = { 0 };
    eg_host_node* node = host_node(ctx, "square_in_place", hn_square_in_place, NULL, &st);
    eg_plane* decl[] = { py1 };
    const uint32_t access[] = { RW_ };
    eg_graph* open = NULL;
    static const char scope[] = "inside a conditional arm, a fork branch or a WHILE";
    int ok = f1 && f2 && node && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_IF, 0, pf, 0, 0) == EG_OK;
    verdict(r, ok && refused(host_call(ctx, node, 1, decl, access), EG_NOT_WRITTEN, ctx, "host node 'square_in_place'") && reason_has(ctx, scope) && eg_launch(ctx, f1, NULL) == EG_OK &&
                   eg_record_cond_end(ctx) == EG_OK && eg_record_end(ctx, NULL, &open) == EG_OK,
            "a host node inside an IF arm of a device recording: refused NOT_WRITTEN by name, the recording left open");
    eg_graph_free(ctx, open);
    ok = ok && eg_record_begin(ctx) == EG_OK && eg_record_fork_begin(ctx) == EG_OK;
    verdict(r, ok && refused(host_call(ctx, node, 1, decl, access), EG_NOT_WRITTEN, ctx, scope) && eg_launch(ctx, f1, NULL) == EG_OK && eg_record_fork_branch(ctx) == EG_OK &&
                   eg_launch(ctx, f2, NULL) == EG_OK && eg_record_fork_end(ctx) == EG_OK && eg_record_end(ctx, NULL, &open) == EG_OK,
            "a host node inside a fork branch of a device recording: refused NOT_WRITTEN by name, the recording left open");
    eg_graph_free(ctx, open);
    ok = ok && eg_record_begin(ctx) == EG_OK && cond_open(ctx, EG_COND_WHILE, 0, pf, 0, 3) == EG_OK;
    verdict(r, ok && refused(host_call(ctx, node, 1, decl, access), EG_NOT_WRITTEN, ctx, scope) && eg_launch(ctx, f1, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK &&
                   eg_record_end(ctx, NULL, &open) == EG_OK,
            "a host node inside a WHILE of a device recording: refused NOT_WRITTEN by name, the recording left open");
    eg_graph_free(ctx, open);
    eg_host_node_free(ctx, node);
    eg_plan_free(ctx, f1);
    eg_plan_free(ctx, f2);
    release(r);
}

static void host_node_rows(struct rig* r, const struct units* u) {
    host_leaf_row(r, u);
    host_in_place_row(r, u);
    host_int_row(r);
    host_no_vjp_row(r, u);
    host_segment_row(r, u);
}

static struct units load_units(struct rig* r) {
    const struct units u = { load(r, "scale", 0x10), load(r, "scale_vjp", 0x11), load(r, "square", 0x12), load(r, "square_vjp", 0x13),
                             load(r, "mul2", 0x14), load(r, "mul2_vjp", 0x15), load(r, "uscale", 0x16), load(r, "uscale_vjp", 0x17),
                             load(r, "scale_tape_vjp", 0x18), load(r, "scale_open_vjp", 0x19),
                             load(r, "fan", 0x1a), load(r, "fan_vjp", 0x1b), load(r, "norm2", 0x1c), load(r, "norm2_vjp", 0x1d), load(r, "tick", 0x1e), load(r, "scale_scatter_vjp", 0x1f),
                             load(r, "iscale", 0x20), load(r, "iscale_vjp", 0x21), load(r, "scale_vjp_comp", 0x22) };
    verdict(r, u.scale && u.scale_vjp && u.square && u.square_vjp && u.mul2 && u.mul2_vjp && u.uscale && u.uscale_vjp && u.tape_vjp &&
                   u.open_vjp && u.fan && u.fan_vjp && u.norm2 && u.norm2_vjp && u.tick && u.scatter_vjp && u.iscale && u.iscale_vjp && u.scale_vjp_comp,
            "the graph fixture's nineteen units load");
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
    if (argc != 3) {
        fprintf(stderr, "usage: %s <fixture_graph lib> <fixture_reverse lib>\n", argv[0]);
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
    struct rig host = { .ctx = ctx, .lib = argv[1], .run_lib = argv[2] };
    const struct units u = load_units(&host);
    gradient_rows(&host, &u);
    refusal_rows(&host, &u);
    conditional_rows(&host, &u);
    option_rows(&host, &u);
    leaf_rows(&host, &u);
    int_read_rows(&host, &u);
    run_rows(&host, &u);
    host_node_rows(&host, &u);
    host_while_refusal_row(&host, &u);

    int32_t devices = 0;
    if (eg_device_count(ctx, &devices) != EG_OK || devices < 1) {
        check(1, "graph reverse rows on a DEVICE context: skipped, no device present");
    } else {
        st.flags = EG_CTX_DEVICE;
        eg_ctx* dctx = NULL;
        check(eg_ctx_new(&st, &dctx) == EG_OK, "eg_ctx_new DEVICE context");
        struct rig dev = { .ctx = dctx, .lib = argv[1], .run_lib = argv[2], .device = 1 };
        const struct units du = load_units(&dev);
        gradient_rows(&dev, &du);
        device_refusal_rows(&dev, &du);
        conditional_rows(&dev, &du);
        option_rows(&dev, &du);
        leaf_rows(&dev, &du);
        int_read_rows(&dev, &du);
        run_rows(&dev, &du);
        host_node_rows(&dev, &du);
        device_host_scope_row(&dev, &du);
        policy_row(&host, &u, &dev, &du);
        eg_ctx_free(dctx);
    }
    eg_ctx_free(ctx);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures == 0 ? 0 : 1;
}
