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

struct rig {
    eg_ctx* ctx;
    const char* lib;
};

static eg_plane* host_plane(eg_ctx* ctx, struct tensor* t, int64_t n, double* data) {
    tensor_fill(t, n, data, kDLFloat, 64);
    t->managed.dl_tensor.device.device_type = kDLCPU;
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

static eg_unit* load(const struct rig* r, const char* name, uint8_t byte) {
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
static eg_plan* plan_with(const struct rig* r, eg_unit* unit, int32_t n_slots, eg_plane* const* p, const double* uniform_k) {
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = EG_STRUCTURE_HOST_SERIAL;
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
static void chain_row(const struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], z[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1];
    struct tensor tx, ty, tz, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane* px = host_plane(ctx, &tx, N, x);
    eg_plane* py = host_plane(ctx, &ty, N, y);
    eg_plane* pz = host_plane(ctx, &tz, N, z);
    eg_plane* pw = host_plane(ctx, &tw, 1, w);
    eg_plane* p1 = host_plane(ctx, &t1, N, one);
    eg_plane* pgx = host_plane(ctx, &tgx, N, gx);
    eg_plane* pgw = host_plane(ctx, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, pz };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->square, 2, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->square_vjp } };
    eg_graph* graph = NULL;
    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             end_diff(ctx, 2, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, pz, p1, 2, in, gr) == EG_OK;
    ww[0] = 0.0;
    for (int i = 0; i < N; ++i) {
        wx[i] = 2.0 * w[0] * w[0] * x[i];
        ww[0] += 2.0 * w[0] * x[i] * x[i];
    }
    check(ok && close_to(gx, wx, N) && close_to(gw, ww, 1), "eg_graph_reverse of y = w x, z = y^2: dJ/dx per sample and dJ/dw (a shared parameter plane) match the closed form");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    eg_plane* all[] = { px, py, pz, pw, p1, pgx, pgw };
    drop(ctx, 7, all);
}

/* y = w x, then x <- y^2 in place: node 0 reads x, node 1 overwrites it. J = sum x_new = sum y^2. */
static void overwrite_row(const struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1];
    struct tensor tx, ty, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane* px = host_plane(ctx, &tx, N, x);
    eg_plane* py = host_plane(ctx, &ty, N, y);
    eg_plane* pw = host_plane(ctx, &tw, 1, w);
    eg_plane* p1 = host_plane(ctx, &t1, N, one);
    eg_plane* pgx = host_plane(ctx, &tgx, N, gx);
    eg_plane* pgw = host_plane(ctx, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, px };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->square, 2, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->square_vjp } };
    eg_graph* graph = NULL;
    double x0[N];
    memcpy(x0, x, sizeof(x0));
    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             end_diff(ctx, 2, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    ok = ok && reverse(ctx, graph, px, p1, 2, in, gr) == EG_OK;
    ww[0] = 0.0;
    for (int i = 0; i < N; ++i) {
        wx[i] = 2.0 * w[0] * w[0] * x0[i];
        ww[0] += 2.0 * w[0] * x0[i] * x0[i];
    }
    check(ok && close_to(gx, wx, N) && close_to(gw, ww, 1), "eg_graph_reverse with the second launch overwriting the plane the first read: the snapshot gives the gradient at the original x");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    eg_plane* all[] = { px, py, pw, p1, pgx, pgw };
    drop(ctx, 6, all);
}

/* y = w x, z = y x: x is read by both nodes. dJ/dx = 2 w x, dJ/dw = sum x^2. A second launch with other x gives its own gradient. */
static void shared_read_row(const struct rig* r, const struct units* u) {
    eg_ctx* ctx = r->ctx;
    double x[N], y[N], z[N], w[1], one[N], gx[N], gw[1], wx[N], ww[1];
    struct tensor tx, ty, tz, tw, t1, tgx, tgw;
    fill(x, w, 0.5);
    for (int i = 0; i < N; ++i) one[i] = 1.0;
    eg_plane* px = host_plane(ctx, &tx, N, x);
    eg_plane* py = host_plane(ctx, &ty, N, y);
    eg_plane* pz = host_plane(ctx, &tz, N, z);
    eg_plane* pw = host_plane(ctx, &tw, 1, w);
    eg_plane* p1 = host_plane(ctx, &t1, N, one);
    eg_plane* pgx = host_plane(ctx, &tgx, N, gx);
    eg_plane* pgw = host_plane(ctx, &tgw, 1, gw);
    eg_plane* a[] = { px, pw, py };
    eg_plane* b[] = { py, px, pz };
    eg_plan* f = plan_with(r, u->scale, 3, a, NULL);
    eg_plan* g = plan_with(r, u->mul2, 3, b, NULL);
    const struct eg_node_vjp vjps[] = { { 0, u->scale_vjp }, { 1, u->mul2_vjp } };
    eg_graph* graph = NULL;
    int ok = f && g && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             end_diff(ctx, 2, vjps, &graph) == EG_OK;
    eg_plane* in[] = { px, pw };
    eg_plane* gr[] = { pgx, pgw };
    for (int round = 0; round < 2; ++round) {
        fill(x, w, round == 0 ? 0.5 : 1.25);
        ok = ok && eg_graph_launch(ctx, graph) == EG_OK && reverse(ctx, graph, pz, p1, 2, in, gr) == EG_OK;
        ww[0] = 0.0;
        for (int i = 0; i < N; ++i) {
            wx[i] = 2.0 * w[0] * x[i];
            ww[0] += x[i] * x[i];
        }
        check(ok && close_to(gx, wx, N) && close_to(gw, ww, 1),
              round == 0 ? "eg_graph_reverse of a plane read by two nodes: the two adjoints are summed"
                         : "a second eg_graph_launch with other inputs, then eg_graph_reverse: the gradient of the second launch");
    }
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    eg_plane* all[] = { px, py, pz, pw, p1, pgx, pgw };
    drop(ctx, 7, all);
}

/* v = fan(x) = (x, 2x, 3x), o = |v|^2, J = sum o: dJ/dx = 28 x. With @p overwrite, a third launch rewrites v from z = 5x after
 * the norm read it, so the norm's vjp needs the snapshot of v. */
static void vector_row(const struct rig* r, const struct units* u, int overwrite) {
    eg_ctx* ctx = r->ctx;
    double x[N], z[N], o[N], v[3 * N], one[N], gx[N], want[N];
    struct tensor tx, tz, to, t1, tgx;
    struct tensor2 tv;
    fill(x, o, 0.5);
    for (int i = 0; i < N; ++i) {
        one[i] = 1.0;
        z[i] = 5.0 * x[i];
    }
    eg_plane* px = host_plane(ctx, &tx, N, x);
    eg_plane* pz = host_plane(ctx, &tz, N, z);
    eg_plane* po = host_plane(ctx, &to, N, o);
    eg_plane* p1 = host_plane(ctx, &t1, N, one);
    eg_plane* pgx = host_plane(ctx, &tgx, N, gx);
    eg_plane* pv = host_plane2(ctx, &tv, 3, v);
    eg_plane* a[] = { px, pv };
    eg_plane* b[] = { pv, po };
    eg_plane* c[] = { pz, pv };
    eg_plan* f = plan_with(r, u->fan, 2, a, NULL);
    eg_plan* g = plan_with(r, u->norm2, 2, b, NULL);
    eg_plan* h = overwrite ? plan_with(r, u->fan, 2, c, NULL) : NULL;
    const struct eg_node_vjp vjps[] = { { 0, u->fan_vjp }, { 1, u->norm2_vjp } };
    eg_graph* graph = NULL;
    int ok = f && g && (h || !overwrite) && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK && eg_launch(ctx, g, NULL) == EG_OK &&
             (!overwrite || eg_launch(ctx, h, NULL) == EG_OK) && end_diff(ctx, 2, vjps, &graph) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK;
    eg_plane* in[] = { px };
    eg_plane* gr[] = { pgx };
    ok = ok && reverse(ctx, graph, po, p1, 1, in, gr) == EG_OK;
    for (int i = 0; i < N; ++i) want[i] = 28.0 * x[i];
    check(ok && close_to(gx, want, N),
          overwrite ? "eg_graph_reverse through a width-3 vector plane that a later launch overwrites: the snapshot keeps the plane's shape and the gradient is right"
                    : "eg_graph_reverse through a width-3 vector plane between two launches: dJ/dx matches the closed form");
    eg_graph_free(ctx, graph);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, g);
    if (h) eg_plan_free(ctx, h);
    eg_plane* all[] = { px, pz, po, p1, pgx, pv };
    drop(ctx, 6, all);
}

static void refusal_rows(const struct rig* r, const struct units* u) {
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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture_graph lib>\n", argv[0]);
        return 2;
    }
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new host context");
    (void)parse_digest;    /* shared helper this test does not need */
    const struct rig r = { ctx, argv[1] };
    const struct units u = { load(&r, "scale", 0x10), load(&r, "scale_vjp", 0x11), load(&r, "square", 0x12), load(&r, "square_vjp", 0x13),
                             load(&r, "mul2", 0x14), load(&r, "mul2_vjp", 0x15), load(&r, "uscale", 0x16), load(&r, "uscale_vjp", 0x17),
                             load(&r, "scale_tape_vjp", 0x18), load(&r, "scale_open_vjp", 0x19),
                             load(&r, "fan", 0x1a), load(&r, "fan_vjp", 0x1b), load(&r, "norm2", 0x1c), load(&r, "norm2_vjp", 0x1d) };
    check(u.scale && u.scale_vjp && u.square && u.square_vjp && u.mul2 && u.mul2_vjp && u.uscale && u.uscale_vjp && u.tape_vjp && u.open_vjp && u.fan && u.fan_vjp && u.norm2 && u.norm2_vjp,
          "the graph fixture's fourteen units load");
    chain_row(&r, &u);
    overwrite_row(&r, &u);
    shared_read_row(&r, &u);
    vector_row(&r, &u, 0);
    vector_row(&r, &u, 1);
    refusal_rows(&r, &u);
    eg_ctx_free(ctx);
    printf("%s: %d failure(s)\n", argv[0], failures);
    return failures == 0 ? 0 : 1;
}
