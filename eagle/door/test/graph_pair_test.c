/*
 * Device recording and the caller's stream. N launches of the recorded step unit osc (call 2014362.459) recorded
 * into an eagle graph run nothing; K replays leave every plane BITWISE equal to the old device graph's
 * (GraphPipeline: the same launches captured and replayed K times) and to N*K plain launches (closed form:
 * t = min(stop, N*K) * dt). A plan rebound since the recording is refused at replay; what a capture cannot hold
 * is refused by name while recording, and a free asked for meanwhile waits for the capture's end. eg_stream_use:
 * launches and replays on a stream the test makes with the runtime are ordered on it between the test's own
 * copies there.
 *
 * usage: graph_pair_test <osc library> <osc digest> <host fixture library> <inputs dir> <oracle dir>
 * inputs dir: {t_final,x,v}_in.f64; oracle dir: {t,v,x}_replayed.f64, term_replayed.u8 (N 3, K 5).
 */
#define _POSIX_C_SOURCE 200809L
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdlib.h>

#define N 48
#define DT 0.0625
#define LAUNCHES 3
#define REPLAYS 5
enum { S_T, S_V, S_X, S_TFINAL, S_COUNTER, S_TERM, S_DT };

static double t_final[N], x_in[N], v_in[N];

/* t, v, x, t_final (f64) and the mask (bool) on the device, a DEVICE plan of osc bound to them. */
struct set {
    void* d[5];
    struct tensor t[5];
    eg_plane* p[5];
    eg_plan* plan;
};

static const void* init_of(int i) {
    static const double zeros[N];
    static const uint8_t none[N];
    const void* init[5] = { zeros, v_in, x_in, t_final, none };
    return init[i];
}

static size_t bytes_of(int i) { return i < 4 ? N * sizeof(double) : N; }

static eg_status bind(eg_ctx* ctx, eg_plan* plan, eg_plane* const* p) {
    const int32_t slots[5] = { S_T, S_V, S_X, S_TFINAL, S_TERM }, uslot[1] = { S_DT };
    const double dt = DT;
    const void* values[1] = { &dt };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 5;
    b.plane_slots = slots;
    b.planes = p;
    b.n_uniforms = 1;
    b.uniform_slots = uslot;
    b.uniform_values = values;
    return eg_bind(ctx, plan, &b);
}

/* @p stream 0: the inputs are copied synchronously; else asynchronously on @p stream (the caller orders the rest). */
static int set_new(eg_ctx* ctx, eg_unit* unit, struct set* s, cudaStream_t stream) {
    int ok = 1;
    for (int i = 0; i < 5; ++i) {
        ok &= cudaMalloc(&s->d[i], bytes_of(i)) == cudaSuccess;
        ok &= (stream ? cudaMemcpyAsync(s->d[i], init_of(i), bytes_of(i), cudaMemcpyHostToDevice, stream)
                      : cudaMemcpy(s->d[i], init_of(i), bytes_of(i), cudaMemcpyHostToDevice)) == cudaSuccess;
        struct tensor* t = &s->t[i];
        tensor_fill(t, N, s->d[i], i < 4 ? kDLFloat : kDLBool, i < 4 ? 64 : 8);
        s->p[i] = NULL;
        ok = ok && eg_plane_wrap(ctx, &t->managed, &s->p[i]) == EG_OK;
    }
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = unit;
    pd.structure = EG_STRUCTURE_DEVICE;
    pd.n_samples = N;
    s->plan = NULL;
    return ok && eg_plan_new(ctx, &pd, &s->plan) == EG_OK && bind(ctx, s->plan, s->p) == EG_OK;
}

static void set_free(eg_ctx* ctx, struct set* s) {
    eg_plan_free(ctx, s->plan);
    for (int i = 0; i < 5; ++i) {
        eg_plane_free(ctx, s->p[i]);
        cudaFree(s->d[i]);
    }
}

/* The planes of @p s on the host (out[i] holds bytes_of(i)); @p stream as in set_new. */
static int set_get(const struct set* s, void* const* out, cudaStream_t stream) {
    int ok = 1;
    for (int i = 0; i < 5; ++i)
        ok &= (stream ? cudaMemcpyAsync(out[i], s->d[i], bytes_of(i), cudaMemcpyDeviceToHost, stream)
                      : cudaMemcpy(out[i], s->d[i], bytes_of(i), cudaMemcpyDeviceToHost)) == cudaSuccess;
    return ok && (!stream || cudaStreamSynchronize(stream) == cudaSuccess);
}

struct host_planes {
    double t[N], v[N], x[N], tf[N];
    uint8_t term[N];
};

static void* const* slots_of(struct host_planes* h) {
    static void* out[5];
    out[0] = h->t;
    out[1] = h->v;
    out[2] = h->x;
    out[3] = h->tf;
    out[4] = h->term;
    return out;
}

static int same(const struct host_planes* a, const struct host_planes* b) {
    return memcmp(a->t, b->t, sizeof(a->t)) == 0 && memcmp(a->v, b->v, sizeof(a->v)) == 0 &&
           memcmp(a->x, b->x, sizeof(a->x)) == 0 && memcmp(a->term, b->term, sizeof(a->term)) == 0;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <osc lib> <osc digest> <host fixture lib> <inputs dir> <oracle dir>\n", argv[0]);
        return 2;
    }
    if (!read_file(argv[4], "t_final_in.f64", t_final, sizeof(t_final)) || !read_file(argv[4], "x_in.f64", x_in, sizeof(x_in)) ||
        !read_file(argv[4], "v_in.f64", v_in, sizeof(v_in))) {
        fprintf(stderr, "inputs missing in %s\n", argv[4]);
        return 2;
    }
    struct host_planes old;
    const int old_ok = read_file(argv[5], "t_replayed.f64", old.t, sizeof(old.t)) && read_file(argv[5], "v_replayed.f64", old.v, sizeof(old.v)) &&
                       read_file(argv[5], "x_replayed.f64", old.x, sizeof(old.x)) && read_file(argv[5], "term_replayed.u8", old.term, sizeof(old.term));
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.flags = EG_CTX_DEVICE;
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = "osc";
    struct eg_unit_load_desc ld;
    memset(&ld, 0, sizeof(ld));
    ld.struct_size = sizeof(ld);
    ld.path = argv[1];
    ld.unit = &ref;
    eg_unit* osc = NULL;
    check(parse_digest(argv[2], ref.digest) && eg_unit_load(ctx, &ld, &osc) == EG_OK, "eg_unit_load of osc (PTX image)");
    if (osc == NULL) return 1;

    /* the plain-C oracle: N*K launches */
    struct set direct;
    struct host_planes want, got, zero;
    int ok = set_new(ctx, osc, &direct, 0);
    for (int i = 0; ok && i < LAUNCHES * REPLAYS; ++i) ok = eg_launch(ctx, direct.plan, NULL) == EG_OK;
    ok = ok && eg_sync(ctx) == EG_OK && set_get(&direct, slots_of(&want), 0);
    int closed = ok;
    for (int i = 0; closed && i < N; ++i) {
        const double stop = round(t_final[i] / DT), k = LAUNCHES * REPLAYS;
        closed = want.t[i] == (stop < k ? stop : k) * DT && (want.term[i] != 0) == (stop <= k);
    }
    check(ok && closed, "15 plain launches: t = min(stop, 15) * dt and the mask set at the stop (closed form)");

    /* record N launches, replay K */
    struct set rec;
    eg_graph* graph = NULL;
    ok = set_new(ctx, osc, &rec, 0) && eg_record_begin(ctx) == EG_OK;
    for (int i = 0; ok && i < LAUNCHES; ++i) ok = eg_launch(ctx, rec.plan, NULL) == EG_OK;
    ok = ok && eg_record_end(ctx, NULL, &graph) == EG_OK && graph != NULL && set_get(&rec, slots_of(&got), 0);
    int untouched = ok;
    for (int i = 0; untouched && i < N; ++i) untouched = got.t[i] == 0.0 && got.x[i] == x_in[i] && got.term[i] == 0;
    check(untouched, "recording 3 DEVICE launches: eg_record_end OK, the capture ran nothing");
    for (int r = 0; ok && r < REPLAYS; ++r) ok = eg_graph_launch(ctx, graph) == EG_OK;
    ok = ok && eg_sync(ctx) == EG_OK && set_get(&rec, slots_of(&got), 0);
    check(ok && same(&got, &want), "5 replays of the 3 recorded launches: t, v, x, mask bitwise equal to 15 plain launches");
    check(ok && old_ok && same(&got, &old), "5 replays: t, v, x, mask bitwise equal to the old device graph (GraphPipeline, 3 launches, launch(5))");

    /* a rebound plan is refused at replay */
    struct set other;
    ok = set_new(ctx, osc, &other, 0) && bind(ctx, rec.plan, other.p) == EG_OK;
    check(ok && eg_graph_launch(ctx, graph) == EG_BAD_REQUEST && reason_has(ctx, "rebound since the recording"),
          "a plan rebound since the recording: eg_graph_launch refused BAD_REQUEST");
    check(bind(ctx, rec.plan, rec.p) == EG_OK && eg_graph_launch(ctx, graph) == EG_OK && eg_sync(ctx) == EG_OK,
          "bound back to the recorded planes: the replay runs again");
    set_free(ctx, &other);

    /* what a capture cannot hold, refused by name while recording; a free waits for the capture's end */
    struct eg_unit_ref href;
    memset(&href, 0, sizeof(href));
    href.struct_size = sizeof(href);
    href.name = "axpb";
    memset(href.digest, 0xA1, EG_DIGEST_BYTES);
    ld.path = argv[3];
    ld.unit = &href;
    eg_unit* host_unit = NULL;
    struct eg_plan_desc hd;
    memset(&hd, 0, sizeof(hd));
    hd.struct_size = sizeof(hd);
    hd.structure = EG_STRUCTURE_HOST_TEAM;
    hd.n_samples = N;
    eg_plan* host_plan = NULL;
    ok = eg_unit_load(ctx, &ld, &host_unit) == EG_OK && (hd.unit = host_unit, eg_plan_new(ctx, &hd, &host_plan) == EG_OK);
    check(ok, "a host unit and plan on the device context (made before recording)");
    struct set spare;
    ok = set_new(ctx, osc, &spare, 0) && eg_record_begin(ctx) == EG_OK;
    check(ok && eg_record_begin(ctx) == EG_BAD_REQUEST && reason_has(ctx, "already recording"), "eg_record_begin twice: BAD_REQUEST");
    check(eg_sync(ctx) == EG_BAD_REQUEST && reason_has(ctx, "eg_sync while recording"), "eg_sync while recording: BAD_REQUEST");
    check(eg_stream_use(ctx, 0) == EG_BAD_REQUEST && reason_has(ctx, "eg_stream_use while recording"),
          "eg_stream_use while recording: BAD_REQUEST");
    eg_plan* none = NULL;
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = osc;
    pd.structure = EG_STRUCTURE_DEVICE;
    pd.n_samples = N;
    check(eg_plan_new(ctx, &pd, &none) == EG_BAD_REQUEST && reason_has(ctx, "eg_plan_new while recording"),
          "eg_plan_new while recording on a device: BAD_REQUEST");
    eg_runner* runner = NULL;
    struct eg_runner_desc rd;
    memset(&rd, 0, sizeof(rd));
    rd.struct_size = sizeof(rd);
    rd.n_plans = 1;
    rd.plans = &rec.plan;
    rd.max_steps = 10;
    check(eg_runner_new(ctx, &rd, &runner) == EG_BAD_REQUEST && reason_has(ctx, "eg_runner_new while recording"),
          "eg_runner_new while recording on a device: BAD_REQUEST");
    check(eg_launch(ctx, host_plan, NULL) == EG_NOT_WRITTEN && reason_has(ctx, "host work inside a device graph"),
          "a host launch while recording on a device: NOT_WRITTEN by name");
    struct eg_cond_desc cd;
    memset(&cd, 0, sizeof(cd));
    cd.struct_size = sizeof(cd);
    check(eg_record_cond_begin(ctx, &cd) == EG_BAD_REQUEST && reason_has(ctx, "eg_cond_desc.cell"),
          "a conditional with no cell in a device recording: BAD_REQUEST by name");
    check(eg_graph_launch(ctx, graph) == EG_NOT_WRITTEN && reason_has(ctx, "a graph inside a device graph"),
          "replaying a graph while recording on a device: NOT_WRITTEN by name");
    check(eg_launch(ctx, rec.plan, NULL) == EG_OK, "a DEVICE launch is still recorded after the refusals");
    eg_plan_free(ctx, spare.plan);    /* its finished counter is a device cell: freed after the capture */
    spare.plan = NULL;
    eg_graph* g2 = NULL;
    check(eg_record_end(ctx, NULL, &g2) == EG_OK && g2 != NULL, "eg_record_end OK after a DEVICE plan was freed while recording");
    set_free(ctx, &spare);
    ok = set_new(ctx, osc, &other, 0);
    check(ok && eg_graph_free(ctx, g2) == EG_OK, "the second graph frees; a plan made after it works");
    set_free(ctx, &other);

    /* the caller's stream */
    cudaStream_t s = NULL;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess && eg_stream_use(ctx, (uint64_t)(uintptr_t)s) == EG_OK,
          "eg_stream_use of a stream the test made with the runtime");
    struct set on;
    ok = set_new(ctx, osc, &on, s);    /* inputs copied asynchronously on s, then the launches, then the copies back */
    for (int i = 0; ok && i < LAUNCHES * REPLAYS; ++i) ok = eg_launch(ctx, on.plan, NULL) == EG_OK;
    ok = ok && set_get(&on, slots_of(&got), s);
    check(ok && same(&got, &want), "15 launches ordered on the caller's stream between its copies: equal to the oracle");
    struct set onrec;
    eg_graph* g3 = NULL;
    ok = set_new(ctx, osc, &onrec, s) && eg_record_begin(ctx) == EG_OK;
    for (int i = 0; ok && i < LAUNCHES; ++i) ok = eg_launch(ctx, onrec.plan, NULL) == EG_OK;
    ok = ok && eg_record_end(ctx, NULL, &g3) == EG_OK;
    for (int r = 0; ok && r < REPLAYS; ++r) ok = eg_graph_launch(ctx, g3) == EG_OK;
    ok = ok && set_get(&onrec, slots_of(&got), s);
    check(ok && same(&got, &want), "recorded on the caller's stream, 5 replays on it: equal to the oracle");
    check(eg_stream_use(ctx, 0) == EG_OK && eg_graph_launch(ctx, g3) == EG_OK && eg_sync(ctx) == EG_OK,
          "eg_stream_use 0: back on eagle's own stream, a replay and eg_sync");
    set_free(ctx, &on);
    eg_graph_free(ctx, g3);
    set_free(ctx, &onrec);
    cudaStreamDestroy(s);
    eg_ctx* host = NULL;
    st.flags = 0;
    check(eg_ctx_new(&st, &host) == EG_OK && eg_stream_use(host, 1) == EG_BAD_REQUEST && reason_has(host, "host-only context"),
          "eg_stream_use on a host-only context: BAD_REQUEST");
    eg_ctx_free(host);

    /* freeing the context while recording, with a graph never freed */
    check(eg_record_begin(ctx) == EG_OK && eg_launch(ctx, rec.plan, NULL) == EG_OK, "recording again at the end");
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free while recording and with a graph not freed");
    for (int i = 0; i < 5; ++i) {    /* the planes went with the context; their memory is the test's */
        cudaFree(direct.d[i]);
        cudaFree(rec.d[i]);
    }
    (void)zero;
    (void)none;
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
