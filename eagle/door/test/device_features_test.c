/*
 * What a DEVICE plan takes beyond a plain unit, each against the old device path on the same inputs, BITWISE: an index
 * set (a lane set: the live count and the map the caller binds; the kernel maps the lanes), a thread box (the launch
 * counts threads, ceil(n / (t * stride)) * stride of them), and a BANDED unit (an ordinary launch over planes of 8-byte
 * band cells). Beside the oracle: the closed form of the lane set, and what stays refused by name (a live count above
 * the map, explicit partitions of a boxed unit, each of the three inside a runner loop).
 *
 * usage: device_features_test <_plain lib> <digest> <kg_fwd lib> <digest> <kq lib> <digest> <oracle dir>
 * oracle dir: is_x.f64, is_y_{map,none}.f64, is_map_{map,none}.i32, box_{feat,th,y}_<rows>.f64, band_{x,y,out}.u64.
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdlib.h>

/* A device plane of @p n elements (@p code, @p bits) holding @p init (n * bits / 8 bytes; NULL: zeros). */
struct dplane {
    void* d;
    struct tensor t;
    eg_plane* p;
};

static int dplane_new(eg_ctx* ctx, struct dplane* s, int64_t n, uint8_t code, uint8_t bits, const void* init) {
    const size_t bytes = (size_t)n * bits / 8;
    s->p = NULL;
    if (cudaMalloc(&s->d, bytes ? bytes : 1) != cudaSuccess) return 0;
    if (init != NULL ? cudaMemcpy(s->d, init, bytes, cudaMemcpyHostToDevice) != cudaSuccess : cudaMemset(s->d, 0, bytes) != cudaSuccess) return 0;
    tensor_fill(&s->t, n, s->d, code, bits);
    return eg_plane_wrap(ctx, &s->t.managed, &s->p) == EG_OK;
}

static void dplane_free(eg_ctx* ctx, struct dplane* s) {
    if (s->p != NULL) eg_plane_free(ctx, s->p);
    cudaFree(s->d);
}

static void* read_all(const char* dir, const char* file, size_t size) {
    void* buf = malloc(size);
    if (buf != NULL && !read_file(dir, file, buf, size)) {
        free(buf);
        return NULL;
    }
    return buf;
}

static eg_unit* load_unit(eg_ctx* ctx, const char* lib, const char* name, const char* hex) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    struct eg_unit_load_desc ld;
    memset(&ld, 0, sizeof(ld));
    ld.struct_size = sizeof(ld);
    ld.path = lib;
    ld.unit = &ref;
    eg_unit* unit = NULL;
    char what[160];
    snprintf(what, sizeof(what), "%s: eg_unit_load of the recorded unit on the device context", name);
    const int ok = parse_digest(hex, ref.digest) && eg_unit_load(ctx, &ld, &unit) == EG_OK;
    check(ok, what);
    return ok ? unit : NULL;
}

static eg_plan* plan_of(eg_ctx* ctx, eg_unit* unit, int64_t n) {
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = unit;
    pd.structure = EG_STRUCTURE_DEVICE;
    pd.n_samples = n;
    eg_plan* plan = NULL;
    return eg_plan_new(ctx, &pd, &plan) == EG_OK ? plan : NULL;
}

static eg_status bind(eg_ctx* ctx, eg_plan* plan, int n, const int32_t* slots, struct dplane* const* planes) {
    eg_plane* handles[4];
    for (int i = 0; i < n; ++i) handles[i] = planes[i]->p;
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = n;
    b.plane_slots = slots;
    b.planes = handles;
    return eg_bind(ctx, plan, &b);
}

/* Launch @p plan and read @p out (device) back into @p got (@p bytes). */
static int launch_read(eg_ctx* ctx, eg_plan* plan, const struct dplane* out, void* got, size_t bytes) {
    return eg_launch(ctx, plan, NULL) == EG_OK && eg_sync(ctx) == EG_OK && cudaMemcpy(got, out->d, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

static void runner_refused(eg_ctx* ctx, eg_plan* plan, const char* part, const char* what) {
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_plans = 1;
    d.plans = &plan;
    d.max_steps = 1;
    d.steps_reverse = EG_STEPS_REVERSE_PER_STEP;
    d.events = EG_EVENTS_STEP;
    d.on_no_event = EG_NO_EVENT_REFUSE;
    eg_runner* runner = NULL;
    check(eg_runner_new(ctx, &d, &runner) == EG_NOT_WRITTEN && reason_has(ctx, part) && runner == NULL, what);
}

/* ---- an index set: slots y, x, active_count, active_map -------------------------------------------------------- */
#define LANES 40
static void index_set(eg_ctx* ctx, const char* lib, const char* hex, const char* dir) {
    eg_unit* unit = load_unit(ctx, lib, "_plain", hex);
    double x[LANES], want[LANES], got[LANES], seven[LANES];
    int32_t map[5];
    char file[64];
    if (unit == NULL || !read_file(dir, "is_x.f64", x, sizeof(x))) return;
    for (int i = 0; i < LANES; ++i) seven[i] = -7.0;
    const struct { const char* tag; int live; } cases[2] = { { "map", 5 }, { "none", 0 } };
    for (int c = 0; c < 2; ++c) {
        const int live = cases[c].live, entries = live ? 5 : 1;
        snprintf(file, sizeof(file), "is_map_%s.i32", cases[c].tag);
        int ok = read_file(dir, file, map, entries * sizeof(int32_t));
        snprintf(file, sizeof(file), "is_y_%s.f64", cases[c].tag);
        ok = ok && read_file(dir, file, want, sizeof(want));
        const uint32_t count = (uint32_t)live;
        struct dplane y = { 0 }, px = { 0 }, pc = { 0 }, pm = { 0 };
        eg_plan* plan = plan_of(ctx, unit, LANES);
        const int32_t slots[4] = { 0, 1, 2, 3 };
        struct dplane* planes[4] = { &y, &px, &pc, &pm };
        ok = ok && plan && dplane_new(ctx, &y, LANES, kDLFloat, 64, seven) && dplane_new(ctx, &px, LANES, kDLFloat, 64, x) &&
             dplane_new(ctx, &pc, 1, kDLUInt, 32, &count) && dplane_new(ctx, &pm, entries, kDLInt, 32, map) &&
             bind(ctx, plan, 4, slots, planes) == EG_OK && launch_read(ctx, plan, &y, got, sizeof(got));
        char what[200];
        snprintf(what, sizeof(what), "index set (%d live lanes of %d): DEVICE launch OK, y bitwise vs the old device path", live, LANES);
        check(ok && memcmp(got, want, sizeof(got)) == 0, what);
        int closed = ok;
        for (int i = 0; closed && i < LANES; ++i) {
            int listed = 0;
            for (int k = 0; k < live; ++k) listed |= map[k] == i;
            closed = listed ? fabs(got[i] - (x[i] * x[i] + 1.0)) <= 1e-13 : got[i] == -7.0;
        }
        snprintf(what, sizeof(what), "index set (%d live lanes): closed form, listed lanes x*x+1, the rest untouched", live);
        check(closed, what);
        if (c == 0) {    /* a live count above the map's entries */
            const uint32_t six = 6;
            check(cudaMemcpy(pc.d, &six, sizeof(six), cudaMemcpyHostToDevice) == cudaSuccess && eg_launch(ctx, plan, NULL) == EG_BAD_REQUEST &&
                      reason_has(ctx, "index count 6"),
                  "index set: a live count above the map's entries refused BAD_REQUEST");
            runner_refused(ctx, plan, "an index set", "index set: a runner over it refused NOT_WRITTEN by name");
        }
        if (plan) eg_plan_free(ctx, plan);
        for (int i = 0; i < 4; ++i) dplane_free(ctx, planes[i]);
    }
    eg_unit_free(ctx, unit);
}

/* ---- a thread box: slots y, lane, feat, th ---------------------------------------------------------------------- */
#define FEAT 16384
static void thread_box(eg_ctx* ctx, const char* lib, const char* hex, const char* dir) {
    eg_unit* unit = load_unit(ctx, lib, "kg_fwd", hex);
    if (unit == NULL) return;
    double* feat = read_all(dir, "box_feat_1.f64", FEAT * sizeof(double));
    for (int rows = 1; feat != NULL && rows <= 3; ++rows) {
        const int64_t n = rows * 256, th_n = (int64_t)rows * 72;
        char file[64], what[200];
        snprintf(file, sizeof(file), "box_feat_%d.f64", rows);
        double* feat_r = read_all(dir, file, FEAT * sizeof(double));
        snprintf(file, sizeof(file), "box_th_%d.f64", rows);
        double* th = read_all(dir, file, th_n * sizeof(double));
        snprintf(file, sizeof(file), "box_y_%d.f64", rows);
        double* want = read_all(dir, file, n * sizeof(double));
        double* got = malloc(n * sizeof(double));
        int32_t* lane = malloc(n * sizeof(int32_t));
        double* nans = malloc(n * sizeof(double));
        int ok = feat_r && th && want && got && lane && nans;
        for (int64_t i = 0; ok && i < n; ++i) {
            lane[i] = (int32_t)i;
            nans[i] = NAN;
        }
        struct dplane y, pl, pf, pt;
        eg_plan* plan = ok ? plan_of(ctx, unit, n) : NULL;
        const int32_t slots[4] = { 0, 1, 2, 3 };
        struct dplane* planes[4] = { &y, &pl, &pf, &pt };
        ok = ok && plan && dplane_new(ctx, &y, n, kDLFloat, 64, nans) && dplane_new(ctx, &pl, n, kDLInt, 32, lane) &&
             dplane_new(ctx, &pf, FEAT, kDLFloat, 64, feat_r) && dplane_new(ctx, &pt, th_n, kDLFloat, 64, th) &&
             bind(ctx, plan, 4, slots, planes) == EG_OK && launch_read(ctx, plan, &y, got, n * sizeof(double));
        snprintf(what, sizeof(what), "thread box (%lld lanes, t 16 stride 256): DEVICE launch OK, y bitwise vs the old device path", (long long)n);
        check(ok && memcmp(got, want, n * sizeof(double)) == 0, what);
        if (rows == 1) {
            struct eg_plan_desc pd;
            const int64_t base = 0, count = 64;
            memset(&pd, 0, sizeof(pd));
            pd.struct_size = sizeof(pd);
            pd.unit = unit;
            pd.structure = EG_STRUCTURE_DEVICE;
            pd.n_samples = n;
            pd.n_partitions = 1;
            pd.part_base = &base;
            pd.part_count = &count;
            eg_plan* none = NULL;
            check(eg_plan_new(ctx, &pd, &none) == EG_BAD_REQUEST && reason_has(ctx, "thread box"),
                  "thread box: explicit partitions refused BAD_REQUEST (they count threads, not lanes)");
            if (plan) runner_refused(ctx, plan, "a device thread box", "thread box: a runner over it refused NOT_WRITTEN by name");
        }
        if (plan) eg_plan_free(ctx, plan);
        if (ok) for (int i = 0; i < 4; ++i) dplane_free(ctx, planes[i]);
        free(feat_r); free(th); free(want); free(got); free(lane); free(nans);
    }
    free(feat);
    eg_unit_free(ctx, unit);
}

/* ---- a BANDED unit: slots out, x, y over planes of 8-byte band cells ------------------------------------------- */
#define CELLS 4096
static void banded(eg_ctx* ctx, const char* lib, const char* hex, const char* dir) {
    eg_unit* unit = load_unit(ctx, lib, "kq", hex);
    uint64_t *x = read_all(dir, "band_x.u64", CELLS * 8), *y = read_all(dir, "band_y.u64", CELLS * 8),
             *want = read_all(dir, "band_out.u64", CELLS * 8), *got = malloc(CELLS * 8);
    if (unit == NULL) return;
    struct dplane out, px, py;
    eg_plan* plan = plan_of(ctx, unit, CELLS);
    const int32_t slots[3] = { 0, 1, 2 };
    struct dplane* planes[3] = { &out, &px, &py };
    const int ok = x && y && want && got && plan && dplane_new(ctx, &out, CELLS, kDLFloat, 64, NULL) &&
                   dplane_new(ctx, &px, CELLS, kDLFloat, 64, x) && dplane_new(ctx, &py, CELLS, kDLFloat, 64, y) &&
                   bind(ctx, plan, 3, slots, planes) == EG_OK && launch_read(ctx, plan, &out, got, CELLS * 8);
    check(ok && memcmp(got, want, CELLS * 8) == 0, "BANDED unit (4096 band cells): DEVICE launch OK, out cells bitwise vs the old device path");
    if (plan) eg_plan_free(ctx, plan);
    if (ok) for (int i = 0; i < 3; ++i) dplane_free(ctx, planes[i]);
    free(x); free(y); free(want); free(got);
    eg_unit_free(ctx, unit);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <_plain lib> <digest> <kg_fwd lib> <digest> <kq lib> <digest> <oracle dir>\n", argv[0]);
        return 2;
    }
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.flags = EG_CTX_DEVICE;
    st.device = 0;
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    index_set(ctx, argv[1], argv[2], argv[7]);
    thread_box(ctx, argv[3], argv[4], argv[7]);
    banded(ctx, argv[5], argv[6], argv[7]);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
