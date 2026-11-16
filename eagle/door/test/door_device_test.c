/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The door's device rows: a device context, a device unit loaded from the device fixture (a fatbin for sm_61), one
 * plain DEVICE launch against a plain-C oracle (bitwise), and every DEVICE refusal by status and reason. The
 * host fixture's units serve the refusals of the features a DEVICE plan does not run yet.
 *
 * Usage: door_device_test <libfixture_device.so> <libfixture_unit.so>  (run it through the GPU pool)
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#define N 1000

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

/* True when the context's reason contains @p part (the reason is printed when it does not). */
static int reason_has(eg_ctx* ctx, const char* part) {
    char buf[1024];
    size_t len = 0;
    if (eg_error(ctx, buf, sizeof(buf), &len) != EG_OK) return 0;
    if (strstr(buf, part) != NULL) return 1;
    printf("      reason: %s\n", buf);
    return 0;
}

static int refused(eg_status got, eg_status want, eg_ctx* ctx, const char* part) {
    return got == want && reason_has(ctx, part);
}

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[2];
};

static eg_plane* wrap(eg_ctx* ctx, struct tensor* t, void* data, int ndim, int64_t d0, int64_t d1, int cuda, int device_id) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = d0;
    t->shape[1] = d1;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = cuda ? kDLCUDA : kDLCPU;
    t->managed.dl_tensor.device.device_id = device_id;
    t->managed.dl_tensor.ndim = ndim;
    t->managed.dl_tensor.dtype.code = kDLFloat;
    t->managed.dl_tensor.dtype.bits = 64;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

static eg_ctx* context(uint32_t flags, int32_t device, eg_status* status) {
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.flags = flags;
    s.device = device;
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    *status = eg_ctx_new(&s, &ctx);
    return ctx;
}

static eg_status load(eg_ctx* ctx, const char* path, const char* name, uint8_t digest_byte, eg_unit** out) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memset(ref.digest, digest_byte, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = path;
    d.unit = &ref;
    *out = NULL;
    return eg_unit_load(ctx, &d, out);
}

static eg_status plan_n(eg_ctx* ctx, eg_unit* unit, eg_structure s, int64_t n, int64_t parts, const int64_t* base,
                        const int64_t* count, eg_plan** out) {
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = s;
    d.n_samples = n;
    d.n_partitions = parts;
    d.part_base = base;
    d.part_count = count;
    *out = NULL;
    return eg_plan_new(ctx, &d, out);
}

static eg_status plan(eg_ctx* ctx, eg_unit* unit, eg_structure s, int64_t parts, const int64_t* base,
                      const int64_t* count, eg_plan** out) {
    return plan_n(ctx, unit, s, N, parts, base, count, out);
}

/* A device buffer of @p count doubles holding @p values (NULL: zeros). */
static double* dev_doubles(const double* values, int count) {
    double* d = NULL;
    if (cudaMalloc((void**)&d, (size_t)count * sizeof(double)) != cudaSuccess) return NULL;
    if (values) cudaMemcpy(d, values, (size_t)count * sizeof(double), cudaMemcpyHostToDevice);
    else cudaMemset(d, 0, (size_t)count * sizeof(double));
    return d;
}

static eg_status bind_planes(eg_ctx* ctx, eg_plan* p, int n, const int32_t* slots, eg_plane* const* planes) {
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = n;
    b.plane_slots = slots;
    b.planes = (eg_plane**)planes;
    return eg_bind(ctx, p, &b);
}

/* MIN and MAX cells over rows of 4 samples, and a companion folded after each of two launches, on the device,
 * against plain-C oracles (bitwise); the refusals of the made planes' units. */
static void made_planes(eg_ctx* ctx, const char* dev_lib) {
    static const double xv[12] = { 3.5, -1.25, 7.0, 0.5, -9.75, 2.0, 2.0, 11.5, 0.125, -0.5, 6.25, 4.0 };
    eg_unit *cells = NULL, *keyed = NULL, *bad = NULL;
    check(load(ctx, dev_lib, "dcells", 0xD2, &cells) == EG_OK && load(ctx, dev_lib, "dkeyed", 0xD3, &keyed) == EG_OK,
          "eg_unit_load dcells and dkeyed (cells, a companion and its device fold function)");
    check(refused(load(ctx, dev_lib, "dkeyednf", 0xD4, &bad), EG_FAULT, ctx, "lacks the fold function 'absent__fold'"),
          "a device fold function the image lacks: FAULT naming it");
    eg_unit* host_fold = NULL;
    check(load(ctx, dev_lib, "dkeyedh", 0xD5, &host_fold) == EG_OK, "eg_unit_load of a unit with a host fold entry and no device fold");
    eg_plan* p = NULL;
    check(refused(plan_n(ctx, host_fold, EG_STRUCTURE_DEVICE, 12, 0, NULL, NULL, &p), EG_NOT_WRITTEN, ctx, "no device fold function"),
          "DEVICE plan of a companion with no device fold function refused NOT_WRITTEN by name");
    if (cells == NULL || keyed == NULL) return;
    double* dx = dev_doubles(xv, 12);
    struct tensor tx, tx10, tt, tm, ta;
    eg_plane* px = wrap(ctx, &tx, dx, 1, 12, 0, 1, 0);
    eg_plane* px10 = wrap(ctx, &tx10, dx, 1, 10, 0, 1, 0);
    /* cells over 10 samples (rows 4, 4, 2): whole, then two partitions that start on a row; x changes between runs */
    double *dt = dev_doubles(NULL, 1), *dm = dev_doubles(NULL, 1);
    eg_plane* pt = wrap(ctx, &tt, dt, 1, 1, 0, 1, 0);
    eg_plane* pm = wrap(ctx, &tm, dm, 1, 1, 0, 1, 0);
    const int64_t base[2] = { 0, 4 }, count[2] = { 4, 6 };
    int ok = px && px10 && pt && pm;
    for (int r = 0; r < 2 && ok; ++r) {
        double lo = 1e300, hi = -1e300, got[2] = { 0, 0 };
        for (int i = 0; i < 10; ++i) {
            const double v = xv[(i + 5 * r) % 12];
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        double shifted[12];
        for (int i = 0; i < 12; ++i) shifted[i] = xv[(i + 5 * r) % 12];
        cudaMemcpy(dx, shifted, sizeof(shifted), cudaMemcpyHostToDevice);
        const int32_t slots[3] = { 0, 1, 2 };
        eg_plane* planes[3] = { pt, pm, px10 };
        eg_status s = plan_n(ctx, cells, EG_STRUCTURE_DEVICE, 10, 2 * r, base, count, &p);
        if (s == EG_OK) s = bind_planes(ctx, p, 3, slots, planes);
        if (s == EG_OK) s = eg_launch(ctx, p, NULL);
        if (s == EG_OK) s = eg_sync(ctx);
        if (s != EG_OK) reason_has(ctx, "(the cells launch was refused)");
        cudaMemcpy(&got[0], dt, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&got[1], dm, sizeof(double), cudaMemcpyDeviceToHost);
        ok = s == EG_OK && memcmp(&got[0], &lo, sizeof(double)) == 0 && memcmp(&got[1], &hi, sizeof(double)) == 0;
        if (!ok) printf("      run %d: min %.17g (want %.17g), max %.17g (want %.17g)\n", r, got[0], lo, got[1], hi);
        if (p) eg_plan_free(ctx, p);
    }
    check(ok, "dcells: MIN and MAX of the rows folded into the caller's (1,) cells, whole and over two partitions, bitwise == the plain-C oracle");
    /* a companion over 12 samples (3 keys): two launches, the fold after each; acc keeps its value between them */
    double want[3] = { 0, 0, 0 }, comp[3] = { 0, 0, 0 };
    for (int launch = 0; launch < 2; ++launch) {
        for (int i = 0; i < 12; ++i) want[i / 4] += xv[i], comp[i / 4] += 0.5;
        for (int k = 0; k < 3; ++k) want[k] += comp[k], comp[k] = 0.125;
    }
    double* da = dev_doubles(NULL, 3);
    eg_plane* pa = wrap(ctx, &ta, da, 1, 3, 0, 1, 0);
    cudaMemcpy(dx, xv, sizeof(xv), cudaMemcpyHostToDevice);
    int match = pa != NULL;
    {
        const int32_t slots[2] = { 0, 2 };
        eg_plane* planes[2] = { pa, px };
        double got[3] = { -1, -1, -1 };
        cudaMemset(da, 0, sizeof(got));
        eg_status s = plan_n(ctx, keyed, EG_STRUCTURE_DEVICE, 12, 0, NULL, NULL, &p);
        if (s == EG_OK) s = bind_planes(ctx, p, 2, slots, planes);
        for (int launch = 0; launch < 2 && s == EG_OK; ++launch) s = eg_launch(ctx, p, NULL);
        if (s == EG_OK) s = eg_sync(ctx);
        if (s != EG_OK) reason_has(ctx, "(the companion launch was refused)");
        cudaMemcpy(got, da, sizeof(got), cudaMemcpyDeviceToHost);
        match = s == EG_OK && memcmp(got, want, sizeof(got)) == 0;
        if (!match) printf("      acc: %.17g %.17g %.17g, want %.17g %.17g %.17g\n", got[0], got[1], got[2], want[0], want[1], want[2]);
        if (p) eg_plan_free(ctx, p);
    }
    check(match, "dkeyed: the companion made zeroed at bind, folded into acc after each of two launches, bitwise == the plain-C oracle");
    eg_plane* cell_bad = NULL;
    struct tensor t2;
    double* d2 = dev_doubles(NULL, 2);
    cell_bad = wrap(ctx, &t2, d2, 1, 2, 0, 1, 0);
    {
        const int32_t slots[3] = { 0, 1, 2 };
        eg_plane* planes[3] = { cell_bad, pm, px10 };
        check(plan_n(ctx, cells, EG_STRUCTURE_DEVICE, 10, 0, NULL, NULL, &p) == EG_OK &&
                  refused(bind_planes(ctx, p, 3, slots, planes), EG_BAD_REQUEST, ctx, "a cell is one element"),
              "dcells: a cell of another shape than (1,) refused BAD_REQUEST by name");
        if (p) eg_plan_free(ctx, p);
    }
    {   /* a runner over a plan with a companion, or with cells, is refused by name (the loop's folds have no old-path oracle) */
        const int32_t kslots[2] = { 0, 2 };
        eg_plane* kplanes[2] = { pa, px };
        eg_plan *kp = NULL, *cp = NULL;
        struct eg_runner_desc rn;
        memset(&rn, 0, sizeof(rn));
        rn.struct_size = sizeof(rn);
        rn.n_plans = 1;
        rn.max_steps = 1;
        eg_runner* runner = NULL;
        int rok = plan_n(ctx, keyed, EG_STRUCTURE_DEVICE, 12, 0, NULL, NULL, &kp) == EG_OK &&
                  bind_planes(ctx, kp, 2, kslots, kplanes) == EG_OK;
        rn.plans = &kp;
        rok = rok && refused(eg_runner_new(ctx, &rn, &runner), EG_NOT_WRITTEN, ctx, "companion or cell folding inside a device runner loop");
        rok = rok && plan_n(ctx, cells, EG_STRUCTURE_DEVICE, 10, 0, NULL, NULL, &cp) == EG_OK;
        rn.plans = &cp;
        rok = rok && refused(eg_runner_new(ctx, &rn, &runner), EG_NOT_WRITTEN, ctx, "companion or cell folding inside a device runner loop");
        check(rok && runner == NULL, "eg_runner_new over a DEVICE plan with a companion, or with cells: NOT_WRITTEN by name");
        if (kp) eg_plan_free(ctx, kp);
        if (cp) eg_plan_free(ctx, cp);
    }
    eg_unit_free(ctx, cells);
    eg_unit_free(ctx, keyed);
    eg_unit_free(ctx, host_fold);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, px10);
    eg_plane_free(ctx, pt);
    eg_plane_free(ctx, pm);
    eg_plane_free(ctx, pa);
    eg_plane_free(ctx, cell_bad);
    cudaFree(dx);
    cudaFree(dt);
    cudaFree(dm);
    cudaFree(da);
    cudaFree(d2);
}

/* Bind y, x, v (planes, slots 0-2) and a (slot 3). */
static eg_status bind(eg_ctx* ctx, eg_plan* p, eg_plane* y, eg_plane* x, eg_plane* v, double a) {
    eg_plane* planes[3] = { y, x, v };
    const int32_t slots[3] = { 0, 1, 2 }, uslot[1] = { 3 };
    const void* values[1] = { &a };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 3;
    b.plane_slots = slots;
    b.planes = planes;
    b.n_uniforms = 1;
    b.uniform_slots = uslot;
    b.uniform_values = values;
    return eg_bind(ctx, p, &b);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <libfixture_device.so> <libfixture_unit.so>\n", argv[0]);
        return 2;
    }
    const char* dev_lib = argv[1];
    const char* host_lib = argv[2];
    static double x[N], v[2 * N], y[N], want[N];
    const double a = 1.0 / 3.0;
    for (int i = 0; i < N; ++i) {
        x[i] = 0.001 * i - 0.37;
        v[i] = 1.0 / (i + 1.0);
        v[N + i] = -0.25 * i;
    }
    for (int i = 0; i < N; ++i) want[i] = fma(a, x[i], v[i]) + (double)N * v[N + i];

    /* Contexts. */
    eg_status s = EG_OK;
    eg_ctx* host = context(0, 0, &s);
    check(s == EG_OK, "eg_ctx_new host context");
    eg_ctx* bad = context(EG_CTX_DEVICE, 99, &s);
    check(refused(s, EG_BAD_REQUEST, bad, "the driver sees"), "EG_CTX_DEVICE device 99 refused BAD_REQUEST (the driver sees fewer)");
    check(eg_ctx_free(bad) == EG_OK, "eg_ctx_free of the refused context");
    eg_ctx* ctx = context(EG_CTX_DEVICE, 0, &s);
    check(s == EG_OK, "eg_ctx_new EG_CTX_DEVICE device 0");
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    check(eg_ctx_info(ctx, &info) == EG_OK && info.device == 0 && (info.capabilities & EG_CAP_DEVICE) &&
              info.device_type == kDLCUDA && info.stream != -1 && info.stream != 0,
          "eg_ctx_info: device 0, EG_CAP_DEVICE, kDLCUDA, the context's own stream");
    check(eg_sync(host) == EG_OK && eg_sync(ctx) == EG_OK, "eg_sync on a host and an idle device context");
    check(eg_stream_use(ctx, 0) == EG_OK, "eg_stream_use 0 (eagle's own stream) on a device context");

    /* Units. */
    eg_unit *axpy = NULL, *axpy64 = NULL, *u = NULL;
    check(load(ctx, dev_lib, "axpy", 0xC1, &axpy) == EG_OK && load(ctx, dev_lib, "axpy64", 0xC2, &axpy64) == EG_OK,
          "eg_unit_load axpy and axpy64 (fatbin image, module and function per unit)");
    check(refused(load(ctx, dev_lib, "nofn", 0xC5, &u), EG_FAULT, ctx, "lacks the function 'missing'"),
          "a device function the image lacks: FAULT naming it");
    check(refused(load(ctx, dev_lib, "noimg", 0xC6, &u), EG_FAULT, ctx, "lacks its device image 'absent__image'"),
          "a device image symbol the library lacks: FAULT naming it");
    eg_unit* on_host = NULL;
    check(load(host, dev_lib, "axpy", 0xC1, &on_host) == EG_OK, "eg_unit_load axpy on a host context (no image loaded)");
    eg_plan* p = NULL;
    check(refused(plan(host, on_host, EG_STRUCTURE_DEVICE, 0, NULL, NULL, &p), EG_BAD_REQUEST, host, "host-only context"),
          "structure DEVICE on a host-only context refused BAD_REQUEST");

    /* Device planes. */
    double *dx = NULL, *dv = NULL, *dy = NULL;
    check(cudaMalloc((void**)&dx, sizeof(x)) == cudaSuccess && cudaMalloc((void**)&dv, sizeof(v)) == cudaSuccess &&
              cudaMalloc((void**)&dy, sizeof(y)) == cudaSuccess &&
              cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(dv, v, sizeof(v), cudaMemcpyHostToDevice) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
          "device buffers for x, v, y");
    struct tensor tx, tv, ty, tother, thost;
    eg_plane* px = wrap(ctx, &tx, dx, 1, N, 0, 1, 0);
    eg_plane* pv = wrap(ctx, &tv, dv, 2, 2, N, 1, 0);
    eg_plane* py = wrap(ctx, &ty, dy, 1, N, 0, 1, 0);
    check(px && pv && py, "eg_plane_wrap kDLCUDA planes of the context's device");
    eg_plane* other = NULL;
    memset(&tother, 0, sizeof(tother));
    tother.shape[0] = N;
    tother.managed.version.major = DLPACK_MAJOR_VERSION;
    tother.managed.dl_tensor.data = dy;
    tother.managed.dl_tensor.device.device_type = kDLCUDA;
    tother.managed.dl_tensor.device.device_id = 1;
    tother.managed.dl_tensor.ndim = 1;
    tother.managed.dl_tensor.dtype.code = kDLFloat;
    tother.managed.dl_tensor.dtype.bits = 64;
    tother.managed.dl_tensor.dtype.lanes = 1;
    tother.managed.dl_tensor.shape = tother.shape;
    check(refused(eg_plane_wrap(ctx, &tother.managed, &other), EG_BAD_REQUEST, ctx, "on the context of device 0"),
          "a kDLCUDA plane of another device refused BAD_REQUEST");
    eg_plane* hy = wrap(ctx, &thost, y, 1, N, 0, 0, 0);

    /* Launches against the plain-C oracle. */
    const int64_t base[2] = { 0, 300 }, count[2] = { 300, N - 300 };
    struct { eg_unit* unit; int64_t parts; const char* what; } runs[3] = {
        { axpy, 0, "axpy, DEVICE whole, eagle's block" },
        { axpy, 2, "axpy, DEVICE (0,300),(300,700)" },
        { axpy64, 0, "axpy64, DEVICE whole, the meta's block 64" },
    };
    for (int r = 0; r < 3; ++r) {
        memset(y, 0, sizeof(y));
        cudaMemset(dy, 0, sizeof(y));
        cudaDeviceSynchronize();
        uint64_t tiles = 7;
        s = plan(ctx, runs[r].unit, EG_STRUCTURE_DEVICE, runs[r].parts, runs[r].parts ? base : NULL,
                 runs[r].parts ? count : NULL, &p);
        if (s == EG_OK) s = bind(ctx, p, py, px, pv, a);
        if (s == EG_OK) s = eg_launch(ctx, p, &tiles);
        if (s == EG_OK) s = eg_sync(ctx);
        if (s != EG_OK) reason_has(ctx, "(the launch was refused)");
        cudaMemcpy(y, dy, sizeof(y), cudaMemcpyDeviceToHost);
        char what[160];
        snprintf(what, sizeof(what), "%s: launch, eg_sync, tiles 0, y bitwise == the plain-C oracle", runs[r].what);
        check(s == EG_OK && tiles == 0 && memcmp(y, want, sizeof(y)) == 0, what);
        if (r < 2) eg_plan_free(ctx, p);
    }

    /* Binding and operation refusals (p: the axpy64 DEVICE plan, bound). */
    eg_plan* hp = NULL;
    check(plan(ctx, axpy, EG_STRUCTURE_DEVICE, 0, NULL, NULL, &hp) == EG_OK &&
              refused(bind(ctx, hp, hy, px, pv, a), EG_BAD_REQUEST, ctx, "a host plane in a DEVICE plan"),
          "a host plane in a DEVICE plan refused BAD_REQUEST");
    eg_plan_free(ctx, hp);
    eg_unit* axpb = NULL;
    check(load(ctx, host_lib, "axpb", 0xA1, &axpb) == EG_OK, "eg_unit_load axpb (host fixture) on the device context");
    check(plan(ctx, axpb, EG_STRUCTURE_HOST_TEAM, 0, NULL, NULL, &hp) == EG_OK &&
              refused(bind(ctx, hp, py, px, NULL, a), EG_BAD_REQUEST, ctx, "a device plane in a host plan"),
          "a device plane in a host plan refused BAD_REQUEST");
    eg_plan_free(ctx, hp);
    check(refused(eg_copy(ctx, py, hy), EG_NOT_WRITTEN, ctx, "eg_copy of a device plane"), "eg_copy of a device plane: NOT_WRITTEN");
    struct eg_reduce_desc rd;
    memset(&rd, 0, sizeof(rd));
    rd.struct_size = sizeof(rd);
    rd.in = py;
    rd.out = hy;
    check(refused(eg_reduce(ctx, &rd), EG_NOT_WRITTEN, ctx, "eg_reduce of a device plane"), "eg_reduce of a device plane: NOT_WRITTEN");
    struct eg_runner_desc rn;
    memset(&rn, 0, sizeof(rn));
    rn.struct_size = sizeof(rn);
    rn.n_plans = 1;
    rn.plans = &p;
    rn.max_steps = 1;
    eg_runner* runner = NULL;
    check(refused(eg_runner_new(ctx, &rn, &runner), EG_BAD_REQUEST, ctx, "has no finish") && runner == NULL,
          "eg_runner_new over a DEVICE plan without a finish: BAD_REQUEST (the device runner checks as the host one)");
    eg_graph* g = NULL;
    check(eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, &g) == EG_OK &&
              eg_graph_free(ctx, g) == EG_OK,
          "recording a DEVICE launch: captured, eg_record_end and eg_graph_free OK");

    /* DEVICE units refused by name (no device image), and the features a DEVICE plan now takes: a thread box and a
     * BANDED scalar type (their pairs run in device_features_test). */
    struct { const char* name; uint8_t byte; const char* reason; } rows[8] = {
        { "boxdev", 0xC3, NULL },
        { "banded", 0xC4, NULL },
        { "count1", 0xD1, "the unit has no device image" },   /* a fixed-step finish runs on a device */
        { "count1a", 0xE1, "the unit has no device image" },
        { "lanes", 0x91, "the unit has no device image" },
        { "keyed", 0x68, "the unit has no device image" },
        { "cells", 0x67, "the unit has no device image" },
        { "axpb", 0xA1, "the unit has no device image" },
    };
    for (int r = 0; r < 8; ++r) {
        eg_unit* unit = NULL;
        const char* path = r < 2 ? dev_lib : host_lib;
        s = load(ctx, path, rows[r].name, rows[r].byte, &unit);
        char what[160];
        eg_plan* made = NULL;
        if (rows[r].reason == NULL) {
            snprintf(what, sizeof(what), "DEVICE plan of %s accepted", rows[r].name);
            check(s == EG_OK && plan(ctx, unit, EG_STRUCTURE_DEVICE, 0, NULL, NULL, &made) == EG_OK && eg_plan_free(ctx, made) == EG_OK, what);
        } else {
            snprintf(what, sizeof(what), "DEVICE plan of %s refused NOT_WRITTEN: %s", rows[r].name, rows[r].reason);
            check(s == EG_OK && refused(plan(ctx, unit, EG_STRUCTURE_DEVICE, 0, NULL, NULL, &hp), EG_NOT_WRITTEN, ctx, rows[r].reason),
                  what);
        }
        eg_unit_free(ctx, unit);
    }

    made_planes(ctx, dev_lib);

    /* The host still runs on a device context. */
    static double hx[N], hyv[N];
    for (int i = 0; i < N; ++i) hx[i] = x[i];
    struct tensor thx, thy;
    eg_plane* phx = wrap(ctx, &thx, hx, 1, N, 0, 0, 0);
    eg_plane* phy = wrap(ctx, &thy, hyv, 1, N, 0, 0, 0);
    eg_plane* hplanes[2] = { phy, phx };
    const int32_t hslots[2] = { 0, 1 }, huslots[2] = { 2, 3 };
    const double b = 0.5;
    const void* hvalues[2] = { &a, &b };
    struct eg_bind_desc hb;
    memset(&hb, 0, sizeof(hb));
    hb.struct_size = sizeof(hb);
    hb.n_planes = 2;
    hb.plane_slots = hslots;
    hb.planes = hplanes;
    hb.n_uniforms = 2;
    hb.uniform_slots = huslots;
    hb.uniform_values = hvalues;
    s = plan(ctx, axpb, EG_STRUCTURE_HOST_TEAM, 0, NULL, NULL, &hp);
    if (s == EG_OK) s = eg_bind(ctx, hp, &hb);
    if (s == EG_OK) s = eg_launch(ctx, hp, NULL);
    int same = 1;
    for (int i = 0; i < N; ++i) same &= hyv[i] == a * hx[i] + b;
    check(s == EG_OK && same, "axpb HOST_TEAM with host planes on the device context: y == a * x + b");

    /* Freeing with live handles: modules unloaded, stream destroyed, the primary context released. */
    check(eg_unit_free(ctx, axpy) == EG_OK && eg_unit_free(ctx, axpy64) == EG_OK && eg_unit_free(ctx, axpb) == EG_OK &&
              eg_plane_free(ctx, px) == EG_OK,
          "eg_unit_free and eg_plane_free while plans still hold them");
    check(eg_ctx_free(ctx) == EG_OK && eg_ctx_free(host) == EG_OK, "eg_ctx_free of the device context with live handles");
    cudaFree(dx);
    cudaFree(dv);
    cudaFree(dy);
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
