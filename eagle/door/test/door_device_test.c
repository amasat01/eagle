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
#include "seg_model.h"

#ifndef EG_TEST_DIR
#define EG_TEST_DIR "test" /* the directory of the recorded decisions; the Makefile gives its absolute path */
#endif

#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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
    {   /* eg_clear on the device: acc and its companion restart in stream order; a launch after the clear == the first launch of a fresh plan */
        double first[3], got[3] = { -1, -1, -1 };
        for (int k = 0; k < 3; ++k) first[k] = xv[4 * k] + xv[4 * k + 1] + xv[4 * k + 2] + xv[4 * k + 3] + 2.0;
        const int32_t slots[2] = { 0, 2 }, out[1] = { 0 }, comp_slot[1] = { 1 };
        eg_plane* planes[2] = { pa, px };
        double zeroed[3] = { -1, -1, -1 };
        cudaMemset(da, 0, sizeof(got));
        eg_status s = plan_n(ctx, keyed, EG_STRUCTURE_DEVICE, 12, 0, NULL, NULL, &p);
        if (s == EG_OK) s = bind_planes(ctx, p, 2, slots, planes);
        for (int launch = 0; launch < 2 && s == EG_OK; ++launch) s = eg_launch(ctx, p, NULL);
        if (s == EG_OK) s = eg_clear(ctx, p, 1, out);
        if (s == EG_OK) s = eg_sync(ctx);
        cudaMemcpy(zeroed, da, sizeof(zeroed), cudaMemcpyDeviceToHost);
        if (s == EG_OK) s = eg_launch(ctx, p, NULL);
        if (s == EG_OK) s = eg_sync(ctx);
        cudaMemcpy(got, da, sizeof(got), cudaMemcpyDeviceToHost);
        int ok = s == EG_OK && zeroed[0] == 0.0 && zeroed[1] == 0.0 && zeroed[2] == 0.0 && memcmp(got, first, sizeof(got)) == 0;
        ok = ok && eg_clear(ctx, p, 0, NULL) == EG_OK;
        ok = ok && refused(eg_clear(ctx, p, 1, comp_slot), EG_BAD_REQUEST, ctx, "'acc_c') is a companion or compensation plane");
        ok = ok && refused(eg_clear(ctx, p, 1, slots + 1), EG_BAD_REQUEST, ctx, "is not an output plane");
        if (!ok) printf("      cleared: %.17g %.17g %.17g, after relaunch %.17g %.17g %.17g\n", zeroed[0], zeroed[1], zeroed[2], got[0], got[1], got[2]);
        check(ok, "eg_clear on a DEVICE plan: the output zero and the companion restarted in stream order (a launch after == a fresh plan, bitwise), "
                  "n_slots 0 a no-op, a companion or an input slot refused BAD_REQUEST by reason");
        if (p) eg_plan_free(ctx, p);
    }
    {   /* the free of an eagle-made plane is stream-ordered: it returns while work queued before it still runs */
        cudaStream_t own = NULL;
        void *big_a = NULL, *big_b = NULL;
        const size_t big = (size_t)256 << 20;
        int ok = cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc(&big_a, big) == cudaSuccess &&
                 cudaMalloc(&big_b, big) == cudaSuccess && eg_stream_use(ctx, (uint64_t)(uintptr_t)own) == EG_OK;
        const int32_t slots[2] = { 0, 2 };
        eg_plane* planes[2] = { pa, px };
        ok = ok && plan_n(ctx, keyed, EG_STRUCTURE_DEVICE, 12, 0, NULL, NULL, &p) == EG_OK && bind_planes(ctx, p, 2, slots, planes) == EG_OK;
        for (int k = 0; ok && k < 100; ++k) ok = cudaMemcpyAsync(big_b, big_a, big, cudaMemcpyDeviceToDevice, own) == cudaSuccess;
        ok = ok && cudaStreamQuery(own) == cudaErrorNotReady;   /* the queued copies are still running before the free */
        ok = ok && eg_plan_free(ctx, p) == EG_OK;
        p = NULL;
        const int returned_early = ok && cudaStreamQuery(own) == cudaErrorNotReady;
        ok = ok && cudaStreamSynchronize(own) == cudaSuccess && eg_stream_use(ctx, 0) == EG_OK;
        check(ok && returned_early, "freeing a plan's eagle-made companion plane returns while the stream's queued work still runs (stream-ordered free)");
        cudaFree(big_a);
        cudaFree(big_b);
        cudaStreamDestroy(own);
    }
    {   /* a plane allocated on stream A and used there is freed after A's work although stream B is in scope: B reuses its bytes */
        cudaStream_t sa = NULL, sb = NULL;
        void *big_a = NULL, *big_b = NULL, *reuse = NULL;
        const size_t big = (size_t)256 << 20;
        double first[3], got[3] = { -1, -1, -1 };
        for (int k = 0; k < 3; ++k) first[k] = xv[4 * k] + xv[4 * k + 1] + xv[4 * k + 2] + xv[4 * k + 3] + 2.0;
        int ok = cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking) == cudaSuccess &&
                 cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc(&big_a, big) == cudaSuccess &&
                 cudaMalloc(&big_b, big) == cudaSuccess && eg_stream_use(ctx, (uint64_t)(uintptr_t)sa) == EG_OK;
        const int32_t slots[2] = { 0, 2 };
        eg_plane* planes[2] = { pa, px };
        cudaMemset(da, 0, sizeof(got));
        ok = ok && plan_n(ctx, keyed, EG_STRUCTURE_DEVICE, 12, 0, NULL, NULL, &p) == EG_OK && bind_planes(ctx, p, 2, slots, planes) == EG_OK;
        for (int k = 0; ok && k < 100; ++k) ok = cudaMemcpyAsync(big_b, big_a, big, cudaMemcpyDeviceToDevice, sa) == cudaSuccess;
        ok = ok && eg_launch(ctx, p, NULL) == EG_OK;    /* on A after the copies: reads and folds the companion plane */
        ok = ok && eg_stream_use(ctx, (uint64_t)(uintptr_t)sb) == EG_OK && eg_plan_free(ctx, p) == EG_OK;
        p = NULL;
        ok = ok && cudaMallocAsync(&reuse, 24, sb) == cudaSuccess && cudaMemsetAsync(reuse, 0xFF, 24, sb) == cudaSuccess;
        ok = ok && cudaStreamSynchronize(sa) == cudaSuccess && cudaStreamSynchronize(sb) == cudaSuccess && eg_stream_use(ctx, 0) == EG_OK;
        cudaMemcpy(got, da, sizeof(got), cudaMemcpyDeviceToHost);
        check(ok && memcmp(got, first, sizeof(got)) == 0,
              "a plane made on stream A, used there and freed with stream B in scope: A's work completes with the right values (B reusing the bytes cannot corrupt them)");
        if (reuse) cudaFreeAsync(reuse, sb);
        cudaStreamSynchronize(sb);
        cudaFree(big_a);
        cudaFree(big_b);
        cudaStreamDestroy(sa);
        cudaStreamDestroy(sb);
    }
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

/* A free asked while a launch of @p keep is recorded waits for eg_record_end: the capture is not invalidated by it. The freed
 * plan holds an eagle-made companion plane, so its free reaches the device allocator. */
static void free_while_recording(eg_ctx* ctx, const char* dev_lib, eg_plan* keep) {
    static const double xv[12] = { 3.5, -1.25, 7.0, 0.5, -9.75, 2.0, 2.0, 11.5, 0.125, -0.5, 6.25, 4.0 };
    eg_unit* keyed = NULL;
    eg_plan* drop = NULL;
    eg_graph* g = NULL;
    struct tensor tx, ta;
    double *dx = dev_doubles(xv, 12), *da = dev_doubles(NULL, 3);
    eg_plane* px = wrap(ctx, &tx, dx, 1, 12, 0, 1, 0);
    eg_plane* pa = wrap(ctx, &ta, da, 1, 3, 0, 1, 0);
    const int32_t slots[2] = { 0, 2 };
    eg_plane* planes[2] = { pa, px };
    eg_status s = load(ctx, dev_lib, "dkeyed", 0xD3, &keyed);
    if (s == EG_OK) s = plan_n(ctx, keyed, EG_STRUCTURE_DEVICE, 12, 0, NULL, NULL, &drop);
    if (s == EG_OK) s = bind_planes(ctx, drop, 2, slots, planes);
    if (s == EG_OK) s = eg_record_begin(ctx);
    if (s == EG_OK) {
        s = eg_launch(ctx, keep, NULL);
        if (s == EG_OK) s = eg_plan_free(ctx, drop);
        if (s == EG_OK) drop = NULL;
        const eg_status ended = eg_record_end(ctx, NULL, &g);
        if (s == EG_OK) s = ended;
    }
    if (s == EG_OK) s = eg_graph_launch(ctx, g);
    if (s == EG_OK) s = eg_sync(ctx);
    if (s == EG_OK) s = eg_graph_free(ctx, g);
    if (s != EG_OK) reason_has(ctx, "(the recording row was refused)");
    check(s == EG_OK, "a plan with a companion freed while another launch is recorded: the free waits for eg_record_end, the graph is recorded, launched and freed OK");
    if (drop) eg_plan_free(ctx, drop);
    eg_unit_free(ctx, keyed);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, pa);
    cudaFree(dx);
    cudaFree(da);
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

/* eg_set_count on a DEVICE prefix set: the write is queued on the stream (no sync between it and the launches); the lanes below
 * the count take the new values and the rest keep theirs, against the plain-C values bitwise. */
static void set_count_device(eg_ctx* ctx, const char* dev_lib) {
    enum { SM = 64 };
    double x[SM], y0[SM], want[SM], got[SM];
    eg_unit* unit = NULL;
    check(load(ctx, dev_lib, "dprefix", 0xD6, &unit) == EG_OK, "eg_unit_load dprefix (a prefix set on the device)");
    if (unit == NULL) return;
    for (int i = 0; i < SM; ++i) x[i] = (double)i * 0.5 - 3.0, y0[i] = -1.0;
    double *dy = dev_doubles(y0, SM), *dx = dev_doubles(x, SM);
    uint32_t* dc = NULL;
    cudaMalloc((void**)&dc, sizeof(uint32_t));
    struct tensor ty, tx, tc;
    eg_plane *py = wrap(ctx, &ty, dy, 1, SM, 0, 1, 0), *px = wrap(ctx, &tx, dx, 1, SM, 0, 1, 0);
    memset(&tc, 0, sizeof(tc));
    tc.shape[0] = 1;
    tc.managed.version.major = DLPACK_MAJOR_VERSION;
    tc.managed.dl_tensor.data = dc;
    tc.managed.dl_tensor.device.device_type = kDLCUDA;
    tc.managed.dl_tensor.ndim = 1;
    tc.managed.dl_tensor.dtype.code = kDLUInt;
    tc.managed.dl_tensor.dtype.bits = 32;
    tc.managed.dl_tensor.dtype.lanes = 1;
    tc.managed.dl_tensor.shape = tc.shape;
    eg_plane* pc = NULL;
    int ok = py && px && eg_plane_wrap(ctx, &tc.managed, &pc) == EG_OK;
    eg_plan* p = NULL;
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* planes[3] = { py, px, pc };
    ok = ok && plan_n(ctx, unit, EG_STRUCTURE_DEVICE, SM, 0, NULL, NULL, &p) == EG_OK && bind_planes(ctx, p, 3, slots, planes) == EG_OK;
    uint32_t cell = 99;
    ok = ok && eg_set_count(ctx, p, SM) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
    for (int i = 0; i < SM; ++i) want[i] = 2.0 * x[i] + 1.0;
    ok = ok && memcmp(got, want, sizeof(got)) == 0;
    for (int i = 0; i < SM; ++i) x[i] = -x[i] - 7.0;
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
    ok = ok && eg_set_count(ctx, p, 21) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cell, dc, sizeof(cell), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 21; ++i) want[i] = 2.0 * x[i] + 1.0;
    ok = ok && cell == 21 && memcmp(got, want, sizeof(got)) == 0;
    for (int i = 0; i < SM; ++i) x[i] += 100.0;
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
    ok = ok && eg_set_count(ctx, p, 0) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
    ok = ok && memcmp(got, want, sizeof(got)) == 0;
    if (!ok) printf("      lane 0 %.17g (want %.17g), lane 40 %.17g (want %.17g)\n", got[0], want[0], got[40], want[40]);
    check(ok, "eg_set_count on a DEVICE prefix set: lanes below the count take the new values, the others keep theirs bitwise, 0 runs nothing");
    ok = refused(eg_set_count(ctx, p, SM + 1), EG_BAD_REQUEST, ctx, "above the sample count's 64 entries");
    cudaMemcpy(&cell, dc, sizeof(cell), cudaMemcpyDeviceToHost);
    check(ok && cell == 0, "eg_set_count above the capacity on a DEVICE plan: refused BAD_REQUEST, the cell unchanged");
    {   /* a graph recorded at one count replays at each later count: the kernel reads the count plane, nothing is baked in */
        const uint32_t counts[4] = { 40, 10, 0, SM };
        eg_graph* g = NULL;
        for (int i = 0; i < SM; ++i) x[i] = 0.25 * i + 1.0;
        cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
        ok = eg_set_count(ctx, p, counts[0]) == EG_OK && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK &&
             eg_record_end(ctx, NULL, &g) == EG_OK;
        for (int k = 0; ok && k < 4; ++k) {
            ok = (k == 0 || eg_set_count(ctx, p, counts[k]) == EG_OK) && cudaMemcpy(dy, y0, sizeof(y0), cudaMemcpyHostToDevice) == cudaSuccess &&
                 eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
            cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
            for (int i = 0; i < SM; ++i) want[i] = (uint32_t)i < counts[k] ? 2.0 * x[i] + 1.0 : y0[i];
            ok = ok && memcmp(got, want, sizeof(got)) == 0;
        }
        check(ok, "a DEVICE graph of a prefix set recorded at count 40 replays at 40, 10, 0 and the capacity 64 (eg_set_count between replays, no new recording): "
                  "lanes below the count new, the others old, bitwise == the plain-C values");
        eg_graph_free(ctx, g);
    }
    if (p) eg_plan_free(ctx, p);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, pc);
    eg_unit_free(ctx, unit);
}

/* ---- host steps inside a device recording ---------------------------------------------------------------------------- */

/* A one-dimensional host or device plane of @p n elements of type (@p code, @p bits) over @p data. */
static eg_plane* wrap_typed(eg_ctx* ctx, struct tensor* t, void* data, uint8_t code, uint8_t bits, int64_t n, int cuda) {
    eg_plane* p = wrap(ctx, t, data, 1, n, 0, cuda, 0);
    if (p == NULL) return NULL;
    eg_plane_free(ctx, p);    /* re-wrap with the element type: wrap() makes float64 */
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
    p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

/* A device plan, a host plan, a device plan recorded once: each replay runs the three in stream order. The host plan counts
 * its launches in a host cell; the device plans chain y2 = 2 (2 x + 1) + 1. A host step whose launch is refused at the replay
 * (an index count above its entries) reports FAULT naming it at the next eg_sync, once. The graph freed while replays are queued
 * waits for them: their host steps run before the references go. */
static void host_steps_in_graph(eg_ctx* ctx, const char* dev_lib, const char* host_lib) {
    enum { HN = 64, REPLAYS = 3 };
    eg_unit *dp = NULL, *tick = NULL, *prefix = NULL;
    check(load(ctx, dev_lib, "dprefix", 0xD6, &dp) == EG_OK && load(ctx, host_lib, "tick", 0xB8, &tick) == EG_OK &&
              load(ctx, host_lib, "prefix", 0x92, &prefix) == EG_OK,
          "eg_unit_load dprefix (device), tick and prefix (host) for host steps in a device graph");
    if (dp == NULL || tick == NULL || prefix == NULL) return;
    double x[HN], got[HN], want[HN];
    for (int i = 0; i < HN; ++i) x[i] = 0.5 * i - 7.0, want[i] = 2.0 * (2.0 * x[i] + 1.0) + 1.0;
    uint32_t count = HN;
    int64_t ticks = 0;
    double *dx = dev_doubles(x, HN), *d1 = dev_doubles(NULL, HN), *d2 = dev_doubles(NULL, HN);
    uint32_t* dc = NULL;
    cudaMalloc((void**)&dc, sizeof(count));
    cudaMemcpy(dc, &count, sizeof(count), cudaMemcpyHostToDevice);
    struct tensor tx, t1, t2, tc, tt;
    eg_plane *px = wrap(ctx, &tx, dx, 1, HN, 0, 1, 0), *p1 = wrap(ctx, &t1, d1, 1, HN, 0, 1, 0), *p2 = wrap(ctx, &t2, d2, 1, HN, 0, 1, 0),
             *pc = wrap_typed(ctx, &tc, dc, kDLUInt, 32, 1, 1), *pt = wrap_typed(ctx, &tt, &ticks, kDLInt, 64, 1, 0);
    eg_plan *a = NULL, *b = NULL, *h = NULL;
    const int32_t slots[3] = { 0, 1, 2 }, cell_slot[1] = { 0 };
    eg_plane* a_planes[3] = { p1, px, pc };
    eg_plane* b_planes[3] = { p2, p1, pc };
    eg_plane* h_planes[1] = { pt };
    int ok = px && p1 && p2 && pc && pt && plan_n(ctx, dp, EG_STRUCTURE_DEVICE, HN, 0, NULL, NULL, &a) == EG_OK &&
             plan_n(ctx, dp, EG_STRUCTURE_DEVICE, HN, 0, NULL, NULL, &b) == EG_OK && bind_planes(ctx, a, 3, slots, a_planes) == EG_OK &&
             bind_planes(ctx, b, 3, slots, b_planes) == EG_OK && plan_n(ctx, tick, EG_STRUCTURE_HOST_SERIAL, 1, 0, NULL, NULL, &h) == EG_OK &&
             bind_planes(ctx, h, 1, cell_slot, h_planes) == EG_OK;
    check(ok, "host steps: two device plans and a host plan made and bound on a device context");
    if (!ok) return;
    /* the eager sequence, once */
    ok = eg_launch(ctx, a, NULL) == EG_OK && eg_launch(ctx, h, NULL) == EG_OK && eg_launch(ctx, b, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, d2, sizeof(got), cudaMemcpyDeviceToHost);
    ok = ok && ticks == 1 && memcmp(got, want, sizeof(got)) == 0;
    check(ok, "host steps: the eager sequence (device plan, host plan, device plan) gives the chained values and one tick");
    ticks = 0;
    cudaMemset(d1, 0, sizeof(got));    /* the legacy stream is not ordered against the context's stream: settle the reset first */
    cudaMemset(d2, 0, sizeof(got));
    cudaDeviceSynchronize();
    eg_graph* g = NULL;
    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, a, NULL) == EG_OK && eg_launch(ctx, h, NULL) == EG_OK &&
         eg_launch(ctx, b, NULL) == EG_OK && eg_record_end(ctx, NULL, &g) == EG_OK;
    if (!ok) reason_has(ctx, "(the recording was refused)");
    check(ok && ticks == 0, "host steps: a host launch recorded inside a device recording runs nothing at record time");
    for (int k = 0; ok && k < REPLAYS; ++k) ok = eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, d2, sizeof(got), cudaMemcpyDeviceToHost);
    int first_diff = -1;
    for (int i = 0; i < HN && first_diff < 0; ++i)
        if (memcmp(&got[i], &want[i], sizeof(double)) != 0) first_diff = i;
    if (ticks != REPLAYS || first_diff >= 0)
        printf("      ticks %lld (want %d), first differing value at %d: got %.17g want %.17g\n", (long long)ticks, REPLAYS, first_diff,
               first_diff >= 0 ? got[first_diff] : 0.0, first_diff >= 0 ? want[first_diff] : 0.0);
    check(ok && ticks == REPLAYS, "host steps: 3 replays of device plan, host plan, device plan: the host plan ran 3 times");
    check(ok && first_diff < 0, "host steps: after the 3 replays the values equal the eager sequence bitwise");
    /* freed while replays are queued */
    ticks = 0;
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && eg_graph_launch(ctx, g) == EG_OK && eg_graph_launch(ctx, g) == EG_OK &&
         eg_graph_free(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
    check(ok && ticks == REPLAYS, "host steps: a graph freed with 3 replays queued is safe: its host steps ran 3 times before the references went");
    /* a failing step */
    double hx[8] = { 1, 2, 3, 4, 5, 6, 7, 8 }, hy[8] = { 0 };
    uint32_t hcount = 8;
    struct tensor thx, thy, thc;
    eg_plane *phx = wrap(ctx, &thx, hx, 1, 8, 0, 0, 0), *phy = wrap(ctx, &thy, hy, 1, 8, 0, 0, 0),
             *phc = wrap_typed(ctx, &thc, &hcount, kDLUInt, 32, 1, 0);
    eg_plane* f_planes[3] = { phy, phx, phc };
    eg_plan* f = NULL;
    eg_graph* gf = NULL;
    ok = phx && phy && phc && plan_n(ctx, prefix, EG_STRUCTURE_HOST_SERIAL, 8, 0, NULL, NULL, &f) == EG_OK &&
         bind_planes(ctx, f, 3, slots, f_planes) == EG_OK && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, f, NULL) == EG_OK &&
         eg_record_end(ctx, NULL, &gf) == EG_OK;
    hcount = 9;    /* above the 8 entries: the host launch is refused when the step runs */
    ok = ok && eg_graph_launch(ctx, gf) == EG_OK;
    check(ok && refused(eg_sync(ctx), EG_FAULT, ctx, "eg_launch of unit 'prefix'") && reason_has(ctx, "index count 9"),
          "host steps: a step refused at the replay reports FAULT at the next eg_sync, naming the step and the cause");
    check(eg_sync(ctx) == EG_OK, "host steps: the failure is cleared once reported");
    hcount = 8;
    check(ok && eg_graph_launch(ctx, gf) == EG_OK && eg_sync(ctx) == EG_OK && hy[7] == 17.0, "host steps: the graph replays OK once the cause is gone");
    if (gf) eg_graph_free(ctx, gf);
    eg_plan_free(ctx, f);
    eg_plan_free(ctx, h);
    eg_plan_free(ctx, a);
    eg_plan_free(ctx, b);
    eg_plane_free(ctx, phx), eg_plane_free(ctx, phy), eg_plane_free(ctx, phc);
    eg_plane_free(ctx, px), eg_plane_free(ctx, p1), eg_plane_free(ctx, p2), eg_plane_free(ctx, pc), eg_plane_free(ctx, pt);
    eg_unit_free(ctx, dp), eg_unit_free(ctx, tick), eg_unit_free(ctx, prefix);
    cudaFree(dx), cudaFree(d1), cudaFree(d2), cudaFree(dc);
}

/* ---- host nodes on a device context ---- */

struct node_log {
    int calls, device_views, released;
};

/* y *= 3 in place over HN device doubles, through the CUDA runtime: the view is device memory the door made ready by synchronizing the stream. */
static int32_t node_triple(void* user, int32_t n, const void* const* planes) {
    struct node_log* log = (struct node_log*)user;
    const DLTensor* t = (const DLTensor*)planes[0];
    double y[64];
    ++log->calls;
    if (n != 1) return 1;
    log->device_views += t->device.device_type == kDLCUDA;
    char* data = (char*)t->data + t->byte_offset;
    cudaMemcpy(y, data, sizeof(y), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 64; ++i) y[i] *= 3.0;
    cudaMemcpy(data, y, sizeof(y), cudaMemcpyHostToDevice);
    return 0;
}

static void node_release(void* user) { ++((struct node_log*)user)->released; }

/* launch (p1 = 2 x + 1) -> host node (p1 *= 3) -> launch (p2 = 2 p1 + 1): the node is a segment between two captures; recorded and replayed, bitwise equal to the same
 * work done eagerly; the description names it; a node inside a fork branch is refused by name; the release runs once. */
static void host_nodes_device(eg_ctx* ctx, const char* dev_lib) {
    enum { HN = 64, REPLAYS = 3 };
    eg_unit* dp = NULL;
    check(load(ctx, dev_lib, "dprefix", 0xD6, &dp) == EG_OK, "eg_unit_load dprefix for host nodes in a device graph");
    if (dp == NULL) return;
    double x[HN], got[HN], want[HN], eager[HN];
    for (int i = 0; i < HN; ++i) x[i] = 0.5 * i - 7.0, want[i] = 2.0 * (3.0 * (2.0 * x[i] + 1.0)) + 1.0;
    uint32_t count = HN;
    double *dx = dev_doubles(x, HN), *d1 = dev_doubles(NULL, HN), *d2 = dev_doubles(NULL, HN);
    uint32_t* dc = NULL;
    cudaMalloc((void**)&dc, sizeof(count));
    cudaMemcpy(dc, &count, sizeof(count), cudaMemcpyHostToDevice);
    struct tensor tx, t1, t2, tc;
    eg_plane *px = wrap(ctx, &tx, dx, 1, HN, 0, 1, 0), *p1 = wrap(ctx, &t1, d1, 1, HN, 0, 1, 0), *p2 = wrap(ctx, &t2, d2, 1, HN, 0, 1, 0),
             *pc = wrap_typed(ctx, &tc, dc, kDLUInt, 32, 1, 1);
    eg_plan *a = NULL, *b = NULL;
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* a_planes[3] = { p1, px, pc };
    eg_plane* b_planes[3] = { p2, p1, pc };
    struct node_log log = { 0 };
    struct eg_host_node_desc nd;
    memset(&nd, 0, sizeof(nd));
    nd.struct_size = sizeof(nd);
    nd.name = "triple";
    nd.fn = node_triple;
    nd.user = &log;
    nd.release = node_release;
    eg_host_node* node = NULL;
    int ok = px && p1 && p2 && pc && plan_n(ctx, dp, EG_STRUCTURE_DEVICE, HN, 0, NULL, NULL, &a) == EG_OK && plan_n(ctx, dp, EG_STRUCTURE_DEVICE, HN, 0, NULL, NULL, &b) == EG_OK &&
             bind_planes(ctx, a, 3, slots, a_planes) == EG_OK && bind_planes(ctx, b, 3, slots, b_planes) == EG_OK && eg_host_node_new(ctx, &nd, &node) == EG_OK;
    check(ok, "host nodes: two device plans, a host node and the planes made on a device context");
    if (!ok) return;
    eg_plane* decl[1] = { p1 };
    const uint32_t rw = EG_ACCESS_READ | EG_ACCESS_WRITE;
    ok = eg_launch(ctx, a, NULL) == EG_OK && eg_host_call(ctx, node, 1, decl, &rw) == EG_OK && eg_launch(ctx, b, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(eager, d2, sizeof(eager), cudaMemcpyDeviceToHost);
    check(ok && log.calls == 1 && log.device_views == 1 && memcmp(eager, want, sizeof(want)) == 0,
          "an eager host call between two device launches: the callback saw a device plane with the first launch's values, and the chain equals the closed form bitwise");
    cudaMemset(d1, 0, sizeof(got));
    cudaMemset(d2, 0, sizeof(got));
    cudaDeviceSynchronize();
    log.calls = 0;
    eg_graph* g = NULL;
    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, a, NULL) == EG_OK && eg_host_call(ctx, node, 1, decl, &rw) == EG_OK && eg_launch(ctx, b, NULL) == EG_OK &&
         eg_record_end(ctx, NULL, &g) == EG_OK && log.calls == 0;
    for (int k = 0; ok && k < REPLAYS; ++k) ok = eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, d2, sizeof(got), cudaMemcpyDeviceToHost);
    check(ok && log.calls == REPLAYS && memcmp(got, eager, sizeof(got)) == 0,
          "launch -> host node -> launch recorded on a device context and replayed 3 times: the callback ran once per replay and the values equal the eager work, bitwise");
    struct eg_graph_description d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    ok = ok && eg_graph_describe(ctx, g, &d) == EG_OK && d.n_nodes == 3 && d.nodes[1].kind == EG_GNODE_HOST && d.nodes[1].unit != NULL && strcmp(d.nodes[1].unit, "triple") == 0 &&
         d.nodes[1].n_planes == 1 && d.nodes[1].planes[0] == p1 && d.nodes[1].access[0] == 3;
    check(ok, "eg_graph_describe of a device graph: the host node is EG_GNODE_HOST named 'triple' over its declared plane");
    ok = eg_record_begin(ctx) == EG_OK && eg_record_fork_begin(ctx) == EG_OK && eg_launch(ctx, a, NULL) == EG_OK && eg_record_fork_branch(ctx) == EG_OK;
    check(ok && refused(eg_host_call(ctx, node, 1, decl, &rw), EG_NOT_WRITTEN, ctx, "host node 'triple'") && reason_has(ctx, "a fork branch") && eg_record_abort(ctx) == EG_OK,
          "a host node inside a fork branch of a device recording is refused NOT_WRITTEN by name");
    check(eg_host_node_free(ctx, node) == EG_OK && log.released == 0 && eg_graph_free(ctx, g) == EG_OK && log.released == 1,
          "a host node freed while a device graph holds it is released once, when the graph is freed");
    eg_plan_free(ctx, a);
    eg_plan_free(ctx, b);
    eg_plane_free(ctx, px), eg_plane_free(ctx, p1), eg_plane_free(ctx, p2), eg_plane_free(ctx, pc);
    eg_unit_free(ctx, dp);
    cudaFree(dx), cudaFree(d1), cudaFree(d2), cudaFree(dc);
}

/* A launch writes each cell as the reduction over the lanes it RAN: a DEVICE prefix set of live count 8, 3, 0, 5, 0 over a SUM cell of rows
 * of 4 gives the sum of the first count lanes (0 for none), never the cell's old value or a stale row. */
static void cell_rows_device(eg_ctx* ctx, const char* dev_lib) {
    enum { CN = 8 };
    eg_unit* unit = NULL;
    check(load(ctx, dev_lib, "dprefixsum", 0xDB, &unit) == EG_OK, "eg_unit_load dprefixsum (a prefix set with a SUM cell of rows of 4)");
    if (unit == NULL) return;
    double x[CN], tot = 0.0;
    for (int i = 0; i < CN; ++i) x[i] = (double)(i + 1);
    const double nine = 9.0;
    double *dx = dev_doubles(x, CN), *dt = dev_doubles(&nine, 1);
    uint32_t* dc = NULL;
    cudaMalloc((void**)&dc, sizeof(uint32_t));
    struct tensor tt, tx, tc;
    eg_plane *pt = wrap(ctx, &tt, dt, 1, 1, 0, 1, 0), *px = wrap(ctx, &tx, dx, 1, CN, 0, 1, 0), *pc = wrap_typed(ctx, &tc, dc, kDLUInt, 32, 1, 1);
    const int32_t slots[3] = { 0, 2, 3 };
    eg_plane* planes[3] = { pt, px, pc };
    eg_plan* p = NULL;
    int ok = pt && px && pc && plan_n(ctx, unit, EG_STRUCTURE_DEVICE, CN, 0, NULL, NULL, &p) == EG_OK && bind_planes(ctx, p, 3, slots, planes) == EG_OK;
    const uint32_t counts[5] = { 8, 3, 0, 5, 0 };
    for (int k = 0; ok && k < 5; ++k) {
        double want = 0.0;
        for (uint32_t i = 0; i < counts[k]; ++i) want += x[i];
        cudaMemcpy(dc, &counts[k], sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dt, &nine, sizeof(double), cudaMemcpyHostToDevice);
        ok = eg_launch(ctx, p, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
        cudaMemcpy(&tot, dt, sizeof(double), cudaMemcpyDeviceToHost);
        ok = ok && tot == want;
        if (!ok) printf("      count %u: cell %.17g, want %.17g\n", counts[k], tot, want);
    }
    check(ok, "a DEVICE SUM cell over a prefix set of live count 8, 3, 0, 5, 0: the sum of the lanes that ran, 0 for none");
    if (p) eg_plan_free(ctx, p);
    eg_plane_free(ctx, pt), eg_plane_free(ctx, px), eg_plane_free(ctx, pc);
    eg_unit_free(ctx, unit);
    cudaFree(dx), cudaFree(dt), cudaFree(dc);
}

/* eg_plane_new / eg_plane_dlpack on a device context: a DEVICE plane (kDLCUDA) and a PINNED plane (kDLCUDAHost) by their export fields and
 * the driver's own reading of the pointers; a DEVICE launch on made planes against the plain-C oracle (bitwise); the refusals of the
 * device kinds. */
static void alloc_planes(eg_ctx* ctx, eg_ctx* host, eg_unit* axpy, const double* x, const double* v, const double* want, double a) {
    const int64_t shape1[1] = { N }, shape2[2] = { 2, N }, shapep[2] = { 4, 5 };
    struct eg_plane_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = EG_PLANE_ZERO;
    d.dtype_code = kDLFloat;
    d.dtype_bits = 64;
    d.kind = EG_MEM_DEVICE;
    d.ndim = 1;
    d.shape = shape1;
    eg_plane *pdev = NULL, *ppin = NULL;
    struct DLManagedTensorVersioned *edev = NULL, *epin = NULL;
    double back[N];
    memset(back, 0x5a, sizeof(back));
    int ok = eg_plane_new(ctx, &d, &pdev) == EG_OK && eg_plane_dlpack(ctx, pdev, &edev) == EG_OK && edev->dl_tensor.data != NULL &&
             edev->dl_tensor.device.device_type == kDLCUDA && edev->dl_tensor.device.device_id == 0 && edev->dl_tensor.ndim == 1 &&
             edev->dl_tensor.shape[0] == N && edev->dl_tensor.dtype.code == kDLFloat && edev->dl_tensor.dtype.bits == 64 &&
             cudaMemcpy(back, edev->dl_tensor.data, sizeof(back), cudaMemcpyDeviceToHost) == cudaSuccess;
    for (int i = 0; i < N; ++i) ok = ok && back[i] == 0.0;
    check(ok, "eg_plane_new DEVICE with EG_PLANE_ZERO: the export is kDLCUDA of device 0 with shape (1000), float64, and the driver reads zeros back");

    d.kind = EG_MEM_PINNED;
    d.ndim = 2;
    d.shape = shapep;
    struct cudaPointerAttributes attr;
    ok = eg_plane_new(ctx, &d, &ppin) == EG_OK && eg_plane_dlpack(ctx, ppin, &epin) == EG_OK && epin->dl_tensor.data != NULL &&
         epin->dl_tensor.device.device_type == kDLCUDAHost && epin->dl_tensor.ndim == 2 && epin->dl_tensor.shape[0] == 4 &&
         epin->dl_tensor.shape[1] == 5 && epin->dl_tensor.dtype.code == kDLFloat && epin->dl_tensor.dtype.bits == 64 &&
         (epin->dl_tensor.strides == NULL || (epin->dl_tensor.strides[0] == 5 && epin->dl_tensor.strides[1] == 1)) &&
         cudaPointerGetAttributes(&attr, epin->dl_tensor.data) == cudaSuccess && attr.type == cudaMemoryTypeHost;
    for (int i = 0; ok && i < 20; ++i) ok = ((const double*)epin->dl_tensor.data)[i] == 0.0;
    check(ok, "eg_plane_new PINNED with EG_PLANE_ZERO: the export is kDLCUDAHost with shape (4, 5), float64, zeroed, and the driver reports page-locked host memory");

    eg_plane *pm_y = NULL, *pm_x = NULL, *pm_v = NULL;
    struct DLManagedTensorVersioned *ey = NULL, *ex = NULL, *ev = NULL;
    d.kind = EG_MEM_DEVICE;
    d.ndim = 1;
    d.shape = shape1;
    ok = eg_plane_new(ctx, &d, &pm_y) == EG_OK && eg_plane_new(ctx, &d, &pm_x) == EG_OK && eg_plane_dlpack(ctx, pm_y, &ey) == EG_OK &&
         eg_plane_dlpack(ctx, pm_x, &ex) == EG_OK;
    d.ndim = 2;
    d.shape = shape2;
    ok = ok && eg_plane_new(ctx, &d, &pm_v) == EG_OK && eg_plane_dlpack(ctx, pm_v, &ev) == EG_OK;
    ok = ok && cudaMemcpy(ex->dl_tensor.data, x, N * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(ev->dl_tensor.data, v, 2 * N * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
    eg_plan* p = NULL;
    uint64_t tiles = 7;
    if (ok) ok = plan(ctx, axpy, EG_STRUCTURE_DEVICE, 0, NULL, NULL, &p) == EG_OK && bind(ctx, p, pm_y, pm_x, pm_v, a) == EG_OK &&
                 eg_launch(ctx, p, &tiles) == EG_OK && eg_sync(ctx) == EG_OK;
    memset(back, 0, sizeof(back));
    ok = ok && tiles == 0 && cudaMemcpy(back, ey->dl_tensor.data, sizeof(back), cudaMemcpyDeviceToHost) == cudaSuccess &&
         memcmp(back, want, sizeof(back)) == 0;
    check(ok, "axpy DEVICE on eg_plane_new planes: launch, eg_sync, y bitwise == the plain-C oracle (the value the caller-plane run gives)");
    eg_plan_free(ctx, p);

    eg_graph* g = NULL;
    d.ndim = 1;
    d.shape = shape1;
    eg_plane* q = NULL;
    ok = eg_record_begin(ctx) == EG_OK && refused(eg_plane_new(ctx, &d, &q), EG_BAD_REQUEST, ctx, "outside the capture") && q == NULL &&
         eg_record_end(ctx, NULL, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK;
    check(ok, "eg_plane_new inside a DEVICE recording refused BAD_REQUEST by reason; the recording still ends");
    d.kind = EG_MEM_PINNED;
    ok = refused(eg_plane_new(host, &d, &q), EG_NOT_WRITTEN, host, "PINNED memory on a host-only context");
    d.kind = EG_MEM_DEVICE;
    ok = ok && refused(eg_plane_new(host, &d, &q), EG_NOT_WRITTEN, host, "DEVICE memory on a host-only context");
    check(ok, "eg_plane_new PINNED and DEVICE on a host-only context of a CUDA build refused NOT_WRITTEN by reason");

    /* The memory survives eg_plane_free while the export is held, and goes with the last release. */
    ok = eg_plane_free(ctx, pdev) == EG_OK && eg_plane_free(ctx, ppin) == EG_OK && cudaMemcpy(back, edev->dl_tensor.data, sizeof(back), cudaMemcpyDeviceToHost) == cudaSuccess &&
         ((double*)epin->dl_tensor.data)[19] == 0.0;
    edev->deleter(edev);
    epin->deleter(epin);
    ey->deleter(ey);
    ex->deleter(ex);
    ev->deleter(ev);
    ok = ok && eg_plane_free(ctx, pm_y) == EG_OK && eg_plane_free(ctx, pm_x) == EG_OK && eg_plane_free(ctx, pm_v) == EG_OK && eg_sync(ctx) == EG_OK &&
         cudaGetLastError() == cudaSuccess;
    check(ok, "exports of device and pinned planes are readable after eg_plane_free; their release and the planes' frees leave the driver clean");
}

/* A DEVICE plane exported, the context freed: the memory is still readable, and its release then frees it on the stream it was made on. */
static void check_close_with_device_export(void) {
    eg_status s = EG_OK;
    eg_ctx* ctx = context(EG_CTX_DEVICE, 0, &s);
    const int64_t shape1[1] = { N };
    struct eg_plane_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = EG_PLANE_ZERO;
    d.dtype_code = kDLFloat;
    d.dtype_bits = 64;
    d.kind = EG_MEM_DEVICE;
    d.ndim = 1;
    d.shape = shape1;
    eg_plane* p = NULL;
    struct DLManagedTensorVersioned* e = NULL;
    double fill[N], back[N];
    for (int i = 0; i < N; ++i) fill[i] = 0.5 * i;
    memset(back, 0, sizeof(back));
    int ok = s == EG_OK && eg_plane_new(ctx, &d, &p) == EG_OK && eg_plane_dlpack(ctx, p, &e) == EG_OK &&
             cudaMemcpy(e->dl_tensor.data, fill, sizeof(fill), cudaMemcpyHostToDevice) == cudaSuccess && eg_plane_free(ctx, p) == EG_OK &&
             eg_ctx_free(ctx) == EG_OK && eg_sync(ctx) == EG_BAD_REQUEST && reason_has(ctx, "the context is closed") &&
             cudaMemcpy(back, e->dl_tensor.data, sizeof(back), cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(back, fill, sizeof(back)) == 0;
    if (e != NULL) e->deleter(e);
    ok = ok && cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess;
    check(ok, "a DEVICE export outlives eg_ctx_free (closed, readable); its release frees the memory and the driver stays clean");
}

/* One attribute of the device function @p name of the fixture's image, read through the driver API on its own: the oracle
 * of the figures the door reports (-1: the image or the driver could not be read). */
static int driver_attribute(const char* dev_lib, const char* name, int attribute) {
    void* lib = dlopen(dev_lib, RTLD_NOW);
    void* driver = dlopen("libcuda.so.1", RTLD_NOW);
    int value = -1;
    if (lib != NULL && driver != NULL) {
        typedef int (*load_fn)(void**, const void*);
        typedef int (*find_fn)(void**, void*, const char*);
        typedef int (*attr_fn)(int*, int, void*);
        const void* image = dlsym(lib, "fixture__image");
        load_fn load_data;
        find_fn find;
        attr_fn get;
        *(void**)&load_data = dlsym(driver, "cuModuleLoadData");
        *(void**)&find = dlsym(driver, "cuModuleGetFunction");
        *(void**)&get = dlsym(driver, "cuFuncGetAttribute");
        void *module = NULL, *function = NULL;
        int got = 0;
        if (image != NULL && load_data(&module, image) == 0 && find(&function, module, name) == 0 && get(&got, attribute, function) == 0)
            value = got;
    }
    return value;
}

/* The CUfunction_attribute values of cuda.h, named here as this test links the runtime only. */
enum { ATTR_MAX_THREADS = 0, ATTR_SHARED_BYTES = 1, ATTR_CONST_BYTES = 2, ATTR_LOCAL_BYTES = 3, ATTR_REGS = 4 };

/* eg_plan_entry_figures against the driver's own reads of the same functions, and eg_runner_figures of a finishing unit
 * against the launch its kernel saw (dcount writes its blockDim and gridDim into geom). */
static void figures_device(eg_ctx* ctx, const char* dev_lib) {
    enum { FN = 1000 };
    eg_unit *axpy = NULL, *count = NULL;
    check(load(ctx, dev_lib, "axpy", 0xC1, &axpy) == EG_OK && load(ctx, dev_lib, "dcount", 0xD7, &count) == EG_OK,
          "eg_unit_load axpy and dcount (a finishing device unit) for the figures");
    if (axpy == NULL || count == NULL) return;
    const char* names[2] = { "axpy", "dcount" };
    eg_unit* units[2] = { axpy, count };
    int ok = 1;
    for (int u = 0; u < 2; ++u) {
        eg_plan* p = NULL;
        struct eg_entry_figures f;
        memset(&f, 0, sizeof(f));
        f.struct_size = sizeof(f);
        ok = ok && plan_n(ctx, units[u], EG_STRUCTURE_DEVICE, FN, 0, NULL, NULL, &p) == EG_OK && eg_plan_entry_figures(ctx, p, &f) == EG_OK;
        ok = ok && f.struct_size == sizeof(f) && f.flags == EG_ENTRY_DEVICE && f.persist_regs == 0 && f.persist_max_threads == 0;
        ok = ok && f.regs > 0 && f.regs == driver_attribute(dev_lib, names[u], ATTR_REGS) &&
             f.max_threads == driver_attribute(dev_lib, names[u], ATTR_MAX_THREADS) &&
             f.local_bytes == driver_attribute(dev_lib, names[u], ATTR_LOCAL_BYTES) &&
             f.shared_bytes == driver_attribute(dev_lib, names[u], ATTR_SHARED_BYTES) &&
             f.const_bytes == driver_attribute(dev_lib, names[u], ATTR_CONST_BYTES);
        if (p) eg_plan_free(ctx, p);
    }
    check(ok, "eg_plan_entry_figures on DEVICE plans: EG_ENTRY_DEVICE, registers, max threads, local, shared and constant bytes == the driver's own reads");

    static double limit[FN];
    for (int i = 0; i < FN; ++i) limit[i] = 1.0 + i % 3;
    double *dk = dev_doubles(NULL, FN), *dl = dev_doubles(limit, FN), *dg = dev_doubles(NULL, 2);
    unsigned char* dm = NULL;
    cudaMalloc((void**)&dm, FN);
    cudaMemset(dm, 0, FN);
    struct tensor tk, tl, tg, tm;
    eg_plane *pk = wrap(ctx, &tk, dk, 1, FN, 0, 1, 0), *pl = wrap(ctx, &tl, dl, 1, FN, 0, 1, 0), *pg = wrap(ctx, &tg, dg, 1, 2, 0, 1, 0);
    memset(&tm, 0, sizeof(tm));
    tm.shape[0] = FN;
    tm.managed.version.major = DLPACK_MAJOR_VERSION;
    tm.managed.dl_tensor.data = dm;
    tm.managed.dl_tensor.device.device_type = kDLCUDA;
    tm.managed.dl_tensor.ndim = 1;
    tm.managed.dl_tensor.dtype.code = kDLBool;
    tm.managed.dl_tensor.dtype.bits = 8;
    tm.managed.dl_tensor.dtype.lanes = 1;
    tm.managed.dl_tensor.shape = tm.shape;
    eg_plane* pm = NULL;
    eg_plan* p = NULL;
    eg_runner* r = NULL;
    const int32_t slots[4] = { 0, 1, 2, 4 };
    eg_plane* planes[4] = { pk, pl, NULL, pg };
    ok = pk && pl && pg && eg_plane_wrap(ctx, &tm.managed, &pm) == EG_OK;
    planes[2] = pm;
    struct eg_runner_desc rn;
    memset(&rn, 0, sizeof(rn));
    rn.struct_size = sizeof(rn);
    rn.n_plans = 1;
    rn.plans = &p;
    rn.max_steps = 10;
    ok = ok && plan_n(ctx, count, EG_STRUCTURE_DEVICE, FN, 0, NULL, NULL, &p) == EG_OK && bind_planes(ctx, p, 4, slots, planes) == EG_OK &&
         eg_runner_new(ctx, &rn, &r) == EG_OK;
    struct eg_runner_figures f;
    memset(&f, 0, sizeof(f));
    f.struct_size = sizeof(f);
    ok = ok && eg_runner_figures(ctx, r, &f) == EG_OK && f.mode == EG_MODE_BAND && f.block == -1 && f.grid == -1 && f.chunk == -1;
    check(ok, "eg_runner_figures on a DEVICE runner before its first run: mode BAND, block, grid and chunk -1");
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    double geom[2] = { 0, 0 };
    ok = ok && eg_run(ctx, r, &rep) == EG_OK && eg_sync(ctx) == EG_OK && eg_runner_figures(ctx, r, &f) == EG_OK;
    cudaMemcpy(geom, dg, sizeof(geom), cudaMemcpyDeviceToHost);
    ok = ok && rep.finished == FN && f.mode == EG_MODE_BAND && f.chunk == -1 && f.block > 0 && f.block % 32 == 0 && (double)f.block == geom[0] &&
         (double)f.grid == geom[1] && f.grid == (FN + f.block - 1) / f.block;
    if (!ok) printf("      block %d grid %lld, the kernel saw %g x %g\n", f.block, (long long)f.grid, geom[0], geom[1]);
    check(ok, "eg_runner_figures after eg_run: block and grid == the launch the kernel saw (blockDim, gridDim), grid covers the samples");
    struct eg_plan_counters counters;
    int32_t n_mix = -1;
    memset(&counters, 0, sizeof(counters));
    counters.struct_size = sizeof(counters);
    check(p != NULL && eg_plan_counters(ctx, p, &counters, NULL, 0, &n_mix) == EG_OK && counters.finished == FN && counters.steps == -1 && n_mix == 0,
          "eg_plan_counters on a DEVICE plan after eg_run: the finished counter is the mask count, no steps word, no mix (a synchronous device read)");
    if (r) eg_runner_free(ctx, r);
    int64_t at[4] = { 0, 0, 0, 0 };    /* at[L]: the samples of limit L, each finished by the L-th launch from k = 0 */
    for (int i = 0; i < FN; ++i) ++at[(int)limit[i]];
    cudaMemset(dk, 0, FN * sizeof(double));
    cudaMemset(dm, 0, FN);
    ok = p != NULL;
    for (int launch = 1; ok && launch <= 3; ++launch)
        ok = eg_launch(ctx, p, NULL) == EG_OK && eg_plan_counters(ctx, p, &counters, NULL, 0, &n_mix) == EG_OK && counters.finished == at[launch];
    check(ok, "eg_plan_counters on a DEVICE plan: each of 3 eg_launch calls reads the samples that launch finished");
    cudaMemset(dk, 0, FN * sizeof(double));
    cudaMemset(dm, 0, FN);
    eg_graph* g = NULL;
    ok = p != NULL && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, NULL, &g) == EG_OK;
    for (int replay = 1; ok && replay <= 3; ++replay)
        ok = eg_graph_launch(ctx, g) == EG_OK && eg_plan_counters(ctx, p, &counters, NULL, 0, &n_mix) == EG_OK && counters.finished == at[replay];
    check(ok, "eg_plan_counters on a DEVICE plan: each replay of a recorded launch reads the samples that replay finished");
    if (g) eg_graph_free(ctx, g);
    if (p) eg_plan_free(ctx, p);
    eg_plane_free(ctx, pk);
    eg_plane_free(ctx, pl);
    eg_plane_free(ctx, pm);
    eg_plane_free(ctx, pg);
    eg_unit_free(ctx, axpy);
    eg_unit_free(ctx, count);
    cudaFree(dk);
    cudaFree(dl);
    cudaFree(dg);
    cudaFree(dm);
}

/* ---- eg_runner_reset on the device ---- */

#define DN 5000

struct dcount {
    eg_plan* plan;
    eg_runner* runner;
    double *dk, *dl;
    unsigned char* dm;
    double k[DN], limit[DN];
    unsigned char mask[DN];
    struct tensor tk, tl, tm;
    eg_plane *pk, *pl, *pm;
};

/* A plane of @p n elements of @p code / @p bits over device memory @p data. */
static eg_plane* wrap_device_typed(eg_ctx* ctx, struct tensor* t, void* data, int64_t n, uint8_t code, uint8_t bits) {
    eg_plane* p = wrap(ctx, t, data, 1, n, 0, 1, 0);
    if (p == NULL) return NULL;
    eg_plane_free(ctx, p);
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
    p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

/* limit_i in 1..50; every 97th sample starts finished. */
static void dcount_inputs(struct dcount* c) {
    for (int i = 0; i < DN; ++i) c->k[i] = 0.0, c->limit[i] = (double)(1 + (i * 37) % 50), c->mask[i] = i % 97 == 0;
    cudaMemcpy(c->dk, c->k, sizeof(c->k), cudaMemcpyHostToDevice);
    cudaMemcpy(c->dl, c->limit, sizeof(c->limit), cudaMemcpyHostToDevice);
    cudaMemcpy(c->dm, c->mask, sizeof(c->mask), cudaMemcpyHostToDevice);
}

static eg_status dcount_new(eg_ctx* ctx, eg_unit* unit, int64_t every, double theta, struct dcount* c) {
    memset(c, 0, sizeof(*c));
    cudaMalloc((void**)&c->dk, sizeof(c->k));
    cudaMalloc((void**)&c->dl, sizeof(c->limit));
    cudaMalloc((void**)&c->dm, sizeof(c->mask));
    c->pk = wrap(ctx, &c->tk, c->dk, 1, DN, 0, 1, 0);
    c->pl = wrap(ctx, &c->tl, c->dl, 1, DN, 0, 1, 0);
    c->pm = wrap_device_typed(ctx, &c->tm, c->dm, DN, kDLBool, 8);
    if (!(c->pk && c->pl && c->pm)) return EG_FAULT;
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* planes[3] = { c->pk, c->pl, c->pm };
    eg_status s = plan_n(ctx, unit, EG_STRUCTURE_DEVICE, DN, 0, NULL, NULL, &c->plan);
    if (s == EG_OK) s = bind_planes(ctx, c->plan, 3, slots, planes);
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_plans = 1;
    d.plans = &c->plan;
    d.max_steps = 1000;
    d.every = every;
    if (theta > 0) d.flags = EG_RUN_REORDER, d.reorder_theta = theta;
    if (s == EG_OK) s = eg_runner_new(ctx, &d, &c->runner);
    return s;
}

static eg_status dcount_run(eg_ctx* ctx, struct dcount* c, struct eg_run_report* rep) {
    memset(rep, 0, sizeof(*rep));
    rep->struct_size = sizeof(*rep);
    eg_status s = eg_run(ctx, c->runner, rep);
    if (s == EG_OK) s = eg_sync(ctx);
    cudaMemcpy(c->k, c->dk, sizeof(c->k), cudaMemcpyDeviceToHost);
    cudaMemcpy(c->mask, c->dm, sizeof(c->mask), cudaMemcpyDeviceToHost);
    return s;
}

static void dcount_free(eg_ctx* ctx, struct dcount* c) {
    eg_runner_free(ctx, c->runner);
    eg_plan_free(ctx, c->plan);
    eg_plane_free(ctx, c->pk);
    eg_plane_free(ctx, c->pl);
    eg_plane_free(ctx, c->pm);
    cudaFree(c->dk);
    cudaFree(c->dl);
    cudaFree(c->dm);
}

/* Run, eg_runner_reset, write the inputs again, run again: planes and report bitwise those of a fresh runner's run. */
static void reset_device_case(eg_ctx* ctx, eg_unit* unit, int64_t every, double theta, const char* what) {
    static struct dcount fresh, reused;
    struct eg_run_report rf, r1, r2;
    eg_status s = dcount_new(ctx, unit, every, theta, &fresh);
    if (s == EG_OK) dcount_inputs(&fresh), s = dcount_run(ctx, &fresh, &rf);
    if (s == EG_OK) s = dcount_new(ctx, unit, every, theta, &reused);
    if (s == EG_OK) dcount_inputs(&reused), s = dcount_run(ctx, &reused, &r1);
    if (s == EG_OK) s = eg_runner_reset(ctx, reused.runner);
    if (s == EG_OK) s = eg_sync(ctx);
    cudaMemcpy(reused.mask, reused.dm, sizeof(reused.mask), cudaMemcpyDeviceToHost);
    int cleared = 1;
    for (int i = 0; i < DN; ++i) cleared &= reused.mask[i] == 0;
    if (s == EG_OK) dcount_inputs(&reused), s = dcount_run(ctx, &reused, &r2);
    int same = memcmp(reused.k, fresh.k, sizeof(fresh.k)) == 0 && memcmp(reused.mask, fresh.mask, sizeof(fresh.mask)) == 0 &&
               r2.n == rf.n && r2.finished == rf.finished && r2.launches == rf.launches && r2.steps == rf.steps &&
               r2.mapped == rf.mapped && r2.compactions == rf.compactions && r2.reorders == rf.reorders && r2.flags == rf.flags;
    if (!same) printf("      fresh: %lld finished %lld launches %lld compactions %lld reorders; again: %lld %lld %lld %lld\n", (long long)rf.finished,
                      (long long)rf.launches, (long long)rf.compactions, (long long)rf.reorders, (long long)r2.finished,
                      (long long)r2.launches, (long long)r2.compactions, (long long)r2.reorders);
    check(s == EG_OK && cleared && same, what);
    dcount_free(ctx, &fresh);
    dcount_free(ctx, &reused);
}

/* ---- SEGMENTED groups under a runner on the device (host-stepped) ---- */

struct dset {
    double *dk, *dl, *dp;
    unsigned char* dm;
    double k[SGN], limit[SGN], pos[SGN];
    unsigned char mask[SGN];
    struct tensor t[4];
    eg_plane* pl[4];
    eg_plan* plan;
};

static void dset_upload(struct dset* q, int dynamic) {
    for (int i = 0; i < SGN; ++i) {
        q->k[i] = dynamic ? DG_K0(i) : 0.0;
        q->limit[i] = dynamic ? 7.0 : (double)(2 + i % 9);
        q->pos[i] = -1.0;
        q->mask[i] = 0;
    }
    cudaMemcpy(q->dk, q->k, sizeof(q->k), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dl, q->limit, sizeof(q->limit), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dp, q->pos, sizeof(q->pos), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dm, q->mask, sizeof(q->mask), cudaMemcpyHostToDevice);
}

static int dset_new(eg_ctx* ctx, eg_unit* unit, int dynamic, int with_pos, struct dset* q) {
    memset(q, 0, sizeof(*q));
    cudaMalloc((void**)&q->dk, sizeof(q->k));
    cudaMalloc((void**)&q->dl, sizeof(q->limit));
    cudaMalloc((void**)&q->dp, sizeof(q->pos));
    cudaMalloc((void**)&q->dm, sizeof(q->mask));
    dset_upload(q, dynamic);
    q->pl[0] = wrap(ctx, &q->t[0], q->dk, 1, SGN, 0, 1, 0);
    q->pl[1] = wrap(ctx, &q->t[1], q->dl, 1, SGN, 0, 1, 0);
    q->pl[2] = wrap_device_typed(ctx, &q->t[2], q->dm, SGN, kDLBool, 8);
    q->pl[3] = with_pos ? wrap(ctx, &q->t[3], q->dp, 1, SGN, 0, 1, 0) : NULL;
    const int32_t slots[4] = { 0, 1, 2, 7 };
    const int n = with_pos ? 4 : 3;
    eg_status s = q->pl[0] && q->pl[1] && q->pl[2] && (q->pl[3] || !with_pos) ? plan_n(ctx, unit, EG_STRUCTURE_DEVICE, SGN, 0, NULL, NULL, &q->plan) : EG_FAULT;
    if (s == EG_OK) s = bind_planes(ctx, q->plan, n, slots, q->pl);
    return s == EG_OK;
}

static void dset_read(eg_ctx* ctx, struct dset* q) {
    eg_sync(ctx);
    cudaMemcpy(q->k, q->dk, sizeof(q->k), cudaMemcpyDeviceToHost);
    cudaMemcpy(q->pos, q->dp, sizeof(q->pos), cudaMemcpyDeviceToHost);
    cudaMemcpy(q->mask, q->dm, sizeof(q->mask), cudaMemcpyDeviceToHost);
}

static void dset_free(eg_ctx* ctx, struct dset* q) {
    eg_plan_free(ctx, q->plan);
    for (int i = 0; i < 4; ++i)
        if (q->pl[i] != NULL) eg_plane_free(ctx, q->pl[i]);
    cudaFree(q->dk), cudaFree(q->dl), cudaFree(q->dp), cudaFree(q->dm);
}

static int dset_same(const struct dset* a, const struct dset* b, int with_pos) {
    return memcmp(a->k, b->k, sizeof(a->k)) == 0 && memcmp(a->mask, b->mask, sizeof(a->mask)) == 0 &&
           (!with_pos || memcmp(a->pos, b->pos, sizeof(a->pos)) == 0);
}

/* The group's plan launched one step at a time by the caller, with the runner's finish rule. */
static int64_t dset_launch_loop(eg_ctx* ctx, struct dset* q, int64_t max_steps) {
    int64_t launches = 0;
    uint64_t tiles = 0;
    dset_read(ctx, q);
    while (launches < max_steps && seg_done(q->mask) != SGN && eg_launch(ctx, q->plan, &tiles) == EG_OK) ++launches, dset_read(ctx, q);
    return launches;
}

static eg_status dset_run(eg_ctx* ctx, struct dset* q, int64_t max_steps, eg_runner** r, struct eg_run_report* rep) {
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_plans = 1;
    d.plans = &q->plan;
    d.max_steps = max_steps;
    memset(rep, 0, sizeof(*rep));
    rep->struct_size = sizeof(*rep);
    eg_status s = eg_runner_new(ctx, &d, r);
    if (s == EG_OK) s = eg_run(ctx, *r, rep);
    dset_read(ctx, q);
    return s;
}

/* The report of two runs is the same record (the wall time of a capture is no part of a host-stepped group's). */
static int same_report(const struct eg_run_report* a, const struct eg_run_report* b) {
    return a->n == b->n && a->finished == b->finished && a->launches == b->launches && a->steps == b->steps && a->mapped == b->mapped &&
           a->flags == b->flags && a->mode == b->mode && a->compactions == b->compactions && a->reorders == b->reorders;
}

/* The largest |fused.k - segmented.k| over the samples, against the bound n eps M. The two are different compiled units, so their k is not bitwise.
 * Derivation: every addition into k is a multiple of 0.125 of at least 0.125 and at most 0.25 and a sample stops at its first k >= limit, so a sample takes at
 * most n = Lmax / 0.125 + 1 additions over the run, each rounded once (error <= eps / 2 of a value below M) in each of the two orders:
 * |diff| <= n eps M with M = Lmax + 0.25 the cap of k, eps = DBL_EPSILON, Lmax the largest limit. @p bound and @p worst receive the two sides. */
static int fused_k_within_bound(const struct dset* fused, const struct dset* segmented, double* bound, double* worst) {
    double lmax = 0.0;
    for (int i = 0; i < SGN; ++i) lmax = fmax(lmax, segmented->limit[i]);
    *bound = (lmax / 0.125 + 1.0) * DBL_EPSILON * (lmax + 0.25);
    *worst = 0.0;
    for (int i = 0; i < SGN; ++i) *worst = fmax(*worst, fabs(fused->k[i] - segmented->k[i]));
    return *worst <= *bound;
}

/* One SEGMENTED group (static or dynamic selector) under a DEVICE runner, host-stepped: the runner against the plan launched by hand, the fused unit within the
 * derived bound, reset and a fresh runner, a cut at 3 steps, the refusals and the figures. Same code and same data are compared bitwise; the fused unit is another computation. */
static void segmented_group_rows(eg_ctx* ctx, const char* name, int dynamic, eg_unit* segmented, eg_unit* fused) {
    static struct dset hand, run, fus, fresh, cut_hand, cut_run, first;
    struct eg_run_report rep, rep2, repf, repc, repd;
    eg_runner *r = NULL, *rf = NULL, *rn = NULL, *rc = NULL;
    char what[320];
    check(dset_new(ctx, segmented, dynamic, dynamic, &hand) && dset_new(ctx, segmented, dynamic, dynamic, &run) && dset_new(ctx, fused, dynamic, 0, &fus) &&
              dset_new(ctx, segmented, dynamic, dynamic, &fresh) && dset_new(ctx, segmented, dynamic, dynamic, &cut_hand) && dset_new(ctx, segmented, dynamic, dynamic, &cut_run),
          "the SEGMENTED group's plans and the fused unit's plan on the device");
    const int64_t want = dset_launch_loop(ctx, &hand, 1000);
    snprintf(what, sizeof(what), "the %s device group launched by hand until done: %lld launches", name, (long long)want);
    check(seg_done(hand.mask) == SGN && (dynamic || want == 5), what);
    /* (a) the runner against the hand loop, report in full */
    int ok = dset_run(ctx, &run, 1000, &r, &rep) == EG_OK && dset_same(&run, &hand, dynamic);
    snprintf(what, sizeof(what), "%s: a DEVICE runner over a SEGMENTED group equals the plan launched step by step (planes bitwise) and reports launches = steps = %lld, finished = lanes, mapped = launches x lanes, mode BAND, DONE|EXACT_STEPS", name, (long long)want);
    check(ok && rep.launches == want && rep.steps == want && rep.finished == SGN && rep.mapped == want * SGN && rep.mode == EG_MODE_BAND &&
              rep.flags == (EG_REPORT_DONE | EG_REPORT_EXACT_STEPS), what);
    /* (b) the fused unit: mask and launches are integers, k is within the derived bound */
    double bound = 0.0, worst = 0.0;
    ok = dset_run(ctx, &fus, 1000, &rf, &repf) == EG_OK && fused_k_within_bound(&fus, &hand, &bound, &worst) && memcmp(fus.mask, hand.mask, sizeof(hand.mask)) == 0 &&
         repf.launches == want && repf.mode == EG_MODE_BAND;
    snprintf(what, sizeof(what), "%s: the fused unit (the captured loop) equals it: mask and launches exact, k within n eps M (max |diff| %.3e, bound %.3e)", name, worst, bound);
    check(ok, what);
    /* (c) reset and a second run equal the first, report included, and a fresh runner on the same data */
    memcpy(first.k, run.k, sizeof(run.k)), memcpy(first.mask, run.mask, sizeof(run.mask)), memcpy(first.pos, run.pos, sizeof(run.pos));
    dset_upload(&run, dynamic);
    memset(&rep2, 0, sizeof(rep2)), rep2.struct_size = sizeof(rep2);
    ok = eg_runner_reset(ctx, r) == EG_OK && eg_run(ctx, r, &rep2) == EG_OK && (dset_read(ctx, &run), dset_same(&run, &first, dynamic)) && same_report(&rep, &rep2);
    memset(&repc, 0, sizeof(repc)), repc.struct_size = sizeof(repc);
    ok = ok && dset_run(ctx, &fresh, 1000, &rn, &repc) == EG_OK && dset_same(&fresh, &run, dynamic) && same_report(&repc, &rep2);
    snprintf(what, sizeof(what), "%s: eg_runner_reset then a second run equals the first, report included, and a fresh runner on the same data (bitwise)", name);
    check(ok, what);
    /* (d) a cut at 3 steps */
    const int64_t cut = dset_launch_loop(ctx, &cut_hand, 3);
    ok = dset_run(ctx, &cut_run, 3, &rc, &repd) == EG_OK && dset_same(&cut_run, &cut_hand, dynamic);
    snprintf(what, sizeof(what), "%s: max_steps 3 equals the hand loop cut at 3 (planes bitwise), finished %lld < %d, flags EXACT_STEPS only", name, (long long)repd.finished, SGN);
    check(ok && cut == 3 && repd.launches == 3 && repd.finished < SGN && repd.finished == seg_done(cut_hand.mask) && repd.flags == EG_REPORT_EXACT_STEPS, what);
    /* (e) refusals, by status and sentence */
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d), d.n_plans = 1, d.plans = &run.plan, d.max_steps = 10;
    eg_runner* x = NULL;
    eg_plan* two[2] = { run.plan, fresh.plan };
    d.n_plans = 2, d.plans = two;
    check(refused(eg_runner_new(ctx, &d, &x), EG_NOT_WRITTEN, ctx, "is the whole step of its runner, not one plan of several") && x == NULL, "a second plan beside a SEGMENTED group: NOT_WRITTEN");
    d.n_plans = 1, d.plans = &run.plan, d.events = EG_EVENTS_REFINE;
    check(refused(eg_runner_new(ctx, &d, &x), EG_NOT_WRITTEN, ctx, "is not recorded or refined") && x == NULL, "EG_EVENTS_REFINE over a SEGMENTED group: NOT_WRITTEN");
    const eg_derived_role tape_role = EG_DERIVED_TAPE;
    eg_unit* const tape_unit = segmented;   /* a well-formed derived set: the group's own rule refuses before the units are read */
    d.events = 0, d.flags = EG_RUN_RECORD, d.n_derived = 1, d.derived_roles = &tape_role, d.derived_units = &tape_unit;
    check(refused(eg_runner_new(ctx, &d, &x), EG_NOT_WRITTEN, ctx, "is not recorded or refined") && x == NULL, "EG_RUN_RECORD over a SEGMENTED group: NOT_WRITTEN");
    d.n_derived = 0, d.derived_roles = NULL, d.derived_units = NULL;
    d.flags = EG_RUN_REORDER;
    check(refused(eg_runner_new(ctx, &d, &x), EG_BAD_REQUEST, ctx, "nothing to reorder") && x == NULL, "EG_RUN_REORDER over a SEGMENTED group: BAD_REQUEST");
    d.flags = 0, d.every = 4;
    check(refused(eg_runner_new(ctx, &d, &x), EG_BAD_REQUEST, ctx, "steps per launch") && x == NULL, "every over a SEGMENTED group: BAD_REQUEST");
    /* (f) the figures after a run: the report's mode, and no launch of its own (the runner steps the group) */
    struct eg_runner_figures fig;
    memset(&fig, 0, sizeof(fig));
    fig.struct_size = sizeof(fig);
    snprintf(what, sizeof(what), "%s: eg_runner_figures after a SEGMENTED run: mode BAND, block, chunk and grid -1; the report's launches equal the hand count %lld", name, (long long)want);
    check(eg_runner_figures(ctx, r, &fig) == EG_OK && fig.mode == EG_MODE_BAND && fig.block == -1 && fig.chunk == -1 && fig.grid == -1 && rep.launches == want, what);
    eg_runner_free(ctx, r), eg_runner_free(ctx, rf), eg_runner_free(ctx, rn), eg_runner_free(ctx, rc);
    dset_free(ctx, &hand), dset_free(ctx, &run), dset_free(ctx, &fus), dset_free(ctx, &fresh), dset_free(ctx, &cut_hand), dset_free(ctx, &cut_run);
}

static void check_segmented_device(eg_ctx* ctx, const char* dev_lib) {
    eg_unit *cls = NULL, *k1 = NULL, *k0 = NULL, *fused = NULL, *dk1 = NULL, *dsw = NULL, *dk0 = NULL, *dgf = NULL;
    check(load(ctx, dev_lib, "dsg__cls", 0xF1, &cls) == EG_OK && load(ctx, dev_lib, "dsg__k1", 0xF3, &k1) == EG_OK &&
              load(ctx, dev_lib, "dsgf", 0xF4, &fused) == EG_OK && load(ctx, dev_lib, "ddg__k1", 0xF6, &dk1) == EG_OK &&
              load(ctx, dev_lib, "ddg__sweep", 0xF7, &dsw) == EG_OK && load(ctx, dev_lib, "ddgf", 0xF8, &dgf) == EG_OK,
          "eg_unit_load of the device SEGMENTED groups' member units and fused units");
    eg_unit* sdeps[2] = { k1, cls };
    eg_unit* ddeps[2] = { dk1, dsw };
    {
        struct eg_unit_ref r0 = { .struct_size = sizeof(r0), .name = "dsg__k0" }, r1 = { .struct_size = sizeof(r1), .name = "ddg__k0" };
        memset(r0.digest, 0xF2, EG_DIGEST_BYTES), memset(r1.digest, 0xF5, EG_DIGEST_BYTES);
        struct eg_unit_load_desc d = { .struct_size = sizeof(d), .path = dev_lib, .unit = &r0, .deps = sdeps, .n_deps = 2 };
        eg_status a = eg_unit_load(ctx, &d, &k0);
        d.unit = &r1, d.deps = ddeps;
        check(a == EG_OK && eg_unit_load(ctx, &d, &dk0) == EG_OK, "eg_unit_load of both groups' first branch units with the group's units in deps");
    }
    if (k0 == NULL || dk0 == NULL || fused == NULL || dgf == NULL) return;
    segmented_group_rows(ctx, "static", 0, k0, fused);
    segmented_group_rows(ctx, "dynamic", 1, dk0, dgf);
    struct eg_run_report rep;
    {   /* the key refresh schedule, through the position plane: a refresh at launch 1 and at launch 4, none at 2 or 3 */
        static struct dset c;
        double mk[SGN], ml[SGN], mp[SGN], runpos[4][SGN];
        unsigned char mm[SGN];
        int64_t refreshes[16];
        int n_ref = 0;
        for (int i = 0; i < SGN; ++i) mk[i] = DG_K0(i), ml[i] = 7.0, mp[i] = -1.0, mm[i] = 0;
        dg_model_device(mk, ml, mm, mp, 1000, refreshes, &n_ref);
        check(n_ref >= 2 && refreshes[0] == 1 && refreshes[1] == 4, "the device model refreshes its sort key at launch 1 and next at launch 4 (12 stale of 200 live)");
        int agree = 1;
        for (int cut = 2; cut <= 5; ++cut) {
            for (int i = 0; i < SGN; ++i) mk[i] = DG_K0(i), mp[i] = -1.0, mm[i] = 0;
            dg_model_device(mk, ml, mm, mp, cut, refreshes, &n_ref);
            eg_runner* rc = NULL;
            agree &= dset_new(ctx, dk0, 1, 1, &c) && dset_run(ctx, &c, cut, &rc, &rep) == EG_OK && memcmp(c.k, mk, sizeof(mk)) == 0 &&
                     memcmp(c.mask, mm, sizeof(mm)) == 0 && memcmp(c.pos, mp, sizeof(mp)) == 0 && rep.launches == cut;
            if (cut <= 5 && cut >= 2) memcpy(runpos[cut - 2 < 4 ? cut - 2 : 3], c.pos, sizeof(c.pos));
            eg_runner_free(ctx, rc);
            dset_free(ctx, &c);
        }
        check(agree, "the dynamic DEVICE group under a runner for 2 to 5 launches: planes and positions as the device model");
        check(memcmp(runpos[0], runpos[1], sizeof(runpos[0])) == 0 && memcmp(runpos[1], runpos[2], sizeof(runpos[0])) != 0,
              "the positions are the same after 2 and 3 launches and differ after 4: the key is refreshed at launch 4 and not before");
    }
    eg_unit_free(ctx, k0), eg_unit_free(ctx, k1), eg_unit_free(ctx, cls), eg_unit_free(ctx, fused), eg_unit_free(ctx, dk0), eg_unit_free(ctx, dk1),
        eg_unit_free(ctx, dsw), eg_unit_free(ctx, dgf);
}

/* ---- an AUTO group that names its segmented set, on the device ---- */

/* One run's planes on the device: k, limit, how and mask of the "dsga" group (struct dset's pos plane is how: slot 5 here), the units' own slot order. */
static int dauto_new(eg_ctx* ctx, eg_unit* unit, int n1, int l1, int with_how, struct dset* q) {
    memset(q, 0, sizeof(*q));
    cudaMalloc((void**)&q->dk, sizeof(q->k));
    cudaMalloc((void**)&q->dl, sizeof(q->limit));
    cudaMalloc((void**)&q->dp, sizeof(q->pos));
    cudaMalloc((void**)&q->dm, sizeof(q->mask));
    auto_fill(q->k, q->limit, q->pos, q->mask, n1, l1);
    cudaMemcpy(q->dk, q->k, sizeof(q->k), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dl, q->limit, sizeof(q->limit), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dp, q->pos, sizeof(q->pos), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dm, q->mask, sizeof(q->mask), cudaMemcpyHostToDevice);
    q->pl[0] = wrap(ctx, &q->t[0], q->dk, 1, SGN, 0, 1, 0);
    q->pl[1] = wrap(ctx, &q->t[1], q->dl, 1, SGN, 0, 1, 0);
    q->pl[2] = wrap_device_typed(ctx, &q->t[2], q->dm, SGN, kDLBool, 8);
    q->pl[3] = with_how ? wrap(ctx, &q->t[3], q->dp, 1, SGN, 0, 1, 0) : NULL;
    const int32_t slots[4] = { 0, 1, 2, 5 };
    eg_status s = q->pl[0] && q->pl[1] && q->pl[2] && (q->pl[3] || !with_how) ? plan_n(ctx, unit, EG_STRUCTURE_DEVICE, SGN, 0, NULL, NULL, &q->plan) : EG_FAULT;
    if (s == EG_OK) s = bind_planes(ctx, q->plan, with_how ? 4 : 3, slots, q->pl);
    return s == EG_OK;
}

static void check_auto_device(eg_ctx* ctx, const char* dev_lib) {
    eg_unit *cls = NULL, *k1 = NULL, *k0 = NULL, *fused = NULL, *auto_unit[4] = { NULL, NULL, NULL, NULL };
    const char* names[4] = { "dsga_rare", "dsga_even", "dsga_scalar", "dsga_heavy" };
    check(load(ctx, dev_lib, "dsga__cls", 0x91, &cls) == EG_OK && load(ctx, dev_lib, "dsga__k1", 0x93, &k1) == EG_OK && load(ctx, dev_lib, "dsgf", 0xF4, &fused) == EG_OK,
          "eg_unit_load of the device AUTO group's classify and second branch units");
    eg_unit* sdeps[2] = { k1, cls };
    {
        struct eg_unit_ref r0 = { .struct_size = sizeof(r0), .name = "dsga__k0" };
        memset(r0.digest, 0x92, EG_DIGEST_BYTES);
        struct eg_unit_load_desc d = { .struct_size = sizeof(d), .path = dev_lib, .unit = &r0, .deps = sdeps, .n_deps = 2 };
        check(cls && k1 && eg_unit_load(ctx, &d, &k0) == EG_OK, "eg_unit_load of the device AUTO group's first branch unit with the group's units in deps");
    }
    eg_unit* adeps[3] = { k0, k1, cls };
    for (int u = 0; u < 4 && k0 != NULL; ++u) {
        struct eg_unit_ref r = { .struct_size = sizeof(r), .name = names[u] };
        memset(r.digest, 0x94 + u, EG_DIGEST_BYTES);
        struct eg_unit_load_desc d = { .struct_size = sizeof(d), .path = dev_lib, .unit = &r, .deps = adeps, .n_deps = 3 };
        check(eg_unit_load(ctx, &d, &auto_unit[u]) == EG_OK, "eg_unit_load of a device AUTO unit naming its segmented set");
    }
    if (k0 == NULL || fused == NULL || auto_unit[3] == NULL) return;
    /* The recorded runs are the ones the old stack decides the same way on five GPUs from a P2000 to an H100 (test/auto_record.py), so the expectation does
     * not depend on this one; its decision on any weights is proved equal to Dispatch.h's by tests/test_DispatchPolicy.cpp. */
    char path[512];
    snprintf(path, sizeof(path), "%s/auto_decisions_device.txt", EG_TEST_DIR);
    FILE* f = fopen(path, "r");
    check(f != NULL, "the recorded device decisions of the old stack are present");
    char line[256];
    int rows = 0, agree = 0, switched = 0, fused_from_start = 0, segmented_to_end = 0;
    while (f != NULL && fgets(line, sizeof(line), f) != NULL) {
        char name[32];
        int n1, l1, max_steps, want_seg, want_total;
        if (line[0] == '#' || sscanf(line, "%31s %d %d %d %d %d", name, &n1, &l1, &max_steps, &want_seg, &want_total) != 6) continue;
        const int u = strcmp(name, "rare") == 0 ? 0 : strcmp(name, "even") == 0 ? 1 : strcmp(name, "scalar") == 0 ? 2 : 3;
        static struct dset q, ref;
        eg_runner *r = NULL, *rf = NULL;
        struct eg_run_report rep, rep_ref;
        int ok = dauto_new(ctx, auto_unit[u], n1, l1, 1, &q) && dauto_new(ctx, fused, n1, l1, 0, &ref);
        ok = ok && dset_run(ctx, &q, max_steps, &r, &rep) == EG_OK && dset_run(ctx, &ref, max_steps, &rf, &rep_ref) == EG_OK;
        for (int i = 0; ok && i < SGN; ++i)
            ok = q.pos[i] == auto_how(q.limit, i, n1, want_seg, want_total) && q.k[i] == ref.k[i] && q.mask[i] == ref.mask[i];
        ok = ok && rep.launches == want_total && rep.steps == want_total && rep.mode == EG_MODE_BAND && rep.launches == rep_ref.launches;
        ++rows;
        agree += ok;
        switched += want_seg > 0 && want_seg < want_total;
        fused_from_start += want_seg == 0;
        segmented_to_end += want_seg > 0 && want_seg == want_total;
        if (!ok) printf("      case %s n1 %d l1 %d max_steps %d: old stack segmented %d of %d steps\n", name, n1, l1, max_steps, want_seg, want_total);
        eg_runner_free(ctx, r), eg_runner_free(ctx, rf);
        dset_free(ctx, &q), dset_free(ctx, &ref);
    }
    if (f != NULL) fclose(f);
    char what[400];
    snprintf(what, sizeof(what), "the DEVICE AUTO runner takes the old stack's decision in all %d recorded runs (%d fused from the start, %d segmented to the end, %d switched to "
             "the captured loop at a look), its values equal the plain fused unit's bitwise", rows, fused_from_start, segmented_to_end, switched);
    check(rows == 72 && agree == rows && fused_from_start > 0 && segmented_to_end > 0 && switched > 0, what);
    {   /* the same run forced segmented (the branch unit's own runner) gives the AUTO run's values */
        static struct dset a, b;
        eg_runner *ra = NULL, *rb = NULL;
        struct eg_run_report rep_a, rep_b;
        int ok = dauto_new(ctx, auto_unit[3], 20, 100, 1, &a) && dauto_new(ctx, k0, 20, 100, 1, &b);
        ok = ok && dset_run(ctx, &a, 400, &ra, &rep_a) == EG_OK && dset_run(ctx, &b, 400, &rb, &rep_b) == EG_OK;
        ok = ok && memcmp(a.k, b.k, sizeof(a.k)) == 0 && memcmp(a.mask, b.mask, sizeof(a.mask)) == 0 && rep_a.launches == rep_b.launches && rep_a.finished == SGN;
        check(ok, "a DEVICE AUTO run equals the same group forced segmented (the branch unit's runner): planes bitwise, same launches");
        eg_runner_free(ctx, ra), eg_runner_free(ctx, rb);
        dset_free(ctx, &a), dset_free(ctx, &b);
    }
    {   /* a run that switched leaves no stale captured loop: run it again at the full budget and again */
        static struct dset q;
        eg_runner* r = NULL;
        struct eg_run_report rep, rep2;
        int ok = dauto_new(ctx, auto_unit[3], 2, 100, 1, &q) && dset_run(ctx, &q, 400, &r, &rep) == EG_OK;
        auto_fill(q.k, q.limit, q.pos, q.mask, 2, 100);
        cudaMemcpy(q.dk, q.k, sizeof(q.k), cudaMemcpyHostToDevice), cudaMemcpy(q.dp, q.pos, sizeof(q.pos), cudaMemcpyHostToDevice);
        memset(&rep2, 0, sizeof(rep2));
        rep2.struct_size = sizeof(rep2);
        ok = ok && eg_runner_reset(ctx, r) == EG_OK && eg_run(ctx, r, &rep2) == EG_OK && (dset_read(ctx, &q), 1) && rep2.launches == rep.launches &&
             rep2.finished == SGN && rep2.steps == rep.steps;
        check(ok, "eg_runner_reset of a DEVICE AUTO runner then a second run equals the first (launches, steps, done)");
        eg_runner_free(ctx, r);
        dset_free(ctx, &q);
    }
    for (int u = 0; u < 4; ++u) eg_unit_free(ctx, auto_unit[u]);
    eg_unit_free(ctx, k0), eg_unit_free(ctx, k1), eg_unit_free(ctx, cls), eg_unit_free(ctx, fused);
}

static void reset_device(eg_ctx* ctx, const char* dev_lib) {
    eg_unit *plain = NULL, *active = NULL;
    check(load(ctx, dev_lib, "dcount1", 0xD9, &plain) == EG_OK && load(ctx, dev_lib, "dcount1a", 0xDA, &active) == EG_OK,
          "eg_unit_load dcount1 and dcount1a (a finishing unit on the device, plain and over an active set)");
    if (plain == NULL || active == NULL) return;
    reset_device_case(ctx, plain, 0, 0.0, "eg_runner_reset on a DEVICE plain runner: run again == a fresh runner (bitwise planes, same report)");
    reset_device_case(ctx, active, 4, 0.0, "eg_runner_reset on a DEVICE active-set runner: run again == a fresh runner");
    reset_device_case(ctx, active, 4, 0.5, "eg_runner_reset on a DEVICE reordering active-set runner: run again == a fresh runner");
    struct dcount c;
    int ok = dcount_new(ctx, plain, 0, 0.0, &c) == EG_OK;
    ok = ok && refused(eg_runner_reset(ctx, NULL), EG_BAD_REQUEST, ctx, "runner");
    check(ok, "eg_runner_reset of a NULL runner refused BAD_REQUEST on a device context");
    dcount_free(ctx, &c);
    {   /* a runner's cells are freed in stream order: freeing it returns while work queued before it still runs */
        cudaStream_t own = NULL;
        void *big_a = NULL, *big_b = NULL;
        const size_t big = (size_t)256 << 20;
        struct dcount r;
        int ok = cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc(&big_a, big) == cudaSuccess &&
                 cudaMalloc(&big_b, big) == cudaSuccess && eg_stream_use(ctx, (uint64_t)(uintptr_t)own) == EG_OK &&
                 dcount_new(ctx, active, 4, 0.5, &r) == EG_OK;
        for (int k = 0; ok && k < 100; ++k) ok = cudaMemcpyAsync(big_b, big_a, big, cudaMemcpyDeviceToDevice, own) == cudaSuccess;
        ok = ok && cudaStreamQuery(own) == cudaErrorNotReady;   /* the queued copies are still running before the free */
        ok = ok && eg_runner_free(ctx, r.runner) == EG_OK;
        r.runner = NULL;
        const int returned_early = ok && cudaStreamQuery(own) == cudaErrorNotReady;
        ok = ok && cudaStreamSynchronize(own) == cudaSuccess && eg_stream_use(ctx, 0) == EG_OK;
        check(ok && returned_early, "freeing a device runner returns while the stream's queued work still runs (its cells are freed in stream order)");
        dcount_free(ctx, &r);
        cudaFree(big_a);
        cudaFree(big_b);
        cudaStreamDestroy(own);
    }
    {   /* a runner made on stream A, run there and freed with stream B in scope: B reusing the bytes cannot corrupt A's results */
        cudaStream_t sa = NULL, sb = NULL;
        void *big_a = NULL, *big_b = NULL, *reuse = NULL;
        const size_t big = (size_t)256 << 20;
        static struct dcount fresh, r;
        struct eg_run_report rf, rr;
        int ok = cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking) == cudaSuccess &&
                 cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc(&big_a, big) == cudaSuccess &&
                 cudaMalloc(&big_b, big) == cudaSuccess && dcount_new(ctx, active, 4, 0.5, &fresh) == EG_OK;
        if (ok) dcount_inputs(&fresh), ok = dcount_run(ctx, &fresh, &rf) == EG_OK;
        ok = ok && eg_stream_use(ctx, (uint64_t)(uintptr_t)sa) == EG_OK && dcount_new(ctx, active, 4, 0.5, &r) == EG_OK;
        if (ok) dcount_inputs(&r);
        for (int k = 0; ok && k < 100; ++k) ok = cudaMemcpyAsync(big_b, big_a, big, cudaMemcpyDeviceToDevice, sa) == cudaSuccess;
        ok = ok && eg_run(ctx, r.runner, &rr) == EG_OK;    /* on A after the copies: the loop reads and writes the runner's cells */
        ok = ok && eg_stream_use(ctx, (uint64_t)(uintptr_t)sb) == EG_OK && eg_runner_free(ctx, r.runner) == EG_OK;
        r.runner = NULL;
        ok = ok && cudaMallocAsync(&reuse, 4096, sb) == cudaSuccess && cudaMemsetAsync(reuse, 0xFF, 4096, sb) == cudaSuccess;
        ok = ok && cudaStreamSynchronize(sa) == cudaSuccess && cudaStreamSynchronize(sb) == cudaSuccess && eg_stream_use(ctx, 0) == EG_OK;
        cudaMemcpy(r.k, r.dk, sizeof(r.k), cudaMemcpyDeviceToHost);
        cudaMemcpy(r.mask, r.dm, sizeof(r.mask), cudaMemcpyDeviceToHost);
        check(ok && memcmp(r.k, fresh.k, sizeof(r.k)) == 0 && memcmp(r.mask, fresh.mask, sizeof(r.mask)) == 0,
              "a runner made on stream A, run there and freed with stream B in scope: A's results equal a fresh runner's (B reusing the bytes cannot corrupt them)");
        if (reuse) cudaFreeAsync(reuse, sb);
        cudaStreamSynchronize(sb);
        dcount_free(ctx, &r);
        dcount_free(ctx, &fresh);
        cudaFree(big_a);
        cudaFree(big_b);
        cudaStreamDestroy(sa);
        cudaStreamDestroy(sb);
    }
    eg_unit_free(ctx, plain);
    eg_unit_free(ctx, active);
}

/* ---- step graphs on a device runner -------------------------------------- */

/* The runner's own planes (k, limit, mask) and the step work's: a second count plan (k2, limit2, mask2) and a copy of k (kc).
 * The step work is "copy k into kc, launch the second count plan" (the launch inside an IF on `cell` when conditional). */
struct dsteps {
    double *dk, *dl, *dk2, *dl2, *dkc;
    unsigned char *dm, *dm2;
    int32_t* dcell;
    struct tensor t[8];
    eg_plane* pl[8];
    eg_plan *p1, *p2;
    double k[DN], k2[DN], kc[DN];
};

static eg_status dsteps_new(eg_ctx* ctx, eg_unit* unit, eg_unit* plain, int32_t flag, struct dsteps* q) {
    static double limit[DN];
    static unsigned char mask[DN], zero[DN];
    memset(q, 0, sizeof(*q));
    memset(zero, 0, sizeof(zero));
    for (int i = 0; i < DN; ++i) limit[i] = (double)(1 + (i * 37) % 50), mask[i] = i % 97 == 0;
    void** d[7] = { (void**)&q->dk, (void**)&q->dl, (void**)&q->dk2, (void**)&q->dl2, (void**)&q->dkc, (void**)&q->dm, (void**)&q->dm2 };
    const size_t bytes[7] = { sizeof(limit), sizeof(limit), sizeof(limit), sizeof(limit), sizeof(limit), DN, DN };
    for (int i = 0; i < 7; ++i) cudaMalloc(d[i], bytes[i]), cudaMemset(*d[i], 0, bytes[i]);
    cudaMalloc((void**)&q->dcell, sizeof(int32_t));
    cudaMemcpy(q->dl, limit, sizeof(limit), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dl2, limit, sizeof(limit), cudaMemcpyHostToDevice);
    cudaMemcpy(q->dm, mask, DN, cudaMemcpyHostToDevice);
    cudaMemcpy(q->dcell, &flag, sizeof(flag), cudaMemcpyHostToDevice);
    q->pl[0] = wrap(ctx, &q->t[0], q->dk, 1, DN, 0, 1, 0);
    q->pl[1] = wrap(ctx, &q->t[1], q->dl, 1, DN, 0, 1, 0);
    q->pl[2] = wrap_device_typed(ctx, &q->t[2], q->dm, DN, kDLBool, 8);
    q->pl[3] = wrap(ctx, &q->t[3], q->dk2, 1, DN, 0, 1, 0);
    q->pl[4] = wrap(ctx, &q->t[4], q->dl2, 1, DN, 0, 1, 0);
    q->pl[5] = wrap_device_typed(ctx, &q->t[5], q->dm2, DN, kDLBool, 8);
    q->pl[6] = wrap(ctx, &q->t[6], q->dkc, 1, DN, 0, 1, 0);
    q->pl[7] = wrap_device_typed(ctx, &q->t[7], q->dcell, 1, kDLInt, 32);
    for (int i = 0; i < 8; ++i)
        if (q->pl[i] == NULL) return EG_FAULT;
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* first[3] = { q->pl[0], q->pl[1], q->pl[2] };
    eg_plane* second[3] = { q->pl[3], q->pl[4], q->pl[5] };
    eg_status s = plan_n(ctx, unit, EG_STRUCTURE_DEVICE, DN, 0, NULL, NULL, &q->p1);
    if (s == EG_OK) s = bind_planes(ctx, q->p1, 3, slots, first);
    if (s == EG_OK) s = plan_n(ctx, plain, EG_STRUCTURE_DEVICE, DN, 0, NULL, NULL, &q->p2);
    if (s == EG_OK) s = bind_planes(ctx, q->p2, 3, slots, second);
    return s;
}

static void dsteps_free(eg_ctx* ctx, struct dsteps* q) {
    eg_plan_free(ctx, q->p1);
    eg_plan_free(ctx, q->p2);
    for (int i = 0; i < 8; ++i) eg_plane_free(ctx, q->pl[i]);
    cudaFree(q->dk), cudaFree(q->dl), cudaFree(q->dk2), cudaFree(q->dl2), cudaFree(q->dkc), cudaFree(q->dm), cudaFree(q->dm2),
        cudaFree(q->dcell);
}

static eg_graph* dsteps_record(eg_ctx* ctx, struct dsteps* q, int conditional) {
    eg_graph* g = NULL;
    int ok = eg_record_begin(ctx) == EG_OK && eg_copy(ctx, q->pl[0], q->pl[6]) == EG_OK;
    if (ok && conditional) {
        struct eg_cond_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.kind = EG_COND_IF;
        d.cell = q->pl[7];
        ok = eg_record_cond_begin(ctx, &d) == EG_OK && eg_launch(ctx, q->p2, NULL) == EG_OK && eg_record_cond_end(ctx) == EG_OK;
    } else if (ok) {
        ok = eg_launch(ctx, q->p2, NULL) == EG_OK;
    }
    return ok && eg_record_end(ctx, NULL, &g) == EG_OK ? g : NULL;
}

/* One batch: the step work as a step graph of a runner over `unit` (graphed), or as calls ahead of the count launch for
 * `plain_steps` steps (unit is then dcount1). Returns whether every call succeeded; k, k2 and kc are read back into q. */
static int dsteps_case(eg_ctx* ctx, eg_unit* unit, eg_unit* plain, int64_t every, int conditional, int32_t flag, int graphed,
                       int64_t plain_steps, struct dsteps* q) {
    eg_status s = dsteps_new(ctx, unit, plain, flag, q);
    if (s == EG_OK && graphed) {
        eg_graph* g = dsteps_record(ctx, q, conditional);
        eg_runner* r = NULL;
        struct eg_runner_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.n_plans = 1;
        d.plans = &q->p1;
        d.max_steps = 1000;
        d.every = every;
        d.step_graph = g;
        struct eg_run_report rep;
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        s = g ? eg_runner_new(ctx, &d, &r) : EG_FAULT;
        if (s == EG_OK && graphed == 2) s = eg_graph_free(ctx, g);    /* the runner's reference keeps the graph alive */
        if (s == EG_OK) s = eg_run(ctx, r, &rep);
        if (s == EG_OK) s = eg_sync(ctx);
        if (s == EG_OK) s = eg_runner_free(ctx, r);
        if (s == EG_OK && graphed != 2) s = eg_graph_free(ctx, g);
    } else if (s == EG_OK) {
        for (int64_t st = 0; s == EG_OK && st < plain_steps; ++st) {
            s = eg_copy(ctx, q->pl[0], q->pl[6]);
            if (s == EG_OK && (!conditional || flag != 0)) s = eg_launch(ctx, q->p2, NULL);
            if (s == EG_OK) s = eg_launch(ctx, q->p1, NULL);
        }
        if (s == EG_OK) s = eg_sync(ctx);
    }
    if (s != EG_OK) printf("      step graph case failed with status %d\n", (int)s), reason_has(ctx, "\1");
    cudaMemcpy(q->k, q->dk, sizeof(q->k), cudaMemcpyDeviceToHost);
    cudaMemcpy(q->k2, q->dk2, sizeof(q->k2), cudaMemcpyDeviceToHost);
    cudaMemcpy(q->kc, q->dkc, sizeof(q->kc), cudaMemcpyDeviceToHost);
    return s == EG_OK;
}

static void step_graph_device(eg_ctx* ctx, const char* dev_lib) {
    eg_unit *plain = NULL, *active = NULL;
    if (load(ctx, dev_lib, "dcount1", 0xD9, &plain) != EG_OK || load(ctx, dev_lib, "dcount1a", 0xDA, &active) != EG_OK) {
        check(0, "step graph: dcount1 and dcount1a loaded");
        return;
    }
    static struct dsteps g, p;
    static double want_k2[DN], want_kc[DN];
    const int64_t need = 50;    /* the longest limit among the live samples: steps the batch takes */
    for (int i = 0; i < DN; ++i) {
        const double limit = (double)(1 + (i * 37) % 50);
        want_k2[i] = limit;     /* the second count plan counts up to its limit, one a step */
        want_kc[i] = i % 97 == 0 ? 0.0 : (limit < (double)(need - 1) ? limit : (double)(need - 1));    /* k before the last step */
    }
    const struct { int64_t every; double theta; int conditional; int32_t flag; const char* what; } rows[] = {
        { 0, 0.0, 0, 1, "a device step graph of a copy and a launch runs once per step inside the captured loop: bitwise the plain loop, and the closed form" },
    };
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
        const int on = rows[i].flag != 0;
        int ok = dsteps_case(ctx, plain, plain, rows[i].every, rows[i].conditional, rows[i].flag, 1, 0, &g) &&
                 dsteps_case(ctx, plain, plain, rows[i].every, rows[i].conditional, rows[i].flag, 0, on ? need : need, &p);
        ok = ok && memcmp(g.k, p.k, sizeof(g.k)) == 0 && memcmp(g.k2, p.k2, sizeof(g.k2)) == 0 && memcmp(g.kc, p.kc, sizeof(g.kc)) == 0;
        if (on) ok = ok && memcmp(g.k2, want_k2, sizeof(g.k2)) == 0 && memcmp(g.kc, want_kc, sizeof(g.kc)) == 0;
        else for (int j = 0; ok && j < DN; ++j) ok = g.k2[j] == 0.0;
        check(ok, rows[i].what);
        dsteps_free(ctx, &g);
        dsteps_free(ctx, &p);
    }
    {   /* the graph freed by its caller before the run: the runner holds it */
        int ok = dsteps_case(ctx, plain, plain, 0, 0, 1, 2, 0, &g) && dsteps_case(ctx, plain, plain, 0, 0, 1, 0, need, &p);
        ok = ok && memcmp(g.k, p.k, sizeof(g.k)) == 0 && memcmp(g.k2, p.k2, sizeof(g.k2)) == 0 && memcmp(g.kc, p.kc, sizeof(g.kc)) == 0 &&
             memcmp(g.k2, want_k2, sizeof(g.k2)) == 0;
        check(ok, "eg_graph_free of a device step graph before the run: the runner's reference keeps it, values bitwise the plain loop");
        dsteps_free(ctx, &g);
        dsteps_free(ctx, &p);
    }
    {   /* a conditional: refused by name, nothing left held */
        eg_status s = dsteps_new(ctx, plain, plain, 1, &g);
        eg_graph* graph = s == EG_OK ? dsteps_record(ctx, &g, 1) : NULL;
        eg_runner* r = NULL;
        struct eg_runner_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.n_plans = 1;
        d.plans = &g.p1;
        d.max_steps = 1000;
        d.step_graph = graph;
        check(graph != NULL && refused(eg_runner_new(ctx, &d, &r), EG_NOT_WRITTEN, ctx, "a conditional inside a device step graph") && r == NULL &&
                  eg_graph_free(ctx, graph) == EG_OK,
              "a device step graph holding a conditional refused NOT_WRITTEN by name (the host takes it)");
        dsteps_free(ctx, &g);
    }
    {   /* an active set (every 4, a reorder): the graph runs ahead of each launch */
        int ok = dsteps_case(ctx, active, plain, 4, 0, 1, 1, 0, &g) && dsteps_case(ctx, plain, plain, 0, 0, 1, 0, 52, &p);
        /* the captured loop runs whole intervals of 4 launches: 13 of them to finish at step 50, so 52 steps with the graph each */
        ok = ok && memcmp(g.k, p.k, sizeof(g.k)) == 0 && memcmp(g.k2, p.k2, sizeof(g.k2)) == 0 && memcmp(g.kc, p.kc, sizeof(g.kc)) == 0;
        check(ok, "a device step graph on a runner with an active set (every 4): bitwise the plain loop of the 52 steps of its 13 whole intervals");
        dsteps_free(ctx, &g);
        dsteps_free(ctx, &p);
    }
    {   /* refusals */
        eg_status s = dsteps_new(ctx, active, plain, 1, &g);
        eg_graph* graph = s == EG_OK ? dsteps_record(ctx, &g, 0) : NULL;
        eg_runner* r = NULL;
        struct eg_runner_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.n_plans = 1;
        d.plans = &g.p1;
        d.max_steps = 1000;
        d.flags = EG_RUN_REORDER;
        d.reorder_theta = 0.5;
        d.step_graph = graph;
        int ok = graph != NULL && refused(eg_runner_new(ctx, &d, &r), EG_BAD_REQUEST, ctx, "only the step_graph touches") && r == NULL;
        d.flags = 0;
        d.reorder_theta = 0;
        ok = ok && eg_runner_new(ctx, &d, &r) == EG_OK && eg_graph_free(ctx, graph) == EG_OK &&
             refused(eg_graph_free(ctx, graph), EG_BAD_REQUEST, ctx, "already freed");
        ok = ok && eg_runner_free(ctx, r) == EG_OK;
        check(ok, "a device step graph: reorder refused for a per-sample plane only the graph touches, the graph freed once while its runner holds it, by name");
        dsteps_free(ctx, &g);
    }
    eg_unit_free(ctx, plain);
    eg_unit_free(ctx, active);
}

/* A device plane of @p n elements of the given DLPack type, wrapped; @p real_n of them are allocated and zeroed (a refusal
 * row never reads the plane, so it may name a length it does not own). */
static eg_plane* zero_plane(eg_ctx* ctx, struct tensor* t, int64_t n, size_t real_n, int code, int bits, void** dev) {
    cudaMalloc(dev, real_n * (size_t)(bits / 8));
    cudaMemset(*dev, 0, real_n * (size_t)(bits / 8));
    memset(t, 0, sizeof(*t));
    t->shape[0] = n;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = *dev;
    t->managed.dl_tensor.device.device_type = kDLCUDA;
    t->managed.dl_tensor.ndim = 1;
    t->managed.dl_tensor.dtype.code = (uint8_t)code;
    t->managed.dl_tensor.dtype.bits = (uint8_t)bits;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    eg_plane* plane = NULL;
    return eg_plane_wrap(ctx, &t->managed, &plane) == EG_OK ? plane : NULL;
}

/* One automatic unit's plan over n samples (hits, geo, done bound) and a runner over it, on the DEVICE. */
struct auto_run {
    struct tensor t[3];
    void* d[3];
    eg_plane* pl[3];
    eg_plan* plan;
    eg_runner* runner;
    eg_status plan_status, runner_status;
};

static void auto_run_new(eg_ctx* ctx, eg_unit* unit, int64_t n, size_t real_n, int64_t parts, const int64_t* base,
                         const int64_t* count, struct auto_run* r) {
    memset(r, 0, sizeof(*r));
    r->pl[0] = zero_plane(ctx, &r->t[0], n, real_n, kDLUInt, 32, &r->d[0]);
    r->pl[1] = zero_plane(ctx, &r->t[1], n, real_n, kDLInt, 64, &r->d[1]);
    r->pl[2] = zero_plane(ctx, &r->t[2], n, real_n, kDLBool, 8, &r->d[2]);
    const int32_t slots[3] = { 0, 1, 2 };
    r->plan_status = plan_n(ctx, unit, EG_STRUCTURE_DEVICE, n, parts, base, count, &r->plan);
    if (r->plan_status == EG_OK) r->plan_status = bind_planes(ctx, r->plan, 3, slots, r->pl);
    if (r->plan_status != EG_OK) return;
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_plans = 1;
    d.plans = &r->plan;
    d.max_steps = 1;
    r->runner_status = eg_runner_new(ctx, &d, &r->runner);
}

static void auto_run_free(eg_ctx* ctx, struct auto_run* r) {
    if (r->runner) eg_runner_free(ctx, r->runner);
    if (r->plan) eg_plan_free(ctx, r->plan);
    for (int i = 0; i < 3; ++i) {
        eg_plane_free(ctx, r->pl[i]);
        cudaFree(r->d[i]);
    }
}

/* The fast (persist) entry of an automatic unit: every sample visited exactly once at several batch sizes, the chunk and
 * block the door chose obey the rule, the grid fits the device, a count that would wrap the fetch counter is refused, and
 * what the door does not run (a range entry, several partitions) is refused by name. The kernel's own count of visits
 * (hits, uint32 atomics) and its counter are the oracles; the entry's registers and residency come from the CUDA runtime
 * loading the fixture's own image. */
static void fast_entries(eg_ctx* ctx, const char* dev_lib) {
    eg_unit *unit = NULL, *ranged = NULL;
    check(load(ctx, dev_lib, "dauto", 0xDB, &unit) == EG_OK && load(ctx, dev_lib, "dautorange", 0xDC, &ranged) == EG_OK,
          "eg_unit_load dauto and dautorange (automatic units with a persist and a range entry)");
    if (unit == NULL || ranged == NULL) return;
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int64_t sizes[3] = { 100003, 600011, 4099 };
    for (int s = 0; s < 3; ++s) {
        const int64_t n = sizes[s];
        struct auto_run r;
        auto_run_new(ctx, unit, n, (size_t)n, 0, NULL, NULL, &r);
        struct eg_run_report rep;
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        unsigned* hits = (unsigned*)malloc((size_t)n * sizeof(unsigned));
        int64_t geo[3] = { 0, 0, 0 };
        int ok = r.runner_status == EG_OK && eg_run(ctx, r.runner, &rep) == EG_OK && eg_sync(ctx) == EG_OK && hits != NULL;
        if (ok) {
            cudaMemcpy(hits, r.d[0], (size_t)n * sizeof(unsigned), cudaMemcpyDeviceToHost);
            cudaMemcpy(geo, r.d[1], sizeof(geo), cudaMemcpyDeviceToHost);
            int64_t wrong = 0;
            for (int64_t i = 0; i < n; ++i) wrong += hits[i] != 1;
            ok = wrong == 0 && rep.mode == EG_MODE_PERSIST && rep.finished == n && (rep.flags & EG_REPORT_DONE) != 0;
            if (!ok)
                printf("      mode %d finished %lld of %lld, %lld samples not visited exactly once\n", (int)rep.mode,
                       (long long)rep.finished, (long long)n, (long long)wrong);
        } else {
            printf("      runner %d, plan %d\n", (int)r.runner_status, (int)r.plan_status);
            reason_has(ctx, "");
        }
        char what[200];
        snprintf(what, sizeof(what), "persist run of %lld samples (grid %lld, block %lld, chunk %lld): every sample visited exactly once, counter == n",
                 (long long)n, (long long)geo[0], (long long)geo[1], (long long)geo[2]);
        check(ok, what);
        if (ok && s == 0) {
            const int64_t chunk = geo[2];
            void* handle = dlopen(dev_lib, RTLD_NOW);
            const void* image = handle ? dlsym(handle, "fixture__image") : NULL;
            cudaLibrary_t lib;
            cudaKernel_t kernel;
            struct cudaFuncAttributes fa;
            int per_sm = 0;
            const int fit = image != NULL && cudaLibraryLoadData(&lib, image, NULL, NULL, 0, NULL, NULL, 0) == cudaSuccess &&
                            cudaLibraryGetKernel(&kernel, lib, "dauto_persist") == cudaSuccess &&
                            cudaFuncGetAttributes(&fa, (const void*)kernel) == cudaSuccess &&
                            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (const void*)kernel, (int)geo[1], 0) == cudaSuccess;
            check(chunk >= 1 && chunk <= (1 << 20) && (chunk & (chunk - 1)) == 0 && geo[1] >= 32 && geo[1] % 32 == 0,
                  "the persist chunk is a power of two in [1, 2^20] and the block is whole warps");
            check(fit && geo[1] <= 512 && geo[1] <= fa.maxThreadsPerBlock && geo[1] * fa.numRegs <= prop.regsPerBlock &&
                      geo[0] >= 1 && geo[0] <= (int64_t)prop.multiProcessorCount * per_sm,
                  "the persist grid never exceeds the blocks that fit resident, the block is at most 512 threads within the registers");
            struct eg_runner_figures rf;
            memset(&rf, 0, sizeof(rf));
            rf.struct_size = sizeof(rf);
            check(eg_runner_figures(ctx, r.runner, &rf) == EG_OK && rf.mode == EG_MODE_PERSIST && rf.grid == geo[0] &&
                      rf.block == geo[1] && rf.chunk == geo[2],
                  "eg_runner_figures of a persist run: mode PERSIST, grid, block and chunk == the launch the kernel saw");
            struct eg_entry_figures ef;
            memset(&ef, 0, sizeof(ef));
            ef.struct_size = sizeof(ef);
            check(fit && eg_plan_entry_figures(ctx, r.plan, &ef) == EG_OK && (ef.flags & EG_ENTRY_PERSIST) != 0 &&
                      ef.persist_regs == fa.numRegs && ef.persist_max_threads == fa.maxThreadsPerBlock,
                  "eg_plan_entry_figures of an automatic unit: EG_ENTRY_PERSIST, persist registers and max threads == the driver's own reads");
        }
        free(hits);
        auto_run_free(ctx, &r);
    }
    /* a count that would wrap the uint32 fetch counter: n plus one overshooting fetch per lane must stay below 2^32 (2^32
     * itself is already refused by the runner, whose finished counter is a uint32) */
    struct auto_run big;
    auto_run_new(ctx, unit, (int64_t)1 << 32, 1024, 0, NULL, NULL, &big);
    check(big.plan_status == EG_OK && refused(big.runner_status, EG_BAD_REQUEST, ctx, "the finished counter is uint32"), "a runner over 2^32 samples is refused BAD_REQUEST (the finished counter is uint32)");
    auto_run_free(ctx, &big);
    const int64_t huge[2] = { ((int64_t)1 << 32) - 1, ((int64_t)1 << 32) - ((int64_t)1 << 21) };
    for (int i = 0; i < 2; ++i) {
        struct auto_run r;
        auto_run_new(ctx, unit, huge[i], 1024, 0, NULL, NULL, &r);
        char what[120];
        snprintf(what, sizeof(what), "a persist runner over %lld samples is %s", (long long)huge[i],
                 i == 0 ? "refused BAD_REQUEST (the fetch counter would wrap)" : "made");
        check(r.plan_status == EG_OK && (i == 0 ? refused(r.runner_status, EG_BAD_REQUEST, ctx, "fewer than 2^32 samples per launch")
                                                : r.runner_status == EG_OK),
              what);
        auto_run_free(ctx, &r);
    }
    /* the range entry and several partitions are not run by the door: refused by name, no silent fallback */
    struct auto_run r;
    auto_run_new(ctx, ranged, 5000, 5000, 0, NULL, NULL, &r);
    check(r.plan_status == EG_OK && refused(r.runner_status, EG_NOT_WRITTEN, ctx, "the range entry '_range'"),
          "an automatic unit with a range entry: eg_runner_new refused NOT_WRITTEN by name");
    auto_run_free(ctx, &r);
    const int64_t base[2] = { 0, 2500 }, count[2] = { 2500, 2500 };
    auto_run_new(ctx, unit, 5000, 5000, 2, base, count, &r);
    check(r.plan_status == EG_OK && refused(r.runner_status, EG_NOT_WRITTEN, ctx, "2 partitions"),
          "an automatic unit over two partitions: eg_runner_new refused NOT_WRITTEN by name");
    auto_run_free(ctx, &r);
    eg_unit_free(ctx, unit);
    eg_unit_free(ctx, ranged);
}

/* A lookup set on the DEVICE: lane i < *count writes the sample map[i]. A plain launch writes exactly the mapped samples (plain-C
 * oracle over the map); a recorded graph replays with a smaller count the caller writes into its own plane (no re-record);
 * eg_set_count is refused (the count is the caller's plane) and a count above the map's entries is refused at launch. */
static void lookup_device(eg_ctx* ctx, const char* dev_lib) {
    enum { SM = 64, ENTRIES = 16 };
    double x[SM], y0[SM], want[SM], got[SM];
    int32_t map[ENTRIES];
    eg_unit* unit = NULL;
    check(load(ctx, dev_lib, "dlookup", 0xDD, &unit) == EG_OK, "eg_unit_load dlookup (a lookup set on the device)");
    if (unit == NULL) return;
    for (int i = 0; i < SM; ++i) x[i] = (double)i * 0.5 - 3.0, y0[i] = -1.0;
    for (int i = 0; i < ENTRIES; ++i) map[i] = (i * 5 + 3) % SM;
    double *dy = dev_doubles(y0, SM), *dx = dev_doubles(x, SM);
    int32_t* dm = NULL;
    uint32_t* dc = NULL;
    cudaMalloc((void**)&dm, sizeof(map));
    cudaMalloc((void**)&dc, sizeof(uint32_t));
    cudaMemcpy(dm, map, sizeof(map), cudaMemcpyHostToDevice);
    struct tensor ty, tx, tm, tc;
    eg_plane *py = wrap(ctx, &ty, dy, 1, SM, 0, 1, 0), *px = wrap(ctx, &tx, dx, 1, SM, 0, 1, 0);
    eg_plane *pm = wrap_device_typed(ctx, &tm, dm, ENTRIES, kDLInt, 32), *pc = wrap_device_typed(ctx, &tc, dc, 1, kDLUInt, 32);
    int ok = py && px && pm && pc;
    eg_plan* p = NULL;
    const int32_t slots[4] = { 0, 1, 2, 3 };
    eg_plane* planes[4] = { py, px, pm, pc };
    uint32_t count = ENTRIES;
    cudaMemcpy(dc, &count, sizeof(count), cudaMemcpyHostToDevice); /* binding reads the count: it is written first */
    ok = ok && plan_n(ctx, unit, EG_STRUCTURE_DEVICE, SM, 0, NULL, NULL, &p) == EG_OK && bind_planes(ctx, p, 4, slots, planes) == EG_OK;
    ok = ok && eg_launch(ctx, p, NULL) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
    for (int i = 0; i < SM; ++i) want[i] = y0[i];
    for (int i = 0; i < ENTRIES; ++i) want[map[i]] = 2.0 * x[map[i]] + 1.0;
    ok = ok && memcmp(got, want, sizeof(got)) == 0;
    check(ok, "a DEVICE lookup set: a plain launch writes exactly the mapped samples, the others stay untouched (bitwise)");
    for (int i = 0; i < SM; ++i) x[i] = -x[i] - 7.0, y0[i] = -1.0;
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y0, sizeof(y0), cudaMemcpyHostToDevice);
    eg_graph* g = NULL;
    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, NULL, &g) == EG_OK;
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
    for (int i = 0; i < SM; ++i) want[i] = y0[i];
    for (int i = 0; i < ENTRIES; ++i) want[map[i]] = 2.0 * x[map[i]] + 1.0;
    ok = ok && memcmp(got, want, sizeof(got)) == 0;
    for (int i = 0; i < SM; ++i) x[i] += 100.0;
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
    count = 6;
    cudaMemcpy(dc, &count, sizeof(count), cudaMemcpyHostToDevice);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
    cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
    for (int i = 0; i < (int)count; ++i) want[map[i]] = 2.0 * x[map[i]] + 1.0;
    ok = ok && memcmp(got, want, sizeof(got)) == 0;
    if (!ok) printf("      sample %d: %.17g (want %.17g)\n", map[0], got[map[0]], want[map[0]]);
    check(ok, "a DEVICE lookup set recorded once: after the caller writes a smaller count into its plane, the replay refreshes the first entries' samples and the rest keep theirs (bitwise)");
    ok = refused(eg_set_count(ctx, p, 1), EG_BAD_REQUEST, ctx, "is a lookup set: its count is the caller's plane, bind it again");
    count = ENTRIES + 1;
    cudaMemcpy(dc, &count, sizeof(count), cudaMemcpyHostToDevice);
    ok = ok && refused(eg_launch(ctx, p, NULL), EG_BAD_REQUEST, ctx, "above the map's 16 entries");
    check(ok, "a DEVICE lookup set: eg_set_count refused BAD_REQUEST (the count is the caller's plane), a count above the map's entries refused at launch by reason");
    if (g) eg_graph_free(ctx, g);
    if (p) eg_plan_free(ctx, p);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, pm);
    eg_plane_free(ctx, pc);
    eg_unit_free(ctx, unit);
}

/* ---- transforms ---------------------------------------------------------- */

static void fft_move(void* dst, const void* src, size_t bytes) { cudaMemcpy(dst, src, bytes, cudaMemcpyDefault); }

#include "fft_cases.inc"

/* A device plan equals the host plan within the width's tolerance: every kind and adjoint, norm and shape. */
static void fft_cross(eg_ctx* dev, eg_ctx* host) {
    static const struct { eg_fft_kind kind; uint32_t flags; } kinds[] = {
        { EG_FFT_C2C, 0 }, { EG_FFT_C2C, EG_FFT_INVERSE }, { EG_FFT_R2C, 0 }, { EG_FFT_C2R, 0 },
        { EG_FFT_C2C, EG_FFT_ADJOINT }, { EG_FFT_R2C, EG_FFT_ADJOINT }, { EG_FFT_C2R, EG_FFT_ADJOINT } };
    for (int w = 0; w < 2; ++w) {
        const uint32_t bits = w == 0 ? 32 : 64;
        int ok = 1;
        for (int k = 0; k < 7; ++k)
            for (int g = 0; g < FFT_GEOMS && ok; ++g)
                for (int norm = EG_FFT_NORM_BACKWARD; norm <= EG_FFT_NORM_ORTHO && ok; ++norm) {
                    const struct fft_case c = fft_case_of(&fft_geoms[g], kinds[k].kind, kinds[k].flags, norm, bits);
                    int64_t is[4], os[4];
                    const size_t n_in = fft_floats(is, fft_plane(&c, 1, is)), n_out = fft_floats(os, fft_plane(&c, 0, os));
                    double* in = malloc(n_in * sizeof(double));
                    double* on_dev = malloc(n_out * sizeof(double));
                    double* on_host = malloc(n_out * sizeof(double));
                    fft_fill(in, n_in, bits);
                    ok = fft_apply(dev, 1, &c, in, on_dev) == EG_OK && fft_apply(host, 0, &c, in, on_host) == EG_OK;
                    double top = 0;
                    for (size_t i = 0; ok && i < n_out; ++i) top = fabs(on_host[i]) > top ? fabs(on_host[i]) : top;
                    const double tol = 8 * fft_eps(bits) * ((double)fft_points(&c) + 4) * (1 + top);
                    for (size_t i = 0; ok && i < n_out; ++i) ok = fabs(on_dev[i] - on_host[i]) <= tol;
                    if (!ok) printf("      kind %d flags %u geometry %d norm %d: tolerance %g\n", kinds[k].kind, kinds[k].flags, g, norm, tol);
                    free(in);
                    free(on_dev);
                    free(on_host);
                }
        char what[128];
        snprintf(what, sizeof(what), "eg_fft_run float%u: the device plan equals the host plan within 8*eps*(n+4)*(1+max), every kind, adjoint, norm and shape", bits);
        check(ok, what);
    }
}

/* A plane of @p n elements of @p code / @p bits over @p data, on the device (@p cuda) or the host. */
static eg_plane* wrap_as(eg_ctx* ctx, struct tensor* t, void* data, int64_t n, int cuda, uint8_t code, uint8_t bits) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = n;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = cuda ? kDLCUDA : kDLCPU;
    t->managed.dl_tensor.ndim = 1;
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

/* eg_copy across the host and the device: host -> device -> device -> host returns the input bitwise for each dtype
 * (same data, so the oracle is the input bytes); a shape or dtype mismatch and a copy inside a recording are refused by
 * status and sentence; a same-side device copy still records. */
static void copy_across(eg_ctx* ctx) {
    enum { COUNT = 1000 };
    static const struct { uint8_t code, bits; const char* name; } kinds[5] = {
        { kDLFloat, 64, "f64" }, { kDLFloat, 32, "f32" }, { kDLInt, 64, "i64" }, { kDLInt, 32, "i32" }, { kDLInt, 8, "i8" },
    };
    for (int k = 0; k < 5; ++k) {
        const size_t bytes = (size_t)COUNT * (kinds[k].bits / 8);
        unsigned char *in = malloc(bytes), *out = calloc(bytes, 1), *dev_a = NULL, *dev_b = NULL;
        unsigned state = 12345u + (unsigned)k;
        for (size_t i = 0; in && i < bytes; ++i) in[i] = (unsigned char)((state = state * 1664525u + 1013904223u) >> 24);
        cudaMalloc((void**)&dev_a, bytes);
        cudaMalloc((void**)&dev_b, bytes);
        struct tensor th_in, th_out, td_a, td_b;
        eg_plane* h_in = wrap_as(ctx, &th_in, in, COUNT, 0, kinds[k].code, kinds[k].bits);
        eg_plane* h_out = wrap_as(ctx, &th_out, out, COUNT, 0, kinds[k].code, kinds[k].bits);
        eg_plane* d_a = wrap_as(ctx, &td_a, dev_a, COUNT, 1, kinds[k].code, kinds[k].bits);
        eg_plane* d_b = wrap_as(ctx, &td_b, dev_b, COUNT, 1, kinds[k].code, kinds[k].bits);
        char what[160];
        snprintf(what, sizeof(what), "eg_copy host -> device -> device -> host returns the input bitwise (%s)", kinds[k].name);
        int ok = in && out && h_in && h_out && d_a && d_b && eg_copy(ctx, h_in, d_a) == EG_OK && eg_copy(ctx, d_a, d_b) == EG_OK &&
                 eg_copy(ctx, d_b, h_out) == EG_OK && memcmp(in, out, bytes) == 0;
        check(ok, what);
        if (k == 0) {
            eg_plane* shorter = NULL;
            struct tensor ts;
            shorter = wrap_as(ctx, &ts, dev_b, COUNT - 1, 1, kinds[0].code, 64);
            check(shorter && refused(eg_copy(ctx, h_in, shorter), EG_BAD_REQUEST, ctx, "one shape and dtype"),
                  "eg_copy host -> device of another shape: BAD_REQUEST, one shape and dtype");
            eg_plane_free(ctx, shorter);
            shorter = wrap_as(ctx, &ts, dev_b, COUNT, 1, kDLInt, 64);
            check(shorter && refused(eg_copy(ctx, shorter, h_in), EG_BAD_REQUEST, ctx, "one shape and dtype"),
                  "eg_copy device -> host of another dtype of the same width: BAD_REQUEST, one shape and dtype");
            eg_plane_free(ctx, shorter);
            eg_graph* g = NULL;
            cudaMemset(dev_b, 0, bytes);
            ok = eg_record_begin(ctx) == EG_OK &&
                 refused(eg_copy(ctx, h_in, d_a), EG_NOT_WRITTEN, ctx, "a host<->device copy is not recordable; copy before or after the recording") &&
                 refused(eg_copy(ctx, d_a, h_out), EG_NOT_WRITTEN, ctx, "not recordable") && eg_copy(ctx, d_a, d_b) == EG_OK &&
                 eg_record_end(ctx, NULL, &g) == EG_OK;
            ok = ok && eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK;
            unsigned char* back = calloc(bytes, 1);
            ok = ok && back && cudaMemcpy(back, dev_b, bytes, cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(back, in, bytes) == 0;
            check(ok, "inside a recording a host<->device eg_copy is refused NOT_WRITTEN by name and a device -> device one records and replays");
            free(back);
            if (g) eg_graph_free(ctx, g);
        }
        eg_plane_free(ctx, h_in);
        eg_plane_free(ctx, h_out);
        eg_plane_free(ctx, d_a);
        eg_plane_free(ctx, d_b);
        cudaFree(dev_a);
        cudaFree(dev_b);
        free(in);
        free(out);
    }
}


/* The graph builder on a device context: forks and joins, waits, conditionals (IF, IF_ELSE, WHILE with a cap, a negated test, a lane
 * of a flag array, a count against a baseline) and a rebound plan, each replayed and compared with the add-order run bitwise (dprefix:
 * y = 2 x + 1 over the first count lanes). The race row runs on a context of its own: a refused fork_end leaves the recording open. */
enum { BN = 64 };

struct gb {
    eg_ctx* ctx;
    eg_unit* unit;
    struct tensor t[7];
    double *dx, *d1, *d2, *d3;
    uint32_t *dc, *df;
    eg_plane *px, *p1, *p2, *p3, *pc, *pf, *pd;
    eg_plan *a, *b, *c, *d;    /* a: p1 <- x, b: p2 <- x, c: p2 <- p1, d: p1 <- x again */
};

static int gb_make(struct gb* g, const char* dev_lib) {
    static double x[BN];
    for (int i = 0; i < BN; ++i) x[i] = 0.5 * i - 7.0;
    eg_status s = EG_OK;
    memset(g, 0, sizeof(*g));
    g->ctx = context(EG_CTX_DEVICE, 0, &s);
    if (s != EG_OK || load(g->ctx, dev_lib, "dprefix", 0xD6, &g->unit) != EG_OK) return 0;
    uint32_t count = BN;
    g->dx = dev_doubles(x, BN), g->d1 = dev_doubles(NULL, BN), g->d2 = dev_doubles(NULL, BN), g->d3 = dev_doubles(NULL, BN);
    cudaMalloc((void**)&g->dc, sizeof(count));
    cudaMemcpy(g->dc, &count, sizeof(count), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&g->df, 4 * sizeof(uint32_t));
    cudaMemset(g->df, 0, 4 * sizeof(uint32_t));
    eg_ctx* c = g->ctx;
    g->px = wrap(c, &g->t[0], g->dx, 1, BN, 0, 1, 0), g->p1 = wrap(c, &g->t[1], g->d1, 1, BN, 0, 1, 0), g->p2 = wrap(c, &g->t[2], g->d2, 1, BN, 0, 1, 0);
    g->p3 = wrap(c, &g->t[3], g->d3, 1, BN, 0, 1, 0);
    g->pc = wrap_typed(c, &g->t[4], g->dc, kDLUInt, 32, 1, 1), g->pf = wrap_typed(c, &g->t[5], g->df, kDLUInt, 32, 4, 1);
    g->pd = wrap(c, &g->t[6], g->d3, 1, 4, 0, 1, 0);    /* a float64 plane: not a condition cell */
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* in[4][3] = { { g->p1, g->px, g->pc }, { g->p2, g->px, g->pc }, { g->p2, g->p1, g->pc }, { g->p1, g->px, g->pc } };
    eg_plan** plans[4] = { &g->a, &g->b, &g->c, &g->d };
    int ok = g->px && g->p1 && g->p2 && g->p3 && g->pc && g->pf && g->pd;
    for (int k = 0; ok && k < 4; ++k)
        ok = plan_n(c, g->unit, EG_STRUCTURE_DEVICE, BN, 0, NULL, NULL, plans[k]) == EG_OK && bind_planes(c, *plans[k], 3, slots, in[k]) == EG_OK;
    return ok;
}

static void gb_free(struct gb* g) {
    eg_ctx_free(g->ctx);    /* the plans, planes and unit go with it; an open recording is dropped */
    cudaFree(g->dx), cudaFree(g->d1), cudaFree(g->d2), cudaFree(g->d3), cudaFree(g->dc), cudaFree(g->df);
}

static void gb_clear(struct gb* g) {
    cudaMemset(g->d1, 0, BN * sizeof(double)), cudaMemset(g->d2, 0, BN * sizeof(double)), cudaMemset(g->d3, 0, BN * sizeof(double));
    cudaDeviceSynchronize();
}

static void gb_flags(struct gb* g, uint32_t f0, uint32_t f1, uint32_t f2) {
    const uint32_t f[4] = { f0, f1, f2, 0 };
    cudaMemcpy(g->df, f, sizeof(f), cudaMemcpyHostToDevice);
}

/* 1: the plane holds 2 x + 1 (what: 1) or zeros (what: 0), bitwise. */
static int gb_holds(const double* dev, int what) {
    double got[BN];
    cudaDeviceSynchronize();
    cudaMemcpy(got, dev, sizeof(got), cudaMemcpyDeviceToHost);
    for (int i = 0; i < BN; ++i) {
        const double want = what ? 2.0 * (0.5 * i - 7.0) + 1.0 : 0.0;
        if (memcmp(&got[i], &want, sizeof(want)) != 0) return 0;
    }
    return 1;
}

static eg_status gb_cond(struct gb* g, int kind, uint32_t flags, int32_t arms, eg_plane* cell, int64_t index, eg_plane* baseline, int64_t base_index,
                         int64_t max_iters) {
    struct eg_cond_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.kind = kind, d.flags = flags, d.n_arms = arms, d.cell = cell, d.cell_index = index;
    d.baseline = baseline, d.baseline_index = base_index, d.max_iters = max_iters;
    return eg_record_cond_begin(g->ctx, &d);
}

/* One replay of the graph, then eg_sync: OK. */
/* A description's node count, edge count and node kinds as a string, to compare two recordings. */
static int gb_shape(eg_ctx* c, const eg_graph* gr, int* n_nodes, int* n_edges, char* kinds, size_t room) {
    struct eg_graph_description d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    if (eg_graph_describe(c, gr, &d) != EG_OK || (size_t)d.n_nodes + 1 > room) return 0;
    for (int i = 0; i < d.n_nodes; ++i) kinds[i] = (char)('A' + ((const struct eg_graph_node*)((const char*)d.nodes + (size_t)i * d.node_size))->kind);
    kinds[d.n_nodes] = 0;
    *n_nodes = d.n_nodes;
    *n_edges = d.n_edges;
    return 1;
}

static int gb_replay(struct gb* g, eg_graph* gr) { return eg_graph_launch(g->ctx, gr) == EG_OK && eg_sync(g->ctx) == EG_OK; }

static int gb_edge(const struct eg_graph_description* d, int from, int to, eg_gedge_kind kind, const eg_plane* plane) {
    for (int i = 0; i < d->n_edges; ++i)
        if (d->edges[i].from == from && d->edges[i].to == to && d->edges[i].kind == kind && d->edges[i].plane == plane) return 1;
    return 0;
}

static void gb_dump(const struct eg_graph_description* d) {
    printf("      description: %d nodes, %d edges, %d levels\n", d->n_nodes, d->n_edges, d->n_levels);
    for (int i = 0; i < d->n_nodes; ++i)
        printf("      node %d: kind %d parent %d level %d arm %d unit %s\n", i, d->nodes[i].kind, d->nodes[i].parent, d->nodes[i].level, d->nodes[i].arm,
               d->nodes[i].unit ? d->nodes[i].unit : "-");
    for (int i = 0; i < d->n_edges; ++i)
        printf("      edge %d -> %d: kind %d plane %p\n", d->edges[i].from, d->edges[i].to, d->edges[i].kind, (void*)d->edges[i].plane);
}

/* A recording a failed row left open would fail every row after it: start over on a fresh context. */
#define GB_SECTION_END()                                         \
    do {                                                         \
        if (failures != mark) {                                  \
            gb_free(&g);                                         \
            if (!gb_make(&g, dev_lib)) return;                   \
            c = g.ctx, gr = NULL;                                \
        }                                                        \
        mark = failures;                                         \
    } while (0)

static void graph_builder_device(const char* dev_lib) {
    struct gb g;
    check(gb_make(&g, dev_lib), "graph builder: a device context, dprefix and its planes made");
    if (g.ctx == NULL || g.a == NULL || g.d == NULL) return;
    eg_ctx* c = g.ctx;
    eg_graph* gr = NULL;
    int ok, mark = failures;

    /* fork and join */
    gb_clear(&g);
    ok = eg_record_begin(c) == EG_OK && eg_record_fork_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_fork_branch(c) == EG_OK &&
         eg_launch(c, g.b, NULL) == EG_OK && eg_record_fork_end(c) == EG_OK && eg_record_end(c, NULL, &gr) == EG_OK;
    if (!ok) reason_has(c, "(the fork recording was refused)");
    ok = ok && gb_replay(&g, gr) && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 1) && gb_holds(g.d2, 1), "device fork/join: two branches writing disjoint planes replay to the add-order values");
    if (gr) eg_graph_free(c, gr);
    gr = NULL;
    GB_SECTION_END();

    /* wait: its refusals, then a dependent branch ordered after the producer */
    gb_clear(&g);
    eg_plane* w1[1] = { g.p1 };
    eg_plane* wx[1] = { g.px };
    eg_plane* w2[1] = { g.p2 };
    ok = eg_record_begin(c) == EG_OK && refused(eg_record_wait(c, 1, w1), EG_BAD_REQUEST, c, "no fork is open") &&
         eg_record_fork_begin(c) == EG_OK && refused(eg_record_fork_branch(c), EG_BAD_REQUEST, c, "empty") && eg_launch(c, g.a, NULL) == EG_OK &&
         refused(eg_record_wait(c, 1, w1), EG_BAD_REQUEST, c, "written by this branch") && eg_record_fork_branch(c) == EG_OK &&
         refused(eg_record_wait(c, 1, wx), EG_BAD_REQUEST, c, "written by no earlier sibling branch") && eg_record_wait(c, 1, w1) == EG_OK &&
         eg_launch(c, g.c, NULL) == EG_OK && refused(eg_record_wait(c, 1, w2), EG_BAD_REQUEST, c, "written by this branch") &&
         eg_record_fork_end(c) == EG_OK && eg_record_end(c, NULL, &gr) == EG_OK;
    if (!ok) reason_has(c, "(the wait recording was refused)");
    check(ok, "device wait: refused outside a branch, for an empty branch, for a plane this branch writes and one no earlier branch writes; accepted after its producer");
    for (int k = 0; ok && k < 3; ++k) ok = gb_replay(&g, gr);
    {
        double got[BN];
        cudaMemcpy(got, g.d2, sizeof(got), cudaMemcpyDeviceToHost);
        int same = 1;
        for (int i = 0; i < BN; ++i) {
            const double want = 2.0 * (2.0 * (0.5 * i - 7.0) + 1.0) + 1.0;
            same &= memcmp(&got[i], &want, sizeof(want)) == 0;
        }
        check(ok && same, "device wait: the dependent branch ran after its producer, 3 replays: 2 (2 x + 1) + 1 bitwise");
    }
    {   /* the description the door built at record time: a fork of two branches, a wait, a launch ordered after its producer */
        struct eg_graph_description d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        const int described = ok && eg_graph_describe(c, gr, &d) == EG_OK && d.n_nodes == 7;
        const int dumped_before = failures;
        check(described && d.nodes[0].kind == EG_GNODE_FORK && d.nodes[1].kind == EG_GNODE_BRANCH && d.nodes[1].parent == 0 && d.nodes[2].kind == EG_GNODE_LAUNCH &&
                  d.nodes[2].parent == 1 && d.nodes[3].kind == EG_GNODE_BRANCH && d.nodes[3].parent == 0 && d.nodes[4].kind == EG_GNODE_WAIT && d.nodes[4].parent == 3 &&
                  d.nodes[5].kind == EG_GNODE_LAUNCH && d.nodes[5].parent == 3 && d.nodes[6].kind == EG_GNODE_JOIN && d.nodes[6].parent == -1,
              "device description: fork, two branches, a wait, two launches and the join, with their parents");
        check(described && gb_edge(&d, 2, 4, EG_GEDGE_WAIT, g.p1) && gb_edge(&d, 4, 5, EG_GEDGE_ORDER, NULL) && gb_edge(&d, 2, 6, EG_GEDGE_JOIN, NULL) && gb_edge(&d, 5, 6, EG_GEDGE_JOIN, NULL) && d.nodes[5].level > d.nodes[2].level && d.n_levels >= 3 &&
                  d.nodes[2].unit != NULL && strcmp(d.nodes[2].unit, "dprefix") == 0,
              "device description: the WAIT edge from the producer to the wait names its plane, the reader follows the wait, both launches join, and the reader sits on a later level");
        if (failures != dumped_before && ok) {
            printf("      the planes: p1 %p\n", (void*)g.p1);
            gb_dump(&d);
        }
    }
    if (gr) eg_graph_free(c, gr);
    gr = NULL;
    GB_SECTION_END();

    /* conditionals over a lane of a flag array */
    gb_clear(&g);
    gb_flags(&g, 0, 0, 1);
    ok = eg_record_begin(c) == EG_OK && gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 2, NULL, 0, 0) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK &&
         eg_record_cond_end(c) == EG_OK && eg_record_end(c, NULL, &gr) == EG_OK;
    if (!ok) reason_has(c, "(the IF recording was refused)");
    ok = ok && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 1), "device IF on lane 2 of a flag array: set, the arm runs");
    gb_clear(&g);
    gb_flags(&g, 0, 0, 0);
    ok = ok && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 0), "device IF: the caller rewrites lane 2 to 0 between replays, the arm does not run");
    gb_clear(&g);
    gb_flags(&g, 0, 0, 7);
    ok = ok && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 1), "device IF: lane 2 rewritten to 7, the arm runs again (the test is read at each replay)");
    {   /* a conditional's test, its arm and the launch in it */
        struct eg_graph_description d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        check(ok && eg_graph_describe(c, gr, &d) == EG_OK && d.n_nodes == 3 && d.nodes[0].kind == EG_GNODE_COND && d.nodes[0].arm == EG_COND_IF && d.nodes[0].cell_index == 2 &&
                  d.nodes[1].kind == EG_GNODE_ARM && d.nodes[1].parent == 0 && d.nodes[2].kind == EG_GNODE_LAUNCH && d.nodes[2].parent == 1,
              "device description: a conditional's test (its kind and cell index), its arm and the launch inside it");
    }
    if (gr) eg_graph_free(c, gr);
    gr = NULL;
    GB_SECTION_END();

    gb_clear(&g);
    gb_flags(&g, 0, 0, 0);
    ok = eg_record_begin(c) == EG_OK && gb_cond(&g, EG_COND_IF, EG_COND_NEGATE, 0, g.pf, 2, NULL, 0, 0) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK &&
         eg_record_cond_end(c) == EG_OK && eg_record_end(c, NULL, &gr) == EG_OK && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 1), "device negated IF: lane 2 is 0, the arm runs");
    gb_clear(&g);
    gb_flags(&g, 0, 0, 3);
    ok = ok && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 0), "device negated IF: lane 2 is 3, the arm does not run");
    if (gr) eg_graph_free(c, gr);
    gr = NULL;
    GB_SECTION_END();

    /* a count against a baseline: equal skips the arm, different runs it; negated, the other way */
    for (int negate = 0; negate < 2; ++negate) {
        gb_clear(&g);
        gb_flags(&g, 5, 5, 0);
        ok = eg_record_begin(c) == EG_OK && gb_cond(&g, EG_COND_IF, negate ? EG_COND_NEGATE : 0, 0, g.pf, 0, g.pf, 1, 0) == EG_OK &&
             eg_launch(c, g.a, NULL) == EG_OK && eg_record_cond_end(c) == EG_OK && eg_record_end(c, NULL, &gr) == EG_OK && gb_replay(&g, gr);
        const int equal_runs = ok && gb_holds(g.d1, 1);
        gb_clear(&g);
        gb_flags(&g, 6, 5, 0);
        ok = ok && gb_replay(&g, gr);
        const int differ_runs = ok && gb_holds(g.d1, 1);
        check(ok && equal_runs == negate && differ_runs == !negate,
              negate ? "device negated count/baseline test: equal runs the arm, different skips it" : "device count/baseline test: equal skips the arm, different runs it");
        if (gr) eg_graph_free(c, gr);
        gr = NULL;
        GB_SECTION_END();
    }

    /* IF_ELSE: by the capability, refused by name without it */
    gb_clear(&g);
    gb_flags(&g, 0, 0, 1);
    if (eg_record_begin(c) == EG_OK && gb_cond(&g, EG_COND_IF_ELSE, 0, 0, g.pf, 2, NULL, 0, 0) == EG_NOT_WRITTEN && reason_has(c, "conditional arms")) {
        check(eg_record_end(c, NULL, &gr) == EG_OK && eg_graph_free(c, gr) == EG_OK, "device IF_ELSE refused NOT_WRITTEN by name: the driver has no conditional arms (skipped)");
        gr = NULL;
    } else {
        ok = eg_launch(c, g.a, NULL) == EG_OK && eg_record_cond_arm(c) == EG_OK && eg_launch(c, g.b, NULL) == EG_OK && eg_record_cond_end(c) == EG_OK &&
             eg_record_end(c, NULL, &gr) == EG_OK && gb_replay(&g, gr);
        check(ok && gb_holds(g.d1, 1) && gb_holds(g.d2, 0), "device IF_ELSE: lane 2 set, the first arm runs and the second does not");
        gb_clear(&g);
        gb_flags(&g, 0, 0, 0);
        ok = ok && gb_replay(&g, gr);
        check(ok && gb_holds(g.d1, 0) && gb_holds(g.d2, 1), "device IF_ELSE: lane 2 rewritten to 0, the second arm runs");
        if (gr) eg_graph_free(c, gr);
        gr = NULL;
    }

    GB_SECTION_END();

    /* WHILE: the test never changes, so only max_iters ends it; a test that is false runs nothing */
    gb_clear(&g);
    gb_flags(&g, 0, 0, 1);
    ok = eg_record_begin(c) == EG_OK && gb_cond(&g, EG_COND_WHILE, 0, 0, g.pf, 2, NULL, 0, 3) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK &&
         eg_record_cond_end(c) == EG_OK && eg_record_end(c, NULL, &gr) == EG_OK && gb_replay(&g, gr) && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 1), "device WHILE max_iters 3 over a test that stays true: it stops at the cap without error and the body ran");
    gb_clear(&g);
    gb_flags(&g, 0, 0, 0);
    ok = ok && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 0), "device WHILE with a false test runs no iteration");
    if (gr) eg_graph_free(c, gr);
    gr = NULL;
    GB_SECTION_END();

    /* refusals of a conditional (nothing is opened by them), then a fork inside an arm, a conditional inside a branch */
    gb_clear(&g);
    gb_flags(&g, 0, 0, 1);
    ok = eg_record_begin(c) == EG_OK &&
         refused(gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 0, NULL, 0, 2), EG_BAD_REQUEST, c, "max_iters is for WHILE only") &&
         refused(gb_cond(&g, EG_COND_SWITCH, EG_COND_NEGATE, 2, g.pf, 0, NULL, 0, 0), EG_BAD_REQUEST, c, "NEGATE on a SWITCH") &&
         refused(gb_cond(&g, EG_COND_SWITCH, 0, 2, g.pf, 0, g.pf, 1, 0), EG_BAD_REQUEST, c, "SWITCH reads its arm from the cell alone") &&
         refused(gb_cond(&g, EG_COND_IF, 0, 0, g.pd, 0, NULL, 0, 0), EG_BAD_REQUEST, c, "int32 or uint32") &&
         refused(gb_cond(&g, EG_COND_IF, 0, 0, g.px, 0, NULL, 0, 0), EG_BAD_REQUEST, c, "int32 or uint32") &&
         refused(gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 4, NULL, 0, 0), EG_BAD_REQUEST, c, "outside the plane") &&
         refused(gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 0, g.pf, -1, 0), EG_BAD_REQUEST, c, "outside the plane") &&
         refused(eg_record_cond_arm(c), EG_BAD_REQUEST, c, "no conditional is open");
    check(ok, "device conditional refusals by name: cap off WHILE, NEGATE or a baseline on SWITCH, a double or oversized cell, an index outside it, an arm with none open");
    ok = ok && gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 2, NULL, 0, 0) == EG_OK && eg_record_fork_begin(c) == EG_OK && refused(eg_record_cond_end(c), EG_BAD_REQUEST, c, "fork is still open") &&
         eg_launch(c, g.a, NULL) == EG_OK && eg_record_fork_branch(c) == EG_OK && eg_launch(c, g.b, NULL) == EG_OK &&
         refused(eg_record_end(c, NULL, &gr), EG_BAD_REQUEST, c, "is still open") && eg_record_fork_end(c) == EG_OK && eg_record_cond_end(c) == EG_OK &&
         eg_record_fork_begin(c) == EG_OK && eg_launch(c, g.d, NULL) == EG_OK && refused(eg_record_end(c, NULL, &gr), EG_BAD_REQUEST, c, "fork is still open") &&
         eg_record_fork_branch(c) == EG_OK && gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 2, NULL, 0, 0) == EG_OK && refused(eg_record_fork_branch(c), EG_BAD_REQUEST, c, "conditional is still open") &&
         eg_launch(c, g.c, NULL) == EG_OK && eg_record_cond_end(c) == EG_OK;
    if (!ok) reason_has(c, "(the nested recording was refused)");
    check(ok, "device nesting: a fork inside a conditional arm and a conditional inside a fork branch recorded; the open ones refuse the end and the wrong close by name");
    if (c != NULL) {    /* the last fork's branch 1 reads p1 that branch 0 writes: a race refused at its end */
        check(ok && refused(eg_record_fork_end(c), EG_BAD_REQUEST, c, "race between node") && reason_has(c, "node ") && reason_has(c, "slot ") &&
                  eg_record_abort(c) == EG_OK,
              "device race: branches touching one plane with no wait are refused at fork_end, naming both nodes and slots; the fork stays open, as before the call");
        /* eg_record_abort from every state of a capture, each followed by a launch outside any recording and a recording that ends */
        check(refused(eg_record_abort(c), EG_BAD_REQUEST, c, "no open recording"), "device eg_record_abort with nothing open: refused BAD_REQUEST by name");
        eg_graph* ag = NULL;
        ok = eg_record_begin(c) == EG_OK && eg_record_fork_begin(c) == EG_OK && refused(eg_record_wait(c, 1, &g.p1), EG_BAD_REQUEST, c, "no earlier sibling") &&
             refused(eg_record_fork_branch(c), EG_BAD_REQUEST, c, "branch is empty") && refused(eg_record_fork_end(c), EG_BAD_REQUEST, c, "branch is empty") &&
             eg_record_abort(c) == EG_OK;
        ok = ok && eg_launch(c, g.a, NULL) == EG_OK && eg_sync(c) == EG_OK && eg_record_begin(c) == EG_OK && eg_launch(c, g.b, NULL) == EG_OK &&
             eg_record_end(c, NULL, &ag) == EG_OK && gb_replay(&g, ag) && eg_graph_free(c, ag) == EG_OK;
        if (!ok) reason_has(c, "(the abort of a fork with a refused wait failed)");
        check(ok, "device eg_record_abort of a fork whose branch holds only a refused wait: OK, then a launch outside a recording and a new recording replay");
        ok = eg_record_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 2, NULL, 0, 0) == EG_OK && eg_launch(c, g.b, NULL) == EG_OK &&
             eg_record_abort(c) == EG_OK && eg_record_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_end(c, NULL, &ag) == EG_OK && gb_replay(&g, ag) &&
             eg_graph_free(c, ag) == EG_OK;
        if (!ok) reason_has(c, "(the abort inside an IF arm failed)");
        check(ok, "device eg_record_abort inside an open IF arm (a capture with a conditional woven in): OK, a new recording replays");
        ok = eg_record_begin(c) == EG_OK && gb_cond(&g, EG_COND_WHILE, 0, 0, g.pf, 2, NULL, 0, 4) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_abort(c) == EG_OK &&
             eg_record_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_end(c, NULL, &ag) == EG_OK && gb_replay(&g, ag) && eg_graph_free(c, ag) == EG_OK;
        if (!ok) reason_has(c, "(the abort inside a WHILE body failed)");
        check(ok, "device eg_record_abort inside an open WHILE body: OK, a new recording replays");
        ok = eg_record_begin(c) == EG_OK && eg_record_abort(c) == EG_OK && eg_record_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_end(c, NULL, &ag) == EG_OK &&
             eg_graph_free(c, ag) == EG_OK;
        check(ok, "device eg_record_abort of a recording of one launch and of an empty one: OK, the context records again");
        /* a refused eg_record_cond_end (an arm with no work) changes nothing: the arm takes its launch and the conditional closes */
        ok = eg_record_begin(c) == EG_OK && gb_cond(&g, EG_COND_IF, 0, 0, g.pf, 2, NULL, 0, 0) == EG_OK && refused(eg_record_cond_end(c), EG_BAD_REQUEST, c, "holds no node") &&
             eg_launch(c, g.a, NULL) == EG_OK && eg_record_cond_end(c) == EG_OK && eg_record_end(c, NULL, &ag) == EG_OK && eg_graph_free(c, ag) == EG_OK;
        if (!ok) reason_has(c, "(the refused empty-arm end failed)");
        check(ok, "device eg_record_cond_end over an empty arm: refused BAD_REQUEST by name with the recording as before; the arm takes a launch and the conditional closes");
        /* refused calls leave the shape as it was: the description equals that of the recording without them */
        eg_graph *ra = NULL, *rb = NULL;
        int n1 = 0, n2 = 0, e1 = 0, e2 = 0;
        char k1[32] = "", k2[32] = "";
        ok = eg_record_begin(c) == EG_OK && eg_record_fork_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_wait(c, 1, &g.p2) == EG_BAD_REQUEST &&
             eg_record_cond_arm(c) == EG_BAD_REQUEST && eg_record_cond_end(c) == EG_BAD_REQUEST && eg_record_fork_branch(c) == EG_OK && eg_record_wait(c, 1, &g.p3) == EG_BAD_REQUEST &&
             eg_launch(c, g.b, NULL) == EG_OK && eg_record_fork_end(c) == EG_OK && eg_record_end(c, NULL, &ra) == EG_OK;
        ok = ok && eg_record_begin(c) == EG_OK && eg_record_fork_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_fork_branch(c) == EG_OK && eg_launch(c, g.b, NULL) == EG_OK &&
             eg_record_fork_end(c) == EG_OK && eg_record_end(c, NULL, &rb) == EG_OK;
        ok = ok && gb_shape(c, ra, &n1, &e1, k1, sizeof(k1)) && gb_shape(c, rb, &n2, &e2, k2, sizeof(k2));
        check(ok && n1 == n2 && e1 == e2 && strcmp(k1, k2) == 0, "device refused wait, arm and conditional-end calls leave the shape as it was: the description equals that of the recording without them");
        if (ra) eg_graph_free(c, ra);
        if (rb) eg_graph_free(c, rb);
    }
    gb_free(&g);

    /* a plain graph whose plan is rebound outside the recording reads the new binding */
    if (!gb_make(&g, dev_lib)) return;
    c = g.ctx;
    gb_clear(&g);
    ok = eg_record_begin(c) == EG_OK && eg_launch(c, g.a, NULL) == EG_OK && eg_record_end(c, NULL, &gr) == EG_OK && gb_replay(&g, gr);
    check(ok && gb_holds(g.d1, 1), "device rebind: the plain graph first writes the plane it was recorded with");
    {
        const int32_t slot0[1] = { 0 };
        eg_plane* moved[1] = { g.p3 };
        gb_clear(&g);
        ok = ok && bind_planes(c, g.a, 1, slot0, moved) == EG_OK && gb_replay(&g, gr);
        const int wrote_new = gb_holds(g.d3, 1), left_old = gb_holds(g.d1, 0);
        if (!ok) reason_has(c, "(the rebound replay was refused)");
        else if (!wrote_new || !left_old) printf("      rebind: new plane written %d, old plane left %d\n", wrote_new, left_old);
        check(ok && wrote_new && left_old, "device rebind: after eg_bind the next replay writes the new plane and leaves the old one (never stale)");
    }
    if (gr) eg_graph_free(c, gr);
    gb_free(&g);
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
    {   /* the device group on a host context of a door built with it */
        int32_t count = 0;
        check(eg_sync(host) == EG_OK && refused(eg_stream_use(host, 0), EG_BAD_REQUEST, host, "host-only context") &&
                  eg_device_count(host, &count) == EG_OK && count >= 1,
              "a host context of a door with the device group: eg_sync OK, eg_stream_use BAD_REQUEST (host-only context), "
              "eg_device_count OK with a count of 1 or more");
    }
    {   /* eg_device_props against the CUDA runtime's own attributes, and the derived figures against their formulas */
        struct eg_device_props p;
        memset(&p, 0, sizeof(p));
        p.struct_size = sizeof(p);
        int32_t count = 0;
        int ok = eg_device_props(host, 0, &p) == EG_OK && eg_device_count(host, &count) == EG_OK;
        int a[11];
        const enum cudaDeviceAttr attrs[11] = { cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
            cudaDevAttrMultiProcessorCount, cudaDevAttrWarpSize, cudaDevAttrMaxRegistersPerMultiprocessor,
            cudaDevAttrMaxRegistersPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
            cudaDevAttrClockRate, cudaDevAttrMemoryClockRate, cudaDevAttrGlobalMemoryBusWidth };
        for (int i = 0; i < 11; ++i) ok = ok && cudaDeviceGetAttribute(&a[i], attrs[i], 0) == cudaSuccess;
        struct cudaDeviceProp rp;
        ok = ok && cudaGetDeviceProperties(&rp, 0) == cudaSuccess;
        ok = ok && p.cc_major == a[0] && p.cc_minor == a[1] && p.sm_count == a[2] && p.warp_size == a[3] && p.regs_per_sm == a[4] &&
             p.regs_per_block == a[5] && p.shared_mem_per_block == a[6] && p.shared_mem_per_sm == a[7] && p.clock_rate_khz == a[8] &&
             p.memory_clock_rate_khz == a[9] && p.memory_bus_width_bits == a[10] && strcmp(p.name, rp.name) == 0;
        const double bytes = 2.0 * a[9] * 1e3 * (a[10] / 8.0);
        ok = ok && p.peak_bytes_per_s == bytes && p.fp64_ratio >= 1.0 && p.peak_flops_dp == p.peak_flops_sp / p.fp64_ratio &&
             p.ridge_flops_per_byte_sp == p.peak_flops_sp / bytes && p.ridge_flops_per_byte_dp == p.peak_flops_dp / bytes;
        if (a[0] == 6 && a[1] == 1) ok = ok && p.fp64_ratio == 32.0;    /* sm_61 runs double precision at 1/32 */
        check(ok, "eg_device_props: every raw field equals the CUDA runtime's attribute (and name); peak bandwidth, the "
                  "double-precision peak and both ridge points equal their formulas over those fields");
        struct eg_device_props small;
        memset(&small, 0, sizeof(small));
        small.struct_size = 8;
        check(refused(eg_device_props(host, 0, &small), EG_BAD_REQUEST, host, "struct_size") &&
                  refused(eg_device_props(host, count, &p), EG_BAD_REQUEST, host, "the driver sees"),
              "eg_device_props: a short struct_size and an ordinal the driver does not see are refused BAD_REQUEST by name");
    }

    /* Units. */
    eg_unit *axpy = NULL, *axpy64 = NULL, *u = NULL, *fat129 = NULL;
    check(load(ctx, dev_lib, "axpy", 0xC1, &axpy) == EG_OK && load(ctx, dev_lib, "axpy64", 0xC2, &axpy64) == EG_OK,
          "eg_unit_load axpy and axpy64 (fatbin image, module and function per unit)");
    check(refused(load(ctx, dev_lib, "nofn", 0xC5, &u), EG_FAULT, ctx, "lacks the function 'missing'"),
          "a device function the image lacks: FAULT naming it");
    check(refused(load(ctx, dev_lib, "noimg", 0xC6, &u), EG_FAULT, ctx, "lacks its device image 'absent__image'"),
          "a device image symbol the library lacks: FAULT naming it");
    /* R4: the driver's own answer to the image, mapped. The images are nvcc 12.9's; this driver may be older. */
    check(refused(load(ctx, dev_lib, "ptx129", 0xE2, &u), EG_BAD_REQUEST, ctx, "newer driver") && reason_has(ctx, "PTX 8.8") &&
              reason_has(ctx, "nvcc 12.9"),
          "a PTX-only image newer than the driver: BAD_REQUEST naming 'newer driver', 'PTX 8.8' and the compiler");
    check(refused(load(ctx, dev_lib, "sass70", 0xE3, &u), EG_BAD_REQUEST, ctx, "no machine code"),
          "a fatbin with sm_70 code only and no PTX on this device: BAD_REQUEST naming 'no machine code'");
    check(load(ctx, dev_lib, "fat129", 0xE1, &fat129) == EG_OK, "eg_unit_load fat129 (nvcc 12.9 fatbin, sm_61 code and compute_61 PTX)");
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
    struct { eg_unit* unit; int64_t parts; const char* what; } runs[4] = {
        { axpy, 0, "axpy, DEVICE whole, eagle's block" },
        { axpy, 2, "axpy, DEVICE (0,300),(300,700)" },
        { fat129, 0, "fat129 (nvcc 12.9 fatbin), DEVICE whole" },
        { axpy64, 0, "axpy64, DEVICE whole, the meta's block 64" },
    };
    for (int r = 0; r < 4; ++r) {
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
        if (r < 3) eg_plan_free(ctx, p);
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
    copy_across(ctx);
    struct eg_reduce_desc rd;
    memset(&rd, 0, sizeof(rd));
    rd.struct_size = sizeof(rd);
    rd.in = py;
    rd.out = hy;
    check(refused(eg_reduce(ctx, &rd), EG_BAD_REQUEST, ctx, "both sides"), "eg_reduce of a device plane into a host plane: BAD_REQUEST (one side per operation)");
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
    check(eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, NULL, &g) == EG_OK &&
              eg_graph_free(ctx, g) == EG_OK,
          "recording a DEVICE launch: captured, eg_record_end and eg_graph_free OK");
    check(eg_record_begin(ctx) == EG_OK && refused(bind(ctx, p, py, px, pv, a), EG_BAD_REQUEST, ctx, "eg_bind while recording") &&
              eg_record_end(ctx, NULL, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK,
          "eg_bind inside a DEVICE recording is refused BAD_REQUEST by name; the recording still ends");
    free_while_recording(ctx, dev_lib, p);

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
    set_count_device(ctx, dev_lib);
    lookup_device(ctx, dev_lib);
    figures_device(ctx, dev_lib);
    host_steps_in_graph(ctx, dev_lib, host_lib);
    host_nodes_device(ctx, dev_lib);
    cell_rows_device(ctx, dev_lib);
    reset_device(ctx, dev_lib);
    step_graph_device(ctx, dev_lib);
    check_segmented_device(ctx, dev_lib);
    check_auto_device(ctx, dev_lib);
    fast_entries(ctx, dev_lib);
    graph_builder_device(dev_lib);

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

    alloc_planes(ctx, host, axpy, x, v, want, a);
    check_close_with_device_export();

    fft_values(ctx, 1);
    fft_adjoint(ctx, 1);
    fft_in_place(ctx, 1);
    fft_recorded(ctx, 1);
    fft_refusals(ctx, 1);
    fft_cross(ctx, host);

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
