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
        const eg_status ended = eg_record_end(ctx, &g);
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
             eg_record_end(ctx, &g) == EG_OK;
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
         eg_record_end(ctx, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK;
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
    if (r) eg_runner_free(ctx, r);
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
    static struct dset ref, run, fus;
    struct eg_run_report rep, rep2;
    eg_runner *r = NULL, *rf = NULL;
    check(dset_new(ctx, k0, 0, 0, &ref) && dset_new(ctx, k0, 0, 0, &run) && dset_new(ctx, fused, 0, 0, &fus), "the static group's plans and the fused unit's plan on the device");
    const int64_t want = dset_launch_loop(ctx, &ref, 1000);
    check(want == 5 && seg_done(ref.mask) == SGN, "the device group launched by hand until done: 5 launches");
    check(dset_run(ctx, &run, 1000, &r, &rep) == EG_OK && dset_same(&run, &ref, 0) && rep.launches == want && rep.steps == want && rep.finished == SGN &&
              rep.mapped == want * SGN && rep.mode == EG_MODE_BAND && rep.flags == (EG_REPORT_DONE | EG_REPORT_EXACT_STEPS),
          "a DEVICE runner over a SEGMENTED group (host-stepped): planes and report equal the plan launched step by step");
    check(dset_run(ctx, &fus, 1000, &rf, &rep2) == EG_OK && dset_same(&fus, &ref, 0) && rep2.launches == want && rep2.mode == EG_MODE_BAND,
          "the same run of the fused unit (the captured loop) equals it bitwise");
    dset_upload(&run, 0);
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    check(eg_runner_reset(ctx, r) == EG_OK && eg_run(ctx, r, &rep) == EG_OK && (dset_read(ctx, &run), dset_same(&run, &ref, 0)) && rep.launches == want &&
              rep.finished == SGN,
          "eg_runner_reset then a second run of the SEGMENTED runner equals the first, report included");
    eg_runner_free(ctx, r), eg_runner_free(ctx, rf);
    {
        struct eg_runner_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d), d.n_plans = 1, d.plans = &run.plan, d.max_steps = 10;
        eg_runner* rr = NULL;
        d.flags = EG_RUN_REORDER;
        check(refused(eg_runner_new(ctx, &d, &rr), EG_BAD_REQUEST, ctx, "nothing to reorder"), "EG_RUN_REORDER over a DEVICE SEGMENTED group: BAD_REQUEST");
        d.flags = 0, d.every = 4;
        check(refused(eg_runner_new(ctx, &d, &rr), EG_BAD_REQUEST, ctx, "steps per launch"), "every over a DEVICE SEGMENTED group: BAD_REQUEST");
        d.every = 0, d.flags = EG_RUN_RECORD;
        check(eg_runner_new(ctx, &d, &rr) != EG_OK && rr == NULL, "EG_RUN_RECORD over a DEVICE SEGMENTED group is refused");
    }
    dset_free(ctx, &ref), dset_free(ctx, &run), dset_free(ctx, &fus);

    /* the dynamic selector: the runner, the hand-launched plan and the fused unit agree bitwise; the sort schedule is the host test's */
    static struct dset a, b, f;
    check(dset_new(ctx, dk0, 1, 1, &a) && dset_new(ctx, dk0, 1, 1, &b) && dset_new(ctx, dgf, 1, 0, &f), "the dynamic group's plans on the device");
    const int64_t total = dset_launch_loop(ctx, &a, 1000);
    eg_runner *rb = NULL, *rfu = NULL;
    check(dset_run(ctx, &b, 1000, &rb, &rep) == EG_OK && dset_same(&b, &a, 1) && rep.launches == total && rep.finished == SGN,
          "the dynamic DEVICE group under a runner equals the plan launched step by step (planes, positions, launches)");
    check(dset_run(ctx, &f, 1000, &rfu, &rep2) == EG_OK && memcmp(f.k, a.k, sizeof(a.k)) == 0 && rep2.launches == total,
          "the dynamic group equals the fused unit's run bitwise");
    dset_upload(&b, 1);
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    check(eg_runner_reset(ctx, rb) == EG_OK && eg_run(ctx, rb, &rep) == EG_OK && (dset_read(ctx, &b), dset_same(&b, &a, 1)) && rep.launches == total,
          "eg_runner_reset of a dynamic DEVICE group, then a second run equals the first (the sort starts over)");
    eg_runner_free(ctx, rb), eg_runner_free(ctx, rfu);
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
    dset_free(ctx, &a), dset_free(ctx, &b), dset_free(ctx, &f);
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
    ok = eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, &g) == EG_OK;
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

#include "fft_cases.h"

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
                 eg_record_end(ctx, &g) == EG_OK;
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
                  refused(eg_device_count(host, &count), EG_NOT_WRITTEN, host, "not written in this build"),
              "a host context of a door with the device group: eg_sync OK, eg_stream_use BAD_REQUEST (host-only context), "
              "eg_device_count NOT_WRITTEN (not written in either build)");
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
    check(eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, &g) == EG_OK &&
              eg_graph_free(ctx, g) == EG_OK,
          "recording a DEVICE launch: captured, eg_record_end and eg_graph_free OK");
    check(eg_record_begin(ctx) == EG_OK && refused(bind(ctx, p, py, px, pv, a), EG_BAD_REQUEST, ctx, "eg_bind while recording") &&
              eg_record_end(ctx, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK,
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
    reset_device(ctx, dev_lib);
    check_segmented_device(ctx, dev_lib);
    check_auto_device(ctx, dev_lib);
    fast_entries(ctx, dev_lib);

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
