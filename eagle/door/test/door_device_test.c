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

#include <dlfcn.h>
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
    eg_unit_free(ctx, plain);
    eg_unit_free(ctx, active);
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
    check(refused(eg_copy(ctx, py, hy), EG_BAD_REQUEST, ctx, "both sides"), "eg_copy of a device plane onto a host plane: BAD_REQUEST (one side per operation)");
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
    figures_device(ctx, dev_lib);
    reset_device(ctx, dev_lib);

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
