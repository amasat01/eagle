/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The first pair check: one GENERATED unit (its kernel text and its generated
 * meta source, built into one library) run through the door, compared BITWISE
 * with the outputs the old path wrote for the same recorded case.
 *
 * The case: y = x * a + p over 64 lanes, x = 0, 1, ..., 63, p = 1. Case A
 * passes a as the number 2.0 (the old path spreads it over every lane); case B
 * passes a as a plane of 2.0. The unit's slots: 0 y (MUTABLE), 1 a
 * (PER_SAMPLE), 2 x (PER_SAMPLE), 3 p (UNIFORM).
 *
 * Usage: pair_test <unit library> <oracle dir> <unit name> <digest hex>
 *   <oracle dir> holds y_scalar_a.f64 and y_plane_a.f64 (64 float64 each).
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 64

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[1];
    int64_t strides[1];
};

static void tensor_init(struct tensor* t, double* data, int64_t n) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = n;
    t->strides[0] = 1;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCPU;
    t->managed.dl_tensor.ndim = 1;
    t->managed.dl_tensor.dtype.code = kDLFloat;
    t->managed.dl_tensor.dtype.bits = 64;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    t->managed.dl_tensor.strides = t->strides;
}

static int read_oracle(const char* dir, const char* file, double* out) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    const size_t got = fread(out, sizeof(double), N, f);
    fclose(f);
    return got == N;
}

static int parse_digest(const char* hex, uint8_t* out) {
    if (strlen(hex) != 2 * EG_DIGEST_BYTES) return 0;
    for (int i = 0; i < EG_DIGEST_BYTES; ++i) {
        unsigned v = 0;
        if (sscanf(hex + 2 * i, "%2x", &v) != 1) return 0;
        out[i] = (uint8_t)v;
    }
    return 1;
}

/* Plan the unit under one structure, bind y, a, x, p, launch; OK or the first refusal. */
static eg_status run(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t n_parts,
                     double* y, double* a, double* x, double p) {
    static const int64_t base[2] = { 0, N / 2 }, count[2] = { N / 2, N / 2 };
    struct tensor ty, ta, tx;
    tensor_init(&ty, y, N);
    tensor_init(&ta, a, N);
    tensor_init(&tx, x, N);
    eg_plane *py = NULL, *pa = NULL, *px = NULL;
    eg_plan* plan = NULL;
    eg_status s = eg_plane_wrap(ctx, &ty.managed, &py);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &ta.managed, &pa);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &tx.managed, &px);
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = structure;
    d.n_samples = N;
    d.n_partitions = n_parts;
    d.part_base = n_parts ? base : NULL;
    d.part_count = n_parts ? count : NULL;
    if (s == EG_OK) s = eg_plan_new(ctx, &d, &plan);
    const int32_t plane_slots[3] = { 0, 1, 2 };
    eg_plane* const planes[3] = { py, pa, px };
    const int32_t uniform_slots[1] = { 3 };
    const void* const values[1] = { &p };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 3;
    b.n_uniforms = 1;
    b.plane_slots = plane_slots;
    b.planes = planes;
    b.uniform_slots = uniform_slots;
    b.uniform_values = values;
    if (s == EG_OK) s = eg_bind(ctx, plan, &b);
    uint64_t tiles = 0;
    if (s == EG_OK) s = eg_launch(ctx, plan, &tiles);
    eg_plan_free(ctx, plan);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, pa);
    eg_plane_free(ctx, px);
    return s;
}

static void print_reason(eg_ctx* ctx) {
    char buf[1024];
    size_t len = 0;
    if (eg_error(ctx, buf, sizeof(buf), &len) == EG_OK && len > 0) printf("      reason: %s\n", buf);
}

/* One case: HOST_TEAM whole, HOST_TEAM over two partitions and HOST_SERIAL, each bitwise == the old path. */
static void run_case(eg_ctx* ctx, eg_unit* unit, const char* label, const double* want) {
    static const struct { eg_structure s; int64_t parts; const char* name; } shapes[3] = {
        { EG_STRUCTURE_HOST_TEAM, 0, "HOST_TEAM whole" },
        { EG_STRUCTURE_HOST_TEAM, 2, "HOST_TEAM (0,32),(32,32)" },
        { EG_STRUCTURE_HOST_SERIAL, 0, "HOST_SERIAL" },
    };
    for (int k = 0; k < 3; ++k) {
        double x[N], a[N], y[N];
        for (int i = 0; i < N; ++i) x[i] = (double)i, a[i] = 2.0, y[i] = 0.0;
        const eg_status s = run(ctx, unit, shapes[k].s, shapes[k].parts, y, a, x, 1.0);
        char what[160];
        snprintf(what, sizeof(what), "%s, %s: launch OK and y bitwise == the old path", label, shapes[k].name);
        check(s == EG_OK && memcmp(y, want, sizeof(y)) == 0, what);
        if (s != EG_OK) print_reason(ctx);
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <unit library> <oracle dir> <unit name> <digest hex>\n", argv[0]);
        return 2;
    }
    double want_a[N], want_b[N], exact[N];
    check(read_oracle(argv[2], "y_scalar_a.f64", want_a) && read_oracle(argv[2], "y_plane_a.f64", want_b),
          "old path outputs read (2 x 64 float64)");
    for (int i = 0; i < N; ++i) exact[i] = (double)i * 2.0 + 1.0;
    check(memcmp(want_a, exact, sizeof(exact)) == 0 && memcmp(want_b, exact, sizeof(exact)) == 0,
          "old path outputs == x * 2 + 1 (the recorded test's own assertion)");

    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new host context");

    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = argv[3];
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = argv[1];
    d.unit = &ref;
    eg_unit* unit = NULL;
    const int digest_ok = parse_digest(argv[4], ref.digest);
    const eg_status ls = digest_ok ? eg_unit_load(ctx, &d, &unit) : EG_BAD_REQUEST;
    check(ls == EG_OK && unit != NULL, "eg_unit_load of the generated meta (name + digest)");
    if (ls != EG_OK) {
        print_reason(ctx);
        eg_ctx_free(ctx);
        return 1;
    }

    run_case(ctx, unit, "case A (a = 2.0 spread by the caller)", want_a);
    run_case(ctx, unit, "case B (a = a plane of 2.0)", want_b);

    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
