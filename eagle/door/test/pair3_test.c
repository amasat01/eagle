/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The door's third proof on a unit the old stack built: _wide_math (y = real(idx * 3e9 // 3e9)), a MUTABLE
 * float64 y and an int32 PER_SAMPLE index plane (meta dtype INT32; the kernel text reads std::int32_t). The
 * index values reach +-1e9, so a plane bound at another width reads other bits. The door loads the unit by
 * name + digest, binds the old path's index plane, launches under HOST_TEAM and HOST_SERIAL and compares y
 * BITWISE with the old path's output.
 *
 * Usage: pair3_test <unit library> <oracle dir> <unit name> <digest hex>
 *   <oracle dir> holds idx_in.i32 (64 x int32) and y_out.f64 (64 x float64).
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
};

static void tensor_init(struct tensor* t, void* data, uint8_t code, uint8_t bits) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = N;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCPU;
    t->managed.dl_tensor.ndim = 1;
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
}

static int read_file(const char* dir, const char* file, void* out, size_t size, size_t count) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    const size_t got = fread(out, size, count, f);
    fclose(f);
    return got == count;
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

static void print_reason(eg_ctx* ctx) {
    char buf[1024];
    size_t len = 0;
    if (eg_error(ctx, buf, sizeof(buf), &len) == EG_OK && len > 0) printf("      reason: %s\n", buf);
}

/* Plan, bind y (slot 0) and idx (slot 1), launch; the first differing element is printed. */
static void run_case(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t n_parts, const char* name,
                     const int32_t* idx_in, const double* want) {
    static const int64_t base[2] = { 0, N / 2 }, count[2] = { N / 2, N / 2 };
    int32_t idx[N];
    double y[N];
    memcpy(idx, idx_in, sizeof(idx));
    for (int i = 0; i < N; ++i) y[i] = -1.0;
    struct tensor ty, ti;
    tensor_init(&ty, y, kDLFloat, 64);
    tensor_init(&ti, idx, kDLInt, 32);
    eg_plane *py = NULL, *pi = NULL;
    eg_plan* plan = NULL;
    eg_status s = eg_plane_wrap(ctx, &ty.managed, &py);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &ti.managed, &pi);
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
    const int32_t slots[2] = { 0, 1 };
    eg_plane* const planes[2] = { py, pi };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 2;
    b.plane_slots = slots;
    b.planes = planes;
    if (s == EG_OK) s = eg_bind(ctx, plan, &b);
    if (s == EG_OK) s = eg_launch(ctx, plan, NULL);
    if (s != EG_OK) print_reason(ctx);
    eg_plan_free(ctx, plan);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, pi);
    int first = -1;
    for (int i = 0; i < N && first < 0; ++i)
        if (memcmp(&y[i], &want[i], sizeof(double)) != 0) first = i;
    if (first >= 0)
        printf("      first difference at sample %d: idx %d, door %.17g, old path %.17g\n", first, idx_in[first],
               y[first], want[first]);
    char what[160];
    snprintf(what, sizeof(what), "_wide_math, %s: launch OK and y bitwise == the old path", name);
    check(s == EG_OK && first < 0, what);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <unit library> <oracle dir> <unit name> <digest hex>\n", argv[0]);
        return 2;
    }
    static int32_t idx[N];
    static double want[N];
    check(read_file(argv[2], "idx_in.i32", idx, sizeof(int32_t), N) && read_file(argv[2], "y_out.f64", want, sizeof(double), N),
          "old path input and output read (idx 64 x int32, y 64 x float64)");
    int exact = 1, wide = 0;
    for (int i = 0; i < N; ++i) {
        exact &= want[i] == (double)idx[i];
        wide |= idx[i] > 100000000 || idx[i] < -100000000;
    }
    check(exact && wide, "old path y == real(idx) (the recorded kernel's identity) and idx reaches +-1e8 and beyond");

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
    const eg_status ls = parse_digest(argv[4], ref.digest) ? eg_unit_load(ctx, &d, &unit) : EG_BAD_REQUEST;
    check(ls == EG_OK && unit != NULL, "eg_unit_load of the generated meta (name + digest)");
    if (ls != EG_OK) {
        print_reason(ctx);
        eg_ctx_free(ctx);
        return 1;
    }
    eg_host_threads(ctx, 3);
    run_case(ctx, unit, EG_STRUCTURE_HOST_TEAM, 0, "HOST_TEAM whole", idx, want);
    run_case(ctx, unit, EG_STRUCTURE_HOST_TEAM, 2, "HOST_TEAM (0,32),(32,32)", idx, want);
    run_case(ctx, unit, EG_STRUCTURE_HOST_SERIAL, 0, "HOST_SERIAL", idx, want);
    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
