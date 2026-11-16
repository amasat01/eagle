/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The door's fourth proof on a unit the old stack built: _flat (y = x * 3 + lane.item) emitted as a caller-built index
 * set: slots y (MUTABLE), x (PER_SAMPLE), the index count (uint32 LOOKUP) and the index map (int32 LOOKUP). The door
 * loads the unit by name + digest, binds the old path's inputs, launches under HOST_TEAM and HOST_SERIAL and compares
 * all 40 elements of y BITWISE with the old path's output (the listed lanes written, every other element still 0).
 *
 * Usage: pair4_test <unit library> <oracle dir> <unit name> <digest hex>
 *   <oracle dir> holds x_in.f64 (40 x float64), map_in.i32 (5 x int32) and y_out.f64 (40 x float64).
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 40
#define LIVE 5

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[1];
};

static void tensor_init(struct tensor* t, void* data, int64_t n, uint8_t code, uint8_t bits) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = n;
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

/* Plan, bind y, x, the count and the map (slots 0 to 3), launch; the first differing element is printed. */
static void run_case(eg_ctx* ctx, eg_unit* unit, eg_structure structure, const char* name, const double* x_in,
                     const int32_t* map_in, const double* want) {
    double y[N], x[N];
    int32_t map[LIVE];
    uint32_t count = LIVE;
    memcpy(x, x_in, sizeof(x));
    memcpy(map, map_in, sizeof(map));
    for (int i = 0; i < N; ++i) y[i] = 0.0;
    struct tensor ty, tx, tc, tm;
    tensor_init(&ty, y, N, kDLFloat, 64);
    tensor_init(&tx, x, N, kDLFloat, 64);
    tensor_init(&tc, &count, 1, kDLUInt, 32);
    tensor_init(&tm, map, LIVE, kDLInt, 32);
    eg_plane *py = NULL, *px = NULL, *pc = NULL, *pm = NULL;
    eg_plan* plan = NULL;
    eg_status s = eg_plane_wrap(ctx, &ty.managed, &py);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &tx.managed, &px);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &tc.managed, &pc);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &tm.managed, &pm);
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = structure;
    d.n_samples = N;
    if (s == EG_OK) s = eg_plan_new(ctx, &d, &plan);
    const int32_t slots[4] = { 0, 1, 2, 3 };
    eg_plane* const planes[4] = { py, px, pc, pm };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 4;
    b.plane_slots = slots;
    b.planes = planes;
    if (s == EG_OK) s = eg_bind(ctx, plan, &b);
    if (s == EG_OK) s = eg_launch(ctx, plan, NULL);
    if (s != EG_OK) print_reason(ctx);
    eg_plan_free(ctx, plan);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, pc);
    eg_plane_free(ctx, pm);
    int first = -1;
    for (int i = 0; i < N && first < 0; ++i)
        if (memcmp(&y[i], &want[i], sizeof(double)) != 0) first = i;
    if (first >= 0)
        printf("      first difference at sample %d: door %.17g, old path %.17g\n", first, y[first], want[first]);
    char what[160];
    snprintf(what, sizeof(what), "_flat over an index set, %s: launch OK and y bitwise == the old path", name);
    check(s == EG_OK && first < 0, what);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <unit library> <oracle dir> <unit name> <digest hex>\n", argv[0]);
        return 2;
    }
    static double x[N], want[N];
    static int32_t map[LIVE];
    check(read_file(argv[2], "x_in.f64", x, sizeof(double), N) && read_file(argv[2], "map_in.i32", map, sizeof(int32_t), LIVE) &&
              read_file(argv[2], "y_out.f64", want, sizeof(double), N),
          "old path inputs and output read (x 40 x float64, map 5 x int32, y 40 x float64)");
    int listed = 0, zero = 0;
    for (int i = 0; i < N; ++i) zero += want[i] == 0.0;
    for (int t = 0; t < LIVE; ++t) listed += want[map[t]] != 0.0;
    check(listed == LIVE && zero == N - LIVE, "old path wrote exactly the 5 listed lanes");

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
    run_case(ctx, unit, EG_STRUCTURE_HOST_TEAM, "HOST_TEAM", x, map, want);
    run_case(ctx, unit, EG_STRUCTURE_HOST_SERIAL, "HOST_SERIAL", x, map, want);
    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
