/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The door's second proof on a unit the old stack built: mat_apply (y = m @ v), one generated unit with a
 * rank-1 MUTABLE y, a rank-1 VEC_IN v and a rank-2 MAT_IN m. The door loads it by name + digest, binds the
 * inputs the old path ran on, launches under HOST_TEAM and HOST_SERIAL and compares y BITWISE with the old
 * path's output. The old path binds m as the flat (9, n) plane; the door binds the same memory as (3, 3, n).
 *
 * Usage: pair2_test <unit library> <oracle dir> <unit name> <digest hex>
 *   <oracle dir> holds m_in.f64 (9 x 64), v_in.f64 (3 x 64) and y_out.f64 (3 x 64).
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
    int64_t shape[3];
};

static void tensor_init(struct tensor* t, double* data, int ndim, int64_t d0, int64_t d1, int64_t d2) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = d0;
    t->shape[1] = d1;
    t->shape[2] = d2;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCPU;
    t->managed.dl_tensor.ndim = ndim;
    t->managed.dl_tensor.dtype.code = kDLFloat;
    t->managed.dl_tensor.dtype.bits = 64;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
}

static int read_file(const char* dir, const char* file, double* out, size_t count) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    const size_t got = fread(out, sizeof(double), count, f);
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

/* Plan, bind y (slot 0), v (slot 1) and m (slot 2), launch; the first differing element is printed. */
static void run_case(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t n_parts, const char* name,
                     const double* m, const double* v, const double* want) {
    static const int64_t base[2] = { 0, N / 2 }, count[2] = { N / 2, N / 2 };
    double mm[9 * N], vv[3 * N], y[3 * N];
    memcpy(mm, m, sizeof(mm));
    memcpy(vv, v, sizeof(vv));
    for (int i = 0; i < 3 * N; ++i) y[i] = -1.0;
    struct tensor ty, tv, tm;
    tensor_init(&ty, y, 2, 3, N, 0);
    tensor_init(&tv, vv, 2, 3, N, 0);
    tensor_init(&tm, mm, 3, 3, 3, N);
    eg_plane *py = NULL, *pv = NULL, *pm = NULL;
    eg_plan* plan = NULL;
    eg_status s = eg_plane_wrap(ctx, &ty.managed, &py);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &tv.managed, &pv);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &tm.managed, &pm);
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
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* const planes[3] = { py, pv, pm };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 3;
    b.plane_slots = slots;
    b.planes = planes;
    if (s == EG_OK) s = eg_bind(ctx, plan, &b);
    if (s == EG_OK) s = eg_launch(ctx, plan, NULL);
    if (s != EG_OK) print_reason(ctx);
    eg_plan_free(ctx, plan);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, pv);
    eg_plane_free(ctx, pm);
    int first = -1;
    for (int i = 0; i < 3 * N && first < 0; ++i)
        if (memcmp(&y[i], &want[i], sizeof(double)) != 0) first = i;
    if (first >= 0)
        printf("      first difference at row %d sample %d: door %.17g, old path %.17g\n", first / N, first % N, y[first],
               want[first]);
    char what[160];
    snprintf(what, sizeof(what), "mat_apply, %s: launch OK and y bitwise == the old path", name);
    check(s == EG_OK && first < 0, what);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <unit library> <oracle dir> <unit name> <digest hex>\n", argv[0]);
        return 2;
    }
    static double m[9 * N], v[3 * N], want[3 * N];
    check(read_file(argv[2], "m_in.f64", m, 9 * N) && read_file(argv[2], "v_in.f64", v, 3 * N) &&
              read_file(argv[2], "y_out.f64", want, 3 * N),
          "old path inputs and output read (m 9 x 64, v 3 x 64, y 3 x 64 float64)");
    int plain = 1; /* the old output is the plain loop over the same inputs, to a few ulp */
    for (int i = 0; i < N && plain; ++i)
        for (int r = 0; r < 3; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 3; ++c) acc += m[(r * 3 + c) * N + i] * v[c * N + i];
            const double diff = acc - want[r * N + i];
            if (diff > 1e-9 * (acc < 0 ? -acc : acc) + 1e-12 || -diff > 1e-9 * (acc < 0 ? -acc : acc) + 1e-12) plain = 0;
        }
    check(plain, "old path output == m @ v (plain loop, to 1e-9)");

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
    run_case(ctx, unit, EG_STRUCTURE_HOST_TEAM, 0, "HOST_TEAM whole", m, v, want);
    run_case(ctx, unit, EG_STRUCTURE_HOST_TEAM, 2, "HOST_TEAM (0,32),(32,32)", m, v, want);
    run_case(ctx, unit, EG_STRUCTURE_HOST_SERIAL, 0, "HOST_SERIAL", m, v, want);
    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
