/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The door's sixth proof on a unit the old stack built: mv (out = sum(M @ v)) with M a GATHER matrix, compact over the lane
 * axes a and c of Grid[a, b:4, c:6]: the plane is (8, 8, 18), 18 = 3 * 6 columns, not the 72 lanes. The unit's meta is the
 * recorded one with the gather pattern the sidecar records ({true, false, true}) written in by hand (the replay tool's
 * request does not carry it). y is compared BITWISE with the old path's output on HOST_TEAM and HOST_SERIAL.
 *
 * Usage: pair6_test <unit library> <oracle dir> <unit name> <digest hex>
 *   <oracle dir> holds v_in.f64 (8 x 72), m_in.f64 (8 x 8 x 3 x 1 x 6) and y_out.f64 (72).
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 72
#define YLEN N
#define ALEN (8 * N)
#define BLEN (64 * 18)

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[3];
};

static void tensor_init(struct tensor* t, void* data, int ndim, int64_t d0, int64_t d1, int64_t d2) {
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

static int read_file(const char* dir, const char* file, void* out, size_t count) {
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

/* Plan, bind the three planes (slots 0 to 2), launch; the first differing element is printed. */
static void run_case(eg_ctx* ctx, eg_unit* unit, eg_structure structure, const char* name, const double* in0,
                     const double* in1, const double* want) {
    static double y[YLEN], a[ALEN], b[BLEN];
    memcpy(a, in0, sizeof(a));
    memcpy(b, in1, sizeof(b));
    for (int i = 0; i < YLEN; ++i) y[i] = -1.0;
    struct tensor ty, ta, tb;
    tensor_init(&ty, y, 1, N, 0, 0);
    tensor_init(&ta, a, 2, 8, N, 0);
    tensor_init(&tb, b, 3, 8, 8, 18);
    eg_plane *py = NULL, *pa = NULL, *pb = NULL;
    eg_plan* plan = NULL;
    eg_status s = eg_plane_wrap(ctx, &ty.managed, &py);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &ta.managed, &pa);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &tb.managed, &pb);
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = structure;
    d.n_samples = N;
    if (s == EG_OK) s = eg_plan_new(ctx, &d, &plan);
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* const planes[3] = { py, pa, pb };
    struct eg_bind_desc bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 3;
    bd.plane_slots = slots;
    bd.planes = planes;
    if (s == EG_OK) s = eg_bind(ctx, plan, &bd);
    if (s == EG_OK) s = eg_launch(ctx, plan, NULL);
    if (s != EG_OK) print_reason(ctx);
    eg_plan_free(ctx, plan);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, pa);
    eg_plane_free(ctx, pb);
    int first = -1;
    for (int i = 0; i < YLEN && first < 0; ++i)
        if (memcmp(&y[i], &want[i], sizeof(double)) != 0) first = i;
    if (first >= 0) printf("      first difference at element %d: door %.17g, old path %.17g\n", first, y[first], want[first]);
    char what[160];
    snprintf(what, sizeof(what), "mv with a gathered M, %s: launch OK and y bitwise == the old path", name);
    check(s == EG_OK && first < 0, what);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <unit library> <oracle dir> <unit name> <digest hex>\n", argv[0]);
        return 2;
    }
    static double in0[ALEN], in1[BLEN], want[YLEN];
    check(read_file(argv[2], "v_in.f64", in0, ALEN) && read_file(argv[2], "m_in.f64", in1, BLEN) &&
              read_file(argv[2], "y_out.f64", want, YLEN),
          "old path inputs and output read");
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
    run_case(ctx, unit, EG_STRUCTURE_HOST_TEAM, "HOST_TEAM", in0, in1, want);
    run_case(ctx, unit, EG_STRUCTURE_HOST_SERIAL, "HOST_SERIAL", in0, in1, want);
    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
