/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The door's pair proofs: one unit the old stack built (its kernel text and its generated meta source, built
 * into one library) is loaded by name + digest, bound to the inputs the old path ran on, launched under
 * HOST_TEAM and HOST_SERIAL (the first three units also over two partitions) and y is compared BITWISE with the
 * old path's output. One table row describes each unit: its planes (slot = position, y first), where each
 * plane's values come from, the oracle outputs and the cases to run.
 *
 *   ss_bcast   y = x * a + p: PER_SAMPLE a and x, UNIFORM p; case A passes a as the number 2.0 spread by the
 *              caller, case B as a plane of 2.0.
 *   mat_apply  y = m @ v: rank-1 MUTABLE y, VEC_IN v and a rank-2 MAT_IN m (the old path binds m as the flat
 *              (9, n) plane; the door binds the same memory as (3, 3, n)).
 *   _wide_math y = real(idx * 3e9 // 3e9): an int32 PER_SAMPLE index plane reaching +-1e9, so a plane bound at
 *              another width reads other bits.
 *   _flat      y = x * 3 + lane.item over a caller-built index set: x, the uint32 index count and the int32
 *              index map (LOOKUP); the listed lanes are written, every other element stays 0.
 *   rotate     y = A @ x with a SHARED parameter plane A (a LOOKUP slot, one 2 x 2 matrix for every lane).
 *   mv         out = sum(M @ v) with M a GATHER matrix compact over the lane axes a and c of Grid[a, b:4, c:6]
 *              (the unit's meta carries the gather pattern by hand).
 *   ksum_c     out[key] = a compensated sum of x over 8192 samples: a keyed accumulate; eagle makes the companion
 *              (slot 1, zeroed) and folds it into out after the launch.
 *   fixed_count acc[a] += th[b] over Grid[a:4, b:8]: the same companion over a static plane axis, th a LOOKUP table.
 *
 * Usage: pair_test <unit library> <oracle dir> <unit name> <digest hex>
 *   <oracle dir> holds the files named in the unit's row (float64 and int32 raw arrays).
 *
 * The device column (built as pair_device_test with EG_PAIR_DEVICE, run through the GPU pool): the unit's cuda
 * library on a device context, its planes copied to device memory, launched under DEVICE whole and over two
 * halves, y compared BITWISE with the old path's device output. Rows with ``device`` set only (mv, ksum_c,
 * fixed_count: the last two make their companion on the device and fold it with the unit's device fold function).
 */
#define _POSIX_C_SOURCE 200809L

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef EG_PAIR_DEVICE
#include <cuda_runtime_api.h>
#endif

#define MAXP 4
#define MAXV 16384 /* elements per plane buffer */

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[3];
    int64_t strides[3];
};

static void tensor_init(struct tensor* t, void* data, int ndim, const int64_t* dims, uint8_t code, uint8_t bits,
                        int unit_stride) {
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < ndim; ++i) t->shape[i] = dims[i];
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCPU;
    t->managed.dl_tensor.ndim = ndim;
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
    if (unit_stride) {
        t->strides[0] = 1;
        t->managed.dl_tensor.strides = t->strides;
    }
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

/* ---- the unit table ---- */

struct plane_spec {
    uint8_t code, bits;
    int ndim;
    int64_t dims[3];
    const char* file; /* the old path's input; NULL = filled by gen */
    void (*gen)(void* buf);
};

struct shape_spec {
    eg_structure s;
    int64_t parts;
    const char* name;
};

struct case_spec {
    const char* label; /* first half of the check text */
    int want;          /* index into the unit's oracle outputs */
};

/* The oracle inputs and outputs, as read. */
struct data {
    double in[MAXP][MAXV];
    double want[2][MAXV];
};

struct unit_spec {
    const char* name;
    int64_t n;                  /* samples */
    int nplanes;                /* planes the caller binds, in slot order, skipping the slots of ``made``; plane 0 is y (float64, yinit) */
    unsigned made;              /* bit s set: slot s is eagle's own (a companion, compensation rows) and is never bound */
    struct plane_spec planes[MAXP];
    double yinit;
    int has_uniform;            /* one float64 UNIFORM at slot nplanes, value 1.0 */
    int unit_stride;            /* the planes carry an explicit stride of 1 (1-d only) */
    int threads;                /* eg_host_threads, 0 = leave */
    const char* want_file[2];
    int nwant;
    const char* read_msg;
    int (*sanity)(const struct data*, const char** msg);
    int nshapes;
    const struct shape_spec* shapes;
    int ncases;
    struct case_spec cases[2];
    int device;                 /* the row has a device column */
};

static const struct shape_spec three_shapes[3] = {
    { EG_STRUCTURE_HOST_TEAM, 0, "HOST_TEAM whole" },
    { EG_STRUCTURE_HOST_TEAM, 2, "HOST_TEAM (0,32),(32,32)" },
    { EG_STRUCTURE_HOST_SERIAL, 0, "HOST_SERIAL" },
};
static const struct shape_spec two_shapes[2] = {
    { EG_STRUCTURE_HOST_TEAM, 0, "HOST_TEAM" },
    { EG_STRUCTURE_HOST_SERIAL, 0, "HOST_SERIAL" },
};
#ifdef EG_PAIR_DEVICE
static const struct shape_spec device_shapes[2] = {
    { EG_STRUCTURE_DEVICE, 0, "DEVICE whole" },
    { EG_STRUCTURE_DEVICE, 2, "DEVICE over two halves" },
};
#endif

static void gen_two(void* buf) {
    for (int i = 0; i < 64; ++i) ((double*)buf)[i] = 2.0;
}
static void gen_iota(void* buf) {
    for (int i = 0; i < 64; ++i) ((double*)buf)[i] = (double)i;
}
static void gen_count5(void* buf) { *(uint32_t*)buf = 5; }

static int sanity_ss_bcast(const struct data* d, const char** msg) {
    double exact[64];
    for (int i = 0; i < 64; ++i) exact[i] = (double)i * 2.0 + 1.0;
    *msg = "old path outputs == x * 2 + 1 (the recorded test's own assertion)";
    return memcmp(d->want[0], exact, sizeof(exact)) == 0 && memcmp(d->want[1], exact, sizeof(exact)) == 0;
}

static int sanity_mat_apply(const struct data* d, const char** msg) {
    const double* v = d->in[1];
    const double* m = d->in[2];
    const double* want = d->want[0];
    int plain = 1; /* the old output is the plain loop over the same inputs, to a few ulp */
    for (int i = 0; i < 64 && plain; ++i)
        for (int r = 0; r < 3; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 3; ++c) acc += m[(r * 3 + c) * 64 + i] * v[c * 64 + i];
            const double diff = acc - want[r * 64 + i];
            const double tol = 1e-9 * (acc < 0 ? -acc : acc) + 1e-12;
            if (diff > tol || -diff > tol) plain = 0;
        }
    *msg = "old path output == m @ v (plain loop, to 1e-9)";
    return plain;
}

static int sanity_wide_math(const struct data* d, const char** msg) {
    const int32_t* idx = (const int32_t*)d->in[1];
    int exact = 1, wide = 0;
    for (int i = 0; i < 64; ++i) {
        exact &= d->want[0][i] == (double)idx[i];
        wide |= idx[i] > 100000000 || idx[i] < -100000000;
    }
    *msg = "old path y == real(idx) (the recorded kernel's identity) and idx reaches +-1e8 and beyond";
    return exact && wide;
}

static int sanity_flat(const struct data* d, const char** msg) {
    const int32_t* map = (const int32_t*)d->in[3];
    int listed = 0, zero = 0;
    for (int i = 0; i < 40; ++i) zero += d->want[0][i] == 0.0;
    for (int t = 0; t < 5; ++t) listed += d->want[0][map[t]] != 0.0;
    *msg = "old path wrote exactly the 5 listed lanes";
    return listed == 5 && zero == 40 - 5;
}

#define F64 kDLFloat, 64
#define I32 kDLInt, 32

static int sanity_ksum(const struct data* d, const char** msg) {
    int ok = 1;
    for (int k = 0; k < 2; ++k) {
        double e = 0.0;
        for (int i = 0; i < 8192; ++i) e += d->in[1][k * 8192 + i];
        ok &= d->want[0][k] - e <= 1e-9 && e - d->want[0][k] <= 1e-9;
    }
    *msg = "old path out[key] == the sum of x over the key's 8192 samples (plain loop, to 1e-9)";
    return ok;
}

static int sanity_fixed_count(const struct data* d, const char** msg) {
    double e = 0.0;
    for (int i = 0; i < 8; ++i) e += d->in[1][i];
    int ok = 1;
    for (int a = 0; a < 4; ++a) ok &= d->want[0][a] - e <= 1e-12 && e - d->want[0][a] <= 1e-12;
    *msg = "old path acc[a] == the sum of th for each of the 4 values of a (plain loop, to 1e-12)";
    return ok;
}

static const struct unit_spec units[] = {
    { .name = "ss_bcast", .n = 64, .nplanes = 3,
      .planes = { { F64, 1, { 64 }, NULL, NULL },
                  { F64, 1, { 64 }, NULL, gen_two },
                  { F64, 1, { 64 }, NULL, gen_iota } },
      .yinit = 0.0, .has_uniform = 1, .unit_stride = 1,
      .want_file = { "y_scalar_a.f64", "y_plane_a.f64" }, .nwant = 2,
      .read_msg = "old path outputs read (2 x 64 float64)", .sanity = sanity_ss_bcast,
      .nshapes = 3, .shapes = three_shapes,
      .ncases = 2, .cases = { { "case A (a = 2.0 spread by the caller)", 0 }, { "case B (a = a plane of 2.0)", 1 } } },
    { .name = "mat_apply", .n = 64, .nplanes = 3,
      .planes = { { F64, 2, { 3, 64 }, NULL, NULL },
                  { F64, 2, { 3, 64 }, "v_in.f64", NULL },
                  { F64, 3, { 3, 3, 64 }, "m_in.f64", NULL } },
      .yinit = -1.0, .threads = 3,
      .want_file = { "y_out.f64" }, .nwant = 1,
      .read_msg = "old path inputs and output read (m 9 x 64, v 3 x 64, y 3 x 64 float64)",
      .sanity = sanity_mat_apply,
      .nshapes = 3, .shapes = three_shapes, .ncases = 1, .cases = { { "mat_apply", 0 } } },
    { .name = "_wide_math", .n = 64, .nplanes = 2,
      .planes = { { F64, 1, { 64 }, NULL, NULL }, { I32, 1, { 64 }, "idx_in.i32", NULL } },
      .yinit = -1.0, .threads = 3,
      .want_file = { "y_out.f64" }, .nwant = 1,
      .read_msg = "old path input and output read (idx 64 x int32, y 64 x float64)",
      .sanity = sanity_wide_math,
      .nshapes = 3, .shapes = three_shapes, .ncases = 1, .cases = { { "_wide_math", 0 } } },
    { .name = "_flat", .n = 40, .nplanes = 4,
      .planes = { { F64, 1, { 40 }, NULL, NULL },
                  { F64, 1, { 40 }, "x_in.f64", NULL },
                  { kDLUInt, 32, 1, { 1 }, NULL, gen_count5 },
                  { I32, 1, { 5 }, "map_in.i32", NULL } },
      .yinit = 0.0, .threads = 3,
      .want_file = { "y_out.f64" }, .nwant = 1,
      .read_msg = "old path inputs and output read (x 40 x float64, map 5 x int32, y 40 x float64)",
      .sanity = sanity_flat,
      .nshapes = 2, .shapes = two_shapes, .ncases = 1, .cases = { { "_flat over an index set", 0 } } },
    { .name = "rotate", .n = 64, .nplanes = 3,
      .planes = { { F64, 2, { 2, 64 }, NULL, NULL },
                  { F64, 2, { 2, 64 }, "x_in.f64", NULL },
                  { F64, 2, { 2, 2 }, "a_in.f64", NULL } },
      .yinit = -1.0, .threads = 3,
      .want_file = { "y_out.f64" }, .nwant = 1, .read_msg = "old path inputs and output read",
      .nshapes = 2, .shapes = two_shapes, .ncases = 1, .cases = { { "rotate with a shared A", 0 } } },
    { .name = "mv", .n = 72, .nplanes = 3,
      .planes = { { F64, 1, { 72 }, NULL, NULL },
                  { F64, 2, { 8, 72 }, "v_in.f64", NULL },
                  { F64, 3, { 8, 8, 18 }, "m_in.f64", NULL } },
      .yinit = -1.0, .threads = 3,
      .want_file = { "y_out.f64" }, .nwant = 1, .read_msg = "old path inputs and output read",
      .nshapes = 2, .shapes = two_shapes, .ncases = 1, .cases = { { "mv with a gathered M", 0 } }, .device = 1 },
    { .name = "ksum_c", .n = 16384, .nplanes = 2, .made = 1u << 1,
      .planes = { { F64, 1, { 2 }, NULL, NULL }, { F64, 1, { 16384 }, "x_in.f64", NULL } },
      .yinit = 0.0, .threads = 3,
      .want_file = { "y_out.f64" }, .nwant = 1, .read_msg = "old path input and output read (x 2 x 8192, out 2 float64)",
      .sanity = sanity_ksum,
      .nshapes = 2, .shapes = two_shapes, .ncases = 1, .cases = { { "ksum_c: a compensated keyed sum", 0 } }, .device = 1 },
    { .name = "fixed_count", .n = 32, .nplanes = 2, .made = 1u << 1,
      .planes = { { F64, 1, { 4 }, NULL, NULL }, { F64, 1, { 8 }, "th_in.f64", NULL } },
      .yinit = 0.0, .threads = 3,
      .want_file = { "y_out.f64" }, .nwant = 1, .read_msg = "old path input and output read (th 8, acc 4 float64)",
      .sanity = sanity_fixed_count,
      .nshapes = 2, .shapes = two_shapes, .ncases = 1, .cases = { { "fixed_count: a companion over a static axis", 0 } }, .device = 1 },
};

static int64_t elems(const struct plane_spec* p) {
    int64_t n = 1;
    for (int i = 0; i < p->ndim; ++i) n *= p->dims[i];
    return n;
}

/* Plan, bind every plane (and the uniform), launch; the first differing element of y is printed. */
static void run_case(eg_ctx* ctx, eg_unit* unit, const struct unit_spec* u, const struct shape_spec* sh,
                     const char* label, const struct data* data, const double* want) {
    static double buf[MAXP][MAXV];
    const int64_t ny = elems(&u->planes[0]);
    struct tensor t[MAXP];
    eg_plane* planes[MAXP] = { NULL };
    int32_t slots[MAXP];
    eg_status s = EG_OK;
    int next = 0;
#ifdef EG_PAIR_DEVICE
    void* dev[MAXP] = { NULL };
#endif
    for (int i = 0; i < u->nplanes; ++i) {
        const struct plane_spec* p = &u->planes[i];
        if (i == 0)
            for (int64_t k = 0; k < ny; ++k) buf[0][k] = u->yinit;
        else
            memcpy(buf[i], data->in[i], (size_t)elems(p) * p->bits / 8);
        tensor_init(&t[i], buf[i], p->ndim, p->dims, p->code, p->bits, u->unit_stride);
#ifdef EG_PAIR_DEVICE
        const size_t bytes = (size_t)elems(p) * p->bits / 8;
        if (cudaMalloc(&dev[i], bytes) != cudaSuccess || cudaMemcpy(dev[i], buf[i], bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            s = EG_FAULT;
        t[i].managed.dl_tensor.data = dev[i];
        t[i].managed.dl_tensor.device.device_type = kDLCUDA;
#endif
        while ((u->made >> next) & 1u) ++next;
        slots[i] = next++;
        if (s == EG_OK) s = eg_plane_wrap(ctx, &t[i].managed, &planes[i]);
    }
#ifdef EG_PAIR_DEVICE
    if (cudaDeviceSynchronize() != cudaSuccess) s = EG_FAULT; /* the copies land before the context's stream reads them */
#endif
    const int64_t base[2] = { 0, u->n / 2 }, count[2] = { u->n / 2, u->n / 2 };
    eg_plan* plan = NULL;
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = sh->s;
    d.n_samples = u->n;
    d.n_partitions = sh->parts;
    d.part_base = sh->parts ? base : NULL;
    d.part_count = sh->parts ? count : NULL;
    if (s == EG_OK) s = eg_plan_new(ctx, &d, &plan);
    const int32_t uniform_slots[1] = { u->nplanes + __builtin_popcount(u->made) };
    const double p = 1.0;
    const void* const values[1] = { &p };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = u->nplanes;
    b.plane_slots = slots;
    b.planes = planes;
    b.n_uniforms = u->has_uniform;
    b.uniform_slots = u->has_uniform ? uniform_slots : NULL;
    b.uniform_values = u->has_uniform ? values : NULL;
    if (s == EG_OK) s = eg_bind(ctx, plan, &b);
    if (s == EG_OK) s = eg_launch(ctx, plan, NULL);
#ifdef EG_PAIR_DEVICE
    if (s == EG_OK) s = eg_sync(ctx);
    if (s == EG_OK && cudaMemcpy(buf[0], dev[0], (size_t)ny * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) s = EG_FAULT;
#endif
    if (s != EG_OK) print_reason(ctx);
    eg_plan_free(ctx, plan);
    for (int i = 0; i < u->nplanes; ++i) eg_plane_free(ctx, planes[i]);
#ifdef EG_PAIR_DEVICE
    for (int i = 0; i < u->nplanes; ++i) cudaFree(dev[i]);
#endif
    int64_t first = -1;
    for (int64_t i = 0; i < ny && first < 0; ++i)
        if (memcmp(&buf[0][i], &want[i], sizeof(double)) != 0) first = i;
    if (first >= 0)
        printf("      first difference at element %lld: door %.17g, old path %.17g\n", (long long)first, buf[0][first],
               want[first]);
    char what[160];
    snprintf(what, sizeof(what), "%s, %s: launch OK and y bitwise == the old path", label, sh->name);
    check(s == EG_OK && first < 0, what);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <unit library> <oracle dir> <unit name> <digest hex>\n", argv[0]);
        return 2;
    }
    const struct unit_spec* u = NULL;
    for (size_t k = 0; k < sizeof(units) / sizeof(units[0]); ++k)
        if (strcmp(units[k].name, argv[3]) == 0) u = &units[k];
    if (u == NULL) {
        fprintf(stderr, "no pair row for unit %s\n", argv[3]);
        return 2;
    }
    static struct data data;
    int read_ok = 1;
    for (int i = 0; i < u->nplanes; ++i) {
        const struct plane_spec* p = &u->planes[i];
        if (p->file) read_ok &= read_file(argv[2], p->file, data.in[i], p->bits / 8, (size_t)elems(p));
        if (p->gen) p->gen(data.in[i]);
    }
    for (int i = 0; i < u->nwant; ++i)
        read_ok &= read_file(argv[2], u->want_file[i], data.want[i], sizeof(double), (size_t)elems(&u->planes[0]));
    check(read_ok, u->read_msg);
    if (u->sanity) {
        const char* msg = "";
        const int ok = u->sanity(&data, &msg);
        check(ok, msg);
    }

    int nshapes = u->nshapes;
    const struct shape_spec* shapes = u->shapes;
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
#ifdef EG_PAIR_DEVICE
    if (!u->device) {
        fprintf(stderr, "no device column for unit %s\n", argv[3]);
        return 2;
    }
    st.flags = EG_CTX_DEVICE;
    nshapes = u->made ? 1 : 2; /* a cross-sample write is not split across partitions */
    shapes = device_shapes;
#endif
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, st.flags ? "eg_ctx_new device context (device 0)" : "eg_ctx_new host context");
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
    if (u->threads) eg_host_threads(ctx, u->threads);
    for (int c = 0; c < u->ncases; ++c)
        for (int k = 0; k < nshapes; ++k)
            run_case(ctx, unit, u, &shapes[k], u->cases[c].label, &data, data.want[u->cases[c].want]);
    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
