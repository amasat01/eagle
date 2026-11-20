/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The transform rows of the door (eg_fft_new, eg_fft_run, eg_fft_free), written once for the host test (door_test.c) and
 * the device test (door_device_test.c). The includer defines, before this file, check(ok, what), reason_has(ctx, part)
 * and fft_move(dst, src, bytes): a copy between host memory and the memory of a plane of the context under test.
 *
 * The oracle is independent of the door: a direct O(n^2) DFT per line, written here in long double. Every value test
 * states its tolerance from the point count n and the width's epsilon: 4 * eps * (n + 4) * (1 + max |result|).
 */
#include <float.h>
#include <math.h>

#define FFT_PI 3.14159265358979323846264338327950288L

struct fft_case {
    eg_fft_kind kind;
    uint32_t flags;
    eg_fft_norm norm;
    uint32_t bits;
    int ndim;
    int64_t shape[3];
    int n_axes;
    int32_t axes[3];
};

/* The shapes and axes the value rows run: n even and odd and 1, two axes in both orders, one axis, three axes. */
struct fft_geom {
    int ndim;
    int64_t shape[3];
    int n_axes;
    int32_t axes[3];
};

static const struct fft_geom fft_geoms[] = {
    { 1, { 1 }, 1, { 0 } },       { 1, { 2 }, 1, { 0 } },       { 1, { 7 }, 1, { 0 } },
    { 1, { 8 }, 1, { 0 } },       { 1, { 12 }, 1, { 0 } },      { 2, { 4, 6 }, 2, { 0, 1 } },
    { 2, { 4, 6 }, 2, { 1, 0 } }, { 2, { 4, 6 }, 1, { 0 } },    { 2, { 4, 6 }, 1, { 1 } },
    { 2, { 5, 7 }, 2, { 0, 1 } }, { 3, { 3, 4, 5 }, 1, { 1 } }, { 3, { 3, 4, 5 }, 2, { 0, 2 } },
};
#define FFT_GEOMS ((int)(sizeof(fft_geoms) / sizeof(fft_geoms[0])))

static struct fft_case fft_case_of(const struct fft_geom* g, eg_fft_kind kind, uint32_t flags, eg_fft_norm norm, uint32_t bits)
{
    struct fft_case c;
    memset(&c, 0, sizeof(c));
    c.kind = kind;
    c.flags = flags;
    c.norm = norm;
    c.bits = bits;
    c.ndim = g->ndim;
    memcpy(c.shape, g->shape, sizeof(c.shape));
    c.n_axes = g->n_axes;
    memcpy(c.axes, g->axes, sizeof(c.axes));
    return c;
}

/* A float64 plane over the memory of @p plane, from @p byte_offset on, with its own shape and DLPack flags. It borrows the
 * memory: @p plane outlives it. */
static void* fft_address(eg_ctx* ctx, eg_plane* plane);

struct fft_view_holder {
    struct DLManagedTensorVersioned managed;
    int64_t shape[4];
};

static void fft_view_free(struct DLManagedTensorVersioned* t) { free(t->manager_ctx); }

static eg_plane* fft_view(eg_ctx* ctx, int device, eg_plane* plane, uint64_t byte_offset, int nd, const int64_t* shape, uint64_t flags)
{
    struct fft_view_holder* h = calloc(1, sizeof(*h));
    eg_plane* view = NULL;
    memcpy(h->shape, shape, (size_t)nd * sizeof(int64_t));
    h->managed.version.major = DLPACK_MAJOR_VERSION;
    h->managed.version.minor = DLPACK_MINOR_VERSION;
    h->managed.manager_ctx = h;
    h->managed.deleter = fft_view_free;
    h->managed.flags = flags;
    h->managed.dl_tensor.data = fft_address(ctx, plane);
    h->managed.dl_tensor.byte_offset = byte_offset;
    h->managed.dl_tensor.device.device_type = device ? kDLCUDA : kDLCPU;
    h->managed.dl_tensor.ndim = nd;
    h->managed.dl_tensor.dtype.code = kDLFloat;
    h->managed.dl_tensor.dtype.bits = 64;
    h->managed.dl_tensor.dtype.lanes = 1;
    h->managed.dl_tensor.shape = h->shape;
    if (eg_plane_wrap(ctx, &h->managed, &view) == EG_OK) return view;
    free(h);
    return NULL;
}

static eg_fft_norm fft_swapped(eg_fft_norm norm)
{
    return norm == EG_FFT_NORM_ORTHO ? norm : norm == EG_FFT_NORM_BACKWARD ? EG_FFT_NORM_FORWARD : EG_FFT_NORM_BACKWARD;
}

static double fft_eps(uint32_t bits) { return bits == 32 ? (double)FLT_EPSILON : DBL_EPSILON; }

/* A deterministic value in [-1, 1). */
static double fft_rand(void)
{
    static uint64_t state = 0x9e3779b97f4a7c15ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(int64_t)(state >> 11) / (double)(1ULL << 52) - 1.0;
}

/* @p v rounded to the plane's width, so the oracle reads what the plane holds. */
static void fft_fill(double* v, size_t n, uint32_t bits)
{
    for (size_t i = 0; i < n; ++i) v[i] = bits == 32 ? (double)(float)fft_rand() : fft_rand();
}

/* The shape of the plane the door reads (@p input) or writes for @p c: the transform's input and output, swapped for an
 * ADJOINT plan. A complex side carries the trailing 2 and, for a real transform, the halved last axis. */
static int fft_plane(const struct fft_case* c, int input, int64_t* shape)
{
    const int adjoint = (c->flags & EG_FFT_ADJOINT) != 0;
    const int of_transform_input = input != adjoint;
    const int complex_side = c->kind == EG_FFT_C2C || (c->kind == EG_FFT_R2C) != of_transform_input;
    memcpy(shape, c->shape, (size_t)c->ndim * sizeof(int64_t));
    if (!complex_side) return c->ndim;
    if (c->kind != EG_FFT_C2C) shape[c->axes[c->n_axes - 1]] = shape[c->axes[c->n_axes - 1]] / 2 + 1;
    shape[c->ndim] = 2;
    return c->ndim + 1;
}

static size_t fft_floats(const int64_t* shape, int nd)
{
    size_t n = 1;
    for (int i = 0; i < nd; ++i) n *= (size_t)shape[i];
    return n;
}

static size_t fft_points(const struct fft_case* c)
{
    size_t n = 1;
    for (int i = 0; i < c->n_axes; ++i) n *= (size_t)c->shape[c->axes[i]];
    return n;
}

/* ---- the oracle: direct DFTs in long double ---------------------------------------------------------------------------- */

/* In place, along @p axis of the complex array @p a (re, im pairs) of shape @p sh: sign -1 forward, +1 inverse, unscaled. */
static void fft_dft_axis(long double* a, const int64_t* sh, int nd, int axis, int sign)
{
    int64_t inner = 1, outer = 1, n = sh[axis];
    for (int k = axis + 1; k < nd; ++k) inner *= sh[k];
    for (int k = 0; k < axis; ++k) outer *= sh[k];
    long double* line = malloc((size_t)(2 * n) * sizeof(long double));
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                line[2 * j] = a[2 * ((o * n + j) * inner + i)];
                line[2 * j + 1] = a[2 * ((o * n + j) * inner + i) + 1];
            }
            for (int64_t k = 0; k < n; ++k) {
                long double re = 0, im = 0;
                for (int64_t j = 0; j < n; ++j) {
                    const long double angle = sign * 2 * FFT_PI * (long double)((j * k) % n) / (long double)n;
                    re += line[2 * j] * cosl(angle) - line[2 * j + 1] * sinl(angle);
                    im += line[2 * j] * sinl(angle) + line[2 * j + 1] * cosl(angle);
                }
                a[2 * ((o * n + k) * inner + i)] = re;
                a[2 * ((o * n + k) * inner + i) + 1] = im;
            }
        }
    free(line);
}

/* numpy's factor of the norm over the point count n of the transform (forward unless an inverse). */
static long double fft_norm_factor(const struct fft_case* c, int forward)
{
    const long double n = (long double)fft_points(c);
    if (c->norm == EG_FFT_NORM_ORTHO) return 1.0L / sqrtl(n);
    return forward == (c->norm == EG_FFT_NORM_FORWARD) ? 1.0L / n : 1.0L;
}

/* The result the transform @p c (no ADJOINT) gives @p in, into @p out (the output plane's floats). */
static void fft_oracle(const struct fft_case* c, const double* in, long double* out)
{
    int64_t cs[4], os[4], is[4];
    const int cn = fft_plane(c, 1, is), on = fft_plane(c, 0, os);
    const size_t n_in = fft_floats(is, cn), n_out = fft_floats(os, on);
    const int last = c->axes[c->n_axes - 1];
    if (c->kind == EG_FFT_C2C) {
        const int forward = (c->flags & EG_FFT_INVERSE) == 0;
        for (size_t i = 0; i < n_in; ++i) out[i] = in[i];
        for (int a = 0; a < c->n_axes; ++a) fft_dft_axis(out, os, on - 1, c->axes[a], forward ? -1 : 1);
        const long double f = fft_norm_factor(c, forward);
        for (size_t i = 0; i < n_out; ++i) out[i] *= f;
    } else if (c->kind == EG_FFT_R2C) {
        const size_t points = fft_floats(is, cn);
        long double* full = calloc(2 * points, sizeof(long double));
        for (size_t i = 0; i < points; ++i) full[2 * i] = in[i];
        memcpy(cs, c->shape, sizeof(c->shape));
        for (int a = 0; a < c->n_axes; ++a) fft_dft_axis(full, cs, c->ndim, c->axes[a], -1);
        const int64_t m = c->shape[last] / 2 + 1;
        int64_t inner = 1, outer = 1;
        for (int k = last + 1; k < c->ndim; ++k) inner *= c->shape[k];
        for (int k = 0; k < last; ++k) outer *= c->shape[k];
        const long double f = fft_norm_factor(c, 1);
        for (int64_t o = 0; o < outer; ++o)
            for (int64_t k = 0; k < m; ++k)
                for (int64_t i = 0; i < inner; ++i)
                    for (int p = 0; p < 2; ++p)
                        out[2 * ((o * m + k) * inner + i) + p] = f * full[2 * ((o * c->shape[last] + k) * inner + i) + p];
        free(full);
    } else {
        long double* z = malloc(n_in * sizeof(long double));
        for (size_t i = 0; i < n_in; ++i) z[i] = in[i];
        for (int a = 0; a + 1 < c->n_axes; ++a) fft_dft_axis(z, is, cn - 1, c->axes[a], 1);
        const int64_t n = c->shape[last], m = n / 2 + 1;
        int64_t inner = 1, outer = 1;
        for (int k = last + 1; k < c->ndim; ++k) inner *= c->shape[k];
        for (int k = 0; k < last; ++k) outer *= c->shape[k];
        const long double f = fft_norm_factor(c, 0);
        for (int64_t o = 0; o < outer; ++o)
            for (int64_t i = 0; i < inner; ++i)
                for (int64_t j = 0; j < n; ++j) {
                    long double sum = 0;
                    for (int64_t k = 0; k < n; ++k) {
                        const int64_t from = k < m ? k : n - k;
                        const long double zr = z[2 * ((o * m + from) * inner + i)];
                        const long double zi = (k < m ? 1 : -1) * z[2 * ((o * m + from) * inner + i) + 1];
                        const long double angle = 2 * FFT_PI * (long double)((j * k) % n) / (long double)n;
                        sum += zr * cosl(angle) - zi * sinl(angle);
                    }
                    out[(o * n + j) * inner + i] = f * sum;
                }
        free(z);
    }
}

/* ---- running the door ------------------------------------------------------------------------------------------------ */

static eg_plane* fft_make_plane(eg_ctx* ctx, int device, uint32_t bits, int nd, const int64_t* shape)
{
    struct eg_plane_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = EG_PLANE_ZERO;
    d.kind = device ? EG_MEM_DEVICE : EG_MEM_HOST;
    d.ndim = nd;
    d.dtype_code = kDLFloat;
    d.dtype_bits = bits;
    d.shape = shape;
    eg_plane* p = NULL;
    return eg_plane_new(ctx, &d, &p) == EG_OK ? p : NULL;
}

/* The memory of @p plane (the plane stays alive: the export is released at once). */
static void* fft_address(eg_ctx* ctx, eg_plane* plane)
{
    struct DLManagedTensorVersioned* t = NULL;
    if (eg_plane_dlpack(ctx, plane, &t) != EG_OK) return NULL;
    void* data = (char*)t->dl_tensor.data + t->dl_tensor.byte_offset;
    t->deleter(t);
    return data;
}

/* Write @p v (n values) into @p plane at the plane's width. */
static void fft_put(eg_ctx* ctx, eg_plane* plane, uint32_t bits, const double* v, size_t n)
{
    void* staging = malloc(n * (bits / 8) + 1);
    for (size_t i = 0; i < n; ++i)
        if (bits == 32) ((float*)staging)[i] = (float)v[i];
        else ((double*)staging)[i] = v[i];
    eg_sync(ctx);
    fft_move(fft_address(ctx, plane), staging, n * (bits / 8));
    free(staging);
}

/* Read @p plane (n values) into @p v, once the context's queued work is done. */
static void fft_get(eg_ctx* ctx, eg_plane* plane, uint32_t bits, double* v, size_t n)
{
    void* staging = malloc(n * (bits / 8) + 1);
    eg_sync(ctx);
    fft_move(staging, fft_address(ctx, plane), n * (bits / 8));
    for (size_t i = 0; i < n; ++i) v[i] = bits == 32 ? (double)((float*)staging)[i] : ((double*)staging)[i];
    free(staging);
}

static eg_status fft_new(eg_ctx* ctx, const struct fft_case* c, eg_fft** plan)
{
    struct eg_fft_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = c->flags;
    d.kind = c->kind;
    d.norm = c->norm;
    d.dtype_bits = c->bits;
    d.ndim = c->ndim;
    d.shape = c->shape;
    d.n_axes = c->n_axes;
    d.axes = c->axes;
    *plan = NULL;
    return eg_fft_new(ctx, &d, plan);
}

/* The planes and plan of @p c, made once. */
struct fft_rig {
    eg_plane *in, *out;
    eg_fft* plan;
    size_t n_in, n_out;
    eg_status status; /* of eg_fft_new */
};

static struct fft_rig fft_rig_new(eg_ctx* ctx, int device, const struct fft_case* c)
{
    struct fft_rig r;
    int64_t is[4], os[4];
    const int in_nd = fft_plane(c, 1, is), out_nd = fft_plane(c, 0, os);
    r.n_in = fft_floats(is, in_nd);
    r.n_out = fft_floats(os, out_nd);
    r.in = fft_make_plane(ctx, device, c->bits, in_nd, is);
    r.out = fft_make_plane(ctx, device, c->bits, out_nd, os);
    r.status = fft_new(ctx, c, &r.plan);
    return r;
}

static void fft_rig_free(eg_ctx* ctx, struct fft_rig* r)
{
    eg_fft_free(ctx, r->plan);
    eg_plane_free(ctx, r->in);
    eg_plane_free(ctx, r->out);
}

/* Run @p c on @p in into @p out (the output's floats). The status of the first call that is not OK, else EG_OK. */
static eg_status fft_apply(eg_ctx* ctx, int device, const struct fft_case* c, const double* in, double* out)
{
    struct fft_rig r = fft_rig_new(ctx, device, c);
    eg_status status = r.in != NULL && r.out != NULL ? r.status : EG_FAULT;
    if (status == EG_OK) {
        fft_put(ctx, r.in, c->bits, in, r.n_in);
        status = eg_fft_run(ctx, r.plan, r.in, r.out);
    }
    if (status == EG_OK) fft_get(ctx, r.out, c->bits, out, r.n_out);
    fft_rig_free(ctx, &r);
    return status;
}

static double fft_max_abs(const long double* v, size_t n)
{
    long double m = 0;
    for (size_t i = 0; i < n; ++i) m = fabsl(v[i]) > m ? fabsl(v[i]) : m;
    return (double)m;
}

static const char* fft_kind_name(eg_fft_kind kind, uint32_t flags)
{
    if (kind == EG_FFT_R2C) return "r2c";
    if (kind == EG_FFT_C2R) return "c2r";
    return (flags & EG_FFT_INVERSE) ? "c2c inverse" : "c2c forward";
}

/* ---- the rows ------------------------------------------------------------------------------------------------------- */

/* Every kind, width, norm and shape of fft_geoms equals the oracle within the stated tolerance. */
static void fft_values(eg_ctx* ctx, int device)
{
    static const struct { eg_fft_kind kind; uint32_t flags; } kinds[] = { { EG_FFT_C2C, 0 }, { EG_FFT_C2C, EG_FFT_INVERSE }, { EG_FFT_R2C, 0 }, { EG_FFT_C2R, 0 } };
    static const uint32_t widths[] = { 32, 64 };
    for (int k = 0; k < 4; ++k)
        for (int w = 0; w < 2; ++w) {
            int ok = 1, runs = 0;
            double worst = 0;
            for (int g = 0; g < FFT_GEOMS && ok; ++g)
                for (int norm = EG_FFT_NORM_BACKWARD; norm <= EG_FFT_NORM_ORTHO && ok; ++norm) {
                    const struct fft_case c = fft_case_of(&fft_geoms[g], kinds[k].kind, kinds[k].flags, norm, widths[w]);
                    int64_t is[4], os[4];
                    const size_t n_in = fft_floats(is, fft_plane(&c, 1, is)), n_out = fft_floats(os, fft_plane(&c, 0, os));
                    double* in = malloc(n_in * sizeof(double));
                    double* got = malloc(n_out * sizeof(double));
                    long double* want = malloc(n_out * sizeof(long double));
                    fft_fill(in, n_in, c.bits);
                    fft_oracle(&c, in, want);
                    const double tol = 4 * fft_eps(c.bits) * ((double)fft_points(&c) + 4) * (1 + fft_max_abs(want, n_out));
                    ok = fft_apply(ctx, device, &c, in, got) == EG_OK;
                    for (size_t i = 0; ok && i < n_out; ++i) {
                        const double e = fabs((double)((long double)got[i] - want[i]));
                        worst = e > worst ? e : worst;
                        ok = e <= tol;
                    }
                    if (!ok) printf("      geometry %d norm %d: tolerance %g\n", g, norm, tol);
                    ++runs;
                    free(in);
                    free(got);
                    free(want);
                }
            char what[160];
            snprintf(what, sizeof(what), "eg_fft %s float%u: %d shapes x norms equal the O(n^2) long double DFT within 4*eps*(n+4)*(1+max) (worst %.3g)",
                     fft_kind_name(kinds[k].kind, kinds[k].flags), widths[w], runs, worst);
            check(ok, what);
        }
}

/* Interior bins (0 < k < last along the halved axis) of the half spectrum @p v of @p c times @p factor. */
static void fft_weigh_interior(double* v, const struct fft_case* c, double factor)
{
    int64_t shape[4];
    const int nd = fft_plane(c, c->kind == EG_FFT_C2R, shape);
    const int last = c->axes[c->n_axes - 1];
    const int64_t n = c->shape[last], m = n / 2 + 1, edge = n % 2 == 0 ? m - 1 : m;
    int64_t run = 2;
    for (int k = last + 1; k < nd - 1; ++k) run *= shape[k];
    for (size_t i = 0; i < fft_floats(shape, nd); ++i) {
        const int64_t bin = ((int64_t)i / run) % m;
        if (bin > 0 && bin < edge) v[i] *= factor;
    }
}

/* <T x, y> == <x, T^H y> for every kind and norm, and the adjoint of the adjoint is the transform. */
static void fft_adjoint(eg_ctx* ctx, int device)
{
    static const struct { eg_fft_kind kind; uint32_t flags; } kinds[] = { { EG_FFT_C2C, 0 }, { EG_FFT_C2C, EG_FFT_INVERSE }, { EG_FFT_R2C, 0 }, { EG_FFT_C2R, 0 } };
    for (int k = 0; k < 4; ++k)
        for (int w = 0; w < 2; ++w) {
            const uint32_t bits = w == 0 ? 32 : 64;
            int ok = 1, twice = 1;
            for (int g = 0; g < FFT_GEOMS && ok && twice; ++g)
                for (int norm = EG_FFT_NORM_BACKWARD; norm <= EG_FFT_NORM_ORTHO && ok && twice; ++norm) {
                    const struct fft_case t = fft_case_of(&fft_geoms[g], kinds[k].kind, kinds[k].flags, norm, bits);
                    struct fft_case h = t;
                    h.flags |= EG_FFT_ADJOINT;
                    int64_t is[4], os[4];
                    const size_t n_in = fft_floats(is, fft_plane(&t, 1, is)), n_out = fft_floats(os, fft_plane(&t, 0, os));
                    double *x = malloc(n_in * sizeof(double)), *y = malloc(n_out * sizeof(double));
                    double *tx = malloc(n_out * sizeof(double)), *hy = malloc(n_in * sizeof(double));
                    fft_fill(x, n_in, bits);
                    fft_fill(y, n_out, bits);
                    ok = fft_apply(ctx, device, &t, x, tx) == EG_OK && fft_apply(ctx, device, &h, y, hy) == EG_OK;
                    long double lhs = 0, rhs = 0, nx = 0, ny = 0;
                    for (size_t i = 0; ok && i < n_out; ++i) lhs += (long double)tx[i] * y[i], nx += (long double)tx[i] * tx[i], ny += (long double)y[i] * y[i];
                    for (size_t i = 0; ok && i < n_in; ++i) rhs += (long double)x[i] * hy[i];
                    const double tol = 16 * fft_eps(bits) * ((double)fft_points(&t) + 4) * (1 + (double)sqrtl(nx * ny));
                    ok = ok && fabsl(lhs - rhs) <= tol;
                    if (!ok) printf("      geometry %d norm %d: <Tx,y> %.9Lg <x,T^H y> %.9Lg tolerance %g\n", g, norm, lhs, rhs, tol);
                    /* T^H of T^H is T: the adjoint plan of the explicit adjoint */
                    double* again = malloc(n_out * sizeof(double));
                    long double* want = malloc(n_out * sizeof(long double));
                    struct fft_case b = t;
                    b.norm = fft_swapped(norm);
                    if (t.kind == EG_FFT_C2C) {
                        b.flags = (t.flags ^ EG_FFT_INVERSE) | EG_FFT_ADJOINT;
                        twice = ok && fft_apply(ctx, device, &b, x, again) == EG_OK && memcmp(again, tx, n_out * sizeof(double)) == 0;
                    } else {
                        /* R2C^H = C2R(norm swapped) after halving; C2R^H = R2C(norm swapped) before doubling. The test applies the
                         * other weight itself and the result is the oracle's transform. */
                        b.kind = t.kind == EG_FFT_R2C ? EG_FFT_C2R : EG_FFT_R2C;
                        b.flags = EG_FFT_ADJOINT;
                        double* given = malloc(n_in * sizeof(double));
                        memcpy(given, x, n_in * sizeof(double));
                        if (t.kind == EG_FFT_C2R) fft_weigh_interior(given, &t, 2.0);
                        twice = ok && fft_apply(ctx, device, &b, given, again) == EG_OK;
                        if (t.kind == EG_FFT_R2C) fft_weigh_interior(again, &t, 0.5);
                        fft_oracle(&t, x, want);
                        const double tolerance = 4 * fft_eps(bits) * ((double)fft_points(&t) + 4) * (1 + fft_max_abs(want, n_out));
                        for (size_t i = 0; twice && i < n_out; ++i) twice = fabs((double)((long double)again[i] - want[i])) <= tolerance;
                        free(given);
                    }
                    if (!twice) printf("      geometry %d norm %d: the adjoint of the adjoint differs\n", g, norm);
                    free(x);
                    free(y);
                    free(tx);
                    free(hy);
                    free(again);
                    free(want);
                }
            char what[160];
            snprintf(what, sizeof(what), "eg_fft_new ADJOINT of %s float%u: <Tx,y> == <x,T^H y> (all shapes and norms) and T^H of T^H is T", fft_kind_name(kinds[k].kind, kinds[k].flags), bits);
            check(ok && twice, what);
        }
}

/* One case, run in place on one plane (C2C) and the refusal of the real kinds. */
static void fft_in_place(eg_ctx* ctx, int device)
{
    const struct fft_case c = fft_case_of(&fft_geoms[5], EG_FFT_C2C, 0, EG_FFT_NORM_ORTHO, 64);
    int64_t s[4];
    const size_t n = fft_floats(s, fft_plane(&c, 1, s));
    double *x = malloc(n * sizeof(double)), *apart = malloc(n * sizeof(double)), *inplace = malloc(n * sizeof(double));
    fft_fill(x, n, 64);
    eg_plane* p = fft_make_plane(ctx, device, 64, 3, s);
    eg_fft* plan = NULL;
    int ok = fft_apply(ctx, device, &c, x, apart) == EG_OK && p != NULL && fft_new(ctx, &c, &plan) == EG_OK;
    if (ok) {
        fft_put(ctx, p, 64, x, n);
        ok = eg_fft_run(ctx, plan, p, p) == EG_OK;
        fft_get(ctx, p, 64, inplace, n);
    }
    check(ok && memcmp(apart, inplace, n * sizeof(double)) == 0, "eg_fft_run C2C with in == out gives the bits of the run on two planes");
    eg_fft_free(ctx, plan);
    eg_plane_free(ctx, p);
    free(x);
    free(apart);
    free(inplace);
}

/* A recorded run replays to the bits of the direct run, also after the caller freed the plan; eg_fft_new is refused inside. */
static void fft_recorded(eg_ctx* ctx, int device)
{
    static const struct { eg_fft_kind kind; uint32_t flags; int geom; } rows[] = { { EG_FFT_C2R, 0, 6 }, { EG_FFT_R2C, EG_FFT_ADJOINT, 5 } };
    int ok = 1;
    for (int r = 0; r < 2 && ok; ++r) {
        const struct fft_case c = fft_case_of(&fft_geoms[rows[r].geom], rows[r].kind, rows[r].flags, EG_FFT_NORM_FORWARD, 64);
        struct fft_rig rig = fft_rig_new(ctx, device, &c);
        double *x = malloc(rig.n_in * sizeof(double)), *direct = malloc(rig.n_out * sizeof(double)), *replay = malloc(rig.n_out * sizeof(double));
        double* zeros = calloc(rig.n_out, sizeof(double));
        fft_fill(x, rig.n_in, 64);
        fft_put(ctx, rig.in, 64, x, rig.n_in);
        ok = rig.status == EG_OK && eg_fft_run(ctx, rig.plan, rig.in, rig.out) == EG_OK;
        fft_get(ctx, rig.out, 64, direct, rig.n_out);
        fft_put(ctx, rig.out, 64, zeros, rig.n_out);
        eg_graph* graph = NULL;
        eg_fft* again = NULL;
        ok = ok && eg_record_begin(ctx) == EG_OK && eg_fft_run(ctx, rig.plan, rig.in, rig.out) == EG_OK;
        ok = ok && fft_new(ctx, &c, &again) == EG_BAD_REQUEST && reason_has(ctx, "while recording");
        ok = ok && eg_record_end(ctx, NULL, &graph) == EG_OK && eg_fft_free(ctx, rig.plan) == EG_OK;
        rig.plan = NULL;
        if (!device) { /* a host recording runs nothing until it replays */
            fft_get(ctx, rig.out, 64, replay, rig.n_out);
            ok = ok && memcmp(replay, zeros, rig.n_out * sizeof(double)) == 0;
        }
        for (int launch = 0; ok && launch < 2; ++launch) {
            fft_put(ctx, rig.out, 64, zeros, rig.n_out);
            ok = eg_graph_launch(ctx, graph) == EG_OK;
            fft_get(ctx, rig.out, 64, replay, rig.n_out);
            ok = ok && memcmp(replay, direct, rig.n_out * sizeof(double)) == 0;
        }
        eg_graph_free(ctx, graph);
        fft_rig_free(ctx, &rig);
        free(x);
        free(direct);
        free(replay);
        free(zeros);
    }
    check(ok, "a recorded eg_fft_run (C2R with its input copy, adjoint R2C with its weight) replays twice to the bits of the direct run after eg_fft_free; eg_fft_new while recording is refused");
}

/* ---- refusals ------------------------------------------------------------------------------------------------------- */

static eg_status fft_refuse_new(eg_ctx* ctx, struct fft_case c, int64_t* shape, int32_t* axes, int bad)
{
    struct eg_fft_desc d;
    eg_fft* plan = NULL;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = c.flags;
    d.kind = c.kind;
    d.norm = c.norm;
    d.dtype_bits = c.bits;
    d.ndim = c.ndim;
    d.shape = shape;
    d.n_axes = c.n_axes;
    d.axes = axes;
    if (bad == 1) d.shape = NULL;
    if (bad == 2) d.axes = NULL;
    const eg_status s = eg_fft_new(ctx, &d, &plan);
    if (s == EG_OK) eg_fft_free(ctx, plan);
    return s;
}

/* Every refusal of eg_fft_new, eg_fft_run and eg_fft_free, by status and reason. */
static void fft_refusals(eg_ctx* ctx, int device)
{
    int64_t shape[3] = { 4, 6, 0 }, negative[3] = { 4, -1, 0 }, empty[3] = { 0, 6, 0 };
    int32_t axes[3] = { 0, 1, 0 }, twice[3] = { 1, 1, 0 }, outside[3] = { 0, 2, 0 };
    const struct fft_case base = fft_case_of(&fft_geoms[5], EG_FFT_C2C, 0, EG_FFT_NORM_BACKWARD, 64);
    struct fft_case c;
    int all = 1;
#define FFT_REFUSED(case_expr, shp, axs, bad, status, part)                                            \
    do {                                                                                              \
        c = base;                                                                                     \
        case_expr;                                                                                    \
        all = all && fft_refuse_new(ctx, c, shp, axs, bad) == (status) && reason_has(ctx, part);      \
    } while (0)
    FFT_REFUSED(c.flags = 1u << 5, shape, axes, 0, EG_BAD_REQUEST, "unknown bits");
    FFT_REFUSED(c.kind = 7, shape, axes, 0, EG_BAD_REQUEST, "kind 7");
    FFT_REFUSED(c.norm = 5, shape, axes, 0, EG_BAD_REQUEST, "norm 5");
    FFT_REFUSED(c.bits = 16, shape, axes, 0, EG_NOT_WRITTEN, "32 and 64 are written");
    FFT_REFUSED(c.bits = 8, shape, axes, 0, EG_BAD_REQUEST, "not a float width");
    FFT_REFUSED(c.ndim = 0, shape, axes, 0, EG_BAD_REQUEST, "ndim 0");
    FFT_REFUSED((void)0, shape, axes, 1, EG_BAD_REQUEST, "without a shape");
    FFT_REFUSED((void)0, negative, axes, 0, EG_BAD_REQUEST, "negative");
    FFT_REFUSED(c.n_axes = 0, shape, axes, 0, EG_BAD_REQUEST, "no axes");
    FFT_REFUSED((void)0, shape, axes, 2, EG_BAD_REQUEST, "no axes");
    FFT_REFUSED((void)0, shape, twice, 0, EG_BAD_REQUEST, "twice");
    FFT_REFUSED((void)0, shape, outside, 0, EG_BAD_REQUEST, "outside");
    FFT_REFUSED((void)0, empty, axes, 0, EG_BAD_REQUEST, "length 0");
    FFT_REFUSED((c.kind = EG_FFT_R2C, c.flags = EG_FFT_INVERSE), shape, axes, 0, EG_BAD_REQUEST, "INVERSE is for C2C");
    FFT_REFUSED((c.kind = EG_FFT_C2R, c.flags = EG_FFT_INVERSE | EG_FFT_ADJOINT), shape, axes, 0, EG_BAD_REQUEST, "INVERSE is for C2C");
#undef FFT_REFUSED
    check(all, "eg_fft_new refuses an unknown flag, kind or norm, a float width other than 32 and 64, a bad shape or axis list, and INVERSE on a real kind, by status and reason");

    /* eg_fft_run: the planes */
    int64_t s2[3] = { 4, 6, 2 }, wrong[3] = { 4, 7, 2 }, real[2] = { 4, 6 }, half[3] = { 4, 4, 2 }, wide[3] = { 4, 7, 2 };
    eg_plane* good = fft_make_plane(ctx, device, 64, 3, s2);
    eg_plane* other = fft_make_plane(ctx, device, 64, 3, s2);
    eg_plane* narrow = fft_make_plane(ctx, device, 32, 3, s2);
    eg_plane* bad_shape = fft_make_plane(ctx, device, 64, 3, wrong);
    eg_plane* plain = fft_make_plane(ctx, device, 64, 2, real);
    eg_plane* spectrum = fft_make_plane(ctx, device, 64, 3, half);
    eg_plane* big = fft_make_plane(ctx, device, 64, 3, wide);
    eg_fft *c2c = NULL, *r2c = NULL;
    struct fft_case rc = fft_case_of(&fft_geoms[5], EG_FFT_R2C, 0, EG_FFT_NORM_BACKWARD, 64);
    all = good && other && narrow && bad_shape && plain && spectrum && big && fft_new(ctx, &base, &c2c) == EG_OK && fft_new(ctx, &rc, &r2c) == EG_OK;
    /* views of one buffer: a real plane under its half spectrum, and two complex planes shifted by one float */
    eg_plane* under = all ? fft_view(ctx, device, spectrum, 0, 2, real, 0) : NULL;
    eg_plane* first = all ? fft_view(ctx, device, big, 0, 3, s2, 0) : NULL;
    eg_plane* shifted = all ? fft_view(ctx, device, big, sizeof(double), 3, s2, 0) : NULL;
    eg_plane* frozen = all ? fft_view(ctx, device, other, 0, 3, s2, DLPACK_FLAG_BITMASK_READ_ONLY) : NULL;
    all = all && under && first && shifted && frozen;
    all = all && eg_fft_run(ctx, c2c, narrow, other) == EG_BAD_REQUEST && reason_has(ctx, "must be float64");
    all = all && eg_fft_run(ctx, c2c, good, bad_shape) == EG_BAD_REQUEST && reason_has(ctx, "must have shape (4, 6, 2), got (4, 7, 2)");
    all = all && eg_fft_run(ctx, c2c, plain, other) == EG_BAD_REQUEST && reason_has(ctx, "must have shape (4, 6, 2), got (4, 6)");
    all = all && eg_fft_run(ctx, r2c, plain, other) == EG_BAD_REQUEST && reason_has(ctx, "must have shape (4, 4, 2), got (4, 6, 2)");
    all = all && eg_fft_run(ctx, r2c, plain, spectrum) == EG_OK;
    all = all && eg_fft_run(ctx, r2c, under, spectrum) == EG_BAD_REQUEST && reason_has(ctx, "only C2C runs in place");
    all = all && eg_fft_run(ctx, c2c, first, shifted) == EG_BAD_REQUEST && reason_has(ctx, "without being one range");
    all = all && eg_fft_run(ctx, c2c, good, first) == EG_OK && eg_fft_run(ctx, c2c, first, first) == EG_OK;
    all = all && eg_fft_run(ctx, c2c, good, frozen) == EG_BAD_REQUEST && reason_has(ctx, "read-only");
    all = all && eg_fft_run(ctx, NULL, good, other) == EG_BAD_REQUEST && eg_fft_run(ctx, c2c, NULL, other) == EG_BAD_REQUEST &&
         eg_fft_run(ctx, c2c, good, NULL) == EG_BAD_REQUEST;
    if (device) { /* the plan runs where its context does: a host plane is the wrong target */
        eg_plane* host_plane = fft_make_plane(ctx, 0, 64, 3, s2);
        all = all && host_plane && eg_fft_run(ctx, c2c, host_plane, other) == EG_BAD_REQUEST && reason_has(ctx, "plan runs on the device");
        eg_plane_free(ctx, host_plane);
    }
    check(all, "eg_fft_run refuses a plane of the wrong width, shape, last axis, target or access, an overlap other than one C2C range, and a NULL handle, by status and reason");

    /* handles */
    eg_ctx* foreign = NULL;
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    all = eg_ctx_new(&st, &foreign) == EG_OK;
    all = all && eg_fft_free(ctx, NULL) == EG_OK;
    all = all && eg_fft_free(foreign, c2c) == EG_BAD_REQUEST && reason_has(foreign, "not a live handle");
    all = all && eg_fft_free(ctx, c2c) == EG_OK && eg_fft_free(ctx, c2c) == EG_BAD_REQUEST;
    all = all && eg_fft_run(ctx, c2c, good, other) == EG_BAD_REQUEST && reason_has(ctx, "not a live handle");
    check(all, "eg_fft_free is NULL-safe, refuses a handle of another context, and a freed plan is refused by eg_fft_run");
    eg_ctx_free(foreign);
    eg_fft_free(ctx, r2c);
    eg_plane_free(ctx, good);
    eg_plane_free(ctx, other);
    eg_plane_free(ctx, narrow);
    eg_plane_free(ctx, bad_shape);
    eg_plane_free(ctx, plain);
    eg_plane_free(ctx, spectrum);
    eg_plane_free(ctx, big);
    eg_plane_free(ctx, under);
    eg_plane_free(ctx, first);
    eg_plane_free(ctx, shifted);
    eg_plane_free(ctx, frozen);
}
