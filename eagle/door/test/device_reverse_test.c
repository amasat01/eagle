// SPDX-License-Identifier: Apache-2.0
/*
 * A recorded run and its reverse on the device (EG_RUN_RECORD over a DEVICE plan, per-step reverse): the one-step taping
 * twin and the step vjp of a fixed-step unit, recorded until done (the finished count read back once per window) and
 * reversed, the end states, the report's steps and launches and the gradients compared BITWISE with the old device
 * path's on the same inputs. Beside the oracle: a second record and reverse on the same runner repeats it, and what a
 * device record does not run yet is refused by name.
 *
 * usage: device_reverse_test <library> <oracle dir> <primal digest> <twin digest> <vjp digest> <primal name> <window>
 * oracle dir: x0, v0, tf, end_{x,v,t}, grad_{x,v,t,c,dt}, steps and run (steps, launches) .bin.
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 64
#define STEPS 400

/* Device planes made from host values; freed together. */
struct dplanes {
    void* d[16];
    struct tensor t[16];
    eg_plane* p[16];
    int k;
};

static eg_plane* dplane(eg_ctx* ctx, struct dplanes* s, const void* init, int64_t n, uint8_t code, uint8_t bits) {
    const int i = s->k++;
    const size_t bytes = (size_t)n * bits / 8;
    s->d[i] = NULL;
    s->p[i] = NULL;
    if (cudaMalloc(&s->d[i], bytes) != cudaSuccess || cudaMemcpy(s->d[i], init, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    s->p[i] = wrap(ctx, &s->t[i], n, s->d[i], code, bits);
    return s->p[i];
}

static void dfree(eg_ctx* ctx, struct dplanes* s) {
    for (int i = 0; i < s->k; ++i) {
        if (s->p[i]) eg_plane_free(ctx, s->p[i]);
        if (s->d[i]) cudaFree(s->d[i]);
    }
    s->k = 0;
}

static int dget(const struct dplanes* s, int i, void* host, size_t bytes) {
    return s->d[i] && cudaMemcpy(host, s->d[i], bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

static int dput(const struct dplanes* s, int i, const void* host, size_t bytes) {
    return s->d[i] && cudaMemcpy(s->d[i], host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

static eg_unit* load(eg_ctx* ctx, const char* path, const char* name, const uint8_t* digest) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memcpy(ref.digest, digest, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = path;
    d.unit = &ref;
    eg_unit* u = NULL;
    return eg_unit_load(ctx, &d, &u) == EG_OK ? u : NULL;
}

static struct eg_bind_desc binds(int32_t n_planes, const int32_t* slots, eg_plane* const* planes, int32_t n_uniforms,
                                 const int32_t* uslots, const void* const* values) {
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = n_planes;
    d.plane_slots = slots;
    d.planes = planes;
    d.n_uniforms = n_uniforms;
    d.uniform_slots = uslots;
    d.uniform_values = values;
    return d;
}

static double x0[N], v0[N], tf[N], ex[N], ev[N], et[N], gx[N], gv[N], gt[N], gc, gdt;
static int64_t run_ref[2];

/* One record (states and report against the oracle) and its reverse (gradients against the oracle); @p seed_host: the
 * seeds are host planes, refused by name. */
static void record_and_reverse(eg_ctx* ctx, eg_runner* r, const struct dplanes* s, const char* what0, int seed_host) {
    static const double zeros[N];
    static const uint8_t none[N];
    char what[320];
    int ok = dput(s, 0, zeros, sizeof(zeros)) && dput(s, 1, v0, sizeof(v0)) && dput(s, 2, x0, sizeof(x0)) && dput(s, 4, none, sizeof(none));
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    ok = ok && eg_run(ctx, r, &rep) == EG_OK;
    static double x[N], v[N], t[N];
    ok = ok && dget(s, 0, t, sizeof(t)) && dget(s, 1, v, sizeof(v)) && dget(s, 2, x, sizeof(x));
    int s_ok = ok && rep.steps == run_ref[0] && rep.launches == run_ref[1] && rep.finished == N && (rep.flags & EG_REPORT_DONE) &&
               rep.mode == EG_MODE_BAND;
    for (int i = 0; s_ok && i < N; ++i) s_ok = memcmp(&x[i], &ex[i], 8) == 0 && memcmp(&v[i], &ev[i], 8) == 0 && memcmp(&t[i], &et[i], 8) == 0;
    snprintf(what, sizeof(what), "%s: the recorded run's states equal the old device path's bitwise (%lld steps, %lld launches, "
             "%lld finished)", what0, (long long)rep.steps, (long long)rep.launches, (long long)rep.finished);
    check(s_ok, what);
    static double sx[N], sv[N], ox[N], ov[N], ot[N];
    double oc = 0.0, odt = 0.0;
    for (int i = 0; i < N; ++i) {
        sx[i] = 1.0;
        sv[i] = 0.5;
    }
    struct dplanes g = { .k = 0 };
    eg_plane* sp[2];
    struct tensor host_t[2];
    if (seed_host) {
        for (int i = 0; i < 2; ++i) {
            tensor_fill(&host_t[i], N, i ? sv : sx, kDLFloat, 64);
            host_t[i].managed.dl_tensor.device.device_type = kDLCPU;
            sp[i] = NULL;
            eg_plane_wrap(ctx, &host_t[i].managed, &sp[i]);
        }
    } else {
        sp[0] = dplane(ctx, &g, sx, N, kDLFloat, 64);
        sp[1] = dplane(ctx, &g, sv, N, kDLFloat, 64);
    }
    eg_plane* const gp[5] = { dplane(ctx, &g, ox, N, kDLFloat, 64), dplane(ctx, &g, ov, N, kDLFloat, 64), dplane(ctx, &g, ot, N, kDLFloat, 64),
                              dplane(ctx, &g, &oc, 1, kDLFloat, 64), dplane(ctx, &g, &odt, 1, kDLFloat, 64) };
    const int32_t sslot[2] = { 2, 1 }, gslot[5] = { 2, 1, 0, 6, 7 };
    const struct eg_bind_desc sd = binds(2, sslot, sp, 0, NULL, NULL), gd = binds(5, gslot, gp, 0, NULL, NULL);
    const int base = seed_host ? 0 : 2;
    if (seed_host) {
        snprintf(what, sizeof(what), "%s: host seed planes for a device reverse refused BAD_REQUEST by name", what0);
        check(ok && sp[0] && sp[1] && eg_reverse(ctx, r, &sd, &gd) == EG_BAD_REQUEST && reason_has(ctx, "a device reverse takes device planes"), what);
        for (int i = 0; i < 2; ++i)
            if (sp[i]) eg_plane_free(ctx, sp[i]);
        dfree(ctx, &g);
        return;
    }
    const int rok = ok && eg_reverse(ctx, r, &sd, &gd) == EG_OK && dget(&g, base, ox, sizeof(ox)) && dget(&g, base + 1, ov, sizeof(ov)) &&
                    dget(&g, base + 2, ot, sizeof(ot)) && dget(&g, base + 3, &oc, sizeof(oc)) && dget(&g, base + 4, &odt, sizeof(odt));
    int g_ok = rok && memcmp(&oc, &gc, sizeof(gc)) == 0 && memcmp(&odt, &gdt, sizeof(gdt)) == 0;
    for (int i = 0; g_ok && i < N; ++i) g_ok = memcmp(&ox[i], &gx[i], 8) == 0 && memcmp(&ov[i], &gv[i], 8) == 0 && memcmp(&ot[i], &gt[i], 8) == 0;
    snprintf(what, sizeof(what), "%s: eg_reverse's gradients (x, v, t per sample; c, dt summed) equal the old device path's "
             "bitwise (c %.17g, dt %.17g)", what0, oc, odt);
    check(g_ok, what);
    dfree(ctx, &g);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <library> <oracle dir> <primal digest> <twin digest> <vjp digest> <primal name> <window>\n", argv[0]);
        return 2;
    }
    const char *lib = argv[1], *dir = argv[2], *base = argv[6];
    const long window = atol(argv[7]);
    uint8_t dg[3][EG_DIGEST_BYTES];
    int64_t steps[N];
    if (!parse_digest(argv[3], dg[0]) || !parse_digest(argv[4], dg[1]) || !parse_digest(argv[5], dg[2]) ||
        !read_file(dir, "x0.bin", x0, sizeof(x0)) || !read_file(dir, "v0.bin", v0, sizeof(v0)) || !read_file(dir, "tf.bin", tf, sizeof(tf)) ||
        !read_file(dir, "end_x.bin", ex, sizeof(ex)) || !read_file(dir, "end_v.bin", ev, sizeof(ev)) || !read_file(dir, "end_t.bin", et, sizeof(et)) ||
        !read_file(dir, "grad_x.bin", gx, sizeof(gx)) || !read_file(dir, "grad_v.bin", gv, sizeof(gv)) || !read_file(dir, "grad_t.bin", gt, sizeof(gt)) ||
        !read_file(dir, "grad_c.bin", &gc, sizeof(gc)) || !read_file(dir, "grad_dt.bin", &gdt, sizeof(gdt)) ||
        !read_file(dir, "steps.bin", steps, sizeof(steps)) || !read_file(dir, "run.bin", run_ref, sizeof(run_ref))) {
        fprintf(stderr, "digests or oracle files missing in %s\n", dir);
        return 2;
    }
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.flags = EG_CTX_DEVICE;
    st.device = 0;
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    char tn[64], vn[64], what[200];
    snprintf(tn, sizeof(tn), "%s_tape", base);
    snprintf(vn, sizeof(vn), "%s_step_vjp", base);
    eg_unit* p = load(ctx, lib, base, dg[0]);
    eg_unit* units[2] = { load(ctx, lib, tn, dg[1]), load(ctx, lib, vn, dg[2]) };
    snprintf(what, sizeof(what), "%s: the primal, its taping twin and its step vjp load (PTX images) on the device context", base);
    check(p && units[0] && units[1], what);
    if (p && units[0] && units[1]) {
        static const double zeros[N];
        static const uint8_t none[N];
        struct dplanes s = { .k = 0 };
        eg_plane* const planes[5] = { dplane(ctx, &s, zeros, N, kDLFloat, 64), dplane(ctx, &s, v0, N, kDLFloat, 64),
                                      dplane(ctx, &s, x0, N, kDLFloat, 64), dplane(ctx, &s, tf, N, kDLFloat, 64),
                                      dplane(ctx, &s, none, N, kDLBool, 8) };
        const int32_t slots[5] = { 0, 1, 2, 3, 5 }, us[2] = { 6, 7 };
        const double c = 0.3, dt = 0.01;
        const void* const uv[2] = { &c, &dt };
        const struct eg_bind_desc bd = binds(5, slots, planes, 2, us, uv);
        struct eg_plan_desc pd;
        memset(&pd, 0, sizeof(pd));
        pd.struct_size = sizeof(pd);
        pd.unit = p;
        pd.structure = EG_STRUCTURE_DEVICE;
        pd.policy = EG_REDUCE_COMPENSATED;
        pd.n_samples = N;
        eg_plan* plan = NULL;
        const eg_derived_role roles[3] = { EG_DERIVED_TAPE, EG_DERIVED_STEP_VJP, EG_DERIVED_FUSED_VJP };
        eg_unit* const derived[3] = { units[0], units[1], units[1] };
        struct eg_runner_desc d = runner_desc(&plan, STEPS);
        d.flags = EG_RUN_RECORD;
        d.n_derived = 2;
        d.derived_roles = roles;
        d.derived_units = derived;
        d.window = window;
        eg_runner* r = NULL;
        const int ok = planes[4] && eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &bd) == EG_OK && eg_runner_new(ctx, &d, &r) == EG_OK;
        snprintf(what, sizeof(what), "%s: eg_runner_new with EG_RUN_RECORD over a DEVICE plan (window %ld)", base, window);
        check(ok, what);
        if (ok) {
            for (int k = 0; k < 2; ++k) {
                snprintf(what, sizeof(what), "%s device window %ld%s", base, window, k ? ", recorded again on the same runner" : "");
                record_and_reverse(ctx, r, &s, what, 0);
            }
            snprintf(what, sizeof(what), "%s device window %ld", base, window);
            record_and_reverse(ctx, r, &s, what, 1);
            eg_runner* none_r = NULL;
            struct eg_runner_desc bad = d;
            const eg_derived_role fused_roles[2] = { EG_DERIVED_TAPE_FUSED, EG_DERIVED_STEP_VJP };
            bad.derived_roles = fused_roles;
            check(eg_runner_new(ctx, &bad, &none_r) == EG_NOT_WRITTEN && reason_has(ctx, "K-step twin") && reason_has(ctx, "on a device") && none_r == NULL,
                  "device record: the K-step twin (EG_DERIVED_TAPE_FUSED) refused NOT_WRITTEN by name");
            bad = d;
            bad.steps_reverse = EG_STEPS_REVERSE_FUSED;
            bad.n_derived = 3;
            check(eg_runner_new(ctx, &bad, &none_r) == EG_NOT_WRITTEN && reason_has(ctx, "steps_reverse FUSED on a device") && none_r == NULL,
                  "device record: steps_reverse FUSED refused NOT_WRITTEN by name");
            check(eg_runner_free(ctx, r) == EG_OK, "eg_runner_free of the recorded device runner (a record left live)");
        }
        if (plan) eg_plan_free(ctx, plan);
        dfree(ctx, &s);
    }
    for (int i = 0; i < 2; ++i)
        if (units[i]) eg_unit_free(ctx, units[i]);
    if (p) eg_unit_free(ctx, p);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
