/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The split-unit pair proof (host): a source unit with its stage units, loaded by name + digest (stages first, passed as
 * deps), planned under HOST_TEAM and HOST_SERIAL, launched twice, every output plane compared BITWISE with the old path's.
 *
 *   split_pair_test <library> <oracle dir> <row> <source name> <source digest> [<stage name> <stage digest>]...
 *   split_pair_test <library> -refuse:<NOT_WRITTEN|BAD_REQUEST>:<reason text> <row> <source name> <source digest> [<stage>...]
 *     (the stages load, the source's load must be refused with that status and a reason holding that text)
 *
 * <oracle dir>/<row>.spec lists the source slots in order, one per line:
 *   n <lanes>
 *   in  <name> <f64|f32|i32|u32|b8> <ndim> <extents...>      (file <row>.<name>.in: the old path's input, extents without the lane axis)
 *   out <name> <f64|f32|i32> <ndim> <extents...>      (starts zeroed; file <row>.<name>.out: the old path's result)
 *   ink  <name> <f64|f32|i32> <count>                 (a plane of <count> elements with no lane axis: a table; file <row>.<name>.in)
 *   outk <name> <f64|f32|i32> <count>                 (the same, an output: a block sum's keys; file <row>.<name>.out)
 *   uni <name> <value>                            (a float64 uniform)
 *   cell <name>                                   (a cell the caller receives: one float64, the fold of the stages' rows; file <row>.<name>.out)
 *   skip <name>                                   (a slot eagle makes: the stage's own scratch; takes a slot index)
 * Every plane is (extents..., lanes).
 *
 * Built with -DEG_PAIR_DEVICE (split_device_test, run under the GPU pool) the units are device units on a device context: the planes are
 * copied to the device, the plan is DEVICE, and the outputs are copied back. -refuse:<STATUS>:<text>|<N> then also plans N lanes and
 * expects that refusal (the scratch byte sum).
 */
#define _POSIX_C_SOURCE 200809L

#include "pair_util.h"

#include <stdlib.h>

#ifdef EG_PAIR_DEVICE
#include <cuda_runtime_api.h>
#endif

#define MAXS 16
#define MAXD 3

struct pl {
    char kind[8], name[48];
    int is_int, is_f32, ndim;   /* is_f32: a float32 plane (4 bytes) */
    int64_t dims[MAXD + 1];
    int64_t count;
    void* buf;
    void* want;
    double uni;
};

/* the element types a plane may have: 0 f64, 1 i32, 2 u32, 3 b8 (a bool byte) */
static int ty_of(const char* ty) { return strcmp(ty, "i32") == 0 ? 1 : strcmp(ty, "u32") == 0 ? 2 : strcmp(ty, "b8") == 0 ? 3 : 0; }
static size_t esz(const struct pl* q) { return q->is_f32 ? 4 : q->is_int == 0 ? 8 : q->is_int == 3 ? 1 : 4; }
static int is_out(const struct pl* q) { return strcmp(q->kind, "out") == 0 || strcmp(q->kind, "outk") == 0 || strcmp(q->kind, "cell") == 0; }
static int is_flat(const struct pl* q) { return strcmp(q->kind, "ink") == 0 || strcmp(q->kind, "outk") == 0; }

struct pt {
    struct DLManagedTensorVersioned managed;
    int64_t shape[MAXD + 1];
};

static int read_spec(const char* dir, const char* row, int64_t* n, struct pl* p, int* np) {
    char path[1024], kind[8];
    snprintf(path, sizeof(path), "%s/%s.spec", dir, row);
    FILE* f = fopen(path, "r");
    if (f == NULL) return 0;
    *np = 0;
    while (fscanf(f, "%7s", kind) == 1) {
        if (strcmp(kind, "n") == 0) {
            long long v;
            if (fscanf(f, "%lld", &v) != 1) return 0;
            *n = v;
            continue;
        }
        struct pl* q = &p[(*np)++];
        char ty[8];
        strcpy(q->kind, kind);
        if (strcmp(kind, "skip") == 0) {
            if (fscanf(f, "%47s", q->name) != 1) return 0;
            continue;
        }
        if (strcmp(kind, "cell") == 0) {
            if (fscanf(f, "%47s", q->name) != 1) return 0;
            q->ndim = 0;
            continue;
        }
        if (strcmp(kind, "uni") == 0) {
            if (fscanf(f, "%47s %lf", q->name, &q->uni) != 2) return 0;
            continue;
        }
        if (strcmp(kind, "ink") == 0 || strcmp(kind, "outk") == 0) {
            long long v;
            if (fscanf(f, "%47s %7s %lld", q->name, ty, &v) != 3) return 0;
            q->is_int = ty_of(ty);
            q->is_f32 = strcmp(ty, "f32") == 0;
            q->ndim = 0;
            q->dims[0] = v;
            continue;
        }
        if (fscanf(f, "%47s %7s %d", q->name, ty, &q->ndim) != 3) return 0;
        q->is_int = ty_of(ty);
        q->is_f32 = strcmp(ty, "f32") == 0;
        for (int i = 0; i < q->ndim; ++i) {
            long long v;
            if (fscanf(f, "%lld", &v) != 1) return 0;
            q->dims[i] = v;
        }
    }
    fclose(f);
    return 1;
}

static void load(eg_ctx* ctx, const char* lib, const char* name, const char* hex, eg_unit* const* deps, int n_deps, eg_unit** out, const char* what) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = lib;
    d.unit = &ref;
    d.deps = deps;
    d.n_deps = n_deps;
    const eg_status s = parse_digest(hex, ref.digest) ? eg_unit_load(ctx, &d, out) : EG_BAD_REQUEST;
    check(s == EG_OK && *out != NULL, what);
    if (s != EG_OK) reason_has(ctx, "\001");
}

static int refused(int argc, char** argv) {
    const char* spec = argv[2] + 8;
    const char* colon = strchr(spec, ':');
    if (colon == NULL) return 2;
    const eg_status want = strncmp(spec, "NOT_WRITTEN", (size_t)(colon - spec)) == 0 ? EG_NOT_WRITTEN : EG_BAD_REQUEST;
    const char* bar = strchr(colon, '|');   /* text|N: the source loads and the PLAN of N lanes is refused */
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
#ifdef EG_PAIR_DEVICE
    st.flags = EG_CTX_DEVICE;
#endif
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new host context");
    const int ns = (argc - 6) / 2;
    eg_unit* deps[MAXS] = { NULL };
    for (int i = 0; i < ns; ++i) load(ctx, argv[1], argv[6 + 2 * i], argv[7 + 2 * i], NULL, 0, &deps[i], "a stage unit loaded (name + digest)");
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = argv[4];
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = argv[1];
    d.unit = &ref;
    d.deps = deps;
    d.n_deps = ns;
    eg_unit* src = NULL;
    eg_status s = parse_digest(argv[5], ref.digest) ? eg_unit_load(ctx, &d, &src) : EG_FAULT;
    if (bar != NULL) {
        check(s == EG_OK && src != NULL, "the split source unit loaded");
        eg_plan* plan = NULL;
        struct eg_plan_desc pd;
        memset(&pd, 0, sizeof(pd));
        pd.struct_size = sizeof(pd);
        pd.unit = src;
#ifdef EG_PAIR_DEVICE
        pd.structure = EG_STRUCTURE_DEVICE;
#else
        pd.structure = EG_STRUCTURE_HOST_TEAM;
#endif
        pd.n_samples = atoll(bar + 1);
        s = eg_plan_new(ctx, &pd, &plan);
        if (s == EG_OK) eg_plan_free(ctx, plan);
        eg_unit_free(ctx, src);
        src = NULL;
    }
    char buf[1024];
    size_t len = 0;
    buf[0] = 0;
    if (eg_error(ctx, buf, sizeof(buf), &len) != EG_OK) buf[0] = 0;
    const char* text_end = bar != NULL ? bar : colon + strlen(colon);
    char need[256];
    snprintf(need, sizeof(need), "%.*s", (int)(text_end - (colon + 1)), colon + 1);
    printf("      reason: %s\n", buf);
    char what[420];
    snprintf(what, sizeof(what), "the split unit's %s refused (%.*s) with a reason holding '%s'", bar != NULL ? "plan" : "load", (int)(colon - spec), spec, need);
    check(s == want && src == NULL && strstr(buf, need) != NULL, what);
    for (int i = 0; i < ns; ++i) eg_unit_free(ctx, deps[i]);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc < 6 || (argc - 6) % 2 != 0) {
        fprintf(stderr, "usage: %s <library> <oracle dir> <row> <source name> <source digest> [<stage name> <stage digest>]...\n", argv[0]);
        return 2;
    }
    if (strncmp(argv[2], "-refuse:", 8) == 0) return refused(argc, argv);
    int64_t n = 0;
    static struct pl pls[MAXS];
    int np = 0;
    check(read_spec(argv[2], argv[3], &n, pls, &np), "the row's spec read");
    if (failures) return 1;
    for (int i = 0; i < np; ++i) {
        struct pl* q = &pls[i];
        if (strcmp(q->kind, "uni") == 0 || strcmp(q->kind, "skip") == 0) continue;
        q->count = strcmp(q->kind, "cell") == 0 ? 1 : is_flat(q) ? q->dims[0] : n;
        if (strcmp(q->kind, "cell") != 0 && !is_flat(q))
            for (int d = 0; d < q->ndim; ++d) q->count *= q->dims[d];
        const size_t bytes = (size_t)q->count * esz(q);
        q->buf = calloc(1, bytes);
        q->want = calloc(1, bytes);
        char file[160];
        if (strcmp(q->kind, "in") == 0 || strcmp(q->kind, "ink") == 0) {
            snprintf(file, sizeof(file), "%s.%s.in", argv[3], q->name);
            check(read_file(argv[2], file, q->buf, bytes), "an input read");
        } else {
            snprintf(file, sizeof(file), "%s.%s.out", argv[3], q->name);
            check(read_file(argv[2], file, q->want, bytes), "an output read");
        }
    }
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
#ifdef EG_PAIR_DEVICE
    st.flags = EG_CTX_DEVICE;
#endif
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new context");
    const int ns = (argc - 6) / 2;
    eg_unit* deps[MAXS] = { NULL };
    for (int i = 0; i < ns; ++i) load(ctx, argv[1], argv[6 + 2 * i], argv[7 + 2 * i], NULL, 0, &deps[i], "a stage unit loaded (name + digest)");
    eg_unit* src = NULL;
    load(ctx, argv[1], argv[4], argv[5], deps, ns, &src, "the split source unit loaded with its stages as deps");
    if (src == NULL || failures) return 1;
    eg_host_threads(ctx, 3);
#ifdef EG_PAIR_DEVICE
    const eg_structure structures[1] = { EG_STRUCTURE_DEVICE };
    const char* names[1] = { "DEVICE" };
    const int nstruct = 1;
#else
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_TEAM, EG_STRUCTURE_HOST_SERIAL };
    const char* names[2] = { "HOST_TEAM", "HOST_SERIAL" };
    const int nstruct = 2;
#endif
    for (int k = 0; k < nstruct; ++k) {
        eg_plan* plan = NULL;
        struct eg_plan_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.unit = src;
        d.structure = structures[k];
        d.n_samples = n;
        eg_status s = eg_plan_new(ctx, &d, &plan);
        struct pt t[MAXS];
        eg_plane* planes[MAXS] = { NULL };
        int32_t slots[MAXS], uslots[MAXS];
        double uvals[MAXS];
        const void* uptr[MAXS];
        int nplanes = 0, nuni = 0, slot = 0;
#ifdef EG_PAIR_DEVICE
        void* dev[MAXS] = { NULL };
#endif
        for (int i = 0; i < np; ++i) {
            struct pl* q = &pls[i];
            if (strcmp(q->kind, "skip") == 0) {
                ++slot;
                continue;
            }
            if (strcmp(q->kind, "uni") == 0) {
                uvals[nuni] = q->uni;
                uptr[nuni] = &uvals[nuni];
                uslots[nuni++] = slot++;
                continue;
            }
            if (is_out(q)) memset(q->buf, 0, (size_t)q->count * esz(q));
            memset(&t[i], 0, sizeof(t[i]));
            for (int d2 = 0; d2 < q->ndim; ++d2) t[i].shape[d2] = q->dims[d2];
            if (strcmp(q->kind, "cell") == 0) { t[i].shape[0] = 1; } else if (is_flat(q)) { t[i].shape[0] = q->dims[0]; } else t[i].shape[q->ndim] = n;
            t[i].managed.version.major = DLPACK_MAJOR_VERSION;
            t[i].managed.dl_tensor.data = q->buf;
            t[i].managed.dl_tensor.device.device_type = kDLCPU;
#ifdef EG_PAIR_DEVICE
            const size_t pbytes = (size_t)q->count * esz(q);
            if (cudaMalloc(&dev[i], pbytes) != cudaSuccess || cudaMemcpy(dev[i], q->buf, pbytes, cudaMemcpyHostToDevice) != cudaSuccess) s = EG_FAULT;
            t[i].managed.dl_tensor.data = dev[i];
            t[i].managed.dl_tensor.device.device_type = kDLCUDA;
#endif
            t[i].managed.dl_tensor.ndim = strcmp(q->kind, "cell") == 0 || is_flat(q) ? 1 : q->ndim + 1;
            t[i].managed.dl_tensor.dtype.code = q->is_int == 0 ? kDLFloat : q->is_int == 1 ? kDLInt : q->is_int == 2 ? kDLUInt : kDLBool;
            t[i].managed.dl_tensor.dtype.bits = (uint8_t)(esz(q) * 8);
            t[i].managed.dl_tensor.dtype.lanes = 1;
            t[i].managed.dl_tensor.shape = t[i].shape;
            if (s == EG_OK) s = eg_plane_wrap(ctx, &t[i].managed, &planes[nplanes]);
            slots[nplanes++] = slot++;
        }
        struct eg_bind_desc b;
        memset(&b, 0, sizeof(b));
        b.struct_size = sizeof(b);
        b.n_planes = nplanes;
        b.plane_slots = slots;
        b.planes = planes;
        b.n_uniforms = nuni;
        b.uniform_slots = uslots;
        b.uniform_values = uptr;
#ifdef EG_PAIR_DEVICE
        if (cudaDeviceSynchronize() != cudaSuccess) s = EG_FAULT;   /* the copies land before the context's stream reads them */
#endif
        if (s == EG_OK) s = eg_bind(ctx, plan, &b);
        for (int rep = 0; rep < 2; ++rep) {
#ifndef EG_PAIR_DEVICE
            if (rep > 0)    /* a plane the caller binds as a counter accumulates over launches: every launch starts from the zeroed output */
                for (int i = 0; i < np; ++i)
                    if (is_out(&pls[i])) memset(pls[i].buf, 0, (size_t)pls[i].count * esz(&pls[i]));
#endif
            if (s == EG_OK) s = eg_launch(ctx, plan, NULL);
#ifdef EG_PAIR_DEVICE
            if (s == EG_OK) s = eg_sync(ctx);
            for (int i = 0; i < np && s == EG_OK; ++i)
                if (is_out(&pls[i]) &&
                    cudaMemcpy(pls[i].buf, dev[i], (size_t)pls[i].count * esz(&pls[i]), cudaMemcpyDeviceToHost) != cudaSuccess) s = EG_FAULT;
#endif
            if (s != EG_OK) reason_has(ctx, "\001");
            int bad = 0;
            for (int i = 0; i < np; ++i) {
                struct pl* q = &pls[i];
                if (!is_out(q)) continue;
                const size_t bytes = (size_t)q->count * esz(q);
                if (memcmp(q->buf, q->want, bytes) != 0) {
                    ++bad;
                    int64_t at = 0;
                    const size_t w = esz(q);
                    while (memcmp((char*)q->buf + (size_t)at * w, (char*)q->want + (size_t)at * w, w) == 0) ++at;
                    if (q->is_int == 1) printf("      output '%s' differs from the old path at element %lld: %d, old path %d\n", q->name, (long long)at, ((int32_t*)q->buf)[at], ((int32_t*)q->want)[at]);
                    else if (q->is_int) printf("      output '%s' differs from the old path at element %lld\n", q->name, (long long)at);
                    else if (q->is_f32) printf("      output '%s' differs from the old path at element %lld: %.9g, old path %.9g\n", q->name, (long long)at, (double)((float*)q->buf)[at], (double)((float*)q->want)[at]);
                    else printf("      output '%s' differs from the old path at element %lld: %.17g, old path %.17g\n", q->name, (long long)at, ((double*)q->buf)[at], ((double*)q->want)[at]);
                }
            }
            char what[200];
            snprintf(what, sizeof(what), "%s, launch %d: OK and %d output plane(s) bitwise == the old path", names[k], rep + 1, nplanes);
            check(s == EG_OK && bad == 0, what);
        }
        eg_plan_free(ctx, plan);
        for (int i = 0; i < nplanes; ++i) eg_plane_free(ctx, planes[i]);
#ifdef EG_PAIR_DEVICE
        for (int i = 0; i < np; ++i) cudaFree(dev[i]);
#endif
    }
    check(eg_unit_free(ctx, src) == EG_OK, "eg_unit_free of the source");
    for (int i = 0; i < ns; ++i) check(eg_unit_free(ctx, deps[i]) == EG_OK, "eg_unit_free of a stage");
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
