/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The door's first proof, in C11: load the fixture unit library through the
 * door, run y = a*x + b under HOST_TEAM whole, HOST_TEAM over two partitions
 * and HOST_SERIAL, and require every result BITWISE equal to a*x + b computed
 * here and to each other; then the refusals. Prints PASS/FAIL per check and
 * exits non-zero on any failure.
 *
 * Usage: door_test <fixture> <fixture: wrong size> <fixture: no size block> <fixture: v1 tag>
 */
#define _POSIX_C_SOURCE 200809L
/* macOS hides dlopen's RTLD_NOLOAD under a strict POSIX request unless its own extensions are asked for too */
#define _DARWIN_C_SOURCE

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"
#include "seg_model.h"

#ifndef EG_TEST_DIR
#define EG_TEST_DIR "test" /* the directory of the recorded decisions; the Makefile gives its absolute path */
#endif

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int failures = 0;
/* A build fact, read once from the door (main): the library carries the device group, so it answers the device calls and
 * refuses a device context for want of a device, where a door without it refuses them NOT_WRITTEN. */
static int device_group = 0;
static const char* library_path = NULL;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

/* ---- DLPack tensors over test-owned buffers ------------------------------ */

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[3];
    int deleted; /* how many times the deleter ran */
};

static void count_delete(struct DLManagedTensorVersioned* self) {
    ((struct tensor*)self->manager_ctx)->deleted += 1;
}

static void tensor_init(struct tensor* t, void* data, int64_t n, uint8_t bits) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = n;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.version.minor = DLPACK_MINOR_VERSION;
    t->managed.manager_ctx = t;
    t->managed.deleter = count_delete;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCPU;
    t->managed.dl_tensor.ndim = 1;
    t->managed.dl_tensor.dtype.code = kDLFloat;
    t->managed.dl_tensor.dtype.bits = bits;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
}

/* ---- door helpers -------------------------------------------------------- */

static eg_ctx* make_ctx(void) {
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    return eg_ctx_new(&s, &ctx) == EG_OK ? ctx : NULL;
}

static int reason_has(eg_ctx* ctx, const char* needle);

static eg_status load(eg_ctx* ctx, const char* name, uint8_t digest_byte, eg_unit** out) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memset(ref.digest, digest_byte, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = library_path;
    d.unit = &ref;
    return eg_unit_load(ctx, &d, out);
}

static eg_status plan(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t n,
                      int64_t n_parts, const int64_t* base, const int64_t* count, eg_plan** out) {
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = structure;
    d.n_samples = n;
    d.n_partitions = n_parts;
    d.part_base = base;
    d.part_count = count;
    return eg_plan_new(ctx, &d, out);
}

/* Bind y (slot 0) and x (slot 1) as planes, a (slot 2) and b (slot 3) as uniforms. */
static eg_status bind(eg_ctx* ctx, eg_plan* p, eg_plane* y, eg_plane* x, const double* a,
                      const double* b) {
    const int32_t plane_slots[2] = { 0, 1 };
    eg_plane* const planes[2] = { y, x };
    const int32_t uniform_slots[2] = { 2, 3 };
    const void* const values[2] = { a, b };
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = 2;
    d.n_uniforms = 2;
    d.plane_slots = plane_slots;
    d.planes = planes;
    d.uniform_slots = uniform_slots;
    d.uniform_values = values;
    return eg_bind(ctx, p, &d);
}

/* Run the unit under one plan shape into y; returns the launch status. */
static eg_status run_into(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t n,
                          int64_t n_parts, const int64_t* base, const int64_t* count,
                          double* x, double* y, struct tensor* tx, struct tensor* ty) {
    const double a = 2.0, b = -1.5;
    eg_plane *px = NULL, *py = NULL;
    eg_plan* p = NULL;
    tensor_init(tx, x, n, 64);
    tensor_init(ty, y, n, 64);
    eg_status s = eg_plane_wrap(ctx, &tx->managed, &px);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &ty->managed, &py);
    if (s == EG_OK) s = plan(ctx, unit, structure, n, n_parts, base, count, &p);
    if (s == EG_OK) s = bind(ctx, p, py, px, &a, &b);
    uint64_t tiles = 0;
    if (s == EG_OK) s = eg_launch(ctx, p, &tiles);
    /* the caller drops its references first: the plan still holds the planes */
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, py);
    if (tx->deleted != 0 || ty->deleted != 0) s = EG_FAULT;
    eg_plan_free(ctx, p);
    return s;
}

/* ---- the oracle ---------------------------------------------------------- */

static void oracle(eg_ctx* ctx, eg_unit* unit, int64_t n, int64_t split, const char* label) {
    double* x = malloc(sizeof(double) * (size_t)n);
    double* want = malloc(sizeof(double) * (size_t)n);
    double* whole = calloc((size_t)n, sizeof(double));
    double* parted = calloc((size_t)n, sizeof(double));
    double* serial = calloc((size_t)n, sizeof(double));
    const double a = 2.0, b = -1.5;
    for (int64_t i = 0; i < n; ++i) {
        x[i] = (double)((i * 7919) % 1000) / 37.0 - 13.0;
        want[i] = a * x[i] + b;
    }
    const int64_t base[2] = { 0, split }, count[2] = { split, n - split };
    struct tensor tx[3], ty[3];
    char what[160];
    snprintf(what, sizeof(what), "%s: HOST_TEAM whole launches", label);
    check(run_into(ctx, unit, EG_STRUCTURE_HOST_TEAM, n, 0, NULL, NULL, x, whole, &tx[0], &ty[0]) == EG_OK, what);
    snprintf(what, sizeof(what), "%s: HOST_TEAM partitions (0,%lld),(%lld,%lld) launch", label,
             (long long)split, (long long)split, (long long)(n - split));
    check(run_into(ctx, unit, EG_STRUCTURE_HOST_TEAM, n, 2, base, count, x, parted, &tx[1], &ty[1]) == EG_OK, what);
    snprintf(what, sizeof(what), "%s: HOST_SERIAL launches", label);
    check(run_into(ctx, unit, EG_STRUCTURE_HOST_SERIAL, n, 0, NULL, NULL, x, serial, &tx[2], &ty[2]) == EG_OK, what);
    const size_t bytes = sizeof(double) * (size_t)n;
    snprintf(what, sizeof(what), "%s: whole bitwise == a*x+b", label);
    check(memcmp(whole, want, bytes) == 0, what);
    snprintf(what, sizeof(what), "%s: whole bitwise == partitioned", label);
    check(memcmp(whole, parted, bytes) == 0, what);
    snprintf(what, sizeof(what), "%s: whole bitwise == HOST_SERIAL", label);
    check(memcmp(whole, serial, bytes) == 0, what);
    int once = 1;
    for (int i = 0; i < 3; ++i) once = once && tx[i].deleted == 1 && ty[i].deleted == 1;
    snprintf(what, sizeof(what), "%s: each deleter ran exactly once, at plan free", label);
    check(once, what);
    free(x), free(want), free(whole), free(parted), free(serial);
}

/* ---- refusals ------------------------------------------------------------ */

static void check_error_protocol(eg_ctx* ctx) {
    eg_unit* u = NULL;
    check(load(ctx, "no_such_unit", 0x00, &u) == EG_BAD_REQUEST && u == NULL,
          "unknown unit name refused BAD_REQUEST");
    size_t len = 0;
    check(eg_error(ctx, NULL, 0, &len) == EG_OK && len > 0, "eg_error call 1 gives the length");
    char* buf = malloc(len + 1);
    size_t len2 = 0;
    check(eg_error(ctx, buf, len, &len2) == EG_BAD_REQUEST && len2 == len,
          "eg_error with cap == len refused BAD_REQUEST");
    check(eg_error(ctx, buf, len + 1, &len2) == EG_OK && strlen(buf) == len &&
              strstr(buf, "no_such_unit") != NULL,
          "eg_error call 2 writes the reason (kept across the refused read)");
    free(buf);
}

/* Settings: zero-filled is host only; EG_CTX_DEVICE is not built; a device without the flag is refused. */
static eg_status settings_status(uint32_t size, uint32_t flags, int32_t device) {
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = size;
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    s.flags = flags;
    s.device = device;
    eg_ctx* c = NULL;
    const eg_status st = eg_ctx_new(&s, &c);
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    if (st == EG_OK && (eg_ctx_info(c, &info) != EG_OK || info.device != -1)) {
        eg_ctx_free(c);
        return EG_FAULT;
    }
    eg_ctx_free(c);
    return st;
}

static void check_settings(void) {
    const uint32_t short_size = (uint32_t)(offsetof(struct eg_ctx_settings, abi_major) + sizeof(uint32_t));
    check(settings_status(short_size, 0, 0) == EG_OK,
          "settings with struct_size + abi_major only: a host-only context (info.device -1)");
    if (!device_group)    /* with the group, a device context is made or refused for want of a device: check_device_count */
        check(settings_status(sizeof(struct eg_ctx_settings), EG_CTX_DEVICE, 0) == EG_NOT_WRITTEN,
              "EG_CTX_DEVICE refused NOT_WRITTEN (device group not built)");
    check(settings_status(sizeof(struct eg_ctx_settings), 0, 1) == EG_BAD_REQUEST,
          "device != 0 without EG_CTX_DEVICE refused BAD_REQUEST");
}

static void check_bad_input(eg_ctx* ctx, eg_unit* unit) {
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    s.flags = 1u << 7;
    eg_ctx* bad = NULL;
    const eg_status st = eg_ctx_new(&s, &bad);
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    check(st == EG_BAD_REQUEST && bad != NULL && eg_ctx_info(bad, &info) == EG_BAD_REQUEST,
          "unknown settings flag refused BAD_REQUEST; the refused context keeps refusing");
    eg_ctx_free(bad);

    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = 1u << 5;
    d.unit = unit;
    d.n_samples = 4;
    eg_plan* p = NULL;
    check(eg_plan_new(ctx, &d, &p) == EG_BAD_REQUEST && p == NULL, "unknown plan flag refused BAD_REQUEST");

    const int64_t base[2] = { 0, 2 }, count[2] = { 3, 2 };
    check(plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, 4, 2, base, count, &p) == EG_BAD_REQUEST,
          "overlapping partitions refused BAD_REQUEST");
    check(plan(ctx, unit, EG_STRUCTURE_DEVICE, 4, 0, NULL, NULL, &p) == (device_group ? EG_BAD_REQUEST : EG_NOT_WRITTEN),
          "DEVICE structure on a host context refused: NOT_WRITTEN without the device group, BAD_REQUEST (host-only context) with it");

    double x32[4] = { 0 }, y[4] = { 0 };
    struct tensor tx, ty;
    tensor_init(&tx, x32, 4, 32);
    tensor_init(&ty, y, 4, 64);
    eg_plane *px = NULL, *py = NULL;
    const double a = 1.0, b = 0.0;
    eg_status s1 = eg_plane_wrap(ctx, &tx.managed, &px);
    eg_status s2 = eg_plane_wrap(ctx, &ty.managed, &py);
    eg_status s3 = plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, 4, 0, NULL, NULL, &p);
    check(s1 == EG_OK && s2 == EG_OK && s3 == EG_OK && bind(ctx, p, py, px, &a, &b) == EG_BAD_REQUEST,
          "float32 plane bound to a float64 slot refused BAD_REQUEST");
    uint64_t tiles = 0;
    check(eg_launch(ctx, p, &tiles) == EG_BAD_REQUEST, "launch with unbound slots refused BAD_REQUEST");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, py);
    check(tx.deleted == 1 && ty.deleted == 1, "refused bind takes no reference: deleters ran once");

    eg_unit* split = NULL;
    check(load(ctx, "axpb_split", 0xC3, &split) == EG_BAD_REQUEST && split == NULL && reason_has(ctx, "is not in deps"),
          "split unit with its stage missing from deps refused BAD_REQUEST naming it");
    {
        eg_unit* stage = NULL;
        struct eg_unit_ref ref;
        memset(&ref, 0, sizeof(ref));
        ref.struct_size = sizeof(ref);
        ref.name = "axpb_split";
        memset(ref.digest, 0xC3, EG_DIGEST_BYTES);
        struct eg_unit_load_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.path = library_path;
        d.unit = &ref;
        d.deps = &stage;
        d.n_deps = 1;
        check(load(ctx, "axpb", 0xA1, &stage) == EG_OK && eg_unit_load(ctx, &d, &split) == EG_OK && split != NULL,
              "split unit loaded with its stage in deps");
        check(eg_unit_free(ctx, split) == EG_OK && eg_unit_free(ctx, stage) == EG_OK, "split unit and its stage freed");
        split = NULL;
    }
    eg_unit* wrong = NULL;
    check(load(ctx, "axpb", 0xC3, &wrong) == EG_BAD_REQUEST, "right name, wrong digest refused BAD_REQUEST");
    check(eg_run(ctx, NULL, NULL) == EG_BAD_REQUEST, "eg_run of a NULL runner refused BAD_REQUEST");
    check(eg_host_threads(ctx, -1) == EG_BAD_REQUEST, "eg_host_threads(-1) refused BAD_REQUEST");
}

/* A plane of another context is refused at bind. */
static void check_foreign_handle(eg_ctx* ctx, eg_unit* unit) {
    eg_ctx* other = make_ctx();
    double x[4] = { 0 }, y[4] = { 0 };
    struct tensor tx, ty;
    tensor_init(&tx, x, 4, 64);
    tensor_init(&ty, y, 4, 64);
    eg_plane *px = NULL, *py = NULL;
    eg_plan* p = NULL;
    const double a = 1.0, b = 0.0;
    eg_plane_wrap(other, &tx.managed, &px);
    eg_plane_wrap(ctx, &ty.managed, &py);
    plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, 4, 0, NULL, NULL, &p);
    check(bind(ctx, p, py, px, &a, &b) == EG_BAD_REQUEST, "plane of another context refused BAD_REQUEST");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, py);
    eg_ctx_free(other); /* frees px: its deleter runs here */
    check(tx.deleted == 1 && ty.deleted == 1, "eg_ctx_free runs the deleter of a plane still held");
}

/* Load "axpb" from @p path; BAD_REQUEST with a reason holding @p need1 and @p need2. */
static int refused_with(eg_ctx* ctx, const char* path, const char* need1, const char* need2) {
    const char* saved = library_path;
    library_path = path;
    eg_unit* u = NULL;
    const eg_status st = load(ctx, "axpb", 0xA1, &u);
    library_path = saved;
    char buf[1024];
    size_t len = 0;
    const int read = eg_error(ctx, buf, sizeof(buf), &len) == EG_OK;
    if (st != EG_BAD_REQUEST || !read) return 0;
    printf("      reason: %s\n", buf);
    return u == NULL && strstr(buf, need1) != NULL && strstr(buf, need2) != NULL;
}

/* The exported size block beside a host entry is checked at load. */
static void check_size_block(eg_ctx* ctx, char** paths) {
    check(refused_with(ctx, paths[0], "sizeof(ScalarHandle)", "48"),
          "a wrong sizeof(ScalarHandle) in eagle_layout_sizes refused BAD_REQUEST, naming field and values");
    check(refused_with(ctx, paths[1], "eagle_abi_tag", "does not export"),
          "a library exporting no size block refused BAD_REQUEST, naming the symbol");
    check(refused_with(ctx, paths[2], "aether-abi/1", "aether-abi/2"),
          "a v1 eagle_abi_tag refused BAD_REQUEST, naming both tags");
}

/* A forked child is refused every call but eg_error and eg_ctx_free. */
static void check_fork(eg_ctx* ctx) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        struct eg_ctx_info info;
        memset(&info, 0, sizeof(info));
        info.struct_size = sizeof(info);
        int ok = eg_ctx_info(ctx, &info) == EG_BAD_REQUEST;
        size_t len = 0;
        ok = ok && eg_error(ctx, NULL, 0, &len) == EG_OK && len > 0;
        ok = ok && eg_ctx_free(ctx) == EG_OK;
        _exit(ok ? 0 : 1);
    }
    int status = -1;
    waitpid(pid, &status, 0);
    check(pid > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "forked child refused BAD_REQUEST (eg_error and eg_ctx_free still work)");
}

/* ---- runners: a finishing unit stepped until done, against a closed form -- */

#define RN 4099

static int reason_has(eg_ctx* ctx, const char* needle) {
    char buf[512];
    size_t len = 0;
    return eg_error(ctx, buf, sizeof(buf), &len) == EG_OK && strstr(buf, needle) != NULL;
}

static struct eg_runner_desc runner_desc(eg_plan* const* plans, int32_t n_plans, int64_t max_steps) {
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_plans = n_plans;
    d.plans = plans;
    d.max_steps = max_steps;
    return d;
}

/* Bind k, limit and mask (slots 0, 1, 2) of a count plan. */
static eg_status bind_count(eg_ctx* ctx, eg_plan* p, eg_plane* k, eg_plane* limit, eg_plane* mask) {
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* const planes[3] = { k, limit, mask };
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = 3;
    d.plane_slots = slots;
    d.planes = planes;
    return eg_bind(ctx, p, &d);
}

static void mask_init(struct tensor* t, unsigned char* data, int64_t n) {
    tensor_init(t, data, n, 8);
    t->managed.dl_tensor.dtype.code = kDLBool;
}

/* A count plan of `unit` over k, limit, mask; the caller's plane references are dropped. */
static eg_plan* count_plan(eg_ctx* ctx, eg_unit* unit, eg_structure structure, double* k, double* limit,
                           unsigned char* mask, struct tensor* t) {
    eg_plane *pk = NULL, *pl = NULL, *pm = NULL;
    eg_plan* p = NULL;
    tensor_init(&t[0], k, RN, 64);
    tensor_init(&t[1], limit, RN, 64);
    mask_init(&t[2], mask, RN);
    eg_status s = eg_plane_wrap(ctx, &t[0].managed, &pk);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &t[1].managed, &pl);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &t[2].managed, &pm);
    if (s == EG_OK) s = plan(ctx, unit, structure, RN, 0, NULL, NULL, &p);
    if (s == EG_OK) s = bind_count(ctx, p, pk, pl, pm);
    eg_plane_free(ctx, pk);
    eg_plane_free(ctx, pl);
    eg_plane_free(ctx, pm);
    return s == EG_OK ? p : NULL;
}

/* limit_i in 1..50; every 97th sample starts finished (its k stays 0). */
static void count_inputs(double* k, double* limit, unsigned char* mask) {
    for (int64_t i = 0; i < RN; ++i) {
        k[i] = 0.0;
        limit[i] = (double)(1 + (i * 37) % 50);
        mask[i] = i % 97 == 0;
    }
}

/* The closed form: a live sample needs ceil(L / K) launches; the loop runs the most of
 * them, capped at ceil(max_steps / K); k_i = min(L, launches * K). */
static void count_oracle(const double* limit, const unsigned char* mask0, int64_t steps, int64_t max_steps,
                         double* k, int64_t* launches, int64_t* finished) {
    const int64_t cap = (max_steps + steps - 1) / steps;
    int64_t need = 0, done = 0;
    for (int64_t i = 0; i < RN; ++i)
        if (!mask0[i] && ((int64_t)limit[i] + steps - 1) / steps > need) need = ((int64_t)limit[i] + steps - 1) / steps;
    *launches = need < cap ? need : cap;
    for (int64_t i = 0; i < RN; ++i) {
        const int64_t l = (int64_t)limit[i];
        k[i] = mask0[i] ? 0.0 : (double)(l < *launches * steps ? l : *launches * steps);
        done += mask0[i] || (l + steps - 1) / steps <= *launches;
    }
    *finished = done;
}

static void run_count(eg_ctx* ctx, eg_unit* unit, int64_t steps, eg_structure structure, int32_t threads,
                      int64_t max_steps) {
    static double k[RN], limit[RN], want_k[RN];
    static unsigned char mask[RN], mask0[RN];
    struct tensor t[3];
    char what[160];
    count_inputs(k, limit, mask);
    memcpy(mask0, mask, RN);
    int64_t want_launches = 0, want_finished = 0;
    count_oracle(limit, mask0, steps, max_steps, want_k, &want_launches, &want_finished);
    eg_host_threads(ctx, threads);
    eg_plan* p = count_plan(ctx, unit, structure, k, limit, mask, t);
    eg_runner* r = NULL;
    const struct eg_runner_desc d = runner_desc(&p, 1, max_steps);
    eg_status s = p ? eg_runner_new(ctx, &d, &r) : EG_FAULT;
    if (s == EG_OK) s = eg_plan_free(ctx, p); /* the runner keeps the plan alive */
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    if (s == EG_OK) s = eg_run(ctx, r, &rep);
    int planes_ok = 1;
    for (int64_t i = 0; i < RN; ++i)
        planes_ok &= memcmp(&k[i], &want_k[i], sizeof(double)) == 0 && (mask[i] != 0) == (want_k[i] >= limit[i] || mask0[i]);
    const int done = want_finished == RN;
    snprintf(what, sizeof(what), "eg_run K=%lld %s threads=%d max_steps=%lld: launches %lld finished %lld, "
             "steps, mapped, flags and k as the closed form", (long long)steps,
             structure == EG_STRUCTURE_HOST_SERIAL ? "HOST_SERIAL" : "HOST_TEAM", threads,
             (long long)max_steps, (long long)want_launches, (long long)want_finished);
    check(s == EG_OK && planes_ok && rep.struct_size == sizeof(rep) && rep.mode == EG_MODE_HOST &&
              rep.n == RN && rep.finished == want_finished && rep.launches == want_launches &&
              rep.steps == want_launches * steps && rep.mapped == want_launches * RN &&
              rep.compactions == 0 && rep.reorders == 0 && rep.build_s == 0.0 &&
              rep.flags == ((done ? EG_REPORT_DONE : 0u) | (steps == 1 ? EG_REPORT_EXACT_STEPS : 0u)),
          what);
    if (done && s == EG_OK) { /* a second run re-syncs the counter to the mask: nothing to launch */
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        check(eg_run(ctx, r, &rep) == EG_OK && rep.launches == 0 && rep.finished == RN,
              "a second eg_run of a finished batch launches nothing");
    }
    check(eg_runner_free(ctx, r) == EG_OK && t[0].deleted == 1 && t[2].deleted == 1,
          "eg_runner_free drops the last plan reference: the planes' deleters run once");
}

static void check_runner(eg_ctx* ctx, eg_unit* axpb) {
    eg_unit *c1 = NULL, *c3 = NULL;
    check(load(ctx, "count1", 0xD1, &c1) == EG_OK && load(ctx, "count3", 0xD3, &c3) == EG_OK,
          "eg_unit_load count1 and count3 (TERMINATED mask, eagle-made counter, finish)");
    if (c1 == NULL || c3 == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int32_t threads[2] = { 1, 3 };
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            run_count(ctx, c1, 1, structures[si], threads[ti], 1000);
            run_count(ctx, c3, 3, structures[si], threads[ti], 1000);
            run_count(ctx, c1, 1, structures[si], threads[ti], 20);
            run_count(ctx, c3, 3, structures[si], threads[ti], 20);
        }

    /* a two-plan step: the finishing plan, then y = a*x + b */
    static double k[RN], limit[RN], x[RN], y[RN], want_k[RN];
    static unsigned char mask[RN];
    struct tensor t[3], t3[3], tx, ty;
    count_inputs(k, limit, mask);
    int64_t want_launches = 0, want_finished = 0;
    count_oracle(limit, mask, 1, 1000, want_k, &want_launches, &want_finished);
    for (int64_t i = 0; i < RN; ++i) x[i] = (double)i, y[i] = 0.0;
    const double a = 2.0, b = -1.5;
    eg_plane *px = NULL, *py = NULL;
    eg_plan* plans[2] = { count_plan(ctx, c1, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t), NULL };
    tensor_init(&tx, x, RN, 64);
    tensor_init(&ty, y, RN, 64);
    eg_status s = eg_plane_wrap(ctx, &tx.managed, &px);
    if (s == EG_OK) s = eg_plane_wrap(ctx, &ty.managed, &py);
    if (s == EG_OK) s = plan(ctx, axpb, EG_STRUCTURE_HOST_TEAM, RN, 0, NULL, NULL, &plans[1]);
    if (s == EG_OK) s = bind(ctx, plans[1], py, px, &a, &b);
    eg_runner* r = NULL;
    struct eg_runner_desc d = runner_desc(plans, 2, 1000);
    if (s == EG_OK) s = eg_runner_new(ctx, &d, &r);
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    if (s == EG_OK) s = eg_run(ctx, r, &rep);
    int ok = s == EG_OK && rep.launches == want_launches && rep.finished == RN;
    for (int64_t i = 0; i < RN; ++i) ok &= y[i] == a * x[i] + b && k[i] == want_k[i];
    check(ok, "a two-plan step launches both plans, in order, until the first plan's finish");
    eg_runner_free(ctx, r);
    r = NULL;

    /* refusals: each status and a reason naming the feature */
    const struct { uint32_t flags; int32_t events, steps_reverse, n_derived; int64_t every, window, max_steps;
                   int graph; eg_status status; const char* reason; } bad[] = {
        { EG_RUN_RECORD, 0, 0, 0, 0, 0, 10, 0, EG_BAD_REQUEST, "needs its derived units" },
        { EG_RUN_REORDER, 0, 0, 0, 0, 0, 10, 0, EG_BAD_REQUEST, "no active set" },
        { 1u << 7, 0, 0, 0, 0, 0, 10, 0, EG_BAD_REQUEST, "unknown bits" },
        { 0, EG_EVENTS_REFINE, 0, 0, 0, 0, 10, 0, EG_BAD_REQUEST, "events refine needs the" },
        { 0, 0, EG_STEPS_REVERSE_FUSED, 0, 0, 0, 10, 0, EG_BAD_REQUEST, "steps_reverse FUSED needs EG_RUN_RECORD" },
        { 0, 0, 0, 1, 0, 0, 10, 0, EG_NOT_WRITTEN, "derived units" },
        { 0, 0, 0, 0, 0, 0, 10, 1, EG_BAD_REQUEST, "step_graph" },
        { 0, 0, 0, 0, 5, 0, 10, 0, EG_BAD_REQUEST, "every" },
        { 0, 0, 0, 0, 0, 4, 10, 0, EG_BAD_REQUEST, "window" },
        { 0, 0, 0, 0, 0, 0, 0, 0, EG_BAD_REQUEST, "max_steps" },
    };
    int all = 1;
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        d = runner_desc(plans, 1, bad[i].max_steps);
        d.flags = bad[i].flags;
        d.events = bad[i].events;
        d.steps_reverse = bad[i].steps_reverse;
        d.n_derived = bad[i].n_derived;
        d.every = bad[i].every;
        d.window = bad[i].window;
        d.step_graph = bad[i].graph ? (eg_graph*)&d : NULL;
        const int ok1 = eg_runner_new(ctx, &d, &r) == bad[i].status && r == NULL && reason_has(ctx, bad[i].reason);
        if (!ok1) printf("      refusal %zu ('%s') did not hold\n", i, bad[i].reason);
        all &= ok1;
    }
    check(all, "eg_runner_new refuses RECORD, REORDER, unknown flags, events REFINE, steps_reverse FUSED, "
               "derived units, a step_graph, every, window and max_steps < 1, each by name");
    eg_plan* swapped[2] = { plans[1], plans[0] };
    d = runner_desc(swapped, 2, 10);
    check(eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "has no finish"),
          "a first plan without a finish refused BAD_REQUEST");
    eg_plan* twice[2] = { plans[0], plans[0] };
    d = runner_desc(twice, 2, 10);
    check(eg_runner_new(ctx, &d, &r) == EG_NOT_WRITTEN && reason_has(ctx, "second finishing plan"),
          "a second finishing plan in one step refused NOT_WRITTEN");
    eg_plan* p3 = count_plan(ctx, c3, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t3);
    eg_plan* k_step[2] = { p3, plans[1] };
    d = runner_desc(k_step, 2, 10);
    check(eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "one step per launch"),
          "a several-plan step over a K=3 unit refused BAD_REQUEST");
    const int32_t counter_slot = 3;
    struct eg_bind_desc bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 1;
    bd.plane_slots = &counter_slot;
    bd.planes = &px;
    check(eg_bind(ctx, plans[0], &bd) == EG_BAD_REQUEST && reason_has(ctx, "finished counter"),
          "binding the eagle-made finished counter refused BAD_REQUEST");
    eg_plan* dev = NULL;
    check(plan(ctx, c1, EG_STRUCTURE_DEVICE, RN, 0, NULL, NULL, &dev) == (device_group ? EG_BAD_REQUEST : EG_NOT_WRITTEN) &&
              reason_has(ctx, "DEVICE"),
          "a DEVICE plan of a finishing unit on a host context refused: NOT_WRITTEN without the device group, BAD_REQUEST with it");
    d = runner_desc(plans, 1, 10);
    check(eg_runner_new(ctx, &d, &r) == EG_OK && eg_run_events(ctx, r, NULL, NULL) == EG_BAD_REQUEST &&
              reason_has(ctx, "no refine run") && eg_reverse(ctx, r, NULL, NULL) == EG_BAD_REQUEST &&
              reason_has(ctx, "not recorded"),
          "eg_run_events of a runner that never refined is refused BAD_REQUEST; eg_reverse of a runner without EG_RUN_RECORD is refused BAD_REQUEST");
    check(eg_runner_free(ctx, r) == EG_OK && eg_plan_free(ctx, plans[0]) == EG_OK &&
              eg_plan_free(ctx, plans[1]) == EG_OK && eg_plan_free(ctx, p3) == EG_OK &&
              eg_plane_free(ctx, px) == EG_OK && eg_plane_free(ctx, py) == EG_OK &&
              eg_unit_free(ctx, c1) == EG_OK && eg_unit_free(ctx, c3) == EG_OK,
          "runner, plans, planes and units freed");
    eg_host_threads(ctx, 1);
}

/* ---- reduce, copy, recording --------------------------------------------- */

static void tensor_as(struct tensor* t, uint8_t code, uint8_t bits) {
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
}

/* A 2-D (d0, d1) view over the same data (samples last). */
static void tensor_2d(struct tensor* t, int64_t d0, int64_t d1) {
    t->managed.dl_tensor.ndim = 2;
    t->shape[0] = d0;
    t->shape[1] = d1;
}

static eg_plane* wrap(eg_ctx* ctx, struct tensor* t, void* data, int64_t n, uint8_t code, uint8_t bits) {
    eg_plane* p = NULL;
    tensor_init(t, data, n, bits);
    tensor_as(t, code, bits);
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

static eg_status reduce_do(eg_ctx* ctx, eg_plane* in, eg_plane* out, eg_plane* comp, eg_reduce_op op,
                           eg_reduce_policy policy, uint32_t flags) {
    struct eg_reduce_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.flags = flags;
    d.in = in;
    d.out = out;
    d.comp = comp;
    d.op = op;
    d.policy = policy;
    return eg_reduce(ctx, &d);
}

#define NAN_D (0.0 / 0.0)

static void check_reduce(eg_ctx* ctx) {
    enum { N = 100001, M = 20001, NCOL = 3, K = 1000 };
    static double x[N], big[K + 1], zeros[K];
    static float xf[M], bigf[K + 1];
    static int64_t xi[M];
    static int32_t xi32[M];
    static double cols[NCOL * K];
    struct tensor tx, to;
    double out = NAN_D, want_sum = 0, want_max = -1e300, want_min = 1e300;
    for (int i = 0; i < N; ++i) {
        x[i] = (double)((i * 7919) % 1000) - 500.0;
        want_sum += x[i]; /* integers: exact in any order */
        if (x[i] > want_max) want_max = x[i];
        if (x[i] < want_min) want_min = x[i];
    }
    eg_plane* in = wrap(ctx, &tx, x, N, kDLFloat, 64);
    eg_plane* po = wrap(ctx, &to, &out, 1, kDLFloat, 64);
    int ok = in && po;
    ok = ok && reduce_do(ctx, in, po, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK && out == want_sum;
    out = NAN_D;
    ok = ok && reduce_do(ctx, in, po, NULL, EG_REDUCE_SUM, EG_REDUCE_FIXED, 0) == EG_OK && out == want_sum;
    out = NAN_D;
    ok = ok && reduce_do(ctx, in, po, NULL, EG_REDUCE_MAX, EG_REDUCE_COMPENSATED, 0) == EG_OK && out == want_max;
    out = NAN_D;
    ok = ok && reduce_do(ctx, in, po, NULL, EG_REDUCE_MIN, EG_REDUCE_COMPENSATED, 0) == EG_OK && out == want_min;
    check(ok, "eg_reduce float64 sum (both policies), max, min over 100001 samples equal the plain loop");

    /* float32 */
    float outf = 0.0f, fsum = 0, fmax = -1e30f, fmin = 1e30f;
    for (int i = 0; i < M; ++i) {
        xf[i] = (float)(i % 13) - 6.0f;
        fsum += xf[i]; /* |partial sum| < 2^24: exact */
        if (xf[i] > fmax) fmax = xf[i];
        if (xf[i] < fmin) fmin = xf[i];
    }
    struct tensor tf, tof;
    eg_plane* pf = wrap(ctx, &tf, xf, M, kDLFloat, 32);
    eg_plane* pof = wrap(ctx, &tof, &outf, 1, kDLFloat, 32);
    ok = pf && pof;
    ok = ok && reduce_do(ctx, pf, pof, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK && outf == fsum;
    outf = 0.0f;
    ok = ok && reduce_do(ctx, pf, pof, NULL, EG_REDUCE_MAX, EG_REDUCE_COMPENSATED, 0) == EG_OK && outf == fmax;
    outf = 0.0f;
    ok = ok && reduce_do(ctx, pf, pof, NULL, EG_REDUCE_MIN, EG_REDUCE_COMPENSATED, 0) == EG_OK && outf == fmin;
    check(ok, "eg_reduce float32 sum, max, min equal the plain loop");

    /* a (3, n) samples-last plane reduces column by column */
    double colout[NCOL], wsum[NCOL], wmax[NCOL];
    for (int j = 0; j < NCOL; ++j) {
        wsum[j] = 0;
        wmax[j] = -1e300;
        for (int i = 0; i < K; ++i) {
            cols[j * K + i] = (double)((i * (j + 3)) % 17) - 8.0;
            wsum[j] += cols[j * K + i];
            if (cols[j * K + i] > wmax[j]) wmax[j] = cols[j * K + i];
        }
    }
    struct tensor tcols, tcout;
    eg_plane* pcols = wrap(ctx, &tcols, cols, NCOL * K, kDLFloat, 64);
    eg_plane* pcout = wrap(ctx, &tcout, colout, NCOL, kDLFloat, 64);
    tensor_2d(&tcols, NCOL, K);
    ok = pcols && pcout && reduce_do(ctx, pcols, pcout, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK;
    for (int j = 0; j < NCOL; ++j) ok = ok && colout[j] == wsum[j];
    ok = ok && reduce_do(ctx, pcols, pcout, NULL, EG_REDUCE_MAX, EG_REDUCE_COMPENSATED, 0) == EG_OK;
    for (int j = 0; j < NCOL; ++j) ok = ok && colout[j] == wmax[j];
    check(ok, "eg_reduce of a (3, 1000) samples-last plane: each column's sum and max equal the plain loop");

    /* compensated: 2^53 then 1000 ones. The exact sum 2^53 + 1000 is representable; a plain sum is not. */
    big[0] = 9007199254740992.0;
    for (int i = 1; i <= K; ++i) big[i] = 1.0;
    struct tensor tb, thi, tcomp;
    double hi = 0, lo = 0;
    eg_plane* pb = wrap(ctx, &tb, big, K + 1, kDLFloat, 64);
    eg_plane* phi = wrap(ctx, &thi, &hi, 1, kDLFloat, 64);
    eg_plane* plo = wrap(ctx, &tcomp, &lo, 1, kDLFloat, 64);
    ok = pb && phi && plo && reduce_do(ctx, pb, phi, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK &&
         hi == 9007199254741992.0;
    hi = lo = NAN_D;
    ok = ok && reduce_do(ctx, pb, phi, plo, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, EG_REDUCE_PAIR) == EG_OK &&
         (long double)hi + (long double)lo == 9007199254741992.0L;
    check(ok, "eg_reduce compensated sum is the exact 2^53 + 1000; the PAIR (hi, comp) sums to it exactly");
    float bf = 0, lf = 0;
    bigf[0] = 16777216.0f;
    for (int i = 1; i <= K; ++i) bigf[i] = 1.0f;
    struct tensor tbf, thf, tlf;
    eg_plane* pbf = wrap(ctx, &tbf, bigf, K + 1, kDLFloat, 32);
    eg_plane* phf = wrap(ctx, &thf, &bf, 1, kDLFloat, 32);
    eg_plane* plf = wrap(ctx, &tlf, &lf, 1, kDLFloat, 32);
    ok = pbf && phf && plf && reduce_do(ctx, pbf, phf, plf, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, EG_REDUCE_PAIR) == EG_OK &&
         (double)bf + (double)lf == 16778216.0;
    check(ok, "eg_reduce float32 PAIR (hi, comp) sums exactly to 2^24 + 1000");

    /* a companion input: each term is x + comp */
    for (int i = 0; i < K; ++i) zeros[i] = 0.25;
    struct tensor tz, tk, tsk;
    double sk = 0.0, want = 0.0;
    for (int i = 0; i < K; ++i) want += (double)i + 0.25;
    static double ramp[K];
    for (int i = 0; i < K; ++i) ramp[i] = (double)i;
    eg_plane* pr = wrap(ctx, &tk, ramp, K, kDLFloat, 64);
    eg_plane* pz = wrap(ctx, &tz, zeros, K, kDLFloat, 64);
    eg_plane* psk = wrap(ctx, &tsk, &sk, 1, kDLFloat, 64);
    ok = pr && pz && psk && reduce_do(ctx, pr, psk, pz, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK && sk == want;
    check(ok, "eg_reduce with a companion input sums x + comp (499750)");

    /* integers and ALL */
    int64_t oi = 0, wi = 0, wimax = -1;
    int32_t o32 = 0, w32max = -1000;
    for (int i = 0; i < M; ++i) {
        xi[i] = (int64_t)i * 3000000007LL % 1000003 - 400000;
        wi += xi[i];
        if (xi[i] > wimax) wimax = xi[i];
        xi32[i] = (int32_t)(i % 101) - 50;
        if (xi32[i] > w32max) w32max = xi32[i];
    }
    struct tensor ti, toi, t32, to32;
    eg_plane* pi = wrap(ctx, &ti, xi, M, kDLInt, 64);
    eg_plane* poi = wrap(ctx, &toi, &oi, 1, kDLInt, 64);
    eg_plane* p32 = wrap(ctx, &t32, xi32, M, kDLInt, 32);
    eg_plane* po32 = wrap(ctx, &to32, &o32, 1, kDLInt, 32);
    ok = pi && poi && p32 && po32 && reduce_do(ctx, pi, poi, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK && oi == wi;
    ok = ok && reduce_do(ctx, pi, poi, NULL, EG_REDUCE_MAX, EG_REDUCE_COMPENSATED, 0) == EG_OK && oi == wimax;
    ok = ok && reduce_do(ctx, p32, po32, NULL, EG_REDUCE_MAX, EG_REDUCE_COMPENSATED, 0) == EG_OK && o32 == w32max;
    static unsigned char mask[M];
    unsigned char mo = 7;
    memset(mask, 1, sizeof(mask));
    struct tensor tm, tmo;
    eg_plane* pm = wrap(ctx, &tm, mask, M, kDLBool, 8);
    eg_plane* pmo = wrap(ctx, &tmo, &mo, 1, kDLBool, 8);
    ok = ok && pm && pmo && reduce_do(ctx, pm, pmo, NULL, EG_REDUCE_ALL, EG_REDUCE_COMPENSATED, 0) == EG_OK && mo == 1;
    mask[M - 3] = 0;
    ok = ok && reduce_do(ctx, pm, pmo, NULL, EG_REDUCE_ALL, EG_REDUCE_COMPENSATED, 0) == EG_OK && mo == 0;
    check(ok, "eg_reduce int64 sum and max, int32 max, and ALL over a bool mask equal the plain loop");

    /* refusals */
    unsigned char dummy = 0;
    uint32_t u32[4] = { 1, 2, 3, 4 }, u32o[1] = { 0 };
    struct tensor tu, tuo, te, tbadout;
    eg_plane* pu = wrap(ctx, &tu, u32, 4, kDLUInt, 32);
    eg_plane* puo = wrap(ctx, &tuo, u32o, 1, kDLUInt, 32);
    eg_plane* pe = wrap(ctx, &te, &dummy, 0, kDLFloat, 64);
    eg_plane* pbad = wrap(ctx, &tbadout, &colout, 3, kDLFloat, 64);
    int all = pu && puo && pe && pbad;
    all = all && reduce_do(ctx, pm, pmo, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_NOT_WRITTEN && reason_has(ctx, "bool plane takes op ALL");
    all = all && reduce_do(ctx, in, po, NULL, EG_REDUCE_ALL, EG_REDUCE_COMPENSATED, 0) == EG_NOT_WRITTEN && reason_has(ctx, "ALL takes a bool plane");
    all = all && reduce_do(ctx, pu, puo, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_NOT_WRITTEN && reason_has(ctx, "dtype");
    all = all && reduce_do(ctx, in, po, plo, EG_REDUCE_MAX, EG_REDUCE_COMPENSATED, 0) == EG_NOT_WRITTEN && reason_has(ctx, "comp takes");
    all = all && reduce_do(ctx, pi, poi, poi, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, EG_REDUCE_PAIR) == EG_NOT_WRITTEN && reason_has(ctx, "comp takes");
    all = all && reduce_do(ctx, pb, phi, plf, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, EG_REDUCE_PAIR) == EG_BAD_REQUEST && reason_has(ctx, "comp dtype");
    all = all && reduce_do(ctx, pb, phi, plo, EG_REDUCE_SUM, EG_REDUCE_FIXED, EG_REDUCE_PAIR) == EG_NOT_WRITTEN && reason_has(ctx, "policy FIXED");
    all = all && reduce_do(ctx, pb, phi, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, EG_REDUCE_PAIR) == EG_BAD_REQUEST && reason_has(ctx, "comp, which is NULL");
    all = all && reduce_do(ctx, in, poi, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_BAD_REQUEST && reason_has(ctx, "out dtype");
    all = all && reduce_do(ctx, in, pbad, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_BAD_REQUEST && reason_has(ctx, "out must have shape");
    all = all && reduce_do(ctx, pe, po, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_BAD_REQUEST && reason_has(ctx, "no samples");
    all = all && reduce_do(ctx, in, in, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_BAD_REQUEST;
    all = all && reduce_do(ctx, in, po, NULL, (eg_reduce_op)9, EG_REDUCE_COMPENSATED, 0) == EG_BAD_REQUEST && reason_has(ctx, "unknown reduce op");
    all = all && reduce_do(ctx, in, po, NULL, EG_REDUCE_SUM, (eg_reduce_policy)9, 0) == EG_BAD_REQUEST && reason_has(ctx, "unknown reduce policy");
    all = all && reduce_do(ctx, in, po, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 1u << 5) == EG_BAD_REQUEST && reason_has(ctx, "unknown bits");
    all = all && reduce_do(ctx, NULL, po, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_BAD_REQUEST;
    check(all, "eg_reduce refuses each op/dtype/policy pair the host does not serve NOT_WRITTEN and each "
               "shape, dtype, flag or handle fault BAD_REQUEST, by name");

    /* copy */
    static double src[K], dst[K];
    static float dstf[K];
    for (int i = 0; i < K; ++i) src[i] = (double)i * 0.1, dst[i] = -1.0;
    struct tensor ts, td, tdf;
    eg_plane* ps = wrap(ctx, &ts, src, K, kDLFloat, 64);
    eg_plane* pd = wrap(ctx, &td, dst, K, kDLFloat, 64);
    eg_plane* pdf = wrap(ctx, &tdf, dstf, K, kDLFloat, 32);
    ok = ps && pd && pdf && eg_copy(ctx, ps, pd) == EG_OK && memcmp(src, dst, sizeof(src)) == 0;
    ok = ok && eg_copy(ctx, ps, ps) == EG_OK;
    check(ok, "eg_copy moves a float64 plane bitwise (and onto itself does nothing)");
    td.managed.flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
    all = eg_copy(ctx, ps, pdf) == EG_BAD_REQUEST && reason_has(ctx, "one shape and dtype");
    all = all && eg_copy(ctx, ps, pe) == EG_BAD_REQUEST;
    all = all && eg_copy(ctx, ps, pd) == EG_BAD_REQUEST && reason_has(ctx, "read-only");
    all = all && eg_copy(ctx, ps, NULL) == EG_BAD_REQUEST;
    check(all, "eg_copy refuses a dtype or shape mismatch, a read-only dst and a NULL plane BAD_REQUEST");
    eg_plane* planes[] = { in, po, pf, pof, pcols, pcout, pb, phi, plo, pbf, phf, plf, pr, pz, psk, pi, poi, p32, po32,
                           pm, pmo, pu, puo, pe, pbad, ps, pd, pdf };
    all = 1;
    for (size_t i = 0; i < sizeof(planes) / sizeof(planes[0]); ++i) all &= eg_plane_free(ctx, planes[i]) == EG_OK;
    check(all, "reduce and copy planes freed");
}

/* A cell plane over one int32 of the caller's. */
static eg_plane* cell_at(eg_ctx* ctx, struct tensor* t, int32_t* p) { return wrap(ctx, t, p, 1, kDLInt, 32); }

/* Record one conditional of `kind` whose arm i copies src[i] into out; returns the graph. */
static eg_graph* record_copy_cond(eg_ctx* ctx, eg_cond_kind kind, int32_t n_arms, eg_plane* cell, eg_plane* const* src,
                                  eg_plane* out) {
    struct eg_cond_desc d;
    eg_graph* g = NULL;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.kind = kind;
    d.n_arms = n_arms;
    d.cell = cell;
    int32_t arms = kind == EG_COND_IF_ELSE ? 2 : kind == EG_COND_SWITCH ? n_arms : 1;
    int ok = eg_record_begin(ctx) == EG_OK && eg_record_cond_begin(ctx, &d) == EG_OK;
    for (int32_t i = 0; ok && i < arms; ++i) {
        ok = eg_copy(ctx, src[i], out) == EG_OK;
        if (ok && i + 1 < arms) ok = eg_record_cond_arm(ctx) == EG_OK;
    }
    ok = ok && eg_record_cond_end(ctx) == EG_OK && eg_record_end(ctx, &g) == EG_OK;
    return ok ? g : NULL;
}

static void check_conditionals(eg_ctx* ctx) {
    int32_t cell = 0;
    double vals[3] = { 11.0, 22.0, 33.0 }, out = 0.0;
    struct tensor tc, tv[3], to;
    eg_plane* pc = cell_at(ctx, &tc, &cell);
    eg_plane* src[3];
    for (int i = 0; i < 3; ++i) src[i] = wrap(ctx, &tv[i], &vals[i], 1, kDLFloat, 64);
    eg_plane* po = wrap(ctx, &to, &out, 1, kDLFloat, 64);
    int ok = pc && po && src[0] && src[1] && src[2];
    if (!ok) {
        check(0, "conditional planes wrapped");
        return;
    }

    eg_graph* g_if = record_copy_cond(ctx, EG_COND_IF, 0, pc, src, po);
    eg_graph* g_ie = record_copy_cond(ctx, EG_COND_IF_ELSE, 0, pc, src, po);
    eg_graph* g_sw = record_copy_cond(ctx, EG_COND_SWITCH, 3, pc, src, po);
    ok = g_if && g_ie && g_sw && out == 0.0;
    check(ok, "a recorded IF, IF_ELSE and SWITCH run nothing while recording");

    const struct { int32_t cell; double if_out, ie_out, sw_out; } rows[] = {
        { 1, 11.0, 11.0, 22.0 }, { 0, -5.0, 22.0, 11.0 }, { 7, 11.0, 11.0, -5.0 }, { 2, 11.0, 11.0, 33.0 },
        { -1, 11.0, 11.0, -5.0 }, { 3, 11.0, 11.0, -5.0 },
    };
    ok = 1;
    for (size_t i = 0; ok && i < sizeof(rows) / sizeof(rows[0]); ++i) {
        cell = rows[i].cell;
        out = -5.0;
        ok = eg_graph_launch(ctx, g_if) == EG_OK && out == rows[i].if_out;
        out = -5.0;
        ok = ok && eg_graph_launch(ctx, g_ie) == EG_OK && out == rows[i].ie_out;
        out = -5.0;
        ok = ok && eg_graph_launch(ctx, g_sw) == EG_OK && out == rows[i].sw_out;
        if (!ok) printf("      cell %d: if %g if_else %g switch %g\n", cell, rows[i].if_out, rows[i].ie_out, rows[i].sw_out);
    }
    check(ok, "IF takes its arm when the cell is nonzero; IF_ELSE arm 0 on nonzero, arm 1 on zero; SWITCH arm k "
              "on k and nothing out of range (0, 1, 2, -1, 3, 7)");

    /* WHILE: the cell is b[0] of a shift register {1,1,1,0,0}; the body shifts it left one place.
     * The body runs while b[0] != 0: three times here. */
    int32_t b[5];
    struct tensor tb[5];
    eg_plane* pb[5];
    ok = 1;
    for (int i = 0; i < 5; ++i) ok = ok && (pb[i] = cell_at(ctx, &tb[i], &b[i]));
    eg_graph* g_wh = NULL;
    if (ok) {
        struct eg_cond_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.kind = EG_COND_WHILE;
        d.cell = pb[0];
        ok = eg_record_begin(ctx) == EG_OK && eg_record_cond_begin(ctx, &d) == EG_OK;
        for (int i = 0; ok && i < 4; ++i) ok = eg_copy(ctx, pb[i + 1], pb[i]) == EG_OK;
        ok = ok && eg_record_cond_end(ctx) == EG_OK && eg_record_end(ctx, &g_wh) == EG_OK;
    }
    if (ok) {
        const int32_t start[5] = { 1, 1, 1, 0, 0 };
        memcpy(b, start, sizeof(b));
        ok = eg_graph_launch(ctx, g_wh) == EG_OK;
        int32_t want[5] = { 0, 0, 0, 0, 0 }; /* three shifts: the ones leave, zeros fill */
        ok = ok && memcmp(b, want, sizeof(b)) == 0;
        const int32_t start2[5] = { 0, 1, 1, 1, 1 };
        memcpy(b, start2, sizeof(b));
        ok = ok && eg_graph_launch(ctx, g_wh) == EG_OK && memcmp(b, start2, sizeof(b)) == 0; /* zero cell: no pass */
        const int32_t start3[5] = { 1, 0, 1, 1, 1 };
        memcpy(b, start3, sizeof(b));
        const int32_t want3[5] = { 0, 1, 1, 1, 1 };
        ok = ok && eg_graph_launch(ctx, g_wh) == EG_OK && memcmp(b, want3, sizeof(b)) == 0; /* one pass */
    }
    check(ok, "WHILE repeats its body while the cell is nonzero (three passes, none, one)");

    /* a graph keeps its planes alive past the caller's references */
    eg_plane_free(ctx, pc);
    eg_plane_free(ctx, po);
    for (int i = 0; i < 3; ++i) eg_plane_free(ctx, src[i]);
    ok = tc.deleted == 0 && to.deleted == 0 && tv[0].deleted == 0;
    cell = 1;
    out = -5.0;
    ok = ok && eg_graph_launch(ctx, g_ie) == EG_OK && out == 11.0;
    check(ok, "a recorded graph keeps its planes alive after the caller frees them");
    ok = eg_graph_free(ctx, g_if) == EG_OK && eg_graph_free(ctx, g_ie) == EG_OK && to.deleted == 0 &&
         eg_graph_free(ctx, g_sw) == EG_OK && tc.deleted == 1 && to.deleted == 1 && tv[0].deleted == 1 &&
         tv[2].deleted == 1 && eg_graph_free(ctx, NULL) == EG_OK;
    check(ok, "eg_graph_free drops the last references: the planes' deleters run once");
    check(eg_graph_free(ctx, g_if) == EG_BAD_REQUEST && reason_has(ctx, "graph") &&
              eg_graph_launch(ctx, g_if) == EG_BAD_REQUEST,
          "a freed graph is refused BAD_REQUEST by launch and by free");
    ok = eg_graph_free(ctx, g_wh) == EG_OK;
    for (int i = 0; i < 5; ++i) ok = ok && eg_plane_free(ctx, pb[i]) == EG_OK;
    check(ok, "while graph and planes freed");
}

static void check_recording_states(eg_ctx* ctx) {
    int32_t cell = 1;
    double f = 0.0;
    struct tensor tc, tf;
    eg_plane* pc = cell_at(ctx, &tc, &cell);
    eg_plane* pf = wrap(ctx, &tf, &f, 1, kDLFloat, 64);
    eg_graph *g = NULL, *none = NULL;
    struct eg_cond_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.cell = pc;
    int all = pc && pf;
    all = all && eg_record_end(ctx, &g) == EG_BAD_REQUEST && reason_has(ctx, "not recording");
    all = all && eg_record_cond_begin(ctx, &d) == EG_BAD_REQUEST && reason_has(ctx, "not recording");
    all = all && eg_record_cond_arm(ctx) == EG_BAD_REQUEST && eg_record_cond_end(ctx) == EG_BAD_REQUEST;
    all = all && eg_record_begin(ctx) == EG_OK && eg_record_begin(ctx) == EG_BAD_REQUEST && reason_has(ctx, "already recording");
    all = all && eg_record_cond_arm(ctx) == EG_BAD_REQUEST && reason_has(ctx, "no conditional is open");
    all = all && eg_record_cond_end(ctx) == EG_BAD_REQUEST && reason_has(ctx, "no conditional is open");
    d.cell = pf;
    all = all && eg_record_cond_begin(ctx, &d) == EG_BAD_REQUEST && reason_has(ctx, "int32");
    d.cell = pc;
    d.flags = 1;
    all = all && eg_record_cond_begin(ctx, &d) == EG_BAD_REQUEST && reason_has(ctx, "unknown bits");
    d.flags = 0;
    d.kind = 9;
    all = all && eg_record_cond_begin(ctx, &d) == EG_BAD_REQUEST && reason_has(ctx, "unknown conditional kind");
    d.kind = EG_COND_IF;
    d.n_arms = 2;
    all = all && eg_record_cond_begin(ctx, &d) == EG_BAD_REQUEST && reason_has(ctx, "SWITCH only");
    d.kind = EG_COND_SWITCH;
    d.n_arms = 0;
    all = all && eg_record_cond_begin(ctx, &d) == EG_BAD_REQUEST && reason_has(ctx, "n_arms");
    d.kind = EG_COND_IF;
    d.n_arms = 0;
    all = all && eg_record_cond_begin(ctx, &d) == EG_OK;
    all = all && eg_record_cond_arm(ctx) == EG_BAD_REQUEST && reason_has(ctx, "all are open");
    all = all && eg_record_end(ctx, &g) == EG_BAD_REQUEST && reason_has(ctx, "still open") && g == NULL;
    all = all && eg_record_cond_end(ctx) == EG_OK;
    d.kind = EG_COND_IF_ELSE;
    all = all && eg_record_cond_begin(ctx, &d) == EG_OK && eg_record_cond_end(ctx) == EG_BAD_REQUEST &&
         reason_has(ctx, "1 arm(s) of the 2");
    all = all && eg_record_cond_arm(ctx) == EG_OK && eg_record_cond_arm(ctx) == EG_BAD_REQUEST;
    all = all && eg_record_cond_end(ctx) == EG_OK;
    d.kind = EG_COND_SWITCH;
    d.n_arms = 3;
    all = all && eg_record_cond_begin(ctx, &d) == EG_OK && eg_record_cond_arm(ctx) == EG_OK && eg_record_cond_arm(ctx) == EG_OK &&
         eg_record_cond_arm(ctx) == EG_BAD_REQUEST && reason_has(ctx, "takes 3 arm(s)");
    all = all && eg_record_cond_end(ctx) == EG_OK;
    d.n_arms = 0;
    /* conditionals nest: an IF inside the second arm of an IF_ELSE */
    d.kind = EG_COND_IF_ELSE;
    all = all && eg_record_cond_begin(ctx, &d) == EG_OK && eg_record_cond_arm(ctx) == EG_OK;
    d.kind = EG_COND_IF;
    all = all && eg_record_cond_begin(ctx, &d) == EG_OK && eg_record_cond_end(ctx) == EG_OK &&
         eg_record_cond_end(ctx) == EG_OK;
    all = all && eg_record_end(ctx, &g) == EG_OK && g != NULL && eg_record_end(ctx, &none) == EG_BAD_REQUEST;
    check(all, "recording states: not recording, already recording, no conditional open, arm misuse, bad cell, "
               "flags, kind and n_arms, a conditional still open at end, nesting, each BAD_REQUEST by name");
    all = eg_graph_launch(ctx, g) == EG_OK && eg_graph_free(ctx, g) == EG_OK && eg_plane_free(ctx, pc) == EG_OK &&
          eg_plane_free(ctx, pf) == EG_OK;
    check(all && tc.deleted == 1 && tf.deleted == 1,
          "an empty-armed graph replays and frees its planes");
}

/* y = a*y + b in place over RN samples (axpb with x = y), then sum y into s, then copy s into t:
 * recorded once and replayed N times must equal the same calls run directly N times, bitwise. */
static void check_recorded_list(eg_ctx* ctx, eg_unit* axpb, eg_unit* count1) {
    enum { N = 5, K = RN };
    static double y_rec[K], y_dir[K], k[K], limit[K], want_k[K];
    static unsigned char mask[K];
    double s_rec = 0, s_dir = 0, t_rec = 0, t_dir = 0;
    const double a = 0.5, b = 1.0;
    struct tensor ty_r, ty_d, ts_r, ts_d, tt_r, tt_d, tcnt[3];
    for (int i = 0; i < K; ++i) y_rec[i] = y_dir[i] = (double)(i % 29) * 0.37 - 3.0;
    eg_plane *pyr = wrap(ctx, &ty_r, y_rec, K, kDLFloat, 64), *pyd = wrap(ctx, &ty_d, y_dir, K, kDLFloat, 64);
    eg_plane *psr = wrap(ctx, &ts_r, &s_rec, 1, kDLFloat, 64), *psd = wrap(ctx, &ts_d, &s_dir, 1, kDLFloat, 64);
    eg_plane *ptr = wrap(ctx, &tt_r, &t_rec, 1, kDLFloat, 64), *ptd = wrap(ctx, &tt_d, &t_dir, 1, kDLFloat, 64);
    eg_plan *plr = NULL, *pld = NULL;
    int ok = pyr && pyd && psr && psd && ptr && ptd &&
             plan(ctx, axpb, EG_STRUCTURE_HOST_TEAM, K, 0, NULL, NULL, &plr) == EG_OK &&
             plan(ctx, axpb, EG_STRUCTURE_HOST_SERIAL, K, 0, NULL, NULL, &pld) == EG_OK &&
             bind(ctx, plr, pyr, pyr, &a, &b) == EG_OK && bind(ctx, pld, pyd, pyd, &a, &b) == EG_OK;
    if (!ok) {
        check(0, "recorded list: planes and plans made");
        return;
    }

    /* the recorded list: launch, reduce into s, copy s into t, and a runner of the count unit */
    count_inputs(k, limit, mask);
    unsigned char mask0[K];
    memcpy(mask0, mask, sizeof(mask));
    int64_t want_launches = 0, want_finished = 0;
    count_oracle(limit, mask0, 1, 1000, want_k, &want_launches, &want_finished);
    eg_plan* cp = count_plan(ctx, count1, EG_STRUCTURE_HOST_TEAM, k, limit, mask, tcnt);
    eg_runner* rn = NULL;
    const struct eg_runner_desc rd = runner_desc(&cp, 1, 1000);
    ok = cp && eg_runner_new(ctx, &rd, &rn) == EG_OK;
    eg_graph* g = NULL;
    uint64_t tiles = 99;
    ok = ok && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, plr, &tiles) == EG_OK && tiles == 0 &&
         reduce_do(ctx, pyr, psr, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK &&
         eg_copy(ctx, psr, ptr) == EG_OK && eg_run(ctx, rn, NULL) == EG_OK && eg_record_end(ctx, &g) == EG_OK;
    ok = ok && memcmp(y_rec, y_dir, sizeof(y_rec)) == 0 && s_rec == 0.0 && t_rec == 0.0 && k[1] == 0.0 &&
         limit[1] == (double)(1 + 37 % 50);
    check(ok, "while recording, eg_launch, eg_reduce, eg_copy and eg_run run nothing");
    /* the caller's references go: the graph keeps plan, runner and planes */
    ok = ok && eg_plan_free(ctx, plr) == EG_OK && eg_plan_free(ctx, cp) == EG_OK && eg_runner_free(ctx, rn) == EG_OK &&
         eg_plane_free(ctx, pyr) == EG_OK && ty_r.deleted == 0;
    for (int i = 0; ok && i < N; ++i) {
        ok = eg_graph_launch(ctx, g) == EG_OK;
        ok = ok && eg_launch(ctx, pld, NULL) == EG_OK &&
             reduce_do(ctx, pyd, psd, NULL, EG_REDUCE_SUM, EG_REDUCE_COMPENSATED, 0) == EG_OK && eg_copy(ctx, psd, ptd) == EG_OK;
        ok = ok && memcmp(y_rec, y_dir, sizeof(y_rec)) == 0 && memcmp(&s_rec, &s_dir, sizeof(double)) == 0 &&
             memcmp(&t_rec, &t_dir, sizeof(double)) == 0;
    }
    check(ok, "a recorded launch + reduce + copy replayed 5 times equals the same calls run directly 5 times, bitwise");
    eg_graph* gb = NULL;
    check(eg_record_begin(ctx) == EG_OK && bind(ctx, pld, pyd, pyd, &a, &b) == EG_BAD_REQUEST && reason_has(ctx, "eg_bind while recording") &&
              eg_record_end(ctx, &gb) == EG_OK && eg_graph_free(ctx, gb) == EG_OK && bind(ctx, pld, pyd, pyd, &a, &b) == EG_OK,
          "eg_bind between eg_record_begin and eg_record_end is refused BAD_REQUEST by name; after the end it binds again");
    int kok = 1;
    for (int i = 0; i < K; ++i) kok &= k[i] == want_k[i];
    check(kok, "the recorded eg_run reached the closed-form count (replayed 5 times: a finished batch launches nothing)");
    ok = eg_graph_free(ctx, g) == EG_OK && ty_r.deleted == 1 && tcnt[0].deleted == 1 && tcnt[2].deleted == 1;
    check(ok, "eg_graph_free drops its plan, runner and plane references: the deleters run once");
    ok = eg_plan_free(ctx, pld) == EG_OK;
    eg_plane* rest[] = { pyd, psr, psd, ptr, ptd };
    for (int i = 0; i < 5; ++i) ok = ok && eg_plane_free(ctx, rest[i]) == EG_OK;
    check(ok, "recorded list leftovers freed");
}

/* An open recording and a live graph at eg_ctx_free are freed with the context. */
/* Two exports released together from two threads: the one that drops the count to zero frees the context. */
struct export_release {
    struct DLManagedTensorVersioned* export;
    volatile int* go;
};

static void* release_export(void* arg) {
    struct export_release* r = arg;
    while (!*r->go) sched_yield();
    r->export->deleter(r->export);
    return NULL;
}

/* eg_ctx_free with exports outstanding closes the context; the last export's release frees it. The oracle of "freed" is the
 * wrapped tensors' own deleter, which runs only in the context's teardown, and valgrind's leak check over this whole test. */
static void check_free_with_exports(void) {
    double a = 1.0, b = 2.0;
    struct tensor ta, tb;
    struct DLManagedTensorVersioned *ea = NULL, *eb = NULL;
    eg_ctx* ctx = make_ctx();
    eg_plane* pa = ctx ? wrap(ctx, &ta, &a, 1, kDLFloat, 64) : NULL;
    int ok = pa != NULL && eg_plane_dlpack(ctx, pa, &ea) == EG_OK && eg_plane_free(ctx, pa) == EG_OK && eg_ctx_free(ctx) == EG_OK;
    struct eg_plane_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    eg_plane* q = NULL;
    ok = ok && eg_plane_new(ctx, &d, &q) == EG_BAD_REQUEST && q == NULL && reason_has(ctx, "the context is closed") &&
         eg_ctx_free(ctx) == EG_BAD_REQUEST && ta.deleted == 0 && ((double*)ea->dl_tensor.data)[0] == 1.0;
    if (ea != NULL) ea->deleter(ea);
    check(ok && ta.deleted == 1, "eg_ctx_free with an export outstanding returns OK, a later call is refused by reason (closed), and the consumer's release frees the context");

    ctx = make_ctx();
    eg_plane* pa2 = ctx ? wrap(ctx, &ta, &a, 1, kDLFloat, 64) : NULL;
    eg_plane* pb = ctx ? wrap(ctx, &tb, &b, 1, kDLFloat, 64) : NULL;
    ok = pa2 != NULL && pb != NULL && eg_plane_dlpack(ctx, pa2, &ea) == EG_OK && eg_plane_dlpack(ctx, pb, &eb) == EG_OK &&
         eg_ctx_free(ctx) == EG_OK;
    volatile int go = 0;
    struct export_release ra = { ea, &go }, rb = { eb, &go };
    pthread_t ths[2];
    ok = ok && pthread_create(&ths[0], NULL, release_export, &ra) == 0 && pthread_create(&ths[1], NULL, release_export, &rb) == 0;
    go = 1;
    if (ok) {
        pthread_join(ths[0], NULL);
        pthread_join(ths[1], NULL);
    }
    check(ok && ta.deleted == 1 && tb.deleted == 1, "two exports released on two threads after eg_ctx_free: one teardown, every tensor released once");

    ctx = make_ctx();
    pa = ctx ? wrap(ctx, &ta, &a, 1, kDLFloat, 64) : NULL;
    ok = pa != NULL && eg_plane_dlpack(ctx, pa, &ea) == EG_OK;
    if (ok) ea->deleter(ea);
    ok = ok && eg_ctx_free(ctx) == EG_OK && ta.deleted == 1;
    check(ok, "eg_ctx_free with every export already released frees at once, as without exports");
}

static void check_free_with_graph(void) {
    eg_ctx* ctx = make_ctx();
    double v = 1.0, w = 0.0;
    struct tensor tv, tw;
    eg_plane* pv = ctx ? wrap(ctx, &tv, &v, 1, kDLFloat, 64) : NULL;
    eg_plane* pw = ctx ? wrap(ctx, &tw, &w, 1, kDLFloat, 64) : NULL;
    eg_graph* g = NULL;
    int ok = pv && pw && eg_record_begin(ctx) == EG_OK && eg_copy(ctx, pv, pw) == EG_OK && eg_record_end(ctx, &g) == EG_OK &&
             eg_record_begin(ctx) == EG_OK && eg_copy(ctx, pw, pv) == EG_OK;
    ok = ok && eg_ctx_free(ctx) == EG_OK && tv.deleted == 1 && tw.deleted == 1;
    check(ok, "eg_ctx_free with a live graph and an open recording runs each plane's deleter once");
}

/* ---- the active set: compaction and reorder ------------------------------- */

struct sim_out {
    int64_t iters, compactions, reorders, finished;
};

/* The old host rule on a plain model of the batch (a launch lowers `left` by one for a live slot; a
 * sample is done when it reaches 0): `every` steps per iteration, a compaction when the finished count
 * moved, the reorder when the last compaction found live < theta * 32 * live32 and live < span >= 4096. */
static void sim_active(const double* limit, const unsigned char* mask0, int steps, int64_t every, double theta,
                       int64_t max_steps, struct sim_out* o) {
    static int64_t left[RN], left2[RN];
    static unsigned char done[RN], done2[RN];
    static int32_t map[RN];
    int64_t finished = 0, count = RN, span = RN, at_last = 0;
    int fire = 0;
    const int64_t lpc = every / steps, cap = (max_steps + every - 1) / every;
    for (int64_t i = 0; i < RN; ++i) {
        done[i] = mask0[i];
        left[i] = mask0[i] ? 0 : ((int64_t)limit[i] + steps - 1) / steps;
        finished += mask0[i];
        map[i] = (int32_t)i;
    }
    memset(o, 0, sizeof(*o));
    while (o->iters < cap && finished != RN) {
        for (int64_t l = 0; l < lpc && count > 0 && finished != RN; ++l)
            for (int64_t t = 0; t < count; ++t) {
                const int64_t s = map[t];
                if (!done[s] && --left[s] == 0) done[s] = 1, ++finished;
            }
        if (finished != at_last) {
            int64_t k = 0, live32 = 0;
            for (int64_t i = 0; i < RN; ++i)
                if (!done[i]) map[k++] = (int32_t)i;
            for (int64_t g = 0; g < RN; g += 32) {
                int any = 0;
                for (int64_t i = g; i < g + 32 && i < RN; ++i) any |= !done[i];
                live32 += any;
            }
            count = k;
            ++o->compactions;
            at_last = finished;
            fire = theta > 0 && RN >= 4096 && (double)k < theta * 32.0 * (double)live32 && k < span && span >= 4096;
        }
        if (fire) { /* live slots of [0, span) to the front, stable */
            int64_t live = 0, a = 0, b;
            for (int64_t i = 0; i < span; ++i) live += !done[i];
            b = live;
            for (int64_t i = 0; i < span; ++i) {
                const int64_t d = done[i] ? b++ : a++;
                left2[d] = left[i];
                done2[d] = done[i];
            }
            for (int64_t i = 0; i < span; ++i) left[i] = left2[i], done[i] = done2[i];
            for (int64_t i = 0; i < live; ++i) map[i] = (int32_t)i;
            count = span = live;
            fire = 0;
            ++o->reorders;
        }
        ++o->iters;
    }
    o->finished = finished;
}

static int64_t sim_reorders_seen = 0;

/* One run of an active unit: planes, report and counters against the model; `every` 0 is the default. */
static void run_active_case(eg_ctx* ctx, eg_unit* unit, int64_t steps, eg_structure structure, int32_t threads,
                            int64_t every, double theta, int64_t max_steps) {
    static double k[RN], limit[RN], limit0[RN], want_k[RN];
    static unsigned char mask[RN], mask0[RN];
    struct tensor t[3];
    char what[200];
    count_inputs(k, limit, mask);
    memcpy(limit0, limit, sizeof(limit));
    memcpy(mask0, mask, RN);
    const int64_t eff = every ? every : steps * ((16 + steps - 1) / steps);
    struct sim_out sim;
    sim_active(limit, mask0, (int)steps, eff, theta, max_steps, &sim);
    int64_t launches = 0, finished = 0;
    count_oracle(limit, mask0, steps, sim.iters * eff, want_k, &launches, &finished);
    eg_host_threads(ctx, threads);
    eg_plan* p = count_plan(ctx, unit, structure, k, limit, mask, t);
    eg_runner* r = NULL;
    struct eg_runner_desc d = runner_desc(&p, 1, max_steps);
    d.every = every;
    if (theta > 0) d.flags = EG_RUN_REORDER, d.reorder_theta = theta;
    eg_status s = p ? eg_runner_new(ctx, &d, &r) : EG_FAULT;
    if (s == EG_OK) s = eg_plan_free(ctx, p);
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    if (s == EG_OK) s = eg_run(ctx, r, &rep);
    int planes_ok = 1;
    for (int64_t i = 0; i < RN; ++i)
        planes_ok &= memcmp(&k[i], &want_k[i], sizeof(double)) == 0 && memcmp(&limit[i], &limit0[i], sizeof(double)) == 0 &&
                     (mask[i] != 0) == (mask0[i] || want_k[i] >= limit[i]);
    snprintf(what, sizeof(what), "active K=%lld %s threads=%d every=%lld theta=%.2f max_steps=%lld: %lld iterations, %lld "
             "compactions, %lld reorders as the model; k, limit, mask in sample order", (long long)steps,
             structure == EG_STRUCTURE_HOST_SERIAL ? "SERIAL" : "TEAM", threads, (long long)every, theta,
             (long long)max_steps, (long long)sim.iters, (long long)sim.compactions, (long long)sim.reorders);
    check(s == EG_OK && planes_ok && rep.mode == EG_MODE_HOST && rep.n == RN && rep.finished == sim.finished &&
              rep.launches == sim.iters && rep.steps == sim.iters * eff && rep.mapped == sim.iters * RN &&
              rep.compactions == sim.compactions && rep.reorders == sim.reorders &&
              rep.flags == (sim.finished == RN ? EG_REPORT_DONE : 0u),
          what);
    sim_reorders_seen += sim.reorders;
    if (s == EG_OK && sim.finished == RN) {
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        check(eg_run(ctx, r, &rep) == EG_OK && rep.launches == 0 && rep.compactions == 0 && rep.finished == RN,
              "a second eg_run of a finished active batch launches nothing");
    }
    check(eg_runner_free(ctx, r) == EG_OK && t[0].deleted == 1, "active runner freed: the planes' deleters run once");
}

static void check_active_set(eg_ctx* ctx) {
    eg_unit *a1 = NULL, *a3 = NULL, *c1 = NULL;
    check(load(ctx, "count1a", 0xE1, &a1) == EG_OK && load(ctx, "count3a", 0xE3, &a3) == EG_OK &&
              load(ctx, "count1", 0xD1, &c1) == EG_OK,
          "eg_unit_load count1a and count3a (an active map and count, LOOKUP slots 4 and 5)");
    if (a1 == NULL || a3 == NULL || c1 == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int32_t threads[2] = { 1, 3 };
    const int64_t every1[3] = { 0, 4, 40 }, every3[3] = { 0, 6, 48 };
    const double thetas[3] = { 0.0, 0.5, 0.75 };
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti)
            for (int ei = 0; ei < 3; ++ei)
                for (int hi = 0; hi < 3; ++hi) {
                    run_active_case(ctx, a1, 1, structures[si], threads[ti], every1[ei], thetas[hi], 1000);
                    run_active_case(ctx, a3, 3, structures[si], threads[ti], every3[ei], thetas[hi], 1000);
                }
    run_active_case(ctx, a1, 1, EG_STRUCTURE_HOST_TEAM, 3, 4, 0.5, 20);
    run_active_case(ctx, a3, 3, EG_STRUCTURE_HOST_SERIAL, 1, 6, 0.0, 20);
    check(sim_reorders_seen > 0, "the reorder fired in the runs above (the model counted at least one)");

    /* refusals */
    static double k[RN], limit[RN], x[RN];
    static unsigned char mask[RN];
    struct tensor t[3], tx, t2[3];
    count_inputs(k, limit, mask);
    eg_plan* pa = count_plan(ctx, a1, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
    eg_plan* pa3 = count_plan(ctx, a3, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t2);
    eg_plan* pc = count_plan(ctx, c1, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t2);
    const struct { uint32_t flags; int64_t every; double theta; int use3; const char* reason; } bad[] = {
        { 0, 3, 0, 0, "below 4 steps" },
        { 0, -4, 0, 0, "< 0" },
        { 0, 7, 0, 1, "whole number of launches" },
        { EG_RUN_REORDER, 0, 0.1, 0, "is outside [" },
        { EG_RUN_REORDER, 0, 0.9, 0, "is outside [" },
    };
    int all = pa && pa3 && pc;
    for (size_t i = 0; all && i < sizeof(bad) / sizeof(bad[0]); ++i) {
        eg_plan* use = bad[i].use3 ? pa3 : pa;
        eg_runner* r = NULL;
        struct eg_runner_desc d = runner_desc(&use, 1, 100);
        d.every = bad[i].every;
        d.flags = bad[i].flags;
        d.reorder_theta = bad[i].theta;
        const int ok1 = eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && r == NULL && reason_has(ctx, bad[i].reason);
        if (!ok1) printf("      refusal %zu ('%s') did not hold\n", i, bad[i].reason);
        all &= ok1;
    }
    check(all, "eg_runner_new refuses every below 4 steps, negative or not a whole number of launches, and a "
               "theta outside [0.25, 0.75], each BAD_REQUEST by name");
    eg_unit* ax = NULL;
    eg_plan* pax = NULL;
    check(load(ctx, "axpb", 0xA1, &ax) == EG_OK && plan(ctx, ax, EG_STRUCTURE_HOST_TEAM, RN, 0, NULL, NULL, &pax) == EG_OK,
          "an axpb plan for the mixed step");
    eg_plan* mixed[2] = { pa, pax };
    eg_runner *r1 = NULL, *r2 = NULL;
    struct eg_runner_desc d = runner_desc(mixed, 2, 100);
    check(eg_runner_new(ctx, &d, &r1) == EG_BAD_REQUEST && reason_has(ctx, "all read an active set or none"),
          "plans that do and do not read an active set in one step refused BAD_REQUEST");
    d = runner_desc(&pc, 1, 100);
    d.flags = EG_RUN_REORDER;
    d.reorder_theta = 0.5;
    check(eg_runner_new(ctx, &d, &r1) == EG_BAD_REQUEST && reason_has(ctx, "active set"),
          "EG_RUN_REORDER over a unit with no active set refused BAD_REQUEST");
    d = runner_desc(&pa, 1, 100);
    eg_status s1 = eg_runner_new(ctx, &d, &r1);
    check(s1 == EG_OK && eg_runner_new(ctx, &d, &r2) == EG_BAD_REQUEST && reason_has(ctx, "another runner"),
          "a plan already reading one runner's active set refused by a second runner BAD_REQUEST");
    tensor_init(&tx, x, RN, 64);
    eg_plane *px = NULL;
    const int32_t map_slot = 4;
    struct eg_bind_desc bd;
    eg_plane_wrap(ctx, &tx.managed, &px);
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 1;
    bd.plane_slots = &map_slot;
    bd.planes = &px;
    check(eg_bind(ctx, pa, &bd) == EG_BAD_REQUEST && reason_has(ctx, "active set"),
          "binding the active map refused BAD_REQUEST");
    check(eg_launch(ctx, pa3, NULL) == EG_BAD_REQUEST && reason_has(ctx, "unbound"),
          "launching a plan that reads an active set without a runner refused BAD_REQUEST (its map is unbound)");
    all = eg_runner_free(ctx, r1) == EG_OK && eg_plan_free(ctx, pa) == EG_OK && eg_plan_free(ctx, pa3) == EG_OK &&
          eg_plan_free(ctx, pc) == EG_OK && eg_plan_free(ctx, pax) == EG_OK && eg_unit_free(ctx, ax) == EG_OK &&
          eg_plane_free(ctx, px) == EG_OK && eg_unit_free(ctx, a1) == EG_OK &&
          eg_unit_free(ctx, a3) == EG_OK && eg_unit_free(ctx, c1) == EG_OK;
    check(all, "active runner, plans, planes and units freed");
    eg_host_threads(ctx, 1);
}

/* ---- steps AUTO ------------------------------------------------------------ */

/* band_k as its doc states it: the power of two nearest sqrt(2R / h), h = newly / (live * k), in [lo, hi]. */
static int64_t model_band_k(double rate, int64_t newly, int64_t live, int64_t k, int64_t hi, int64_t lo) {
    if (newly <= 0) return hi;
    const double e2 = 2.0 * rate * (double)live * (double)k / (double)newly;
    int64_t p = 1;
    while (p < (1LL << 30) && e2 >= 2.0 * (double)p * (double)p) p *= 2;
    return p < lo ? lo : p > hi ? hi : p;
}

struct auto_out {
    int64_t iters, steps, mapped, compactions, finished, last_word;
};

/* The old pacing (first launch 16 steps, then the band, the last cut to the steps left; with an active set a
 * compaction after each launch in which the finished count moved) on the plain per-slot model. */
static void model_auto(const double* limit, const unsigned char* mask0, int64_t max_steps, int active, struct auto_out* o) {
    static int64_t left[RN];
    static unsigned char done[RN];
    static int32_t map[RN];
    int64_t finished = 0, count = RN, at_last = 0, k = 16, fin_at, word = max_steps < 16 ? max_steps : 16, steps_done = 0;
    for (int64_t i = 0; i < RN; ++i) {
        done[i] = mask0[i];
        left[i] = mask0[i] ? 0 : (int64_t)limit[i]; /* steps still needed */
        finished += mask0[i];
        map[i] = (int32_t)i;
    }
    fin_at = finished;
    memset(o, 0, sizeof(*o));
    int go = finished != RN && max_steps > 0;
    const int64_t cap_iters = (max_steps + 7) / 8 + 1;
    while (o->iters < cap_iters && go) {
        const int64_t live = active ? (count < RN ? count : RN) : RN;
        for (int64_t t = 0; t < live; ++t) {
            const int64_t s = active ? map[t] : t;
            for (int64_t w = 0; w < word && !done[s]; ++w)
                if (--left[s] == 0) done[s] = 1, ++finished;
        }
        if (active && finished != at_last) {
            int64_t c = 0;
            for (int64_t i = 0; i < RN; ++i)
                if (!done[i]) map[c++] = (int32_t)i;
            count = c;
            ++o->compactions;
            at_last = finished;
        }
        steps_done += word;
        o->mapped += RN - fin_at;
        const int64_t next = model_band_k(8.0, finished - fin_at, RN - finished, k, 64, 8);
        fin_at = finished;
        k = next;
        go = finished != RN && max_steps - steps_done > 0;
        if (go) word = next < max_steps - steps_done ? next : max_steps - steps_done;
        ++o->iters;
    }
    o->steps = steps_done;
    o->finished = finished;
    o->last_word = word;
}

static void run_auto_case(eg_ctx* ctx, eg_unit* unit, int active, eg_structure structure, int32_t threads, int64_t max_steps) {
    static double k[RN], limit[RN], want_k[RN];
    static unsigned char mask[RN], mask0[RN];
    struct tensor t[3];
    char what[200];
    count_inputs(k, limit, mask);
    memcpy(mask0, mask, RN);
    struct auto_out m;
    model_auto(limit, mask0, max_steps, active, &m);
    for (int64_t i = 0; i < RN; ++i) {
        const double ran = (double)m.steps; /* every live sample steps while it is live: k = min(limit, steps run) */
        want_k[i] = mask0[i] ? 0.0 : (limit[i] < ran ? limit[i] : ran);
    }
    eg_host_threads(ctx, threads);
    eg_plan* p = count_plan(ctx, unit, structure, k, limit, mask, t);
    eg_runner* r = NULL;
    const struct eg_runner_desc d = runner_desc(&p, 1, max_steps);
    eg_status s = p ? eg_runner_new(ctx, &d, &r) : EG_FAULT;
    if (s == EG_OK) s = eg_plan_free(ctx, p);
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    if (s == EG_OK) s = eg_run(ctx, r, &rep);
    int planes_ok = 1;
    for (int64_t i = 0; i < RN; ++i) planes_ok &= memcmp(&k[i], &want_k[i], sizeof(double)) == 0;
    const uint32_t flags = (m.finished == RN ? EG_REPORT_DONE : 0u) |
                           ((m.iters == 0 || m.finished != RN || m.last_word == 1) ? EG_REPORT_EXACT_STEPS : 0u);
    snprintf(what, sizeof(what), "steps AUTO%s %s threads=%d max_steps=%lld: %lld launches, %lld steps, %lld compactions as the "
             "model; k as the closed form", active ? " + active set" : "",
             structure == EG_STRUCTURE_HOST_SERIAL ? "SERIAL" : "TEAM", threads, (long long)max_steps, (long long)m.iters,
             (long long)m.steps, (long long)m.compactions);
    check(s == EG_OK && planes_ok && rep.mode == EG_MODE_HOST && rep.finished == m.finished && rep.launches == m.iters &&
              rep.steps == m.steps && rep.compactions == m.compactions && rep.flags == flags &&
              rep.mapped == (active ? m.mapped : m.iters * RN),
          what);
    check(eg_runner_free(ctx, r) == EG_OK && t[0].deleted == 1, "AUTO runner freed");
}

static void check_auto(eg_ctx* ctx) {
    eg_unit *ua = NULL, *uaa = NULL;
    check(load(ctx, "countA", 0xF1, &ua) == EG_OK && load(ctx, "countAa", 0xF2, &uaa) == EG_OK,
          "eg_unit_load countA and countAa (finish steps AUTO, steps_max 64, an eagle-made int64 steps word)");
    if (ua == NULL || uaa == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int32_t threads[2] = { 1, 3 };
    const int64_t caps[3] = { 1000, 40, 5 };
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti)
            for (int ci = 0; ci < 3; ++ci) {
                run_auto_case(ctx, ua, 0, structures[si], threads[ti], caps[ci]);
                run_auto_case(ctx, uaa, 1, structures[si], threads[ti], caps[ci]);
            }
    static double k[RN], limit[RN];
    static unsigned char mask[RN];
    struct tensor t[3], t2[3], t3[3];
    count_inputs(k, limit, mask);
    eg_plan* pa = count_plan(ctx, ua, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
    eg_plan* pb = count_plan(ctx, ua, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t2);
    eg_plan* two[2] = { pa, pb };
    eg_runner* r = NULL;
    struct eg_runner_desc d = runner_desc(two, 2, 100);
    int ok = pa && pb;
    check(ok && eg_runner_new(ctx, &d, &r) == EG_NOT_WRITTEN && reason_has(ctx, "steps AUTO in a several-plan step"),
          "steps AUTO in a several-plan step refused NOT_WRITTEN");
    eg_plan* pc2 = count_plan(ctx, uaa, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t3);
    d = runner_desc(&pc2, 1, 100);
    d.every = 8;
    check(pc2 && eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && reason_has(ctx, "steps AUTO"),
          "every with a steps AUTO unit refused BAD_REQUEST");
    const int32_t steps_slot = 4;
    eg_plane* pl = NULL;
    struct tensor tl;
    static double w[RN];
    tensor_init(&tl, w, RN, 64);
    eg_plane_wrap(ctx, &tl.managed, &pl);
    struct eg_bind_desc bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 1;
    bd.plane_slots = &steps_slot;
    bd.planes = &pl;
    check(eg_bind(ctx, pa, &bd) == EG_BAD_REQUEST && reason_has(ctx, "'steps'"),
          "binding a float64 plane of RN cells to the steps word refused BAD_REQUEST, naming the slot");
    ok = eg_plan_free(ctx, pa) == EG_OK && eg_plan_free(ctx, pb) == EG_OK && eg_plan_free(ctx, pc2) == EG_OK && eg_plane_free(ctx, pl) == EG_OK &&
         eg_unit_free(ctx, ua) == EG_OK && eg_unit_free(ctx, uaa) == EG_OK;
    check(ok, "AUTO plans, plane and units freed");
    eg_host_threads(ctx, 1);
}

/* ---- vector, matrix, table, int and count slots ---------------------------- */

static eg_plane* wrap_nd(eg_ctx* ctx, struct tensor* t, void* data, uint8_t code, uint8_t bits, int ndim, int64_t d0,
                         int64_t d1, int64_t d2) {
    eg_plane* p = NULL;
    tensor_init(t, data, 1, bits);
    tensor_as(t, code, bits);
    t->managed.dl_tensor.ndim = ndim;
    t->shape[0] = d0;
    t->shape[1] = d1;
    t->shape[2] = d2;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

static void check_slot_kinds(eg_ctx* ctx) {
    enum { M = 1000, TL = 7 };
    static double v[3 * M], v0[3 * M], w[3 * M], mm[4 * M], o[M], table[TL];
    static int32_t idx[M];
    const double s = 0.75;
    for (int i = 0; i < M; ++i) {
        idx[i] = (i * 5) % 23 - 3 + 3; /* >= 0 */
        for (int c = 0; c < 3; ++c) v[c * M + i] = v0[c * M + i] = (double)((i + c) % 9) * 0.5, w[c * M + i] = (double)(c + 1) * 0.25 + i * 1e-3;
        for (int c = 0; c < 4; ++c) mm[c * M + i] = (double)(c + 1) * 0.5 - i * 1e-3;
    }
    for (int i = 0; i < TL; ++i) table[i] = 100.0 + i;
    eg_unit* unit = NULL;
    struct tensor tv, tw, tmm, ti, tt, to, tbad;
    eg_plane *pv = wrap_nd(ctx, &tv, v, kDLFloat, 64, 2, 3, M, 0), *pw = wrap_nd(ctx, &tw, w, kDLFloat, 64, 2, 3, M, 0),
             *pmm = wrap_nd(ctx, &tmm, mm, kDLFloat, 64, 3, 2, 2, M), *pi = wrap_nd(ctx, &ti, idx, kDLInt, 32, 1, M, 0, 0),
             *pt = wrap_nd(ctx, &tt, table, kDLFloat, 64, 1, TL, 0, 0), *po = wrap_nd(ctx, &to, o, kDLFloat, 64, 1, M, 0, 0);
    int ok = pv && pw && pmm && pi && pt && po && load(ctx, "vecs", 0x71, &unit) == EG_OK;
    if (!ok) {
        check(0, "vecs: unit and planes made");
        return;
    }
    const int32_t slots[7] = { 0, 1, 2, 3, 4, 6, 0 };
    eg_plane* planes[6] = { pv, pw, pmm, pi, pt, po };
    const int32_t plane_slots[6] = { 0, 1, 2, 3, 4, 6 };
    const int32_t uslot = 7;
    const void* uvals[1] = { &s };
    (void)slots;
    struct eg_bind_desc bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 6;
    bd.plane_slots = plane_slots;
    bd.planes = planes;
    bd.n_uniforms = 1;
    bd.uniform_slots = &uslot;
    bd.uniform_values = uvals;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    for (int si = 0; si < 2; ++si) {
        for (int i = 0; i < 3 * M; ++i) v[i] = v0[i];
        eg_plan* p = NULL;
        ok = plan(ctx, unit, structures[si], M, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
             eg_launch(ctx, p, NULL) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK;
        /* the plain loop, twice */
        static double want_v[3 * M], want_o[M];
        for (int i = 0; i < 3 * M; ++i) want_v[i] = v0[i];
        for (int rep = 0; rep < 2; ++rep)
            for (int i = 0; i < M; ++i) {
                const double add = table[(uint32_t)idx[i] % TL];
                for (int c = 0; c < 3; ++c) want_v[c * M + i] += w[c * M + i] * s + add;
                double m = 0.0;
                for (int c = 0; c < 4; ++c) m += mm[c * M + i];
                want_o[i] = (want_v[i] + want_v[M + i] + want_v[2 * M + i]) + m + (double)M;
            }
        ok = ok && memcmp(v, want_v, sizeof(v)) == 0 && memcmp(o, want_o, sizeof(o)) == 0;
        check(ok, "vecs: rank-1 MUTABLE and VEC_IN, rank-2 MAT_IN and rank-0 OUT as mirrors, an int32 PER_SAMPLE, a "
                  "LOOKUP table read by element count and eagle's NSAMPLES equal the plain loop (bitwise, twice)");
        eg_plan_free(ctx, p);
    }

    /* refusals */
    eg_plan* p = NULL;
    ok = plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, M, 0, NULL, NULL, &p) == EG_OK;
    int all = ok;
    const int32_t one_slot[1] = { 0 };
    struct eg_bind_desc b1;
    memset(&b1, 0, sizeof(b1));
    b1.struct_size = sizeof(b1);
    b1.n_planes = 1;
    b1.plane_slots = one_slot;
    eg_plane* pbad = wrap_nd(ctx, &tbad, v, kDLFloat, 64, 1, M, 0, 0); /* v as (n,): rank 1 wants (3, n) */
    b1.planes = &pbad;
    all = all && pbad && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "a plane of shape (3, ");
    eg_plane_free(ctx, pbad);
    pbad = wrap_nd(ctx, &tbad, idx, kDLFloat, 64, 1, M, 0, 0);
    const int32_t idx_slot[1] = { 3 };
    b1.plane_slots = idx_slot;
    b1.planes = &pbad;
    all = all && pbad && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "is not the slot's");
    eg_plane_free(ctx, pbad);
    const int32_t nn_slot[1] = { 5 };
    b1.plane_slots = nn_slot;
    b1.planes = &pi;
    all = all && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "eagle passes it");
    const int32_t tab_slot[1] = { 4 };
    b1.plane_slots = tab_slot;
    b1.planes = &pi;
    all = all && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "is not the slot's");
    all = all && eg_launch(ctx, p, NULL) == EG_BAD_REQUEST && reason_has(ctx, "unbound");
    check(all, "vecs refusals: a plane of the wrong rank or shape, a wrong dtype, binding the sample count, an unbound "
               "slot, each BAD_REQUEST by name");
    ok = eg_plan_free(ctx, p) == EG_OK && eg_unit_free(ctx, unit) == EG_OK;
    for (int i = 0; i < 6; ++i) ok = ok && eg_plane_free(ctx, planes[i]) == EG_OK;
    check(ok, "vecs plan, planes and unit freed");
}

/* ---- an int32 TERMINATED mask, and a finish flag the host reads and ignores ---------- */

/* count1i / count1ia over an int32 mask: planes in sample order after the run (and after a reorder), the report
 * as the closed form (plain) or the model of the old rule (active). */
static void run_i32_mask_case(eg_ctx* ctx, eg_unit* unit, int active, eg_structure structure, int32_t threads,
                              int64_t every, double theta) {
    static double k[RN], limit[RN], limit0[RN], want_k[RN];
    static unsigned char mask8[RN], mask0[RN];
    static int32_t mask[RN];
    struct tensor tk, tl, tm;
    char what[200];
    count_inputs(k, limit, mask8);
    memcpy(limit0, limit, sizeof(limit));
    memcpy(mask0, mask8, RN);
    for (int64_t i = 0; i < RN; ++i) mask[i] = mask8[i];
    struct sim_out sim;
    int64_t launches = 0, finished = 0, eff = 1;
    if (active) {
        eff = every ? every : 16;
        sim_active(limit, mask0, 1, eff, theta, 1000, &sim);
        count_oracle(limit, mask0, 1, sim.iters * eff, want_k, &launches, &finished);
    } else {
        count_oracle(limit, mask0, 1, 1000, want_k, &launches, &finished);
    }
    eg_host_threads(ctx, threads);
    eg_plane* pk = wrap(ctx, &tk, k, RN, kDLFloat, 64);
    eg_plane* pl = wrap(ctx, &tl, limit, RN, kDLFloat, 64);
    eg_plane* pm = wrap(ctx, &tm, mask, RN, kDLInt, 32);
    eg_plan* p = NULL;
    eg_runner* r = NULL;
    eg_status s = pk && pl && pm ? plan(ctx, unit, structure, RN, 0, NULL, NULL, &p) : EG_FAULT;
    if (s == EG_OK) s = bind_count(ctx, p, pk, pl, pm);
    struct eg_runner_desc d = runner_desc(&p, 1, 1000);
    d.every = every;
    if (theta > 0) d.flags = EG_RUN_REORDER, d.reorder_theta = theta;
    if (s == EG_OK) s = eg_runner_new(ctx, &d, &r);
    eg_plan_free(ctx, p);
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    if (s == EG_OK) s = eg_run(ctx, r, &rep);
    int ok = s == EG_OK;
    for (int64_t i = 0; ok && i < RN; ++i)
        ok = memcmp(&k[i], &want_k[i], sizeof(double)) == 0 && memcmp(&limit[i], &limit0[i], sizeof(double)) == 0 &&
             (mask[i] != 0) == (mask0[i] || want_k[i] >= limit[i]);
    if (active)
        ok = ok && rep.launches == sim.iters && rep.compactions == sim.compactions && rep.reorders == sim.reorders &&
             rep.finished == sim.finished;
    else
        ok = ok && rep.launches == launches && rep.finished == finished;
    snprintf(what, sizeof(what), "int32 mask%s %s threads=%d every=%lld theta=%.2f: planes in sample order, report as "
             "the model", active ? " + active set" : "", structure == EG_STRUCTURE_HOST_SERIAL ? "SERIAL" : "TEAM",
             threads, (long long)every, theta);
    check(ok, what);
    if (active) sim_reorders_seen += sim.reorders;
    eg_runner_free(ctx, r);
    eg_plane_free(ctx, pk);
    eg_plane_free(ctx, pl);
    eg_plane_free(ctx, pm);
}

static void check_mask_dtypes(eg_ctx* ctx) {
    eg_unit *ui = NULL, *uia = NULL, *ue = NULL;
    check(load(ctx, "count1i", 0x81, &ui) == EG_OK && load(ctx, "count1ia", 0x82, &uia) == EG_OK &&
              load(ctx, "count1e", 0x83, &ue) == EG_OK,
          "eg_unit_load count1i and count1ia (an int32 TERMINATED mask) and count1e (finish flag ENTRY_COUNTS)");
    if (ui == NULL || uia == NULL || ue == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int32_t threads[2] = { 1, 3 };
    const int64_t before = sim_reorders_seen;
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            run_count(ctx, ue, 1, structures[si], threads[ti], 1000); /* the flag changes nothing on the host */
            run_i32_mask_case(ctx, ui, 0, structures[si], threads[ti], 0, 0.0);
            for (int ei = 0; ei < 2; ++ei)
                for (int hi = 0; hi < 2; ++hi)
                    run_i32_mask_case(ctx, uia, 1, structures[si], threads[ti], ei ? 4 : 0, hi ? 0.5 : 0.0);
        }
    check(sim_reorders_seen > before, "the reorder fired over the int32 mask (the model counted at least one)");
    eg_host_threads(ctx, 1);
    check(eg_unit_free(ctx, ui) == EG_OK && eg_unit_free(ctx, uia) == EG_OK && eg_unit_free(ctx, ue) == EG_OK,
          "int32-mask and ENTRY_COUNTS units freed");
}

/* ---- index sets and prefix sets ------------------------------------------- */

static void check_index_sets(eg_ctx* ctx) {
    enum { M = 1000, CAP = 400 };
    static double x[M], y[M], want[M];
    static int32_t map[CAP];
    static int64_t map64[CAP];
    uint32_t count = 0;
    eg_unit *ul = NULL, *up = NULL;
    check(load(ctx, "lanes", 0x91, &ul) == EG_OK && load(ctx, "prefix", 0x92, &up) == EG_OK,
          "eg_unit_load lanes (an index map and count) and prefix (a count only)");
    if (ul == NULL || up == NULL) return;
    for (int i = 0; i < M; ++i) x[i] = (double)i * 0.25 - 7.0;
    for (int i = 0; i < CAP; ++i) map[i] = (int32_t)((i * 7919 + 13) % M), map64[i] = map[i];
    /* 7919 is prime to 1000: the first 400 lanes are distinct */
    struct tensor ty, tx, tm, tc, tm64;
    eg_plane *py = wrap(ctx, &ty, y, M, kDLFloat, 64), *px = wrap(ctx, &tx, x, M, kDLFloat, 64),
             *pm = wrap(ctx, &tm, map, CAP, kDLInt, 32), *pm64 = wrap(ctx, &tm64, map64, CAP, kDLInt, 64),
             *pc = wrap(ctx, &tc, &count, 1, kDLUInt, 32);
    int ok = py && px && pm && pm64 && pc;
    if (!ok) {
        check(0, "index set planes wrapped");
        return;
    }
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int32_t threads[2] = { 1, 3 };
    const uint32_t counts[4] = { 0, 1, 137, CAP };
    int all = 1;
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti) {
            eg_host_threads(ctx, threads[ti]);
            eg_plan* p = NULL;
            const int32_t slots[4] = { 0, 1, 2, 3 };
            eg_plane* planes[4] = { py, px, pm, pc };
            struct eg_bind_desc bd;
            memset(&bd, 0, sizeof(bd));
            bd.struct_size = sizeof(bd);
            bd.n_planes = 4;
            bd.plane_slots = slots;
            bd.planes = planes;
            count = 0;
            all &= plan(ctx, ul, structures[si], M, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK;
            for (int ci = 0; all && ci < 4; ++ci) {
                count = counts[ci]; /* the count cell is read at every launch */
                for (int i = 0; i < M; ++i) y[i] = want[i] = -1.0;
                for (uint32_t t = 0; t < count; ++t) want[map[t]] = 2.0 * x[map[t]] + 1.0;
                all &= eg_launch(ctx, p, NULL) == EG_OK && memcmp(y, want, sizeof(y)) == 0;
            }
            eg_plan_free(ctx, p);
        }
    check(all, "an index set runs exactly the lanes map[0..count) (count 0, 1, 137, 400; read at each launch) and leaves "
               "every other lane alone, equal to the plain loop on SERIAL and TEAM, 1 and 3 threads");
    /* one partition is a range of POSITIONS of the set, cut at the live count: it runs map[base .. min(base + count, live)) and equals the whole run restricted to them */
    {
        const struct { int64_t base, count; } ranges[4] = { { 0, 60 }, { 50, 150 }, { 137, 50 }, { 300, 100 } };
        int in_range = 1;
        for (int si = 0; si < 2; ++si)
            for (int ci = 0; ci < 2; ++ci) {
                const int32_t slots[4] = { 0, 1, 2, 3 };
                eg_plane* planes[4] = { py, px, pm, pc };
                struct eg_bind_desc bd;
                memset(&bd, 0, sizeof(bd));
                bd.struct_size = sizeof(bd);
                bd.n_planes = 4;
                bd.plane_slots = slots;
                bd.planes = planes;
                count = counts[2 + ci];
                for (int ri = 0; ri < 4; ++ri) {
                    eg_plan* p = NULL;
                    const int64_t pb[1] = { ranges[ri].base }, pc_[1] = { ranges[ri].count };
                    in_range &= plan(ctx, ul, structures[si], M, 1, pb, pc_, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK;
                    for (int i = 0; i < M; ++i) y[i] = want[i] = -1.0;
                    for (int64_t t = ranges[ri].base; t < ranges[ri].base + ranges[ri].count && t < count; ++t)
                        want[map[t]] = 2.0 * x[map[t]] + 1.0;
                    in_range &= eg_launch(ctx, p, NULL) == EG_OK && memcmp(y, want, sizeof(y)) == 0;
                    eg_plan_free(ctx, p);
                }
            }
        check(in_range, "an index set over one partition [base, base + count) runs map[base .. min(base + count, live)) only: "
                        "ranges (0, 60), (50, 150), (137, 50), (300, 100) at live 137 and 400, SERIAL and TEAM");
        eg_plan* out = NULL;
        const int64_t past[1] = { M - 10 }, wide_count[1] = { 11 };
        check(plan(ctx, ul, EG_STRUCTURE_HOST_TEAM, M, 1, past, wide_count, &out) == EG_BAD_REQUEST && reason_has(ctx, "leaves the range"),
              "an index set partition leaving [0, n) refused BAD_REQUEST");
    }

    /* a prefix set: lanes [0, count) */
    all = 1;
    for (int si = 0; si < 2; ++si) {
        eg_plan* p = NULL;
        const int32_t slots[3] = { 0, 1, 2 };
        eg_plane* planes[3] = { py, px, pc };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 3;
        bd.plane_slots = slots;
        bd.planes = planes;
        all &= plan(ctx, up, structures[si], M, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK;
        for (int ci = 0; all && ci < 4; ++ci) {
            count = ci == 3 ? M : counts[ci];
            for (int i = 0; i < M; ++i) y[i] = want[i] = -1.0;
            for (uint32_t t = 0; t < count; ++t) want[t] = 2.0 * x[t] + 1.0;
            all &= eg_launch(ctx, p, NULL) == EG_OK && memcmp(y, want, sizeof(y)) == 0;
        }
        count = M + 1;
        all &= eg_launch(ctx, p, NULL) == EG_BAD_REQUEST && reason_has(ctx, "above the sample count's");
        eg_plan_free(ctx, p);
    }
    check(all, "a prefix set runs lanes [0, count) equal to the plain loop, and a count above the sample count is "
               "refused BAD_REQUEST at launch");

    /* refusals at bind and launch */
    eg_plan* p = NULL;
    all = plan(ctx, ul, EG_STRUCTURE_HOST_TEAM, M, 0, NULL, NULL, &p) == EG_OK;
    const int32_t map_slot[1] = { 2 }, count_slot[1] = { 3 }, y_slot[1] = { 0 }, x_slot[1] = { 1 };
    struct eg_bind_desc b1;
    memset(&b1, 0, sizeof(b1));
    b1.struct_size = sizeof(b1);
    b1.n_planes = 1;
    eg_plane* one[1];
    one[0] = py; b1.plane_slots = y_slot; b1.planes = one; all = all && eg_bind(ctx, p, &b1) == EG_OK;
    one[0] = px; b1.plane_slots = x_slot; all = all && eg_bind(ctx, p, &b1) == EG_OK;
    count = 3;
    one[0] = pc; b1.plane_slots = count_slot; all = all && eg_bind(ctx, p, &b1) == EG_OK;
    one[0] = pm64; b1.plane_slots = map_slot;
    all = all && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "is not the slot's");
    map[1] = map[0];
    one[0] = pm;
    all = all && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "repeats an earlier entry");
    map[1] = M;
    all = all && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "is outside the dense lanes");
    map[1] = 5;
    count = CAP + 1;
    all = all && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "the index map holds");
    count = 3;
    all = all && eg_bind(ctx, p, &b1) == EG_OK;
    count = CAP + 1; /* rewritten after the bind: refused at launch, nothing runs past the map */
    all = all && eg_launch(ctx, p, NULL) == EG_BAD_REQUEST && reason_has(ctx, "above the map's");
    one[0] = pm; b1.plane_slots = count_slot; /* a map where the count goes: dtype */
    all = all && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST;
    const int64_t part_base[2] = { 0, 500 }, part_count[2] = { 500, 500 };
    eg_plan* pp = NULL;
    all = all && plan(ctx, ul, EG_STRUCTURE_HOST_TEAM, M, 2, part_base, part_count, &pp) == EG_BAD_REQUEST &&
          reason_has(ctx, "one partition (positions of the set)");
    check(all, "index set refusals by name: a map of the wrong width, a repeated lane, a lane outside the dense lanes, a "
               "count above the map (at bind and at launch), a wrong plane in the count slot, two partitions");
    eg_plan_free(ctx, p);
    ok = eg_plane_free(ctx, py) == EG_OK && eg_plane_free(ctx, px) == EG_OK && eg_plane_free(ctx, pm) == EG_OK &&
         eg_plane_free(ctx, pm64) == EG_OK && eg_plane_free(ctx, pc) == EG_OK && eg_unit_free(ctx, ul) == EG_OK &&
         eg_unit_free(ctx, up) == EG_OK;
    check(ok, "index set planes and units freed");
    eg_host_threads(ctx, 1);
}

/* ---- an active set over groups of lanes, and the wide roles -------------------- */

/* The old group rule on a plain model: a launch lowers `left` by one for each live lane of a live group; a
 * compaction (when the finished count moved) keeps a group while any lane of it is live and launches
 * groups * G lanes. */
static void sim_groups(const double* limit, const unsigned char* mask0, int64_t group, int64_t every, int64_t max_steps,
                       struct sim_out* o) {
    static int64_t left[RN];
    static unsigned char done[RN];
    int64_t finished = 0, at_last = 0;
    const int64_t ng = (RN + group - 1) / group, cap = (max_steps + every - 1) / every;
    int64_t live_groups = ng;
    for (int64_t i = 0; i < RN; ++i) {
        done[i] = mask0[i];
        left[i] = mask0[i] ? 0 : (int64_t)limit[i];
        finished += mask0[i];
    }
    memset(o, 0, sizeof(*o));
    while (o->iters < cap && finished != RN) {
        for (int64_t l = 0; l < every && live_groups > 0 && finished != RN; ++l) {
            for (int64_t i = 0; i < RN; ++i)
                if (!done[i] && --left[i] == 0) done[i] = 1, ++finished;
        }
        if (finished != at_last) {
            live_groups = 0;
            for (int64_t g = 0; g < ng; ++g) {
                int any = 0;
                for (int64_t i = g * group; i < (g + 1) * group && i < RN; ++i) any |= !done[i];
                live_groups += any;
            }
            ++o->compactions;
            at_last = finished;
        }
        ++o->iters;
    }
    o->finished = finished;
}

static void check_groups(eg_ctx* ctx) {
    eg_unit* unit = NULL;
    check(load(ctx, "count1g", 0x84, &unit) == EG_OK, "eg_unit_load count1g (an active set over groups of 5 lanes)");
    if (unit == NULL) return;
    void* lib = dlopen(library_path, RTLD_LAZY | RTLD_NOLOAD);
    int64_t* past = lib ? (int64_t*)dlsym(lib, "fixture_past_n") : NULL;
    check(past != NULL, "the fixture's past-n counter is reachable");
    if (past == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int32_t threads[2] = { 1, 3 };
    const int64_t everys[3] = { 0, 4, 40 };
    for (int si = 0; si < 2; ++si)
        for (int ti = 0; ti < 2; ++ti)
            for (int ei = 0; ei < 3; ++ei) {
                static double k[RN], limit[RN], limit0[RN], want_k[RN];
                static unsigned char mask[RN], mask0[RN];
                struct tensor t[3];
                char what[200];
                count_inputs(k, limit, mask);
                memcpy(limit0, limit, sizeof(limit));
                memcpy(mask0, mask, RN);
                const int64_t every = everys[ei], eff = every ? every : 16, max_steps = 1000;
                struct sim_out sim;
                int64_t launches = 0, finished = 0;
                sim_groups(limit, mask0, 5, eff, max_steps, &sim);
                count_oracle(limit, mask0, 1, sim.iters * eff, want_k, &launches, &finished);
                eg_host_threads(ctx, threads[ti]);
                *past = 0;
                eg_plan* p = count_plan(ctx, unit, structures[si], k, limit, mask, t);
                eg_runner* r = NULL;
                struct eg_runner_desc d = runner_desc(&p, 1, max_steps);
                d.every = every;
                eg_status s = p ? eg_runner_new(ctx, &d, &r) : EG_FAULT;
                if (s == EG_OK) s = eg_plan_free(ctx, p);
                struct eg_run_report rep;
                memset(&rep, 0, sizeof(rep));
                rep.struct_size = sizeof(rep);
                if (s == EG_OK) s = eg_run(ctx, r, &rep);
                int planes_ok = 1;
                for (int64_t i = 0; i < RN; ++i)
                    planes_ok &= memcmp(&k[i], &want_k[i], sizeof(double)) == 0 && memcmp(&limit[i], &limit0[i], sizeof(double)) == 0 &&
                                 (mask[i] != 0) == (mask0[i] || want_k[i] >= limit[i]);
                /* every live group launches 5 lanes; the last group holds 4 samples, so its fifth lane is past n */
                snprintf(what, sizeof(what), "groups of 5 %s threads=%d every=%lld: %lld iterations, %lld compactions as the model, "
                         "planes as the closed form, lanes past n handed to the entry", structures[si] == EG_STRUCTURE_HOST_SERIAL ? "SERIAL" : "TEAM",
                         threads[ti], (long long)every, (long long)sim.iters, (long long)sim.compactions);
                check(s == EG_OK && planes_ok && rep.finished == sim.finished && rep.launches == sim.iters && rep.steps == sim.iters * eff &&
                          rep.compactions == sim.compactions && rep.reorders == 0 && *past > 0 &&
                          rep.flags == (sim.finished == RN ? EG_REPORT_DONE : 0u),
                      what);
                if (s == EG_OK) check(eg_runner_free(ctx, r) == EG_OK, "groups runner freed");
            }
    eg_host_threads(ctx, 1);
    static double k[RN], limit[RN];
    static unsigned char mask[RN];
    struct tensor t[3];
    count_inputs(k, limit, mask);
    eg_plan* p = count_plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
    eg_runner* r = NULL;
    struct eg_runner_desc d = runner_desc(&p, 1, 100);
    d.flags = EG_RUN_REORDER;
    d.reorder_theta = 0.5;
    check(p && eg_runner_new(ctx, &d, &r) == EG_BAD_REQUEST && r == NULL && reason_has(ctx, "compacts whole groups"),
          "EG_RUN_REORDER over a unit with groups of lanes refused BAD_REQUEST by name");
    check(eg_plan_free(ctx, p) == EG_OK && eg_unit_free(ctx, unit) == EG_OK, "groups plan and unit freed");
}

static void check_wide(eg_ctx* ctx) {
    enum { M = 1000 };
    static double wi[5 * M], wo[5 * M], want[5 * M], y[M], keep[5 * M];
    for (int i = 0; i < 5 * M; ++i) wi[i] = (double)(i % 13) * 0.5 - 3.0, keep[i] = -7.0;
    eg_unit* unit = NULL;
    check(load(ctx, "widemul", 0x72, &unit) == EG_OK, "eg_unit_load widemul (WIDE_OUT, WIDE_IN)");
    if (unit == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int64_t rows[2] = { 3, 5 };    /* the rows are the caller's: the entry reads three */
    for (int si = 0; si < 2; ++si)
        for (int ri = 0; ri < 2; ++ri) {
            struct tensor two, twi, ty;
            eg_plane *pwo = wrap_nd(ctx, &two, wo, kDLFloat, 64, 2, rows[ri], M, 0),
                     *pwi = wrap_nd(ctx, &twi, wi, kDLFloat, 64, 2, rows[ri], M, 0), *py = wrap(ctx, &ty, y, M, kDLFloat, 64);
            memcpy(wo, keep, sizeof(wo));
            memcpy(want, keep, sizeof(want));
            for (int r = 0; r < 3; ++r)
                for (int i = 0; i < M; ++i) want[r * M + i] = 2.0 * wi[r * M + i] + r;
            const int32_t slots[3] = { 0, 1, 2 };
            eg_plane* planes[3] = { pwo, pwi, py };
            struct eg_bind_desc bd;
            memset(&bd, 0, sizeof(bd));
            bd.struct_size = sizeof(bd);
            bd.n_planes = 3;
            bd.plane_slots = slots;
            bd.planes = planes;
            eg_plan* p = NULL;
            int ok = pwo && pwi && py && plan(ctx, unit, structures[si], M, 0, NULL, NULL, &p) == EG_OK &&
                     eg_bind(ctx, p, &bd) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK;
            for (int i = 0; ok && i < M; ++i) ok = y[i] == (double)M + 1000.0 * (double)M;
            check(ok && memcmp(wo, want, sizeof(wo)) == 0,
                  "wide: WIDE_OUT = 2 * WIDE_IN + row over (rows, n) planes equal the plain loop (bitwise), both handles "
                  "carry the trailing extent, rows past the entry's are untouched");
            eg_plan_free(ctx, p);
            eg_plane_free(ctx, pwo);
            eg_plane_free(ctx, pwi);
            eg_plane_free(ctx, py);
        }
    /* refusals */
    struct tensor tb;
    eg_plan* p = NULL;
    int all = plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, M, 0, NULL, NULL, &p) == EG_OK;
    const int32_t one[1] = { 1 };
    struct eg_bind_desc b1;
    memset(&b1, 0, sizeof(b1));
    b1.struct_size = sizeof(b1);
    b1.n_planes = 1;
    b1.plane_slots = one;
    eg_plane* pbad = wrap_nd(ctx, &tb, wi, kDLFloat, 64, 0, 0, 0, 0);    /* no dimension */
    b1.planes = &pbad;
    all = all && pbad && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "at least one dimension");
    eg_plane_free(ctx, pbad);
    pbad = wrap_nd(ctx, &tb, wo, kDLFloat, 64, 2, 3, M, 0);              /* WIDE_OUT is written */
    tb.managed.flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
    const int32_t zero[1] = { 0 };
    b1.plane_slots = zero;
    b1.planes = &pbad;
    all = all && pbad && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "read-only");
    eg_plane_free(ctx, pbad);
    check(all, "wide refusals: a plane without a dimension, a read-only WIDE_OUT, each BAD_REQUEST by name");
    check(eg_plan_free(ctx, p) == EG_OK && eg_unit_free(ctx, unit) == EG_OK, "wide plan and unit freed");
}

/* A flat wide plane longer than n: every read at e * 3 + c stays inside it (the handle carries the plane's trailing extent, not n). */
static void check_wide_flat(eg_ctx* ctx) {
    enum { N = 512, W = 3 };
    static double theta[N * W], y[N];
    eg_unit* unit = NULL;
    check(load(ctx, "widesum", 0xBB, &unit) == EG_OK, "eg_unit_load widesum (a WIDE_IN read at e * 3 + c)");
    if (unit == NULL) return;
    for (int i = 0; i < N * W; ++i) theta[i] = 0.5 * i;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    int all = 1;
    for (int si = 0; si < 2; ++si) {
        struct tensor tt, ty;
        eg_plane *pt = wrap(ctx, &tt, theta, N * W, kDLFloat, 64), *py = wrap(ctx, &ty, y, N, kDLFloat, 64);
        const int32_t slots[2] = { 0, 1 };
        eg_plane* planes[2] = { py, pt };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 2;
        bd.plane_slots = slots;
        bd.planes = planes;
        eg_plan* p = NULL;
        memset(y, 0, sizeof(y));
        all &= pt && py && plan(ctx, unit, structures[si], N, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
               eg_launch(ctx, p, NULL) == EG_OK;
        for (int e = 0; e < N; ++e)    /* the closed form, from e = 0 through the e >= 171 where an n-element extent would clamp */
            all &= y[e] == 0.5 * (3.0 * e) + 10.0 * 0.5 * (3.0 * e + 1.0) + 100.0 * 0.5 * (3.0 * e + 2.0);
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, pt);
        eg_plane_free(ctx, py);
    }
    check(all, "wide: a flat (1536,) plane at n = 512, W = 3 reads every member at e * 3 + c, e up to 511 (SERIAL and TEAM)");
    eg_unit_free(ctx, unit);
}

/* ---- GATHER and SHARED slots, and a host thread box ---------------------------- */

static void check_gather(eg_ctx* ctx) {
    enum { N = 40, MAXM = 12 };
    static double y[N], m[4 * MAXM], want[N];
    static const struct { const char* name; uint8_t byte; int64_t cols; } units[3] = {
        { "gathb", 0x61, 4 }, { "gatha", 0x62, 10 }, { "gathn", 0x63, 7 } };   /* m: b (4), a (n / 4), any */
    for (int ui = 0; ui < 3; ++ui) {
        eg_unit* unit = NULL;
        const int64_t cols = units[ui].cols;
        struct tensor ty, tm, tbad;
        for (int i = 0; i < 4 * MAXM; ++i) m[i] = (double)(i % 17) * 0.25 - 1.0;
        for (int i = 0; i < N; ++i) {
            double v = 0.0, w = 1.0;
            for (int c = 0; c < 4; ++c, w *= 10.0) v += w * m[c * cols + i % cols];
            want[i] = v;
            y[i] = 0.0;
        }
        eg_plane *py = wrap(ctx, &ty, y, N, kDLFloat, 64), *pm = wrap_nd(ctx, &tm, m, kDLFloat, 64, 3, 2, 2, cols);
        const int32_t slots[2] = { 0, 1 };
        eg_plane* planes[2] = { py, pm };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 2;
        bd.plane_slots = slots;
        bd.planes = planes;
        eg_plan* p = NULL;
        int ok = py && pm && load(ctx, units[ui].name, units[ui].byte, &unit) == EG_OK &&
                 plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, N, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
                 eg_launch(ctx, p, NULL) == EG_OK && memcmp(y, want, sizeof(y)) == 0;
        check(ok, "gather: a compact (2, 2, m) matrix is read at column lane % m, m the columns of the lane axes it varies along "
                  "(static axis, runtime axis, or any m without a pattern), equal the plain loop (bitwise)");
        if (ui < 2) {   /* a column count other than the pattern's */
            eg_plane* pbad = wrap_nd(ctx, &tbad, m, kDLFloat, 64, 3, 2, 2, cols + 1);
            const int32_t one[1] = { 1 };
            eg_plane* bad[1] = { pbad };
            bd.n_planes = 1;
            bd.plane_slots = one;
            bd.planes = bad;
            check(pbad && eg_bind(ctx, p, &bd) == EG_BAD_REQUEST && reason_has(ctx, "is expected (m: the columns"),
                  "gather: a column count other than the lane axes' product refused BAD_REQUEST by name");
            eg_plane_free(ctx, pbad);
        }
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, py);
        eg_plane_free(ctx, pm);
        eg_unit_free(ctx, unit);
    }
}

static void check_shared(eg_ctx* ctx) {
    enum { N = 100 };
    static double y[N], x[N], a[6];
    struct tensor ty, tx, ta;
    for (int i = 0; i < N; ++i) x[i] = (double)i * 0.5 - 3.0;
    for (int i = 0; i < 6; ++i) a[i] = 1.5 + i;
    eg_unit* unit = NULL;
    eg_plane *py = wrap(ctx, &ty, y, N, kDLFloat, 64), *px = wrap(ctx, &tx, x, N, kDLFloat, 64),
             *pa = wrap_nd(ctx, &ta, a, kDLFloat, 64, 2, 2, 2, 0);
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* planes[3] = { py, px, pa };
    struct eg_bind_desc bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 3;
    bd.plane_slots = slots;
    bd.planes = planes;
    eg_plan* p = NULL;
    int ok = py && px && pa && load(ctx, "shr", 0x66, &unit) == EG_OK &&
             plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, N, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
             eg_launch(ctx, p, NULL) == EG_OK;
    for (int i = 0; ok && i < N; ++i) ok = y[i] == x[i] * a[0] + a[3];
    check(ok, "shared: a (2, 2) parameter plane with no lane axis is read flat by every lane (bitwise, N != its extent)");
    struct tensor tb;
    const int32_t two[1] = { 2 };
    int all = 1;
    const int64_t shapes[2][2] = { { 4, 0 }, { 2, 3 } };   /* flat (4,), and (2, 3) */
    for (int k = 0; k < 2; ++k) {
        eg_plane* pb = wrap_nd(ctx, &tb, a, kDLFloat, 64, k == 0 ? 1 : 2, shapes[k][0], shapes[k][1], 0);
        eg_plane* one[1] = { pb };
        struct eg_bind_desc b1 = bd;
        b1.n_planes = 1;
        b1.plane_slots = two;
        b1.planes = one;
        all = all && pb && eg_bind(ctx, p, &b1) == EG_BAD_REQUEST && reason_has(ctx, "shared parameter of its declared shape");
        eg_plane_free(ctx, pb);
    }
    check(all, "shared: a plane of another shape than the declared one refused BAD_REQUEST by name");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, pa);
    eg_unit_free(ctx, unit);
}

/* The box unit over n = 100 lanes (27 threads) under @p desc: y, or NULL when the plan is refused. */
static eg_status box_run(eg_ctx* ctx, eg_unit* unit, const struct eg_plan_desc* desc, double* x, double* y) {
    struct tensor ty, tx;
    eg_plane *py = wrap(ctx, &ty, y, 100, kDLFloat, 64), *px = wrap(ctx, &tx, x, 100, kDLFloat, 64);
    const int32_t slots[2] = { 0, 1 };
    eg_plane* planes[2] = { py, px };
    struct eg_bind_desc bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 2;
    bd.plane_slots = slots;
    bd.planes = planes;
    struct eg_plan_desc d = *desc;
    d.unit = unit;
    eg_plan* p = NULL;
    eg_status s = py && px ? eg_plan_new(ctx, &d, &p) : EG_FAULT;
    if (s == EG_OK) s = eg_bind(ctx, p, &bd);
    if (s == EG_OK) s = eg_launch(ctx, p, NULL);
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, px);
    return s;
}

/* A thread box's partitions count threads: explicit ones and the even split equal the whole-range result bitwise. */
static void check_box_partitions(eg_ctx* ctx, eg_unit* unit) {
    static double x[100], y[100], whole[100];
    for (int i = 0; i < 100; ++i) x[i] = (double)i * 0.25 - 9.0;
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_samples = 100;
    int all = 1;
    for (int si = 0; si < 2; ++si) {
        d.structure = si == 0 ? EG_STRUCTURE_HOST_SERIAL : EG_STRUCTURE_HOST_TEAM;
        for (int i = 0; i < 100; ++i) whole[i] = -5.0;
        all &= box_run(ctx, unit, &d, x, whole) == EG_OK;
        const int64_t one_base[1] = { 0 }, one_count[1] = { 27 };
        const int64_t three_base[3] = { 0, 9, 18 }, three_count[3] = { 9, 9, 9 };
        const int64_t gap_base[2] = { 3, 20 }, gap_count[2] = { 17, 7 };
        const struct { int64_t n; const int64_t *base, *count; uint32_t flags; } cases[4] = {
            { 1, one_base, one_count, 0 }, { 3, three_base, three_count, 0 }, { 3, NULL, NULL, EG_PLAN_SPLIT_EVENLY },
            { 4, NULL, NULL, EG_PLAN_SPLIT_EVENLY } };
        for (int c = 0; c < 4; ++c) {
            d.n_partitions = cases[c].n, d.part_base = cases[c].base, d.part_count = cases[c].count, d.flags = cases[c].flags;
            for (int i = 0; i < 100; ++i) y[i] = -5.0;
            all &= box_run(ctx, unit, &d, x, y) == EG_OK && memcmp(y, whole, sizeof(y)) == 0;
        }
        d.n_partitions = 2, d.part_base = gap_base, d.part_count = gap_count, d.flags = 0;     /* threads [3, 20) and [20, 27) only */
        for (int i = 0; i < 100; ++i) y[i] = -5.0;
        all &= box_run(ctx, unit, &d, x, y) == EG_OK;
        int served = 0;
        for (int i = 0; i < 100; ++i) {
            const int th = (i / 12) * 3 + (i % 12) % 3;       /* the lane's thread */
            served += th >= 3;
            all &= y[i] == (th >= 3 ? whole[i] : -5.0);
        }
        all &= served > 0 && served < 100;
        d.flags = 0, d.n_partitions = 0, d.part_base = d.part_count = NULL;
    }
    check(all, "box: 1 and 3 explicit thread partitions, and EG_PLAN_SPLIT_EVENLY into 3 and 4, equal the whole-range result bitwise "
               "(a partial range runs only its threads; SERIAL and TEAM)");
    eg_plan* p = NULL;
    d.structure = EG_STRUCTURE_HOST_TEAM, d.unit = unit;
    const int64_t over_base[1] = { 0 }, over_count[1] = { 28 }, late_base[1] = { 27 }, late_count[1] = { 1 };
    d.n_partitions = 1, d.part_base = over_base, d.part_count = over_count;
    check(eg_plan_new(ctx, &d, &p) == EG_BAD_REQUEST && reason_has(ctx, "thread range [0, 27)") && reason_has(ctx, "4 lanes per thread, stride 3"),
          "box: a thread partition past the thread total refused BAD_REQUEST, naming the unit, t, stride and the total");
    d.part_base = late_base, d.part_count = late_count;
    check(eg_plan_new(ctx, &d, &p) == EG_BAD_REQUEST && reason_has(ctx, "thread range [0, 27)"), "box: a thread partition starting at the total refused BAD_REQUEST");
    const int64_t lap_base[2] = { 0, 4 }, lap_count[2] = { 5, 5 };
    d.n_partitions = 2, d.part_base = lap_base, d.part_count = lap_count;
    check(eg_plan_new(ctx, &d, &p) == EG_BAD_REQUEST && reason_has(ctx, "overlap"), "box: overlapping thread partitions refused BAD_REQUEST");
}

/* EG_PLAN_SPLIT_EVENLY cuts a range as eagle's Python plan does; its refusals by name. */
static void check_even_split(eg_ctx* ctx) {
    void* lib = dlopen(library_path, RTLD_LAZY | RTLD_NOLOAD);
    int64_t* spans = lib ? (int64_t*)dlsym(lib, "fixture_spans") : NULL;
    int64_t* span_n = lib ? (int64_t*)dlsym(lib, "fixture_span_n") : NULL;
    eg_unit* unit = NULL;
    check(spans != NULL && span_n != NULL && load(ctx, "spans", 0xB9, &unit) == EG_OK, "eg_unit_load spans");
    if (spans == NULL || span_n == NULL || unit == NULL) return;
    /* the Python plan's (n, parts) splits: ceil(remaining / left) each, the remainder spread early; empty parts launch nothing */
    static const int64_t want_10_3[] = { 0, 4, 4, 3, 7, 3 };
    static const int64_t want_7_7[] = { 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1 };
    static const int64_t want_1_4[] = { 0, 1 };
    const struct { int64_t n, parts, calls; const int64_t* want; } cases[3] = {
        { 10, 3, 3, want_10_3 }, { 7, 7, 7, want_7_7 }, { 1, 4, 1, want_1_4 } };
    int all = 1;
    for (int c = 0; c < 3; ++c) {
        int64_t cell = 0;
        struct tensor t;
        eg_plane* pc = wrap(ctx, &t, &cell, 1, kDLInt, 64);
        struct eg_plan_desc d;
        memset(&d, 0, sizeof(d));
        d.struct_size = sizeof(d);
        d.unit = unit;
        d.structure = EG_STRUCTURE_HOST_SERIAL;
        d.n_samples = cases[c].n;
        d.n_partitions = cases[c].parts;
        d.flags = EG_PLAN_SPLIT_EVENLY;
        eg_plan* p = NULL;
        const int32_t slot = 0;
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 1;
        bd.plane_slots = &slot;
        bd.planes = &pc;
        *span_n = 0;
        uint64_t tiles = 0;
        all &= pc && eg_plan_new(ctx, &d, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK && eg_launch(ctx, p, &tiles) == EG_OK &&
               *span_n == cases[c].calls && tiles == (uint64_t)cases[c].calls &&
               memcmp(spans, cases[c].want, 2 * sizeof(int64_t) * (size_t)cases[c].calls) == 0;
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, pc);
    }
    check(all, "EG_PLAN_SPLIT_EVENLY: the bases and counts of the Python plan's even split for (n, parts) = (10, 3), (7, 7), (1, 4)");
    eg_plan* p = NULL;
    const int64_t base[1] = { 0 }, count[1] = { 4 };
    struct eg_plan_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.unit = unit;
    d.structure = EG_STRUCTURE_HOST_SERIAL;
    d.n_samples = 10;
    d.flags = EG_PLAN_SPLIT_EVENLY;
    d.n_partitions = 1, d.part_base = base, d.part_count = count;
    check(eg_plan_new(ctx, &d, &p) == EG_BAD_REQUEST && reason_has(ctx, "takes no part_base or part_count"),
          "EG_PLAN_SPLIT_EVENLY with part_base and part_count refused BAD_REQUEST by name");
    d.part_base = d.part_count = NULL;
    d.n_partitions = 0;
    check(eg_plan_new(ctx, &d, &p) == EG_BAD_REQUEST && reason_has(ctx, "needs n_partitions >= 1"),
          "EG_PLAN_SPLIT_EVENLY with n_partitions 0 refused BAD_REQUEST by name");
    d.flags = 1u << 1, d.n_partitions = 1;
    check(eg_plan_new(ctx, &d, &p) == EG_BAD_REQUEST && reason_has(ctx, "unknown bits"), "an unknown plan flag bit refused BAD_REQUEST");
    eg_unit_free(ctx, unit);
}

static void check_box(eg_ctx* ctx) {
    void* lib = dlopen(library_path, RTLD_LAZY | RTLD_NOLOAD);
    int64_t* threads_run = lib ? (int64_t*)dlsym(lib, "fixture_box_threads") : NULL;
    eg_unit* unit = NULL;
    check(threads_run != NULL && load(ctx, "boxed", 0x64, &unit) == EG_OK, "eg_unit_load boxed (a thread box of 4 lanes 3 apart)");
    if (threads_run == NULL || unit == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    const int64_t sizes[4] = { 1, 12, 1000, 1001 };
    int all = 1;
    for (int si = 0; si < 2; ++si)
        for (int zi = 0; zi < 4; ++zi) {
            enum { MAXN = 1001 };
            static double x[MAXN], y[MAXN];
            const int64_t n = sizes[zi];
            struct tensor ty, tx;
            for (int64_t i = 0; i < n; ++i) x[i] = (double)i * 0.25 - 9.0, y[i] = -5.0;
            eg_plane *py = wrap(ctx, &ty, y, n, kDLFloat, 64), *px = wrap(ctx, &tx, x, n, kDLFloat, 64);
            const int32_t slots[2] = { 0, 1 };
            eg_plane* planes[2] = { py, px };
            struct eg_bind_desc bd;
            memset(&bd, 0, sizeof(bd));
            bd.struct_size = sizeof(bd);
            bd.n_planes = 2;
            bd.plane_slots = slots;
            bd.planes = planes;
            eg_plan* p = NULL;
            eg_host_threads(ctx, 3);
            *threads_run = 0;
            all &= py && px && plan(ctx, unit, structures[si], n, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
                   eg_launch(ctx, p, NULL) == EG_OK;
            all &= *threads_run == (n + 11) / 12 * 3;          /* ceil(n / (4 * 3)) * 3 threads */
            for (int64_t i = 0; i < n; ++i) all &= y[i] == 2.0 * x[i] + 1.0;
            eg_plan_free(ctx, p);
            eg_plane_free(ctx, py);
            eg_plane_free(ctx, px);
        }
    check(all, "box: ceil(n / (T * stride)) * stride threads run, and every lane is served once (y = 2x + 1 for n = 1, 12, 1000, 1001, "
               "SERIAL and TEAM)");
    check_box_partitions(ctx, unit);
    eg_unit* bad = NULL;
    check(load(ctx, "boxbad", 0x65, &bad) == EG_BAD_REQUEST && reason_has(ctx, "host_box needs"),
          "box: a thread box of fewer than 2 lanes refused BAD_REQUEST by name");
    eg_unit_free(ctx, unit);
}

/* cell rows of 4 samples folded by their op, and a keyed accumulate with an eagle-made companion folded after each launch */
static void check_made(eg_ctx* ctx) {
    static double x[10], tot[1], mx[1], acc[4], xk[12];
    eg_unit* cellsum = NULL;
    struct tensor tx, tt, tm, ta, tk;
    eg_unit *cells = NULL, *keyed = NULL, *bad = NULL;
    check(load(ctx, "cells", 0x67, &cells) == EG_OK && load(ctx, "keyed", 0x68, &keyed) == EG_OK, "eg_unit_load cells and keyed");
    check(load(ctx, "keyedbad", 0x69, &bad) == EG_NOT_WRITTEN && reason_has(ctx, "no host fold entry"),
          "companion: a unit whose companion has no host fold entry refused NOT_WRITTEN by name");
    check(load(ctx, "cellsum", 0x6a, &cellsum) == EG_NOT_WRITTEN && reason_has(ctx, "compensation slot is not stated"),
          "cells: a SUM cell, whose compensation slot the meta does not state, refused NOT_WRITTEN by name");
    if (cells == NULL || keyed == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_TEAM, EG_STRUCTURE_HOST_SERIAL };
    int all = 1;
    for (int si = 0; si < 2; ++si) {
        eg_plane *px = wrap(ctx, &tx, x, 10, kDLFloat, 64), *pt = wrap(ctx, &tt, tot, 1, kDLFloat, 64),
                 *pm = wrap(ctx, &tm, mx, 1, kDLFloat, 64);
        const int32_t slots[3] = { 2, 0, 1 };
        eg_plane* planes[3] = { px, pt, pm };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 3;
        bd.plane_slots = slots;
        bd.planes = planes;
        eg_plan* p = NULL;
        tot[0] = mx[0] = -1.0;
        for (int i = 0; i < 10; ++i) x[i] = (double)(i + 1);
        all &= px && pt && pm && plan(ctx, cells, structures[si], 10, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
               eg_launch(ctx, p, NULL) == EG_OK;
        all &= tot[0] == 1.0 && mx[0] == 10.0;          /* the min of the row minima 1, 5, 9; the max of the row maxima 4, 8, 10 */
        for (int i = 0; i < 10; ++i) x[i] = (double)(2 * i + 2);
        all &= eg_launch(ctx, p, NULL) == EG_OK && tot[0] == 2.0 && mx[0] == 20.0;    /* a second launch refolds the rows */
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, px);
        eg_plane_free(ctx, pt);
        eg_plane_free(ctx, pm);
    }
    check(all, "cells: rows of 4 samples folded into the cells by their op after each launch (MIN 1, MAX 10; "
               "again 2 and 20; SERIAL and TEAM)");
    {
        double two[2] = { 0.0, 0.0 };
        eg_plane *px = wrap(ctx, &tx, x, 10, kDLFloat, 64), *pt = wrap(ctx, &tt, tot, 1, kDLFloat, 64),
                 *p2 = wrap(ctx, &tm, two, 2, kDLFloat, 64);
        eg_plan* p = NULL;
        const int64_t base[1] = { 2 }, count[1] = { 4 };
        const int32_t s1[2] = { 2, 0 }, s_two[1] = { 0 };
        eg_plane* both[2] = { px, pt };
        eg_plane* one_two[1] = { p2 };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 2;
        bd.plane_slots = s1;
        bd.planes = both;
        int ok = px && pt && p2 && plan(ctx, cells, EG_STRUCTURE_HOST_TEAM, 10, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK;
        ok = ok && eg_launch(ctx, p, NULL) == EG_BAD_REQUEST && reason_has(ctx, "'mx') is unbound");
        check(ok, "cells: a cell the caller left unbound refused BAD_REQUEST by name at the launch");
        bd.n_planes = 1;
        bd.plane_slots = s_two;
        bd.planes = one_two;
        ok = eg_bind(ctx, p, &bd) == EG_BAD_REQUEST && reason_has(ctx, "shape (1,)");
        check(ok, "cells: a cell of another shape than (1,) refused BAD_REQUEST by name");
        eg_plan* q = NULL;
        check(plan(ctx, cells, EG_STRUCTURE_HOST_TEAM, 10, 1, base, count, &q) == EG_BAD_REQUEST && reason_has(ctx, "rows of 4"),
              "cells: a partition that starts inside a row refused BAD_REQUEST by name");
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, px);
        eg_plane_free(ctx, pt);
        eg_plane_free(ctx, p2);
    }
    all = 1;
    for (int si = 0; si < 2; ++si) {
        eg_plane *px = wrap(ctx, &tk, xk, 12, kDLFloat, 64), *pa = wrap_nd(ctx, &ta, acc, kDLFloat, 64, 1, 3, 0, 0);
        const int32_t slots[2] = { 2, 0 };
        eg_plane* planes[2] = { px, pa };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 2;
        bd.plane_slots = slots;
        bd.planes = planes;
        eg_plan* p = NULL;
        for (int i = 0; i < 12; ++i) xk[i] = (double)(i + 1);
        acc[0] = acc[1] = acc[2] = 0.0, acc[3] = 99.0;
        all &= px && pa && plan(ctx, keyed, structures[si], 12, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
               eg_launch(ctx, p, NULL) == EG_OK;
        for (int k = 0; k < 3; ++k) all &= acc[k] == (double)(16 * k + 10) + 2.0;    /* the fold added the companion's 4 x 0.5 */
        all &= eg_launch(ctx, p, NULL) == EG_OK;
        for (int k = 0; k < 3; ++k) all &= acc[k] == 2.0 * (double)(16 * k + 10) + 4.125;   /* + 4 x 0.5 on the residual 0.125 the fold left */
        all &= acc[3] == 99.0;                                                          /* the fold stays inside the target */
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, px);
        eg_plane_free(ctx, pa);
    }
    check(all, "companion: eagle makes the zeroed companion with the target, runs the unit's fold entry over the target's 3 slots after each "
               "launch (SERIAL and TEAM, two launches, the 4th element untouched)");
    {
        eg_plane *px = wrap(ctx, &tk, xk, 12, kDLFloat, 64), *pa = wrap(ctx, &ta, acc, 4, kDLFloat, 64);
        const int32_t slots[2] = { 2, 0 };
        eg_plane* planes[2] = { px, pa };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 2;
        bd.plane_slots = slots;
        bd.planes = planes;
        eg_plan* p = NULL;
        check(px && pa && plan(ctx, keyed, EG_STRUCTURE_HOST_TEAM, 12, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_BAD_REQUEST &&
                  reason_has(ctx, "the unit's named axis has 3"),
              "companion: a plane whose key axis is not the lane axis's extent (4, not 12 / 4) refused BAD_REQUEST by name");
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, px);
        eg_plane_free(ctx, pa);
    }
    eg_unit_free(ctx, cells);
    eg_unit_free(ctx, keyed);
}

/* eg_clear: an output and its eagle-made companion restart together; every refusal by status and reason. */
static void check_clear(eg_ctx* ctx, eg_unit* axpb) {
    static double x[12], acc[4], y[8], xs[8];
    struct tensor tx, ta, ty, txs;
    eg_unit* keyed = NULL;
    check(load(ctx, "keyed", 0x68, &keyed) == EG_OK, "eg_unit_load keyed for eg_clear");
    if (keyed == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    int all = 1;
    for (int si = 0; si < 2; ++si) {
        eg_plane *px = wrap(ctx, &tx, x, 12, kDLFloat, 64), *pa = wrap_nd(ctx, &ta, acc, kDLFloat, 64, 1, 3, 0, 0);
        const int32_t slots[2] = { 2, 0 }, out[1] = { 0 };
        eg_plane* planes[2] = { px, pa };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 2;
        bd.plane_slots = slots;
        bd.planes = planes;
        eg_plan* p = NULL;
        for (int i = 0; i < 12; ++i) x[i] = (double)(i + 1);
        acc[0] = acc[1] = acc[2] = 0.0, acc[3] = 99.0;
        all &= px && pa && plan(ctx, keyed, structures[si], 12, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK &&
               eg_launch(ctx, p, NULL) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK;
        all &= eg_clear(ctx, p, 1, out) == EG_OK;
        for (int k = 0; k < 3; ++k) all &= acc[k] == 0.0;
        all &= acc[3] == 99.0;    /* the plane ends at its third element */
        all &= eg_launch(ctx, p, NULL) == EG_OK;
        /* the values of the first launch of a fresh plan (a companion left at 0.125 would add 0.125 x 3 elements' worth) */
        for (int k = 0; k < 3; ++k) all &= acc[k] == (double)(16 * k + 10) + 2.0;
        all &= eg_clear(ctx, p, 0, NULL) == EG_OK && acc[0] == (double)10 + 2.0;    /* n_slots == 0 changes nothing */
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, px);
        eg_plane_free(ctx, pa);
    }
    check(all, "eg_clear: zeroes the output in place and restarts its companion (SERIAL and TEAM: a launch after the clear == the first launch of a fresh plan); "
               "n_slots 0 is a no-op");
    {
        eg_plane *px = wrap(ctx, &txs, xs, 8, kDLFloat, 64), *py = wrap(ctx, &ty, y, 8, kDLFloat, 64);
        const double a = 1.0, b = 0.0;
        eg_plan* p = NULL;
        eg_plan* kp = NULL;
        int ok = px && py && plan(ctx, axpb, EG_STRUCTURE_HOST_SERIAL, 8, 0, NULL, NULL, &p) == EG_OK &&
                 bind(ctx, p, py, px, &a, &b) == EG_OK;
        const int32_t s_in[1] = { 1 }, s_unif[1] = { 2 }, s_none[1] = { 99 }, s_neg[1] = { -1 }, s_twice[2] = { 0, 0 };
        ok = ok && eg_clear(ctx, p, -1, s_in) == EG_BAD_REQUEST && reason_has(ctx, "n_slots without its array");
        ok = ok && eg_clear(ctx, p, 1, NULL) == EG_BAD_REQUEST && reason_has(ctx, "n_slots without its array");
        ok = ok && eg_clear(ctx, p, 1, s_none) == EG_BAD_REQUEST && reason_has(ctx, "unknown slot 99");
        ok = ok && eg_clear(ctx, p, 1, s_neg) == EG_BAD_REQUEST && reason_has(ctx, "unknown slot -1");
        ok = ok && eg_clear(ctx, p, 1, s_in) == EG_BAD_REQUEST && reason_has(ctx, "'x') is not an output plane");
        ok = ok && eg_clear(ctx, p, 1, s_unif) == EG_BAD_REQUEST && reason_has(ctx, "'a') is not an output plane");
        ok = ok && eg_clear(ctx, p, 2, s_twice) == EG_BAD_REQUEST && reason_has(ctx, "'y') cleared twice");
        ok = ok && plan(ctx, keyed, EG_STRUCTURE_HOST_SERIAL, 12, 0, NULL, NULL, &kp) == EG_OK;
        const int32_t s_comp[1] = { 1 }, s_acc[1] = { 0 };
        ok = ok && eg_clear(ctx, kp, 1, s_comp) == EG_BAD_REQUEST && reason_has(ctx, "'acc_c') is a companion or compensation plane");
        ok = ok && eg_clear(ctx, kp, 1, s_acc) == EG_BAD_REQUEST && reason_has(ctx, "'acc') is unbound");
        ok = ok && eg_record_begin(ctx) == EG_OK && eg_clear(ctx, p, 0, NULL) == EG_BAD_REQUEST && reason_has(ctx, "eg_clear while recording");
        eg_graph* g = NULL;
        ok = ok && eg_record_end(ctx, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK;
        check(ok, "eg_clear: n_slots < 0, slots NULL, a slot out of range, an input, a uniform, a companion, an unbound slot, the same slot twice "
                  "and a call while recording each refused BAD_REQUEST by reason");
        eg_plan_free(ctx, p);
        eg_plan_free(ctx, kp);
        eg_plane_free(ctx, px);
        eg_plane_free(ctx, py);
    }
    eg_unit_free(ctx, keyed);
}

/* eg_set_count: a prefix set and a lookup set; the lanes below the count take the new values, the others keep theirs. */
static void check_set_count(eg_ctx* ctx, eg_unit* axpb) {
    enum { SM = 64, SCAP = 16 };
    static double y[SM], x[SM], want[SM];
    static int32_t map[SCAP];
    static uint32_t count;
    struct tensor ty, tx, tm, tc;
    eg_unit *up = NULL, *ul = NULL;
    check(load(ctx, "prefix", 0x92, &up) == EG_OK && load(ctx, "lanes", 0x91, &ul) == EG_OK, "eg_unit_load prefix and lanes for eg_set_count");
    if (up == NULL || ul == NULL) return;
    for (int i = 0; i < SM; ++i) x[i] = (double)i * 0.5 - 3.0;
    for (int i = 0; i < SCAP; ++i) map[i] = (i * 5 + 3) % SM;
    eg_plane *py = wrap(ctx, &ty, y, SM, kDLFloat, 64), *px = wrap(ctx, &tx, x, SM, kDLFloat, 64),
             *pm = wrap(ctx, &tm, map, SCAP, kDLInt, 32), *pc = wrap(ctx, &tc, &count, 1, kDLUInt, 32);
    if (!(py && px && pm && pc)) {
        check(0, "eg_set_count planes wrapped");
        return;
    }
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    int all = 1;
    for (int si = 0; si < 2; ++si) {
        eg_plan* p = NULL;
        const int32_t slots[3] = { 0, 1, 2 };
        eg_plane* planes[3] = { py, px, pc };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 3;
        bd.plane_slots = slots;
        bd.planes = planes;
        all &= plan(ctx, up, structures[si], SM, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK;
        for (int i = 0; i < SM; ++i) y[i] = -1.0;
        all &= eg_set_count(ctx, p, SM) == EG_OK && count == SM && eg_launch(ctx, p, NULL) == EG_OK;
        for (int i = 0; i < SM; ++i) want[i] = 2.0 * x[i] + 1.0;
        all &= memcmp(y, want, sizeof(y)) == 0;
        for (int i = 0; i < SM; ++i) x[i] = -x[i] - 7.0;
        all &= eg_set_count(ctx, p, 21) == EG_OK && count == 21 && eg_launch(ctx, p, NULL) == EG_OK;
        for (int i = 0; i < 21; ++i) want[i] = 2.0 * x[i] + 1.0;    /* the lanes from 21 keep the first launch's values */
        all &= memcmp(y, want, sizeof(y)) == 0;
        for (int i = 0; i < SM; ++i) x[i] += 100.0;
        all &= eg_set_count(ctx, p, 0) == EG_OK && count == 0 && eg_launch(ctx, p, NULL) == EG_OK && memcmp(y, want, sizeof(y)) == 0;
        all &= eg_set_count(ctx, p, SM + 1) == EG_BAD_REQUEST && reason_has(ctx, "above the sample count's") && count == 0;
        eg_plan_free(ctx, p);
        for (int i = 0; i < SM; ++i) x[i] = (double)i * 0.5 - 3.0;
    }
    check(all, "eg_set_count on a prefix set: lanes below the count take the new values, the others keep theirs bitwise, 0 runs nothing, "
               "a count above the sample count is refused BAD_REQUEST and leaves the cell (SERIAL and TEAM)");
    all = 1;
    for (int si = 0; si < 2; ++si) {    /* a list recorded at one count replays at each later count: the kernel reads the count plane at replay */
        eg_plan* p = NULL;
        eg_graph* g = NULL;
        const uint32_t counts[4] = { 40, 10, 0, SM };
        const int32_t slots[3] = { 0, 1, 2 };
        eg_plane* planes[3] = { py, px, pc };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 3;
        bd.plane_slots = slots;
        bd.planes = planes;
        all &= plan(ctx, up, structures[si], SM, 0, NULL, NULL, &p) == EG_OK && eg_bind(ctx, p, &bd) == EG_OK && eg_set_count(ctx, p, counts[0]) == EG_OK &&
               eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, &g) == EG_OK;
        for (int k = 0; k < 4 && all; ++k) {
            all &= k == 0 || eg_set_count(ctx, p, counts[k]) == EG_OK;
            for (int i = 0; i < SM; ++i) y[i] = -1.0;
            all &= eg_graph_launch(ctx, g) == EG_OK;
            for (int i = 0; i < SM; ++i) want[i] = (uint32_t)i < counts[k] ? 2.0 * x[i] + 1.0 : -1.0;
            all &= memcmp(y, want, sizeof(y)) == 0;
        }
        eg_graph_free(ctx, g);
        eg_plan_free(ctx, p);
    }
    check(all, "eg_set_count on a prefix set recorded at count 40: the replays at 40, 10, 0 and the capacity 64 run at the count last written "
               "with no new recording, bitwise == the plain-C values (SERIAL and TEAM)");
    {
        eg_plan *p = NULL, *q = NULL, *r = NULL, *l = NULL;
        const int32_t s_part[3] = { 0, 1, 3 }, s_lane[4] = { 0, 1, 2, 3 };
        eg_plane* part[3] = { py, px, pc };
        eg_plane* lane[4] = { py, px, pm, pc };
        struct eg_bind_desc bd;
        memset(&bd, 0, sizeof(bd));
        bd.struct_size = sizeof(bd);
        bd.n_planes = 3;
        bd.plane_slots = s_part;
        bd.planes = part;
        count = 5;
        int ok = plan(ctx, ul, EG_STRUCTURE_HOST_TEAM, SM, 0, NULL, NULL, &q) == EG_OK;
        ok = ok && eg_set_count(ctx, q, 1) == EG_BAD_REQUEST && reason_has(ctx, "is a lookup set");
        bd.n_planes = 4;
        bd.plane_slots = s_lane;
        bd.planes = lane;
        ok = ok && plan(ctx, ul, EG_STRUCTURE_HOST_TEAM, SM, 0, NULL, NULL, &l) == EG_OK && eg_bind(ctx, l, &bd) == EG_OK;
        ok = ok && eg_set_count(ctx, l, 1) == EG_BAD_REQUEST && reason_has(ctx, "is a lookup set: its count is the caller's plane, bind it again") && count == 5;
        const int32_t s_p[3] = { 0, 1, 2 };
        bd.n_planes = 3;
        bd.plane_slots = s_p;
        bd.planes = part;
        ok = ok && plan(ctx, up, EG_STRUCTURE_HOST_SERIAL, SM, 0, NULL, NULL, &p) == EG_OK;
        ok = ok && eg_set_count(ctx, p, 1) == EG_BAD_REQUEST && reason_has(ctx, "('index_count') is unbound");
        ok = ok && plan(ctx, axpb, EG_STRUCTURE_HOST_SERIAL, SM, 0, NULL, NULL, &r) == EG_OK;
        ok = ok && eg_set_count(ctx, r, 1) == EG_BAD_REQUEST && reason_has(ctx, "has no index count slot");
        ok = ok && eg_bind(ctx, p, &bd) == EG_OK && eg_record_begin(ctx) == EG_OK && eg_set_count(ctx, p, 1) == EG_BAD_REQUEST && reason_has(ctx, "eg_set_count while recording");
        eg_graph* g = NULL;
        ok = ok && eg_record_end(ctx, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK;
        check(ok, "eg_set_count: a lookup set (bound or not), a unit with no index count slot, an unbound count slot and a call while "
                  "recording each refused BAD_REQUEST by reason");
        eg_plan_free(ctx, p);
        eg_plan_free(ctx, q);
        eg_plan_free(ctx, r);
        eg_plan_free(ctx, l);
    }
    eg_plane_free(ctx, py);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, pm);
    eg_plane_free(ctx, pc);
    eg_unit_free(ctx, up);
    eg_unit_free(ctx, ul);
}

/* EG_UNIT_CALLER_CARDS: a unit lowered under caller-stated cards loads as any other; an unknown unit flag is refused by name. */
static void check_unit_flags(eg_ctx* ctx) {
    eg_unit* cards = NULL;
    eg_unit* bad = NULL;
    check(load(ctx, "axpb_cards", 0xD4, &cards) == EG_OK && cards != NULL, "a unit with EG_UNIT_CALLER_CARDS loads");
    check(load(ctx, "axpb_badflag", 0xD5, &bad) == EG_NOT_WRITTEN && bad == NULL && reason_has(ctx, "unknown flag bits 2"),
          "a unit flag bit eagle does not know is refused NOT_WRITTEN, naming the bits");
    eg_unit_free(ctx, cards);
}

/* eg_banded_*: the cells of known values are aether's own words (0, 1, -1, 0.5, 3 and 2^99 + a fraction); the band is [2^-94, 2^126). */
static void check_banded(eg_ctx* ctx) {
    const double in[5] = {0.0, 1.0, -1.0, 0.5, 3.0};
    const uint64_t want[5] = {0x0000000000000000ull, 0x3f80000000000000ull, 0xbf80000000000000ull, 0x3f00000000000000ull, 0x4040000000000000ull};
    uint64_t cells[5], one = 7, sentinel = 0x1234;
    double back[5];
    int ok = eg_banded_pack(ctx, in, cells, 5) == EG_OK && memcmp(cells, want, sizeof(want)) == 0;
    ok = ok && eg_banded_unpack(ctx, want, back, 5) == EG_OK && memcmp(back, in, sizeof(in)) == 0;
    ok = ok && eg_banded_pack_scalar(ctx, -1.0, &one) == EG_OK && one == want[2];
    check(ok, "eg_banded_pack / unpack / pack_scalar: known cells bitwise and the round trip");
    const double wide = 0x1.93e5939a08ceap+99;
    uint64_t wide_cell = 0;
    check(eg_banded_pack(ctx, &wide, &wide_cell, 1) == EG_OK && wide_cell == 0x7149f2c9cd046750ull &&
              eg_banded_unpack(ctx, &wide_cell, back, 1) == EG_OK && back[0] == wide,
          "eg_banded_pack: a value near the top of the band keeps its cell and its value");
    const double bad[3] = {1.0, 1e-30, 1e40};
    cells[1] = sentinel;
    check(eg_banded_pack(ctx, bad, cells, 3) == EG_BAD_REQUEST && reason_has(ctx, "element 1") && reason_has(ctx, "outside tier 1's exact window"),
          "eg_banded_pack refuses the first value outside the band by element and reason");
    one = sentinel;
    check(eg_banded_pack_scalar(ctx, 1e40, &one) == EG_BAD_REQUEST && reason_has(ctx, "outside tier 1's exact window") && one == sentinel,
          "eg_banded_pack_scalar refuses a value outside the band and leaves out as it was");
    check(eg_banded_pack(ctx, NULL, NULL, 0) == EG_OK && eg_banded_unpack(ctx, NULL, NULL, 0) == EG_OK, "eg_banded_pack / unpack with n == 0 are OK");
    check(eg_banded_pack(ctx, NULL, cells, 1) == EG_BAD_REQUEST && reason_has(ctx, "in is NULL") &&
              eg_banded_pack(ctx, in, NULL, 1) == EG_BAD_REQUEST && reason_has(ctx, "out is NULL") &&
              eg_banded_unpack(ctx, NULL, back, 1) == EG_BAD_REQUEST && reason_has(ctx, "in is NULL") &&
              eg_banded_unpack(ctx, want, NULL, 1) == EG_BAD_REQUEST && reason_has(ctx, "out is NULL") &&
              eg_banded_pack_scalar(ctx, 1.0, NULL) == EG_BAD_REQUEST && reason_has(ctx, "out is NULL"),
          "eg_banded_*: NULL with n > 0 is BAD_REQUEST by reason");
}

/* eg_plane_new / eg_plane_dlpack: the made plane's export fields, its release, a launch on made planes against caller planes, every refusal by reason. */
static void check_alloc(eg_ctx* ctx, eg_unit* axpb) {
    enum { N = 8 };
    static double xc[N], yc[N];
    struct tensor txc, tyc;
    const int64_t shape[2] = { 2, 3 };
    struct eg_plane_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.kind = EG_MEM_HOST;
    d.ndim = 2;
    d.dtype_code = kDLFloat;
    d.dtype_bits = 64;
    d.shape = shape;
    d.flags = EG_PLANE_ZERO;
    eg_plane* p = NULL;
    struct DLManagedTensorVersioned* m = NULL;
    int ok = eg_plane_new(ctx, &d, &p) == EG_OK && p != NULL && eg_plane_dlpack(ctx, p, &m) == EG_OK && m != NULL;
    if (ok) {
        const DLTensor* t = &m->dl_tensor;
        const double* v = (const double*)t->data;
        ok = m->version.major == DLPACK_MAJOR_VERSION && t->data != NULL && t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3 &&
             t->dtype.code == kDLFloat && t->dtype.bits == 64 && t->dtype.lanes == 1 && t->device.device_type == kDLCPU &&
             (t->strides == NULL || (t->strides[0] == 3 && t->strides[1] == 1));
        for (int i = 0; i < 6; ++i) ok = ok && v[i] == 0.0;
    }
    check(ok, "eg_plane_new HOST with EG_PLANE_ZERO: the export has its data, shape (2, 3), float64, kDLCPU, dense strides and zeroed bytes");
    struct DLManagedTensorVersioned* again = NULL;
    ok = ok && eg_plane_dlpack(ctx, p, &again) == EG_OK && again != NULL && again->dl_tensor.data == m->dl_tensor.data;
    if (m != NULL) m->deleter(m);
    if (again != NULL) again->deleter(again);
    check(ok && eg_plane_free(ctx, p) == EG_OK, "eg_plane_dlpack twice gives one memory; both exports released, then the plane freed");

    const struct { uint8_t code, bits; } dtypes[7] = { { kDLFloat, 64 }, { kDLFloat, 32 }, { kDLInt, 64 }, { kDLInt, 32 },
                                                       { kDLInt, 8 },    { kDLUInt, 32 },  { kDLBool, 8 } };
    ok = 1;
    for (int i = 0; i < 7; ++i) {
        eg_plane* q = NULL;
        struct DLManagedTensorVersioned* e = NULL;
        d.dtype_code = dtypes[i].code;
        d.dtype_bits = dtypes[i].bits;
        ok = ok && eg_plane_new(ctx, &d, &q) == EG_OK && eg_plane_dlpack(ctx, q, &e) == EG_OK && e->dl_tensor.dtype.code == dtypes[i].code &&
             e->dl_tensor.dtype.bits == dtypes[i].bits;
        if (e != NULL) e->deleter(e);
        eg_plane_free(ctx, q);
    }
    d.dtype_code = kDLFloat;
    d.dtype_bits = 64;
    check(ok, "eg_plane_new takes each dtype eg_plane_wrap takes (float64/32, int64/32/8, uint32, bool)");

    {   /* the memory outlives eg_plane_free and eg_ctx_free while an export is held */
        eg_ctx* other = make_ctx();
        eg_plane* q = NULL;
        struct DLManagedTensorVersioned* e = NULL;
        ok = other != NULL && eg_plane_new(other, &d, &q) == EG_OK && eg_plane_dlpack(other, q, &e) == EG_OK && eg_plane_free(other, q) == EG_OK;
        if (ok) ((double*)e->dl_tensor.data)[5] = 42.5;
        ok = ok && eg_ctx_free(other) == EG_OK && ((double*)e->dl_tensor.data)[5] == 42.5;
        if (e != NULL) e->deleter(e);
        check(ok, "a plane freed while exported keeps its memory, also across eg_ctx_free, until the export is released");
    }

    {   /* a launch on made planes == the same launch on caller planes, bitwise */
        eg_plane *pxc = wrap(ctx, &txc, xc, N, kDLFloat, 64), *pyc = wrap(ctx, &tyc, yc, N, kDLFloat, 64);
        eg_plane *pxm = NULL, *pym = NULL;
        struct DLManagedTensorVersioned *exm = NULL, *eym = NULL;
        const int64_t one[1] = { N };
        d.ndim = 1;
        d.shape = one;
        const double a = 1.25, b = -0.5;
        ok = pxc && pyc && eg_plane_new(ctx, &d, &pxm) == EG_OK && eg_plane_new(ctx, &d, &pym) == EG_OK && eg_plane_dlpack(ctx, pxm, &exm) == EG_OK &&
             eg_plane_dlpack(ctx, pym, &eym) == EG_OK;
        for (int i = 0; ok && i < N; ++i) xc[i] = ((double*)exm->dl_tensor.data)[i] = 0.1 * (i + 1);
        const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
        for (int si = 0; ok && si < 2; ++si) {
            eg_plan *pc = NULL, *pm = NULL;
            ok = plan(ctx, axpb, structures[si], N, 0, NULL, NULL, &pc) == EG_OK && bind(ctx, pc, pyc, pxc, &a, &b) == EG_OK &&
                 eg_launch(ctx, pc, NULL) == EG_OK && plan(ctx, axpb, structures[si], N, 0, NULL, NULL, &pm) == EG_OK &&
                 bind(ctx, pm, pym, pxm, &a, &b) == EG_OK && eg_launch(ctx, pm, NULL) == EG_OK &&
                 memcmp(yc, eym->dl_tensor.data, sizeof(yc)) == 0 && yc[N - 1] != 0.0;
            eg_plan_free(ctx, pc);
            eg_plan_free(ctx, pm);
        }
        check(ok, "a plan bound to eg_plane_new planes and launched (SERIAL and TEAM) writes the bytes a plan on caller planes writes");
        if (exm != NULL) exm->deleter(exm);
        if (eym != NULL) eym->deleter(eym);
        eg_plane_free(ctx, pxc);
        eg_plane_free(ctx, pyc);
        eg_plane_free(ctx, pxm);
        eg_plane_free(ctx, pym);
    }

    {   /* every refusal, by status and reason */
        eg_plane* q = NULL;
        const int64_t neg[2] = { 4, -1 }, big[2] = { INT64_MAX / 2, 4 }, good[1] = { 4 };
        struct eg_plane_desc r = d;
        r.ndim = 1;
        r.shape = good;
        ok = eg_plane_new(ctx, &r, NULL) == EG_BAD_REQUEST && reason_has(ctx, "out is NULL");
        r.kind = 9;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_BAD_REQUEST && reason_has(ctx, "unknown kind 9") && q == NULL;
        r.kind = EG_MEM_HOST;
        r.flags = 2;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_BAD_REQUEST && reason_has(ctx, "unknown flag bits 2");
        r.flags = 0;
        r.ndim = -1;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_BAD_REQUEST && reason_has(ctx, "ndim -1 without a shape");
        r.ndim = 1;
        r.shape = NULL;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_BAD_REQUEST && reason_has(ctx, "ndim 1 without a shape");
        r.ndim = 2;
        r.shape = neg;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_BAD_REQUEST && reason_has(ctx, "shape[1] is negative");
        r.shape = big;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_BAD_REQUEST && reason_has(ctx, "the size overflows");
        r.ndim = 1;
        r.shape = good;
        r.dtype_bits = 16;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_NOT_WRITTEN && reason_has(ctx, "dtype (code 2, bits 16) is not one eagle runs");
        r.dtype_bits = 64;
        r.kind = EG_MEM_DEVICE;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_NOT_WRITTEN && reason_has(ctx, "DEVICE memory on a host-only context");
        r.kind = EG_MEM_PINNED;
        ok = ok && eg_plane_new(ctx, &r, &q) == EG_NOT_WRITTEN && reason_has(ctx, "PINNED memory on a host-only context");
        ok = ok && eg_plane_new(ctx, NULL, &q) == EG_BAD_REQUEST && q == NULL;
        check(ok, "eg_plane_new: a NULL out, a bad kind, an unknown flag bit, a bad ndim or shape, a negative extent, an overflow, an unknown dtype, and "
                  "PINNED or DEVICE on a host-only context each refused by status and reason");
    }
    {
        eg_plane* q = NULL;
        struct DLManagedTensorVersioned* e = NULL;
        d.ndim = 1;
        d.shape = shape;
        ok = eg_plane_new(ctx, &d, &q) == EG_OK && eg_plane_dlpack(ctx, q, NULL) == EG_BAD_REQUEST && reason_has(ctx, "out is NULL");
        ok = ok && eg_plane_dlpack(ctx, NULL, &e) == EG_BAD_REQUEST && e == NULL && reason_has(ctx, "plane is NULL");
        ok = ok && eg_plane_free(ctx, q) == EG_OK && eg_plane_dlpack(ctx, q, &e) == EG_BAD_REQUEST && e == NULL;
        check(ok, "eg_plane_dlpack: a NULL out, a NULL plane and a freed plane each refused BAD_REQUEST by reason");
    }
}

/* eg_runner_figures and eg_plan_entry_figures on the host: no launch figures and no device entry, by a flag and -1, not by
 * zeros; the OUTPUT struct rule and the refusals by status and reason. */
static void check_figures(eg_ctx* ctx) {
    static double k[RN], limit[RN];
    static unsigned char mask[RN];
    struct tensor t[3];
    count_inputs(k, limit, mask);
    eg_unit* c1 = NULL;
    check(load(ctx, "count1", 0xD1, &c1) == EG_OK, "eg_unit_load count1 for the figures");
    if (c1 == NULL) return;
    eg_plan* p = count_plan(ctx, c1, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
    eg_runner* r = NULL;
    const struct eg_runner_desc d = runner_desc(&p, 1, 1000);
    int ok = p != NULL && eg_runner_new(ctx, &d, &r) == EG_OK;
    struct eg_runner_figures f;
    struct eg_entry_figures e;
    memset(&f, 0x55, sizeof(f));
    f.struct_size = sizeof(f);
    memset(&e, 0x55, sizeof(e));
    e.struct_size = sizeof(e);
    struct eg_run_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    for (int pass = 0; pass < 2; ++pass) { /* before and after a run: a host runner has no launch either way */
        ok = ok && eg_runner_figures(ctx, r, &f) == EG_OK && f.struct_size == sizeof(f) && f.mode == EG_MODE_HOST &&
             f.block == -1 && f.chunk == -1 && f.grid == -1 && f.steps_reverse == EG_STEPS_REVERSE_PER_STEP;
        if (pass == 0) ok = ok && eg_run(ctx, r, &rep) == EG_OK;
    }
    check(ok, "eg_runner_figures on a host runner, before and after eg_run: mode HOST, block, chunk and grid -1");
    ok = eg_plan_entry_figures(ctx, p, &e) == EG_OK && e.struct_size == sizeof(e) && e.flags == 0 && e.regs == 0 &&
         e.max_threads == 0 && e.persist_regs == 0 && e.persist_max_threads == 0 && e.local_bytes == 0 &&
         e.shared_bytes == 0 && e.const_bytes == 0;
    check(ok, "eg_plan_entry_figures on a host plan: EG_ENTRY_DEVICE clear, every figure 0");
    struct eg_runner_figures shorter;
    memset(&shorter, 0x55, sizeof(shorter));
    shorter.struct_size = 2 * sizeof(uint32_t);
    ok = eg_runner_figures(ctx, r, &shorter) == EG_OK && shorter.struct_size == 2 * sizeof(uint32_t) &&
         shorter.mode == (eg_run_mode)0x55555555;
    check(ok, "eg_runner_figures writes min(caller, eagle) bytes: the fields past a short struct_size are untouched");
    struct eg_runner_figures zero;
    memset(&zero, 0, sizeof(zero));
    ok = eg_runner_figures(ctx, r, NULL) == EG_BAD_REQUEST && reason_has(ctx, "eg_runner_figures is NULL");
    ok = ok && eg_runner_figures(ctx, r, &zero) == EG_BAD_REQUEST && reason_has(ctx, "struct_size 0 is too small");
    ok = ok && eg_runner_figures(ctx, NULL, &f) == EG_BAD_REQUEST && reason_has(ctx, "runner is NULL");
    ok = ok && eg_plan_entry_figures(ctx, p, NULL) == EG_BAD_REQUEST && reason_has(ctx, "eg_entry_figures is NULL");
    ok = ok && eg_plan_entry_figures(ctx, NULL, &e) == EG_BAD_REQUEST && reason_has(ctx, "plan is NULL");
    check(ok, "eg_runner_figures and eg_plan_entry_figures: a NULL out, a zero struct_size and a NULL handle refused BAD_REQUEST by reason");
    check(eg_runner_free(ctx, r) == EG_OK && eg_plan_free(ctx, p) == EG_OK && eg_unit_free(ctx, c1) == EG_OK,
          "eg_runner_free, eg_plan_free and eg_unit_free after the figures");
}

/* One run of `unit` on fresh planes, the oracle of the reset rows: its report and planes. */
static eg_status reset_fresh_run(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t every, double theta,
                                 int64_t max_steps, double* k, double* limit, unsigned char* mask, struct eg_run_report* rep) {
    struct tensor t[3];
    count_inputs(k, limit, mask);
    eg_plan* p = count_plan(ctx, unit, structure, k, limit, mask, t);
    eg_runner* r = NULL;
    struct eg_runner_desc d = runner_desc(&p, 1, max_steps);
    d.every = every;
    if (theta > 0) d.flags = EG_RUN_REORDER, d.reorder_theta = theta;
    eg_status s = p ? eg_runner_new(ctx, &d, &r) : EG_FAULT;
    if (s == EG_OK) s = eg_plan_free(ctx, p);
    memset(rep, 0, sizeof(*rep));
    rep->struct_size = sizeof(*rep);
    if (s == EG_OK) s = eg_run(ctx, r, rep);
    return s == EG_OK ? eg_runner_free(ctx, r) : s;
}

/* Run to completion, eg_runner_reset, write the inputs again, run again: planes and report bitwise those of a fresh runner. */
static void reset_case(eg_ctx* ctx, eg_unit* unit, eg_structure structure, int64_t every, double theta, int64_t max_steps,
                       const char* what) {
    static double k[RN], limit[RN], kf[RN], limitf[RN];
    static unsigned char mask[RN], maskf[RN];
    struct tensor t[3];
    struct eg_run_report fresh, again, first;
    eg_status s = reset_fresh_run(ctx, unit, structure, every, theta, max_steps, kf, limitf, maskf, &fresh);
    count_inputs(k, limit, mask);
    eg_plan* p = count_plan(ctx, unit, structure, k, limit, mask, t);
    eg_runner* r = NULL;
    struct eg_runner_desc d = runner_desc(&p, 1, max_steps);
    d.every = every;
    if (theta > 0) d.flags = EG_RUN_REORDER, d.reorder_theta = theta;
    if (s == EG_OK) s = p ? eg_runner_new(ctx, &d, &r) : EG_FAULT;
    if (s == EG_OK) s = eg_plan_free(ctx, p);
    memset(&first, 0, sizeof(first));
    first.struct_size = sizeof(first);
    again = first;
    if (s == EG_OK) s = eg_run(ctx, r, &first);
    if (s == EG_OK) s = eg_runner_reset(ctx, r);
    int cleared = 1;
    for (int64_t i = 0; i < RN; ++i) cleared &= mask[i] == 0;    /* every sample live again, read back from the caller's plane */
    count_inputs(k, limit, mask);
    if (s == EG_OK) s = eg_run(ctx, r, &again);
    int same = memcmp(k, kf, sizeof(k)) == 0 && memcmp(limit, limitf, sizeof(limit)) == 0 && memcmp(mask, maskf, sizeof(mask)) == 0;
    same = same && again.mode == fresh.mode && again.n == fresh.n && again.finished == fresh.finished && again.launches == fresh.launches &&
           again.steps == fresh.steps && again.mapped == fresh.mapped && again.compactions == fresh.compactions &&
           again.reorders == fresh.reorders && again.flags == fresh.flags;
    check(s == EG_OK && cleared && same, what);
    eg_runner_free(ctx, r);
}

/* eg_runner_reset: plain, active-set, reordering-set and steps-AUTO runners start over; the refusals. */
static void check_reset(eg_ctx* ctx) {
    eg_unit *c1 = NULL, *a1 = NULL, *ua = NULL, *uaa = NULL;
    check(load(ctx, "count1", 0xD1, &c1) == EG_OK && load(ctx, "count1a", 0xE1, &a1) == EG_OK && load(ctx, "countA", 0xF1, &ua) == EG_OK &&
              load(ctx, "countAa", 0xF2, &uaa) == EG_OK,
          "eg_unit_load count1, count1a, countA and countAa for eg_runner_reset");
    if (c1 == NULL || a1 == NULL || ua == NULL || uaa == NULL) return;
    const eg_structure structures[2] = { EG_STRUCTURE_HOST_SERIAL, EG_STRUCTURE_HOST_TEAM };
    for (int si = 0; si < 2; ++si) {
        eg_host_threads(ctx, si == 0 ? 1 : 3);
        reset_case(ctx, c1, structures[si], 0, 0.0, 1000, "eg_runner_reset: a plain runner run again equals a fresh runner (bitwise planes, same report)");
        reset_case(ctx, a1, structures[si], 4, 0.0, 1000, "eg_runner_reset: an active-set runner run again equals a fresh runner");
        reset_case(ctx, a1, structures[si], 4, 0.5, 1000, "eg_runner_reset: a reordering active-set runner run again equals a fresh runner");
        reset_case(ctx, ua, structures[si], 0, 0.0, 1000, "eg_runner_reset: a steps AUTO runner run again equals a fresh runner");
        reset_case(ctx, uaa, structures[si], 0, 0.0, 1000, "eg_runner_reset: a steps AUTO runner with an active set run again equals a fresh runner");
    }
    static double k[RN], limit[RN];
    static unsigned char mask[RN];
    struct tensor t[3];
    count_inputs(k, limit, mask);
    eg_plan* p = count_plan(ctx, c1, EG_STRUCTURE_HOST_SERIAL, k, limit, mask, t);
    eg_runner* r = NULL;
    const struct eg_runner_desc d = runner_desc(&p, 1, 100);
    int ok = p != NULL && eg_runner_new(ctx, &d, &r) == EG_OK;
    ok = ok && eg_runner_reset(ctx, NULL) == EG_BAD_REQUEST && reason_has(ctx, "runner");
    ok = ok && eg_record_begin(ctx) == EG_OK && eg_runner_reset(ctx, r) == EG_BAD_REQUEST && reason_has(ctx, "eg_runner_reset while recording");
    eg_graph* g = NULL;
    ok = ok && eg_record_end(ctx, &g) == EG_OK && eg_graph_free(ctx, g) == EG_OK;
    check(ok, "eg_runner_reset: a NULL runner and a call while recording refused BAD_REQUEST by reason");
    eg_runner_free(ctx, r);
    eg_plan_free(ctx, p);
    eg_unit_free(ctx, c1);
    eg_unit_free(ctx, a1);
    eg_unit_free(ctx, ua);
    eg_unit_free(ctx, uaa);
}

/* A door built without the device group: every device-group call is NOT_WRITTEN by the same reason, but eg_sync, which is OK. */
static void check_no_device_group(eg_ctx* ctx) {
    int32_t count = -1;
    struct eg_device_props props;
    memset(&props, 0, sizeof(props));
    props.struct_size = sizeof(props);
    const char* none = "the device group is not built";
    int ok = eg_device_count(ctx, &count) == EG_OK && count == 0;
    ok = ok && eg_device_props(ctx, 0, &props) == EG_NOT_WRITTEN && reason_has(ctx, none);
    ok = ok && eg_stream_use(ctx, 0) == EG_NOT_WRITTEN && reason_has(ctx, "eg_stream_use: the device group is not built");
    ok = ok && eg_record_child(ctx, 0) == EG_NOT_WRITTEN && reason_has(ctx, none);
    ok = ok && eg_sync(ctx) == EG_OK;
    check(ok, "a door without the device group: eg_device_count OK with 0, eg_device_props, eg_stream_use and eg_record_child "
              "NOT_WRITTEN (\"the device group is not built\"), eg_sync OK");
}

/* eg_device_count answers on a host context in either build; a device context is made when the driver sees a device and
 * refused BAD_REQUEST naming the count (or the driver) when it sees none. A door without the device group refuses it
 * NOT_WRITTEN. */
static void check_device_count(eg_ctx* ctx) {
    int32_t count = -1;
    check(eg_device_count(ctx, &count) == EG_OK && count >= 0 && eg_device_count(ctx, NULL) == EG_BAD_REQUEST,
          "eg_device_count on a host context: OK with a count of 0 or more, a NULL count BAD_REQUEST");
    const int built = device_group;
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    st.flags = EG_CTX_DEVICE;
    eg_ctx* dev = NULL;
    const eg_status made = eg_ctx_new(&st, &dev);
    int ok;
    if (!built)
        ok = made == EG_NOT_WRITTEN && reason_has(dev, "the device group is not built");
    else if (count == 0)
        ok = made == EG_BAD_REQUEST && (reason_has(dev, "the driver sees 0 device(s)") || reason_has(dev, "no CUDA driver could be opened"));
    else
        ok = made == EG_OK;
    check(ok, "a device context: made when the driver sees a device, BAD_REQUEST naming the count or the driver when it sees none, "
              "NOT_WRITTEN without the device group");
    eg_ctx_free(dev);
    eg_ctx* host = make_ctx();
    int32_t again = -1;
    check(host != NULL && eg_device_count(host, &again) == EG_OK && again == count, "host contexts work and see the same count");
    eg_ctx_free(host);
}

/* ---- transforms ---------------------------------------------------------- */

static void fft_move(void* dst, const void* src, size_t bytes) { memcpy(dst, src, bytes); }

#include "fft_cases.inc"

/* A context of 1 thread and one of 3 give bitwise identical outputs, for every kind, adjoints included, at both widths. */
static void check_fft_threads(void) {
    static const struct { eg_fft_kind kind; uint32_t flags; } kinds[] = {
        { EG_FFT_C2C, 0 }, { EG_FFT_C2C, EG_FFT_INVERSE | EG_FFT_ADJOINT }, { EG_FFT_R2C, 0 }, { EG_FFT_C2R, 0 },
        { EG_FFT_R2C, EG_FFT_ADJOINT }, { EG_FFT_C2R, EG_FFT_ADJOINT } };
    static const struct fft_geom geoms[] = { { 2, { 96, 40 }, 2, { 0, 1 } }, { 2, { 96, 40 }, 2, { 1, 0 } }, { 2, { 48, 36 }, 1, { 0 } } };
    eg_ctx* one = make_ctx();
    eg_ctx* three = make_ctx();
    int ok = one != NULL && three != NULL && eg_host_threads(one, 1) == EG_OK && eg_host_threads(three, 3) == EG_OK;
    for (int k = 0; ok && k < 6; ++k)
        for (int g = 0; ok && g < 3; ++g)
            for (int w = 0; ok && w < 2; ++w) {
                const struct fft_case c = fft_case_of(&geoms[g], kinds[k].kind, kinds[k].flags, EG_FFT_NORM_ORTHO, w == 0 ? 32 : 64);
                int64_t is[4], os[4];
                const size_t n_in = fft_floats(is, fft_plane(&c, 1, is)), n_out = fft_floats(os, fft_plane(&c, 0, os));
                double* in = malloc(n_in * sizeof(double));
                double* a = malloc(n_out * sizeof(double));
                double* b = malloc(n_out * sizeof(double));
                fft_fill(in, n_in, c.bits);
                ok = fft_apply(one, 0, &c, in, a) == EG_OK && fft_apply(three, 0, &c, in, b) == EG_OK && memcmp(a, b, n_out * sizeof(double)) == 0;
                free(in);
                free(a);
                free(b);
            }
    check(ok, "eg_fft_run on a context of 1 thread and one of 3 gives bitwise identical outputs (every kind and adjoint, float32 and float64)");
    eg_ctx_free(one);
    eg_ctx_free(three);
}

/* ---- SEGMENTED groups under a runner ---------------------------------------- */

static eg_status load_with(eg_ctx* ctx, const char* name, uint8_t digest_byte, eg_unit* const* deps, int32_t n_deps, eg_unit** out) {
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = name;
    memset(ref.digest, digest_byte, EG_DIGEST_BYTES);
    struct eg_unit_load_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.path = library_path;
    d.unit = &ref;
    d.deps = deps;
    d.n_deps = n_deps;
    return eg_unit_load(ctx, &d, out);
}

/* One run's planes: k, limit, mask, pos over SGN samples, wrapped and bound to slots 0, 1, 2 (and 7 when @p with_pos). */
struct seg_set {
    double k[SGN], limit[SGN], pos[SGN];
    unsigned char mask[SGN];
    struct tensor t[4];
    eg_plan* plan;
};

static void seg_inputs(struct seg_set* q, int dynamic) {
    for (int i = 0; i < SGN; ++i) {
        q->k[i] = dynamic ? DG_K0(i) : 0.0;    /* dynamic: the samples cross the class edge k = 4 one after another */
        q->limit[i] = dynamic ? 7.0 : (double)(2 + i % 9);
        q->pos[i] = -1.0;
        q->mask[i] = 0;
    }
}

static int seg_plan(eg_ctx* ctx, eg_unit* unit, struct seg_set* q, int with_pos) {
    eg_plane* pl[4] = { NULL, NULL, NULL, NULL };
    tensor_init(&q->t[0], q->k, SGN, 64);
    tensor_init(&q->t[1], q->limit, SGN, 64);
    mask_init(&q->t[2], q->mask, SGN);
    tensor_init(&q->t[3], q->pos, SGN, 64);
    eg_status s = plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, SGN, 0, NULL, NULL, &q->plan);
    for (int i = 0; i < 3 + (with_pos ? 1 : 0) && s == EG_OK; ++i) s = eg_plane_wrap(ctx, &q->t[i].managed, &pl[i]);
    const int32_t slots[4] = { 0, 1, 2, 7 };
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = 3 + (with_pos ? 1 : 0);
    d.plane_slots = slots;
    d.planes = pl;
    if (s == EG_OK) s = eg_bind(ctx, q->plan, &d);
    for (int i = 0; i < 4; ++i)
        if (pl[i] != NULL) eg_plane_free(ctx, pl[i]);
    return s == EG_OK;
}

static int seg_same(const struct seg_set* a, const struct seg_set* b, int with_pos) {
    return memcmp(a->k, b->k, sizeof(a->k)) == 0 && memcmp(a->mask, b->mask, sizeof(a->mask)) == 0 &&
           (!with_pos || memcmp(a->pos, b->pos, sizeof(a->pos)) == 0);
}

static int64_t seg_finished(const struct seg_set* q) { return seg_done(q->mask); }

/* The group's plan launched one step at a time by the caller, with the runner's finish rule (until every sample finished or max_steps launches). */
static int64_t seg_launch_loop(eg_ctx* ctx, struct seg_set* q, int64_t max_steps) {
    int64_t launches = 0;
    uint64_t tiles = 0;
    while (launches < max_steps && seg_finished(q) != SGN && eg_launch(ctx, q->plan, &tiles) == EG_OK) ++launches;
    return launches;
}

static eg_status seg_run(eg_ctx* ctx, struct seg_set* q, int64_t max_steps, eg_runner** out, struct eg_run_report* rep) {
    struct eg_runner_desc d = runner_desc(&q->plan, 1, max_steps);
    memset(rep, 0, sizeof(*rep));
    rep->struct_size = sizeof(*rep);
    eg_status s = eg_runner_new(ctx, &d, out);
    return s == EG_OK ? eg_run(ctx, *out, rep) : s;
}

/* The dispatch weights row is looked up once, by the class of this machine, and shown with the class key. No class row is written (a row equal to the
 * built-in one states nothing), so every machine takes the built-in row, -1; an older caller's smaller struct is filled as far as it goes. */
static void check_weight_row(eg_ctx* ctx) {
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    check(eg_ctx_info(ctx, &info) == EG_OK && info.struct_size == sizeof(info) && info.weight_vendor >= 0 && info.weight_row == -1,
          "eg_ctx_info: this machine takes the built-in weights row (-1) and states the class key it looked up");
    struct eg_ctx_info old;
    memset(&old, 0, sizeof(old));
    old.struct_size = (uint32_t)offsetof(struct eg_ctx_info, weight_vendor);
    old.weight_row = 77;
    check(eg_ctx_info(ctx, &old) == EG_OK && old.struct_size == offsetof(struct eg_ctx_info, weight_vendor) && old.weight_row == 77,
          "eg_ctx_info of a caller that predates the weights members fills none of them");
}

/* ---- SEGMENTED state: a drifted sort, the class a group ran by, and the cut across threads ---------------------------- */

/* Load the SEGMENTED group's branch unit of @p ctx with its members in deps: the static group "sg" (classify unit) or the dynamic "dg" (sweep unit). */
static eg_unit* seg_group(eg_ctx* ctx, int dynamic) {
    eg_unit *first = NULL, *second = NULL, *top = NULL;
    const int ok = dynamic ? load(ctx, "dg__k1", 0xE6, &first) == EG_OK && load(ctx, "dg__sweep", 0xE7, &second) == EG_OK
                           : load(ctx, "sg__k1", 0xE3, &first) == EG_OK && load(ctx, "sg__cls", 0xE1, &second) == EG_OK;
    eg_unit* deps[2] = { first, second };
    if (!ok || load_with(ctx, dynamic ? "dg__k0" : "sg__k0", dynamic ? 0xE5 : 0xE2, deps, 2, &top) != EG_OK) return NULL;
    return top;
}

/* The caller changes the samples between two launches, so the group's kept sort no longer matches them: the next launch must detect it and the step complete
 * as the fused unit's, which has no sort: k and the finished marks stay bitwise equal launch by launch. The static group drifts through the key its classify
 * unit reads (limit), the dynamic one through the state its class is read from (k). */
static void check_segmented_drift(eg_ctx* ctx) {
    for (int dynamic = 0; dynamic < 2; ++dynamic) {
        eg_unit* group = seg_group(ctx, dynamic);
        eg_unit* fused = NULL;
        check(group != NULL && load(ctx, dynamic ? "dgf" : "sgf", dynamic ? 0xE8 : 0xE4, &fused) == EG_OK,
              dynamic ? "eg_unit_load of the dynamic group and its fused unit for the drift rows" : "eg_unit_load of the static group and its fused unit for the drift rows");
        if (group == NULL || fused == NULL) return;
        static struct seg_set a, f;
        seg_inputs(&a, dynamic), seg_inputs(&f, dynamic);
        int ok = seg_plan(ctx, group, &a, dynamic) && seg_plan(ctx, fused, &f, 0);
        uint64_t tiles = 0;
        int same = ok, launches = 0;
        for (; ok && launches < 400 && !(seg_finished(&a) == SGN && seg_finished(&f) == SGN); ++launches) {
            ok = eg_launch(ctx, a.plan, &tiles) == EG_OK && eg_launch(ctx, f.plan, &tiles) == EG_OK;
            same = same && ok && seg_same(&a, &f, 0);
            if (launches == 1) {    /* after the second launch: a third of the live samples change class */
                for (int i = 0; i < SGN; i += 3) {
                    if (a.mask[i] || f.mask[i]) continue;
                    if (dynamic) a.k[i] += 2.0, f.k[i] += 2.0;
                    else a.limit[i] = f.limit[i] = (double)(2 + (i * 5) % 9);
                }
            }
        }
        check(same && launches < 400 && seg_finished(&a) == SGN,
              dynamic ? "dynamic group: samples whose class changed between launches (a stale sort) are caught, every step equals the fused unit's bitwise to the end"
                      : "static group: samples whose class key changed between launches (a stale sort) are caught, every step equals the fused unit's bitwise to the end");
        eg_plan_free(ctx, a.plan), eg_plan_free(ctx, f.plan);
        eg_unit_free(ctx, fused);
    }
}

/* One launch of a fresh group from the fixture inputs: each sample's k moved by exactly the step of the class the group ran it by, once. The classes are
 * door-internal (no call reads them), so this pins them through their effect: the static group's classify unit sends limit >= 6 to the branch that steps
 * 2 and the others to the one that steps 1 (it writes no class plane for a caller); the dynamic group's class is k >= 4, stepping 0.125 (class + 1), and
 * its branches record the sample's slot in pos, so every sample was visited. */
static int one_launch_visits(eg_ctx* ctx, int dynamic, struct seg_set* q, eg_unit* group) {
    seg_inputs(q, dynamic);
    uint64_t tiles = 0;
    int ok = seg_plan(ctx, group, q, dynamic) && eg_launch(ctx, q->plan, &tiles) == EG_OK;
    for (int i = 0; ok && i < SGN; ++i) {
        const double step = dynamic ? 0.125 * ((DG_K0(i) >= 4.0 ? 1 : 0) + 1) : (q->limit[i] >= 6.0 ? 2.0 : 1.0);
        ok = q->k[i] == (dynamic ? DG_K0(i) : 0.0) + step && (!dynamic || q->pos[i] == (double)i);
    }
    return ok;
}

/* Each segmented group on contexts of 1, 4 and 7 threads: one launch visits every sample once, and the run to the end equals the 1-thread run bitwise. */
static void check_segmented_threads(void) {
    for (int dynamic = 0; dynamic < 2; ++dynamic) {
        static struct seg_set one, run;
        int visits = 1, same = 1;
        for (int ti = 0; ti < 3; ++ti) {
            const int threads[3] = { 1, 4, 7 };
            eg_ctx* ctx = make_ctx();
            eg_host_threads(ctx, threads[ti]);
            eg_unit* group = seg_group(ctx, dynamic);
            if (group == NULL) { visits = same = 0; eg_ctx_free(ctx); continue; }
            visits = visits && one_launch_visits(ctx, dynamic, &run, group);
            eg_plan_free(ctx, run.plan);
            seg_inputs(&run, dynamic);
            eg_runner* r = NULL;
            struct eg_run_report rep;
            same = same && seg_plan(ctx, group, &run, dynamic) && seg_run(ctx, &run, 1000, &r, &rep) == EG_OK && seg_finished(&run) == SGN;
            if (ti == 0) one = run;
            else same = same && seg_same(&run, &one, dynamic);
            eg_runner_free(ctx, r);
            eg_plan_free(ctx, run.plan);
            eg_ctx_free(ctx);
        }
        check(visits, dynamic ? "dynamic group on 1, 4 and 7 threads: one launch moves every sample by its class's step once and records its slot in pos"
                              : "static group on 1, 4 and 7 threads: one launch moves every sample by its class's step exactly once");
        check(same, dynamic ? "dynamic group run to the end on 1, 4 and 7 threads: planes bitwise equal to the 1-thread run"
                            : "static group run to the end on 1, 4 and 7 threads: planes bitwise equal to the 1-thread run");
    }
}

/* ---- an AUTO group that names its segmented set ---------------------------- */

/* The planes of one run of the "sga" group: k, limit, mask and how (the path each step took: 1 segmented, 1000 fused) over SGN samples, wrapped and bound to
 * slots 0, 1, 2 and 5 (both the AUTO units and the branch units hold them there). */
struct auto_set {
    double k[SGN], limit[SGN], how[SGN];
    unsigned char mask[SGN];
    struct tensor t[4];
    eg_plan* plan;
};

static void auto_inputs(struct auto_set* q, int n1, int l1) { auto_fill(q->k, q->limit, q->how, q->mask, n1, l1); }

static int auto_plan(eg_ctx* ctx, eg_unit* unit, struct auto_set* q, int with_how) {
    eg_plane* pl[4] = { NULL, NULL, NULL, NULL };
    tensor_init(&q->t[0], q->k, SGN, 64);
    tensor_init(&q->t[1], q->limit, SGN, 64);
    mask_init(&q->t[2], q->mask, SGN);
    tensor_init(&q->t[3], q->how, SGN, 64);
    eg_status s = plan(ctx, unit, EG_STRUCTURE_HOST_TEAM, SGN, 0, NULL, NULL, &q->plan);
    for (int i = 0; i < 3 + with_how && s == EG_OK; ++i) s = eg_plane_wrap(ctx, &q->t[i].managed, &pl[i]);
    const int32_t slots[4] = { 0, 1, 2, 5 };
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = 3 + with_how;
    d.plane_slots = slots;
    d.planes = pl;
    if (s == EG_OK) s = eg_bind(ctx, q->plan, &d);
    for (int i = 0; i < 4; ++i)
        if (pl[i] != NULL) eg_plane_free(ctx, pl[i]);
    return s == EG_OK;
}

static void check_auto_look(eg_ctx* ctx) {
    eg_unit *cls = NULL, *k1 = NULL, *k0 = NULL, *auto_unit[4] = { NULL, NULL, NULL, NULL }, *fused = NULL;
    const char* names[4] = { "sga_rare", "sga_even", "sga_scalar", "sga_heavy" };
    check(load(ctx, "sga__cls", 0xC1, &cls) == EG_OK && load(ctx, "sga__k1", 0xC3, &k1) == EG_OK && load(ctx, "sgf", 0xE4, &fused) == EG_OK,
          "eg_unit_load of the AUTO group's classify and second branch units");
    eg_unit* sdeps[2] = { k1, cls };
    check(cls && k1 && load_with(ctx, "sga__k0", 0xC2, sdeps, 2, &k0) == EG_OK, "eg_unit_load of its first branch unit with the group's units in deps");
    eg_unit* adeps[3] = { k0, k1, cls };
    for (int u = 0; u < 4 && k0 != NULL; ++u)
        check(load_with(ctx, names[u], (uint8_t)(0xC4 + u), adeps, 3, &auto_unit[u]) == EG_OK, "eg_unit_load of an AUTO unit naming its segmented set");
    if (k0 == NULL || auto_unit[3] == NULL) return;

    /* The runs of test/auto_decisions.txt are the ones the old stack (test/auto_record.py) decides the same way on every machine of a grid of core counts and
     * cache sizes, so no machine's weights enter the expectation: the segmented steps and the total steps of each run, and so the path of each step in how.
     * The old stack's decision on any weights is proved equal to Dispatch.h's by tests/test_DispatchPolicy.cpp. */
    char path[512];
    snprintf(path, sizeof(path), "%s/auto_decisions.txt", EG_TEST_DIR);
    FILE* f = fopen(path, "r");
    check(f != NULL, "the recorded decisions of the old stack are present");
    char line[256];
    int rows = 0, agree = 0, switched = 0, fused_from_start = 0, segmented_to_end = 0;
    while (f != NULL && fgets(line, sizeof(line), f) != NULL) {
        char name[32];
        int n1, l1, max_steps, want_seg, want_total;
        if (line[0] == '#' || sscanf(line, "%31s %d %d %d %d %d", name, &n1, &l1, &max_steps, &want_seg, &want_total) != 6) continue;
        const int u = strcmp(name, "rare") == 0 ? 0 : strcmp(name, "even") == 0 ? 1 : strcmp(name, "scalar") == 0 ? 2 : 3;
        static struct auto_set q, ref;
        auto_inputs(&q, n1, l1);
        auto_inputs(&ref, n1, l1);
        eg_runner *r = NULL, *rf = NULL;
        struct eg_run_report rep, rep_ref;
        memset(&rep_ref, 0, sizeof(rep_ref));
        rep_ref.struct_size = sizeof(rep_ref);
        int ok = auto_plan(ctx, auto_unit[u], &q, 1) && auto_plan(ctx, fused, &ref, 0);
        struct eg_runner_desc d = runner_desc(&q.plan, 1, max_steps);
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        ok = ok && eg_runner_new(ctx, &d, &r) == EG_OK && eg_run(ctx, r, &rep) == EG_OK;
        struct eg_runner_desc dr = runner_desc(&ref.plan, 1, max_steps);
        ok = ok && eg_runner_new(ctx, &dr, &rf) == EG_OK && eg_run(ctx, rf, &rep_ref) == EG_OK;
        for (int i = 0; ok && i < SGN; ++i)    /* the path of every sample's steps is the old stack's decision, its values the plain fused unit's */
            ok = q.how[i] == auto_how(q.limit, i, n1, want_seg, want_total) && q.k[i] == ref.k[i] && q.mask[i] == ref.mask[i];
        ok = ok && rep.launches == want_total && rep.steps == want_total && rep.mode == EG_MODE_HOST && rep.launches == rep_ref.launches;
        ++rows;
        agree += ok;
        switched += want_seg > 0 && want_seg < want_total;
        fused_from_start += want_seg == 0;
        segmented_to_end += want_seg > 0 && want_seg == want_total;
        if (!ok) printf("      case %s n1 %d l1 %d max_steps %d: old stack segmented %d of %d steps\n", name, n1, l1, max_steps, want_seg, want_total);
        eg_runner_free(ctx, r);
        eg_runner_free(ctx, rf);
        eg_plan_free(ctx, q.plan);
        eg_plan_free(ctx, ref.plan);
    }
    if (f != NULL) fclose(f);
    {   /* the same run forced segmented (the branch unit's own runner) gives the AUTO run's values; reset then a second run equals the first */
        static struct auto_set a, b, first;
        auto_inputs(&a, 20, 100), auto_inputs(&b, 20, 100);
        eg_runner *ra = NULL, *rb = NULL;
        struct eg_run_report rep_a, rep_b, rep2;
        memset(&rep_a, 0, sizeof(rep_a)), memset(&rep_b, 0, sizeof(rep_b)), memset(&rep2, 0, sizeof(rep2));
        rep_a.struct_size = rep_b.struct_size = rep2.struct_size = sizeof(rep_a);
        struct eg_runner_desc da, db;
        int ok = auto_plan(ctx, auto_unit[0], &a, 1) && auto_plan(ctx, k0, &b, 1);
        da = runner_desc(&a.plan, 1, 400), db = runner_desc(&b.plan, 1, 400);
        ok = ok && eg_runner_new(ctx, &da, &ra) == EG_OK && eg_run(ctx, ra, &rep_a) == EG_OK && eg_runner_new(ctx, &db, &rb) == EG_OK && eg_run(ctx, rb, &rep_b) == EG_OK;
        ok = ok && memcmp(a.k, b.k, sizeof(a.k)) == 0 && memcmp(a.mask, b.mask, sizeof(a.mask)) == 0 && rep_a.launches == rep_b.launches && rep_a.finished == SGN;
        check(ok, "an AUTO run equals the same group forced segmented (the branch unit's runner): planes bitwise, same launches");
        memcpy(&first, &a, sizeof(a));
        auto_inputs(&a, 20, 100);
        ok = eg_runner_reset(ctx, ra) == EG_OK && eg_run(ctx, ra, &rep2) == EG_OK && memcmp(a.k, first.k, sizeof(a.k)) == 0 &&
             memcmp(a.how, first.how, sizeof(a.how)) == 0 && rep2.launches == rep_a.launches && rep2.steps == rep_a.steps && rep2.flags == rep_a.flags;
        check(ok, "eg_runner_reset of an AUTO runner then a second run equals the first, how included");
        eg_runner_free(ctx, ra), eg_runner_free(ctx, rb);
        eg_plan_free(ctx, a.plan), eg_plan_free(ctx, b.plan);
    }
    char what[400];
    snprintf(what, sizeof(what), "the AUTO runner takes the old stack's decision in all %d recorded runs (%d fused from the start, %d segmented to the end, %d switched to fused at a look), "
             "its values equal the plain fused unit's bitwise", rows, fused_from_start, segmented_to_end, switched);
    check(rows == 36 && agree == rows && fused_from_start > 0 && segmented_to_end > 0 && switched > 0, what);
    /* the refusals of a runner over an AUTO group with a segmented set */
    {
        eg_unit *steps_auto = NULL, *k3 = NULL;
        eg_runner* rr = NULL;
        eg_plan *ps = NULL, *pk = NULL, *pp = NULL;
        const int64_t base[2] = { 0, SGN / 2 }, count[2] = { SGN / 2, SGN / 2 };
        check(load_with(ctx, "auto_steps", 0xB6, adeps, 3, &steps_auto) == EG_OK && load_with(ctx, "auto_k3", 0xB7, adeps, 3, &k3) == EG_OK,
              "eg_unit_load of the AUTO units with steps AUTO and with 3 steps a launch");
        struct eg_runner_desc d;
        check(steps_auto && plan(ctx, steps_auto, EG_STRUCTURE_HOST_TEAM, SGN, 0, NULL, NULL, &ps) == EG_OK && (d = runner_desc(&ps, 1, 10), 1) &&
                  eg_runner_new(ctx, &d, &rr) == EG_NOT_WRITTEN && reason_has(ctx, "needs a fixed step count per launch"),
              "a runner over an AUTO group with a segmented set and steps AUTO: NOT_WRITTEN");
        check(k3 && plan(ctx, k3, EG_STRUCTURE_HOST_TEAM, SGN, 0, NULL, NULL, &pk) == EG_OK && (d = runner_desc(&pk, 1, 10), 1) &&
                  eg_runner_new(ctx, &d, &rr) == EG_NOT_WRITTEN && reason_has(ctx, "needs a fixed step count per launch"),
              "a runner over an AUTO group whose steps per launch differ from its segmented set's: NOT_WRITTEN");
        check(plan(ctx, auto_unit[0], EG_STRUCTURE_HOST_TEAM, SGN, 2, base, count, &pp) == EG_OK && (d = runner_desc(&pp, 1, 10), 1) &&
                  eg_runner_new(ctx, &d, &rr) == EG_NOT_WRITTEN && reason_has(ctx, "runs one whole partition"),
              "a runner over an AUTO group with a segmented set and two partitions: NOT_WRITTEN");
        eg_plan_free(ctx, ps), eg_plan_free(ctx, pk), eg_plan_free(ctx, pp);
        eg_unit_free(ctx, steps_auto), eg_unit_free(ctx, k3);
    }
    /* the refusals of an AUTO record, by status and reason */
    {
        struct { const char* name; uint8_t byte; const char* reason; } bad[5] = {
            { "auto_nocls", 0xB1, "is AUTO and names branches without a classify unit" },
            { "auto_nobr", 0xB2, "is AUTO and names a classify unit without branches" },
            { "auto_sweep", 0xB3, "is AUTO with a sweep: a dynamic selector runs fused" },
            { "auto_nocost", 0xB4, "names its segmented set but states no branch costs: the choice cannot be priced" },
            { "auto_group", 0xB5, "branch 0's record is group 'sg' k 2, the AUTO record says 'sga' k 2" } };
        eg_unit *sg_cls = NULL, *sg_k1 = NULL, *sg_k0 = NULL;
        load(ctx, "sg__cls", 0xE1, &sg_cls), load(ctx, "sg__k1", 0xE3, &sg_k1);
        eg_unit* gdeps[2] = { sg_k1, sg_cls };
        load_with(ctx, "sg__k0", 0xE2, gdeps, 2, &sg_k0);
        eg_unit* bdeps[3] = { sg_k0, sg_k1, sg_cls };
        for (int b = 0; b < 5; ++b) {
            eg_unit* out = NULL;
            char what[200];
            snprintf(what, sizeof(what), "eg_unit_load of '%s' refused BAD_REQUEST: %s", bad[b].name, bad[b].reason);
            check(load_with(ctx, bad[b].name, bad[b].byte, b == 4 ? bdeps : adeps, 3, &out) == EG_BAD_REQUEST &&
                      out == NULL && reason_has(ctx, bad[b].reason),
                  what);
        }
        eg_unit_free(ctx, sg_k0), eg_unit_free(ctx, sg_k1), eg_unit_free(ctx, sg_cls);
    }
    for (int u = 0; u < 4; ++u) eg_unit_free(ctx, auto_unit[u]);
    eg_unit_free(ctx, k0), eg_unit_free(ctx, k1), eg_unit_free(ctx, cls), eg_unit_free(ctx, fused);
}

static void check_segmented_runner(eg_ctx* ctx) {
    eg_unit *cls = NULL, *k1 = NULL, *k0 = NULL, *fused = NULL;
    check(load(ctx, "sg__cls", 0xE1, &cls) == EG_OK && load(ctx, "sg__k1", 0xE3, &k1) == EG_OK && load(ctx, "sgf", 0xE4, &fused) == EG_OK,
          "eg_unit_load of the static group's classify and second branch unit, and its fused unit");
    eg_unit* deps[2] = { k1, cls };
    check(cls && k1 && load_with(ctx, "sg__k0", 0xE2, deps, 2, &k0) == EG_OK, "eg_unit_load of the first branch unit with the group's units in deps");
    if (k0 == NULL || fused == NULL) return;
    static struct seg_set ref, run, fus, again;
    struct eg_run_report rep, rep2;
    eg_runner *r = NULL, *rf = NULL;
    seg_inputs(&ref, 0), seg_inputs(&run, 0), seg_inputs(&fus, 0), seg_inputs(&again, 0);
    check(seg_plan(ctx, k0, &ref, 0) && seg_plan(ctx, k0, &run, 0) && seg_plan(ctx, fused, &fus, 0), "the group's plans and the fused unit's plan");
    const int64_t want = seg_launch_loop(ctx, &ref, 1000);
    check(want == 5 && seg_finished(&ref) == SGN, "the group launched by hand until done: 5 launches (limit 10 at 2 per step)");
    check(seg_run(ctx, &run, 1000, &r, &rep) == EG_OK && seg_same(&run, &ref, 0) && rep.launches == want && rep.steps == want &&
              rep.finished == SGN && rep.mapped == want * SGN && rep.mode == EG_MODE_HOST &&
              rep.flags == (EG_REPORT_DONE | EG_REPORT_EXACT_STEPS),
          "a runner over a SEGMENTED group: its planes and report equal the plan launched step by step");
    check(seg_run(ctx, &fus, 1000, &rf, &rep2) == EG_OK && seg_same(&fus, &ref, 0) && rep2.launches == want,
          "the same run of the fused unit (the same per-sample steps) equals it bitwise");
    memset(run.k, 0, sizeof(run.k));
    memset(run.mask, 0, sizeof(run.mask));
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    check(eg_runner_reset(ctx, r) == EG_OK && eg_run(ctx, r, &rep) == EG_OK && seg_same(&run, &ref, 0) && rep.launches == want && rep.finished == SGN,
          "eg_runner_reset then a second run equals the first run, report included");
    check(eg_runner_free(ctx, r) == EG_OK && eg_runner_free(ctx, rf) == EG_OK, "the group's runners freed");
    {   /* a run cut short by max_steps: the same cut as the hand-launched loop */
        static struct seg_set cut_ref, cut;
        seg_inputs(&cut_ref, 0), seg_inputs(&cut, 0);
        seg_plan(ctx, k0, &cut_ref, 0), seg_plan(ctx, k0, &cut, 0);
        const int64_t part = seg_launch_loop(ctx, &cut_ref, 3);
        eg_runner* rc = NULL;
        check(seg_run(ctx, &cut, 3, &rc, &rep) == EG_OK && seg_same(&cut, &cut_ref, 0) && rep.launches == part && rep.finished < SGN &&
                  rep.flags == EG_REPORT_EXACT_STEPS,
              "max_steps 3 stops a SEGMENTED run after 3 launches, not done, as the hand-launched loop");
        eg_runner_free(ctx, rc);
        eg_plan_free(ctx, cut_ref.plan);
    }
    /* the refusals, by status and reason */
    {
        eg_unit* axpb = NULL;
        eg_plan* two[2] = { run.plan, NULL };
        struct eg_runner_desc d = runner_desc(two, 2, 10);
        eg_runner* rr = NULL;
        check(load(ctx, "axpb", 0xA1, &axpb) == EG_OK && plan(ctx, axpb, EG_STRUCTURE_HOST_TEAM, SGN, 0, NULL, NULL, &two[1]) == EG_OK &&
                  eg_runner_new(ctx, &d, &rr) == EG_NOT_WRITTEN && reason_has(ctx, "the whole step of its runner"),
              "a SEGMENTED group beside another plan: NOT_WRITTEN, it is the whole step");
        d = runner_desc(&run.plan, 1, 10);
        d.flags = EG_RUN_REORDER;
        check(eg_runner_new(ctx, &d, &rr) == EG_BAD_REQUEST && reason_has(ctx, "nothing to reorder"), "EG_RUN_REORDER over a SEGMENTED group: BAD_REQUEST");
        d.flags = 0, d.every = 4;
        check(eg_runner_new(ctx, &d, &rr) == EG_BAD_REQUEST && reason_has(ctx, "compaction does not apply"), "every over a SEGMENTED group: BAD_REQUEST");
        d.every = 0, d.events = EG_EVENTS_REFINE;
        check(eg_runner_new(ctx, &d, &rr) != EG_OK && rr == NULL, "events REFINE over a SEGMENTED group is refused");
        eg_plan_free(ctx, two[1]);
        eg_unit_free(ctx, axpb);
    }
    eg_plan_free(ctx, ref.plan), eg_plan_free(ctx, run.plan), eg_plan_free(ctx, fus.plan), eg_plan_free(ctx, again.plan);
    eg_unit_free(ctx, fused);

    /* the dynamic selector: re-sorts at the model's launches and not before; the fused unit's run equals it bitwise */
    eg_unit *dk1 = NULL, *dsw = NULL, *dk0 = NULL, *dgf = NULL;
    check(load(ctx, "dg__k1", 0xE6, &dk1) == EG_OK && load(ctx, "dg__sweep", 0xE7, &dsw) == EG_OK && load(ctx, "dgf", 0xE8, &dgf) == EG_OK,
          "eg_unit_load of the dynamic group's second branch, sweep and fused units");
    eg_unit* ddeps[2] = { dk1, dsw };
    check(dk1 && dsw && load_with(ctx, "dg__k0", 0xE5, ddeps, 2, &dk0) == EG_OK, "eg_unit_load of the dynamic group's first branch unit");
    if (dk0 == NULL || dgf == NULL) return;
    static struct seg_set model, a, b, f;
    int64_t sorts[16];
    int n_sorts = 0;
    seg_inputs(&model, 1);
    const int64_t total = dg_model(model.k, model.limit, model.mask, model.pos, 1000, sorts, &n_sorts);
    /* Launch L steps the samples of run 0 whose k (k0 + 0.125 (L - 1) while in run 0) is still below 4 and marks the rest stale: 6 samples
     * (0.02 * ((37 i) mod 200) >= 3.875) in launch 2, 12 in launch 3. The sort is kept while the stale are at most 5% of the 200 live (10), so launch 3 keeps
     * it (6 stale) and launch 4 sorts again (12 stale): the sorts are at launches 1 and 4. */
    check(n_sorts >= 2 && sorts[0] == 1 && sorts[1] == 4, "the model sorts at launch 1 and next at launch 4 (12 stale of 200 live), not at launch 2 or 3");
    for (int64_t cut = sorts[1] - 1; cut <= sorts[1] + 1; ++cut) {
        seg_inputs(&model, 1), seg_inputs(&a, 1);
        int64_t m_sorts[16];
        int m_n = 0;
        dg_model(model.k, model.limit, model.mask, model.pos, cut, m_sorts, &m_n);
        eg_runner* rd = NULL;
        char what[160];
        snprintf(what, sizeof(what), "dynamic group, run of %lld launches (the second sort at launch %lld): planes, positions and report as the model",
                 (long long)cut, (long long)sorts[1]);
        check(seg_plan(ctx, dk0, &a, 1) && seg_run(ctx, &a, cut, &rd, &rep) == EG_OK && seg_same(&a, &model, 1) && rep.launches == cut, what);
        eg_runner_free(ctx, rd);
        eg_plan_free(ctx, a.plan);
    }
    seg_inputs(&model, 1), seg_inputs(&b, 1), seg_inputs(&f, 1);
    dg_model(model.k, model.limit, model.mask, model.pos, 1000, sorts, &n_sorts);
    eg_runner *rb = NULL, *rfu = NULL;
    check(seg_plan(ctx, dk0, &b, 1) && seg_plan(ctx, dgf, &f, 0) && seg_run(ctx, &b, 1000, &rb, &rep) == EG_OK && seg_run(ctx, &f, 1000, &rfu, &rep2) == EG_OK &&
              seg_same(&b, &model, 1) && memcmp(b.k, f.k, sizeof(b.k)) == 0 && rep.launches == total && rep2.launches == total,
          "the dynamic group to completion equals the model and the fused unit bitwise");
    memset(b.k, 0, sizeof(b.k));
    for (int i = 0; i < SGN; ++i) b.k[i] = DG_K0(i), b.pos[i] = -1.0;
    memset(b.mask, 0, sizeof(b.mask));
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = sizeof(rep);
    check(eg_runner_reset(ctx, rb) == EG_OK && eg_run(ctx, rb, &rep) == EG_OK && seg_same(&b, &model, 1) && rep.launches == total,
          "eg_runner_reset of a dynamic group, then a second run equals the first (the sort starts over)");
    eg_runner_free(ctx, rb), eg_runner_free(ctx, rfu);
    eg_plan_free(ctx, b.plan), eg_plan_free(ctx, f.plan);
    eg_unit_free(ctx, k0), eg_unit_free(ctx, k1), eg_unit_free(ctx, cls), eg_unit_free(ctx, dk0), eg_unit_free(ctx, dk1), eg_unit_free(ctx, dsw), eg_unit_free(ctx, dgf);
}

/* ---- calls on one context queue; a reason belongs to its thread -------- */

struct queued {
    eg_ctx* ctx;
    eg_plan* plan;
    eg_status status;
    int done;
};

static void* queued_launch(void* arg) {
    struct queued* r = arg;
    uint64_t tiles = 0;
    r->status = eg_launch(r->ctx, r->plan, &tiles);
    return NULL;
}

static void* queued_info(void* arg) {
    struct queued* r = arg;
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    r->status = eg_ctx_info(r->ctx, &info);
    __atomic_store_n(&r->done, 1, __ATOMIC_SEQ_CST);
    return NULL;
}

/* A second thread entering a context while a launch runs waits for it, then runs. */
static void check_second_thread(eg_ctx* ctx) {
    eg_unit* wait = NULL;
    double x[2] = { 0, 0 }, y[2] = { 0, 0 };
    struct tensor tx, ty;
    tensor_init(&tx, x, 2, 64);
    tensor_init(&ty, y, 2, 64);
    eg_plane *px = NULL, *py = NULL;
    eg_plan* p = NULL;
    const double a = 1.0, b = 0.0;
    int ready = load(ctx, "wait", 0xB2, &wait) == EG_OK && eg_plane_wrap(ctx, &tx.managed, &px) == EG_OK &&
                eg_plane_wrap(ctx, &ty.managed, &py) == EG_OK &&
                plan(ctx, wait, EG_STRUCTURE_HOST_SERIAL, 2, 0, NULL, NULL, &p) == EG_OK &&
                bind(ctx, p, py, px, &a, &b) == EG_OK;
    check(ready, "queue fixture: wait unit loaded, planned and bound");
    if (!ready) return;
    volatile double* flag = y;
    struct queued first = { ctx, p, -1, 0 }, second = { ctx, NULL, -1, 0 };
    pthread_t th, th2;
    pthread_create(&th, NULL, queued_launch, &first);
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int entered = 0;
    for (;;) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (flag[1] != 0.0) { entered = 1; break; }
        clock_gettime(CLOCK_MONOTONIC, &t);
        if (t.tv_sec - t0.tv_sec > 10) break;
    }
    pthread_create(&th2, NULL, queued_info, &second);
    const struct timespec pause = { 0, 200 * 1000 * 1000 };
    nanosleep(&pause, NULL);
    const int waited = __atomic_load_n(&second.done, __ATOMIC_SEQ_CST) == 0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    flag[0] = 1.0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    pthread_join(th, NULL);
    pthread_join(th2, NULL);
    check(entered && waited && second.status == EG_OK, "a second thread waits for the running call, then runs OK");
    check(first.status == EG_OK, "the first thread's launch finished OK");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, py);
    eg_unit_free(ctx, wait);
}

enum { TICK_THREADS = 8, TICK_LAUNCHES = 250 };

struct ticking {
    eg_ctx* ctx;
    eg_plan* plan;
    int failed;
};

static void* tick_launches(void* arg) {
    struct ticking* r = arg;
    for (int i = 0; i < TICK_LAUNCHES; ++i) {
        uint64_t tiles = 0;
        if (eg_launch(r->ctx, r->plan, &tiles) != EG_OK) ++r->failed;
    }
    return NULL;
}

/* A plan of "tick" on @p ctx over @p cell (one int64 cell); NULL when it cannot be made. */
static eg_plan* tick_plan(eg_ctx* ctx, eg_unit* tick, struct tensor* t, int64_t* cell) {
    eg_plane* pc = wrap(ctx, t, cell, 1, kDLInt, 64);
    eg_plan* p = NULL;
    const int32_t slot = 0;
    eg_plane* const planes[1] = { pc };
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = 1;
    d.plane_slots = &slot;
    d.planes = planes;
    const int ok = pc != NULL && plan(ctx, tick, EG_STRUCTURE_HOST_SERIAL, 1, 0, NULL, NULL, &p) == EG_OK &&
                   eg_bind(ctx, p, &d) == EG_OK;
    eg_plane_free(ctx, pc);
    if (!ok) {
        eg_plan_free(ctx, p);
        return NULL;
    }
    return p;
}

struct reasoner {
    eg_ctx* ctx;
    int step;     /* 0 none, 1: A has failed, 2: B has finished, 3: A may read */
    char text[256];
};

static void failing_call(eg_ctx* ctx, int variant) {
    eg_plane* out = NULL;
    if (variant == 0) eg_plane_wrap(ctx, NULL, &out);
    else eg_plane_wrap(ctx, NULL, NULL);
}

static void wait_step(struct reasoner* r, int step) {
    while (__atomic_load_n(&r->step, __ATOMIC_SEQ_CST) < step) sched_yield();
}

static void* reason_a(void* arg) {
    struct reasoner* r = arg;
    failing_call(r->ctx, 0);
    __atomic_store_n(&r->step, 1, __ATOMIC_SEQ_CST);
    wait_step(r, 3);
    size_t len = 0;
    r->text[0] = '\0';
    if (eg_error(r->ctx, r->text, sizeof(r->text), &len) != EG_OK) r->text[0] = '\0';
    return NULL;
}

static void* reason_b(void* arg) {
    struct reasoner* r = arg;
    wait_step(r, 1);
    failing_call(r->ctx, 1);
    __atomic_store_n(&r->step, 2, __ATOMIC_SEQ_CST);
    return NULL;
}

/* Planes of a context's own export are refused: their release would wait on the call that holds the context. */
static void check_own_export(eg_ctx* ctx) {
    double v[2] = { 0, 0 };
    struct tensor t;
    eg_plane* pl = wrap(ctx, &t, v, 2, kDLFloat, 64);
    struct DLManagedTensorVersioned* exported = NULL;
    eg_plane* again = NULL;
    const int made = pl != NULL && eg_plane_dlpack(ctx, pl, &exported) == EG_OK;
    check(made && eg_plane_wrap(ctx, exported, &again) == EG_BAD_REQUEST && again == NULL &&
              reason_has(ctx, "an export of this context"),
          "wrapping a context's own export is refused BAD_REQUEST (its release would wait on itself)");
    if (made) exported->deleter(exported);
    eg_plane_free(ctx, pl);
}

static void check_queued_calls(eg_ctx* ctx, eg_ctx* other) {
    eg_unit *tick = NULL, *tick_other = NULL;
    check(load(ctx, "tick", 0xB8, &tick) == EG_OK && load(other, "tick", 0xB8, &tick_other) == EG_OK, "eg_unit_load tick on two contexts");
    if (tick == NULL || tick_other == NULL) return;
    int64_t cell = 0, cell_other = 0;
    struct tensor t, t_other;
    eg_plan* p = tick_plan(ctx, tick, &t, &cell);
    eg_plan* p_other = tick_plan(other, tick_other, &t_other, &cell_other);
    check(p != NULL && p_other != NULL, "tick plans bound on two contexts");
    if (p == NULL || p_other == NULL) return;
    struct ticking many[TICK_THREADS], lone[2];
    pthread_t th[TICK_THREADS], th_lone[2];
    for (int i = 0; i < TICK_THREADS; ++i) many[i] = (struct ticking){ ctx, p, 0 };
    for (int i = 0; i < TICK_THREADS; ++i) pthread_create(&th[i], NULL, tick_launches, &many[i]);
    for (int i = 0; i < TICK_THREADS; ++i) pthread_join(th[i], NULL);
    int failed = 0;
    for (int i = 0; i < TICK_THREADS; ++i) failed += many[i].failed;
    check(failed == 0 && cell == (int64_t)TICK_THREADS * TICK_LAUNCHES,
          "8 threads x 250 launches on one context: every launch ran alone, the cell holds exactly 2000");
    cell = 0;
    lone[0] = (struct ticking){ ctx, p, 0 };
    lone[1] = (struct ticking){ other, p_other, 0 };
    for (int i = 0; i < 2; ++i) pthread_create(&th_lone[i], NULL, tick_launches, &lone[i]);
    for (int i = 0; i < 2; ++i) pthread_join(th_lone[i], NULL);
    check(lone[0].failed == 0 && lone[1].failed == 0 && cell == TICK_LAUNCHES && cell_other == TICK_LAUNCHES,
          "two contexts launched at once from two threads both count exactly");
    struct reasoner r = { ctx, 0, "" };
    pthread_t ta, tb;
    pthread_create(&ta, NULL, reason_a, &r);
    pthread_create(&tb, NULL, reason_b, &r);
    wait_step(&r, 2);
    __atomic_store_n(&r.step, 3, __ATOMIC_SEQ_CST);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    check(strstr(r.text, "tensor is NULL") != NULL, "a thread's reason survives another thread's failing call with another reason");
    check_own_export(ctx);
    eg_plan_free(ctx, p);
    eg_plan_free(other, p_other);
    eg_unit_free(ctx, tick);
    eg_unit_free(other, tick_other);
}

/* ---- a plan's counters, and a steps word the caller binds ---------------- */

static eg_status counters_of(eg_ctx* ctx, eg_plan* p, struct eg_plan_counters* c, uint32_t* mix, int32_t cap, int32_t* n_mix) {
    memset(c, 0, sizeof(*c));
    c->struct_size = sizeof(*c);
    return eg_plan_counters(ctx, p, c, mix, cap, n_mix);
}

/* Bind the int64 steps word (slot 4 of countA) to the caller's cell. */
static eg_status bind_steps(eg_ctx* ctx, eg_plan* p, eg_plane* cell) {
    const int32_t slot = 4;
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = 1;
    d.plane_slots = &slot;
    d.planes = &cell;
    return eg_bind(ctx, p, &d);
}

static int64_t count_at_most(const double* limit, double bound) {
    int64_t n = 0;
    for (int64_t i = 0; i < RN; ++i) n += limit[i] <= bound;
    return n;
}

/* The finished counter after launches of count1, and a launch of countA over the caller's steps word. */
static void check_counters_finished(eg_ctx* ctx) {
    static double k[RN], limit[RN];
    static unsigned char mask[RN];
    eg_unit *c1 = NULL, *ua = NULL, *axpb = NULL;
    check(load(ctx, "count1", 0xD1, &c1) == EG_OK && load(ctx, "countA", 0xF1, &ua) == EG_OK && load(ctx, "axpb", 0xA1, &axpb) == EG_OK,
          "eg_unit_load count1, countA and axpb for eg_plan_counters");
    if (c1 == NULL || ua == NULL || axpb == NULL) return;
    struct tensor t[3];
    struct eg_plan_counters c;
    int32_t n_mix = -1;
    count_inputs(k, limit, mask);
    memset(mask, 0, sizeof(mask));
    eg_plan* p = count_plan(ctx, c1, EG_STRUCTURE_HOST_SERIAL, k, limit, mask, t);
    int ok = p != NULL;
    for (int launch = 1; ok && launch <= 3; ++launch) {
        int64_t masked = 0;
        ok = eg_launch(ctx, p, NULL) == EG_OK && counters_of(ctx, p, &c, NULL, 0, &n_mix) == EG_OK;
        for (int64_t i = 0; i < RN; ++i) masked += mask[i] != 0;
        ok = ok && masked == count_at_most(limit, launch) && c.finished == masked - count_at_most(limit, launch - 1) &&
             c.steps == -1 && n_mix == 0 && c.struct_size == sizeof(c);
    }
    check(ok, "eg_plan_counters: each of 3 launches reads the samples that launch finished (limit in (launch - 1, launch]), no steps word, no mix");
    eg_plan_free(ctx, p);

    count_inputs(k, limit, mask);    /* every 97th sample's mask set before the first launch */
    int64_t preset = 0, summed = 0, masked = 0;
    for (int64_t i = 0; i < RN; ++i) preset += mask[i] != 0;
    p = count_plan(ctx, c1, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
    ok = p != NULL && preset > 0;
    for (int launch = 1; ok && launch <= 50; ++launch) {
        ok = eg_launch(ctx, p, NULL) == EG_OK && counters_of(ctx, p, &c, NULL, 0, &n_mix) == EG_OK;
        summed += c.finished;
    }
    for (int64_t i = 0; i < RN; ++i) masked += mask[i] != 0;
    check(ok && masked == RN && summed == RN - preset,
          "eg_plan_counters: samples masked before the first launch are not counted; the 50 per-launch figures add up to the rest");
    eg_plan_free(ctx, p);

    count_inputs(k, limit, mask);
    memset(mask, 0, sizeof(mask));
    p = count_plan(ctx, c1, EG_STRUCTURE_HOST_SERIAL, k, limit, mask, t);
    eg_graph* g = NULL;
    ok = p != NULL && eg_record_begin(ctx) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && eg_record_end(ctx, &g) == EG_OK;
    for (int replay = 1; ok && replay <= 3; ++replay)
        ok = eg_graph_launch(ctx, g) == EG_OK && counters_of(ctx, p, &c, NULL, 0, &n_mix) == EG_OK &&
             c.finished == count_at_most(limit, replay) - count_at_most(limit, replay - 1);
    check(ok, "eg_plan_counters: each replay of a recorded launch reads the samples that replay finished");
    eg_graph_free(ctx, g);
    eg_plan_free(ctx, p);

    int64_t word = 2;
    struct tensor tw;
    eg_plane* pw = wrap(ctx, &tw, &word, 1, kDLInt, 64);
    count_inputs(k, limit, mask);
    memset(mask, 0, sizeof(mask));
    p = count_plan(ctx, ua, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
    ok = p != NULL && pw != NULL && counters_of(ctx, p, &c, NULL, 0, &n_mix) == EG_OK && c.finished == 0 && c.steps == 0;
    check(ok, "eg_plan_counters: an unbound steps word of a fresh AUTO plan reads 0");
    ok = ok && bind_steps(ctx, p, pw) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK && counters_of(ctx, p, &c, NULL, 0, &n_mix) == EG_OK;
    for (int64_t i = 0; ok && i < RN; ++i) ok = k[i] == (limit[i] < 2.0 ? limit[i] : 2.0);
    check(ok && c.steps == 2 && c.finished == count_at_most(limit, 2.0),
          "a bound steps word: a plain launch steps every sample by the caller's 2, and the counters read it back");
    word = 5;
    ok = eg_launch(ctx, p, NULL) == EG_OK && counters_of(ctx, p, &c, NULL, 0, &n_mix) == EG_OK && c.steps == 5 &&
         c.finished == count_at_most(limit, 7.0) - count_at_most(limit, 2.0);
    check(ok, "a bound steps word is read at each launch (5 more steps: this launch finished limit in (2, 7])");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, pw);

    int64_t two[2] = { 1, 1 };
    struct tensor t2;
    eg_plane* p2 = wrap(ctx, &t2, two, 2, kDLInt, 64);
    p = count_plan(ctx, ua, EG_STRUCTURE_HOST_SERIAL, k, limit, mask, t);
    ok = p != NULL && p2 != NULL && bind_steps(ctx, p, p2) == EG_BAD_REQUEST && reason_has(ctx, "one cell");
    check(ok, "a steps word of two cells refused BAD_REQUEST by name");
    eg_plane_free(ctx, p2);
    int64_t ro = 1;
    struct tensor tr;
    eg_plane* pr = NULL;
    tensor_init(&tr, &ro, 1, 64);
    tensor_as(&tr, kDLInt, 64);
    tr.managed.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
    eg_plane_wrap(ctx, &tr.managed, &pr);
    ok = p != NULL && pr != NULL && bind_steps(ctx, p, pr) == EG_BAD_REQUEST && reason_has(ctx, "read-only");
    check(ok, "a read-only steps word refused BAD_REQUEST (the runner writes it)");
    eg_plane_free(ctx, pr);
    struct tensor tf;
    eg_plane* pf = wrap(ctx, &tf, k, 1, kDLFloat, 64);
    ok = p != NULL && pf != NULL && bind_steps(ctx, p, pf) == EG_BAD_REQUEST;
    check(ok, "a steps word of another element type refused BAD_REQUEST");
    eg_plane_free(ctx, pf);
    eg_plan_free(ctx, p);

    p = NULL;
    double x[4] = { 0 }, y[4] = { 0 };
    const double a = 1.0, b = 0.0;
    struct tensor tx, ty;
    eg_plane *px = wrap(ctx, &tx, x, 4, kDLFloat, 64), *py = wrap(ctx, &ty, y, 4, kDLFloat, 64);
    ok = px && py && plan(ctx, axpb, EG_STRUCTURE_HOST_SERIAL, 4, 0, NULL, NULL, &p) == EG_OK && bind(ctx, p, py, px, &a, &b) == EG_OK &&
         counters_of(ctx, p, &c, NULL, 0, &n_mix) == EG_OK && c.finished == -1 && c.steps == -1 && n_mix == 0;
    check(ok, "eg_plan_counters: a unit with no finish reads -1 and -1, no mix");
    check(eg_plan_counters(ctx, p, &c, NULL, 0, NULL) == EG_BAD_REQUEST && reason_has(ctx, "n_mix is NULL") &&
              eg_plan_counters(ctx, p, NULL, NULL, 0, &n_mix) == EG_BAD_REQUEST && eg_plan_counters(ctx, NULL, &c, NULL, 0, &n_mix) == EG_BAD_REQUEST,
          "eg_plan_counters: n_mix NULL, out NULL and a NULL plan refused BAD_REQUEST");
    c.struct_size = 2;
    check(eg_plan_counters(ctx, p, &c, NULL, 0, &n_mix) == EG_BAD_REQUEST && reason_has(ctx, "too small"), "eg_plan_counters: a struct_size below 4 refused BAD_REQUEST");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, py);
    eg_unit_free(ctx, c1);
    eg_unit_free(ctx, ua);
    eg_unit_free(ctx, axpb);
}

/* eg_run over a bound steps word writes its budgets there as into its own: the planes and the counters agree. */
static void check_counters_run(eg_ctx* ctx) {
    static double k[RN], limit[RN], k_own[RN];
    static unsigned char mask[RN];
    eg_unit* ua = NULL;
    check(load(ctx, "countA", 0xF1, &ua) == EG_OK, "eg_unit_load countA for a run over a bound steps word");
    if (ua == NULL) return;
    int64_t steps_seen[2] = { -1, -1 };
    int ok = 1;
    for (int bound = 0; bound < 2; ++bound) {
        struct tensor t[3], tw;
        int64_t word = 7;
        eg_plane* pw = wrap(ctx, &tw, &word, 1, kDLInt, 64);
        count_inputs(k, limit, mask);
        memset(mask, 0, sizeof(mask));
        eg_plan* p = count_plan(ctx, ua, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
        eg_runner* r = NULL;
        const struct eg_runner_desc d = runner_desc(&p, 1, 1000);
        struct eg_run_report rep;
        memset(&rep, 0, sizeof(rep));
        rep.struct_size = sizeof(rep);
        struct eg_plan_counters c;
        ok = ok && p && pw && (!bound || bind_steps(ctx, p, pw) == EG_OK) && eg_runner_new(ctx, &d, &r) == EG_OK && eg_run(ctx, r, &rep) == EG_OK &&
             counters_of(ctx, p, &c, NULL, 0, &(int32_t){ 0 }) == EG_OK && c.finished == RN && rep.finished == RN && (!bound || word == c.steps);
        steps_seen[bound] = c.steps;
        if (!bound) memcpy(k_own, k, sizeof(k));
        else ok = ok && memcmp(k, k_own, sizeof(k)) == 0;
        eg_runner_free(ctx, r);
        eg_plan_free(ctx, p);
        eg_plane_free(ctx, pw);
    }
    check(ok && steps_seen[0] > 0 && steps_seen[0] == steps_seen[1],
          "eg_run over a caller-bound steps word: the planes, the last budget and the finished count equal the run over eagle's own word");
    eg_unit_free(ctx, ua);
}

/* The mix counters of a two-group AUTO unit, eagle's and the caller's. */
static void check_counters_mix(eg_ctx* ctx) {
    static double k[RN], limit[RN];
    static unsigned char mask[RN];
    eg_unit* um = NULL;
    check(load(ctx, "mix2", 0xBA, &um) == EG_OK, "eg_unit_load mix2 (two AUTO dispatch groups)");
    if (um == NULL) return;
    struct tensor t[3];
    count_inputs(k, limit, mask);
    memset(mask, 0, sizeof(mask));
    uint32_t want[5] = { 0, 0, 0, 0, 0 };    /* group a: limit < 6, >= 6; group b: limit mod 3 */
    for (int64_t i = 0; i < RN; ++i) {
        ++want[limit[i] >= 6.0 ? 1 : 0];
        ++want[2 + (int64_t)limit[i] % 3];
    }
    eg_plan* p = count_plan(ctx, um, EG_STRUCTURE_HOST_TEAM, k, limit, mask, t);
    struct eg_plan_counters c;
    uint32_t mix[5] = { 9, 9, 9, 9, 9 };
    int32_t n_mix = -1;
    int ok = p != NULL && eg_launch(ctx, p, NULL) == EG_OK && counters_of(ctx, p, &c, mix, 0, &n_mix) == EG_OK && n_mix == 5 && mix[0] == 9;
    check(ok, "eg_plan_counters: mix_cap 0 reports the count (2 + 3) and writes nothing");
    ok = p != NULL && counters_of(ctx, p, &c, mix, 4, &n_mix) == EG_BAD_REQUEST && n_mix == 5 && reason_has(ctx, "mix_cap 4") && mix[0] == 9;
    check(ok, "eg_plan_counters: mix_cap below the count refused BAD_REQUEST, the count still reported, nothing written");
    ok = p != NULL && counters_of(ctx, p, &c, mix, 5, &n_mix) == EG_OK && n_mix == 5 && memcmp(mix, want, sizeof(want)) == 0 &&
         c.finished == count_at_most(limit, 1.0);
    check(ok, "eg_plan_counters: the two groups' mix counters, concatenated in group order, equal the counts of the branch choices");
    ok = p != NULL && counters_of(ctx, p, &c, NULL, 5, &n_mix) == EG_BAD_REQUEST && reason_has(ctx, "mix is NULL");
    check(ok, "eg_plan_counters: a NULL mix with mix_cap > 0 refused BAD_REQUEST");
    eg_plan_free(ctx, p);

    /* the second group's counters bound by the caller: read from there, and the launch fills them (eagle zeroes only its own) */
    uint32_t own_b[3] = { 100, 100, 100 };
    struct tensor tb;
    eg_plane* pb = wrap(ctx, &tb, own_b, 3, kDLUInt, 32);
    count_inputs(k, limit, mask);
    memset(mask, 0, sizeof(mask));
    p = count_plan(ctx, um, EG_STRUCTURE_HOST_SERIAL, k, limit, mask, t);
    const int32_t slot = 5;
    struct eg_bind_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_planes = 1;
    d.plane_slots = &slot;
    d.planes = &pb;
    ok = p != NULL && pb != NULL && eg_bind(ctx, p, &d) == EG_OK && eg_launch(ctx, p, NULL) == EG_OK &&
         counters_of(ctx, p, &c, mix, 5, &n_mix) == EG_OK && mix[0] + mix[1] == (uint32_t)RN && mix[2] == own_b[0] && mix[3] == own_b[1] &&
         mix[4] == own_b[2] && own_b[0] + own_b[1] + own_b[2] == 300u + RN;
    check(ok, "eg_plan_counters reads a caller-bound mix counter from the caller's plane");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, pb);
    eg_unit_free(ctx, um);
}

static void check_counters(eg_ctx* ctx) {
    check_counters_finished(ctx);
    check_counters_run(ctx);
    check_counters_mix(ctx);
}

/* A launch writes each cell as the reduction over the lanes it RAN: a prefix set of live count 8, then 3, 0 and 5, over a SUM cell of
 * rows of 4, gives the sum of the first count lanes each time (the identity 0 for none), never the cell's old value or a stale row. */
static void check_cell_rows(eg_ctx* ctx) {
    enum { N = 8 };
    eg_unit* unit = NULL;
    check(load(ctx, "prefixsum", 0xBC, &unit) == EG_OK, "eg_unit_load prefixsum (a prefix set with a SUM cell of rows of 4)");
    if (unit == NULL) return;
    double x[N], tot = 0.0;
    uint32_t count = 0;
    for (int i = 0; i < N; ++i) x[i] = (double)(i + 1);
    struct tensor tt, tx, tc;
    eg_plane *pt = wrap(ctx, &tt, &tot, 1, kDLFloat, 64), *px = wrap(ctx, &tx, x, N, kDLFloat, 64), *pc = wrap(ctx, &tc, &count, 1, kDLUInt, 32);
    const int32_t slots[3] = { 0, 2, 3 };
    eg_plane* planes[3] = { pt, px, pc };
    struct eg_bind_desc bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.n_planes = 3;
    bd.plane_slots = slots;
    bd.planes = planes;
    const uint32_t counts[5] = { 8, 3, 0, 5, 0 };
    for (int si = 0; si < 2; ++si) {
        eg_plan* p = NULL;
        int ok = pt && px && pc && plan(ctx, unit, si == 0 ? EG_STRUCTURE_HOST_SERIAL : EG_STRUCTURE_HOST_TEAM, N, 0, NULL, NULL, &p) == EG_OK &&
                 eg_bind(ctx, p, &bd) == EG_OK;
        for (int k = 0; ok && k < 5; ++k) {
            count = counts[k];
            tot = 9.0;    /* what the cell held: never part of the result */
            double want = 0.0;
            for (uint32_t i = 0; i < count; ++i) want += x[i];
            ok = eg_launch(ctx, p, NULL) == EG_OK && tot == want;
            if (!ok) printf("      count %u: cell %.17g, want %.17g\n", count, tot, want);
        }
        check(ok, si == 0 ? "a SUM cell over a prefix set of live count 8, 3, 0, 5, 0 (SERIAL): the sum of the lanes that ran, 0 for none"
                          : "a SUM cell over a prefix set of live count 8, 3, 0, 5, 0 (TEAM): the sum of the lanes that ran, 0 for none");
        eg_plan_free(ctx, p);
    }
    eg_plane_free(ctx, pt), eg_plane_free(ctx, px), eg_plane_free(ctx, pc);
    eg_unit_free(ctx, unit);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <fixture> <wrong size> <no size block> <v1 tag>\n", argv[0]);
        return 2;
    }
    library_path = argv[1];
    eg_ctx* ctx = make_ctx();
    check(ctx != NULL, "eg_ctx_new host context");
    if (ctx == NULL) return 1;
    device_group = eg_stream_use(ctx, 0) != EG_NOT_WRITTEN;    /* a host context has no stream: BAD_REQUEST with the group built */
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    check(eg_ctx_info(ctx, &info) == EG_OK && info.abi_major == EG_DOOR_VERSION_MAJOR &&
              info.device == -1 && (info.capabilities & EG_CAP_HOST) && info.host_threads >= 1 &&
              info.struct_size == sizeof(info),
          "eg_ctx_info: major 1, host only, threads >= 1");
    check(eg_host_threads(ctx, 4) == EG_OK && eg_ctx_info(ctx, &info) == EG_OK && info.host_threads == 4,
          "eg_host_threads(4) is in effect for this context");
    eg_unit* unit = NULL;
    check(load(ctx, "axpb", 0xA1, &unit) == EG_OK && unit != NULL, "eg_unit_load axpb");
    if (unit == NULL) return 1;

    oracle(ctx, unit, 64, 32, "n=64");
    oracle(ctx, unit, 100001, 50000, "n=100001");
    eg_host_threads(ctx, 1);
    oracle(ctx, unit, 100001, 50000, "n=100001 threads=1");
    check_runner(ctx, unit);
    check_weight_row(ctx);
    check_auto_look(ctx);
    check_segmented_runner(ctx);
    check_segmented_drift(ctx);
    check_segmented_threads();
    check_active_set(ctx);
    check_auto(ctx);
    check_counters(ctx);
    check_cell_rows(ctx);
    check_slot_kinds(ctx);
    check_mask_dtypes(ctx);
    check_index_sets(ctx);
    check_groups(ctx);
    check_wide(ctx);
    check_wide_flat(ctx);
    check_gather(ctx);
    check_shared(ctx);
    check_box(ctx);
    check_even_split(ctx);
    check_made(ctx);
    check_alloc(ctx, unit);
    check_clear(ctx, unit);
    check_set_count(ctx, unit);
    check_unit_flags(ctx);
    check_banded(ctx);
    check_figures(ctx);
    check_reset(ctx);
    if (!device_group) check_no_device_group(ctx);
    check_device_count(ctx);

    check_reduce(ctx);
    fft_values(ctx, 0);
    fft_adjoint(ctx, 0);
    fft_in_place(ctx, 0);
    fft_recorded(ctx, 0);
    fft_refusals(ctx, 0);
    check_fft_threads();
    check_conditionals(ctx);
    check_recording_states(ctx);
    {
        eg_unit *c1 = NULL;
        check(load(ctx, "count1", 0xD1, &c1) == EG_OK, "eg_unit_load count1 for the recorded list");
        if (c1 != NULL) {
            eg_host_threads(ctx, 3);
            check_recorded_list(ctx, unit, c1);
            eg_host_threads(ctx, 1);
            eg_unit_free(ctx, c1);
        }
    }
    check_free_with_graph();
    check_free_with_exports();
    check_settings();
    check_error_protocol(ctx);
    check_bad_input(ctx, unit);
    check_foreign_handle(ctx, unit);
    check_second_thread(ctx);
    {
        eg_ctx* other = make_ctx();
        check(other != NULL, "eg_ctx_new second host context");
        if (other != NULL) check_queued_calls(ctx, other);
        eg_ctx_free(other);
    }
    check_size_block(ctx, argv + 2);
    check_fork(ctx);

    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
