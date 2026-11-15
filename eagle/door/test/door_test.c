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

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"

#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int failures = 0;
static const char* library_path = NULL;

static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

/* ---- DLPack tensors over test-owned buffers ------------------------------ */

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[1];
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
    check(plan(ctx, unit, EG_STRUCTURE_DEVICE, 4, 0, NULL, NULL, &p) == EG_NOT_WRITTEN,
          "DEVICE structure refused NOT_WRITTEN");

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
    check(load(ctx, "axpb_split", 0xC3, &split) == EG_NOT_WRITTEN && split == NULL,
          "split unit (n_stages > 0) refused NOT_WRITTEN");
    eg_unit* wrong = NULL;
    check(load(ctx, "axpb", 0xC3, &wrong) == EG_BAD_REQUEST, "right name, wrong digest refused BAD_REQUEST");
    check(eg_run(ctx, NULL, NULL) == EG_NOT_WRITTEN, "eg_run refused NOT_WRITTEN");
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

struct busy_run {
    eg_ctx* ctx;
    eg_plan* plan;
    eg_status status;
};

static void* busy_launch(void* arg) {
    struct busy_run* r = arg;
    uint64_t tiles = 0;
    r->status = eg_launch(r->ctx, r->plan, &tiles);
    return NULL;
}

/* A second thread entering a context while a launch runs is refused. */
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
    check(ready, "busy fixture: wait unit loaded, planned and bound");
    if (!ready) return;
    volatile double* flag = y;
    struct busy_run r = { ctx, p, -1 };
    pthread_t th;
    pthread_create(&th, NULL, busy_launch, &r);
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int entered = 0;
    for (;;) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (flag[1] != 0.0) { entered = 1; break; }
        clock_gettime(CLOCK_MONOTONIC, &t);
        if (t.tv_sec - t0.tv_sec > 10) break;
    }
    struct eg_ctx_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    const eg_status during = eg_ctx_info(ctx, &info);
    size_t len = 0;
    const eg_status err_during = eg_error(ctx, NULL, 0, &len);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    flag[0] = 1.0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    pthread_join(th, NULL);
    check(entered && during == EG_BAD_REQUEST && err_during == EG_BAD_REQUEST,
          "second thread entering a busy context refused BAD_REQUEST");
    check(r.status == EG_OK && eg_ctx_info(ctx, &info) == EG_OK, "the first thread's launch finished OK");
    eg_plan_free(ctx, p);
    eg_plane_free(ctx, px);
    eg_plane_free(ctx, py);
    eg_unit_free(ctx, wait);
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

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <fixture> <wrong size> <no size block> <v1 tag>\n", argv[0]);
        return 2;
    }
    library_path = argv[1];
    eg_ctx* ctx = make_ctx();
    check(ctx != NULL, "eg_ctx_new host context");
    if (ctx == NULL) return 1;
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

    check_settings();
    check_error_protocol(ctx);
    check_bad_input(ctx, unit);
    check_foreign_handle(ctx, unit);
    check_second_thread(ctx);
    check_size_block(ctx, argv + 2);
    check_fork(ctx);

    check(eg_unit_free(ctx, unit) == EG_OK && eg_ctx_free(ctx) == EG_OK, "eg_unit_free and eg_ctx_free");
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
