/*
 * The no-graph entries of a device runner over a recorded automatic unit (finish steps AUTO): a batch the size
 * rule sends to the fused-one entry and one it sends to the persist entry, run twice through eg_runner_new /
 * eg_run, every plane and the report compared BITWISE with the old device runner's. With "extras": the entry the
 * rule picks at the sizes between, read off the report (closed form: one launch over every sample, a second run
 * of a finished batch takes no step), and what such a runner refuses by name.
 *
 * usage: fast_pair_test <unit library> <digest hex> <unit> <oracle case dir> <max_steps> <mode 2|3> [extras]
 * oracle case dir: planes.txt (slot name dlpack-code bits | uniform slot value), <name>_in.bin,
 * <name>_out{1,2}.bin, report{1,2}.i64 = [n, finished, launches, steps, mapped, done, exact, mode].
 */
#define _POSIX_C_SOURCE 200809L
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <stdlib.h>

#define MAXP 12

struct spec {
    int slot, code, bits;
    char name[32];
};
static struct spec specs[MAXP];
static int nspecs = 0, uslot = -1;
static double uvalue = 0.0;
static const char* dir;

struct run_planes {
    void* d[MAXP];
    struct tensor t[MAXP];
    eg_plane* p[MAXP];
};

static int read_specs(void) {
    char path[1024], word[32];
    snprintf(path, sizeof(path), "%s/planes.txt", dir);
    FILE* f = fopen(path, "r");
    if (f == NULL) return 0;
    while (fscanf(f, "%31s", word) == 1) {
        if (strcmp(word, "uniform") == 0) {
            if (fscanf(f, "%d %lf", &uslot, &uvalue) != 2) break;
        } else if (nspecs < MAXP) {
            struct spec* s = &specs[nspecs++];
            s->slot = atoi(word);
            if (fscanf(f, "%31s %d %d", s->name, &s->code, &s->bits) != 3) break;
        }
    }
    fclose(f);
    return nspecs > 0 && uslot >= 0;
}

/* The first @p n samples of each recorded input plane, on the device. */
static int planes_new(eg_ctx* ctx, struct run_planes* r, int64_t n, int64_t n_file) {
    for (int i = 0; i < nspecs; ++i) {
        const size_t width = (size_t)specs[i].bits / 8, bytes = (size_t)n * width;
        char file[64];
        snprintf(file, sizeof(file), "%s_in.bin", specs[i].name);
        void* host = malloc((size_t)n_file * width);
        int ok = host != NULL && read_file(dir, file, host, (size_t)n_file * width) && cudaMalloc(&r->d[i], bytes) == cudaSuccess &&
                 cudaMemcpy(r->d[i], host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        free(host);
        if (!ok) return 0;
        struct tensor* t = &r->t[i];
        tensor_fill(t, n, r->d[i], (uint8_t)specs[i].code, (uint8_t)specs[i].bits);
        r->p[i] = NULL;
        if (eg_plane_wrap(ctx, &t->managed, &r->p[i]) != EG_OK) return 0;
    }
    return 1;
}

static void planes_free(eg_ctx* ctx, struct run_planes* r) {
    for (int i = 0; i < nspecs; ++i) {
        eg_plane_free(ctx, r->p[i]);
        cudaFree(r->d[i]);
    }
}

static eg_plan* plan_bound(eg_ctx* ctx, eg_unit* unit, int64_t n, int64_t parts, const struct run_planes* r) {
    const int64_t base[2] = { 0, n / 2 }, count[2] = { n / 2, n - n / 2 };
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = unit;
    pd.structure = EG_STRUCTURE_DEVICE;
    pd.n_samples = n;
    pd.n_partitions = parts;
    pd.part_base = parts ? base : NULL;
    pd.part_count = parts ? count : NULL;
    eg_plan* plan = NULL;
    if (eg_plan_new(ctx, &pd, &plan) != EG_OK) return NULL;
    int32_t slots[MAXP];
    for (int i = 0; i < nspecs; ++i) slots[i] = specs[i].slot;
    const int32_t us[1] = { uslot };
    const void* values[1] = { &uvalue };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = nspecs;
    b.plane_slots = slots;
    b.planes = r->p;
    b.n_uniforms = 1;
    b.uniform_slots = us;
    b.uniform_values = values;
    if (eg_bind(ctx, plan, &b) != EG_OK) {
        eg_plan_free(ctx, plan);
        return NULL;
    }
    return plan;
}

static int run(eg_ctx* ctx, eg_runner* runner, struct eg_run_report* r) {
    memset(r, 0, sizeof(*r));
    r->struct_size = sizeof(*r);
    return eg_run(ctx, runner, r) == EG_OK;
}

static void compare(const char* unit, int run_no, int64_t n, const struct run_planes* rp, const struct eg_run_report* r) {
    char file[96], what[320];
    for (int i = 0; i < nspecs; ++i) {
        const size_t bytes = (size_t)n * (size_t)specs[i].bits / 8;
        void* got = malloc(bytes);
        void* want = malloc(bytes);
        snprintf(file, sizeof(file), "%s_out%d.bin", specs[i].name, run_no);
        const int ok = got && want && read_file(dir, file, want, bytes) && cudaMemcpy(got, rp->d[i], bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                       memcmp(got, want, bytes) == 0;
        snprintf(what, sizeof(what), "%s n %lld run %d: %s bitwise vs the old device runner", unit, (long long)n, run_no, specs[i].name);
        check(ok, what);
        free(got);
        free(want);
    }
    int64_t want[8];
    snprintf(file, sizeof(file), "report%d.i64", run_no);
    const int64_t have[8] = { r->n, r->finished, r->launches, r->steps, r->mapped, (r->flags & EG_REPORT_DONE) != 0,
                              (r->flags & EG_REPORT_EXACT_STEPS) != 0, r->mode };
    snprintf(what, sizeof(what), "%s n %lld run %d: report n %lld finished %lld launches %lld steps %lld mapped %lld done %lld "
             "exact %lld mode %lld equals the old device runner's", unit, (long long)n, run_no, (long long)have[0], (long long)have[1],
             (long long)have[2], (long long)have[3], (long long)have[4], (long long)have[5], (long long)have[6], (long long)have[7]);
    check(read_file(dir, file, want, sizeof(want)) && memcmp(want, have, sizeof(have)) == 0 && r->compactions == 0 &&
              r->build_s == 0.0, what);
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        fprintf(stderr, "usage: %s <lib> <digest> <unit> <oracle case dir> <max_steps> <mode> [extras]\n", argv[0]);
        return 2;
    }
    dir = argv[4];
    const int64_t max_steps = atoll(argv[5]), want_mode = atoll(argv[6]);
    int64_t rep1[8];
    if (!read_specs() || !read_file(dir, "report1.i64", rep1, sizeof(rep1))) {
        fprintf(stderr, "oracle files missing in %s\n", dir);
        return 2;
    }
    const int64_t n = rep1[0];
    struct eg_ctx_settings st;
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    st.flags = EG_CTX_DEVICE;
    st.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    check(eg_ctx_new(&st, &ctx) == EG_OK, "eg_ctx_new device context (device 0)");
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = argv[3];
    struct eg_unit_load_desc ld;
    memset(&ld, 0, sizeof(ld));
    ld.struct_size = sizeof(ld);
    ld.path = argv[1];
    ld.unit = &ref;
    eg_unit* unit = NULL;
    check(parse_digest(argv[2], ref.digest) && eg_unit_load(ctx, &ld, &unit) == EG_OK,
          "eg_unit_load of the recorded automatic unit (PTX image, its persist entry)");
    if (unit == NULL) return 1;
    char what[256];
    struct run_planes rp;
    eg_plan* plan = planes_new(ctx, &rp, n, n) ? plan_bound(ctx, unit, n, 0, &rp) : NULL;
    eg_runner* runner = NULL;
    struct eg_runner_desc rd = runner_desc(&plan, max_steps);
    struct eg_run_report r;
    snprintf(what, sizeof(what), "%s n %lld: DEVICE plan, eg_runner_new and eg_run OK, mode %lld (2 fused-one, 3 persist)",
             argv[3], (long long)n, (long long)want_mode);
    int ok = plan != NULL && eg_runner_new(ctx, &rd, &runner) == EG_OK && run(ctx, runner, &r);
    check(ok && r.mode == want_mode, what);
    if (ok) compare(argv[3], 1, n, &rp, &r);
    ok = ok && run(ctx, runner, &r);
    if (ok) compare(argv[3], 2, n, &rp, &r);
    if (argc == 8) {
        /* refusals by name */
        struct eg_runner_desc bad = rd;
        eg_runner* none = NULL;
        bad.every = 3;
        check(eg_runner_new(ctx, &bad, &none) == EG_BAD_REQUEST && reason_has(ctx, "picks the steps of each launch") && none == NULL,
              "automatic unit: every refused BAD_REQUEST (the runner picks the steps)");
        bad = rd;
        bad.flags = EG_RUN_REORDER;
        check(eg_runner_new(ctx, &bad, &none) == EG_NOT_WRITTEN && reason_has(ctx, "EG_RUN_REORDER"),
              "automatic unit: EG_RUN_REORDER refused NOT_WRITTEN by name");
        bad = rd;
        bad.flags = EG_RUN_RECORD;
        const eg_derived_role tape_role[1] = { EG_DERIVED_TAPE };
        bad.n_derived = 1;
        bad.derived_roles = tape_role;
        bad.derived_units = &unit;
        check(eg_runner_new(ctx, &bad, &none) == EG_BAD_REQUEST && reason_has(ctx, "has no derived record"),
              "automatic unit: EG_RUN_RECORD over a unit with no derived record refused BAD_REQUEST by name");
        struct run_planes two;
        eg_plan* halves = planes_new(ctx, &two, 1000, n) ? plan_bound(ctx, unit, 1000, 2, &two) : NULL;
        bad = runner_desc(&halves, max_steps);
        check(halves != NULL && eg_runner_new(ctx, &bad, &none) == EG_NOT_WRITTEN && reason_has(ctx, "automatic band loop"),
              "automatic unit over two partitions: the policy band loop refused NOT_WRITTEN by name");
        eg_plan_free(ctx, halves);
        planes_free(ctx, &two);
        /* the size rule between the sure sizes: at most one wave fused-one, above it persist */
        const int64_t sizes[2] = { 1000, 5000 }, modes[2] = { EG_MODE_FUSED_ONE, EG_MODE_PERSIST };
        for (int s = 0; s < 2; ++s) {
            struct run_planes sp;
            eg_plan* sized = planes_new(ctx, &sp, sizes[s], n) ? plan_bound(ctx, unit, sizes[s], 0, &sp) : NULL;
            eg_runner* sr = NULL;
            struct eg_runner_desc sd = runner_desc(&sized, max_steps);
            struct eg_run_report a, b;
            ok = sized != NULL && eg_runner_new(ctx, &sd, &sr) == EG_OK && run(ctx, sr, &a) && run(ctx, sr, &b);
            snprintf(what, sizeof(what), "n %lld: mode %lld, one launch over every sample, all finished; the second run takes "
                     "%s", (long long)sizes[s], (long long)modes[s], modes[s] == EG_MODE_PERSIST ? "no step" : "the budget offered");
            check(ok && a.mode == modes[s] && a.launches == 1 && a.mapped == sizes[s] && a.finished == sizes[s] &&
                      (a.flags & EG_REPORT_DONE) && a.steps <= max_steps && b.finished == sizes[s] &&
                      b.steps == (modes[s] == EG_MODE_PERSIST ? 0 : max_steps),
                  what);
            eg_runner_free(ctx, sr);
            eg_plan_free(ctx, sized);
            planes_free(ctx, &sp);
        }
    }
    check(eg_runner_free(ctx, runner) == EG_OK && eg_plan_free(ctx, plan) == EG_OK && eg_unit_free(ctx, unit) == EG_OK,
          "eg_runner_free, eg_plan_free, eg_unit_free");
    planes_free(ctx, &rp);
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
