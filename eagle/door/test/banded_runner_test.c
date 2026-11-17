/*
 * A device runner over a BANDED finishing step unit: planes of 8-byte band cells (wrapped as 64-bit planes), the dt uniform
 * a packed band word. The recorded unit (osc in the banded scalar mode) is run until done by eg_runner_new / eg_run and
 * compared BITWISE, planes, mask and report counts, with the old device runner on the same cells, at a cap that finishes
 * every sample and one that does not; beside it the report's own identities (steps = launches, mapped = launches * n, done
 * iff every sample finished, the mask counts the finished).
 *
 * usage: banded_runner_test <library> <digest hex> <oracle dir>
 * oracle dir: in_{t_final,x,v}.u64, dt.u64, out_{x,v,t}_<cap>.u64, term_<cap>.u8, report_<cap>.i64.
 */
#define _POSIX_C_SOURCE 200809L
#include "eagle/door/eg_door.h"
#include "pair_util.h"

#include <cuda_runtime_api.h>

#include <stdlib.h>

#define N 48
enum { S_T, S_V, S_X, S_TFINAL, S_COUNTER, S_TERM, S_DT };

static void one_cap(eg_ctx* ctx, eg_unit* unit, const char* dir, int64_t cap) {
    uint64_t cells[3][N], want[N], got[N], dt = 0;
    uint8_t term[N], want_term[N];
    char file[64], what[256];
    static const char* in_names[3] = { "in_v.u64", "in_x.u64", "in_t_final.u64" };    /* slots V, X, TFINAL */
    int ok = read_file(dir, "dt.u64", &dt, sizeof(dt));
    for (int i = 0; ok && i < 3; ++i) ok = read_file(dir, in_names[i], cells[i], sizeof(cells[i]));
    void* d[5] = { NULL, NULL, NULL, NULL, NULL };
    struct tensor t[5];
    eg_plane* p[5] = { NULL, NULL, NULL, NULL, NULL };
    static const uint64_t zeros[N];
    static const uint8_t none[N];
    const void* init[5] = { zeros, cells[0], cells[1], cells[2], none };    /* T, V, X, TFINAL, TERM */
    for (int i = 0; ok && i < 5; ++i) {
        const size_t bytes = i < 4 ? sizeof(zeros) : sizeof(none);
        ok = cudaMalloc(&d[i], bytes) == cudaSuccess && cudaMemcpy(d[i], init[i], bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
             (p[i] = wrap(ctx, &t[i], N, d[i], i < 4 ? kDLUInt : kDLBool, i < 4 ? 64 : 8)) != NULL;
    }
    struct eg_plan_desc pd;
    memset(&pd, 0, sizeof(pd));
    pd.struct_size = sizeof(pd);
    pd.unit = unit;
    pd.structure = EG_STRUCTURE_DEVICE;
    pd.n_samples = N;
    eg_plan* plan = NULL;
    const int32_t slots[5] = { S_T, S_V, S_X, S_TFINAL, S_TERM }, uslot[1] = { S_DT };
    const void* values[1] = { &dt };
    struct eg_bind_desc b;
    memset(&b, 0, sizeof(b));
    b.struct_size = sizeof(b);
    b.n_planes = 5;
    b.plane_slots = slots;
    b.planes = p;
    b.n_uniforms = 1;
    b.uniform_slots = uslot;
    b.uniform_values = values;
    eg_runner* runner = NULL;
    struct eg_run_report r;
    memset(&r, 0, sizeof(r));
    r.struct_size = sizeof(r);
    ok = ok && eg_plan_new(ctx, &pd, &plan) == EG_OK && eg_bind(ctx, plan, &b) == EG_OK;
    const struct eg_runner_desc rd = runner_desc(&plan, cap);
    ok = ok && eg_runner_new(ctx, &rd, &runner) == EG_OK && eg_run(ctx, runner, &r) == EG_OK;
    snprintf(what, sizeof(what), "banded cap %lld: DEVICE plan over band cells, eg_runner_new and eg_run OK", (long long)cap);
    check(ok, what);
    if (!ok) {
        reason_has(ctx, "\x01");
        return;
    }
    static const char* outs[3] = { "t", "v", "x" };
    for (int i = 0; i < 3; ++i) {
        snprintf(file, sizeof(file), "out_%s_%lld.u64", outs[i], (long long)cap);
        const int good = read_file(dir, file, want, sizeof(want)) && cudaMemcpy(got, d[i], sizeof(got), cudaMemcpyDeviceToHost) == cudaSuccess;
        snprintf(what, sizeof(what), "banded cap %lld: %s cells bitwise vs the old device runner", (long long)cap, outs[i]);
        check(good && memcmp(got, want, sizeof(got)) == 0, what);
    }
    snprintf(file, sizeof(file), "term_%lld.u8", (long long)cap);
    int good = read_file(dir, file, want_term, sizeof(want_term)) && cudaMemcpy(term, d[4], sizeof(term), cudaMemcpyDeviceToHost) == cudaSuccess;
    int64_t set = 0;
    for (int i = 0; good && i < N; ++i) {
        good = (term[i] != 0) == (want_term[i] != 0);
        set += term[i] != 0;
    }
    snprintf(what, sizeof(what), "banded cap %lld: the mask equals the old device runner's", (long long)cap);
    check(good, what);
    int64_t rep[7];
    snprintf(file, sizeof(file), "report_%lld.i64", (long long)cap);
    good = read_file(dir, file, rep, sizeof(rep));
    const int64_t have[7] = { r.n, r.finished, r.launches, r.steps, r.mapped, (r.flags & EG_REPORT_DONE) != 0, (r.flags & EG_REPORT_EXACT_STEPS) != 0 };
    snprintf(what, sizeof(what), "banded cap %lld: report n %lld finished %lld launches %lld steps %lld mapped %lld done %lld exact %lld "
             "equals the old device runner's, mode BAND", (long long)cap, (long long)have[0], (long long)have[1], (long long)have[2],
             (long long)have[3], (long long)have[4], (long long)have[5], (long long)have[6]);
    check(good && memcmp(rep, have, sizeof(rep)) == 0 && r.mode == EG_MODE_BAND, what);
    snprintf(what, sizeof(what), "banded cap %lld: steps = launches <= cap, mapped = launches * n, done iff all finished, the mask counts the finished",
             (long long)cap);
    check(r.steps == r.launches && r.launches <= cap && r.mapped == r.launches * N && ((r.flags & EG_REPORT_DONE) != 0) == (r.finished == N) &&
              set == (int64_t)r.finished,
          what);
    check(eg_runner_free(ctx, runner) == EG_OK && eg_plan_free(ctx, plan) == EG_OK, "eg_runner_free and eg_plan_free");
    for (int i = 0; i < 5; ++i) {
        eg_plane_free(ctx, p[i]);
        cudaFree(d[i]);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <library> <digest> <oracle dir>\n", argv[0]);
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
    struct eg_unit_ref ref;
    memset(&ref, 0, sizeof(ref));
    ref.struct_size = sizeof(ref);
    ref.name = "bosc";
    struct eg_unit_load_desc ld;
    memset(&ld, 0, sizeof(ld));
    ld.struct_size = sizeof(ld);
    ld.path = argv[1];
    ld.unit = &ref;
    eg_unit* unit = NULL;
    const int ok = parse_digest(argv[2], ref.digest) && eg_unit_load(ctx, &ld, &unit) == EG_OK;
    check(ok, "bosc: eg_unit_load of the recorded BANDED unit (PTX image) on the device context");
    if (ok) {
        one_cap(ctx, unit, argv[3], 400);
        one_cap(ctx, unit, argv[3], 20);
        eg_unit_free(ctx, unit);
    }
    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "OK", failures);
    return failures != 0;
}
