/* SPDX-License-Identifier: Apache-2.0 */

/*
 * The device forward pieces of a reverse through conditionals and loops, checked without the reverse: the cells of a conditional's test saved at
 * test time on each replay (the caller rewrites the cell afterwards and the saved value stays), the copy a WHILE body makes into the buffer of its
 * iteration, and the read-back of both with the iteration count. The test links the door's objects for the seam (test/device_forward_seam.cpp) and
 * reuses the device fixture helpers of door_device_test.c.
 *
 * Usage: device_forward_test <libfixture_device.so> <libfixture_unit.so>  (run it through the GPU pool)
 */
#define main door_device_main
#include "door_device_test.c"
#undef main

extern eg_status df_scratch(eg_ctx*, size_t, void**);
extern eg_status df_save_test(eg_ctx*, eg_plane*, int64_t, eg_plane*, int64_t, int32_t*);
extern eg_status df_loop_copy(eg_ctx*, eg_plane*, void*, size_t);
extern eg_status df_saved(eg_ctx*, eg_graph*, int, uint32_t*, uint32_t*, int32_t*, uint32_t*, int32_t*);
extern eg_status df_read(eg_ctx*, const void*, void*, size_t);

enum { FN = 64, STRIDE = 8 };

static void say(const char* what, eg_status st, const uint32_t* cells, const int32_t* n_slots, const uint32_t* iterations, const int32_t* n_loops) {
    printf("      %s: status %d, slots %d, cells %u %u %u, loops %d, iterations %u\n", what, (int)st, *n_slots, cells[0], cells[1], cells[2], *n_loops,
           iterations[0]);
}

static eg_status cond_open(eg_ctx* ctx, int kind, eg_plane* cell, int64_t index, int64_t max_iters) {
    struct eg_cond_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.kind = kind;
    d.cell = cell;
    d.cell_index = index;
    d.max_iters = max_iters;
    return eg_record_cond_begin(ctx, &d);
}

/* One call of a chain: its status, printed with the call's text when it is not OK. */
static int step(eg_ctx* ctx, eg_status st, const char* call) {
    if (st == EG_OK) return 1;
    char buf[512];
    size_t len = 0;
    buf[0] = 0;
    eg_error(ctx, buf, sizeof(buf), &len);
    printf("      %s: status %d, reason '%s'\n", call, (int)st, buf);
    return 0;
}
#define STEP(call) step(ctx, (call), #call)

static void set_flags(uint32_t* df, uint32_t f2) {
    const uint32_t f[4] = { 0, 0, f2, 0 };
    cudaMemcpy(df, f, sizeof(f), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <libfixture_device.so> <libfixture_unit.so>\n", argv[0]);
        return 2;
    }
    eg_status s = EG_OK;
    eg_ctx* ctx = context(EG_CTX_DEVICE, 0, &s);
    eg_unit* unit = NULL;
    check(s == EG_OK && load(ctx, argv[1], "dprefix", 0xD6, &unit) == EG_OK, "a device context and the dprefix unit");
    if (unit == NULL) return 1;
    double x[FN], want[3][FN];
    for (int i = 0; i < FN; ++i) x[i] = 0.5 * i - 7.0;
    double *dx = dev_doubles(x, FN), *d1 = dev_doubles(x, FN), *d2 = dev_doubles(NULL, FN);
    uint32_t count = FN, *dc = NULL, *df = NULL;
    cudaMalloc((void**)&dc, sizeof(count));
    cudaMemcpy(dc, &count, sizeof(count), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&df, 4 * sizeof(uint32_t));
    struct tensor tx, t1, t2, tc, tf;
    eg_plane *px = wrap(ctx, &tx, dx, 1, FN, 0, 1, 0), *p1 = wrap(ctx, &t1, d1, 1, FN, 0, 1, 0), *p2 = wrap(ctx, &t2, d2, 1, FN, 0, 1, 0),
             *pc = wrap_typed(ctx, &tc, dc, kDLUInt, 32, 1, 1), *pf = wrap_typed(ctx, &tf, df, kDLUInt, 32, 4, 1);
    const int32_t slots[3] = { 0, 1, 2 };
    eg_plane* a_planes[3] = { p1, px, pc };    /* p1 <- 2 x + 1 */
    eg_plane* w_planes[3] = { p1, p1, pc };    /* p1 <- 2 p1 + 1: a value that differs at each iteration */
    eg_plan *a = NULL, *w = NULL;
    int ok = px && p1 && p2 && pc && pf && plan_n(ctx, unit, EG_STRUCTURE_DEVICE, FN, 0, NULL, NULL, &a) == EG_OK &&
             plan_n(ctx, unit, EG_STRUCTURE_DEVICE, FN, 0, NULL, NULL, &w) == EG_OK && bind_planes(ctx, a, 3, slots, a_planes) == EG_OK &&
             bind_planes(ctx, w, 3, slots, w_planes) == EG_OK;
    check(ok, "the planes and the two plans made and bound");
    if (!ok) return 1;

    /* the test cell saved at test time: the caller rewrites it after the replay, the saved value stays; each replay saves again */
    eg_graph* g = NULL;
    int32_t slot = -1, n_slots = 0, n_loops = 0;
    uint32_t cells[4 * STRIDE], bases[4 * STRIDE], iterations[4];
    set_flags(df, 1);
    ok = STEP(eg_record_begin(ctx)) && STEP(df_save_test(ctx, pf, 2, NULL, 0, &slot)) && STEP(cond_open(ctx, EG_COND_IF, pf, 2, 0)) &&
         STEP(eg_launch(ctx, a, NULL)) && STEP(eg_record_cond_end(ctx)) && STEP(eg_record_end(ctx, NULL, &g));
    ok = ok && STEP(eg_graph_launch(ctx, g)) && STEP(eg_sync(ctx));
    set_flags(df, 0);    /* rewritten after the replay */
    memset(cells, 0xFF, sizeof(cells));
    if (!ok) reason_has(ctx, "\x01");
    ok = ok && (s = df_saved(ctx, g, STRIDE, cells, bases, &n_slots, iterations, &n_loops)) == EG_OK;
    if (!ok) reason_has(ctx, "\x01");
    say("after replay 1", s, cells, &n_slots, iterations, &n_loops);
    check(ok && slot == 0 && n_slots == 1 && n_loops == 0 && cells[0] == 1, "device saved test: lane 2 was 1 at the replay and the save still holds 1 after the caller rewrote it to 0");
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK && df_saved(ctx, g, STRIDE, cells, bases, &n_slots, iterations, &n_loops) == EG_OK;
    check(ok && cells[0] == 0, "device saved test: the next replay saves the lane again (now 0)");
    if (g) eg_graph_free(ctx, g);
    g = NULL;

    /* a three-iteration WHILE: buffer k holds iteration k's plane, the test inside the body is saved per iteration, the count reads back as 3 */
    void* buffer = NULL;
    set_flags(df, 1);
    cudaMemcpy(d1, x, sizeof(x), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    for (int i = 0; i < FN; ++i) {
        want[0][i] = x[i];
        want[1][i] = 2.0 * want[0][i] + 1.0;
        want[2][i] = 2.0 * want[1][i] + 1.0;
    }
    ok = STEP(eg_record_begin(ctx)) && STEP(df_scratch(ctx, 3 * sizeof(x), &buffer)) && STEP(cond_open(ctx, EG_COND_WHILE, pf, 2, 3)) &&
         STEP(df_save_test(ctx, pf, 2, NULL, 0, &slot)) && STEP(df_loop_copy(ctx, p1, buffer, sizeof(x))) && STEP(eg_launch(ctx, w, NULL)) &&
         STEP(eg_record_cond_end(ctx)) && STEP(eg_record_end(ctx, NULL, &g));
    ok = ok && STEP(eg_graph_launch(ctx, g)) && STEP(eg_sync(ctx));
    memset(cells, 0xFF, sizeof(cells));
    if (!ok) reason_has(ctx, "\x01");
    ok = ok && (s = df_saved(ctx, g, STRIDE, cells, bases, &n_slots, iterations, &n_loops)) == EG_OK;
    if (!ok) reason_has(ctx, "\x01");
    say("while", s, cells, &n_slots, iterations, &n_loops);
    check(ok && n_loops == 1 && iterations[0] == 3, "device WHILE max_iters 3 over a test that stays true: the iteration count reads back as 3");
    check(ok && n_slots == 1 && cells[0] == 1 && cells[1] == 1 && cells[2] == 1, "device WHILE: the test saved inside the body holds the cell of each of the 3 iterations");
    double got[3][FN];
    ok = ok && df_read(ctx, buffer, got, sizeof(got)) == EG_OK;
    check(ok && memcmp(got, want, sizeof(got)) == 0, "device WHILE: buffer k holds the plane as iteration k found it (k = 0, 1, 2), bitwise");
    set_flags(df, 0);
    ok = ok && eg_graph_launch(ctx, g) == EG_OK && eg_sync(ctx) == EG_OK && df_saved(ctx, g, STRIDE, cells, bases, &n_slots, iterations, &n_loops) == EG_OK;
    check(ok && iterations[0] == 0, "device WHILE with the test false at the next replay: the iteration count reads back as 0");
    if (g) eg_graph_free(ctx, g);

    check(eg_ctx_free(ctx) == EG_OK, "eg_ctx_free");
    cudaFree(dx), cudaFree(d1), cudaFree(d2), cudaFree(dc), cudaFree(df);
    printf("%s: %d check(s) failed\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
