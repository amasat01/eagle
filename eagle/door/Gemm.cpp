// SPDX-License-Identifier: Apache-2.0

// Matrix products (eg_gemm_new, eg_gemm_run, eg_gemm_free): the descriptor's checks, the planes' checks at each run and
// the handle, and the shape rules a gemm unit's record shares with them. The numerics are behind the seams: the
// reference loop in host/Gemm.cpp, cuBLAS in device/Gemm.cpp.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/host/Ops.h"

#include <memory>

namespace eagle {
namespace door {

void gemm_shape(GemmCall& c, const std::string& at) {
    const bool ta = c.flags & EG_GEMM_TRANS_A, tb = c.flags & EG_GEMM_TRANS_B, tc = c.flags & EG_GEMM_TRANS_C;
    const std::int64_t k_a = ta ? c.a_rows : c.a_cols, k_b = tb ? c.b_cols : c.b_rows;
    c.m = ta ? c.a_cols : c.a_rows;
    c.n = tb ? c.b_rows : c.b_cols;
    c.k = k_a;
    if (k_a != k_b)
        refuse(EG_BAD_REQUEST, at + ": the contracted extents disagree: A stored (" + std::to_string(c.a_rows) + ", " + std::to_string(c.a_cols) +
               ") gives " + std::to_string(k_a) + ", B stored (" + std::to_string(c.b_rows) + ", " + std::to_string(c.b_cols) + ") gives " + std::to_string(k_b));
    const std::int64_t want_rows = tc ? c.n : c.m, want_cols = tc ? c.m : c.n;
    if (c.c_rows != want_rows || c.c_cols != want_cols)
        refuse(EG_BAD_REQUEST, at + ": C is stored (" + std::to_string(c.c_rows) + ", " + std::to_string(c.c_cols) + ") but the product of (" + std::to_string(c.m) +
               ", " + std::to_string(c.k) + ") and (" + std::to_string(c.k) + ", " + std::to_string(c.n) + ") is stored (" + std::to_string(want_rows) + ", " +
               std::to_string(want_cols) + ")");
}

void check_gemm_plane(const std::string& at, std::int64_t rows, std::int64_t cols, const DLTensor& t) {
    const std::int64_t count = elements(t);
    if (count % rows != 0)
        refuse(EG_BAD_REQUEST, at + ": a plane of " + std::to_string(count) + " elements is not a whole number of rows of its " + std::to_string(rows) + " stated rows");
    const std::int64_t stride = count / rows;
    if (stride < cols)
        refuse(EG_BAD_REQUEST, at + ": the row stride " + std::to_string(stride) + " is smaller than the " + std::to_string(cols) + " stated columns");
    if (stride > kGemmMaxExtent)
        refuse(EG_BAD_REQUEST, at + ": the row stride " + std::to_string(stride) + " is beyond " + std::to_string(kGemmMaxExtent));
}

namespace {

/** @brief Refuse the descriptor @p d unless it is well formed; the plan's call with its extents resolved. */
GemmCall check_desc(const eg_gemm_desc& d) {
    const std::uint32_t known = EG_GEMM_TRANS_A | EG_GEMM_TRANS_B | EG_GEMM_TRANS_C | EG_GEMM_ACCUMULATE;
    if (d.flags & ~known) refuse(EG_BAD_REQUEST, "eg_gemm_desc.flags: unknown bits " + std::to_string(d.flags & ~known));
    if (d.dtype_bits != 32 && d.dtype_bits != 64)
        refuse(EG_BAD_REQUEST, "eg_gemm_desc.dtype_bits " + std::to_string(d.dtype_bits) + " is not a float width (32 or 64)");
    GemmCall c;
    c.flags = d.flags;
    c.a_rows = d.a_rows, c.a_cols = d.a_cols, c.b_rows = d.b_rows, c.b_cols = d.b_cols, c.c_rows = d.c_rows, c.c_cols = d.c_cols;
    const char* const names[] = { "a_rows", "a_cols", "b_rows", "b_cols", "c_rows", "c_cols" };
    const std::int64_t extents[] = { c.a_rows, c.a_cols, c.b_rows, c.b_cols, c.c_rows, c.c_cols };
    for (int i = 0; i < 6; ++i)
        if (extents[i] < 1 || extents[i] > kGemmMaxExtent)
            refuse(EG_BAD_REQUEST, std::string("eg_gemm_desc.") + names[i] + " " + std::to_string(extents[i]) + " is outside 1 .. " + std::to_string(kGemmMaxExtent));
    gemm_shape(c, "eg_gemm_desc");
    return c;
}

/** @brief Refuse plane @p p (named @p what, of @p rows x @p cols stored) unless the product @p g can read, or write, it. */
void check_plane(const eg_gemm_s& g, const eg_plane_s& p, std::int64_t rows, std::int64_t cols, bool write, const char* what) {
    const std::string name = std::string("eg_gemm_run ") + what;
    host::check_access(p, write, name.c_str());
    const DLTensor& t = p.tensor->dl_tensor;
    if ((t.device.device_type != kDLCPU) != g.device)
        refuse(EG_BAD_REQUEST, name + ": the product runs on the " + (g.device ? "device" : "host") + " and this plane is on the " + (g.device ? "host" : "device"));
    if (t.dtype.code != kDLFloat || t.dtype.bits != (g.f32 ? 32u : 64u) || t.dtype.lanes != 1)
        refuse(EG_BAD_REQUEST, name + ": the plane must be float" + (g.f32 ? "32" : "64") + " (the product's width)");
    if (t.ndim != 2 || t.shape[0] != rows)
        refuse(EG_BAD_REQUEST, name + ": the plane must be 2-D with " + std::to_string(rows) + " rows");
    check_gemm_plane(name, rows, cols, t);
}

}  // namespace

void gemm_new(eg_ctx& ctx, const eg_gemm_desc* desc, eg_gemm** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_gemm_new: out is NULL");
    *out = nullptr;
    const eg_gemm_desc d = read_input(desc, "eg_gemm_desc");
    const GemmCall call = check_desc(d);
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, "eg_gemm_new while recording (eg_record_begin .. eg_record_end): make the product before recording");
    auto g = std::make_unique<eg_gemm_s>();
    g->ctx = &ctx;
    g->device = ctx.dev.ordinal >= 0;
    g->f32 = d.dtype_bits == 32;
    g->call = call;
    if (g->device) device_gemm_ready(ctx);
    ctx.gemms.insert(g.get());
    *out = g.release();
}

std::pair<eg_gemm_s, eg_gemm_s> gemm_adjoints(const eg_gemm_s& g) {
    const GemmCall& f = g.call;
    const bool ta = f.flags & EG_GEMM_TRANS_A, tb = f.flags & EG_GEMM_TRANS_B, tc = f.flags & EG_GEMM_TRANS_C;
    eg_gemm_s a = g, b = g;
    // dA = dC op(B)^T, stored as A is; dB = op(A)^T dC, stored as B is. Each reads the stored planes of the forward, so only the flags change.
    a.call = GemmCall{};
    a.call.flags = (tc ? EG_GEMM_TRANS_A : 0u) | (tb ? 0u : EG_GEMM_TRANS_B) | (ta ? EG_GEMM_TRANS_C : 0u);
    a.call.a_rows = f.c_rows, a.call.a_cols = f.c_cols, a.call.b_rows = f.b_rows, a.call.b_cols = f.b_cols, a.call.c_rows = f.a_rows, a.call.c_cols = f.a_cols;
    gemm_shape(a.call, "eg_record_end: the product of the gradient of A");
    b.call = GemmCall{};
    b.call.flags = (ta ? 0u : EG_GEMM_TRANS_A) | (tc ? EG_GEMM_TRANS_B : 0u) | (tb ? EG_GEMM_TRANS_C : 0u);
    b.call.a_rows = f.a_rows, b.call.a_cols = f.a_cols, b.call.b_rows = f.c_rows, b.call.b_cols = f.c_cols, b.call.c_rows = f.b_rows, b.call.c_cols = f.b_cols;
    gemm_shape(b.call, "eg_record_end: the product of the gradient of B");
    return { a, b };
}

void gemm_run(eg_ctx& ctx, eg_gemm_s* gemm, eg_plane_s* a, eg_plane_s* b, eg_plane_s* c) {
    check_handle(ctx.gemms, gemm, "gemm");
    for (const auto& p : { std::pair<eg_plane_s*, const char*>{ a, "a" }, { b, "b" }, { c, "c" } }) check_handle(ctx.planes, p.first, p.second);
    const GemmCall& k = gemm->call;
    check_plane(*gemm, *a, k.a_rows, k.a_cols, false, "a");
    check_plane(*gemm, *b, k.b_rows, k.b_cols, false, "b");
    check_plane(*gemm, *c, k.c_rows, k.c_cols, true, "c");
    if (host::overlaps(c->tensor->dl_tensor, a->tensor->dl_tensor) || host::overlaps(c->tensor->dl_tensor, b->tensor->dl_tensor))
        refuse(EG_BAD_REQUEST, "eg_gemm_run: c overlaps a or b");
    if (gemm->device) device_gemm_run(ctx, *gemm, a, b, c);
    else if (!record_gemm(ctx, gemm, a, b, c)) host::gemm_exec(*gemm, *a, *b, *c);
}

void gemm_release(eg_ctx& ctx, eg_gemm_s* gemm) {
    if (--gemm->refs > 0) return;
    ctx.gemms.erase(gemm);
    delete gemm;
}

}  // namespace door
}  // namespace eagle
