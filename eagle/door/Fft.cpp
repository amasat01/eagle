// SPDX-License-Identifier: Apache-2.0

// Transform plans (eg_fft_new, eg_fft_run, eg_fft_free): the descriptor's
// checks, the plan as it runs (an adjoint resolved to the transform that
// computes it), the planes' checks at each run and the handle. The numerics
// are behind the seams: pocketfft in host/Fft.cpp, cuFFT in device/Fft.cpp.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Team.h"

#include <memory>

namespace eagle {
namespace door {

namespace {

/** @brief The norm of the adjoint: BACKWARD and FORWARD swap, ORTHO stays. */
eg_fft_norm adjoint_norm(eg_fft_norm norm)
{
    if (norm == EG_FFT_NORM_ORTHO)
        return norm;
    return norm == EG_FFT_NORM_BACKWARD ? EG_FFT_NORM_FORWARD
                                        : EG_FFT_NORM_BACKWARD;
}

/** @brief @p shape as the text "(4, 5, 2)". */
std::string shape_text(const std::vector<std::int64_t>& shape)
{
    std::string text = "(";
    for (std::size_t i = 0; i < shape.size(); ++i)
        text += (i > 0 ? ", " : "") + std::to_string(shape[i]);
    return text + ")";
}

/** @brief Refuse the descriptor @p d unless it is well formed. */
void check_desc(const eg_fft_desc& d)
{
    if (d.flags & ~(EG_FFT_INVERSE | EG_FFT_ADJOINT))
        refuse(EG_BAD_REQUEST,
            "eg_fft_desc.flags: unknown bits "
                + std::to_string(d.flags & ~(EG_FFT_INVERSE | EG_FFT_ADJOINT)));
    if (d.kind < EG_FFT_C2C || d.kind > EG_FFT_C2R)
        refuse(EG_BAD_REQUEST,
            "eg_fft_desc.kind " + std::to_string(d.kind)
                + " is not C2C, R2C or C2R");
    if (d.norm < EG_FFT_NORM_BACKWARD || d.norm > EG_FFT_NORM_ORTHO)
        refuse(EG_BAD_REQUEST,
            "eg_fft_desc.norm " + std::to_string(d.norm)
                + " is not BACKWARD, FORWARD or ORTHO");
    if ((d.flags & EG_FFT_INVERSE) && d.kind != EG_FFT_C2C)
        refuse(EG_BAD_REQUEST,
            "EG_FFT_INVERSE is for C2C: R2C is forward and C2R inverse");
    if (d.dtype_bits == 16 || d.dtype_bits == 128)
        refuse(EG_NOT_WRITTEN,
            "eg_fft_desc.dtype_bits " + std::to_string(d.dtype_bits)
                + ": float widths 32 and 64 are written");
    if (d.dtype_bits != 32 && d.dtype_bits != 64)
        refuse(EG_BAD_REQUEST,
            "eg_fft_desc.dtype_bits " + std::to_string(d.dtype_bits)
                + " is not a float width (32 or 64)");
    if (d.ndim < 1 || d.shape == nullptr)
        refuse(EG_BAD_REQUEST,
            "eg_fft_desc: ndim " + std::to_string(d.ndim) + " without a shape");
    if (d.n_axes < 1 || d.axes == nullptr)
        refuse(EG_BAD_REQUEST, "eg_fft_desc: no axes to transform");
    for (int i = 0; i < d.ndim; ++i)
        if (d.shape[i] < 0)
            refuse(EG_BAD_REQUEST,
                "eg_fft_desc.shape[" + std::to_string(i) + "] is negative");
    for (int i = 0; i < d.n_axes; ++i) {
        const std::int32_t a = d.axes[i];
        if (a < 0 || a >= d.ndim)
            refuse(EG_BAD_REQUEST,
                "eg_fft_desc.axes[" + std::to_string(i)
                    + "] = " + std::to_string(a) + " is outside [0, "
                    + std::to_string(d.ndim) + ")");
        for (int j = 0; j < i; ++j)
            if (d.axes[j] == a)
                refuse(EG_BAD_REQUEST,
                    "eg_fft_desc.axes names axis " + std::to_string(a)
                        + " twice");
        if (d.shape[a] < 1)
            refuse(EG_BAD_REQUEST,
                "eg_fft_desc: transformed axis " + std::to_string(a)
                    + " has length 0");
    }
}

/** @brief The plan @p d describes as it runs: an adjoint becomes the transform
 * that computes it. */
std::unique_ptr<eg_fft_s> resolve(eg_ctx& ctx, const eg_fft_desc& d)
{
    auto f     = std::make_unique<eg_fft_s>();
    f->ctx     = &ctx;
    f->device  = ctx.dev.ordinal >= 0;
    f->kind    = d.kind;
    f->forward = (d.flags & EG_FFT_INVERSE) == 0;
    f->norm    = d.norm;
    f->bits    = d.dtype_bits;
    f->shape.assign(d.shape, d.shape + d.ndim);
    f->axes.assign(d.axes, d.axes + d.n_axes);
    if (d.flags & EG_FFT_ADJOINT) {
        f->norm = adjoint_norm(d.norm);
        if (d.kind == EG_FFT_C2C) {
            f->forward = !f->forward;
        } else {
            f->kind   = d.kind == EG_FFT_R2C ? EG_FFT_C2R : EG_FFT_R2C;
            f->weight = d.kind == EG_FFT_R2C ? 0.5 : 2.0;
        }
    }
    if (f->kind != EG_FFT_C2C) {
        const std::int64_t n = f->shape[f->axes.back()];
        f->bins              = n / 2 + 1;
        f->edge              = n % 2 == 0 ? f->bins - 1 : f->bins;
        f->run               = 2;
        for (std::size_t a = f->axes.back() + 1; a < f->shape.size(); ++a)
            f->run *= f->shape[a];
    }
    return f;
}

/** @brief The shape of the plane of @p f that is read (@p input) or written. */
std::vector<std::int64_t> plane_shape(const eg_fft_s& f, bool input)
{
    std::vector<std::int64_t> shape = f.shape;
    const bool complex_side
        = f.kind == EG_FFT_C2C || (f.kind == EG_FFT_C2R) == input;
    if (!complex_side)
        return shape;
    if (f.kind != EG_FFT_C2C)
        shape[f.axes.back()] = f.bins;
    shape.push_back(2);
    return shape;
}

/** @brief Refuse plane @p p (named @p what) unless the plan @p f can read, or
 * write, it. */
void check_plane(
    const eg_fft_s& f, const eg_plane_s& p, bool input, const char* what)
{
    const std::string name = std::string("eg_fft_run ") + what;
    host::check_access(p, !input, name.c_str());
    const DLTensor& t = p.tensor->dl_tensor;
    if ((t.device.device_type != kDLCPU) != f.device)
        refuse(EG_BAD_REQUEST,
            name + ": the plan runs on the " + (f.device ? "device" : "host")
                + " and this plane is on the "
                + (f.device ? "host" : "device"));
    if (t.dtype.code != kDLFloat || t.dtype.bits != f.bits
        || t.dtype.lanes != 1)
        refuse(EG_BAD_REQUEST,
            name + ": the plane must be float" + std::to_string(f.bits)
                + " (the plan's width)");
    const std::vector<std::int64_t> want = plane_shape(f, input);
    const std::vector<std::int64_t> have(t.shape, t.shape + t.ndim);
    if (have != want)
        refuse(EG_BAD_REQUEST,
            name + ": the plane must have shape " + shape_text(want) + ", got "
                + shape_text(have));
}

} // namespace

/** @brief eg_fft_new: check, resolve, build the device plans, register. */
void fft_new(eg_ctx& ctx, const eg_fft_desc* desc, eg_fft** out)
{
    if (out == nullptr)
        refuse(EG_BAD_REQUEST, "eg_fft_new: out is NULL");
    *out                = nullptr;
    const eg_fft_desc d = read_input(desc, "eg_fft_desc");
    check_desc(d);
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST,
            "eg_fft_new while recording (eg_record_begin .. eg_record_end): "
            "make the plan before recording");
    std::unique_ptr<eg_fft_s> f = resolve(ctx, d);
    if (f->device)
        device_fft_new(ctx, *f);
    ctx.ffts.insert(f.get());
    *out = f.release();
}

/** @brief eg_fft_run: check the planes, then run on the plan's target (or keep
 * the run while recording). */
void fft_run(eg_ctx& ctx, eg_fft_s* plan, eg_plane_s* in, eg_plane_s* out)
{
    check_handle(ctx.ffts, plan, "plan");
    check_handle(ctx.planes, in, "in");
    check_handle(ctx.planes, out, "out");
    check_plane(*plan, *in, true, "in");
    check_plane(*plan, *out, false, "out");
    const DLTensor& a = in->tensor->dl_tensor;
    const DLTensor& b = out->tensor->dl_tensor;
    if (host::overlaps(a, b)
        && (plan->kind != EG_FFT_C2C || a.data != b.data
            || a.byte_offset != b.byte_offset))
        refuse(EG_BAD_REQUEST,
            plan->kind == EG_FFT_C2C
                ? "eg_fft_run: in and out overlap without being one range: C2C "
                  "runs in place only on the same plane"
                : "eg_fft_run: in and out overlap: only C2C runs in place");
    if (plan->device)
        device_fft_run(ctx, *plan, in, out);
    else if (!record_fft(ctx, plan, in, out))
        host::fft_exec(*plan, *in, *out, host::team_threads(ctx));
}

/** @brief The adjoint of the transform as run: its direction and norm swap; a real transform becomes its partner with the Hermitian weight
 * (R2C's adjoint is C2R reading half of each interior bin, C2R's is R2C writing twice each interior bin). */
eg_fft_s* fft_adjoint(eg_ctx& ctx, const eg_fft_s& plan)
{
    auto f        = std::make_unique<eg_fft_s>(plan);
    f->dev        = nullptr;
    f->work       = nullptr;
    f->refs       = 1;
    f->caller_ref = false;
    f->norm       = adjoint_norm(plan.norm);
    if (plan.kind == EG_FFT_C2C) {
        f->forward = !plan.forward;
    } else if (plan.kind == EG_FFT_R2C) {
        f->kind   = EG_FFT_C2R;
        f->weight = 0.5 * plan.weight;
    } else {
        f->kind   = EG_FFT_R2C;
        f->weight = 2.0 * plan.weight;
    }
    if (f->device)
        device_fft_new(ctx, *f);
    ctx.ffts.insert(f.get());
    return f.release();
}

/** @brief Drop one reference to @p plan; the last one frees it. */
void fft_release(eg_ctx& ctx, eg_fft_s* plan)
{
    if (--plan->refs > 0)
        return;
    ctx.ffts.erase(plan);
    if (plan->device)
        device_fft_free(ctx, *plan);
    delete plan;
}

/** @brief Free every plan of @p ctx; a forked child makes no device call. */
void fft_free_all(eg_ctx& ctx)
{
    for (eg_fft_s* plan : ctx.ffts) {
        if (plan->device && !forked(ctx))
            device_fft_free(ctx, *plan);
        delete plan;
    }
    ctx.ffts.clear();
}

} // namespace door
} // namespace eagle
