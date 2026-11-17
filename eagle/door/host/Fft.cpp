// SPDX-License-Identifier: Apache-2.0

// The host transform: aether's batched pocketfft (aether/fft/Fft.h) on the
// thread count of the launch, then the plan's Hermitian weight. The only file
// of the door that includes Fft.h.
#include "aether/fft/Fft.h"
#include "eagle/door/host/Ops.h"

namespace eagle {
namespace door {
namespace host {

namespace {

aether::fft::Norm norm_of(eg_fft_norm norm)
{
    switch (norm) {
    case EG_FFT_NORM_FORWARD:
        return aether::fft::Norm::Forward;
    case EG_FFT_NORM_ORTHO:
        return aether::fft::Norm::Ortho;
    default:
        return aether::fft::Norm::Backward;
    }
}

/** @brief The first float of the plane @p p. */
template<class T>
T* floats_of(const eg_plane_s& p)
{
    const DLTensor& t = p.tensor->dl_tensor;
    return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

/** @brief Multiply the interior bins (0 < k < edge along the halved axis) of
 * the @p count floats of a half spectrum by the plan's weight. */
template<class T>
void weigh(T* data, std::int64_t count, const eg_fft_s& plan)
{
    const T weight = static_cast<T>(plan.weight);
    for (std::int64_t first = 0, bin = 0; first < count;
        first += plan.run, bin       = (bin + 1) % plan.bins)
        if (bin > 0 && bin < plan.edge)
            for (std::int64_t i = first; i < first + plan.run; ++i)
                data[i] *= weight;
}

template<class T>
void exec(const eg_fft_s& plan, const eg_plane_s& in, const eg_plane_s& out,
    int threads)
{
    const std::vector<std::size_t> shape(plan.shape.begin(), plan.shape.end());
    const std::vector<std::size_t> axes(plan.axes.begin(), plan.axes.end());
    const aether::fft::Norm norm = norm_of(plan.norm);
    const T* src                 = floats_of<T>(in);
    T* dst                       = floats_of<T>(out);
    if (plan.kind == EG_FFT_C2C) {
        aether::fft::c2c(src, dst, shape, axes, plan.forward, norm, threads);
    } else if (plan.kind == EG_FFT_R2C) {
        aether::fft::r2c(src, dst, shape, axes, norm, threads);
        if (plan.weight != 1.0)
            weigh(dst, elements(out.tensor->dl_tensor), plan);
    } else if (plan.weight != 1.0) {
        // the input is read-only: the weighted spectrum is a copy
        const std::int64_t count = elements(in.tensor->dl_tensor);
        std::vector<T> weighted(src, src + count);
        weigh(weighted.data(), count, plan);
        aether::fft::c2r(weighted.data(), dst, shape, axes, norm, threads);
    } else {
        aether::fft::c2r(src, dst, shape, axes, norm, threads);
    }
}

} // namespace

void fft_exec(const eg_fft_s& plan, const eg_plane_s& in, const eg_plane_s& out,
    int threads)
{
    if (plan.bits == 32)
        exec<float>(plan, in, out, threads);
    else
        exec<double>(plan, in, out, threads);
}

} // namespace host
} // namespace door
} // namespace eagle
