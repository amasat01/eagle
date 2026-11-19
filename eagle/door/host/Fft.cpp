// SPDX-License-Identifier: Apache-2.0

// The host transform: aether's batched pocketfft (aether/fft/Fft.h), its
// chunks run on eagle's host pool with the launch's team, then the plan's
// Hermitian weight. The only file of the door that includes Fft.h.
#include "aether/fft/Fft.h"
#include "eagle/door/host/Ops.h"
#include "eagle/exec/HostPool.h"

namespace eagle {
namespace door {
namespace host {

namespace {

/** @brief aether's FFT loop on eagle's host pool: each chunk once, on a team of @p team threads. aether's
 *         chunk body keeps its own first failure, so the body the pool runs does not throw. */
struct PoolLoop {
    int team;

    template<class F>
    void operator()(std::size_t count, F&& chunkBody) const
    {
        eagle::exec::hostFor(static_cast<std::int64_t>(count), team,
            [&](std::int64_t chunk) { chunkBody(static_cast<std::size_t>(chunk)); });
    }
};

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
    PoolLoop loop{ threads };
    if (plan.kind == EG_FFT_C2C) {
        aether::fft::c2c(src, dst, shape, axes, plan.forward, norm, loop);
    } else if (plan.kind == EG_FFT_R2C) {
        aether::fft::r2c(src, dst, shape, axes, norm, loop);
        if (plan.weight != 1.0)
            weigh(dst, elements(out.tensor->dl_tensor), plan);
    } else if (plan.weight != 1.0) {
        // the input is read-only: the weighted spectrum is a copy
        const std::int64_t count = elements(in.tensor->dl_tensor);
        std::vector<T> weighted(src, src + count);
        weigh(weighted.data(), count, plan);
        aether::fft::c2r(weighted.data(), dst, shape, axes, norm, loop);
    } else {
        aether::fft::c2r(src, dst, shape, axes, norm, loop);
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
