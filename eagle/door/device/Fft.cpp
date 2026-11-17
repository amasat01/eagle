// SPDX-License-Identifier: Apache-2.0

// The device transform: one batched cuFFT plan per transformed axis, run on
// the context's stream, then one kernel (device/Loop.cu) for the norm and the
// Hermitian weight. cuFFT is not linked: its C ABI is the same across majors,
// so the door opens libcufft.so.12 or .11 at the first plan and declares the
// few calls it makes, and a machine without cuFFT still loads the door.
#include "eagle/door/device/Fft.h"

#include "eagle/door/device/Loop.h"

#include <climits>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

#include <dlfcn.h>

namespace eagle {
namespace door {

/** @brief One batched 1-D cuFFT plan along an axis, run once per outer block
 * when the axis has elements after it. */
struct FftStep {
    int handle             = 0;
    std::int64_t loops     = 1;
    std::int64_t in_bytes  = 0,
                 out_bytes = 0; ///< the offset of each outer block
};

struct DeviceFft {
    std::vector<FftStep> c2c; ///< the complex axes, in order
    FftStep real;             ///< the halved axis of a real transform
    double scale = 1.0;       ///< the norm's factor
    bool empty   = false;     ///< nothing to transform
};

namespace {

// cuFFT's cufftType values and transform directions.
enum : int {
    kR2C     = 0x2a,
    kC2R     = 0x2c,
    kC2C     = 0x29,
    kD2Z     = 0x6a,
    kZ2D     = 0x6c,
    kZ2Z     = 0x69,
    kForward = -1,
    kInverse = 1
};

/** @brief The cuFFT calls the door makes. */
struct Library {
    int (*plan_many)(int*, int, int*, int*, int, int, int*, int, int, int, int)
        = nullptr;
    int (*c2c)(int, void*, void*, int) = nullptr;
    int (*z2z)(int, void*, void*, int) = nullptr;
    int (*r2c)(int, void*, void*)      = nullptr;
    int (*d2z)(int, void*, void*)      = nullptr;
    int (*c2r)(int, void*, void*)      = nullptr;
    int (*z2d)(int, void*, void*)      = nullptr;
    int (*set_stream)(int, void*)      = nullptr;
    int (*destroy)(int)                = nullptr;
};

/** @brief The library, opened once: cuFFT 12 ships with CUDA 13 and cuFFT 11
 * with CUDA 12. */
const Library& library()
{
    static std::once_flag once;
    static Library lib;
    static std::string failure;
    std::call_once(once, [] {
        std::string tried;
        void* handle = nullptr;
        for (const char* name : { "libcufft.so.12", "libcufft.so.11" }) {
            handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (handle != nullptr)
                break;
            const char* why = dlerror();
            tried += std::string("\n  ") + name + ": "
                + (why != nullptr ? why : "not loadable");
        }
        if (handle == nullptr) {
            failure
                = "cuFFT is not installed (libcufft.so.12 and .11 were tried):"
                + tried
                + "\ninstall CUDA's cuFFT (pip: nvidia-cufft-cu12 or "
                  "nvidia-cufft-cu13) or put it on the loader path";
            return;
        }
        const auto load = [handle](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
                dlsym(handle, name));
            return fn != nullptr;
        };
        const bool whole = load(lib.plan_many, "cufftPlanMany")
            && load(lib.c2c, "cufftExecC2C") && load(lib.z2z, "cufftExecZ2Z")
            && load(lib.r2c, "cufftExecR2C") && load(lib.d2z, "cufftExecD2Z")
            && load(lib.c2r, "cufftExecC2R") && load(lib.z2d, "cufftExecZ2D")
            && load(lib.set_stream, "cufftSetStream")
            && load(lib.destroy, "cufftDestroy");
        if (!whole) {
            failure = "the cuFFT library that was opened lacks calls the door "
                      "makes";
            lib     = Library{};
        }
    });
    if (lib.plan_many == nullptr)
        refuse(EG_FAULT, failure);
    return lib;
}

void check(int code, const char* call)
{
    if (code != 0)
        refuse(EG_FAULT,
            std::string(call) + " failed with code " + std::to_string(code));
}

/**
 * @brief The batched plan of one axis: @p n points, @p in_len and @p out_len
 * elements along the axis on the two sides,
 * @p inner elements after it and @p outer before it, element sizes @p in_size
 * and @p out_size in bytes. An axis with nothing after it is one call over all
 * @p outer lines; otherwise one call per outer block with the lines strided.
 */
FftStep make_step(const Library& lib, int type, std::int64_t n,
    std::int64_t in_len, std::int64_t out_len, std::int64_t inner,
    std::int64_t outer, std::int64_t in_size, std::int64_t out_size)
{
    FftStep step;
    int length       = static_cast<int>(n);
    int in_embed     = static_cast<int>(in_len * inner),
        out_embed    = static_cast<int>(out_len * inner);
    const bool lines = inner == 1;
    const int stride = static_cast<int>(inner);
    check(lib.plan_many(&step.handle, 1, &length, &in_embed, stride,
              lines ? static_cast<int>(in_len) : 1, &out_embed, stride,
              lines ? static_cast<int>(out_len) : 1, type,
              static_cast<int>(lines ? outer : inner)),
        "cufftPlanMany");
    if (!lines) {
        step.loops     = outer;
        step.in_bytes  = in_embed * in_size;
        step.out_bytes = out_embed * out_size;
    }
    return step;
}

/** @brief The product of the extents of @p shape in [@p first, @p last). */
std::int64_t extent(
    const std::vector<std::int64_t>& shape, std::size_t first, std::size_t last)
{
    std::int64_t n = 1;
    for (std::size_t a = first; a < last; ++a)
        n *= shape[a];
    return n;
}

/** @brief The factor of the norm of @p plan over its transformed lengths. */
double norm_scale(const eg_fft_s& plan)
{
    double points = 1;
    for (std::int32_t a : plan.axes)
        points *= static_cast<double>(plan.shape[a]);
    const bool forward
        = plan.kind == EG_FFT_R2C || (plan.kind == EG_FFT_C2C && plan.forward);
    if (plan.norm == EG_FFT_NORM_ORTHO)
        return 1.0 / std::sqrt(points);
    return forward == (plan.norm == EG_FFT_NORM_FORWARD) ? 1.0 / points : 1.0;
}

/** @brief Run the complex step @p step from @p in to @p out in direction @p
 * direction. */
void exec_c2c(const Library& lib, const FftStep& step, bool dbl, int direction,
    char* in, char* out, void* stream)
{
    check(lib.set_stream(step.handle, stream), "cufftSetStream");
    for (std::int64_t o = 0; o < step.loops; ++o) {
        void* a = in + o * step.in_bytes;
        void* b = out + o * step.out_bytes;
        check(dbl ? lib.z2z(step.handle, a, b, direction)
                  : lib.c2c(step.handle, a, b, direction),
            "cufftExecC2C");
    }
}

/** @brief Run the real step of @p plan from @p in to @p out. */
void exec_real(const Library& lib, const FftStep& step, const eg_fft_s& plan,
    char* in, char* out, void* stream)
{
    const bool dbl = plan.bits == 64;
    check(lib.set_stream(step.handle, stream), "cufftSetStream");
    for (std::int64_t o = 0; o < step.loops; ++o) {
        void* a = in + o * step.in_bytes;
        void* b = out + o * step.out_bytes;
        if (plan.kind == EG_FFT_R2C)
            check(dbl ? lib.d2z(step.handle, a, b) : lib.r2c(step.handle, a, b),
                "cufftExecR2C");
        else
            check(dbl ? lib.z2d(step.handle, a, b) : lib.c2r(step.handle, a, b),
                "cufftExecC2R");
    }
}

} // namespace

std::size_t fft_work_bytes(const eg_fft_s& plan)
{
    if (plan.kind != EG_FFT_C2R)
        return 0;
    std::vector<std::int64_t> half = plan.shape;
    half[plan.axes.back()]         = plan.bins;
    return static_cast<std::size_t>(extent(half, 0, half.size())) * 2
        * (plan.bits / 8);
}

DeviceFft* fft_build(const eg_fft_s& plan)
{
    const Library& lib       = library();
    auto fft                 = std::make_unique<DeviceFft>();
    fft->scale               = norm_scale(plan);
    const std::int64_t total = extent(plan.shape, 0, plan.shape.size());
    fft->empty               = total == 0;
    if (fft->empty)
        return fft.release();
    if (total > INT_MAX)
        refuse(EG_NOT_WRITTEN,
            "eg_fft_new: " + std::to_string(total)
                + " elements are more than the 2^31 - 1 cuFFT counts in");
    const bool dbl               = plan.bits == 64;
    const std::int64_t real_size = plan.bits / 8, complex_size = 2 * real_size;
    std::vector<std::int64_t> spectrum = plan.shape; // the complex side
    const bool real_kind               = plan.kind != EG_FFT_C2C;
    if (real_kind)
        spectrum[plan.axes.back()] = plan.bins;
    const std::size_t complex_axes
        = real_kind ? plan.axes.size() - 1 : plan.axes.size();
    try {
        for (std::size_t i = 0; i < complex_axes; ++i) {
            const std::size_t a = static_cast<std::size_t>(plan.axes[i]);
            fft->c2c.push_back(
                make_step(lib, dbl ? kZ2Z : kC2C, spectrum[a], spectrum[a],
                    spectrum[a], extent(spectrum, a + 1, spectrum.size()),
                    extent(spectrum, 0, a), complex_size, complex_size));
        }
        if (real_kind) {
            const std::size_t a  = static_cast<std::size_t>(plan.axes.back());
            const std::int64_t n = plan.shape[a], bins = plan.bins,
                               inner = extent(spectrum, a + 1, spectrum.size()),
                               outer = extent(spectrum, 0, a);
            fft->real                = plan.kind == EG_FFT_R2C
                               ? make_step(lib, dbl ? kD2Z : kR2C, n, n, bins, inner, outer,
                                     real_size, complex_size)
                               : make_step(lib, dbl ? kZ2D : kC2R, n, bins, n, inner, outer,
                                     complex_size, real_size);
        }
    } catch (...) {
        fft_drop(fft.get());
        throw;
    }
    return fft.release();
}

void fft_enqueue(const DeviceFft& fft, const eg_fft_s& plan, const void* in,
    void* out, void* work, std::uint64_t stream)
{
    if (fft.empty)
        return;
    const Library& lib  = library();
    const bool dbl      = plan.bits == 64;
    void* queue         = reinterpret_cast<void*>(stream);
    char* src           = const_cast<char*>(static_cast<const char*>(in));
    char* dst           = static_cast<char*>(out);
    std::int64_t floats = 0; // of the plane written
    if (plan.kind == EG_FFT_C2R) {
        const std::size_t bytes = fft_work_bytes(plan);
        char* copy              = static_cast<char*>(work);
        copy_step(stream, copy, src, bytes);
        if (plan.weight != 1.0)
            fft_weigh(stream, plan.bits, copy,
                static_cast<std::int64_t>(bytes / (plan.bits / 8)), plan.run,
                plan.bins, plan.edge, 1.0, plan.weight);
        for (const FftStep& step : fft.c2c)
            exec_c2c(lib, step, dbl, kInverse, copy, copy, queue);
        exec_real(lib, fft.real, plan, copy, dst, queue);
        floats = extent(plan.shape, 0, plan.shape.size());
    } else if (plan.kind == EG_FFT_R2C) {
        exec_real(lib, fft.real, plan, src, dst, queue);
        for (const FftStep& step : fft.c2c)
            exec_c2c(lib, step, dbl, kForward, dst, dst, queue);
        std::vector<std::int64_t> spectrum = plan.shape;
        spectrum[plan.axes.back()]         = plan.bins;
        floats = 2 * extent(spectrum, 0, spectrum.size());
    } else {
        char* from = src;
        for (const FftStep& step : fft.c2c) {
            exec_c2c(lib, step, dbl, plan.forward ? kForward : kInverse, from,
                dst, queue);
            from = dst;
        }
        floats = 2 * extent(plan.shape, 0, plan.shape.size());
    }
    const bool weighs = plan.kind == EG_FFT_R2C && plan.weight != 1.0;
    if (fft.scale != 1.0
        || weighs) { // the other kinds' weight is on their input
        if (plan.kind == EG_FFT_R2C)
            fft_weigh(stream, plan.bits, dst, floats, plan.run, plan.bins,
                plan.edge, fft.scale, plan.weight);
        else
            fft_weigh(
                stream, plan.bits, dst, floats, floats, 1, 0, fft.scale, 1.0);
    }
}

void fft_drop(DeviceFft* fft) noexcept
{
    if (fft == nullptr)
        return;
    const Library& lib = library(); // opened: a plan exists only after it
    for (const FftStep& step : fft->c2c)
        lib.destroy(step.handle);
    if (fft->real.handle != 0)
        lib.destroy(fft->real.handle);
    delete fft;
}

} // namespace door
} // namespace eagle
