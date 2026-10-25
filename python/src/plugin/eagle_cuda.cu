// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file eagle_cuda.cu
 * @brief ``libeagle_cuda.so``: the CUDA backend of the ``eagle-backend/1`` seam.
 *
 * Implements the C seam (``seam/eagle_backend.h``) over the ``eagle::cuda``
 * classes. Built once with nvcc and copied into every per-Python wheel: it has
 * no Python dependency and no CUDA runtime dependency. The CUDA runtime calls
 * eagle's headers make are served by the plugin's own translation onto the
 * driver API (``cuda_driver.cpp``), so ``libcuda.so.1`` is the only CUDA library
 * it loads. Every entry is a
 * ``try``/``catch`` over its whole body; no exception crosses the seam, and the
 * message of a nonzero status is kept in a thread-local buffer
 * (``eagle_backend_last_error``).
 *
 * Devices: every handle remembers the device it was created on, and every call
 * on it runs with that device current (restored afterwards), so a handle keeps
 * working whatever the caller's current device is later.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eagle/compose/GraphComposer.h"
#include "eagle/cuda/CaptureAttribution.h"
#include "eagle/cuda/CaptureConditional.h"
#include "eagle/cuda/CaptureFork.h"
#include "eagle/cuda/DeviceProps.h"
#include "eagle/cuda/Graph.h"
#include "eagle/diff/Reverse.h"
#include "eagle/events/Sensitivity.h"
#include "eagle/exec/DeviceKernel.h"
#include "eagle/exec/Transfer.h"
#include "eagle/filtering/Compact.h"
#include "eagle/filtering/Reorder.h"
#include "eagle/filtering/ReorderClasses.h"
#include "eagle/reduce/canonical.h"
#include "eagle/util/CaptureGuard.h"
#include "plugin/cuda_driver.h"
#include "plugin/interop.h"
#include "seam/eagle_backend.h"

#ifndef EAGLE_PYTHON_VERSION
#define EAGLE_PYTHON_VERSION "unknown"
#endif
#ifndef EAGLE_CUDA_PLUGIN_ARCHS
#define EAGLE_CUDA_PLUGIN_ARCHS "unknown"
#endif

using eagle::CountGuard;
using eagle::compose::GraphComposer;
using eagle::cuda::CaptureConditional;
using eagle::cuda::CondKind;
using eagle::cuda::CaptureFork;
using eagle::cuda::CapturedGraph;
using eagle::cuda::Graph;
using eagle::cuda::Launcher;
using eagle::cuda::Stream;
using eagle::cuda::StreamCapturer;

// The opaque handle types of the seam, defined here and nowhere else.
struct eagle_backend_stream_s {
    int device;
    Stream stream;
};
struct eagle_backend_capturer_s {
    int device;
    StreamCapturer capturer;
};
struct eagle_backend_captured_s {
    int device;
    CapturedGraph graph;
};
struct eagle_backend_fork_s {
    int device;
    CaptureFork fork;
};
struct eagle_backend_conditional_s {
    int device;
    CaptureConditional conditional;
};
struct eagle_backend_graph_s {
    int device;
    Graph graph;
};
struct eagle_backend_launcher_s {
    int device;
    Launcher launcher;
};
struct eagle_backend_composer_s {
    int device;
    std::shared_ptr<GraphComposer> composer;
    std::vector<std::string> names; // member_name storage, refreshed per call
};

namespace {

thread_local std::string g_lastError;

/** A callback of the core failed; unwind to the entry. */
struct CallbackFailed {
};

/** The device runtime cannot be used here (no driver, no device, ...). */
struct DeviceUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/** The backend provides the group but not this option. */
struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/** A null or already-consumed handle. */
struct BadHandle : std::runtime_error {
    using std::runtime_error::runtime_error;
};

bool isUnavailableCode(cudaError_t e)
{
    return e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || e == cudaErrorInitializationError
        || e == cudaErrorSystemDriverMismatch || e == cudaErrorCompatNotSupportedOnDevice;
}

/** Throw DeviceUnavailable unless a CUDA device is usable (cached per process). */
void requireDevice()
{
    static const cudaError_t status = [] {
        int n = 0;
        cudaError_t e = cudaGetDeviceCount(&n);
        if (e == cudaSuccess && n == 0)
            e = cudaErrorNoDevice;
        return e;
    }();
    if (status == cudaSuccess)
        return;
    std::string text = std::string("eagle: the CUDA backend cannot run here: ") + cudaGetErrorName(status)
        + " (" + cudaGetErrorString(status) + ")";
    if (status == cudaErrorNoDevice)
        text = "eagle: no CUDA GPU visible (cudaErrorNoDevice). Check nvidia-smi and CUDA_VISIBLE_DEVICES. "
               "To run on the CPU instead, pass NumPy arrays (not CuPy arrays): the same eagle call then runs on CPU threads.";
    else if (status == cudaErrorInsufficientDriver || status == cudaErrorSystemDriverMismatch)
        text = std::string("eagle: the NVIDIA driver cannot run this CUDA build (") + cudaGetErrorName(status)
            + "). Install the extra that matches your driver: pip install \"raptor-eagle[cuda12]\" for a CUDA 12 "
              "driver (R525 or newer), \"raptor-eagle[cuda13]\" for a CUDA 13 driver (R580 or newer); "
              "run nvidia-smi to see which CUDA version your driver supports.";
    if (isUnavailableCode(status))
        throw DeviceUnavailable(text);
    throw std::runtime_error(text);
}

void checkCuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("eagle: ") + what + ": " + cudaGetErrorString(e));
}

/** The calling thread's current device (the static runtime follows the driver's
 *  per-thread current context, which every runtime in the process shares). */
int currentDevice()
{
    int d = 0;
    checkCuda(cudaGetDevice(&d), "cudaGetDevice");
    return d;
}

/** A driver entry point at its CUDA 12.0 ABI (never a later _vN revision). */
template <class Fn>
Fn driverEntry12(const char* symbol)
{
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult found{};
    const cudaError_t err = cudaGetDriverEntryPointByVersion(symbol, &fn, 12000,
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
        cudaEnablePerThreadDefaultStream,
#else
        cudaEnableLegacyStream,
#endif
        &found);
    if (err != cudaSuccess || found != cudaDriverEntryPointSuccess || fn == nullptr)
        throw DeviceUnavailable(std::string("eagle: the CUDA driver does not provide ") + symbol);
    return reinterpret_cast<Fn>(fn);
}

/** The device @p stream belongs to; the current device for the default streams. */
int streamDevice(std::uint64_t stream)
{
    if (stream == 0 || stream == reinterpret_cast<std::uint64_t>(cudaStreamLegacy)
        || stream == reinterpret_cast<std::uint64_t>(cudaStreamPerThread))
        return currentDevice();
    // Pinned to the 12.0 ABI of each entry: asked for at this toolkit's version,
    // "cuStreamGetCtx" resolves to cuStreamGetCtx_v2 (12.5+, one more argument).
    static const auto getCtx = driverEntry12<CUresult (*)(CUstream, CUcontext*)>("cuStreamGetCtx");
    static const auto push = driverEntry12<CUresult (*)(CUcontext)>("cuCtxPushCurrent");
    static const auto pop = driverEntry12<CUresult (*)(CUcontext*)>("cuCtxPopCurrent");
    static const auto getDev = driverEntry12<CUresult (*)(CUdevice*)>("cuCtxGetDevice");
    CUcontext ctx = nullptr;
    if (getCtx(reinterpret_cast<CUstream>(stream), &ctx) != CUDA_SUCCESS || ctx == nullptr)
        throw std::invalid_argument("eagle: the stream handle passed is not a valid CUDA stream");
    if (push(ctx) != CUDA_SUCCESS)
        throw std::runtime_error("eagle: cuCtxPushCurrent failed");
    CUdevice dev = 0;
    const CUresult r = getDev(&dev);
    CUcontext popped = nullptr;
    pop(&popped);
    if (r != CUDA_SUCCESS)
        throw std::runtime_error("eagle: cuCtxGetDevice failed");
    return static_cast<int>(dev);
}

/** Resolve a desc's ``device`` (-1 = the stream's, or the current one). */
int resolveDevice(int device, std::optional<std::uint64_t> stream)
{
    if (device >= 0)
        return device;
    if (device != -1)
        throw std::invalid_argument("eagle: device must be >= 0 or -1, got " + std::to_string(device));
    return stream ? streamDevice(*stream) : currentDevice();
}

/** Makes @p device current for the scope (restores the previous one). */
using DeviceScope = eagle::interop::detail::DeviceScope;

/** A size-prefixed input desc, with the fields the caller did not send zeroed. */
template <class D>
D readDesc(const D* in, std::uint32_t knownFlags)
{
    if (in == nullptr)
        throw std::invalid_argument("eagle: null desc");
    if (in->struct_size < 3 * sizeof(std::uint32_t))
        throw std::invalid_argument("eagle: desc struct_size " + std::to_string(in->struct_size) + " is too small");
    D d{};
    std::memcpy(&d, in, std::min<std::size_t>(in->struct_size, sizeof(D)));
    if ((d.flags & ~knownFlags) != 0)
        throw Unsupported("eagle: the CUDA backend does not know desc flag bits "
            + std::to_string(d.flags & ~knownFlags));
    return d;
}

template <class H>
H* live(H* h)
{
    if (h == nullptr)
        throw BadHandle("eagle: null or already-consumed handle (a moved-from object?)");
    return h;
}

template <class T>
void out(T* p, T v)
{
    if (p == nullptr)
        throw std::invalid_argument("eagle: null output pointer");
    *p = v;
}

/** Run @p body, mapping whatever it throws to a status + thread-local message. */
template <class Body>
eagle_backend_status guarded(Body&& body) noexcept
{
    try {
        body();
        g_lastError.clear();
        return EAGLE_BACKEND_OK;
    } catch (const CallbackFailed&) {
        g_lastError = "a callback raised";
        return EAGLE_BACKEND_CALLBACK_FAILED;
    } catch (const DeviceUnavailable& e) {
        g_lastError = e.what();
        return EAGLE_BACKEND_UNAVAILABLE;
    } catch (const Unsupported& e) {
        g_lastError = e.what();
        return EAGLE_BACKEND_UNSUPPORTED;
    } catch (const BadHandle& e) {
        g_lastError = e.what();
        return EAGLE_BACKEND_BAD_HANDLE;
    } catch (const eagle::interop::InteropError& e) {
        g_lastError = e.what();
        return EAGLE_BACKEND_INTEROP;
    } catch (const std::out_of_range& e) {
        g_lastError = e.what();
        return EAGLE_BACKEND_INDEX;
    } catch (const std::invalid_argument& e) {
        g_lastError = e.what();
        return EAGLE_BACKEND_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return EAGLE_BACKEND_ERROR;
    } catch (...) {
        g_lastError = "eagle: unknown C++ exception in the CUDA backend";
        return EAGLE_BACKEND_ERROR;
    }
}

/** Destroy @p h with its device current; never throws. */
template <class H>
void release(H* h) noexcept
{
    if (h == nullptr)
        return;
    try {
        DeviceScope scope(h->device);
        delete h;
    } catch (...) {
        delete h;
    }
}

// kept in the binary as one literal (the release check reads it with strings)
[[gnu::used]] const char kPluginArchs[] = EAGLE_CUDA_PLUGIN_ARCHS;
static_assert(sizeof kPluginArchs <= sizeof eagle_backend_build_info::archs,
              "the arch list does not fit eagle_backend_build_info::archs");

void copyText(char* dst, std::size_t n, const char* src)
{
    std::snprintf(dst, n, "%s", src);
}

/** Run @p call between cudaGetLastError clear-before / check-after, so a failure
 *  the release build of eagle::cuda would swallow is reported. */
template <class Call>
auto bracketed(const char* what, const char* verb, Call&& call)
{
    cudaGetLastError(); // clear-before
    auto result = call();
    const cudaError_t e = cudaGetLastError(); // check-after
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": CUDA error after " + verb + " (code "
            + std::to_string(static_cast<int>(e)) + ": " + cudaGetErrorString(e) + ")");
    return result;
}

std::function<void()> preLaunch(eagle_backend_void_fn pre, void* ctx)
{
    if (pre == nullptr)
        return nullptr;
    return [pre, ctx] {
        if (pre(ctx) != 0)
            throw CallbackFailed{};
    };
}

constexpr std::uint64_t kCapabilities = EAGLE_BACKEND_CAP_STREAM | EAGLE_BACKEND_CAP_CAPTURER
    | EAGLE_BACKEND_CAP_CAPTURED | EAGLE_BACKEND_CAP_FORK | EAGLE_BACKEND_CAP_CONDITIONAL
    | EAGLE_BACKEND_CAP_ATTRIBUTION | EAGLE_BACKEND_CAP_GRAPH | EAGLE_BACKEND_CAP_LAUNCHER | EAGLE_BACKEND_CAP_EXEC
    | EAGLE_BACKEND_CAP_DEVICE | EAGLE_BACKEND_CAP_INTEROP | EAGLE_BACKEND_CAP_FILTERING;

/** The DLPack device types this backend serves: CUDA, CUDAHost, CUDAManaged. */
constexpr std::int32_t kDeviceTypes[] = { kDLCUDA, kDLCUDAHost, kDLCUDAManaged };

} // namespace

#define EAGLE_BACKEND_API extern "C" __attribute__((visibility("default")))

// EAGLE_BACKEND_INJECT_MISSING builds a deliberately broken plugin for the
// symbol-gate injection row: eagle_backend_fork_join is exported under a wrong
// name, so the manifest gate must go red and the loader must name the symbol.
#ifdef EAGLE_BACKEND_INJECT_MISSING
#define EAGLE_BACKEND_INJECTED_NAME(name) name##_x
#else
#define EAGLE_BACKEND_INJECTED_NAME(name) name
#endif

// ---- mandatory core ---------------------------------------------------------

EAGLE_BACKEND_API std::uint32_t eagle_backend_version(void)
{
    return EAGLE_BACKEND_VERSION;
}

EAGLE_BACKEND_API const char* eagle_backend_last_error(void)
{
    return g_lastError.c_str();
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_build_info(struct eagle_backend_build_info* info)
{
    return guarded([&] {
        if (info == nullptr)
            throw std::invalid_argument("eagle_backend_build_info: null info");
        struct eagle_backend_build_info full {};
        full.abi_version = EAGLE_BACKEND_VERSION;
        full.cudart_version = CUDART_VERSION;
        copyText(full.eagle_version, sizeof full.eagle_version, EAGLE_PYTHON_VERSION);
        copyText(full.archs, sizeof full.archs, kPluginArchs);
        copyText(full.backend_kind, sizeof full.backend_kind, "cuda");
        const std::uint32_t want = info->struct_size;
        if (want < sizeof(std::uint32_t))
            throw std::invalid_argument("eagle_backend_build_info: struct_size is smaller than the size field");
        const std::uint32_t n = std::min<std::uint32_t>(want, sizeof full);
        full.struct_size = n;
        std::memcpy(info, &full, n);
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_probe(void)
{
    return guarded([] { requireDevice(); });
}

EAGLE_BACKEND_API void eagle_backend_free(void* p)
{
    std::free(p);
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_capabilities(std::uint64_t* groups)
{
    // A group whose driver entry points this driver lacks is not offered.
    return guarded([&] {
        std::uint64_t groups_ = eagle_cuda_plugin::driverGroups(kCapabilities);
        // IF/ELSE and SWITCH add no driver entry (the same cuGraphAddNode): the
        // driver's version is their capability.
        if ((groups_ & EAGLE_BACKEND_CAP_CONDITIONAL) && eagle::cuda::conditionalArmsSupported())
            groups_ |= EAGLE_BACKEND_CAP_CONDITIONAL_IF_ELSE | EAGLE_BACKEND_CAP_CONDITIONAL_SWITCH;
        out(groups, groups_);
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_device_types(std::int32_t* types, std::int32_t cap,
    std::int32_t* count)
{
    return guarded([&] {
        const std::int32_t n = static_cast<std::int32_t>(sizeof kDeviceTypes / sizeof kDeviceTypes[0]);
        for (std::int32_t i = 0; i < n && i < cap && types != nullptr; ++i)
            types[i] = kDeviceTypes[i];
        out(count, n);
    });
}

// ---- stream ------------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_stream_create(const struct eagle_backend_stream_desc* desc,
    eagle_backend_stream* h)
{
    return guarded([&] {
        const auto d = readDesc(desc, EAGLE_BACKEND_STREAM_NON_BLOCKING);
        requireDevice();
        const int device = resolveDevice(d.device, std::nullopt);
        DeviceScope scope(device);
        out(h, new eagle_backend_stream_s{ device, Stream((d.flags & EAGLE_BACKEND_STREAM_NON_BLOCKING) != 0) });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_stream_ptr(eagle_backend_stream h, std::uint64_t* stream)
{
    return guarded([&] { out(stream, reinterpret_cast<std::uint64_t>(live(h)->stream.cuda())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_stream_synchronize(eagle_backend_stream h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->stream.synchronize();
    });
}

EAGLE_BACKEND_API void eagle_backend_stream_release(eagle_backend_stream h)
{
    release(h);
}

// ---- capturer ----------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_capturer_create(
    const struct eagle_backend_capturer_desc* desc, eagle_backend_capturer* h)
{
    return guarded([&] {
        const auto d = readDesc(desc, 0);
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        out(h, new eagle_backend_capturer_s{ device, StreamCapturer(reinterpret_cast<cudaStream_t>(d.stream)) });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_capturer_begin(eagle_backend_capturer h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->capturer.begin();
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_capturer_end(eagle_backend_capturer h,
    eagle_backend_captured* captured)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        if (captured == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        auto box = std::make_unique<eagle_backend_captured_s>(eagle_backend_captured_s{ h->device,
            CapturedGraph(h->capturer.end()) });
        *captured = box.release();
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_capturer_graph(eagle_backend_capturer h, std::uint64_t* graph)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        out(graph, reinterpret_cast<std::uint64_t>(h->capturer.graph()));
    });
}

EAGLE_BACKEND_API void eagle_backend_capturer_release(eagle_backend_capturer h)
{
    release(h);
}

// ---- captured ----------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_captured_is_valid(eagle_backend_captured h, std::int32_t* valid)
{
    return guarded([&] { out(valid, static_cast<std::int32_t>(bool(live(h)->graph))); });
}

EAGLE_BACKEND_API void eagle_backend_captured_release(eagle_backend_captured h)
{
    release(h);
}

// ---- fork --------------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_create(const struct eagle_backend_fork_desc* desc,
    eagle_backend_fork* h)
{
    return guarded([&] {
        const auto d = readDesc(desc, 0);
        requireDevice();
        const int device = resolveDevice(d.device, d.origin);
        DeviceScope scope(device);
        if (h == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        *h = new eagle_backend_fork_s{ device,
            CaptureFork(reinterpret_cast<cudaStream_t>(d.origin), static_cast<std::size_t>(d.branch_count),
                static_cast<std::size_t>(d.wait_count)) };
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_fork(eagle_backend_fork h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->fork.fork();
    });
}

EAGLE_BACKEND_API eagle_backend_status EAGLE_BACKEND_INJECTED_NAME(eagle_backend_fork_join)(eagle_backend_fork h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->fork.join();
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_branch(eagle_backend_fork h, std::uint64_t i,
    std::uint64_t* stream)
{
    return guarded([&] {
        if (i >= live(h)->fork.size())
            throw std::out_of_range("branch index out of range for this CaptureFork");
        out(stream, reinterpret_cast<std::uint64_t>(h->fork.branch(static_cast<std::size_t>(i))));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_origin(eagle_backend_fork h, std::uint64_t* stream)
{
    return guarded([&] { out(stream, reinterpret_cast<std::uint64_t>(live(h)->fork.origin())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_forked(eagle_backend_fork h, std::int32_t* forked)
{
    return guarded([&] { out(forked, static_cast<std::int32_t>(live(h)->fork.forked())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_size(eagle_backend_fork h, std::uint64_t* n)
{
    return guarded([&] { out(n, static_cast<std::uint64_t>(live(h)->fork.size())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_mark(eagle_backend_fork h, std::uint64_t producer,
    std::uint64_t* mark)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        out(mark, static_cast<std::uint64_t>(h->fork.mark(static_cast<std::size_t>(producer))));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fork_wait(eagle_backend_fork h, std::uint64_t waiter,
    std::uint64_t mark)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->fork.wait(static_cast<std::size_t>(waiter), static_cast<std::size_t>(mark));
    });
}

EAGLE_BACKEND_API void eagle_backend_fork_release(eagle_backend_fork h)
{
    release(h);
}

// ---- conditional -------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_conditional_create(
    const struct eagle_backend_conditional_desc* desc, eagle_backend_conditional* h)
{
    return guarded([&] {
        const auto d = readDesc(desc, 0);
        requireDevice();
        const int device = resolveDevice(d.device, d.origin);
        DeviceScope scope(device);
        if (h == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        const cudaStream_t origin = reinterpret_cast<cudaStream_t>(d.origin);
        CountGuard guard{
            reinterpret_cast<const unsigned int*>(d.count_ptr),
            d.baseline_ptr ? reinterpret_cast<const unsigned int*>(d.baseline_ptr) : nullptr,
        };
        CondKind kind = d.counter_ptr ? CondKind::While : CondKind::If;
        switch (d.kind) {
        case 0: break;
        case EAGLE_BACKEND_COND_IF: kind = CondKind::If; break;
        case EAGLE_BACKEND_COND_IF_ELSE: kind = CondKind::IfElse; break;
        case EAGLE_BACKEND_COND_WHILE: kind = CondKind::While; break;
        case EAGLE_BACKEND_COND_SWITCH: kind = CondKind::Switch; break;
        default: throw Unsupported("eagle_backend_conditional_create: unknown kind " + std::to_string(d.kind));
        }
        const eagle::conditional::LoopTerms loop{ d.loop_cap, reinterpret_cast<unsigned int*>(d.counter_ptr),
            reinterpret_cast<const unsigned int*>(d.limit_ptr), reinterpret_cast<unsigned int*>(d.brk_ptr) };
        *h = new eagle_backend_conditional_s{ device,
            CaptureConditional(origin, kind, guard, d.arms, d.negate, loop) };
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_conditional_begin(eagle_backend_conditional h,
    std::uint64_t* body)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        out(body, reinterpret_cast<std::uint64_t>(h->conditional.begin()));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_conditional_body_stream(eagle_backend_conditional h,
    std::uint64_t* body)
{
    return guarded([&] { out(body, reinterpret_cast<std::uint64_t>(live(h)->conditional.bodyStream())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_conditional_is_loop(eagle_backend_conditional h,
    std::int32_t* isLoop)
{
    return guarded([&] { out(isLoop, static_cast<std::int32_t>(live(h)->conditional.isLoop())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_conditional_end(eagle_backend_conditional h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->conditional.end();
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_conditional_arm_stream(eagle_backend_conditional h,
    std::uint64_t k, std::uint64_t* stream)
{
    return guarded([&] {
        if (k >= live(h)->conditional.arms())
            throw std::out_of_range("arm index out of range for this CaptureConditional");
        out(stream, reinterpret_cast<std::uint64_t>(h->conditional.armStream(static_cast<unsigned int>(k))));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_conditional_nodes(eagle_backend_conditional h,
    std::int64_t which, std::uint64_t** nodes, std::int64_t* count)
{
    return guarded([&] {
        if (nodes == nullptr || count == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        DeviceScope scope(live(h)->device);
        if (which < -1 || which >= static_cast<std::int64_t>(h->conditional.arms()))
            throw std::out_of_range("arm index out of range for this CaptureConditional");
        const std::vector<cudaGraphNode_t> found = which < 0
            ? h->conditional.setterNodes()
            : h->conditional.bodyNodes(static_cast<unsigned int>(which));
        auto* buf = static_cast<std::uint64_t*>(std::malloc(std::max<std::size_t>(1, found.size()) * 8));
        if (buf == nullptr)
            throw std::bad_alloc();
        for (std::size_t i = 0; i < found.size(); ++i)
            buf[i] = reinterpret_cast<std::uint64_t>(found[i]);
        *nodes = buf;
        *count = static_cast<std::int64_t>(found.size());
    });
}

EAGLE_BACKEND_API void eagle_backend_conditional_release(eagle_backend_conditional h)
{
    release(h);
}

// ---- attribution ---------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_capture_snapshot_nodes(std::uint64_t stream,
    std::uint64_t** nodes, std::int64_t* count)
{
    return guarded([&] {
        if (nodes == nullptr || count == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        requireDevice();
        DeviceScope scope(resolveDevice(-1, stream));
        const std::vector<cudaGraphNode_t> found
            = eagle::cuda::captureSnapshotNodes(reinterpret_cast<cudaStream_t>(stream));
        auto* buf = static_cast<std::uint64_t*>(std::malloc(std::max<std::size_t>(1, found.size()) * 8));
        if (buf == nullptr)
            throw std::bad_alloc();
        for (std::size_t i = 0; i < found.size(); ++i)
            buf[i] = reinterpret_cast<std::uint64_t>(found[i]);
        *nodes = buf;
        *count = static_cast<std::int64_t>(found.size());
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_is_node_toggleable(std::uint64_t node, std::int32_t* ok)
{
    return guarded([&] {
        out(ok, static_cast<std::int32_t>(eagle::cuda::isNodeToggleable(reinterpret_cast<cudaGraphNode_t>(node))));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_capture_add_child(std::uint64_t stream, std::uint64_t child,
    std::int32_t conditionalBody)
{
    return guarded([&] {
        if (child == 0)
            throw std::invalid_argument("eagle_backend_capture_add_child: null graph");
        requireDevice();
        DeviceScope scope(resolveDevice(-1, stream));
        eagle::cuda::captureAddChild(reinterpret_cast<cudaStream_t>(stream),
            reinterpret_cast<cudaGraph_t>(child), conditionalBody != 0);
    });
}

// ---- graph -------------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_graph_create(const struct eagle_backend_graph_desc* desc,
    eagle_backend_graph* h)
{
    return guarded([&] {
        const auto d = readDesc(desc, 0);
        requireDevice();
        const int device = resolveDevice(d.device, std::nullopt);
        DeviceScope scope(device);
        if (h == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        *h = new eagle_backend_graph_s{ device, Graph() };
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_graph_from_captured(eagle_backend_captured c,
    eagle_backend_graph* h)
{
    return guarded([&] {
        if (h == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        DeviceScope scope(live(c)->device);
        // Defense in depth: a falsy CapturedGraph wraps a null cudaGraph_t — the
        // shape a failed/aborted capture leaves behind. Adopting it would build a
        // Graph that cannot be instantiated, the error surfacing far from its cause.
        if (!c->graph)
            throw std::invalid_argument(
                "Graph.from_captured: the CapturedGraph is empty (null "
                "cudaGraph_t) and cannot be adopted. The capture that "
                "produced it did not yield a graph — typically because "
                "it was never begun, was already consumed by an earlier "
                "from_captured/add_node call, or ended in a failed "
                "state (e.g. a forked branch that was never joined).");
        *h = new eagle_backend_graph_s{ c->device, Graph(std::move(c->graph)) };
        delete c; // consumed on OK
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_graph_stream(eagle_backend_graph h, std::uint64_t stream)
{
    return guarded([&] { live(h)->graph.stream(reinterpret_cast<cudaStream_t>(stream)); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_graph_add_node(eagle_backend_graph h, eagle_backend_captured c)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        live(c);
        h->graph.addNode(std::move(c->graph));
        delete c; // consumed on OK
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_graph_launcher(eagle_backend_graph h,
    eagle_backend_launcher* l)
{
    return guarded([&] {
        if (l == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        DeviceScope scope(live(h)->device);
        // The swallowed-error site an earlier gate found: a rejected
        // cudaGraphInstantiate (e.g. a second instantiate of a conditional-bearing
        // graph) would otherwise yield a silently INERT Launcher.
        auto box = bracketed("Graph.launcher", "instantiate", [&] {
            return std::make_unique<eagle_backend_launcher_s>(eagle_backend_launcher_s{ h->device,
                h->graph.launcher() });
        });
        *l = box.release();
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_graph_last_node(eagle_backend_graph h, std::int64_t* index)
{
    return guarded([&] { out(index, static_cast<std::int64_t>(live(h)->graph.lastNode())); });
}

EAGLE_BACKEND_API void eagle_backend_graph_release(eagle_backend_graph h)
{
    release(h);
}

// ---- launcher ----------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_launch(eagle_backend_launcher h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        // Launcher::launch's release build uses the debug-only EAGLE_CHECK, so a
        // failing cudaGraphLaunch is otherwise swallowed and replay returns
        // silently wrong/inert. Caveat (accepted): this may attribute an unrelated
        // earlier async error to this call; the clear-before bounds it.
        bracketed("Launcher.launch", "launch", [&] {
            h->launcher.launch();
            return 0;
        });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_synchronize(eagle_backend_launcher h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->launcher.synchronize();
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_stream(eagle_backend_launcher h, std::uint64_t stream)
{
    return guarded([&] { live(h)->launcher.stream(reinterpret_cast<cudaStream_t>(stream)); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_set_logical_size(eagle_backend_launcher h,
    std::int64_t logicalSize)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->launcher.setLogicalSize(static_cast<eagle::idx_t>(logicalSize));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_kernel_node_count(eagle_backend_launcher h,
    std::int64_t* count)
{
    return guarded([&] { out(count, static_cast<std::int64_t>(live(h)->launcher.kernelNodeCount())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_set_node_enabled(eagle_backend_launcher h,
    std::uint64_t node, std::int32_t enabled)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        eagle::cuda::setNodeEnabled(h->launcher.execHandle(), reinterpret_cast<cudaGraphNode_t>(node), enabled != 0);
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_set_kernel_params(eagle_backend_launcher h,
    std::uint64_t node, const std::uint64_t* params, std::int64_t n)
{
    return guarded([&] {
        if (n < 0 || (n > 0 && params == nullptr))
            throw std::invalid_argument("eagle_backend_launcher_set_kernel_params: bad params");
        DeviceScope scope(live(h)->device);
        std::vector<void*> args(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < args.size(); ++i)
            args[i] = reinterpret_cast<void*>(params[i]);
        h->launcher.setKernelParams(reinterpret_cast<cudaGraphNode_t>(node), args.data(), args.size());
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_set_memcpy_source(eagle_backend_launcher h,
    std::uint64_t node, std::uint64_t src)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->launcher.setMemcpySource(reinterpret_cast<cudaGraphNode_t>(node), reinterpret_cast<const void*>(src));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_launcher_exec_update(eagle_backend_launcher h,
    eagle_backend_captured captured, std::int32_t* accepted)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        if (!live(captured)->graph)
            throw std::invalid_argument("eagle_backend_launcher_exec_update: the CapturedGraph is empty");
        out(accepted, static_cast<std::int32_t>(h->launcher.tryExecUpdate(captured->graph.graph())));
    });
}

EAGLE_BACKEND_API void eagle_backend_launcher_release(eagle_backend_launcher h)
{
    release(h);
}

// ---- exec --------------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_run_device(const struct eagle_backend_launch_desc* desc,
    std::int32_t* launched)
{
    return guarded([&] {
        const auto d = readDesc(desc, 0);
        requireDevice();
        if (d.nparams < 0 || (d.nparams > 0 && d.params == nullptr))
            throw std::invalid_argument("eagle: run_device needs nparams >= 0 and a params array");
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        std::vector<void*> params;
        params.reserve(static_cast<std::size_t>(d.nparams));
        for (std::int64_t i = 0; i < d.nparams; ++i)
            params.push_back(reinterpret_cast<void*>(d.params[i]));
        const eagle::exec::Partition part{ d.base, d.count, d.n_samples };
        try {
            out(launched,
                static_cast<std::int32_t>(eagle::exec::DeviceKernel::run(reinterpret_cast<CUfunction>(d.function),
                    params, part, reinterpret_cast<CUstream>(d.stream), d.block)));
        } catch (const std::runtime_error&) {
            // Cold path: a stream of another device is the one launch failure left to name.
            int streamOn = device;
            try { streamOn = streamDevice(d.stream); } catch (...) {}
            if (streamOn != device)
                throw std::runtime_error("eagle: the stream belongs to device " + std::to_string(streamOn)
                    + "; this launch runs on device " + std::to_string(device));
            throw;
        }
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_time_launch(const struct eagle_backend_launch_desc* desc,
    std::int32_t reps, double* seconds)
{
    return guarded([&] {
        const auto d = readDesc(desc, 0);
        requireDevice();
        if (reps < 1 || seconds == nullptr)
            throw std::invalid_argument("eagle: x_time_launch needs reps >= 1 and a seconds array");
        if (d.nparams < 0 || (d.nparams > 0 && d.params == nullptr))
            throw std::invalid_argument("eagle: x_time_launch needs nparams >= 0 and a params array");
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        std::vector<void*> params;
        params.reserve(static_cast<std::size_t>(d.nparams));
        for (std::int64_t i = 0; i < d.nparams; ++i)
            params.push_back(reinterpret_cast<void*>(d.params[i]));
        const eagle::exec::Partition part{ d.base, d.count, d.n_samples };
        // a raw 0 is the legacy default stream here, as for run_device's cuLaunchKernel
        const cudaStream_t stream = d.stream == 0 ? cudaStreamLegacy : reinterpret_cast<cudaStream_t>(d.stream);
        std::vector<cudaEvent_t> events(2 * static_cast<std::size_t>(reps));
        for (auto& e : events) EAGLE_CHECK_ALWAYS(cudaEventCreateWithFlags(&e, cudaEventDefault));
        for (std::int32_t r = 0; r < reps; ++r) {
            EAGLE_CHECK_ALWAYS(cudaEventRecord(events[2 * r], stream));
            eagle::exec::DeviceKernel::run(reinterpret_cast<CUfunction>(d.function), params, part,
                reinterpret_cast<CUstream>(d.stream), d.block);
            EAGLE_CHECK_ALWAYS(cudaEventRecord(events[2 * r + 1], stream));
        }
        EAGLE_CHECK_ALWAYS(cudaEventSynchronize(events.back()));
        for (std::int32_t r = 0; r < reps; ++r) {
            float ms = 0.0f;
            EAGLE_CHECK_ALWAYS(cudaEventElapsedTime(&ms, events[2 * r], events[2 * r + 1]));
            seconds[r] = 1e-3 * static_cast<double>(ms);
        }
        for (auto& e : events) EAGLE_CHECK_ALWAYS(cudaEventDestroy(e));
    });
}

// ---- filtering -----------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_compact_device(struct eagle_backend_compact_desc* desc)
{
    return guarded([&] {
        const auto d = readDesc(desc, EAGLE_BACKEND_COMPACT_MASK_IS_DROP);
        if (d.n < 0)
            throw std::invalid_argument("eagle_backend_compact_device: n must be >= 0");
        if (d.n > eagle::filtering::kCompactMaxSamples)
            throw Unsupported("eagle_backend_compact_device: the int32 index map addresses at most 2^31-1 samples, got n = "
                + std::to_string(d.n));
        if (d.scratch == 0) {
            // The query: the size the caller must provide. Written back only
            // when the caller's desc is new enough to carry the field.
            constexpr std::size_t end
                = offsetof(eagle_backend_compact_desc, scratch_bytes) + sizeof(desc->scratch_bytes);
            if (desc->struct_size < end)
                throw std::invalid_argument("eagle_backend_compact_device: the scratch query needs a desc carrying "
                                            "scratch_bytes");
            desc->scratch_bytes = static_cast<std::int64_t>(eagle::filtering::compactScratchBytes(d.n));
            return;
        }
        if (d.count == 0 || (d.n > 0 && (d.mask == 0 || d.index_map == 0)))
            throw std::invalid_argument("eagle_backend_compact_device: mask, index_map and count must be device "
                                        "addresses");
        // The appended trigger fields: absent (an older caller's desc ends
        // before them, and readDesc zeroed them) or live32 == 0 = no trigger.
        eagle::filtering::ReorderTrigger trig{};
        const bool withTrigger = d.live32 != 0;
        if (withTrigger) {
            if (d.span == 0 || d.fire == 0)
                throw std::invalid_argument("eagle_backend_compact_device: a trigger needs live32, span and fire");
            if (d.reserved2 != 0)
                throw std::invalid_argument("eagle_backend_compact_device: reserved2 must be zero");
            trig = { reinterpret_cast<std::uint32_t*>(d.live32), reinterpret_cast<const std::uint32_t*>(d.span),
                reinterpret_cast<std::uint32_t*>(d.fire), d.theta };
        }
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed("eagle_backend_compact_device", "enqueue", [&] {
            eagle::filtering::compactDevice(reinterpret_cast<const std::uint8_t*>(d.mask), d.flags,
                reinterpret_cast<std::int32_t*>(d.index_map), reinterpret_cast<std::uint32_t*>(d.count),
                reinterpret_cast<void*>(d.scratch), d.n,
                // a raw 0 is the LEGACY default stream at this seam (as for
                // run_device's cuLaunchKernel), never the per-thread one the
                // plugin's own <<<>>> would read it as
                d.stream == 0 ? cudaStreamLegacy : reinterpret_cast<cudaStream_t>(d.stream),
                withTrigger ? &trig : nullptr);
            return 0;
        });
    });
}

namespace {

/** The plane table of a reorder desc, validated; the widest element size out. */
std::vector<eagle::filtering::ReorderPlane> reorderPlanes(const eagle_backend_reorder_desc& d, const char* who,
    std::uint32_t& maxElem)
{
    if (d.n < 0)
        throw std::invalid_argument(std::string(who) + ": n must be >= 0");
    if (d.n > eagle::filtering::kCompactMaxSamples)
        throw Unsupported(std::string(who) + ": the int32 indices address at most 2^31-1 samples, got n = "
            + std::to_string(d.n));
    if (d.n_planes > 0 && d.planes == 0)
        throw std::invalid_argument(std::string(who) + ": n_planes > 0 needs a plane table");
    const auto* table = reinterpret_cast<const eagle_backend_reorder_plane*>(d.planes);
    std::vector<eagle::filtering::ReorderPlane> planes;
    planes.reserve(d.n_planes);
    maxElem = 4;
    for (std::uint32_t p = 0; p < d.n_planes; ++p) {
        const std::uint32_t b = table[p].elem_bytes;
        if (b != 1 && b != 2 && b != 4 && b != 8 && b != 16)
            throw std::invalid_argument(std::string(who) + ": plane " + std::to_string(p) + " has "
                + std::to_string(b) + "-byte elements; supported are 1, 2, 4, 8 and 16");
        maxElem = std::max(maxElem, b);
        planes.push_back({ reinterpret_cast<void*>(table[p].data), b });
    }
    return planes;
}

/** Answer the scratch query when scratch == 0; true when it did. */
bool reorderQuery(eagle_backend_reorder_desc* desc, const eagle_backend_reorder_desc& d, std::uint32_t maxElem,
    const char* who)
{
    if (d.scratch != 0)
        return false;
    constexpr std::size_t end = offsetof(eagle_backend_reorder_desc, scratch_bytes) + sizeof(desc->scratch_bytes);
    if (desc->struct_size < end)
        throw std::invalid_argument(std::string(who) + ": the scratch query needs a desc carrying scratch_bytes");
    desc->scratch_bytes = static_cast<std::int64_t>(eagle::filtering::reorderScratchBytes(d.n, maxElem));
    return true;
}

cudaStream_t seamStream(std::uint64_t stream)
{
    return stream == 0 ? cudaStreamLegacy : reinterpret_cast<cudaStream_t>(stream);
}

} // namespace

EAGLE_BACKEND_API eagle_backend_status eagle_backend_reorder_device(struct eagle_backend_reorder_desc* desc)
{
    return guarded([&] {
        const auto d = readDesc(desc, EAGLE_BACKEND_COMPACT_MASK_IS_DROP);
        std::uint32_t maxElem = 0;
        const auto planes     = reorderPlanes(d, "eagle_backend_reorder_device", maxElem);
        if (reorderQuery(desc, d, maxElem, "eagle_backend_reorder_device"))
            return;
        if (d.n > 0
            && (d.mask == 0 || d.perm == 0 || d.inv == 0 || d.index_map == 0 || d.count == 0 || d.span == 0))
            throw std::invalid_argument("eagle_backend_reorder_device: mask, perm, inv, index_map, count and span "
                                        "must be device addresses");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed("eagle_backend_reorder_device", "enqueue", [&] {
            eagle::filtering::reorderDevice(planes.data(), static_cast<std::uint32_t>(planes.size()),
                reinterpret_cast<const std::uint8_t*>(d.mask), d.flags, reinterpret_cast<std::int32_t*>(d.perm),
                reinterpret_cast<std::int32_t*>(d.inv), reinterpret_cast<std::int32_t*>(d.index_map),
                reinterpret_cast<std::uint32_t*>(d.count), reinterpret_cast<std::uint32_t*>(d.span),
                reinterpret_cast<std::uint32_t*>(d.fire), reinterpret_cast<void*>(d.scratch), d.n,
                seamStream(d.stream));
            return 0;
        });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_reset_device(struct eagle_backend_reorder_desc* desc)
{
    return guarded([&] {
        constexpr const char* who = "eagle_backend_reset_device";
        const auto d = readDesc(desc, EAGLE_BACKEND_COMPACT_MASK_IS_DROP);
        if (d.n < 0)
            throw std::invalid_argument(std::string(who) + ": n must be >= 0");
        if (d.n > eagle::filtering::kCompactMaxSamples)
            throw Unsupported(std::string(who) + ": the int32 indices address at most 2^31-1 samples, got n = "
                + std::to_string(d.n));
        if (d.count == 0 || d.span == 0 || (d.n > 0 && (d.perm == 0 || d.inv == 0)))
            throw std::invalid_argument(std::string(who) + ": perm, inv, count and span must be device addresses");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed(who, "enqueue", [&] {
            eagle::filtering::resetDevice(reinterpret_cast<std::int32_t*>(d.perm),
                reinterpret_cast<std::int32_t*>(d.inv), reinterpret_cast<std::int32_t*>(d.index_map),
                reinterpret_cast<std::uint32_t*>(d.count), reinterpret_cast<std::uint32_t*>(d.span), d.n,
                seamStream(d.stream));
            return 0;
        });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_pointer_residency(
    std::int32_t device, std::uint64_t ptr, std::int32_t* cls)
{
    return guarded([&] {
        if (cls == nullptr)
            throw std::invalid_argument("eagle_backend_pointer_residency: null cls");
        requireDevice();
        const int dev = resolveDevice(device, std::nullopt);
        DeviceScope scope(dev);
        // The plugin links the driver only (no cudart): the classes of
        // eagle::filtering::classifyPointer, read through cuPointerGetAttribute.
        // A pageable pointer is an INVALID_VALUE here, which is the class, not a fault.
        using eagle::filtering::Residency;
        static const auto attr = driverEntry12<CUresult (*)(void*, CUpointer_attribute, CUdeviceptr)>(
            "cuPointerGetAttribute");
        const auto p = static_cast<CUdeviceptr>(ptr);
        Residency r = Residency::Pageable;
        unsigned memType = 0;
        int managed = 0;
        if (attr(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, p) != CUDA_SUCCESS) {
            r = Residency::Pageable;
        } else if (attr(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, p) == CUDA_SUCCESS && managed != 0) {
            r = Residency::Managed;
        } else if (memType == CU_MEMORYTYPE_DEVICE) {
            int ordinal = -1;
            r = attr(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p) == CUDA_SUCCESS && ordinal != dev
                ? Residency::OtherDevice : Residency::Device;
        } else if (memType == CU_MEMORYTYPE_HOST) {
            CUdeviceptr mapped = 0;
            r = attr(&mapped, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, p) == CUDA_SUCCESS && mapped != 0
                ? Residency::MappedHost : Residency::Pageable;
        }
        *cls = static_cast<std::int32_t>(r);
    });
}

// ---- filtering: K classes ----------------------------------------------------

namespace {

// the class descs repeat the base descs verbatim and in order (the common prefix parses identically)
static_assert(offsetof(eagle_backend_reorder_classes_desc, scratch_bytes) == offsetof(eagle_backend_reorder_desc, scratch_bytes)
        && offsetof(eagle_backend_reorder_classes_desc, class_plane) == sizeof(eagle_backend_reorder_desc),
    "eagle_backend_reorder_classes_desc must start with eagle_backend_reorder_desc");
static_assert(offsetof(eagle_backend_compact_classes_desc, reserved2) == offsetof(eagle_backend_compact_desc, reserved2)
        && offsetof(eagle_backend_compact_classes_desc, class_plane) == sizeof(eagle_backend_compact_desc),
    "eagle_backend_compact_classes_desc must start with eagle_backend_compact_desc");

/** A class desc must carry every class field (a shorter one is UNSUPPORTED, never zero-defaulted). */
template <class D>
void requireClassFields(const D* in, std::size_t end, const char* who)
{
    if (in != nullptr && in->struct_size < end)
        throw Unsupported(std::string(who) + ": struct_size " + std::to_string(in->struct_size)
            + " ends before the class fields (" + std::to_string(end) + " bytes needed)");
}

} // namespace

EAGLE_BACKEND_API eagle_backend_status eagle_backend_compact_classes_device(
    struct eagle_backend_compact_classes_desc* desc)
{
    return guarded([&] {
        constexpr const char* who = "eagle_backend_compact_classes_device";
        requireClassFields(desc, offsetof(eagle_backend_compact_classes_desc, reserved3) + sizeof(desc->reserved3), who);
        const auto d = readDesc(desc, EAGLE_BACKEND_COMPACT_MASK_IS_DROP);
        if (d.n < 0)
            throw std::invalid_argument(std::string(who) + ": n must be >= 0");
        if (d.n > eagle::filtering::kCompactMaxSamples)
            throw Unsupported(std::string(who) + ": the int32 index map addresses at most 2^31-1 samples, got n = "
                + std::to_string(d.n));
        if (d.scratch == 0) {
            desc->scratch_bytes = static_cast<std::int64_t>(eagle::filtering::compactScratchBytes(d.n));
            return;
        }
        if (d.count == 0 || (d.n > 0 && (d.mask == 0 || d.index_map == 0)))
            throw std::invalid_argument(std::string(who) + ": mask, index_map and count must be device addresses");
        if (d.live32 == 0 || d.span == 0 || d.fire == 0 || d.mixed32 == 0 || d.class_fire == 0
            || (d.n > 0 && d.class_plane == 0))
            throw std::invalid_argument(std::string(who) + ": live32, span, fire, class_plane, mixed32 and class_fire "
                                                           "are required");
        if (d.reserved2 != 0 || d.reserved3 != 0)
            throw std::invalid_argument(std::string(who) + ": reserved fields must be zero");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed(who, "enqueue", [&] {
            eagle::filtering::ReorderTrigger trig{ reinterpret_cast<std::uint32_t*>(d.live32),
                reinterpret_cast<const std::uint32_t*>(d.span), reinterpret_cast<std::uint32_t*>(d.fire), d.theta };
            const auto stream = seamStream(d.stream);
            eagle::filtering::compactDevice(reinterpret_cast<const std::uint8_t*>(d.mask), d.flags,
                reinterpret_cast<std::int32_t*>(d.index_map), reinterpret_cast<std::uint32_t*>(d.count),
                reinterpret_cast<void*>(d.scratch), d.n, stream, &trig);
            eagle::filtering::ClassTrigger ct{ reinterpret_cast<const std::uint8_t*>(d.class_plane), d.classes,
                d.class_theta, reinterpret_cast<std::uint32_t*>(d.mixed32),
                reinterpret_cast<const std::uint32_t*>(d.stale_count), reinterpret_cast<std::uint32_t*>(d.class_fire) };
            eagle::filtering::classTriggerDevice(reinterpret_cast<const std::uint8_t*>(d.mask), d.flags, trig,
                reinterpret_cast<const std::uint32_t*>(d.count), ct, d.n, stream);
            return 0;
        });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_reorder_classes_device(
    struct eagle_backend_reorder_classes_desc* desc)
{
    return guarded([&] {
        constexpr const char* who = "eagle_backend_reorder_classes_device";
        requireClassFields(desc, offsetof(eagle_backend_reorder_classes_desc, class_sorts) + sizeof(desc->class_sorts),
            who);
        const auto d = readDesc(desc,
            EAGLE_BACKEND_COMPACT_MASK_IS_DROP | EAGLE_BACKEND_CLASSES_WITHIN_LIVE);
        if (d.classes < 2 || d.classes > EAGLE_BACKEND_MAX_CLASSES)
            throw std::invalid_argument(std::string(who) + ": classes must lie in [2, 255]");
        if (d.reserved3 != 0)
            throw std::invalid_argument(std::string(who) + ": reserved3 must be zero");
        // the base validation shares the reorder_device descriptor shape
        eagle_backend_reorder_desc base{};
        std::memcpy(&base, &d, sizeof base);
        std::uint32_t maxElem = 0;
        const auto planes     = reorderPlanes(base, who, maxElem);
        if (d.scratch == 0) {
            desc->scratch_bytes = static_cast<std::int64_t>(
                eagle::filtering::classSortScratchBytes(d.n, maxElem, d.classes));
            return;
        }
        if (d.n > 0
            && (d.mask == 0 || d.perm == 0 || d.inv == 0 || d.index_map == 0 || d.count == 0 || d.span == 0
                || d.class_plane == 0 || d.offsets == 0 || d.class_sorts == 0))
            throw std::invalid_argument(std::string(who) + ": mask, perm, inv, index_map, count, span, class_plane, "
                                                           "offsets and class_sorts must be device addresses");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed(who, "enqueue", [&] {
            eagle::filtering::reorderClassesDevice(planes.data(), static_cast<std::uint32_t>(planes.size()),
                reinterpret_cast<const std::uint8_t*>(d.mask), d.flags, d.classes,
                reinterpret_cast<const std::uint8_t*>(d.class_plane), reinterpret_cast<std::int32_t*>(d.perm),
                reinterpret_cast<std::int32_t*>(d.inv), reinterpret_cast<std::int32_t*>(d.index_map),
                reinterpret_cast<std::uint32_t*>(d.count), reinterpret_cast<std::uint32_t*>(d.span),
                reinterpret_cast<std::uint32_t*>(d.fire), reinterpret_cast<std::uint32_t*>(d.offsets),
                reinterpret_cast<std::uint32_t*>(d.class_sorts), reinterpret_cast<void*>(d.scratch), d.n,
                seamStream(d.stream));
            return 0;
        });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_restore_device(struct eagle_backend_reorder_desc* desc)
{
    return guarded([&] {
        const auto d = readDesc(desc, EAGLE_BACKEND_COMPACT_MASK_IS_DROP);
        std::uint32_t maxElem = 0;
        const auto planes     = reorderPlanes(d, "eagle_backend_x_restore_device", maxElem);
        if (reorderQuery(desc, d, maxElem, "eagle_backend_x_restore_device"))
            return;
        if (d.n > 0 && (d.perm == 0 || d.inv == 0))
            throw std::invalid_argument("eagle_backend_x_restore_device: perm and inv must be device addresses");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed("eagle_backend_x_restore_device", "enqueue", [&] {
            eagle::filtering::restoreDevice(planes.data(), static_cast<std::uint32_t>(planes.size()),
                reinterpret_cast<std::int32_t*>(d.perm), reinterpret_cast<std::int32_t*>(d.inv),
                reinterpret_cast<void*>(d.scratch), d.n, seamStream(d.stream));
            return 0;
        });
    });
}

namespace {

/** The fields a newer caller appends to the reduce desc; defaults for an older one. */
struct ReduceExt {
    std::uint32_t policy = EAGLE_BACKEND_REDUCE_ORDERED;
    std::uint32_t flags = 0;
    std::int64_t columns = 1;
    std::uint64_t comp_in = 0;
};

/** One canonical reduction of element type @p T under @p OP into @p out
 *  (capturable). */
template <class T, class OP>
void reduceCanonical(const eagle_backend_reduce_desc& d, const ReduceExt& e, cudaStream_t stream, bool fixed)
{
    namespace canon = eagle::reduce::canonical;
    const canon::DevicePtr<T> in{ reinterpret_cast<const T*>(d.in) };
    const auto policy = static_cast<eagle::reduce::Policy>(e.policy);
    const auto rows = static_cast<eagle::idx_t>(d.n), cols = static_cast<eagle::idx_t>(e.columns);
    const bool pair = (e.flags & EAGLE_BACKEND_REDUCE_PAIR_OUT) != 0;
    auto* wa = reinterpret_cast<T*>(d.work_a);
    auto* wb = reinterpret_cast<T*>(d.work_b);
    auto* out = reinterpret_cast<T*>(d.out);
    if (e.comp_in != 0)
        canon::enqueueTo<T, OP>(in, canon::DevicePtr<T>{ reinterpret_cast<const T*>(e.comp_in) }, rows,
            cols, policy, wa, wb, out, pair, stream, fixed);
    else
        canon::enqueueTo<T, OP>(in, canon::NoComp{}, rows, cols, policy, wa, wb, out, pair, stream, fixed);
}

} // namespace

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_reduce_canonical(struct eagle_backend_reduce_desc* desc)
{
    return guarded([&] {
        constexpr const char* who = "eagle_backend_x_reduce_canonical";
        constexpr std::size_t base = offsetof(eagle_backend_reduce_desc, work_elems) + sizeof(desc->work_elems);
        if (desc == nullptr || desc->struct_size < base)
            throw std::invalid_argument(std::string(who) + ": null desc or struct_size too small");
        const auto& d = *desc;
        if (d.n < 1)
            throw std::invalid_argument(std::string(who) + ": n must be >= 1");
        const bool all = d.op == EAGLE_BACKEND_REDUCE_OP_ALL;
        if (d.op > EAGLE_BACKEND_REDUCE_OP_MIN || d.dtype > EAGLE_BACKEND_REDUCE_U8)
            throw std::invalid_argument(std::string(who) + ": unknown dtype or op");
        if (all != (d.dtype == EAGLE_BACKEND_REDUCE_U8))
            throw std::invalid_argument(std::string(who) + ": the U8 element type goes with OP_ALL, and only with it");
        ReduceExt e;
        if (d.struct_size >= offsetof(eagle_backend_reduce_desc, comp_in) + sizeof(d.comp_in))
            e = { d.policy, d.flags, d.columns, d.comp_in };
        if (e.columns < 1 || e.policy > EAGLE_BACKEND_REDUCE_AUTO || e.flags > EAGLE_BACKEND_REDUCE_PAIR_OUT)
            throw std::invalid_argument(std::string(who) + ": columns must be >= 1; unknown policy or flags");
        const bool fsum = d.op == EAGLE_BACKEND_REDUCE_OP_SUM
            && (d.dtype == EAGLE_BACKEND_REDUCE_F32 || d.dtype == EAGLE_BACKEND_REDUCE_F64);
        if ((e.comp_in != 0 || e.flags != 0) && !fsum)
            throw std::invalid_argument(std::string(who) + ": comp_in and PAIR_OUT take a float32/float64 sum");
        if (d.work_a == 0) {
            desc->work_elems = static_cast<std::int64_t>(eagle::reduce::canonical::deviceWorkElems(
                static_cast<eagle::idx_t>(d.n), static_cast<eagle::idx_t>(e.columns)));
            return;
        }
        if (d.in == 0 || d.out == 0 || d.work_b == 0)
            throw std::invalid_argument(std::string(who) + ": in, out, work_a and work_b must be device addresses");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed(who, "enqueue", [&] {
            const auto stream = seamStream(d.stream);
            // double sums on a slow-FP64 device take the fixed-point leaf
            const bool fixed = d.dtype == EAGLE_BACKEND_REDUCE_F64 && d.op == EAGLE_BACKEND_REDUCE_OP_SUM
                && eagle::reduce::canonical::fixedLeafDevice(device);
            using namespace aether;
            const auto go = [&](auto tag) {
                using T = decltype(tag);
                switch (d.op) {
                case EAGLE_BACKEND_REDUCE_OP_SUM: reduceCanonical<T, SumOp<T>>(d, e, stream, fixed); break;
                case EAGLE_BACKEND_REDUCE_OP_MAX: reduceCanonical<T, MaxOp<T>>(d, e, stream, false); break;
                default: reduceCanonical<T, eagle::reduce::canonical::MinOp<T>>(d, e, stream, false);
                }
            };
            switch (d.dtype) {
            case EAGLE_BACKEND_REDUCE_F32: go(float{}); break;
            case EAGLE_BACKEND_REDUCE_F64: go(double{}); break;
            case EAGLE_BACKEND_REDUCE_I32: go(std::int32_t{}); break;
            case EAGLE_BACKEND_REDUCE_I64: go(std::int64_t{}); break;
            default:
                reduceCanonical<std::uint8_t, LogicalAndOp>(d, e, stream, false);
            }
            return 0;
        });
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_diff_columns(struct eagle_backend_diff_desc* desc)
{
    return guarded([&] {
        constexpr const char* who = "eagle_backend_x_diff_columns";
        if (desc == nullptr || desc->struct_size < sizeof(eagle_backend_diff_desc))
            throw std::invalid_argument(std::string(who) + ": null desc or struct_size too small");
        const auto& d = *desc;
        if (d.count < 0 || d.rows < 0 || d.op > EAGLE_BACKEND_DIFF_READ)
            throw std::invalid_argument(std::string(who) + ": unknown op, or a negative count/rows");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed(who, "enqueue", [&] {
            namespace dd = eagle::diff::device;
            const auto stream = seamStream(d.stream);
            void* dst = reinterpret_cast<void*>(d.dst);
            const void* src = reinterpret_cast<const void*>(d.src);
            const auto* index = reinterpret_cast<const std::int64_t*>(d.index);
            const std::uint32_t b = d.elem_bytes;
            const std::size_t bytes = static_cast<std::size_t>(d.count) * b;
            const auto check = [&](cudaError_t e, const char* what) {
                if (e != cudaSuccess)
                    throw std::runtime_error(std::string(who) + ": " + what + " failed: " + cudaGetErrorString(e));
            };
            if (d.op == EAGLE_BACKEND_DIFF_COPY) {
                if (bytes) check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream), "copy");
            } else if (d.op == EAGLE_BACKEND_DIFF_READ) {
                if (bytes) check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), "read");
                check(cudaStreamSynchronize(stream), "stream wait");
            } else if (d.op == EAGLE_BACKEND_DIFF_GATHER) {
                if (b == 1) dd::gatherColumns<std::uint8_t>(dst, d.dst_ld, src, d.src_ld, index, d.count, d.rows, stream);
                else if (b == 2) dd::gatherColumns<std::uint16_t>(dst, d.dst_ld, src, d.src_ld, index, d.count, d.rows, stream);
                else if (b == 4) dd::gatherColumns<std::uint32_t>(dst, d.dst_ld, src, d.src_ld, index, d.count, d.rows, stream);
                else if (b == 8) dd::gatherColumns<std::uint64_t>(dst, d.dst_ld, src, d.src_ld, index, d.count, d.rows, stream);
                else throw std::invalid_argument(std::string(who) + ": gather elem_bytes must be 1, 2, 4 or 8");
            } else if (d.op == EAGLE_BACKEND_DIFF_ACCUMULATE) {
                if (b == 8) dd::accumulatePrefix(static_cast<double*>(dst), static_cast<const double*>(src), d.dst_ld, d.count, d.rows, stream);
                else if (b == 4) dd::accumulatePrefix(static_cast<float*>(dst), static_cast<const float*>(src), d.dst_ld, d.count, d.rows, stream);
                else throw std::invalid_argument(std::string(who) + ": accumulate elem_bytes must be 4 or 8");
            } else {
                if (b == 8) dd::fillPrefix(static_cast<std::int64_t*>(dst), static_cast<std::int64_t>(d.value), d.count, stream);
                else if (b == 4) dd::fillPrefix(static_cast<std::int32_t*>(dst), static_cast<std::int32_t>(d.value), d.count, stream);
                else throw std::invalid_argument(std::string(who) + ": fill elem_bytes must be 4 or 8");
            }
            return 0;
        });
    });
}

namespace {

template <class T>
eagle::events::EventPlanes<T> eventPlanes(const eagle_backend_event_desc& d)
{
    eagle::events::EventPlanes<T> p;
    p.n = d.n;
    p.guards = d.guards;
    p.components = d.components;
    for (int j = 0; j < d.guards; ++j) {
        p.g[j] = reinterpret_cast<const T*>(d.g[j]);
        p.dg[j] = reinterpret_cast<const T*>(d.dg[j]);
        p.gbar[j] = reinterpret_cast<T*>(d.gbar[j]);
    }
    for (int r = 0; r < d.components; ++r) {
        p.a[r] = reinterpret_cast<T*>(d.a[r]);
        p.b[r] = reinterpret_cast<const T*>(d.b[r]);
        p.c[r] = reinterpret_cast<const T*>(d.c[r]);
    }
    p.terminated = reinterpret_cast<const std::uint8_t*>(d.terminated);
    p.applied = reinterpret_cast<const std::int64_t*>(d.applied);
    p.t0 = reinterpret_cast<const T*>(d.t0);
    p.h = reinterpret_cast<const T*>(d.h);
    p.status = reinterpret_cast<std::int8_t*>(d.status);
    p.which = reinterpret_cast<std::int8_t*>(d.which);
    p.tau = reinterpret_cast<T*>(d.tau);
    p.lo = reinterpret_cast<T*>(d.lo);
    p.hi = reinterpret_cast<T*>(d.hi);
    p.tStop = reinterpret_cast<T*>(d.t_stop);
    return p;
}

}  // namespace

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_event_columns(struct eagle_backend_event_desc* desc)
{
    return guarded([&] {
        constexpr const char* who = "eagle_backend_x_event_columns";
        if (desc == nullptr || desc->struct_size < sizeof(eagle_backend_event_desc))
            throw std::invalid_argument(std::string(who) + ": null desc or struct_size too small");
        const auto& d = *desc;
        if (d.n < 0 || d.op > eagle::events::kOpSelect || d.guards < 0
            || d.guards > EAGLE_BACKEND_EVENT_GUARDS || d.components < 0
            || d.components > EAGLE_BACKEND_EVENT_COMPONENTS)
            throw std::invalid_argument(std::string(who) + ": unknown op, or n/guards/components out of range");
        requireDevice();
        const int device = resolveDevice(d.device, d.stream);
        DeviceScope scope(device);
        bracketed(who, "enqueue", [&] {
            const auto stream = seamStream(d.stream);
            if (d.elem_bytes == 8)
                eagle::events::device::apply(eventPlanes<double>(d), d.op, stream);
            else if (d.elem_bytes == 4)
                eagle::events::device::apply(eventPlanes<float>(d), d.op, stream);
            else
                throw std::invalid_argument(std::string(who) + ": elem_bytes must be 4 or 8");
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(who) + ": launch failed: " + cudaGetErrorString(e));
            return 0;
        });
    });
}

// ---- device ------------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_device_props(std::int32_t device,
    struct eagle_backend_device_props* props)
{
    return guarded([&] {
        if (props == nullptr || props->struct_size < sizeof(std::uint32_t))
            throw std::invalid_argument("eagle_backend_device_props: null props or struct_size too small");
        requireDevice();
        const int dev = resolveDevice(device, std::nullopt);
        const eagle::DeviceProps p = eagle::cuda::deviceProps(dev);
        struct eagle_backend_device_props full {};
        full.device = dev;
        copyText(full.name, sizeof full.name, p.name.c_str());
        full.cc_major = p.cc_major;
        full.cc_minor = p.cc_minor;
        full.sm_count = p.sm_count;
        full.clock_rate_khz = p.clock_rate_khz;
        full.memory_clock_rate_khz = p.memory_clock_rate_khz;
        full.memory_bus_width_bits = p.memory_bus_width_bits;
        full.regs_per_block = p.regs_per_block;
        full.regs_per_sm = p.regs_per_sm;
        full.warp_size = p.warp_size;
        full.shared_mem_per_block = static_cast<std::int64_t>(p.shared_mem_per_block);
        full.shared_mem_per_sm = static_cast<std::int64_t>(p.shared_mem_per_sm);
        full.peak_bytes_per_s = p.peak_bytes_per_s;
        full.peak_flops_sp = p.peak_flops_sp;
        full.peak_flops_dp = p.peak_flops_dp;
        full.fp64_ratio = p.fp64_ratio;
        full.ridge_flops_per_byte_sp = p.ridgeFlopsPerByte("float32");
        full.ridge_flops_per_byte_dp = p.ridgeFlopsPerByte("float64");
        const std::uint32_t n = std::min<std::uint32_t>(props->struct_size, sizeof full);
        full.struct_size = n;
        std::memcpy(props, &full, n);
    });
}

// ---- interop -----------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_fence(std::int32_t dlDeviceType, std::int32_t deviceId,
    std::int64_t producer, std::int64_t consumer)
{
    return guarded([&] {
        if (std::find(std::begin(kDeviceTypes), std::end(kDeviceTypes), dlDeviceType) == std::end(kDeviceTypes))
            throw Unsupported("eagle: the CUDA backend does not serve DLPack device type "
                + std::to_string(dlDeviceType));
        requireDevice();
        auto code = [](std::int64_t c) {
            return c == EAGLE_BACKEND_STREAM_NONE ? std::optional<std::intptr_t>()
                                                  : std::optional<std::intptr_t>(static_cast<std::intptr_t>(c));
        };
        eagle::interop::fence(code(producer), code(consumer), deviceId);
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_event_pool_created(std::int64_t* count)
{
    return guarded([&] { out(count, static_cast<std::int64_t>(eagle::interop::EventPool::instance().created())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_memset_device(std::uint64_t ptr, std::uint64_t bytes,
    std::uint64_t stream)
{
    return guarded([&] {
        if (bytes == 0)
            return;
        if (ptr == 0)
            throw std::invalid_argument("eagle_backend_memset_device: null address");
        requireDevice();
        const int device = resolveDevice(-1, stream);
        DeviceScope scope(device);
        EAGLE_CHECK_ALWAYS(cudaMemsetAsync(reinterpret_cast<void*>(ptr), 0, static_cast<std::size_t>(bytes),
            seamStream(stream)));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_memcpy_device(std::uint64_t dst, std::uint64_t src,
    std::uint64_t bytes, std::uint64_t stream)
{
    return guarded([&] {
        if (bytes == 0)
            return;
        if (dst == 0 || src == 0)
            throw std::invalid_argument("eagle_backend_memcpy_device: null address");
        requireDevice();
        const int device = resolveDevice(-1, stream);
        DeviceScope scope(device);
        EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
            static_cast<std::size_t>(bytes), cudaMemcpyDeviceToDevice, seamStream(stream)));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_memcpy(std::uint64_t dst, std::uint64_t src, std::uint64_t bytes,
    std::int32_t kind, std::uint64_t stream)
{
    return guarded([&] {
        if (!eagle::exec::validCopyKind(kind))
            throw std::invalid_argument("eagle_backend_memcpy: unknown copy kind " + std::to_string(kind));
        if (bytes == 0)
            return;
        if (dst == 0 || src == 0)
            throw std::invalid_argument("eagle_backend_memcpy: null address");
        requireDevice();
        const int device = resolveDevice(-1, stream);
        DeviceScope scope(device);
        eagle::exec::memcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
            static_cast<std::size_t>(bytes), static_cast<eagle::exec::CopyKind>(kind), seamStream(stream));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_host_alloc(std::uint64_t bytes, std::uint64_t* addr)
{
    return guarded([&] {
        if (addr == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        *addr = 0;
        if (bytes == 0)
            return;
        requireDevice();
        *addr = reinterpret_cast<std::uint64_t>(eagle::exec::hostAlloc(static_cast<std::size_t>(bytes)));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_host_free(std::uint64_t ptr)
{
    return guarded([&] {
        if (ptr != 0)
            eagle::exec::hostFree(reinterpret_cast<void*>(ptr));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_host_func(std::uint64_t stream, void (*fn)(void*), void* arg)
{
    return guarded([&] {
        if (fn == nullptr)
            throw std::invalid_argument("eagle_backend_host_func: null function");
        requireDevice();
        const int device = resolveDevice(-1, stream);
        DeviceScope scope(device);
        eagle::exec::hostFunc(seamStream(stream), fn, arg);
    });
}

// ---- experimental --------------------------------------------------------------

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_captured_debug_dot(eagle_backend_captured h, const char* path,
    std::uint32_t flags)
{
    return guarded([&] {
        cudaGraph_t g = live(h)->graph.graph();
        if (g == nullptr)
            throw std::runtime_error("CapturedGraph is empty (no handle to dot)");
        if (path == nullptr)
            throw std::invalid_argument("eagle: null path");
        const cudaError_t e = cudaGraphDebugDotPrint(g, path, flags);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("cudaGraphDebugDotPrint failed: ") + cudaGetErrorString(e));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_capture_guard_depth(std::int64_t* depth)
{
    return guarded([&] { out(depth, static_cast<std::int64_t>(eagle::util::CaptureGuardState::instance().depth())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_capture_guard_pending_count(std::int64_t* count)
{
    return guarded([&] {
        out(count, static_cast<std::int64_t>(eagle::util::CaptureGuardState::instance().pendingCount()));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_create(
    const struct eagle_backend_composer_desc* desc, eagle_backend_composer* h)
{
    return guarded([&] {
        const auto d = readDesc(desc, 0);
        if (h == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        requireDevice();
        const int device = resolveDevice(d.device, std::nullopt);
        DeviceScope scope(device);
        auto c = std::make_shared<GraphComposer>(std::string(d.mode != nullptr ? d.mode : "sequenced"));
        *h = new eagle_backend_composer_s{ device, std::move(c), {} };
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_register_launcher(eagle_backend_composer h,
    eagle_backend_launcher launcher, const char* name, eagle_backend_void_fn pre, void* preCtx, std::int64_t* index)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        live(launcher);
        const auto i = h->composer->registerLauncher(std::move(launcher->launcher),
            name != nullptr ? std::string(name) : std::string(), preLaunch(pre, preCtx));
        delete launcher; // consumed on OK
        out(index, static_cast<std::int64_t>(i));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_register_callable(eagle_backend_composer h,
    eagle_backend_step_fn step, void* stepCtx, const char* name, eagle_backend_void_fn pre, void* preCtx,
    std::int64_t* index)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        if (step == nullptr)
            throw std::invalid_argument("eagle: null step callback");
        const auto i = h->composer->registerCallable(
            [step, stepCtx](cudaStream_t s) {
                if (step(stepCtx, reinterpret_cast<std::uint64_t>(s)) != 0)
                    throw CallbackFailed{};
            },
            name != nullptr ? std::string(name) : std::string(), preLaunch(pre, preCtx));
        out(index, static_cast<std::int64_t>(i));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_register_nested(eagle_backend_composer h,
    eagle_backend_composer nested, const char* name, eagle_backend_void_fn pre, void* preCtx, std::int64_t* index)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        live(nested);
        const auto i = h->composer->registerNested(nested->composer,
            name != nullptr ? std::string(name) : std::string(), preLaunch(pre, preCtx));
        out(index, static_cast<std::int64_t>(i));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_build(eagle_backend_composer h)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->composer->build();
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_set_routing(eagle_backend_composer h,
    const std::uint8_t* pattern, std::int64_t n)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        if (n < 0 || (n > 0 && pattern == nullptr))
            throw std::invalid_argument("eagle: bad routing pattern");
        std::vector<bool> p(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            p[static_cast<std::size_t>(i)] = pattern[i] != 0;
        h->composer->setRouting(p);
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_launch(eagle_backend_composer h, std::int64_t n)
{
    return guarded([&] {
        DeviceScope scope(live(h)->device);
        h->composer->launch(static_cast<eagle::idx_t>(n));
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_fired_history(eagle_backend_composer h,
    std::int64_t** flat, std::int64_t** offsets, std::int64_t* calls)
{
    return guarded([&] {
        if (flat == nullptr || offsets == nullptr || calls == nullptr)
            throw std::invalid_argument("eagle: null output pointer");
        const auto& hist = live(h)->composer->firedHistory();
        std::size_t total = 0;
        for (const auto& v : hist)
            total += v.size();
        auto* f = static_cast<std::int64_t*>(std::malloc(std::max<std::size_t>(1, total) * 8));
        auto* o = static_cast<std::int64_t*>(std::malloc((hist.size() + 1) * 8));
        if (f == nullptr || o == nullptr) {
            std::free(f);
            std::free(o);
            throw std::bad_alloc();
        }
        std::size_t k = 0;
        o[0] = 0;
        for (std::size_t c = 0; c < hist.size(); ++c) {
            for (auto v : hist[c])
                f[k++] = static_cast<std::int64_t>(v);
            o[c + 1] = static_cast<std::int64_t>(k);
        }
        *flat = f;
        *offsets = o;
        *calls = static_cast<std::int64_t>(hist.size());
    });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_reset_fired_history(eagle_backend_composer h)
{
    return guarded([&] { live(h)->composer->resetFiredHistory(); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_mode(eagle_backend_composer h, const char** mode)
{
    return guarded([&] { out(mode, live(h)->composer->mode().c_str()); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_num_members(eagle_backend_composer h,
    std::int64_t* n)
{
    return guarded([&] { out(n, static_cast<std::int64_t>(live(h)->composer->numMembers())); });
}

EAGLE_BACKEND_API eagle_backend_status eagle_backend_x_composer_member_name(eagle_backend_composer h, std::int64_t i,
    const char** name)
{
    return guarded([&] {
        live(h)->names = h->composer->memberNames();
        if (i < 0 || static_cast<std::size_t>(i) >= h->names.size())
            throw std::out_of_range("member index out of range");
        out(name, h->names[static_cast<std::size_t>(i)].c_str());
    });
}

EAGLE_BACKEND_API void eagle_backend_x_composer_release(eagle_backend_composer h)
{
    release(h);
}
