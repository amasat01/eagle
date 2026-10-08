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
#include "eagle/exec/DeviceKernel.h"
#include "eagle/filtering/Compact.h"
#include "eagle/filtering/Reorder.h"
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
    const std::string text = std::string("eagle: the CUDA backend cannot run here: ") + cudaGetErrorName(status)
        + " (" + cudaGetErrorString(status) + ")";
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
        copyText(full.archs, sizeof full.archs, EAGLE_CUDA_PLUGIN_ARCHS);
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
    return guarded([&] { out(groups, eagle_cuda_plugin::driverGroups(kCapabilities)); });
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
            CaptureFork(reinterpret_cast<cudaStream_t>(d.origin), static_cast<std::size_t>(d.branch_count)) };
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
        if (d.counter_ptr)
            *h = new eagle_backend_conditional_s{ device,
                CaptureConditional(origin, guard, d.loop_cap, reinterpret_cast<unsigned int*>(d.counter_ptr)) };
        else
            *h = new eagle_backend_conditional_s{ device, CaptureConditional(origin, guard) };
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
        out(launched,
            static_cast<std::int32_t>(eagle::exec::DeviceKernel::run(reinterpret_cast<CUfunction>(d.function), params,
                part, reinterpret_cast<CUstream>(d.stream), d.block)));
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
