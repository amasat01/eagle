// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cuda_driver.cpp
 * @brief The CUDA runtime functions ``libeagle_cuda.so`` calls, implemented
 *        over the CUDA DRIVER API, so the plugin needs ``libcuda.so.1`` and no
 *        CUDA runtime library.
 *
 * eagle's C++ headers call the CUDA runtime (``cudaStreamCreate``,
 * ``cudaGraphLaunch``, ...). The plugin is linked without any CUDA runtime
 * (nvcc ``-cudart none``); this translation unit defines exactly the runtime
 * entry points the plugin references, each a thin translation onto its driver
 * counterpart. Everything here is hidden: the version script exports the
 * ``eagle_backend_*`` seam and nothing else, so these definitions never
 * interpose another library's CUDA runtime in the same process.
 *
 * DRIVER. ``libcuda.so.1`` is opened on first use (``dlopen``); every driver
 * function is then resolved one by one through ``cuGetProcAddress`` at the
 * CUDA version whose ABI this file was written against (the ``PFN_*_vNNNNN``
 * typedefs), so a newer driver hands back that same ABI. A missing entry point
 * leaves its slot empty: the runtime function returns
 * ``cudaErrorCallRequiresNewerDriver`` and ``driverGroups`` clears every
 * capability group that needs it. No driver at all maps to
 * ``cudaErrorInsufficientDriver``, as the CUDA runtime reports it.
 *
 * CONTEXTS follow the CUDA runtime's rules, so the plugin shares the device's
 * PRIMARY context with every other runtime in the process (torch, CuPy, a
 * toolkit): the current device is the device of the calling thread's current
 * driver context; a call that needs a context and finds none makes the primary
 * context of the thread's selected device (initially 0) current;
 * ``cudaSetDevice`` retains and binds that device's primary context.
 *
 * STREAMS. The plugin is compiled with per-thread default-stream semantics, so
 * it references the ``*_ptsz`` spellings of the stream-ordered calls; in those,
 * stream 0 is the per-thread default stream (``CU_STREAM_PER_THREAD``). The
 * plain spellings keep the legacy meaning of 0. Both are provided.
 *
 * KERNELS. nvcc compiles the plugin's own kernels ahead of time into a fat
 * binary embedded in the object, and registers it through the
 * ``__cudaRegister*`` hooks defined here: the image is loaded with
 * ``cuModuleLoadData`` once per context, the kernels are looked up by their
 * device names, and a launch (``cudaLaunchKernel``, which is also what a
 * ``<<<...>>>`` lowers to) is a ``cuLaunchKernel``.
 */

// The plugin TU is compiled with per-thread default-stream semantics; this one
// defines BOTH spellings of each stream-ordered call explicitly.
#undef CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "plugin/cuda_driver.h"
#include "seam/eagle_backend.h"

namespace {

// ---- the driver --------------------------------------------------------------

using GetProc2 = CUresult (*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
using GetProc1 = CUresult (*)(const char*, void**, int, cuuint64_t);

constexpr std::uint64_t kCore = ~UINT64_C(0); // needed by every group

/** One driver entry point: name, the CUDA version of its ABI, the groups using it. */
struct Entry {
    const char* name;
    int version;
    std::uint64_t groups;
    void* fn;
};

enum Slot : int {
    sInit, sDeviceGetCount, sDeviceGet, sDevicePrimaryCtxRetain, sCtxGetCurrent,
    sCtxSetCurrent, sCtxGetDevice, sGetErrorName, sGetErrorString, sStreamGetCtx,
    sDeviceGetAttribute, sDeviceGetName, sDeviceTotalMem,
    sModuleLoadData, sModuleGetFunction, sLaunchKernel, sKernelGetFunction,
    sStreamCreate, sStreamDestroy, sStreamSynchronize, sStreamWaitEvent,
    sStreamBeginCapture, sStreamEndCapture, sStreamBeginCaptureToGraph, sStreamGetCaptureInfo,
    sStreamUpdateCaptureDependencies,
    sEventCreate, sEventDestroy, sEventRecord, sMemFree,
    sGraphCreate, sGraphDestroy, sGraphExecDestroy, sGraphInstantiateWithFlags, sGraphLaunch, sGraphGetNodes,
    sGraphNodeGetType, sGraphAddKernelNode, sGraphKernelNodeGetParams, sGraphExecKernelNodeSetParams,
    sGraphAddEmptyNode, sGraphAddChildGraphNode, sGraphChildGraphNodeGetGraph, sGraphNodeSetEnabled,
    sGraphDebugDotPrint, sGraphConditionalHandleCreate, sGraphAddNode,
    kSlots
};

constexpr std::uint64_t STREAM = EAGLE_BACKEND_CAP_STREAM, CAPTURER = EAGLE_BACKEND_CAP_CAPTURER,
                        CAPTURED = EAGLE_BACKEND_CAP_CAPTURED, FORK = EAGLE_BACKEND_CAP_FORK,
                        COND = EAGLE_BACKEND_CAP_CONDITIONAL, ATTR = EAGLE_BACKEND_CAP_ATTRIBUTION,
                        GRAPH = EAGLE_BACKEND_CAP_GRAPH, LAUNCHER = EAGLE_BACKEND_CAP_LAUNCHER,
                        EXEC = EAGLE_BACKEND_CAP_EXEC, DEVICE = EAGLE_BACKEND_CAP_DEVICE,
                        INTEROP = EAGLE_BACKEND_CAP_INTEROP, FILTER = EAGLE_BACKEND_CAP_FILTERING;

// Indexed by Slot; the versions match the PFN_*_vNNNNN typedef each call uses.
Entry g_entries[kSlots] = {
    { "cuInit", 2000, kCore, nullptr },
    { "cuDeviceGetCount", 2000, kCore, nullptr },
    { "cuDeviceGet", 2000, kCore, nullptr },
    { "cuDevicePrimaryCtxRetain", 7000, kCore, nullptr },
    { "cuCtxGetCurrent", 4000, kCore, nullptr },
    { "cuCtxSetCurrent", 4000, kCore, nullptr },
    { "cuCtxGetDevice", 2000, kCore, nullptr },
    { "cuGetErrorName", 6000, kCore, nullptr },
    { "cuGetErrorString", 6000, kCore, nullptr },
    { "cuStreamGetCtx", 9020, kCore, nullptr },
    { "cuDeviceGetAttribute", 2000, DEVICE | EXEC, nullptr },
    { "cuDeviceGetName", 2000, DEVICE, nullptr },
    { "cuDeviceTotalMem", 3020, DEVICE, nullptr },
    { "cuModuleLoadData", 2000, COND | FILTER, nullptr },
    { "cuModuleGetFunction", 2000, COND | FILTER, nullptr },
    { "cuLaunchKernel", 4000, COND | EXEC | FILTER, nullptr },
    { "cuKernelGetFunction", 12000, 0, nullptr },
    { "cuStreamCreate", 2000, STREAM | FORK | COND, nullptr },
    { "cuStreamDestroy", 4000, STREAM | FORK | COND, nullptr },
    { "cuStreamSynchronize", 2000, STREAM | GRAPH | LAUNCHER, nullptr },
    { "cuStreamWaitEvent", 3020, FORK | INTEROP, nullptr },
    { "cuStreamBeginCapture", 10010, CAPTURER, nullptr },
    { "cuStreamEndCapture", 10000, CAPTURER | COND, nullptr },
    { "cuStreamBeginCaptureToGraph", 12030, COND, nullptr },
    { "cuStreamGetCaptureInfo", 12030, COND | ATTR, nullptr },
    { "cuStreamUpdateCaptureDependencies", 11030, COND, nullptr },
    { "cuEventCreate", 2000, FORK | INTEROP, nullptr },
    { "cuEventDestroy", 4000, FORK | INTEROP, nullptr },
    { "cuEventRecord", 2000, FORK | INTEROP, nullptr },
    { "cuMemFree", 3020, 0, nullptr },
    { "cuGraphCreate", 10000, GRAPH | COND, nullptr },
    { "cuGraphDestroy", 10000, CAPTURER | CAPTURED | GRAPH | COND, nullptr },
    { "cuGraphExecDestroy", 10000, LAUNCHER, nullptr },
    { "cuGraphInstantiateWithFlags", 11040, LAUNCHER | GRAPH, nullptr },
    { "cuGraphLaunch", 10000, LAUNCHER, nullptr },
    { "cuGraphGetNodes", 10000, GRAPH | COND | ATTR, nullptr },
    { "cuGraphNodeGetType", 10000, GRAPH | COND | ATTR, nullptr },
    { "cuGraphAddKernelNode", 12000, GRAPH | COND, nullptr },
    { "cuGraphKernelNodeGetParams", 12000, GRAPH, nullptr },
    { "cuGraphExecKernelNodeSetParams", 12000, LAUNCHER, nullptr },
    { "cuGraphAddEmptyNode", 10000, GRAPH, nullptr },
    { "cuGraphAddChildGraphNode", 10000, GRAPH | COND, nullptr },
    { "cuGraphChildGraphNodeGetGraph", 10000, GRAPH, nullptr },
    { "cuGraphNodeSetEnabled", 11060, LAUNCHER | ATTR, nullptr },
    { "cuGraphDebugDotPrint", 11030, 0, nullptr },
    { "cuGraphConditionalHandleCreate", 12030, COND, nullptr },
    { "cuGraphAddNode", 12020, COND, nullptr },
};

struct Driver {
    bool loaded = false;
    GetProc2 getProc = nullptr; // cuGetProcAddress_v2 (CUDA 12.0+ drivers)
};

/** Names listed (comma separated) in EAGLE_CUDA_DRIVER_HIDE are treated as absent. */
bool hidden(const char* name)
{
    const char* list = std::getenv("EAGLE_CUDA_DRIVER_HIDE");
    if (list == nullptr)
        return false;
    const std::size_t n = std::strlen(name);
    for (const char* p = list; *p != '\0';) {
        const char* end = std::strchr(p, ',');
        const std::size_t len = end ? static_cast<std::size_t>(end - p) : std::strlen(p);
        if (len == n && std::strncmp(p, name, n) == 0)
            return true;
        if (end == nullptr)
            break;
        p = end + 1;
    }
    return false;
}

const Driver& driver()
{
    static const Driver d = [] {
        Driver out;
        void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr)
            return out;
        auto get2 = reinterpret_cast<GetProc2>(dlsym(lib, "cuGetProcAddress_v2"));
        auto get1 = get2 ? nullptr : reinterpret_cast<GetProc1>(dlsym(lib, "cuGetProcAddress"));
        if (get2 == nullptr && get1 == nullptr)
            return out;
        out.getProc = get2;
        for (Entry& e : g_entries) {
            void* fn = nullptr;
            if (get2) {
                CUdriverProcAddressQueryResult found{};
                if (get2(e.name, &fn, e.version, CU_GET_PROC_ADDRESS_LEGACY_STREAM, &found) != CUDA_SUCCESS
                    || found != CU_GET_PROC_ADDRESS_SUCCESS)
                    fn = nullptr;
            } else if (get1(e.name, &fn, e.version, CU_GET_PROC_ADDRESS_LEGACY_STREAM) != CUDA_SUCCESS) {
                fn = nullptr;
            }
            e.fn = hidden(e.name) ? nullptr : fn;
        }
        out.loaded = g_entries[sInit].fn != nullptr;
        return out;
    }();
    return d;
}

template <class Fn>
Fn fn(Slot s)
{
    return reinterpret_cast<Fn>(g_entries[s].fn);
}

// ---- runtime error state -----------------------------------------------------

thread_local cudaError_t t_lastError = cudaSuccess;

cudaError_t record(cudaError_t e)
{
    if (e != cudaSuccess)
        t_lastError = e;
    return e;
}

// The runtime's error codes take the driver's values for every condition the
// two share; the codes below are the runtime-only ones this layer produces.
cudaError_t fromDriver(CUresult r)
{
    return record(static_cast<cudaError_t>(r));
}

/** Initialise the driver once; the runtime's status for "no usable driver". */
cudaError_t initDriver()
{
    static const cudaError_t status = [] {
        const Driver& d = driver();
        if (!d.loaded)
            return cudaErrorInsufficientDriver;
        return static_cast<cudaError_t>(fn<PFN_cuInit_v2000>(sInit)(0));
    }();
    return record(status);
}

/** The slot's function, or nullptr after recording why it cannot be called. */
template <class Fn>
Fn need(Slot s, cudaError_t& err)
{
    err = initDriver();
    if (err != cudaSuccess)
        return nullptr;
    if (g_entries[s].fn == nullptr) {
        err = record(cudaErrorCallRequiresNewerDriver);
        return nullptr;
    }
    return fn<Fn>(s);
}

#define EAGLE_DRV(var, slot, type)                                                                         \
    cudaError_t var##_err = cudaSuccess;                                                                   \
    const auto var = need<type>(slot, var##_err);                                                          \
    if (var == nullptr)                                                                                    \
        return var##_err

// ---- contexts ----------------------------------------------------------------

thread_local int t_device = 0; // the thread's selected device when no context is current

struct Primary {
    std::mutex mutex;
    std::map<int, CUcontext> contexts; // retained once, never released (as the runtime does)
};

Primary& primary()
{
    static Primary* p = new Primary();
    return *p;
}

cudaError_t bindPrimary(int device)
{
    EAGLE_DRV(retain, sDevicePrimaryCtxRetain, PFN_cuDevicePrimaryCtxRetain_v7000);
    EAGLE_DRV(set, sCtxSetCurrent, PFN_cuCtxSetCurrent_v4000);
    EAGLE_DRV(get, sDeviceGet, PFN_cuDeviceGet_v2000);
    CUcontext ctx = nullptr;
    {
        Primary& pr = primary();
        std::lock_guard<std::mutex> lock(pr.mutex);
        auto it = pr.contexts.find(device);
        if (it == pr.contexts.end()) {
            CUdevice dev = 0;
            if (CUresult r = get(&dev, device); r != CUDA_SUCCESS)
                return record(
                    r == CUDA_ERROR_INVALID_VALUE ? cudaErrorInvalidDevice : static_cast<cudaError_t>(static_cast<int>(r)));
            if (CUresult r = retain(&ctx, dev); r != CUDA_SUCCESS)
                return fromDriver(r);
            pr.contexts.emplace(device, ctx);
        } else {
            ctx = it->second;
        }
    }
    if (CUresult r = set(ctx); r != CUDA_SUCCESS)
        return fromDriver(r);
    t_device = device;
    return cudaSuccess;
}

/** Make sure the calling thread has a current context (the runtime's lazy init). */
cudaError_t ensureContext()
{
    EAGLE_DRV(getCurrent, sCtxGetCurrent, PFN_cuCtxGetCurrent_v4000);
    CUcontext ctx = nullptr;
    if (CUresult r = getCurrent(&ctx); r != CUDA_SUCCESS)
        return fromDriver(r);
    if (ctx != nullptr)
        return cudaSuccess;
    return bindPrimary(t_device);
}

#define EAGLE_CTX()                                                                                        \
    do {                                                                                                   \
        if (cudaError_t ctx_err_ = ensureContext(); ctx_err_ != cudaSuccess)                               \
            return ctx_err_;                                                                               \
    } while (0)

CUcontext currentContext()
{
    CUcontext ctx = nullptr;
    if (g_entries[sCtxGetCurrent].fn)
        fn<PFN_cuCtxGetCurrent_v4000>(sCtxGetCurrent)(&ctx);
    return ctx;
}

CUstream perThread(cudaStream_t s)
{
    return s == nullptr ? CU_STREAM_PER_THREAD : reinterpret_cast<CUstream>(s);
}

CUstream legacy(cudaStream_t s)
{
    return reinterpret_cast<CUstream>(s);
}

// ---- the plugin's own kernels (nvcc's registration hooks) ---------------------

struct FatbinWrapper {
    int magic;
    int version;
    const void* data;
    void* filenameOrFatbins;
};

struct Kernel {
    std::size_t image; // index into Registry::images
    std::string name;
};

/** The kernel registry. Constructed on first use: nvcc's registration runs from
 *  a static initializer of another translation unit, in no defined order with
 *  this one's. */
struct Registry {
    std::mutex mutex;
    std::vector<const void*> images;                               // registered fat binaries
    std::map<const void*, Kernel> kernels;                          // host stub -> device kernel
    std::map<std::pair<CUcontext, std::size_t>, CUmodule> modules; // per context and image
    std::map<std::pair<CUcontext, const void*>, CUfunction> functions;
    std::map<CUfunction, const void*> stubs; // reverse, for kernel-node parameter read-back
};

Registry& registry()
{
    static Registry* r = new Registry(); // never destroyed: used until the process ends
    return *r;
}

/** The CUfunction of host stub @p stub in the current context, loading its image once. */
cudaError_t functionOf(const void* stub, CUfunction* out)
{
    EAGLE_CTX();
    const CUcontext ctx = currentContext();
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto k = reg.kernels.find(stub);
    if (k == reg.kernels.end())
        return record(cudaErrorInvalidDeviceFunction);
    if (auto f = reg.functions.find({ ctx, stub }); f != reg.functions.end()) {
        *out = f->second;
        return cudaSuccess;
    }
    EAGLE_DRV(load, sModuleLoadData, PFN_cuModuleLoadData_v2000);
    EAGLE_DRV(getFunction, sModuleGetFunction, PFN_cuModuleGetFunction_v2000);
    CUmodule& mod = reg.modules[{ ctx, k->second.image }];
    if (mod == nullptr) {
        if (CUresult r = load(&mod, reg.images[k->second.image]); r != CUDA_SUCCESS) {
            mod = nullptr;
            return fromDriver(r);
        }
    }
    CUfunction f = nullptr;
    if (CUresult r = getFunction(&f, mod, k->second.name.c_str()); r != CUDA_SUCCESS)
        return fromDriver(r);
    reg.functions[{ ctx, stub }] = f;
    reg.stubs[f] = stub;
    *out = f;
    return cudaSuccess;
}

/** A kernel-node ``func``: one of the plugin's stubs, or a driver function read back from a graph. */
cudaError_t resolveFunc(const void* func, CUfunction* out)
{
    {
        Registry& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        if (reg.kernels.find(func) == reg.kernels.end()) {
            *out = reinterpret_cast<CUfunction>(const_cast<void*>(func));
            return func ? cudaSuccess : record(cudaErrorInvalidDeviceFunction);
        }
    }
    return functionOf(func, out);
}

void* stubOf(CUfunction f)
{
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.stubs.find(f);
    return it == reg.stubs.end() ? reinterpret_cast<void*>(f) : const_cast<void*>(it->second);
}

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t shared;
    cudaStream_t stream;
};
thread_local std::vector<LaunchConfig> t_launchConfigs;

cudaError_t launch(const void* func, dim3 grid, dim3 block, void** args, std::size_t shared, CUstream stream)
{
    CUfunction f = nullptr;
    if (cudaError_t e = functionOf(func, &f); e != cudaSuccess)
        return e;
    EAGLE_DRV(launchKernel, sLaunchKernel, PFN_cuLaunchKernel_v4000);
    return fromDriver(launchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z,
        static_cast<unsigned>(shared), stream, args, nullptr));
}

CUDA_KERNEL_NODE_PARAMS_v2 driverParams(const cudaKernelNodeParams& p, CUfunction f)
{
    CUDA_KERNEL_NODE_PARAMS_v2 k{};
    k.func = f;
    k.gridDimX = p.gridDim.x;
    k.gridDimY = p.gridDim.y;
    k.gridDimZ = p.gridDim.z;
    k.blockDimX = p.blockDim.x;
    k.blockDimY = p.blockDim.y;
    k.blockDimZ = p.blockDim.z;
    k.sharedMemBytes = p.sharedMemBytes;
    k.kernelParams = p.kernelParams;
    k.extra = p.extra;
    return k;
}

// ---- error text --------------------------------------------------------------

struct ErrorName {
    int code;
    const char* name;
};

constexpr ErrorName kErrorNames[] = {
    { 0, "cudaSuccess" },
    { 1, "cudaErrorInvalidValue" },
    { 2, "cudaErrorMemoryAllocation" },
    { 3, "cudaErrorInitializationError" },
    { 4, "cudaErrorCudartUnloading" },
    { 5, "cudaErrorProfilerDisabled" },
    { 6, "cudaErrorProfilerNotInitialized" },
    { 7, "cudaErrorProfilerAlreadyStarted" },
    { 8, "cudaErrorProfilerAlreadyStopped" },
    { 9, "cudaErrorInvalidConfiguration" },
    { 12, "cudaErrorInvalidPitchValue" },
    { 13, "cudaErrorInvalidSymbol" },
    { 16, "cudaErrorInvalidHostPointer" },
    { 17, "cudaErrorInvalidDevicePointer" },
    { 18, "cudaErrorInvalidTexture" },
    { 19, "cudaErrorInvalidTextureBinding" },
    { 20, "cudaErrorInvalidChannelDescriptor" },
    { 21, "cudaErrorInvalidMemcpyDirection" },
    { 22, "cudaErrorAddressOfConstant" },
    { 23, "cudaErrorTextureFetchFailed" },
    { 24, "cudaErrorTextureNotBound" },
    { 25, "cudaErrorSynchronizationError" },
    { 26, "cudaErrorInvalidFilterSetting" },
    { 27, "cudaErrorInvalidNormSetting" },
    { 28, "cudaErrorMixedDeviceExecution" },
    { 31, "cudaErrorNotYetImplemented" },
    { 32, "cudaErrorMemoryValueTooLarge" },
    { 34, "cudaErrorStubLibrary" },
    { 35, "cudaErrorInsufficientDriver" },
    { 36, "cudaErrorCallRequiresNewerDriver" },
    { 37, "cudaErrorInvalidSurface" },
    { 43, "cudaErrorDuplicateVariableName" },
    { 44, "cudaErrorDuplicateTextureName" },
    { 45, "cudaErrorDuplicateSurfaceName" },
    { 46, "cudaErrorDevicesUnavailable" },
    { 49, "cudaErrorIncompatibleDriverContext" },
    { 52, "cudaErrorMissingConfiguration" },
    { 53, "cudaErrorPriorLaunchFailure" },
    { 65, "cudaErrorLaunchMaxDepthExceeded" },
    { 66, "cudaErrorLaunchFileScopedTex" },
    { 67, "cudaErrorLaunchFileScopedSurf" },
    { 68, "cudaErrorSyncDepthExceeded" },
    { 69, "cudaErrorLaunchPendingCountExceeded" },
    { 98, "cudaErrorInvalidDeviceFunction" },
    { 100, "cudaErrorNoDevice" },
    { 101, "cudaErrorInvalidDevice" },
    { 102, "cudaErrorDeviceNotLicensed" },
    { 103, "cudaErrorSoftwareValidityNotEstablished" },
    { 127, "cudaErrorStartupFailure" },
    { 200, "cudaErrorInvalidKernelImage" },
    { 201, "cudaErrorDeviceUninitialized" },
    { 205, "cudaErrorMapBufferObjectFailed" },
    { 206, "cudaErrorUnmapBufferObjectFailed" },
    { 207, "cudaErrorArrayIsMapped" },
    { 208, "cudaErrorAlreadyMapped" },
    { 209, "cudaErrorNoKernelImageForDevice" },
    { 210, "cudaErrorAlreadyAcquired" },
    { 211, "cudaErrorNotMapped" },
    { 212, "cudaErrorNotMappedAsArray" },
    { 213, "cudaErrorNotMappedAsPointer" },
    { 214, "cudaErrorECCUncorrectable" },
    { 215, "cudaErrorUnsupportedLimit" },
    { 216, "cudaErrorDeviceAlreadyInUse" },
    { 217, "cudaErrorPeerAccessUnsupported" },
    { 218, "cudaErrorInvalidPtx" },
    { 219, "cudaErrorInvalidGraphicsContext" },
    { 220, "cudaErrorNvlinkUncorrectable" },
    { 221, "cudaErrorJitCompilerNotFound" },
    { 222, "cudaErrorUnsupportedPtxVersion" },
    { 223, "cudaErrorJitCompilationDisabled" },
    { 224, "cudaErrorUnsupportedExecAffinity" },
    { 225, "cudaErrorUnsupportedDevSideSync" },
    { 300, "cudaErrorInvalidSource" },
    { 301, "cudaErrorFileNotFound" },
    { 302, "cudaErrorSharedObjectSymbolNotFound" },
    { 303, "cudaErrorSharedObjectInitFailed" },
    { 304, "cudaErrorOperatingSystem" },
    { 400, "cudaErrorInvalidResourceHandle" },
    { 401, "cudaErrorIllegalState" },
    { 402, "cudaErrorLossyQuery" },
    { 500, "cudaErrorSymbolNotFound" },
    { 600, "cudaErrorNotReady" },
    { 700, "cudaErrorIllegalAddress" },
    { 701, "cudaErrorLaunchOutOfResources" },
    { 702, "cudaErrorLaunchTimeout" },
    { 703, "cudaErrorLaunchIncompatibleTexturing" },
    { 704, "cudaErrorPeerAccessAlreadyEnabled" },
    { 705, "cudaErrorPeerAccessNotEnabled" },
    { 708, "cudaErrorSetOnActiveProcess" },
    { 709, "cudaErrorContextIsDestroyed" },
    { 710, "cudaErrorAssert" },
    { 711, "cudaErrorTooManyPeers" },
    { 712, "cudaErrorHostMemoryAlreadyRegistered" },
    { 713, "cudaErrorHostMemoryNotRegistered" },
    { 714, "cudaErrorHardwareStackError" },
    { 715, "cudaErrorIllegalInstruction" },
    { 716, "cudaErrorMisalignedAddress" },
    { 717, "cudaErrorInvalidAddressSpace" },
    { 718, "cudaErrorInvalidPc" },
    { 719, "cudaErrorLaunchFailure" },
    { 720, "cudaErrorCooperativeLaunchTooLarge" },
    { 800, "cudaErrorNotPermitted" },
    { 801, "cudaErrorNotSupported" },
    { 802, "cudaErrorSystemNotReady" },
    { 803, "cudaErrorSystemDriverMismatch" },
    { 804, "cudaErrorCompatNotSupportedOnDevice" },
    { 805, "cudaErrorMpsConnectionFailed" },
    { 806, "cudaErrorMpsRpcFailure" },
    { 807, "cudaErrorMpsServerNotReady" },
    { 808, "cudaErrorMpsMaxClientsReached" },
    { 809, "cudaErrorMpsMaxConnectionsReached" },
    { 810, "cudaErrorMpsClientTerminated" },
    { 811, "cudaErrorCdpNotSupported" },
    { 812, "cudaErrorCdpVersionMismatch" },
    { 900, "cudaErrorStreamCaptureUnsupported" },
    { 901, "cudaErrorStreamCaptureInvalidated" },
    { 902, "cudaErrorStreamCaptureMerge" },
    { 903, "cudaErrorStreamCaptureUnmatched" },
    { 904, "cudaErrorStreamCaptureUnjoined" },
    { 905, "cudaErrorStreamCaptureIsolation" },
    { 906, "cudaErrorStreamCaptureImplicit" },
    { 907, "cudaErrorCapturedEvent" },
    { 908, "cudaErrorStreamCaptureWrongThread" },
    { 909, "cudaErrorTimeout" },
    { 910, "cudaErrorGraphExecUpdateFailure" },
    { 911, "cudaErrorExternalDevice" },
    { 912, "cudaErrorInvalidClusterSize" },
    { 913, "cudaErrorFunctionNotLoaded" },
    { 914, "cudaErrorInvalidResourceType" },
    { 915, "cudaErrorInvalidResourceConfiguration" },
    { 999, "cudaErrorUnknown" },
    { 10000, "cudaErrorApiFailureBase" },
};

const char* runtimeName(cudaError_t e)
{
    for (const ErrorName& n : kErrorNames)
        if (n.code == static_cast<int>(e))
            return n.name;
    return "cudaErrorUnknown";
}

} // namespace

namespace eagle_cuda_plugin {

std::uint64_t driverGroups(std::uint64_t wanted)
{
    if (!driver().loaded)
        return wanted;
    std::uint64_t missing = 0;
    for (const Entry& e : g_entries)
        if (e.fn == nullptr)
            missing |= e.groups;
    return wanted & ~missing;
}

} // namespace eagle_cuda_plugin

// =============================================================================
// The CUDA runtime entry points the plugin references. Hidden (the version
// script keeps them local to libeagle_cuda.so).
// =============================================================================

extern "C" {

// ---- nvcc registration hooks -------------------------------------------------

void** __cudaRegisterFatBinary(void* fatCubin)
{
    const auto* w = static_cast<const FatbinWrapper*>(fatCubin);
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.images.push_back(w->data);
    // The handle nvcc passes back is only ever handed to the hooks below.
    return reinterpret_cast<void**>(reg.images.size());
}

void __cudaRegisterFatBinaryEnd(void**) { }

// Modules die with their contexts; nothing is unloaded at process exit, when
// the driver may already be shutting down.
void __cudaUnregisterFatBinary(void**) { }

void __cudaRegisterFunction(void** handle, const char* hostFun, char*, const char* deviceName, int, uint3*, uint3*,
    dim3*, dim3*, int*)
{
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.kernels[hostFun] = Kernel{ reinterpret_cast<std::size_t>(handle) - 1, deviceName };
}

// The plugin reads no __device__ variable from the host.
void __cudaRegisterVar(void**, char*, char*, const char*, int, std::size_t, int, int) { }

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, std::size_t sharedMem, CUstream_st* stream)
{
    t_launchConfigs.push_back(LaunchConfig{ gridDim, blockDim, sharedMem, stream });
    return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, std::size_t* sharedMem, void* stream)
{
    if (t_launchConfigs.empty())
        return record(cudaErrorMissingConfiguration);
    const LaunchConfig c = t_launchConfigs.back();
    t_launchConfigs.pop_back();
    *gridDim = c.grid;
    *blockDim = c.block;
    *sharedMem = c.shared;
    *static_cast<cudaStream_t*>(stream) = c.stream;
    return cudaSuccess;
}

// ---- errors ------------------------------------------------------------------

cudaError_t cudaGetLastError(void)
{
    const cudaError_t e = t_lastError;
    t_lastError = cudaSuccess;
    return e;
}

const char* cudaGetErrorName(cudaError_t error)
{
    return runtimeName(error);
}

const char* cudaGetErrorString(cudaError_t error)
{
    switch (error) {
    case cudaSuccess:
        return "no error";
    case cudaErrorInsufficientDriver:
        return "no usable CUDA driver: libcuda.so.1 could not be loaded";
    case cudaErrorCallRequiresNewerDriver:
        return "the installed CUDA driver does not provide this entry point";
    case cudaErrorInvalidDeviceFunction:
        return "invalid device function";
    case cudaErrorMissingConfiguration:
        return "__global__ function call is not configured";
    default:
        break;
    }
    const cudaError_t saved = t_lastError; // a lookup is not the caller's error
    const char* text = nullptr;
    if (initDriver() == cudaSuccess && g_entries[sGetErrorString].fn != nullptr
        && fn<PFN_cuGetErrorString_v6000>(sGetErrorString)(static_cast<CUresult>(error), &text) == CUDA_SUCCESS
        && text != nullptr) {
        t_lastError = saved;
        return text;
    }
    t_lastError = saved;
    return "unrecognized error code";
}

// ---- devices and contexts ----------------------------------------------------

cudaError_t cudaGetDeviceCount(int* count)
{
    EAGLE_DRV(getCount, sDeviceGetCount, PFN_cuDeviceGetCount_v2000);
    int n = 0;
    if (CUresult r = getCount(&n); r != CUDA_SUCCESS)
        return fromDriver(r);
    *count = n;
    return n == 0 ? record(cudaErrorNoDevice) : cudaSuccess;
}

cudaError_t cudaGetDevice(int* device)
{
    EAGLE_DRV(getCurrent, sCtxGetCurrent, PFN_cuCtxGetCurrent_v4000);
    EAGLE_DRV(getDevice, sCtxGetDevice, PFN_cuCtxGetDevice_v2000);
    CUcontext ctx = nullptr;
    if (CUresult r = getCurrent(&ctx); r != CUDA_SUCCESS)
        return fromDriver(r);
    if (ctx == nullptr) {
        *device = t_device;
        return cudaSuccess;
    }
    CUdevice dev = 0;
    if (CUresult r = getDevice(&dev); r != CUDA_SUCCESS)
        return fromDriver(r);
    *device = static_cast<int>(dev);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device)
{
    int n = 0;
    if (cudaError_t e = cudaGetDeviceCount(&n); e != cudaSuccess)
        return e;
    if (device < 0 || device >= n)
        return record(cudaErrorInvalidDevice);
    return bindPrimary(device);
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
{
    EAGLE_DRV(getAttribute, sDeviceGetAttribute, PFN_cuDeviceGetAttribute_v2000);
    EAGLE_DRV(get, sDeviceGet, PFN_cuDeviceGet_v2000);
    CUdevice dev = 0;
    if (get(&dev, device) != CUDA_SUCCESS)
        return record(cudaErrorInvalidDevice);
    return fromDriver(getAttribute(value, static_cast<CUdevice_attribute>(attr), dev));
}

cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device)
{
    EAGLE_DRV(getAttribute, sDeviceGetAttribute, PFN_cuDeviceGetAttribute_v2000);
    EAGLE_DRV(getName, sDeviceGetName, PFN_cuDeviceGetName_v2000);
    EAGLE_DRV(totalMem, sDeviceTotalMem, PFN_cuDeviceTotalMem_v3020);
    EAGLE_DRV(get, sDeviceGet, PFN_cuDeviceGet_v2000);
    CUdevice dev = 0;
    if (get(&dev, device) != CUDA_SUCCESS)
        return record(cudaErrorInvalidDevice);
    *prop = cudaDeviceProp{};
    if (CUresult r = getName(prop->name, sizeof prop->name, dev); r != CUDA_SUCCESS)
        return fromDriver(r);
    std::size_t bytes = 0;
    if (CUresult r = totalMem(&bytes, dev); r != CUDA_SUCCESS)
        return fromDriver(r);
    prop->totalGlobalMem = bytes;
    auto attr = [&](CUdevice_attribute a) {
        int v = 0;
        return getAttribute(&v, a, dev) == CUDA_SUCCESS ? v : 0;
    };
    prop->major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    prop->minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    prop->multiProcessorCount = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
#if CUDART_VERSION < 13000 // CUDA 13 dropped both fields (cudaDeviceGetAttribute answers them)
    prop->clockRate = attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    prop->memoryClockRate = attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
#endif
    prop->memoryBusWidth = attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
    prop->sharedMemPerBlock = static_cast<std::size_t>(attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK));
    prop->sharedMemPerMultiprocessor
        = static_cast<std::size_t>(attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR));
    prop->sharedMemPerBlockOptin = static_cast<std::size_t>(attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN));
    prop->regsPerBlock = attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    prop->regsPerMultiprocessor = attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    prop->warpSize = attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    prop->maxThreadsPerBlock = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    prop->maxThreadsPerMultiProcessor = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    prop->maxThreadsDim[0] = attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    prop->maxThreadsDim[1] = attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    prop->maxThreadsDim[2] = attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
    prop->maxGridSize[0] = attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    prop->maxGridSize[1] = attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    prop->maxGridSize[2] = attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    prop->totalConstMem = static_cast<std::size_t>(attr(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY));
    prop->l2CacheSize = attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
    prop->pciBusID = attr(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
    prop->pciDeviceID = attr(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
    prop->pciDomainID = attr(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
    prop->concurrentKernels = attr(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
    prop->asyncEngineCount = attr(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
    prop->unifiedAddressing = attr(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
    prop->managedMemory = attr(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
    prop->integrated = attr(CU_DEVICE_ATTRIBUTE_INTEGRATED);
    prop->ECCEnabled = attr(CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
    return cudaSuccess;
}

#if CUDART_VERSION >= 13000
// CUDA 13 declares the unversioned name (12.x maps it to _v2 with a macro).
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
{
    return cudaGetDeviceProperties_v2(prop, device);
}
#endif

cudaError_t cudaFree(void* devPtr)
{
    EAGLE_CTX();
    if (devPtr == nullptr)
        return cudaSuccess;
    EAGLE_DRV(memFree, sMemFree, PFN_cuMemFree_v3020);
    return fromDriver(memFree(reinterpret_cast<CUdeviceptr>(devPtr)));
}

static cudaError_t entryPoint(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags,
    cudaDriverEntryPointQueryResult* status)
{
    if (cudaError_t e = initDriver(); e != cudaSuccess)
        return e;
    const GetProc2 get2 = driver().getProc;
    if (get2 == nullptr)
        return record(cudaErrorCallRequiresNewerDriver);
    CUdriverProcAddressQueryResult found{};
    const CUresult r = get2(symbol, funcPtr, static_cast<int>(cudaVersion), flags, &found);
    if (status != nullptr)
        *status = static_cast<cudaDriverEntryPointQueryResult>(found);
    if (hidden(symbol)) {
        *funcPtr = nullptr;
        if (status != nullptr)
            *status = cudaDriverEntryPointSymbolNotFound;
    }
    return fromDriver(r);
}

cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion,
    unsigned long long flags, cudaDriverEntryPointQueryResult* status)
{
    return entryPoint(symbol, funcPtr, cudaVersion, flags, status);
}

// Per-thread spelling: cudaEnableDefault asks for the per-thread default stream.
cudaError_t cudaGetDriverEntryPointByVersion_ptsz(const char* symbol, void** funcPtr, unsigned int cudaVersion,
    unsigned long long flags, cudaDriverEntryPointQueryResult* status)
{
    return entryPoint(symbol, funcPtr, cudaVersion,
        flags == cudaEnableDefault ? static_cast<unsigned long long>(CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
                                   : flags,
        status);
}

// ---- streams and events ------------------------------------------------------

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags)
{
    EAGLE_CTX();
    EAGLE_DRV(create, sStreamCreate, PFN_cuStreamCreate_v2000);
    return fromDriver(create(reinterpret_cast<CUstream*>(stream), flags));
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    EAGLE_DRV(destroy, sStreamDestroy, PFN_cuStreamDestroy_v4000);
    return fromDriver(destroy(legacy(stream)));
}

static cudaError_t streamSynchronize(CUstream s)
{
    EAGLE_CTX();
    EAGLE_DRV(sync, sStreamSynchronize, PFN_cuStreamSynchronize_v2000);
    return fromDriver(sync(s));
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { return streamSynchronize(legacy(stream)); }
cudaError_t cudaStreamSynchronize_ptsz(cudaStream_t stream) { return streamSynchronize(perThread(stream)); }

static cudaError_t streamWaitEvent(CUstream s, cudaEvent_t event, unsigned int flags)
{
    EAGLE_CTX();
    EAGLE_DRV(wait, sStreamWaitEvent, PFN_cuStreamWaitEvent_v3020);
    return fromDriver(wait(s, reinterpret_cast<CUevent>(event), flags));
}
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return streamWaitEvent(legacy(stream), event, flags);
}
cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return streamWaitEvent(perThread(stream), event, flags);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
{
    EAGLE_CTX();
    EAGLE_DRV(create, sEventCreate, PFN_cuEventCreate_v2000);
    return fromDriver(create(reinterpret_cast<CUevent*>(event), flags));
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    EAGLE_DRV(destroy, sEventDestroy, PFN_cuEventDestroy_v4000);
    return fromDriver(destroy(reinterpret_cast<CUevent>(event)));
}

static cudaError_t eventRecord(cudaEvent_t event, CUstream s)
{
    EAGLE_CTX();
    EAGLE_DRV(rec, sEventRecord, PFN_cuEventRecord_v2000);
    return fromDriver(rec(reinterpret_cast<CUevent>(event), s));
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) { return eventRecord(event, legacy(stream)); }
cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream)
{
    return eventRecord(event, perThread(stream));
}

// ---- stream capture ----------------------------------------------------------

static cudaError_t beginCapture(CUstream s, cudaStreamCaptureMode mode)
{
    EAGLE_CTX();
    EAGLE_DRV(begin, sStreamBeginCapture, PFN_cuStreamBeginCapture_v10010);
    return fromDriver(begin(s, static_cast<CUstreamCaptureMode>(mode)));
}
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    return beginCapture(legacy(stream), mode);
}
cudaError_t cudaStreamBeginCapture_ptsz(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    return beginCapture(perThread(stream), mode);
}

static cudaError_t beginCaptureToGraph(CUstream s, cudaGraph_t graph, const cudaGraphNode_t* deps,
    const cudaGraphEdgeData* edges, std::size_t n, cudaStreamCaptureMode mode)
{
    EAGLE_CTX();
    EAGLE_DRV(begin, sStreamBeginCaptureToGraph, PFN_cuStreamBeginCaptureToGraph_v12030);
    return fromDriver(begin(s, graph, deps, reinterpret_cast<const CUgraphEdgeData*>(edges), n,
        static_cast<CUstreamCaptureMode>(mode)));
}
cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* deps,
    const cudaGraphEdgeData* edges, std::size_t n, cudaStreamCaptureMode mode)
{
    return beginCaptureToGraph(legacy(stream), graph, deps, edges, n, mode);
}
cudaError_t cudaStreamBeginCaptureToGraph_ptsz(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* deps,
    const cudaGraphEdgeData* edges, std::size_t n, cudaStreamCaptureMode mode)
{
    return beginCaptureToGraph(perThread(stream), graph, deps, edges, n, mode);
}

static cudaError_t endCapture(CUstream s, cudaGraph_t* graph)
{
    EAGLE_CTX();
    EAGLE_DRV(end, sStreamEndCapture, PFN_cuStreamEndCapture_v10000);
    return fromDriver(end(s, graph));
}
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) { return endCapture(legacy(stream), graph); }
cudaError_t cudaStreamEndCapture_ptsz(cudaStream_t stream, cudaGraph_t* graph)
{
    return endCapture(perThread(stream), graph);
}

static cudaError_t captureInfo(CUstream s, cudaStreamCaptureStatus* status, unsigned long long* id,
    cudaGraph_t* graph, const cudaGraphNode_t** deps, const cudaGraphEdgeData** edges, std::size_t* n)
{
    EAGLE_CTX();
    EAGLE_DRV(info, sStreamGetCaptureInfo, PFN_cuStreamGetCaptureInfo_v12030);
    CUstreamCaptureStatus st{};
    cuuint64_t cid = 0;
    const CUresult r = info(s, &st, &cid, graph, deps, reinterpret_cast<const CUgraphEdgeData**>(edges), n);
    if (status != nullptr)
        *status = static_cast<cudaStreamCaptureStatus>(st);
    if (id != nullptr)
        *id = cid;
    return fromDriver(r);
}
// CUDA 13 renamed cudaStreamGetCaptureInfo_v3 to cudaStreamGetCaptureInfo (same arguments).
#if CUDART_VERSION >= 13000
#define EAGLE_CAPTURE_INFO cudaStreamGetCaptureInfo
#define EAGLE_CAPTURE_INFO_PTSZ cudaStreamGetCaptureInfo_ptsz
#else
#define EAGLE_CAPTURE_INFO cudaStreamGetCaptureInfo_v3
#define EAGLE_CAPTURE_INFO_PTSZ cudaStreamGetCaptureInfo_v3_ptsz
#endif
cudaError_t EAGLE_CAPTURE_INFO(cudaStream_t stream, cudaStreamCaptureStatus* status, unsigned long long* id,
    cudaGraph_t* graph, const cudaGraphNode_t** deps, const cudaGraphEdgeData** edges, std::size_t* n)
{
    return captureInfo(legacy(stream), status, id, graph, deps, edges, n);
}
cudaError_t EAGLE_CAPTURE_INFO_PTSZ(cudaStream_t stream, cudaStreamCaptureStatus* status,
    unsigned long long* id, cudaGraph_t* graph, const cudaGraphNode_t** deps, const cudaGraphEdgeData** edges,
    std::size_t* n)
{
    return captureInfo(perThread(stream), status, id, graph, deps, edges, n);
}

static cudaError_t updateCaptureDeps(CUstream s, cudaGraphNode_t* deps, std::size_t n, unsigned int flags)
{
    EAGLE_CTX();
    EAGLE_DRV(update, sStreamUpdateCaptureDependencies, PFN_cuStreamUpdateCaptureDependencies_v11030);
    return fromDriver(update(s, deps, n, flags));
}
// CUDA 13 adds an edge-data argument; eagle never passes edge data, so a non-null one is refused.
#if CUDART_VERSION >= 13000
#define EAGLE_EDGES_ARG const cudaGraphEdgeData* edges,
#define EAGLE_NO_EDGES() \
    if (edges != nullptr)  \
        return record(cudaErrorNotSupported);
#else
#define EAGLE_EDGES_ARG
#define EAGLE_NO_EDGES()
#endif
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* deps, EAGLE_EDGES_ARG
    std::size_t n, unsigned int flags)
{
    EAGLE_NO_EDGES()
    return updateCaptureDeps(legacy(stream), deps, n, flags);
}
cudaError_t cudaStreamUpdateCaptureDependencies_ptsz(cudaStream_t stream, cudaGraphNode_t* deps, EAGLE_EDGES_ARG
    std::size_t n, unsigned int flags)
{
    EAGLE_NO_EDGES()
    return updateCaptureDeps(perThread(stream), deps, n, flags);
}

// ---- kernels -----------------------------------------------------------------

cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, std::size_t shared,
    cudaStream_t stream)
{
    return launch(func, grid, block, args, shared, legacy(stream));
}

cudaError_t cudaLaunchKernel_ptsz(const void* func, dim3 grid, dim3 block, void** args, std::size_t shared,
    cudaStream_t stream)
{
    return launch(func, grid, block, args, shared, perThread(stream));
}

#if CUDART_VERSION >= 13000
// CUDA 13's compiler-generated launch stubs fetch a kernel handle once, then launch through it. The handle is
// only ever handed back to the two entries below, so it is the host function pointer itself.
cudaError_t __cudaGetKernel(cudaKernel_t* kernel, const void* func)
{
    *kernel = reinterpret_cast<cudaKernel_t>(const_cast<void*>(func));
    return cudaSuccess;
}

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void** args, std::size_t shared,
    cudaStream_t stream)
{
    return launch(reinterpret_cast<const void*>(kernel), grid, block, args, shared, legacy(stream));
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid, dim3 block, void** args, std::size_t shared,
    cudaStream_t stream)
{
    return launch(reinterpret_cast<const void*>(kernel), grid, block, args, shared, perThread(stream));
}
#endif

// ---- graphs ------------------------------------------------------------------

cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags)
{
    EAGLE_CTX();
    EAGLE_DRV(create, sGraphCreate, PFN_cuGraphCreate_v10000);
    return fromDriver(create(graph, flags));
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    EAGLE_DRV(destroy, sGraphDestroy, PFN_cuGraphDestroy_v10000);
    return fromDriver(destroy(graph));
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec)
{
    EAGLE_DRV(destroy, sGraphExecDestroy, PFN_cuGraphExecDestroy_v10000);
    return fromDriver(destroy(exec));
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags)
{
    EAGLE_CTX();
    EAGLE_DRV(instantiate, sGraphInstantiateWithFlags, PFN_cuGraphInstantiateWithFlags_v11040);
    return fromDriver(instantiate(exec, graph, flags));
}

static cudaError_t graphLaunch(cudaGraphExec_t exec, CUstream s)
{
    EAGLE_CTX();
    EAGLE_DRV(launchGraph, sGraphLaunch, PFN_cuGraphLaunch_v10000);
    return fromDriver(launchGraph(exec, s));
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) { return graphLaunch(exec, legacy(stream)); }
cudaError_t cudaGraphLaunch_ptsz(cudaGraphExec_t exec, cudaStream_t stream)
{
    return graphLaunch(exec, perThread(stream));
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* n)
{
    EAGLE_DRV(getNodes, sGraphGetNodes, PFN_cuGraphGetNodes_v10000);
    return fromDriver(getNodes(graph, nodes, n));
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type)
{
    EAGLE_DRV(getType, sGraphNodeGetType, PFN_cuGraphNodeGetType_v10000);
    CUgraphNodeType t{};
    const CUresult r = getType(node, &t);
    *type = static_cast<cudaGraphNodeType>(t);
    return fromDriver(r);
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
    std::size_t n)
{
    EAGLE_DRV(add, sGraphAddEmptyNode, PFN_cuGraphAddEmptyNode_v10000);
    return fromDriver(add(node, graph, deps, n));
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
    std::size_t n, cudaGraph_t child)
{
    EAGLE_DRV(add, sGraphAddChildGraphNode, PFN_cuGraphAddChildGraphNode_v10000);
    return fromDriver(add(node, graph, deps, n, child));
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* graph)
{
    EAGLE_DRV(get, sGraphChildGraphNodeGetGraph, PFN_cuGraphChildGraphNodeGetGraph_v10000);
    return fromDriver(get(node, graph));
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
    std::size_t n, const cudaKernelNodeParams* params)
{
    EAGLE_CTX();
    EAGLE_DRV(add, sGraphAddKernelNode, PFN_cuGraphAddKernelNode_v12000);
    CUfunction f = nullptr;
    if (cudaError_t e = resolveFunc(params->func, &f); e != cudaSuccess)
        return e;
    const CUDA_KERNEL_NODE_PARAMS_v2 k = driverParams(*params, f);
    return fromDriver(add(node, graph, deps, n, &k));
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* params)
{
    EAGLE_CTX();
    EAGLE_DRV(get, sGraphKernelNodeGetParams, PFN_cuGraphKernelNodeGetParams_v12000);
    CUDA_KERNEL_NODE_PARAMS_v2 k{};
    if (CUresult r = get(node, &k); r != CUDA_SUCCESS)
        return fromDriver(r);
    CUfunction f = k.func;
    if (f == nullptr && k.kern != nullptr) {
        EAGLE_DRV(kernelFunction, sKernelGetFunction, PFN_cuKernelGetFunction_v12000);
        if (CUresult r = kernelFunction(&f, k.kern); r != CUDA_SUCCESS)
            return fromDriver(r);
    }
    params->func = stubOf(f);
    params->gridDim = dim3(k.gridDimX, k.gridDimY, k.gridDimZ);
    params->blockDim = dim3(k.blockDimX, k.blockDimY, k.blockDimZ);
    params->sharedMemBytes = k.sharedMemBytes;
    params->kernelParams = k.kernelParams;
    params->extra = k.extra;
    return cudaSuccess;
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec, cudaGraphNode_t node,
    const cudaKernelNodeParams* params)
{
    EAGLE_CTX();
    EAGLE_DRV(set, sGraphExecKernelNodeSetParams, PFN_cuGraphExecKernelNodeSetParams_v12000);
    CUfunction f = nullptr;
    if (cudaError_t e = resolveFunc(params->func, &f); e != cudaSuccess)
        return e;
    const CUDA_KERNEL_NODE_PARAMS_v2 k = driverParams(*params, f);
    return fromDriver(set(exec, node, &k));
}

cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t exec, cudaGraphNode_t node, unsigned int enabled)
{
    EAGLE_DRV(set, sGraphNodeSetEnabled, PFN_cuGraphNodeSetEnabled_v11060);
    return fromDriver(set(exec, node, enabled));
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags)
{
    EAGLE_DRV(print, sGraphDebugDotPrint, PFN_cuGraphDebugDotPrint_v11030);
    return fromDriver(print(graph, path, flags));
}

cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* handle, cudaGraph_t graph,
    unsigned int defaultLaunchValue, unsigned int flags)
{
    EAGLE_CTX();
    EAGLE_DRV(create, sGraphConditionalHandleCreate, PFN_cuGraphConditionalHandleCreate_v12030);
    CUgraphConditionalHandle h = 0;
    const CUresult r = create(&h, graph, currentContext(), defaultLaunchValue, flags);
    if (r == CUDA_SUCCESS)
        *handle = static_cast<cudaGraphConditionalHandle>(h);
    return fromDriver(r);
}

cudaError_t cudaGraphAddNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps, EAGLE_EDGES_ARG
    std::size_t n, cudaGraphNodeParams* params)
{
    EAGLE_NO_EDGES()
    EAGLE_CTX();
    EAGLE_DRV(add, sGraphAddNode, PFN_cuGraphAddNode_v12020);
    CUgraphNodeParams p{};
    p.type = static_cast<CUgraphNodeType>(params->type);
    switch (params->type) {
    case cudaGraphNodeTypeConditional:
        p.conditional.handle = params->conditional.handle;
        p.conditional.type = static_cast<CUgraphConditionalNodeType>(params->conditional.type);
        p.conditional.size = params->conditional.size;
        p.conditional.phGraph_out = params->conditional.phGraph_out;
        p.conditional.ctx = currentContext();
        break;
    case cudaGraphNodeTypeKernel: {
        CUfunction f = nullptr;
        if (cudaError_t e = resolveFunc(params->kernel.func, &f); e != cudaSuccess)
            return e;
        p.kernel.func = f;
        p.kernel.gridDimX = params->kernel.gridDim.x;
        p.kernel.gridDimY = params->kernel.gridDim.y;
        p.kernel.gridDimZ = params->kernel.gridDim.z;
        p.kernel.blockDimX = params->kernel.blockDim.x;
        p.kernel.blockDimY = params->kernel.blockDim.y;
        p.kernel.blockDimZ = params->kernel.blockDim.z;
        p.kernel.sharedMemBytes = params->kernel.sharedMemBytes;
        p.kernel.kernelParams = params->kernel.kernelParams;
        p.kernel.extra = params->kernel.extra;
        break;
    }
    case cudaGraphNodeTypeEmpty:
        break;
    default:
        return record(cudaErrorNotSupported);
    }
    const CUresult r = add(node, graph, deps, n, &p);
    // The driver hands a conditional node's body graphs back through the
    // parameters (a driver-owned array): pass them on to the caller's.
    if (r == CUDA_SUCCESS && params->type == cudaGraphNodeTypeConditional)
        params->conditional.phGraph_out = p.conditional.phGraph_out;
    return fromDriver(r);
}

} // extern "C"
