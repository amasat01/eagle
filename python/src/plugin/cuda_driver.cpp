// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cuda_driver.cpp
 * @brief The CUDA runtime functions ``libeagle_cuda.so`` calls that the shared aether driver-runtime shim
 *        (``aether/backend/cuda/DriverRuntime.inl``) does not provide: graphs, capture, stream and event
 *        creation, host functions, device attributes and properties, error names, the CUDA 13 kernel handle.
 *
 * The shim carries the driver (``libcuda.so.1`` resolved entry by entry through ``cuGetProcAddress``), the
 * primary-context rules, the error state, memory, the nvcc kernel-registration hooks and the plain and
 * ``_ptsz`` spellings of the stream-ordered calls; this file uses its ``Slots``, ``need``, ``AETHER_DRV`` and
 * ``AETHER_CTX``, so there is ONE context and error state per process. ``libcuda.so.1`` is the only library the
 * plugin needs; ``driverGroups`` clears every capability group whose driver entry point is absent (a name listed in
 * ``EAGLE_CUDA_DRIVER_HIDE`` counts as absent). Everything is hidden by the version script, so nothing here
 * interposes another library's CUDA runtime in the same process.
 */

// The plugin TU is compiled with per-thread default-stream semantics; this one defines BOTH spellings explicitly.
#undef CUDA_API_PER_THREAD_DEFAULT_STREAM

#define AETHER_DRIVER_HIDE_ENV "EAGLE_CUDA_DRIVER_HIDE"
#include "aether/backend/cuda/DriverRuntime.inl"

#include <cuda_runtime_api.h>

#include "plugin/cuda_driver.h"
#include "seam/eagle_backend.h"

namespace {

constexpr std::uint64_t kCore = ~UINT64_C(0); // needed by every group
constexpr std::uint64_t STREAM = EAGLE_BACKEND_CAP_STREAM, CAPTURER = EAGLE_BACKEND_CAP_CAPTURER,
                        CAPTURED = EAGLE_BACKEND_CAP_CAPTURED, FORK = EAGLE_BACKEND_CAP_FORK,
                        COND = EAGLE_BACKEND_CAP_CONDITIONAL, ATTR = EAGLE_BACKEND_CAP_ATTRIBUTION,
                        GRAPH = EAGLE_BACKEND_CAP_GRAPH, LAUNCHER = EAGLE_BACKEND_CAP_LAUNCHER,
                        EXEC = EAGLE_BACKEND_CAP_EXEC, DEVICE = EAGLE_BACKEND_CAP_DEVICE,
                        INTEROP = EAGLE_BACKEND_CAP_INTEROP, FILTER = EAGLE_BACKEND_CAP_FILTERING;

/** The capability groups using one driver entry point (entries no group needs are not listed). */
struct Use {
    const char* name;
    std::uint64_t groups;
};
constexpr Use kUses[] = {
    { "cuInit", kCore },
    { "cuDeviceGetCount", kCore },
    { "cuDeviceGet", kCore },
    { "cuDevicePrimaryCtxRetain", kCore },
    { "cuCtxGetCurrent", kCore },
    { "cuCtxSetCurrent", kCore },
    { "cuCtxGetDevice", kCore },
    { "cuGetErrorName", kCore },
    { "cuGetErrorString", kCore },
    { "cuStreamGetCtx", kCore },
    { "cuDeviceGetAttribute", DEVICE | EXEC },
    { "cuDeviceGetName", DEVICE },
    { "cuDeviceTotalMem", DEVICE },
    { "cuModuleLoadData", COND | FILTER },
    { "cuModuleGetFunction", COND | FILTER },
    { "cuLaunchKernel", COND | EXEC | FILTER },
    { "cuFuncGetAttribute", COND | EXEC | FILTER },
    { "cuStreamCreate", STREAM | FORK | COND },
    { "cuStreamDestroy", STREAM | FORK | COND },
    { "cuStreamSynchronize", STREAM | GRAPH | LAUNCHER },
    { "cuStreamWaitEvent", FORK | INTEROP },
    { "cuStreamBeginCapture", CAPTURER },
    { "cuStreamEndCapture", CAPTURER | COND },
    { "cuStreamBeginCaptureToGraph", COND },
    { "cuStreamGetCaptureInfo", COND | ATTR },
    { "cuStreamUpdateCaptureDependencies", COND },
    { "cuEventCreate", FORK | INTEROP },
    { "cuEventDestroy", FORK | INTEROP },
    { "cuEventRecord", FORK | INTEROP },
    { "cuGraphCreate", GRAPH | COND },
    { "cuGraphDestroy", CAPTURER | CAPTURED | GRAPH | COND },
    { "cuGraphExecDestroy", LAUNCHER },
    { "cuGraphInstantiateWithFlags", LAUNCHER | GRAPH },
    { "cuGraphLaunch", LAUNCHER },
    { "cuGraphGetNodes", GRAPH | COND | ATTR },
    { "cuGraphNodeGetType", GRAPH | COND | ATTR },
    { "cuGraphAddKernelNode", GRAPH | COND },
    { "cuGraphKernelNodeGetParams", GRAPH },
    { "cuGraphExecKernelNodeSetParams", LAUNCHER },
    { "cuGraphAddEmptyNode", GRAPH },
    { "cuGraphAddChildGraphNode", GRAPH | COND },
    { "cuGraphChildGraphNodeGetGraph", GRAPH },
    { "cuGraphNodeSetEnabled", LAUNCHER | ATTR },
    { "cuGraphConditionalHandleCreate", COND },
    { "cuGraphAddNode", COND },
};

/** Is driver entry point ``name`` resolved? */
bool present(const char* name)
{
    const Slots& d = driver();
#define EAGLE_PRESENT(entry, ver)                                                                          \
    if (std::strcmp(name, #entry) == 0)                                                                    \
        return d.entry != nullptr;
    AETHER_DRIVER_ENTRIES(EAGLE_PRESENT)
#undef EAGLE_PRESENT
    return false;
}

CUcontext currentContext()
{
    CUcontext ctx = nullptr;
    if (driver().cuCtxGetCurrent)
        driver().cuCtxGetCurrent(&ctx);
    return ctx;
}

CUstream legacy(cudaStream_t s) { return reinterpret_cast<CUstream>(s); }
CUstream perThread(cudaStream_t s) { return reinterpret_cast<CUstream>(perThreadStream(s)); }

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
    for (const Use& u : kUses)
        if (!present(u.name))
            missing |= u.groups;
    return wanted & ~missing;
}

int makeContextCurrent() { return static_cast<int>(aether_driver_runtime::ensureContext()); }

} // namespace eagle_cuda_plugin

// =============================================================================
// The CUDA runtime entry points the plugin references that the shim lacks. Hidden (the version script keeps
// them local to libeagle_cuda.so).
// =============================================================================

extern "C" {

// ---- nvcc registration hooks -------------------------------------------------

// ---- errors ------------------------------------------------------------------

const char* cudaGetErrorName(cudaError_t error)
{
    return runtimeName(error);
}

// ---- devices and contexts ----------------------------------------------------

cudaError_t cudaDriverGetVersion(int* version)
{
    AETHER_DRV(get, cuDriverGetVersion);
    return fromDriver(get(version));
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
{
    AETHER_DRV(getAttribute, cuDeviceGetAttribute);
    AETHER_DRV(get, cuDeviceGet);
    CUdevice dev = 0;
    if (get(&dev, device) != CUDA_SUCCESS)
        return record(cudaErrorInvalidDevice);
    return fromDriver(getAttribute(value, static_cast<CUdevice_attribute>(attr), dev));
}

cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device)
{
    AETHER_DRV(getAttribute, cuDeviceGetAttribute);
    AETHER_DRV(getName, cuDeviceGetName);
    AETHER_DRV(totalMem, cuDeviceTotalMem);
    AETHER_DRV(get, cuDeviceGet);
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

static cudaError_t launchHostFunc(CUstream s, cudaHostFn_t hostFn, void* arg)
{
    AETHER_CTX();
    AETHER_DRV(launch, cuLaunchHostFunc);
    return fromDriver(launch(s, reinterpret_cast<CUhostFn>(hostFn), arg));
}
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t hostFn, void* arg)
{
    return launchHostFunc(legacy(stream), hostFn, arg);
}
cudaError_t cudaLaunchHostFunc_ptsz(cudaStream_t stream, cudaHostFn_t hostFn, void* arg)
{
    return launchHostFunc(perThread(stream), hostFn, arg);
}

static cudaError_t entryPoint(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags,
    cudaDriverEntryPointQueryResult* status)
{
    if (cudaError_t e = initDriver(); e != cudaSuccess)
        return e;
    const auto get2 = driver().getProc;
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
    AETHER_CTX();
    AETHER_DRV(create, cuStreamCreate);
    return fromDriver(create(reinterpret_cast<CUstream*>(stream), flags));
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    AETHER_DRV(destroy, cuStreamDestroy);
    return fromDriver(destroy(legacy(stream)));
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
    AETHER_DRV(elapsed, cuEventElapsedTime);
    return fromDriver(elapsed(ms, reinterpret_cast<CUevent>(start), reinterpret_cast<CUevent>(end)));
}

// ---- stream capture ----------------------------------------------------------

static cudaError_t beginCapture(CUstream s, cudaStreamCaptureMode mode)
{
    AETHER_CTX();
    AETHER_DRV(begin, cuStreamBeginCapture);
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
    AETHER_CTX();
    AETHER_DRV(begin, cuStreamBeginCaptureToGraph);
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
    AETHER_CTX();
    AETHER_DRV(end, cuStreamEndCapture);
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
    AETHER_CTX();
    AETHER_DRV(info, cuStreamGetCaptureInfo);
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
    AETHER_CTX();
    AETHER_DRV(update, cuStreamUpdateCaptureDependencies);
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
    return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, args, shared, stream);
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid, dim3 block, void** args, std::size_t shared,
    cudaStream_t stream)
{
    return cudaLaunchKernel_ptsz(reinterpret_cast<const void*>(kernel), grid, block, args, shared, stream);
}
#endif

// ---- graphs ------------------------------------------------------------------

cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags)
{
    AETHER_CTX();
    AETHER_DRV(create, cuGraphCreate);
    return fromDriver(create(graph, flags));
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    AETHER_DRV(destroy, cuGraphDestroy);
    return fromDriver(destroy(graph));
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec)
{
    AETHER_DRV(destroy, cuGraphExecDestroy);
    return fromDriver(destroy(exec));
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags)
{
    AETHER_CTX();
    AETHER_DRV(instantiate, cuGraphInstantiateWithFlags);
    return fromDriver(instantiate(exec, graph, flags));
}

static cudaError_t graphLaunch(cudaGraphExec_t exec, CUstream s)
{
    AETHER_CTX();
    AETHER_DRV(launchGraph, cuGraphLaunch);
    return fromDriver(launchGraph(exec, s));
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) { return graphLaunch(exec, legacy(stream)); }
cudaError_t cudaGraphLaunch_ptsz(cudaGraphExec_t exec, cudaStream_t stream)
{
    return graphLaunch(exec, perThread(stream));
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* n)
{
    AETHER_DRV(getNodes, cuGraphGetNodes);
    return fromDriver(getNodes(graph, nodes, n));
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type)
{
    AETHER_DRV(getType, cuGraphNodeGetType);
    CUgraphNodeType t{};
    const CUresult r = getType(node, &t);
    *type = static_cast<cudaGraphNodeType>(t);
    return fromDriver(r);
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
    std::size_t n)
{
    AETHER_DRV(add, cuGraphAddEmptyNode);
    return fromDriver(add(node, graph, deps, n));
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
    std::size_t n, cudaGraph_t child)
{
    AETHER_DRV(add, cuGraphAddChildGraphNode);
    return fromDriver(add(node, graph, deps, n, child));
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* graph)
{
    AETHER_DRV(get, cuGraphChildGraphNodeGetGraph);
    return fromDriver(get(node, graph));
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
    std::size_t n, const cudaKernelNodeParams* params)
{
    AETHER_CTX();
    AETHER_DRV(add, cuGraphAddKernelNode);
    CUfunction f = nullptr;
    if (cudaError_t e = resolveFunc(params->func, &f); e != cudaSuccess)
        return e;
    const CUDA_KERNEL_NODE_PARAMS_v2 k = driverParams(*params, f);
    return fromDriver(add(node, graph, deps, n, &k));
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* params)
{
    AETHER_CTX();
    AETHER_DRV(get, cuGraphKernelNodeGetParams);
    CUDA_KERNEL_NODE_PARAMS_v2 k{};
    if (CUresult r = get(node, &k); r != CUDA_SUCCESS)
        return fromDriver(r);
    CUfunction f = k.func;
    if (f == nullptr && k.kern != nullptr) {
        AETHER_DRV(kernelFunction, cuKernelGetFunction);
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
    AETHER_CTX();
    AETHER_DRV(set, cuGraphExecKernelNodeSetParams);
    CUfunction f = nullptr;
    if (cudaError_t e = resolveFunc(params->func, &f); e != cudaSuccess)
        return e;
    const CUDA_KERNEL_NODE_PARAMS_v2 k = driverParams(*params, f);
    return fromDriver(set(exec, node, &k));
}

cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t exec, cudaGraphNode_t node, unsigned int enabled)
{
    AETHER_DRV(set, cuGraphNodeSetEnabled);
    return fromDriver(set(exec, node, enabled));
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags)
{
    AETHER_DRV(print, cuGraphDebugDotPrint);
    return fromDriver(print(graph, path, flags));
}

cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* handle, cudaGraph_t graph,
    unsigned int defaultLaunchValue, unsigned int flags)
{
    AETHER_CTX();
    AETHER_DRV(create, cuGraphConditionalHandleCreate);
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
    AETHER_CTX();
    AETHER_DRV(add, cuGraphAddNode);
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
