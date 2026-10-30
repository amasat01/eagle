// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/Graph.h"

#include <string>
#include <unordered_map>
#include <vector>

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/**
 * @brief Instantiated CUDA graph executor.
 *
 * Holds a ``cudaGraphExec_t`` (instantiated from a ``cudaGraph_t``)
 * and provides ``launch()`` / ``synchronize()`` / ``setLogicalSize()``
 * for repeated execution on the associated stream. Constructed by
 * ``Graph::launcher()`` only.
 *
 * Lifetime: shares ownership of the producing ``Graph::Storage`` via
 * ``std::shared_ptr<Graph::Storage> storage_``. The source graph and
 * every captured-child original it owns stay alive for as long as
 * any ``Launcher`` (or the producing ``Graph``) holds a reference.
 * That guarantee keeps the kernel-node handles and ``kernelParams``
 * pointers in ``kernelNodes_`` valid for any subsequent
 * ``setLogicalSize`` call.
 *
 * Multi-launcher fan-out: any number of Launchers can share the same
 * ``Storage``. Each instantiates its own independent
 * ``cudaGraphExec_t``.
 *
 * Copy construction and copy assignment are disabled; use move semantics.
 *
 * @note Available only when ``EAGLE_CPU_ONLY`` is not defined.
 * @see Graph — builds the graph and produces a Launcher.
 */
class Launcher {
    using Self = Launcher;
    friend class Graph;

public:
    /** @brief Default constructor: empty Launcher (no source storage,
     *  no exec instance). Useful as a member that gets move-assigned
     *  later. */
    Launcher() = default;

    /** @brief Copy constructor is forbidden */
    Launcher(Launcher& other)       = delete;
    Launcher(const Launcher& other) = delete;

    /** @brief Move constructor */
    Launcher(Launcher&& other) noexcept
        : storage_{ std::move(other.storage_) }
        , instance_{ other.instance_ }
        , instantiated_{ other.instantiated_ }
        , stream_{ other.stream_ }
        , kernelNodes_{ std::move(other.kernelNodes_) }
        , kernelSnapshots_{ std::move(other.kernelSnapshots_) }
    {
        other.instance_     = nullptr;
        other.instantiated_ = false;
        other.stream_       = 0;
    }

    /** @brief Copy assignment is forbidden */
    Self& operator=(Launcher& other)       = delete;
    Self& operator=(const Launcher& other) = delete;

    /** @brief Move assignment */
    Self& operator=(Launcher&& other) noexcept
    {
        /* Free our exec instance before taking over. The shared_ptr
         * release below decrements our previous Storage's ref-count;
         * if we held the last reference the Storage destructor runs
         * here and tears down the source CUDA graph. */
        destroyInstance_();

        storage_      = std::move(other.storage_);
        instance_     = other.instance_;
        instantiated_ = other.instantiated_;
        stream_       = other.stream_;
        kernelNodes_  = std::move(other.kernelNodes_);
        kernelSnapshots_ = std::move(other.kernelSnapshots_);

        other.instance_     = nullptr;
        other.instantiated_ = false;
        other.stream_       = 0;
        return *this;
    }

    /** @brief Update the stream */
    Self& stream(const cudaStream_t& stream)
    {
        stream_ = stream;
        return *this;
    }

    /** @brief Expose the stream */
    const cudaStream_t& stream() const { return stream_; }

    /** @brief Launch the graph */
    void launch()
    {
        EAGLE_KERNEL_PRE();
        /* Measured (5 reps x 200k iters, warmed up, P2000/CUDA
         * 12.6) -- the launch+sync round-trip is ~4.5-4.6 us; the delta
         * between EAGLE_CHECK and EAGLE_CHECK_ALWAYS here was 5-42 ns
         * (<1%), smaller than the ~80 ns (~1.8%) run-to-run measurement
         * noise itself (one rep even measured negative). Lost in the
         * noise -- flipped rather than left silent. */
        EAGLE_CHECK_ALWAYS(cudaGraphLaunch(instance_, stream_));
        EAGLE_KERNEL_POST();
    }

    /** @brief Synchronize */
    inline void synchronize() { EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(stream_)); }

    /** @brief Patch each registered kernel node's launch params for
     *  the new ``logicalSize``.
     *
     *  Per-record: ``idealBlockSize == 0`` keeps capture-time blockDim
     *  and shrinks only ``gridDim`` (required for fixed-layout
     *  reduction-tree kernels); ``> 0`` runs ``computeBlocks`` to
     *  re-tune both. Microsecond-class — no re-capture.
     *
     *  Never calls ``cudaGraphKernelNodeGetParams`` at runtime: the
     *  exec graph round-trip rejects templated kernels with
     *  ``InvalidDeviceFunction``. Snapshotted ``capturedParams`` is
     *  copied and mutated instead.
     *
     *  Lifetime: walks our own ``kernelNodes_`` snapshot. Both the
     *  node handles and the ``kernelParams`` pointers reference
     *  internals of ``storage_->graph``; safe because ``storage_``
     *  keeps Storage alive for our lifetime.
     */
    void setLogicalSize(idx_t logicalSize)
    {
        EAGLE_ASSERT(instantiated_,
            "Launcher::setLogicalSize requires an instantiated graph");

        /* Skip altogether if logicalSize is 0 */
        if (logicalSize == 0)
            return;

        for (const KernelNodeRecord& rec : kernelNodes_) {
            /* Layout-locked kernels (e.g. scan-tree recursion levels)
             * advertise ``kFixedSize`` to opt out of patching entirely:
             * their grid/block were sized at capture for a buffer
             * smaller than the launcher's logicalSize, and re-deriving
             * gridDim from logicalSize would over-write internal
             * fixed-size scratch (verified to corrupt
             * ``Slice::scanGraph`` recursion's per-block sums). */
            if (rec.idealBlockSize == kFixedSize)
                continue;

            idx_t newBlock;
            idx_t newGrid;
            if (rec.idealBlockSize == 0) {
                newBlock = rec.capturedParams.blockDim.x;
#ifndef EAGLE_CPU_ONLY
                if (rec.driverLaunched)
                    newBlock = rec.driverParams.blockDimX;
#endif
                newGrid  = (logicalSize + newBlock - 1) / newBlock;
            } else {
                computeBlocks(
                    logicalSize, newGrid, newBlock, rec.idealBlockSize);
            }
#ifndef EAGLE_CPU_ONLY
            if (rec.driverLaunched) {
                patchDriverNode_(rec, newGrid, newBlock);
                continue;
            }
#endif
            cudaKernelNodeParams params = rec.capturedParams;
            params.blockDim = dim3(newBlock, 1, 1);
            params.gridDim  = dim3(newGrid, 1, 1);
            EAGLE_CHECK_ALWAYS(
                cudaGraphExecKernelNodeSetParams(instance_, rec.node, &params));
        }
    }


    /** @brief Patch what kernel node @p node reads, in this exec only.
     *
     *  Argument ``i < n`` takes the value stored at ``params[i]``
     *  (``nullptr`` keeps the captured value); arguments past @p n keep
     *  theirs (the partition tail). Function, grid, block and shared memory
     *  stay as captured: the node's driver parameters are snapshotted once,
     *  on first patch, from the source graph (the driver call, so a
     *  driver-launched ``CUfunction`` node is accepted). The values are
     *  copied at the call. @p node may live in the root graph or in any
     *  conditional body graph. Microsecond-class -- no re-capture. */
    void setKernelParams(cudaGraphNode_t node, void* const* params, std::size_t n)
    {
        EAGLE_ASSERT(instantiated_,
            "Launcher::setKernelParams requires an instantiated graph");
        static const auto set = detail::driver_entry<decltype(&::cuGraphExecKernelNodeSetParams)>(
            "cuGraphExecKernelNodeSetParams");
        const KernelSnapshot& snap = kernelSnapshot_(node);
        if (n > snap.args.size())
            throw std::invalid_argument("eagle: Launcher::setKernelParams got "
                + std::to_string(n) + " arguments for a kernel taking "
                + std::to_string(snap.args.size()) + ".");
        std::vector<void*> args(snap.args);
        for (std::size_t i = 0; i < n; ++i)
            if (params[i] != nullptr)
                args[i] = params[i];
        CUDA_KERNEL_NODE_PARAMS p = snap.params;
        p.kernelParams = args.data();
        p.extra        = nullptr;
        driverCheck_(set(reinterpret_cast<CUgraphExec>(instance_),
            reinterpret_cast<CUgraphNode>(node), &p), "patching a kernel node's arguments");
    }

    /** @brief Re-point captured memcpy node @p node at source @p src, in this
     *  exec only. Size, destination and the source's memory type stay as
     *  captured (the runtime's 1-D setter would re-derive the kind and
     *  refuses a device-to-device one). */
    void setMemcpySource(cudaGraphNode_t node, const void* src)
    {
        EAGLE_ASSERT(instantiated_,
            "Launcher::setMemcpySource requires an instantiated graph");
        static const auto get = detail::driver_entry<decltype(&::cuGraphMemcpyNodeGetParams)>(
            "cuGraphMemcpyNodeGetParams");
        static const auto set = detail::driver_entry<decltype(&::cuGraphExecMemcpyNodeSetParams)>(
            "cuGraphExecMemcpyNodeSetParams");
        static const auto current = detail::driver_entry<decltype(&::cuCtxGetCurrent)>("cuCtxGetCurrent");
        CUDA_MEMCPY3D p{};
        driverCheck_(get(reinterpret_cast<CUgraphNode>(node), &p), "reading a memcpy node");
        if (p.srcMemoryType == CU_MEMORYTYPE_HOST)
            p.srcHost = src;
        else if (p.srcMemoryType == CU_MEMORYTYPE_DEVICE || p.srcMemoryType == CU_MEMORYTYPE_UNIFIED)
            p.srcDevice = reinterpret_cast<CUdeviceptr>(src);
        else
            throw std::invalid_argument("eagle: Launcher::setMemcpySource: the node copies from an array.");
        CUcontext ctx = nullptr;
        driverCheck_(current(&ctx), "reading the current context");
        driverCheck_(set(reinterpret_cast<CUgraphExec>(instance_), reinterpret_cast<CUgraphNode>(node), &p, ctx),
            "re-pointing a memcpy node");
    }

    /** @brief Update this exec in place from @p sameTopology (a re-capture of
     *  the same shape): the fallback when a patch is refused. Returns false
     *  when the driver rejects the update (the exec is unchanged: rebuild).
     *  Node handles of the original graph keep addressing the exec. */
    bool tryExecUpdate(cudaGraph_t sameTopology)
    {
        EAGLE_ASSERT(instantiated_,
            "Launcher::tryExecUpdate requires an instantiated graph");
        static const auto update = detail::driver_entry<decltype(&::cuGraphExecUpdate)>("cuGraphExecUpdate");
        CUgraphExecUpdateResultInfo info{};
        const CUresult r = update(reinterpret_cast<CUgraphExec>(instance_),
            reinterpret_cast<CUgraph>(sameTopology), &info);
        if (r == CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE)
            return false;
        driverCheck_(r, "updating the exec graph");
        return true;
    }

    /** @brief Number of registered kernel records (for tests / debug). */
    idx_t kernelNodeCount() const
    {
        return static_cast<idx_t>(kernelNodes_.size());
    }

    /** @brief Read-only accessor to the registered kernel records
     *  (tests / diagnostics). */
    const std::vector<KernelNodeRecord>& kernelNodes() const
    {
        return kernelNodes_;
    }

    /** @brief Destructor — destroys the exec instance.
     *
     *  ``storage_`` (the shared ``Graph::Storage`` reference) is
     *  released after; if we hold the last reference the Storage
     *  destructor runs here and tears down the source CUDA graph. */
    ~Launcher() { destroyInstance_(); }

    /** @brief Expose the source graph (lifetime tied to ``storage_``). */
    cudaGraph_t graph() const
    {
        return storage_ ? storage_->graph : nullptr;
    }

    /** @brief Raw ``cudaGraphExec_t`` handle of the instantiated exec
     *  graph -- escape hatch for facilities (member-enable
     *  toggling, ``eagle::cuda::setNodeEnabled`` in
     *  ``CaptureAttribution.h``) that must call ``cudaGraphExec*`` APIs
     *  directly, mirroring ``Graph::nodeHandle()``'s escape-hatch
     *  precedent. A bare accessor -- no CUDA API call here, so this adds
     *  no new version floor to this shared header. */
    cudaGraphExec_t execHandle() const { return instance_; }

private:
#ifndef EAGLE_CPU_ONLY
    /** @brief A kernel node's driver parameters and argument pointers, as
     *  captured (the pointers address the source graph's own storage, alive
     *  with ``storage_``). */
    struct KernelSnapshot {
        CUDA_KERNEL_NODE_PARAMS params;
        std::vector<void*> args;
    };

    const KernelSnapshot& kernelSnapshot_(cudaGraphNode_t node)
    {
        if (auto it = kernelSnapshots_.find(node); it != kernelSnapshots_.end())
            return it->second;
        static const auto get = detail::driver_entry<decltype(&::cuGraphKernelNodeGetParams)>(
            "cuGraphKernelNodeGetParams");
        KernelSnapshot snap{};
        driverCheck_(get(reinterpret_cast<CUgraphNode>(node), &snap.params), "reading a kernel node");
        CUfunction f = snap.params.func;
        if (f == nullptr) {
            static const auto fromKernel = detail::driver_entry<decltype(&::cuKernelGetFunction)>(
                "cuKernelGetFunction");
            driverCheck_(fromKernel(&f, snap.params.kern), "resolving a kernel node's function");
        }
        const std::size_t count = paramCount_(f);
        auto** captured = static_cast<void**>(snap.params.kernelParams);
        snap.args.assign(captured, captured + count);
        return kernelSnapshots_.emplace(node, std::move(snap)).first->second;
    }

    /** @brief Number of parameters @p f takes (``cuFuncGetParamInfo``, a
     *  CUDA 12.4 driver entry, asked for at its own ABI). */
    static std::size_t paramCount_(CUfunction f)
    {
        using Info = CUresult (*)(CUfunction, std::size_t, std::size_t*, std::size_t*);
        static const Info info = [] {
            void* fn = nullptr;
#if CUDART_VERSION >= 12050
            cudaDriverEntryPointQueryResult found{};
            if (cudaGetDriverEntryPointByVersion("cuFuncGetParamInfo", &fn, 12040,
                    cudaEnableDefault, &found) != cudaSuccess
                || found != cudaDriverEntryPointSuccess)
                fn = nullptr;
#endif
            return reinterpret_cast<Info>(fn);
        }();
        if (info == nullptr)
            throw std::runtime_error("eagle: patching kernel arguments needs an NVIDIA driver "
                ">= 12.4 and a CUDA >= 12.5 build (cuFuncGetParamInfo).");
        std::size_t n = 0, offset = 0, size = 0;
        while (info(f, n, &offset, &size) == CUDA_SUCCESS)
            ++n;
        return n;
    }

    static void driverCheck_(CUresult r, const char* what)
    {
        if (r != CUDA_SUCCESS)
            throw std::runtime_error(std::string("eagle: ") + what + " failed ("
                + detail::driver_error_text(r) + ").");
    }

    /** @brief Patch a driver-launched kernel node's grid and block through the
     *  driver API (the runtime refuses its ``CUfunction``). */
    void patchDriverNode_(const KernelNodeRecord& rec, idx_t grid, idx_t block)
    {
        static const auto set = detail::driver_entry<decltype(&::cuGraphExecKernelNodeSetParams)>(
            "cuGraphExecKernelNodeSetParams");
        CUDA_KERNEL_NODE_PARAMS params = rec.driverParams;
        params.gridDimX  = static_cast<unsigned int>(grid);
        params.gridDimY  = 1;
        params.gridDimZ  = 1;
        params.blockDimX = static_cast<unsigned int>(block);
        params.blockDimY = 1;
        params.blockDimZ = 1;
        if (const CUresult r = set(reinterpret_cast<CUgraphExec>(instance_),
                reinterpret_cast<CUgraphNode>(rec.node), &params); r != CUDA_SUCCESS)
            throw std::runtime_error(std::string("eagle: patching a driver-launched kernel node failed (")
                + detail::driver_error_text(r) + ").");
    }
#endif

protected:
    /** @brief Construct from shared Storage and a kernelNodes_
     *  snapshot. Called only by ``Graph::launcher()``. */
    explicit Launcher(const cudaStream_t& stream,
        std::shared_ptr<Graph::Storage> storage,
        std::vector<KernelNodeRecord> kernelNodes)
        : storage_{ std::move(storage) }
        , stream_{ stream }
        , kernelNodes_{ std::move(kernelNodes) }
    {
        instantiate_();
    }

    /** @brief What an out-of-memory failure needs: the largest local memory per thread
     *  among this graph's driver-launched kernels, and free/total GPU memory now. */
    std::string oomContext_() const
    {
        std::string kernels;
        int widest = 0;
#ifndef EAGLE_CPU_ONLY
        static const auto attr = detail::driver_entry<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
        for (const auto& rec : kernelNodes_) {
            int local = 0;     // eagle's kernels are driver-launched (a plugin's cuLaunchKernel)
            if (rec.driverLaunched && rec.driverParams.func != nullptr &&
                attr(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, rec.driverParams.func) != CUDA_SUCCESS)
                local = 0;
            widest = std::max(widest, local);
        }
#endif
        kernels = std::to_string(kernelNodes_.size()) + " kernel node(s), largest local memory "
            + std::to_string(widest) + " bytes per thread; ";
        return kernels + detail::oom_context(nullptr);
    }

    /** @brief Create the instance */
    void instantiate_()
    {
        if (!instantiated_) {
            EAGLE_ASSERT(storage_ && storage_->graph != nullptr,
                "Cannot instantiate a non-created graph");
            const cudaError_t made = cudaGraphInstantiate(
                &instance_, storage_->graph, NULL, NULL, 0);
            if (made == cudaErrorMemoryAllocation)
                ::eagle::err::detail::throwAt<::aether::Error>(__FILE__, __LINE__, __func__,
                    "cudaErrorMemoryAllocation (2): out of memory at cudaGraphInstantiate; "
                    + oomContext_());
            EAGLE_CHECK_ALWAYS(made);
            instantiated_ = true;
        }
    }

    /** @brief Destroy the instance.
     *
     *  A ``Launcher`` that outlives its CUDA context segfaults on teardown:
     *  a ``Launcher`` reachable only from Python (e.g. ``GraphPipeline``'s
     *  ``self._launcher``) becomes garbage at a time entirely governed by
     *  Python's GC -- which can land THIS destructor's ``cudaGraphExecDestroy``
     *  call in the middle of an unrelated, LATER, still-active capture
     *  elsewhere in the process. That call always succeeded at swallowing
     *  ITS OWN error (``EAGLE_CHECK_NOTHROW``, correct and unchanged by this
     *  fix), but the CUDA call itself still ran and invalidated the ambient
     *  capture (``cudaErrorStreamCaptureInvalidated``) -- the next launch on
     *  it then failed, and the poisoned native state crashed at a later,
     *  unrelated GC or interpreter exit. Fix: defer the actual driver call
     *  via ``eagle::util::CaptureGuardState::destroyOrDefer`` -- runs
     *  immediately when no capture is active anywhere in the process
     *  (today's exact timing), or queues until the outermost active capture
     *  scope exits. This object's own state (``instantiated_``/``instance_``)
     *  retires immediately either way, so a second call here is still the
     *  existing no-op guard, and the queued closure owns exactly one destroy. */
    void destroyInstance_()
    {
        if (instantiated_) {
            const cudaGraphExec_t handle = instance_;
            instantiated_                = false;
            instance_                    = nullptr;
            ::eagle::util::CaptureGuardState::instance().destroyOrDefer(
                [handle]() { EAGLE_CHECK_NOTHROW(cudaGraphExecDestroy(handle)); });
        }
    }

    /* Data members. ``storage_`` is declared first so it is destroyed
     * last (after ``destroyInstance_`` runs in ``~Launcher``). */
    std::shared_ptr<Graph::Storage> storage_;
    cudaGraphExec_t instance_     = nullptr;
    bool instantiated_            = false;
    cudaStream_t stream_          = 0;

    /** @brief Snapshot of ``Graph::kernelNodes_`` at ``launcher()``
     *  call time. Subsequent ``addNode`` calls on the producing Graph
     *  do not affect this list. Handles inside reference internals
     *  of ``storage_->graph``. */
    std::vector<KernelNodeRecord> kernelNodes_;

    /** @brief Per-node snapshots taken by ``setKernelParams``. */
    std::unordered_map<cudaGraphNode_t, KernelSnapshot> kernelSnapshots_;
};

/* Inline definition of Graph::launcher(); requires the full Launcher
 * definition above. */
inline Launcher Graph::launcher() const
{
    return Launcher(stream_, storage_, kernelNodes_);
}

} // namespace cuda
} // namespace eagle

#endif
