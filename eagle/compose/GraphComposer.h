// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file GraphComposer.h
 * @brief C++-native, dual-mode composition of independently-built launchable
 *        units into ONE launch list.
 *
 * @par ODR structure (load-bearing -- read before editing)
 * Unlike ``eagle::conditional::ConditionalGroup`` (which varies ONE member's
 * TYPE and several method BODIES by ``EAGLE_CPU_ONLY`` inside a single class),
 * this header is compiled under BOTH macro settings in the SAME link (the
 * CUDA gate binary links ~180 default-mode TUs alongside the CPU-face test
 * TU, which forces ``EAGLE_CPU_ONLY=1`` on itself) -- a class with the same
 * mangled name but a macro-varying member layout or method body is an ODR
 * violation the linker resolves by silently keeping ONE face's code for
 * every call site, corrupting the other face's objects. ``GraphComposer``
 * therefore has exactly ONE class definition with ONE member layout, valid
 * under both settings: every face-divergent behavior (a built ``Launcher``
 * vs a built ``cpu::Graph``; captured-super-graph toggling vs plain host
 * dispatch) is captured, AT REGISTRATION TIME, behind the macro-free
 * abstract interfaces ``MemberEngine`` / ``FlatCaptureEngine`` below. The
 * concrete implementations of those interfaces (``detail::Cuda*`` /
 * ``detail::Cpu*``) use FACE-PREFIXED, mutually exclusive names, so they
 * never collide even though only one face's set exists in any given
 * compiled TU. The handful of registration methods whose PARAMETER TYPES
 * differ by face (``registerLauncher(cuda::Launcher)`` vs
 * ``registerGraph(cpu::Graph)``; ``registerCallable``'s ``StepFn`` signature)
 * are naturally non-colliding (different mangled names already), so they may
 * freely differ in body -- everything else (``build``/``setRouting``/
 * ``launch``/``registerNested``/introspection) is byte-identical text on
 * both faces, dispatching only through the abstract interfaces.
 */

#include "eagle/cpu/Graph.h"
#include "eagle/cuda.h"
#include "eagle/cuda/CaptureAttribution.h"
#include "eagle/cuda/CaptureFork.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/throw.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace compose {

/**
 * @brief Abstract, macro-free per-member replay engine -- the type erasure
 *        seam a ``Launcher``/``cpu::Graph``/raw-callable member is wrapped
 *        behind at registration time (see the file doc's "ODR structure").
 *        This class's definition contains no ``EAGLE_CPU_ONLY`` branch at
 *        all: it compiles to byte-identical code in both faces.
 */
class MemberEngine {
public:
    virtual ~MemberEngine() = default;
    /** @brief Replay this member ``n`` times, host-blocking (synchronize
     *  before returning if it owns a stream). */
    virtual void replayBlocking(idx_t n) = 0;
    /** @brief Eligible for the cross-member overlap group (``mode=
     *  "sequenced"``, CUDA only -- always ``false`` on the CPU face and for
     *  a plain callable on either face, matching Python's semantics). */
    virtual bool ownsStream() const = 0;
    /** @brief Enqueue ``n`` replays WITHOUT synchronizing. Only called when
     *  @ref ownsStream is true. */
    virtual void enqueueAsync(idx_t n) = 0;
    /** @brief Synchronize after a prior @ref enqueueAsync. Only called when
     *  @ref ownsStream is true. */
    virtual void syncOwn() = 0;
};

/**
 * @brief Abstract, macro-free per-composer flat-capture engine
 *        (``mode="enabled"``/``"rebuild"``). A composer's ``flatCapture_``
 *        is null unless a CUDA-face registration installed one -- i.e.
 *        always null on the CPU face, and always null for a
 *        ``mode="sequenced"`` composer on either face (see the class doc's
 *        note). No ``EAGLE_CPU_ONLY`` branch in this definition either.
 */
class FlatCaptureEngine {
public:
    virtual ~FlatCaptureEngine() = default;
    /** @brief Register one more raw, kernel-issuing member (its target
     *  stream crosses as a ``uintptr_t`` so this interface stays CUDA-type-
     *  free). Returns its slot index, which MUST equal the caller's own
     *  ``Member`` registration index (one ``addRawMember`` call per
     *  ``registerCallable`` call, same order). */
    virtual idx_t addRawMember(std::function<void(std::uintptr_t)> step) = 0;
    /** @brief ``mode="enabled"``: capture every registered member ONCE, at
     *  full sibling concurrency, tracking per-member node attribution for
     *  later @ref setEnabled calls. */
    virtual void buildAll() = 0;
    /** @brief ``mode="rebuild"``: recapture from scratch containing ONLY
     *  the members marked active in @p active (index-aligned with every
     *  ``addRawMember`` call so far). An all-false pattern captures
     *  nothing. */
    virtual void rebuildActive(const std::vector<bool>& active) = 0;
    /** @brief ``mode="enabled"`` only: flip member @p idx's whole
     *  contributed node set on/off in the ONE instantiated exec graph.
     *  @throws std::runtime_error if that member's node set is not
     *          entirely toggleable. */
    virtual void setEnabled(idx_t idx, bool enabled) = 0;
    /** @brief Replay the ONE flat graph ``n`` times (a no-op if nothing is
     *  currently captured, e.g. ``mode="rebuild"`` with an all-inactive
     *  pattern). */
    virtual void launch(idx_t n) = 0;
};

namespace detail {

#ifndef EAGLE_CPU_ONLY

/** @brief CUDA face: wraps a built, already-instantiated ``Launcher``. */
class CudaLauncherEngine : public MemberEngine {
public:
    explicit CudaLauncherEngine(cuda::Launcher launcher)
        : launcher_{ std::move(launcher) }
    {
    }
    void replayBlocking(idx_t n) override
    {
        for (idx_t i = 0; i < n; i++)
            launcher_.launch();
        launcher_.synchronize();
    }
    bool ownsStream() const override { return launcher_.stream() != 0; }
    void enqueueAsync(idx_t n) override
    {
        for (idx_t i = 0; i < n; i++)
            launcher_.launch();
    }
    void syncOwn() override { launcher_.synchronize(); }

private:
    cuda::Launcher launcher_;
};

/** @brief CUDA face: a raw, kernel-issuing callable under mode="sequenced"
 *  -- owns a private, non-blocking stream to issue its own kernels onto.
 *  Never overlap-eligible (matches Python: a plain callable member has no
 *  ``_launch_handles``, so it is always dispatched blocking). */
class CudaCallableEngine : public MemberEngine {
public:
    explicit CudaCallableEngine(std::function<void(cudaStream_t)> step)
        : step_{ std::move(step) }
        , stream_{ /*nonBlocking=*/true }
    {
    }
    void replayBlocking(idx_t n) override
    {
        for (idx_t i = 0; i < n; i++)
            step_(stream_.cuda());
        stream_.synchronize();
    }
    bool ownsStream() const override { return false; }
    void enqueueAsync(idx_t) override { }
    void syncOwn() override { }

private:
    std::function<void(cudaStream_t)> step_;
    cuda::Stream stream_;
};

/** @brief CUDA face: the ``mode="enabled"``/``"rebuild"`` flat super-graph
 *  builder -- a ``CaptureFork`` sibling per raw member when there are 2+,
 *  attributing each member's contributed node set via
 *  ``eagle::cuda::captureSnapshotNodes`` snapshots taken immediately
 *  before/after that member's own callable call on its own branch stream
 *  -- the exact mechanism ``GraphPipeline.build()``
 *  (``eagle/python/eagle/pipeline.py``) uses, ported to C++ directly. */
class CudaFlatCaptureEngine : public FlatCaptureEngine {
public:
    idx_t addRawMember(std::function<void(std::uintptr_t)> step) override
    {
        const idx_t idx = idx_t(raw_.size());
        raw_.push_back(std::move(step));
        nodes_.emplace_back();
        toggleable_.push_back(true);
        return idx;
    }

    void buildAll() override
    {
        std::vector<idx_t> all(raw_.size());
        for (idx_t i = 0; i < idx_t(raw_.size()); i++)
            all[i] = i;
        capture_(all, /*trackToggle=*/true);
    }

    void rebuildActive(const std::vector<bool>& active) override
    {
        std::vector<idx_t> idxs;
        for (idx_t i = 0; i < idx_t(active.size()); i++)
            if (active[i])
                idxs.push_back(i);
        if (idxs.empty()) {
            graph_.reset();
            launcher_.reset();
            return;
        }
        capture_(idxs, /*trackToggle=*/false);
    }

    void setEnabled(idx_t idx, bool enabled) override
    {
        if (!toggleable_[idx])
            EAGLE_THROW(std::runtime_error,
                "GraphComposer.setRouting: member " + std::to_string(idx)
                    + " is not toggleable under mode=\"enabled\" -- its "
                      "captured node set contains something other than a "
                      "kernel/memcpy/memset node; eagle can only toggle the "
                      "node types CUDA itself supports (see "
                      "eagle::cuda::isNodeToggleable)");
        for (cudaGraphNode_t node : nodes_[idx])
            eagle::cuda::setNodeEnabled(launcher_->execHandle(), node, enabled);
    }

    void launch(idx_t n) override
    {
        if (!launcher_)
            return;
        for (idx_t i = 0; i < n; i++)
            launcher_->launch();
        launcher_->synchronize();
    }

private:
    void capture_(const std::vector<idx_t>& indices, bool trackToggle)
    {
        const cudaStream_t origin = ownStream_.cuda();
        const bool useFork        = indices.size() >= 2;
        std::unique_ptr<cuda::CaptureFork> fork;
        if (useFork)
            fork = std::make_unique<cuda::CaptureFork>(origin, indices.size());

        cuda::StreamCapturer capturer(origin);
        capturer.begin();
        if (useFork)
            fork->fork();

        for (std::size_t k = 0; k < indices.size(); k++) {
            const idx_t idx = indices[k];
            const cudaStream_t memberStream
                = useFork ? fork->branch(k) : origin;

            std::vector<cudaGraphNode_t> before;
            if (trackToggle)
                before = eagle::cuda::captureSnapshotNodes(memberStream);

            raw_[idx](reinterpret_cast<std::uintptr_t>(memberStream));

            if (trackToggle) {
                const std::vector<cudaGraphNode_t> after
                    = eagle::cuda::captureSnapshotNodes(memberStream);
                const std::set<cudaGraphNode_t> beforeSet(
                    before.begin(), before.end());
                nodes_[idx].clear();
                for (cudaGraphNode_t node : after)
                    if (beforeSet.find(node) == beforeSet.end())
                        nodes_[idx].push_back(node);
                toggleable_[idx] = std::all_of(nodes_[idx].begin(),
                    nodes_[idx].end(), [](cudaGraphNode_t node) {
                        return eagle::cuda::isNodeToggleable(node);
                    });
            }
        }

        if (useFork)
            fork->join();
        cudaGraph_t g = capturer.end();

        graph_    = std::make_shared<cuda::Graph>(cuda::CapturedGraph{ g });
        launcher_ = std::make_shared<cuda::Launcher>(graph_->launcher());
        launcher_->stream(origin);
    }

    std::vector<std::function<void(std::uintptr_t)>> raw_;
    std::vector<std::vector<cudaGraphNode_t>> nodes_;
    std::vector<bool> toggleable_;
    cuda::Stream ownStream_{ /*nonBlocking=*/true };
    std::shared_ptr<cuda::Graph> graph_;
    std::shared_ptr<cuda::Launcher> launcher_;
};

#else // EAGLE_CPU_ONLY

/** @brief CPU face: wraps a built ``cpu::Graph``. Never overlap-eligible --
 *  there is no stream/capture concept on the host. */
class CpuGraphEngine : public MemberEngine {
public:
    explicit CpuGraphEngine(cpu::Graph graph)
        : graph_{ std::make_shared<cpu::Graph>(std::move(graph)) }
    {
    }
    void replayBlocking(idx_t n) override
    {
        for (idx_t i = 0; i < n; i++)
            graph_->run();
    }
    bool ownsStream() const override { return false; }
    void enqueueAsync(idx_t) override { }
    void syncOwn() override { }

private:
    std::shared_ptr<cpu::Graph> graph_;
};

/** @brief CPU face: a raw callable member -- valid in every mode (
 *  there is no captured-super-graph distinction to make on the host, so
 *  every mode wraps a callable the exact same way). */
class CpuCallableEngine : public MemberEngine {
public:
    explicit CpuCallableEngine(std::function<void()> step)
        : step_{ std::move(step) }
    {
    }
    void replayBlocking(idx_t n) override
    {
        for (idx_t i = 0; i < n; i++)
            step_();
    }
    bool ownsStream() const override { return false; }
    void enqueueAsync(idx_t) override { }
    void syncOwn() override { }

private:
    std::function<void()> step_;
};

#endif // EAGLE_CPU_ONLY

} // namespace detail

/**
 * @brief Compose N already-built launchables into ONE launch list.
 *
 * @par The pitch (for a reader who knows Python + a little GPU, not CUDA
 *      internals)
 * A ``GraphComposer`` does not know or care what its members compute. The
 * motivating shape is an ENSEMBLE of independently-captured launchables that
 * a caller wants to fire together, selectively, and cheaply re-route between
 * replays without re-capturing everything from scratch -- e.g. a batch of
 * independent spacecraft-trajectory propagation pipelines,
 * where "which trajectories are still active this step" changes over time.
 * This is the C++-native engine behind ``eagle.compose.GraphComposer``
 * (Python) -- landed C++-native so a C++ consumer gets
 * it for free.
 *
 * @par The three modes this build supports (``mode=`` at construction)
 * - ``"sequenced"`` (DEFAULT, CORE): every member keeps its own private
 *   engine (own stream, if any), and this composer just decides, per
 *   launch, which of them fire -- including a built ``mode="sequenced"``
 *   ``GraphComposer`` AS a member of another (the RECURSION LOCK).
 * - ``"enabled"`` (CUDA face only -- see below): every registered raw
 *   callable is captured ONCE into one flat super-graph at full sibling
 *   concurrency, and switching a member on/off between replays is a
 *   microsecond host-side driver-flag flip, no recapture ever.
 * - ``"rebuild"`` (CUDA face only): every @ref setRouting call recaptures a
 *   fresh super-graph containing ONLY the active members.
 *
 * @par CPU face: the REFERENCE implementation
 * ``eagle::cpu::Graph`` has no capture/instantiate/replay split at all, so
 * there is no host analogue of a flat super-graph or an in-graph node to
 * toggle. On this face every mode resolves to the SAME ordered host
 * dispatch: visit every active member, in registration order.
 * ``"enabled"`` degenerates to a plain schedule-SKIP; ``"rebuild"``
 * degenerates to a schedule-REBUILD (the active set is just recomputed) --
 * no node-enable/recapture concept is invented on this face.
 *
 * @par Deferred: ``mode="conditional"``
 * Python's device-paced fourth mode is NOT implemented on this face this
 * package (pre-declared deferral, not an oversight) -- it remains available
 * in ``eagle.compose.GraphComposer`` (Python).
 *
 * @par The RECURSION LOCK
 * A built ``mode="sequenced"`` ``GraphComposer`` is itself a valid member of
 * another ``mode="sequenced"`` composer (@ref registerNested) -- composers
 * of composers, to any depth. Nesting under, or of, any OTHER mode is
 * rejected loudly at registration time.
 *
 * @par Fired-member recording
 * Every @ref launch call appends the active-member index set to a history
 * (@ref firedHistory / @ref resetFiredHistory).
 *
 * @par The overlap seam (``mode="sequenced"``, CUDA)
 * A member whose engine owns its own, non-default stream
 * (@ref MemberEngine::ownsStream) is eligible for the cross-member OVERLAP
 * GROUP: every eligible active member's replay(s) are enqueued FIRST, on
 * its own stream, and only THEN is every stream synchronized, in a second
 * pass -- mirroring ``compose.py``'s ``_async_launch_group`` enqueue-all-
 * then-join-all ordering. Everything else (a plain callable, a nested
 * composer -- always opaque and blocking, since its own ``launch()``
 * already owns its internal join) runs BLOCKING, in registration order,
 * first. On the CPU face NOTHING ever owns a stream, so this group is
 * always empty there -- the same code degenerates to pure ordered
 * dispatch automatically.
 *
 * @note DEVIATION from ``compose.py``: the Python overlap seam also
 *       brackets replay with a cupy event wait, ordering replay after
 *       the CALLER'S ambient current stream -- necessary there because
 *       cupy gives every kernel call an IMPLICIT current-stream target.
 *       Raw CUDA C++ has no such ambient concept, so there is nothing
 *       implicit to race against here; a caller sequencing device work
 *       against a composer's replay does so with its own explicit
 *       stream/event wait before calling @ref launch.
 * @note DEVIATION: a raw callable member has no ambient "current stream"
 *       to issue kernels onto (unlike Python) -- its signature takes the
 *       target stream explicitly (as a ``uintptr_t`` at the type-erasure
 *       seam; ``cudaStream_t`` at the public @ref registerCallable API).
 *
 * @note Binding note: inherits ``std::enable_shared_from_this`` purely for
 *       the nanobind ``shared_ptr<GraphComposer>`` interop @ref
 *       registerNested's Python binding relies on (the RECURSION LOCK
 *       passes a Python-owned composer into this same-typed C++ argument).
 *       Without it, nanobind's ``shared_ptr<T>`` type caster manufactures a
 *       fresh, Python-refcount-backed bridge for every conversion instead
 *       of reusing this object's own control block once one exists, and
 *       skips the ``keep_shared_from_this_alive`` safety net nanobind
 *       installs specifically for types that support it -- confirmed via
 *       ``nanobind/stl/shared_ptr.h`` and ``nb_class.h`` (both read
 *       directly). Purely additive: no public member, no size-sensitive
 *       layout change relative to itself across the two build faces (both
 *       compile this same base uniformly), no behavior change outside the
 *       binding's shared_ptr conversions.
 */
class GraphComposer : public std::enable_shared_from_this<GraphComposer> {
public:
    using Self         = GraphComposer;
    using PreLaunchFn  = std::function<void()>;

#ifndef EAGLE_CPU_ONLY
    /** @brief A raw, kernel-issuing member step. Takes the target stream
     *  EXPLICITLY (see the class doc's DEVIATION note). */
    using StepFn = std::function<void(cudaStream_t)>;
#else
    /** @brief A raw member step -- no stream parameter on this face. */
    using StepFn = std::function<void()>;
#endif

    /**
     * @brief Construct an empty composer.
     * @param mode One of ``"sequenced"`` (default), ``"enabled"``,
     *        ``"rebuild"``. ``"conditional"`` is DEFERRED (see the class
     *        doc) and rejected with a message naming the deferral.
     * @throws std::invalid_argument for an unrecognized mode.
     */
    explicit GraphComposer(std::string mode = "sequenced")
        : mode_{ std::move(mode) }
    {
        if (mode_ != "sequenced" && mode_ != "enabled" && mode_ != "rebuild") {
            const std::string extra = (mode_ == "conditional")
                ? " -- \"conditional\" (device-paced, per-member IF node) is "
                  "DEFERRED on this C++-native engine; it remains available "
                  "in eagle.compose.GraphComposer (Python)"
                : "";
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer: mode must be one of {\"sequenced\", "
                "\"enabled\", \"rebuild\"}; got \""
                    + mode_ + "\"" + extra);
        }
    }

    GraphComposer(const GraphComposer&)            = delete;
    GraphComposer& operator=(const GraphComposer&) = delete;
    GraphComposer(GraphComposer&&)                 = default;
    GraphComposer& operator=(GraphComposer&&)      = default;

    // ------------------------------------------------------------ registration

#ifndef EAGLE_CPU_ONLY
    /** @brief Register a built ``Launcher`` as a member (``mode=
     *  "sequenced"`` only -- other modes fuse every member into one
     *  captured/flat step; register a raw callable via
     *  @ref registerCallable instead). Returns the registration-order
     *  index.
     *  @throws std::runtime_error if called after @ref build.
     *  @throws std::invalid_argument if this composer is not
     *          ``mode="sequenced"``, if ``pre_launch`` is set incorrectly,
     *          or on a duplicate ``name``. */
    idx_t registerLauncher(cuda::Launcher launcher, std::string name = {},
        PreLaunchFn pre_launch = nullptr)
    {
        beforeRegister_(name, static_cast<bool>(pre_launch));
        if (mode_ != "sequenced")
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.registerLauncher: a built Launcher member is "
                "only accepted under mode=\"sequenced\" (mode=\""
                    + mode_
                    + "\" fuses every member into one captured/flat step -- "
                      "register a raw callable via registerCallable() "
                      "instead)");
        Member m;
        m.kind      = Member::Kind::Engine;
        m.name      = name;
        m.preLaunch = std::move(pre_launch);
        m.engine
            = std::make_shared<detail::CudaLauncherEngine>(std::move(launcher));
        return finishRegister_(std::move(m), name);
    }
#else
    /** @brief Register a built ``cpu::Graph`` as a member -- the CPU
     *  counterpart of the CUDA face's ``registerLauncher`` (``mode=
     *  "sequenced"`` only). Returns the registration-order index.
     *  @throws std::runtime_error if called after @ref build.
     *  @throws std::invalid_argument if this composer is not
     *          ``mode="sequenced"``, if ``pre_launch`` is set incorrectly,
     *          or on a duplicate ``name``. */
    idx_t registerGraph(cpu::Graph graph, std::string name = {},
        PreLaunchFn pre_launch = nullptr)
    {
        beforeRegister_(name, static_cast<bool>(pre_launch));
        if (mode_ != "sequenced")
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.registerGraph: a built cpu::Graph member is "
                "only accepted under mode=\"sequenced\" -- register a raw "
                "callable via registerCallable() instead");
        Member m;
        m.kind      = Member::Kind::Engine;
        m.name      = name;
        m.preLaunch = std::move(pre_launch);
        m.engine    = std::make_shared<detail::CpuGraphEngine>(std::move(graph));
        return finishRegister_(std::move(m), name);
    }
#endif

    /** @brief Register a raw, kernel-issuing callable as a member -- valid
     *  in EVERY mode. Under ``mode="sequenced"`` it is replayed directly
     *  (blocking, in registration order, unless ``pre_launch`` seeds it
     *  first); under the CUDA face's flat modes it is captured into this
     *  composer's own super-graph at @ref build / @ref setRouting time.
     *  Returns the registration-order index.
     *  @throws std::runtime_error if called after @ref build.
     *  @throws std::invalid_argument if ``pre_launch`` is set on a
     *          non-``"sequenced"`` composer, or on a duplicate ``name``. */
    idx_t registerCallable(
        StepFn step, std::string name = {}, PreLaunchFn pre_launch = nullptr)
    {
        beforeRegister_(name, static_cast<bool>(pre_launch));
#ifndef EAGLE_CPU_ONLY
        if (mode_ == "sequenced") {
            Member m;
            m.kind      = Member::Kind::Engine;
            m.name      = name;
            m.preLaunch = std::move(pre_launch);
            m.engine
                = std::make_shared<detail::CudaCallableEngine>(std::move(step));
            return finishRegister_(std::move(m), name);
        }
        if (!flatCapture_)
            flatCapture_ = std::make_shared<detail::CudaFlatCaptureEngine>();
        flatCapture_->addRawMember([step](std::uintptr_t s) {
            step(reinterpret_cast<cudaStream_t>(s));
        });
        Member m;
        m.kind = Member::Kind::FlatCaptured;
        m.name = name;
        return finishRegister_(std::move(m), name);
#else
        Member m;
        m.kind      = Member::Kind::Engine;
        m.name      = name;
        m.preLaunch = std::move(pre_launch);
        m.engine    = std::make_shared<detail::CpuCallableEngine>(std::move(step));
        return finishRegister_(std::move(m), name);
#endif
    }

    /** @brief Register a nested, already-``build()``-ed ``mode="sequenced"``
     *  composer as a member -- the RECURSION LOCK. Both this (OUTER)
     *  composer and @p nested must be ``mode="sequenced"``. Identical on
     *  both faces. Returns the registration-order index.
     *  @throws std::runtime_error if called after @ref build.
     *  @throws std::invalid_argument if either composer is not
     *          ``mode="sequenced"``, if @p nested is not yet built, if
     *          @p nested is null or is this composer itself, if
     *          ``pre_launch`` is set incorrectly, on a duplicate ``name``,
     *          or if @p nested is already registered here. */
    idx_t registerNested(std::shared_ptr<GraphComposer> nested,
        std::string name = {}, PreLaunchFn pre_launch = nullptr)
    {
        beforeRegister_(name, static_cast<bool>(pre_launch));
        if (mode_ != "sequenced")
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.registerNested: nesting a composer is only "
                "supported under a mode=\"sequenced\" OUTER composer (the "
                "RECURSION LOCK covers sequenced-mode nesting only this "
                "phase)");
        if (!nested)
            EAGLE_THROW(
                std::invalid_argument, "GraphComposer.registerNested: nested "
                                        "composer must not be null");
        if (nested.get() == this)
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.registerNested: a composer cannot nest "
                "itself");
        if (nested->mode_ != "sequenced")
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.registerNested: nesting a composer requires "
                "BOTH composers be mode=\"sequenced\" (this composer is \""
                    + mode_ + "\", the registered one is \"" + nested->mode_
                    + "\")");
        if (!nested->built_)
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.registerNested: a nested composer must be "
                "build()-ed before registration -- its own launch() "
                "contract requires it");
        for (const Member& existing : members_)
            if (existing.kind == Member::Kind::Nested
                && existing.nested.get() == nested.get())
                EAGLE_THROW(std::invalid_argument,
                    "GraphComposer.registerNested: this nested composer is "
                    "already registered -- each member may be registered "
                    "at most once");
        Member m;
        m.kind      = Member::Kind::Nested;
        m.name      = name;
        m.preLaunch = std::move(pre_launch);
        m.nested    = std::move(nested);
        return finishRegister_(std::move(m), name);
    }

    // ------------------------------------------------------------------ build

    /** @brief Perform whatever one-time setup ``mode`` needs. Identical on
     *  both faces: dispatches through @ref FlatCaptureEngine when one was
     *  installed (CUDA face, flat modes only); pure bookkeeping otherwise
     *  (every other case, including the whole CPU face).
     *  @throws std::runtime_error if already built, or if no members are
     *          registered.
     *  @throws std::runtime_error (``mode="enabled"``) if any member's
     *          captured node set is not entirely toggleable. */
    Self& build()
    {
        if (built_)
            EAGLE_THROW(std::runtime_error,
                "GraphComposer.build: already built (one capture)");
        if (members_.empty())
            EAGLE_THROW(std::runtime_error,
                "GraphComposer.build: no members registered -- call "
                "register...() at least once before build()");
        activeMask_.assign(members_.size(), true);
        if (flatCapture_) {
            if (mode_ == "enabled") {
                flatCapture_->buildAll();
                for (idx_t i = 0; i < idx_t(members_.size()); i++)
                    flatCapture_->setEnabled(i, true);
            } else { // "rebuild"
                flatCapture_->rebuildActive(activeMask_);
            }
        }
        built_ = true;
        return *this;
    }

    // ---------------------------------------------------------------- routing

    /** @brief Update which members are active. Identical on both faces.
     *  @throws std::runtime_error if called before @ref build.
     *  @throws std::invalid_argument if ``pattern.size()`` does not equal
     *          the registered member count.
     *  @throws std::runtime_error (``mode="enabled"``) if toggling a
     *          non-toggleable member. */
    Self& setRouting(const std::vector<bool>& pattern)
    {
        if (!built_)
            EAGLE_THROW(std::runtime_error,
                "GraphComposer.setRouting: call build() before setRouting()");
        if (pattern.size() != members_.size())
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.setRouting: routing pattern length "
                    + std::to_string(pattern.size())
                    + " != registered member count "
                    + std::to_string(members_.size()));
        if (flatCapture_) {
            if (mode_ == "enabled") {
                for (idx_t i = 0; i < idx_t(members_.size()); i++)
                    flatCapture_->setEnabled(i, pattern[i]);
            } else { // "rebuild"
                flatCapture_->rebuildActive(pattern);
            }
        }
        activeMask_.assign(pattern.begin(), pattern.end());
        return *this;
    }

    // ------------------------------------------------------------------ replay

    /** @brief Replay ``n`` times. Identical on both faces: delegates to
     *  @ref FlatCaptureEngine when one was installed; ordered per-member
     *  dispatch (@ref launchOrdered_) otherwise -- which is EVERY case on
     *  the CPU face, and the ``mode="sequenced"`` case on the CUDA face.
     *  @throws std::runtime_error if called before @ref build. */
    Self& launch(idx_t n = 1)
    {
        if (!built_)
            EAGLE_THROW(
                std::runtime_error, "GraphComposer.launch: call build() "
                                     "before launch()");
        if (flatCapture_) {
            std::vector<idx_t> fired;
            for (idx_t i = 0; i < idx_t(members_.size()); i++)
                if (activeMask_[i])
                    fired.push_back(i);
            firedHistory_.push_back(std::move(fired));
            flatCapture_->launch(n);
            return *this;
        }
        return launchOrdered_(n);
    }

    // ------------------------------------------------------------ introspection

    const std::string& mode() const { return mode_; }
    idx_t numMembers() const { return idx_t(members_.size()); }

    /** @brief Registration-order member names (empty string for an
     *  unnamed member). */
    std::vector<std::string> memberNames() const
    {
        std::vector<std::string> out;
        out.reserve(members_.size());
        for (const Member& m : members_)
            out.push_back(m.name);
        return out;
    }

    /** @brief Per-``launch()``-call fired-member index sets, oldest first,
     *  since construction or the last @ref resetFiredHistory. */
    const std::vector<std::vector<idx_t>>& firedHistory() const
    {
        return firedHistory_;
    }

    void resetFiredHistory() { firedHistory_.clear(); }

private:
    struct Member {
        enum class Kind { Engine, Nested, FlatCaptured } kind = Kind::Engine;
        std::string name;
        PreLaunchFn preLaunch;
        std::shared_ptr<MemberEngine> engine;   // Kind::Engine
        std::shared_ptr<GraphComposer> nested;  // Kind::Nested
        // Kind::FlatCaptured members carry no per-member state here at all
        // -- their replay logic lives entirely inside flatCapture_, and
        // launchOrdered_ (below) never iterates a flat-mode composer's
        // members (launch() dispatches to flatCapture_ first).
    };

    void beforeRegister_(const std::string& name, bool hasPreLaunch)
    {
        if (built_)
            EAGLE_THROW(std::runtime_error,
                "GraphComposer.register: cannot register after build() -- "
                "this composer's capture is already committed");
        if (hasPreLaunch && mode_ != "sequenced")
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.register: pre_launch is only accepted for "
                "mode=\"sequenced\" members -- the other modes fuse every "
                "member into ONE captured/flat step already");
        if (!name.empty() && names_.count(name))
            EAGLE_THROW(std::invalid_argument,
                "GraphComposer.register: duplicate member name \"" + name
                    + "\"");
    }

    idx_t finishRegister_(Member&& m, const std::string& name)
    {
        const idx_t index = idx_t(members_.size());
        members_.push_back(std::move(m));
        if (!name.empty())
            names_.insert(name);
        return index;
    }

    /** @brief THE ordered-dispatch path: ``mode="sequenced"`` on the CUDA
 *  face, and EVERY mode on the CPU face (no ``flatCapture_``
     *  ever exists there, so @ref launch always reaches this). Splits
     *  active members into the cross-member overlap group (engines that
     *  own their own stream -- always empty on the CPU face) and blocking
     *  members (a plain callable, a stream-less engine, or a nested
     *  composer -- its own ``launch()`` already owns its internal join, so
     *  it is one opaque blocking unit), runs blocking members first, then
     *  enqueues the overlap group and synchronizes it in a second pass. */
    Self& launchOrdered_(idx_t n)
    {
        std::vector<idx_t> active;
        for (idx_t i = 0; i < idx_t(members_.size()); i++)
            if (activeMask_[i])
                active.push_back(i);
        firedHistory_.push_back(active);
        if (active.empty())
            return *this;

        std::vector<idx_t> overlapGroup;
        std::vector<idx_t> blocking;
        for (idx_t i : active) {
            const Member& m = members_[i];
            if (m.kind == Member::Kind::Engine && m.engine
                && m.engine->ownsStream())
                overlapGroup.push_back(i);
            else
                blocking.push_back(i);
        }

        for (idx_t i : blocking) {
            Member& m = members_[i];
            if (m.preLaunch)
                m.preLaunch();
            switch (m.kind) {
            case Member::Kind::Engine:
                m.engine->replayBlocking(n);
                break;
            case Member::Kind::Nested:
                m.nested->launch(n);
                break;
            default:
                break; // unreachable: FlatCaptured members never appear here
            }
        }

        if (!overlapGroup.empty()) {
            for (idx_t i : overlapGroup) {
                Member& m = members_[i];
                if (m.preLaunch)
                    m.preLaunch();
                m.engine->enqueueAsync(n);
            }
            for (idx_t i : overlapGroup)
                members_[i].engine->syncOwn();
        }
        return *this;
    }

    std::string mode_;
    std::vector<Member> members_;
    std::set<std::string> names_;
    std::vector<bool> activeMask_;
    std::vector<std::vector<idx_t>> firedHistory_;
    bool built_ = false;

    /** @brief Null except on the CUDA face, for a composer that has
     *  registered at least one raw callable under ``mode="enabled"``/
     *  ``"rebuild"`` -- see @ref registerCallable. Always null on the CPU
 *  face and for ``mode="sequenced"`` on either face. */
    std::shared_ptr<FlatCaptureEngine> flatCapture_;
};

} // namespace compose
} // namespace eagle
