// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file eagle_core.cpp
 * @brief nanobind binding of the ``eagle::cuda`` CUDA-graph core.
 *
 * The compiled module of the EAGLE Python package. It exposes exactly the
 * host-side graph primitives the Python ``GraphPipeline`` needs to capture a
 * sequence of externally-issued launches, instantiate the resulting graph, and
 * replay it — the SAME ``eagle::cuda`` classes a native C++ consumer uses, so
 * Python and C++ share one graph implementation.
 *
 * Two images: this module is plain C++ (g++, ``EAGLE_CPU_ONLY``), with no CUDA
 * runtime, toolkit or driver dependency; every CUDA object lives in the backend
 * plugin ``libeagle_cuda.so`` behind the ``eagle-backend/1`` C seam
 * (``src/seam/eagle_backend.h``), loaded on the first call that needs it. The
 * classes below own opaque handles into it. Without a usable backend (no plugin,
 * an incompatible one, no driver or device) those calls raise
 * ``eagle.BackendUnavailable``; the host structures and loader facts never need it.
 *
 * Design boundary: the core owns *capture / instantiate / replay*; the kernels
 * themselves are always issued by the caller (cupy ``RawKernel``, a torch/jax
 * launch, …) onto the stream this module hands out as a raw ``uintptr_t``.
 *
 * Bound surface (minimal, parity-driven):
 *   Stream          — owns a ``cudaStream_t``; ``ptr()`` exposes it to Python.
 *   StreamCapturer  — ``begin()`` / ``end()`` around the captured launches.
 *   CaptureFork     — ``fork()`` / ``join()`` around independent branch work,
 *                     so one capture records siblings instead of a chain.
 *   CaptureConditional — ``begin()`` / ``end()`` weaving a device-evaluated
 *                     IF node around a skippable region's
 *                     launches, on the origin stream OR a CaptureFork branch.
 *   CapturedGraph   — move-only owner of the captured ``cudaGraph_t``;
 *                     ``debug_dot()`` for node introspection (matches cupy).
 *   Graph           — parent container; ``add_node`` folds in the capture and
 *                     harvests its kernel records; ``launcher()`` instantiates.
 *   Launcher        — ``launch()`` / ``synchronize()`` / ``set_logical_size()`` /
 *                     ``set_node_enabled()``.
 *   capture_snapshot_nodes / is_node_toggleable — free functions backing
 *                     ``GraphPipeline.build()``'s per-member node attribution
 *                     (the member-enable "mode=enabled" graph-runtime
 *                     control plane; see ``eagle::cuda::CaptureAttribution.h``).
 *   capture_guard_depth / capture_guard_pending_count — test/diagnostic-only
 *                     introspection of ``eagle::util::CaptureGuardState``
 *                     (STOP-THE-LINE incident fix: a process-wide
 *                     capture-depth counter + deferred-destroy queue so a
 *                     GC-triggered teardown never invalidates an unrelated
 *                     active capture; see ``eagle/util/CaptureGuard.h``).
 *   device_props    — ``eagle::cuda::deviceProps(device)`` as a plain
 *                     dict, every raw + derived field, zero Python-side
 *                     computation (see ``eagle/DeviceProps.h``).
 *   InteropBuffer / fence / event_pool_created — the generic DLPack layer
 *                     (``plugin/interop.h``), see ``interop_binding.h``.
 *   GraphComposer   — the C++-native
 *                     ``eagle::compose::GraphComposer`` parity surface
 *                     (construct/register/build/set_routing/launch/fired/
 *                     reset + introspection; see
 *                     ``eagle/compose/GraphComposer.h``). ``compose.py``'s
 *                     own Python ``GraphComposer`` stays byte-frozen this
 *                     package — folding its launch path onto this engine is
 *                     the named post-boundary follow-on, not this binding.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  // std::optional <-> None (GraphComposer named args)
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>  // std::vector<uintptr_t> <-> list[int] (capture_snapshot_nodes)
#include <nanobind/stl/pair.h>  // (address, element bytes) plane pairs (reorder_device)
#include <nanobind/stl/tuple.h>  // graph-view rows (nodes, edges, plane refs)
#include <nanobind/stl/function.h>  // the recording forward's checkpoint callback

// The heterogeneous execution contract's C++ half.
// `eagle.plan` (Python) decides a Plan; these are the structures it drives, plus
// the loader facts the plan must agree with (the ABI tags and the layout
// self-check sizes). All of it is host code: this module is compiled without a
// device runtime (EAGLE_CPU_ONLY) and reaches the GPU only through the backend
// plugin (src/seam/).
#include "eagle/diff/Record.h"
#include "eagle/diff/Reverse.h"
#include "eagle/diff/Transpose.h"
#include "eagle/diff/Window.h"
#include "eagle/exec/HostSchedule.h"
#include "eagle/exec/HostTeam.h"
#include "eagle/exec/Partition.h"
#include "eagle/exec/TimeLaunch.h"
#include "eagle/filtering/ReorderClasses.h"
#include "eagle/graph/Edges.h"
#include "eagle/graph/Layout.h"
#include "eagle/graph/Render.h"
#include "eagle/cpu/DeviceProps.h"
#include "eagle/policy/Dispatch.h"
#include "eagle/reduce/canonical.h"
#include "eagle/util/Busy.h"
#include "plugin/gref_abi.h"
#include "plugin/roles.h"

#include "interop_binding.h"  // the generic DLPack layer (eagle.interop)
#include "seam/DeviceKernel.h"
#include "seam/Groups.h"
#include "seam/Loader.h"
#include "seam/Status.h"

namespace nb = nanobind;

using eagle::exec::HostTeam;
using eagle::exec::Partition;

namespace {

// The documented FT-2 instrument: a counter that is DELIBERATELY plain, so a
// free-threading run can prove its schedule is adversarial enough to lose
// updates before it certifies anything else. Nothing real reads or writes it.
// The read-modify-write is split by a short busy gap so the lost-update window
// is wide enough to expose on every free-threaded build.
volatile std::uint64_t g_unsynchronised = 0;

inline void unsynchronisedBump()
{
    const std::uint64_t seen = g_unsynchronised;
    for (volatile int spin = 0; spin < 2048; spin = spin + 1) {}
    g_unsynchronised = seen + 1;
}

/** @brief Turn a Python list of integer pointers into the ``params[]`` array a
 *  plugin entry receives. The caller owns every pointer; this only re-types the
 *  list, exactly as ``eagle.host_launch``'s ctypes marshalling does. */
inline std::vector<void*> asParams(const std::vector<std::uintptr_t>& args)
{
    std::vector<void*> params;
    params.reserve(args.size());
    for (std::uintptr_t a : args) params.push_back(reinterpret_cast<void*>(a));
    return params;
}

/** @brief The CUDA backend (loaded on first use; see src/seam/Loader.h). */
inline eagle_seam::Backend& cuda()
{
    return eagle_seam::backend("cuda");
}

/** @brief Raise the Python exception a nonzero status of the CUDA backend means. */
inline void ck(eagle_backend_status status)
{
    eagle_seam::check(cuda(), status);
}

#define EB(group, sym) EAGLE_SEAM_FN(group, sym)
#define EBX(sym) EAGLE_SEAM_XFN(sym)

/** @brief A zero-filled, size-prefixed desc naming device @p device. */
template <class D>
D makeDesc(std::int32_t device)
{
    D d{};
    d.struct_size = sizeof(D);
    d.device = device;
    return d;
}

// -- Owning wrappers of the backend's opaque handles ------------------------
// Each Python object owns one handle and releases it with its own lifetime. A
// handle CONSUMED by another call (Graph.from_captured / add_node take a
// CapturedGraph, GraphComposer.register_launcher takes a Launcher) is nulled
// here, and any later use raises instead of touching a moved-from object.

struct PyStream {
    eagle_backend_stream h = nullptr;
    PyStream() = default;
    PyStream(const PyStream&) = delete;
    PyStream& operator=(const PyStream&) = delete;
    ~PyStream() { if (h) EB("stream", eagle_backend_stream_release)(h); }
};

struct PyCaptured {
    eagle_backend_captured h = nullptr;
    PyCaptured() = default;
    PyCaptured(const PyCaptured&) = delete;
    PyCaptured& operator=(const PyCaptured&) = delete;
    ~PyCaptured() { if (h) EB("captured", eagle_backend_captured_release)(h); }
};

struct PyCapturer {
    eagle_backend_capturer h = nullptr;
    PyCapturer() = default;
    PyCapturer(const PyCapturer&) = delete;
    PyCapturer& operator=(const PyCapturer&) = delete;
    ~PyCapturer() { if (h) EB("capturer", eagle_backend_capturer_release)(h); }
};

struct PyFork {
    eagle_backend_fork h = nullptr;
    PyFork() = default;
    PyFork(const PyFork&) = delete;
    PyFork& operator=(const PyFork&) = delete;
    ~PyFork() { if (h) EB("fork", eagle_backend_fork_release)(h); }
};

struct PyConditional {
    eagle_backend_conditional h = nullptr;
    PyConditional() = default;
    PyConditional(const PyConditional&) = delete;
    PyConditional& operator=(const PyConditional&) = delete;
    ~PyConditional() { if (h) EB("conditional", eagle_backend_conditional_release)(h); }
};

struct PyLauncher {
    eagle_backend_launcher h = nullptr;
    PyLauncher() = default;
    PyLauncher(const PyLauncher&) = delete;
    PyLauncher& operator=(const PyLauncher&) = delete;
    ~PyLauncher() { if (h) EB("launcher", eagle_backend_launcher_release)(h); }
};

struct PyGraph {
    eagle_backend_graph h = nullptr;
    PyGraph() = default;
    PyGraph(const PyGraph&) = delete;
    PyGraph& operator=(const PyGraph&) = delete;
    ~PyGraph() { if (h) EB("graph", eagle_backend_graph_release)(h); }
};

struct PyComposer {
    eagle_backend_composer h = nullptr;
    // Python callables and nested composers the backend calls back into: kept
    // alive here for as long as this composer can launch them.
    std::vector<nb::object> keep;
    PyComposer() = default;
    PyComposer(const PyComposer&) = delete;
    PyComposer& operator=(const PyComposer&) = delete;
    ~PyComposer() { if (h) EBX(eagle_backend_x_composer_release)(h); }
};

/** The handle of a wrapper, or a RuntimeError naming the consumed object. */
template <class H>
H need(H h, const char* what)
{
    if (h == nullptr)
        throw std::runtime_error(std::string("eagle: this ") + what
            + " was consumed by an earlier call (moved into a Graph or GraphComposer) and can no longer be used");
    return h;
}

// -- Callback trampolines: Python callables the backend calls back into ------
// They run on the calling thread inside the seam call; a Python exception is
// stored, the call unwinds inside the backend, and eagle_seam::check re-raises it.

std::int32_t stepTrampoline(void* ctx, std::uint64_t stream)
{
    nb::gil_scoped_acquire gil;
    try {
        nb::handle(static_cast<PyObject*>(ctx))(static_cast<std::uintptr_t>(stream));
        return 0;
    } catch (nb::python_error& e) {
        eagle_seam::storeCallbackError(std::move(e));
        return 1;
    } catch (...) {
        return 1;
    }
}

std::int32_t voidTrampoline(void* ctx)
{
    nb::gil_scoped_acquire gil;
    try {
        nb::handle(static_cast<PyObject*>(ctx))();
        return 0;
    } catch (nb::python_error& e) {
        eagle_seam::storeCallbackError(std::move(e));
        return 1;
    } catch (...) {
        return 1;
    }
}

/**
 * @brief Write a captured graph to Graphviz ``dot`` and return the text.
 *
 * Mirrors cupy's ``Graph.debug_dot_str``: both call
 * ``cudaGraphDebugDotPrint`` on the captured ``cudaGraph_t``, so the Python
 * ``num_nodes()`` / ``node_labels()`` parsers produce byte-identical results
 * whichever implementation captured the graph. Called on the CapturedGraph
 * *before* it is consumed by ``Graph::add_node``.
 */
std::string capturedDebugDot(const PyCaptured& c, const std::string& path, unsigned int flags)
{
    if (c.h == nullptr)
        throw std::runtime_error("CapturedGraph is empty (no handle to dot)");
    ck(EBX(eagle_backend_x_captured_debug_dot)(c.h, path.c_str(), flags));
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("could not read dot file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// -- The CUDA fence, as the interop layer's provider for the CUDA device types --

void cudaFence(std::int32_t type, std::int32_t device, std::optional<std::intptr_t> producer,
    std::optional<std::intptr_t> consumer)
{
    eagle_seam::Backend& b = cuda();
    b.require();
    if (!b.servesDeviceType(type))
        throw eagle_seam::Unavailable("eagle: the " + b.label() + " does not serve DLPack device type "
            + std::to_string(type));
    const auto code = [](std::optional<std::intptr_t> s) {
        return s ? static_cast<std::int64_t>(*s) : static_cast<std::int64_t>(EAGLE_BACKEND_STREAM_NONE);
    };
    ck(EB("interop", eagle_backend_fence)(type, device, code(producer), code(consumer)));
}

/** A buffer of a device type no loaded backend routes. */
void noBackendFence(std::int32_t type, std::int32_t, std::optional<std::intptr_t>, std::optional<std::intptr_t>)
{
    throw eagle_seam::Unavailable(
        "eagle: no eagle backend serves DLPack device type " + std::to_string(type) + " (only kind 'cuda' exists)");
}


// -- The graph view (eagle/graph/*.h): rows in, text and numbers out --
namespace eg = eagle::graph;

using NodeRow  = std::tuple<std::string, std::string, std::string, std::int64_t, bool>;
using EdgeRow  = std::tuple<std::int64_t, std::int64_t, std::string, std::string>;
using FrameRow = std::tuple<std::string, std::string, std::int64_t>;
using RefRow   = std::tuple<std::string, std::int64_t, std::uint64_t,
    std::vector<std::int64_t>, std::vector<std::int64_t>, std::int64_t, int>;

eg::PlaneRef planeRefOf(const RefRow& r)
{
    const auto& [name, node, base, shape, strides, itemsize, access] = r;
    if (shape.size() != strides.size())
        throw std::invalid_argument("plane '" + name + "': shape and strides differ in length");
    if (access < 0 || access > 2)
        throw std::invalid_argument("plane '" + name + "': access must be 0 (read), 1 (write) or 2 (in place)");
    return { name, node, base, shape, strides, itemsize, access };
}

std::vector<eg::PlaneRef> planeRefsOf(const std::vector<RefRow>& rows)
{
    std::vector<eg::PlaneRef> out;
    out.reserve(rows.size());
    for (const RefRow& r : rows) out.push_back(planeRefOf(r));
    return out;
}

std::vector<eg::SiblingGroup> groupsOf(const std::vector<std::vector<std::vector<std::int64_t>>>& g)
{
    std::vector<eg::SiblingGroup> out;
    for (const auto& branches : g) out.push_back({ branches });
    return out;
}

/** @brief An immutable Description (built once from rows, validated). */
struct PyDescription {
    eg::Description d;
};

} // namespace

NB_MODULE(_core, m)
{
    m.doc() = "eagle._core — nanobind binding of the eagle::cuda CUDA-graph "
              "core (capture / instantiate / replay).";

    m.def("_unsynchronised_bump", [] { unsynchronisedBump(); },
          "Free-threading canary: increments a deliberately PLAIN counter. "
          "Concurrent calls lose updates; a run that cannot observe the loss "
          "cannot certify the real counters. An instrument, not an API.");
    m.def("_unsynchronised_count", [] { return std::uint64_t{g_unsynchronised}; },
          "The canary counter `_unsynchronised_bump` increments.");

    // The generic DLPack layer: buffer import/validation/export and the
    // stream contract (plugin/interop.h). See interop_binding.h.
    eagle_interop_binding::bind(m);

    // -- The device backend (eagle-backend/1 seam, see src/seam/) -------------
    eagle_seam::registerTranslators();
    m.attr("BackendUnavailable") = eagle_seam::unavailableType();
    // The interop layer re-fences a device buffer through a provider per DLPack
    // device type: the CUDA types route to the cuda backend (loaded on that first
    // use), every other device type raises BackendUnavailable naming it.
    for (std::int32_t type = 2; type < 64; ++type)
        eagle::interop::set_fence_provider(type, &noBackendFence);
    for (std::int32_t type : { std::int32_t(kDLCUDA), std::int32_t(kDLCUDAHost), std::int32_t(kDLCUDAManaged) })
        eagle::interop::set_fence_provider(type, &cudaFence);
    m.def(
        "event_pool_created",
        [] {
            std::int64_t n = 0;
            ck(EB("interop", eagle_backend_event_pool_created)(&n));
            return n;
        },
        "How many events the fence pool has created.");
    m.def(
        "cuda_backend",
        [] {
            eagle_seam::Backend& b = eagle_seam::backend("cuda");
            nb::dict d;
            d["kind"] = b.kind();
            d["loaded"] = b.loaded();
            d["path"] = b.path();
            d["version"] = b.loaded()
                ? std::to_string(b.version() >> 16) + "." + std::to_string(b.version() & 0xffffu)
                : std::string();
            d["error"] = b.error();
            d["capabilities"] = b.capabilities();
            d["device_types"] = b.deviceTypes();
            nb::dict info;
            if (b.loaded()) {
                // Fields past the size the backend wrote back are ABSENT.
                const auto& i = b.buildInfo();
                const auto has = [&](std::size_t end) { return i.struct_size >= end; };
                using BI = struct eagle_backend_build_info;
                info["struct_size"] = i.struct_size;
                if (has(offsetof(BI, abi_version) + sizeof i.abi_version))
                    info["abi_version"] = i.abi_version;
                if (has(offsetof(BI, cudart_version) + sizeof i.cudart_version))
                    info["cudart_version"] = i.cudart_version;
                if (has(offsetof(BI, eagle_version) + sizeof i.eagle_version))
                    info["eagle_version"] = std::string(i.eagle_version);
                if (has(offsetof(BI, archs) + sizeof i.archs))
                    info["archs"] = std::string(i.archs);
                if (has(offsetof(BI, backend_kind) + sizeof i.backend_kind))
                    info["backend_kind"] = std::string(i.backend_kind);
            }
            d["build_info"] = info;
            nb::dict groups;
            for (const eagle_seam::GroupSpec& g : eagle_seam::groups())
                groups[g.name] = b.groupError(g);
            d["groups"] = groups;
            nb::dict experimental;
            for (const char* s : eagle_seam::experimentalSymbols())
                experimental[s] = b.optional(s) != nullptr;
            d["experimental"] = experimental;
            return d;
        },
        "Load (once) and describe the CUDA backend plugin: whether it loaded, "
        "from where, its seam version and capability bits, its build info, "
        "and per capability group the reason it is unavailable ('' when "
        "bound). Diagnostics only; never raises.");

    // -----------------------------------------------------------------------
    // The HETEROGENEOUS EXECUTION CONTRACT (aether-abi/2).
    //
    // The plan goes in Python and the execution in C++:
    // `eagle.plan` maps (manifest execution axis x residency x structures) to a
    // Plan, and these are what it drives. Nothing here decides anything -- the
    // placement RULES are here only so the Python planner and the C++ loader
    // cannot disagree about them (one spelling, `eagle/exec/Partition.h`).
    // -----------------------------------------------------------------------
    nb::class_<Partition>(m, "Partition")
        .def(nb::init<>())
        .def("__init__",
            [](Partition* self, std::int64_t base, std::int64_t count,
               std::int64_t n_samples) {
                new (self) Partition{ base, count, n_samples };
            },
            nb::arg("base"), nb::arg("count"), nb::arg("n_samples"),
            "The launch partition {base, count, nSamples}. `base` is this "
            "partition's first global sample, `count` how many it covers, and "
            "`n_samples` the TRUE total the run spans -- the number every wide / "
            "accumulate plane is addressed against, whatever the cut.")
        .def_static("whole", &Partition::whole, nb::arg("n"),
            "The legacy shape: one partition covering everything.")
        .def_rw("base", &Partition::base)
        .def_rw("count", &Partition::count)
        .def_rw("n_samples", &Partition::nSamples)
        .def("is_whole", &Partition::is_whole,
            "Is this the whole view -- the only shape a legacy aether-abi/1 "
            "plugin may be handed?")
        .def("__repr__", [](const Partition& p) {
            return "Partition(base=" + std::to_string(p.base) + ", count=" +
                   std::to_string(p.count) + ", n_samples=" +
                   std::to_string(p.nSamples) + ")";
        });

    m.def("check_placement",
        [](const std::string& access, const std::string& structure,
           int npartitions) {
            eagle::exec::check_placement(access, structure, npartitions);
        },
        nb::arg("access"), nb::arg("structure"), nb::arg("npartitions") = 1,
        "Is running a body of declared `access` under `structure` over "
        "`npartitions` partitions LEGAL? Raises naming the rule. "
        "`access` is the manifest's exec_access; `structure` is one of "
        "device_kernel / host_team / rank_partition / device_group.");

    m.def("compact_device",
        [](std::uintptr_t mask, std::uintptr_t index_map, std::uintptr_t count, std::uintptr_t scratch,
           std::int64_t n, std::uintptr_t stream, std::uint32_t flags, std::int32_t device,
           std::uintptr_t live32, std::uintptr_t span, std::uintptr_t fire, float theta) {
            auto d = makeDesc<eagle_backend_compact_desc>(device);
            d.flags = flags;
            d.mask = mask;
            d.index_map = index_map;
            d.count = count;
            d.scratch = scratch;
            d.n = n;
            d.stream = stream;
            d.live32 = live32;
            d.span = span;
            d.fire = fire;
            d.theta = theta;
            ck(EB("filtering", eagle_backend_compact_device)(&d));
            return static_cast<std::int64_t>(d.scratch_bytes);
        },
        nb::arg("mask"), nb::arg("index_map"), nb::arg("count"), nb::arg("scratch"), nb::arg("n"),
        nb::arg("stream") = 0, nb::arg("flags") = 0, nb::arg("device") = -1, nb::arg("live32") = 0,
        nb::arg("span") = 0, nb::arg("fire") = 0, nb::arg("theta") = 0.0f,
        "Active-set compaction (the `filtering` group): enqueue predicate -> scan -> "
        "scatter of `n` samples on `stream`, writing the ascending kept indices to "
        "`index_map` (int32) and their number to `count` (uint32); every buffer is a "
        "device address as an integer. With `live32` (and `span`, `fire`, `theta`) the "
        "compaction also computes the reorder trigger: `live32` the 32-sample groups "
        "holding a kept sample, `fire` = count < theta * 32 * live32, count < span and "
        "span >= 4096. "
        "With `scratch == 0` nothing is enqueued and the "
        "return value is the scratch size in bytes the call needs; otherwise it returns 0.");

    // A reorder entry: the plane table crosses as (address, element bytes) pairs.
    auto reorderDesc = [](const std::vector<std::pair<std::uintptr_t, std::uint32_t>>& planes,
                           std::vector<eagle_backend_reorder_plane>& table, std::int64_t n, std::uintptr_t scratch,
                           std::uintptr_t stream, std::uint32_t flags, std::int32_t device) {
        table.clear();
        for (const auto& [data, bytes] : planes) {
            eagle_backend_reorder_plane p{};
            p.data = data;
            p.elem_bytes = bytes;
            table.push_back(p);
        }
        auto d = makeDesc<eagle_backend_reorder_desc>(device);
        d.flags = flags;
        d.n_planes = static_cast<std::uint32_t>(table.size());
        d.planes = reinterpret_cast<std::uint64_t>(table.data());
        d.n = n;
        d.scratch = scratch;
        d.stream = stream;
        return d;
    };

    m.def("reorder_device",
        [reorderDesc](const std::vector<std::pair<std::uintptr_t, std::uint32_t>>& planes, std::uintptr_t mask,
            std::uintptr_t perm, std::uintptr_t inv, std::uintptr_t index_map, std::uintptr_t count,
            std::uintptr_t span, std::uintptr_t fire, std::uintptr_t scratch, std::int64_t n, std::uintptr_t stream,
            std::uint32_t flags, std::int32_t device) {
            // the group first, so a backend without `filtering` names the group
            (void)EB("filtering", eagle_backend_compact_device);
            std::vector<eagle_backend_reorder_plane> table;
            auto d = reorderDesc(planes, table, n, scratch, stream, flags, device);
            d.mask = mask;
            d.perm = perm;
            d.inv = inv;
            d.index_map = index_map;
            d.count = count;
            d.span = span;
            d.fire = fire;
            ck(EBX(eagle_backend_reorder_device)(&d));
            return static_cast<std::int64_t>(d.scratch_bytes);
        },
        nb::arg("planes"), nb::arg("mask"), nb::arg("perm"), nb::arg("inv"), nb::arg("index_map"),
        nb::arg("count"), nb::arg("span"), nb::arg("fire"), nb::arg("scratch"), nb::arg("n"),
        nb::arg("stream") = 0, nb::arg("flags") = 0, nb::arg("device") = -1,
        "Physical reorder (the `filtering` group): enqueue the in-place permutation of "
        "every plane in `planes` ((address, element bytes) pairs; the mask among them) "
        "over [0, *span) so the kept samples come first, in order; `perm`/`inv` follow, "
        "`index_map` becomes the identity over [0, *count), `*span = *count`. With "
        "`fire` != 0 the kernels exit while *fire == 0 and the call clears it. With "
        "`scratch == 0` nothing is enqueued and the return value is the scratch size "
        "in bytes; otherwise it returns 0.");

    m.def(
        "backend_has",
        [](const std::string& symbol, const std::string& kind) {
            return eagle_seam::backend(kind).optional(symbol.c_str()) != nullptr;
        },
        nb::arg("symbol"), nb::arg("kind") = "cuda",
        "Whether the loaded backend `kind` resolves the seam symbol `symbol` (a symbol bound "
        "one by one: its presence is the capability). False when the backend is not loaded.");

    m.def("compact_classes_device",
        [](std::uintptr_t mask, std::uintptr_t index_map, std::uintptr_t count, std::uintptr_t scratch,
           std::int64_t n, std::uintptr_t stream, std::uint32_t flags, std::int32_t device,
           std::uintptr_t live32, std::uintptr_t span, std::uintptr_t fire, float theta,
           std::uintptr_t class_plane, std::uint32_t classes, float class_theta, std::uintptr_t mixed32,
           std::uintptr_t stale_count, std::uintptr_t class_fire) {
            auto d = makeDesc<eagle_backend_compact_classes_desc>(device);
            d.flags = flags;
            d.mask = mask;
            d.index_map = index_map;
            d.count = count;
            d.scratch = scratch;
            d.n = n;
            d.stream = stream;
            d.live32 = live32;
            d.span = span;
            d.fire = fire;
            d.theta = theta;
            d.class_plane = class_plane;
            d.classes = classes;
            d.class_theta = class_theta;
            d.mixed32 = mixed32;
            d.stale_count = stale_count;
            d.class_fire = class_fire;
            ck(EBX(eagle_backend_compact_classes_device)(&d));
            return static_cast<std::int64_t>(d.scratch_bytes);
        },
        nb::arg("mask"), nb::arg("index_map"), nb::arg("count"), nb::arg("scratch"), nb::arg("n"),
        nb::arg("stream"), nb::arg("flags"), nb::arg("device"), nb::arg("live32"), nb::arg("span"),
        nb::arg("fire"), nb::arg("theta"), nb::arg("class_plane"), nb::arg("classes"), nb::arg("class_theta"),
        nb::arg("mixed32"), nb::arg("stale_count") = 0, nb::arg("class_fire") = 0,
        "K-class compaction (the `filtering` group, bound per symbol): `compact_device` with the "
        "liveness trigger plus `mixed32` and `class_fire`. With `scratch == 0` nothing is enqueued "
        "and the return value is the scratch size in bytes; otherwise it returns 0.");

    m.def("reorder_classes_device",
        [reorderDesc](const std::vector<std::pair<std::uintptr_t, std::uint32_t>>& planes, std::uintptr_t mask,
            std::uintptr_t perm, std::uintptr_t inv, std::uintptr_t index_map, std::uintptr_t count,
            std::uintptr_t span, std::uintptr_t fire, std::uintptr_t scratch, std::int64_t n,
            std::uintptr_t stream, std::uint32_t flags, std::int32_t device, std::uintptr_t class_plane,
            std::uint32_t classes, std::uintptr_t offsets, std::uintptr_t class_sorts) {
            std::vector<eagle_backend_reorder_plane> table;
            auto b = reorderDesc(planes, table, n, scratch, stream, flags, device);
            auto d = makeDesc<eagle_backend_reorder_classes_desc>(device);
            d.flags = b.flags;
            d.n_planes = b.n_planes;
            d.planes = b.planes;
            d.mask = mask;
            d.perm = perm;
            d.inv = inv;
            d.index_map = index_map;
            d.count = count;
            d.span = span;
            d.fire = fire;
            d.scratch = scratch;
            d.n = n;
            d.stream = stream;
            d.class_plane = class_plane;
            d.classes = classes;
            d.offsets = offsets;
            d.class_sorts = class_sorts;
            ck(EBX(eagle_backend_reorder_classes_device)(&d));
            return static_cast<std::int64_t>(d.scratch_bytes);
        },
        nb::arg("planes"), nb::arg("mask"), nb::arg("perm"), nb::arg("inv"), nb::arg("index_map"),
        nb::arg("count"), nb::arg("span"), nb::arg("fire"), nb::arg("scratch"), nb::arg("n"),
        nb::arg("stream"), nb::arg("flags"), nb::arg("device"), nb::arg("class_plane"), nb::arg("classes"),
        nb::arg("offsets"), nb::arg("class_sorts"),
        "K-class stable sort (the `filtering` group, bound per symbol): the in-place permutation of "
        "every plane in `planes` (the mask and the class plane among them) by (live, class) over "
        "[0, *span), or by class over [0, *count) with flags bit 1 (WITHIN_LIVE); `offsets` is the "
        "live class prefix, `class_sorts` is incremented. With `scratch == 0` nothing is enqueued "
        "and the return value is the scratch size in bytes; otherwise it returns 0.");

    m.def("restore_device",
        [reorderDesc](const std::vector<std::pair<std::uintptr_t, std::uint32_t>>& planes, std::uintptr_t perm,
            std::uintptr_t inv, std::uintptr_t scratch, std::int64_t n, std::uintptr_t stream,
            std::int32_t device) {
            (void)EB("filtering", eagle_backend_compact_device);
            std::vector<eagle_backend_reorder_plane> table;
            auto d = reorderDesc(planes, table, n, scratch, stream, 0u, device);
            d.perm = perm;
            d.inv = inv;
            ck(EBX(eagle_backend_x_restore_device)(&d));
            return static_cast<std::int64_t>(d.scratch_bytes);
        },
        nb::arg("planes"), nb::arg("perm"), nb::arg("inv"), nb::arg("scratch"), nb::arg("n"),
        nb::arg("stream") = 0, nb::arg("device") = -1,
        "Undo every reorder since the last restore (experimental at the seam: "
        "`eagle_backend_x_restore_device`): each plane back in sample order, `perm` "
        "and `inv` the identity. Enqueue only; the scratch query as for reorder_device.");

    // -- canonical reduction (eagle/reduce/canonical.h): host tree + device seam symbol --
    m.def("reduce_canonical_host",
        [](std::uintptr_t in, std::int64_t n, std::uint32_t dtype, std::uint32_t op, std::uintptr_t out) {
            if (n < 1)
                throw std::invalid_argument("reduce_canonical_host: n must be >= 1");
            if (op > EAGLE_BACKEND_REDUCE_OP_ALL || dtype > EAGLE_BACKEND_REDUCE_U8
                || (op == EAGLE_BACKEND_REDUCE_OP_ALL) != (dtype == EAGLE_BACKEND_REDUCE_U8))
                throw std::invalid_argument("reduce_canonical_host: unknown dtype/op, or U8 not paired with OP_ALL");
            namespace canon = eagle::reduce::canonical;
            const bool sum = op == EAGLE_BACKEND_REDUCE_OP_SUM;
            auto go = [&](auto tag, auto sumOp, auto maxOp) {
                using T = decltype(tag);
                struct Ptr {
                    const T* p;
                    T operator()(eagle::idx_t i) const { return p[i]; }
                } src{ reinterpret_cast<const T*>(in) };
                *reinterpret_cast<T*>(out) = sum ? canon::reduce<T, decltype(sumOp)>(src, n)
                                                 : canon::reduce<T, decltype(maxOp)>(src, n);
            };
            switch (dtype) {
            case EAGLE_BACKEND_REDUCE_F32: go(float{}, aether::SumOp<float>{}, aether::MaxOp<float>{}); break;
            case EAGLE_BACKEND_REDUCE_F64: go(double{}, aether::SumOp<double>{}, aether::MaxOp<double>{}); break;
            case EAGLE_BACKEND_REDUCE_I32:
                go(std::int32_t{}, aether::SumOp<std::int32_t>{}, aether::MaxOp<std::int32_t>{});
                break;
            case EAGLE_BACKEND_REDUCE_I64:
                go(std::int64_t{}, aether::SumOp<std::int64_t>{}, aether::MaxOp<std::int64_t>{});
                break;
            default: {
                struct Ptr8 {
                    const std::uint8_t* p;
                    std::uint8_t operator()(eagle::idx_t i) const { return p[i]; }
                } src{ reinterpret_cast<const std::uint8_t*>(in) };
                *reinterpret_cast<std::uint8_t*>(out) = canon::reduce<std::uint8_t, aether::LogicalAndOp>(src, n);
            }
            }
        },
        nb::arg("in"), nb::arg("n"), nb::arg("dtype"), nb::arg("op"), nb::arg("out"),
        "The canonical fixed-order reduction (leaf blocks of 256, then pairwise levels) of `n` host "
        "elements at address `in` into the one element at `out`. dtype: 0 f32, 1 f64, 2 i32, 3 i64, "
        "4 u8 (OP_ALL only); op: 0 sum, 1 max, 2 all. Bitwise independent of the OpenMP thread count.");

    m.def("reduce_canonical_device",
        [](std::uintptr_t in, std::int64_t n, std::uint32_t dtype, std::uint32_t op, std::uintptr_t out,
            std::uintptr_t work_a, std::uintptr_t work_b, std::uintptr_t stream, std::int32_t device) {
            auto d = makeDesc<eagle_backend_reduce_desc>(device);
            d.dtype = dtype;
            d.op = op;
            d.in = in;
            d.out = out;
            d.work_a = work_a;
            d.work_b = work_b;
            d.n = n;
            d.stream = stream;
            ck(EBX(eagle_backend_x_reduce_canonical)(&d));
            return static_cast<std::int64_t>(d.work_elems);
        },
        nb::arg("in"), nb::arg("n"), nb::arg("dtype"), nb::arg("op"), nb::arg("out"), nb::arg("work_a"),
        nb::arg("work_b"), nb::arg("stream") = 0, nb::arg("device") = -1,
        "The same canonical reduction on the device (experimental at the seam: "
        "`eagle_backend_x_reduce_canonical`), enqueued on `stream` and capturable. With `work_a == 0` "
        "nothing is enqueued and the return value is the element count EACH of the two work buffers "
        "must hold; otherwise it returns 0. The result equals `reduce_canonical_host` bit for bit.");

    m.def("run_device",
        [](std::uintptr_t function, const std::vector<std::uintptr_t>& params,
           const Partition& part, std::uintptr_t stream, unsigned block) {
            std::vector<std::uint64_t> p(params.begin(), params.end());
            auto d = makeDesc<eagle_backend_launch_desc>(-1);
            d.block = block;
            d.function = function;
            d.params = p.data();
            d.nparams = static_cast<std::int64_t>(p.size());
            d.base = part.base;
            d.count = part.count;
            d.n_samples = part.nSamples;
            d.stream = stream;
            std::int32_t launched = 0;
            ck(EB("exec", eagle_backend_run_device)(&d, &launched));
            return static_cast<int>(launched);
        },
        nb::arg("function"), nb::arg("params"), nb::arg("partition"),
        nb::arg("stream") = 0, nb::arg("block") = 256,
        "ONE cuLaunchKernel over one partition, with the int64 triple appended "
        "after the role args and the grid derived from the partition's `count` "
        "`function` is a CUfunction handle and `params` a list of "
        "pointers, both as integers. Returns 1, or 0 for an empty partition.");

    m.def("time_launch",
        [](std::uintptr_t entry, const std::vector<std::uintptr_t>& params, const Partition& part,
           int reps, bool device, std::uintptr_t stream, unsigned block) {
            if (!device) {
                const std::vector<void*> p = asParams(params);
                return eagle::exec::timeHostLaunch(reinterpret_cast<eagle::exec::HostEntryV2>(entry),
                                                   p.data(), part, reps);
            }
            if (reps < 1) throw std::invalid_argument("time_launch: reps must be at least 1");
            std::vector<std::uint64_t> p(params.begin(), params.end());
            auto d = makeDesc<eagle_backend_launch_desc>(-1);
            d.block = block;
            d.function = entry;
            d.params = p.data();
            d.nparams = static_cast<std::int64_t>(p.size());
            d.base = part.base;
            d.count = part.count;
            d.n_samples = part.nSamples;
            d.stream = stream;
            std::vector<double> seconds(static_cast<std::size_t>(reps));
            ck(EBX(eagle_backend_x_time_launch)(&d, reps, seconds.data()));
            return seconds;
        },
        nb::arg("entry"), nb::arg("params"), nb::arg("partition"), nb::arg("reps"),
        nb::arg("device") = false, nb::arg("stream") = 0, nb::arg("block") = 256,
        "Seconds of each of `reps` launches of one entry over one partition, every repetition "
        "kept: the host team run timed with steady_clock, or (device=True; `entry` a CUfunction) "
        "a CUDA event pair around each launch on `stream` (experimental at the seam: "
        "`eagle_backend_x_time_launch`; one stream sync after the last). For calibration.");

    m.def("reorder_classes_host",
        [](const std::vector<std::pair<std::uintptr_t, std::uint32_t>>& planes, std::uintptr_t mask,
           std::uintptr_t perm, std::uintptr_t inv, std::uintptr_t index_map, std::uintptr_t count,
           std::uintptr_t span, std::uintptr_t fire, std::int64_t n, std::uint32_t flags,
           std::uintptr_t class_plane, std::uint32_t classes, std::uintptr_t offsets,
           std::uintptr_t class_sorts) {
            std::vector<eagle::filtering::ReorderPlane> table;
            table.reserve(planes.size());
            for (const auto& [address, bytes] : planes)
                table.push_back({ reinterpret_cast<void*>(address), bytes });
            eagle::filtering::reorderClassesHost(table.data(), static_cast<std::uint32_t>(table.size()),
                reinterpret_cast<const std::uint8_t*>(mask), flags, classes,
                reinterpret_cast<const std::uint8_t*>(class_plane), reinterpret_cast<std::int32_t*>(perm),
                reinterpret_cast<std::int32_t*>(inv), reinterpret_cast<std::int32_t*>(index_map),
                reinterpret_cast<std::uint32_t*>(count), reinterpret_cast<std::uint32_t*>(span),
                reinterpret_cast<std::uint32_t*>(fire), reinterpret_cast<std::uint32_t*>(offsets),
                reinterpret_cast<std::uint32_t*>(class_sorts), n);
        },
        nb::arg("planes"), nb::arg("mask"), nb::arg("perm"), nb::arg("inv"), nb::arg("index_map"),
        nb::arg("count"), nb::arg("span"), nb::arg("fire"), nb::arg("n"), nb::arg("flags"),
        nb::arg("class_plane"), nb::arg("classes"), nb::arg("offsets"), nb::arg("class_sorts"),
        "The host face of reorder_classes_device: the same stable counting sort by (live, class), "
        "in place, now; the planes are (address, element bytes) pairs of host memory.");

    m.def("device_grid", &eagle::exec::launch_grid,
        nb::arg("count"), nb::arg("block"),
        "The grid eagle derives from a partition's `count` (never nSamples).");

    m.def("run_host",
        [](std::uintptr_t entry, const std::vector<std::uintptr_t>& params,
           const Partition& part, std::size_t bytes_per_sample) {
            const std::vector<void*> p = asParams(params);
            return HostTeam::run(
                reinterpret_cast<eagle::exec::HostEntryV2>(entry), p.data(), part,
                bytes_per_sample);
        },
        nb::arg("entry"), nb::arg("params"), nb::arg("partition"),
        nb::arg("bytes_per_sample") = 0,
        "Run a v2 host entry over one partition as an OpenMP team of contiguous "
        "tiles, each tile carrying its OWN triple. "
        "Returns the number of tiles run.");

    m.def("run_host_serial",
        [](std::uintptr_t entry, const std::vector<std::uintptr_t>& params,
           const Partition& part) {
            const std::vector<void*> p = asParams(params);
            HostTeam::run_serial(
                reinterpret_cast<eagle::exec::HostEntryV2>(entry), p.data(), part);
        },
        nb::arg("entry"), nb::arg("params"), nb::arg("partition"),
        "One call, one triple -- the SERIAL arm, and the correctness oracle the "
        "band rows are judged against (fork F-c).");

    m.def("host_threads",
        [] {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        },
        "The size of the host team a launch would use right now.");

    m.attr("BRANCH_TASKS_MAX_SAMPLES") = eagle::exec::BRANCH_TASKS_MAX_SAMPLES;

    m.def("resolve_host_schedule",
        [](std::int64_t n, std::int64_t branches, std::int64_t threads, bool native_only) {
            return eagle::exec::resolve(n, branches, threads, native_only) ==
                           eagle::exec::Schedule::BranchTasks
                       ? "branch_tasks" : "work_share";
        },
        nb::arg("n"), nb::arg("branches"), nb::arg("threads"), nb::arg("native_only"),
        "How the host runs a stage: 'branch_tasks' iff it has >= 2 independent "
        "branches, a team of >= 2 threads, n <= BRANCH_TASKS_MAX_SAMPLES and only "
        "native nodes; else 'work_share'. Pure.");

    // -- the splitting rule (eagle/policy/Dispatch.h) ------------------------------
    {
        namespace pol = eagle::policy;
        using BranchRow = std::tuple<double, double, double, double, double, double>;

        // Only the derived fields the rule reads cross the boundary.
        auto propsOf = [](const nb::dict& d) {
            eagle::DeviceProps p;
            p.peak_flops_sp = nb::cast<double>(d["peak_flops_sp"]);
            p.peak_bytes_per_s = nb::cast<double>(d["peak_bytes_per_s"]);
            p.fp64_ratio = nb::cast<double>(d["fp64_ratio"]);
            return p;
        };
        auto targetOf = [](const std::string& name) {
            if (name == "host") return pol::Target::Host;
            if (name == "device") return pol::Target::Device;
            throw std::invalid_argument("target must be 'host' or 'device', got '" + name + "'");
        };
        auto costOf = [](const std::vector<BranchRow>& rows, bool fusedVectorised) {
            if (rows.size() < 2 || rows.size() > static_cast<std::size_t>(pol::MAX_BRANCHES))
                throw std::invalid_argument("a dispatch group has 2 to 16 branches, got " +
                                            std::to_string(rows.size()));
            pol::CostVector c;
            c.fusedVectorised = fusedVectorised;
            for (const auto& [a32, a64, s32, s64, loads, gathers] : rows)
                c.branch[c.count++] = { a32, a64, s32, s64, loads, gathers };
            return c;
        };
        auto mixOf = [](const std::vector<std::int64_t>& counts, std::int64_t fusedSteps,
                        std::int64_t remainingSteps, double mixed32) {
            pol::Mix mix;
            for (std::size_t j = 0; j < counts.size() && j < pol::MAX_BRANCHES; ++j)
                mix.counts[j] = counts[j];
            mix.fusedSteps = fusedSteps;
            mix.remainingSteps = remainingSteps;
            mix.mixed32 = mixed32;
            return mix;
        };

        m.attr("DISPATCH_ENTER_GAIN") = pol::ENTER_GAIN;
        m.attr("DISPATCH_LEAVE_GAIN") = pol::LEAVE_GAIN;
        m.attr("DISPATCH_EVERY") = pol::DECISION_EVERY;

        m.def("cpu_device_props",
            [] {
                const eagle::DeviceProps p = eagle::cpu::deviceProps();
                nb::dict d;
                d["peak_flops_sp"] = p.peak_flops_sp;
                d["peak_bytes_per_s"] = p.peak_bytes_per_s;
                d["fp64_ratio"] = p.fp64_ratio;
                return d;
            },
            "The host's documented device properties, in the dict form the splitting rule reads.");

        m.def("dispatch_gain",
            [=](const std::vector<BranchRow>& branches, const std::vector<std::int64_t>& counts,
                const nb::dict& props, const std::string& target, std::int64_t fused_steps,
                bool fused_vectorised, std::int64_t remaining_steps, double mixed32) {
                return pol::predictedGain(costOf(branches, fused_vectorised),
                                          mixOf(counts, fused_steps, remaining_steps, mixed32),
                                          propsOf(props), pol::defaultWeights(targetOf(target)));
            },
            nb::arg("branches"), nb::arg("counts"), nb::arg("props"), nb::arg("target"),
            nb::arg("fused_steps") = 1, nb::arg("fused_vectorised") = false,
            nb::arg("remaining_steps") = pol::DECISION_EVERY, nb::arg("mixed32") = -1.0,
            "Predicted time of the fused switch kernel over the split one for this branch mix "
            "(above 1 the split wins). branches: per branch (arith32, arith64, special32, "
            "special64, loads, gathers); counts: samples that took each branch; target: "
            "'host' or 'device'; fused_vectorised: the fused host loop runs every branch; "
            "remaining_steps: steps a class sort is paid over; mixed32: measured share of "
            "32-lane groups holding two or more branches (negative: unmeasured). Pure.");

        m.def("resolve_dispatch_policy",
            [=](const std::vector<BranchRow>& branches, const std::vector<std::int64_t>& counts,
                const nb::dict& props, const std::string& target, const std::string& current,
                std::int64_t fused_steps, bool fused_vectorised, std::int64_t remaining_steps,
                double mixed32) {
                if (current != "switch" && current != "segmented")
                    throw std::invalid_argument("current must be 'switch' or 'segmented', got '" + current + "'");
                pol::HysteresisState state;
                state.current = current == "segmented" ? pol::Policy::Segmented : pol::Policy::Switch;
                return pol::resolve(costOf(branches, fused_vectorised),
                                    mixOf(counts, fused_steps, remaining_steps, mixed32), propsOf(props),
                                    pol::defaultWeights(targetOf(target)), state) == pol::Policy::Segmented
                           ? "segmented" : "switch";
            },
            nb::arg("branches"), nb::arg("counts"), nb::arg("props"), nb::arg("target"),
            nb::arg("current") = "switch", nb::arg("fused_steps") = 1,
            nb::arg("fused_vectorised") = false, nb::arg("remaining_steps") = pol::DECISION_EVERY,
            nb::arg("mixed32") = -1.0,
            "'segmented' once the predicted gain reaches DISPATCH_ENTER_GAIN, 'switch' once it falls "
            "under DISPATCH_LEAVE_GAIN, else `current`; an empty mix keeps `current`. Pure.");

        m.def("dispatch_decide",
            [](double gain, const std::string& current) {
                if (current != "switch" && current != "segmented")
                    throw std::invalid_argument("current must be 'switch' or 'segmented', got '" + current + "'");
                return pol::decide(gain, current == "segmented" ? pol::Policy::Segmented
                                                                : pol::Policy::Switch) == pol::Policy::Segmented
                           ? "segmented" : "switch";
            },
            nb::arg("gain"), nb::arg("current"),
            "The band alone: 'segmented' at gain >= DISPATCH_ENTER_GAIN, 'switch' under DISPATCH_LEAVE_GAIN, "
            "else `current`. Pure.");

        m.def("dispatch_due",
            [](std::int64_t since, std::int64_t steps) {
                pol::HysteresisState state;
                state.sinceDecision = static_cast<int>(since);
                const bool look = pol::due(state, static_cast<int>(steps));
                return std::make_pair(look, static_cast<std::int64_t>(state.sinceDecision));
            },
            nb::arg("since"), nb::arg("steps") = 1,
            "(look, since): count `steps` more steps after `since`; look is True every DISPATCH_EVERY steps.");
    }

    m.def("run_host_branches",
        [](const std::vector<std::vector<std::tuple<std::uintptr_t,
               std::vector<std::uintptr_t>, std::vector<Partition>>>>& stage) {
            std::vector<eagle::exec::HostBranch> branches;
            for (const auto& branch : stage) {
                auto& out = branches.emplace_back();
                for (const auto& [entry, params, parts] : branch)
                    out.push_back({ reinterpret_cast<eagle::exec::HostEntryV2>(entry),
                                    asParams(params), parts });
            }
            nb::gil_scoped_release release;
            eagle::exec::runHostBranches(branches);
        },
        nb::arg("stage"),
        "Run one stage's independent branches at once, one per thread, with the GIL "
        "released. `stage` is a list of branches; a branch is a list of "
        "(entry, params, partitions) host nodes run in order on one thread.");

    m.def("host_tile_size", &HostTeam::tile_size, nb::arg("bytes_per_sample") = 0,
        "The team's tile size in samples -- `Host::launch`'s own rule "
        "(aether::optimalTileSize); 0 falls back to aether's DEFAULT_TILE_SIZE.");
    m.def("host_tile_count", &HostTeam::tile_count, nb::arg("partition"),
        nb::arg("bytes_per_sample") = 0,
        "How many tiles a partition would be cut into.");

    m.def("fold",
        [](const std::string& op, const std::vector<double>& values) {
            return eagle::exec::fold(eagle::exec::op_from_string(op),
                                     values.data(), values.size());
        },
        nb::arg("op"), nb::arg("values"),
        "Combine partials in eagle's FIXED ascending order -- never "
        "atomics, so the deterministic arm is reproducible. `op` is the "
        "manifest's exec_op (sum|times|max|land).");

    // -- Loader facts the Python plan surface must agree with -----------------
    m.attr("ABI_TAG_V1") = EAGLE_AETHER_ABI;
    m.attr("ABI_TAG_V2") = EAGLE_AETHER_ABI_V2;
    m.attr("SCHEMA_VERSION") = eagle::plugin::kPluginSchemaVersion;
    m.attr("MAX_SCHEMA_VERSION") = eagle::plugin::kPluginMaxSchemaVersion;
    m.attr("LAYOUT_SYMBOL") = eagle::plugin::kEagleLayoutSymbol;

    m.def("abi_version_of", &eagle::plugin::abi_version_of, nb::arg("tag"),
        "1 for 'aether-abi/1', 2 for 'aether-abi/2', 0 for anything else.");

    m.def("layout_sizes", [] {
        std::uint64_t want[eagle::plugin::kEagleLayoutFieldCount];
        eagle::plugin::expected_layout_sizes(want);
        return std::vector<std::uint64_t>(
            want, want + eagle::plugin::kEagleLayoutFieldCount);
    },
        "This build's aether-abi/2 layout self-check sizes, in field order: "
        "sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle), "
        "sizeof(aether::idx_t), sizeof(partition triple). An artifact exporting "
        "'eagle_layout_sizes' must match these exactly.");

    m.def("layout_field_name", [](std::size_t i) {
        return std::string(eagle::plugin::layout_field_name(i));
    },
        nb::arg("index"), "The human name of layout field `index`.");

    m.def("arg_roles", [] {
        std::vector<std::string> roles;
        for (const char* r : eagle::plugin::kPluginArgRoles) roles.push_back(r);
        return roles;
    },
        "The canonical arg-spec role vocabulary (the C++ half of the register).");

    // -- Stream: owns a cudaStream_t, hands its raw handle to Python ----------
    // Every CUDA object below lives in the backend plugin behind an opaque
    // handle; these classes own the handle. `device` (default -1) names the
    // device the object is created on: -1 is the device of the stream passed
    // in or, for an object created without one, the calling thread's current
    // device (the one cupy/torch made current).
    nb::class_<PyStream>(m, "Stream")
        .def(
            "__init__",
            [](PyStream* self, bool non_blocking, int device) {
                auto d = makeDesc<eagle_backend_stream_desc>(device);
                d.flags = non_blocking ? EAGLE_BACKEND_STREAM_NON_BLOCKING : 0u;
                eagle_backend_stream h = nullptr;
                ck(EB("stream", eagle_backend_stream_create)(&d, &h));
                new (self) PyStream();
                self->h = h;
            },
            nb::arg("non_blocking") = false, nb::arg("device") = -1,
            "Create a stream; non_blocking=True (recommended for graph "
            "capture) avoids an implicit dependency on the legacy default "
            "stream that cupy/torch launch on.")
        .def(
            "ptr",
            [](const PyStream& s) {
                std::uint64_t p = 0;
                ck(EB("stream", eagle_backend_stream_ptr)(need(s.h, "Stream"), &p));
                return static_cast<std::uintptr_t>(p);
            }, nb::lock_self(),
            "Raw cudaStream_t as a Python int (wrap with "
            "cupy.cuda.ExternalStream).")
        .def("synchronize", [](PyStream& s) {
            ck(EB("stream", eagle_backend_stream_synchronize)(need(s.h, "Stream")));
        }, nb::lock_self());

    // -- CapturedGraph: move-only owner of the captured cudaGraph_t -----------
    nb::class_<PyCaptured>(m, "CapturedGraph")
        .def("__bool__", [](const PyCaptured& c) {
            if (c.h == nullptr)
                return false;
            std::int32_t valid = 0;
            ck(EB("captured", eagle_backend_captured_is_valid)(c.h, &valid));
            return valid != 0;
        }, nb::lock_self())
        .def("debug_dot", &capturedDebugDot, nb::lock_self(), nb::arg("path"),
            nb::arg("flags") = 0u,
            "Write this captured graph to Graphviz dot at `path` and return "
            "the text (matches cupy Graph.debug_dot_str).");

    // -- StreamCapturer: begin()/end() around the captured launches ----------
    nb::class_<PyCapturer>(m, "StreamCapturer")
        .def(
            "__init__",
            [](PyCapturer* self, std::uintptr_t stream, int device) {
                auto d = makeDesc<eagle_backend_capturer_desc>(device);
                d.stream = stream;
                eagle_backend_capturer h = nullptr;
                ck(EB("capturer", eagle_backend_capturer_create)(&d, &h));
                new (self) PyCapturer();
                self->h = h;
            },
            nb::arg("stream"), nb::arg("device") = -1)
        .def("begin", [](PyCapturer& c) {
            ck(EB("capturer", eagle_backend_capturer_begin)(need(c.h, "StreamCapturer")));
        }, nb::lock_self())
        .def(
            "end",
            [](PyCapturer& c) {
                eagle_backend_captured g = nullptr;
                ck(EB("capturer", eagle_backend_capturer_end)(need(c.h, "StreamCapturer"), &g));
                auto* out = new PyCaptured();
                out->h = g;
                return out;
            }, nb::lock_self(),
            nb::rv_policy::take_ownership,
            "End capture and return a CapturedGraph owning the cudaGraph_t.");

    // -- CaptureFork: sibling branches inside one capture --------------------
    // Mirrors the C++ contract exactly: forking grants the scheduler PERMISSION
    // to co-execute the branches, never an obligation, and the relative order
    // among branches is unobservable. Branch streams are handed to Python as
    // raw integer handles (same convention as Stream.ptr()), to be wrapped in
    // cupy.cuda.ExternalStream — the caller issues the branch work, exactly as
    // it issues the origin-stream work.
    nb::class_<PyFork>(m, "CaptureFork")
        .def(
            "__init__",
            [](PyFork* self, std::uintptr_t origin, std::size_t branch_count, int device) {
                auto d = makeDesc<eagle_backend_fork_desc>(device);
                d.origin = origin;
                d.branch_count = branch_count;
                eagle_backend_fork h = nullptr;
                ck(EB("fork", eagle_backend_fork_create)(&d, &h));
                new (self) PyFork();
                self->h = h;
            },
            nb::arg("origin"), nb::arg("branch_count"), nb::arg("device") = -1,
            "Create branch streams/events for a later fork. MUST be "
            "constructed BEFORE StreamCapturer.begin(): stream and event "
            "creation is illegal while a capture is in flight.")
        .def("fork", [](PyFork& f) { ck(EB("fork", eagle_backend_fork_fork)(need(f.h, "CaptureFork"))); }, nb::lock_self(),
            "Open the fork: every branch becomes a sibling of every other, and "
            "everything captured so far becomes a predecessor of all branches. "
            "Calling twice is a no-op.")
        .def("join", [](PyFork& f) { ck(EB("fork", eagle_backend_fork_join)(need(f.h, "CaptureFork"))); }, nb::lock_self(),
            "Close the fork: the origin waits for every branch, so work issued "
            "after the join depends on all of them. Idempotent. An unjoined "
            "branch makes StreamCapturer.end() fail and discard the graph.")
        .def(
            "branch",
            [](const PyFork& f, std::size_t index) {
                std::uint64_t s = 0;
                ck(EB("fork", eagle_backend_fork_branch)(need(f.h, "CaptureFork"), index, &s));
                return static_cast<std::uintptr_t>(s);
            }, nb::lock_self(),
            nb::arg("index"),
            "Raw cudaStream_t of branch `index` as a Python int (wrap with "
            "cupy.cuda.ExternalStream). Only meaningful between fork() and "
            "join().")
        .def(
            "origin",
            [](const PyFork& f) {
                std::uint64_t s = 0;
                ck(EB("fork", eagle_backend_fork_origin)(need(f.h, "CaptureFork"), &s));
                return static_cast<std::uintptr_t>(s);
            }, nb::lock_self(),
            "Raw cudaStream_t of the origin stream this fork branches from.")
        .def(
            "forked",
            [](const PyFork& f) {
                std::int32_t v = 0;
                ck(EB("fork", eagle_backend_fork_forked)(need(f.h, "CaptureFork"), &v));
                return v != 0;
            }, nb::lock_self(),
            "True once fork() has run and join() has not.")
        .def("__len__",
            [](const PyFork& f) {
                std::uint64_t n = 0;
                ck(EB("fork", eagle_backend_fork_size)(need(f.h, "CaptureFork"), &n));
                return static_cast<std::size_t>(n);
            }, nb::lock_self())
        .def(
            "size",
            [](const PyFork& f) {
                std::uint64_t n = 0;
                ck(EB("fork", eagle_backend_fork_size)(need(f.h, "CaptureFork"), &n));
                return static_cast<std::size_t>(n);
            }, nb::lock_self(),
            "Number of branches.");

    // -- CaptureConditional: weave a device-evaluated IF node into capture ---
    // The Python surface's mechanism (see eagle._conditional):
    // SkipGuard computes two raw device uint32 pointers (count/baseline,
    // CountGuard, 0 == nullptr == "no baseline") and GraphPipeline.build
    // constructs one of these per Skippable, in the same pre-capture pass
    // that builds CaptureFork -- both create resources illegal mid-capture.
    // A RepeatWhile (eagle._conditional) adds the two loop keywords:
    // `loop_cap` (>= 1, frozen at build) and `counter_ptr` (the raw device
    // pointer of its uint32[2] [remaining, ran] cell). counter_ptr == 0 keeps
    // the IF kind, so every existing 3-argument call site is unchanged.
    nb::class_<PyConditional>(m, "CaptureConditional")
        .def(
            "__init__",
            [](PyConditional* self, std::uintptr_t origin, std::uintptr_t count_ptr, std::uintptr_t baseline_ptr,
                unsigned int loop_cap, std::uintptr_t counter_ptr, int device) {
                auto d = makeDesc<eagle_backend_conditional_desc>(device);
                d.origin = origin;
                d.count_ptr = count_ptr;
                d.baseline_ptr = baseline_ptr;
                d.loop_cap = loop_cap;
                d.counter_ptr = counter_ptr;
                eagle_backend_conditional h = nullptr;
                ck(EB("conditional", eagle_backend_conditional_create)(&d, &h));
                new (self) PyConditional();
                self->h = h;
            },
            nb::arg("origin"), nb::arg("count_ptr"), nb::arg("baseline_ptr") = 0,
            nb::arg("loop_cap") = 0, nb::arg("counter_ptr") = 0, nb::arg("device") = -1,
            "Create the guarded region's body stream. MUST be constructed "
            "BEFORE StreamCapturer.begin() -- stream creation is illegal "
            "while a capture is in flight on the calling thread. `origin` is the "
            "stream the region would have been captured on directly (the "
            "pipeline's main stream, a CaptureFork branch, or another "
            "CaptureConditional's body_stream()); `count_ptr` / "
            "`baseline_ptr` are raw device uint32 pointers (0 == nullptr == "
            "no baseline, reads as 0). With `counter_ptr` (raw device pointer "
            "of a uint32[2] [remaining, ran] cell) the weave is a WHILE loop "
            "capped at `loop_cap` (>= 1) iterations per replay; without it, "
            "an IF.")
        .def(
            "begin",
            [](PyConditional& c) {
                std::uint64_t s = 0;
                ck(EB("conditional", eagle_backend_conditional_begin)(need(c.h, "CaptureConditional"), &s));
                return static_cast<std::uintptr_t>(s);
            }, nb::lock_self(),
            "Weave the setter kernel + conditional node (IF, or WHILE for a "
            "loop) at the origin's current capture tip and open capture of "
            "the body on the pre-created body stream. Returns the body "
            "stream as a raw int -- wrap with cupy.cuda.ExternalStream and "
            "launch the guarded region's work on it.")
        .def(
            "body_stream",
            [](const PyConditional& c) {
                std::uint64_t s = 0;
                ck(EB("conditional", eagle_backend_conditional_body_stream)(need(c.h, "CaptureConditional"), &s));
                return static_cast<std::uintptr_t>(s);
            }, nb::lock_self(),
            "The pre-created body stream as a raw int, valid from "
            "construction: the `origin` a NESTED weave (a skippable region "
            "inside a loop body) must be constructed against, before "
            "capture begins.")
        .def(
            "is_loop",
            [](const PyConditional& c) {
                std::int32_t v = 0;
                ck(EB("conditional", eagle_backend_conditional_is_loop)(need(c.h, "CaptureConditional"), &v));
                return v != 0;
            }, nb::lock_self(),
            "True for a WHILE (loop) weave, False for an IF weave.")
        .def(
            "end",
            [](PyConditional& c) {
                ck(EB("conditional", eagle_backend_conditional_end)(need(c.h, "CaptureConditional")));
            }, nb::lock_self(),
            "Close the body capture. Idempotent, dtor-safe.");

    // -- Capture attribution (member-enable "mode=enabled"): mid- ------
    // capture node-set snapshotting + toggleability classification. Free
    // functions (no owned state), thin wrappers 1:1 over
    // eagle::cuda::CaptureAttribution.h -- GraphPipeline.build() calls
    // capture_snapshot_nodes() immediately before and after each registered
    // member's launches and diffs the two snapshots to attribute that
    // member's contributed nodes, then classifies each with
    // is_node_toggleable() to decide whether the member may later be
    // toggled via Launcher.set_node_enabled(). Deliberately never touches
    // cudaGraphKernelNodeGetParams (the templated-kernel trap).
    m.def(
        "capture_snapshot_nodes",
        [](std::uintptr_t stream) {
            std::uint64_t* nodes = nullptr;
            std::int64_t count = 0;
            ck(EB("attribution", eagle_backend_capture_snapshot_nodes)(stream, &nodes, &count));
            std::vector<std::uintptr_t> out(nodes, nodes + count);
            cuda().meta().free(nodes);
            return out;
        },
        nb::arg("stream"),
        "Node handles (as ints) of the graph currently being captured "
        "into, as seen from `stream` -- which must be part of an ACTIVE "
        "ThreadLocal-mode capture (the pipeline's main stream, or a "
        "CaptureFork branch). Diff two snapshots taken immediately before "
        "and after one member's launches to attribute its contributed "
        "nodes.");
    m.def(
        "is_node_toggleable",
        [](std::uintptr_t node) {
            std::int32_t v = 0;
            ck(EB("attribution", eagle_backend_is_node_toggleable)(node, &v));
            return v != 0;
        },
        nb::arg("node"),
        "Whether `node` (a handle from capture_snapshot_nodes()) is a "
        "kernel, memcpy or memset node -- the only types "
        "Launcher.set_node_enabled() may toggle (CUDA's own supported "
        "set for cudaGraphNodeSetEnabled). False for e.g. a "
        "CaptureConditional IF node.");

    // -- Capture-guard introspection (STOP-THE-LINE incident fix, test/ ------
    // diagnostic only): eagle::util::CaptureGuardState's process-wide
    // capture-depth counter and deferred-destroy queue length, which live in
    // the backend (every CUDA-touching teardown happens there). See
    // eagle/util/CaptureGuard.h for the full incident writeup. Not part of the
    // product surface; exists so a red-proof test can observe the queue
    // draining without racing on stderr text alone.
    m.def(
        "capture_guard_depth",
        []() {
            std::int64_t v = 0;
            ck(EBX(eagle_backend_x_capture_guard_depth)(&v));
            return static_cast<int>(v);
        },
        "Capture-depth counter of the calling thread (test/diagnostic only).");
    m.def(
        "capture_guard_pending_count",
        []() {
            std::int64_t v = 0;
            ck(EBX(eagle_backend_x_capture_guard_pending_count)(&v));
            return static_cast<std::size_t>(v);
        },
        "Number of CUDA-touching teardowns currently deferred because a "
        "capture is open on the calling thread (test/diagnostic only).");

    // -- Device properties: a plain dict, -
    // every eagle::DeviceProps field, raw and derived. ridge_flops_per_byte is
    // dtype-parametric in C++ (eagle::DeviceProps::ridgeFlopsPerByte), so both
    // precomputed values are exposed as separate keys -- the Python
    // eagle._device_props.DeviceProps view selects between them by dtype
    // name, which is dispatch, never arithmetic (the "no Python-side
    // computation" contract this binding exists to uphold).
    m.def(
        "device_props",
        [](int device) {
            struct eagle_backend_device_props p {};
            p.struct_size = sizeof p;
            ck(EB("device", eagle_backend_device_props)(device, &p));
            p.name[sizeof p.name - 1] = '\0';
            nb::dict d;
            d["name"]                     = std::string(p.name);
            d["cc_major"]                 = p.cc_major;
            d["cc_minor"]                 = p.cc_minor;
            d["sm_count"]                 = p.sm_count;
            d["clock_rate_khz"]           = p.clock_rate_khz;
            d["memory_clock_rate_khz"]    = p.memory_clock_rate_khz;
            d["memory_bus_width_bits"]    = p.memory_bus_width_bits;
            d["shared_mem_per_block"]     = static_cast<std::size_t>(p.shared_mem_per_block);
            d["shared_mem_per_sm"]        = static_cast<std::size_t>(p.shared_mem_per_sm);
            d["regs_per_block"]           = p.regs_per_block;
            d["regs_per_sm"]              = p.regs_per_sm;
            d["warp_size"]                = p.warp_size;
            d["peak_bytes_per_s"]         = p.peak_bytes_per_s;
            d["peak_flops_sp"]            = p.peak_flops_sp;
            d["peak_flops_dp"]            = p.peak_flops_dp;
            d["fp64_ratio"]               = p.fp64_ratio;
            d["ridge_flops_per_byte_sp"]  = p.ridge_flops_per_byte_sp;
            d["ridge_flops_per_byte_dp"]  = p.ridge_flops_per_byte_dp;
            return d;
        },
        nb::arg("device") = 0,
        "eagle::cuda::deviceProps(device) as a plain dict -- every struct "
        "field, raw and derived, with ZERO Python-side computation "
        "(see eagle/DeviceProps.h for every "
        "formula). ridge_flops_per_byte is dtype-parametric in C++, so both "
        "precomputed values are exposed (ridge_flops_per_byte_sp/_dp); "
        "eagle._device_props.DeviceProps.ridge_flops_per_byte(dtype) selects "
        "between them by dtype name.");

    // -- Launcher: instantiated exec graph; replay ---------------------------
    nb::class_<PyLauncher>(m, "Launcher")
        .def(
            "launch",
            [](PyLauncher& l) { ck(EB("launcher", eagle_backend_launcher_launch)(need(l.h, "Launcher"))); }, nb::lock_self(),
            "Replay the instantiated exec graph once. Raises RuntimeError if "
            "cudaGetLastError() is nonzero after the replay; see "
            "eagle/python/eagle/pipeline.py's "
            "launch() for the companion stream-ordering fix).")
        .def("synchronize",
            [](PyLauncher& l) { ck(EB("launcher", eagle_backend_launcher_synchronize)(need(l.h, "Launcher"))); }, nb::lock_self())
        .def(
            "stream",
            [](PyLauncher& l, std::uintptr_t s) {
                ck(EB("launcher", eagle_backend_launcher_stream)(need(l.h, "Launcher"), s));
            }, nb::lock_self(),
            nb::arg("stream"))
        .def(
            "set_logical_size",
            [](PyLauncher& l, std::int64_t logical_size) {
                ck(EB("launcher", eagle_backend_launcher_set_logical_size)(need(l.h, "Launcher"), logical_size));
            }, nb::lock_self(),
            nb::arg("logical_size"),
            "Patch every kernel node's grid/block for a new logical size.")
        .def(
            "kernel_node_count",
            [](const PyLauncher& l) {
                std::int64_t n = 0;
                ck(EB("launcher", eagle_backend_launcher_kernel_node_count)(need(l.h, "Launcher"), &n));
                return n;
            }, nb::lock_self(),
            "Number of harvested kernel-node records (cross-check for "
            "num_nodes()).")
        .def(
            "set_node_enabled",
            [](PyLauncher& l, std::uintptr_t node, bool enabled) {
                ck(EB("launcher", eagle_backend_launcher_set_node_enabled)(need(l.h, "Launcher"), node,
                    enabled ? 1 : 0));
            }, nb::lock_self(),
            nb::arg("node"), nb::arg("enabled"),
            "Enable/disable one node (by raw handle, as returned by "
            "capture_snapshot_nodes()) in this Launcher's instantiated "
            "exec graph. Legal between replays; no recapture, no "
            "structure change (mode=\"enabled\"). "
            "Raises RuntimeError (name+code) if the node's type "
            "is not one CUDA supports toggling -- check "
            "is_node_toggleable() ahead of time.");

    // -- Graph: parent container; folds in the capture, instantiates ---------
    nb::class_<PyGraph>(m, "Graph")
        .def(
            "__init__",
            [](PyGraph* self, int device) {
                auto d = makeDesc<eagle_backend_graph_desc>(device);
                eagle_backend_graph h = nullptr;
                ck(EB("graph", eagle_backend_graph_create)(&d, &h));
                new (self) PyGraph();
                self->h = h;
            },
            nb::arg("device") = -1)
        .def_static(
            "from_captured",
            [](PyCaptured& c) {
                // Defense in depth: a falsy CapturedGraph wraps a null
                // cudaGraph_t — the shape a failed/aborted capture leaves
                // behind (or a handle an earlier call already consumed).
                // Refuse here, naming the likely reason.
                if (c.h == nullptr)
                    throw std::invalid_argument(
                        "Graph.from_captured: the CapturedGraph is empty (null "
                        "cudaGraph_t) and cannot be adopted. The capture that "
                        "produced it did not yield a graph — typically because "
                        "it was never begun, was already consumed by an earlier "
                        "from_captured/add_node call, or ended in a failed "
                        "state (e.g. a forked branch that was never joined).");
                eagle_backend_graph h = nullptr;
                ck(EB("graph", eagle_backend_graph_from_captured)(c.h, &h));
                c.h = nullptr; // consumed
                auto* g = new PyGraph();
                g->h = h;
                return g;
            },
            nb::arg("captured").lock(), nb::rv_policy::take_ownership,
            "Adopt a CapturedGraph as the source graph and instantiate it "
            "directly (no clone, no kernel harvest) — the single-sequence "
            "replay path. Consumes the CapturedGraph.")
        .def(
            "stream",
            [](PyGraph& g, std::uintptr_t s) { ck(EB("graph", eagle_backend_graph_stream)(need(g.h, "Graph"), s)); }, nb::lock_self(),
            nb::arg("stream"))
        .def(
            "add_node",
            [](PyGraph& g, PyCaptured& c) {
                ck(EB("graph", eagle_backend_graph_add_node)(need(g.h, "Graph"), need(c.h, "CapturedGraph")));
                c.h = nullptr; // consumed
            }, nb::lock_self(),
            nb::arg("captured").lock(),
            "Fold a CapturedGraph in as a child-graph node (consumes it) and "
            "harvest its kernel records.")
        .def(
            "launcher",
            [](const PyGraph& g) {
                eagle_backend_launcher h = nullptr;
                ck(EB("graph", eagle_backend_graph_launcher)(need(g.h, "Graph"), &h));
                auto* l = new PyLauncher();
                l->h = h;
                return l;
            }, nb::lock_self(),
            nb::rv_policy::take_ownership,
            "Instantiate an exec graph and return a Launcher. Raises "
            "RuntimeError if cudaGetLastError() is nonzero after instantiate.")
        .def("last_node", [](const PyGraph& g) {
            std::int64_t i = 0;
            ck(EB("graph", eagle_backend_graph_last_node)(need(g.h, "Graph"), &i));
            return i;
        }, nb::lock_self());

    // -- GraphComposer: the C++-native parity surface --
    // Experimental at the seam (eagle_backend_x_composer_*): it is bound one
    // symbol at a time and raises BackendUnavailable naming the symbol on a
    // backend without it. A Launcher member is CONSUMED (its Python object is
    // left moved-from, any later use raises); a nested composer is SHARED (the
    // backend co-owns it, and this object keeps the nested Python object alive,
    // with the callables it registered). A raw callable member's target stream
    // crosses as a uintptr_t (same convention as Stream.ptr() /
    // CaptureFork.branch() above).
    nb::class_<PyComposer>(m, "GraphComposer")
        .def(
            "__init__",
            [](PyComposer* self, const std::string& mode, int device) {
                auto d = makeDesc<eagle_backend_composer_desc>(device);
                d.mode = mode.c_str();
                eagle_backend_composer h = nullptr;
                ck(EBX(eagle_backend_x_composer_create)(&d, &h));
                new (self) PyComposer();
                self->h = h;
            },
            nb::arg("mode") = "sequenced", nb::arg("device") = -1,
            "Construct an empty composer. mode is one of \"sequenced\" "
            "(default), \"enabled\", \"rebuild\" -- \"conditional\" is "
            "DEFERRED on this C++-native engine (compose.py's own "
            "GraphComposer keeps it); raises ValueError naming the "
            "deferral if requested here.")
        .def(
            "register_launcher",
            [](PyComposer& c, PyLauncher& launcher, std::optional<std::string> name, nb::object pre_launch) {
                const bool hasPre = !pre_launch.is_none();
                std::int64_t index = 0;
                ck(EBX(eagle_backend_x_composer_register_launcher)(need(c.h, "GraphComposer"),
                    need(launcher.h, "Launcher"), name ? name->c_str() : nullptr,
                    hasPre ? &voidTrampoline : nullptr, hasPre ? pre_launch.ptr() : nullptr, &index));
                launcher.h = nullptr; // consumed
                if (hasPre)
                    c.keep.push_back(pre_launch);
                return index;
            }, nb::lock_self(),
            nb::arg("launcher").lock(), nb::arg("name") = nb::none(),
            nb::arg("pre_launch") = nb::none(),
            "Register a built Launcher as a member (mode=\"sequenced\" "
            "only). Consumes `launcher` (move-only, C++ side) -- the "
            "Python Launcher object is left in its moved-from state, "
            "exactly like eagle::cuda::Graph's own move-only handling "
            "elsewhere in this binding. LIFETIME CONTRACT: the composer "
            "owns the moved Launcher (and its exec), but NOT the capture "
            "SOURCE it was built from -- the caller must keep the source "
            "GraphPipeline (or equivalent capture machinery) alive for as "
            "long as this composer launches. C++ RAII scoping and "
            "eagle.GraphComposer (compose.py, which retains registered "
            "members) both satisfy this naturally; only this raw binding "
            "can violate it, and violating it is undefined behavior "
            "(probe-verified 2026-08-04: launch after source teardown "
            "faults). Returns the member's registration-order index.")
        .def(
            "register_callable",
            [](PyComposer& c, nb::object step, std::optional<std::string> name, nb::object pre_launch) {
                const bool hasPre = !pre_launch.is_none();
                std::int64_t index = 0;
                ck(EBX(eagle_backend_x_composer_register_callable)(need(c.h, "GraphComposer"), &stepTrampoline,
                    step.ptr(), name ? name->c_str() : nullptr, hasPre ? &voidTrampoline : nullptr,
                    hasPre ? pre_launch.ptr() : nullptr, &index));
                c.keep.push_back(step);
                if (hasPre)
                    c.keep.push_back(pre_launch);
                return index;
            }, nb::lock_self(),
            nb::arg("step"), nb::arg("name") = nb::none(),
            nb::arg("pre_launch") = nb::none(),
            "Register a raw, kernel-issuing member: `step(stream: int)` "
            "must issue every kernel it owns onto exactly the stream "
            "handle it is given (wrap with cupy.cuda.ExternalStream). "
            "Valid in every mode. Returns the member's registration-order "
            "index.")
        .def(
            "register_nested",
            [](PyComposer& c, nb::object nested, std::optional<std::string> name, nb::object pre_launch) {
                PyComposer& n = nb::cast<PyComposer&>(nested);
                const bool hasPre = !pre_launch.is_none();
                std::int64_t index = 0;
                ck(EBX(eagle_backend_x_composer_register_nested)(need(c.h, "GraphComposer"),
                    need(n.h, "GraphComposer"), name ? name->c_str() : nullptr,
                    hasPre ? &voidTrampoline : nullptr, hasPre ? pre_launch.ptr() : nullptr, &index));
                c.keep.push_back(nested);
                if (hasPre)
                    c.keep.push_back(pre_launch);
                return index;
            }, nb::lock_self(),
            nb::arg("nested").lock(), nb::arg("name") = nb::none(),
            nb::arg("pre_launch") = nb::none(),
            "Register an already-built() mode=\"sequenced\" GraphComposer "
            "as a member of THIS mode=\"sequenced\" composer -- the "
            "RECURSION LOCK. Returns the member's registration-order "
            "index.")
        .def(
            "build", [](PyComposer& c) { ck(EBX(eagle_backend_x_composer_build)(need(c.h, "GraphComposer"))); }, nb::lock_self(),
            "Perform whatever one-time setup this composer's mode needs "
            "(capture, for the two flat modes; bookkeeping only for "
            "mode=\"sequenced\"). May be called at most once.")
        .def(
            "set_routing",
            [](PyComposer& c, const std::vector<bool>& pattern) {
                std::vector<std::uint8_t> p(pattern.begin(), pattern.end());
                ck(EBX(eagle_backend_x_composer_set_routing)(need(c.h, "GraphComposer"), p.data(),
                    static_cast<std::int64_t>(p.size())));
            }, nb::lock_self(),
            nb::arg("pattern"),
            "Update which members are active. Mechanism + cost depend on "
            "mode -- see eagle/compose/GraphComposer.h's class doc.")
        .def(
            "launch",
            [](PyComposer& c, int n) { ck(EBX(eagle_backend_x_composer_launch)(need(c.h, "GraphComposer"), n)); }, nb::lock_self(),
            nb::arg("n") = 1, "Replay n times.")
        .def(
            "fired_history",
            [](const PyComposer& c) {
                std::int64_t* flat = nullptr;
                std::int64_t* offsets = nullptr;
                std::int64_t calls = 0;
                ck(EBX(eagle_backend_x_composer_fired_history)(need(c.h, "GraphComposer"), &flat, &offsets, &calls));
                std::vector<std::vector<std::int64_t>> out(static_cast<std::size_t>(calls));
                for (std::int64_t k = 0; k < calls; ++k)
                    out[static_cast<std::size_t>(k)].assign(flat + offsets[k], flat + offsets[k + 1]);
                cuda().meta().free(flat);
                cuda().meta().free(offsets);
                return out;
            }, nb::lock_self(),
            "Per-launch()-call fired-member index lists, oldest first "
            "(list[list[int]] -- eagle._core's parity surface; compose.py's "
            "own GraphComposer returns frozenset per call instead).")
        .def("reset_fired_history",
            [](PyComposer& c) { ck(EBX(eagle_backend_x_composer_reset_fired_history)(need(c.h, "GraphComposer"))); }, nb::lock_self())
        .def_prop_ro(
            "mode",
            [](const PyComposer& c) {
                const char* mode = nullptr;
                ck(EBX(eagle_backend_x_composer_mode)(need(c.h, "GraphComposer"), &mode));
                return std::string(mode ? mode : "");
            }, nb::lock_self())
        .def("member_names",
            [](const PyComposer& c) {
                std::int64_t n = 0;
                ck(EBX(eagle_backend_x_composer_num_members)(need(c.h, "GraphComposer"), &n));
                std::vector<std::string> out;
                for (std::int64_t i = 0; i < n; ++i) {
                    const char* name = nullptr;
                    ck(EBX(eagle_backend_x_composer_member_name)(c.h, i, &name));
                    out.emplace_back(name ? name : "");
                }
                return out;
            }, nb::lock_self())
        .def("num_members", [](const PyComposer& c) {
            std::int64_t n = 0;
            ck(EBX(eagle_backend_x_composer_num_members)(need(c.h, "GraphComposer"), &n));
            return n;
        }, nb::lock_self());

    // -- The graph view: Description, edge derivation, layout, renderers, and
    // the busy guard behind eagle.Busy. All of it is the header-only C++ of
    // eagle/graph/ and eagle/util/Busy.h; this is the thin layer over it.
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const eagle::util::Busy& e) {
            nb::object type = nb::module_::import_("eagle._busy").attr("Busy");
            PyErr_SetString(type.ptr(), e.what());
        }
    });
    m.attr("OVERLAP_MAX_WORK") = eg::kOverlapMaxWork;
    m.def("kind_meaning", [](const std::string& kind) { return std::string(eg::kindMeaning(eg::parseEdgeKind(kind))); },
        nb::arg("kind"), "Plain-words meaning of a hazard kind ('RAW', 'WAR', 'WAW').");

    nb::class_<PyDescription>(m, "GraphDescription")
        .def("__init__",
            [](PyDescription* self, const std::vector<NodeRow>& nodes,
                const std::vector<EdgeRow>& edges, const std::vector<FrameRow>& frames) {
                new (self) PyDescription();
                eg::Description& d = self->d;
                for (const auto& [label, kind, stage, frame, opaque] : nodes)
                    d.nodes.push_back({ eg::gidx(d.nodes.size()), label, kind, stage, frame, opaque });
                for (const auto& [src, dst, plane, kind] : edges)
                    d.edges.push_back({ src, dst, plane, eg::parseEdgeKind(kind) });
                for (const auto& [kind, label, parent] : frames)
                    d.frames.push_back({ eg::gidx(d.frames.size()), kind, label, parent });
                d.validate();
            },
            nb::arg("nodes"), nb::arg("edges") = std::vector<EdgeRow>{},
            nb::arg("frames") = std::vector<FrameRow>{},
            "nodes: (label, kind, stage, frame, opaque); edges: (src, dst, plane, kind "
            "'RAW'|'WAR'|'WAW'); frames: (kind, label, parent). Ids are positions; "
            "frame/parent -1 means none. Raises ValueError on a malformed graph.")
        .def("validate", [](const PyDescription& s) { s.d.validate(); })
        .def("num_nodes", [](const PyDescription& s) { return s.d.nodes.size(); })
        .def("num_edges", [](const PyDescription& s) { return s.d.edges.size(); })
        .def("layout",
            [](const PyDescription& s, int sweeps) {
                const eg::Layout l = eg::layout(s.d, sweeps);
                std::vector<std::pair<double, double>> xy;
                for (const auto& p : l.xy) xy.emplace_back(p[0], p[1]);
                return std::make_tuple(l.layer, l.order, xy);
            },
            nb::arg("sweeps") = 4,
            "(layer, order, xy) per node: layer = stage, order = position in it, "
            "xy = unit-grid coordinates (x centred per layer, y = layer, downwards).")
        .def("ascii",
            [](const PyDescription& s, bool unicode, std::size_t max_label) {
                return eg::ascii(s.d, eg::layout(s.d), unicode, max_label);
            },
            nb::arg("unicode") = true, nb::arg("max_label") = 24,
            "The box block diagram (unicode=False: + - | only).")
        .def("mermaid", [](const PyDescription& s) { return eg::mermaid(s.d); })
        .def("dot", [](const PyDescription& s) { return eg::dot(s.d); });

    m.def("derive_edges",
        [](const std::vector<RefRow>& refs,
            const std::vector<std::vector<std::vector<std::int64_t>>>& groups) {
            std::vector<EdgeRow> out;
            for (const eg::Edge& e : eg::deriveEdges(planeRefsOf(refs), groupsOf(groups)))
                out.emplace_back(e.src, e.dst, e.plane, eg::kindName(e.kind));
            return out;
        },
        nb::arg("plane_refs"), nb::arg("groups") = std::vector<std::vector<std::vector<std::int64_t>>>{},
        "Hazard edges between nodes from the planes they touch, stride-aware. "
        "plane_refs: (name, node, base, shape, strides_bytes, itemsize, access "
        "0 read | 1 write | 2 in place). groups: per fork/join group, its "
        "branches, each a list of node ids; siblings get no edge.");
    m.def("sibling_conflicts",
        [](const std::vector<RefRow>& refs,
            const std::vector<std::vector<std::vector<std::int64_t>>>& groups) {
            std::vector<std::tuple<std::int64_t, std::int64_t, std::string, std::string, std::string>> out;
            for (const eg::Conflict& c : eg::siblingConflicts(planeRefsOf(refs), groupsOf(groups)))
                out.emplace_back(c.a, c.b, c.planeA, c.planeB, eg::kindName(c.kind));
            return out;
        },
        nb::arg("plane_refs"), nb::arg("groups"),
        "Overlaps between siblings of one fork/join group with a writer: "
        "(a, b, plane_a, plane_b, kind).");
    m.def("planes_overlap",
        [](const RefRow& a, const RefRow& b) {
            return eg::detail::overlap(planeRefOf(a), planeRefOf(b));
        },
        nb::arg("a"), nb::arg("b"), "Do two plane references share a byte?");

    // -- The reverse of a graph (eagle/diff/Transpose.h): eagle.diff plans its
    // derivative steps, summing joins and snapshots here.
    using VjpRow = std::tuple<std::int64_t, std::vector<std::string>, std::vector<std::string>>;
    auto vjpsOf = [](const std::vector<VjpRow>& rows) {
        std::vector<eagle::diff::NodeVjp> out;
        for (const auto& [node, reads, writes] : rows) out.push_back({ node, reads, writes });
        return out;
    };
    auto snapshotRows = [](const std::vector<eagle::diff::Snapshot>& s) {
        std::vector<std::pair<std::int64_t, std::string>> out;
        for (const auto& x : s) out.emplace_back(x.node, x.plane);
        return out;
    };
    m.attr("DIFF_SEED") = eagle::diff::kSeed;
    m.attr("DIFF_ZERO") = eagle::diff::kZero;
    m.def("diff_snapshots",
        [=](const std::vector<VjpRow>& vjps) { return snapshotRows(eagle::diff::snapshots(vjpsOf(vjps))); },
        nb::arg("vjps"),
        "The planes a reverse needs copied during the forward: (node, plane) = copy `plane` just "
        "before forward node `node` runs. vjps: (node, reads, writes) per node, in node order.");
    m.def("diff_transpose",
        [=](const PyDescription& forward, const std::vector<VjpRow>& vjps, const std::vector<std::string>& seeded) {
            const eagle::diff::ReversePlan p = eagle::diff::transpose(forward.d, vjpsOf(vjps), seeded);
            std::vector<NodeRow> nodes;
            for (const eg::Node& n : p.reversed.nodes) nodes.emplace_back(n.label, n.kind, n.stage, n.frame, n.opaque);
            std::vector<EdgeRow> edges;
            for (const eg::Edge& e : p.reversed.edges) edges.emplace_back(e.src, e.dst, e.plane, eg::kindName(e.kind));
            std::vector<std::tuple<std::int64_t, std::string, std::vector<std::int64_t>>> joins;
            for (const auto& j : p.joins) joins.emplace_back(j.target, j.plane, j.sources);
            nb::dict out;
            out["nodes"] = nodes;
            out["edges"] = edges;
            out["forward"] = p.forward;
            out["joins"] = joins;
            out["snapshots"] = snapshotRows(p.snapshots);
            out["seed_of"] = p.seedOf;
            out["read_of"] = p.readOf;
            out["gradients"] = p.gradients;
            return out;
        },
        nb::arg("forward"), nb::arg("vjps"), nb::arg("seeded"),
        "The reverse of `forward` (a GraphDescription) as a dict: nodes/edges of the reverse "
        "(run in order), forward (its forward node, -1 for a join), joins (target, plane, "
        "sources in summing order), snapshots, seed_of / read_of per reverse node, and gradients "
        "(plane, source) per graph input. Sources: a reverse node, DIFF_SEED or DIFF_ZERO. "
        "Raises ValueError naming a node that cannot be reversed.");
    m.def("ordered_sum",
        [](std::uintptr_t dst, const std::vector<std::uintptr_t>& src, std::int64_t n, std::uint32_t dtype) {
            auto go = [&](auto tag) {
                using T = decltype(tag);
                std::vector<const T*> in;
                for (std::uintptr_t a : src) in.push_back(reinterpret_cast<const T*>(a));
                eagle::diff::orderedSum<T>(reinterpret_cast<T*>(dst), in, n);
            };
            if (dtype == EAGLE_BACKEND_REDUCE_F64) go(double{});
            else if (dtype == EAGLE_BACKEND_REDUCE_F32) go(float{});
            else throw std::invalid_argument("ordered_sum: dtype must be 0 (f32) or 1 (f64)");
        },
        nb::arg("dst"), nb::arg("src"), nb::arg("n"), nb::arg("dtype"),
        "dst = ((src[0] + src[1]) + ...) elementwise over n host elements, in exactly this order: "
        "the same bits for every OpenMP thread count. dtype: 0 f32, 1 f64 (the reduce codes).");

    // -- The windowed reverse of a stepped run (eagle/diff/{Window,Record,Reverse}.h).
    m.def("diff_default_window", &eagle::diff::defaultWindow, nb::arg("steps"),
        "The default reverse window for runs of up to `steps` steps: clamp(ceil(sqrt(steps)), 16, 1024).");
    nb::class_<eagle::diff::RecordSlot>(m, "RecordSlot")
        .def(nb::init<>())
        .def("begin", [](eagle::diff::RecordSlot& r) { return static_cast<int>(r.begin()); },
             "Start a recording: 0 ok (generation moves on), 1 a reverse holds the last record, "
             "2 the last record is still live.")
        .def("abandon", &eagle::diff::RecordSlot::abandon)
        .def("hold", &eagle::diff::RecordSlot::hold, nb::arg("generation"))
        .def("release", &eagle::diff::RecordSlot::release, nb::arg("generation"), nb::arg("consumed"))
        .def_prop_ro("generation", &eagle::diff::RecordSlot::generation)
        .def_prop_ro("live", &eagle::diff::RecordSlot::live)
        .def_prop_ro("held", &eagle::diff::RecordSlot::held);
    m.def("diff_stop_order",
        [](std::uintptr_t steps, std::int64_t n) {
            return eagle::diff::stopOrder(reinterpret_cast<const std::int64_t*>(steps), n);
        },
        nb::arg("steps"), nb::arg("n"),
        "The reverse's sample order: samples sorted by stop step, longest first (stable). "
        "`steps` is the address of n int64 host values.");
    m.def("diff_live_counts",
        [](std::uintptr_t steps, std::int64_t n) {
            return eagle::diff::liveCounts(reinterpret_cast<const std::int64_t*>(steps), n);
        },
        nb::arg("steps"), nb::arg("n"),
        "live[j] = samples whose stop step exceeds j, for j below the longest stop.");
    m.def("diff_chunk_samples", &eagle::diff::chunkSamples, nb::arg("budget_bytes"),
        nb::arg("per_sample_bytes"), nb::arg("n"), nb::arg("leaf"),
        "Samples per reverse pass under a byte budget: n when all fit, else the largest "
        "leaf * 2^q that fits, 0 when not even one leaf block fits.");
    m.def("diff_combine_chunks",
        [](std::vector<double> partials, std::uint32_t dtype) -> double {
            if (partials.empty()) throw std::invalid_argument("diff_combine_chunks: no partials");
            if (dtype == EAGLE_BACKEND_REDUCE_F32) {
                std::vector<float> f(partials.begin(), partials.end());
                return eagle::reduce::canonical::combine<float, aether::SumOp<float>>(f);
            }
            return eagle::reduce::canonical::combine<double, aether::SumOp<double>>(partials);
        },
        nb::arg("partials"), nb::arg("dtype"),
        "The canonical pairwise combine of per-pass sums (in pass order): bitwise the "
        "unchunked canonical sum when every pass but the last holds leaf * 2^q samples.");
    m.def("diff_record_host",
        [](std::uintptr_t entry, const std::vector<std::uintptr_t>& params, std::int64_t n,
           std::int64_t maxSteps, std::int64_t window, std::int64_t maxK, std::uintptr_t word,
           std::uintptr_t finished, const std::function<void(std::int64_t)>& checkpoint,
           const std::optional<std::function<void(std::int64_t)>>& after) {
            const std::vector<void*> p = asParams(params);
            const auto fn = reinterpret_cast<eagle::exec::HostEntryV2>(entry);
            const auto* count = reinterpret_cast<const std::int32_t*>(finished);
            auto* steps = reinterpret_cast<std::int64_t*>(word);
            nb::gil_scoped_release unlocked;
            return eagle::diff::recordForward(
                maxSteps, window, maxK, [&] { return *count >= n; },
                [&](std::int64_t k) { nb::gil_scoped_acquire locked; checkpoint(k); },
                [&](std::int64_t j, std::int64_t k) {
                    if (steps != nullptr) *steps = k;
                    HostTeam::run(fn, p.data(), Partition::whole(n));
                    if (after) { nb::gil_scoped_acquire locked; (*after)(j + k); }
                });
        },
        nb::arg("entry"), nb::arg("params"), nb::arg("n"), nb::arg("max_steps"),
        nb::arg("window"), nb::arg("max_k"), nb::arg("word"), nb::arg("finished"),
        nb::arg("checkpoint"), nb::arg("after").none() = nb::none(),
        "The recording forward on the host: launches of the host entry that never cross a "
        "window boundary (k <= max_k steps, written to the int64 `word` first unless it is 0) "
        "until the int32 finished count at `finished` reaches n or max_steps, calling "
        "checkpoint(k) before the first launch of window k and after(steps done) after every "
        "launch when given (a compaction). Returns the steps run.");
    using Gather = std::tuple<std::uintptr_t, std::vector<std::uintptr_t>, std::int64_t, std::int64_t>;
    using Accum = std::tuple<std::uintptr_t, std::uintptr_t, std::int64_t, std::uint32_t>;
    m.def("diff_reverse_host",
        [](std::int64_t n, std::int64_t srcN, std::int64_t window, const std::vector<std::int64_t>& live,
           const std::vector<std::uintptr_t>& orders, const std::vector<Gather>& gathers,
           std::uintptr_t recomputeEntry, const std::vector<std::uintptr_t>& recomputeParams,
           std::uintptr_t recomputeWord, std::uintptr_t vjpEntry, const std::vector<std::vector<std::uintptr_t>>& vjpParams,
           const std::vector<Accum>& accums, std::uintptr_t stepPlane, std::int64_t stepBytes) {
            if (static_cast<std::int64_t>(vjpParams.size()) != 2 * window)
                throw std::invalid_argument("diff_reverse_host: vjp_params holds 2 lists per tape row");
            const auto recompute = reinterpret_cast<eagle::exec::HostEntryV2>(recomputeEntry);
            const auto vjp = reinterpret_cast<eagle::exec::HostEntryV2>(vjpEntry);
            auto* word = reinterpret_cast<std::int64_t*>(recomputeWord);
            const std::vector<void*> rp = asParams(recomputeParams);
            std::vector<std::vector<void*>> vp;
            for (const auto& v : vjpParams) vp.push_back(asParams(v));
            const auto prefix = [n](std::int64_t count) { return Partition{ 0, count, n }; };
            const auto at = [&](std::int64_t j) { return live[static_cast<std::size_t>(j)]; };
            nb::gil_scoped_release unlocked;
            eagle::diff::reverseWindows(
                window, live,
                [&](std::int64_t k, std::int64_t count) {
                    for (const auto& [dst, sources, rows, bytes] : gathers)
                        eagle::diff::gatherColumns(reinterpret_cast<void*>(dst), n,
                            reinterpret_cast<const void*>(sources.at(static_cast<std::size_t>(k))), srcN,
                            reinterpret_cast<const std::int64_t*>(orders.at(static_cast<std::size_t>(k))),
                            count, rows, static_cast<std::size_t>(bytes));
                },
                [&](std::int64_t c, std::int64_t len) {
                    if (word != nullptr) {   // one fused launch; each sample stops itself
                        *word = len;
                        HostTeam::run(recompute, rp.data(), prefix(at(c)));
                        return;
                    }
                    for (std::int64_t j = c; j < c + len; ++j) HostTeam::run(recompute, rp.data(), prefix(at(j)));
                },
                [&](std::int64_t j, std::int64_t count) {
                    if (stepBytes == 8)
                        eagle::diff::fillPrefix(reinterpret_cast<std::int64_t*>(stepPlane), j, count);
                    else if (stepBytes == 4)
                        eagle::diff::fillPrefix(reinterpret_cast<std::int32_t*>(stepPlane),
                                                static_cast<std::int32_t>(j), count);
                    HostTeam::run(vjp, vp[static_cast<std::size_t>(2 * (j % window) + (j & 1))].data(), prefix(count));
                    for (const auto& [acc, g, rows, dtype] : accums)
                        for (std::int64_t r = 0; r < rows; ++r) {
                            if (dtype == EAGLE_BACKEND_REDUCE_F64)
                                eagle::diff::accumulatePrefix(reinterpret_cast<double*>(acc) + r * n,
                                                              reinterpret_cast<const double*>(g) + r * n, count);
                            else
                                eagle::diff::accumulatePrefix(reinterpret_cast<float*>(acc) + r * n,
                                                              reinterpret_cast<const float*>(g) + r * n, count);
                        }
                });
        },
        nb::arg("n"), nb::arg("src_n"), nb::arg("window"), nb::arg("live"), nb::arg("orders"),
        nb::arg("gathers"), nb::arg("recompute_entry"), nb::arg("recompute_params"),
        nb::arg("recompute_word"), nb::arg("vjp_entry"), nb::arg("vjp_params"), nb::arg("accums"),
        nb::arg("step_plane"), nb::arg("step_bytes"),
        "The windowed reverse of one pass of a recorded host run, all in C++. Planes are n "
        "samples wide, checkpoints src_n. gathers: (dst, checkpoint address per window, rows, "
        "element bytes), window k's columns read through orders[k] (int64 checkpoint column "
        "per reverse slot, the window's sample order composed with its checkpoint's slot order). The recompute entry replays a window as one fused launch (its int64 steps word "
        "at recompute_word) or step by step (recompute_word 0), over the live prefix; "
        "vjp_params[2*row + (j & 1)] binds step j's vjp; step_plane (0: none) receives j "
        "before it; accums: (acc, g, rows, dtype 0 f32 / 1 f64) add each step's share.");

    m.def("diff_record_device",
        [](std::uintptr_t entry, const std::vector<std::uintptr_t>& params, std::int64_t n,
           std::int64_t maxSteps, std::int64_t window, std::int64_t maxK, std::uintptr_t word,
           std::uintptr_t finished, std::uint32_t block,
           const std::vector<std::pair<std::uintptr_t, std::int64_t>>& state,
           const std::function<std::vector<std::uintptr_t>(std::int64_t)>& checkpoint,
           const std::optional<std::function<void(std::int64_t)>>& after, std::uintptr_t stream,
           std::int32_t device) {
            const std::vector<void*> p = asParams(params);
            const auto op = [&](std::uint32_t code, std::uint64_t dst, std::uint64_t src, std::int64_t count,
                                std::uint32_t bytes, std::int64_t value) {
                auto d = makeDesc<eagle_backend_diff_desc>(device);
                d.op = code;
                d.dst = dst;
                d.src = src;
                d.dst_ld = count;
                d.count = count;
                d.rows = 1;
                d.elem_bytes = bytes;
                d.value = value;
                d.stream = stream;
                ck(EBX(eagle_backend_x_diff_columns)(&d));
            };
            std::int32_t done = 0;
            std::int64_t ran = 0;
            const eagle::exec::Partition whole{ 0, n, n };
            return eagle::diff::recordForward(
                maxSteps, window, maxK,
                [&] {   // read back once per window: a finished sample's launch is a no-op
                    if (ran % window != 0) return false;
                    op(EAGLE_BACKEND_DIFF_READ, reinterpret_cast<std::uint64_t>(&done), finished, 1, 4, 0);
                    return done >= n;
                },
                [&](std::int64_t k) {
                    const std::vector<std::uintptr_t> dst = checkpoint(k);
                    if (dst.size() != state.size())
                        throw std::invalid_argument("diff_record_device: checkpoint(k) returns one address per state plane");
                    for (std::size_t i = 0; i < state.size(); ++i)
                        op(EAGLE_BACKEND_DIFF_COPY, dst[i], state[i].first, state[i].second, 1, 0);
                },
                [&](std::int64_t j, std::int64_t k) {
                    if (word != 0) op(EAGLE_BACKEND_DIFF_FILL, word, 0, 1, 8, k);
                    eagle_seam::DeviceKernel::run(entry, p, whole, stream, block);
                    ran = j + k;
                    if (after) (*after)(ran);
                });
        },
        nb::arg("entry"), nb::arg("params"), nb::arg("n"), nb::arg("max_steps"), nb::arg("window"),
        nb::arg("max_k"), nb::arg("word"), nb::arg("finished"), nb::arg("block"), nb::arg("state"),
        nb::arg("checkpoint"), nb::arg("after").none() = nb::none(), nb::arg("stream") = 0,
        nb::arg("device") = -1,
        "diff_record_host on the device: launches through the backend's launcher, the steps word "
        "written on the stream, checkpoints copied on the stream into the addresses checkpoint(k) "
        "returns (state: (device address, bytes) per plane), the int32 finished count read back "
        "through `eagle_backend_x_diff_columns`. Returns the steps run.");
    m.def("diff_reverse_device",
        [](std::int64_t n, std::int64_t srcN, std::int64_t window, const std::vector<std::int64_t>& live,
           const std::vector<std::uintptr_t>& orders, const std::vector<Gather>& gathers,
           std::uintptr_t recomputeEntry, const std::vector<std::uintptr_t>& recomputeParams,
           std::uintptr_t recomputeWord, std::uint32_t recomputeBlock, std::uintptr_t vjpEntry,
           const std::vector<std::vector<std::uintptr_t>>& vjpParams, std::uint32_t vjpBlock,
           const std::vector<Accum>& accums, std::uintptr_t stepPlane, std::int64_t stepBytes,
           std::uintptr_t stream, std::int32_t device) {
            if (static_cast<std::int64_t>(vjpParams.size()) != 2 * window)
                throw std::invalid_argument("diff_reverse_device: vjp_params holds 2 lists per tape row");
            const std::vector<void*> rp = asParams(recomputeParams);
            std::vector<std::vector<void*>> vp;
            for (const auto& v : vjpParams) vp.push_back(asParams(v));
            const auto columns = [&](std::uint32_t op, std::uint64_t dst, std::uint64_t src, std::int64_t dstLd,
                                     std::int64_t srcLd, std::int64_t count, std::int64_t rows,
                                     std::uint32_t bytes, std::int64_t value, std::uint64_t index) {
                auto d = makeDesc<eagle_backend_diff_desc>(device);
                d.op = op;
                d.dst = dst;
                d.src = src;
                d.index = index;
                d.dst_ld = dstLd;
                d.src_ld = srcLd;
                d.count = count;
                d.rows = rows;
                d.elem_bytes = bytes;
                d.value = value;
                d.stream = stream;
                ck(EBX(eagle_backend_x_diff_columns)(&d));
            };
            const auto prefix = [n](std::int64_t count) { return Partition{ 0, count, n }; };
            const auto at = [&](std::int64_t j) { return live[static_cast<std::size_t>(j)]; };
            eagle::diff::reverseWindows(
                window, live,
                [&](std::int64_t k, std::int64_t count) {
                    for (const auto& [dst, sources, rows, bytes] : gathers)
                        columns(EAGLE_BACKEND_DIFF_GATHER, dst, sources.at(static_cast<std::size_t>(k)), n, srcN,
                                count, rows, static_cast<std::uint32_t>(bytes), 0,
                                orders.at(static_cast<std::size_t>(k)));
                },
                [&](std::int64_t c, std::int64_t len) {
                    if (recomputeWord != 0) {   // one fused launch; each sample stops itself
                        columns(EAGLE_BACKEND_DIFF_FILL, recomputeWord, 0, 1, 0, 1, 1, 8, len, 0);
                        eagle_seam::DeviceKernel::run(recomputeEntry, rp, prefix(at(c)), stream, recomputeBlock);
                        return;
                    }
                    for (std::int64_t j = c; j < c + len; ++j)
                        eagle_seam::DeviceKernel::run(recomputeEntry, rp, prefix(at(j)), stream, recomputeBlock);
                },
                [&](std::int64_t j, std::int64_t count) {
                    if (stepPlane != 0)
                        columns(EAGLE_BACKEND_DIFF_FILL, stepPlane, 0, n, 0, count, 1,
                                static_cast<std::uint32_t>(stepBytes), j, 0);
                    eagle_seam::DeviceKernel::run(
                        vjpEntry, vp[static_cast<std::size_t>(2 * (j % window) + (j & 1))], prefix(count), stream, vjpBlock);
                    for (const auto& [acc, g, rows, dtype] : accums)
                        columns(EAGLE_BACKEND_DIFF_ACCUMULATE, acc, g, n, n, count, rows,
                                dtype == EAGLE_BACKEND_REDUCE_F64 ? 8u : 4u, 0, 0);
                });
        },
        nb::arg("n"), nb::arg("src_n"), nb::arg("window"), nb::arg("live"), nb::arg("orders"),
        nb::arg("gathers"), nb::arg("recompute_entry"), nb::arg("recompute_params"),
        nb::arg("recompute_word"), nb::arg("recompute_block"), nb::arg("vjp_entry"),
        nb::arg("vjp_params"), nb::arg("vjp_block"), nb::arg("accums"), nb::arg("step_plane"),
        nb::arg("step_bytes"), nb::arg("stream") = 0, nb::arg("device") = -1,
        "diff_reverse_host on the device: the same window driver, every operation enqueued on "
        "`stream` (launches through the backend's launcher, column ops through "
        "`eagle_backend_x_diff_columns`); `orders` are device int64 arrays. Enqueue only.");

    nb::class_<eagle::util::BusyGuard>(m, "BusyGuard")
        .def(nb::init<std::string>(), nb::arg("owner"))
        .def("enter", [](eagle::util::BusyGuard& g, const std::string& method) { g.enter(method); },
            nb::arg("method"), "Take the guard or raise eagle.Busy naming the method in flight.")
        .def("leave", [](eagle::util::BusyGuard& g) { g.leave(); })
        .def("current", &eagle::util::BusyGuard::current)
        .def("owner", &eagle::util::BusyGuard::owner);
}
