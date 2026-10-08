// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file eagle_mpi.cpp
 * @brief nanobind binding of ``eagle::exec::RankPartition`` — the OPTIONAL
 *        ``eagle._mpi`` extension (plain MPI, MECHANISM only).
 *
 * WHY A SECOND MODULE, AND NOT A FEW MORE ENTRY POINTS IN ``_core``.
 * ``eagle/exec/RankPartition.h`` ``#error``s unless ``EAGLE_MPI`` is defined, and
 * the define exists precisely so that eagle never links ``libmpi``: the C++ side
 * puts it PRIVATELY on one test target (``tests/mpi/CMakeLists.txt``) and nowhere
 * else. Folding this surface into ``_core`` would drag ``libmpi`` into the
 * extension every eagle Python user imports — including the ones with no MPI
 * installed at all, for whom the import would simply stop working. So the rank
 * structure gets its OWN extension, built only when an MPI is present
 * (``-DEAGLE_PYTHON_MPI=ON|AUTO``), and ``_core``'s freedom from ``libmpi`` is an
 * executable audit row in the main Python suite
 * (``tests/test_rank_partition.py::test_core_extension_does_not_link_libmpi``)
 * rather than a promise in a comment.
 *
 * THE TRIPLE CROSSES AS THREE INTEGERS, NOT AS ``_core.Partition``. Two nanobind
 * extensions in one process share nanobind's type registry only when their ABI
 * tags agree, and a second ``nb::class_<Partition>`` would be a duplicate
 * registration outright. Neither failure is worth risking for a struct that is
 * three ``int64``s: this module takes and returns ``(base, count, n_samples)``
 * tuples, and :mod:`eagle.exec` — which owns both spellings — converts. The two
 * modules therefore share no C++ type at all, only the header they were both
 * compiled from.
 *
 * WHAT IS BOUND, AND WHAT DELIBERATELY IS NOT.
 *   ``world`` / ``local_partition``   the world size + this rank's contiguous share
 *   ``run_rank_partition``            this rank's slice through HostTeam or DeviceKernel
 *   ``allgather_plane``               the host-staged output gather (MPI_Allgatherv)
 *   ``allgather_fold``                the mapreduce combine, in RANK ORDER
 *
 * ``RankPartition::allgather_staged`` (the device-resident gather, which stages
 * through an owning ``aether::Array``) is NOT bound. The Python plan's device path
 * already lands every output plane on the HOST — ``eagle.plan._run_device_plan``
 * returns ``cupy.asnumpy(...)`` planes — so the Python gather is a host gather by
 * construction, which is exactly the shape ruled (host-staged, and deliberately
 * no CUDA-aware MPI). Binding the staged twin would mean handing it a cupy
 * allocation dressed as an aether array, which is neither true nor needed.
 *
 * MPI LIFECYCLE. ``MPI_Init`` is LAZY (first call into this module) and conditional
 * on ``MPI_Initialized``, so a host process that already initialised MPI is left
 * alone; ``MPI_Finalize`` is registered with the ``atexit`` MODULE (see
 * ``finalizeMpi`` for why NOT ``Py_AtExit``) and runs only if THIS module was the
 * one that initialised. A singleton world — no ``mpirun``, size 1 — is a supported
 * configuration, not a degraded one: it is what makes the rank structure reachable
 * from the ordinary single-process test suite.
 */

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <dlfcn.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <mpi.h>

// The rank structure itself. `EAGLE_MPI` is put on this target by
// python/CMakeLists.txt and on no other; the header's own `#error` is what turns an
// accidental include elsewhere into a compile failure rather than a link surprise.
#include "eagle/exec/HostTeam.h"
#include "eagle/exec/Partition.h"
#include "eagle/exec/RankPartition.h"

// A plain C++ TU (EAGLE_CPU_ONLY): the device inner is eagle_seam::DeviceKernel,
// which forwards each launch to the CUDA backend plugin (the same file eagle._core
// loads; dlopen of the same path returns the same mapping).
#include "seam/DeviceKernel.h"

namespace nb = nanobind;

using eagle::exec::Partition;
using RankHost = eagle::exec::RankPartition<eagle::exec::HostTeam>;
using RankDevice = eagle::exec::RankPartition<eagle_seam::DeviceKernel>;

namespace {

/** @brief The launch-partition triple as it crosses this module's boundary. */
using Triple = std::tuple<std::int64_t, std::int64_t, std::int64_t>;

inline Partition asPartition(const Triple& t)
{
    return Partition{ std::get<0>(t), std::get<1>(t), std::get<2>(t) };
}

inline Triple asTriple(const Partition& p)
{
    return Triple{ p.base, p.count, p.nSamples };
}

/** @brief Turn a Python list of integer pointers into the ``params[]`` array a
 *  plugin entry receives — byte for byte ``eagle_core.cu``'s own ``asParams``,
 *  because the two modules must hand a body the SAME argument block. */
inline std::vector<void*> asParams(const std::vector<std::uintptr_t>& args)
{
    std::vector<void*> params;
    params.reserve(args.size());
    for (std::uintptr_t a : args) params.push_back(reinterpret_cast<void*>(a));
    return params;
}

//: Did THIS module call `MPI_Init`? Only then may it call `MPI_Finalize`.
bool g_weInitialised = false;

/**
 * @brief Finalize MPI at interpreter shutdown, but only what we started.
 *
 * Registered through the `atexit` MODULE, NOT `Py_AtExit`, and that is a measured
 * choice rather than a stylistic one. `Py_AtExit` runs its callbacks from inside
 * `Py_FinalizeEx`, by which point the interpreter has already torn most of itself
 * down; `MPI_Finalize` there corrupted the heap in roughly half of full-suite runs
 * ("munmap_chunk(): invalid pointer", exit 139/134 AFTER pytest had printed
 * "506 passed") — a verdict that reached the terminal but not the caller's exit
 * code, which is the one failure mode a gate cannot see. `atexit` callbacks run
 * BEFORE that teardown, with the process still whole.
 *
 * Registration is also LAZY — it happens in `ensureInit`, not at module import —
 * so that `atexit`'s LIFO order puts this callback AHEAD of the CUDA/torch
 * teardown those libraries register when they are imported: MPI is finalized while
 * the accelerator runtimes it may still touch are alive.
 *
 * WHERE THIS MUST NOT BE ASKED TO RUN AT ALL. `MPI_Init` is not a passive call:
 * Open MPI installs process-wide memory-management hooks (the `opal/mca/patcher`
 * and accelerator machinery) that do not coexist with a long-lived process already
 * running the CUDA driver, cupy and torch allocators. Measured on eagle's own
 * Python suite: with MPI initialised in the pytest process, the run crashed in
 * roughly one attempt in three — either inside Open MPI's own teardown
 * (`opal_output_finalize <- opal_finalize_util <- ompi_mpi_finalize`, exit 139
 * AFTER "506 passed" had been printed: a verdict reaching the terminal but not the
 * caller's exit code) or, with finalize suppressed, later and deeper, inside
 * `libcuda`'s `cuModuleGetFunction`. The SAME suite with the MPI rows removed was
 * clean every time. So eagle's single-process rows drive this module in a CHILD
 * process (`tests/rank_singleton_cases.py`), and the distributed bed is its own
 * `mpirun` world — MPI never shares a process with an unrelated long session.
 *
 * The `MPI_Finalized` re-check is not belt-and-braces: an embedding host that
 * initialised MPI itself may also have finalized it before the interpreter goes
 * down, and calling `MPI_Finalize` twice is undefined.
 */
void finalizeMpi()
{
    if (!g_weInitialised) return;
    g_weInitialised = false;
    int finalized = 0;
    if (MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized) MPI_Finalize();
}

/**
 * @brief Promote ``libmpi`` to the global symbol namespace before initialising.
 *
 * A Python extension is dlopen'ed ``RTLD_LOCAL``, so everything it links comes in
 * local too. Open MPI then dlopen's its own MCA components, which resolve their
 * ``MPI_*``/``opal_*`` symbols against the GLOBAL namespace and find nothing — the
 * classic "mca_base_component_repository_open: unable to open" failure that
 * mpi4py's own loader exists to avoid. Re-opening the library the build linked
 * against with ``RTLD_GLOBAL`` promotes the already-mapped object in place
 * (``RTLD_NOLOAD``); the non-NOLOAD retry covers a loader that will not promote.
 *
 * Deliberately best-effort and SILENT on failure: an Open MPI built with its
 * components inside ``libmpi`` needs none of this, and a real initialisation
 * problem must surface as ``MPI_Init``'s own error below, not as a dlopen message
 * about a path this module guessed. ``EAGLE_MPI_LIBRARY`` is the path CMake
 * resolved (never a guessed SONAME); it is absent only if MPI::MPI_CXX carried no
 * file path at all, in which case there is nothing honest to open.
 */
void promoteMpiSymbols()
{
#ifdef EAGLE_MPI_LIBRARY
    if (std::string(EAGLE_MPI_LIBRARY).empty()) return;
    if (::dlopen(EAGLE_MPI_LIBRARY, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD) != nullptr)
        return;
    ::dlopen(EAGLE_MPI_LIBRARY, RTLD_NOW | RTLD_GLOBAL);
#endif
}

/** @brief Initialise MPI once, if nobody else already has. */
void ensureInit()
{
    int initialised = 0;
    eagle::exec::mpi_check(MPI_Initialized(&initialised), "MPI_Initialized");
    if (initialised) return;

    int finalized = 0;
    eagle::exec::mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized)
        throw std::runtime_error(
            "eagle._mpi: MPI has already been finalized in this process; the rank "
            "structure cannot be used after MPI_Finalize");

    promoteMpiSymbols();
    eagle::exec::mpi_check(MPI_Init(nullptr, nullptr), "MPI_Init");
    g_weInitialised = true;
    nb::module_::import_("atexit").attr("register")(
        nb::cpp_function([] { finalizeMpi(); }));
}

}  // namespace

NB_MODULE(_mpi, m)
{
    m.doc() = "eagle._mpi — nanobind binding of eagle::exec::RankPartition (the "
              "MPI execution structure). Optional: built only where an MPI is "
              "present (-DEAGLE_PYTHON_MPI=ON|AUTO).";

    // Whether this build carries the DEVICE inner. It always does: the inner
    // forwards to the CUDA backend, and a missing or unusable backend raises
    // eagle.BackendUnavailable at the launch, by name.
    eagle_seam::registerTranslators();
    m.attr("HAS_DEVICE_INNER") = true;

    m.def("world", [] {
        ensureInit();
        return std::tuple<int, int>{ RankHost::comm_rank(MPI_COMM_WORLD),
                                     RankHost::comm_size(MPI_COMM_WORLD) };
    },
        "This process's `(rank, size)` in MPI_COMM_WORLD, initialising MPI on the "
        "first call. A singleton world (no mpirun) reports (0, 1).");

    m.def("local_partition",
        [](const Triple& whole) {
            ensureInit();
            return asTriple(RankHost::local(asPartition(whole), MPI_COMM_WORLD));
        },
        nb::arg("whole"),
        "This rank's CONTIGUOUS share of `whole`, as a (base, count, n_samples) "
        "triple. `n_samples` is unchanged: a rank is one more cut of the same run, "
        "never a smaller run. Uneven splits give the first `n % size` ranks "
        "one extra sample.");

    m.def("run_rank_partition",
        [](std::uintptr_t entry, const std::vector<std::uintptr_t>& params,
           const Triple& whole, const std::string& inner,
           std::size_t bytes_per_sample, std::uintptr_t stream, unsigned block) {
            ensureInit();
            const std::vector<void*> p = asParams(params);
            const Partition part = asPartition(whole);
            // One return type for both inners: HostTeam counts TILES (size_t) and
            // DeviceKernel counts LAUNCHES (int), and the caller only ever reads
            // "how many units of work this rank issued".
            if (inner == "host_team")
                return std::int64_t(RankHost::run(
                    reinterpret_cast<eagle::exec::HostEntryV2>(entry), p.data(),
                    part, MPI_COMM_WORLD, bytes_per_sample));
            if (inner == "device_kernel")
                return std::int64_t(RankDevice::run(entry, p, part, MPI_COMM_WORLD, stream, block));
            throw std::runtime_error(
                "eagle._mpi.run_rank_partition: inner structure '" + inner +
                "' is not one of host_team, device_kernel (a device inner needs a "
                "CUDA build of this extension)");
        },
        nb::arg("entry"), nb::arg("params"), nb::arg("partition"), nb::arg("inner"),
        nb::arg("bytes_per_sample") = 0, nb::arg("stream") = 0,
        nb::arg("block") = 256,
        "Run THIS RANK's contiguous share of `partition` through `inner` "
        "(host_team | device_kernel). `params` is the SAME argument block the "
        "single-process structures take: for a host entry the pointer values, for "
        "a device kernel the addresses of the by-value argument storage. eagle does "
        "no pointer arithmetic on them on any rank — the body slices by the triple "
        "it is handed. Returns the inner structure's own count (tiles run, or "
        "1/0 for a device launch).");

    m.def("allgather_plane",
        [](std::uintptr_t plane, const Triple& whole, int elems_per_sample) {
            ensureInit();
            RankHost::allgather_plane(reinterpret_cast<double*>(plane),
                                      asPartition(whole), MPI_COMM_WORLD,
                                      elems_per_sample);
        },
        nb::arg("plane"), nb::arg("partition"), nb::arg("elems_per_sample") = 1,
        "Gather a HOST per-sample float64 output plane in place, so every rank "
        "ends with the WHOLE of it (MPI_Allgatherv). `plane` is the address of "
        "element 0 of a buffer addressed by GLOBAL sample index, holding "
        "`elems_per_sample` contiguous doubles per sample. The wire type is "
        "MPI_DOUBLE — a plane of any other element type must not be gathered "
        "through this call.");

    m.def("allgather_fold",
        [](const std::string& op, double partial) {
            ensureInit();
            return RankHost::allgather_fold(eagle::exec::op_from_string(op), partial,
                                            MPI_COMM_WORLD);
        },
        nb::arg("op"), nb::arg("partial"),
        "Combine the R per-rank mapreduce partials under `op` in a FIXED RANK "
        "ORDER (MPI_Allgather + eagle::exec::fold, never MPI_Allreduce, whose "
        "combine order is the implementation's business — two ranks could then "
        "disagree in the last bits). Returns the same value on every rank.");
}
