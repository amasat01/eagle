// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Vocabulary for the heterogeneous execution contract's PARTITION and
// PLACEMENT rules (aether-abi/2).
//
// A partition is an explicit int64 TRIPLE `{base, count, nSamples}`, never a
// sub-view: eagle slices only PER-SAMPLE roles and passes SHARED roles WHOLE, and the body's
// global index is `base + local`, guarded by `count`, while every expression that
// bakes the sample count receives the TRUE `nSamples`. A sub-view would silently
// re-target the cross-sample reads (Staged/lookup) and the accumulate planes that
// a generated kernel already emits against.
//
// Deliberately DEPENDENCY-FREE, exactly like `plugin/gref_abi.h`: no aether, no
// CUDA, no eagle headers. `DeviceKernel.h` (CUDA) and `HostTeam.h` (OpenMP) both
// include it, and so does `plugin/host_registry.h`'s legacy-partition guard — a
// header any of those can include must cost none of them a dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace eagle {
namespace exec {

/**
 * @brief The launch partition: `{base, count, nSamples}`.
 *
 * `base` is the global index of this partition's first sample, `count` how many
 * samples it covers, and `nSamples` the TRUE total the whole run spans — the
 * number every wide/accum plane is addressed against, independent of how the run
 * was cut. `whole(n)` is the legacy (aether-abi/1) shape: one partition covering
 * everything.
 *
 * int64 throughout: the v1 wire packed the sample count as a 4-byte `unsigned`
 * (`registry.h`'s `counts.push_back(unsigned(n))`), which silently caps at 2^31.
 */
struct Partition {
    std::int64_t base     = 0;
    std::int64_t count    = 0;
    std::int64_t nSamples = 0;

    static Partition whole(std::int64_t n) { return Partition{ 0, n, n }; }

    /** @brief Is this the whole view — the only shape a legacy v1 plugin may be
     *  handed? */
    bool is_whole() const { return base == 0 && count == nSamples; }
};

/** @brief The launch grid eagle derives from a partition's `count` (never
 *  `nSamples`): one thread per sample of THIS partition, `block` per block.
 *  0 for an empty partition. Device-runtime free, so the host build and every
 *  device backend share one spelling (`DeviceKernel::grid` delegates here). */
inline unsigned launch_grid(std::int64_t count, unsigned block) {
    if (block == 0) throw std::runtime_error("DeviceKernel: block must be > 0");
    if (count <= 0) return 0u;
    return static_cast<unsigned>((static_cast<std::uint64_t>(count) + block - 1u) / block);
}

/** @brief The aether-abi/2 HOST plugin entry: a SERIAL range over
 *  `[base, base + count)` of views whose plane addressing spans `nSamples`.
 *
 *  The v1 shape (`void(void* const*, int32_t)`, `eagle::cpu::HostEntry`) lives on
 *  for legacy artifacts; this is its REPLACEMENT, not an overload of it. Declared
 *  in this dependency-free header rather than in `HostTeam.h` so the plugin
 *  registry (which must not see aether) and the execution structure (which uses
 *  aether's tiling) share ONE spelling of the type. */
using HostEntryV2 = void (*)(void* const* params, std::int64_t base,
                             std::int64_t count, std::int64_t nSamples);

/** @brief The DECLARED access class (L3) — how the body touches sample-local vs
 *  cross-sample data. Classified by the emitter from the trace, never guessed;
 *  the wire spelling is the manifest's `exec_access` (raptor
 *  `schema/manifest.py::EXEC_ACCESS_CLASSES`). */
enum class Access { SampleLocal, CrossSampleRead, CrossSampleWrite, MapReduce };

/** @brief The EXECUTION STRUCTURE eagle may drive the body with. */
enum class Structure { DeviceKernel, HostTeam, RankPartition, DeviceGroup };

/** @brief The declared reduction operator of a `mapreduce` body — the manifest's
 *  `exec_op` (raptor `EXEC_OPS`). Eagle owns the COMBINE; the body only produces
 *  per-partition partials. */
enum class ReduceOp { Sum, Times, Max, LAnd };

/** @brief Parse the wire spelling of an access class; throws naming the supported
 *  set (the wire vocabulary is raptor's, and an unknown value must be refused
 *  rather than dispatched on). */
inline Access access_from_string(const std::string& s) {
    if (s == "sample_local")       return Access::SampleLocal;
    if (s == "cross_sample_read")  return Access::CrossSampleRead;
    if (s == "cross_sample_write") return Access::CrossSampleWrite;
    if (s == "mapreduce")          return Access::MapReduce;
    throw std::runtime_error("exec_access '" + s + "' is not a supported access "
        "class (supported: cross_sample_read, cross_sample_write, mapreduce, "
        "sample_local)");
}

/** @brief Parse the wire spelling of an execution structure. */
inline Structure structure_from_string(const std::string& s) {
    if (s == "device_kernel")   return Structure::DeviceKernel;
    if (s == "host_team")       return Structure::HostTeam;
    if (s == "rank_partition")  return Structure::RankPartition;
    if (s == "device_group")    return Structure::DeviceGroup;
    throw std::runtime_error("execution structure '" + s + "' is not a supported "
        "structure (supported: device_group, device_kernel, host_team, "
        "rank_partition)");
}

/** @brief Parse the wire spelling of a reduction operator (`exec_op`). */
inline ReduceOp op_from_string(const std::string& s) {
    if (s == "sum")   return ReduceOp::Sum;
    if (s == "times") return ReduceOp::Times;
    if (s == "max")   return ReduceOp::Max;
    if (s == "land")  return ReduceOp::LAnd;
    throw std::runtime_error("exec_op '" + s + "' is not a supported reduction op "
        "(supported: land, max, sum, times)");
}

inline const char* to_string(Access a) {
    switch (a) {
        case Access::SampleLocal:      return "sample_local";
        case Access::CrossSampleRead:  return "cross_sample_read";
        case Access::CrossSampleWrite: return "cross_sample_write";
        case Access::MapReduce:        return "mapreduce";
    }
    return "?";
}

inline const char* to_string(Structure s) {
    switch (s) {
        case Structure::DeviceKernel:  return "device_kernel";
        case Structure::HostTeam:      return "host_team";
        case Structure::RankPartition: return "rank_partition";
        case Structure::DeviceGroup:   return "device_group";
    }
    return "?";
}

/** @brief The identity element of @p op — what eagle seeds a partial plane with
 *  before the run, so an untouched slot cannot perturb the fold. */
inline double identity(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:   return 0.0;
        case ReduceOp::Times: return 1.0;
        case ReduceOp::Max:   return -std::numeric_limits<double>::infinity();
        case ReduceOp::LAnd:  return 1.0;
    }
    return 0.0;
}

/** @brief Combine two partials under @p op. */
inline double combine(ReduceOp op, double a, double b) {
    switch (op) {
        case ReduceOp::Sum:   return a + b;
        case ReduceOp::Times: return a * b;
        case ReduceOp::Max:   return (b > a) ? b : a;
        case ReduceOp::LAnd:  return (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
    }
    return a;
}

/**
 * @brief Fold @p n partials under @p op in a FIXED ASCENDING order.
 *
 * The order is the contract, not an implementation detail: a mapreduce result is
 * band-gated rather than bit-exact precisely BECAUSE the combine order differs
 * between a whole run and a partitioned one, so the order eagle itself uses must
 * at least be deterministic and identical on every arm — never an atomic race.
 */
inline double fold(ReduceOp op, const double* partials, std::size_t n) {
    double acc = identity(op);
    for (std::size_t i = 0; i < n; ++i) acc = combine(op, acc, partials[i]);
    return acc;
}

/**
 * @brief Is running a body of declared @p access under @p structure over
 *        @p npartitions partitions LEGAL?
 *
 * Refuses NAMING THE RULE. The rules, verbatim from the contract:
 *   * `sample_local` / `cross_sample_read` / `mapreduce` -> every implemented
 *     structure (a `cross_sample_read` body reads shared roles, which eagle
 *     passes WHOLE on the executing device by construction);
 *   * `RankPartition` is now IMPLEMENTED and takes `cross_sample_read` with
 *     it: its inputs are REPLICATED, so a rank's lookup/Staged tables are whole on
 *     its own device, which is exactly the condition placement legality states. `npartitions` for
 *     this structure is the communicator's size;
 *   * `cross_sample_write` -> single-device structures ONLY, and only as ONE
 *     partition: two partitions accumulating into one shared target would need a
 *     partial-accum combine that is not specified, so it is ruled ILLEGAL. Under
 *     `RankPartition` it is refused even at ONE rank — a run that is legal at
 *     `-np 1` and illegal at `-np 2` is a trap, and the structure's whole purpose
 *     is to be run at more;
 *   * `DeviceGroup` -> refused outright, "NCCL not implemented": that structure
 *     does not exist in this build.
 */
inline void check_placement(Access access, Structure structure, int npartitions) {
    if (npartitions < 1)
        throw std::runtime_error("illegal placement: npartitions must be >= 1, got "
            + std::to_string(npartitions));
    if (structure == Structure::DeviceGroup)
        throw std::runtime_error(std::string("illegal placement: execution structure '")
            + to_string(structure) + "' is not implemented in this build "
            "(NCCL not implemented)");
    if (access == Access::CrossSampleWrite) {
        if (structure == Structure::RankPartition)
            throw std::runtime_error(std::string("illegal placement: exec_access '")
                + to_string(access) + "' may not be placed on '"
                + to_string(structure) + "': ranks accumulating into one shared "
                "target need a partial-accum combine, which is not specified (F-e)");
        if (npartitions > 1)
            throw std::runtime_error(std::string("illegal placement: exec_access '")
                + to_string(access) + "' is legal on single-device structures only and "
                "may not be split across " + std::to_string(npartitions) + " partitions "
                "(a partial-accum combine is not specified; F-e)");
    }
}

/** @brief String-spelled twin of :func:`check_placement` — the form the manifest's
 *  own values and the Python plan surface hand over. */
inline void check_placement(const std::string& access, const std::string& structure,
                            int npartitions) {
    check_placement(access_from_string(access), structure_from_string(structure),
                    npartitions);
}

/** @brief Guard the legacy bridge: an `aether-abi/1` plugin knows nothing of the
 *  partition triple, so it may only ever be driven WHOLE-VIEW, single-structure. */
inline void check_legacy_whole_view(int abi_version, const Partition& p,
                                    const std::string& what) {
    if (abi_version >= 2) return;
    if (!p.is_whole())
        throw std::runtime_error(what + ": a legacy aether-abi/1 plugin takes no "
            "partition triple and may only be launched WHOLE-VIEW, single-partition "
            "(L13); got base=" + std::to_string(p.base) + " count=" +
            std::to_string(p.count) + " nSamples=" + std::to_string(p.nSamples));
}

/** @brief Guard the legacy bridge on the STRUCTURE axis: an `aether-abi/1`
 *  plugin is a whole-view, SINGLE-device artifact, so the multi-process and
 *  multi-device structures may not drive it at all.
 *
 *  Without this the refusal would still happen, but one layer down and for a
 *  misleading reason: `RankPartition` would hand a legacy entry a non-whole
 *  triple, `check_legacy_whole_view` would refuse THAT, and at a world size of 1
 *  — where the rank's share happens to be the whole view — nothing would refuse
 *  anything and R identical writers would race on the gather. The rule belongs on
 *  the structure, where it holds at every world size. */
inline void check_legacy_structure(int abi_version, Structure structure,
                                   const std::string& what) {
    if (abi_version >= 2) return;
    if (structure != Structure::RankPartition && structure != Structure::DeviceGroup)
        return;
    throw std::runtime_error(what + ": a legacy aether-abi/1 plugin is a WHOLE-VIEW, "
        "SINGLE-device artifact and may not be driven by execution structure '"
        + to_string(structure) + "' (L13); re-emit it as aether-abi/2");
}

}  // namespace exec
}  // namespace eagle
