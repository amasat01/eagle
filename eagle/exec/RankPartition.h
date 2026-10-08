// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `eagle::exec::RankPartition` — the MULTI-PROCESS execution structure of the
// heterogeneous execution contract (aether-abi/2,
// Plain MPI, MECHANISM only -- no performance claims, and deliberately
// NO CUDA-aware MPI).
//
// THE SHAPE. Rank `r` of `R` runs the r-th CONTIGUOUS sub-partition of one whole
// partition through an INNER structure (`HostTeam` or `DeviceKernel`) with its OWN
// triple `{base_r, count_r, nSamples}`. `nSamples` never moves: it is the TRUE
// total, exactly as it is for a tile of a HostTeam (L2) — a rank is one more cut of
// the same run, not a smaller run.
//
// INPUTS ARE REPLICATED. Every rank holds the whole inputs, so the SHARED roles a
// `cross_sample_read` body reads (lookup / Staged tables) are whole on every rank
// by construction. That is the ruling that makes `cross_sample_read` LEGAL here
// while it would not be under a structure that partitions its tables — L3 states
// the condition ("every structure whose shared roles are WHOLE on the executing
// device"), and replication is how RankPartition meets it. `cross_sample_write`
// stays ILLEGAL (fork F-e): two ranks accumulating into one shared target need a
// partial-accum combine that is not specified.
//
// OUTPUTS ARE GATHERED HOST-STAGED. Each rank contributes its own slice of a
// per-sample output plane and the wire is a plain `MPI_Allgatherv` on HOST buffers,
// so every rank ends with the WHOLE output — bit-identical to a single whole run
// because no rank ever recomputes another rank's samples. A device-resident
// plane is staged D2H and back H2D through AETHER (`Array::download`/`upload`,
// which are `aether::copyAsync` -- no raw memcpy on aether data), never through
// a hand-rolled `cudaMemcpy` in this header.
//
// MAPREDUCE. Each rank folds its own partial with the inner structure's fold, the R
// partials cross the wire as one `MPI_Allgather`, and every rank folds THEM in
// RANK ORDER (`exec::fold`). Rank order is the contract for the same reason
// ascending tile order is in `HostTeam`: a mapreduce result is band-gated rather
// than bit-exact BECAUSE the combine order differs between a whole run and a
// partitioned one, so the order eagle itself uses must at least be deterministic
// and identical on every rank. `MPI_Allreduce(MPI_SUM)` is deliberately NOT used —
// its combine order is the implementation's business, so two ranks could disagree
// in the last bits.
//
// COMPILED ONLY WHERE THE BED IS. `EAGLE_MPI` is defined PRIVATELY on
// `tests/mpi/eagle_mpi_bed` and nowhere else, so eagle never links MPI and no other
// translation unit pays for `<mpi.h>`. The `#error` below makes an accidental
// include a loud compile failure rather than a silent one.
#pragma once

#ifndef EAGLE_MPI
#error "eagle/exec/RankPartition.h requires EAGLE_MPI (define it PRIVATELY on the target that needs MPI; see tests/mpi/CMakeLists.txt)"
#endif

#include "eagle/exec/HostTeam.h"
#include "eagle/exec/Partition.h"

#ifndef EAGLE_CPU_ONLY
#include "eagle/exec/DeviceKernel.h"
#include <aether/array/Array.h>
#include <cuda_runtime.h>
#endif

#include <mpi.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace eagle {
namespace exec {

/** @brief Throw naming the MPI call that failed. A collective that returns an
 *  error and is ignored leaves the ranks disagreeing about what they hold, which
 *  is exactly the class of defect the bed exists to catch — so no call here is
 *  unchecked. */
inline void mpi_check(int rc, const char* what)
{
    if (rc == MPI_SUCCESS) return;
    char msg[MPI_MAX_ERROR_STRING] = {};
    int len = 0;
    if (MPI_Error_string(rc, msg, &len) != MPI_SUCCESS) len = 0;
    throw std::runtime_error(std::string("RankPartition: ") + what + " failed: "
        + (len > 0 ? std::string(msg, std::size_t(len)) : std::to_string(rc)));
}

/**
 * @brief The rank execution structure: one contiguous sub-partition per MPI rank,
 *        driven through @p Inner.
 *
 * @tparam Inner the per-rank execution structure — `HostTeam` or `DeviceKernel`.
 */
template <class Inner>
struct RankPartition {

    /** @brief The number of ranks in @p comm — the `npartitions` every placement
     *  question about this structure is asked with). */
    static int comm_size(MPI_Comm comm)
    {
        int n = 0;
        mpi_check(MPI_Comm_size(comm, &n), "MPI_Comm_size");
        return n;
    }

    /** @brief This process's rank in @p comm. */
    static int comm_rank(MPI_Comm comm)
    {
        int r = 0;
        mpi_check(MPI_Comm_rank(comm, &r), "MPI_Comm_rank");
        return r;
    }

    /**
     * @brief The sub-partition rank @p rank of @p size covers.
     *
     * Contiguous, and UNEVEN splits are allowed: with `n = q*size + rem`, the first
     * `rem` ranks take `q + 1` samples and the rest take `q`. The alternative —
     * padding every rank to `ceil(n/size)` — would hand the last rank a `count`
     * running past `nSamples`, and a body that early-outs on `count` would
     * then write past the plane. Every sample of @p whole is covered exactly once,
     * which is what `UnevenSplitCoversEverySample` pins.
     *
     * A pure function of its arguments, with no MPI in it, so the coverage property
     * can be enumerated without a communicator.
     */
    static Partition local(const Partition& whole, int rank, int size)
    {
        if (size < 1)
            throw std::runtime_error("RankPartition::local: size must be >= 1, got "
                + std::to_string(size));
        if (rank < 0 || rank >= size)
            throw std::runtime_error("RankPartition::local: rank " + std::to_string(rank)
                + " is outside [0, " + std::to_string(size) + ")");
        const std::int64_t n   = whole.count;
        const std::int64_t q   = n / size;
        const std::int64_t rem = n % size;
        const std::int64_t r   = rank;
        const std::int64_t count = q + (r < rem ? 1 : 0);
        const std::int64_t base  = whole.base + r * q + (r < rem ? r : rem);
        return Partition{ base, count, whole.nSamples };
    }

    /** @overload Resolves the rank and size from @p comm. */
    static Partition local(const Partition& whole, MPI_Comm comm)
    {
        return local(whole, comm_rank(comm), comm_size(comm));
    }

    /**
     * @brief The placement + transition guard for a run over @p comm.
     *
     * Two rules, both refused NAMING the rule:
     *   * the L3/F-e placement legality of @p access on this structure over
     *     `comm_size(comm)` partitions — `sample_local`, `cross_sample_read`
     *     (replicated tables) and `mapreduce` are legal, `cross_sample_write` is
     *     not;
     *   * a legacy `aether-abi/1` plugin may not be driven by this structure at all:
     *     it takes no partition triple, so every rank would run the WHOLE
     *     view and the gather would then be a race between R identical writers.
     */
    static void check(Access access, int abi_version, MPI_Comm comm)
    {
        check_placement(access, Structure::RankPartition, comm_size(comm));
        check_legacy_structure(abi_version, Structure::RankPartition,
                               "RankPartition::check");
    }

    /**
     * @brief Run @p plugin over THIS RANK's share of @p whole through @p Inner.
     *
     * @param plugin the inner structure's entry — a `HostEntryV2` for `HostTeam`,
     *               a `CUfunction` for `DeviceKernel`.
     * @param args   the packed role args, passed WHOLE (L2): eagle does no pointer
     *               arithmetic on them on any rank, and the body slices by its own
     *               triple.
     * @param whole  the partition the WHOLE world covers.
     * @param comm   the communicator whose size is the partition count.
     * @param extra  whatever the inner structure takes after the partition —
     *               `bytesPerSample` for `HostTeam`, `(stream, block)` for
     *               `DeviceKernel`. Forwarded verbatim; RankPartition adds nothing
     *               to the inner launch but the triple it hands it.
     * @return the inner structure's own return value for this rank's launch.
     */
    template <class Entry, class Args, class... Extra>
    static auto run(Entry plugin, const Args& args, const Partition& whole,
                    MPI_Comm comm, Extra... extra)
    {
        return Inner::run(plugin, args, local(whole, comm), extra...);
    }

    /**
     * @brief Gather a per-sample HOST output plane so every rank ends with the
     *        WHOLE of it.
     *
     * @param plane          the plane's element 0 — addressed by GLOBAL sample
     *                       index, spanning `whole.nSamples` samples, exactly as
     *                       the body addressed it.
     * @param whole          the partition the world covered.
     * @param comm           the communicator.
     * @param elemsPerSample how many `double`s one sample occupies in this plane
     *                       (1 for a scalar plane).
     *
     * `MPI_IN_PLACE`: each rank already wrote its own slice where it belongs, so
     * there is nothing to copy into a send buffer — and a send buffer would be one
     * more place for a slice to be staged out of the wrong offset.
     */
    static void allgather_plane(double* plane, const Partition& whole, MPI_Comm comm,
                                int elemsPerSample = 1)
    {
        if (plane == nullptr)
            throw std::runtime_error("RankPartition::allgather_plane: null plane");
        if (elemsPerSample < 1)
            throw std::runtime_error("RankPartition::allgather_plane: elemsPerSample "
                "must be >= 1, got " + std::to_string(elemsPerSample));
        const int size = comm_size(comm);
        std::vector<int> counts(static_cast<std::size_t>(size));
        std::vector<int> displs(static_cast<std::size_t>(size));
        for (int r = 0; r < size; ++r) {
            const Partition p = local(whole, r, size);
            counts[std::size_t(r)] = int(p.count * elemsPerSample);
            displs[std::size_t(r)] = int(p.base * elemsPerSample);
        }
        mpi_check(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, plane,
                                 counts.data(), displs.data(), MPI_DOUBLE, comm),
                  "MPI_Allgatherv");
    }

    /**
     * @brief Combine the R per-rank partials under @p op in a FIXED RANK ORDER.
     *
     * `MPI_Allgather` + `exec::fold`, never `MPI_Allreduce`: the reduction order of
     * a library reduce is the implementation's business, so two ranks may return
     * results differing in the last bits — and "identical on every rank" is the
     * property this structure owes its caller.
     */
    static double allgather_fold(ReduceOp op, double partial, MPI_Comm comm)
    {
        const int size = comm_size(comm);
        std::vector<double> partials(std::size_t(size), identity(op));
        mpi_check(MPI_Allgather(&partial, 1, MPI_DOUBLE, partials.data(), 1,
                                MPI_DOUBLE, comm),
                  "MPI_Allgather");
        return fold(op, partials.data(), partials.size());
    }

#ifndef EAGLE_CPU_ONLY
    /**
     * @brief The device-resident twin of :func:`allgather_plane`: stage D2H, gather
     *        on the host, stage back H2D.
     *
     * @param arr    an owning aether array whose device copy the inner
     *               `DeviceKernel` launches wrote into. Its HOST copy is aether's
     *               own pinned chunk, which is what makes `copyAsync` legal
     *               (aether refuses a direct CPU<->CUDA pair) and what the wire
     *               reads from.
     * @param whole  the partition the world covered.
     * @param comm   the communicator.
     * @param stream the stream the staging rides.
     *
     * This design explicitly DROPS CUDA-aware MPI, so the device pointer never reaches
     * the wire. Raw memcpy is forbidden on aether data, so the staging is
     * `Array::download`/`Array::upload` — `aether::copyAsync` underneath — and this
     * header contains no `cudaMemcpy` of its own. The two synchronisations are
     * unavoidable and deliberate: MPI must not read a buffer a D2H copy is still
     * filling, and the caller must not launch against a device plane an H2D copy is
     * still filling.
     */
    template <class ArrayT>
    static void allgather_staged(ArrayT& arr, const Partition& whole, MPI_Comm comm,
                                 aether::Stream stream)
    {
        arr.download(stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            throw std::runtime_error("RankPartition::allgather_staged: the D2H "
                "staging stream did not synchronise");
        auto host = arr.hostView();
        const int elems = int(host.size() / (whole.nSamples > 0
                                             ? std::size_t(whole.nSamples) : 1));
        allgather_plane(host.data(), whole, comm, elems < 1 ? 1 : elems);
        arr.upload(stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            throw std::runtime_error("RankPartition::allgather_staged: the H2D "
                "staging stream did not synchronise");
    }
#endif
};

}  // namespace exec
}  // namespace eagle
