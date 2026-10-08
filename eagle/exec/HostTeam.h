// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `eagle::exec::HostTeam` — the HOST execution structure of the heterogeneous
// execution contract (aether-abi/2).
//
// WHAT MOVED. Under aether-abi/1 the plugin owned its own threading: the generator emitted
// `{name}_host(void* const* params, int32_t n)` with an INTERNAL
// `#pragma omp parallel for`, and eagle's host
// registry just called it once. Under v2 the entry is a SERIAL range over
// `[base, base+count)` and the threading is EAGLE'S: this class cuts the partition
// into contiguous TILES and calls the entry once per tile, each tile carrying its
// OWN triple: with the plugin's own loop
// removed but the plane addressing left alone, every tile would have re-based its
// `t` at 0 and every tile would have written accum/wide column `t`. A per-tile
// triple makes the global index `base + local` on every tile, so the columns are
// disjoint by construction.
//
// Deliberately minimal: team + contiguous tiles ONLY. No barrier/single,
// no `nowait` chaining, no SIMD-inner arm -- those arrive when a downstream DSL
// rework consumes this API.
//
// SCHEDULE is dynamic: a thread takes the next tile when it finishes its last, so a
// tile that runs long (its samples take more steps, or its thread was descheduled
// on a shared machine) does not hold the team's one barrier while the others idle.
// Tiles are still contiguous and each still gets its own triple, so which thread
// ran a tile changes no result.
//
// TILE SIZE is the `Host::launch` rule, not a new one: `aether::optimalTileSize<T>
// (bytesPerSample)` (aether `backend/cpu/Tiled.h`), the same L2-derived sizing
// `eagle::cpu::Host::launch` passes through — pass the SAME `bytesPerSample` across
// every host kernel in a step for cross-kernel L2 residency, exactly as documented
// on `Host::launch` above.
#pragma once

#include "eagle/exec/Partition.h"
#include "eagle/typedefs.h"

#include <aether/backend/cpu/Tiled.h>

#include <cstdint>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eagle {
namespace exec {

/**
 * @brief The host execution structure: OpenMP over contiguous tiles of one
 *        partition, each tile driving the SAME body through its own triple.
 */
struct HostTeam {

    /** @brief The tile size (in samples) this team would use for @p bytesPerSample
     *  — `Host::launch`'s own L2-derived rule (`aether::optimalTileSize<double>`),
     *  exposed so a test can fix the schedule rather than infer it. `0` falls back
     *  to aether's compile-time `DEFAULT_TILE_SIZE`, exactly as `Host::launch` does. */
    static std::size_t tile_size(std::size_t bytesPerSample = 0) {
        return aether::optimalTileSize<double>(bytesPerSample);
    }

    /** @brief How many tiles @p part would be cut into. */
    static std::size_t tile_count(const Partition& part,
                                  std::size_t bytesPerSample = 0) {
        if (part.count <= 0) return 0;
        const std::size_t tile = tile_size(bytesPerSample);
        return (static_cast<std::size_t>(part.count) + tile - 1) / tile;
    }

    /**
     * @brief Run @p fn over @p part as a team of contiguous tiles.
     *
     * @param fn              the plugin's SERIAL v2 entry.
     * @param args            the packed role args (`params[]`), passed WHOLE to
     *                        every tile — eagle does NO pointer arithmetic on
     *                        them (L2); the body slices by its own triple.
     * @param part            the partition this team covers.
     * @param bytesPerSample  the caller's per-sample working-set estimate, driving
     *                        the L2-derived tile size (`Host::launch`'s rule).
     * @return the number of tiles actually run.
     */
    static std::size_t run(HostEntryV2 fn, void* const* args, const Partition& part,
                           std::size_t bytesPerSample = 0) {
        if (fn == nullptr)
            throw std::runtime_error("HostTeam::run: null host entry");
        if (part.count <= 0) return 0;
        const std::int64_t tile =
            static_cast<std::int64_t>(tile_size(bytesPerSample));
        const std::int64_t ntiles = (part.count + tile - 1) / tile;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (std::int64_t t = 0; t < ntiles; ++t) {
            const std::int64_t base  = part.base + t * tile;
            const std::int64_t count =
                (t + 1 == ntiles) ? (part.base + part.count - base) : tile;
            fn(args, base, count, part.nSamples);
        }
        return static_cast<std::size_t>(ntiles);
    }

    /** @brief Vector-args convenience twin of :func:`run`. */
    static std::size_t run(HostEntryV2 fn, const std::vector<void*>& args,
                           const Partition& part, std::size_t bytesPerSample = 0) {
        return run(fn, args.data(), part, bytesPerSample);
    }

    /**
     * @brief Run @p fn SERIALLY over @p part — one call, one triple.
     *
     * The correctness REFERENCE the band rows are judged against (the generator's
     * eagle-free host runtime becomes exactly this), and the shape a
     * legacy whole-view launch takes.
     */
    static void run_serial(HostEntryV2 fn, void* const* args, const Partition& part) {
        if (fn == nullptr)
            throw std::runtime_error("HostTeam::run_serial: null host entry");
        if (part.count <= 0) return;
        fn(args, part.base, part.count, part.nSamples);
    }
};

}  // namespace exec
}  // namespace eagle
