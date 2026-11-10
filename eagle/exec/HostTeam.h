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

#include "eagle/exec/HostPool.h"
#include "eagle/exec/Partition.h"
#include "eagle/typedefs.h"

#include <aether/backend/cpu/Tiled.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace eagle {
namespace exec {

/**
 * @brief The host execution structure: eagle's host pool over contiguous tiles
 *        of one partition, each tile driving the SAME body through its own triple.
 */
struct HostTeam {

    /** @brief How long the caller runs the first tiles itself to learn what a tile costs (microseconds). */
    static constexpr double HOST_PROBE_US = 2.0;
    /** @brief The least work one claim of a team thread carries (microseconds). */
    static constexpr double HOST_LANE_US = 5.0;

    /// What one tile of an entry cost when this thread last probed it.
    struct Cost { HostEntryV2 fn = nullptr; std::int64_t tile = 0; double perTileUs = 0.0; std::uint32_t launches = 0; };
    static Cost& remembered(HostEntryV2 fn) {
        static thread_local Cost table[16];
        return table[(reinterpret_cast<std::uintptr_t>(fn) >> 4) & 15];
    }

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
        struct Tiles { HostEntryV2 fn; void* const* args; Partition part; std::int64_t tile, ntiles, first, grain; };
        Tiles tiles{ fn, args, part, tile, ntiles, 0, 1 };
        const auto oneTile = [](const Tiles& s, std::int64_t t) {
            const std::int64_t base = s.part.base + t * s.tile;
            const std::int64_t count = (t + 1 == s.ntiles) ? (s.part.base + s.part.count - base) : s.tile;
            s.fn(s.args, base, count, s.part.nSamples);
        };
        if (ntiles == 1 || host_max_threads() <= 1) {   // no thread to wake
            for (std::int64_t t = 0; t < ntiles; ++t) oneTile(tiles, t);
            return static_cast<std::size_t>(ntiles);
        }
        // COST. What a tile costs is not known before it runs: a tile of a cheap kernel is tens of nanoseconds, of
        // a heavy one tens of microseconds. The calling thread remembers it per entry; with none remembered (and
        // again every 64th launch, since a kernel's cost can move with its data) it PROBES: it runs the first
        // tiles itself, in order, 1 then 2 then 4 ..., until HOST_PROBE_US is spent. The tiles and their triples
        // never change, only who runs them, so no result depends on any of this.
        Cost& cost = remembered(fn);
        std::int64_t done = 0;
        if (cost.fn != fn || cost.tile != tile || (++cost.launches & 63) == 0) {
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            double spent = 0.0;
            for (std::int64_t step = 1; done < ntiles && spent < HOST_PROBE_US; step *= 2) {
                const std::int64_t stop = std::min(ntiles, done + step);
                for (; done < stop; ++done) oneTile(tiles, done);
                spent = std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            }
            cost = Cost{ fn, tile, spent / static_cast<double>(done), 0 };
        }
        const std::int64_t rest = ntiles - done;
        const double perTile = cost.perTileUs;
        if (perTile * static_cast<double>(rest) < 2 * detail::HOST_FLOOR_US) {   // waking the team costs more than it saves
            for (; done < ntiles; ++done) oneTile(tiles, done);
            return static_cast<std::size_t>(ntiles);
        }
        // GRAIN. A thread takes `grain` consecutive tiles per claim, so the claim (two shared atomics) is paid
        // once per HOST_LANE_US of work and not once per tile; never fewer than 4 claims per thread, so a
        // long tile still cannot hold the team.
        const std::int64_t team = host_max_threads();
        tiles.first = done;
        tiles.grain = std::clamp<std::int64_t>(static_cast<std::int64_t>(HOST_LANE_US / perTile) + 1, 1,
                                               std::max<std::int64_t>(1, rest / (4 * team)));
        const auto runLane = [](const void* ctx, std::uint32_t lane) {
            const auto& s = *static_cast<const Tiles*>(ctx);
            const std::int64_t from = s.first + lane * s.grain;
            const std::int64_t to = std::min(s.ntiles, from + s.grain);
            for (std::int64_t t = from; t < to; ++t) {
                const std::int64_t base = s.part.base + t * s.tile;
                const std::int64_t count = (t + 1 == s.ntiles) ? (s.part.base + s.part.count - base) : s.tile;
                s.fn(s.args, base, count, s.part.nSamples);
            }
        };
        const std::int64_t lanes = (rest + tiles.grain - 1) / tiles.grain;
        const detail::HostPool::Job job{ static_cast<std::uint32_t>(lanes), nullptr, runLane, &tiles };
        detail::HostPool::instance().run(job, static_cast<int>(team));
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
