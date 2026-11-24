// SPDX-License-Identifier: Apache-2.0

// The recordable host operations that are not launches: reduce, copy, fft and gemm. Each
// is checked once (`*_check`, which refuses as eg_door.h states) and then run
// any number of times (`*_exec`): a recording keeps the checked form.
#pragma once

#include "eagle/door/Context.h"

#include <utility>
#include <vector>

namespace eagle {
namespace door {
namespace host {

/** @brief A checked reduce: the planes, the op and the shape (columns, n samples-last). */
struct ReduceOp {
    eg_plane_s* in = nullptr;
    eg_plane_s* out = nullptr;
    eg_plane_s* comp = nullptr;   ///< input companion, or (with ``pair``) the output compensation
    eg_plane_s* index = nullptr;  ///< min and max only: receives the lane of each column's winner, or nullptr
    eg_plane_s* lanes = nullptr;  ///< min and max over the rows of a cell: the lane that won each row (same type as ``in``), or nullptr
    bool position = false;        ///< with ``lanes``: the cell takes the lane of the overall winner instead of its value
    std::int64_t samples = 0;     ///< with ``lanes``: the sample count, the lane of "no winner"
    eg_reduce_op op = EG_REDUCE_SUM;
    bool pair = false;
    std::int32_t code = 0;        ///< the element type, see Ops.cpp
    std::int64_t n = 0, cols = 0;
    std::int64_t first = 0;       ///< a one-column reduce reads the samples [first, first + n) of ``in`` (and ``comp``)
};

ReduceOp reduce_check(const eg_ctx& ctx, const eg_reduce_desc& desc);
void reduce_exec(const ReduceOp& r, int threads);

/** @brief Check the fold of the rows of a cell: ``desc.comp`` holds the compensation rows of a sum, or the rows of the lanes that won (@p lanes
 *  is then the same plane) of a minimum or maximum, which folds pairs (value, lane) and writes the value, or with @p position the lane of the
 *  winner (@p samples when there is none). */
ReduceOp cell_reduce_check(const eg_ctx& ctx, eg_reduce_desc desc, bool position, std::int64_t samples);

/** @brief A checked scan: the two planes, the op, the form and the element type (see Ops.cpp). */
struct ScanOp {
    eg_plane_s* in = nullptr;
    eg_plane_s* out = nullptr;
    eg_scan_op op = EG_SCAN_SUM;
    bool exclusive = false;
    std::int32_t code = 0;
    std::int64_t n = 0;
};

ScanOp scan_check(const eg_ctx& ctx, const eg_scan_desc& desc);
/** @brief Run the checked scan @p s on at most @p threads threads: the blocks and the order of their totals depend on
 *  the extent alone, so every result is the same bits at any thread count. */
void scan_exec(const ScanOp& s, int threads);

/** @brief A checked compaction: the three planes and the extent. */
struct CompactOp {
    eg_plane_s* keep = nullptr;
    eg_plane_s* kept = nullptr;
    eg_plane_s* count = nullptr;
    std::int64_t n = 0;
};

CompactOp compact_check(const eg_ctx& ctx, const eg_compact_desc& desc);
void compact_exec(const CompactOp& c, int threads);

/** @brief A checked sort by key: the three planes, the key bound and the
 * extent. */
struct OrderOp {
    eg_plane_s* keys   = nullptr;
    eg_plane_s* order  = nullptr;
    eg_plane_s* starts = nullptr;
    std::int64_t bound = 0;
    std::int64_t n     = 0;
};

OrderOp order_check(const eg_ctx& ctx, const eg_order_desc& desc);
/** @brief Run the checked sort @p o on at most @p threads threads: the stable
 * permutation is one value, so every thread count gives the same bits. Refuses
 * (EG_BAD_REQUEST) with the number of keys outside [0, bound). */
void order_exec(const OrderOp& o, int threads);

/** @brief The one compaction of the host runtime: the indices of the lanes of @p mask that are kept (a set byte keeps, or with
 *  @p drop drops) into @p kept[0, count), ascending, and their number into @p count; the entries from @p count on are left as
 *  they were. The same result at any @p threads. @p scratch is the working memory: grown when @p n needs more, kept by a caller that compacts again. */
void compact_lanes(const std::uint8_t* mask, bool drop, std::int32_t* kept, std::uint32_t* count, std::int64_t n, int threads,
                   std::vector<std::uint64_t>& scratch);

/** @brief After one launch of @p plan: run each companion's fold entry over its target, then fold each cell's rows
 *  [@p first_row, @p end_row) into the cell the caller bound (the cell's op). A cell is the reduction over the lanes
 *  the launch ran: with no row written it takes the op's identity (0, +inf, -inf, true), never what it held. */
void fold_plan(const eg_plan_s& plan, int threads, std::int64_t first_row, std::int64_t end_row);

/** @brief Whether the memory of @p a and @p b overlaps. */
bool overlaps(const DLTensor& a, const DLTensor& b);

/** @brief Refuse @p plane unless it is compact and (when @p write) writable; @p what names it in the reason. */
void check_access(const eg_plane_s& plane, bool write, const char* what);

void copy_check(const eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst);
void copy_exec(const eg_plane_s& src, const eg_plane_s& dst);

/** @brief Run the checked transform @p plan once, @p in -> @p out, on at most @p threads threads (host/Fft.cpp):
 *  pocketfft, then the plan's Hermitian weight. The result is bitwise the same for any thread count. */
void fft_exec(const eg_fft_s& plan, const eg_plane_s& in, const eg_plane_s& out, int threads);

/** @brief Run the checked product @p gemm once, C (+)= op(@p a) op(@p b) into @p c (host/Gemm.cpp): an ordered triple loop. */
void gemm_exec(const eg_gemm_s& gemm, const eg_plane_s& a, const eg_plane_s& b, const eg_plane_s& c);

}  // namespace host
}  // namespace door
}  // namespace eagle
