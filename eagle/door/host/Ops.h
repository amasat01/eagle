// SPDX-License-Identifier: Apache-2.0

// The recordable host operations that are not launches: reduce, copy and fft. Each
// is checked once (`*_check`, which refuses as eg_door.h states) and then run
// any number of times (`*_exec`): a recording keeps the checked form.
#pragma once

#include "eagle/door/Context.h"

namespace eagle {
namespace door {
namespace host {

/** @brief A checked reduce: the planes, the op and the shape (columns, n samples-last). */
struct ReduceOp {
    eg_plane_s* in = nullptr;
    eg_plane_s* out = nullptr;
    eg_plane_s* comp = nullptr;   ///< input companion, or (with ``pair``) the output compensation
    eg_reduce_op op = EG_REDUCE_SUM;
    bool pair = false;
    std::int32_t code = 0;        ///< the element type, see Ops.cpp
    std::int64_t n = 0, cols = 0;
};

ReduceOp reduce_check(const eg_ctx& ctx, const eg_reduce_desc& desc);
void reduce_exec(const ReduceOp& r, int threads);

/** @brief After one launch of @p plan: run each companion's fold entry over its target, then fold each
 *  cell's rows into the cell the caller bound (the cell's op). */
void fold_plan(const eg_plan_s& plan, int threads);

/** @brief Whether the memory of @p a and @p b overlaps. */
bool overlaps(const DLTensor& a, const DLTensor& b);

/** @brief Refuse @p plane unless it is compact and (when @p write) writable; @p what names it in the reason. */
void check_access(const eg_plane_s& plane, bool write, const char* what);

void copy_check(const eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst);
void copy_exec(const eg_plane_s& src, const eg_plane_s& dst);

/** @brief Run the checked transform @p plan once, @p in -> @p out, on at most @p threads threads (host/Fft.cpp):
 *  pocketfft, then the plan's Hermitian weight. The result is bitwise the same for any thread count. */
void fft_exec(const eg_fft_s& plan, const eg_plane_s& in, const eg_plane_s& out, int threads);

}  // namespace host
}  // namespace door
}  // namespace eagle
