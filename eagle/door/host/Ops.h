// SPDX-License-Identifier: Apache-2.0

// The recordable host operations that are not launches: reduce and copy. Each
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

void copy_check(const eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst);
void copy_exec(const eg_plane_s& src, const eg_plane_s& dst);

}  // namespace host
}  // namespace door
}  // namespace eagle
