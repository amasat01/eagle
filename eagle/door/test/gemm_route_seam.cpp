// SPDX-License-Identifier: Apache-2.0

// The gemm route rows' test seam: a test that links the door's objects (not the shared library, which exports eg_* names only) reads a synthetic route
// table into a fresh device context through the same function eg_ctx_new calls. It is no part of the library.
#include "eagle/door/Context.h"

#include <vector>

/** @brief Read @p count rows of (target, family, model, dtype_bits, route) into @p ctx as eg_ctx_new reads the built-in table; EG_FAULT when the table cannot route. */
extern "C" eg_status gemm_route_rows_set(eg_ctx* ctx, int (*rows)[5], int count) {
    std::vector<eagle::policy::GemmRouteRow> table;
    for (int i = 0; i < count; ++i) table.push_back({ rows[i][0], rows[i][1], rows[i][2], rows[i][3], rows[i][4] });
    return eagle::door::call(ctx, [&](eg_ctx& c) { eagle::door::resolve_gemm_routes(c, table.data(), count); });
}
