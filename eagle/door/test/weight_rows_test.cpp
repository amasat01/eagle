// SPDX-License-Identifier: Apache-2.0

// The door's built-in weights rows against the table the Python binding serves (python/src/DispatchDefaults.h): every weight of the host and the
// device row equal, so a row edited on one side and not the other fails here. The rows are the header the build force-includes (DOOR_ROWS).
#include "eagle/policy/Dispatch.h"
#include "python/src/DispatchDefaults.h"

#include <cstdio>

namespace pol = eagle::policy;

namespace {

#define EG_WEIGHT_ROW_INIT(...) { __VA_ARGS__ },
const pol::WeightRow kRows[] = { EG_WEIGHT_ROWS(EG_WEIGHT_ROW_INIT) };
#undef EG_WEIGHT_ROW_INIT

/** @brief The built-in row of @p target as the door reads it: the class-all-zero row. */
const pol::WeightRow* builtinRow(pol::Target target) {
    for (const pol::WeightRow& r : kRows)
        if (r.target == static_cast<int>(target) && !r.vendor && !r.family && !r.model) return &r;
    return nullptr;
}

bool same(const char* what, const pol::Weights& a, const pol::Weights& b) {
    const double x[] = { a.arith32, a.special32, a.special64, a.elemBytes, a.launchSeconds, a.splitSeconds, a.sortSeconds,
                         a.laneRate, a.memoryRate, a.cacheBytes, a.cacheRatio, a.chainSeconds, a.vectorWidth };
    const double y[] = { b.arith32, b.special32, b.special64, b.elemBytes, b.launchSeconds, b.splitSeconds, b.sortSeconds,
                         b.laneRate, b.memoryRate, b.cacheBytes, b.cacheRatio, b.chainSeconds, b.vectorWidth };
    bool ok = a.target == b.target;
    for (int i = 0; i < 13; ++i) {
        if (x[i] != y[i]) std::printf("      %s: weight %d is %.17g in the door row and %.17g in the binding table\n", what, i, x[i], y[i]);
        ok = ok && x[i] == y[i];
    }
    return ok;
}

}  // namespace

int main() {
    int failures = 0;
    const struct { const char* what; pol::Target target; pol::Machine machine; } cases[] = {
        { "host", pol::Target::Host, { 4, 12.7e6 } }, { "device", pol::Target::Device, { 1, 0 } } };
    for (const auto& c : cases) {
        const pol::WeightRow* row = builtinRow(c.target);
        if (row == nullptr && c.target == pol::Target::Device) {   // a host-only door's header has no device row, and its door never reads one
            std::printf("SKIP  this row header has no device weights row (a host-only door)\n");
            continue;
        }
        const bool ok = row != nullptr && same(c.what, pol::weightsOfRow(*row, c.machine), eagle_python::defaultDispatchWeights(c.target));
        std::printf("%s  the door's built-in %s weights row equals the Python binding's default weights, value for value\n", ok ? "PASS" : "FAIL", c.what);
        failures += !ok;
    }
    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures != 0;
}
