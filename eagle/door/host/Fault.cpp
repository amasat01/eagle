// SPDX-License-Identifier: Apache-2.0

// Checked units: a unit built with the checks of its generated code (eg_unit_meta.fault_slot) gets a two-cell int64 fault cell from the door per plan.
// The door sets both cells to -1 before each launch, graph replay and run, and reads them after: the generated code claims cell 0 with one atomic
// compare-and-swap from -1 to the violated slot and only the winner writes the offending index to cell 1, so the pair is always one violation.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"

namespace eagle {
namespace door {

namespace {

constexpr std::int64_t kClear[2] = { -1, -1 };

bool on_device(const eg_plan_s& plan) { return plan.structure == EG_STRUCTURE_DEVICE; }

}  // namespace

void fault_reset(eg_ctx& ctx, const eg_plan_s& plan) {
    if (plan.fault == nullptr) return;
    const DLTensor& t = plan.fault->tensor->dl_tensor;
    void* cells = static_cast<char*>(t.data) + t.byte_offset;
    if (on_device(plan)) device_write(ctx, cells, kClear, sizeof(kClear));
    else std::memcpy(cells, kClear, sizeof(kClear));
}

void fault_check(eg_ctx& ctx, const eg_plan_s& plan) {
    if (plan.fault == nullptr) return;
    const DLTensor& t = plan.fault->tensor->dl_tensor;
    const void* cells = static_cast<const char*>(t.data) + t.byte_offset;
    std::int64_t seen[2];
    if (on_device(plan)) device_read(ctx, cells, seen, sizeof(seen));
    else std::memcpy(seen, cells, sizeof(seen));
    if (seen[0] == -1) return;    // cell 1 is read only once cell 0 is claimed
    const eg_unit_s& unit = *plan.unit;
    const std::int64_t slot = seen[0];
    if (slot < 0 || slot >= static_cast<std::int64_t>(unit.slots.size()))
        refuse(EG_FAULT, "unit '" + unit.name + "': the fault cell names slot " + std::to_string(slot) + ", which the unit does not have");
    const eg_plane_s* bound = plan.bindings[static_cast<std::size_t>(slot)].plane;
    const std::string extent = bound != nullptr ? std::to_string(elements(bound->tensor->dl_tensor)) : std::to_string(plan.bindings[static_cast<std::size_t>(slot)].handle.samples);
    refuse(EG_FAULT, "unit '" + unit.name + "': slot " + std::to_string(slot) + " ('" + unit.slots[static_cast<std::size_t>(slot)].name + "'): index " +
           std::to_string(seen[1]) + " outside its " + extent + " elements");
}

void fault_reset(eg_ctx& ctx, const eg_runner_s& runner) {
    for (const eg_plan_s* p : runner.plans) fault_reset(ctx, *p);
}

void fault_check(eg_ctx& ctx, const eg_runner_s& runner) {
    for (const eg_plan_s* p : runner.plans) fault_check(ctx, *p);
}

}  // namespace door
}  // namespace eagle
