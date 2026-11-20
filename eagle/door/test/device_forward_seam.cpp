// SPDX-License-Identifier: Apache-2.0

// The device forward pieces' test seam: the save of a conditional's test cells, the per-iteration copy of a WHILE body and the read-back, called as
// the door's own reverse calls them. A test that links the door's objects reaches them here; they are no part of the library.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"

using namespace eagle::door;

extern "C" eg_status df_scratch(eg_ctx* ctx, size_t bytes, void** out) {
    return call(ctx, [&](eg_ctx& c) { *out = device_record_scratch(c, bytes); });
}

extern "C" eg_status df_save_test(eg_ctx* ctx, eg_plane* cell, int64_t cell_index, eg_plane* baseline, int64_t baseline_index, int32_t* slot) {
    return call(ctx, [&](eg_ctx& c) { *slot = device_record_save_test(c, cell, cell_index, baseline, baseline_index); });
}

extern "C" eg_status df_loop_copy(eg_ctx* ctx, eg_plane* src, void* buffer, size_t bytes) {
    return call(ctx, [&](eg_ctx& c) {
        const DLTensor& t = src->tensor->dl_tensor;
        device_record_loop_copy(c, static_cast<const char*>(t.data) + t.byte_offset, buffer, bytes);
    });
}

/** @brief The saved tests of @p graph: slot s entry k at ``cells[s * stride + k]`` (``bases`` alike), ``n_slots`` slots, and the iteration count of each WHILE. */
extern "C" eg_status df_saved(eg_ctx* ctx, eg_graph* graph, int stride, uint32_t* cells, uint32_t* bases, int32_t* n_slots, uint32_t* iterations, int32_t* n_loops) {
    return call(ctx, [&](eg_ctx& c) {
        DeviceSaved saved;
        device_saved_read(c, graph, saved);
        *n_slots = static_cast<int32_t>(saved.cell.size());
        for (std::size_t s = 0; s < saved.cell.size(); ++s)
            for (std::size_t k = 0; k < saved.cell[s].size() && static_cast<int>(k) < stride; ++k) {
                cells[s * static_cast<std::size_t>(stride) + k] = saved.cell[s][k];
                bases[s * static_cast<std::size_t>(stride) + k] = saved.baseline[s][k];
            }
        *n_loops = static_cast<int32_t>(saved.iterations.size());
        for (std::size_t i = 0; i < saved.iterations.size(); ++i) iterations[i] = saved.iterations[i];
    });
}

extern "C" eg_status df_read(eg_ctx* ctx, const void* device, void* host, size_t bytes) {
    return call(ctx, [&](eg_ctx& c) { device_read(c, device, host, bytes); });
}
