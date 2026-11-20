// SPDX-License-Identifier: Apache-2.0

// The door's device seam: the functions the device group (device/Device.cpp) provides to the core. The core
// reaches the device group only through these declarations. A build without the group provides the
// NOT_WRITTEN stubs for them, so the core, the host group and the library load the same either way.
#pragma once

#include "eagle/door/Context.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Shape.h"

namespace eagle {
namespace door {

// The device group (device/Device.cpp); the host-only build refuses EG_CTX_DEVICE and structure DEVICE.
void device_open(eg_ctx& ctx, std::int32_t device);
void device_close(eg_ctx& ctx) noexcept;
/** @brief The devices the driver sees (device/Device.cpp); 0 with no driver, no device, or no device group built. */
std::int32_t device_count(eg_ctx& ctx);
void device_unit_load(eg_unit_s& unit, const eg_unit_meta& meta, void* library);
void device_unit_free(eg_unit_s& unit) noexcept;
void device_plan_check(const eg_ctx& ctx, const eg_unit_s& unit);
/** @brief eg_plan_new of a routable DEVICE plan of a gemm unit, and eg_gemm_new of a device product: open cuBLAS (EG_FAULT naming the libraries tried) and make the handle of the stream in scope. */
void device_gemm_ready(eg_ctx& ctx);
void device_launch(eg_plan_s& plan);
/** @brief eg_launch of a DEVICE plan: its eagle-made finished counter zeroed on the stream, then device_launch, so the
 *  counter holds the samples this launch finished (a runner's launches go through device_launch and keep counting). */
void device_launch_call(eg_plan_s& plan);
void device_classify(eg_plan_s& plan);
void device_loop_drop(eg_runner_s& runner);
void device_sync(eg_ctx& ctx);
void device_record_begin(eg_ctx& ctx);
void device_record_launch(eg_ctx& ctx, eg_plan_s& plan);
/** @brief eg_run of a DEVICE runner while recording: the capture ends and the run becomes a step of the graph that eg_graph_launch runs eagerly
 *  (reset, then the run) between that capture and the next; refused NOT_WRITTEN inside a conditional arm, a fork branch or a WHILE. */
void device_record_run(eg_ctx& ctx, eg_runner_s* runner);
/** @brief eg_record_abort on a device context: end and destroy the open capture, release what the recording holds. */
void device_record_abort(eg_ctx& ctx);
/** @brief eg_record_end on a device context; with @p diff (the tape of EG_RECORD_DIFF, planned by diff_plan over the recorded launches, which
 *  must be all it holds) the capture is made again with the snapshot copies in front of the launches that overwrite a plane. */
void device_record_end(eg_ctx& ctx, const eg_record_end_desc* desc, eg_graph** out);
/** @brief The recording calls of a graph builder on a device context (device/Device.cpp), the device half of ``eg_record_cond_begin``,
 *  ``_cond_arm``, ``_cond_end``, ``eg_record_fork_begin``, ``_fork_branch``, ``_fork_end`` and ``eg_record_wait``: conditional nodes
 *  and parallel branches woven into the capture. Each is called instead of the host's when ``ctx.dev_rec`` is set. */
void device_record_cond_begin(eg_ctx& ctx, const eg_cond_desc* desc);
void device_record_cond_arm(eg_ctx& ctx);
void device_record_cond_end(eg_ctx& ctx);
void device_record_fork_begin(eg_ctx& ctx);
void device_record_fork_branch(eg_ctx& ctx);
void device_record_fork_end(eg_ctx& ctx);
void device_record_wait(eg_ctx& ctx, std::int32_t n_planes, eg_plane_s* const* planes);
/** @brief Whether the driver has IF_ELSE and SWITCH conditional nodes: the truth of the capability bits EG_CAP_COND_IF_ELSE and EG_CAP_COND_SWITCH on a device context. */
bool device_cond_arms_supported();
/** @brief The shape of the device recording in progress; the host calls it keeps as nodes (a host launch, run, copy ...) add themselves to it. */
GraphShape& device_shape(eg_ctx& ctx);
/** @brief eg_graph_describe of device graph @p graph: the description built at record time. */
void device_graph_describe(eg_ctx& ctx, const eg_graph* graph, eg_graph_description* out);
/** @brief Bring the tests a replay of device graph @p diff saved (see device_record_end) into ``diff.saved``, in one sync: the reverse reads it
 *  through ``leaf_ran``. Called by ``eg_graph_reverse`` of a device graph. */
void device_diff_read_saved(eg_ctx& ctx, GraphDiff& diff);
/** @brief The tape of device graph @p graph (null: it is a plain graph). */
GraphDiff* device_graph_diff(eg_ctx& ctx, eg_graph* graph);
/** @brief Record a HOST node on the capture in scope: when the graph replays, the driver's host-function launch calls
 *  @p run(@p user, failure) on a driver thread, ordered on the stream. @p run must not throw and makes no CUDA call (the
 *  driver forbids it); it stores a failure in @p failure, which the context's next ``eg_graph_launch`` or ``eg_sync``
 *  returns. The graph owns @p user until it is freed, when @p release(ctx, user) drops it (after the stream is idle). */
void device_record_host_step(eg_ctx& ctx, void (*run)(void* user, StepFailure& failure) noexcept, void* user,
                             void (*release)(eg_ctx& ctx, void* user));
bool device_graph(const eg_ctx& ctx, const eg_graph* graph);
void device_graph_launch(eg_ctx& ctx, eg_graph* graph);
void device_graph_free(eg_ctx& ctx, eg_graph* graph);
void device_graphs_drop(eg_ctx& ctx) noexcept;
void device_stream_use(eg_ctx& ctx, std::uint64_t stream);
void device_quiet(const eg_ctx& ctx, const char* call, eg_status status, const char* why);
/** @brief The refusal reason of the host calls a device recording does not keep (eg_reverse). */
inline constexpr const char* kHostWorkInGraph = "host work inside a device graph is not written in this build of the door";
void device_plan_cells(eg_plan_s& plan);
void device_plan_free(eg_plan_s& plan) noexcept;
/** @brief Device memory of @p bytes on the context's device, zeroed when @p zero (device/Device.cpp). */
void* device_alloc(eg_ctx& ctx, std::size_t bytes, bool zero = true);
void device_free(eg_ctx& ctx, void* data) noexcept;
/** @brief Page-locked host memory of @p bytes, zeroed when @p zero; freed by pinned_free (a forked child makes no driver call). */
void* pinned_alloc(eg_ctx& ctx, std::size_t bytes, bool zero);
void pinned_free(eg_ctx& ctx, void* data) noexcept;
void device_cell_work(eg_plan_s& plan);
/** @brief eg_reduce and eg_copy over device planes, on the context's stream (device/Device.cpp); checked as the host's. */
void device_reduce(eg_ctx& ctx, const eg_reduce_desc& desc);
/** @brief The already checked reduce @p r of device planes on the context's stream; kept in the recording if there is one. */
void device_reduce_leaf(eg_ctx& ctx, const host::ReduceOp& r);
void device_copy(eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst);
/** @brief The device half of a transform plan (device/Fft.cpp): build @p plan's cuFFT plans and the input copy of a C2R
 *  (FAULT when cuFFT cannot be opened); run it on the context's stream, kept alive by the capture in progress; free it. */
void device_fft_new(eg_ctx& ctx, eg_fft_s& plan);
/** @brief eg_gemm_run of a device product: queued on the context's stream (never makes a handle), legal in a capture. */
void device_gemm_run(eg_ctx& ctx, const eg_gemm_s& gemm, eg_plane_s* a, eg_plane_s* b, eg_plane_s* c);
void device_fft_run(eg_ctx& ctx, eg_fft_s& plan, eg_plane_s* in, eg_plane_s* out);
void device_fft_free(eg_ctx& ctx, eg_fft_s& plan) noexcept;
/** @brief Read @p bytes (a multiple of 4) of device memory at @p device into @p host, ordered after the context's queued work. */
void device_read(eg_ctx& ctx, const void* device, void* host, std::size_t bytes);
/** @brief The device ops of a recorded DEVICE run and its reverse (device/Device.cpp), on the context's stream:
 *  a host-to-device write (waits), a device copy, a zeroing, one column op (``diff_columns``: 0 gather, 1 accumulate,
 *  2 fill, 3 scatter), one launch of @p plan over its first @p count samples with the block resolved at @p of samples,
 *  the mask count into a device counter, the canonical compensated pair of @p n values (waits), the free bytes. */
void device_write(eg_ctx& ctx, void* device, const void* host, std::size_t bytes);
void device_copy_bytes(eg_ctx& ctx, void* dst, const void* src, std::size_t bytes);
void device_zero(eg_ctx& ctx, void* data, std::size_t bytes);
void device_columns(eg_ctx& ctx, int op, void* dst, std::int64_t dst_ld, const void* src, std::int64_t src_ld,
                    const std::int64_t* index, std::int64_t count, std::int64_t rows, unsigned bytes, std::int64_t value);
void device_launch_prefix(eg_plan_s& plan, std::int64_t count, std::int64_t of);
void device_mask_count(eg_ctx& ctx, const void* mask, std::int64_t n, unsigned width, std::uint64_t counter);
void device_pair_sum(eg_ctx& ctx, bool f64, const void* in, std::int64_t n, void* hi, void* comp);
std::int64_t device_free_bytes(eg_ctx& ctx);
/** @brief Queue on the context's stream one forward event operation over the device planes @p planes (an EventPlanes of @p bytes wide reals). */
void device_event_op(eg_ctx& ctx, unsigned bytes, unsigned op, const void* planes);
/** @brief A recorded DEVICE run's active set (device/Device.cpp): queue the compaction of @p mask into the runner's device map and
 *  count (the reorder trigger too, for a reordering set); with @p reorder and a fired trigger, the reorder of @p owned (the
 *  mask first) and its count; returns whether it reordered. */
bool device_active_compact(eg_ctx& ctx, eg_runner_s& runner, const void* mask, const std::vector<Owned>& owned, bool reorder);
/** @brief Put every plane of @p owned back in sample order and recompute the map (the end of a reordered run). */
void device_active_restore(eg_ctx& ctx, eg_runner_s& runner, const void* mask, const std::vector<Owned>& owned);
void device_runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out);
/** @brief A DEVICE runner's cells, live-slot map and reordering state written as eg_runner_new leaves them (stream order). */
void device_runner_initial(eg_ctx& ctx, eg_runner_s& runner);
void device_run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report);
/** @brief Refuse NOT_WRITTEN "the device group is not built" for the device-group call @p call, in a door built without it. */
void device_group_check(const char* call);

}  // namespace door
}  // namespace eagle
