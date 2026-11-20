// SPDX-License-Identifier: Apache-2.0

// The captured run-until-done loop of a DEVICE runner (device/Loop.cu, built by
// nvcc over eagle's capture seam). Plain types only: no CUDA or eagle header
// reaches the files that include this one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace eagle {
namespace door {

struct DeviceLoop;  ///< an instantiated graph: one WHILE node over the step

/** @brief A step inside the loop body that runs, on the stream it is given, only while ``*count != *base``. */
struct LoopGate {
    std::function<void(std::uint64_t)> run;
    const unsigned* count;
    const unsigned* base;
};

/** @brief A per-sample plane a reorder moves: its device address and element size. */
struct DevicePlane {
    void* data;
    unsigned bytes;
};

/**
 * @brief Capture on @p stream one WHILE node that runs @p body (it queues the step on the stream it is
 * given) while ``*finished != *total``, at most @p cap times per launch; @p cells is the device
 * ``[remaining, ran]`` pair the node keeps (``ran``: the iterations of the last launch). Each of @p gates is an IF
 * node that follows the body, in order, in each iteration. Instantiated.
 */
DeviceLoop* loop_capture(std::uint64_t stream, const unsigned* finished, const unsigned* total, unsigned* cells,
                         unsigned cap, const std::function<void(std::uint64_t)>& body, const std::vector<LoopGate>& gates = {});

/** @brief Queue one launch of @p loop on @p stream. */
void loop_launch(DeviceLoop* loop, std::uint64_t stream);

/** @brief Page-locked host memory of @p bytes; freed by host_pinned_free. */
void* host_pinned_new(std::size_t bytes);

/** @brief Free host_pinned_new memory (NULL-safe). */
void host_pinned_free(void* data) noexcept;

/** @brief A device allocation of @p bytes from the stream-ordered pool, zeroed when @p zero, ready for work queued on
 *  @p stream; no call waits on the device. */
void* alloc_async(std::uint64_t stream, std::size_t bytes, bool zero);

/** @brief Release an alloc_async allocation made on @p alloc_stream once the work queued on @p alloc_stream and on
 *  @p current before this call has run (0-safe); no host wait. */
void free_async(std::uint64_t alloc_stream, std::uint64_t current, void* data) noexcept;

/** @brief Write @p count host values into device @p cells, stream-ordered on @p stream. */
void cells_write(std::uint64_t stream, std::uint64_t cells, const unsigned* values, unsigned count);

/** @brief Read @p count device @p cells into @p values once @p stream has drained to them. */
void cells_read(std::uint64_t stream, std::uint64_t cells, unsigned* values, unsigned count);

/** @brief Queue on @p stream: ``*counter`` = the samples of the @p n-sample mask whose @p width-byte element
 *  is nonzero (the old path's mask count, re-syncing the finished counter to the mask). */
void mask_count(std::uint64_t stream, const void* mask, std::int64_t n, unsigned width, unsigned* counter);

/** @brief Elements of the cell's element type each of the two work buffers of a ``cell_fold`` over @p rows rows holds. */
std::int64_t cell_work_elems(std::int64_t rows);

/** @brief Queue on @p stream: ``*out`` = the sum (@p op 0, with the compensation rows @p comp, or none when null), max (1) or min (2)
 *  of the @p rows float values at @p in (@p bits wide, 32 or 64), by eagle's canonical reduction tree; @p wa and @p wb are the work buffers. */
void cell_fold(std::uint64_t stream, const void* in, const void* comp, std::int64_t rows, unsigned bits, int op, void* out, void* wa,
               void* wb);

/** @brief Bytes of scratch the compaction of @p n samples needs. */
std::size_t compact_scratch_bytes(std::int64_t n);

/** @brief Queue on @p stream the compaction of the @p n-sample mask (a set byte = finished) into the ascending live
 *  index @p map and its @p count.
 *  With @p live32 set the reorder trigger is computed too (the three words and @p theta). */
void compact_step(std::uint64_t stream, const void* mask, std::int64_t n, std::int32_t* map, unsigned* count,
                  void* scratch, unsigned* live32, const unsigned* span, unsigned* fire, float theta);

/** @brief Queue on @p stream ``*done_at_last = *finished`` and ``++*compactions``. */
void compact_mark(std::uint64_t stream, const unsigned* finished, unsigned* done_at_last, unsigned* compactions);

/** @brief Bytes of scratch a reorder or restore over @p planes of @p n samples needs. */
std::size_t reorder_scratch_bytes(std::int64_t n, const std::vector<DevicePlane>& planes);

/** @brief Queue on @p stream ``*reorders += *fire != 0``, then the reorder of @p planes (the mask among them) of the
 *  @p n samples by the mask: live samples first, ``perm``/``inv``/@p map/@p count/@p span follow, ``*fire`` cleared. */
void reorder_step(std::uint64_t stream, const std::vector<DevicePlane>& planes, const void* mask, std::int32_t* perm,
                  std::int32_t* inv, std::int32_t* map, unsigned* count, unsigned* span, unsigned* fire, void* scratch,
                  std::int64_t n, unsigned* reorders);

/** @brief Queue on @p stream the undoing of every reorder: each of @p planes back in sample order, ``perm``/``inv`` identity. */
void restore_step(std::uint64_t stream, const std::vector<DevicePlane>& planes, std::int32_t* perm, std::int32_t* inv,
                  void* scratch, std::int64_t n);

/** @brief Bytes of scratch a K-class sort of @p n samples over @p planes needs (``classes`` classes). */
std::size_t class_sort_scratch_bytes(std::int64_t n, const std::vector<DevicePlane>& planes, unsigned classes);

/** @brief Queue on @p stream eagle's stable K-class sort: the slots of the @p n samples by ``(live, class)`` (a set mask byte = dead, last), every
 *  plane of @p planes moving with them (the mask and the uint8 class plane @p class_plane among them); ``perm``/``inv`` follow, ``offsets``
 *  (``classes + 1`` uint32) holds the first slot of each class's live run, ``*count`` and ``*span`` become the live count, ``*sorts`` is incremented. */
void class_sort_step(std::uint64_t stream, const std::vector<DevicePlane>& planes, const void* mask, unsigned classes, const void* class_plane,
                     std::int32_t* perm, std::int32_t* inv, std::int32_t* map, unsigned* count, unsigned* span, unsigned* offsets, unsigned* sorts,
                     void* scratch, std::int64_t n);

/** @brief Queue on @p stream the compaction of the @p n-sample mask whose SET bytes are kept (a stale mask) into the ascending index
 *  @p map and its @p count; @p scratch is ``compact_scratch_bytes(n)``. */
void compact_keep_step(std::uint64_t stream, const void* mask, std::int64_t n, std::int32_t* map, unsigned* count, void* scratch);

/** @brief Queue on @p stream: clear every byte of the @p n-byte stale mask whose sample is dead (a set byte of @p dead): ``stale &= live``. */
void stale_live_step(std::uint64_t stream, void* stale, const void* dead, std::int64_t n);

/** @brief Queue on @p stream ``perm``/``inv``/@p map the identity over @p n samples, ``*count`` = ``*span`` = @p n. */
void reset_step(std::uint64_t stream, std::int32_t* perm, std::int32_t* inv, std::int32_t* map, unsigned* count, unsigned* span, std::int64_t n);

/** @brief Reduce each of the @p cols columns of the (cols, n) device plane @p in into @p out on @p stream by eagle's
 *  canonical ordered tree (the order the host runs; ``code``: 0 f32, 1 f64, 2 i32, 3 i64, 4 u8 with op 3 ALL; ``op``: 0 SUM,
 *  1 MAX, 2 MIN, 3 ALL). @p comp: a float sum's companion input, or with @p pair the output of the compensation beside
 *  ``out``. Waits for the result (its work buffers are freed here). */
void reduce_step(std::uint64_t stream, int code, int op, const void* in, std::int64_t n, std::int64_t cols, void* out, void* comp,
                 bool pair);

/** @brief Queue on @p stream a device-to-device copy of @p bytes. */
void copy_step(std::uint64_t stream, void* dst, const void* src, std::size_t bytes);

/** @brief Queue on @p stream the norm and Hermitian weight of a transform's @p count floats (@p bits wide, 32 or 64): every
 *  float is scaled by @p scale, and by @p scale * @p weight when its bin is interior, that is 0 < bin < @p edge with
 *  ``bin = i / run % bins`` (@p run floats per bin, @p bins bins along the halved axis). */
void fft_weigh(std::uint64_t stream, unsigned bits, void* data, std::int64_t count, std::int64_t run, std::int64_t bins,
               std::int64_t edge, double scale, double weight);

/** @brief Queue on @p stream a host-to-device copy of @p bytes and wait for it (the host bytes are the caller's). */
void write_bytes(std::uint64_t stream, void* dst, const void* src, std::size_t bytes);

/** @brief Queue on @p stream a device-to-host copy of @p bytes and wait for it (the host bytes are the caller's, possibly pageable). */
void read_bytes(std::uint64_t stream, void* dst, const void* src, std::size_t bytes);

/** @brief Queue on @p stream the zeroing of @p bytes at @p dst. */
void zero_bytes(std::uint64_t stream, void* dst, std::size_t bytes);

/** @brief Queue on @p stream one column op over sample-last planes (the old path's ``eagle_backend_x_diff_columns``), for
 *  c < @p count and r < @p rows: @p op 0 gathers ``dst[r*dst_ld + c] = src[r*src_ld + index[c]]``, 1 accumulates
 *  ``dst[r*dst_ld + c] += src[r*dst_ld + c]`` (@p bytes 4: f32, 8: f64), 2 fills ``dst[c] = value`` (4: int32, 8: int64),
 *  3 scatters ``dst[r*dst_ld + index[c]] = src[r*src_ld + c]``. */
void diff_columns(std::uint64_t stream, int op, void* dst, std::int64_t dst_ld, const void* src, std::int64_t src_ld,
                  const std::int64_t* index, std::int64_t count, std::int64_t rows, unsigned bytes, std::int64_t value);

/** @brief Queue on @p stream one event operation (classify to commit, seed, select) over the planes @p planes points
 *  to: an ``eagle::events::EventPlanes`` of @p bytes (4 or 8) wide reals, device addresses. */
void event_op(std::uint64_t stream, unsigned bytes, unsigned op, const void* planes);

/** @brief The free bytes of the current device. */
std::int64_t free_bytes();

/** @brief Wait until @p stream is idle. */
void stream_wait(std::uint64_t stream);

struct DeviceCapture;  ///< a stream capture held open across door calls

/** @brief Begin capturing @p stream (StreamCapturer, thread-local mode): work queued on it is recorded, not run. */
DeviceCapture* capture_begin(std::uint64_t stream);

/** @brief End @p capture and instantiate its graph (replayed by loop_launch, freed by loop_free); throws when the
 *  capture failed. @p capture is freed either way. */
DeviceLoop* capture_end(DeviceCapture* capture);

/** @brief End @p capture and drop what it recorded (NULL-safe). */
void capture_abort(DeviceCapture* capture) noexcept;

/** @brief Destroy @p loop (NULL-safe). */
void loop_free(DeviceLoop* loop) noexcept;

/** @brief The instantiated graph and its template as the driver's handles (CUgraphExec, CUgraph), for node updates. */
std::uint64_t loop_exec(const DeviceLoop* loop);
std::uint64_t loop_graph(const DeviceLoop* loop);

/** @brief Instantiate @p loop's template again, after its nodes were changed, and drop the old instance. */
void loop_reinstantiate(DeviceLoop* loop);

/** @brief The last @p count kernel nodes the capture on @p stream recorded, oldest first: they form one chain, each after the
 *  one before (a plan's partitions launched in turn). */
std::vector<std::uint64_t> capture_kernels(std::uint64_t stream, unsigned count);

struct DevicePool;   ///< the streams and events a recording's conditionals and forks use, made before the capture opens (none can be made inside it)

/** @brief Make @p streams streams and @p events events; free them with pool_free once the capture has ended. */
DevicePool* pool_new(unsigned streams, unsigned events);
void pool_free(DevicePool* pool) noexcept;

/** @brief Whether @p pool still has @p streams streams and @p events events to hand out. */
bool pool_has(const DevicePool* pool, unsigned streams, unsigned events);

/** @brief Queue on @p stream (inside a capture) the save of one conditional test: entry ``min(*iter, capacity - 1)`` of @p dst (three words: 1 for
 *  "evaluated", the cell, the baseline or 0 for none) takes ``*cell`` and ``*baseline``. A null @p iter is entry 0. */
void save_test_step(std::uint64_t stream, const unsigned* cell, const unsigned* baseline, const unsigned* iter, unsigned* dst, unsigned capacity);

/** @brief Queue on @p stream (inside a capture) the copy of @p bytes at @p src to ``table[min(*iter, capacity - 1)]``, @p table being a device array of
 *  device addresses. */
void table_copy_step(std::uint64_t stream, const void* src, std::size_t bytes, void* const* table, const unsigned* iter, unsigned capacity);

/** @brief One device-to-host read of read_many. */
struct ReadItem {
    void* host;
    const void* device;
    std::size_t bytes;
};

/** @brief Queue every read of @p items on @p stream and wait once. */
void read_many(std::uint64_t stream, const std::vector<ReadItem>& items);

struct DeviceWeave;  ///< a conditional woven into the capture in progress: its arm streams and its nodes
struct DeviceFork;   ///< parallel branches of the capture in progress: one stream each, joined back at the end

/** @brief Whether the driver has IF/ELSE and SWITCH conditional nodes (12.8). */
bool cond_arms_supported();

/**
 * @brief Weave a conditional into the capture on @p origin. @p kind: 0 IF, 1 IF_ELSE, 2 WHILE, 3 SWITCH. The test is
 * ``*cell != (baseline ? *baseline : 0)`` over 32-bit device words (``*cell`` is the arm index for SWITCH, which takes no
 * baseline), inverted when @p negate. A WHILE stops after @p cap iterations (0: no cap) and keeps
 * its state in the two device words @p loop_cells. Every arm is being captured on return; @p arms is the arm count.
 */
DeviceWeave* cond_begin(DevicePool* pool, std::uint64_t origin, int kind, const unsigned* cell, const unsigned* baseline, bool negate,
                        unsigned arms, unsigned cap, unsigned* loop_cells);

/** @brief The stream that captures arm @p arm of @p weave. */
std::uint64_t cond_stream(const DeviceWeave* weave, unsigned arm);

/** @brief Close every arm of @p weave and free it; throws when an arm captured no work (the weave is freed either way). */
void cond_end(DeviceWeave* weave);

/** @brief Free @p weave, closing what is still open (NULL-safe, never throws). */
void cond_drop(DeviceWeave* weave) noexcept;

/** @brief Fork the capture on @p origin: branch 0 starts after everything recorded so far. A fork takes an event, and each branch a
 *  stream and an event; a wait one event; they come back to @p pool when the fork or conditional ends (the caller checks pool_has). */
DeviceFork* fork_begin(DevicePool* pool, std::uint64_t origin);

/** @brief Add a branch to @p fork, also starting after everything recorded before the fork; its stream. */
std::uint64_t fork_branch(DeviceFork* fork);

/** @brief The stream that captures branch @p branch of @p fork. */
std::uint64_t fork_stream(const DeviceFork* fork, unsigned branch);

/** @brief Make branch @p waiter run after everything branch @p producer has captured so far. */
void fork_wait(DeviceFork* fork, unsigned waiter, unsigned producer);

/** @brief Join every branch back into the origin stream and free @p fork. */
void fork_end(DeviceFork* fork);

/** @brief Free @p fork (NULL-safe, never throws). */
void fork_drop(DeviceFork* fork) noexcept;

}  // namespace door
}  // namespace eagle
