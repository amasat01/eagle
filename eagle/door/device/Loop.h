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

/** @brief A zeroed device array of @p count uint32 cells (its address). */
std::uint64_t cells_new(unsigned count);

/** @brief Free cells made by cells_new (0-safe). */
void cells_free(std::uint64_t cells) noexcept;

/** @brief Write @p count host values into device @p cells, stream-ordered on @p stream. */
void cells_write(std::uint64_t stream, std::uint64_t cells, const unsigned* values, unsigned count);

/** @brief Read @p count device @p cells into @p values once @p stream has drained to them. */
void cells_read(std::uint64_t stream, std::uint64_t cells, unsigned* values, unsigned count);

/** @brief Queue on @p stream: ``*counter`` = the samples of the @p n-sample mask whose @p width-byte element
 *  is nonzero (the old path's mask count, re-syncing the finished counter to the mask). */
void mask_count(std::uint64_t stream, const void* mask, std::int64_t n, unsigned width, unsigned* counter);

/** @brief Elements of the cell's element type each of the two work buffers of a ``cell_fold`` over @p rows rows holds. */
std::int64_t cell_work_elems(std::int64_t rows);

/** @brief Queue on @p stream: ``*out`` = the max (@p max) or min of the @p rows float values at @p in
 *  (@p bits wide, 32 or 64), by eagle's canonical reduction tree; @p wa and @p wb are the work buffers. */
void cell_fold(std::uint64_t stream, const void* in, std::int64_t rows, unsigned bits, bool max, void* out, void* wa,
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

/** @brief Wait until @p stream is idle. */
void stream_wait(std::uint64_t stream);

/** @brief Destroy @p loop (NULL-safe). */
void loop_free(DeviceLoop* loop) noexcept;

}  // namespace door
}  // namespace eagle
