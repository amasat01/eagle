/* Copyright 2026 Alessandro Masat
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file eagle_xla.h
 * @brief The C face of ``libeagle_xla.so``: one generic XLA FFI custom call that
 *        runs any registered eagle kernel on the buffers XLA hands it.
 *
 * The library exports three symbols and nothing else:
 *
 * - ``eagle_xla_launch_cpu`` / ``eagle_xla_launch_cuda``: the FFI handlers (plain
 *   ``XLA_FFI_Handler`` functions over XLA's C API), registered by Python with
 *   ``jax.ffi.register_ffi_target`` for the ``cpu`` and ``CUDA`` platforms. Their
 *   only attribute is the string ``digest`` naming a registered recipe, so no
 *   address ever appears in a lowered module.
 * - ``eagle_xla_register(digest, recipe)``: put a recipe in the process-wide
 *   table (behind a reader-writer lock; a later registration of the same digest
 *   replaces it, a call already holding the old one finishes with it).
 *
 * A call looks its digest up, checks every buffer's element type (and every
 * per-sample buffer's sample count) against the recipe, copies each aliased
 * operand into its result buffer when XLA gave the call a different one (the
 * handler honours the alias itself, so a fused or re-buffered call still reads
 * the carried state), zero-fills the results the recipe flags, builds the by-value argument PODs from the buffer
 * dimensions (``plugin/gref_layout.h``: a ``(width, n)`` row-major buffer is a
 * ``GRefMirror`` with ``samples = compStride = n``, ``sampleStride = 1``; an
 * ``(n,)`` or one-cell buffer a ``ScalarHandle``), and launches ONE serial
 * call over ``[0, n)``:
 *
 * After the launch, each cell the recipe lists is finished into its one-element
 * result from its rows (and companion rows) with the canonical reduction
 * (``eagle/reduce/canonical.h``): inline on the CPU, the seam's
 * ``eagle_backend_x_reduce_canonical`` on the same stream on CUDA, in the work
 * results XLA allocated (the handler allocates nothing).
 *
 * - CPU: the kernel's host entry ``void entry(void* const*, int64 base, int64
 *   count, int64 n)``, inline on XLA's thread; alias copy and zero-fill are ``memcpy`` and
 *   ``memset``.
 * - CUDA: the seam's ``eagle_backend_run_device`` on XLA's stream (never a host
 *   synchronisation); the alias copy and the zero-fill are the seam's
 *   ``eagle_backend_memcpy_device`` and ``eagle_backend_memset_device`` on the
 *   same stream. The call's device ordinal must be the recipe's.
 *
 * Plain C++, no CUDA toolchain, no Python: every device entry point is an
 * address the recipe carries, resolved by Python from the loaded CUDA plugin.
 */
#ifndef EAGLE_XLA_H
#define EAGLE_XLA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** ``eagle_xla_recipe.platform``. */
#define EAGLE_XLA_PLATFORM_CPU 0
#define EAGLE_XLA_PLATFORM_CUDA 1

/** ``eagle_xla_recipe.slot_kind``: the by-value shape one kernel slot is
 *  passed as. INOUT is a ``ScalarHandle`` over a RESULT buffer that XLA
 *  aliases to an operand (the ``Terminated`` mask a kernel finishes): it is
 *  never zero-filled. */
#define EAGLE_XLA_SLOT_GREF 1
#define EAGLE_XLA_SLOT_SCALAR_HANDLE 2
#define EAGLE_XLA_SLOT_INT_HANDLE 3
#define EAGLE_XLA_SLOT_NSAMPLES 4
#define EAGLE_XLA_SLOT_INOUT 5

/** ``eagle_xla_recipe.cell_op``: how a cell folds its rows (sum and mean rows
 *  already carry the 1/n); ``policy``: the float-sum policy, the seam's
 *  EAGLE_BACKEND_REDUCE_COMPENSATED / ORDERED / AUTO values. */
/** OR'd into ``policy`` of a CPU recipe: float64 sum cells take the
 *  fixed-point leaf, as the machine's GPU does (``canonical::fixedLeaf``). */
#define EAGLE_XLA_POLICY_FIXED_F64 0x100
#define EAGLE_XLA_CELL_SUM 0
#define EAGLE_XLA_CELL_MAX 1
#define EAGLE_XLA_CELL_MIN 2

/** ``eagle_xla_register`` status codes. */
#define EAGLE_XLA_OK 0
#define EAGLE_XLA_BAD_RECIPE 1   /**< null pointer, wrong struct_size or platform */
#define EAGLE_XLA_BAD_SLOT 2     /**< a slot kind, buffer index or width out of range */

/** One kernel, as the handler runs it. Every array is read during
 *  ``eagle_xla_register`` and copied; the caller may free it afterwards. */
struct eagle_xla_recipe {
    uint32_t struct_size;        /**< sizeof(struct eagle_xla_recipe), exactly */
    uint32_t platform;           /**< EAGLE_XLA_PLATFORM_* */
    int32_t ordinal;             /**< CUDA: the device the function was loaded on */
    uint32_t block;              /**< CUDA: threads per block */
    uint64_t entry;              /**< CPU: the host entry; CUDA: eagle_backend_run_device */
    uint64_t last_error;         /**< CUDA: eagle_backend_last_error */
    uint64_t memset;             /**< CUDA: eagle_backend_memset_device */
    uint64_t memcpy;             /**< CUDA: eagle_backend_memcpy_device */
    uint64_t function;           /**< CUDA: the CUfunction */
    int32_t nargs;               /**< XLA operands */
    int32_t nrets;               /**< XLA results */
    const int32_t* dtype;        /**< nargs + nrets XLA_FFI_DataType codes, operands first */
    const uint8_t* zero;         /**< nrets flags: 1 = zero-fill this result before the launch */
    int32_t nslots;              /**< the kernel's arg_spec slots */
    int32_t sample_buffer;       /**< the buffer (operands, then results) whose last extent is n */
    const int32_t* slot_kind;    /**< nslots EAGLE_XLA_SLOT_* */
    const int32_t* slot_buffer;  /**< nslots buffer indices (-1 for NSAMPLES) */
    const uint8_t* slot_cell;    /**< nslots flags: 1 = a lookup cell (its own extent), 0 = per sample */
    int32_t index_bytes;         /**< NSAMPLES width: 4 or 8 */
    int32_t nalias;              /**< operand/result pairs XLA aliases */
    const int32_t* alias_operand; /**< nalias operand indices */
    const int32_t* alias_result;  /**< nalias result indices (into the results, not the buffers) */
    uint64_t reduce;             /**< CUDA: eagle_backend_x_reduce_canonical (the host finishes inline) */
    int32_t ncells;              /**< sample cells the call finishes after the launch */
    int32_t cell_row;            /**< samples each row of a cell stands for (rows = ceil(n / cell_row)) */
    int32_t policy;              /**< EAGLE_XLA_POLICY_* of the float sums */
    const int32_t* cell_rows;    /**< ncells buffer indices of the row planes */
    const int32_t* cell_comp;    /**< ncells buffer indices of the companion rows (-1: none) */
    const int32_t* cell_out;     /**< ncells buffer indices of the (1,) cell results */
    const int32_t* cell_op;      /**< ncells EAGLE_XLA_CELL_* */
    int32_t work_a;              /**< CUDA: buffer index of the first reduction work result (-1: no cells) */
    int32_t work_b;              /**< CUDA: the second */
};

int32_t eagle_xla_register(const char* digest, const struct eagle_xla_recipe* recipe);

#ifdef __cplusplus
}
#endif

#endif /* EAGLE_XLA_H */
