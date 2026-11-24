// SPDX-License-Identifier: Apache-2.0

// The device group's matrix products: those a gemm record (eg_unit_meta.h) states and those of eg_gemm_run, run through cuBLAS on a
// stream. cuBLAS is opened at run time; nothing here links it. Every function is called with the context's device
// current (device/Device.cpp), and only by that file.
#pragma once

#include "eagle/door/Context.h"

namespace eagle {
namespace door {

/** @brief Make the cuBLAS handle of @p stream if the context has none: opens cuBLAS the first time (EG_FAULT naming the libraries tried). */
void gemm_handle_make(eg_ctx& ctx, std::uint64_t stream);
/** @brief Destroy the cuBLAS handle of @p stream, if the context made one (the stream is about to be destroyed). */
void gemm_handle_drop(eg_ctx& ctx, std::uint64_t stream) noexcept;
/** @brief Destroy every cuBLAS handle of @p ctx. */
void gemm_handles_free(eg_ctx& ctx) noexcept;
/** @brief Queue one row-major product on @p stream, with the handle made for that stream (never makes one): C (+)= op(A) op(B) as
 *  @p c states it, over device planes of the row strides @p ld_a, @p ld_b, @p ld_c and the width @p f32. The one routine of a gemm
 *  unit's products and of eg_gemm_run. */
void gemm_product(const eg_ctx& ctx, std::uint64_t stream, const GemmCall& c, bool f32, const void* a, int ld_a, const void* b, int ld_b, void* out, int ld_c);
/** @brief Queue the products of @p plan's gemm record on @p stream (gemm_product over each call). */
void gemm_launch(const eg_plan_s& plan, std::uint64_t stream);

}  // namespace door
}  // namespace eagle
