// SPDX-License-Identifier: Apache-2.0

// The device group's gemm unit: the matrix products a gemm record (eg_unit_meta.h) states, run through cuBLAS on a
// stream. cuBLAS is opened at run time; nothing here links it. Every function is called with the context's device
// current (device/Device.cpp), and only by that file.
#pragma once

#include "eagle/door/Context.h"

namespace eagle {
namespace door {

/** @brief Make the cuBLAS handle of @p stream if the context has none: opens cuBLAS the first time (EG_FAULT naming the libraries tried). */
void gemm_handle_make(eg_ctx& ctx, std::uint64_t stream);
/** @brief Destroy every cuBLAS handle of @p ctx. */
void gemm_handles_free(eg_ctx& ctx) noexcept;
/** @brief Queue the products of @p plan's gemm record on @p stream, with the handle made for that stream (never makes one). */
void gemm_launch(const eg_plan_s& plan, std::uint64_t stream);

}  // namespace door
}  // namespace eagle
