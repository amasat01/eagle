// SPDX-License-Identifier: Apache-2.0

// The device transform engine (device/Fft.cpp): the cuFFT plans of one
// transform plan and their queued run. cuFFT is opened by dlopen at the first
// plan, never linked. Device/Device.cpp makes the context current around each
// call and owns the plan's input copy; no CUDA header reaches this file.
#pragma once

#include "eagle/door/Context.h"

namespace eagle {
namespace door {

/** @brief Bytes of the copy of its input that a C2R of @p plan keeps (cuFFT's
 * c2r destroys its input): 0 for the others. */
std::size_t fft_work_bytes(const eg_fft_s& plan);

/** @brief The cuFFT plans of @p plan. Refuses FAULT naming the libraries tried
 * when cuFFT cannot be opened, and NOT_WRITTEN a transform of more than 2^31 -
 * 1 elements (cuFFT counts in 32 bits). */
DeviceFft* fft_build(const eg_fft_s& plan);

/** @brief Queue one run of @p plan from device memory @p in to @p out on @p
 * stream: no allocation, no wait, legal in a capture.
 *  @p work is the C2R's input copy, ``fft_work_bytes`` long. */
void fft_enqueue(const DeviceFft& fft, const eg_fft_s& plan, const void* in,
    void* out, void* work, std::uint64_t stream);

/** @brief Destroy the cuFFT plans (NULL-safe). */
void fft_drop(DeviceFft* fft) noexcept;

} // namespace door
} // namespace eagle
