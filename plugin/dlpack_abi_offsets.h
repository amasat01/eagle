// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Layout guard for the vendored DLPack ABI header next to this file.
//
// WHY THIS EXISTS. The plugin boundary passes DLPack structs by address across
// a compiled wall, so their layout IS the contract -- but the export-set gate
// compares SYMBOL NAMES and cannot see a struct whose fields moved underneath a
// name that never changed. A signature or layout change therefore passes that
// gate silently. These assertions are the gate for it.
//
// They also pin the boundary against a bad re-vendoring: every copy of the
// upstream header in this ecosystem shares upstream's own DLPACK_DLPACK_H_
// include guard, so when two copies meet in one translation unit the first one
// included wins and the second is skipped entirely. If the copies ever
// disagree, the winner is decided by include order rather than by intent. A
// compile-time failure here is that disagreement becoming visible.
//
// Included from a translation unit that is always built, so it cannot be
// skipped. Numbers are the x86-64 LP64 layout; a port to an ABI with different
// pointer size or alignment is expected to fail here and be re-derived
// deliberately rather than relaxed.

#pragma once

#include "dlpack.h"

#include <cstddef>
#include <cstdint>

// ---- enum values the wall depends on -------------------------------------
// Device codes: the bridge maps kDLCUDA onto aether's Device::CUDA_DEVICE by
// value identity, with no translation table, so this is load-bearing.
static_assert(static_cast<int>(kDLCPU) == 1, "DLPack kDLCPU must be 1");
static_assert(static_cast<int>(kDLCUDA) == 2, "DLPack kDLCUDA must be 2");
static_assert(static_cast<int>(kDLCUDAHost) == 3, "DLPack kDLCUDAHost must be 3");
static_assert(static_cast<int>(kDLCUDAManaged) == 13, "DLPack kDLCUDAManaged must be 13");
// dtype codes: the column producer emits these three; kDLBool is asserted
// because the versioned export policy uses it where the legacy one used
// kDLUInt, and a silent swap would mis-type every boolean column.
static_assert(static_cast<int>(kDLInt) == 0, "DLPack kDLInt must be 0");
static_assert(static_cast<int>(kDLUInt) == 1, "DLPack kDLUInt must be 1");
static_assert(static_cast<int>(kDLFloat) == 2, "DLPack kDLFloat must be 2");
static_assert(static_cast<int>(kDLBool) == 6, "DLPack kDLBool must be 6");

// DLDeviceType is explicitly int32_t under C++ upstream; the pre-1.0 subset
// this header replaced left the underlying type implementation-defined.
static_assert(sizeof(DLDeviceType) == 4, "DLDeviceType must be 4 bytes");

// ---- DLPackVersion --------------------------------------------------------
static_assert(sizeof(DLPackVersion) == 8, "DLPackVersion size");
static_assert(offsetof(DLPackVersion, major) == 0, "DLPackVersion::major");
static_assert(offsetof(DLPackVersion, minor) == 4, "DLPackVersion::minor");

// ---- DLDevice -------------------------------------------------------------
static_assert(sizeof(DLDevice) == 8, "DLDevice size");
static_assert(offsetof(DLDevice, device_type) == 0, "DLDevice::device_type");
static_assert(offsetof(DLDevice, device_id) == 4, "DLDevice::device_id");

// ---- DLDataType -----------------------------------------------------------
static_assert(sizeof(DLDataType) == 4, "DLDataType size");
static_assert(offsetof(DLDataType, code) == 0, "DLDataType::code");
static_assert(offsetof(DLDataType, bits) == 1, "DLDataType::bits");
static_assert(offsetof(DLDataType, lanes) == 2, "DLDataType::lanes");

// ---- DLTensor -------------------------------------------------------------
static_assert(sizeof(DLTensor) == 48, "DLTensor size");
static_assert(offsetof(DLTensor, data) == 0, "DLTensor::data");
static_assert(offsetof(DLTensor, device) == 8, "DLTensor::device");
static_assert(offsetof(DLTensor, ndim) == 16, "DLTensor::ndim");
static_assert(offsetof(DLTensor, dtype) == 20, "DLTensor::dtype");
static_assert(offsetof(DLTensor, shape) == 24, "DLTensor::shape");
static_assert(offsetof(DLTensor, strides) == 32, "DLTensor::strides");
static_assert(offsetof(DLTensor, byte_offset) == 40, "DLTensor::byte_offset");

// ---- DLManagedTensor (legacy; still upstream, still exchanged) ------------
static_assert(sizeof(DLManagedTensor) == 64, "DLManagedTensor size");
static_assert(offsetof(DLManagedTensor, dl_tensor) == 0, "DLManagedTensor::dl_tensor");
static_assert(offsetof(DLManagedTensor, manager_ctx) == 48, "DLManagedTensor::manager_ctx");
static_assert(offsetof(DLManagedTensor, deleter) == 56, "DLManagedTensor::deleter");

// ---- DLManagedTensorVersioned (the exchanged struct) ----------------------
// Note the field order: version/manager_ctx/deleter/flags come BEFORE the
// tensor, the reverse of the legacy struct. A consumer that casts one to the
// other reads the version words as a data pointer.
static_assert(sizeof(DLManagedTensorVersioned) == 80, "DLManagedTensorVersioned size");
static_assert(offsetof(DLManagedTensorVersioned, version) == 0, "DLManagedTensorVersioned::version");
static_assert(offsetof(DLManagedTensorVersioned, manager_ctx) == 8, "DLManagedTensorVersioned::manager_ctx");
static_assert(offsetof(DLManagedTensorVersioned, deleter) == 16, "DLManagedTensorVersioned::deleter");
static_assert(offsetof(DLManagedTensorVersioned, flags) == 24, "DLManagedTensorVersioned::flags");
static_assert(offsetof(DLManagedTensorVersioned, dl_tensor) == 32, "DLManagedTensorVersioned::dl_tensor");

// ---- version macros -------------------------------------------------------
static_assert(DLPACK_MAJOR_VERSION == 1, "vendored DLPack major version");
static_assert(DLPACK_MINOR_VERSION == 0, "vendored DLPack minor version");
