// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// View mirror <-> DLPack bridge at the plugin deployment boundary.
//
// The CuPy test harness deliberately stays thin (contiguous numpy struct
// mirror, no DLPack). The *production* surface — the precompiled host that
// one-to-one matches a host's acted-on state & output arrays — is where DLPack
// interop belongs, because aether's Device taxonomy IS DLPack's (aether::Device
// is a thin DLDevice wrapper, aether/device/Device.h). This header converts
// between a vector View mirror (see gref_abi.h) and a DLPack dense device
// tensor describing a contiguous (W, N) SoA buffer.
//
// The struct eagle vendors here is the PRE-1.0 `DLManagedTensor` (plugin/
// dlpack.h) and it STAYS: aether ships `aether::interop::toDLPackLegacy` /
// `fromDLPack` against exactly that generation (aether/interop/DLPackLegacy.h),
// on the documented ground that the inner structs are ABI-identical between
// aether's own vendored legacy subset and eagle's.
//
// Scope: contiguous float64 (W, N) tensors on a CPU or CUDA device, from
// either DLPack generation, with the access flag carried both ways. The
// general layer (any dtype/shape, Requirements, keep-alive, stream contract)
// is plugin/interop.h; this header is its dependency-free subset for the
// precompiled host, which builds as C++17 without aether.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "dlpack.h"
#include "gref_abi.h"

namespace eagle {
namespace plugin {

// aether's Device wraps DLDevice verbatim, so a mirror's `deviceType_` IS a
// DLDeviceType value; the bridge relies on that identity so no translation
// table is needed. gref_abi.h spells the same code as kEagleAbiDeviceCUDA
// (it must stay dependency-free, so it cannot name the enum itself).
static_assert(static_cast<int>(kDLCUDA) == 2,
    "kDLCUDA must equal aether Device(kDLCUDA) == 2");
static_assert(kEagleAbiDeviceCUDA == static_cast<int>(kDLCUDA),
    "gref_abi.h's device code must be DLPack's");

// The access a DLPack tensor declares: versioned tensors carry the read-only
// flag, the legacy struct carries none. Same spelling as the generic layer
// (aether/interop/Buffer.h, eagle/plugin/interop.h), which this header cannot
// include: it stays C++17 and free of aether for the precompiled host.
enum class BridgeAccess { ReadWrite, ReadOnly, Unknown };

namespace bridge_detail {

[[noreturn]] inline void refuse(const std::string& check, const std::string& want, const std::string& got)
{
    throw std::runtime_error("DLPack: " + check + ": required " + want + ", got " + got);
}

inline std::string dtype_text(DLDataType d)
{
    if (d.lanes == 1 && d.code == kDLFloat)
        return "float" + std::to_string(d.bits);
    if (d.lanes == 1 && d.code == kDLInt)
        return "int" + std::to_string(d.bits);
    if (d.lanes == 1 && d.code == kDLUInt)
        return "uint" + std::to_string(d.bits);
    return "dtype(code=" + std::to_string(d.code) + ", bits=" + std::to_string(d.bits)
        + ", lanes=" + std::to_string(d.lanes) + ")";
}

inline std::string device_text(DLDeviceType t)
{
    if (t == kDLCPU)
        return "cpu";
    if (t == kDLCUDA)
        return "cuda";
    return "device_type " + std::to_string(static_cast<int>(t));
}

// The checks of the generic layer's Requirements that a (W, N) mirror needs:
// rank 2, float64, C-contiguous, 8-byte aligned, CPU or CUDA. Each refusal
// names its check and both values.
inline GRefMirror mirror_of(const DLTensor& t)
{
    if (t.device.device_type != kDLCUDA && t.device.device_type != kDLCPU)
        refuse("device", "cpu or cuda", device_text(t.device.device_type));
    if (t.ndim != 2)
        refuse("shape", "a rank-2 (W, N) tensor", "rank " + std::to_string(t.ndim));
    if (t.dtype.code != kDLFloat || t.dtype.bits != 64 || t.dtype.lanes != 1)
        refuse("dtype", "float64", dtype_text(t.dtype));
    const int64_t W = t.shape[0];
    const int64_t N = t.shape[1];
    const bool contiguous = t.strides == nullptr
        || ((N <= 1 || t.strides[1] == 1) && (W <= 1 || t.strides[0] == N));
    if (!contiguous)
        refuse("stride", "C-contiguous with unit stride",
            "strides (" + std::to_string(t.strides[0]) + ", " + std::to_string(t.strides[1]) + ")");
    char* base = static_cast<char*>(t.data) + t.byte_offset;
    if (reinterpret_cast<std::uintptr_t>(base) % 8 != 0)
        refuse("alignment", "8-byte aligned", "address " + std::to_string(reinterpret_cast<std::uintptr_t>(base)));
    GRefMirror g;
    g.data_         = reinterpret_cast<double*>(base);
    g.samples_      = static_cast<std::uint64_t>(N);
    g.compStride_   = static_cast<std::uint64_t>(N);
    g.sampleStride_ = 1;
    g.deviceType_   = t.device.device_type == kDLCPU ? kEagleAbiDeviceCPU : kEagleAbiDeviceCUDA;
    g.deviceId_     = t.device.device_id;
    return g;
}

} // namespace bridge_detail

// The access a versioned tensor declares (its read-only flag).
inline BridgeAccess access_of(const DLManagedTensorVersioned* mt)
{
    return (mt->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0 ? BridgeAccess::ReadOnly : BridgeAccess::ReadWrite;
}

// The legacy struct carries no access flag.
inline BridgeAccess access_of(const DLManagedTensor*) { return BridgeAccess::Unknown; }

// Build a GRefMirror from a DLPack tensor describing a contiguous (W, N)
// float64 array (SoA) on a CPU or CUDA device. nVecs_ == dimOffset_ == N (the
// sample count, shape[1]); the vector width W (shape[0]) only affects
// in-kernel indexing, not the view. `byte_offset` is folded into the pointer.
// The caller decides what access it needs (see access_of).
inline GRefMirror gref_from_dlpack(const DLManagedTensorVersioned* mt)
{
    if (mt->version.major != DLPACK_MAJOR_VERSION)
        throw std::runtime_error("DLPack: major version mismatch — the only "
                                 "legal action is to call the deleter");
    return bridge_detail::mirror_of(mt->dl_tensor);
}

// The legacy-struct overload of gref_from_dlpack (same checks).
inline GRefMirror gref_from_dlpack(const DLManagedTensor* mt)
{
    return bridge_detail::mirror_of(mt->dl_tensor);
}

// Fill a DLPack tensor view of a (W, N) mirror. ``shape`` is caller-owned
// storage (int64_t[2]) that must outlive any consumer of the tensor; strides
// are left NULL (row-major contiguous). No ownership transfer — deleter is
// null, and the view aliases the GRef's own memory rather than copying it.
// The read-only flag is set iff ``read_only`` (a `lookup` role's buffer); the
// device follows the mirror's own device code.
inline DLManagedTensorVersioned dlpack_from_gref(
    const GRefMirror& g, int64_t width, int64_t shape[2], bool read_only = false)
{
    if (g.compStride_ != g.samples_ || g.sampleStride_ != 1)
        throw std::runtime_error(
            "DLPack: only a C-contiguous (W, N) mirror can be exported "
            "without strides");
    shape[0] = width;
    shape[1] = static_cast<int64_t>(g.samples_);
    DLManagedTensorVersioned mt{};
    mt.version.major = DLPACK_MAJOR_VERSION;
    mt.version.minor = DLPACK_MINOR_VERSION;
    mt.flags = read_only ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
    mt.dl_tensor.data        = const_cast<double*>(g.data_);
    mt.dl_tensor.device      = DLDevice{
        g.deviceType_ == kEagleAbiDeviceCPU ? kDLCPU : kDLCUDA, g.deviceId_ };
    mt.dl_tensor.ndim        = 2;
    mt.dl_tensor.dtype       = DLDataType{ kDLFloat, 64, 1 };
    mt.dl_tensor.shape       = shape;
    mt.dl_tensor.strides     = nullptr;
    mt.dl_tensor.byte_offset = 0;
    mt.manager_ctx           = nullptr;
    mt.deleter               = nullptr;
    return mt;
}

} // namespace plugin
} // namespace eagle
