// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Device-clean half of the AETHER array-view ABI mirrors.
//
// This header holds the layout only; `gref_abi.h` adds the host-only
// validation functions (`std::string`, `std::runtime_error`). Under the NVRTC
// profile a device translation unit reaches every payload header from memory
// and NVRTC's EDG front end
// has no host standard library at all, so a device TU may never REACH
// `<stdexcept>`/`<string>` even behind an unused function — a HAWK-emitted
// kernel and eagle's own device-side plugin fixtures now include ONLY this
// file. `gref_abi.h` still exists, `#include`s this file, and adds the
// host-only validation surface (`abi_version_of`, `check_aether_abi`,
// `check_layout_sizes`) — nothing here is renamed, so every existing host
// caller of those functions is unaffected.
//
// Deliberately includes **no aether headers** — the whole point of the
// precompiled host is that accepting a kernelized plugin needs only the
// binary ABI (these PODs), the PTX, and its metadata sidecar; not the aether
// source, nvcc, or CuPy.
//
// The layouts are pinned by static_asserts here and cross-checked against the
// real aether types by tests/test_gref_layout.py and tests/test_handle_layout.py.
//
// PORT NOTE (one-way). These PODs used to mirror
// a two-tier view family -- a vector view (32 B) and
// a scalar view (a bare 8-byte device pointer).
// aether collapsed that family into ONE `aether::View<T, Extents, Layout>`
// (`aether/view/View.h`): the View IS the descriptor, and it carries its own
// mapping (extents + per-mode strides) and its own `Device` tag. There is no
// HandleT any more, so a per-sample scalar plane is a rank-1 View — 32 B, not
// 8. The sizes below are MEASURED off the installed aether, never carried
// over: 32 -> 40 for the vector mirror, 8 -> 32 for the scalar one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eagle {
namespace plugin {

// The index width the WIRE silently depends on. `nsamples` is
// packed as `aether::idx_t` and read back as `aether::idx_t` in the
// kernel prologue, so a build of aether with a 64-bit index shifts the whole
// parameter block — and a GRefMirror/ScalarHandle/IntHandle size triple would NOT
// catch it. It therefore enters the layout self-check as its own field.
//
// Mirrored, never included: this header is deliberately aether-free (see the file
// header), so the width is taken from aether's OWN switch macro when the TU has
// already seen it (or a -D on the command line) and otherwise defaults to aether's
// documented default (`aether/typedefs.h`: `#ifndef AETHER_INDEX_T -> uint32_t`).
// Because both spellings read the same macro, a `-DAETHER_INDEX_T=...` build keeps
// the two in step; a mismatch is exactly what the self-check exists to catch.
#ifndef EAGLE_ABI_INDEX_T
#  ifdef AETHER_INDEX_T
#    define EAGLE_ABI_INDEX_T AETHER_INDEX_T
#  else
#    define EAGLE_ABI_INDEX_T std::uint32_t
#  endif
#endif

// The DLPack device-type codes a mirror's `Device` tag carries. Spelled as
// plain integers so this header keeps its "no dependencies" property (the
// DLPack enum itself lives in plugin/dlpack.h, whose own static_assert already
// pins kDLCUDA == 2); aether's `Device` is a thin `DLDevice` wrapper
// (aether/device/Device.h), so these ARE the values it stores.
inline constexpr std::int32_t kEagleAbiDeviceCPU  = 1;  // kDLCPU
inline constexpr std::int32_t kEagleAbiDeviceCUDA = 2;  // kDLCUDA

// Mirror of aether::Array<Real, N>::ViewT — i.e. aether::View<double,
// aether::extents<N, aether::dyn>, aether::layout_stride> (a 40-byte POD).
//
// The struct is width-independent: for a
// contiguous (N_vec, N) buffer the component pitch is N regardless of the
// vector width N_vec (width only changes in-kernel indexing). What replaced
// An earlier `nVecs_`/`dimOffset_` pair is now aether's `layout_stride` mapping: ONE
// dynamic extent (the trailing/batch mode — the sample count) plus one stride
// per mode. The unused texture fields are simply gone: aether's View has no
// texture path at all, so there is nothing to leave zero.
struct GRefMirror {
    double*       data_        = nullptr;              // 0  device pointer to the SoA buffer
    std::uint64_t samples_     = 0;                    // 8  extents<N,dyn>'s ONE dynamic extent
    std::uint64_t compStride_  = 0;                    // 16 mapping().stride(0): component pitch
    std::uint64_t sampleStride_= 1;                    // 24 mapping().stride(1): sample pitch
    std::int32_t  deviceType_  = kEagleAbiDeviceCUDA;  // 32 Device::raw.device_type
    std::int32_t  deviceId_    = 0;                    // 36 Device::raw.device_id
};
static_assert(sizeof(GRefMirror) == 40, "GRefMirror must match aether View<double,extents<N,dyn>,layout_stride> (40 B)");
static_assert(alignof(GRefMirror) == 8, "GRefMirror alignment");
// offsetof is NOT a constant expression under NVRTC (no __builtin_offsetof):
// these layout pins are host-only. sizeof/alignof/is_standard_layout
// above and below stay unconditional — they ARE constant expressions under
// NVRTC and are exactly what a device TU needs to trust the ABI.
#ifndef __CUDACC_RTC__
static_assert(offsetof(GRefMirror, data_) == 0, "data_ offset");
static_assert(offsetof(GRefMirror, samples_) == 8, "samples_ offset");
static_assert(offsetof(GRefMirror, compStride_) == 16, "compStride_ offset");
static_assert(offsetof(GRefMirror, sampleStride_) == 24, "sampleStride_ offset");
static_assert(offsetof(GRefMirror, deviceType_) == 32, "deviceType_ offset");
static_assert(offsetof(GRefMirror, deviceId_) == 36, "deviceId_ offset");
#endif
static_assert(std::is_standard_layout<GRefMirror>::value, "GRefMirror layout");
static_assert(std::is_trivially_copyable<GRefMirror>::value, "GRefMirror POD");

// Mirror of aether::Array<T>::ViewT — aether::View<T, aether::extents<
// aether::dyn>, aether::layout_stride> (a 32-byte POD): the rank-1 sibling of
// GRefMirror, one mode instead of two. Used for the per-sample spacecraft
// scalars (mass/area/cr/cd) and the bool termination mask.
//
// This is the mirror that CHANGED SHAPE, not just size: the earlier `HandleT` was a
// bare device pointer with no extent and no device tag, so the kernel had to
// be told the sample count some other way. aether's View carries both.
struct ScalarHandle {
    void*         data       = nullptr;              // 0
    std::uint64_t samples    = 0;                    // 8
    std::uint64_t stride     = 1;                    // 16
    std::int32_t  deviceType = kEagleAbiDeviceCUDA;  // 24
    std::int32_t  deviceId   = 0;                    // 28
};
static_assert(sizeof(ScalarHandle) == 32, "ScalarHandle must match aether View<T,extents<dyn>,layout_stride> (32 B)");
static_assert(alignof(ScalarHandle) == 8, "ScalarHandle alignment");
#ifndef __CUDACC_RTC__
static_assert(offsetof(ScalarHandle, data) == 0, "ScalarHandle data offset");
static_assert(offsetof(ScalarHandle, samples) == 8, "ScalarHandle samples offset");
static_assert(offsetof(ScalarHandle, stride) == 16, "ScalarHandle stride offset");
static_assert(offsetof(ScalarHandle, deviceType) == 24, "ScalarHandle deviceType offset");
static_assert(offsetof(ScalarHandle, deviceId) == 28, "ScalarHandle deviceId offset");
#endif
static_assert(std::is_standard_layout<ScalarHandle>::value, "ScalarHandle layout");
static_assert(std::is_trivially_copyable<ScalarHandle>::value, "ScalarHandle POD");

// Mirror of aether::Array<Int>::ViewT: structurally identical to ScalarHandle
// but kept as a distinct type for an int-valued slot — a pure kernel's
// `Mutable[int]` state. Naming it apart from the Real/bool ScalarHandle makes
// the int element type explicit at the call site.
struct IntHandle {
    void*         data       = nullptr;
    std::uint64_t samples    = 0;
    std::uint64_t stride     = 1;
    std::int32_t  deviceType = kEagleAbiDeviceCUDA;
    std::int32_t  deviceId   = 0;
};
static_assert(sizeof(IntHandle) == 32, "IntHandle must match aether View<Int,extents<dyn>,layout_stride> (32 B)");
#ifndef __CUDACC_RTC__
static_assert(offsetof(IntHandle, data) == 0, "IntHandle data offset");
static_assert(offsetof(IntHandle, samples) == 8, "IntHandle samples offset");
static_assert(offsetof(IntHandle, stride) == 16, "IntHandle stride offset");
static_assert(offsetof(IntHandle, deviceType) == 24, "IntHandle deviceType offset");
static_assert(offsetof(IntHandle, deviceId) == 28, "IntHandle deviceId offset");
#endif
static_assert(std::is_standard_layout<IntHandle>::value, "IntHandle layout");
static_assert(std::is_trivially_copyable<IntHandle>::value, "IntHandle POD");

// The `dev` argument is EXPLICIT at every call site rather than defaulted: the
// device registry binds CUDA buffers and the host registry binds plain host
// ones, and aether refuses a transfer whose Device tag disagrees with the
// memory — a silently-defaulted tag would be a lie the kernel cannot detect.
inline GRefMirror make_gref(double* data, std::uint64_t n, std::int32_t dev) {
    GRefMirror g;
    g.data_ = data;
    g.samples_ = n;
    // A contiguous (W, N) SoA buffer: component c starts at data + c * N and
    // successive samples are adjacent — exactly what packedSpanView<W> builds.
    g.compStride_ = n;
    g.sampleStride_ = 1;
    g.deviceType_ = dev;
    return g;
}

inline ScalarHandle make_handle(void* ptr, std::uint64_t n, std::int32_t dev) {
    ScalarHandle h;
    h.data = ptr;
    h.samples = n;
    h.stride = 1;
    h.deviceType = dev;
    return h;
}

inline IntHandle make_int_handle(void* ptr, std::uint64_t n, std::int32_t dev) {
    IntHandle h;
    h.data = ptr;
    h.samples = n;
    h.stride = 1;
    h.deviceType = dev;
    return h;
}

// ---------------------------------------------------------------------------
// THE LAYOUT SELF-CHECK's WIRE SHAPE;
// the PartitionTriple struct and the field-count
// constants are device-clean (no std::string); the comparison FUNCTIONS that
// read and report them (`expected_layout_sizes`, `layout_field_name`,
// `check_layout_sizes`) stay in `gref_abi.h`, the host-only half, because a
// mismatch message is built with `std::string`/`std::to_string`.
// ---------------------------------------------------------------------------
// A tag says "this artifact was built against the same CONTRACT". It cannot say
// "…and against the same LAYOUT": every POD above is a mirror of an aether type
// whose size is a build-time fact, and `aether::idx_t`'s width is a compile-time
// SWITCH (`AETHER_INDEX_T`). A plugin built against a differently-configured
// aether carries a perfectly correct `aether-abi/2` tag and then decodes the
// parameter block at the wrong offsets — silently, arithmetically wrong.
//
// So every aether-abi/2 artifact MUST export an array of five unsigned 64-bit
// sizes under the fixed symbol `eagle_layout_sizes` (below), and the loader
// reads it BEFORE binding anything:
//
//   [0] sizeof(GRefMirror)        the (W, N) vector-view mirror        (40)
//   [1] sizeof(ScalarHandle)      the rank-1 per-sample view mirror    (32)
//   [2] sizeof(IntHandle)         its int-valued twin                  (32)
//   [3] sizeof(aether::idx_t)     the WIRE INDEX WIDTH     (4)
//   [4] sizeof(partition triple)  three int64s, L2's `{base,count,n}`  (24)
//
// HOW IT IS READ differs per target and is NOT incidental:
// a host plugin is a `.so`, so the loader `dlsym`s the symbol and reads the
// array directly; a device plugin is a PTX module with no dlsym at all, so the
// loader resolves it with `cuModuleGetGlobal` and stages the 40 bytes to the
// host. That staging is a DRIVER-MODULE global read, not a transfer on aether
// data, so it is outside the "no raw memcpy on aether data" rule —
// nothing in the plugin's own buffers is touched, and
// the read happens before any binding exists to touch.

// The fixed export symbol every aether-abi/2 plugin carries.
inline constexpr const char* kEagleLayoutSymbol = "eagle_layout_sizes";

// How many fields the array carries.
inline constexpr std::size_t kEagleLayoutFieldCount = 5;

// The int64 partition triple `{base, count, nSamples}` (L2) as it rides the wire —
// declared here so its size is a MEASURED fact of this build rather than a
// hand-written 24 in two places. Device fixtures build their `eagle_layout_sizes`
// export from `sizeof(PartitionTriple)` directly (see
// tests/fixtures/device_plugin_execv2.cu), so this struct must be device-clean.
struct PartitionTriple {
    std::int64_t base     = 0;
    std::int64_t count    = 0;
    std::int64_t nSamples = 0;
};
static_assert(sizeof(PartitionTriple) == 24, "the L2 partition triple is three int64s");

}  // namespace plugin
}  // namespace eagle
