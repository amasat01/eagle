// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <limits>

#include <aether/aether.h>

/* Every alias below is aether's: eagle/cpu, eagle/filtering, eagle/cuda,
 * eagle/util/ObservableArray.h, plugin/plugin_registry, and the
 * DLPack/nanobind ABI all consume it through this header. */

/** @brief Top-level eagle namespace. */
namespace eagle {

/** @brief Re-export aether's per-sample index type. */
using aether::SampleIndex;

/** @brief Scalar floating-point type used throughout eagle. */
using Real     = double;
/** @brief Index type (from aether -- `aether::idx_t`). */
using idx_t    = aether::idx_t;
/** @brief General-purpose signed integer type for metadata fields. */
using mInt_t   = int;
/** @brief Unsigned integer type. */
using uint_t   = unsigned int;
/** @brief Dimension type (from aether). */
using dims_t   = aether::dims_t;
/** @brief Vector-dimension type (alias for ``dims_t``). */
using vecdim_t = aether::dims_t;

/** @brief aether SoA array with element type ``DT`` and vector extent ``VD``.
 *
 *  aether merges a two-tier `scalar::Array`/`vector::Array` design into ONE
 *  `Array<T, Es...>` whose extents pack carries the static (vector) modes and
 *  whose trailing dynamic mode is the batch -- so a fixed-width `vector::Array<DT,VD>`
 *  is spelled `Array<DT, VD>` and its scalar sibling is simply `Array<DT>`. */
template<typename DT, idx_t VD>
using VecNTArray = aether::Array<DT, VD>;
/** @brief aether register-resident item with element type ``DT`` and
 *         dimension ``VD`` (the per-item vector width). */
template<typename DT, idx_t VD>
using VecNT = aether::Item<DT, VD>;

/** @brief 3-D double-precision vector. Same type as `aether::Vec3d`
 *         (`aether/aether.h`) under a different name. */
using Vec3R      = VecNT<Real, 3>;
/** @brief 6-D double-precision vector (position + velocity). */
using Vec6R      = VecNT<Real, 6>;
/** @brief SoA array of 3-D double-precision vectors. */
using Vec3RArray = VecNTArray<Real, 3>;
/** @brief SoA array of 6-D double-precision vectors. */
using Vec6RArray = VecNTArray<Real, 6>;

/** @brief Shorthand for the device-side view type of an aether scalar array.
 *
 *  aether collapses an earlier `GRef`/`WRef`/`GRef::HandleT` two-tier into the one
 *  `aether::View` (`aether/view/View.h`): the View IS the lightweight
 *  descriptor, and "global vs work reference" is now just which chunk the View
 *  was built over. `Array<T>::ViewT` is that View type. */
template<typename ComponentT>
using GRefArrT = typename aether::Array<ComponentT>::ViewT;

/** @brief Read-only counterpart of ``GRefArrT`` — `Array<T>::ConstViewT`.
 *
 *  aether has no implicit `View<T>` -> `View<const T>` conversion; call
 *  `.as_const()` on a mutable view to reach this type. */
template<typename ComponentT>
using CRefArrT = typename aether::Array<ComponentT>::ConstViewT;

/** @brief The aether `Device` eagle's HOST-resident buffers live on — the
 *         same one `aether::Array` gives its own host chunk. */
inline aether::Device hostDevice()
{
#ifdef EAGLE_CPU_ONLY
    return aether::Device(kDLCPU);
#else
    return aether::Device(kDLCUDAHost);
#endif
}

/** @brief The aether `Device` eagle's DEVICE-resident buffers live on. In an
 *         `EAGLE_CPU_ONLY` build there is only one copy, so this is
 *         `hostDevice()` — mirroring `aether::Array`'s own single-chunk
 *         `AETHER_CPP_MODE` arm. */
inline aether::Device deviceDevice()
{
#ifdef EAGLE_CPU_ONLY
    return aether::Device(kDLCPU);
#else
    return aether::Device(kDLCUDA);
#endif
}

/** @brief Non-owning scalar `View` over a raw, unit-stride span of @p n
 *         elements -- an earlier `scalar::Array<T>::GRef(ptr, size)` constructor.
 *
 *  aether's `make_view()` is HOST-ONLY (it throws `aether::Error` on a bad
 *  span, and device code cannot throw — `aether/view/MakeView.h`), so a span
 *  view that must also be buildable inside a kernel is assembled from `View`'s
 *  public `(data, mapping, device)` constructor, exactly as
 *  `aether::make_work_view` does for shared memory. */
template<typename T>
AETHER_DEVICEHOST() inline GRefArrT<T> spanView(T* data, idx_t n, aether::Device dev)
{
    using ExtT = aether::extents<aether::dyn>;
    using MapT = typename aether::layout_stride::template mapping<ExtT>;
    return GRefArrT<T>(data, MapT(ExtT(n), { std::size_t{ 1 } }), dev);
}

/** @brief `spanView` over a component-packed (SoA) span: @p C components of
 *         @p n samples each, component `c` starting at `data + c * n`.
 *
 *  The pitch is @p n exactly — NOT `aether::Array`'s quantised `capacity()`.
 *  This is the shape a caller carving a `C*n` scratch slot chooses for itself,
 *  and `aether::View::component<I>()` reads the pitch back off the mapping, so
 *  the two never disagree. */
template<std::size_t C, typename T>
AETHER_DEVICEHOST() inline auto packedSpanView(T* data, idx_t n, aether::Device dev)
{
    using ExtT = aether::extents<C, aether::dyn>;
    using MapT = typename aether::layout_stride::template mapping<ExtT>;
    return aether::View<T, ExtT, aether::layout_stride>(
        data, MapT(ExtT(n), { std::size_t(n), std::size_t{ 1 } }), dev);
}

/** @brief Native stream handle threaded through eagle's copy paths.
 *
 *  aether declares `aether::Stream` only in CUDA-backed builds
 *  (`aether/chunk/Copy.h`); the pure-C++ arm keeps the neutral integer tag
 *  `eagle::cpu::StreamRef::native()` already returns, so a "stream" argument
 *  threads through `EAGLE_CPU_ONLY` code unchanged. */
#ifdef EAGLE_CPU_ONLY
using nativeStream_t = int;
#else
using nativeStream_t = aether::Stream;
#endif

/** @brief Allocate an `aether::Array` of `n` samples with every element set
 *         to `initVal`.
 *
 *  An earlier `Array(size, initVal = 0)` ALWAYS filled (zero by default); aether's
 *  `Array(n)` deliberately leaves the storage UNINITIALISED (it allocates a
 *  `Chunk` and nothing else). Every eagle site that relied on the earlier implicit
 *  zero-fill goes through here, so the fill is explicit and greppable rather
 *  than an inherited accident. Fills the HOST copy; call `upload()` to reach
 *  the device one, exactly as aether's explicit-transfer contract requires. */
template<typename T, std::size_t... Es>
inline aether::Array<T, Es...> makeArray(std::size_t n, const T& initVal = T{})
{
    aether::Array<T, Es...> arr(n);
    /* Fill the WHOLE backing span, `capacity()` (not `samples()`) per static
     * mode: aether quantises capacity up to `kCapacityAlignment` and pitches
     * component `c` at `c * capacity()`, so a flat `samples()`-long loop would
     * both miss the tail components and leave the reserved spare dirty. */
    constexpr std::size_t itemSize = (Es * ... * std::size_t{ 1 });
    T* base = arr.hostView().data();
    for (std::size_t i = 0; i < itemSize * arr.capacity(); ++i)
        base[i] = initVal;
    return arr;
}


/** @brief Deep copy of an `aether::Array`. 
 *
 *  `aether::Array` owns move-only `Chunk`s, so it has no copy constructor and
 *  no `clone()`. Both arrays get the SAME quantised `capacity()` for the same
 *  `samples()`, so one flat span copy reproduces the pitch exactly. Copies the
 *  HOST copy and then `upload()`s, matching aether's explicit-transfer
 *  contract (a device-only write not yet downloaded is NOT carried over —
 *  An earlier clone copied both sides). */
template<typename T, std::size_t... Es>
inline aether::Array<T, Es...> cloneArray(const aether::Array<T, Es...>& src)
{
    aether::Array<T, Es...> out(src.samples());
    constexpr std::size_t itemSize = (Es * ... * std::size_t{ 1 });
    const T* from = src.hostView().data();
    T* to         = out.hostView().data();
    for (std::size_t i = 0; i < itemSize * src.capacity(); ++i)
        to[i] = from[i];
    out.upload();
    return out;
}

} // namespace eagle
