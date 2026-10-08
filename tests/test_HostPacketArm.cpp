// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The RED-arm test for eagle::cpu::Host's packet
 * arm (packetLoad/packetStore/loadMask/applyTail), which an earlier fix addressed.
 * Before that fix, `loadMask` assumed `typename MaskT::StorageT`, which
 * `aether::simd::PacketMask<DataT,1>` does not define (bare `bool mask_`
 * only -- `aether::simd::PacketMask<DataT, 1>`), breaking every `DataT` at
 * width 1 -- including `double` itself, not only the emulated carrier -- and
 * `applyTail` was hard-typed to `PacketMask<eagle::Real, W>` (Real =
 * double), the one function in the file not templated on `DataT`. No
 * eagle test instantiated any of these four methods with a non-`double`
 * `DataT` before this file.
 *
 * Three arms, mirroring `launchIfNot`'s own composition
 * (loadMask -> applyTail -> {packetLoad,packetStore}, generalized to
 * `DataT`) at the package level directly, since `launchIfNot`/`launchIf`
 * themselves stay hard-typed to `Real` (Host.h, unchanged by that fix):
 *   (i)   aether::banded::BandedReal at its PreferredWidth resolution
 *         (== 1: the RED arm -- broken pre-fix).
 *   (ii)  double at an explicit, forced W=1 (ALSO broken pre-fix: the
 *         scalar fallback specialization is DataT-agnostic, so native
 *         double at width 1 hit the exact same missing-StorageT defect).
 *   (iii) double at its native/preferred width (the unchanged production
 *         path this fix must not disturb).
 *
 * Deliberately NOT "TestBase.h": that header pulls in the full eagle.h
 * umbrella (cuda.h/reduce.h/...), whose __device__/__host__-qualified
 * bodies only nvcc parses -- and in CUDA mode this .cpp is compiled by CXX
 * (g++), not nvcc (same trap test_HostDispatchContract.cpp documents).
 * eagle/cpu/Host.h itself is CUDA-free, but it reaches eagle/typedefs.h,
 * whose eagle-side arm selection keys off EAGLE_CPU_ONLY -- so this TU is
 * explicitly compiled with -DEAGLE_CPU_ONLY (CMakeLists.txt
 * COMPILE_DEFINITIONS on this one source), independent of whichever
 * EAGLE_CPP_MODE the ambient build uses -- dual-registered into BOTH build
 * modes per test_HostDispatchContract.cpp's own recipe (the packet arm is
 * equally CUDA-free under either mode). aether itself needs no per-source
 * escape any more: aether/macros.h has a __CUDACC__ branch, so a
 * host-compiled .cpp inside a CUDA-mode build includes aether unchanged and
 * the former CPU-only build twin is gone. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/cpu/Host.h"

#include <aether/backend/cpu/simd/PacketTraits.h>

/* The emulated family is what this arm exercises ("emulation is
 * an alternative arm"): neither `double` nor `float`, so
 * `aether::simd::PreferredWidth` resolves it to 1 and the W==1 specialization
 * below is the one under test.
 *
 * AETHER_GAP: the carrier `aether::banded::Band` CANNOT be the DataT here.
 * `aether::simd::Packet<DataT, 1>::zero()` is spelled `Packet{ DataT{ 0 } }`
 * (`Packet<DataT, 1>::zero()`) and `maskLoad` — which
 * `eagle::cpu::Host::packetLoad` calls unconditionally — instantiates it, but
 * `Band` deliberately has NO scalar constructor at all: it is the
 * metadata-free 3-limb register carrier and its operator wall deletes every
 * mix with a native scalar on purpose (Band.h). Every banded WORKING carrier
 * (`Band`, `Ff1`, `Ff2`) is in the same position, so the scalar Packet
 * fallback is closed to the whole rung. aether needs a width-1 `zero()` that
 * does not assume `DataT{0}` (value-initialise `DataT{}`, or a
 * `packet_zero<DataT>` customisation point).
 *
 * Worked around eagle-side by carrying the family's STORAGE sibling,
 * `aether::banded::BandedReal`: an 8-byte codec word that IS
 * value-constructible, carries the same certified 53-bit numerics (its
 * operators route through `Band` and pack back on assignment,
 * `aether/banded/BandedRealOps.h`), and resolves to the same PreferredWidth
 * == 1. The RED arm's subject — a non-double/non-float DataT through the
 * scalar Packet/PacketMask specialization — is unchanged. Its
 * `fromDouble`/`toDouble` are also the PUBLIC host-only ingest/egress
 * terminals used for the double reference comparison below (the carrier's own
 * `bandFromIEEE`/`bandToIEEE` live in `aether::banded::detail` and are not
 * consumer surface). */
#include <aether/banded/banded.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace eagle_tests {
namespace HostPacketArmTest {

using eagle::idx_t;
using SdT = aether::banded::BandedReal;

// `BandedReal` is neither `double` nor `float`, so PreferredWidth (aether/backend/cpu/
// simd/PacketTraits.h) resolves it to the scalar (Width==1) fallback --
// this is the compile-time fact that makes the emulated carrier land in
// exactly the specialization that earlier fix addressed. Checked here so a future
// PreferredWidth change that widened this would fail loudly, not silently
// narrow the RED-arm's coverage.
static_assert(aether::simd::PreferredWidth<SdT> == 1,
    "BandedReal must resolve to the scalar PacketMask/Packet "
    "specialization (Width==1) for this test to cover the RED arm "
    "the earlier defect class fixed.");

class HostPacketArmTest : public ::testing::Test { };

namespace {

/** @brief `double` -> the emulated carrier, through aether's PUBLIC host-only
 *  ingest terminal. There is no implicit route: `Band`'s operator wall
 *  (`aether/banded/Band.h`) deletes every banded<->native mix on purpose, and
 *  `BandedReal`'s own `double` entry points are NAMED, never conversions.
 *
 *  Unlike the carrier this replaces, `BandedReal::fromDouble` THROWS
 *  `aether::Error` outside tier 1's exact window [2^-94, 2^126)
 *  (`aether/banded/BandCell8.h`): the carrier needs no FP64 hardware, and its
 *  exponent range is FP32's, not FP64's. Every value this file feeds it is
 *  inside that window. */
static SdT bandOf(double v) { return SdT::fromDouble(v); }

/** @brief …and back, through the matching PUBLIC egress terminal. */
static double asDouble(SdT b) { return b.toDouble(); }

constexpr std::size_t kN = 7; // deliberately not a multiple of any W
                               // tested below, so every width>=2 case
                               // below drives a genuine tail packet.

// Values whose squares are NOT exactly representable in binary double
// (unlike e.g. 0.5*0.5), so the banded arm below exercises a genuine
// rounding step rather than an operation that would be exact under any
// correctly-rounded implementation regardless of the fix.
constexpr std::array<double, kN> kValues
    = { 0.1, 1.1, -0.3, 2.7, -1.9, 3.3, 4.4 };

// Marks lanes 1, 4 and 6 as "terminated" (skip): must pass through
// unchanged rather than being squared -- exactly the semantics
// launchIfNot's own `~loadMask(...)` composition gives its callers.
constexpr std::array<bool, kN> kTerminated
    = { false, true, false, false, true, false, true };

/* This file's gap is now closed: `aether::simd::Packet` now carries
 * a `select(mask, a, b)` lane-wise blend again, so eagle's `selectMasked`
 * workaround (built by hand from `store`/`maskStore`/`load`) is DELETED. It is
 * a HIDDEN FRIEND on purpose (aether/backend/cpu/simd/Packet.h), reachable
 * only by ADL on a `Packet` argument — which is exactly what lets it coexist
 * with 5-parameter POSIX `::select` from <sys/select.h> in this same TU. So
 * the call below is deliberately UNQUALIFIED; `aether::simd::select(...)`
 * would not find it. */

/** @brief Drive packetLoad -> loadMask -> applyTail -> packetStore for a
 *  given DataT/W exactly the way launchIfNot's body composes them
 *  (Host.h: loadMask<Real,W> then applyTail<W> then a masked kernel
 *  call), generalized here to any DataT -- the capability that fix unblocks.
 *  Active (non-terminated) lanes get squared in-place; terminated lanes
 *  keep their input value. Loops in W-sized packets so `n % W != 0`
 *  drives a genuine tail (`PacketIndex::makeTail`), exercising both
 *  PacketMask specializations' `firstN()` as well as the W==1 branch
 *  `loadMask` now takes. */
template<typename DataT, std::size_t W>
std::array<DataT, kN> runMaskedSquareKernel(
    const std::array<DataT, kN>& in, const std::array<bool, kN>& terminated)
{
    using eagle::cpu::Host;
    std::array<DataT, kN> out = in;

    for (idx_t base = 0; base < static_cast<idx_t>(kN); base += W) {
        const idx_t remaining
            = std::min<idx_t>(W, static_cast<idx_t>(kN) - base);
        const auto pi = (remaining == W)
            ? aether::PacketIndex<W>::make(base)
            : aether::PacketIndex<W>::makeTail(base, remaining);

        const auto vals    = Host::packetLoad<DataT, W>(in.data(), pi);
        const auto squared = vals * vals;

        const auto skipMask = Host::loadMask<DataT, W>(terminated.data(), pi);
        const auto active   = Host::applyTail<W>(~skipMask, pi);

        const auto chosen = select(active, squared, vals);
        Host::packetStore<DataT, W>(out.data(), pi, chosen);
    }
    return out;
}

std::array<double, kN> scalarSquareReference(
    const std::array<double, kN>& in, const std::array<bool, kN>& terminated)
{
    std::array<double, kN> ref{};
    for (std::size_t i = 0; i < kN; ++i)
        ref[i] = terminated[i] ? in[i] : in[i] * in[i];
    return ref;
}

} // namespace

TEST_F(HostPacketArmTest, DoubleAtExplicitWidthOneMatchesScalarReferenceExactly)
{
    // Forces the scalar (Width==1) specialization for native `double` --
    // the SECOND RED arm the pre-fix `MaskT::StorageT` assumption broke
    // (it is DataT-agnostic: any DataT at W=1 hit it, not only the emulated carrier).
    constexpr std::size_t W = 1;
    const auto out = runMaskedSquareKernel<double, W>(kValues, kTerminated);
    const auto ref = scalarSquareReference(kValues, kTerminated);

    for (std::size_t i = 0; i < kN; ++i)
        // Same type, same single multiply, no reduction/associativity in
        // play -- IEEE 754 correctly-rounded double * double is bit-exact
        // regardless of scalar vs. packet form, so exact equality (not a
        // tolerance) is the right assertion here.
        EXPECT_EQ(out[i], ref[i]) << "lane " << i;
}

TEST_F(HostPacketArmTest, WideDoubleReferenceMatchesScalarReferenceExactly)
{
    // The unchanged production path: double at its native/preferred SIMD
    // width (PreferredWidth<double>; SSE2-or-wider on any x86_64 build,
    // 2 on this box's default -msse2 baseline). Already worked pre-fix
    // this is the "zero behavior change" regression guard.
    constexpr std::size_t W = aether::simd::PreferredWidth<double>;
    const auto out = runMaskedSquareKernel<double, W>(kValues, kTerminated);
    const auto ref = scalarSquareReference(kValues, kTerminated);

    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_EQ(out[i], ref[i]) << "lane " << i;
}

TEST_F(HostPacketArmTest,
    SoftDoubleAtPreferredWidthMatchesScalarReferenceWithinDerivedTolerance)
{
    // The RED arm: the emulated carrier instantiated through the SAME
    // PreferredWidth resolution any non-double/float DataT goes through
    // (== 1, pinned by the static_assert above) -- this failed to COMPILE
    // before that fix.
    constexpr std::size_t W = aether::simd::PreferredWidth<SdT>;

    std::array<SdT, kN> sdValues{};
    for (std::size_t i = 0; i < kN; ++i)
        sdValues[i] = bandOf(kValues[i]);

    const auto out = runMaskedSquareKernel<SdT, W>(sdValues, kTerminated);
    const auto ref = scalarSquareReference(kValues, kTerminated);

    for (std::size_t i = 0; i < kN; ++i) {
        if (kTerminated[i]) {
            // A terminated lane is a pure copy through `select` -- no
            // arithmetic touches it, so it is bit-exact regardless of
            // DataT (the banded value was itself constructed directly
            // from `kValues[i]`, a lossless round trip: `BandedReal::
            // toDouble` is documented as the exact inverse of `fromDouble`
            // on every tier-1 value, and all of `kValues` is tier-1).
            EXPECT_EQ(asDouble(out[i]), kValues[i]) << "lane " << i;
            continue;
        }
        // A single correctly-rounded multiply on each side (native FP64
        // hardware for `ref`, the banded carrier's own certified 53-bit
        // emulation for `out`) is independently within 0.5ULP of the
        // exact mathematical product, so the two can differ by at most
        // ~1ULP even when both round correctly -- mirrored from
        // test_Reduction.cu's CUDA_SoftDoubleSum_MatchesDoubleTwin
        // ("emulated adds are correctly rounded but can flip RNE ties --
        // never assert bit-exact arithmetic parity", the numerics
        // contract's item 1). A 4x safety factor over that 1ULP bound absorbs the
        // tie-flip without hiding a real defect at this magnitude.
        const double tol = 4.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(ref[i]));
        EXPECT_NEAR(asDouble(out[i]), ref[i], tol) << "lane " << i;
    }
}

} // namespace HostPacketArmTest
} // namespace eagle_tests
