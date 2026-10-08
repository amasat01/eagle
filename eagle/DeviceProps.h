// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>

/**
 * @file DeviceProps.h
 * @brief Mode-neutral device-property facility: raw ``cudaDeviceProp``-derived
 * fields plus the derived roofline quantities a static classifier
 * needs, computed HERE ONCE so
 * both mode-specific factories share a single source of truth for every
 * formula: ``eagle::cuda::deviceProps(device)`` (real hardware,
 * ``eagle/cuda/DeviceProps.h``) and ``eagle::cpu::deviceProps()`` (documented
 * host values, ``eagle/cpu/DeviceProps.h``).
 *
 * CUDA-free by design (no ``EAGLE_CPU_ONLY`` guard, no CUDA includes) —
 * mirrors ``eagle/launch/Traits.h``'s own "dual-mode consumers need this
 * visible in CPP_MODE too" reasoning: the struct and its formulas are plain
 * arithmetic over already-known integers, never a CUDA API call.
 */

namespace eagle {

/**
 * @brief FMA-capable lanes per SM, by compute capability (major, minor) — the
 * multiplier ``peak_flops_sp`` uses. CONSERVATIVE DEFAULT 64 for an unlisted
 * compute capability {record-only}: UNDERESTIMATES peak throughput, which
 * keeps the derived ridge point LOW — so a roofline classifier calls unknown
 * hardware "compute-bound" (class L, never-fuse) SOONER. The safe failure
 * direction: it over-protects, never over-fuses. Known table entries are
 * record-only (NVIDIA architecture whitepapers) — not individually
 * load-bearing, unlike the fallback.
 */
inline int fmaLanesPerSM(int ccMajor, int ccMinor)
{
    if (ccMajor == 6 && ccMinor == 0)
        return 64; // {record-only} Pascal GP100 (datacenter): 64 SP cores/SM
    if (ccMajor == 6 && ccMinor == 1)
        return 128; // {record-only} Pascal GP10x (consumer, e.g. Quadro P2000)
    if (ccMajor == 7)
        return 64; // {record-only} Volta / Turing: 64 SP cores/SM
    if (ccMajor == 8 && ccMinor == 0)
        return 64; // {record-only} Ampere GA100 (datacenter): 64 SP cores/SM
    if (ccMajor == 8)
        return 128; // {record-only} Ampere GA10x (consumer) / Ada
    if (ccMajor == 9)
        return 128; // {record-only} Hopper
    return 64; // {floor} conservative default for an unlisted cc
}

/**
 * @brief FP32:FP64 throughput ratio, by compute capability — divides
 * ``peak_flops_sp`` down to ``peak_flops_dp``. CONSERVATIVE DEFAULT 32 for an
 * unlisted compute capability {floor}: assumes the WORST commonly-seen FP64
 * throughput class (consumer Pascal/Turing-style 1:32), keeping the derived
 * DP ridge point LOW for the same "over-protect, never over-fuse" reason as
 * :func:`fmaLanesPerSM`. Known table entries are record-only. The (6, 1)
 * entry is the standing known-answer for the local dev box (Quadro P2000).
 */
inline int fp64Ratio(int ccMajor, int ccMinor)
{
    if (ccMajor == 6 && ccMinor == 0)
        return 2; // {record-only} Pascal GP100 (datacenter): 1:2
    if (ccMajor == 6 && ccMinor == 1)
        return 32; // {record-only} Pascal GP10x (consumer, e.g. Quadro P2000): 1:32
    if (ccMajor == 7 && ccMinor == 0)
        return 2; // {record-only} Volta: 1:2
    if (ccMajor == 7)
        return 32; // {record-only} Turing (consumer): 1:32
    if (ccMajor == 8 && ccMinor == 0)
        return 2; // {record-only} Ampere GA100 (datacenter): 1:2
    if (ccMajor == 8)
        return 64; // {record-only} Ampere GA10x / Ada (consumer): 1:64
    if (ccMajor == 9)
        return 2; // {record-only} Hopper: 1:2
    return 32; // {floor} conservative default for an unlisted cc
}

/**
 * @brief Device-property facility: raw ``cudaDeviceProp``-derived
 * fields plus the static roofline quantities derived from them — the SINGLE
 * source of truth for the formulas a static classifier consumes
 * (op count + bytes -> static arithmetic intensity
 * vs the device ridge point). Built by ``eagle::cuda::deviceProps(device)``
 * (real ``cudaGetDeviceProperties``) or ``eagle::cpu::deviceProps()``
 * (documented host values, no GPU needed) — never hand-constructed with
 * derived fields already filled in; see :func:`deriveFields`.
 */
struct DeviceProps {
    // -- raw fields (cudaDeviceProp-derived; eagle::cpu::deviceProps()
    // documents its own host-analogue value for each, field-by-field) ------
    std::string name;
    int cc_major              = 0; ///< ``cudaDeviceProp::major``
    int cc_minor               = 0; ///< ``cudaDeviceProp::minor``
    int sm_count                = 0; ///< ``multiProcessorCount``
    int clock_rate_khz         = 0; ///< ``clockRate`` (GPU core clock, kHz)
    int memory_clock_rate_khz = 0; ///< ``memoryClockRate`` (kHz)
    int memory_bus_width_bits = 0; ///< ``memoryBusWidth`` (bits)
    std::size_t shared_mem_per_block = 0; ///< ``sharedMemPerBlock`` (bytes)
    std::size_t shared_mem_per_sm    = 0; ///< ``sharedMemPerMultiprocessor`` (bytes)
    int regs_per_block          = 0; ///< ``regsPerBlock`` (32-bit registers)
    int regs_per_sm              = 0; ///< ``regsPerMultiprocessor`` (32-bit registers)
    int warp_size                 = 0; ///< ``warpSize``

    // -- derived fields (filled by deriveFields(), never set by hand) ------
    double peak_bytes_per_s = 0.0; ///< 2 * memory_clock_rate_hz * (memory_bus_width_bits / 8)
    double peak_flops_sp    = 0.0; ///< sm_count * clock_rate_hz * fmaLanesPerSM(cc) * 2
    double peak_flops_dp    = 0.0; ///< peak_flops_sp / fp64_ratio
    double fp64_ratio       = 0.0; ///< fp64Ratio(cc_major, cc_minor) {record-only table lookup}

    /**
     * @brief flops/byte the device balances at, for ``dtype`` ("float32" or
     * "float64") — ``ridge = peak_flops(dtype) / peak_bytes_per_s``, the
     * roofline break-even point that classifies a node's
     * static arithmetic intensity against. Returns 0 when
     * ``peak_bytes_per_s`` is 0 (a degenerate/synthetic profile with no
     * modeled memory bandwidth) rather than dividing by zero.
     */
    double ridgeFlopsPerByte(const std::string& dtype) const
    {
        const double flops = (dtype == "float32") ? peak_flops_sp : peak_flops_dp;
        return peak_bytes_per_s > 0.0 ? flops / peak_bytes_per_s : 0.0;
    }
};

/**
 * @brief Fill every derived field of @p props from its already-populated raw
 * fields — the ONE place every formula this facility exposes is evaluated
 * ("the SINGLE source of truth for the formulas"). Both
 * ``eagle::cuda::deviceProps`` and ``eagle::cpu::deviceProps`` populate the
 * raw fields their own way and then call this before returning, so the two
 * modes can never drift on how a derived field is computed.
 */
inline DeviceProps deriveFields(DeviceProps props)
{
    // peak_bytes_per_s: 2x for double-data-rate memory, /8 to convert the bus
    // width from bits to bytes, *1000 to convert the clock from kHz to Hz.
    props.peak_bytes_per_s = 2.0
        * (static_cast<double>(props.memory_clock_rate_khz) * 1000.0)
        * (static_cast<double>(props.memory_bus_width_bits) / 8.0);

    const int lanes = fmaLanesPerSM(props.cc_major, props.cc_minor);
    // peak_flops_sp: 2 flops per fused-multiply-add.
    props.peak_flops_sp = static_cast<double>(props.sm_count)
        * (static_cast<double>(props.clock_rate_khz) * 1000.0)
        * static_cast<double>(lanes) * 2.0;

    props.fp64_ratio = static_cast<double>(fp64Ratio(props.cc_major, props.cc_minor));
    props.peak_flops_dp
        = props.fp64_ratio > 0.0 ? props.peak_flops_sp / props.fp64_ratio : 0.0;

    return props;
}

} // namespace eagle
