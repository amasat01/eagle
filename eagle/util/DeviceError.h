// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>
#include <map>
#include <string>   // EAGLE_CHECK_ALWAYS formats its own message
#include <vector>

#include <aether/err/Error.h>

#include "eagle/util/throw.h"

#if defined(EAGLE_DEBUG_MODE) && !defined(EAGLE_CPU_ONLY)

/* The trailing cudaGetLastError() is a DELIBERATE, status-discarded CLEAR of
 * the CUDA runtime's per-thread last-error slot. The slot is set by ANY
 * failing runtime call and is only ever cleared by reading it — successful
 * calls leave it untouched — so without this, EAGLE_KERNEL_POST's
 * EAGLE_CHECK_ALWAYS(cudaGetLastError()) throws a STALE error from arbitrary
 * earlier code, misattributed to THIS span's file/line/function. Measured
 * the first time the debug arm ever ran: every test that deliberately
 * provokes a failing CUDA call (the teardown-poison suite, the
 * conditional-graph 801 probes) poisoned the NEXT PRE/POST site — a fixture
 * constructor or an unrelated launch(). PRE clearing the slot makes POST
 * report exactly the errors arising between PRE and POST, which is what
 * launch instrumentation means. The device-side flag protocol (reset just
 * above) is a separate channel and is not affected. */
#define EAGLE_KERNEL_PRE()                                                     \
    {                                                                          \
        ::eagle::err::detail::resetDeviceErrors(::eagle::err::deviceErrorFlag);\
        cudaGetLastError();                                                    \
    }

#define EAGLE_KERNEL_POST()                                                    \
    {                                                                          \
        EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());                           \
        ::eagle::err::detail::checkDeviceErrors(::eagle::err::deviceErrorFlag, \
            ::eagle::err::errorMessages, __FILE__, __LINE__,                   \
            static_cast<const char*>(__func__));                               \
    }

#define EAGLE_GPU_THROW(errCode)                                               \
    {                                                                          \
        ::eagle::err::detail::reportDeviceError(                               \
            errCode, ::eagle::err::deviceErrorFlag);                           \
    }

#define EAGLE_GPU_ASSERT(condition, errCode)                                   \
    {                                                                          \
        if (!(condition))                                                      \
            ::eagle::err::detail::reportDeviceError(                           \
                errCode, ::eagle::err::deviceErrorFlag);                       \
    }

#else // EAGLE_DEBUG_MODE && !EAGLE_CPU_ONLY

#define EAGLE_KERNEL_PRE()                                                     \
    {                                                                          \
    }

#define EAGLE_KERNEL_POST()                                                    \
    {                                                                          \
    }

#define EAGLE_GPU_THROW(errCode)                                               \
    {                                                                          \
    }

#define EAGLE_GPU_ASSERT(condition, errCode)                                   \
    {                                                                          \
    }

#endif // EAGLE_DEBUG_MODE && !EAGLE_CPU_ONLY

/* ------------------------------------------------------------------------- *
 * EAGLE_CHECK_ALWAYS — the CUDA API check for eagle's host-API call sites.
 *
 * eagle's plain ``EAGLE_CHECK`` — compiled out to a bare expression statement
 * unless ``EAGLE_DEBUG_MODE`` was set — was retired: a release-mode discard
 * of a CUDA API status is a silent-wrong-data defect, not an acceptable
 * hot-path trade. Every eagle
 * product call site guarded this way was a host-API call (stream/event sync,
 * async memcpy, event record/wait, ``cudaLaunchHostFunc``, allocation) —
 * none is per-sample device code — so release-mode discard bought no
 * measurable per-call savings against a real risk: a fail-silent status such
 * as ``cudaStreamEndCapture``'s ``cudaErrorStreamCaptureUnjoined`` (which
 * *also* hands back a null ``cudaGraph_t``) would propagate a null into graph
 * adoption and only surface much later, far from its cause.
 *
 * ``EAGLE_CHECK_ALWAYS`` is therefore now THE check for these call sites, and
 * is defined IDENTICALLY in both build modes.
 *
 * **Why eagle formats this message itself.** It used
 * to delegate wholesale to an upstream helper, which
 * formats with ``cudaGetErrorString`` alone. CUDA puts NO digits in either the
 * string or the name (measured: code 801 → string ``"operation not supported"``,
 * name ``"cudaErrorNotSupported"``), so a caller could not anchor a test on the
 * error's identity — only on an English sentence. That sentence is
 * CUDA-version-dependent and carries no stability guarantee, whereas
 * ``cudaErrorNotSupported`` is the stable identifier.
 *
 * eagle holds the ``cudaError_t`` at the raise site, so it prepends the stable
 * NAME and the numeric code and then hands the enriched text to eagle's own
 * thrower (``eagle/util/throw.h``). The exception type is ``aether::Error``:
 * aether collapsed an earlier CUDA-specific error type into the one unified
 * host exception, so there is no CUDA-specific subtype to name any more.
 * ------------------------------------------------------------------------- */
#ifndef EAGLE_CPU_ONLY

#define EAGLE_CHECK_ALWAYS(expr)                                               \
    {                                                                          \
        const cudaError_t _eagle_err = (expr);                                 \
        if (_eagle_err != cudaSuccess) {                                       \
            ::eagle::err::detail::throwAt<::aether::Error>(                    \
                __FILE__, __LINE__, __func__,                                  \
                std::string(cudaGetErrorName(_eagle_err)) + " ("               \
                + std::to_string(static_cast<int>(_eagle_err)) + "): "         \
                + cudaGetErrorString(_eagle_err));                             \
        }                                                                      \
    }

#else // EAGLE_CPU_ONLY

/* No CUDA API exists to check in a CPU-only build. */
#define EAGLE_CHECK_ALWAYS(expr)                                               \
    {                                                                          \
        expr;                                                                  \
    }

#endif // EAGLE_CPU_ONLY

/* ------------------------------------------------------------------------- *
 * EAGLE_CHECK_NOTHROW — an unconditional, non-throwing CUDA API check.
 *
 * Destructor/teardown call sites (``~Event``, ``~Stream``,
 * ``~Launcher`` and the shared ``destroy_()``/``destroyInstance_()`` helpers
 * those destructors call — also reached from move-assignment, which is fine:
 * both paths are "retire this resource", never "build a new one") must NEVER
 * throw. A throwing destructor unwinding during driver shutdown aborts the
 * process instead of reporting an error (the facade-global-teardown lesson).
 *
 * ``EAGLE_CHECK_NOTHROW`` always evaluates ``expr`` — never compiled out,
 * same as ``EAGLE_CHECK_ALWAYS`` — and on a non-``cudaSuccess`` return logs to
 * ``stderr`` and swallows: visible, never fatal.
 *
 * Purely additive: it changes no existing macro's semantics. Do not reach
 * for it on a build/capture-time call — those want ``EAGLE_CHECK_ALWAYS`` so
 * a failure is still reported loudly; reserve ``EAGLE_CHECK_NOTHROW`` for
 * code that runs at, or is shared with, object teardown.
 * ------------------------------------------------------------------------- */
#ifndef EAGLE_CPU_ONLY

#define EAGLE_CHECK_NOTHROW(expr)                                             \
    {                                                                         \
        cudaError_t eagleCheckNothrowErr_ = (expr);                          \
        if (eagleCheckNothrowErr_ != cudaSuccess) {                          \
            std::fprintf(stderr,                                             \
                "EAGLE_CHECK_NOTHROW: CUDA error (swallowed, teardown) at "  \
                "%s:%d (%s): %s\n",                                           \
                __FILE__, __LINE__, __func__,                                 \
                cudaGetErrorString(eagleCheckNothrowErr_));                   \
        }                                                                     \
    }

#else // EAGLE_CPU_ONLY

/* No CUDA API exists to check in a CPU-only build. */
#define EAGLE_CHECK_NOTHROW(expr)                                             \
    {                                                                         \
        expr;                                                                \
    }

#endif // EAGLE_CPU_ONLY

/* Define the required routines calling the CUDA API if not in CPU only mode*/
#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace err {

/* The DEVICE-side error protocol is EAGLE-SIDE, not aether's.
 * It is the execution layer's contract with its plugins — a domain library
 * layered on eagle (e.g. downstream trajectory libraries) keeps its OWN flag + code set and wraps
 * these same macros. aether's
 * ``aether::DeviceFlag`` (``aether/err/DeviceFlag.h``) stays the ONE-BIT
 * primitive it was designed to be: a class owning a sticky ``uint32_t`` word
 * in a ``Chunk`` pair, set with ``atomicOr(ptr, 1u)`` and read back on the
 * host — it answers "did anything go wrong", not "which of 64 things".
 *
 * What eagle needs, and now owns outright:
 *   * a 64-bit MULTI-CODE bitmask (one bit per error kind, OR-ed from any
 *     number of threads — a kernel that trips three distinct conditions
 *     reports three, not "something happened"),
 *   * a code -> message map so the raise site's identity survives to the host
 *     as text rather than a number, and
 *   * a plain ``__device__`` SYMBOL (not a Chunk-backed object), because
 *     ``cudaMemcpyToSymbol``/``cudaMemcpyFromSymbol`` on a header-declared
 *     symbol is what lets a kernel raise with a bare ``atomicOr`` and no
 *     launch-parameter plumbing at all.
 *
 * The thrown type is ``aether::Error``: aether collapsed earlier separate
 * ``err::DeviceError``/``err::CUDAError`` into the one host exception, so
 * there is no device-specific subtype to name any more (see below --
 * the generic thrower is eagle-side, ``eagle/util/throw.h``). */

/** @brief Unsigned integer type used as a bitmask for device-side error flags. */
using DeviceFlag      = unsigned long long int;
/** @brief Map from individual error flag bits to human-readable messages. */
using ErrorMessageMap = std::map<DeviceFlag, const char*>;

/* Leverage macros to detect if we are compiling into a shared module. If so,
 * make the variable extern (will be defined later) to keep a single instance */
#ifndef EAGLE_INTO_SHARED_LIBRARY
[[maybe_unused]] static __device__ DeviceFlag deviceErrorFlag;
#else
[[maybe_unused]] extern __device__ DeviceFlag deviceErrorFlag;
#endif

/** @brief Bit-flag error codes reported from device kernels via
 *  ``EAGLE_GPU_THROW``. Domain libraries layered on eagle keep
 *  their own flag + codes and wrap these macros, exactly as eagle used to
 *  wrap this header's. */
enum EagleErrorTypes : DeviceFlag {
    NO_ERROR           = (DeviceFlag)0,      ///< No error.
    FAILED_WARP_LEADER = (DeviceFlag)1 << 0, ///< Warp leader failed a required condition.
    ERR_MAX = (DeviceFlag)1 << 63            ///< Sentinel — maximum error flag value.
};

const static ErrorMessageMap errorMessages = {
    { FAILED_WARP_LEADER,
        "Warp leader error: the warp leader should always meet the condition, "
        "so this should never happen." }
    // Error messages corresponding to error variants
    // { MY_ERROR, "you did a thing wrong" },
};

namespace detail {

/**
 * @brief Discard the CUDA runtime's per-thread last-error slot.
 *
 * The slot is set by ANY failing runtime call and is cleared only by reading
 * it, so an entry point that launches kernels and then checks
 * ``EAGLE_CHECK_ALWAYS(cudaGetLastError())`` would otherwise throw an error
 * left behind by unrelated earlier code (one already reported, or caught and
 * handled by its caller), misattributed to its own launch. Calling this on
 * entry makes the post-launch checks report exactly the errors of this entry
 * point's own launches — the same contract ``EAGLE_KERNEL_PRE`` gives the
 * debug arm. A sticky (context-corrupting) error is not hidden by this: every
 * later runtime call returns it again.
 */
inline void clearStaleLastError()
{
    static_cast<void>(cudaGetLastError());
}

/**
 * @brief Decode a bitmask device flag into a human-readable message string.
 *
 * Every SET bit is reported, including one with no map entry (as
 * ``[unknown error]``) — a multi-code flag whose decode dropped the codes it
 * did not recognise would be a silent-wrong-diagnosis, which is the whole
 * reason the flag is a bitmask and not a single value.
 *
 * @param[in] deviceFlag  Bitmask of ``EagleErrorTypes`` values.
 * @param[in] messageMap  Map from individual flag bits to message strings.
 * @return Concatenated error description.
 */
[[maybe_unused]] static std::string flagToMessage(
    const DeviceFlag& deviceFlag, const ErrorMessageMap& messageMap)
{
    DeviceFlag errorKind = 1;
    std::vector<const char*> messages;
    for (std::size_t i = 0; i < 8 * sizeof(DeviceFlag); i++, errorKind <<= 1) {
        if (deviceFlag & errorKind) {
            const auto it = messageMap.find(errorKind);
            messages.push_back(
                it != messageMap.end() ? it->second : "[unknown error]");
        }
    }

    if (messages.size() == 1)
        return "GPU error: " + std::string(messages[0]);
    if (messages.size() > 1) {
        std::string message = "Multiple GPU errors: ";
        for (const char* m : messages)
            message += std::string(m) + "; ";
        return message;
    }
    return "GPU error (empty flag)";
}

/** @brief Reset the device flag used to report errors in kernels.
 *  @param[in,out] flag  Device-side error flag to zero out.
 */
[[maybe_unused]] static void resetDeviceErrors(DeviceFlag& flag)
{
    DeviceFlag zero = 0;
    EAGLE_CHECK_ALWAYS(cudaMemcpyToSymbol(flag, &zero, sizeof(DeviceFlag)));
}

/**
 * @brief Check for error flags raised during kernel execution.
 *
 * @param[in] flag        Device-side error flag to read.
 * @param[in] messageMap  Map from flag bits to error descriptions.
 * @param[in] file        Source file name.
 * @param[in] line        Source line number.
 * @param[in] func        Enclosing function name.
 * @throws aether::Error  If any error bits are set in ``flag``.
 */
[[maybe_unused]] static void checkDeviceErrors(DeviceFlag& flag,
    const ErrorMessageMap& messageMap, const std::string& file, int line,
    const std::string& func)
{
    // Check for kernel launch errors first: a launch that never ran cannot
    // have raised a device flag, so reporting the flag first would report
    // "no error" for a kernel that did not execute.
    EAGLE_CHECK_ALWAYS(cudaGetLastError());

    DeviceFlag error = 0;
    EAGLE_CHECK_ALWAYS(
        cudaMemcpyFromSymbol(&error, flag, sizeof(DeviceFlag)));

    if (error != 0) {
        ::eagle::err::detail::throwAt<::aether::Error>(
            file, line, func, flagToMessage(error, messageMap));
    }
}

/** @brief Raise an error flag from device code.
 *  @param[in] errorCode  Error bit(s) to set (from ``EagleErrorTypes``).
 *  @param[in,out] flag   Device-side flag; bits are atomically OR-ed in.
 */
[[maybe_unused]] __device__ static void reportDeviceError(
    DeviceFlag errorCode, DeviceFlag& flag)
{
    atomicOr(&flag, errorCode);
}

} // namespace detail

} // namespace err
} // namespace eagle

#endif
