// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace eagle {
namespace cpu {

/** @brief Host stand-in for a CUDA stream handle.
 *
 *  The CPU backend runs every kernel synchronously through
 *  ``eagle::cpu::Host::launch``, so there is no stream to reference and nothing
 *  to wait on: ``synchronize()`` is a no-op and ``native()`` yields the neutral
 *  ``0`` that host copy paths accept and ignore. It is the
 *  backend-generic ``eagle::Stream`` on the host side, letting interface code
 *  thread a "stream" argument through unchanged in ``EAGLE_CPU_ONLY`` builds.
 *  Mirrors the surface of ::eagle::cuda::StreamRef so both resolve behind the
 *  same ``eagle::Stream`` alias. Pure standard C++ — no CUDA dependency — and
 *  therefore available in both build modes. */
class StreamRef {
public:
    /** @brief Underlying native "stream" type on the host (an integer tag). */
    using NativeT = int;

    /** @brief Default constructor. */
    StreamRef() = default;

    /** @brief No-op: host launches are already synchronous. */
    void synchronize() const { }

    /** @brief Neutral stream tag accepted (and ignored) by host copies. */
    int native() const { return 0; }
};

} // namespace cpu
} // namespace eagle
