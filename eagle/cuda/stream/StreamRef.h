// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/** @brief Non-owning handle to a CUDA stream.
 *
 *  Where ::eagle::cuda::Stream is an owning RAII wrapper that creates and
 *  destroys a real ``cudaStream_t``, this handle merely *references* an
 *  existing stream — the legacy default stream (``0``) unless constructed from
 *  another. It is the backend-generic ``eagle::Stream`` on the CUDA side:
 *  cheap to default-construct, trivially copyable, and exposing exactly the
 *  surface interface code needs — ``synchronize()`` plus a ``native()``
 *  accessor returning the underlying ``cudaStream_t`` for stream-aware
 *  async copies. Downstream code names it only through the ``eagle::Stream``
 *  alias so a single ``#ifdef`` selects the host variant; no raw
 *  ``cudaStream_t`` is guarded at any call site. */
class StreamRef {
public:
    /** @brief Underlying native stream type. */
    using NativeT = cudaStream_t;

    /** @brief Default handle to the legacy default stream (``0``). */
    StreamRef() = default;

    /** @brief Wrap an existing native stream (e.g. an owning Stream's
     *  ``cuda()``), so a caller holding a real stream can pass a lightweight
     *  reference without transferring ownership. */
    StreamRef(cudaStream_t stream)
        : stream_{ stream }
    {
    }

    /** @brief Block the host until all work on the referenced stream
     *  completes. */
    void synchronize() const { EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(stream_)); }

    /** @brief Return the underlying ``cudaStream_t`` for async memory copies. */
    cudaStream_t native() const { return stream_; }

private:
    cudaStream_t stream_ = 0;
};

} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
