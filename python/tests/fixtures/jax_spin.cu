// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A JAX custom call that stays busy on the GPU for a chosen number of clock
// cycles, then returns x + 1. It writes -1 into the result first, so an
// early reader sees -1 and a reader that waited sees x + 1. It is the
// "operation still running" that python/tests/test_interop_jax.py needs
// (built there with nvcc at test time).

#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

__global__ void spinAddOne(const double* x, double* y, int64_t n, long long cycles) {
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) y[i] = -1.0;
  __syncthreads();
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] + 1.0;
}

static ffi::Error spinImpl(cudaStream_t stream, int64_t cycles,
                           ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> y) {
  spinAddOne<<<1, 128, 0, stream>>>(x.typed_data(), y->typed_data(),
                                    static_cast<int64_t>(x.element_count()), cycles);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SpinHandler, spinImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<int64_t>("cycles")
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());
