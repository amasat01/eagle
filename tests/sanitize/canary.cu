// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// Sanitizer non-vacuity canary (device). Each mode commits ONE defect that
// compute-sanitizer must report; the local gate (tests/sanitize/sanitize_gate.sh)
// asserts memcheck >= 1 error on dev-oob and racecheck >= 1 hazard on dev-race.
// Run bare, both modes exit 0: a quiet tool run on a canary means the tool is
// not instrumenting the binary.
//
//   canary dev-oob    32-bit word-widened atomic OR on a 1-byte cudaMalloc
//                     (the widening a byte-flag "set to 1" must never use)
//                     -> memcheck: Invalid __global__ atomic of size 4 ... out of bounds
//   canary dev-race   one thread writes a shared word, another warp reads it, no __syncthreads
//                     -> racecheck: write-write hazard

#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

namespace {

// The widening pattern: addresses the containing 32-bit word of a byte slot.
__global__ void widenedOr(unsigned char* flag) {
    unsigned* word = reinterpret_cast<unsigned*>(
        reinterpret_cast<unsigned long long>(flag) & ~3ull);
    atomicOr(word, 1u);
}

__global__ void sharedWriteWrite(int* out) {
    __shared__ volatile int slot;
    if (threadIdx.x == 0)
        slot = 1;              // one writer ...
    out[threadIdx.x] = slot;   // ... read by the other warp with no __syncthreads in between
}

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

int devOob() {
    unsigned char* flag = nullptr;
    if (!ok(cudaMalloc(&flag, 1), "cudaMalloc")) return 1;
    cudaMemset(flag, 0, 1);
    widenedOr<<<1, 1>>>(flag);
    bool good = ok(cudaDeviceSynchronize(), "widenedOr");
    cudaFree(flag);
    return good ? 0 : 1;
}

int devRace() {
    int* out = nullptr;
    if (!ok(cudaMalloc(&out, 64 * sizeof(int)), "cudaMalloc")) return 1;
    sharedWriteWrite<<<1, 64>>>(out);
    bool good = ok(cudaDeviceSynchronize(), "sharedWriteWrite");
    cudaFree(out);
    return good ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s dev-oob|dev-race\n", argv[0]);
        return 64;
    }
    if (!std::strcmp(argv[1], "dev-oob"))  return devOob();
    if (!std::strcmp(argv[1], "dev-race")) return devRace();
    std::fprintf(stderr, "unknown mode '%s'\n", argv[1]);
    return 64;
}
