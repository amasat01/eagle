// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The DLPack stream contract of plugin/interop.h on a real device.
//
// Adversarial schedule: a slow kernel on non-blocking stream A spins, then
// writes a sentinel; the consumer reads the buffer with an asynchronous copy
// on non-blocking stream B. Nothing but the layer's fence orders the two.
// The RED twin runs the identical schedule without the fence and must see
// the stale value, which proves the schedule can bite at all.

#include <cstdint>
#include <optional>
#include <string>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "plugin/interop.h"

namespace dlpack_stream_tests {

constexpr double kSentinel = 42.0;
// ~0.2 s on a 1.5 GHz part: far longer than a launch plus an 8-byte copy.
constexpr long long kSpinCycles = 300000000LL;

__global__ void slow_write(double* out, long long cycles, double value)
{
    const long long start = clock64();
    while (clock64() - start < cycles) { }
    *out = value;
}

struct Schedule {
    cudaStream_t a = nullptr;
    cudaStream_t b = nullptr;
    double* device = nullptr;
    double* host = nullptr;

    Schedule()
    {
        EXPECT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
        EXPECT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&device, sizeof(double)), cudaSuccess);
        EXPECT_EQ(cudaMallocHost(&host, sizeof(double)), cudaSuccess);
        EXPECT_EQ(cudaMemset(device, 0, sizeof(double)), cudaSuccess);
        *host = -1.0;
    }

    ~Schedule()
    {
        cudaStreamSynchronize(a);
        cudaStreamSynchronize(b);
        cudaFree(device);
        cudaFreeHost(host);
        cudaStreamDestroy(a);
        cudaStreamDestroy(b);
    }

    // An external buffer over `device`, ordered on stream A.
    eagle::interop::OrderedBuffer buffer() const
    {
        aether::interop::ArrayInterface ai;
        ai.data = device;
        ai.shape = { 1 };
        ai.typestr = "<f8";
        ai.device = DLDevice{ kDLCUDA, 0 };
        ai.producer = "test.Schedule";
        return eagle::interop::OrderedBuffer{ aether::interop::fromArrayInterface(ai),
            reinterpret_cast<std::intptr_t>(a) };
    }

    // Launch the slow writer on A, then the consumer's read on B. `between`
    // runs after the launch and before the read is queued.
    template<class F>
    double run(F&& between)
    {
        // The check below is about THIS launch: clear whatever an earlier
        // call on this thread left in the runtime's last-error slot first.
        (void)cudaGetLastError();
        slow_write<<<1, 1, 0, a>>>(device, kSpinCycles, kSentinel);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        between();
        EXPECT_EQ(cudaMemcpyAsync(host, device, sizeof(double), cudaMemcpyDeviceToHost, b), cudaSuccess);
        // Waits for B only: whatever B was ordered after is what B saw.
        EXPECT_EQ(cudaStreamSynchronize(b), cudaSuccess);
        return *host;
    }
};

std::intptr_t handle(cudaStream_t s) { return reinterpret_cast<std::intptr_t>(s); }

TEST(DLPackStream, ExportFenceOrdersTheConsumerAfterTheProducer)
{
    for (int round = 0; round < 3; ++round) {
        Schedule s;
        const auto buf = s.buffer();
        const double seen = s.run([&] {
            DLManagedTensorVersioned* t = eagle::interop::export_versioned(buf, handle(s.b));
            t->deleter(t);
        });
        EXPECT_EQ(seen, kSentinel) << "round " << round;
    }
}

TEST(DLPackStream, RedTwinWithoutTheFenceReadsTheStaleValue)
{
    Schedule s;
    const double seen = s.run([] {});
    EXPECT_EQ(seen, 0.0) << "the schedule must bite: without a fence B reads before A writes";
}

TEST(DLPackStream, NoOrderingCodeLeavesTheReadUnordered)
{
    Schedule s;
    const auto buf = s.buffer();
    const std::size_t before = eagle::interop::EventPool::instance().created();
    const double seen = s.run([&] {
        DLManagedTensorVersioned* t = eagle::interop::export_versioned(buf, eagle::interop::kStreamNoOrder);
        t->deleter(t);
    });
    EXPECT_EQ(seen, 0.0) << "-1 performs no ordering";
    EXPECT_EQ(eagle::interop::EventPool::instance().created(), before);
}

TEST(DLPackStream, RefenceOrdersALaterLaunch)
{
    Schedule s;
    const auto buf = s.buffer();
    const double seen = s.run([&] { eagle::interop::refence(buf, handle(s.b)); });
    EXPECT_EQ(seen, kSentinel);
}

TEST(DLPackStream, StreamZeroIsRefused)
{
    Schedule s;
    const auto buf = s.buffer();
    try {
        (void)eagle::interop::export_versioned(buf, 0);
        FAIL() << "stream 0 must be refused";
    } catch (const eagle::interop::InteropError& e) {
        EXPECT_NE(std::string(e.what()).find("stream: 0 is ambiguous"), std::string::npos);
    }
}

TEST(DLPackStream, LegacyAndPerThreadCodesAreAccepted)
{
    Schedule s;
    const auto buf = s.buffer();
    for (std::intptr_t code : { eagle::interop::kStreamLegacy, eagle::interop::kStreamPerThread }) {
        DLManagedTensorVersioned* t = eagle::interop::export_versioned(buf, code);
        EXPECT_EQ(t->dl_tensor.data, s.device);
        t->deleter(t);
    }
    DLManagedTensorVersioned* t = eagle::interop::export_versioned(buf, std::nullopt);
    t->deleter(t);
}

TEST(DLPackStream, FencesReuseEventsFromThePool)
{
    Schedule s;
    const auto buf = s.buffer();
    eagle::interop::refence(buf, handle(s.b));
    const std::size_t before = eagle::interop::EventPool::instance().created();
    for (int i = 0; i < 100; ++i)
        eagle::interop::refence(buf, handle(s.b));
    EXPECT_EQ(eagle::interop::EventPool::instance().created(), before) << "no event created per call";
}

TEST(DLPackStream, CpuStreamArgumentIsRefusedForACudaRequirement)
{
    Schedule s;
    const auto buf = s.buffer();
    aether::interop::Requirements r;
    r.deviceType = kDLCPU;
    const auto out = aether::interop::refusals(buf.buffer, r);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], "device: required cpu, got cuda");
}

} // namespace dlpack_stream_tests
