// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The host arm of the generic DLPack layer (plugin/interop.h) and the
// generalised plugin bridge (plugin/dlpack_bridge.h). CUDA-free: compiled in
// BOTH build modes with EAGLE_CPU_ONLY forced (see tests/CMakeLists.txt). The
// stream contract on a real device is test_DLPackStream.cu.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "plugin/dlpack_bridge.h"
#include "plugin/interop.h"

namespace {

using eagle::interop::Access;
using eagle::interop::InteropError;
using eagle::interop::OrderedBuffer;

struct HostProducer {
    std::vector<double> storage = std::vector<double>(6, 1.5);
    std::int64_t shape[2] = { 2, 3 };
    int deletes = 0;
    DLManagedTensorVersioned versioned{};
    DLManagedTensor legacy{};

    DLManagedTensorVersioned* make(bool readOnly)
    {
        versioned = DLManagedTensorVersioned{};
        versioned.version = DLPackVersion{ 1, 0 };
        versioned.manager_ctx = this;
        versioned.deleter = [](DLManagedTensorVersioned* s) { ++static_cast<HostProducer*>(s->manager_ctx)->deletes; };
        versioned.flags = readOnly ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
        fill(versioned.dl_tensor);
        return &versioned;
    }

    DLManagedTensor* makeLegacy()
    {
        legacy = DLManagedTensor{};
        legacy.manager_ctx = this;
        legacy.deleter = [](DLManagedTensor* s) { ++static_cast<HostProducer*>(s->manager_ctx)->deletes; };
        fill(legacy.dl_tensor);
        return &legacy;
    }

private:
    void fill(DLTensor& t)
    {
        t.data = storage.data();
        t.device = DLDevice{ kDLCPU, 0 };
        t.ndim = 2;
        t.dtype = DLDataType{ kDLFloat, 64, 1 };
        t.shape = shape;
        t.strides = nullptr;
        t.byte_offset = 0;
    }
};

template<class F>
std::string message_of(F&& fn)
{
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

TEST(DLPackInteropHost, StreamCodesFollowDLPack)
{
    EXPECT_TRUE(eagle::interop::resolve_stream(std::nullopt).order);
    EXPECT_EQ(eagle::interop::resolve_stream(std::nullopt).handle, eagle::interop::kStreamLegacy);
    EXPECT_EQ(eagle::interop::resolve_stream(2).handle, eagle::interop::kStreamPerThread);
    EXPECT_FALSE(eagle::interop::resolve_stream(-1).order) << "-1 asks for no ordering";
    EXPECT_NE(message_of([] { (void)eagle::interop::resolve_stream(0); }).find("stream: 0 is ambiguous"),
        std::string::npos);
    EXPECT_NE(message_of([] { (void)eagle::interop::resolve_stream(-7); }).find("stream: required"),
        std::string::npos);
}

TEST(DLPackInteropHost, CpuBufferExportsWithoutStreamAndRefusesOne)
{
    HostProducer p;
    {
        OrderedBuffer b{ aether::interop::importDLPack(p.make(false), "test.Host"), std::nullopt };
        EXPECT_EQ(b.buffer.access, Access::ReadWrite);
        DLManagedTensorVersioned* out = eagle::interop::export_versioned(b, std::nullopt);
        EXPECT_EQ(out->dl_tensor.data, p.storage.data());
        EXPECT_EQ(out->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0u);
        out->deleter(out);
        const std::string msg = message_of([&] { (void)eagle::interop::export_versioned(b, 1); });
        EXPECT_NE(msg.find("stream: a CPU buffer takes no stream, got 1"), std::string::npos) << msg;
    }
    EXPECT_EQ(p.deletes, 1);
}

TEST(DLPackInteropHost, ReadOnlyCpuBufferKeepsItsFlagThroughExport)
{
    HostProducer p;
    OrderedBuffer b{ aether::interop::importDLPack(p.make(true)), std::nullopt };
    DLManagedTensorVersioned* out = eagle::interop::export_versioned(b, std::nullopt);
    EXPECT_NE(out->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0u);
    out->deleter(out);
    EXPECT_NE(message_of([&] { (void)eagle::interop::export_legacy(b, std::nullopt); }).find("read-only"),
        std::string::npos);
}

TEST(DLPackInteropHost, CudaRequirementRefusesACpuBuffer)
{
    HostProducer p;
    aether::interop::BufferView b = aether::interop::importDLPack(p.make(false));
    aether::interop::Requirements r;
    r.deviceType = kDLCUDA;
    const auto out = aether::interop::refusals(b, r);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], "device: required cuda, got cpu");
}

TEST(DLPackInteropHost, CudaBufferIsRefusedByACpuOnlyBuild)
{
    aether::interop::BufferView b;
    b.view.device = aether::Device(DLDevice{ kDLCUDA, 0 });
    OrderedBuffer ob{ b, 1 };
    EXPECT_NE(message_of([&] { eagle::interop::refence(ob, 1); }).find("CPU-only"), std::string::npos);
}

TEST(DLPackInteropHost, GrefFromBufferValidatesThroughTheLayer)
{
    HostProducer p;
    aether::interop::BufferView b = aether::interop::importDLPack(p.make(true));
    const eagle::plugin::GRefMirror g = eagle::interop::gref_from_buffer(b, 2, false);
    EXPECT_EQ(g.data_, p.storage.data());
    EXPECT_EQ(g.samples_, 3u);
    EXPECT_EQ(g.deviceType_, eagle::plugin::kEagleAbiDeviceCPU);
    const std::string msg = message_of([&] { (void)eagle::interop::gref_from_buffer(b, 2, true); });
    EXPECT_NE(msg.find("access: required read-write, got read-only"), std::string::npos) << msg;
    EXPECT_NE(message_of([&] { (void)eagle::interop::gref_from_buffer(b, 3, false); }).find("shape: required 3"),
        std::string::npos);
}

TEST(DLPackInteropHost, BridgeAcceptsBothGenerationsOnTheHost)
{
    HostProducer p;
    DLManagedTensorVersioned* v = p.make(true);
    EXPECT_EQ(eagle::plugin::access_of(v), eagle::plugin::BridgeAccess::ReadOnly);
    eagle::plugin::GRefMirror g = eagle::plugin::gref_from_dlpack(v);
    EXPECT_EQ(g.deviceType_, eagle::plugin::kEagleAbiDeviceCPU);
    DLManagedTensor* l = p.makeLegacy();
    EXPECT_EQ(eagle::plugin::access_of(l), eagle::plugin::BridgeAccess::Unknown);
    EXPECT_EQ(eagle::plugin::gref_from_dlpack(l).data_, p.storage.data());

    std::int64_t shape[2];
    DLManagedTensorVersioned ro = eagle::plugin::dlpack_from_gref(g, 2, shape, true);
    EXPECT_NE(ro.flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0u);
    EXPECT_EQ(ro.dl_tensor.device.device_type, kDLCPU);
    DLManagedTensorVersioned rw = eagle::plugin::dlpack_from_gref(g, 2, shape);
    EXPECT_EQ(rw.flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0u);

    p.versioned.dl_tensor.dtype.bits = 32;
    EXPECT_NE(message_of([&] { (void)eagle::plugin::gref_from_dlpack(v); }).find("dtype: required float64, got float32"),
        std::string::npos);
    p.versioned.dl_tensor.dtype.bits = 64;
    std::int64_t strides[2] = { 6, 2 };
    p.versioned.dl_tensor.strides = strides;
    EXPECT_NE(message_of([&] { (void)eagle::plugin::gref_from_dlpack(v); }).find("stride: required C-contiguous"),
        std::string::npos);
}

} // namespace
