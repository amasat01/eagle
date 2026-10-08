// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Reorder.cu
 * @brief ``eagle::filtering::reorderDevice`` / ``restoreDevice`` and the
 *        device trigger, against the host face (ReorderHostCases.h, also run
 *        here).
 *
 * The device face matches the host face byte for byte on every plane, ``perm``,
 * ``inv``, the map and the words, over two reorders of a thinning mask and a
 * restore; the device trigger equals the host trigger (random and grouped
 * live sets); a captured reorder guarded by its ``fire`` word is a no-op on a
 * replay with ``fire == 0`` and reorders on a replay with ``fire == 1`` (RED: a
 * kernel that ignored the word would move the planes on the first replay); and
 * every device entry point ignores an error an earlier call left in the
 * runtime's last-error slot.
 */

#include <cuda_runtime_api.h>

#include "ReorderHostCases.h"

namespace eagle_tests {
namespace ReorderTest {

using eagle::filtering::compactDevice;
using eagle::filtering::reorderDevice;
using eagle::filtering::restoreDevice;

template<typename T>
T* upload(const std::vector<T>& v)
{
    T* d                = nullptr;
    const std::size_t m = v.empty() ? 1 : v.size();
    EXPECT_EQ(cudaMalloc(&d, m * sizeof(T)), cudaSuccess);
    if (!v.empty())
        EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

template<typename T>
void download(std::vector<T>& v, const T* d)
{
    if (!v.empty())
        EXPECT_EQ(cudaMemcpy(v.data(), d, v.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
}

/** @brief A HostBatch mirrored on the device. */
struct DeviceBatch {
    std::int64_t n;
    std::uint8_t* mask;
    std::uint16_t* p2;
    std::uint32_t* p4;
    double* p8;
    Elem16* p16;
    std::int32_t *perm, *inv, *map;
    std::uint32_t* words; // count, span, fire, live32
    void* scratch = nullptr;

    explicit DeviceBatch(const HostBatch& h)
        : n(h.n)
        , mask(upload(h.mask))
        , p2(upload(h.p2))
        , p4(upload(h.p4))
        , p8(upload(h.p8))
        , p16(upload(h.p16))
        , perm(upload(h.perm))
        , inv(upload(h.inv))
        , map(upload(h.map))
        , words(upload(std::vector<std::uint32_t> { h.count, h.span, h.fire, 0u }))
    {
        EXPECT_EQ(cudaMalloc(&scratch, reorderScratchBytes(n, 16)), cudaSuccess);
    }
    ~DeviceBatch()
    {
        for (void* p : { (void*)mask, (void*)p2, (void*)p4, (void*)p8, (void*)p16, (void*)perm,
                 (void*)inv, (void*)map, (void*)words, scratch })
            cudaFree(p);
    }
    std::array<ReorderPlane, 5> planes()
    {
        return { ReorderPlane { p2, 2 }, ReorderPlane { mask, 1 }, ReorderPlane { p4, 4 },
            ReorderPlane { p8, 8 }, ReorderPlane { p16, 16 } };
    }
    void reorder(cudaStream_t s, bool guarded = true)
    {
        auto pl = planes();
        reorderDevice(pl.data(), std::uint32_t(pl.size()), mask, kCompactMaskIsDrop, perm, inv, map,
            words, words + 1, guarded ? words + 2 : nullptr, scratch, n, s);
    }
    // Each copy rides the per-thread default stream, which has no ordering
    // with a replay on another stream: finish it before returning.
    void setMask(const std::vector<std::uint8_t>& m)
    {
        EXPECT_EQ(cudaMemcpy(mask, m.data(), m.size(), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    void setWord(int w, std::uint32_t v)
    {
        EXPECT_EQ(cudaMemcpy(words + w, &v, 4, cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    /** Copy everything back into a HostBatch for comparison. */
    HostBatch read() const
    {
        HostBatch h(n);
        download(h.mask, mask);
        download(h.p2, p2);
        download(h.p4, p4);
        download(h.p8, p8);
        download(h.p16, p16);
        download(h.perm, perm);
        download(h.inv, inv);
        download(h.map, map);
        std::vector<std::uint32_t> w(4);
        download(w, words);
        h.count = w[0];
        h.span  = w[1];
        h.fire  = w[2];
        return h;
    }
};

void expectSame(const HostBatch& a, const HostBatch& b)
{
    ASSERT_EQ(a.count, b.count);
    ASSERT_EQ(a.span, b.span);
    ASSERT_EQ(a.fire, b.fire);
    ASSERT_EQ(a.mask, b.mask);
    ASSERT_EQ(a.p2, b.p2);
    ASSERT_EQ(a.p4, b.p4);
    ASSERT_EQ(std::memcmp(a.p8.data(), b.p8.data(), a.p8.size() * 8), 0);
    ASSERT_EQ(std::memcmp(a.p16.data(), b.p16.data(), a.p16.size() * 16), 0);
    ASSERT_EQ(a.perm, b.perm);
    ASSERT_EQ(a.inv, b.inv);
    for (std::uint32_t t = 0; t < a.count; ++t)
        ASSERT_EQ(a.map[t], b.map[t]);
}

class ReorderDeviceSizes : public Test, public ::testing::WithParamInterface<std::int64_t> { };

TEST_P(ReorderDeviceSizes, DeviceMatchesHostByteForByte)
{
    const std::int64_t n = GetParam();
    HostBatch h(n);
    DeviceBatch d(h);
    const auto drop1 = randomDrop(n, 0.5, 21u + unsigned(n));
    h.finish(drop1);
    d.setMask(h.mask);
    h.reorder();
    d.reorder(0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expectSame(d.read(), h);
    auto drop2 = randomDrop(n, 0.5, 23u + unsigned(n));
    for (std::size_t i = 0; i < drop2.size(); ++i)
        drop2[i] = std::uint8_t(drop2[i] | drop1[i]);
    h.finish(drop2);
    d.setMask(h.mask);
    h.fire = 1;
    d.setWord(2, 1u);
    h.reorder();
    d.reorder(0, false); // unconditional form
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    h.fire = 0; // the unconditional call leaves the (unread) word alone
    d.setWord(2, 0u);
    expectSame(d.read(), h);
    h.restore();
    auto pl = d.planes();
    restoreDevice(pl.data(), std::uint32_t(pl.size()), d.perm, d.inv, d.scratch, n, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expectSame(d.read(), h);
}

INSTANTIATE_TEST_SUITE_P(Edges, ReorderDeviceSizes,
    ::testing::Values(std::int64_t { 0 }, std::int64_t { 1 }, std::int64_t { 7 },
        std::int64_t { 256 }, std::int64_t { 257 }, std::int64_t { 100003 }));

class ReorderDevice : public Test { };

TEST_F(ReorderDevice, TriggerMatchesTheHostFace)
{
    const std::int64_t n = 100000;
    std::vector<std::uint8_t> grouped(std::size_t(n), 1);
    for (std::int64_t i = 0; i < n / 5; ++i)
        grouped[std::size_t(i)] = 0;
    for (const auto& m : { randomDrop(n, 0.8, 5u), grouped, randomDrop(n, 0.3, 6u) }) {
        for (const float theta : { 0.25f, 0.5f, 0.75f }) {
            std::uint32_t hostLive = 0;
            const std::uint32_t want = hostFire(m, std::uint32_t(n), theta, &hostLive);
            std::uint8_t* dm         = upload(m);
            std::int32_t* dmap       = upload(std::vector<std::int32_t>(std::size_t(n)));
            std::uint32_t* w         = upload(std::vector<std::uint32_t> { 0u, std::uint32_t(n), 7u, 99u });
            void* sc                 = nullptr;
            ASSERT_EQ(cudaMalloc(&sc, compactScratchBytes(n)), cudaSuccess);
            const ReorderTrigger trig { w + 3, w + 1, w + 2, theta };
            compactDevice(dm, kCompactMaskIsDrop, dmap, w, sc, n, 0, &trig);
            std::vector<std::uint32_t> got(4);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            download(got, w);
            EXPECT_EQ(got[3], hostLive) << "theta " << theta;
            EXPECT_EQ(got[2], want) << "theta " << theta;
            cudaFree(dm);
            cudaFree(dmap);
            cudaFree(w);
            cudaFree(sc);
        }
    }
}

TEST_F(ReorderDevice, CapturedReorderObeysItsFireWord)
{
    const std::int64_t n = 5000;
    HostBatch h(n);
    DeviceBatch d(h);
    const auto drop = randomDrop(n, 0.7, 31u);
    h.finish(drop);
    d.setMask(h.mask);
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    d.reorder(s);
    cudaGraph_t g = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&exec, g, 0), cudaSuccess);
    d.setWord(2, 0u);
    ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    HostBatch untouched(n);
    untouched.finish(drop);
    untouched.fire = 0;
    expectSame(d.read(), untouched);
    d.setWord(2, 1u);
    ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    h.reorder();
    expectSame(d.read(), h);
    expectReordered(d.read(), drop);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
}

/** @brief Leave the runtime's last-error slot set, as an earlier failed call
 *         (already reported, or caught by its caller) does. */
void poisonLastError()
{
    ASSERT_NE(cudaSetDevice(-1), cudaSuccess);
    ASSERT_NE(cudaPeekAtLastError(), cudaSuccess) << "the slot must be set for this row to mean anything";
}

// Every device entry point checks its own launches with cudaGetLastError(), so
// it must not report an error some unrelated earlier call left in the slot (RED
// before the entry points cleared it: cudaErrorInvalidDevice thrown from the
// first post-launch check, misattributed to the launch).
TEST_F(ReorderDevice, EntryPointsIgnoreAStaleRuntimeError)
{
    const std::int64_t n = 257;
    const auto drop      = randomDrop(n, 0.5, 41u);
    std::uint32_t kept   = 0;
    for (const auto b : drop)
        kept += b == 0 ? 1u : 0u;
    std::uint8_t* dm   = upload(drop);
    std::int32_t* dmap = upload(std::vector<std::int32_t>(std::size_t(n)));
    std::uint32_t* w   = upload(std::vector<std::uint32_t> { 0u, std::uint32_t(n), 0u, 0u });
    void* sc           = nullptr;
    ASSERT_EQ(cudaMalloc(&sc, compactScratchBytes(n)), cudaSuccess);
    const ReorderTrigger trig { w + 3, w + 1, w + 2, 0.5f };
    poisonLastError();
    EXPECT_NO_THROW(compactDevice(dm, kCompactMaskIsDrop, dmap, w, sc, n, 0));
    poisonLastError();
    EXPECT_NO_THROW(compactDevice(dm, kCompactMaskIsDrop, dmap, w, sc, n, 0, &trig));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<std::uint32_t> got(4);
    download(got, w);
    EXPECT_EQ(got[0], kept);
    cudaFree(dm);
    cudaFree(dmap);
    cudaFree(w);
    cudaFree(sc);

    HostBatch h(n);
    DeviceBatch d(h);
    h.finish(drop);
    d.setMask(h.mask);
    poisonLastError();
    EXPECT_NO_THROW(d.reorder(0));
    h.reorder();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expectSame(d.read(), h);
    h.restore();
    auto pl = d.planes();
    poisonLastError();
    EXPECT_NO_THROW(restoreDevice(pl.data(), std::uint32_t(pl.size()), d.perm, d.inv, d.scratch, n, 0));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expectSame(d.read(), h);
    cudaGetLastError(); // leave the slot clean whatever happened above
}

} // namespace ReorderTest
} // namespace eagle_tests
