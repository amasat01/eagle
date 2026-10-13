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

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "ReorderHostCases.h"

namespace eagle_tests {
namespace ReorderTest {

using eagle::filtering::compactDevice;
using eagle::filtering::reorderDevice;
using eagle::filtering::resetDevice;
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
        EXPECT_EQ(cudaMemcpy(v.data(), d, v.size() * sizeof(T), cudaMemcpyDefault), cudaSuccess);
}

/** @brief Where the 4-, 8- and 16-byte planes of a DeviceBatch live. */
enum class Mem { Device, HostAlloc, HostRegister };

/** @brief A HostBatch mirrored on the device. */
struct DeviceBatch {
    Mem mem;
    std::vector<std::pair<void*, Mem>> pinned; // host buffers to release
    std::int64_t n;
    std::uint8_t* mask;
    std::uint16_t* p2;
    std::uint32_t* p4;
    double* p8;
    Elem16* p16;
    std::int32_t *perm, *inv, *map;
    std::uint32_t* words; // count, span, fire, live32
    void* scratch = nullptr;

    /** The plane copy of @p v: device memory, a mapped cudaHostAlloc buffer, or a
     *  registered (mapped) malloc buffer; always returned as the device address. */
    template<typename T>
    T* place(const std::vector<T>& v)
    {
        if (mem == Mem::Device)
            return upload(v);
        const std::size_t bytes = std::max<std::size_t>(v.size(), 1) * sizeof(T);
        void* host              = nullptr;
        if (mem == Mem::HostAlloc) {
            EXPECT_EQ(cudaHostAlloc(&host, bytes, cudaHostAllocMapped), cudaSuccess);
        } else {
            host = std::malloc(bytes);
            EXPECT_EQ(cudaHostRegister(host, bytes, cudaHostRegisterMapped), cudaSuccess);
        }
        pinned.emplace_back(host, mem);
        std::memcpy(host, v.data(), v.size() * sizeof(T));
        void* dev = nullptr;
        EXPECT_EQ(cudaHostGetDevicePointer(&dev, host, 0), cudaSuccess);
        return static_cast<T*>(dev);
    }

    explicit DeviceBatch(const HostBatch& h, Mem where = Mem::Device)
        : mem(where)
        , n(h.n)
        , mask(upload(h.mask))
        , p2(upload(h.p2))
        , p4(place(h.p4))
        , p8(place(h.p8))
        , p16(place(h.p16))
        , perm(upload(h.perm))
        , inv(upload(h.inv))
        , map(upload(h.map))
        , words(upload(std::vector<std::uint32_t> { h.count, h.span, h.fire, 0u }))
    {
        EXPECT_EQ(cudaMalloc(&scratch, reorderScratchBytes(n, 16)), cudaSuccess);
    }
    ~DeviceBatch()
    {
        for (void* p : { (void*)mask, (void*)p2, (void*)perm, (void*)inv, (void*)map, (void*)words, scratch })
            cudaFree(p);
        if (mem == Mem::Device)
            for (void* p : { (void*)p4, (void*)p8, (void*)p16 })
                cudaFree(p);
        for (auto& [host, kind] : pinned) {
            if (kind == Mem::HostAlloc) {
                cudaFreeHost(host);
            } else {
                cudaHostUnregister(host);
                std::free(host);
            }
        }
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

void expectDeviceMatchesHost(std::int64_t n, Mem mem)
{
    HostBatch h(n);
    DeviceBatch d(h, mem);
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

TEST_P(ReorderDeviceSizes, DeviceMatchesHostByteForByte)
{
    expectDeviceMatchesHost(GetParam(), Mem::Device);
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

/** @brief The kernel nodes of the work ``reorderDevice`` does before its first
 *  plane group: the keep predicate and the inclusive scan. */
std::size_t fixedReorderNodes(std::int64_t n)
{
    using eagle::idx_t;
    using eagle::spanView;
    using eagle::deviceDevice;
    std::uint8_t* mask = nullptr;
    idx_t* scr         = nullptr;
    cudaMalloc(&mask, std::size_t(n));
    cudaMalloc(&scr, 3 * std::size_t(n) * sizeof(idx_t));
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    auto keep      = spanView<idx_t>(scr, idx_t(n), deviceDevice());
    auto inclusive = spanView<idx_t>(scr + n, idx_t(n), deviceDevice());
    auto sums      = spanView<idx_t>(scr + 2 * n, idx_t(n), deviceDevice());
    eagle::filtering::detail::compactPredicate<true><<<unsigned((n + 255) / 256), 256, 0, s>>>(mask, keep);
    eagle::cuda::Scan::enqueue<idx_t, aether::SumOp<idx_t>, true>(
        keep.as_const(), inclusive, sums, idx_t(0), s);
    cudaGraph_t g = nullptr;
    cudaStreamEndCapture(s, &g);
    std::size_t nodes = 0;
    cudaGraphGetNodes(g, nullptr, &nodes);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
    cudaFree(mask);
    cudaFree(scr);
    return nodes;
}

/** @brief A captured reorder of @p nPlanes 8-byte planes staging @p stage bytes
 *  per sample: the graph it produced. */
struct ManyPlanes {
    std::int64_t n;
    std::vector<void*> bufs;
    std::vector<eagle::filtering::ReorderPlane> planes;
    std::uint8_t* mask;
    std::int32_t *perm, *inv, *map;
    std::uint32_t* words;
    void* scratch = nullptr;
    cudaGraph_t graph = nullptr;

    ManyPlanes(std::int64_t n_, int nPlanes, std::uint32_t stage)
        : n(n_)
    {
        for (int p = 0; p < nPlanes; ++p) {
            void* d = nullptr;
            cudaMalloc(&d, std::size_t(n) * 8);
            bufs.push_back(d);
            planes.push_back({ d, 8 });
        }
        mask  = upload(std::vector<std::uint8_t>(std::size_t(n), 0));
        perm  = upload(std::vector<std::int32_t>(std::size_t(n)));
        inv   = upload(std::vector<std::int32_t>(std::size_t(n)));
        map   = upload(std::vector<std::int32_t>(std::size_t(n)));
        words = upload(std::vector<std::uint32_t> { 0u, std::uint32_t(n), 1u, 0u });
        planes.push_back({ mask, 1 });
        cudaMalloc(&scratch, eagle::filtering::reorderScratchBytes(n, planes.data(), std::uint32_t(planes.size()), stage));
        cudaStream_t s = nullptr;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        reorderDevice(planes.data(), std::uint32_t(planes.size()), mask, kCompactMaskIsDrop, perm, inv, map,
            words, words + 1, words + 2, scratch, n, s, stage);
        cudaStreamEndCapture(s, &graph);
        cudaStreamDestroy(s);
    }
    ~ManyPlanes()
    {
        cudaGraphDestroy(graph);
        for (void* p : bufs)
            cudaFree(p);
        for (void* p : { (void*)mask, (void*)perm, (void*)inv, (void*)map, (void*)words, scratch })
            cudaFree(p);
    }
    std::size_t nodes() const
    {
        std::size_t k = 0;
        cudaGraphGetNodes(graph, nullptr, &k);
        return k;
    }
};

// 40 planes of 8 bytes, the 1-byte mask and the 4-byte perm: 325 bytes per sample
// staged 128 at a time is three groups, so six group kernels and the words kernel.
TEST_F(ReorderDevice, GroupsLaunchTwoKernelsPerGroup)
{
    const std::int64_t n = 4096;
    ManyPlanes grouped(n, 40, 128);
    EXPECT_EQ(grouped.nodes() - fixedReorderNodes(n), std::size_t(2 * 3 + 1));
    ManyPlanes whole(n, 40, 512); // 332 bytes per sample fit one group
    EXPECT_EQ(whole.nodes() - fixedReorderNodes(n), std::size_t(2 * 1 + 1));
}

TEST_F(ReorderDevice, GridIsBoundedBySms)
{
    int dev = 0, sms = 0;
    ASSERT_EQ(cudaGetDevice(&dev), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), cudaSuccess);
    const std::int64_t n = std::int64_t(sms) * 4 * 256 * 8; // eight times the cap in blocks
    ManyPlanes m(n, 3, 64);
    std::size_t count = 0;
    ASSERT_EQ(cudaGraphGetNodes(m.graph, nullptr, &count), cudaSuccess);
    std::vector<cudaGraphNode_t> nodes(count);
    ASSERT_EQ(cudaGraphGetNodes(m.graph, nodes.data(), &count), cudaSuccess);
    unsigned atCap = 0;
    for (auto node : nodes) {
        cudaGraphNodeType t;
        ASSERT_EQ(cudaGraphNodeGetType(node, &t), cudaSuccess);
        if (t != cudaGraphNodeTypeKernel)
            continue;
        cudaKernelNodeParams kp {};
        ASSERT_EQ(cudaGraphKernelNodeGetParams(node, &kp), cudaSuccess);
        if (kp.gridDim.x == 4u * unsigned(sms))
            ++atCap;
    }
    // the predicate and scan size to n; the group kernels stop at four blocks per SM
    EXPECT_EQ(atCap, 2u); // the stage and place kernels of the single group
}

void expectCapturedReorderObeysItsFireWord(Mem mem)
{
    const std::int64_t n = 5000;
    HostBatch h(n);
    DeviceBatch d(h, mem);
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

TEST_F(ReorderDevice, CapturedReorderObeysItsFireWord)
{
    expectCapturedReorderObeysItsFireWord(Mem::Device);
}

// Planes in mapped pinned host memory: the kernels reach them over the bus and
// the result is byte-identical to the device-memory path.
TEST_F(ReorderDevice, MappedHostAllocPlanesMatchTheHostFace)
{
    for (const std::int64_t n : { std::int64_t { 257 }, std::int64_t { 100003 } })
        expectDeviceMatchesHost(n, Mem::HostAlloc);
}

TEST_F(ReorderDevice, RegisteredHostPlanesMatchTheHostFace)
{
    for (const std::int64_t n : { std::int64_t { 257 }, std::int64_t { 100003 } })
        expectDeviceMatchesHost(n, Mem::HostRegister);
}

TEST_F(ReorderDevice, CapturedReorderOnPinnedPlanesObeysItsFireWord)
{
    expectCapturedReorderObeysItsFireWord(Mem::HostAlloc);
    expectCapturedReorderObeysItsFireWord(Mem::HostRegister);
}

TEST_F(ReorderDevice, ResidencyClassifiesEveryKindOfPointer)
{
    using eagle::filtering::classifyPointer;
    using eagle::filtering::Residency;
    void *dev = nullptr, *managed = nullptr, *mapped = nullptr;
    ASSERT_EQ(cudaMalloc(&dev, 64), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&managed, 64), cudaSuccess);
    ASSERT_EQ(cudaHostAlloc(&mapped, 64, cudaHostAllocMapped), cudaSuccess);
    std::vector<char> pageable(64);
    EXPECT_EQ(classifyPointer(dev), Residency::Device);
    EXPECT_EQ(classifyPointer(managed), Residency::Managed);
    EXPECT_EQ(classifyPointer(mapped), Residency::MappedHost);
    EXPECT_EQ(classifyPointer(pageable.data()), Residency::Pageable);
    int cur = 0;
    ASSERT_EQ(cudaGetDevice(&cur), cudaSuccess);
    EXPECT_EQ(classifyPointer(dev, cur), Residency::Device);
    EXPECT_EQ(classifyPointer(dev, cur + 1), Residency::OtherDevice);
    cudaFree(dev);
    cudaFree(managed);
    cudaFreeHost(mapped);
}

TEST_F(ReorderDevice, PageablePlaneIsRefusedByIndex)
{
    using eagle::filtering::checkPlanesResident;
    const std::int64_t n = 64;
    HostBatch h(n);
    DeviceBatch d(h, Mem::HostAlloc);
    auto pl = d.planes();
    EXPECT_NO_THROW(checkPlanesResident(pl.data(), std::uint32_t(pl.size())));
    std::vector<double> unpinnedBuf(static_cast<std::size_t>(n), 0.0);
    pl[3] = ReorderPlane { unpinnedBuf.data(), 8 };
    try {
        checkPlanesResident(pl.data(), std::uint32_t(pl.size()));
        FAIL() << "a pageable plane must be refused";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("plane 3"), std::string::npos) << what;
        EXPECT_NE(what.find("pageable"), std::string::npos) << what;
    }
}

TEST_F(ReorderDevice, ResetGivesTheIdentityAndTheFullSpan)
{
    const std::int64_t n = 100003;
    for (const bool withMap : { true, false }) {
        HostBatch h(n);
        DeviceBatch d(h);
        h.finish(randomDrop(n, 0.37, 51u));
        d.setMask(h.mask);
        h.reorder();
        d.reorder(0, false);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const HostBatch dropped = d.read();
        ASSERT_LT(dropped.count, std::uint32_t(n)) << "the reorder must have dropped samples";
        resetDevice(d.perm, d.inv, withMap ? d.map : nullptr, d.words, d.words + 1, n, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const HostBatch got = d.read();
        EXPECT_EQ(got.count, std::uint32_t(n));
        EXPECT_EQ(got.span, std::uint32_t(n));
        EXPECT_EQ(got.fire, dropped.fire) << "no fire word is touched";
        for (std::int64_t i = 0; i < n; ++i) {
            const auto s = std::size_t(i);
            ASSERT_EQ(got.perm[s], std::int32_t(i));
            ASSERT_EQ(got.inv[s], std::int32_t(i));
            if (withMap)
                ASSERT_EQ(got.map[s], std::int32_t(i));
            else
                ASSERT_EQ(got.map[s], dropped.map[s]) << "a null map is left alone";
        }
    }
}

TEST_F(ReorderDevice, CapturedResetReplaysTheIdentity)
{
    const std::int64_t n = 5000;
    HostBatch h(n);
    DeviceBatch d(h);
    h.finish(randomDrop(n, 0.5, 61u));
    d.setMask(h.mask);
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    resetDevice(d.perm, d.inv, d.map, d.words, d.words + 1, n, s);
    cudaGraph_t g = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&exec, g, 0), cudaSuccess);
    for (int rep = 0; rep < 2; ++rep) {
        d.reorder(0, false); // scramble perm/inv/map/words, then replay the reset
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_LT(d.read().count, std::uint32_t(n));
        ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        const HostBatch got = d.read();
        EXPECT_EQ(got.count, std::uint32_t(n));
        EXPECT_EQ(got.span, std::uint32_t(n));
        for (std::int64_t i = 0; i < n; ++i) {
            const auto k = std::size_t(i);
            ASSERT_EQ(got.perm[k], std::int32_t(i));
            ASSERT_EQ(got.inv[k], std::int32_t(i));
            ASSERT_EQ(got.map[k], std::int32_t(i));
        }
    }
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
