// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Test-only stand-ins for the two launch entries libeagle_xla.so calls, which
// RECORD the by-value argument PODs they are handed instead of running a
// kernel. python/tests/test_frameworks_jax.py builds this file with the host
// C++ compiler at test time and compares the recorded bytes with what eagle's
// own packer (eagle._plan_pack) builds for the same planes.
//
//   probe_host       the host entry shape: void(void* const*, int64 base, int64 count, int64 n)
//   probe_run_device the seam's eagle_backend_run_device shape
//   probe_read       copy the last record out: nslots, then 40 bytes per slot, then base, count, n

#include <cstdint>
#include <cstring>
#include <mutex>

namespace {

constexpr int kMaxSlots = 64;
constexpr int kBox = 40;

std::mutex g_mutex;
std::int32_t g_nslots = 0;
unsigned char g_boxes[kMaxSlots * kBox];
std::int64_t g_triple[3];

void record(void* const* params, std::int32_t nslots, std::int64_t base, std::int64_t count, std::int64_t n)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_nslots = nslots < kMaxSlots ? nslots : kMaxSlots;
    for (std::int32_t s = 0; s < g_nslots; ++s)
        std::memcpy(g_boxes + s * kBox, params[s], kBox);
    g_triple[0] = base;
    g_triple[1] = count;
    g_triple[2] = n;
}

struct LaunchDesc {
    std::uint32_t struct_size, flags;
    std::int32_t device;
    std::uint32_t block;
    std::uint64_t function;
    const std::uint64_t* params;
    const std::uint64_t* param_sizes;
    std::int64_t nparams;
    std::int64_t base, count, n_samples;
    std::uint64_t stream;
};

std::int32_t g_slots_for_host = 0;

} // namespace

extern "C" __attribute__((visibility("default"))) void probe_set_slots(std::int32_t nslots)
{
    g_slots_for_host = nslots;
}

extern "C" __attribute__((visibility("default"))) void probe_host(void* const* params, std::int64_t base,
    std::int64_t count, std::int64_t n)
{
    record(params, g_slots_for_host, base, count, n);
}

extern "C" __attribute__((visibility("default"))) std::int32_t probe_run_device(const LaunchDesc* d,
    std::int32_t* launched)
{
    void* params[kMaxSlots];
    const std::int32_t nslots = static_cast<std::int32_t>(d->nparams < kMaxSlots ? d->nparams : kMaxSlots);
    for (std::int32_t s = 0; s < nslots; ++s)
        params[s] = reinterpret_cast<void*>(d->params[s]);
    record(params, nslots, d->base, d->count, d->n_samples);
    *launched = 1;
    return 0;
}

extern "C" __attribute__((visibility("default"))) std::int32_t probe_read(unsigned char* out, std::int32_t cap)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::int32_t need = 4 + g_nslots * kBox + 24;
    if (cap < need)
        return -need;
    std::memcpy(out, &g_nslots, 4);
    std::memcpy(out + 4, g_boxes, static_cast<std::size_t>(g_nslots) * kBox);
    std::memcpy(out + 4 + g_nslots * kBox, g_triple, 24);
    return need;
}
