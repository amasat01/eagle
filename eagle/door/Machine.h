// SPDX-License-Identifier: Apache-2.0

// The live machine as the dispatch rule's weights need it: the class that picks a weights row (CPUID vendor, display family and display
// model of the host), the physical cores and the cache a gathered plane may sit in. Read once, at eg_ctx_new.
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace eagle {
namespace door {

/** @brief The class of the host: vendor 1 Intel, 2 AMD, 0 any other (or a non-x86 host, whose rows are the built-in ones). */
struct HostClass {
    int vendor = 0, family = 0, model = 0;
};

inline HostClass host_class() {
    HostClass c;
#if defined(__x86_64__) || defined(__i386__)
    unsigned a = 0, b = 0, cx = 0, d = 0;
    if (!__get_cpuid(0, &a, &b, &cx, &d)) return c;
    char name[13] = {};
    __builtin_memcpy(name, &b, 4);
    __builtin_memcpy(name + 4, &d, 4);
    __builtin_memcpy(name + 8, &cx, 4);
    const std::string vendor = name;
    c.vendor = vendor == "GenuineIntel" ? 1 : vendor == "AuthenticAMD" ? 2 : 0;
    if (!__get_cpuid(1, &a, &b, &cx, &d)) return c;
    const int base_family = static_cast<int>((a >> 8) & 0xF);
    c.family = base_family == 0xF ? base_family + static_cast<int>((a >> 20) & 0xFF) : base_family;
    c.model = static_cast<int>((a >> 4) & 0xF);
    if (base_family == 0x6 || base_family == 0xF) c.model += static_cast<int>((a >> 16) & 0xF) << 4;
#endif
    return c;
}

inline std::string read_line(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

/** @brief Physical cores: the distinct thread-sibling sets of the online CPUs (sysfs); the logical CPU count when sysfs does not say. */
inline int physical_cores() {
    std::set<std::string> cores;
    const unsigned logical = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned cpu = 0; cpu < logical; ++cpu) {
        const std::string siblings = read_line("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        if (!siblings.empty()) cores.insert(siblings);
    }
    return cores.empty() ? static_cast<int>(logical) : static_cast<int>(cores.size());
}

/** @brief The size of a sysfs cache entry ("32K", "8448K", "1M") in bytes, 0 when unreadable. */
inline std::int64_t cache_size(const std::string& text) {
    if (text.empty()) return 0;
    const std::int64_t value = std::stoll(text);
    const char unit = text.back();
    return value * (unit == 'K' ? 1024 : unit == 'M' ? 1024 * 1024 : 1);
}

/** @brief The cache a gathered plane may sit in: the last level once, plus each lower data or unified level for every physical core (sysfs, cpu0's view); 0 when unreadable. */
inline double host_cache_bytes(int cores) {
    int last = 0;
    std::int64_t last_bytes = 0, per_core = 0;
    for (int i = 0; i < 16; ++i) {
        const std::string dir = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        const std::string level = read_line(dir + "level"), type = read_line(dir + "type");
        if (level.empty() || type == "Instruction") continue;
        const std::int64_t bytes = cache_size(read_line(dir + "size"));
        if (std::stoi(level) > last) last = std::stoi(level), last_bytes = bytes;
    }
    for (int i = 0; i < 16; ++i) {
        const std::string dir = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        const std::string level = read_line(dir + "level"), type = read_line(dir + "type");
        if (!level.empty() && type != "Instruction" && std::stoi(level) < last) per_core += cache_size(read_line(dir + "size"));
    }
    return static_cast<double>(last_bytes) + static_cast<double>(per_core) * cores;
}

}  // namespace door
}  // namespace eagle
