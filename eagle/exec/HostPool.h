// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `eagle::exec::detail::HostPool` -- the host threads eagle owns, and the process-wide
// thread budget they follow.
//
// One pool per process. Its workers are plain `std::thread`s that spin for at most
// `HOST_SPIN_US` after a launch, then sleep in `std::atomic::wait`: back-to-back
// launches find them awake, and on a shared machine an idle worker gives its core
// back instead of holding it for an OS time slice.
//
// On Linux each worker is pinned to the sibling set of one physical core (the caller's core is left alone),
// so a team never doubles up on a core while another sits idle.
//
// A JOB is a list of lanes in phases. Lanes of one phase run in any order, on any
// thread, each lane as a whole on one thread; a phase starts when every lane of the
// phases before it is done. The caller runs lanes too. Lanes are handed out by one
// counter, so a late or descheduled worker simply takes fewer of them: the launch
// waits only for the lanes and for the workers that actually joined it.
//
// Safety rules (each one is a hang or a corrupted counter otherwise):
//  - the pool is created at the first team launch and never destroyed; its
//    workers are detached, so process exit never joins a thread;
//  - a launch in a forked child sees another pid and builds a fresh pool;
//  - one launch at a time: a launch that cannot take the pool (another thread
//    holds it, or it is issued from inside a lane or a worker) runs in place on its
//    own thread;
//  - the launch waits on the count of FINISHED lanes and of workers that LEFT,
//    never on the count of lanes handed out.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace eagle {
namespace exec {

namespace detail {

/** @brief Below this much serial work per lane (microseconds) a stage runs in place:
 *  waking a thread costs more than it saves. */
inline constexpr double HOST_FLOOR_US = 40.0;

/** @brief How long an idle worker or a waiting caller spins before it sleeps (microseconds). */
inline constexpr double HOST_SPIN_US = 30.0;

/** @brief The CPU cap of a cgroup v2 `cpu.max` line (`"<quota> <period>"`):
 *  `ceil(quota / period)`, or 0 when there is none (`"max ..."` or unreadable). */
inline int cgroup_cpu_cap(std::string_view cpuMax) {
    std::istringstream in{ std::string(cpuMax) };
    std::string quota;
    double period = 0;
    if (!(in >> quota >> period) || quota == "max" || period <= 0) return 0;
    char* end = nullptr;
    const double q = std::strtod(quota.c_str(), &end);
    if (end == quota.c_str() || q <= 0) return 0;
    return static_cast<int>(std::ceil(q / period));
}

/** @brief This process's cgroup v2 CPU cap (0 = none). */
inline int cgroup_cpu_cap() {
    std::ifstream self("/proc/self/cgroup");
    std::string line, path;
    while (std::getline(self, line))
        if (line.rfind("0::", 0) == 0) path = line.substr(3);
    if (path.empty()) return 0;
    std::ifstream max("/sys/fs/cgroup" + path + "/cpu.max");
    std::string text;
    return std::getline(max, text) ? cgroup_cpu_cap(text) : 0;
}

/** @brief The first value of `OMP_NUM_THREADS` when it is a positive integer, else 0. */
inline int omp_num_threads_env() {
    const char* env = std::getenv("OMP_NUM_THREADS");
    if (env == nullptr) return 0;
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    return (end != env && v > 0 && v < (1 << 15)) ? static_cast<int>(v) : 0;
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

#if defined(__linux__)
/**
 * @brief The CPUs this process may run on, one set per physical core: the affinity mask grouped by the
 *        sibling list the kernel publishes (sysfs), in order of each core's first CPU. Empty when that is
 *        unreadable.
 */
inline std::vector<cpu_set_t> core_masks() {
    cpu_set_t mask;
    std::vector<cpu_set_t> cores;
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return cores;
    std::vector<std::string> names;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                        "/topology/thread_siblings_list");
        std::string siblings;
        if (!std::getline(f, siblings)) return {};
        std::size_t k = 0;
        while (k < names.size() && names[k] != siblings) ++k;
        if (k == names.size()) {
            names.push_back(siblings);
            cores.emplace_back();
            CPU_ZERO(&cores.back());
        }
        CPU_SET(cpu, &cores[k]);
    }
    return cores;
}
#endif

/**
 * @brief The physical cores among the CPUs this process may run on (`core_masks`). Elsewhere, or when
 *        sysfs is unreadable, the logical CPU count.
 */
inline int physical_cores() {
    const int logical = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
#if defined(__linux__)
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return logical;
    const std::size_t cores = core_masks().size();
    return cores == 0 ? CPU_COUNT(&mask) : static_cast<int>(cores);
#else
    return logical;
#endif
}

}  // namespace detail

/** @brief The process-wide cap on host threads; 0 = unset (the default budget). One
 *  atomic for the whole process: it is read at every launch from any thread, so a
 *  launch on a Python worker thread honours it as the main thread does. */
inline std::atomic<int>& host_threads_setting() {
    static std::atomic<int> setting{0};
    return setting;
}

namespace detail {

/** @brief The default host budget: the physical cores of the affinity mask, capped by
 *  the cgroup CPU quota when there is one. Measured once per process. */
inline int host_default_threads() {
    static const int budget = [] {
        const int cores = physical_cores(), cap = detail::cgroup_cpu_cap();
        return cap > 0 ? std::min(cores, cap) : cores;
    }();
    return budget;
}

/** @brief This thread's own team for its host launches; 0 = unset. A run that sizes its
 *  team itself (`eagle.until_done` on the host) sets it for its duration; the process-wide
 *  setting still wins over it. */
inline int& host_thread_team() {
    static thread_local int team = 0;
    return team;
}

}  // namespace detail

/** @brief The most threads a host launch uses now: the process-wide setting, else this
 *  thread's team, else `OMP_NUM_THREADS` (read by eagle once), else the default budget. */
inline int host_max_threads() {
    const int set = host_threads_setting().load(std::memory_order_relaxed);
    if (set > 0) return set;
    if (detail::host_thread_team() > 0) return detail::host_thread_team();
    static const int env = detail::omp_num_threads_env();
    return env > 0 ? env : detail::host_default_threads();
}

namespace detail {

class HostPool {
public:
    /** @brief A job: `lanes` lanes; lane `i` may start once `gate[i]` lanes are done
     *  (its phase's first index; no `gate` = one phase); `run(ctx, i)` runs lane `i`. */
    struct Job {
        std::uint32_t lanes = 0;
        const std::uint32_t* gate = nullptr;
        void (*run)(const void* ctx, std::uint32_t lane) = nullptr;
        const void* ctx = nullptr;
    };

    /** @brief The pool of this process, created on first use (again in a forked child). */
    static HostPool& instance() {
        static std::atomic<HostPool*> current{ nullptr };
        HostPool* pool = current.load(std::memory_order_acquire);
        const long pid = self_pid();
        while (pool == nullptr || pool->owner_ != pid) {
            // A parent's pool in a forked child is leaked on purpose: its threads did not survive
            // the fork, and its memory may be mid-launch. The new pool starts no thread until used.
            auto* fresh = new HostPool(pid);
            if (current.compare_exchange_strong(pool, fresh, std::memory_order_acq_rel)) return *fresh;
            delete fresh;
        }
        return *pool;
    }

    /** @brief True on a pool worker thread. */
    static bool& in_worker() {
        static thread_local bool flag = false;
        return flag;
    }

    /** @brief Run @p job on at most @p team threads, the caller included. In place, in
     *  lane order, when the team is one, the pool is busy, or the caller is a worker,
     *  or inside a lane. Lane bodies must not throw. */
    void run(const Job& job, int team) {
        team = static_cast<int>(std::min<std::int64_t>(team, job.lanes));
        bool idle = false;
        if (team <= 1 || in_worker() ||
            !busy_.compare_exchange_strong(idle, true, std::memory_order_acquire)) {
            run_in_place(job);
            return;
        }
        team = std::min(team, MAX_TEAM);
        if (workers_ == 0) plan_pins();
        while (workers_ < team - 1) {
            Slot& slot = slots_.emplace_back();
            std::thread([this, &slot, k = workers_] { pin(k); work(k, slot); }).detach();
            ++workers_;
        }
        job_ = &job;
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        left_.store(0, std::memory_order_relaxed);
        const std::uint32_t gen = start_.load(std::memory_order_relaxed) + 1;
        door_.store((std::uint64_t{ gen } << 32) | OPEN |
                    (static_cast<std::uint64_t>(team - 1) << CAP_SHIFT), std::memory_order_release);
        start_.store(gen, std::memory_order_seq_cst);
        for (int k = 0; k < team - 1; ++k)   // only the team's sleepers wake; the others stay asleep
            slots_[k].wake();

        drain(job);
        await(done_, job.lanes);
        const std::uint64_t closed = door_.fetch_and(~OPEN, std::memory_order_acq_rel);
        await(left_, static_cast<std::uint32_t>(closed & JOINED_MASK));
        job_ = nullptr;
        busy_.store(false, std::memory_order_release);
    }

    /** @brief How many times worker @p k woke from sleep (0 for a worker that does not exist). */
    std::uint64_t wakeups(int k) const {
        return static_cast<std::size_t>(k) < slots_.size() ? slots_[k].wakeups.load(std::memory_order_relaxed) : 0;
    }

    /** @brief Run every lane of @p job on the calling thread, in order. */
    static void run_in_place(const Job& job) {
        for (std::uint32_t i = 0; i < job.lanes; ++i) job.run(job.ctx, i);
    }

private:
    /// One worker's private sleep word: the pool wakes exactly the sleepers a job has a place for.
    struct Slot {
        std::atomic<std::uint32_t> word{ 0 };
        std::atomic<int> sleeping{ 0 };
        std::atomic<std::uint64_t> wakeups{ 0 };
        void wake() {
            if (sleeping.load(std::memory_order_seq_cst) == 0) return;
            word.fetch_add(1, std::memory_order_seq_cst);
            word.notify_one();
        }
    };

    static constexpr std::uint64_t OPEN = std::uint64_t{ 1 } << 31;
    static constexpr int CAP_SHIFT = 15;
    static constexpr std::uint64_t JOINED_MASK = (std::uint64_t{ 1 } << CAP_SHIFT) - 1;
    static constexpr int MAX_TEAM = 1 << 14;

    explicit HostPool(long pid) : owner_(pid) {}

    // One worker per physical core: lane 0 is the caller's core (never re-pinned), worker k takes the
    // sibling set of the k-th other core, and workers past the cores keep the whole mask. Off Linux, or
    // with fewer than two cores or no sysfs, nothing is pinned.
    void plan_pins() {
#if defined(__linux__)
        pins_ = core_masks();
        if (pins_.size() < 2) { pins_.clear(); return; }
        const int here = sched_getcpu();
        for (auto it = pins_.begin(); it != pins_.end(); ++it)
            if (here >= 0 && CPU_ISSET(here, &*it)) { pins_.erase(it); break; }
#endif
    }

    void pin([[maybe_unused]] int worker) const {
#if defined(__linux__)
        if (static_cast<std::size_t>(worker) < pins_.size())
            sched_setaffinity(0, sizeof(cpu_set_t), &pins_[worker]);
#endif
    }

    static long self_pid() {
#if defined(__unix__) || defined(__APPLE__)
        return static_cast<long>(::getpid());
#else
        return 0;
#endif
    }

    // Spin up to HOST_SPIN_US for `word` to satisfy `ok`, then sleep on it.
    template <class Ok>
    static void wait_for(std::atomic<std::uint32_t>& word, std::atomic<int>& sleepers, Ok ok) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto spin = std::chrono::duration<double, std::micro>(HOST_SPIN_US);
        for (unsigned k = 1;; ++k) {
            std::uint32_t seen = word.load(std::memory_order_acquire);
            if (ok(seen)) return;
            detail::cpu_relax();
            if ((k & 63) == 0 && std::chrono::steady_clock::now() - t0 > spin) {
                sleepers.fetch_add(1, std::memory_order_seq_cst);
                seen = word.load(std::memory_order_seq_cst);
                if (!ok(seen)) word.wait(seen, std::memory_order_acquire);
                sleepers.fetch_sub(1, std::memory_order_seq_cst);
            }
        }
    }

    void await(std::atomic<std::uint32_t>& count, std::uint32_t target) {
        wait_for(count, sync_sleepers_, [target](std::uint32_t v) { return v >= target; });
    }

    void bump(std::atomic<std::uint32_t>& count) {
        count.fetch_add(1, std::memory_order_seq_cst);
        if (sync_sleepers_.load(std::memory_order_seq_cst) > 0) count.notify_all();
    }

    void drain(const Job& job) {
        for (;;) {
            const std::uint32_t lane = next_.fetch_add(1, std::memory_order_relaxed);
            if (lane >= job.lanes) return;
            if (job.gate != nullptr) await(done_, job.gate[lane]);   // the phases before this one are done
            job.run(job.ctx, lane);
            bump(done_);
        }
    }

    // Worker @p k joins only a job whose team has a place for it (k < team - 1): the pinned workers come first.
    bool join(std::uint32_t gen, int k) {
        std::uint64_t door = door_.load(std::memory_order_acquire);
        for (;;) {
            if ((door >> 32) != gen || !(door & OPEN) ||
                static_cast<std::uint64_t>(k) >= ((door >> CAP_SHIFT) & JOINED_MASK))
                return false;
            if (door_.compare_exchange_weak(door, door + 1, std::memory_order_acq_rel,
                                            std::memory_order_acquire))
                return true;
        }
    }

    void work(int k, Slot& slot) {
        in_worker() = true;
        std::uint32_t seen = 0;
        const auto spin = std::chrono::duration<double, std::micro>(HOST_SPIN_US);
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned i = 1; start_.load(std::memory_order_acquire) == seen; ++i) {
                detail::cpu_relax();
                if ((i & 63) != 0 || std::chrono::steady_clock::now() - t0 <= spin) continue;
                // sleep on this worker's own word: a launch wakes it only when the team reaches it
                const std::uint32_t w = slot.word.load(std::memory_order_seq_cst);
                slot.sleeping.store(1, std::memory_order_seq_cst);
                if (start_.load(std::memory_order_seq_cst) == seen) {
                    slot.word.wait(w, std::memory_order_acquire);
                    slot.wakeups.fetch_add(1, std::memory_order_relaxed);
                }
                slot.sleeping.store(0, std::memory_order_seq_cst);
            }
            seen = start_.load(std::memory_order_acquire);
            if (!join(seen, k)) continue;
            drain(*job_);
            bump(left_);
        }
    }

    const long owner_;
    std::atomic<bool> busy_{ false };
    int workers_ = 0;                       // guarded by busy_
    std::deque<Slot> slots_;                // one per worker, guarded by busy_ (a deque: addresses are stable)
#if defined(__linux__)
    std::vector<cpu_set_t> pins_;           // set before the first worker starts, then read-only
#endif
    const Job* job_ = nullptr;              // published by door_, retired after left_
    std::atomic<std::uint64_t> door_{ 0 };  // gen << 32 | OPEN | cap << 15 | joined
    std::atomic<std::uint32_t> start_{ 0 }, next_{ 0 }, done_{ 0 }, left_{ 0 };
    std::atomic<int> sync_sleepers_{ 0 };
};

}  // namespace detail
}  // namespace exec
}  // namespace eagle
