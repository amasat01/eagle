// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file events/Refine.h
 * @brief Locating each sample's event inside its stop step: the per-sample
 *        bracket, tolerance and status logic and the Newton/bisection update,
 *        on the host (OpenMP) and on the device (one thread per sample).
 *
 * A stepped run stops a sample at the first step whose guard @c g reaches
 * the closed test @c g <= tol. The event lies inside that step: from the
 * state held before it, a partial step of size @c tau in [0, h] crosses
 * @c g = 0. The step map with a per-sample @c tau is the caller's (a compiled
 * entry evaluating, per sample, every guard @c G_j(tau) and its derivative
 * @c dG_j/dtau); this header owns what happens between two evaluations:
 *
 * - classify (the evaluation at @c tau = h): a sample that never stopped, or
 *   took no step, has no event; two guards inside the tolerance at once is a
 *   refusal; a guard that sits at zero within the tolerance keeps the step's
 *   own end, @c tau = h, with no iteration;
 * - newton (every later evaluation): converged when |G| <= tol, else the
 *   bracket shrinks to the sign change and @c tau takes the Newton step, or
 *   the bracket midpoint when that step leaves the bracket;
 * - graze: a crossing whose @c |dG/dtau| is tiny next to |g_x|·|Phi_tau|
 *   (over the extended state, time included) has no differentiable event
 *   time;
 * - commit: the refined state replaces the stepped one, and @c t_stop is
 *   written for every sample.
 *
 * The time at which the tolerance is measured is the index form the stepping
 * twin uses, @c t0 + k*h, plus @c tau.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define EAGLE_EVENTS_HD __host__ __device__
#else
#define EAGLE_EVENTS_HD
#endif

namespace eagle {
namespace events {

using idx_t = std::int64_t;

/** @brief Per-sample event status (the ``event_status`` plane, int8). */
enum Status : std::int8_t {
    kPending = 0,   ///< still iterating
    kRefined = 1,   ///< the crossing was located inside the stop step
    kAtStep = 2,    ///< the crossing sits at the step's end within tolerance: tau = h
    kNoEvent = 3,   ///< never stopped (max_steps), stopped before any step, or no guard crossed
    kGraze = 4,     ///< tangent crossing: the event time is not differentiable
    kTwoRules = 5,  ///< two guards inside the tolerance in the same step
};

/** @brief The most Newton/bisection updates one sample takes. */
inline constexpr int kMaxIterations = 8;
/** @brief The closed stop test's tolerance, in ulps of the time times |dG/dtau|. */
inline constexpr double kEventUlps = 4.0;
/** @brief The absolute part of the event tolerance. */
inline constexpr double kGuardTol = 0.0;
/** @brief The most stop rules (guards) one kernel may have. */
inline constexpr int kMaxGuards = 8;
/** @brief The most state components (plane rows) one extended state may have. */
inline constexpr int kMaxComponents = 32;

/** @brief The graze threshold: sqrt(machine epsilon) of @p T. */
template <class T>
EAGLE_EVENTS_HD inline T grazeTol() {
    return std::is_same_v<T, float> ? T(3.4526698e-4) : T(1.4901161193847656e-8);
}

/** @brief sqrt of @p x (no std:: in device code). */
template <class T>
EAGLE_EVENTS_HD inline T root(T x) {
    if constexpr (std::is_same_v<T, float>)
        return ::sqrtf(x);
    else
        return ::sqrt(x);
}

/** @brief The gap from |x| to the next representable value. */
template <class T>
EAGLE_EVENTS_HD inline T spacing(T x) {
    const T a = x < T(0) ? -x : x;
    if constexpr (std::is_same_v<T, float>)
        return ::nextafterf(a, INFINITY) - a;
    else
        return ::nextafter(a, static_cast<double>(INFINITY)) - a;
}

/**
 * @brief The planes an event operation reads and writes, all sample-last of
 *        @c n samples; unused pointers are null. Typed views of the seam's
 *        descriptor, passed by value to a device kernel.
 */
template <class T>
struct EventPlanes {
    idx_t n = 0;
    int guards = 0;                       ///< stop rules
    int components = 0;                   ///< extended-state rows in a/b/c
    const T* g[kMaxGuards] = {};          ///< G_j at the current tau
    const T* dg[kMaxGuards] = {};         ///< dG_j/dtau at the current tau
    T* gbar[kMaxGuards] = {};             ///< the IFT seed of each guard (Sensitivity.h)
    T* a[kMaxComponents] = {};            ///< op-specific component rows
    const T* b[kMaxComponents] = {};
    const T* c[kMaxComponents] = {};
    const std::uint8_t* terminated = nullptr;
    const std::int64_t* applied = nullptr;   ///< steps applied, the stop step included
    const T* t0 = nullptr;
    const T* h = nullptr;                    ///< the grid step, per sample
    std::int8_t* status = nullptr;
    std::int8_t* which = nullptr;            ///< the crossing guard
    T* tau = nullptr;
    T* lo = nullptr;
    T* hi = nullptr;
    T* tStop = nullptr;
};

/** @brief The index-form time at the start of the stop step: t0 + (s-1)*h. */
template <class T>
EAGLE_EVENTS_HD inline T stepStart(const EventPlanes<T>& p, idx_t i) {
    return p.t0[i] + static_cast<T>(p.applied[i] - 1) * p.h[i];
}

/** @brief The event tolerance at time @p t for a guard of slope @p dG. */
template <class T>
EAGLE_EVENTS_HD inline T eventTol(T t, T dG) {
    return T(kEventUlps) * spacing(t) * (dG < T(0) ? -dG : dG) + T(kGuardTol);
}

/**
 * @brief Classify sample @p i from the evaluation at tau = h (the step's end):
 *        writes status, which, tau, lo, hi.
 */
template <class T>
EAGLE_EVENTS_HD inline void classify(const EventPlanes<T>& p, idx_t i) {
    const T h = p.h[i];
    p.tau[i] = h;
    p.lo[i] = T(0);
    p.hi[i] = h;
    p.which[i] = 0;
    if (!p.terminated[i] || p.applied[i] < 1) {
        p.status[i] = kNoEvent;
        return;
    }
    const T t = stepStart(p, i) + h;
    int hits = 0, hit = -1;
    for (int j = 0; j < p.guards; ++j) {
        if (p.g[j][i] <= eventTol(t, p.dg[j][i])) {
            ++hits;
            hit = j;
        }
    }
    if (hits == 0) {
        p.status[i] = kNoEvent;
        return;
    }
    p.which[i] = static_cast<std::int8_t>(hit);
    if (hits > 1) {
        p.status[i] = kTwoRules;
        return;
    }
    const T G = p.g[hit][i], dG = p.dg[hit][i];
    if (-G <= eventTol(t, dG)) {
        p.status[i] = kAtStep;
        return;
    }
    p.status[i] = kPending;
    T next = h - G / dG;
    if (!(next > T(0) && next < h)) next = h / T(2);
    p.tau[i] = next;
}

/**
 * @brief One Newton/bisection update of a pending sample from the evaluation
 *        at its current tau. A converged sample keeps its tau (the evaluation
 *        that converged is the refined state).
 */
template <class T>
EAGLE_EVENTS_HD inline void newton(const EventPlanes<T>& p, idx_t i) {
    if (p.status[i] != kPending) return;
    const int j = p.which[i];
    const T tau = p.tau[i], G = p.g[j][i], dG = p.dg[j][i];
    const T t = stepStart(p, i) + tau;
    if ((G < T(0) ? -G : G) <= eventTol(t, dG)) {
        p.status[i] = kRefined;
        return;
    }
    if (G > T(0))
        p.lo[i] = tau;
    else
        p.hi[i] = tau;
    const T lo = p.lo[i], hi = p.hi[i];
    T next = tau - G / dG;
    if (!(next > lo && next < hi)) next = lo + (hi - lo) / T(2);
    if (hi - lo <= T(kEventUlps) * spacing(t)) {
        p.status[i] = kRefined;   // the bracket is down to the time's rounding
    }
    p.tau[i] = next;
}

/** @brief End of the iterations: a sample still pending takes its last tau. */
template <class T>
EAGLE_EVENTS_HD inline void settle(const EventPlanes<T>& p, idx_t i) {
    if (p.status[i] == kPending) p.status[i] = kRefined;
}

/**
 * @brief The graze test of a located sample: |dG/dtau| <= grazeTol·|g_x|·|Phi_tau|
 *        over the extended state. @c b holds g_x (one row per component),
 *        @c c holds Phi_tau.
 */
template <class T>
EAGLE_EVENTS_HD inline void graze(const EventPlanes<T>& p, idx_t i) {
    const std::int8_t s = p.status[i];
    if (s != kRefined && s != kAtStep) return;
    T gx = T(0), ph = T(0);
    for (int r = 0; r < p.components; ++r) {
        gx += p.b[r][i] * p.b[r][i];
        ph += p.c[r][i] * p.c[r][i];
    }
    const T dG = p.dg[p.which[i]][i];
    if ((dG < T(0) ? -dG : dG) <= grazeTol<T>() * root(gx) * root(ph))
        p.status[i] = kGraze;
}

/**
 * @brief Commit: a refined sample's state rows @c a take the refined rows
 *        @c b; @c t_stop is the event time of a located sample and the
 *        step's end time of every other.
 */
template <class T>
EAGLE_EVENTS_HD inline void commit(const EventPlanes<T>& p, idx_t i) {
    const std::int8_t s = p.status[i];
    if (s == kRefined) {
        for (int r = 0; r < p.components; ++r) p.a[r][i] = p.b[r][i];
        p.tStop[i] = stepStart(p, i) + p.tau[i];
    } else {
        p.tStop[i] = p.t0[i] + static_cast<T>(p.applied[i]) * p.h[i];
    }
}

/** @brief The operations of this header and Sensitivity.h, by code. */
enum Op : std::uint32_t {
    kOpClassify = 0,
    kOpNewton = 1,
    kOpSettle = 2,
    kOpGraze = 3,
    kOpCommit = 4,
    kOpSeed = 5,     ///< Sensitivity.h
    kOpSelect = 6,   ///< Sensitivity.h
};

/** @brief Apply per-sample operation @p f to every sample on the host. */
template <class T, class F>
void forEachSample(const EventPlanes<T>& p, F&& f) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (p.n > 65536)
#endif
    for (idx_t i = 0; i < p.n; ++i) f(p, i);
}

/** @brief Whether any sample is still pending (the host loop's early exit). */
template <class T>
bool anyPending(const EventPlanes<T>& p) {
    for (idx_t i = 0; i < p.n; ++i)
        if (p.status[i] == kPending) return true;
    return false;
}

/**
 * @brief The refine loop: up to kMaxIterations rounds of evaluate + newton,
 *        then settle and one last evaluation at the final tau.
 *
 * @param evaluate  evaluate(): the caller's entry, writing g/dg (and the
 *                  refined state) at the current tau of every sample.
 * @param update    update(op): apply kOpNewton / kOpSettle to every sample.
 * @param pending   pending(): whether any sample is still iterating (a device
 *                  caller may return true always and run every round).
 * @return the rounds run.
 */
template <class Evaluate, class Update, class Pending>
int refineLoop(Evaluate&& evaluate, Update&& update, Pending&& pending) {
    int rounds = 0;
    while (rounds < kMaxIterations && pending()) {
        evaluate();
        update(kOpNewton);
        ++rounds;
    }
    update(kOpSettle);
    evaluate();
    return rounds;
}

#ifdef __CUDACC__
namespace device {

template <class T>
__global__ void refineKernel(EventPlanes<T> p, std::uint32_t op) {
    const idx_t i = static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    switch (op) {
    case kOpClassify: classify(p, i); break;
    case kOpNewton: newton(p, i); break;
    case kOpSettle: settle(p, i); break;
    case kOpGraze: graze(p, i); break;
    case kOpCommit: commit(p, i); break;
    default: break;
    }
}

/** @brief Enqueue a forward event operation over every sample on @p stream. */
template <class T>
void refine(const EventPlanes<T>& p, std::uint32_t op, cudaStream_t stream) {
    if (p.n == 0) return;
    refineKernel<T><<<static_cast<unsigned>((p.n + 255) / 256), 256, 0, stream>>>(p, op);
}

}  // namespace device
#endif

/** @brief Apply a forward event operation (kOpClassify..kOpCommit) on the host. */
template <class T>
void refineHost(const EventPlanes<T>& p, std::uint32_t op) {
    switch (op) {
    case kOpClassify: forEachSample(p, [](const EventPlanes<T>& q, idx_t i) { classify(q, i); }); break;
    case kOpNewton: forEachSample(p, [](const EventPlanes<T>& q, idx_t i) { newton(q, i); }); break;
    case kOpSettle: forEachSample(p, [](const EventPlanes<T>& q, idx_t i) { settle(q, i); }); break;
    case kOpGraze: forEachSample(p, [](const EventPlanes<T>& q, idx_t i) { graze(q, i); }); break;
    case kOpCommit: forEachSample(p, [](const EventPlanes<T>& q, idx_t i) { commit(q, i); }); break;
    default: break;
    }
}

}  // namespace events
}  // namespace eagle
