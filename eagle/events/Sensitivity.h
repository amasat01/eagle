// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file events/Sensitivity.h
 * @brief The event-time sensitivity of a refined stop: the implicit-function
 *        correction as a rank-1 seed of the refined step's vjp, on the host
 *        and on the device.
 *
 * The refined end state is x* = Phi(x_k, theta, tau*) with tau* defined by
 * G(x_k, theta, tau*) = 0, where G is the guard at the end of the partial
 * step and x the EXTENDED state (time included). For an objective with
 * gradient xbar at x*, every input z of the refined step gets
 *
 *     zbar = Phi_z^T xbar + dtau_dz (xbar . Phi_tau)
 *          = Phi_z^T xbar + G_z^T gbar,   gbar = -(xbar . Phi_tau) / (dG/dtau),
 *
 * with dtau_dz = -G_z / (dG/dtau) from G = 0, so ONE vjp of the refined step (state and guards out) seeded with
 * (xbar, gbar) is the whole correction: @c seed writes those seeds, and
 * @c select hands the step-by-step reverse the adjoint of the state before
 * the stop step for a located sample, the end-state adjoint untouched for any
 * other (its event sensitivity is zero).
 */

#include "eagle/events/Refine.h"

namespace eagle {
namespace events {

/** @brief Whether sample @p i has a located event the correction applies to. */
template <class T>
EAGLE_EVENTS_HD inline bool located(const EventPlanes<T>& p, idx_t i) {
    const std::int8_t s = p.status[i];
    return s == kRefined || s == kAtStep;
}

/**
 * @brief The vjp seeds of the refined step for sample @p i: @c a (out) =
 *        xbar (@c b) for a located sample and zero otherwise; @c gbar (out)
 *        the rank-1 guard seed. @c c holds Phi_tau, @c dg the slopes.
 */
template <class T>
EAGLE_EVENTS_HD inline void seed(const EventPlanes<T>& p, idx_t i) {
    for (int j = 0; j < p.guards; ++j) p.gbar[j][i] = T(0);
    if (!located(p, i)) {
        for (int r = 0; r < p.components; ++r) p.a[r][i] = T(0);
        return;
    }
    T lambda = T(0);
    for (int r = 0; r < p.components; ++r) {
        p.a[r][i] = p.b[r][i];
        lambda += p.b[r][i] * p.c[r][i];
    }
    const int j = p.which[i];
    p.gbar[j][i] = -lambda / p.dg[j][i];
}

/**
 * @brief The adjoint the step-by-step reverse starts from: @c a (out) = the
 *        refined step's input adjoint (@c c) for a located sample, the
 *        end-state adjoint (@c b) for any other.
 */
template <class T>
EAGLE_EVENTS_HD inline void select(const EventPlanes<T>& p, idx_t i) {
    const bool on = located(p, i);
    for (int r = 0; r < p.components; ++r) p.a[r][i] = on ? p.c[r][i] : p.b[r][i];
}

/** @brief The steps the step-by-step reverse runs per sample: one fewer for a
 *  located sample (its stop step is the refined one). Host, sample order. */
inline void reverseSteps(std::int64_t* steps, const std::int8_t* status, idx_t n) {
    for (idx_t i = 0; i < n; ++i)
        if ((status[i] == kRefined || status[i] == kAtStep) && steps[i] > 0) --steps[i];
}

#ifdef __CUDACC__
namespace device {

template <class T>
__global__ void sensitivityKernel(EventPlanes<T> p, std::uint32_t op) {
    const idx_t i = static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    if (op == kOpSeed)
        seed(p, i);
    else if (op == kOpSelect)
        select(p, i);
}

/** @brief Enqueue kOpSeed or kOpSelect over every sample on @p stream. */
template <class T>
void sensitivity(const EventPlanes<T>& p, std::uint32_t op, cudaStream_t stream) {
    if (p.n == 0) return;
    sensitivityKernel<T><<<static_cast<unsigned>((p.n + 255) / 256), 256, 0, stream>>>(p, op);
}

/** @brief Any event operation (Refine.h's or this header's) by code. */
template <class T>
void apply(const EventPlanes<T>& p, std::uint32_t op, cudaStream_t stream) {
    if (op == kOpSeed || op == kOpSelect)
        sensitivity(p, op, stream);
    else
        refine(p, op, stream);
}

}  // namespace device
#endif

/** @brief Any event operation (Refine.h's or this header's) on the host. */
template <class T>
void applyHost(const EventPlanes<T>& p, std::uint32_t op) {
    if (op == kOpSeed)
        forEachSample(p, [](const EventPlanes<T>& q, idx_t i) { seed(q, i); });
    else if (op == kOpSelect)
        forEachSample(p, [](const EventPlanes<T>& q, idx_t i) { select(q, i); });
    else
        refineHost(p, op);
}

}  // namespace events
}  // namespace eagle
