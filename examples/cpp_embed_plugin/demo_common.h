// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Inputs and checks shared by the CUDA demo (embedding_demo.cu) and the CPU demo
// (embedding_demo_cpu.cpp): both run the same three plugins on the same data and
// are held to the same references.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace demo {

constexpr double TOL     = 1e-12;  // mattrace / primal: exact closed-form agreement
constexpr double FD_H    = 1e-6;   // central-difference step
constexpr double FD_RTOL = 1e-5;   // VJP-vs-FD relative tolerance
constexpr double FD_ATOL = 1e-8;

// The closed form the toy_energy plugin computes, written independently on the host.
inline double host_energy(double x0, double x1, double x2, double p) {
    const double r2 = x0 * x0 + x1 * x1 + x2 * x2;
    return p * std::exp(-0.5 * r2);
}

// Deterministic inputs (no data files), all flat SoA: component k of sample i is at [k * N + i].
struct Inputs {
    std::uint32_t N;
    double p = 1.7;
    std::vector<double> M;      // (9, N) 3x3 matrices, mattrace input
    std::vector<double> x;      // (3, N) primal input
    std::vector<double> e_bar;  // (N)    upstream cotangent

    explicit Inputs(std::uint32_t n) : N(n), M(9 * std::size_t(n)), x(3 * std::size_t(n)), e_bar(n) {
        std::mt19937_64 rng(20260719);
        std::uniform_real_distribution<double> ux(-1.3, 1.3), ueb(0.3, 2.0);
        for (int d = 0; d < 9; ++d)
            for (std::uint32_t i = 0; i < N; ++i)
                M[std::size_t(d) * N + i] = double(d) + 0.25 * double(i);
        for (int k = 0; k < 3; ++k)
            for (std::uint32_t i = 0; i < N; ++i) x[std::size_t(k) * N + i] = ux(rng);
        for (std::uint32_t i = 0; i < N; ++i) e_bar[i] = ueb(rng);
    }
};

// What the three plugins wrote.
struct Outputs {
    std::vector<double> out, e, x_bar, p_bar;
    explicit Outputs(std::uint32_t n) : out(n), e(n), x_bar(3 * std::size_t(n)), p_bar(n) {}
};

// Print the four error figures and return true when all are inside tolerance.
inline bool check(const Inputs& in, const Outputs& o) {
    const std::uint32_t N = in.N;
    double mat = 0.0, primal = 0.0, vjp_x = 0.0, vjp_p = 0.0;
    for (std::uint32_t i = 0; i < N; ++i) {
        // mattrace vs the direct host sum
        const double tr = in.M[0 * N + i] + in.M[4 * N + i] + in.M[8 * N + i];
        mat = std::max(mat, std::abs(o.out[i] - tr) / std::max(std::abs(tr), 1.0));

        const double x0 = in.x[0 * N + i], x1 = in.x[1 * N + i], x2 = in.x[2 * N + i];
        const double p = in.p, eb = in.e_bar[i];
        // primal vs the closed form
        const double ref = host_energy(x0, x1, x2, p);
        primal = std::max(primal, std::abs(o.e[i] - ref) / std::max(std::abs(ref), 1e-12));

        // VJP vs a central difference of the same closed form (independent of the
        // analytic formula the plugin evaluates)
        const double want_x[3] = {
            eb * (host_energy(x0 + FD_H, x1, x2, p) - host_energy(x0 - FD_H, x1, x2, p)) / (2 * FD_H),
            eb * (host_energy(x0, x1 + FD_H, x2, p) - host_energy(x0, x1 - FD_H, x2, p)) / (2 * FD_H),
            eb * (host_energy(x0, x1, x2 + FD_H, p) - host_energy(x0, x1, x2 - FD_H, p)) / (2 * FD_H)};
        const double want_p =
            eb * (host_energy(x0, x1, x2, p + FD_H) - host_energy(x0, x1, x2, p - FD_H)) / (2 * FD_H);
        for (int k = 0; k < 3; ++k)
            vjp_x = std::max(vjp_x, std::abs(o.x_bar[std::size_t(k) * N + i] - want_x[k]) /
                                        std::max(std::abs(want_x[k]), FD_ATOL));
        vjp_p = std::max(vjp_p, std::abs(o.p_bar[i] - want_p) / std::max(std::abs(want_p), FD_ATOL));
    }
    std::printf("MATTRACE_MAX_REL=%.3e\n", mat);
    std::printf("PRIMAL_MAX_REL=%.3e\n", primal);
    std::printf("VJP_X_MAX_REL_VS_FD=%.3e\n", vjp_x);
    std::printf("VJP_P_MAX_REL_VS_FD=%.3e\n", vjp_p);
    return mat < TOL && primal < TOL && vjp_x < FD_RTOL && vjp_p < FD_RTOL;
}

}  // namespace demo
