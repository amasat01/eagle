// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The same embedding demo with no CUDA anywhere: the three plugins (mattrace,
// toy_energy, toy_energy_vjp) run on the host through `eagle::cpu::PluginRegistry`,
// on the same inputs as embedding_demo.cu and held to the same references.
// The host runs float64 plugins only (the registry refuses any other scalar_type).
//
// Usage: embedding_demo_cpu [artifact_dir] [N]
//   artifact_dir holds host_manifest.json, the sidecars (*.json) and host_plugins.so; it defaults to the
//   directory this target stages at build time (EMBED_DEMO_ART_DIR).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "demo_common.h"
#include "plugin/host_registry.h"

#ifndef EMBED_DEMO_ART_DIR
#define EMBED_DEMO_ART_DIR "art"
#endif

int run(int argc, char** argv) {
    const std::string dir = (argc >= 2) ? argv[1] : EMBED_DEMO_ART_DIR;
    const std::uint32_t N =
        (argc >= 3) ? std::uint32_t(std::strtoul(argv[2], nullptr, 10)) : 4096;

    // Load the three plugins from the manifest: each sidecar (arg_spec, shapes) plus
    // its entry in the one shared object (format "host").
    eagle::cpu::PluginRegistry registry;
    registry.load(eagle::plugin::parse_manifest(dir + "/host_manifest.json"), dir);

    demo::Inputs in(N);
    demo::Outputs o(N);

    // Bind by name, exactly as on the device: host pointers, no upload.
    registry.bind_matrix("M", in.M.data(), N, 3, 3);
    registry.bind_handle("out", o.out.data());
    registry.bind_vector("x", in.x.data(), N);
    registry.bind_handle("e", o.e.data());
    registry.bind_uniform("p", in.p);
    registry.bind_handle("e_bar", in.e_bar.data());
    registry.bind_vector("x_bar", o.x_bar.data(), N);
    registry.bind_handle("p_bar", o.p_bar.data());

    const int launched = registry.run(int(N));

    std::printf("N_PLUGINS=%zu\n", registry.size());
    std::printf("ACTIVE=%d\n", registry.active());
    std::printf("LAUNCHED=%d\n", launched);
    const bool checks_ok = demo::check(in, o);

    // The float32 arm: the same energy law as a float32 plugin, through PluginRegistry32,
    // against the closed form at float32 tolerance.
    std::vector<float> x32(in.x.begin(), in.x.end()), e32(N);
    eagle::cpu::PluginRegistry32 registry32;
    registry32.load(eagle::plugin::parse_manifest(dir + "/host_manifest_f32.json"), dir);
    registry32.bind_vector("x", x32.data(), N);
    registry32.bind_handle("e", e32.data());
    registry32.bind_uniform("p", in.p);
    const int launched32 = registry32.run(int(N));
    double f32_max_rel = 0.0;
    for (std::uint32_t i = 0; i < N; ++i) {
        const double ref = demo::host_energy(in.x[i], in.x[N + i], in.x[2 * N + i], in.p);
        f32_max_rel = std::max(f32_max_rel, std::abs(e32[i] - ref) / std::max(std::abs(ref), 1e-12));
    }
    std::printf("F32_PRIMAL_MAX_REL=%.3e\n", f32_max_rel);
    const bool ok = (launched == 3) && checks_ok && launched32 == 1 && f32_max_rel < 1e-5;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
