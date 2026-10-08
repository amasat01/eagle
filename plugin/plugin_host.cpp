// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Precompiled C++/CUDA "plugin acceptor" for generated force kernels.
//
// This is the deployment surface the CuPy test harness only *stands in* for: a
// standalone host that, using nothing but the CUDA Driver API and the binary
// ABI (see gref_abi.h) — no nvcc, no CuPy at runtime —
//
//   1. driver-loads a generated PTX plugin (cuModuleLoadData + cuModuleGetFunction),
//   2. builds GRef / HandleT views by value over device buffers it owns,
//   3. packs them in the sidecar's arg_spec order and launches the kernel,
//      both standalone (cuLaunchKernel) and as a node in a captured CUDA graph
//      (cuStreamBeginCapture -> cuGraphInstantiate -> cuGraphLaunch),
//   4. verifies the result against a golden reference.
//
// The graph path captures a memset-reset + the kernel launch, then replays
// twice — proving the production reset-then-accumulate idiom survives replay.
//
// Usage:  plugin_host <artifact_dir> <N>
// The directory holds plugin.ptx, plugin.json, one <name>.bin per vec_in /
// per_sample argument (raw little-endian f64, SoA for vectors), uniforms.bin
// (one f64 per uniform, in arg_spec order), golden.bin (expected outVec), and
// optionally terminated.bin (N bytes).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda.h>

#include "dlpack_abi_offsets.h" // layout gate for the vendored DLPack ABI
#include "dlpack_bridge.h"
#include "gref_abi.h"
#include "sidecar.h"

using namespace eagle::plugin;  // the ABI protocol PODs (Sidecar / GRefMirror / handles / parse_*)

namespace {

constexpr double TOL = 1e-12;

void cu_check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(r, &msg);
        std::fprintf(stderr, "CUDA error %d (%s) at %s\n", int(r),
                     msg ? msg : "?", what);
        std::exit(2);
    }
}
#define CU_CHECK(call) cu_check((call), #call)

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return bool(f);
}

std::vector<double> read_doubles(const std::string& p, std::size_t count) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "missing input file: %s\n", p.c_str()); std::exit(2); }
    std::vector<double> v(count);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(count * sizeof(double)));
    if (std::size_t(f.gcount()) != count * sizeof(double)) {
        std::fprintf(stderr, "short read on %s\n", p.c_str()); std::exit(2);
    }
    return v;
}

std::string read_text(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "missing file: %s\n", p.c_str()); std::exit(2); }
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

// Max over samples of the per-sample vector relative error (got/ref are SoA
// (3, N): component d of sample i at index d*N + i).
double max_rel(const std::vector<double>& got, const std::vector<double>& ref,
               std::size_t n) {
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double dn = 0.0, rn = 0.0;
        for (std::size_t d = 0; d < 3; ++d) {
            double e = got[d * n + i] - ref[d * n + i];
            double r = ref[d * n + i];
            dn += e * e; rn += r * r;
        }
        // Relative error where the reference is non-zero; absolute residual
        // where it is zero (e.g. a terminated sample whose golden column is 0).
        const double den = std::sqrt(rn);
        double rel = (den > 0.0) ? std::sqrt(dn) / den : std::sqrt(dn);
        if (rel > worst) worst = rel;
    }
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <artifact_dir> <N>\n", argv[0]);
        return 2;
    }
    const std::string dir = std::string(argv[1]) + "/";
    const std::uint32_t N = std::uint32_t(std::strtoul(argv[2], nullptr, 10));
    const std::size_t vlen = std::size_t(N) * 3;

    const Sidecar sc = parse_sidecar(dir + "plugin.json");

    // ---- door #7 gates: the artifact is vetted BEFORE any CUDA call --------
    // This standalone host is entry point #7 of the seven-door register
    // (plugin/roles.h): it used to driver-load
    // and launched whatever `parse_sidecar` returned, with ZERO gates — the
    // only launching executable in the tree without a registry behind it.
    //
    // Order is the one locked for every launching door: shared
    // `validate_sidecar` -> launch certification (here, the family dispatch
    // below) -> loader-specific gates (`check_aether_abi`). Family refusal must
    // precede ABI-tag complaints, so a multi-defect artifact of the wrong
    // family is told what it actually is rather than nitpicked on its tag.
    //
    // Every rejection returns 2 with the reason on stderr from INSIDE this
    // block, i.e. before `cuInit` — a refusal is GPU-free by construction, not
    // by luck, so these paths are exercisable on a machine with no device.
    try {
        validate_sidecar(sc, "plugin host artifact");
        // Family DISPATCH, not validation (the `_require_pattern` precedent):
        // everything below binds a VECTOR-shaped arg_spec role
        // set (out / vec_in / per_sample / terminated / uniform) and computes a
        // 3-vector relative error against a golden. Absence stays lenient — a
        // pre-freeze sidecar never stamped a `pattern`, and an untagged
        // artifact is a vector kernel by definition (the same default
        // `LoadedVector` takes). A recognized-but-different family used to fall
        // through to the arg_spec loop and die on the first role this host does
        // not bind: a `pure` sidecar reported `unknown arg role: mutable` for a
        // role that is perfectly valid schema-v1 vocabulary. That message named
        // the wrong defect entirely — the artifact is not malformed, it is the
        // wrong KIND for this host, and the gate must say so.
        const std::string pattern = sc.pattern.empty() ? std::string("vector")
                                                       : sc.pattern;
        if (pattern != "vector")
            throw std::runtime_error("plugin host artifact: plugin family '" +
                pattern + "' is recognized but not launchable by this loader "
                "(this standalone host binds vector-shaped artifacts only)");
        check_aether_abi(sc.aether_abi, "plugin host artifact: sidecar");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }

    // ---- driver init + module load (no nvcc) --------------
    CU_CHECK(cuInit(0));
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev, 0));
    // the device's primary context: one call on every CUDA version (CUDA 13
    // gave cuCtxCreate a parameters argument)
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));
    const std::string ptx = read_text(dir + "plugin.ptx");
    CUmodule mod; CU_CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, mod, sc.kernel.c_str()));

    // uniforms (one f64 per 'uniform' arg, in arg_spec order)
    std::size_t n_uniform = 0;
    for (const auto& a : sc.arg_spec) if (a.role == "uniform") ++n_uniform;
    std::vector<double> uniform_vals =
        n_uniform ? read_doubles(dir + "uniforms.bin", n_uniform)
                  : std::vector<double>{};

    // ---- build the launch arguments in arg_spec order ----------------------
    // Storage with stable addresses (reserved so push_back never reallocates),
    // because kernelParams holds pointers into these vectors.
    std::vector<GRefMirror> grefs;   grefs.reserve(sc.arg_spec.size());
    std::vector<ScalarHandle> handles; handles.reserve(sc.arg_spec.size());
    std::vector<double> scalars;     scalars.reserve(sc.arg_spec.size());
    std::vector<void*> params;       params.reserve(sc.arg_spec.size());
    CUdeviceptr d_out = 0;           // remembered for download
    std::size_t uidx = 0;

    auto alloc_upload = [](const std::vector<double>& host) {
        CUdeviceptr d;
        CU_CHECK(cuMemAlloc(&d, host.size() * sizeof(double)));
        CU_CHECK(cuMemcpyHtoD(d, host.data(), host.size() * sizeof(double)));
        return d;
    };

    for (const auto& a : sc.arg_spec) {
        if (a.role == "out") {
            CU_CHECK(cuMemAlloc(&d_out, vlen * sizeof(double)));
            CU_CHECK(cuMemsetD8(d_out, 0, vlen * sizeof(double)));
            grefs.push_back(make_gref(reinterpret_cast<double*>(d_out), N, kEagleAbiDeviceCUDA));
            params.push_back(&grefs.back());
        } else if (a.role == "vec_in") {
            CUdeviceptr d = alloc_upload(read_doubles(dir + a.name + ".bin", vlen));
            grefs.push_back(make_gref(reinterpret_cast<double*>(d), N, kEagleAbiDeviceCUDA));
            params.push_back(&grefs.back());
        } else if (a.role == "per_sample") {
            CUdeviceptr d = alloc_upload(read_doubles(dir + a.name + ".bin", N));
            handles.push_back(make_handle(reinterpret_cast<void*>(d), N, kEagleAbiDeviceCUDA));
            params.push_back(&handles.back());
        } else if (a.role == "terminated") {
            CUdeviceptr d; CU_CHECK(cuMemAlloc(&d, N));
            const std::string tp = dir + "terminated.bin";
            if (file_exists(tp)) {
                std::ifstream f(tp, std::ios::binary);
                std::vector<char> mask((std::istreambuf_iterator<char>(f)),
                                       std::istreambuf_iterator<char>());
                mask.resize(N, 0);
                CU_CHECK(cuMemcpyHtoD(d, mask.data(), N));
            } else {
                CU_CHECK(cuMemsetD8(d, 0, N));   // nothing terminated
            }
            handles.push_back(make_handle(reinterpret_cast<void*>(d), N, kEagleAbiDeviceCUDA));
            params.push_back(&handles.back());
        } else if (a.role == "uniform") {
            scalars.push_back(uniform_vals[uidx++]);
            params.push_back(&scalars.back());
        } else {
            std::fprintf(stderr, "unknown arg role: %s\n", a.role.c_str());
            return 2;
        }
    }

    const std::vector<double> golden = read_doubles(dir + "golden.bin", vlen);
    const unsigned block = 256, grid = (N + block - 1) / block;
    std::vector<double> got(vlen);

    // ---- 1. standalone launch ---------------------------------------------
    CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr,
                            params.data(), nullptr));
    CU_CHECK(cuCtxSynchronize());
    CU_CHECK(cuMemcpyDtoH(got.data(), d_out, vlen * sizeof(double)));
    const double rel_standalone = max_rel(got, golden, N);

    // ---- 2. launch as a node in a captured CUDA graph (replayed twice) -----
    CUstream stream; CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CU_CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    CU_CHECK(cuMemsetD8Async(d_out, 0, vlen * sizeof(double), stream));  // reset accel
    CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream,
                            params.data(), nullptr));
    CUgraph graph; CU_CHECK(cuStreamEndCapture(stream, &graph));
    // cuGraphInstantiateWithFlags has a stable 3-arg signature across CUDA
    // 11.4+ (the bare cuGraphInstantiate macro is remapped between versions).
    CUgraphExec gexec; CU_CHECK(cuGraphInstantiateWithFlags(&gexec, graph, 0));
    CU_CHECK(cuGraphLaunch(gexec, stream));
    CU_CHECK(cuGraphLaunch(gexec, stream));   // replay: reset+accumulate again
    CU_CHECK(cuStreamSynchronize(stream));
    CU_CHECK(cuMemcpyDtoH(got.data(), d_out, vlen * sizeof(double)));
    const double rel_graph = max_rel(got, golden, N);

    // ---- 3. rebuild the vector views through the GRef<->DLPack bridge -------
    // The output and the input arrays are exactly the buffers a
    // deployed host would receive as DLPack tensors from the propagator; round-
    // trip each (GRef -> DLManagedTensorVersioned -> GRef), check the view is
    // then launch from the bridged views to prove they drive the kernel.
    for (auto& g : grefs) {
        int64_t shape[2];
        DLManagedTensorVersioned mt = dlpack_from_gref(g, 3, shape);
        GRefMirror rebuilt = gref_from_dlpack(&mt);
        if (rebuilt.data_ != g.data_ || rebuilt.samples_ != g.samples_ ||
            rebuilt.compStride_ != g.compStride_ ||
            rebuilt.sampleStride_ != g.sampleStride_ ||
            rebuilt.deviceType_ != g.deviceType_) {
            std::fprintf(stderr, "DLPack round-trip mismatch\n");
            return 2;
        }
        g = rebuilt;
    }
    CU_CHECK(cuMemsetD8(d_out, 0, vlen * sizeof(double)));
    CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr,
                            params.data(), nullptr));
    CU_CHECK(cuCtxSynchronize());
    CU_CHECK(cuMemcpyDtoH(got.data(), d_out, vlen * sizeof(double)));
    const double rel_dlpack = max_rel(got, golden, N);

    std::printf("KERNEL=%s\n", sc.kernel.c_str());
    std::printf("STANDALONE_MAX_REL=%.3e\n", rel_standalone);
    std::printf("GRAPH_MAX_REL=%.3e\n", rel_graph);
    std::printf("DLPACK_MAX_REL=%.3e\n", rel_dlpack);
    const bool ok = (rel_standalone < TOL) && (rel_graph < TOL)
                    && (rel_dlpack < TOL);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
