// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "plugin/host_registry.h"

#include "TestBase.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace eagle::plugin;  // the ABI protocol PODs (Sidecar / ArgEntry / handles)

namespace eagle_tests {
namespace HostPluginTest {

// The sidecar the host registry validates + packs against — built in code here,
// exactly what a deployed plugin would stamp for a pure kernel out[i] = a[i]+b[i]:
// a scalar `mutable` output, two `per_sample` inputs, then `nsamples`.
static Sidecar makeAddvecSidecar()
{
    Sidecar sc;
    sc.kernel         = "addvec";
    sc.aether_abi       = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float64";
    sc.arg_spec       = {
        ArgEntry{ "mutable", "out" },
        ArgEntry{ "per_sample", "a" },
        ArgEntry{ "per_sample", "b" },
        ArgEntry{ "nsamples", "n" },
    };
    sc.mutables = { MutableInfo{ "out", "float", 1 } };  // scalar -> flat handle
    return sc;
}

// dlopen the fixture `.so`, bind host buffers by name, run on the host, verify.
// This exercises the full CPU-plugin ABI: dlsym the `<kernel>_host` entry, pack
// the same params[] the device registry builds, and run the plugin's own OpenMP
// loop — with NO CUDA anywhere.
TEST(HostPluginTest, DlopenBindRunAddsVectors)
{
    const std::int32_t n = 4096;
    std::vector<double> a(n), b(n), out(n, -1.0);
    for (std::int32_t i = 0; i < n; ++i) {
        a[i] = double(i);
        b[i] = 2.0 * double(i);
    }

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeAddvecSidecar(), HOST_PLUGIN_ADDVEC_SO);
    reg.bind_handle("out", out.data());
    reg.bind_handle("a", a.data());
    reg.bind_handle("b", b.data());

    const int launched = reg.run(n);
    ASSERT_EQ(launched, 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], 3.0 * double(i));
}

// A disabled registry launches nothing (the enable flag mirrors the device path).
TEST(HostPluginTest, DisabledLaunchesNothing)
{
    const std::int32_t n = 16;
    std::vector<double> a(n, 1.0), b(n, 2.0), out(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeAddvecSidecar(), HOST_PLUGIN_ADDVEC_SO);
    reg.bind_handle("out", out.data());
    reg.bind_handle("a", a.data());
    reg.bind_handle("b", b.data());
    reg.set_enabled(false);

    ASSERT_EQ(reg.run(n), 0);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], -1.0);  // untouched
}

// An ABI-tag mismatch is rejected at load — the artifact/loader boundary guard.
TEST(HostPluginTest, RejectsAbiMismatch)
{
    Sidecar sc  = makeAddvecSidecar();
    sc.aether_abi = "aether-abi/999";
    eagle::cpu::PluginRegistry reg;
    ASSERT_THROW(reg.add_plugin(sc, HOST_PLUGIN_ADDVEC_SO), std::runtime_error);
}

// The sidecar for the matrix-input fixture ``out[i] = trace(M[i])``: a scalar
// `mutable` output, a `mat_in` 3x3 matrix input, then `nsamples`. ``mat_shapes``
// records the declared (R, C) — exactly what a deployed plugin stamps for a mat_in.
static Sidecar makeMattraceSidecar()
{
    Sidecar sc;
    sc.kernel         = "mattrace";
    sc.aether_abi       = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float64";
    sc.arg_spec       = {
        ArgEntry{ "mutable", "out" },
        ArgEntry{ "mat_in", "M" },
        ArgEntry{ "nsamples", "n" },
    };
    sc.mutables   = { MutableInfo{ "out", "float", 1 } };
    sc.mat_shapes = { { "M", { 3, 3 } } };
    return sc;
}

// A flat (R*C, N) SoA 3x3 matrix batch: component (r,c) of sample i at
// M[(r*3 + c) * n + i]. Deterministic fill so the trace has a closed form.
static std::vector<double> makeMatBatch(std::int32_t n)
{
    std::vector<double> M(9 * static_cast<std::size_t>(n));
    for (std::int32_t d = 0; d < 9; ++d)
        for (std::int32_t i = 0; i < n; ++i)
            M[static_cast<std::size_t>(d) * n + i] = double(d) + 0.25 * double(i);
    return M;
}

// The registry PACKS a mat_in through the same GRefMirror as a vector and the plugin
// reads the flat SoA correctly — matrix host parity end to end on the C++ side.
TEST(HostPluginTest, DlopenBindRunMatrixTrace)
{
    const std::int32_t n = 4096;
    std::vector<double> M = makeMatBatch(n);
    std::vector<double> out(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeMattraceSidecar(), HOST_PLUGIN_MATTRACE_SO);
    reg.bind_handle("out", out.data());
    reg.bind_matrix("M", M.data(), std::uint32_t(n), 3, 3);

    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_DOUBLE_EQ(out[i],
            M[std::size_t(0) * n + i] + M[std::size_t(4) * n + i]
                + M[std::size_t(8) * n + i]);
}

// A matrix bound with the wrong (rows, cols) is rejected against the sidecar's
// declared shape at run time — the C++ mirror of the host_launch.py shape guard.
TEST(HostPluginTest, RejectsMatrixShapeMismatch)
{
    const std::int32_t n = 16;
    std::vector<double> M = makeMatBatch(n);
    std::vector<double> out(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeMattraceSidecar(), HOST_PLUGIN_MATTRACE_SO);
    reg.bind_handle("out", out.data());
    reg.bind_matrix("M", M.data(), std::uint32_t(n), 2, 3);  // declared (3, 3)

    ASSERT_THROW(reg.run(n), std::runtime_error);
}

// The sidecar for a matrix *Mutable* ``W`` (dtype "matrix", declared (3, 3)) — the
// same 32-byte GRef ABI as a mat_in, but its shape is carried on the Mutable itself
// (``mutables[].shape``, which parse_sidecar folds into ``mat_shapes``) rather than the
// top-level ``mat_shapes`` object a mat_in uses. Built in code here as parse_sidecar
// would produce it, so the run-time shape guard sees the matrix-Mutable path.
static Sidecar makeMatMutableSidecar()
{
    Sidecar sc;
    sc.kernel         = "mattrace";  // reuse the mattrace `.so` purely to satisfy dlopen
    sc.aether_abi       = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float64";
    sc.arg_spec       = {
        ArgEntry{ "mutable", "W" },
        ArgEntry{ "nsamples", "n" },
    };
    sc.mutables   = { MutableInfo{ "W", "matrix", 9 } };  // R*C = 9
    sc.mat_shapes = { { "W", { 3, 3 } } };  // as parse_sidecar folds from mutables[].shape
    return sc;
}

// A matrix *Mutable* bound with the wrong (rows, cols) is rejected at run against its
// declared (R, C) — the matrix-Mutable arm of ``checked_matrix_gref_`` (role "mutable" +
// dtype "matrix"), a distinct dispatch/shape-source from the mat_in guard above. The
// guard fires while packing params, before the host entry is called, so reusing the
// mattrace `.so` (no matrix-Mutable fixture kernel needed) is sound for the negative path.
TEST(HostPluginTest, RejectsMatrixMutableShapeMismatch)
{
    const std::int32_t n = 16;
    std::vector<double> W(9 * static_cast<std::size_t>(n), 0.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeMatMutableSidecar(), HOST_PLUGIN_MATTRACE_SO);
    reg.bind_matrix("W", W.data(), std::uint32_t(n), 2, 2);  // declared (3, 3)

    ASSERT_THROW(reg.run(n), std::runtime_error);
}

// The sidecar for the lookup fixture ``acc[i] = tab[i % 16]``: a scalar `mutable`
// output, a `lookup` table input, then `nsamples`. ``buffers`` declares ``tab`` (kind
// "lookup", flat ``count``) — exactly what a deployed plugin stamps for a Table[K], and
// what ``declared_tables`` reads to tell the caller what to consolidate.
static Sidecar makeLookupSidecar(std::size_t count = 16)
{
    Sidecar sc;
    sc.kernel         = "lookup";
    sc.aether_abi       = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float64";
    sc.arg_spec       = {
        ArgEntry{ "mutable", "acc" },
        ArgEntry{ "lookup", "tab" },
        ArgEntry{ "nsamples", "n" },
    };
    sc.mutables = { MutableInfo{ "acc", "float", 1 } };  // scalar -> flat handle
    sc.buffers  = { BufferInfo{ "tab", "lookup", "float", count } };
    return sc;
}

// A consolidated lookup table is packed by value as a flat handle over the caller's
// host buffer (no upload — the CPU path reads host memory directly) and the plugin
// gathers from it correctly: lookup host parity end to end on the C++ side.
TEST(HostPluginTest, DlopenConsolidateRunLookup)
{
    const std::int32_t n = 4096;
    std::vector<double> tab(16);
    for (int k = 0; k < 16; ++k)
        tab[k] = 0.5 + 0.1 * double(k);
    std::vector<double> acc(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeLookupSidecar(), HOST_PLUGIN_LOOKUP_SO);

    // declared_tables() reports exactly the (name, count) the caller must consolidate.
    const auto decl = reg.declared_tables();
    ASSERT_EQ(decl.size(), 1u);
    ASSERT_EQ(decl[0].name, "tab");
    ASSERT_EQ(decl[0].count, 16u);

    reg.bind_handle("acc", acc.data());
    reg.consolidate("tab", tab.data(), 16);  // host: record pointer + count, no upload

    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_DOUBLE_EQ(acc[i], tab[i % 16]);
}

// A declared table that was never consolidated fails loudly at run, naming the table —
// the host mirror of the device path's ``test_table_not_consolidated_is_a_clear_error``.
TEST(HostPluginTest, LookupNotConsolidatedIsAClearError)
{
    const std::int32_t n = 16;
    std::vector<double> acc(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeLookupSidecar(), HOST_PLUGIN_LOOKUP_SO);
    reg.bind_handle("acc", acc.data());
    // `tab` deliberately NOT consolidated.
    try {
        reg.run(n);
        FAIL() << "expected run() to throw for the unconsolidated table";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not supplied at consolidation"), std::string::npos) << msg;
        EXPECT_NE(msg.find("tab"), std::string::npos) << msg;
    }
}

// Two plugins declare a table named ``tab`` with DIFFERENT element counts. Tables share
// one global namespace, so declared_tables() must refuse rather than let both bind to
// one undersized buffer — the host mirror of the device path's
// ``test_conflicting_table_shapes_across_plugins_is_error``.
TEST(HostPluginTest, ConflictingTableShapesAcrossPluginsIsError)
{
    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeLookupSidecar(16), HOST_PLUGIN_LOOKUP_SO);
    reg.add_plugin(makeLookupSidecar(8), HOST_PLUGIN_LOOKUP_SO);  // same `tab`, diff count
    try {
        reg.declared_tables();
        FAIL() << "expected declared_tables() to throw on the conflicting counts";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("conflicting element counts"), std::string::npos) << msg;
    }
}

// --- float32 plugins: eagle::cpu::PluginRegistry32 -------------------------------------

static Sidecar f32Sidecar(const std::string& kernel, std::vector<ArgEntry> args)
{
    Sidecar sc;
    sc.kernel         = kernel;
    sc.aether_abi     = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float32";
    sc.arg_spec       = std::move(args);
    sc.mutables       = { MutableInfo{ "out", "float", 1 } };
    return sc;
}

TEST(HostPluginTest, Float32AddvecMatchesNumpyFloat32)
{
    const std::int32_t n = 4096;
    std::vector<float> a(n), b(n), out(n, -1.0f);
    for (std::int32_t i = 0; i < n; ++i) {
        a[i] = 0.1f * float(i);
        b[i] = 1.0f / float(i + 1);
    }
    eagle::cpu::PluginRegistry32 reg;
    reg.add_plugin(f32Sidecar("addvec32", { { "mutable", "out" }, { "per_sample", "a" },
                                           { "per_sample", "b" }, { "nsamples", "n" } }),
                   HOST_PLUGIN_F32_SO);
    reg.bind_handle("out", out.data());
    reg.bind_handle("a", a.data());
    reg.bind_handle("b", b.data());
    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_NEAR(out[i], a[i] + b[i], 1e-6 * std::abs(a[i] + b[i]));
}

TEST(HostPluginTest, Float32MatrixTraceMatchesNumpyFloat32)
{
    const std::int32_t n = 2048;
    std::vector<float> M(9 * std::size_t(n)), out(n, -1.0f);
    for (std::int32_t d = 0; d < 9; ++d)
        for (std::int32_t i = 0; i < n; ++i)
            M[std::size_t(d) * n + i] = float(d) + 0.25f * float(i);
    Sidecar sc = f32Sidecar("mattrace32", { { "mutable", "out" }, { "mat_in", "M" },
                                           { "nsamples", "n" } });
    sc.mat_shapes = { { "M", { 3, 3 } } };
    eagle::cpu::PluginRegistry32 reg;
    reg.add_plugin(sc, HOST_PLUGIN_F32_SO);
    reg.bind_handle("out", out.data());
    reg.bind_matrix("M", M.data(), std::uint32_t(n), 3, 3);
    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i) {
        const float want = M[std::size_t(0) * n + i] + M[std::size_t(4) * n + i] + M[std::size_t(8) * n + i];
        ASSERT_NEAR(out[i], want, 1e-6 * std::abs(want));
    }
}

// vec_in + a real uniform (narrowed once at bind) + a lookup table, all 4-byte floats.
static Sidecar makeVecgain32Sidecar()
{
    Sidecar sc = f32Sidecar("vecgain32", { { "mutable", "out" }, { "vec_in", "x" },
                                          { "uniform", "gain" }, { "lookup", "tab" },
                                          { "nsamples", "n" } });
    sc.buffers = { BufferInfo{ "tab", "lookup", "float", 4 } };
    return sc;
}

TEST(HostPluginTest, Float32VecInUniformAndLookupMatchNumpyFloat32)
{
    const std::int32_t n = 4096;
    std::vector<float> x(n), out(n, -1.0f);
    for (std::int32_t i = 0; i < n; ++i)
        x[i] = 0.01f * float(i);
    std::vector<float> tab = { 0.5f, 1.5f, 2.5f, 3.5f };
    const double gain = 0.3;  // a double at the call, a float in the kernel slot

    eagle::cpu::PluginRegistry32 reg;
    reg.add_plugin(makeVecgain32Sidecar(), HOST_PLUGIN_F32_SO);
    reg.bind_vector("x", x.data(), std::uint32_t(n));
    reg.bind_handle("out", out.data());
    reg.bind_uniform("gain", gain);
    reg.consolidate("tab", tab.data(), 4);
    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i) {
        const float want = x[i] * float(gain) + tab[i % 4];
        ASSERT_NEAR(out[i], want, 1e-6 * std::abs(want));
    }
}

// The scalar must match the registry: a float32 plugin into the float64 registry (and the
// reverse) is refused by name, before dlopen.
TEST(HostPluginTest, Float32PluginIntoFloat64RegistryIsRefusedByName)
{
    eagle::cpu::PluginRegistry reg;
    try {
        reg.add_plugin(makeVecgain32Sidecar(), HOST_PLUGIN_F32_SO);
        FAIL() << "expected the scalar mismatch to be refused";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("float32"), std::string::npos) << msg;
        EXPECT_NE(msg.find("PluginRegistry32"), std::string::npos) << msg;
    }
}

TEST(HostPluginTest, Float64PluginIntoFloat32RegistryIsRefusedByName)
{
    eagle::cpu::PluginRegistry32 reg;
    ASSERT_THROW(reg.add_plugin(makeAddvecSidecar(), HOST_PLUGIN_ADDVEC_SO), std::runtime_error);
}

// The registry owns the finished_count cell: no caller table, zeroed at the start of every run.
static Sidecar makeCountfinSidecar()
{
    Sidecar sc = f32Sidecar("countfin32", { { "mutable", "k" }, { "terminated", "terminated" },
                                           { "lookup", "finished_count" }, { "nsamples", "n" } });
    sc.mutables = { MutableInfo{ "k", "float", 1 } };
    return sc;
}

TEST(HostPluginTest, FinishedCounterIsOwnedAndZeroedEachRun)
{
    const std::int32_t n = 1000;
    std::vector<float> k(n);
    for (std::int32_t i = 0; i < n; ++i)
        k[i] = (i % 4 == 0) ? 1.0f : -1.0f;
    std::vector<unsigned char> mask(n, 0);

    eagle::cpu::PluginRegistry32 reg;
    reg.add_plugin(makeCountfinSidecar(), HOST_PLUGIN_F32_SO);
    reg.bind_handle("k", k.data());
    reg.bind_handle("terminated", mask.data());
    ASSERT_EQ(reg.run(n), 1);  // no consolidate call: the registry supplies the counter
    EXPECT_EQ(reg.finished_count(), 250u);
    ASSERT_EQ(reg.run(n), 1);
    EXPECT_EQ(reg.finished_count(), 250u);  // zeroed again, not 500
}

TEST(HostPluginTest, ConsolidatingTheFinishedCounterIsRefusedByName)
{
    eagle::cpu::PluginRegistry32 reg;
    std::vector<float> t(1, 0.0f);
    try {
        reg.consolidate("finished_count", t.data(), 1);
        FAIL() << "expected the reserved name to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("finished_count"), std::string::npos);
    }
}

}  // namespace HostPluginTest
}  // namespace eagle_tests
