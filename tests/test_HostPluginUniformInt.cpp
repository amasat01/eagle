// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "plugin/host_registry.h"

#include "TestBase.h"

#include <cstdint>
#include <string>
#include <vector>

// The int-role phase — the CPU host registry's
// ADDITIVE int64 uniform path, beside the float64 one.
//
// WHAT THE INT WIDTH IS. A generated kernel's integer quantity is the code generator's
// ``using Int = long long;``, so an integer
// uniform's kernel parameter is ``AETHER_GRID_CONSTANT() Int p_<name>`` — 8 SIGNED bytes.
// ``eagle::plugin::UniformInt`` (plugin/plugin_registry/uniform_binding.h) is that
// type, and ``IntUniformCarriesTheFullSixtyFourBitWidth`` below gates the width with a
// value a 32-bit slot could not carry. (aether has no ``Int`` typedef; its ``idx_t`` /
// ``dims_t`` are 32-bit UNSIGNED index types, which is what the ``nsamples`` role
// packs — not a user integer.)
//
// WHAT IS AND IS NOT DETECTABLE HERE. An ``arg_spec`` row is a ``[role, name]`` pair
// with NO dtype (sidecar.h ``ArgEntry``), and the C++ sidecar reader does not parse the
// sidecar's ``params`` block, so a uniform's type is the one the CALLER declares by
// choosing a binder — exactly as ``bind_vector`` vs ``bind_handle`` already work. The
// registry therefore cannot catch a caller who binds float64 for an int64 slot and
// nothing else; what it CAN and MUST catch is a name given TWO types, which would
// leave the packer to choose which 8 bytes the kernel meant. Both directions of that
// are refused at bind time, before anything is recorded, and both are pinned below.
// (When the sidecar's ``params`` block becomes decl-carrying ``{name, dtype}`` and
// ``parse_sidecar`` grows a reader, the artifact-vs-caller cross-check belongs in
// ``uniform_binding.h`` beside this one — see its header note.)
using namespace eagle::plugin;  // the ABI protocol PODs (Sidecar / ArgEntry / handles)

namespace eagle_tests {
namespace HostPluginUniformIntTest {

// The sidecar for the int-uniform fixture ``out[i] = double(k)``: a scalar `mutable`
// output, one int64 `uniform`, then `nsamples`. Note that NOTHING in this sidecar says
// "int" — that is the finding this window recorded, not an omission: the arg_spec row
// is a [role, name] pair, so the int-ness lives entirely in which binder the caller
// uses (and, on the wire, in the kernel's own compiled signature).
static Sidecar makeIntUniformSidecar()
{
    Sidecar sc;
    sc.kernel         = "intuniform";
    sc.aether_abi       = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float64";
    sc.arg_spec       = {
        ArgEntry{ "mutable", "out" },
        ArgEntry{ "uniform", "k" },
        ArgEntry{ "nsamples", "n" },
    };
    sc.mutables = { MutableInfo{ "out", "float", 1 } };  // scalar -> flat handle
    return sc;
}

// The sidecar for the MIXED fixture ``out[i] = a[i] * gain + double(k)``: a float64
// uniform and an int64 uniform in ONE arg_spec, interleaved with a per_sample input.
static Sidecar makeMixedSidecar()
{
    Sidecar sc;
    sc.kernel         = "mixedscale";
    sc.aether_abi       = EAGLE_AETHER_ABI;
    sc.schema_version = kPluginSchemaVersion;
    sc.scalar_type    = "float64";
    sc.arg_spec       = {
        ArgEntry{ "mutable", "out" },
        ArgEntry{ "per_sample", "a" },
        ArgEntry{ "uniform", "gain" },
        ArgEntry{ "uniform", "k" },
        ArgEntry{ "nsamples", "n" },
    };
    sc.mutables = { MutableInfo{ "out", "float", 1 } };
    return sc;
}

// Registration + binding + packing, end to end: an int64 uniform bound through
// ``bind_uniform_int`` reaches the plugin as the SAME 8 signed bytes. The value is
// deliberately one whose float64 BIT PATTERN is nothing like it (7 as a double's bits
// is a denormal ~3.5e-323), so a registry that packed this slot as a double would fail
// this assertion by many orders of magnitude rather than pass by luck.
TEST(HostPluginUniformIntTest, DlopenBindRunIntUniform)
{
    const std::int32_t n = 4096;
    std::vector<double> out(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeIntUniformSidecar(), HOST_PLUGIN_INTUNIFORM_SO);
    reg.bind_handle("out", out.data());
    reg.bind_uniform_int("k", 7);

    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], 7.0);
}

// THE WIDTH GATE. 2^32 + 1 survives the round trip only if the slot really is 64 bits
// wide: a 32-bit uniform truncates it to 1, and the sign leg (-3) pins that the type is
// SIGNED. This is the executable half of the "which Int?" finding — a 32-bit
// unsigned ``idx_t`` would fail both legs.
TEST(HostPluginUniformIntTest, IntUniformCarriesTheFullSixtyFourBitWidth)
{
    const std::int32_t n = 64;
    std::vector<double> out(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeIntUniformSidecar(), HOST_PLUGIN_INTUNIFORM_SO);
    reg.bind_handle("out", out.data());

    const UniformInt wide = (UniformInt(1) << 32) + 1;  // 4294967297
    reg.bind_uniform_int("k", wide);
    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], double(wide));  // a 32-bit slot would report 1.0

    reg.bind_uniform_int("k", -3);  // re-binding the SAME kind overwrites, as always
    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], -3.0);
}

// A MIXED kernel: float64 and int64 uniforms in one arg_spec, interleaved with a
// per_sample input. The registry packs the two kinds from two separate stable-storage
// vectors, so this is what proves each lands in its OWN params[] slot, in arg_spec
// order, with the right interpretation — and that adding the int arm did not disturb
// the float64 one.
TEST(HostPluginUniformIntTest, DlopenBindRunMixedFloatAndIntUniforms)
{
    const std::int32_t n = 1024;
    std::vector<double> a(n), out(n, -1.0);
    for (std::int32_t i = 0; i < n; ++i)
        a[i] = 0.5 * double(i);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeMixedSidecar(), HOST_PLUGIN_INTUNIFORM_SO);
    reg.bind_handle("out", out.data());
    reg.bind_handle("a", a.data());
    reg.bind_uniform("gain", 0.25);
    reg.bind_uniform_int("k", 11);

    ASSERT_EQ(reg.run(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        ASSERT_DOUBLE_EQ(out[i], a[i] * 0.25 + 11.0);
}

// REFUSAL DIRECTION 1 — int64 over an existing float64 binding of the same name. The
// throw fires at BIND time (before anything is recorded), names both kinds and both
// binders, and needs no plugin at all: the binding maps are registry-level.
TEST(HostPluginUniformIntTest, RefusesIntBindingOverAFloat64Uniform)
{
    eagle::cpu::PluginRegistry reg;
    reg.bind_uniform("k", 1.5);
    try {
        reg.bind_uniform_int("k", 7);
        FAIL() << "expected bind_uniform_int to refuse the double-typed name";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("uniform 'k'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("already bound as float64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("cannot also be bound as int64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("bind_uniform_int"), std::string::npos) << msg;
    }
}

// REFUSAL DIRECTION 2 — float64 over an existing int64 binding of the same name. The
// mirror image, because a one-directional guard is exactly the kind that lets the
// second migration order through.
TEST(HostPluginUniformIntTest, RefusesFloat64BindingOverAnIntUniform)
{
    eagle::cpu::PluginRegistry reg;
    reg.bind_uniform_int("k", 7);
    try {
        reg.bind_uniform("k", 1.5);
        FAIL() << "expected bind_uniform to refuse the int-typed name";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("uniform 'k'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("already bound as int64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("cannot also be bound as float64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("bind_uniform"), std::string::npos) << msg;
    }
}

// The refusal is name-scoped, not global: two DIFFERENT names may take different
// kinds (which is exactly what the mixed kernel above needs), and the guard must not
// turn into "this registry is float64-only once you bind one float64 uniform".
TEST(HostPluginUniformIntTest, DifferentNamesMayTakeDifferentKinds)
{
    eagle::cpu::PluginRegistry reg;
    reg.bind_uniform("gain", 0.25);
    reg.bind_uniform_int("k", 11);
    reg.bind_uniform("gain", 0.5);   // same kind, overwrites
    reg.bind_uniform_int("k", 12);   // same kind, overwrites
    SUCCEED();
}

// The unchanged float64 contract, restated on the int-aware code path: a uniform bound
// through NEITHER binder still fails with the original "unbound uniform" wording. The
// additive rule made visible — the float64 arm's behaviour is what it always was.
TEST(HostPluginUniformIntTest, UnboundUniformKeepsItsOriginalError)
{
    const std::int32_t n = 16;
    std::vector<double> out(n, -1.0);

    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(makeIntUniformSidecar(), HOST_PLUGIN_INTUNIFORM_SO);
    reg.bind_handle("out", out.data());
    // `k` deliberately NOT bound, through either binder.
    try {
        reg.run(n);
        FAIL() << "expected run() to throw for the unbound uniform";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("unbound uniform"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'k'"), std::string::npos) << msg;
    }
}

}  // namespace HostPluginUniformIntTest
}  // namespace eagle_tests
