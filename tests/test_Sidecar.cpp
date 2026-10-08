// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "plugin/sidecar.h"

#include "TestBase.h"

#include <fstream>
#include <string>

// Parse + validate coverage for the optional ``derivative`` block a generated VJP/JVP
// artifact's sidecar carries (schema-v1 additive field). The block is metadata
// only — the launch ignores it — so these tests exercise parse_sidecar / validate_sidecar
// directly over hand-built sidecar JSON, no plugin load. Phase A is recompute-only: a
// populated ``residuals`` list is rejected, naming the schema-bump rule.
using namespace eagle::plugin;

namespace eagle_tests {
namespace SidecarTest {

// A minimal pure-kernel sidecar (the required ``kernel`` + ``arg_spec`` keys); ``extra``
// is spliced in as an additional top-level key (e.g. a ``derivative`` block), or empty.
static std::string sidecarJson(const std::string& extra)
{
    return std::string(R"({
  "format": "host",
  "schema_version": 1,
  "pattern": "pure",
  "aether_abi": "aether-abi/1",
  "scalar_type": "float64",
  "kernel": "raptor_kernel",
  "vector_inputs": [],
  "params": ["s"],
  "per_sample": ["x", "e_bar"],
  "mutables": [
    {"name": "x_bar", "dtype": "float", "width": 1},
    {"name": "s_bar", "dtype": "float", "width": 1}
  ],
  "arg_spec": [
    ["mutable", "x_bar"], ["mutable", "s_bar"],
    ["per_sample", "x"], ["per_sample", "e_bar"],
    ["uniform", "s"], ["terminated", "terminated"], ["nsamples", "nsamples"]
  ],
  "buffers": [])")
        + extra + "\n}\n";
}

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

// A well-formed Phase-A derivative block -> parsed field-for-field, validates cleanly.
TEST(SidecarTest, ParsesDerivativeBlock)
{
    const std::string extra = R"(,
  "derivative": {
    "kind": "vjp",
    "primal": "poly",
    "wrt": ["x", "s"],
    "residual_policy": "recompute",
    "residuals": []
  })";
    const std::string path = writeTmp("deriv_vjp.json", sidecarJson(extra));
    const Sidecar sc = parse_sidecar(path);

    ASSERT_TRUE(sc.derivative.present);
    EXPECT_EQ(sc.derivative.kind, "vjp");
    EXPECT_EQ(sc.derivative.primal, "poly");
    ASSERT_EQ(sc.derivative.wrt.size(), 2u);
    EXPECT_EQ(sc.derivative.wrt[0], "x");
    EXPECT_EQ(sc.derivative.wrt[1], "s");
    EXPECT_EQ(sc.derivative.residual_policy, "recompute");
    EXPECT_FALSE(sc.derivative.residuals_populated);
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'poly_vjp'"));
}

// A JVP block parses just the same (kind-tagged), and exposes its wrt list.
TEST(SidecarTest, ParsesJvpDerivativeBlock)
{
    const std::string extra = R"(,
  "derivative": {"kind": "jvp", "primal": "poly", "wrt": ["x"], "residual_policy": "recompute", "residuals": []})";
    const Sidecar sc = parse_sidecar(writeTmp("deriv_jvp.json", sidecarJson(extra)));
    ASSERT_TRUE(sc.derivative.present);
    EXPECT_EQ(sc.derivative.kind, "jvp");
    ASSERT_EQ(sc.derivative.wrt.size(), 1u);
    EXPECT_EQ(sc.derivative.wrt[0], "x");
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'poly_jvp'"));
}

// An ordinary (primal / non-derivative) sidecar leaves the block inert — backward-lenient.
TEST(SidecarTest, NoDerivativeBlockLeavesItInert)
{
    const Sidecar sc = parse_sidecar(writeTmp("no_deriv.json", sidecarJson("")));
    EXPECT_FALSE(sc.derivative.present);
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'primal'"));
}

// Phase A is recompute-only: a populated residuals list is rejected at validate, naming
// the schema-bump rule (a future revision that stashes residuals bumps schema_version).
TEST(SidecarTest, RejectsPopulatedResiduals)
{
    const std::string extra = R"(,
  "derivative": {
    "kind": "vjp", "primal": "poly", "wrt": ["x", "s"],
    "residual_policy": "recompute", "residuals": [{"name": "r0", "width": 1}]
  })";
    const Sidecar sc = parse_sidecar(writeTmp("bad_residuals.json", sidecarJson(extra)));
    EXPECT_TRUE(sc.derivative.residuals_populated);
    EXPECT_THROW(validate_sidecar(sc, "plugin 'poly_vjp'"), std::runtime_error);
}

// An unknown derivative kind is rejected at validate (kind ∈ {vjp, jvp}).
TEST(SidecarTest, RejectsUnknownKind)
{
    const std::string extra = R"(,
  "derivative": {"kind": "grad", "primal": "poly", "wrt": ["x"], "residual_policy": "recompute", "residuals": []})";
    const Sidecar sc = parse_sidecar(writeTmp("bad_kind.json", sidecarJson(extra)));
    ASSERT_TRUE(sc.derivative.present);
    EXPECT_THROW(validate_sidecar(sc, "plugin 'poly_x'"), std::runtime_error);
}

}  // namespace SidecarTest
}  // namespace eagle_tests
