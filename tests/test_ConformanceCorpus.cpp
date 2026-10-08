// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The shared conformance corpus — C++ driver.
//
// `eagle/tests/conformance/` holds language-neutral fixture JSONs, each paired with
// an expectation file (`{level, expect, error_substring, loaders, overrides}`). TWO
// drivers read that one corpus and assert IDENTICAL outcomes: this file, and the
// pytest `eagle/python/tests/test_conformance_corpus.py`. A divergence between the
// C++ and Python validators — a check present in one language and missing in the
// other, or the same rejection carrying a different message — therefore becomes a
// mechanical test failure rather than an inspection finding
//
//
// Each reject fixture stores ONE `error_substring` that BOTH drivers assert, which is
// what pins message parity across the language boundary.
//
// Validation tier only: no GPU, no CUDA, no real artifact bytes. That is a property
// of the product, not a convenience of the test — `validate_sidecar` / the manifest
// pre-load checks run BEFORE any module load or `dlopen`, so a rejected artifact is
// refused without loading anything. A reject case can therefore
// point `add_plugin`/`load` at a stub `.so` path that does not exist: if the gate
// ever regresses to validate-after-load, the loader reaches the stub and throws a
// dlopen error instead of the expected validation message, and this driver fails.
//
// This is a `.cpp` with no CUDA in it, and it is compiled and run in BOTH build
// modes — `tests/CMakeLists.txt` adds it to the source list explicitly, outside the
// mode-conditional glob (which takes `.cpp` only under EAGLE_CPP_MODE and `.cu` only
// otherwise). Both standing C++ gates therefore carry the corpus. Because this TU has
// no CUDA, it can drive only the HOST registry's manifest loader
// (`eagle::cpu::PluginRegistry::load`) — the device registry's `from_manifest`
// (`plugin_registry/registry.h`) needs `<cuda.h>` + a libcuda link and is exercised
// separately by NAMED CUDA-only `.cu` unit tests in `test_PluginRegistryManifest.cu`
// (an outside-coverage claim must
// name the test, not just the file): `pattern`
// (`RejectsUnknownManifestPatternNeuralBlock`), `format`
// (`RejectsUnknownManifestFormat`), and the manifest-level `schema_version` gate
// (`RejectsNewerManifestSchemaVersion`) — the latter two added alongside this fix
// after an audit found both device call sites, and this very claim,
// untested. A manifest-level fixture therefore names its C++ loader class
// `cpp_load_manifest`, mapped here to
// the host registry; Python's counterpart is `py_load_manifest`
// (`eagle.registry.load_manifest`). Each driver silently SKIPS the other's exclusive
// loader-class name when it appears in a fixture's `loaders` list — that is how a
// cross-language ASYMMETRY is checked with one shared fixture.

#include "plugin/host_registry.h"
#include "plugin/plugin_registry/manifest.h"
#include "plugin/sidecar.h"

// Deliberately NOT "TestBase.h": that header pulls in eagle's
// device headers, whose __device__/__host__ attributes only nvcc parses — and in
// CUDA mode this `.cpp` is compiled by CXX (g++), not nvcc. Including it would make
// this TU unbuildable in exactly the build mode it was added to cover. Nothing here
// needs the `eagle_tests::Test` fixture; bare gtest is the whole dependency.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace eagle::plugin;

namespace eagle_tests {
namespace ConformanceCorpusTest {

// The corpus rows landed so far. An early pass landed { 2, 4, 6, 7, 8, 11 };
// step 1b's atomic widening added row 1 (the `neural_block` golden + its VJP companion)
// and row 5 (the unknown `scatter_policy`); a later volume dispatch adds rows
// { 3, 9, 10, 17, 18, 19, 20, 21 } (the flagged-call fixtures check) and repairs
// the gap by adding { 13, 14, 15, 16 } — those four sat on disk, unnamed by this
// floor, since the first pass. A corpus that silently collects
// nothing reports green and proves nothing — precisely the defect class this machinery
// exists to close — so both drivers assert a floor, and the floor names its rows.
// KEEP IN SYNC with test_conformance_corpus.py.
//
// KEEP IN SYNC with test_conformance_corpus.py: the cross-language
// message contract is exactly one shared substring per row, drawn from the
// language-invariant LEADING clause of each message. Full-message renderings (set
// orderings, `got <value>` tails, etc.) are per-language presentation and are
// PERMANENTLY non-contractual — no `error_substring` may ever span one.
//
// A later pass adds rows { 22, 23, 24 }: the exec-ref
// closed-set hardening, the neural_block integer-field uint32 cap, and the
// `accumulate` scatter_policy acceptance golden.
// A later pass adds row 25: neural-block MANIFEST carriage -- the wire
// dangling-exec-ref reject + the orphaned-blocks[] guard (row 12 stays retired).
static const std::set<int> kRequiredRows = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
};

// A path that deliberately does not exist. Reaching it means validate-before-load
// regressed; `add_plugin`/`load` then throws a dlopen error, not the expected
// validation one.
static const char* const kStubSo = "/nonexistent/conformance-stub.so";

// Loader-class names that belong exclusively to the OTHER driver (the Python pytest
// one). See the file banner: this driver silently skips these.
static bool notMine(const std::string& loader) { return loader == "py_load_manifest"; }

struct Expectation {
    std::string stem;
    int         row = 0;
    std::string level;
    std::string expect;           // "accept" | "reject"
    std::string error_substring;  // meaningful only when expect == "reject"
    std::vector<std::string> loaders;
    // The raw `{ ... }` text of the `overrides` object (empty string if absent or
    // empty) — parsed on demand per loader by overrideFor() below.
    std::string overridesRaw;
};

// The corpus directory, injected by CMake (see tests/CMakeLists.txt).
static std::string corpusDir() { return EAGLE_CONFORMANCE_CORPUS_DIR; }

// Every `*.expect.json` basename in the corpus directory, sorted.
static std::vector<std::string> expectationStems()
{
    std::vector<std::string> stems;
    DIR* d = opendir(corpusDir().c_str());
    if (!d) return stems;
    const std::string suffix = ".expect.json";
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > suffix.size()
            && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
            stems.push_back(n.substr(0, n.size() - suffix.size()));
    }
    closedir(d);
    std::sort(stems.begin(), stems.end());
    return stems;
}

// The `{ ... }` object value of `key` in `s` (searched from `from`); "" if the key
// or its object is not found. Shared by the `overrides` extraction below.
static std::string boundedObject(const std::string& s, const std::string& key,
                                  std::size_t from = 0)
{
    auto k = s.find("\"" + key + "\"", from);
    if (k == std::string::npos) return "";
    auto ob = s.find('{', k);
    if (ob == std::string::npos) return "";
    int d = 0;
    std::size_t oe = std::string::npos;
    for (std::size_t i = ob; i < s.size(); ++i) {
        if (s[i] == '{') ++d;
        else if (s[i] == '}' && --d == 0) { oe = i; break; }
    }
    if (oe == std::string::npos) return "";
    return s.substr(ob, oe - ob + 1);
}

// The depth-1 quoted keys of a `{ ... }` JSON object string `obj` (keys of nested
// objects/arrays are skipped by depth tracking). Used to enumerate the loader names
// an `overrides` object names, and to validate a per-loader override's own keys.
static std::vector<std::string> topLevelKeys(const std::string& obj)
{
    std::vector<std::string> keys;
    int depth = 0;
    for (std::size_t i = 0; i < obj.size(); ++i) {
        const char c = obj[i];
        if (c == '{' || c == '[') { ++depth; continue; }
        if (c == '}' || c == ']') { --depth; continue; }
        if (depth == 1 && c == '"') {
            auto q2 = obj.find('"', i + 1);
            if (q2 == std::string::npos) break;
            auto after = obj.find_first_not_of(" \t\r\n", q2 + 1);
            if (after != std::string::npos && obj[after] == ':')
                keys.push_back(obj.substr(i + 1, q2 - i - 1));
            i = q2;
        }
    }
    return keys;
}

// One loader's parsed `overrides` entry ("per-loader-class
// overrides"): a LOCKED asymmetry pins a DIFFERENT expect/substring for one loader
// than the fixture's top-level default.
struct LoaderOverride {
    bool has_expect = false;
    std::string expect;
    bool has_substring = false;
    std::string error_substring;
};

// Parse the override entry for `loader` out of `x`'s raw `overrides` object, if any.
// Fails loudly on an override key this driver does not understand — the counterpart
// to the old "populated overrides -> hard fail" guard: NOW that overrides are
// parsed, the guard narrows to "an override shape we don't recognize", rather than
// "any override at all" (an earlier placeholder, since replaced).
static LoaderOverride overrideFor(const Expectation& x, const std::string& loader)
{
    LoaderOverride o;
    if (x.overridesRaw.empty()) return o;
    const std::string sub = boundedObject(x.overridesRaw, loader);
    if (sub.empty()) return o;
    for (const auto& k : topLevelKeys(sub)) {
        if (k == "expect") {
            o.has_expect = true;
            o.expect = _string_value(sub, "expect");
        } else if (k == "error_substring") {
            o.has_substring = true;
            o.error_substring = _string_value(sub, "error_substring");
        } else {
            throw std::runtime_error("conformance fixture '" + x.stem +
                "': override for loader '" + loader + "' has an unrecognized key '" +
                k + "'; only 'expect'/'error_substring' are implemented — extend "
                "overrideFor() before landing this override shape");
        }
    }
    return o;
}

// Read one expectation file. Reuses the same hand-written scanners `parse_sidecar`
// uses (`plugin/sidecar.h`) rather than pulling in a JSON dependency — the corpus
// files are our own fixed, machine-read schema, exactly like the sidecars.
static Expectation readExpectation(const std::string& stem)
{
    const std::string s = _slurp(corpusDir() + "/" + stem + ".expect.json");
    Expectation x;
    x.stem  = stem;
    x.level = _string_value(s, "level");
    // Every key read below is declared BEFORE the free-form `why` array in each
    // expectation file, and these scanners take the FIRST match — so prose in `why`
    // can never shadow a real key.
    if (auto r = s.find("\"row\""); r != std::string::npos)
        x.row = std::atoi(s.c_str() + s.find(':', r) + 1);
    x.expect  = _string_value(s, "expect");
    x.loaders = _string_array(s, "loaders");
    if (x.expect == "reject") x.error_substring = _string_value(s, "error_substring");
    // The `overrides` OBJECT's raw text, parsed per-loader on demand by
    // overrideFor(). Every top-level key (a loader name) must be one of this
    // fixture's declared `loaders` — an override for an undeclared loader is
    // almost certainly a typo, and silently ignoring it would be exactly the
    // "asserts the WRONG outcome" hazard DEV-4 flagged.
    x.overridesRaw = boundedObject(s, "overrides");
    for (const auto& loaderKey : topLevelKeys(x.overridesRaw)) {
        if (std::find(x.loaders.begin(), x.loaders.end(), loaderKey) == x.loaders.end())
            throw std::runtime_error("conformance fixture '" + stem +
                "': overrides names loader '" + loaderKey +
                "', which is not in this fixture's `loaders` list");
    }
    return x;
}

// Drive one named loader class over the fixture. Returns false (a no-op) for a
// loader-class name that belongs exclusively to the Python driver — see notMine().
// The loader names are shared with the Python driver so a fixture's `loaders` list
// and its per-loader `overrides` are portable across both languages.
static bool runLoader(const std::string& loader, const std::string& stem,
                       const std::string& level)
{
    if (notMine(loader)) return false;
    if (level == "sidecar") {
        const std::string path = corpusDir() + "/" + stem + ".sidecar.json";
        const Sidecar     sc   = parse_sidecar(path);
        if (loader == "shared_validator") {
            validate_sidecar(sc, "plugin '" + sc.kernel + "'");
        } else if (loader == "host_add_plugin") {
            // The manifest-BYPASSING entry point, and a door found
            // open on the Python side. Its only schema gate IS `validate_sidecar`.
            eagle::cpu::PluginRegistry reg;
            reg.add_plugin(sc, kStubSo);
        } else {
            throw std::runtime_error("unknown conformance loader class '" + loader +
                "' at sidecar level");
        }
    } else if (level == "manifest") {
        const std::string path = corpusDir() + "/" + stem + ".manifest.json";
        if (loader == "cpp_load_manifest") {
            const Manifest m = parse_manifest(path);
            eagle::cpu::PluginRegistry reg;
            reg.load(m, corpusDir());
        } else {
            throw std::runtime_error("unknown conformance loader class '" + loader +
                "' at manifest level");
        }
    } else {
        throw std::runtime_error("unknown conformance fixture level '" + level + "'");
    }
    return true;
}

// ----------------------------------------------------------------------- //
// Corpus-integrity guards. Without these a mis-pathed or empty corpus would
// report green while asserting nothing.
// ----------------------------------------------------------------------- //
TEST(ConformanceCorpusTest, CorpusCollectsItsRequiredRows)
{
    const auto stems = expectationStems();
    ASSERT_FALSE(stems.empty())
        << "the conformance corpus collected ZERO fixtures from " << corpusDir()
        << ". A corpus that collects nothing reports green and proves nothing — "
           "check EAGLE_CONFORMANCE_CORPUS_DIR and the *.expect.json naming.";
    std::set<int> rows;
    for (const auto& stem : stems) rows.insert(readExpectation(stem).row);
    for (const int want : kRequiredRows)
        EXPECT_TRUE(rows.count(want) > 0)
            << "conformance corpus lost required corpus row " << want;
}

TEST(ConformanceCorpusTest, EveryFixturePairsWithAnExpectation)
{
    for (const auto& stem : expectationStems()) {
        const Expectation x = readExpectation(stem);
        const std::string suffix = (x.level == "manifest") ? ".manifest.json"
                                                             : ".sidecar.json";
        const std::string path = corpusDir() + "/" + stem + suffix;
        std::ifstream     f(path, std::ios::binary);
        EXPECT_TRUE(f.good())
            << "expectation '" << stem << ".expect.json' has no matching " << suffix
            << " fixture at " << path << " (it would assert nothing)";
    }
}

// ----------------------------------------------------------------------- //
// The corpus itself: every fixture, every loader class it names.
// ----------------------------------------------------------------------- //
TEST(ConformanceCorpusTest, EveryFixtureMatchesItsExpectation)
{
    const auto stems = expectationStems();
    ASSERT_FALSE(stems.empty()) << "conformance corpus collected ZERO fixtures";

    for (const auto& stem : stems) {
        const Expectation x = readExpectation(stem);
        SCOPED_TRACE("conformance fixture: " + stem);

        ASSERT_TRUE(x.level == "sidecar" || x.level == "manifest")
            << "unknown fixture level '" << x.level << "'";
        ASSERT_FALSE(x.loaders.empty())
            << "fixture '" << stem << "' declares no loaders, so it asserts nothing";

        bool ranAny = false;
        for (const auto& loader : x.loaders) {
            if (notMine(loader)) continue;  // the Python driver's exclusive loader
            ranAny = true;
            SCOPED_TRACE("loader class: " + loader);
            const LoaderOverride ov = overrideFor(x, loader);
            const std::string expect = ov.has_expect ? ov.expect : x.expect;
            const std::string substring =
                ov.has_substring ? ov.error_substring : x.error_substring;

            if (expect == "accept") {
                EXPECT_NO_THROW(runLoader(loader, stem, x.level));
            } else if (expect == "reject") {
                ASSERT_FALSE(substring.empty())
                    << "a reject row needs an error_substring";
                try {
                    runLoader(loader, stem, x.level);
                    ADD_FAILURE() << "expected a rejection carrying '" << substring
                                  << "'";
                } catch (const std::runtime_error& e) {
                    const std::string msg = e.what();
                    EXPECT_NE(msg.find(substring), std::string::npos)
                        << "message parity break: expected substring '" << substring
                        << "' but got: " << msg;
                }
            } else {
                ADD_FAILURE() << "unknown expect value '" << expect << "'";
            }
        }
        EXPECT_TRUE(ranAny) << "fixture '" << stem << "': every declared loader "
            "belongs to the other driver — this fixture asserts nothing in C++; "
            "check its `loaders` list";
    }
}

// ----------------------------------------------------------------------- //
// The top-level key order is a WRITER-side
// contract (raptor.schema.validate_manifest, golden-carried); the C++
// scanner (parse_manifest, plugin/plugin_registry/manifest.h) is a
// deliberately TOLERANT reader — every field is located via
// `s.find("\"key\"")`, independent of position (confirmed by full read,
// Fix that half of the strict-writer/tolerant-reader
// contract here (this TU has no CUDA and is compiled+run in BOTH build
// modes — see the file banner above): a manifest whose top-level keys are
// SHUFFLED still parses successfully and every field reads back correctly,
// so a future C++ tightening is a conscious contract change, not silent
// drift. No change to parse_manifest or validate_manifest semantics — this
// pins current behavior, it does not alter it.
// ----------------------------------------------------------------------- //
TEST(ConformanceCorpusTest, ParseManifestToleratesShuffledTopLevelKeyOrder)
{
    // The canonical order is (schema_version, pattern, aether_abi, plugins)
    // — raptor's `TOP_LEVEL_KEY_ORDER`. This golden
    // reorders every top-level key (plugins last instead of first, aether_abi
    // last instead of third) plus one entry's own subkey order, adversarially.
    const std::string manifestJson = R"({
  "pattern": "vector",
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "plugins": [
    {
      "artifact": "gravity.ptx",
      "id": "gravity",
      "format": "ptx",
      "sidecar": "gravity.json",
      "enabled": true,
      "order": 0
    }
  ]
})";
    const std::string path = ::testing::TempDir() + "d5_shuffled_key_order.json";
    std::ofstream(path, std::ios::binary) << manifestJson;

    const Manifest m = parse_manifest(path);

    EXPECT_EQ(m.aether_abi, "aether-abi/1");
    EXPECT_EQ(m.schema_version, 1);
    EXPECT_EQ(m.pattern, "vector");
    ASSERT_EQ(m.plugins.size(), 1u);
    EXPECT_EQ(m.plugins[0].id, "gravity");
    EXPECT_EQ(m.plugins[0].artifact, "gravity.ptx");
    EXPECT_EQ(m.plugins[0].sidecar, "gravity.json");
    EXPECT_EQ(m.plugins[0].format, "ptx");
    EXPECT_EQ(m.plugins[0].order, 0);
    EXPECT_TRUE(m.plugins[0].enabled);
}

}  // namespace ConformanceCorpusTest
}  // namespace eagle_tests
