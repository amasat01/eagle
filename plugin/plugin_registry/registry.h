// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The independent plugin registry (host-side, ABI-only).
//
// This is the piece a deployment drops in at its force-accumulation opening. It
// is *independent* of any graph machinery: it does NOT own a stream, never calls
// cudaStreamBeginCapture/EndCapture, and knows nothing about any downstream Graph type. The
// host (a downstream consumer, or the inject_demo here) opens its own capture; at the
// opening it calls registry.inject(stream, N). The registry then, for each
// enabled plugin in manifest order, packs the kernel's by-value arguments from
// host-bound PRE-ALLOCATED device buffers and issues a Driver-API cuLaunchKernel
// on that stream. Because stream capture is a driver-level property of the
// shared CUstream, each launch is recorded as a graph node — one more
// outVec[i] += plugin_contribution in the captured pipeline.
//
// Deliberately depends on nothing beyond the ABI: accepting kernelized plugins needs only
// the binary ABI (gref_abi.h PODs), the artifacts, and their sidecars.
//
// Scope note: for the per-sample interface buffers (out/position/velocity/
// mass/.../terminated) the registry only *references* memory the host already
// allocated and bound by name — it never allocates those. The ONE thing it does
// own is *consolidated lookup tables*: read-only "bring your own data" buffers a
// plugin declares (a lookup table or a shared constant — the extension hatch).
// `consolidate(name, host, count)` is the pre-capture phase that allocates device
// memory for such a buffer, uploads the host data once, and holds it for the
// registry's lifetime (freed in the dtor); `inject` then packs it by value as a
// flat handle. These are immutable while the sim runs. A force is stateless pure
// math, so there is no writable plugin buffer; a pure kernel carries its own
// writable `Mutable` per-sample state instead.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>            // Driver API: cuModuleLoadData / cuLaunchKernel

#include "../gref_abi.h"
#include "../roles.h"
#include "../sidecar.h"
#include "manifest.h"
#include "uniform_binding.h"

// The execution structures. Both headers are as
// dependency-free as this one (no aether — Partition.h is plain PODs and
// DeviceKernel.h adds only <cuda.h>, which this file already needs), so the ONE
// spelling of "what a partition means at launch" can live there and be shared with
// the host arm rather than re-written here.
#include "eagle/exec/DeviceKernel.h"

namespace eagle {
namespace cuda {
using namespace eagle::plugin;

// A bound vector buffer: device pointer + sample count N (built into a GRef).
struct VecBinding {
    double* data = nullptr;
    std::uint32_t n = 0;
};

// A bound matrix buffer: device pointer + sample count N + the caller's asserted
// (rows, cols). A matrix binds through the SAME GRef mirror as a vector (a matrix
// GRef is a width-R*C vector GRef, dim = r*C + c, sample-fastest) — (rows, cols) is
// carried here purely so ``inject()`` can validate it against the sidecar's declared
// shape before launch (the device mirror of ``host_registry.h``'s ``MB``).
struct MatBinding {
    double* data = nullptr;
    std::uint32_t n = 0;
    int rows = 0;
    int cols = 0;
};

class PluginRegistry {
public:
    struct Plugin {
        std::string id;
        CUmodule module = nullptr;
        CUfunction fn = nullptr;
        Sidecar sidecar;
        bool enabled = true;
        // Which generation of the binary ABI this artifact speaks (1 or 2), from
        // its own sidecar tag. It DISPATCHES the launch: a v2 kernel takes the
        // int64 partition triple after its role args, a v1 one takes nothing and
        // may only ever be launched whole-view.
        int abi_version = 1;
    };

    // One plugin's packed by-value arguments, in ``arg_spec`` order.
    //
    // The per-kind vectors are STORAGE (reserved, never resized, so the addresses
    // in ``params`` stay valid); ``params`` is what a launch hands the driver. It
    // is returned by value so an execution structure can pack once and launch the
    // SAME args over several partitions — moving a std::vector moves its buffer,
    // so the interior pointers survive the move.
    struct PackedArgs {
        std::vector<GRefMirror> grefs;
        std::vector<ScalarHandle> handles;
        std::vector<double> scalars;
        std::vector<UniformInt> ints;
        std::vector<unsigned> counts;
        std::vector<void*> params;
    };

    // Owns consolidated-table device memory, so the registry is move-only: a copy
    // would alias (and then double-free) those allocations. A defaulted move
    // empties the source's table map, so the moved-from destructor frees nothing.
    PluginRegistry() = default;
    PluginRegistry(const PluginRegistry&) = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;
    PluginRegistry(PluginRegistry&&) = default;
    PluginRegistry& operator=(PluginRegistry&&) = default;
    ~PluginRegistry() { for (auto& kv : tables_) cuMemFree(kv.second.dptr); }

    // Load every plugin in the manifest — one CUmodule each — resolving the fixed
    // the extern-C kernel symbol named in each sidecar. A current CUDA context must
    // already be set (the Runtime primary context is fine; see inject_demo init).
    //
    // Exactly `add_plugin` on a fresh registry: the gate order, the messages and
    // the append order are ONE implementation (v0.2.3), so the whole-manifest
    // door and the merging door cannot drift apart.
    static PluginRegistry from_manifest(const std::string& manifest_path) {
        PluginRegistry reg;
        reg.add_plugin(manifest_path);
        return reg;
    }

    // MERGE one manifest's plugins into THIS registry, appending them after the
    // plugins it already holds (v0.2.3). The device twin of the host registry's
    // `add_plugin`, one level up: the host door takes a single (sidecar, .so)
    // pair because a host plugin has no container to open, while a device plugin
    // is reached through a manifest that names its artifact format — so the
    // manifest IS the device unit, and merging is what a caller composing several
    // BUNDLES at one injection opening needs (a downstream consumer's four hook points).
    //
    // Semantics, deliberately identical to `from_manifest`'s:
    //   * every manifest-level gate runs first (the ABI tag, schema version,
    //     launch-certified pattern, duplicate ids, entry formats), before any
    //     module is loaded;
    //   * ORDER IS APPEND ORDER — this manifest's `plugins[]` order, after
    //     everything already loaded, so `inject`/`inject_plugin` walk the merged
    //     set in the order the caller merged it;
    //   * per-entry `enabled` is carried onto the Plugin, so a disabled entry is
    //     LOADED (its lookup tables still resolve) and never launched.
    //
    // Ids are unique across the WHOLE registry, not merely within one manifest:
    // two plugins sharing an id would be indistinguishable in every diagnostic
    // this class emits, and `declared_tables`' name-namespace reasoning assumes a
    // plugin is identifiable. A collision with an ALREADY-LOADED plugin is
    // refused with its own message (naming the merge), a collision inside the
    // incoming manifest keeps the unchanged "duplicate plugin id" wording.
    //
    // Failure leaves the registry EXACTLY as it was: the incoming plugins are
    // staged in a local vector and spliced in only after the last one has loaded,
    // so a manifest whose third artifact is missing does not leave two plugins
    // half-merged.
    void add_plugin(const std::string& manifest_path) {
        const Manifest m = parse_manifest(manifest_path);
        // Reject a manifest built against a different binary ABI (GRef/HandleT POD
        // layout) than this host — before loading any module or packing any struct.
        // The presence-required semantics and the message live in the ONE shared
        // gate `check_aether_abi` (gref_abi.h); every door calls it rather
        // than re-spelling the compare.
        check_aether_abi(m.aether_abi, "plugin manifest");
        // Forward-strict schema gate (absent -> 0, treated as v1: backward-lenient).
        if (m.schema_version > kPluginMaxSchemaVersion)
            throw std::runtime_error("plugin manifest uses schema v" +
                std::to_string(m.schema_version) + ", newer than this host supports "
                "(v" + std::to_string(kPluginMaxSchemaVersion) + "); upgrade eagle");
        // The schema-v2 EXECUTION AXIS: required and vocabulary-gated on a v2
        // manifest, forbidden on a v1 one, and cross-checked against the ABI tag.
        // Shared with the host registry (manifest.h), one spelling.
        validate_execution_axis(m, "plugin manifest");
        // Manifest-level ``pattern`` DISCRIMINANT ("the
        // discriminant rule"): unlike an ordinary optional key, ``pattern`` SELECTS a
        // schema variant, so a loader that does not recognize its value must refuse
        // it rather than silently bind an unfamiliar variant as an ordinary kernel.
        // The schema_version gate above does NOT catch this: a future variant's own
        // keys (e.g. a neural-ABI bundle's fanin/fanout/activation) are colorably
        // "additive optional" under the backward-lenient rule, so a v1-stamped bundle
        // of a variant this build has never heard of is schema-LEGAL today — and
        // `pattern` is the only field that names which variant it actually is. This
        // registry never dispatches on `pattern` (binding is driven entirely by
        // arg_spec), so absence stays lenient (a pre-freeze manifest never stamped
        // one) — only a launchable value is required, mirroring the scalar_type gate.
        // `from_manifest` LAUNCHES, so it gates on the LAUNCH-CERTIFIED set,
        // the subset of recognized families this registry will actually bind and run;
        // the shared `validate_sidecar` separately gates the sidecar-level `pattern`
        // against the wider RECOGNIZED set.
        // Two branches, ONE spelling (roles.h ``check_launch_certified_pattern``): an
        // UNRECOGNIZED family is told to upgrade eagle; a RECOGNIZED-but-uncertified one
        // (today: ``neural_block``) is told it is not launchable here, with no upgrade
        // suffix — this build knows the family, it simply does not run descriptors.
        check_launch_certified_pattern(m.pattern, "plugin manifest");
        // Manifest-level structural invariant ("duplicate ids are rejected") —
        // a hand-built manifest naming the same plugin id twice is invalid: the
        // The Python producer already guarantees
        // uniqueness for a Bundle-built set (an auto-derived collision gets a
        // numeric suffix), but nothing enforced it here for a manifest assembled
        // by hand. Checked over every declared entry (enabled or not) before
        // loading any module, so a bad manifest fails fast with no partial
        // cuModuleLoadData / consolidation side effect.
        //
        // Seeded with the ids ALREADY loaded (v0.2.3): on a fresh registry the
        // set is empty and this is exactly the historical within-manifest check;
        // on a merge it also catches a collision with a plugin an earlier
        // manifest contributed, which is a different mistake and says so.
        {
            std::set<std::string> incoming;
            for (const auto& e : m.plugins) {
                if (has_id_(e.id))
                    throw std::runtime_error("cannot merge plugin manifest '" +
                        manifest_path + "': its plugin id '" + e.id + "' is "
                        "already loaded in this registry; ids are unique across "
                        "the whole registry, not merely within one manifest");
                if (!incoming.insert(e.id).second)
                    throw std::runtime_error("plugin manifest has duplicate "
                        "plugin id '" + e.id + "'; every entry in plugins[] "
                        "must have a unique id");
            }
        }
            // Manifest-entry `format` DISCRIMINANT: every declared
        // entry's artifact-container tag must be one this build knows how to load,
        // checked before any sidecar parse / module load (manifest.h
        // `validate_manifest_formats`, shared with the host registry).
        validate_manifest_formats(m);
        // STAGED, then spliced (v0.2.3): a failure part-way through the load must
        // leave the registry holding exactly the plugins it held on entry, or a
        // caller merging four bundles and getting a throw on the third would be
        // left with a registry that is neither the old set nor the new one.
        std::vector<Plugin> staged;
        staged.reserve(m.plugins.size());
        for (const auto& e : m.plugins) {
            Plugin p;
            p.id = e.id;
            p.enabled = e.enabled;
            p.sidecar = parse_sidecar(m.dir + e.sidecar);
            // Symmetric with the Python loader: validate this sidecar's OWN binary-ABI
            // tag (not just the manifest's), schema version, and arg roles before
            // binding — so both the manifest and every sidecar are checked on both sides.
            check_aether_abi(p.sidecar.aether_abi, "plugin '" + p.id + "' sidecar");
            validate_sidecar(p.sidecar, "plugin '" + p.id + "'");
            // A sidecar's OWN generation must match the manifest's: the manifest
            // decides how the whole set is launched, the sidecar decides how each
            // artifact was BUILT, and a disagreement means the packing the manifest
            // implies is not the packing the artifact expects.
            p.abi_version = abi_version_of(p.sidecar.aether_abi);
            if (!m.aether_abi.empty() && p.abi_version != abi_version_of(m.aether_abi))
                throw std::runtime_error("plugin '" + p.id + "' sidecar is built for "
                    "aether_abi '" + p.sidecar.aether_abi + "' but its manifest "
                    "declares '" + m.aether_abi + "'; one manifest cannot mix ABI "
                    "generations");
            // float32: this registry's ENTIRE binding surface is double-typed
            // (bind_vector/bind_uniform/consolidate all take double*/double,
            // Buffer::data is double*), and a float32 kernel genuinely needs
            // float-typed buffers and 4-byte uniform packing this registry
            // does not have — a real capability gap, not a build oversight.
            // Name the real reason and the real alternative rather than the
            // CPU-host rejection text this used to carry verbatim (which
            // told the caller to do the thing they were already doing).
            if (p.sidecar.scalar_type == "float32")
                throw std::runtime_error("plugin '" + p.id +
                    "': the C++ CUDA PluginRegistry binds float64 buffers "
                    "only (bind_vector/bind_uniform/consolidate are all "
                    "double-typed); this artifact's scalar_type is "
                    "'float32'. Launch it through eagle's Python/torch "
                    "device path (eagle.LoadedVector / eagle.LoadedPure), "
                    "which supports float32");
            // softdouble: UNLIKE float32, SoftDouble is bit-identical IEEE
            // float64 (SoftDouble static_asserts sizeof == 8; arrays,
            // uniforms, and the GRef ABI all stay float64-shaped), so it
            // loads and launches through this registry's double-typed
            // binding surface with zero widening. Confirmed by a
            // load-and-run probe (eagle/tests/test_PluginRegistryDtype.cu,
            // PluginRegistryDtypeTest.SoftdoubleLoadsAndLaunches): a
            // softdouble artifact loads via cuModuleLoadData, resolves its
            // kernel symbol, and launches through inject() unmodified.
            const std::string art = _slurp(m.dir + e.artifact);
            check_(cuModuleLoadData(&p.module, art.c_str()),
                   "cuModuleLoadData", p.id);
            check_(cuModuleGetFunction(&p.fn, p.module, p.sidecar.kernel.c_str()),
                   "cuModuleGetFunction(" + p.sidecar.kernel + ")", p.id);
            // LAYOUT SELF-CHECK, device face. There is no dlsym on a PTX
            // module, so the exported array is resolved with
            // cuModuleGetGlobal and staged to the host — a driver-module global
            // read, not a transfer on aether data.
            if (p.abi_version >= 2) check_layout_(p);
            staged.push_back(p);
        }
        for (auto& p : staged) plugins_.push_back(std::move(p));
    }

    // Force the plugin at `index` (registry order, i.e. merge order) on or off,
    // overriding the per-entry `enabled` its manifest declared (v0.2.3).
    //
    // The per-plugin twin of `set_enabled`, and the one a caller composing
    // several manifests at one opening needs: `set_enabled` is a single GLOBAL
    // flag, so once two bundles share a registry it can no longer express "this
    // bundle off, that one on". Out of range THROWS, like `inject_plugin` and
    // unlike the total `will_launch` query: naming a plugin that does not exist
    // is a caller defect, not a no-op.
    void set_plugin_enabled(std::size_t index, bool on) {
        if (index >= plugins_.size())
            throw std::out_of_range("plugin index " + std::to_string(index) +
                " is out of range: " + std::to_string(plugins_.size()) +
                " plugin(s) loaded");
        plugins_[index].enabled = on;
    }

    // --- the global runtime flag that gates the whole opening ------------------
    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    std::size_t size() const { return plugins_.size(); }

    // How many plugins would actually launch: the global flag AND each plugin's
    // own manifest enable. This is the node count the opening contributes.
    int active() const {
        if (!enabled_) return 0;
        int k = 0;
        for (const auto& p : plugins_) if (p.enabled) ++k;
        return k;
    }

    // --- name-based binding of PRE-ALLOCATED device memory (no allocation) -----
    // Names match the sidecar arg_spec: "out"/"position"/"velocity" (vectors),
    // "mass"/"area"/"cr"/"cd"/"terminated" (handles), "mu"/... (uniforms).
    void bind_vector(const std::string& name, double* data, std::uint32_t n) {
        vec_[name] = VecBinding{data, n};
    }
    // Bind a matrix (``mat_in`` / matrix ``mutable``) buffer by name — a device
    // flat ``(R*C, N)`` SoA pointer (PRE-ALLOCATED and uploaded by the caller, like
    // ``bind_vector``: the registry only references it), sample count, and the
    // caller's asserted ``(rows, cols)``. Validated against the sidecar's declared
    // ``(R, C)`` at ``inject`` time (loud throw on mismatch) — the device mirror of
    // ``host_registry.h``'s ``bind_matrix`` / ``checked_matrix_gref_``.
    void bind_matrix(const std::string& name, double* data, std::uint32_t n,
                      int rows, int cols) {
        mat_[name] = MatBinding{data, n, rows, cols};
    }
    void bind_handle(const std::string& name, void* ptr) { handle_[name] = ptr; }
    // Bind a float64 ``uniform`` by name — the kernel's ``AETHER_GRID_CONSTANT() Real p_<name>``
    // by-value parameter slot. Re-binding the same name overwrites (unchanged); binding
    // a name already bound as int64 is refused (uniform_binding.h names the reason).
    void bind_uniform(const std::string& name, double value) {
        check_uniform_kind_free(UniformKind::Real, uniform_.count(name) != 0,
                                uniform_int_.count(name) != 0, name, "plugin registry");
        uniform_[name] = value;
    }
    // Bind an int64 ``uniform`` by name — the ADDITIVE int arm of the same role.
    // A generated kernel spells an integer quantity ``Int`` == ``long long``
    //  (the code generator's spellings), so the parameter slot is ``AETHER_GRID_CONSTANT() Int
    // p_<name>`` and this packs the matching 8 signed bytes. Same name-keyed map
    // discipline as ``bind_uniform``, and the same one-type-per-name refusal.
    void bind_uniform_int(const std::string& name, UniformInt value) {
        check_uniform_kind_free(UniformKind::Int, uniform_.count(name) != 0,
                                uniform_int_.count(name) != 0, name, "plugin registry");
        uniform_int_[name] = value;
    }

    // --- consolidation: allocate + upload a read-only lookup table (pre-capture) -
    // The registry OWNS this device memory (unlike the by-name interface bindings
    // above, which it only references). Call once per declared table BEFORE the
    // host opens its stream capture; the table is then immutable for the run.
    // Re-consolidating the same name replaces the prior allocation. ``count`` is
    // the flat element count (== prod(shape)); the table is read as ``handle[k]``.
    void consolidate(const std::string& name, const double* host, std::size_t count) {
        const std::size_t bytes = count * sizeof(double);
        auto it = tables_.find(name);
        // Budget counts what this name occupies AFTER replacement (a re-consolidation
        // reuses its slot), so discount its current size.
        const std::size_t old_bytes = (it != tables_.end()) ? it->second.bytes : 0;
        if (budget_ != 0 && table_bytes() - old_bytes + bytes > budget_)
            throw std::runtime_error("lookup table '" + name + "' (" +
                std::to_string(bytes) + " B) would exceed the declared table "
                "budget (" + std::to_string(budget_) + " B)");
        // Allocate + upload into a FRESH allocation first; commit it only on success.
        // A failure then leaks nothing and never destroys a table already bound to
        // this name (strong exception guarantee on re-consolidation).
        CUdeviceptr d = 0;
        const CUresult r = cuMemAlloc(&d, bytes);
        if (r != CUDA_SUCCESS) {                   // clean OOM (not a raw CUDA code)
            const char* m = nullptr; cuGetErrorString(r, &m);
            throw std::runtime_error("failed to allocate " + std::to_string(bytes) +
                " B for lookup table '" + name + "': " + (m ? m : "?"));
        }
        const CUresult mc = cuMemcpyHtoD(d, host, bytes);
        if (mc != CUDA_SUCCESS) {
            cuMemFree(d);                          // no leak on copy failure
            check_(mc, "cuMemcpyHtoD(table '" + name + "')", name);
        }
        if (it != tables_.end()) cuMemFree(it->second.dptr);  // release old AFTER success
        tables_[name] = OwnedTable{d, bytes};
    }

    // Optional ceiling (bytes) on total consolidated-table memory; 0 = unbounded.
    void set_table_budget(std::size_t bytes) { budget_ = bytes; }
    std::size_t table_bytes() const {
        std::size_t t = 0; for (const auto& kv : tables_) t += kv.second.bytes;
        return t;
    }

    // The distinct lookup tables this plugin set declares (name + flat count),
    // deduplicated by name — exactly what the host must supply at consolidation.
    // Tables share ONE global name namespace, so two plugins declaring the same
    // name with DIFFERENT element counts is a hard error (it would otherwise bind
    // both to one undersized allocation -> a silent out-of-bounds device read).
    std::vector<BufferInfo> declared_tables() const {
        std::vector<BufferInfo> out;
        std::map<std::string, std::size_t> seen;   // name -> flat element count
        for (const auto& p : plugins_) {
            if (!p.enabled) continue;   // a disabled plugin never launches -> no demand
            for (const auto& b : p.sidecar.buffers) {
                if (b.kind != "lookup") continue;
                auto it = seen.find(b.name);
                if (it == seen.end()) {
                    seen.emplace(b.name, b.count);
                    out.push_back(b);
                } else if (it->second != b.count) {
                    throw std::runtime_error("lookup table '" + b.name +
                        "' declared with conflicting element counts (" +
                        std::to_string(it->second) + " vs " +
                        std::to_string(b.count) + ") across plugins; tables share "
                        "one global name namespace");
                }
            }
        }
        return out;
    }

    // The injection opening. For each enabled plugin (manifest order), walk its
    // arg_spec, look each (role, name) up in the bindings, pack by value into
    // per-plugin stable storage, and launch on `stream`. If `stream` is being
    // captured the launch is recorded as a graph node; otherwise it runs eagerly.
    // Returns the number of kernels launched (0 when the flag is off).
    int inject(CUstream stream, int n, int block = 256) {
        return inject_partition(stream, eagle::exec::Partition::whole(n), block);
    }

    /** @brief The PARTITIONED injection opening:
     *  every enabled plugin, in manifest order, over ONE partition of the run.
     *
     *  ``inject(stream, n)`` is exactly this with ``Partition::whole(n)``, so the
     *  legacy opening and the partitioned one are ONE implementation — the packing
     *  rules, the gate order and the manifest order cannot drift between them.
     *  A v2 plugin receives the int64 triple after its role args; a v1 plugin
     *  receives nothing extra and is REFUSED a non-whole partition. */
    int inject_partition(CUstream stream, const eagle::exec::Partition& part,
                         int block = 256) {
        if (!enabled_) return 0;
        int launched = 0;
        for (auto& p : plugins_) {
            if (!p.enabled) continue;
            launch_(p, stream, unsigned(block), part);
            ++launched;
        }
        return launched;
    }

    // Would the plugin at `index` launch? — the per-plugin twin of `active()`:
    // the global flag AND this plugin's own manifest enable. Total (an
    // out-of-range index is simply "no"), because this is the QUERY a caller
    // walking 0..size()-1 uses to decide whether to open a capture at all.
    bool will_launch(std::size_t index) const {
        return enabled_ && index < plugins_.size() && plugins_[index].enabled;
    }

    // Inject exactly ONE plugin — the plugin at `index` in manifest order —
    // into `stream`, packing and launching it exactly as `inject` would.
    // Returns 1 if it launched, 0 if it would not (global flag off, or the
    // plugin's own manifest enable off).
    //
    // Why this exists next to `inject`: a host that records the injection
    // inside a stream capture gets ONE graph node per capture, so a single
    // `inject` of k plugins yields one node whose k kernels run in sequence.
    // A host that wants k INDEPENDENT nodes — siblings that the graph may
    // schedule concurrently, each carrying its own dependency edges — opens
    // one capture per plugin around this call instead. Nothing else differs:
    // same bindings, same packing, same launch geometry, and the manifest
    // order is still the caller's to walk. Out of range THROWS (unlike the
    // total `will_launch` query): a launch aimed at a plugin that does not
    // exist is a caller defect, not a "nothing to do".
    int inject_plugin(CUstream stream, std::size_t index, int n, int block = 256) {
        if (index >= plugins_.size())
            throw std::out_of_range("plugin index " + std::to_string(index) +
                " is out of range: " + std::to_string(plugins_.size()) +
                " plugin(s) loaded");
        if (!will_launch(index)) return 0;
        launch_(plugins_[index], stream, unsigned(block),
                eagle::exec::Partition::whole(n));
        return 1;
    }

    /** @brief Single-plugin twin of :func:`inject_partition`. */
    int inject_plugin_partition(CUstream stream, std::size_t index,
                                const eagle::exec::Partition& part,
                                int block = 256) {
        if (index >= plugins_.size())
            throw std::out_of_range("plugin index " + std::to_string(index) +
                " is out of range: " + std::to_string(plugins_.size()) +
                " plugin(s) loaded");
        if (!will_launch(index)) return 0;
        launch_(plugins_[index], stream, unsigned(block), part);
        return 1;
    }

    /** @brief Which ABI generation the plugin at @p index speaks (1 or 2). */
    int abi_version(std::size_t index) const {
        return at_(index).abi_version;
    }

    /** @brief The resolved ``CUfunction`` of the plugin at @p index — what an
     *  execution structure (``eagle::exec::DeviceKernel``) launches. */
    CUfunction function(std::size_t index) const { return at_(index).fn; }

    /** @brief Pack the plugin at @p index's role args for a run spanning
     *  @p nSamples samples.
     *
     *  Every role is packed WHOLE: eagle does no pointer arithmetic on plugin
     *  buffers, and the body slices per-sample roles by its own triple. The count
     *  that reaches the ``nsamples`` role and every view mirror's extent is the
     *  TRUE ``nSamples`` of the run, never a partition's ``count`` — review
     *  Note: a wide/accum plane's addressing bakes the full sample count,
     *  so handing it a partition's length silently re-targets every column. */
    PackedArgs pack(std::size_t index, std::int64_t nSamples) const {
        return pack_(at_(index), nSamples);
    }

private:
    // Is `id` already loaded? The merge door's collision test (v0.2.3).
    bool has_id_(const std::string& id) const {
        for (const auto& p : plugins_) if (p.id == id) return true;
        return false;
    }

    // The launch geometry both openings derive from `n`: one thread per sample.
    static unsigned grid_(int n, int block) {
        return (unsigned(n) + unsigned(block) - 1u) / unsigned(block);
    }

    // The plugin at `index`, bounds-checked — the const accessor the public
    // per-plugin queries share.
    const Plugin& at_(std::size_t index) const {
        if (index >= plugins_.size())
            throw std::out_of_range("plugin index " + std::to_string(index) +
                " is out of range: " + std::to_string(plugins_.size()) +
                " plugin(s) loaded");
        return plugins_[index];
    }

    // Read the module's exported layout array and compare it with this
    // host's. cuModuleGetGlobal is the PTX face of the host path's dlsym.
    void check_layout_(Plugin& p) const {
        const std::string what = "plugin '" + p.id + "'";
        CUdeviceptr dptr = 0;
        std::size_t bytes = 0;
        if (cuModuleGetGlobal(&dptr, &bytes, p.module, kEagleLayoutSymbol)
                != CUDA_SUCCESS)
            check_layout_sizes(nullptr, what);   // the "does not export it" refusal
        if (bytes < kEagleLayoutFieldCount * sizeof(std::uint64_t))
            throw std::runtime_error(what + ": the exported '" +
                std::string(kEagleLayoutSymbol) + "' is " + std::to_string(bytes) +
                " bytes, but the aether-abi/2 layout self-check is " +
                std::to_string(kEagleLayoutFieldCount) + " x 8 bytes; rebuild the "
                "plugin");
        std::uint64_t got[kEagleLayoutFieldCount] = {};
        check_(cuMemcpyDtoH(got, dptr, kEagleLayoutFieldCount * sizeof(std::uint64_t)),
               "cuMemcpyDtoH(" + std::string(kEagleLayoutSymbol) + ")", p.id);
        check_layout_sizes(got, what);
    }

    // Pack ONE plugin's arg_spec from the bindings. The whole of what `inject`
    // used to do per iteration, factored out so the per-plugin opening cannot
    // drift from the all-plugins one: there is exactly one copy of the packing
    // rules — and now one copy shared with every execution structure.
    PackedArgs pack_(const Plugin& p, std::int64_t nSamples) const {
        PackedArgs out;
        // Per-plugin stable storage: reserved (never resized), so addresses
        // pushed into `params` stay valid through cuLaunchKernel.
        const std::size_t na = p.sidecar.arg_spec.size();
        std::vector<GRefMirror>&   grefs   = out.grefs;   grefs.reserve(na);
        std::vector<ScalarHandle>& handles = out.handles; handles.reserve(na);
        std::vector<double>&       scalars = out.scalars; scalars.reserve(na);
        std::vector<UniformInt>&   ints    = out.ints;    ints.reserve(na);
        std::vector<unsigned>&     counts  = out.counts;  counts.reserve(na);
        std::vector<void*>&        params  = out.params;  params.reserve(na + 3);
        const unsigned n = unsigned(nSamples);
        for (const auto& a : p.sidecar.arg_spec) {
            if (a.role == "out" || a.role == "vec_in") {
                const VecBinding vb = lookup_vec_(p.id, a.name);
                grefs.push_back(make_gref(vb.data, vb.n, kEagleAbiDeviceCUDA));
                params.push_back(&grefs.back());
            } else if (a.role == "mat_in") {
                // A matrix input binds through the SAME GRef mirror as a vector.
                grefs.push_back(checked_matrix_gref_(p.id, p.sidecar, a.name));
                params.push_back(&grefs.back());
            } else if (a.role == "mutable") {
                // A pure kernel's writable per-sample state (the pure output). A
                // vector/matrix Mutable rides the (W, N) / (R*C, N) GRef path exactly
                // like out/vec_in/mat_in; a scalar/int one a flat handle. The host
                // binds it by name like any other buffer — the registry only
                // references it, never resets it (an RMW Mutable carries state; the
                // host owns any seed/reset).
                std::string dt;
                for (const auto& mi : p.sidecar.mutables)
                    if (mi.name == a.name) { dt = mi.dtype; break; }
                if (dt == "vector") {
                    const VecBinding vb = lookup_vec_(p.id, a.name);
                    grefs.push_back(make_gref(vb.data, vb.n, kEagleAbiDeviceCUDA));
                    params.push_back(&grefs.back());
                } else if (dt == "matrix") {
                    grefs.push_back(checked_matrix_gref_(p.id, p.sidecar, a.name));
                    params.push_back(&grefs.back());
                } else {
                    handles.push_back(make_handle(
                        lookup_handle_(p.id, a.name), unsigned(n), kEagleAbiDeviceCUDA));
                    params.push_back(&handles.back());
                }
            } else if (a.role == "nsamples") {
                // A pure kernel has no `out` GRef to carry N, so it takes an explicit
                // sample count (aether::idx_t -> uint32), passed by value.
                counts.push_back(n);
                params.push_back(&counts.back());
            } else if (a.role == "per_sample" || a.role == "terminated") {
                // A read-only per-sample handle: the spacecraft scalars or the
                // termination mask — both rank-1 aether View mirrors bound by
                // name. The sample count comes from the launch's own `n`: an
                // aether View carries its extent explicitly.
                handles.push_back(make_handle(
                        lookup_handle_(p.id, a.name), unsigned(n), kEagleAbiDeviceCUDA));
                params.push_back(&handles.back());
            } else if (a.role == "lookup") {
                // A consolidated read-only table: a flat handle, same 32-byte
                // ABI as a per-sample handle, read at a user-computed index.
                handles.push_back(make_handle(
                    lookup_table_(p.id, a.name), unsigned(n), kEagleAbiDeviceCUDA));
                params.push_back(&handles.back());
            } else if (a.role == "uniform") {
                // A uniform is packed BY VALUE, in one of two same-width but
                // differently-INTERPRETED spellings: float64 (`Real p_<name>`) or
                // int64 (`Int p_<name>`). The arg_spec row carries no dtype
                // (it is a [role, name] pair), so the type is the one the CALLER
                // declared by choosing a binder — and binding one name through both
                // is refused at bind time (uniform_binding.h), so at most one of
                // these maps can hold this name. A uniform bound through NEITHER
                // keeps the unchanged float64 "none was bound" error.
                if (uniform_int_.count(a.name)) {
                    ints.push_back(lookup_uniform_int_(p.id, a.name));
                    params.push_back(&ints.back());
                } else {
                    scalars.push_back(lookup_uniform_(p.id, a.name));
                    params.push_back(&scalars.back());
                }
            } else if (a.role == "wide_in" || a.role == "wide_out" ||
                       a.role == "accum_out") {
                // The WIDE / ACCUMULATE planes: in the role
                // register since the width architecture landed, but unlaunchable
                // by either C++ registry until now — seven arms, then a throw, so
                // the register and its own producer disagreed. All three ride the
                // SAME plain 32-byte scalar-handle ABI as per_sample/lookup (see
                // roles.h and eagle.roles.classify_arg's WIDE_IN/WIDE_OUT tags);
                // what differs is the plane's ADDRESSING, which is the body's
                // business and is what the partition triple makes exact.
                //
                // The handle's extent is the TRUE nSamples of the run, never a
                // partition's count: a wide/accum plane is (rows, nSamples), so a
                // partition's length here would re-target every column.
                handles.push_back(make_handle(
                    lookup_handle_(p.id, a.name), n, kEagleAbiDeviceCUDA));
                params.push_back(&handles.back());
            } else {
                throw std::runtime_error("plugin '" + p.id +
                    "': unknown arg role '" + a.role + "'");
            }
        }
        return out;
    }

    // Pack + launch ONE plugin over ONE partition.
    void launch_(Plugin& p, CUstream stream, unsigned block,
                 const eagle::exec::Partition& part) {
        eagle::exec::check_legacy_whole_view(p.abi_version, part,
                                             "plugin '" + p.id + "'");
        PackedArgs packed = pack_(p, part.nSamples);
        if (p.abi_version >= 2) {
            // ONE spelling of "what a partition means at launch" — the execution
            // structure appends the triple and derives the grid from `count`.
            eagle::exec::DeviceKernel::run(p.fn, packed.params, part, stream, block);
            return;
        }
        // Legacy (aether-abi/1): today's packing, byte for byte — no triple, and
        // the grid derived from the whole view's sample count.
        check_(cuLaunchKernel(p.fn, grid_(int(part.count), int(block)), 1, 1,
                              block, 1, 1, 0, stream, packed.params.data(), nullptr),
               "cuLaunchKernel", p.id);
    }

    static void check_(CUresult r, const std::string& what, const std::string& id) {
        if (r != CUDA_SUCCESS) {
            const char* msg = nullptr; cuGetErrorString(r, &msg);
            throw std::runtime_error(what + " failed for plugin '" + id + "': " +
                                     (msg ? msg : "?"));
        }
    }
    VecBinding lookup_vec_(const std::string& id, const std::string& name) const {
        auto it = vec_.find(name);
        if (it == vec_.end())
            throw std::runtime_error("plugin '" + id + "' needs vector binding '" +
                                     name + "' but none was bound");
        return it->second;
    }
    MatBinding lookup_mat_(const std::string& id, const std::string& name) const {
        auto it = mat_.find(name);
        if (it == mat_.end())
            throw std::runtime_error("plugin '" + id + "' needs matrix binding '" +
                                     name + "' but none was bound");
        return it->second;
    }
    // The bound matrix packed as a vector GRef mirror, after asserting the caller's
    // (rows, cols) matches the sidecar's declared (R, C) for this name (loud throw on
    // mismatch — the device mirror of the host path's ``checked_matrix_gref_``).
    GRefMirror checked_matrix_gref_(const std::string& id, const Sidecar& sc,
                                     const std::string& name) const {
        const MatBinding mb = lookup_mat_(id, name);
        auto ds = sc.mat_shapes.find(name);
        if (ds != sc.mat_shapes.end() &&
            (mb.rows != ds->second.first || mb.cols != ds->second.second))
            throw std::runtime_error("plugin '" + id + "': matrix '" + name +
                "' bound as (" + std::to_string(mb.rows) + ", " +
                std::to_string(mb.cols) + ") but the kernel was compiled for (" +
                std::to_string(ds->second.first) + ", " +
                std::to_string(ds->second.second) + ')');
        return make_gref(mb.data, mb.n, kEagleAbiDeviceCUDA);
    }
    void* lookup_handle_(const std::string& id, const std::string& name) const {
        auto it = handle_.find(name);
        if (it == handle_.end())
            throw std::runtime_error("plugin '" + id + "' needs handle binding '" +
                                     name + "' but none was bound");
        return it->second;
    }
    double lookup_uniform_(const std::string& id, const std::string& name) const {
        auto it = uniform_.find(name);
        if (it == uniform_.end())
            throw std::runtime_error("plugin '" + id + "' needs uniform '" +
                                     name + "' but none was bound");
        return it->second;
    }
    // The int64 twin. Only ever called after ``uniform_int_.count(name)`` said yes, so
    // the miss branch is unreachable through ``inject`` — it is kept (and worded like
    // its float64 sibling) so a future direct caller cannot get a silent zero.
    UniformInt lookup_uniform_int_(const std::string& id, const std::string& name) const {
        auto it = uniform_int_.find(name);
        if (it == uniform_int_.end())
            throw std::runtime_error("plugin '" + id + "' needs uniform '" +
                                     name + "' but none was bound");
        return it->second;
    }
    void* lookup_table_(const std::string& id, const std::string& name) const {
        auto it = tables_.find(name);
        if (it == tables_.end())
            throw std::runtime_error("plugin '" + id + "' needs lookup table '" +
                name + "' but it was not supplied at consolidation "
                "(call registry.consolidate(\"" + name + "\", data, count))");
        return reinterpret_cast<void*>(it->second.dptr);
    }

    // A registry-OWNED device allocation for a consolidated lookup table.
    struct OwnedTable { CUdeviceptr dptr = 0; std::size_t bytes = 0; };

    std::vector<Plugin> plugins_;
    std::map<std::string, VecBinding> vec_;
    std::map<std::string, MatBinding> mat_;
    std::map<std::string, void*> handle_;
    std::map<std::string, double> uniform_;
    std::map<std::string, UniformInt> uniform_int_;  // int64 uniforms
    std::map<std::string, OwnedTable> tables_;   // owned (freed in the destructor)
    std::size_t budget_ = 0;                     // 0 = no table-memory ceiling
    bool enabled_ = true;
};

}  // namespace cuda
}  // namespace eagle
