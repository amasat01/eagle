// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The CPU (host) twin of plugin_registry/registry.h -- the CPU-plugin path.
//
// Where PluginRegistry loads a PTX/cubin module through the CUDA Driver API and
// issues cuLaunchKernel, PluginRegistry dlopen's a shared object and calls a
// plain host entry point. It shares EVERYTHING else with the device path: the
// same gref_abi.h POD mirrors, the same sidecar/manifest schema, the same
// role -> params[] packing. The entry receives the IDENTICAL params[] array the
// device registry builds for cuLaunchKernel — an array of pointers to
// GRefMirror / ScalarHandle / double / unsigned, in arg_spec order — and
// unpacks it against its own (known) signature, running one OpenMP loop over
// the N samples. So a CPU plugin is byte-for-byte the same artifact contract as
// a GPU plugin, minus the PTX: a `.so` exporting `void <kernel>_host(void*
// const* params, int32_t n)`.
//
// Deliberately includes NO CUDA headers — accepting a host plugin
// needs only the binary ABI (gref_abi.h PODs), the `.so`, and its sidecar. That
// keeps this header compilable in EAGLE_CPU_ONLY builds (the whole point of
// this header).
#pragma once

#include "gref_abi.h"
#include "plugin_registry/manifest.h"
#include "plugin_registry/uniform_binding.h"
#include "roles.h"
#include "sidecar.h"

// The partition/placement vocabulary. Plain PODs
// and free functions — no aether, no CUDA, no eagle machinery — so including it
// costs this header none of its "compilable in EAGLE_CPU_ONLY builds" property.
// The TEAM itself (`eagle/exec/HostTeam.h`) deliberately stays ABOVE this header:
// it uses aether's L2 tiling, which this one may not see.
#include "eagle/exec/Partition.h"

#include <cstdint>
#include <dlfcn.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace cpu {
using namespace eagle::plugin;

/** @brief A CPU plugin entry: the same packed args the device path hands
 *  cuLaunchKernel, run on the host over @p n samples. The `.so` exports one of
 *  these per kernel (default symbol ``<kernel>_host``, or the sidecar's
 *  ``host_entry``); its body does its own OpenMP parallel-for. */
using HostEntry = void (*)(void* const* params, std::int32_t n);

/** @brief The aether-abi/2 host entry: a SERIAL range over `[base, base+count)`
 *  The plugin's own `#pragma omp parallel for` is GONE
 *  — threading and tiling are eagle's (`eagle::exec::HostTeam`), and each tile
 *  calls this with its own triple, which is what keeps wide/accum columns disjoint
 *  Single-sourced in `eagle/exec/Partition.h`. */
using HostEntryV2 = eagle::exec::HostEntryV2;

/**
 * @brief Host-side plugin registry — dlopen + dlsym + host-run, ABI-only.
 *
 * Mirrors ``PluginRegistry`` for the CPU graph executor: bind the kernel's
 * by-name buffers from host memory the caller already owns (no device upload —
 * the data is already where the kernel reads it), then ``run(n)`` packs each
 * plugin's args in ``arg_spec`` order and calls its host entry. Move-only; the
 * destructor ``dlclose``s every loaded object.
 */
class PluginRegistry {
    // Forward-declared here (private, as it has always been) so the accessors
    // below can name it in a RETURN type: the definition lives with the rest of
    // this class's private data, far below.
    struct Plugin;

public:
    /** @brief One plugin's packed by-value arguments, in ``arg_spec`` order — the
     *  host twin of the device registry's ``PackedArgs``, and the SAME params[]
     *  array both build.
     *
     *  The per-kind vectors are STORAGE (reserved, never resized, so the addresses
     *  in ``params`` stay valid); ``params`` is what an entry call receives.
     *  Returned by value so an execution structure can pack ONCE and drive the
     *  same args over many tiles — moving a std::vector moves its buffer, so the
     *  interior pointers survive the move. */
    struct PackedArgs {
        std::vector<GRefMirror> grefs;
        std::vector<ScalarHandle> handles;
        std::vector<double> scalars;
        std::vector<UniformInt> ints;
        std::vector<unsigned> counts;
        std::vector<void*> params;
    };

    PluginRegistry()                                 = default;
    PluginRegistry(const PluginRegistry&)            = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;

    PluginRegistry(PluginRegistry&& o) noexcept
        : plugins_(std::move(o.plugins_))
        , vec_(std::move(o.vec_))
        , mat_(std::move(o.mat_))
        , handle_(std::move(o.handle_))
        , uniform_(std::move(o.uniform_))
        , uniform_int_(std::move(o.uniform_int_))
        , tables_(std::move(o.tables_))
        , enabled_(o.enabled_)
    {
        o.plugins_
            .clear(); // moved-from owns nothing -> its dtor dlcloses nothing
    }

    ~PluginRegistry()
    {
        for (auto& p : plugins_)
            if (p.handle)
                dlclose(p.handle);
    }

    /** @brief Load one host plugin from a `.so` path + its parsed sidecar.
     *
     *  The programmatic entry (used by tests and by ``load`` below): validates
     * the ABI tag + schema version exactly like the device registry,
     * ``dlopen``s the object, and resolves the host entry symbol. The CPU
     * runner executes native float64 only (a float32 / SoftDouble kernel is
     * device-only), matching the device host runner's dtype policy.
     *
     *  @p enabled (v0.2.3) mirrors the device registry's per-plugin manifest
     *  enable: a disabled plugin is LOADED — so its lookup tables still resolve
     *  and a bad artifact is still diagnosed — and never RUN. Defaults to true,
     *  so every existing caller of the two-argument form is unchanged. */
    void add_plugin(
        const Sidecar& sc, const std::string& so_path, bool enabled = true)
    {
        // Gate ORDER is LOCKED for every launching door: shared
        // `validate_sidecar` -> launch certification -> loader-specific gates
        // (`check_aether_abi`, the float64 policy). Family refusal must precede
        // ABI-tag complaints, because a descriptor arriving at a host door is a
        // MULTI-DEFECT input — it is the wrong KIND for this loader and it also
        // (correctly) carries no `aether_abi` of its own — and the
        // caller must be told what it actually is rather than nitpicked on a tag
        // it was right not to stamp.
        validate_sidecar(sc, "host plugin '" + sc.kernel + "'");
        // `add_plugin` is the manifest-BYPASSING entry point, so it runs its OWN
        // launch certification: no manifest-level check runs above it, and before
        // step 1b `validate_sidecar` was its only family gate — exactly the hole
        // this check exists to keep shut once RECOGNIZED grew past
        // the launch-certified set.
        check_launch_certified_pattern(sc.pattern,
                                       "host plugin '" + sc.kernel + "':");
        // The binary-ABI gate must fire here too, not just at the two manifest-level
        // doors -- flagged as the one easiest to
        // forget. Presence-required semantics and message shape live in the ONE
        // shared gate (gref_abi.h).
        check_aether_abi(sc.aether_abi, "host plugin '" + sc.kernel + "': sidecar");
        if (!sc.scalar_type.empty() && sc.scalar_type != "float64")
            throw std::runtime_error("host plugin '" + sc.kernel
                + "': the C++ host "
                  "runner executes float64 only (got '"
                + sc.scalar_type + "')");

        Plugin p;
        p.sidecar = sc;
        // Which generation this artifact speaks — it DISPATCHES both the entry
        // signature (v2 takes the partition triple) and whether the layout
        // self-check symbol is required.
        p.abi_version = abi_version_of(sc.aether_abi);
        p.handle  = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!p.handle) {
            const char* dl_err = dlerror();
            throw std::runtime_error("host plugin '" + sc.kernel + "': dlopen('"
                + so_path
                + "') failed: " + std::string(dl_err ? dl_err : "unknown"));
        }

        const std::string sym
            = sc.host_entry.empty() ? (sc.kernel + "_host") : sc.host_entry;
        dlerror(); // clear
        void* fn        = dlsym(p.handle, sym.c_str());
        const char* err = dlerror();
        if (err) {
            const std::string reason(err);
            dlclose(p.handle);
            throw std::runtime_error("host plugin '" + sc.kernel + "': dlsym('"
                + sym + "') failed: " + reason);
        }
        if (p.abi_version >= 2) {
            // LAYOUT SELF-CHECK, host face: a `.so` has dlsym, so the
            // exported array is read directly (the device face resolves the same
            // symbol with cuModuleGetGlobal).
            dlerror();
            void* layout = dlsym(p.handle, kEagleLayoutSymbol);
            const char* lerr = dlerror();
            const std::string what = "host plugin '" + sc.kernel + "'";
            if (lerr != nullptr || layout == nullptr) {
                dlclose(p.handle);
                check_layout_sizes(nullptr, what);  // the "does not export it" refusal
            }
            try {
                check_layout_sizes(static_cast<const std::uint64_t*>(layout), what);
            } catch (...) {
                dlclose(p.handle);
                throw;
            }
            p.fn2 = reinterpret_cast<HostEntryV2>(fn);
        } else {
            p.fn = reinterpret_cast<HostEntry>(fn);
        }
        p.enabled = enabled;
        plugins_.push_back(std::move(p));
    }

    /** @brief Load every plugin in a manifest (parity with the device
     * registry).
     *  @p dir is the directory the manifest's relative artifact/sidecar names
     * are resolved against. */
    void load(const Manifest& m, const std::string& dir)
    {
        // Mirrors the device `PluginRegistry::from_manifest` through the ONE
        // shared binary-ABI gate (gref_abi.h). This door used to be
        // absence-lenient, diverging from the device registry for no principled
        // reason; routing it through the
        // shared helper makes that divergence structurally impossible to
        // reintroduce.
        check_aether_abi(m.aether_abi, "host manifest");
        // Forward-strict schema gate (absent -> 0, treated as v1: backward-lenient),
        // mirroring the device registry's identical check (plugin_registry/registry.h
        // `from_manifest`): `schema_version` is the GATE class (the CLASS
        // column), not a discriminant — a newer version means this host cannot
        // safely interpret the artifact AT ALL, so it is rejected regardless of
        // `pattern`. This door had NO manifest-level schema_version check until now
        // (a v99 manifest with valid ABI/pattern
        // passed every gate below and failed only on the downstream sidecar read) —
        // `parse_manifest` already reads `m.schema_version` (honoring the legacy
        // `version` key fallback too), nothing at this registry gated on it.
        if (m.schema_version > kPluginMaxSchemaVersion)
            throw std::runtime_error("host manifest uses schema v"
                + std::to_string(m.schema_version) + ", newer than this host supports "
                  "(v" + std::to_string(kPluginMaxSchemaVersion) + "); upgrade eagle");
        // The schema-v2 EXECUTION AXIS — the same shared gate the device
        // registry runs, one spelling (plugin_registry/manifest.h).
        validate_execution_axis(m, "host manifest");
        // Manifest-level `pattern` DISCRIMINANT ("the discriminant rule"):
        // mirrors the device registry's identical gate (plugin_registry/registry.h
        // `from_manifest`) — an unrecognized value is rejected naming the
        // supported set; absence stays lenient (this registry never dispatches
        // on `pattern`, and a pre-freeze manifest never stamped one).
        // ``load`` LAUNCHES, so it gates on the launch-certified set, the
        // subset of recognized families this registry will actually bind and run.
        // Two branches, ONE spelling (roles.h ``check_launch_certified_pattern``,
        // shared with the device registry): an UNRECOGNIZED family is told to upgrade
        // eagle; a RECOGNIZED-but-uncertified one (today: ``neural_block``) is told it
        // is not launchable here, with no upgrade suffix.
        check_launch_certified_pattern(m.pattern, "host manifest");
        // The structural invariant ("duplicate ids are rejected"), mirroring the
        // device registry's identical guard — checked over every declared entry
        // before any sidecar/artifact I/O, so a bad manifest fails fast with no
        // partial dlopen.
        {
            std::set<std::string> seen_ids;
            for (const auto& e : m.plugins)
                if (!seen_ids.insert(e.id).second)
                    throw std::runtime_error("host manifest has duplicate "
                        "plugin id '" + e.id + "'; every entry in plugins[] "
                        "must have a unique id");
        }
        // Manifest-entry `format` DISCRIMINANT: every declared
        // entry's artifact-container tag must be one this build knows how to load,
        // checked before any sidecar parse / dlopen (manifest.h
        // `validate_manifest_formats`, shared with the device registry).
        validate_manifest_formats(m);
        for (const auto& e : m.plugins) {
            Sidecar sc = parse_sidecar(dir + "/" + e.sidecar);
            // The per-entry `enabled` flag is CARRIED, not
            // dropped. This door used to load every entry and then run every
            // entry, so the same manifest meant two different things on the two
            // arms — the device registry has honoured `plugins[].enabled` since
            // it was written. Loading a disabled entry (rather than skipping it)
            // is the device twin's choice too: a bad artifact is still diagnosed
            // at load, and a caller can flip it on later.
            add_plugin(sc, dir + "/" + e.artifact, e.enabled);
        }
    }

    void set_enabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    /** @brief How many plugins one ``run()`` would call: the global flag AND
     *  each plugin's own manifest enable (v0.2.3). The host twin of the device
     *  registry's ``active()``, which the host side did not have. */
    int active() const
    {
        if (!enabled_)
            return 0;
        int k = 0;
        for (const auto& p : plugins_)
            if (p.enabled)
                ++k;
        return k;
    }

    /* The device registry's ``will_launch(index)`` / ``set_plugin_enabled``
     * have NO host counterpart here on purpose. This registry runs its plugins
     * itself (``run()``), so nothing outside walks them by index — a host
     * caller states a plugin's enable when it adds it (``add_plugin``'s third
     * argument) and asks ::active how many will run. The device pair exists
     * because a graph-capturing host DOES walk by index, opening one capture
     * per plugin. Adding untested twins here would be surface with no reader. */

    /** @brief Bind a vector (``out`` / ``vec_in`` / vector ``mutable``) buffer
     * by name — a host SoA pointer + sample count. */
    void bind_vector(const std::string& name, double* data, std::uint32_t n)
    {
        vec_[name] = VB{ data, n };
    }
    /** @brief Bind a matrix (``mat_in`` / matrix ``mutable``) buffer by name — a host
     *  flat ``(R*C, N)`` SoA pointer, sample count, and the caller's asserted
     *  ``(rows, cols)``. The shape is validated against the sidecar's declared
     *  ``(R, C)`` at ``run`` time (loud throw on mismatch — the host mirror of the
     *  device path's matrix shape guard). Packing is otherwise identical to a vector:
     *  a matrix GRef is a width-``R*C`` vector GRef (``dim = r*C + c``, sample-fastest). */
    void bind_matrix(
        const std::string& name, double* data, std::uint32_t n, int rows, int cols)
    {
        mat_[name] = MB{ data, n, rows, cols };
    }
    /** @brief Bind a flat handle (``per_sample`` / ``terminated`` / scalar
     * ``mutable``) by name — a host pointer. A ``lookup`` table is NOT bound here:
     * it goes through :func:`consolidate` (the registry-owned "bring your own data"
     * ), so a declared-but-unsupplied table yields a clear table-specific error
     * rather than the generic "unbound handle" one. */
    void bind_handle(const std::string& name, void* ptr)
    {
        handle_[name] = ptr;
    }
    /** @brief Bind a float64 ``uniform`` scalar by name — the kernel's
     *  ``AETHER_GRID_CONSTANT() Real p_<name>`` by-value parameter slot. Re-binding the same
     *  name overwrites (unchanged); binding a name already bound as int64 is refused
     *  (``plugin_registry/uniform_binding.h`` carries the reason and the wording,
     *  shared with the device registry). */
    void bind_uniform(const std::string& name, double value)
    {
        check_uniform_kind_free(UniformKind::Real, uniform_.count(name) != 0,
            uniform_int_.count(name) != 0, name, "host plugin registry");
        uniform_[name] = value;
    }
    /** @brief Bind an int64 ``uniform`` by name — the ADDITIVE int arm of the same
     *  role, the host twin of the device registry's ``bind_uniform_int``.
     *  A generated kernel spells an integer quantity ``Int`` == ``long long``
     *  (the code generator's spellings), so the parameter slot is ``AETHER_GRID_CONSTANT() Int
     *  p_<name>`` and this packs the matching 8 signed bytes into the SAME params[]
     *  array the device path builds for cuLaunchKernel. Same one-type-per-name
     *  refusal, in both directions. */
    void bind_uniform_int(const std::string& name, UniformInt value)
    {
        check_uniform_kind_free(UniformKind::Int, uniform_.count(name) != 0,
            uniform_int_.count(name) != 0, name, "host plugin registry");
        uniform_int_[name] = value;
    }

    /** @brief Consolidate a read-only ``lookup`` table (a table or a shared constant)
     *  — the host twin of the device registry's ``consolidate``. Where the device path
     *  allocates device memory and uploads @p host, the CPU path has nothing to upload:
     *  the table already lives in host memory the kernel reads directly, so this just
     *  RECORDS the caller's pointer + flat element @p count under @p name (immutable for
     *  the run). ``run`` then packs it by value as a flat handle. Re-consolidating the
     *  same name replaces the prior binding. Unlike the interface bindings, a declared
     *  table MUST be consolidated before ``run`` or the launch fails loudly (naming it).
     */
    void consolidate(const std::string& name, const double* host, std::size_t count)
    {
        tables_[name] = TB{ host, count };
    }

    /** @brief The distinct lookup tables the ENABLED plugin set declares (name + flat
     *  count), deduplicated by name — exactly what the caller must consolidate. Tables
     *  share ONE global name namespace, so two plugins declaring the same name with
     *  DIFFERENT element counts is a hard error (it would otherwise bind both to one
     *  undersized buffer -> a silent out-of-bounds read). The host mirror of the device
     *  registry's ``declared_tables`` shape guard. */
    std::vector<BufferInfo> declared_tables() const
    {
        std::vector<BufferInfo> out;
        std::map<std::string, std::size_t> seen; // name -> flat element count
        for (const auto& p : plugins_) {
            for (const auto& b : p.sidecar.buffers) {
                if (b.kind != "lookup")
                    continue;
                auto it = seen.find(b.name);
                if (it == seen.end()) {
                    seen.emplace(b.name, b.count);
                    out.push_back(b);
                } else if (it->second != b.count) {
                    throw std::runtime_error("lookup table '" + b.name
                        + "' declared with conflicting element counts ("
                        + std::to_string(it->second) + " vs "
                        + std::to_string(b.count) + ") across plugins; tables share "
                          "one global name namespace");
                }
            }
        }
        return out;
    }

    /** @brief Run every enabled plugin over @p n samples on the host, packing
     * each kernel's args from the bound buffers in ``arg_spec`` order — the
     * exact same packing the device registry does for cuLaunchKernel. Returns
     * the number of plugins launched (0 when disabled). */
    int run(std::int32_t n)
    {
        return run_partition(eagle::exec::Partition::whole(n));
    }

    /** @brief Run every enabled plugin SERIALLY over ONE partition.
     *
     *
     *  ``run(n)`` is exactly this with ``Partition::whole(n)``, so the legacy
     *  opening and the partitioned one are ONE implementation. A v2 plugin
     *  receives the triple; a v1 plugin receives its ``int32_t n`` and is REFUSED
     *  a non-whole partition. This is the SERIAL arm — the correctness
     *  reference implementation (fork F-c); parallel host execution is ``eagle::exec::HostTeam``'s
     *  alone, driven from :func:`pack` + :func:`entry_v2`. */
    int run_partition(const eagle::exec::Partition& part)
    {
        if (!enabled_)
            return 0;
        int launched = 0;
        for (auto& p : plugins_) {
            if (!p.enabled) // added in v0.2.3: honour the manifest's flag
                continue;
            eagle::exec::check_legacy_whole_view(p.abi_version, part,
                "host plugin '" + p.sidecar.kernel + "'");
            PackedArgs packed = pack_(p, part.nSamples);
            if (p.abi_version >= 2)
                p.fn2(packed.params.data(), part.base, part.count, part.nSamples);
            else
                p.fn(packed.params.data(), std::int32_t(part.count));
            ++launched;
        }
        return launched;
    }

    /** @brief Which ABI generation the plugin at @p index speaks (1 or 2). */
    int abi_version(std::size_t index) const { return at_(index).abi_version; }

    /** @brief The plugin at @p index's aether-abi/2 entry — what an execution
     *  structure (``eagle::exec::HostTeam``) drives. Throws for a v1 plugin: a
     *  legacy artifact has no triple to be driven with. */
    HostEntryV2 entry_v2(std::size_t index) const
    {
        const Plugin& p = at_(index);
        if (p.abi_version < 2 || p.fn2 == nullptr)
            throw std::runtime_error("host plugin '" + p.sidecar.kernel
                + "' is aether-abi/1 and has no partitioned entry; it can only be "
                  "run whole-view (L13)");
        return p.fn2;
    }

    /** @brief The plugin at @p index's legacy entry (null for a v2 plugin). */
    HostEntry entry_v1(std::size_t index) const { return at_(index).fn; }

    /** @brief Pack the plugin at @p index's role args for a run spanning
     *  @p nSamples samples — the host twin of the device registry's ``pack``,
     *  and the SAME params[] array both build. Every role is passed WHOLE; the
     *  count that reaches ``nsamples`` and every view extent is the TRUE
     *  ``nSamples``, never a partition's ``count``. */
    PackedArgs pack(std::size_t index, std::int64_t nSamples) const
    {
        return pack_(at_(index), nSamples);
    }

    std::size_t size() const { return plugins_.size(); }

private:
    const Plugin& at_(std::size_t index) const
    {
        if (index >= plugins_.size())
            throw std::out_of_range("plugin index " + std::to_string(index)
                + " is out of range: " + std::to_string(plugins_.size())
                + " plugin(s) loaded");
        return plugins_[index];
    }

    PackedArgs pack_(const Plugin& p, std::int64_t nSamples) const
    {
        PackedArgs out;
        {
            const std::size_t na = p.sidecar.arg_spec.size();
            // Per-run stable storage: reserved (never resized), so the
            // addresses pushed into `params` stay valid through the entry call.
            std::vector<GRefMirror>& grefs = out.grefs;
            grefs.reserve(na);
            std::vector<ScalarHandle>& handles = out.handles;
            handles.reserve(na);
            std::vector<double>& scalars = out.scalars;
            scalars.reserve(na);
            std::vector<UniformInt>& ints = out.ints;
            ints.reserve(na);
            std::vector<unsigned>& counts = out.counts;
            counts.reserve(na);
            std::vector<void*>& params = out.params;
            params.reserve(na);
            const unsigned n = unsigned(nSamples);
            for (const auto& a : p.sidecar.arg_spec) {
                if (a.role == "out" || a.role == "vec_in") {
                    const VB vb = lookup_vec_(p.sidecar.kernel, a.name);
                    grefs.push_back(make_gref(vb.data, vb.n, kEagleAbiDeviceCPU));
                    params.push_back(&grefs.back());
                } else if (a.role == "mat_in") {
                    // a matrix binds through the SAME GRef mirror as a vector.
                    grefs.push_back(checked_matrix_gref_(p.sidecar, a.name));
                    params.push_back(&grefs.back());
                } else if (a.role == "mutable") {
                    std::string dt;
                    for (const auto& mi : p.sidecar.mutables)
                        if (mi.name == a.name) {
                            dt = mi.dtype;
                            break;
                        }
                    if (dt == "vector") {
                        const VB vb = lookup_vec_(p.sidecar.kernel, a.name);
                        grefs.push_back(make_gref(vb.data, vb.n, kEagleAbiDeviceCPU));
                        params.push_back(&grefs.back());
                    } else if (dt == "matrix") {
                        grefs.push_back(
                            checked_matrix_gref_(p.sidecar, a.name));
                        params.push_back(&grefs.back());
                    } else {
                        handles.push_back(
                            make_handle(lookup_handle_(p.sidecar.kernel, a.name),
                                unsigned(n), kEagleAbiDeviceCPU));
                        params.push_back(&handles.back());
                    }
                } else if (a.role == "nsamples") {
                    counts.push_back(unsigned(n));
                    params.push_back(&counts.back());
                } else if (a.role == "per_sample" || a.role == "terminated") {
                    handles.push_back(
                        make_handle(lookup_handle_(p.sidecar.kernel, a.name),
                            unsigned(n), kEagleAbiDeviceCPU));
                    params.push_back(&handles.back());
                } else if (a.role == "lookup") {
                    // A consolidated read-only table: a flat handle over the host
                    // buffer the caller supplied at consolidation (no upload — the
                    // data is already host-resident), read at a user-computed index.
                    handles.push_back(
                        make_handle(lookup_table_(p.sidecar.kernel, a.name),
                            unsigned(n), kEagleAbiDeviceCPU));
                    params.push_back(&handles.back());
                } else if (a.role == "uniform") {
                    // Packed BY VALUE in one of two spellings — float64
                    // (`Real p_<name>`) or int64 (`Int p_<name>`). The
                    // arg_spec row is a [role, name] pair and carries no dtype,
                    // so the type is the one the CALLER declared by choosing a
                    // binder; binding one name through both is refused at bind
                    // time, so at most one map holds it. Bound through neither
                    // keeps the unchanged float64 "unbound uniform" error.
                    if (uniform_int_.count(a.name)) {
                        ints.push_back(
                            lookup_uniform_int_(p.sidecar.kernel, a.name));
                        params.push_back(&ints.back());
                    } else {
                        scalars.push_back(
                            lookup_uniform_(p.sidecar.kernel, a.name));
                        params.push_back(&scalars.back());
                    }
                } else if (a.role == "wide_in" || a.role == "wide_out"
                    || a.role == "accum_out") {
                    // The WIDE / ACCUMULATE planes: in the
                    // role register but unlaunchable by EITHER C++ registry until
                    // now -- seven arms, then a throw, while the generator's producer was
                    // already emitting a twelfth role. All three ride the SAME
                    // plain 32-byte scalar-handle ABI as per_sample/lookup; what
                    // differs is the plane's ADDRESSING, which is the body's
                    // business and is exactly what the partition triple makes
                    // exact. The extent is the TRUE nSamples of the run, never a
                    // partition's count — a (rows, nSamples) plane handed a
                    // partition length would re-target every column.
                    handles.push_back(
                        make_handle(lookup_handle_(p.sidecar.kernel, a.name),
                            n, kEagleAbiDeviceCPU));
                    params.push_back(&handles.back());
                } else {
                    throw std::runtime_error("host plugin '" + p.sidecar.kernel
                        + "': unknown arg role '" + a.role + "'");
                }
            }
        }
        return out;
    }

    struct VB {
        double* data    = nullptr;
        std::uint32_t n = 0;
    };
    struct MB {
        double* data    = nullptr;
        std::uint32_t n = 0;
        int rows        = 0;
        int cols        = 0;
    };
    // A consolidated lookup table: the caller's host pointer + its flat element count.
    // NOT owned (no upload, no free) — it aliases host memory the caller keeps alive.
    struct TB {
        const double* data = nullptr;
        std::size_t count  = 0;
    };
    struct Plugin {
        Sidecar sidecar;
        void* handle = nullptr;   // dlopen handle
        HostEntry fn = nullptr;   // aether-abi/1 entry (null on a v2 plugin)
        HostEntryV2 fn2 = nullptr;// aether-abi/2 entry (null on a v1 plugin)
        int abi_version = 1;
        bool enabled = true;      // the manifest's per-entry flag (v0.2.3)
    };

    VB lookup_vec_(const std::string& k, const std::string& name) const
    {
        auto it = vec_.find(name);
        if (it == vec_.end())
            throw std::runtime_error(
                "host plugin '" + k + "': unbound vector '" + name + "'");
        return it->second;
    }
    MB lookup_mat_(const std::string& k, const std::string& name) const
    {
        auto it = mat_.find(name);
        if (it == mat_.end())
            throw std::runtime_error(
                "host plugin '" + k + "': unbound matrix '" + name + "'");
        return it->second;
    }
    // The bound matrix packed as a vector GRef mirror, after asserting the caller's
    // (rows, cols) matches the sidecar's declared (R, C) for this name (loud throw on
    // mismatch — the host mirror of the device coerce_mat_inputs shape guard).
    GRefMirror checked_matrix_gref_(
        const Sidecar& sc, const std::string& name) const
    {
        const MB mb = lookup_mat_(sc.kernel, name);
        auto ds = sc.mat_shapes.find(name);
        if (ds != sc.mat_shapes.end()
            && (mb.rows != ds->second.first || mb.cols != ds->second.second))
            throw std::runtime_error("host plugin '" + sc.kernel + "': matrix '"
                + name + "' bound as (" + std::to_string(mb.rows) + ", "
                + std::to_string(mb.cols) + ") but the kernel was compiled for ("
                + std::to_string(ds->second.first) + ", "
                + std::to_string(ds->second.second) + ')');
        return make_gref(mb.data, mb.n, kEagleAbiDeviceCPU);
    }
    void* lookup_handle_(const std::string& k, const std::string& name) const
    {
        auto it = handle_.find(name);
        if (it == handle_.end())
            throw std::runtime_error(
                "host plugin '" + k + "': unbound handle '" + name + "'");
        return it->second;
    }
    // The consolidated table for @p name, as a flat handle. A declared-but-unsupplied
    // table is a clear error naming it (the host mirror of the device registry's
    // "not supplied at consolidation" guard) — the kernel only reads, so the const is
    // cast away purely to fit the 8-byte handle POD.
    void* lookup_table_(const std::string& k, const std::string& name) const
    {
        auto it = tables_.find(name);
        if (it == tables_.end())
            throw std::runtime_error("host plugin '" + k + "': lookup table '" + name
                + "' was not supplied at consolidation "
                  "(call registry.consolidate(\""
                + name + "\", data, count))");
        return const_cast<void*>(static_cast<const void*>(it->second.data));
    }
    double lookup_uniform_(const std::string& k, const std::string& name) const
    {
        auto it = uniform_.find(name);
        if (it == uniform_.end())
            throw std::runtime_error(
                "host plugin '" + k + "': unbound uniform '" + name + "'");
        return it->second;
    }
    // The int64 twin. Only ever called after ``uniform_int_.count(name)`` said yes, so
    // the miss branch is unreachable through ``run`` — kept (and worded like its
    // float64 sibling) so a future direct caller cannot get a silent zero.
    UniformInt lookup_uniform_int_(
        const std::string& k, const std::string& name) const
    {
        auto it = uniform_int_.find(name);
        if (it == uniform_int_.end())
            throw std::runtime_error(
                "host plugin '" + k + "': unbound uniform '" + name + "'");
        return it->second;
    }

    std::vector<Plugin> plugins_;
    std::map<std::string, VB> vec_;
    std::map<std::string, MB> mat_;
    std::map<std::string, void*> handle_;
    std::map<std::string, double> uniform_;
    std::map<std::string, UniformInt> uniform_int_; // int64 uniforms
    std::map<std::string, TB> tables_; // consolidated lookup tables (aliased, not owned)
    bool enabled_ = true;
};

} // namespace cpu
} // namespace eagle
