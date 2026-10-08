# Host plugin parity

This page tracks feature parity between the CPU host plugin path and the
CUDA device plugin path: what a device plugin can do versus what its CPU
host twin can do, feature by feature. It is the **consumer** half of the
parity check — what `eagle::cpu::PluginRegistry` (`plugin/host_registry.h`)
and `eagle.host_launch.HostPluginLibrary` (`python/eagle/host_launch.py`)
accept, bind, and run once a producer hands them a `.so`. The matching
**producer** half — what the code generator's `compile_to_host` emits, and
which roles are still unimplemented in its codegen — lives in the code
generator's own `host-codegen-parity.md`; read the two together, since most
rows depend on both sides.

## Where the host path sits

`eagle::cpu::PluginRegistry` is the CPU twin of the CUDA
`eagle::cuda::PluginRegistry` documented in {doc}`plugin_schema`: same
sidecar/manifest schema, same `arg_spec` role vocabulary, same
`GRefMirror`/`ScalarHandle` binary-ABI PODs. Where the device registry issues
`cuLaunchKernel` against a `dlopen`'d `.ptx`/`.cubin`, the host registry
`dlopen`s a `.so` and calls a plain `void <kernel>_host(void* const*
params, int32_t n)` entry that runs its own `#pragma omp parallel for` — the
**exact same** `params[]` array, packed in `arg_spec` order, that the device
path would hand `cuLaunchKernel`. `eagle.host_launch.HostPluginLibrary` is
the Python peer, `ctypes`-driven, for the same ABI. Both accept host memory
directly — numpy arrays' `.ctypes.data`, or a torch **CPU** tensor's raw
pointer via `eagle.interop.host_ptr` — with no device upload step, since the
data is already where the kernel reads it.

A role that a loader accepts is not automatically a role a *producer* can
emit — a registry may happily bind and launch a matrix-shaped argument, say,
while a producer does not emit that shape for host yet. Conversely, float32
is closed off entirely *by this side*, deliberately, regardless of what any
producer emits.

## How to read the matrix

Rows are the input/output/precision **features** a plugin can carry; columns
cross the two kernel patterns (**pure**, **vector** — [the schema's role vocabulary](https://amasat01.github.io/raptor/content/devguide/plugin_schema.html#role-vocab)'s
8-role and 6-role subsets) with the two native scalar types the C++ host
runner will even load (**f64**, **f32**; SoftDouble is never a host target —
see the SoftDouble row). A cell is:

- **CLOSED** — a compiled host artifact round-trips through this consumer
  and matches the in-process GPU numerics (the code generator's own
  round-trip suite, the `*_matches_direct` pattern).
- **OPEN** — not yet round-trip proven. Some OPEN cells are consumer-ready
  today (the registry already binds the role generically) and are waiting
  purely on a producer; others need consumer-side work too.
- **N/A** — one of two things: the *pattern* has no such role (the vector
  pattern owns no `mutable` sink; see [the schema's role vocabulary](https://amasat01.github.io/raptor/content/devguide/plugin_schema.html#role-vocab)'s per-kind subsets),
  or the *precision* has no host target — every f32 column is N/A because
  the host runs native float64 only.

Every f32 column above is also blocked by a fact this consumer enforces
explicitly: `eagle::cpu::PluginRegistry::add_plugin` throws unless
`sidecar.scalar_type` is empty or exactly `"float64"` —

```cpp
if (!sc.scalar_type.empty() && sc.scalar_type != "float64")
    throw std::runtime_error("host plugin '" + sc.kernel
        + "': the C++ host runner executes float64 only (got '"
        + sc.scalar_type + "')");
```

— a deliberate restriction ("matching the device host runner's dtype
policy" per the source comment). `HostPluginLibrary.__init__` carries the
same `scalar_type != "float64"` gate, raising before `dlopen` with a
matching "float64 only" message
(`test_host_plugin_rejects_nonfloat64_scalar_type`). Both host paths — C++
and Python — refuse a device-only float32 / SoftDouble sidecar identically:
native float64 is the only host target in v1. Widening the mirror to a
genuine f32 host wire ABI stays a possible future evolution if a consumer
ever materializes.

## The matrix

| Feature | Pure f64 | Pure f32 | Vector f64 | Vector f32 |
|---|---|---|---|---|
| Scalar inputs (`per_sample`, read-only) | **CLOSED** | **N/A** — f32 host unsupported | **CLOSED** | **N/A** — f32 host unsupported |
| Vector inputs (`vec_in`) | **CLOSED** | **N/A** — f32 host unsupported | **CLOSED** | **N/A** — f32 host unsupported |
| Matrix inputs (`mat_in`) | **CLOSED** | **N/A** — f32 host unsupported | **CLOSED** | **N/A** — f32 host unsupported |
| Lookup tables (1D/2D) | **CLOSED** | **N/A** — f32 host unsupported | **CLOSED** | **N/A** — f32 host unsupported |
| Mutables — float scalar | **CLOSED** | **N/A** — f32 host unsupported | N/A | N/A |
| Mutables — int scalar | **CLOSED** | **N/A** — f32 host unsupported | N/A | N/A |
| Mutables — vector | **CLOSED** | **N/A** — f32 host unsupported | N/A | N/A |
| Mutables — matrix | **CLOSED** | **N/A** — f32 host unsupported | N/A | N/A |
| Terminated mask | **CLOSED** | **N/A** — f32 host unsupported | **CLOSED** | **N/A** — f32 host unsupported |
| `nsamples` (explicit launch size) | **CLOSED** | **N/A** — f32 host unsupported | **CLOSED** | **N/A** — f32 host unsupported |
| Derivative kernels — generated (VJP/JVP) | **CLOSED** | **N/A** — f32 host unsupported | **N/A** — pure-pattern only | **N/A** — pure-pattern only |
| Derivative kernels — custom | **CLOSED** | **N/A** — f32 host unsupported | **N/A** — pure-pattern only | **N/A** — pure-pattern only |
| numpy / torch CPU zero-copy | **CLOSED** | **N/A** — f32 host unsupported | **CLOSED** | **N/A** — f32 host unsupported |
| SoftDouble — explicit host rejection | **CLOSED** | **N/A** — softdouble is float64-only | **CLOSED** | **N/A** — softdouble is float64-only |

Tally: 22 CLOSED / 0 OPEN / 34 N/A, over 56 cells (14 rows × 4 columns).

## Row notes

**Vocabulary inputs.** Both registries (`PluginRegistry::run` in
`host_registry.h`, `HostPluginLibrary.run` in `host_launch.py`) pack
`per_sample`/`vec_in`/`lookup` identically — one generic role → `ScalarHandle`
or `GRefMirror` branch, no per-feature special-casing. A pure kind can bind a
non-Mutable `per_sample` scalar, and a vector kind can read a bare
`per_sample` scalar alongside its `vec_in`; a pure kind can also read a
`vec_in` with no accompanying `mat_in`. Both roles' packing is role-generic.

**Matrix.** The `mat_in` role is a valid part of the schema-v1 vocabulary —
[the schema's role vocabulary](https://amasat01.github.io/raptor/content/devguide/plugin_schema.html#role-vocab) documents it explicitly — and the reference C++
`inject()` path packs matrix arguments the same way the device GRef ABI
does; Mutable-matrix write-back rides the same packing primitives.

**Mutable write patterns.** The registries bind `mutable` generically
(float/int scalar via `ScalarHandle`, vector/matrix via `GRefMirror`). All
four Pure-f64 mutable shapes round-trip: float scalar (single-assignment
RMW), int scalar (`Int = long long`), a genuine vector RMW, and matrix
write-back. Control-flow write patterns (branch-commit merges, loop-carried
accumulation) and the matrix shape-mismatch guard are covered on top of the
scalar baseline; the int-mutable scope is device scope.

**Terminated / nsamples.** Both roles are bound and run by the existing
round-trip tests (an all-clear terminated mask, an explicit sample count) —
the packing (`counts.push_back(unsigned(n))`,
`make_handle(lookup_handle_(...))` for `terminated`) is generic and shared
with every other role. No test yet drives a mask that actually skips
samples on the host path.

**Derivative kernels.** A generated (VJP/JVP) derivative kernel is an
ordinary pure-pattern artifact from the registry's point of view (the seed /
tangent rides the `per_sample` ABI, each gradient / tangent-output slot is a
`mutable`), so it host-launches through this consumer with no
derivative-specific code, proven end to end against finite differences. The
Vector columns are **N/A**: a generated derivative is *always* pure-pattern
(it commits `mutable` gradient/tangent slots), so there is no vector-pattern
derivative artifact. A custom derivative kernel is a hand-written pure kernel
registered on the primal (`set_custom_vjp` / `set_custom_jvp`) whose binding
shape matches the generated derivative's ABI, so from this consumer's point
of view it is an ordinary pure-pattern artifact — the sidecar simply carries
an additive `derivative` block this reader parses and exposes on the
loaded-kernel metadata (structurally indistinguishable from a generated
derivative's; recompute-only means a populated `residuals` list is rejected
at load). The eagle-side parse / metadata / reject tests are in
`tests/test_Sidecar.cpp` and `python/tests/test_derivative_sidecar.py`. The
three remaining cells are **N/A** for the same reason: derivative kernels
are pure-pattern f64 only, and f32 host is unsupported.

**Zero-copy.** Proven for both f64 patterns. This consumer's zero-copy
capability is proven generically (`test_host_launch_torch.py` drives
`HostPluginLibrary` + `eagle.interop.host_ptr` over both a numpy array and a
torch CPU tensor, writing the caller's buffer in place with no copy), the
numpy leg is zero-copy by construction (`.ctypes.data`), and a deployed
`.so` RMWs a caller's torch CPU tensor in place. The binding path is
pattern-independent, so both Pure and Vector f64 close together.

**SoftDouble.** The producer side rejects `scalar_type=softdouble` up front
with a clear error that names SoftDouble and points at the float64 path, so
no artifact is ever built — and this consumer's Python `HostPluginLibrary`
carries the same `scalar_type != "float64"` gate the C++ registry already
enforces (`test_host_plugin_rejects_nonfloat64_scalar_type`), so a
device-only SoftDouble / float32 sidecar is refused on *both* the C++ and
Python host paths. The f32 columns are **N/A**: SoftDouble emulation runs
over float64 arrays by construction (a float32 + emulation kernel is a
compile-time `TypeError`), so there is no softdouble-f32 artifact to
reject.

## Keeping this page current

Flip exactly the cells a producer/consumer change proves — from `**OPEN**`
to `**CLOSED**`, citing the new test — and update the tally line. Keep this
page and the code generator's `host-codegen-parity.md` in lockstep: a row
closes only when *both* the producer emits the shape and this consumer
round-trips it, so a one-sided fix should still land the cell as OPEN here
with an updated note, not CLOSED.
