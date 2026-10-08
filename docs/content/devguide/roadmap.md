# Roadmap (deferred work)

Consult this page when you are wondering "did anyone already think about
this, and if so why isn't it done yet?" — it is eagle's single list of
capabilities that are deliberately not built yet, what would trigger building
them, and roughly how big each one is. Every item below is a *recorded*
decision, not an oversight: eagle's own docs (linked from each item) already
explain the design reasoning in full; this page exists so the reasoning is
findable in one place instead of only where the gap happens to surface.

## Plugin registry

- **float32 on the C++ `PluginRegistry`.** `eagle::cuda::PluginRegistry`'s
  entire binding surface (`bind_vector`, `bind_uniform`, `consolidate`) is
  `double`-typed, and `from_manifest` rejects any sidecar declaring
  `scalar_type: "float32"` before it ever reaches the loader — a deliberate
  fail-fast, present since eagle's first commit, not a bug found later. The
  reasoning and the exact rejection point are written up in
  {doc}`plugin_schema`'s `float32 on the C++ registry` section, which also
  names the trigger: build the float-typed twin of the binding surface (a
  few hundred lines, on the order of the `double` surface it mirrors) once a
  genuine C++ consumer of an f32 artifact exists — foreseeably a future
  neural-engine deployment. Until that consumer shows up this stays a
  documented capability, not an open task. Note this is a **different** kind
  of gap from SoftDouble, which shares the same registry's `double` binding
  surface and needs no widening — `tests/test_PluginRegistryDtype.cu`'s
  `SoftdoubleLoadsAndLaunches` proves that path already works, because
  SoftDouble is bit-identical float64 underneath.

## GPU architecture coverage

- **SM 100/120 (Blackwell).** `EAGLE_CUDA_ARCHS=all` includes Blackwell
  whenever the `nvcc` in use compiles it (CUDA 13, late CUDA 12), and the
  `native` default builds for whatever GPU the machine has. The CI build
  matrix still targets Pascal through Hopper with CUDA 12.6; a CUDA 13 arm
  and Blackwell measurements are planned.

## Host plugin parity — closed out, nothing carried forward

{doc}`host-plugin-parity` is the row-by-row tracking page for parity between
the CUDA and CPU host plugin paths. Its tally line reads **22 CLOSED / 0
OPEN**, over the full 56-cell matrix — every row that page tracks is closed.
It stays in the devguide as the record of how that closure was reached, and
as the template this page and {doc}`test-map` both follow for "how do I keep
a tracking page honest as work lands."

## What is *not* on this list, on purpose

One thing that might look like a gap is a deliberate design choice instead,
already documented where it lives and not repeated here as an open item. `plugin_schema`'s "C++ reference-consumer scope boundary" section
lists three manifest fields the C++ registry never reads (`vec_widths`,
`accumulate`/`sink`, the convenience arrays) — not a functional gap, since
`arg_spec` alone drives binding.
