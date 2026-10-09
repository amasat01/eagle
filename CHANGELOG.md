# Changelog

## 0.5.0 (2026-10-09)

**Free-threaded CPython, GIL-free.** The compiled core now declares free-threading
support, so on 3.13t and 3.14t eagle runs without the GIL and the GIL stays off after
import. Python-side state is thread-safe: any number of threads may build, plan and
cache concurrently, distinct objects need no synchronisation, and one stateful object
(a stream, capture, launcher, graph, plan and so on) shared between threads is
memory-safe with its calls serialised; the order of those calls is the caller's, as
for a NumPy array or a CuPy stream. CUDA graph capture is per thread: a nested capture
on one thread raises, and two threads may capture concurrently. GPU routes are
supported on free-threaded 3.14; on 3.13t the CPU route runs. CI gained
free-threading stress legs, and `raptor-core>=0.3,<0.4` is now required for the
shared conformance harness.

**Wheel GPU code.** The wheel now ships GPU code for every architecture the CUDA 12.9
toolkit compiles from sm_60 up (Pascal through Blackwell), plus PTX for newer GPUs.

A per-sample plane whose shape reads both as component-major `(w, N)` and as
sample-major `(N, w)` (a `(w, w)` array, or a square matrix head with
`N == R == C`) is now refused, naming the argument, its shape, both readings
and the fix; before, it was silently taken as component-major, which transposed
a C-contiguous sample-major array. Say which axis holds the samples, zero-copy:
per call with `layout="samples_first"` or `layout="samples_last"` on
`Plan.run`, `Plan.bind`, `BoundPlan.rebind`, `eagle.until_done`,
`eagle.run_until_done`, `eagle.simulate`/`eagle.simulation`, the `Loaded*`
kernels and the torch bridge, or per array with the new `eagle.samples_first(x)`
/ `eagle.samples_last(x)` markers, which work for any shape, are refused when
they contradict the shape, and win over the call's `layout=`. hawk's markers are
accepted too (they share a small protocol, `__raptor_samples_axis__` plus
`.array`). Shapes that are not ambiguous behave as before. A plane literally
named `layout` can no longer be passed by keyword to these doors.

## 0.4.1

Wheels now cover CPython 3.9 to 3.14, plus the free-threaded builds 3.13t and
3.14t (manylinux_2_28 x86_64); the package requires Python 3.9 or newer. The
free-threaded wheels first appeared here; 0.5.0 declares GIL-free support for them.
The `cuda12` / `cuda13` extras pick CuPy by Python version (CuPy 14 on 3.10 and
newer, CuPy 13.6 on 3.9); on 3.13t CuPy 14 has no wheel, so those extras do not
resolve there and eagle runs on the CPU route. The package's own code no longer
uses `zip(strict=True)` directly, which needs Python 3.10.
Stream wrapping uses `Stream.from_external` on CuPy 14 (no more `ExternalStream`
DeprecationWarning), falling back to `ExternalStream` on CuPy 13.
Host buffer import on NumPy older than 2.1 (no DLPack read-only signalling, e.g. the
last NumPy for Python 3.9) now reads access from `__array_interface__`, so read-only
arrays import as read-only instead of raising or reporting `unknown`.

## 0.4.0 (first public release)

eagle is the RAPTOR family's GPU execution layer: kernel launch, CUDA
graph capture, a plugin registry, and a host/device dual mode, with a
Python package built on aether.

Adds the generic DLPack layer. `eagle.interop.import_buffer` turns any
DLPack, `__cuda_array_interface__` or `__array_interface__` producer
(numpy, cupy, torch, NVIDIA Warp, ...) into a zero-copy `BufferView` that
reports its `access`, `owner`, `producer` and `stream`, validates
`Requirements` with one named refusal per failed check, keeps the producer
alive, and re-exports through `__dlpack__` under the DLPack 1.0 stream
contract: an event record plus a stream wait, never a host synchronization.
The C++ half is `plugin/interop.h` over aether 0.2.0's
`aether/interop/Buffer.h`; `plugin/dlpack_bridge.h` now accepts CPU and
CUDA tensors of either DLPack generation and carries the read-only flag.
See the documentation's interoperability contract page.

Adds `eagle.frameworks.torch.function`: a hawk per-sample kernel as a
`torch.autograd.Function`. The forward pass runs the kernel, the backward pass
the reverse-mode kernel hawk derives from it, and forward-mode AD
(`torch.autograd.forward_ad`) the derived tangent kernel. CUDA tensors run
through eagle's launch path on torch's current stream, CPU tensors through
hawk's host runtime; planes cross zero-copy through `import_buffer`. A sample
marked by the kernel's `Terminated` mask contributes zero gradient. torch is
an optional extra (`raptor-eagle[torch]`); `import eagle` stays torch-free.

Splits the compiled package in two. `eagle._core` is now plain C++ with no
CUDA toolkit, runtime or driver dependency, so `import eagle` and every host
structure work on a machine without an NVIDIA driver. All CUDA work lives in
`eagle/libeagle_cuda.so`, a plugin that the core loads on first use through the
`eagle-backend/1` C seam (`python/src/seam/eagle_backend.h`). The plugin bundles
no CUDA runtime: the only CUDA library it loads is the NVIDIA driver's
`libcuda.so.1`, it shares the device's primary context with torch, CuPy or any
other CUDA user in the process, and one build serves CUDA 12 and newer drivers
(a capability group whose driver entry points are missing, such as conditional
graph nodes before a 12.3 driver, is reported unavailable on its own). `$EAGLE_BACKEND_CUDA` names a plugin file
to load instead. Without a usable plugin, a device call raises the new
`eagle.BackendUnavailable` (a `RuntimeError`) and states why;
`eagle._core.cuda_backend()` describes what was loaded. The Python API is
unchanged. The creators (`Stream`, `StreamCapturer`, `CaptureFork`,
`CaptureConditional`, `Graph`, `GraphComposer`) gain an optional `device`
argument; the default, -1, keeps today's behaviour: the device of the stream
passed in, or the current device.

Adds active-set compaction. `eagle.ActiveSet` gathers the samples a
`terminated` mask has not finished into an ascending index map and a count, on
the device (the CUDA backend's new `filtering` capability group, one symbol,
`eagle_backend_compact_device`: enqueue-only, so it records into a graph) or on
the host; a hawk kernel built with `Guard(active_set=True)` reads the map, so
live samples share warps while every plane stays in place. A bound host plan
over such a kernel runs only the live range. `eagle.compaction_body` builds the
`repeat_while` body that compacts every 16 steps (fewer than 4 is refused) when
a sample finished since the last compaction, and a `repeat_while` body may now
be a tuple of parts run in order, each a callable or a `Skippable`. The C++
twin is `eagle/filtering/Compact.h`; `cuda::Scan::enqueue` issues the scan
without capturing it.

Renames `GraphPipeline.stream_ptr` to `GraphPipeline.stream`, the family's
shared name for a stream handle. There is no alias.

Requires aether 0.2.

### Earlier development

Before this release, development built the header-only C++23/CUDA launch engine (graph capture,
plugin registry, host/device dual mode, conditional and forked capture, reductions and scans) and its
Python package on aether. This is eagle's first public release.

The finished count of a fast run is a word of the run's counter block: one
memset zeroes every counter and one transfer reads them back, with no
device-to-device copy, and `Simulation.run` no longer synchronises again after
a fast run's own blocking readback.

`EAGLE_CUDA_ARCHS` picks the CUDA architectures when `CMAKE_CUDA_ARCHITECTURES`
is not given: `native` (the default; with no GPU, PTX for the oldest
architecture nvcc compiles, at least `sm_60`), `all` (every architecture from
Pascal to Blackwell that nvcc compiles), or one architecture or a list.
Test fixtures follow the toolkit's oldest architecture, `cudaGraphGetEdges` goes
through the CUDA 13 compatibility shim, and device clocks are read as device
attributes. The Python package requires Python 3.10, and the `cuda12` and
`cuda13` extras pick the matching CuPy (`cuda` stays an alias of `cuda12`).

A fast run on an artifact whose entries count the samples finished on entry
(`entry_counts_finished`) zeroes the finished counter instead of launching the
mask-count kernel: one launch per run, as for a hand-written kernel. The
captured GEMM binds the CUDA 13 cuBLAS (`libcublas.so.13`) as well as CUDA
12's, and the C++ plugin host example uses the device's primary context, so it
builds on CUDA 13.

Above the device's resident capacity, a fast run picks its device entry per
run: the first run takes the size rule (the persistent entry), and when the
batch keeps most lanes busy -- its step fill, the steps run over the batch
times its longest sample, is at least `max(FUSED_MIN_STEP_RATIO, 1 - 12 /
step_ops)` -- the second run measures the one-launch fused entry and the
faster one is kept; a batch whose step fill is lower stays on the persistent
entry without a probe. The fill is read from the kernel's own step-sum
report on every run that decides and on one settled run in
`STEP_FILL_EVERY`; a change of the readback or of the fill re-opens the choice,
and the slower entry is measured again every 64 runs. `RunReport.mode` names
the entry a run took and `RunReport.probe` says whether it was a measuring
run; `_fast_mode=` still pins one entry. Settled runs are not timed, the
launch arguments are packed once per runner, `reset()` clears with two
memsets, and a fast-path runner allocates its active-set map only when a
graph path first needs it (4 MiB less at a million samples).

eagle's default CUDA architecture list keeps only the architectures the
installed `nvcc` still compiles, and the native module links the C++ runtime
statically only when the toolchain has a static `libstdc++`.

A host run's first call no longer rescans the system libraries for `dladdr`
on every entry lookup: it is found once per process, so the first
`until_done(...).run()` in a process is several milliseconds faster.

eagle builds with CUDA 13 as well as CUDA 12: the graph and capture calls whose
signatures CUDA 13 changed go through one wrapper each
(`eagle/cuda/detail/RuntimeCompat.h`), the device clock rates are read as
device attributes, and the driver-only CUDA plugin provides CUDA 13's
kernel-launch entry points and signatures.

With a usable GPU, `eagle.deploy` (and `eagle.simulate`) returns once the
device side is built; the host side builds on its own thread and the first
host run waits for it, raising its error if it failed. The loop's own small
CuPy kernels compile during the deploy too, beside the device build, so the
first device run compiles nothing. Explicit `targets=` build as before. A
default deploy now keeps its device and host builds in two cache folders, so a
cache filled by an earlier version builds once more. A process exits only
after a host build it started has finished, and a child forked while one runs
builds its own. `ActiveSet.reset` writes the identity map in place on the
device, so no batch-sized temporary is left in CuPy's pool.

The persistent launch sizes its threads per SM from the device's issue rate for
the step's precision (its FP32:FP64 ratio and SM layout), the batch size and the
step budget, instead of a fixed 256: float32 runs and short step budgets fill
the GPU, and a small batch keeps enough samples per lane for the drain.

A host run of an automatic artifact picks its pacing from the artifact:
`Runner.host_policy` defaults to `"auto"`, which is `"tiled"` when the host
entry's fused side is HAWK's tiled loop (its shared object exports
`<entry>_tile`, read by `eagle._host_loop.host_tile`) and `"measured"`
otherwise. `"tiled"` launches `steps_max` steps from the first launch, the last
cut to `max_steps`, with no one-step sweep phase and no probe; `"measured"` and
`"band"` are unchanged and can still be set. `HostTeam` hands out its tiles
with `schedule(dynamic)`, so one slow tile no longer holds the team's barrier
while the other threads idle; tiles and their triples are unchanged, so
results are too.

The public surface is trimmed before the first release. `eagle.__all__` drops
the names nothing outside eagle's own tests used: `from_dlpack` (use
`eagle.to_cupy`), `LaunchPlan`, `launch_plan`, `LaunchMixin`, `pure_origin` and
`DEFAULT_BLOCK` (still in `eagle.launch`), and `MemberHandle`, `RecordedSchedule`,
`NonToggleableMemberError` and `UnknownMemberError` (still importable from
`eagle.pipeline`). The empty `eagle.cpu` placeholder is gone; `eagle.cuda` now also
carries `CaptureFork`, `capture_snapshot_nodes` and `is_node_toggleable`. The
cache counters and resets used by eagle's tests are private (`_launch_plan_stats`,
`_coercion_stats`, ...), as are `plan()`'s test seams (`_gather`, `_exec_access`,
`_exec_op`). Removed with no callers: `eagle.exec.abi_version_of` (use
`eagle.abi.abi_version_of`), `HostTeam.tile_size`, `GraphPipeline.is_member_enabled`
and the `eagle.gemm.plan` dtype-name cache hooks. `eagle.dtypes`,
`eagle.roles.RECOGNIZED_PATTERNS`/`MANIFEST_FORMATS` and `eagle.launch.KERNEL_NAME`
are raptor's own objects, re-exported. The role groups live in `eagle.roles`
(`OUTPUT_ROLES`, `INPUT_ROLES`, `PER_SAMPLE_ROLES`, `STATE_ROLES`), and
`eagle.plan.residency(planes)` / `eagle.plan.device_resident(value)` are the one
residency rule `AutoPlan` and `eagle.simulate` share.

`eagle.until_done` takes a list of plans as ONE step that launches each in order
over one namespace of planes; `eagle.simulate` runs a several-kernel model
through it, so `Simulation.runner` is always the `Runner`. A compacting loop checks
`max_steps` once per `every` steps (documented; `RunReport.steps` counts every
step that ran). The plugin schema page moved to raptor's docs; eagle links it.

One sample runs through the same kernel as a batch. A value whose shape is a
plane's per-sample head (a Python number or 0-d array for a scalar plane,
`(w,)` for a vector, `(R, C)` or `(R*C,)` for a matrix) is one sample:
`Plan.run`, `Plan.bind`, `eagle.until_done` and `eagle.run_until_done` run it
as a batch of one through its zero-copy view, and `Plan.run` returns every
output in its head shape (a `numpy.float64` for a scalar, cupy arrays for
device inputs; a supplied head-shaped output is written in place and returned
itself). A shape that is also a batch stays a batch (`(1,)`, `(w, 1)`,
`(w, w)`), and batch calls are unchanged. Refused, naming the argument: a call
mixing one sample with a batch, a Python number under an output name, and a
number for a plane the bind doors would have to allocate (one sample is bound
as 0-d/`(w,)` arrays, written in place; the `terminated` mask is a 0-d bool
array or omitted). `eagle._layout.classify` answers `"single"` for one sample.

`eagle.plan.auto(plugin, **plan_kw)` returns an `AutoPlan` that runs where the
data lives: `.run`/`.bind` select the `HostTeam` plan (`.host`) when every
per-sample value is host-resident and the `DeviceKernel` plan (`.device`)
when any is device-resident, never by sample count. Data on both sides, and a
side the plugin carries no entry for, are refused naming the fix.
`eagle.until_done` and `eagle.run_until_done` accept an `AutoPlan`.

`eagle.deploy` (the same function as `eagle.plan.auto`, bound on first use so
`import eagle` still imports neither `eagle.plan` nor hawk) takes hawk kernels
directly. `deploy(kernel)` returns its `AutoPlan`; `deploy([k1, k2, k3])` builds the kernels into ONE bundle and
returns a tuple of plans in the same order. The build goes through hawk's
build and compile cache under `cache_dir` (default: hawk's cache), so a
repeat call, in the same process or a new one, compiles nothing; `targets`
defaults to `("host", "cuda")` when a CUDA device and a device compiler are
usable, else `("host",)`. Derived kernels (`jvp`/`vjp` built with their
primal, `steps=K`, a default automatic kernel and its `.step`) are inputs
like any other. hawk is imported only inside the call, so eagle imports without
it. Refused, naming the fix: an empty list, a list mixing a kernel and a
plugin, and `targets=`/`cache_dir=` on a built plugin. A built plugin is
taken as before. Tutorial 4 builds its three plans with one `eagle.deploy`
call.

`eagle.simulate(model, *, state, params=None, until=None, max_steps)` runs a
model on every sample until each one finishes, in the model's own words: the
`state` it updates (its `Mutable` planes: the caller's arrays, written in place
and returned; a `Mutable` left out is allocated zero-filled unless a kernel
reads its `.prior`), the `params` it reads (a `Param` takes one value shared by
every sample, a `Scalar`/`Vector` plane one value per sample, as the kernel
declares; the other kind is refused naming the fix, never broadcast) and a
required step cap. `model` is a hawk kernel, a list of hawk kernels (one step
in list order over one namespace of planes) or plans from `eagle.deploy`;
`until=` is the finishing kernel, placed last. One kernel runs through
`eagle.until_done` unchanged; several run as tutorial 4's composition
(`compaction_body` and `repeat_while`, one CUDA graph on the device), one step
per launch of each. The data decides where it runs (numpy and numbers on the
CPU, cupy on the GPU), and plain numbers are one trajectory, returned
head-shaped. The `SimResult` carries the state (as attributes and items),
`finished` per sample, `done`, `steps`, `status` (`"finished"` or
`"max_steps"`: the cap is a result, not an exception) and the `RunReport`;
`eagle.simulation(...)` is the prepared form (`.run()`, `.reset()`). Refused,
each in the caller's words: a name in both dicts or in the wrong one, an array
for a `Param` or a number for a per-sample plane in a batch, the names the door
allocates, a model with no finishing kernel, one name declared two ways across
kernels, kernels under different guards, a string, number or Python function
as `until=`, a state plane named like a result field, and a `.prior`-read
plane without an initial value. Recording along the way, fixed-horizon runs
without a finishing kernel and fusing a list of kernels into one launch are
not part of this release. The Python quickstart opens with it, and tutorial 4
opens with the one call and keeps the explicit loop as what it builds.

Breaking: a Mutable is a plain variable in a kernel body; `x.prior` and
`x[...] = e` are replaced by reading and assigning `x` (the old spellings
raise with the rewrite).
