# Changelog

## 0.6.2 (2026-10-18)

**Graphs run on the GPU their planes live on.** On a machine with several GPUs, a graph now builds and launches on its planes' device from any thread, and a JAX call between eagle launches no longer moves it (before, a launch from another thread could fail with `cudaErrorInvalidResourceHandle`, and a JAX node could hit a capture error).

**JAX is certified.** Five rows (`JX-IN-CUDA-ALIAS`, `JX-IN-WRITE-REFUSED`, `JX-OUT-CUDA-ALIAS`, `JX-IN-CPU-COPY`, `STREAM-JAX-PRODUCER-ORDER`) cover JAX arrays: zero-copy in for read-only roles, zero-copy out (the JAX array is eagle's own result buffer, never written again), and a device-side wait for a JAX operation still running. JAX arrays are immutable, so a JAX array passed as `out=`, as `terminated` (unless the plugin declares it read-only), as a `Mutable` plane, or as a written host buffer (`host_ptr(..., writable=True)`) is refused by name instead of copied. A CPU result for a CPU JAX input is now placed on the CPU device. JAX publishes wheels for Python 3.12 and newer, and no 3.13 free-threaded build; install it with the new `jax` extra on those, plus JAX's own `jax[cuda12]` or `jax[cuda13]` for GPU arrays.

**JAX FFI rows.** Four rows certify `eagle.frameworks.jax`: `JX-FFI-CPU-ALIAS`, `JX-OUT-XLA-OWNED`, `JX-JIT-INVARIANCE` and `STREAM-JAX-FFI-ORDER`, each on the CPU and CUDA legs it declares. The completeness gate now counts a row only if its test ran and passed on every leg (a deselected or skipped row is red), which needs `raptor-core>=0.3.2`.

**hawk kernels are JAX functions.** `eagle.frameworks.jax` runs a hawk kernel under `jit`, `vmap`, `grad`, `jacfwd` and `jacrev` to any order, on the CPU and the GPU, with outputs XLA owns, `donate_argnums` honoured, and the kernel on XLA's own stream (no host wait). Carried-state planes alias input to output, `wrt=()` runs forward-only, a kernel that finishes its own samples needs no `steps=1`, and refusals name your kernel. It needs `jax>=0.11.2` (the `jax` extra also asks for `raptor-hawk>=0.5.1`) and, as for the certified rows, Python 3.12 or newer. Refusals say the way out: mixed float32 and float64 planes (cast to one), forward mode of a kernel that carries state (use reverse mode), and `function()` on an `active_set` kernel (run it with eagle). A new small library, `libeagle_xla.so`, carries the call.

**A carried-state kernel under `vmap` no longer reads zeros.** XLA can hand a batched call a result buffer other than its operand; the handler now copies the operand into the result itself, so the kernel starts from the right state and finishes. The copy is the new seam entry `eagle_backend_memcpy_device` (minor 4).

**`eagle.frameworks.jax.node(fn, reads=, writes=, name=)`.** Runs a JAX step between eagle launches, ordered on the stream, and keeps its results until their copy has run; carried state differentiates through the step's vjp. On the CPU it no longer needs cupy: cupy is imported only for device arrays. On a GPU a JAX node runs between launches, not inside `until_done`'s graph (XLA owns its memory): drive the loop yourself, or put the batch inside `jit`.

**The backend seam is at minor 4.** Minor 2 added `eagle_backend_memset_device` (zero an XLA output without a sync, capture-legal), minor 3 `eagle_backend_capture_add_child` (insert a framework's CUDA graph as one child node) and minor 4 `eagle_backend_memcpy_device`. Each is bound by symbol, so the presence of a symbol is the capability.

**The torch bridge covers more.** One generic launch op gives derivatives to any order, `torch.func`, float32, and `Param` values read from a device plane, loop bodies included; a carried-state kernel steps in place and differentiates through the step vjp; and the op works inside `torch.compile` (first-order reverse mode there; run forward mode and double backward outside it). `torch.compile` needs torch 2.4 or newer (the floor stays 2.2: on an older torch run the step eagerly). A kernel that finishes its own samples returns a fresh `Terminated` mask and leaves yours untouched: keep the returned mask as the next step's.

**Warp programs call hawk kernels.** A hawk kernel runs inside a Warp program: stream-ordered, capturable in `wp.capture_begin` (call `fn.prepare(mode, device)` first, since a build cannot happen inside a capture), writing into Warp arrays given as `out=`. State kernels run in a loop and differentiate through `wp.Tape` (reverse mode only; use the JAX bridge for forward mode). A Warp program that moves its tape fails by name.

**CuPy and NumPy.** `Plan.run` follows the current CuPy or torch stream. An `out=` that is non-contiguous, of the wrong dtype or on the wrong device is now refused by name instead of written back through a copy: pass a contiguous array of the right dtype and device. NumPy calls still block, since host arrays have no stream.

**Your own code in a graph, by name.** `opaque()` and `child()`, and `Graph.add` and `add_node`, take `name=`; a recorded step refuses two nodes with one label. A node declares its tier (captured, stream-ordered or host), and a child graph carrying a node type a loop body forbids is refused by name. `until_done` accepts captured per-step nodes in its step list, and refuses cadence, reorder, tier and record shapes it cannot honour by name. A stream or host node cannot run inside a stepping loop's device graph: capture it, or call `runner.run()` with `max_steps=k` and run your code between.

**Gradients through a recorded run that runs your model.** `reverse(record, seeds, vjps=)` differentiates a recorded stepping loop whose step runs your model before the kernel, through your vjp; the result matches pure torch autograd. A recorded run also accepts stream-ordered and host nodes. In a recorded loop write the step as `[node, plan]`, give a node per-sample planes (not a `Param`) and no plane another node writes, and use `events="step"`; these lift in 0.6.3.

**The shipped CUDA plugin covers every architecture.** `libeagle_cuda` is built for all 15 architectures from sm_60, plus `compute_100` PTX for newer GPUs, for about 130 KB more; the build-info string and the ship-build check agree on that list.

**Performance cards.** Cards kept in git hold the summaries, fits and metadata; the raw rows are release assets. CI downloads the full rows and fails on a missing or altered one. Slim cards keep the top SM clock.

**Interop guides, tutorials and demos.** Guides for JAX, torch, CuPy/NumPy and Warp, tutorials 12 to 18 (torch net in a stepping loop, a Warp kernel inside a graph, a JAX model between launches, the eagle batch inside `jit`), and the early-termination and zero-copy interop demos with notebooks and cards, which moved here from raptor-core. The matching certification rows are in raptor-core 0.3.2.

**Build warnings fixed.** The build-info architectures string now lists SASS and PTX in a short form (`61,70,90/90`) so the full list fits its buffer, with a compile check (before, it was cut at 127 characters); the build links libgomp; test markers are registered; Doxygen builds under strict warnings; the Pages actions use the Node 24 releases.

**Kaggle GPU workflow.** A Kaggle T4 workflow runs the GPU interop rows and the notebooks, from a copy of the suite outside the checkout, and reports results even when the kernel ends in error.

**Efficiency table.** The table quotes `eagle.simulate`, the same arm as the landing site.

**NumPy and torch floors.** `numpy>=1.26` and `torch>=2.2` are declared, with a floor lock and a test that keeps the two in step. `raptor-core>=0.3.2` carries the JAX rows.

**docs-fresh tells its failures apart.** `tools/docs_fresh.sh` exits with a different code for stale docs, a missing card and a missing tool.

**Profiling kit.** The card kit gains a short `nsys` and `ncu` profiling run: launch and chunk sweeps, comparison of every eagle path with Warp, slot fill and crossover reports, clocks locked and chunk lengths pinned, FP64 and FP32 kernels. It supports aarch64 (toolchain and cards-cpu locks picked by `uname -m`) and CUDA 13 (`--cuda 13`); the CUDA 12 profile runs on 12.9 and `--cuda 12.6` keeps the lock for pre-Turing parts. Band captures are profiled as an eager loop, counters are `uint32`, a test guards the locks, and split card runs load the card scripts of the eagle source they measure. The bootstrap uses a conda frontend from `PATH` only when it answers like one.

**Reproduction locks.** The `pypi-*` locks pin eagle 0.6.2 and hawk 0.5.1.

## 0.6.1 (2026-10-15)

**Torch tensors on the CPU under older NumPy.** With NumPy older than 2.2 (Colab ships 2.1), NumPy imports every torch tensor read-only, so `eagle.simulate` refused CPU torch outputs as not writable. The torch bridge now checks the tensor as before and then works on torch's own NumPy view of the same memory, which is always writable. A test makes NumPy's import read-only and checks the outputs are written.

**Colab and Kaggle setup cell.** The first cell of every docs notebook installs the CUDA 12 or CUDA 13 extras to match the runtime (Colab's GPU runtime is CUDA 13) and says plainly when it runs on the CPU and how to get a GPU. It installs the family's own packages with `--upgrade`, so a runtime that kept an older release gets the current one.

**Notebooks checked the way a Colab user runs them.** `make nbcheck-hosted` (and `nbcheck-floor`, `nbcheck-hosted-gpu`) runs every docs notebook against the built wheels in a pinned copy of Colab's environment, setup cell included.

## 0.6.0 (2026-10-14)

**Register gate by toolkit.** The card register gate holds its register and arithmetic pins per CUDA release (12.6, 12.9 and 13.4 measured), compares each build with the pins of the `nvcc` that compiled it, and skips with a reason when that release has no pins.

**Grouped reorder kernels.** `reorderDevice` moves planes in groups: one stage kernel and one place kernel per group of up to 64 planes that fit the staging area (the widest plane's size per sample by default, so existing scratch stays valid), plus a final words kernel, instead of two kernels per plane. Each kernel's grid is `min(ceil(n / 256), 4 * multiprocessors)` blocks striding over `[0, span)`, and `ReorderPlane` order and results are unchanged byte for byte. Passing `stageBytesPerSample` to `reorderDevice` / `reorderHost` and sizing the scratch with the new `reorderScratchBytes(n, planes, nPlanes, stageBytesPerSample)` overload makes the groups larger; the host face processes the same groups.

**Mapped pinned host planes.** A reorder or restore plane may live in mapped pinned host memory (SoA, at most 16 bytes per element, results identical to device memory); `eagle::filtering::classifyPointer` and `checkPlanesResident` classify a pointer for a device and refuse pageable or other-device planes by plane index (per call only under `EAGLE_DEBUG_MODE`), exposed as the plugin symbol `eagle_backend_pointer_residency` (seam minor 1, additive) and `eagle._core.pointer_residency`, and `ActiveSet.own` on a device set accepts a numpy plane that is mapped pinned memory and names the fix for a pageable one.

**resetDevice.** `eagle::filtering::resetDevice(perm, inv, indexMap, count, span, n, stream)` gives `perm`, `inv` and (when non-null) `indexMap` the identity and sets `*count = *span = n` in one capture-legal kernel, with the host twin `resetHost`; it is bound as `eagle._core.reset_device` and the plugin symbol `eagle_backend_reset_device` (seam minor 0 to 1, additive), and `ActiveSet.reset` on a reordering device set uses it.

**Blackwell in the wheel.** The wheel's CUDA plugin carries SASS for Blackwell (sm_100, sm_103, sm_120, sm_121) beside the older architectures, so B200 and RTX PRO 6000 find native code and need no first-use JIT (sm_120 measured: the RTX PRO 6000 Blackwell Server Edition has full RK4 and RK7(8) cards; sm_100, sm_103 and sm_121 are not yet run on hardware), and PTX at compute_100 beside compute_121 so Jetson Thor under a CUDA 13 driver (sm_110) JITs instead of failing; the publish workflow fails a wheel whose plugin does not list `100-real`, `120-real` and `100-virtual`; the install and build-options pages gain a driver-requirements table.

**Launch sizing and cadence.** The persistent launch puts `ceil(1 + 16 / issue)` warps per scheduler partition on each SM (5 on a full-rate FP64 part, 2 on a 1:32 or 1:64 one), held between two warps and the smaller of the register-limited residency and the drain limit `n * f / (2 * SMs)`, where `f` is the step fill read back from the previous run (0.5 on the first); a residency under the saturation count is taken whole as one block. Between the sure fused and sure persist batches (see Automatic entry pick) the fused and persistent entries are timed against each other: a spread batch is where a persistent launch's refills can cost more than the fused entry's idle lanes, and only the wall says which. A banded run compacts every `2^round(log2 sqrt(2R / h))` steps, `R` being a compaction's cost in steps of a full-live launch (priced by the cost model on a device) and `h` the rate samples finish at; a launch in which nothing finished runs the longest allowed. A register bound is attached to an entry only when its launch asks for more threads per SM than it holds, and kept only when it spills nothing new, fits the edge and grows the code by at most 4 %: the oscillator persist entry on an A100 (36 registers, 640 asked, 1536 held) is no longer recompiled. `BoundPlan.entry_reports()` gained `text_bytes` and `demand`, and `Runner.launch_policy` reports the persist block and grid, chunk, fill, `R` and `k`. A persist entry is launched with a chunk length of the power of two nearest `sqrt(8 * m * SMs * claim / (ops * issue))` steps held to 2..32 (a claim-cost model scaled by the SM count; `claim` is the cycles one warp's claim holds the chip, a device-class constant: 12 on Pascal, 1 on later parts; no clock term), `m` being the previous run's mean steps, `ops` the step's floating-point operation count from the artifact's sidecar and `issue` the cycles one warp instruction of the precision takes; once the chunk is known the threads per SM gain one warp per partition for a chunk head's fetch latency. A probed runner refits only when the step fill (to 0.01) or the steps change, and drops the persist entry's stored wall when the refit changes the launch, so it is timed again on the new geometry. The internal runner keyword `_chunk` pins the chunk and turns off its refit. The new rules are plain C++ (`eagle/policy/Launch.h`) and bound as `eagle._core.persist_threads_per_sm`, `persist_chunk` (taking the compute capability, SM count, FP64 ratio, precision, `ops` and mean steps), `band_k`, `band_rate` and `bound_demanded`.
`BoundPlan.launch_persist`/`prepare_persist` take `chunk` (default 32), the steps
between the warp-converged persist entry's chunk heads, a power of two from 1 to
2**20, and refuse a block that is not a multiple of 32. `RunReport.lane_utilisation`
is now `RunReport.slot_fill`: the steps run over the entry's chunk slots (a lane
waiting masked for the next head is an empty slot).

**Compaction cadence.** A banded run's compaction cost now carries the graph gap between a compaction's kernels, and the memory traffic of a step is priced against the device's bandwidth (`memoryClockRate * memoryBusWidth * 2 / 8`, from the owned planes' bytes per sample): a step bound by memory hides most of the arithmetic dead lanes would waste, so the run compacts less often, while a compute-bound step is priced as before. `eagle._core.band_rate` takes the optional `mem_clock_khz`, `bus_width_bits` and `bytes_per_sample`; the band and the persist chunk now read one step operation count (`eagle._until_done.ops_per_step`).

**Automatic entry pick.** A batch of at least eight times the threads the persistent entry keeps resident across the device takes the persistent entry, and one of at most a quarter of what one fused wave holds takes the fused entry, from the first run and without timing the other (`PICK_PERSIST_MULTIPLE` = 8, `PICK_FUSED_DIVISOR` = 4): the entry the size rule names is the one expected to win there by a wide margin. In between, the runner times both entries (the first run takes the size pick, the second the other entry, the 2 % hysteresis keeps the pick on a tie). The other entry is timed again only when the readback moves to another regime (`S` to the nearest power of two below, `f` to one decimal), no longer every 64 runs, so a settled run never repeats the slower entry. A runner that times both entries leaves its pick in a process-wide cache keyed on the device, the kernel, the precision, the batch size and the readback regime, and a later runner with the same key takes it untimed; nothing is written to disk. A model of the persistent entry's wall over the fused entry's (`eagle._core.entry_ratio`) is recorded with each readback in the runner's pick state, for study; it does not pick. A runner's pick is one immutable value replaced whole, so reading it from another thread needs no lock. `RunReport.mapped` counts the samples the launches ran over, summed over the launches, so the bytes a run issued can be computed exactly.

**Host dispatch cost.** On the host a launch now costs the slowest of its arithmetic, its dependent chain and its memory traffic: a branch's longest chain of dependent operations is counted per precision, a loop the compiler leaves scalar pays that chain at `HOST_CHAIN_SECONDS` per operation, and a vectorised loop runs its arithmetic `HOST_VECTOR_WIDTH` times faster and pays no chain. A dispatch rendered as a switch keeps the fused loop scalar. Plane writes are counted apart from reads when hawk reports them (otherwise every plane read is taken to be written), and a float32 calibration keeps the step's own vectorised flag. `benchmarks/reproduce/calibrate.py` writes schema 2: the chain lengths per branch, a four-chain unit that separates chain latency from throughput, and the host profile hawk compiled with, so the fit can be refreshed. The fit was refused on every machine tried, so 0.6.0 ships the default host cost constants; no per-machine calibration is shipped.

**Card scripts.** The perf and RK78 cards write schema 3: the bytes a run issued come from its own report (`RunReport.mapped`), a library that is not installed is listed in `arms_not_run` instead of failing the card, and each arm records `tail_floor_ms` and `first_call_ms`. Counter planes are integers, the slot-fill column uses the `slot_fill` name, and the memory baseline is taken before the warm-up.

**Plane dtype check.** `Plan.bind` (and `rebind`, and `Plan.run` for integer and bool planes) now compares every plane's dtype with the one the artifact's sidecar declares in `arg_dtypes`. Before, an int32 array bound to an `Index` plane declared int64 was accepted and the launch read past the plane and crashed. It is now refused at bind time, naming the plane, both dtypes and the fix (`pass where as int64: where.astype(np.int64)`); nothing is cast silently, and the launch path pays nothing.

## 0.5.1

**Graph-handle leak fix.** `Graph::addNode` rejects borrowed temporaries at compile time; wrap `end()`/`Scan::graph()` in `CapturedGraph` (`addNode(CapturedGraph{ capturer.end() })`). The borrowed `cudaGraph_t` overload never destroyed its input, so a temporary handle leaked; it now takes `const cudaGraph_t&` and the rvalue form is deleted. Lvalue borrows and `CapturedGraph` calls are unchanged. `StreamCapturer::end()` and `Scan::graph()` are now `[[nodiscard]]`, and the internal call sites that leaked (`Scanner`, `Slice`, `Scan::launch`) are fixed. The owned `CapturedGraph` overload of `addNode` gained a scalar `idealBlockSize` (default 0) applied to every kernel when the graph carries no per-kernel table.

**Sanitizer gate.** A CI job runs the CPU suite under AddressSanitizer and UndefinedBehaviorSanitizer, and `tests/sanitize/sanitize_gate.sh` runs the CUDA suite under compute-sanitizer (memcheck, racecheck, synccheck, initcheck) and valgrind. Planted canaries must trip every tool, so a gate that stops instrumenting goes red (`tests/sanitize/README.md`).

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
