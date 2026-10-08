# Test map

Reach for this page when you are about to touch a piece of eagle and want to
know what already tests it — "I'm changing how graph capture handles a
dependent node" or "does anything check the CPU plugin registry's
shape-mismatch rejection?" — without reading through both `tests/` and
`python/tests/` end to end. eagle ships two independent, dual-language test
suites that mirror each other feature-for-feature (the C++ core and its
Python bindings), so this page is organized by *capability*, with both
languages' coverage listed side by side wherever a feature has both.

## How the suites are counted, and why that matters here

Unlike a plain `pytest`/`ctest` invocation, the C++ suite's CI gate
(`tests/check_gate.sh`, invoked as `tests/check_gate.sh <mode>
build/tests/eagle_tests` from CI and `make test`) does not trust a raw pass/fail
exit code — it diffs the actual `--gtest_list_tests` output and the shuffled
run's summary line against a committed name manifest
(`tests/expected_tests_cuda.txt` / `tests/expected_tests_cpp.txt`), so a
silently-skipped or silently-renamed test fails the gate even if every test
that *did* run passed. The two manifests currently commit **274** CUDA-mode
tests and **163** CPP_MODE (OpenMP host) tests — CPP_MODE is smaller because
several GPU-only suites (graph capture internals, device error handling)
have no CPU equivalent to run. `tests/check_gate.sh`'s own header explains the reasoning behind this
design; the short version is: a test *runner* that reports
"0 tests, exit 0" on a typo'd filter is not a defect in the runner, it is a
defect in whichever gate trusted a bare exit code, so the gate here pins the
whole named set instead. The Python suite (`python/tests`, run as `pytest
python/tests -q` in CI) does not yet have an equivalent named-set
gate.

Two tests exist purely to check that other tests are testing the right
thing: `python/tests/test_conformance_corpus.py` and its C++ twin
`tests/test_ConformanceCorpus.cpp` both replay the **same** language-neutral
fixture corpus (`tests/conformance/*.json`), so a C++/Python behavioral
divergence shows up as a mechanical test failure rather than something a
human has to notice by comparing two docs pages.

## Graph capture, lifetime, and ownership (C++, CUDA-only)

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_Graph.cu` | `GraphFixture` | Core `eagle::cuda::Graph`: node dependency wiring, host-node execution, launcher move/replay, child-graph recording, `setLogicalSize` retuning across replays, stream capture. |
| `tests/test_GraphLifetime.cu` | `GraphLifetimeFixture` | A Launcher stays valid after the `Graph` that produced it goes out of scope. |
| `tests/test_GraphOwnership.cu` | `GraphOwnershipFixture` | Ownership/ref-counting semantics of Graph/Launcher handles. |
| `tests/test_GraphPerKernelIdeals.cu` | `GraphPerKernelIdealsFixture` | Per-kernel `idealBlockSize` (the launch policy's chosen thread-block size) is preserved across graph replays. |
| `tests/test_CaptureFork.cu` | `CaptureForkFixture` | `CaptureFork`: forking one capture into sibling graph nodes, auto-join on destructor, replay matching a host oracle. |
| `tests/test_NativeNode.cu` | `NativeNodeTest` | Device-side native graph node behavior. |
| `tests/test_GraphComposer.cu` | `GraphComposerFixture` | `eagle::compose::GraphComposer`'s device face: composing multiple Launchers into one captured graph, replay conformance against a host oracle. |
| `tests/test_GraphComposerCpu.cpp` | `GraphComposerCpuTest` | The CPU parity twin of `GraphComposer`, compiled into both build modes. |
| `python/tests/test_pipeline_concurrent.py` | — | `CaptureFork` + `GraphPipeline.add_concurrent` from the Python side. |
| `python/tests/test_pipeline_parity.py` | — | Bound-core `GraphPipeline` behavior — the durable half of the host/device parity gate. |
| `python/tests/test_u2_interim_error_check.py` | — | The interim `cudaGetLastError()` bracket around conditional-graph launches. |

## Launch policy and block sizing (dual-mode)

*Block size* is how many GPU threads execute together in one scheduling unit
(a *thread block*); picking it well affects *occupancy* — how much of the
GPU's parallel capacity a kernel actually keeps busy.

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_ComputeBlocks.cu` | `ComputeBlocksTest` | `eagle::cuda::computeBlocks` policy invariants — warp lower bound, `idealBlockSize` upper bound, warp-multiple rounding. |
| `tests/test_LaunchTraits.cpp` | `LaunchTraitsTest` | `eagle::launch::Traits`/`minBlocksPerSM`, checked against hand-derived expected values. |
| `python/tests/test_launch_policy.py` | — | Gate A — the `_launch_policy.resolve_block` value table. |
| `python/tests/test_launch_policy_gate_b.py` | — | Gate B — a GPU end-to-end value twin against a real captured graph. |
| `python/tests/test_launch_policy_gate_c.py` | — | Gate C — the nonvacuity ritual: proves the gate can actually fail. |

## Device properties (dual-mode)

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_DeviceProps.cpp` | `DevicePropsTest` | The CUDA-free half: `eagle::deriveFields()`'s formulas and the `fmaLanesPerSM`/`fp64Ratio` compute-capability tables (including the fallback for an unlisted compute capability), plain arithmetic over synthetic inputs. |
| `tests/test_DeviceProps.cu` | `DevicePropsCudaTest` | The hardware half: `eagle::cuda::deviceProps()` against the real device — raw-field sanity, derived-field consistency, and a known-answer check for the dev box's GPU. |

## Conditional / branch execution (dual-mode)

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_ConditionalGroup.cu` | `ConditionalGroupTest` | `eagle::conditional::ConditionalGroup` + `CaptureConditional`, including distinguishing "skipped" from "fired but early-returned." |
| `tests/test_ConditionalGroupCpu.cpp` | `ConditionalGroupCpuTest` | The CPU parity twin of the above, compiled into both build modes. |
| `python/tests/test_conditional.py` | — | `eagle.pipeline`'s `Skippable`/`SkipGuard` from Python. |
| `python/tests/test_repeat_while.py` | — | `eagle.pipeline`'s `RepeatWhile`/`repeat_while` (device-side loop) from Python. |
| `python/tests/test_active_set.py` | — | `eagle.ActiveSet` / `compaction_body`: the active-set index map on device (the `filtering` seam group) and host, captured replays, a sequence loop body, and a hawk `Guard(active_set=True)` kernel bit-identical to its map-free twin; with `reorder=0.5`, the same kernel bit-identical to the map-only and map-free runs (spread, uniform, grouped; host and device), the order invariant after a fire, the reorder-count bounds, a broken reorder caught, and the bind cross-check. |
| `python/tests/test_until_done.py` | — | `eagle.until_done` / `run_until_done` and `eagle.sidecar.read_finish` over a hawk kernel that finishes its own samples: the `finish` reader's refusals, the runner's refusals, the one call bit-equal to the explicit `repeat_while` path (host and device), the loop stopping at the longest sample and the same kernel emitted without its epilogue's count running to the cap, pre-set mask entries untouched then run after `reset()`, and the `Guard(active_set=True)` artifact compacting and, with `reorder=0.5`, reordering through the same call with sample order restored. |
| `python/tests/test_until_done_host.py` | — | The host branch of `eagle.until_done` (`eagle._host_loop`): the bound entry launched on the host team directly, bit-equal to the explicit NumPy-stop-rule path and to the serial oracle (`HostTeam.run_serial`) for the RK4 oscillator and, where the checkout carries it, the RK7(8) card's attempt kernel; launches following the longest sample, a non-counting epilogue caught, pre-set samples and `reset()`, compaction and reorder with sample order restored, and the fallback to the loop's host arm for a step it cannot see through. |
| `python/tests/test_until_done_auto.py` | — | `eagle.until_done` over a `steps="auto"` hawk kernel: `read_finish` accepting `steps: "auto"` only with `steps_max`, both reserved lookups bound and `fused_steps=`/`every=` refused, plain-kind and active-set runs bit-equal to the one-step oracle on the host and to the artifact driven by hand with the policy's k sequence on the device, the exact `max_steps` cap against the fixed `steps=16` overshoot, the band (climbs to 64, halves to 8, nothing launched when all are done) with a constant-K policy turning it red, `reorder=0.5` restoring sample order, and identical `launches_by_k`/compactions/steps on host and device. |
| `python/tests/test_auto_kernels.py` | — | `eagle.deploy` / `eagle.plan.auto` (one function) over hawk kernels: `auto(kernel)`'s host run bit-equal to the hand-assembled plugin's plan (a plain, a vector and a `jvp` kernel) and its device run bit-equal to the hand-assembled device plan and within one ULP of the host; `auto([...])` a tuple of plans in order out of ONE bundle; no rebuild on a second call (the publisher's memo in-process, the unit stamp, zero compile misses and unchanged mtimes in a new process); a default automatic kernel, its `.step` and `hawk.steps(kernel, 4)` through `run_until_done` bit-equal to the hand path; eagle importing and planning a plugin with hawk masked; `eagle.deploy is eagle.plan.auto`, bound without importing `eagle.plan` at `import eagle`; the refusals (empty list, kernel/plugin mix, `targets=` on a plugin, a missing target). |
| `python/tests/test_simulate.py` | — | `eagle.simulate` / `eagle.simulation`: one kernel bit-equal to `run_until_done(deploy(kernel))` with equal reports and the runner's own loop (host and device); tutorial 4's three kernels bit-equal to its hand-composed cell (copied in the test) with equal iteration and settled counts and `energy` allocated, and the same call with the `diagnostic` launch dropped turning that comparison red; `until=` equal to the list form; every refusal's exact text; one sample (numbers in, head-shaped arrays out) equal to its row of a pairwise-distinct batch and different from the others; numpy on `HostTeam`, cupy on `DeviceKernel`, a mix refused; the step cap as `status == "max_steps"`; the artifact byte-equal to hawk's own build and the door loading no native code; the quickstart's lead example (at most 12 lines after the kernel) running as written, and tutorial 4's explicit cell kept. |
| `python/tests/test_reorder.py` | — | The physical reorder of `eagle.ActiveSet`: `theta` bounds, the trigger (random fires, grouped does not, the span floor), every owned plane follows `inv`, device equals host to the bit, `restore`/`in_sample_order`/`reset`, refused exports while permuted, the refusals, and a captured compaction + conditional reorder over changing masks. |

## Device error handling (C++, CUDA-only)

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_DeviceErrorNothrow.cu` | `DeviceErrorNothrowTest` | `EAGLE_CHECK_NOTHROW` never throws, even on a genuine CUDA failure — at the macro level and at the Event/Launcher/Stream object level. |
| `tests/helper_stream_nothrow_reset.cu` | — | Isolated-process helper for the one leg that needs a fresh process after CUDA init. |
| `tests/test_DLPackInterop.cpp` | `DLPackInteropHost` | The host arm of the DLPack layer (`plugin/interop.h`) and the plugin bridge: stream codes, CPU export, refusals, `gref_from_buffer`. Both build modes. |
| `tests/test_DLPackStream.cu` | `DLPackStream` | The DLPack stream contract on a device: export fence and re-fence order a slow producer before the consumer; the RED twin without a fence reads the stale value; `0` refused, `-1` unordered, events pooled. |
| `tests/test_DeviceErrorReleaseGate.cu` | `DeviceErrorReleaseGate` | Every host-API call site that used to be guarded by the deleted, release-mode-silent `EAGLE_CHECK` now goes through `EAGLE_CHECK_ALWAYS` — proves release-mode CUDA-status discard is gone, not merely a debug-mode concern. |

## Reductions, scans, and filtering (dual-mode)

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_Reduction.cu` / `.cpp` | `ReductionTest` | Device and CPU/OpenMP multi-level reductions. |
| `tests/test_Scan.cu` / `.cpp` | `LargeExclusive`, `LargeInclusive`, `LargeScatterSlice`, `SmallExclusive`, `SmallInclusive`, `SmallScatterSlice` | Scan and scatter-compaction, device and host, across small/large regimes. |
| `tests/test_HostNativeNode.cpp` | `HostNativeNodeTest` | Host-graph chained scratch reuse across reduce/scan/filter nodes. |
| `tests/test_Compact.cu` / `.cpp` | `CompactSizes`, `CompactGraph` | Active-set compaction over raw buffers (`eagle/filtering/Compact.h`): the ascending index map and count against a reference across the scan's size edges, drop and keep masks, a captured replay that follows the mask; the CUDA build runs the host face too. |
| `tests/test_Reorder.cu` / `.cpp` (+ `ReorderHostCases.h`) | `ReorderHostSizes`, `ReorderHost`, `ReorderDeviceSizes`, `ReorderDevice` | The physical reorder (`eagle/filtering/Reorder.h`) and the compaction's trigger: every plane size follows `inv`, two thinning reorders and a restore, a clear `fire` is a no-op, the trigger on random and grouped live sets, the refusals; the device face byte-identical to the host face and a captured reorder obeying its `fire` word. |

## Plugin registry and host/device parity (dual-mode)

The place where a *plugin* — a compiled kernel handed to eagle at run time,
rather than one eagle compiled itself — gets loaded, bound, and launched, on
both the CUDA and CPU registries.

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_HostPlugin.cpp` | `HostPluginTest` | CPU `PluginRegistry`: dlopen/bind/run, ABI- and shape-mismatch rejection, disabled-plugin no-launch. |
| `tests/test_PluginRegistryDtype.cu` | `PluginRegistryDtypeTest` | The device registry's deliberate float32 rejection, and SoftDouble load-and-launch. |
| `tests/test_PluginRegistryManifest.cu` | `PluginRegistryManifestTest` | Manifest-level duplicate-plugin-id rejection on the CUDA registry. |
| `tests/test_PluginRegistryMerge.cu` | `PluginRegistryMergeTest` | `eagle::cuda::PluginRegistry::add_plugin` called on an already-populated registry (the merge path) and `set_plugin_enabled`. |
| `tests/test_PluginRegistryUniformInt.cu` | `PluginRegistryUniformIntTest` | The device registry's additive int64 uniform path (`bind_uniform_int`), beside the existing float64 `bind_uniform`. |
| `tests/test_HostPluginUniformInt.cpp` | `HostPluginUniformIntTest` | The CPU host registry's twin of the int64 uniform path above — same binding vocabulary and refusal wording as the device registry. |
| `python/tests/test_int_uniform.py` | — | The Python half of the int64 broadcast uniform. |
| `python/tests/test_plugin_registry.py` | — | An independent multi-plugin registry injecting a plugin set into one graph. |
| `python/tests/test_registry_pattern_loader.py` | — | The registered pattern-loader hook (registry inversion). |
| `python/tests/test_cpp_host.py` | — | The precompiled C++/CUDA plugin host loading and running an eagle PTX plugin. |
| `python/tests/test_host_launch_torch.py` | — | A dlopen'd CPU plugin driven over torch CPU tensors. |
| `python/tests/test_graph_plugin.py` | — | A deployed (PTX) launchable as a capturable CUDA-graph node. |

## `wide_in` / `wide_out` / `accum_out` roles (Python, additive schema-v1)

The three arg-spec roles added after the schema-v1 freeze (see
{doc}`plugin_schema`'s role-vocabulary table) — a named "wide" buffer input/output and
the cross-sample accumulate plane, all three bound as a plain by-value handle.

| Test file | Capability |
|---|---|
| `python/tests/test_wide_arg_classification.py` | `wide_in`/`wide_out` role classification and marshalling (`eagle.roles.classify_arg`), cupy-free. |
| `python/tests/test_wide_capturable_launch.py` | The wide roles on the capturable launch path — `eagle.launch._launch_pure`'s `wide_inputs`/`wide_outputs`/`wide_out_exempt` name tuples and the `launch()` dispatcher forwarding them. |
| `python/tests/test_wide_device_coercions.py` | The device half: `eagle.marshal.coerce_wide_inputs`/`coerce_wide_outputs` and `eagle.launch.pure_prepare` threading them into `assemble_args`. |

## The v2 execution axis: structures, placement, partitioning (dual-mode)

Schema-v2's `exec_targets`/`exec_access`/`exec_op` manifest keys and the Python
`eagle.exec`/`eagle.plan` API that drives them (see {doc}`plugin_schema`'s
execution-axis section) — single-device, host-team, and rank-partitioned runs of the
same body, and the placement/layout self-checks that refuse a mismatch at plan time.

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_ExecContract.cpp` | `ExecContractTest` | The host/CUDA-free execution-contract rows: manifest parsing of the v2 keys, placement legality by `exec_access`, the reduction-operator requirement/prohibition for `mapreduce`. |
| `tests/test_ExecContractDevice.cu` | `ExecContractDeviceTest` | The device rows: the PTX layout self-check, the device launch triple, partition identity under `DeviceKernel`, the mapreduce tolerance band, and the host/device twin comparison. |
| `python/tests/test_schema_v2_load_path.py` | — | RED-first coverage of eagle's manifest load path under schema v2 — the new execution axis through the loader, as opposed to `schema_version` alone. |
| `python/tests/test_exec_contract_rows.py` | — | eagle's four execution-structure interop-certification rows: partition identity, host/device twin, illegal-placement refusal, layout-selfcheck refusal. |
| `python/tests/test_rank_partition.py` | — | The single-process face of `eagle.exec.RankPartition`; the two-rank distributed bed lives in `tests/mpi/` (see {doc}`plugin_schema`'s MPI section) and runs through its own gate. |

## Schema, sidecar, and manifest validation (dual-mode, GPU-free)

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_SchemaHardening.cpp` | `SchemaHardeningTest` | Forward-strict field pinning and unknown-key tolerance across gate/structural/discriminant classes. |
| `tests/test_Sidecar.cpp` | `SidecarTest` | The optional `derivative` sidecar block (VJP/JVP metadata), recompute-only rejection of a populated `residuals`. |
| `tests/test_ConformanceCorpus.cpp` | `ConformanceCorpusTest` | The shared C++/Python conformance corpus driver (see above). |
| `python/tests/test_schema_hardening.py` | — | The Python-side twin of `test_SchemaHardening.cpp`. |
| `python/tests/test_derivative_sidecar.py` | — | The Python-side twin of `test_Sidecar.cpp`. |
| `python/tests/test_conformance_corpus.py` | — | The shared corpus driver, Python side. |
| `python/tests/test_roles_vocab.py` | — | The `arg_spec` role vocabulary is single-sourced; `SCHEMA_VERSION` cross-check. |
| `python/tests/test_roles_raptor_repoint.py` | — | eagle's schema constants re-point at raptor's declared values. |
| `python/tests/test_reserved_words.py` | — | The reserved-word absence test on eagle's own vocabulary. |

## ABI and interop (Python)

| Test file | Capability |
|---|---|
| `python/tests/test_abi_version.py` | The aether-ABI version stamp stays in sync between eagle and the C++ host. |
| `python/tests/test_gref_layout.py` | Compile-time gate: eagle's `GREF_DTYPE` matches aether's `GRef` and the host mirror. |
| `python/tests/test_handle_layout.py` | Compile-time gate: eagle's `HANDLE_DTYPE` matches aether's `HandleT` and the host mirror. |
| `python/tests/test_interop.py` | Framework-free DLPack detection and origin-selection logic. |
| `python/tests/test_interop_buffer.py` | The generic buffer layer: zero-copy import of numpy, torch, cupy and Warp producers, access/owner/producer reporting, re-export, named refusals and the stream contract (fence, RED twin, re-fence). |
| `python/tests/test_interop_matrix.py` | eagle's own 8 declared rows of the cross-repo interop certification matrix. |
| `python/tests/test_interop_matrix_conformance.py` | Completeness gate and no-skip self-test over the matrix above. |
| `python/tests/test_interop_warp.py` | The NVIDIA Warp interop certification rows (`WP-IN-CUDA-ALIAS`, `WP-OUT-CUDA-ALIAS`, `STREAM-WARP-PRODUCER-ORDER`) — kept separate from `test_interop_matrix.py` rather than appended to it. |
| `python/tests/test_arg_classify.py` | The shared role-to-ABI-shape classifier. |

## Python backend seam (`eagle._core` / CUDA plugin boundary)

The `eagle-backend/1` seam between the always-importable g++ core (`eagle._core`) and
the optional CUDA plugin (`libeagle_cuda.so`) — a missing, mismatched or partial
backend is a typed refusal, never a crash.

| Test file | Capability |
|---|---|
| `python/tests/test_backend_unavailable.py` | Every row runs in a fresh subprocess pointed at a plugin through `$EAGLE_BACKEND_CUDA`; a missing, mismatched or partial backend raises `eagle.BackendUnavailable`, never crashes. |
| `python/tests/test_backend_wall.py` | Reads the built artifacts directly: `libeagle_cuda.so` exports exactly the committed seam manifest and nothing else; `eagle._core` neither needs nor references the CUDA runtime or driver; the plugin links nothing outside glibc/OpenMP. |
| `python/tests/test_core_needs_no_driver.py` | The compiled extensions load on a machine with no NVIDIA driver present — only device execution needs one. |
| `python/tests/test_backend_driver_only.py` | The CUDA plugin reaches CUDA through the driver library alone — no undefined CUDA symbol, and no CUDA runtime linked in. |
| `python/tests/test_backend_seam_rows.py` | Behaviour that crosses the seam on a real device: a Python callback the backend calls back into surfaces as its own exception; a handle consumed by another call refuses reuse. |

## PyTorch bridge (Python)

| Test file | Capability |
|---|---|
| `python/tests/test_frameworks_torch.py` | `eagle.frameworks.torch`: a compiled kernel wrapped as a `torch.autograd.Function`, exercised over every role the bridge serves (vector plane, `Param`, `Terminated` mask, vector output). |

## Golden conformance and the neural-block descriptor

| Test file | Capability |
|---|---|
| `python/tests/test_raptor_golden_conformance.py` | eagle's validator accepts raptor's own golden fixtures. |
| `python/tests/test_neural_block_one_implementation.py` | The `neural_block` manifest clause exists in exactly one implementation across the family. |
| `python/tests/test_neural_block_road_equality.py` | Road equality between raptor's entry point and eagle's local one for the same clause. |

## Host dispatch internals (C++, CPU)

| Test file | Suite(s) | Capability |
|---|---|---|
| `tests/test_HostDispatchContract.cpp` | `HostDispatchContractTest` | `eagle::cpu::Host::launch`'s region-required work-sharing contract: it never opens its own `#pragma omp parallel` region, only shares an existing one's threads. |
| `tests/test_HostPacketArm.cpp` | `HostPacketArmTest` | `eagle::cpu::Host`'s SIMD packet arm (`packetLoad`/`packetStore`/`loadMask`/`applyTail`) across scalar types, including width-1 (non-emulated) `DataT`. |

## Docs and utility

| Test file | Capability |
|---|---|
| `python/tests/test_doc_includes.py` | Every doc-included code snippet actually compiles and matches its source. |
| `python/tests/test_perf_card_table.py` | Re-renders every committed `benchmarks/perf_card/card_<device>.json` with the card script's own table generator and diffs it against `docs/content/performance.md`'s committed table. |
| `tests/test_Utils.cu` | `ObsTest`/`SliceTest` — the Observer and reference-holder utility types. |
| `tests/test_ObserverRegistration.cpp` | `ObserverRegistrationTest` — registration-consistency for `eagle::util::Observer` move-assignment and re-observe: the observable's list and the observer's own back-reference must never disagree. |

## Assessment binaries (not gated, informative only)

| File | What it reports |
|---|---|
| `tests/bench_host_graph.cpp` | Host-graph executor overhead vs. direct calls, arena flatness, dlopen'd-plugin vs. native OpenMP scaling. Prints a table; not part of `check_gate.sh`'s pinned count. |
| `tests/probe_LaunchTraitsCpuOnly.cpp` | A compile probe (no `main()`) proving `Traits.h` is genuinely CUDA-free; fails the build on a compile error rather than reporting as a test. |
