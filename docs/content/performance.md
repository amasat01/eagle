# Performance

RAPTOR lets you focus on the application while still making efficient use of
GPUs. This page measures that on two workloads, a fixed-step one and an
adaptive one, against other common ways of writing the same computation, on
both a GPU and a CPU, and publishes every number the run produced, including
the regimes where another approach is faster.

## Why this workload

A large class of scientific code integrates many independent problems at
once: one trajectory, one particle or one parameter sample per GPU thread,
each with its own state, its own control flow and its own stopping point.
Written for the GPU, this class tends to lose efficiency in four places:

- **launch overhead**, when every step is a separate kernel launch from the
  host;
- **host round trips**, when the host has to read a device value every step
  to decide whether to continue;
- **finished samples**, when samples that have stopped keep occupying the
  batch and keep being computed;
- **idle lanes**, when a launch is too small to fill the device.

The benchmark is built to exercise all four. It integrates a batch of damped
harmonic oscillators, each with its own frequency, damping ratio and initial
state, using a fixed-step classical Runge-Kutta (RK4) scheme. Each sample
stops after its **own** number of steps. In the *spread* configurations those
step counts are drawn log-uniformly over two decades, so most of the batch
finishes early; in the *uniform* configuration every sample runs the full
count, a dense, all-active batch. The batch size N runs from 10³ to 10⁶.

The step is written once, as a hawk kernel, one sample at a time:

```{literalinclude} ../../benchmarks/perf_card/perf_card.py
:language: python
:start-after: "# >>> code:hawk_step"
:end-before: "# <<< code:hawk_step"
:dedent: 4
```

`terminated: Terminated` is the per-sample stop mask: a finished sample keeps
its state. `x`, `v` and `k` are read as plain variables holding last launch's
value and reassigned to the next one, which is what updates the state in
place, launch after launch.

## The ways of running it

All arms do the same arithmetic in the same precision (float64) on the same
samples, on the same device; the card's arm table lists each one's loop
structure, and the card records every arm's code.

**hawk + eagle.** The kernel is built with {func}`eagle.deploy` and run until every
sample has finished with {func}`eagle.until_done`: the step, which finishes
its own samples, is captured once as the body of {func}`eagle.repeat_while`,
one CUDA graph whose WHILE node repeats it until the finished count reaches
N. One graph launch runs the whole integration; the host does not read
anything until it is done.

```{literalinclude} ../../benchmarks/perf_card/perf_card.py
:language: python
:start-after: "# >>> code:eagle_graph"
:end-before: "# <<< code:eagle_graph"
:dedent: 8
```

The plain `@hawk.kernel` decorator builds the automatic kernel. By default
{func}`eagle.until_done` auto-routes it past this device-loop/"band" shape
entirely: at or below the device's own latency-regime capacity it takes one
fused launch of the whole run, above it the persistent (work-stealing)
launch -- both no-graph, no-map, one host-device sync total. The band shown
above (each launch running as many steps per sample as the policy picks, 8
to 64, with the state in registers) is still reachable, and is what this
row forces explicitly so its loop-structure description, byte model and
kernel count stay meaningful; it is also what any artifact falls back to
when a reorder is requested, or when the fast entries are unavailable. The
other hawk + eagle arms vary one thing at a time:

- **compaction**: the single step traced under a kind whose guard reads an
  active-set index map (hawk `Guard(active_set=True)`); every 16 steps, when
  a sample finished since the last time, an {class}`eagle.ActiveSet`
  recomputes the map on the device, inside the same graph, so the launch
  covers only the samples still running;
- **compaction + reorder**: the same loop, plus an occasional physical
  reorder of every per-sample plane once the live samples spread thin over
  their warps (`eagle.ActiveSet(mask, reorder=theta)`), restored to sample
  order at the end;
- **K fused steps per launch**: `@hawk.kernel(steps=16)`, with and without
  compaction (every 64 steps);
- **eagle auto (eagle picks the launch mode)**:
  `@hawk.kernel(steps="auto")` on the active-set kind -- this row is the one
  left on `eagle.until_done`'s own default pick, so it takes the fast,
  no-graph launch described above whenever capacity allows and the band
  (compacting after every launch in which a sample finished) only above it;
  the card records which one each run actually took;
- **hawk + eagle persistent**: the plain automatic kernel forced through
  hawk's persist entry -- one launch, no graph, no map, no policy kernel, a
  grid sized from the SM count in which each lane takes the next unfinished
  sample off a counter whenever its own finishes;
- **eagle.simulate**: the same automatic kernel run by {func}`eagle.simulate`
  (its bound form, `eagle.simulation`, so the repetitions reuse one build),
  given its state and parameters by name -- it auto-routes the same way;
- **eager launch loop**: the single step (`.step`, one step per launch)
  launched from a Python loop with a host read of the finished count after
  every step: the most direct loop, every intermediate state one host read
  away.

```{literalinclude} ../../benchmarks/perf_card/perf_card.py
:language: python
:start-after: "# >>> code:eagle_simulate"
:end-before: "# <<< code:eagle_simulate"
:dedent: 8
```

**CuPy and PyTorch, masked.** The same arithmetic as array expressions over
the whole batch, with `where` keeping finished samples unchanged and a host
check that some sample is still running after every step; PyTorch also runs
with a block of 16 steps captured as a CUDA graph and replayed. This is how
the computation is commonly written in an array framework: no kernel to
write, and elementwise kernels that stream memory efficiently. Finished
samples are still computed and then discarded.

**JAX and Warp.** JAX compiles a per-sample `lax.while_loop`, batched with
`vmap`, into one executable; Warp runs an explicit per-thread kernel, one
thread per sample, each stopping at its own step.

**CPU OpenMP.** The same hawk kernel: its planes on the host select the host
side of the same `eagle.deploy`, run by hawk + eagle's OpenMP host team on at most
8 threads. It needs no device and no transfers, and it is the correctness
reference the other arms are checked against.

## Method

- **One block per arm.** For each configuration and N, every arm is timed in
  its own block, one arm after the other, with no interleaving. A block is two
  warm-up runs (excluded; the first also verifies the result), then untimed
  back-to-back runs of that arm until 0.5 s has elapsed (at least one run), so
  the clocks are settled, then 5 timed runs. Every reported time is the mean
  of the middle 3 of the 5 runs (the single highest and the single lowest
  dropped), shown with the range of those 3; the median, the interquartile
  range and every run are kept in the card's JSON.
- **Cool-down gate and clock record.** Before each arm's block the card waits
  until the GPU temperature is within 3 °C of the idle temperature it read at
  start and the GPU reads idle, or for at most 20 s (the wait, the temperature
  and whether it timed out are recorded per block). Right after each timed run
  it reads the SM clock and the clock-throttle reasons through NVML, outside
  the timed region; the tables show the range of the SM clock per arm, and the
  FP32 table adds the ratio in SM clock cycles (wall time times the clock read
  after the run) next to the wall ratio. Wall time stays the headline number.
  Without NVML the gate is skipped and no clock is recorded.
- **Agreement before timing.** The warm-up run is also the verification run:
  every arm must reach exactly the expected step count for every sample, and
  its final state must match the CPU arm within the absolute tolerance
  recorded in the card. The largest difference each arm showed is recorded
  per row.
- **Times.** *Wall* is the integration alone, from inputs resident on the
  device to the device synchronised. *End-to-end* adds uploading the inputs
  from host memory and downloading the results. *Kernel-only* is the sum of
  the durations of the GPU kernels one run executes, measured with Nsight
  Systems in a separate traced pass (for the graph arm it includes hawk + eagle's
  loop-control kernels); tracing makes kernels run slightly longer, so at the
  largest N it can exceed the untraced wall time. For the CPU arm it is the
  time spent inside the host-team launches.
- **Counts.** FLOPs and bytes are counted analytically from the kernel's
  operations, loads and stores; the counting rules are stored in the card.
  *Useful FLOP/s* counts only the steps of samples that are still running,
  the same yardstick for every arm. *Issued B/s* counts the bytes each arm's
  own kernels move, including the passes over finished samples. Both are
  also given as a fraction of the device's peak, computed from its reported
  properties as described in the card. This device does not expose DRAM
  byte counters to the profiler, so bytes are analytic only.
- **Device state.** The card records the device's clock, temperature and
  utilisation, sampled during the repetitions.
- **Compaction cadence sweep.** `eagle_graph` checks its stop guard every
  step; `eagle_graph_compact` checks it once every 16 steps (the cadence its
  compaction runs on) and reads an active-set map built from it. So the main
  table's two graph rows differ in both the cadence and the map, not one
  variable. The sweep at the end of the results below holds the cadence fixed
  at a few values (K = 8, 16, 32) and times two arms at each: the same plain
  step kernel as `eagle_graph`, looped K steps between guard checks with no
  map, against `eagle_graph_compact` at that K. The only structural
  difference within a pair is the active-set map.

## Results

The RK4 oscillator workload runs on both a GPU and a CPU; each device gets
its own card, generated by its own script (`perf_card.py` / `cpu_card.py`),
so the arm lists differ (the GPU card also includes a CPU OpenMP row as its
correctness reference, at 8 threads only; the CPU card is the fuller
CPU-side comparison, with 1- and 8-thread hawk + eagle rows plus Numba, JAX,
multiprocessing and plain Python).

### GPU (Quadro P2000, Tesla T4)

The GPU card is measured on two devices: the Quadro P2000, the reference card for every ratio quoted on this page, and a Tesla T4 (Kaggle, power-capped at 70 W, CUDA 12.6). On the T4 we observed significant SM clock fluctuations between and within arm blocks, so its cells move more from run to run than the P2000's; the card records the SM clock of every timed run.

```{include} _generated/perf_card_gpu.md
```

### CPU (Intel Xeon W-2125)

```{include} _generated/perf_card_cpu.md
```

## Reading the results

<!-- stale-prose-gate:
benchmarks/perf_card/card_quadro-p2000.json: d37837fb31567858848e5f1830f19431
benchmarks/perf_card/cpu_card_intel-xeon-w-2125.json: 5bb591471e7758948a1a2f4994dc575b
-->

Every statement below can be checked against the tables and the
"fastest arm" lines above (GPU card unless marked CPU card).

- **Small batches (N = 10³).** Among the GPU arms the automatic-policy arm
  is fastest in all three configurations (125 µs spread-100, 879 µs
  spread-1000, 893 µs uniform), with `eagle.simulate` next (130 µs, 884 µs
  and 896 µs). `warp_kernel` follows on spread-100 (199 µs, with `hawk +
  eagle persistent` level with it at 203 µs) and on spread-1000 (1.38 ms); on
  uniform the plain `eagle_graph` (1.32 ms) and the K-steps + compaction arm
  (1.38 ms) come next, then `warp_kernel` at 1.49 ms. The CPU OpenMP row,
  which needs no GPU, is faster than every GPU arm on spread-1000 (437 µs
  against `eagle.simulate`'s 884 µs) and on uniform (419 µs against 893 µs),
  as expected on an FP64-weak card; on spread-100 the automatic-policy arm
  leads it (125 µs against 198 µs). At the
  fixed-overhead cell (N = 64, S = 100) `eagle.simulate` takes 117 µs and
  `warp_kernel` 135 µs.
- **Mid-size batches (N = 10⁴-10⁵).** At N = 10⁴ the automatic-policy arm
  leads spread-100 (340 µs, with `eagle.simulate`'s build of the same arm 12 %
  behind at 381 µs, against Warp's 719 µs); the automatic-policy arm also
  leads the GPU arms on spread-1000 (2.54 ms, `eagle.simulate` 2.54 ms,
  against Warp's 6.64 ms), with the CPU OpenMP row ahead of it at 1.78 ms. On uniform,
  `eagle.simulate` (7.22 ms) and `warp_kernel` (7.22 ms) are level; the
  CPU OpenMP row (2.2 ms) is faster than every GPU arm there. At
  N = 10⁵ the automatic-policy arm leads spread-100 (2.02 ms, `eagle.simulate`
  2.02 ms); `eagle.simulate` leads the GPU arms on spread-1000 (16.8 ms, against Warp's
  64 ms; the CPU OpenMP row is faster at 15 ms) and uniform (70.8 ms, against Warp's 71.6 ms, 1.1 % behind); the
  CPU OpenMP row (19.5 ms) is again faster on uniform.
- **Large batches (N = 10⁶).** The automatic-policy arm is the fastest GPU arm on
  spread-100 (17.8 ms, `eagle.simulate` 17.9 ms against Warp's 62.1 ms; the
  CPU OpenMP row is faster at 15.7 ms) and
  `eagle.simulate` ties its hand-built twin on spread-1000 (156 ms each,
  against Warp's 609 ms; the CPU OpenMP row is faster at 149 ms); on uniform `eagle.simulate` and `warp_kernel` tie
  (699 ms and 693 ms) and the CPU OpenMP row is faster than both (191 ms). Across
  every N and configuration in the card, `eagle.simulate`'s policy arm
  tracks its hand-built twin -- state, params and the stop rule given by name
  instead of bound into a plan by hand -- within 1 % in wall time in every cell but
  N = 10³ and 10⁴ spread-100 (125 µs vs 130 µs; 340 µs vs 381 µs,
  `eagle.simulate`'s per-call host work): the problem-level door costs
  next to nothing extra.
- **What compaction adds on a thinning batch.** On spread-1000, compaction
  only pays from about N = 10⁵: at N = 10³ and 10⁴ `eagle_graph_compact`
  costs more than plain `eagle_graph` (6.53 ms vs 1.62 ms at N = 10³; 9.25 ms
  vs 7.61 ms at N = 10⁴ -- the active-set scan has little to recover from at
  this size), crosses over at N = 10⁵ (45.7 ms vs 66.6 ms, compaction now
  1.46× faster) and keeps winning at N = 10⁶ (415 ms vs 625 ms, 1.51×).
- **What compaction costs on a dense batch.** On the uniform configuration,
  where no sample stops early, compaction is overhead with nothing to
  recover at every N measured: 5.01 ms vs 1.32 ms at N = 10³ (3.8×), 12.8 ms
  vs 7.74 ms at N = 10⁴ (1.7×), 95.1 ms vs 72.3 ms at N = 10⁵ (1.32×), and
  862 ms vs 703 ms at N = 10⁶ (1.23×) -- plain `eagle_graph` is the better
  fit for a batch that never thins.
- **What the reorder adds.** On spread-1000, the reorder on top of
  compaction starts paying at N = 10⁵, cutting 45.7 ms to 34.1 ms (1.34×);
  at N = 10⁶ it cuts 415 ms to 279 ms (1.49×). Below that it costs slightly
  more than compaction alone (6.65 ms vs 6.53 ms at N = 10³; 9.64 ms vs
  9.25 ms at N = 10⁴). On the uniform configuration, with nothing to reorder,
  it lands within 4 % of compaction alone at every N (863 ms vs 862 ms at
  N = 10⁶).
- **The compaction cadence sweep.** At N = 10⁶ and K = 16, the cadence-matched
  map-free arm runs 757 ms against `eagle_graph_compact`'s 415 ms (1.82×);
  the main table's `eagle_graph` vs `eagle_graph_compact` difference at the
  same N is 625 ms vs 415 ms (1.51×). Across K = 8, 16 and 32 the sweep's
  ratio climbs from 1.74× to 1.87× at N = 10⁶ and 1.67× to 1.86× at N = 10⁵:
  the active-set map, not the guard-check cadence, accounts for most of the
  difference, and compacting less often gives up little of the benefit over
  this K range.
- **Memory traffic.** PyTorch's CUDA-graph arm reaches the card's full
  issued bandwidth at N = 10⁵ uniform (1.40e11 B/s, "1e+02 %" in the table);
  CuPy masked reaches 87 % of peak at N = 10⁶ uniform. Both arms' wall time
  comes from the number of passes they make over the whole batch every
  step, finished samples included, rather than from slow passes.
- **Device memory.** At N = 10⁶ spread, `eagle.simulate`, the plain graph
  and the persistent arms each hold 50 MiB (1.31× the minimum the planes
  need) and `warp_kernel` 64 MiB (1.68×); CuPy masked holds 194 MiB
  (5.09×) and PyTorch masked 262 MiB (6.87×).
- **Finished samples.** In the hawk step a sample that is already finished
  when a launch starts skips its whole body (no loads, arithmetic or stores),
  but its warp keeps running while any other lane in it is live; the
  active-set map in `eagle_graph_compact` goes further and packs the live
  samples into the first warps, so a finished sample no longer occupies a
  lane once the map drops it. The bullets above on compaction's cost and
  benefit, at the same N on the spread and uniform configurations, show what
  that is worth and what it costs when there is nothing to pack.
- **CPU card.** `eagle host, termination loop, 8 threads` (the same hawk
  step through hawk + eagle's OpenMP team) is the fastest arm at N = 10⁴ and
  10⁵ in all three configurations (270 µs and 1.68 ms spread-100; 2.36 ms
  and 14.9 ms spread-1000; 2.01 ms and 22.5 ms uniform) and at N = 10⁶ (16.5 ms,
  145 ms and 194 ms); at N = 10³ on spread-100 it is also ahead of Numba
  (55.7 µs against 74.8 µs); Numba's `prange` loop, which keeps each sample's
  state in registers across its own steps, is the next-fastest arm at
  N = 10⁶ on both spread configurations (49.9 ms spread-100; 501 ms
  spread-1000). CPU compaction does not track the GPU card's pattern on the
  thinning spread-1000 workload: at N = 10⁶ it costs 11.0× the plain
  termination loop (1.6 s vs 145 ms), where the GPU's compaction instead
  wins. On the dense uniform workload it costs more than the plain
  termination loop as well (2.5 s vs 194 ms, 13×) -- matching the GPU's own
  loss on a batch that never thins, and larger on the CPU.

## A second workload: adaptive RK7(8)

The RK4 oscillator above is a fixed-step problem: every accepted step costs
the same arithmetic, and the four friction points (launch overhead, host
round trips, finished samples, idle lanes) show up against a constant
per-step cost. The second card runs the same class of problem through an
*adaptive* solver instead: a two-body Kepler orbit (float64, eccentricity
uniform in [0, 0.9]) integrated with Fehlberg's embedded RKF7(8) pair to
rtol = atol = 1e-10, so each sample also decides, step by step, whether to
accept or shrink -- a second, data-dependent source of per-sample
divergence on top of each sample's own stopping time. `eagle.simulate`
appears here too, running the same attempt kernel the hand-built `hawk + eagle
graph` arm uses, under its automatic launch policy.

### GPU (Quadro P2000)

```{include} _generated/rk78_card_gpu.md
```

### CPU (Intel Xeon W-2125)

```{include} _generated/rk78_card_cpu.md
```

### Reading the RK7(8) results

<!-- stale-prose-gate:
benchmarks/rk78_card/card_quadro-p2000.json: 5fce90308cf314b6453a47b2e3a8641a
benchmarks/rk78_card/cpu_card_intel-r-xeon-r-w-2125-cpu-4-00ghz.json: 24dbd74c34ae7e754222511411ce945f
-->

- **Fastest arm.** At N = 1,000,000 `eagle.simulate` and `hawk + eagle
  persistent` are level on the GPU card (13.0 s each);
  plain `hawk + eagle graph` takes 1.81× as long (23.5 s). Compaction takes the
  graph to 14.2 s (1.66× faster), and the reorder adds nothing beyond it
  (14.1 s, 1× the compaction arm's time).
- **`eagle.simulate` here.** `eagle.simulate` runs the same attempt kernel
  under eagle's automatic policy, which picks the launch mode from the batch
  (one launch for a small batch, the persistent launch above it; the card
  records the mode taken). From N = 10,000 up it ties `hawk + eagle
  persistent` (160 ms vs 160 ms at N = 10,000; 13.0 s each at N = 1,000,000);
  at N = 1,000 it takes the single launch (29.7 ms against 49.1 ms for the
  persistent arm and 30.7 ms for the plain graph).
  Compile time is its own pass: a cold run of the eagle arms (they all deploy
  the same hawk attempt kernel and its active-set variant) costs 3.36 s,
  falling to 527 ms warm.
- **GPU vs. CPU.** This workload is where the CPU wins outright at every N:
  `hawk + eagle CPU (OpenMP)` reaches 7.15 s at N = 1,000,000 on the GPU
  card's own CPU reference row -- about twice as fast as every GPU arm,
  including the 13.0 s `eagle.simulate` -- and the dedicated CPU card
  confirms it independently at 6.82 s (`hawk + eagle CPU, 8 threads`). (The
  RK4 dense batch above shows a milder version of the same effect.) An
  adaptive, branch-heavy per-sample attempt with only 1106 FLOP leaves little
  for the GPU's wider lanes to amortize against its own launch and
  active-set overhead, on a card whose FP64 throughput is modest; Numba's
  `prange` loop is the fastest *other* CPU arm at 8.81 M attempts-samples/s
  (N = 1,000,000), and JAX's compiled `vmap(while_loop)` is the fastest
  arm that also differentiates and runs on a GPU unmodified (40.2 s on this
  GPU card, 122 s on the CPU card). NVIDIA Warp's per-thread kernel takes
  26.3 s on the GPU.
- **Agreement.** Every arm reaches the same final time and matches the
  analytic Kepler state within the card's derived bound at every N; only at
  N = 1,000,000 do a handful of samples (at most 4 of a million) take a
  different accept/reject path, from a floating-point difference in how
  each framework contracts or rounds `pow`/`sqrt`, not from a tolerance
  violation.

## Reproduce

From the eagle repository root, in an environment with eagle, hawk, CuPy, a
CUDA toolchain and (optionally) Nsight Systems:

```bash
python benchmarks/perf_card/perf_card.py       # RK4 oscillators, GPU
python benchmarks/perf_card/cpu_card.py        # RK4 oscillators, CPU
python benchmarks/rk78_card/rk78_card.py       # RK7(8) Kepler orbits, GPU
python benchmarks/rk78_card/cpu_rk78_card.py   # RK7(8) Kepler orbits, CPU
```

Each run builds the hawk kernel(s), verifies the arms against each other,
times them, and writes its own `card_<device>.json` together with the
`card_<device>.md` table shown above. The JSON carries the md5 of the script
that produced it; a test re-renders the table from the JSON so the two
cannot drift. `--quick` runs a reduced matrix as a smoke test.

To reproduce the cards on another machine (Kaggle, a rented GPU or your own box) with the same pinned software, see [`benchmarks/reproduce/`](https://github.com/amasat01/eagle/tree/main/benchmarks/reproduce).

**Adding a device.** Drop the new `card_<slug>.json`/`.md` (GPU) or
`cpu_card_<slug>.json`/`.md` (CPU) next to the existing ones under
`benchmarks/perf_card/` or `benchmarks/rk78_card/` -- `benchmarks/reproduce/run_cards.sh
--publish` writes them there directly. Rebuilding the docs (`make html` or `make
strict`) then regenerates this page's "Measured on" tables and per-device
sections on its own; re-running a card for a device already listed replaces
that device's rows the same way. No other file on this page needs editing.

## Hardware caveat

The GPU cards above come from a Quadro P2000, a development, consumer-class
card whose FP64 throughput (1:32 of its FP32 rate) and memory bandwidth are
recorded in each card, and from a Kaggle Tesla T4; the CPU cards come from an Intel Xeon W-2125
workstation chip (4 cores, 8 logical CPUs). The method transfers to other
hardware unchanged; the absolute numbers, and possibly the crossover points
between arms, will differ on data-centre GPUs or larger CPUs, whose FP64
throughput, bandwidth and core counts differ from these cards'. A card from
such a device would be added beside these, produced by the same scripts.
