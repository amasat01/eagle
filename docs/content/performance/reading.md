# Reading the numbers

What the results show, with the cards' own figures. Each statement can be checked against the tables on the results pages ([RK4](rk4.md), [RK7(8)](rk78.md)).

## RK4 oscillators

<!-- stale-prose-gate:
benchmarks/perf_card/card_quadro-p2000.json: d37837fb31567858848e5f1830f19431
benchmarks/perf_card/cpu_card_intel-xeon-w-2125.json: 5bb591471e7758948a1a2f4994dc575b
-->

Every statement below can be checked against the tables and the
"fastest arm" lines on the results pages (GPU card unless marked CPU card).

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

## Adaptive RK7(8)

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
