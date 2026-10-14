# Reading the numbers

What the results show, with the benchmark cards' own figures. Each statement can be checked against the tables on the results pages ([RK4](rk4.md), [RK7(8)](rk78.md)). For the mechanisms behind these numbers, see [Where the performance comes from](mechanism.md).

## RK4 oscillators

<!-- stale-prose-gate:
benchmarks/perf_card/card_nvidia-h100-80gb-hbm3.json: e896d6d2050fc792c90b59697547201b
benchmarks/perf_card/cpu_card_amd-epyc-9b45.json: ff15fa43e17e2c9a88f2ba336e8aa476
-->

<!--ctx card=rk4 dev=h100 n=1e6 dist=spread steps=1000-->

Every statement below can be checked against the tables and the
"fastest arm" lines on the results pages. The GPU figures come from the
NVIDIA H100 card and the CPU figures from the AMD EPYC 9B45 card, unless a
sentence names another device; the other devices have their own tabs on the
results pages. Each tool in the card is good at something, and the cells
below say where.

- **Small batches (N = 10³).** Among the GPU arms the automatic-policy arm
  is fastest in all three configurations (<!--c wall arm=eagle_graph_auto n=1000 steps=100-->42.4 µs<!--/c--> spread-100, <!--c wall arm=eagle_graph_auto n=1000-->94.9 µs<!--/c-->
  spread-1000, <!--c wall arm=eagle_graph_auto n=1000 dist=uniform-->99.8 µs<!--/c--> uniform), with `eagle.simulate` second (<!--c wall arm=eagle_simulate n=1000 steps=100-->44 µs<!--/c-->, <!--c wall arm=eagle_simulate n=1000-->103 µs<!--/c-->
  and <!--c wall arm=eagle_simulate n=1000 dist=uniform-->101 µs<!--/c-->) and `warp_kernel` third (<!--c wall arm=warp_kernel n=1000 steps=100-->51.9 µs<!--/c-->, <!--c wall arm=warp_kernel n=1000-->112 µs<!--/c-->
  and <!--c wall arm=warp_kernel n=1000 dist=uniform-->118 µs<!--/c-->). At the fixed-overhead cell (N = 64, S = 100) `eagle.simulate` takes
  <!--c wall arm=eagle_simulate n=64 dist=uniform steps=100 src=fixed-->56.4 µs<!--/c--> and `warp_kernel` <!--c wall arm=warp_kernel n=64 dist=uniform steps=100 src=fixed-->67.5 µs<!--/c-->. The CPU OpenMP row,
  which needs no GPU, takes <!--c wall arm=cpu_openmp n=1000 steps=100-->182 µs<!--/c-->, <!--c wall arm=cpu_openmp n=1000-->406 µs<!--/c--> and <!--c wall arm=cpu_openmp n=1000 dist=uniform-->369 µs<!--/c--> in the same
  three cells: slower than `eagle.simulate` and `warp_kernel` in each of them, and the tool to reach for when there is no GPU at hand.
- **Mid-size batches (N = 10⁴-10⁵).** At N = 10⁴ the automatic-policy arm is
  fastest in all three configurations (<!--c wall arm=eagle_graph_auto n=10000 steps=100-->55.1 µs<!--/c--> spread-100, <!--c wall arm=eagle_graph_auto n=10000-->106 µs<!--/c--> spread-1000,
  <!--c wall arm=eagle_graph_auto n=10000 dist=uniform-->111 µs<!--/c--> uniform). On both spread configurations `warp_kernel` (<!--c wall arm=warp_kernel n=10000 steps=100-->59.2 µs<!--/c-->,
  <!--c wall arm=warp_kernel n=10000-->116 µs<!--/c-->) comes next, a little ahead of `eagle.simulate` (<!--c wall arm=eagle_simulate n=10000 steps=100-->61.4 µs<!--/c-->, <!--c wall arm=eagle_simulate n=10000-->118 µs<!--/c-->);
  on uniform `eagle.simulate` (<!--c wall arm=eagle_simulate n=10000 dist=uniform-->112 µs<!--/c-->) comes next, ahead of `warp_kernel`
  (<!--c wall arm=warp_kernel n=10000 dist=uniform-->121 µs<!--/c-->). At N = 10⁵ the automatic-policy arm leads spread-100 (<!--c wall arm=eagle_graph_auto n=100000 steps=100-->82 µs<!--/c-->, with
  `warp_kernel` at <!--c wall arm=warp_kernel n=100000 steps=100-->83.7 µs<!--/c--> and `eagle.simulate` at <!--c wall arm=eagle_simulate n=100000 steps=100-->87.2 µs<!--/c-->) and spread-1000 (<!--c wall arm=eagle_graph_auto n=100000-->197 µs<!--/c-->,
  `eagle.simulate` <!--c wall arm=eagle_simulate n=100000-->199 µs<!--/c--> and `warp_kernel` <!--c wall arm=warp_kernel n=100000-->237 µs<!--/c--> further back). On uniform at N = 10⁵ the
  per-thread `warp_kernel` is fastest (<!--c wall arm=warp_kernel n=100000 dist=uniform-->245 µs<!--/c-->), with the automatic-policy arm
  (<!--c wall arm=eagle_graph_auto n=100000 dist=uniform-->252 µs<!--/c-->) and `eagle.simulate` (<!--c wall arm=eagle_simulate n=100000 dist=uniform-->260 µs<!--/c-->, <!--c pctgap arm=eagle_simulate den_arm=warp_kernel n=100000 dist=uniform-->6%<!--/c--> longer than `warp_kernel`) close behind.
  The CPU OpenMP row takes <!--c wall arm=cpu_openmp n=100000 dist=uniform-->16.5 ms<!--/c--> on uniform at N = 10⁵, against `eagle.simulate`'s
  <!--c wall arm=eagle_simulate n=100000 dist=uniform-->260 µs<!--/c-->.
- **Large batches (N = 10⁶).** On both spread configurations the automatic-policy arm
  is fastest (<!--c wall arm=eagle_graph_auto steps=100-->163 µs<!--/c--> spread-100, <!--c wall arm=eagle_graph_auto-->601 µs<!--/c--> spread-1000) with `eagle.simulate`
  next (<!--c wall arm=eagle_simulate steps=100-->169 µs<!--/c-->, <!--c wall arm=eagle_simulate-->613 µs<!--/c-->). `warp_kernel` takes <!--c wall arm=warp_kernel steps=100-->234 µs<!--/c--> and
  <!--c wall arm=warp_kernel-->1.54 ms<!--/c-->, which is <!--c x arm=warp_kernel steps=100 den_arm=eagle_simulate-->1.3×<!--/c--> and <!--c x arm=warp_kernel den_arm=eagle_simulate-->2.5×<!--/c--> `eagle.simulate`'s time.
  On uniform, where every sample runs all 1000 steps, the per-thread kernel is the
  fastest arm: `warp_kernel` takes <!--c wall arm=warp_kernel dist=uniform-->1.71 ms<!--/c--> against the automatic-policy arm's
  <!--c wall arm=eagle_graph_auto dist=uniform-->1.74 ms<!--/c--> and `eagle.simulate`'s <!--c wall arm=eagle_simulate dist=uniform-->1.75 ms<!--/c--> (<!--c pctgap arm=eagle_simulate den_arm=warp_kernel dist=uniform-->2.5%<!--/c--> longer than
  `warp_kernel`), so the two sit close together when nothing thins, and Warp is the better fit for that shape. The CPU OpenMP row takes
  <!--c wall arm=cpu_openmp steps=100-->21.1 ms<!--/c-->, <!--c wall arm=cpu_openmp-->199 ms<!--/c--> and <!--c wall arm=cpu_openmp dist=uniform-->170 ms<!--/c--> in the three cells, <!--c x arm=cpu_openmp steps=100 den_arm=eagle_simulate-->120×<!--/c-->,
  <!--c x arm=cpu_openmp den_arm=eagle_simulate-->320×<!--/c--> and <!--c x arm=cpu_openmp dist=uniform den_arm=eagle_simulate-->96×<!--/c--> `eagle.simulate`'s time. The policy arm and its
  hand-built twin (`eagle.simulate` takes state, params and the stop rule by
  name instead of bound into a plan by hand) sit close together; the gap is
  `eagle.simulate`'s per-call host work, which is most visible on small
  batches (<!--c wall arm=eagle_graph_auto n=1000 steps=100-->42.4 µs<!--/c--> vs <!--c wall arm=eagle_simulate n=1000 steps=100-->44 µs<!--/c--> at N = 10³ spread-100;
  <!--c wall arm=eagle_graph_auto n=10000 steps=100-->55.1 µs<!--/c--> vs <!--c wall arm=eagle_simulate n=10000 steps=100-->61.4 µs<!--/c--> at N = 10⁴ spread-100) and
  small at the largest batch (<!--c pctgap arm=eagle_simulate den_arm=eagle_graph_auto-->2%<!--/c--> on spread-1000 at N = 10⁶).
- **Compaction on this card.** The plain graph is faster than the compaction
  arm in every cell of the card, thinning batch or not: on spread-1000,
  <!--c wall arm=eagle_graph n=1000-->498 µs<!--/c--> against <!--c wall arm=eagle_graph_compact n=1000-->3.9 ms<!--/c--> at N = 10³ and <!--c wall arm=eagle_graph-->2.13 ms<!--/c--> against
  <!--c wall arm=eagle_graph_compact-->16 ms<!--/c--> at N = 10⁶; on uniform, <!--c wall arm=eagle_graph n=1000 dist=uniform-->483 µs<!--/c--> against
  <!--c wall arm=eagle_graph_compact n=1000 dist=uniform-->3.03 ms<!--/c--> at N = 10³ and <!--c wall arm=eagle_graph dist=uniform-->2.32 ms<!--/c--> against <!--c wall arm=eagle_graph_compact dist=uniform-->22.9 ms<!--/c-->
  at N = 10⁶. The H100 runs the whole batch so quickly that the active-set
  bookkeeping has little to recover, so plain `eagle_graph` is the better fit here.
  The device tabs show cards where the picture differs: on the P2000, at
  N = 10⁶ on spread-1000, compaction takes <!--c wall dev=p2000 arm=eagle_graph_compact-->374 ms<!--/c--> against the plain graph's
  <!--c wall dev=p2000 arm=eagle_graph-->586 ms<!--/c--> (<!--c x dev=p2000 arm=eagle_graph den_arm=eagle_graph_compact-->1.5×<!--/c--> faster).
- **The reorder on this card.** The reorder on top of compaction is faster
  than compaction alone in one cell of the card, N = 10⁶ on spread-1000, where it
  takes <!--c wall arm=eagle_graph_reorder-->10.7 ms<!--/c--> against <!--c wall arm=eagle_graph_compact-->16 ms<!--/c--> (<!--c x arm=eagle_graph_compact den_arm=eagle_graph_reorder-->1.4×<!--/c-->). In every other
  cell it takes longer than compaction alone, for example
  <!--c wall arm=eagle_graph_reorder n=100000-->5.2 ms<!--/c--> against <!--c wall arm=eagle_graph_compact n=100000-->5.08 ms<!--/c--> at N = 10⁵ on spread-1000 and
  <!--c wall arm=eagle_graph_reorder dist=uniform-->23.3 ms<!--/c--> against <!--c wall arm=eagle_graph_compact dist=uniform-->22.9 ms<!--/c--> at N = 10⁶ on uniform.
- **The compaction cadence sweep.** At N = 10⁶ and K = 16, the cadence-matched
  map-free arm runs <!--c wall src=sweep variant=matched_cadence k=16-->12.6 ms<!--/c--> against `eagle_graph_compact`'s <!--c wall src=sweep variant=compact k=16-->16 ms<!--/c--> (<!--c x src=sweep variant=compact den_variant=matched_cadence k=16-->1.2×<!--/c-->);
  the main table's plain `eagle_graph`, which checks its stop guard every step, takes <!--c wall arm=eagle_graph-->2.13 ms<!--/c-->. Across
  K = 8, 16 and 32 the sweep's ratio falls from <!--c x src=sweep variant=compact den_variant=matched_cadence k=8-->1.4×<!--/c--> to <!--c x src=sweep variant=compact den_variant=matched_cadence k=32-->1.1×<!--/c--> at N = 10⁶ and
  from <!--c x src=sweep variant=compact den_variant=matched_cadence k=8 n=100000-->2.2×<!--/c--> to <!--c x src=sweep variant=compact den_variant=matched_cadence k=32 n=100000-->1.6×<!--/c--> at N = 10⁵. On this card the
  active-set map accounts for the smaller part of the gap to the plain graph;
  most of it is already there once the guard is checked every K steps.
- **Memory traffic.** PyTorch's CUDA-graph arm reaches the device's full
  issued bandwidth at N = 10⁶ uniform (<!--c rate field=issued_bytes_per_s arm=torch_graphed dist=uniform-->3.36e12<!--/c--> B/s, <!--c pct field=fraction_of_peak_dram arm=torch_graphed dist=uniform-->100%<!--/c--> of the DRAM peak, analytic not measured);
  CuPy masked reaches <!--c pct field=fraction_of_peak_dram arm=cupy_masked dist=uniform-->74%<!--/c--> of peak in the same cell. Both arms' wall time
  comes from the number of passes they make over the whole batch every
  step, finished samples included, rather than from slow passes.
- **Device memory.** At N = 10⁶ spread, `eagle.simulate` and the persistent arm
  each hold <!--c mib arm=eagle_graph_persist-->40 MiB<!--/c--> (<!--c xmin arm=eagle_graph_persist-->1.1×<!--/c--> the minimum the planes
  need), the plain graph <!--c mib arm=eagle_graph-->42 MiB<!--/c--> (<!--c xmin arm=eagle_graph-->1.2×<!--/c-->) and `warp_kernel` <!--c mib arm=warp_kernel-->62 MiB<!--/c--> (<!--c xmin arm=warp_kernel-->1.8×<!--/c-->);
  CuPy masked holds <!--c mib arm=cupy_masked-->192 MiB<!--/c--> (<!--c xmin arm=cupy_masked-->5.5×<!--/c-->) and PyTorch masked <!--c mib arm=torch_masked-->350 MiB<!--/c--> (<!--c xmin arm=torch_masked-->10×<!--/c-->).
- **Finished samples.** In the hawk step a sample that is already finished
  when a launch starts skips its whole body (no loads, arithmetic or stores),
  but its warp keeps running while any other lane in it is live; the
  active-set map in `eagle_graph_compact` goes further and packs the live
  samples into the first warps, so a finished sample no longer occupies a
  lane once the map drops it. The bullets above on compaction show what that
  is worth, and what it costs, on the H100; the P2000 tab shows a card where
  the packing pays back its bookkeeping on a thinning batch.
- **CPU card.** <!--ctx card=rk4cpu dev=amd-epyc-9b45-->`eagle host, termination loop, 8 threads` (the same hawk
  step through hawk + eagle's OpenMP team) is the fastest arm at every N
  (N = 10³ to 10⁶) in all three configurations. At N = 10⁴ it takes <!--c wall arm=eagle_term8 n=10000 steps=100-->159 µs<!--/c-->
  spread-100, <!--c wall arm=eagle_term8 n=10000-->1.5 ms<!--/c--> spread-1000 and <!--c wall arm=eagle_term8 n=10000 dist=uniform-->1.94 ms<!--/c--> uniform; at
  N = 10⁶, <!--c wall arm=eagle_term8 steps=100-->14.4 ms<!--/c-->, <!--c wall arm=eagle_term8-->142 ms<!--/c--> and <!--c wall arm=eagle_term8 dist=uniform-->187 ms<!--/c-->. Numba's `prange` loop, which
  keeps each sample's state in registers across its own steps, is the next-fastest arm at
  N = 10⁶ on spread-100 (<!--c wall arm=numba_prange steps=100-->32.4 ms<!--/c-->); on spread-1000 at that N the compacted and reordered
  eagle arms come next (<!--c wall arm=eagle_reorder8-->242 ms<!--/c--> and <!--c wall arm=eagle_compact8-->292 ms<!--/c-->), ahead of Numba (<!--c wall arm=numba_prange-->328 ms<!--/c-->). At N = 10³ on
  spread-100 `eagle host, termination loop, 8 threads` (<!--c wall arm=eagle_term8 n=1000 steps=100-->27.2 µs<!--/c-->) is also ahead of Numba
  (<!--c wall arm=numba_prange n=1000 steps=100-->39 µs<!--/c-->). CPU compaction does not pay back its bookkeeping on this card either: at N = 10⁶ it costs
  <!--c x arm=eagle_compact8 den_arm=eagle_term8-->2.0×<!--/c--> the plain termination loop on spread-1000 (<!--c wall arm=eagle_compact8-->292 ms<!--/c--> vs <!--c wall arm=eagle_term8-->142 ms<!--/c-->) and
  <!--c x arm=eagle_compact8 dist=uniform den_arm=eagle_term8-->3.9×<!--/c--> on the dense uniform workload (<!--c wall arm=eagle_compact8 dist=uniform-->744 ms<!--/c--> vs <!--c wall arm=eagle_term8 dist=uniform-->187 ms<!--/c-->), the same
  direction as on the H100. CPU peak memory is shown for N = 1,000,000 only, net of the benchmark harness's own copies (36 bytes
  per sample on eagle's arms). At smaller N this harness cannot resolve it: each measuring process inherits the launching
  process's peak, so smaller readings show that floor rather than the workload. The cards keep the raw readings; the next
  card run measures every size directly.<!--ctx card=rk4 dev=h100-->

## Adaptive RK7(8)

<!-- stale-prose-gate:
benchmarks/rk78_card/card_nvidia-h100-80gb-hbm3.json: 763658c485f8b65b8ff42c1dc3bba84c
benchmarks/rk78_card/cpu_card_amd-epyc-9b45.json: 07b9ec84b69364c1e746befa19e5c544
-->

<!--ctx card=rk78 dev=h100-->

- **Fastest arm.** At every N on the GPU card `eagle.simulate` is the
  fastest arm, <!--c wall arm=eagle_simulate n=1000-->2.67 ms<!--/c--> at N = 1,000 and <!--c wall arm=eagle_simulate-->43.6 ms<!--/c--> at N = 1,000,000.
  At N = 1,000,000 `hawk + eagle persistent` is level with it (<!--c wall arm=eagle_graph_persist-->43.7 ms<!--/c-->,
  <!--c pctgap arm=eagle_graph_persist den_arm=eagle_simulate-->0.14%<!--/c--> longer); plain `hawk + eagle graph` takes
  <!--c x arm=eagle_graph den_arm=eagle_simulate-->2.1×<!--/c--> as long (<!--c wall arm=eagle_graph-->95.4 ms<!--/c-->) and Warp's per-thread kernel <!--c wall arm=warp_kernel-->101 ms<!--/c-->
  (<!--c x arm=warp_kernel den_arm=eagle_simulate-->2.3×<!--/c-->). Compaction takes <!--c wall arm=eagle_graph_compact-->117 ms<!--/c--> here, longer than the plain graph, and
  the reorder lands between the two (<!--c wall arm=eagle_graph_reorder-->108 ms<!--/c-->).
- **`eagle.simulate` here.** `eagle.simulate` runs the same attempt kernel
  under eagle's automatic policy, which picks the launch mode from the batch
  (one launch for a small batch, the persistent launch above it; the card
  records the mode taken). At N = 10⁵ and above it is level with `hawk + eagle
  persistent` (<!--c wall arm=eagle_simulate n=100000-->7.2 ms<!--/c--> vs <!--c wall arm=eagle_graph_persist n=100000-->7.3 ms<!--/c--> at N = 100,000; <!--c wall arm=eagle_simulate-->43.6 ms<!--/c--> vs <!--c wall arm=eagle_graph_persist-->43.7 ms<!--/c--> at N = 1,000,000);
  at N = 1,000 and 10,000 it takes the single launch (<!--c wall arm=eagle_simulate n=1000-->2.67 ms<!--/c--> and <!--c wall arm=eagle_simulate n=10000-->2.72 ms<!--/c-->, against
  <!--c wall arm=eagle_graph_persist n=1000-->3.17 ms<!--/c--> and <!--c wall arm=eagle_graph_persist n=10000-->3.23 ms<!--/c--> for the persistent arm and <!--c wall arm=eagle_graph n=1000-->3.3 ms<!--/c--> and
  <!--c wall arm=eagle_graph n=10000-->3.34 ms<!--/c--> for the plain graph).
  At N = 1,000,000 it reaches <!--c pct field=fraction_of_peak_fp64 arm=eagle_simulate-->38%<!--/c--> of the card's FP64 peak.
  Compile time is its own pass: a cold run of the eagle arms (they all deploy
  the same hawk attempt kernel and its active-set variant) costs <!--c comp card=rk78 key=hawk_cuda which=cold-->2.48 s<!--/c-->,
  falling to <!--c comp card=rk78 key=hawk_cuda which=warm-->333 ms<!--/c--> warm.
- **GPU and CPU.** This workload is an adaptive, branch-heavy attempt per
  sample, and the GPU arms pull well ahead of the CPU ones at N = 1,000,000.
  `hawk + eagle CPU (OpenMP)` takes <!--c wall arm=eagle_cpu-->6.51 s<!--/c--> on the GPU card's own host row
  (<!--c x arm=eagle_cpu den_arm=eagle_simulate-->140×<!--/c--> the <!--c wall arm=eagle_simulate-->43.6 ms<!--/c--> of `eagle.simulate`), and
  <!--c wall card=rk78cpu dev=amd-epyc-9b45 arm=eagle_cpu8-->3.76 s<!--/c--> on the dedicated CPU card (`hawk + eagle CPU, 8 threads`).
  That path is there for machines without a GPU, and on the CPU card it is the
  fastest arm at every N; Numba's `prange` loop takes
  <!--c wall card=rk78cpu dev=amd-epyc-9b45 arm=numba_prange-->19.9 s<!--/c--> and JAX's compiled `vmap(while_loop)`
  <!--c wall card=rk78cpu dev=amd-epyc-9b45 arm=jax_vmap-->60.5 s<!--/c--> there. On the GPU card JAX is the fastest of the array-framework
  arms (<!--c wall arm=jax_vmap-->1.11 s<!--/c-->, ahead of CuPy at <!--c wall arm=cupy_masked-->9.06 s<!--/c--> and PyTorch at <!--c wall arm=torch_graphed-->7.17 s<!--/c--> with its CUDA graph), and
  NVIDIA Warp's per-thread kernel takes <!--c wall arm=warp_kernel-->101 ms<!--/c-->.
- **Agreement.** Every arm reaches the same final time and matches the
  analytic Kepler state within the card's derived bound at every N; only at
  N = 1,000,000 do a handful of samples take a
  different accept/reject path, from a floating-point difference in how
  each framework contracts or rounds `pow`/`sqrt`, not from a tolerance
  violation.
