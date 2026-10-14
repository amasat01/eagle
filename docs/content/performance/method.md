# Method

- **One block per arm (RK4 GPU cards).** For each configuration and N, every arm is timed in
  its own block, one arm after the other, with no interleaving. A block is two
  warm-up runs (excluded; the first also verifies the result), then untimed
  back-to-back runs of that arm until 0.5 s has elapsed (at least one run), so
  the clocks are settled, then 5 timed runs. Every reported time is the mean
  of the middle 3 of the 5 runs (the single highest and the single lowest
  dropped), shown with the range of those 3; the median, the interquartile
  range and every run are kept in the card's JSON. The RK7(8) cards and the CPU
  cards report the median and interquartile range of their repetitions and rotate
  the arm order every repetition instead; each card's JSON records its own
  `method`, which is the authority for that card.
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
  from host memory and downloading the results. *Kernel-only* (on cards run with Nsight Systems; the others record it as skipped) is the sum of
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

## Terms

- **Useful FLOP/s**: floating-point operations of steps a sample really took, per second of wall time.
- **Issued bytes per second**: analytic count of bytes the loads and stores move, per second; not read from hardware counters.
- **Benchmark card**: one device's report; the **device** is the GPU or CPU it ran on.
