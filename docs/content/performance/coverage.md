# What the cards cover

Every backend feature has a card row, a profile stage, or a reason it has none. Nothing outside this table is measured, and
nothing in it is measured twice.

**Cards.** C1 is the ensemble card: an adaptive RKF7(8) Kepler integration, with its arms and cells. C2 is the feature card: one
row per capability, at one batch size, each arm against a baseline. The profile stages (P1 to P5) are the launch-form sweep and
the per-kernel profiles in `benchmarks/reproduce`.

| group | item | covered by | reason |
|---|---|---|---|
| lowering | per-sample thread form | C2 `wide_forms` (thread arm); C1 every eagle arm | the default form |
| lowering | warp-team form, chosen by lane count | C2 `wide_forms` (team arm); P1 sweep | the 0.7.3 change |
| lowering | staged wide outputs, team form in stepped kernels | C2 `wide_forms` (staged arm); C2 `steps_fused` | one row per form, one per stepped use |
| lowering | host interchange and wide carries | C2 `wide_forms` host arms; the hawk wide-value gate | the same kernels on the host target |
| lowering | emit form, auto or static | covered by `wide_forms` | auto is what every row builds, so a static arm adds no claim |
| lowering | team linear algebra to 4096, one-thread form below 32 | C2 `linalg_lane`, `linalg_team` | both forms, and both precisions on the lane form |
| lowering | thread box, contraction route | C2 `contraction`; P1 sweep | the row is the sweep shape |
| lowering | fused `steps=K`, fused reverse and forward derivatives | C2 `steps_fused` | one kernel body, every K path |
| lowering | complex pairs, rays, pair lane | no timing row | library forms, shown by the docs notebooks, no speed claim |
| lowering | local-memory limit at build | no timing row | a build gate, not a measurement |
| launch | graph capture and replay | C1 arm `eagle_graph`; every C2 row | the plain device loop |
| launch | persistent launch, chunk, geometry | C1 arm `eagle_graph_persist`; P4; P3 | the mechanism behind the headline |
| launch | one-launch or automatic pick, `simulate`, `until_done` | C1 arm `eagle_simulate`; P5 | `simulate` runs the object `until_done` returns |
| launch | band, compaction, reorder | C1 command-line arms; P2000-only profile stage | pays only on small cards, so no committed cell |
| launch | device-aware launches | covered by `mpi_partition` | correctness is a test, speed is the multi-GPU row |
| lanes | lane sets, count between launches | C2 `lane_set` | dense, map and prefix arms |
| reductions | sums over samples, compensated, ordered or auto | C2 `sample_sum` | against `cupy.sum` |
| reductions | keyed, ordered and OR scatter | C2 `scatter` | one row, three policies |
| reductions | cells folded across ranks | C2 `mpi_partition` | the fold is part of the row's graph |
| control | run-time trip count, break, jump, segmented units | C2 `control_flow` | one stepped body |
| derivatives | forward and reverse, through loops and lane sets | C2 `steps_fused`; the hawk gradient corpus | the recorded-loop path is what the automatic arm picks |
| precision | float32 and float64 | C1 float32 cell; C2 `wide_forms`, `linalg_lane`, `contraction` | both dtypes where it matters |
| precision | 32-bit and 64-bit index | covered by `lane_set` | the row uses 32-bit indices |
| host | host and C targets, OpenMP GEMM | C1 CPU card; C2 host arms | the C target emits the same text, one-threaded |
| host | launch overhead at N=1, graph update | no timing row | a local gate, not a cloud row |
| multi-GPU | run follows the arrays' GPU, ranks across GPUs | C2 `mpi_partition` | two GPUs against one, like for like per GPU |
| graph steps | batched FFT | C2 `fft` | against cuFFT |
| graph steps | GEMM through cuBLAS, opt-in | C2 `contraction` (gemm-route arm) | the opt-in route |
| interop | DLPack bridges, streams | covered by C1 | the eagle arms ingest CuPy-owned planes on the card's stream; no timing claim |
| compile | build turnaround | no timing row | a local card, not a cloud row |
