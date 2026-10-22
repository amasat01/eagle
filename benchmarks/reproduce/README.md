# Reproduce the performance cards

eagle publishes four cards, each written by its own script: the RK4 and RK7(8)
**GPU** cards (`perf_card/perf_card.py`, `rk78_card/rk78_card.py`) and their
**CPU** twins (`perf_card/cpu_card.py`, `rk78_card/cpu_rk78_card.py`). The
scripts here set up any Linux x86_64 machine to run them with the same pinned
software and write the cards for that machine.

| File | What it does |
|---|---|
| `bootstrap.sh` | pinned toolchain + Python environment, the four repositories (aether, raptor, eagle, hawk) at their release tags, eagle's `_core` for the GPUs present (CPU-only build without one), hawk, an import smoke test and a machine fact sheet |
| `bootstrap.sh --from-pypi` | the no-compiler route: a plain venv with the released `raptor-eagle` / `raptor-hawk` wheels, pins in `locks/pypi-{gpu,cpu}.txt` |
| `run_cards.sh` | runs the cards (`--which gpu\|cpu\|all`, `--preset smoke\|short\|full`) into `cards/<machine-slug>/` |
| `passes.py` | runs a GPU card one pass at a time (used by `run_cards.sh --split`) |
| `calibrate.sh`, `calibrate.py` | the calibration data collector (see below), with its own notebook `kaggle_calibrate.ipynb` and pins `locks/calibrate-{gpu,cpu}.txt` |
| `profile.sh`, `gpu_profile.py` | the profiling kit: decision probe, Nsight Systems and Nsight Compute captures, one tarball (see below) |
| `kaggle.ipynb` | a thin notebook calling the two scripts |
| `locks/` | the pinned environments and `make_locks.py`, which regenerates them |

## Your machine, or a rented one (vast.ai)

Needs: Linux x86_64 with glibc ≥ 2.28, `git`, `curl`, ~25 GB of disk, and for
the GPU cards an NVIDIA driver ≥ 575.51 (CUDA 12.9 runtime; no CUDA toolkit: the environment brings
its own) and a GPU of compute capability ≥ 7.5. For a pre-Turing GPU (P100, V100; sm_60–sm_70) use
`--cuda 12.6` (driver ≥ 525.60.13). No root. On vast.ai any CUDA 12 image works, e.g.
`nvidia/cuda:12.9.1-devel-ubuntu22.04`.

```bash
git clone --depth 1 --branch v0.6.2 https://github.com/amasat01/eagle.git
eagle/benchmarks/reproduce/bootstrap.sh                 # gpu profile when a GPU is visible
eagle/benchmarks/reproduce/bootstrap.sh --profile cpu   # the CPU cards' environment
eagle/benchmarks/reproduce/run_cards.sh --preset smoke --which all   # minutes: checks the setup
eagle/benchmarks/reproduce/run_cards.sh --which all                  # the full cards
```

`--dry-run` prints every command of either script without running anything.
From inside a family checkout (aether/, eagle/, hawk/, raptor/ side by side) the
checkouts are used as they are; elsewhere they are cloned (`--ref eagle=main`
or `EAGLE_REF=...` picks another ref). With several GPUs, set
`CUDA_VISIBLE_DEVICES` to the one to measure. Cards land in
`~/.cache/raptor-repro/cards/<gpu>--<cpu>/` (`--out` to change);
`--publish` writes them next to the committed cards instead.

Committed cards are slim: the raw rows are release assets. After `--publish`, run
`python benchmarks/slim_card.py --raw-out <dir> <card>.json` to slim a card and keep the full one in `<dir>`;
the tests that read raw rows take `EAGLE_CARD_RAW_DIR=<dir>`, are skipped without it, and fail by file name when it is set
and a file is missing or its md5 is not the card's `slim_of_md5`. CI downloads the release asset `cards.tar.gz` (the
`<dir>` tree: `perf_card/` and `rk78_card/`) from the `perf-cards-raw` release:
`tar -C <dir> -czf cards.tar.gz perf_card rk78_card && gh release create perf-cards-raw cards.tar.gz --latest=false --notes "Full performance cards"`
(a later card run: `gh release upload perf-cards-raw cards.tar.gz --clobber`). After a re-slim, the card md5s that moved are
pinned in the READMEs and notebooks (`python tools/sync_readme_cards.py --check` names them), the site's
`_generated/SOURCES.json` (`tools/sync_cards.py <eagle>`), each repo's docs (`python -m raptor_docs.fresh --write`) and hawk's `docs/perf_claims.json` (`tools/sync_perf_claims.py --eagle-root <eagle>`).

## Google Cloud (or any VM with an NVIDIA driver)

Needs Python >= 3.10 with `venv` and its headers, `git`, a C++ compiler (hawk compiles each kernel the
first time it runs) and a driver >= 575.51 (>= 525.60.13 with `--cuda 12.6`); no conda or CUDA toolkit. On Debian or
Ubuntu: `sudo apt-get install python3-venv python3-dev git build-essential`.

```bash
git clone --depth 1 https://github.com/amasat01/eagle.git
eagle/benchmarks/reproduce/bootstrap.sh --from-pypi          # venv + pinned wheels, ~10 GB of disk
eagle/benchmarks/reproduce/run_cards.sh --split --preset smoke   # minutes: checks the setup
eagle/benchmarks/reproduce/run_cards.sh --split                  # the full cards
```

Cards land in `~/.cache/raptor-repro/cards/<gpu>--<cpu>/` (`--out` to change); copy them
off the VM, then delete it. The fact sheet (`machine_facts.txt`) records `install: pypi`
and the exact package versions. `REPRO_PYTHON=/path/to/python` picks the interpreter.

## Kaggle

Open `kaggle.ipynb` in a Kaggle notebook, choose the **GPU T4 x2**
accelerator, turn Internet on, and run all cells. A session is capped at 12 h:
the run is split into resumable steps (`run_cards.sh --split`) and stops before
the cap. To continue, save the version, attach its output to a new session and
set `RESUME_FROM`. Kaggle's 4 shared vCPUs make its CPU cards a different
machine class from the 8-thread arms the CPU cards time, so run the GPU cards there.

## Calibration data

`calibrate.sh` times a few isolated micro-kernels on this machine and writes one CSV of raw
timings: the cost of each operation class (arithmetic and special functions in both
precisions, plane loads, gathers), the launch and plane-traffic overhead, the extra cost of a
split step and of the class sort, a fused cheap/heavy kernel at four heavy fractions (GPU
only), and held-out fused-vs-split pairs that a later fit must reproduce. Every repetition is
kept (the first of each case is marked cold); nothing is fitted here and nothing on the
machine is changed. The data is fitted later, elsewhere, into the weights the dispatch
chooser uses.

```bash
eagle/benchmarks/reproduce/calibrate.sh --quick     # about a minute: checks the setup
eagle/benchmarks/reproduce/calibrate.sh             # a few minutes per target
```

It installs the pinned pre-release `raptor-eagle` / `raptor-hawk` wheels (`locks/calibrate-*.txt`,
`pip install --pre`) into a private venv under `--work`, then writes
`WORK/calibration/<machine-slug>/calibration-<gpu|cpu>-<date>.csv` (`--out` to change) and prints
its path. The profile follows the visible hardware (`--profile gpu|cpu`); run it once for each
of the GPU and the host. The run takes under 10 minutes per target including kernel compiles
(the CSV header records the compile time, the clocks and temperature before and after, the
thread count and `OMP_PROC_BIND`). It needs a C++ compiler; Debian/Ubuntu:
`sudo apt-get install build-essential python3-venv python3-dev`. `--local` uses the eagle and
hawk already installed for `--python` instead of a venv. On Kaggle open `kaggle_calibrate.ipynb`
(GPU T4, Internet on).

## Profiling (nsys + ncu)

`profile.sh` answers *why* a launch mode or a kernel is fast or slow on this GPU, for the cards' kernel.
It runs after `bootstrap.sh` and the cards, unattended, in about 35 minutes on an A100 or H100 (`profile.sh --dry-run` prints the estimate for the plan; limit 50 by default; `--quick`: under 8):

```bash
eagle/benchmarks/reproduce/profile.sh --env ~/.cache/raptor-repro/env-gpu.sh
# -> ~/.cache/raptor-repro/profiles/<gpu>--<cpu>/profile-<gpu>--<cpu>.tgz  (the last line it prints)
```

| Stage | What it captures |
|---|---|
| `probe` | what eagle reads from the device (SMs, registers, resident threads, the issue knee), each device entry's registers, local bytes, register bound and resident threads per SM (`entry_reports`; "unavailable in this eagle" on releases without it), `device_capacity()`, whether the persist entry exists, the launch mode chosen at N = 10³, 10⁵, 10⁶ and the mode a real run took, and the persist threads per SM eagle picks for the float32 and the float64 oscillator at N = 10⁵, 10⁶ (`probe.txt`) |
| `nsys-grid` | Nsight Systems timelines of eagle auto, persist, band (steps + compaction) and Warp over N = 10³..10⁶, uniform and spread, S = 100 and 1000; kernel time, launch counts and API time as text (`nsys_grid.txt`) |
| `nsys-width` | the same four arms at N = 10⁶, S = 1000 with the stop steps log-uniform over [S/10, S] and [S/1000, S] (the grid holds uniform and the card's [S/100, S]) |
| `ncu-*` | Nsight Compute, one launch per kernel (SpeedOfLight, Occupancy, LaunchStats, WarpStateStats, MemoryWorkloadAnalysis, SchedulerStats) at N = 10⁶ S = 1000 spread (efficiency under thinning), N = 10⁶ S = 1000 uniform (bandwidth baseline) and N = 10⁴ S = 100 spread (latency and launch regime); CSV plus a digest per kernel. Only on compute capability 7.0 and newer |
| `compare` | every comparison case on one matrix, eagle against Warp, in one process (`compare.txt`, `compare.csv`): the oscillator and the RK78 card kernel at N = 10³..10⁶; the oscillator also over S = 100 and 1000 and four stop-step distributions (uniform, the card's spread, narrow `spread:10`, wide `spread:1000`). Every cell runs eagle auto (and records the mode it took), the forced fused one-launch entry, the persistent launch at eagle's own threads per SM and at 320 / 640 / 960 / 1280 (RK78: 256 / 384; a size above the residency limit is skipped and says so), the band path (K-step kernel plus compaction, `every` = eagle's default, 8, 16) and Warp's kernel, each as the median of 3 CUDA-event walls after 2 warm runs (Warp: its own events). Every arm's final state is checked against eagle auto's first (bit for bit when it ran the same kernel through the same launch mode, within the card's tolerance otherwise); a wrong arm is flagged `WRONG` and not timed. The summary lists every cell where auto is slower than Warp and every cell where auto is more than 5 % slower than the fastest eagle path. The oscillator also runs in float32 (`osc32`, `precision=fp32`; every row of the table and CSV is labelled `precision=fp64` or `fp32`) at N = 10⁵, 10⁶, S = 100 and 1000, spread and uniform, with eagle auto, the fused entry, persist at eagle's own size and at 640 / 1024 / 1536 / 2048 and the band path (no Warp arm; each arm's guard is the float32 tolerance against the float32 auto). Warp is the only comparison arm (the cards' torch_compiled and jax_vmap context arms are not in this profile). It replaces the earlier persist and fused sweeps. The persist columns show `wall/chunk`; the persist arms do not count the slot fill (it adds a memset and a sync per run; the chunk sweep measures it in its own arms); `auto mode` is the mode of the last timed run, `modes_per_run` lists every run; eagle auto is timed first and last in each cell (`auto_last`) and the summary lists the cells where the two differ. Every arm records the SM clock (NVML median during its runs) and, where the eagle exposes them, T, c, f, R, k and text_bytes. The literal-table ON/OFF comparison runs only if hawk exposes the option; the probe says whether it does |
| `chunk-sweep` | the forced persist launch over the work-stealing chunk 1, 2, 4, 8, 16, 32 for the float64 and float32 oscillator at S = 100 and 1000, N = 10⁶ spread, at eagle's threads per SM and at twice that (the second is the first thing dropped when the plan is over its time budget), plus RK78 at chunk 8 and 32: wall, the slot fill (counted in its own arms, never in a wall's), and the chunk eagle's rule picks beside the fastest (`chunk_sweep.txt`, `chunk_sweep.csv`). Every forced persist arm passes its chunk explicitly: a forced runner is never refitted, so without a pin it would run the bind-time chunk 32 |
| `crossover` | the fused one-launch entry against the persist launch at N = 3·10⁴, 10⁵, 3·10⁵, S = 100, card spread, with eagle's probe on; records the entry auto settled on after 3 runs (`crossover.txt`) |
| `band` | the band path over the compaction cadence `every` = 16, 32, 64, 128 beside eagle's own pick at N = 10⁵ and 10⁶, S = 1000; R, one compaction group's cost over one full-live step launch, is fitted from the uniform cell's walls over S / every (`band.txt`) |
| `ncu-tsweep` | Nsight Compute achieved warps (with the ratio metrics) of the persist kernel at each persist size (oscillator 384 / 640 / 896 / 1152 / 1536 / 2048, RK78 256 / 384) at N = 10⁶, spread, S = 100 and 1000, plus the float32 oscillator (`precision=fp32`, S = 1000) over the same sizes with the FP32 FMA pipe use beside the achieved warps |
| `cadence-sweep` | the band path (single-step kernel plus compaction) over the compaction cadence `every` = 4, 8, 16, 32 at N = 10⁶, S = 1000, uniform and spread: the graph run gives the wall and the compaction count, an eager run of the same kernels gives the per-kernel sums nsys cannot see inside the graph's conditional loop |
| `ncu-ratio` | Nsight Compute on the persist kernels: threads per issued instruction (about 32 converged, under 16 fragmented), FP64 pipe use, achieved warps and the stall classes, two lane metrics read from ncu alone: `lane_efficiency` (threads per issued instruction over 32) and `fp64_inflation` (FP64 warp-instructions issued over those the sample-steps need at the per-step cost calibrated on the uniform cell of the same kernel and precision; float32 reads the FMA pipe and says so). The kernel's IR operation count (`fp64_ops_ir`, which eagle's chunk rule reads) is printed for reference and never divided into either, for the oscillator (uniform / spread × S = 100 / 1000) and the RK78 card kernel at N = 10⁶ |
| `ncu-chunk` | the chunk sweep's cells under Nsight Compute (eagle's threads per SM): the ratio metrics, `lane_efficiency` and the inflation of the arithmetic pipe's instruction count |
| `ncu-rk78-small` | the RK78 card at N = 10⁴ (critical-path regime) with auto, the fused entry and persist: instructions and FP64 pipe instructions, warp cycles per issued instruction, achieved warps, and the static SASS length of each entry |
| `ncu-band` | the band path's step kernel and its three compaction kernels (predicate, scan, scatter) under Nsight Compute, run as an eager loop because a graph's conditional loop node cannot be profiled |

Needs `nsys` and `ncu` (Nsight Systems and Nsight Compute; found on `PATH` or under `/usr/local/cuda*/bin`; a
missing one skips its stages) and exactly one visible GPU (set `CUDA_VISIBLE_DEVICES` on a multi-GPU machine). GPU
performance counters are usually restricted to administrators: when they are, the ncu stages run under `sudo -n`
(passwordless sudo, as on Google Cloud VMs); without it they are skipped with that reason, and nothing is ever
prompted or changed system-wide.

Every stage runs under its own time cap and a failing one never stops the next; the tarball (reports, raw
`.nsys-rep` / `.ncu-rep` under 200 MB, `status.txt` listing each stage as ok, failed or skipped, `facts.txt` with
versions and the clocks before and after) is written on every exit, Ctrl-C and SIGTERM included. Re-running the
same command after a preemption skips the finished stages (`--fresh` starts over); `--deadline-min` (default 50)
stops new stages from starting; `--only compare,cadence-sweep` runs just those stages (a name or a prefix such as `ncu`).

### The SM clock

A cloud GPU's SM clock moves by tens of percent between arms, more than most of the differences the tables are read for.
`profile.sh` therefore locks it for the whole run (`--lock-clock auto|MHz|off`; `auto` is the device maximum) with
`sudo -n nvidia-smi -lgc`, and releases it on every exit, interrupts included. Without passwordless sudo the clock is
left alone and every wall is normalised to the device maximum clock (wall × sampled MHz ÷ maximum MHz; the CSV keeps the
raw wall). Every table prints which of the two applies and the clock NVML reads.

## Time and cost

On the Quadro P2000 development box, `bootstrap.sh` took about 2 min (cpu profile)
and 4 min (gpu profile), and `--preset smoke` 1.5 min for the CPU cards and 3 min
for the GPU cards; it checks the whole chain. The full matrix
(N = 10³…10⁶, every pass) takes hours per card on a development GPU: the
timing pass is the short part, the memory and compile passes (a fresh process
per arm, size and configuration) and the optional Nsight Systems pass dominate.
`--preset short` (N ≤ 10⁵) gives a first look at a fraction of that. On vast.ai
a consumer GPU rents for roughly $0.2–0.5/h and an A100/H100 for $1–3/h at the
time of writing; Kaggle is free.

## The FP64 class caveat

Every card runs in float64. GPUs differ by more than an order of magnitude in
FP64 rate relative to FP32: data-centre parts (P100, V100, A100, H100) run FP64
at 1/2 of FP32, consumer and workstation parts (T4, P2000, RTX, L4, Ada) at
1/32 to 1/64. Compute-bound arms move with that rate, launch- and memory-bound
arms do not, so the crossovers between arms shift between the two classes. Compare
cards within a class; the card records its device's peak FP64 rate and bandwidth.

## The pinned environments

| Lock | Pins | CUDA floor |
|---|---|---|
| `toolchain-linux-64.lock` | Python 3.12, GCC 14, CMake, Ninja, nvcc 12.9 (conda-forge, explicit URLs + md5) | builds only; shared by CUDA 12 and 12.6; the plugin's cubins load on driver ≥ 525.60 |
| `toolchain-linux-aarch64.lock` | the same toolchain for 64-bit ARM (`make_locks.py toolchain --platform linux-aarch64`); `bootstrap.sh` picks it, and the matching micromamba, from `uname -m` | as above |
| `cards-cpu-linux-aarch64.txt` | the `cards-cpu.txt` packages for 64-bit ARM (`make_locks.py pip cpu --platform linux-aarch64`); `bootstrap.sh` picks it from `uname -m`. There are no aarch64 GPU locks yet, and `--from-pypi` on aarch64 builds raptor-eagle / raptor-hawk from their sdists | none |
| `cards-gpu.txt`, `pypi-gpu.txt` | NumPy, CuPy, PyTorch (cu129), JAX (cuda12), Warp, … (pip, one sha256 each) | CUDA 12.9 runtime libraries (NVRTC included); driver ≥ 575.51; compute capability ≥ 7.5 (the cu129 PyTorch wheel carries sm_75–sm_120) |
| `cards-gpu-cuda126.txt`, `pypi-gpu-cuda126.txt` | the same packages with PyTorch cu126 and the 12.6 libraries (the previous `cards-gpu.txt`) | CUDA 12.6 runtime libraries; driver ≥ 525.60.13; compute capability ≥ 6.0: pre-Turing (sm_60–sm_70), use `--cuda 12.6` |
| `cards-cpu.txt` | NumPy, Numba, PyTorch (CPU), JAX (CPU), … | none |
| `toolchain-linux-64-cuda13.lock` | Python 3.12, GCC 14, CMake, Ninja, nvcc 13.4 (conda-forge, explicit URLs + md5) | builds only; driver ≥ 580, compute capability ≥ 7.5 (CUDA 13 drops offline compilation below sm_75) |
| `cards-gpu-cuda13.txt`, `pypi-gpu-cuda13.txt` | NumPy, CuPy (cuda13x, as eagle's dependency), Warp (CUDA 13.4 wheel), cuda-bindings 13, nvidia-ml-py; no PyTorch, no JAX (pip, one sha256 each) | CUDA 13.4 runtime libraries; driver ≥ 580; compute capability ≥ 7.5 |

### CUDA 12.9 and 12.6

The default `--cuda 12` profile runs one CUDA 12.9 runtime on every GPU from Turing (sm_75) to Blackwell
(sm_120, RTX PRO 6000): PyTorch cu129 and the `nvidia-*-cu12` 12.9 libraries, NVRTC included. Driver ≥ 575.51
(an older CUDA 12 driver may work through minor-version compatibility; only 575 and later is tested). The cu129
PyTorch wheel has no sm_60/61/70 code, so for a pre-Turing GPU (sm_60–sm_70) use `bootstrap.sh --cuda 12.6`: the
previous cu126 locks (`*-cuda126`), an environment of its own (`envs/cards-gpu-cuda126`, `env-gpu-cuda126.sh`),
driver ≥ 525.60.13. Its lock comes from `locks/make_locks.py pip gpu --cuda 12.6`.

### CUDA 13

`bootstrap.sh --cuda 13` (or `REPRO_CUDA=13`; default 12) builds the CUDA 13 profile from the `*-cuda13`
locks into its own environment (`envs/cards-gpu-cuda13`, env script `env-gpu-cuda13.sh`); the CPU profile ignores
the switch. Run `profile.sh --cuda 13` to profile with it (it reads `env-gpu-cuda13.sh` and reports the CUDA 13
package names). An unknown value is refused. The locks come from `locks/make_locks.py toolchain --cuda 13` and
`locks/make_locks.py pip gpu --cuda 13`. The CUDA 13 profile carries eagle, hawk and Warp only: the published
cards' PyTorch and JAX arms need the CUDA 12 profile.

### Comparison arms of `profile.sh`

`profile.sh` compares eagle with Warp alone, on CUDA 12 and 13: the cards' `torch_compiled` and `jax_vmap` context
arms are not run (they are not in this profile; nothing is reported as failed). `run_cards.sh` and the cards keep every arm.

They pin the package versions the committed cards ran with. The CUDA 12.6 locks have been run
end to end on driver 560.35, compute capability 6.1; the CUDA 12.9 locks are new and have not yet been run on a GPU
here. Some builds differ (the committed GPU cards used conda-forge's CUDA 12.9 PyTorch, the lock uses the pip wheel). `nsys` (Nsight Systems) is
optional: without it the cards' kernel-only times are `null`. To bump a version,
edit the `.in` file and run `locks/make_locks.py toolchain` or
`locks/make_locks.py pip gpu|cpu`.
