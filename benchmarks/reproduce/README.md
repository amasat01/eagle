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
| `profile.sh`, `gpu_profile.py` | the profiling kit: decision probe, Nsight Systems and Nsight Compute captures, one tarball (see below) |
| `kaggle.ipynb` | a thin notebook calling the two scripts |
| `locks/` | the pinned environments and `make_locks.py`, which regenerates them |

## Your machine, or a rented one (vast.ai)

Needs: Linux x86_64 with glibc ≥ 2.28, `git`, `curl`, ~25 GB of disk, and for
the GPU cards an NVIDIA driver ≥ 525.60 (no CUDA toolkit: the environment brings
its own). No root. On vast.ai any CUDA 12 image works, e.g.
`nvidia/cuda:12.6.3-devel-ubuntu22.04`.

```bash
git clone --depth 1 --branch v0.5.1 https://github.com/amasat01/eagle.git
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

## Google Cloud (or any VM with an NVIDIA driver)

Needs Python >= 3.10 with `venv` and its headers, `git`, a C++ compiler (hawk compiles each kernel the
first time it runs) and a driver >= 525.60.13; no conda or CUDA toolkit. On Debian or
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

## Profiling (nsys + ncu)

`profile.sh` answers *why* a launch mode or a kernel is fast or slow on this GPU, for the cards' kernel.
It runs after `bootstrap.sh` and the cards, unattended, in about 30 minutes on an A100 (`--quick`: under 8):

```bash
eagle/benchmarks/reproduce/profile.sh --env ~/.cache/raptor-repro/env-gpu.sh
# -> ~/.cache/raptor-repro/profiles/<gpu>--<cpu>/profile-<gpu>--<cpu>.tgz  (the last line it prints)
```

| Stage | What it captures |
|---|---|
| `probe` | what eagle reads from the device (SMs, registers, resident threads, the issue knee), each device entry's registers, local bytes, register bound and resident threads per SM (`entry_reports`; "unavailable in this eagle" on releases without it), `device_capacity()`, whether the persist entry exists, the launch mode chosen at N = 10³, 10⁵, 10⁶ and the mode a real run took (`probe.txt`) |
| `nsys-grid` | Nsight Systems timelines of eagle auto, persist, band (steps + compaction) and Warp over N = 10³..10⁶, uniform and spread, S = 100 and 1000; kernel time, launch counts and API time as text (`nsys_grid.txt`) |
| `nsys-width` | the same four arms at N = 10⁶, S = 1000 with the stop steps log-uniform over [S/10, S] and [S/1000, S] (the grid holds uniform and the card's [S/100, S]) |
| `ncu-*` | Nsight Compute, one launch per kernel (SpeedOfLight, Occupancy, LaunchStats, WarpStateStats, MemoryWorkloadAnalysis, SchedulerStats) at N = 10⁶ S = 1000 spread (efficiency under thinning), N = 10⁶ S = 1000 uniform (bandwidth baseline) and N = 10⁴ S = 100 spread (latency and launch regime); CSV plus a digest per kernel. Only on compute capability 7.0 and newer |
| `persist-sweep` | wall time (CUDA events) and per-lane utilisation of the persistent launch over persist threads per SM (oscillator 320 / 640 / 960 / 1280, RK78 256 / 384, plus eagle's own pick) at N = 10⁶, S = 100 and 1000, uniform and spread; the RK78 card has one workload, so one row per size. `tps_launched` is what really launched (`persist_sweep.txt`). eagle has no knob for the size, so the kit replaces the sizing function for the length of one cell and says so |
| `fused-sweep` | the same points through the fused one-launch entry (`_fast_mode="fused_one"`), for the crossover with the persistent launch |
| `cadence-sweep` | the band path (single-step kernel plus compaction) over the compaction cadence `every` = 4, 8, 16, 32 at N = 10⁶, S = 1000, uniform and spread: the graph run gives the wall and the compaction count, an eager run of the same kernels gives the per-kernel sums nsys cannot see inside the graph's conditional loop |
| `ncu-ratio` | Nsight Compute on the persist kernels: threads per issued instruction (about 32 converged, under 16 fragmented), FP64 pipe use, achieved warps and the stall classes, for the oscillator (uniform / spread × S = 100 / 1000) and the RK78 card kernel at N = 10⁶ |
| `ncu-band` | the band path's step kernel and its three compaction kernels (predicate, scan, scatter) under Nsight Compute, run as an eager loop because a graph's conditional loop node cannot be profiled |

Needs `nsys` and `ncu` (Nsight Systems and Nsight Compute; found on `PATH` or under `/usr/local/cuda*/bin`; a
missing one skips its stages) and exactly one visible GPU (set `CUDA_VISIBLE_DEVICES` on a multi-GPU machine). GPU
performance counters are usually restricted to administrators: when they are, the ncu stages run under `sudo -n`
(passwordless sudo, as on Google Cloud VMs); without it they are skipped with that reason, and nothing is ever
prompted or changed system-wide.

Every stage runs under its own time cap and a failing one never stops the next; the tarball (reports, raw
`.nsys-rep` / `.ncu-rep` under 200 MB, `status.txt` listing each stage as ok, failed or skipped, `facts.txt` with
versions and the clocks before and after) is written on every exit, Ctrl-C and SIGTERM included. Re-running the
same command after a preemption skips the finished stages (`--fresh` starts over); `--deadline-min` (default 35)
stops new stages from starting.

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
| `toolchain-linux-64.lock` | Python 3.12, GCC 14, CMake, Ninja, nvcc 12.9 (conda-forge, explicit URLs + md5) | builds only; the plugin's cubins load on driver ≥ 525.60 |
| `cards-gpu.txt` | NumPy, CuPy, PyTorch (cu126), JAX (cuda12), Warp, … (pip, one sha256 each) | CUDA 12.6 runtime libraries; driver ≥ 525.60.13; compute capability ≥ 6.0 |
| `cards-cpu.txt` | NumPy, Numba, PyTorch (CPU), JAX (CPU), … | none |

They pin the package versions the committed cards ran with and have been run
end to end on driver 560.35, compute capability 6.1; some builds differ (the committed GPU cards used
conda-forge's CUDA 12.9 PyTorch, the lock uses the cu126 wheel). `nsys` (Nsight Systems) is
optional: without it the cards' kernel-only times are `null`. To bump a version,
edit the `.in` file and run `locks/make_locks.py toolchain` or
`locks/make_locks.py pip gpu|cpu`.
