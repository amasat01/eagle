# Reproduce the performance cards

eagle publishes four cards, each written by its own script: the RK4 and RK7(8)
**GPU** cards (`perf_card/perf_card.py`, `rk78_card/rk78_card.py`) and their
**CPU** twins (`perf_card/cpu_card.py`, `rk78_card/cpu_rk78_card.py`). The
scripts here set up any Linux x86_64 machine to run them with the same pinned
software and write the cards for that machine.

| File | What it does |
|---|---|
| `bootstrap.sh` | pinned toolchain + Python environment, the four repositories (aether, raptor, eagle, hawk) at their release tags, eagle's `_core` for the GPUs present (CPU-only build without one), hawk, an import smoke test and a machine fact sheet |
| `run_cards.sh` | runs the cards (`--which gpu\|cpu\|all`, `--preset smoke\|short\|full`) into `cards/<machine-slug>/` |
| `passes.py` | runs a GPU card one pass at a time (used by `run_cards.sh --split`) |
| `kaggle.ipynb` | a thin notebook calling the two scripts |
| `locks/` | the pinned environments and `make_locks.py`, which regenerates them |

## Your machine, or a rented one (vast.ai)

Needs: Linux x86_64 with glibc ≥ 2.28, `git`, `curl`, ~25 GB of disk, and for
the GPU cards an NVIDIA driver ≥ 525.60 (no CUDA toolkit: the environment brings
its own). No root. On vast.ai any CUDA 12 image works, e.g.
`nvidia/cuda:12.6.3-devel-ubuntu22.04`.

```bash
git clone --depth 1 --branch v0.4.0 https://github.com/amasat01/eagle.git
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

## Kaggle

Open `kaggle.ipynb` in a Kaggle notebook, choose the **GPU T4 x2**
accelerator, turn Internet on, and run all cells. A session is capped at 12 h:
the run is split into resumable steps (`run_cards.sh --split`) and stops before
the cap. To continue, save the version, attach its output to a new session and
set `RESUME_FROM`. Kaggle's 4 shared vCPUs make its CPU cards a different
machine class from the 8-thread arms the CPU cards time, so run the GPU cards there.

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
