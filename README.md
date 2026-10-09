<p align="center">
  <img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/brand/glyph_eagle.svg" alt="" height="56">
</p>
<h1 align="center">eagle</h1>
<p align="center">Runs kernels on CPU and GPU, with precise control. Part of the RAPTOR family.</p>

<h3 align="center">eagle: run a million per-sample kernels on the GPU or the CPU</h3>
<p align="center">Each sample stops when it is done. eagle records the loop on the GPU once, replays it,
and skips the samples that already finished.</p>

<p align="center">
  <a href="https://github.com/amasat01/eagle/actions/workflows/ci.yml"><img src="https://github.com/amasat01/eagle/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://amasat01.github.io/eagle/"><img src="https://github.com/amasat01/eagle/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://github.com/amasat01/eagle/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/raptor-eagle/"><img src="https://img.shields.io/pypi/v/raptor-eagle.svg" alt="PyPI"></a>
  <a href="https://doi.org/10.5281/zenodo.23250240"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23250240.svg" alt="DOI"></a>
  <a href="https://www.kaggle.com/kernels/welcome?src=https://github.com/amasat01/eagle/blob/main/docs/content/userguide/quickstart_python.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>
</p>

<p align="center">
  <a href="https://amasat01.github.io/"><b>The RAPTOR family</b></a><br>
  <a href="https://amasat01.github.io/hawk/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_hawk_eagle_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_hawk_eagle_light.svg" alt="hawk" width="430"></picture></a>
  <a href="https://amasat01.github.io/eagle/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_eagle_eagle_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_eagle_eagle_light.svg" alt="eagle" width="430"></picture></a><br>
  <a href="https://amasat01.github.io/aether/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_aether_eagle_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_aether_eagle_light.svg" alt="aether" width="430"></picture></a>
  <a href="https://amasat01.github.io/raptor/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_raptor_eagle_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/ecosystem/ecosystem_card_raptor_eagle_light.svg" alt="raptor" width="430"></picture></a>
</p>

eagle is the execution layer of the [RAPTOR family](https://amasat01.github.io/): it captures and replays kernels
as CUDA graphs to keep work efficient on the GPU, and connects them to the code you already have. See
[where RAPTOR fits](https://amasat01.github.io/where_raptor_fits.html) for the full four-part story.

```python
import cupy as cp

import eagle
import hawk
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel
def oscillator(omega: Scalar, t_end: Param, dt: Param, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    x0, v0 = x, v
    x = x0 + dt * v0
    v = v0 - dt * omega * omega * x0
    t += dt
    terminated = t >= t_end


n = 1_000_000
result = eagle.simulate(
    oscillator,
    omega=cp.linspace(1.0, 3.0, n), t_end=1.0, dt=1e-3,
    x=cp.ones(n), v=cp.zeros(n), t=cp.zeros(n),
    max_steps=10_000,
)
print(result.status, result.steps, result.x[:3])
# finished 1000 [0.54057281 0.54057112 0.54056944]
# (steps counts whole check intervals, so it varies with the launch strategy eagle picks; x does not)
```

NumPy arrays in, and the same call runs on CPU threads.
**[Quickstart](https://amasat01.github.io/eagle/content/userguide/quickstart_python.html)** ·
[Install](#install) · [Performance](#performance) · [C++ API](#c-api)

| **156 ms** | **3.9×–47× faster** | **3.8×–5.2× less GPU memory** |
|:---:|:---:|:---:|
| 1,000,000 RK4 oscillators, each stopping at its own step, up to 1000 steps | than NVIDIA Warp (3.9×), JAX (8.7×), CuPy and PyTorch (47×) — 1,000,000 samples finishing at different times, up to 1000 steps | than CuPy and PyTorch — a million samples in 50 MiB, 1.31× the bare minimum state (CuPy 5.1×, PyTorch 6.9×) |

<sub>Quadro P2000 (development GPU). NVIDIA Warp's per-thread kernel is fastest among the GPU arms in 1 of 12 RK4 cells
(N = 1,000,000, every sample running all 1000 steps: 693 ms vs 699 ms) and ties hawk + eagle at
N = 10,000 on the same batch (7.22 ms each). `eagle.simulate`'s wall time is within 1% of the hand-tuned
version in every cell this card runs but N = 1,000 and 10,000 spread-100; memory at a million spread samples is 50 MiB for `eagle.simulate` against 64 MiB
for NVIDIA Warp. See [every number, every arm](#performance).</sub>

## Threading

Free-threaded CPython (3.13t/3.14t) is supported: the compiled modules declare GIL-free operation and the GIL stays disabled after import. Any number of threads may call module-level functions, build, compile, plan and cache concurrently. Distinct objects may be used from distinct threads without synchronisation. One stateful object (a stream, capture, launcher, graph, composer, plan, pipeline, active set, host kernel or arg block) shared by several threads is memory-safe — its calls serialise and a consumed object raises — but the ORDER of those calls is the caller's responsibility, exactly as for a NumPy array or a CuPy stream. CUDA adds two rules of its own: a stream capture is begun, filled and ended by one thread, and while any capture is open no thread may synchronise the whole device (stream-level synchronisation is fine). GPU routes are supported on free-threaded 3.14; on 3.13t the CPU route runs. CUDA graph capture is per thread: a nested capture on one thread raises, while two threads may capture concurrently.

## Install

```bash
pip install "raptor-eagle[cuda12]"      # or [cuda13]; add [torch] for PyTorch interop
pip install "raptor-hawk[cuda12]"       # the examples below also write kernels with hawk; [cuda13] for CUDA 13
```

> **NVIDIA packages come only through the extras.** `raptor-eagle[cuda12]` / `[cuda13]` pull CuPy (`cupy-cuda12x` /
> `cupy-cuda13x`, with the CUDA headers CuPy compiles against); `raptor-hawk[cuda12]` pulls `cuda-bindings` 12, `nvidia-cuda-nvrtc-cu12` and
> `nvidia-cuda-cccl-cu12`, and `raptor-hawk[cuda13]` pulls `cuda-bindings` 13, `nvidia-cuda-nvrtc` 13 and
> `nvidia-cuda-cccl` 13 (CUDA 13's wheels have no `-cu13` suffix). Without an extra pip installs no NVIDIA package: you get the CPU route, or the GPU route through a CUDA setup you already have. Pick
> the extra matching the CUDA version your driver reports (`nvidia-smi`, top right).

**Platforms:** built and tested on Linux x86_64 only so far (CPython 3.9–3.14, including free-threaded 3.13t and 3.14t), on NVIDIA GPUs from Pascal (Quadro P2000) and Turing (Tesla T4). There are no wheels for macOS, Windows or ARM yet, and WSL2 is untested. `raptor-core` and `aether-dsc` are pure Python and install anywhere.

`raptor-eagle` needs Python 3.9 or newer (CPython 3.9–3.14) and an NVIDIA driver at run time; no `nvcc` or CUDA
toolkit. Free-threaded builds (3.13t, 3.14t) run GIL-free. On 3.13t the `[cuda12]` / `[cuda13]` extras do not resolve (CuPy 14 ships no 3.13t wheel), so eagle runs on the CPU route there; 3.14t has no such limit. The wheel ships GPU code for every NVIDIA architecture from Pascal (sm_60) through Blackwell, plus PTX for newer GPUs. It
brings `raptor-core` along. The C++ library (`find_package(eagle CONFIG REQUIRED)`) is a separate, CMake-only
install; building the Python package from source is covered under [Python package](#python-package) below.

<details>
<summary>C++ library: build, test, install</summary>

Requirements: CMake ≥ 3.20 and a C++23 compiler (GCC ≥ 12 or Clang ≥ 16); CUDA
mode also needs a CUDA toolkit 12.6 or newer (tested with 12.6 and 13.0), whose nvcc accepts host compilers only up to
its own ceiling (GCC 13 for CUDA 12.6 — pass `-DCMAKE_CUDA_HOST_COMPILER=<g++-13>`
if your default is newer). `aether` must already be installed into `PREFIX` (see
aether's README); `PREFIX` is the install directory (inside a conda environment,
use `${CONDA_PREFIX}`).

```bash
# Configure (CUDA/C++ mode, with tests)
cmake -DCMAKE_PREFIX_PATH=${PREFIX} -DEAGLE_BUILD_TESTS=ON -B build .

# Configure (pure C++ mode)
cmake -DCMAKE_PREFIX_PATH=${PREFIX} -DEAGLE_CPP_MODE=ON -DEAGLE_BUILD_TESTS=ON -B build .

# Build
cmake --build build

# Test — the same command CI runs (there is no ctest wiring)
tests/check_gate.sh cuda build/tests/eagle_tests     # or: cpp, for a CPP_MODE build

# Install
cmake --install build --prefix ${PREFIX}
```

The gate checks both the run and the test listing against the committed name list
`tests/expected_tests_<mode>.txt`, so a filtered or empty run cannot pass. After adding or removing tests,
regenerate that list from your build and commit it with the test change (removals also need `--allow-removals`):
`tests/check_gate.sh cuda build/tests/eagle_tests --remint`.

Downstream projects consume it with `find_package(eagle CONFIG REQUIRED)` and link `eagle::eagle`.
</details>

<details>
<summary>Build the Python package from source</summary>

The Python package (`raptor-eagle`, imported as `eagle`) has two compiled
parts: `eagle._core` is plain C++ (g++) and needs no CUDA toolkit, runtime or
driver, while `eagle/libeagle_cuda.so`, the CUDA plugin the core loads on
first use, is built with `nvcc` (pass `-DEAGLE_PYTHON_CUDA_PLUGIN=OFF` to
build the core alone) and at run time needs only the NVIDIA driver — it
bundles no CUDA runtime. The build needs `aether` installed in CUDA mode
(without `-DAETHER_CPP_MODE=ON`) plus eagle's own headers, both in `PREFIX`,
and `raptor` (`pip install raptor-core`, or a clone next to this one: `pip install ../raptor`).

```bash
cmake -DCMAKE_PREFIX_PATH=${PREFIX} -B build-hdr . && cmake --install build-hdr --prefix ${PREFIX}
pip install raptor-core         # or a clone of the raptor repository: pip install ../raptor
export CMAKE_PREFIX_PATH=${PREFIX}
# append ;-DCMAKE_CUDA_HOST_COMPILER=<g++-13> if your default compiler is newer than nvcc accepts
export SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=<your GPU's arch, e.g. 70>"
pip install -e ./python
pytest python/tests -m "not gpu and not interop_matrix"   # drop the -m filter on a GPU machine with cupy and torch
```
</details>

## C++ API

```cpp
#include <eagle/eagle.h>

__global__ void addOne(int* buf, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) buf[tid] += 1;
}

// Record a launch once, replay the captured graph many times.
eagle::cuda::Stream stream;
eagle::cuda::Graph graph;
graph.stream(stream.cuda());

eagle::cuda::StreamCapturer capturer(stream.cuda());
capturer.begin();
addOne<<<1, 64, 0, stream.cuda()>>>(buf, 64);
graph.addNode(capturer.end());

eagle::cuda::Launcher launcher = graph.launcher();
launcher.launch();          // cudaGraphLaunch — no per-call kernel-config overhead
launcher.synchronize();
```

**EAGLE** — *Extensible Adaptive Graph Launch Engine* — is a header-only C++23/CUDA library providing a
parallel-dispatch substrate for GPU batch workloads. It bundles the reusable launch machinery in one place:

- **CUDA-graph capture & launch** — record a stream of kernel launches once
  (`StreamCapturer`), assemble them into a dependency graph (`Graph`), and
  replay via an instantiated `Launcher` with microsecond-class per-launch
  re-tuning (`setLogicalSize`).
- **Stream primitives** — RAII `Stream`, `Event`, and host-callback wrappers.
- **OpenMP + SIMD host dispatch** — `eagle::cpu::Host` packet-batched launchers with
  masked / flag-aware variants for the pure-C++ path.
- **Scan / scatter / compaction** — `filtering::Scanner`, `filtering::Slice`,
  and the prefix-sum + scatter kernels behind stream compaction.
- **Reduction** — `eagle::cuda::Reduction` (multi-level GPU) and
  `eagle::cpu::Reduction` (cache-padded OpenMP).
- **PyTorch bridge** — kernels plug into `loss.backward()` and
  `torch.autograd.forward_ad`, so both reverse-mode and forward-mode
  derivatives run through the same dual-mode engine.

Like its sibling libraries it is **dual-mode**: the same API compiles as CUDA or as pure C++23 (OpenMP), selected
by `EAGLE_CPP_MODE`. The C++ core depends only on `aether`; the Python package (`raptor-eagle`) also depends on
`raptor`, whose schema constants it re-exports.

## Performance

eagle's committed [performance card](https://github.com/amasat01/eagle/tree/main/benchmarks/perf_card) runs 15
ways of integrating a batch of damped harmonic oscillators (fixed-step RK4, float64, each sample stopping at its
own step count) from 1,000 to 1,000,000 samples: `eagle.simulate` (the arm above), the CUDA-graph device loop
alone, the same loop with active-set compaction, the same loop with compaction plus an occasional reorder, two
K-fused-steps variants, the automatic-policy and persistent-launch arms, a plain eager launch loop, a CuPy array implementation, a PyTorch masked implementation, a
PyTorch CUDA-graph implementation, a JAX `jit(vmap(while_loop))` implementation, an NVIDIA Warp per-thread kernel,
and a CPU OpenMP arm (no device or transfers needed, and the correctness reference the rest are checked against).
The five-arm table below is the headline cut (both distributions, N = 1,000,000); the
[full performance page](https://amasat01.github.io/eagle/content/performance.html) has every arm, every N.

<!-- stale-prose-gate:
benchmarks/perf_card/card_quadro-p2000.json: d37837fb31567858848e5f1830f19431
-->

<!-- cards:begin -->
| Distribution (N = 1,000,000) | Arm | wall (range) | sample·steps/s | useful FLOP/s (% of peak) |
|---|---|---:|---:|---:|
| Spread (log-uniform stop steps, up to 1000) | eagle graph (device loop) | 625 ms (625 ms–626 ms) | 3.46e8 | 1.52e10 (16 %) |
| Spread (log-uniform stop steps, up to 1000) | eagle graph + compaction | 415 ms (415 ms–415 ms) | 5.21e8 | 2.29e10 (24 %) |
| Spread (log-uniform stop steps, up to 1000) | eagle graph + compaction + reorder | 279 ms (279 ms–279 ms) | 7.74e8 | 3.41e10 (36 %) |
| Spread (log-uniform stop steps, up to 1000) | CuPy masked | 7.38 s (7.38 s–7.38 s) | 2.93e7 | 1.29e9 (1.4 %) |
| Spread (log-uniform stop steps, up to 1000) | CPU OpenMP | 149 ms (148 ms–150 ms) | 1.45e9 | 6.39e10 (–) |
| Uniform (every sample runs 1000 steps) | eagle graph (device loop) | 703 ms (703 ms–703 ms) | 1.42e9 | 6.26e10 (66 %) |
| Uniform (every sample runs 1000 steps) | eagle graph + compaction | 862 ms (862 ms–862 ms) | 1.16e9 | 5.11e10 (54 %) |
| Uniform (every sample runs 1000 steps) | eagle graph + compaction + reorder | 863 ms (863 ms–863 ms) | 1.16e9 | 5.10e10 (54 %) |
| Uniform (every sample runs 1000 steps) | CuPy masked | 7.38 s (7.38 s–7.38 s) | 1.35e8 | 5.96e9 (6.3 %) |
| Uniform (every sample runs 1000 steps) | CPU OpenMP | 191 ms (190 ms–191 ms) | 5.25e9 | 2.31e11 (–) |

Measured on:

- [Quadro P2000](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-quadro-p2000)
- [Tesla T4](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-tesla-t4)
- [Intel Xeon W-2125 CPU @ 4.00GHz](https://amasat01.github.io/eagle/content/performance.html#perf-card-cpu-intel-xeon-w-2125)
<!-- cards:end -->

Figures are written by `tools/sync_readme_cards.py` from the card JSON (the
same numbers `card_quadro-p2000.md` renders): wall time and rates rounded to 3
significant figures, percent-of-peak to 2; a dash means that arm has no
FP64-peak baseline recorded (CPU). Re-run the script after a card is re-run or
a new `perf_card` device is added; `python/tests/test_readme_cards.py` fails
if the committed block has drifted from a fresh render.

**Honest wins, not buried:** on a dense batch, where every sample runs all 1000 steps, NVIDIA Warp's per-thread kernel is
level with `eagle.simulate` (7.22 ms vs 7.22 ms at N = 10,000;
693 ms vs 699 ms at N = 1,000,000), and at a fixed overhead of 64 samples it takes 135 µs against `eagle.simulate`'s 117 µs. The CPU
OpenMP arm, which needs no GPU, is faster than every GPU arm on the dense batch from N = 1,000 up (419 µs against 893 µs
at N = 1,000; 191 ms against 699 ms at N = 1,000,000) and on the spread-1000 batch at N = 1,000 (437 µs vs 884 µs);
this is an FP64-weak development card. JAX compiles cold faster than hawk + eagle (about 246 ms, against
`eagle.simulate`'s 1.12 s; NVIDIA Warp's cold compile is 1.49 s), and hawk + eagle's own warm recompile is 109 ms. These are
complementary tools: Warp and hawk + eagle both skip nothing — the difference is who writes the per-thread loop and
who decides when to compact.

Compaction's advantage tracks how early the batch thins: on the spread
distribution it cuts the wall time by 1.51× at N = 1,000,000 (625 ms → 415
ms), while on the uniform distribution, where no sample finishes early,
compaction instead costs 1.23× the plain graph's time (862 ms vs 703 ms) —
plain `hawk + eagle graph` is the better fit when nothing thins. The eager launch
loop trails the plain graph by about 1.24-1.25× at this N in both
distributions (774 ms / 875 ms vs 625 ms / 703 ms) — a per-step host read
still costs something, even once the kernel runs long enough that launch
overhead alone would not matter. CuPy masked reaches the highest fraction of
peak memory bandwidth of the arms in this table (up to 87% at N = 1,000,000 uniform,
recorded in the card) — the better fit when array-style code matters more
than wall time on an already bandwidth-bound batch.

The reorder arm is compaction's index map plus an occasional physical
reorder, opt-in through `eagle.ActiveSet(mask, reorder=theta)`. It pays off
from about N = 100,000 on the spread distribution, where the batch keeps
thinning: 45.7 ms → 34.1 ms at N = 100,000 (1.34×), 415 ms → 279 ms at N =
1,000,000 (1.49×). Below that it costs instead — about 2-4% longer than
compaction alone at N = 1,000 and 10,000 — and on the uniform distribution,
where nothing needs reordering, it costs at most about 4% more than
compaction alone at any N (862 ms vs 863 ms at N = 1,000,000; 95.1 ms vs
95.4 ms at N = 100,000).

CPU, no GPU at all: the same kernel on 8 OpenMP threads is 149 ms (spread) / 191 ms (uniform) at N = 1,000,000 —
faster than the best GPU arm on the spread batch (156 ms, `eagle` auto) and faster than every GPU arm on the uniform batch (the best of them, NVIDIA Warp, takes 693 ms),
on this FP64-weak development card; on the harder RK7(8) family (adaptive
orbits, heavier per-step work) the CPU arm is also about twice as fast as the best GPU arm on this same card (7.15 s vs 13.0 s at
N = 1,000,000) — see
[where RAPTOR fits](https://amasat01.github.io/where_raptor_fits.html) for that comparison.

**Hardware caveat:** measured on a Quadro P2000, a development card (8 SMs,
9.48e10 FLOP/s FP64 peak at its rated clock, from the card). The method
carries over unchanged to other GPUs; the absolute numbers, and the
crossover points between arms, will differ on data-centre GPUs.

Full write-up, method and the per-N tables: [the performance
page](https://amasat01.github.io/eagle/content/performance.html). Raw
numbers:
[`card_quadro-p2000.json`](https://github.com/amasat01/eagle/blob/main/benchmarks/perf_card/card_quadro-p2000.json) /
[`card_quadro-p2000.md`](https://github.com/amasat01/eagle/blob/main/benchmarks/perf_card/card_quadro-p2000.md).

---
<p align="center">
  <b>hawk</b> write it ·
  <b>eagle</b> run it (this repo) ·
  <b>aether</b> the numerics core ·
  <b>raptor</b> the shared contracts —
  <a href="https://amasat01.github.io/">the RAPTOR family</a>
</p>

Apache-2.0 (see [`LICENSE`](https://github.com/amasat01/eagle/blob/main/LICENSE) and
[`NOTICE`](https://github.com/amasat01/eagle/blob/main/NOTICE)) · cite via "Cite this repository"
(`CITATION.cff`; every tagged release is archived on Zenodo: [doi:10.5281/zenodo.23250240](https://doi.org/10.5281/zenodo.23250240)) · built to
make GPU computing accessible on modest hardware, for research and education. Collaboration is the point, and a
citation is the currency — get in touch.
