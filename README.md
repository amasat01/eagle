<p align="center">
  <img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/brand/glyph_eagle.svg" alt="" height="56">
</p>
<h1 align="center">eagle</h1>
<p align="center">Runs kernels on CPU and GPU, with precise control. Part of the RAPTOR family.</p>
<p align="center">Linux and macOS wheels; on Windows, use WSL2.</p>

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

Wheels for Linux x86_64 and aarch64 and macOS arm64 (Python 3.9-3.15); the GPU plugin is in the x86_64 wheels only. **On Windows, use WSL2** (the Linux wheels), or open the notebook on Colab or Kaggle (the Kaggle badge above).

```bash
pip install raptor-eagle raptor-hawk                           # CPU threads, no NVIDIA packages
pip install "raptor-eagle[cuda12]" "raptor-hawk[cuda12]"       # NVIDIA GPU ([cuda13] for a CUDA 13 driver)
```

A kernel, run over an array, then two kernels as one graph (NumPy arrays, so it runs on the CPU):

```python
import numpy as np
import eagle
import hawk
from hawk import Param, Scalar


@hawk.kernel
def scale(x: Scalar, a: Param, b: Param) -> Scalar:
    return a * x + b


@hawk.kernel
def square(v: Scalar) -> Scalar:
    return v * v


x = np.arange(4.0)
print(eagle.deploy(scale).run(x=x, a=2.0, b=1.0))        # [1. 3. 5. 7.]


@eagle.Graph.builder
def chain(x):
    g = eagle.Graph()
    g.result = square(scale(x, 2.0, 1.0))
    return g


g = chain(x)                                             # build: records, runs nothing
print(g())                                               # [ 1.  9. 25. 49.]
```

The same lines with CuPy arrays make a CUDA graph: the arrays decide where it runs.
The [graph model](https://amasat01.github.io/eagle/content/userguide/graph_model.html)
has branches, device-side `if` and `while`, `bind` and gradients.

On a GPU, `eagle.simulate` runs a million per-sample loops, each stopping at its own step:

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

With NumPy arrays the same call runs on CPU threads (`steps` then prints 1024, not 1000: it counts whole check intervals, which differ by route).

Errors: `no CUDA GPU visible` means there is no GPU (or `CUDA_VISIBLE_DEVICES` hides it); `cudaErrorInsufficientDriver` means the
installed extra does not match the driver, so install `raptor-eagle[cuda12]` for a CUDA 12 driver or `[cuda13]` for CUDA 13.

With the `[torch]` extra, a hawk kernel works inside PyTorch autograd:

```python
import torch
import hawk
from hawk import Mutable, Scalar
from eagle.frameworks import torch as eagle_torch


@hawk.kernel
def decay(k: Scalar, x: Scalar, x_next: Mutable[Scalar]):
    x_next = x * (1.0 - k)


step = eagle_torch.function(decay, wrt=("k",))
k = torch.full((1000,), 0.5, dtype=torch.float64, requires_grad=True)
x_next = step(k, torch.ones(1000, dtype=torch.float64))
x_next.sum().backward()
print(x_next[0].item(), k.grad[0].item())     # 0.5 -1.0
```

**[Quickstart](https://amasat01.github.io/eagle/content/userguide/quickstart_python.html)** ·
[Install](#install) · [Performance](#performance) · [C++ API](#c-api)

<!-- hero:begin -->
| **613 µs** at N = 1,000,000, 199 µs at N = 100,000 | **2.5×–600× faster** at N = 1,000,000, 1.1×–1700× at N = 100,000 | **4.8×–6.0× less GPU memory** at N = 1,000,000, 5.5×–42× at N = 100,000 |
|:---:|:---:|:---:|
| RK4 oscillators, each stopping at its own step, up to 1000 steps | than NVIDIA Warp (2.5×), JAX (76×), CuPy (600×), PyTorch with torch.compile (130×) at N = 1,000,000; Warp is level or ahead when every sample runs all its steps | than CuPy and PyTorch with torch.compile: 40 MiB for N = 1,000,000, 1.1× the bare minimum state |

<sub>NVIDIA H100 80GB HBM3 (card of 2026-10-13). NVIDIA Warp's per-thread kernel is the fastest GPU arm in 2 of 12 RK4 cells; on the dense batch (every sample running all 1000 steps) at N = 1,000,000 it takes 1.71 ms and `eagle.simulate` 1.75 ms. See [every number, every arm](#performance).</sub>
<!-- hero:end -->

## What's new in 0.7.1

- Planes are allocated through the `aether` arrays from `aether-dsc`, now a base dependency: the GPU needs no extra package.
- The free of a plane is queued on its own stream and deferred during a graph capture.
- `bound.clear(*names)` zeroes a keyed output and its compensation term, also on a split kernel.
- hawk's new features reach eagle through the same kernels: per-lane linear algebra, broadcasting at `eagle.deploy`, a run-time first axis and `if ...: break`.
- Random kernels give different numbers than 0.7.0, because aether's random stream changed.

Full list in the [changelog](CHANGELOG.md).

## Gradients, certified

JAX (`grad`, `jit`, `vmap`, `jacfwd`, `jacrev`), PyTorch (`backward`, `torch.func`) and NVIDIA Warp (`wp.Tape`) differentiate through an eagle kernel, and each is certified against the same law written in that framework alone. NumPy and CuPy have no autodiff: pass your own `vjps=`. The [interoperability page](https://amasat01.github.io/eagle/content/interop.html) lists the certified rows, with tutorials for [JAX](https://amasat01.github.io/eagle/content/userguide/tutorials/jax_vmap_and_jacobians.html), [PyTorch](https://amasat01.github.io/eagle/content/userguide/tutorials/03_torch_training.html), [Warp](https://amasat01.github.io/eagle/content/userguide/tutorials/warp_kernel_in_a_graph.html) and [whole graphs](https://amasat01.github.io/eagle/content/userguide/tutorials/10_gradients_through_a_graph.html). To use your own CUDA graph inside an eagle graph, hand it over as a `child` node (see the Warp tutorial); eagle does not launch inside your capture.

## Threading

Free-threaded CPython (3.13t/3.14t) is supported: the compiled modules declare GIL-free operation and the GIL stays disabled after import. Any number of threads may call module-level functions, build, compile, plan and cache concurrently. Distinct objects may be used from distinct threads without synchronisation. One stateful object (a stream, capture, launcher, graph, composer, plan, pipeline, active set, host kernel or arg block) shared by several threads is memory-safe — its calls serialise and a consumed object raises — but the ORDER of those calls is the caller's responsibility, exactly as for a NumPy array or a CuPy stream. CUDA adds two rules of its own: a stream capture is begun, filled and ended by one thread, and while any capture is open no thread may synchronise the whole device (stream-level synchronisation is fine). GPU routes are supported on free-threaded 3.14; on 3.13t the CPU route runs. CUDA graph capture is per thread: a nested capture on one thread raises, while two threads may capture concurrently.

## Install

```bash
pip install raptor-eagle raptor-hawk                      # CPU threads: no NVIDIA packages, nothing else to install
pip install "raptor-eagle[cuda12]" "raptor-hawk[cuda12]"  # NVIDIA GPU; [cuda13] for CUDA 13; add [torch] for PyTorch interop
```

Extras, platforms, driver and Python versions: [installation](https://amasat01.github.io/eagle/content/userguide/installation.html). Building from source: [build from source](https://amasat01.github.io/build_from_source.html). The C++ library is a separate CMake install; the Python package from source is under [Python package](#python-package) below.

<details>
<summary>C++ library: build, test, install</summary>

Requirements: CMake ≥ 3.20 and a C++23 compiler (GCC ≥ 12 or Clang ≥ 16); CUDA
mode also needs a CUDA toolkit 12.6 or newer (the performance cards record the toolkit each was built with), whose nvcc accepts host compilers only up to
its own ceiling (for example GCC 13 for CUDA 12.6; the ceiling differs per toolkit — pass `-DCMAKE_CUDA_HOST_COMPILER=<g++-13>`
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
graph.addNode(eagle::cuda::CapturedGraph{ capturer.end() });

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
- **PyTorch, JAX and Warp bridges** — kernels plug into `loss.backward()`,
  `torch.autograd.forward_ad`, `jax.grad`/`jacfwd` and `wp.Tape`, so reverse-mode and
  forward-mode derivatives run through the same dual-mode engine.
- **CPU deployment** — `eagle::cpu::PluginRegistry` loads host plugins (float64 and
  float32) from a manifest; see [embed a kernel](https://amasat01.github.io/eagle/content/userguide/cpp/c4_embed_a_kernel.html).

Like its sibling libraries it is **dual-mode**: the same API compiles as CUDA or as pure C++23 (OpenMP), selected
by `EAGLE_CPP_MODE`. The C++ core depends only on `aether`; the Python package (`raptor-eagle`) also depends on
`raptor`, whose schema constants it re-exports.

## Performance

<!--ctx card=rk4 dev=h100 n=1e6 dist=spread steps=1000-->
eagle's committed [performance card](https://github.com/amasat01/eagle/tree/main/benchmarks/perf_card) compares many
ways of integrating a batch of damped harmonic oscillators (fixed-step RK4, float64, each sample stopping at its
own step count) from 1,000 to 1,000,000 samples: `eagle.simulate` (the arm above), the CUDA-graph device loop
alone, the same loop with active-set compaction, the same loop with compaction plus an occasional reorder, two
K-fused-steps variants, the automatic-policy and persistent-launch arms, a plain eager launch loop, a CuPy array
implementation, PyTorch masked, CUDA-graph and `torch.compile` implementations, a JAX `jit(vmap(while_loop))`
implementation, an NVIDIA Warp per-thread kernel, and a CPU OpenMP arm (no device or transfers needed, and the
correctness reference the rest are checked against). The tables below show the NVIDIA H100 card; the five-arm
table is the headline cut (both distributions, N = 1,000,000), and the
[full performance page](https://amasat01.github.io/eagle/content/performance.html) has every arm, every N.

[Where the performance comes from](https://amasat01.github.io/eagle/content/performance/mechanism.html) explains why eagle is ahead where samples finish at different times and level where they do not.

<p align="center"><a href="https://amasat01.github.io/eagle/content/performance.html"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/perf/readme_gpu-dark.svg"><img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/perf/readme_gpu.svg" alt="Wall time per tool for a million RK4 oscillators on the default GPU: eagle.simulate, NVIDIA Warp, JAX, CuPy and PyTorch; the numbers are on the performance page." width="732"></picture></a></p>

The chart shows the H100; [the performance page](https://amasat01.github.io/eagle/content/performance.html) has a tab for every device and CPU measured.

<!-- stale-prose-gate:
benchmarks/perf_card/card_nvidia-h100-80gb-hbm3.json: 86570d9dd9f301b54ae3e50a40ecdfc4
benchmarks/perf_card/card_quadro-p2000.json: e1d57a6c7d57c7cd7efd7bd8e97fcdcf
benchmarks/rk78_card/card_nvidia-h100-80gb-hbm3.json: 4a6847fb1f7815d64d95de8c8b48a527
benchmarks/rk78_card/cpu_card_amd-epyc-9b45.json: 1ef4eda3e4f12f87d3a84585424f1c57
benchmarks/perf_card/cpu_card_aarch64.json: 778ce3a02f2c8c80dd7d23d942630da0
benchmarks/perf_card/cpu_card_amd-epyc-7b13.json: 9b3195ca87f82cf3966a32f30051a03f
benchmarks/perf_card/cpu_card_amd-epyc-9b45.json: 440352ba86c3de47db28e0e9d25fbd6b
benchmarks/perf_card/cpu_card_intel-xeon-platinum-8581c.json: f9c4d4d83d853181142e47e22a9cc994
-->

<!-- cards:begin -->
**NVIDIA H100 80GB HBM3, card of 2026-10-13**

| Distribution (N = 1,000,000) | Arm | wall (range) | sample·steps/s | useful FLOP/s (% of peak) |
|---|---|---:|---:|---:|
| Spread (log-uniform stop steps, up to 1000) | eagle graph (device loop) | 2.13 ms (2.11 ms–2.14 ms) | 1.02e11 | 4.47e12 (13 %) |
| Spread (log-uniform stop steps, up to 1000) | eagle graph + compaction | 16 ms (16 ms–16 ms) | 1.35e10 | 5.96e11 (1.8 %) |
| Spread (log-uniform stop steps, up to 1000) | eagle graph + compaction + reorder | 10.7 ms (10.6 ms–10.7 ms) | 2.03e10 | 8.93e11 (2.7 %) |
| Spread (log-uniform stop steps, up to 1000) | CuPy masked | 368 ms (367 ms–369 ms) | 5.88e8 | 2.59e10 (0.077 %) |
| Spread (log-uniform stop steps, up to 1000) | CPU OpenMP | 199 ms (198 ms–200 ms) | 1.09e9 | 4.79e10 (–) |
| Uniform (every sample runs 1000 steps) | eagle graph (device loop) | 2.32 ms (2.31 ms–2.33 ms) | 4.30e11 | 1.89e13 (57 %) |
| Uniform (every sample runs 1000 steps) | eagle graph + compaction | 22.9 ms (22.9 ms–22.9 ms) | 4.37e10 | 1.92e12 (5.7 %) |
| Uniform (every sample runs 1000 steps) | eagle graph + compaction + reorder | 23.3 ms (23.3 ms–23.3 ms) | 4.30e10 | 1.89e12 (5.7 %) |
| Uniform (every sample runs 1000 steps) | CuPy masked | 365 ms (364 ms–366 ms) | 2.74e9 | 1.21e11 (0.36 %) |
| Uniform (every sample runs 1000 steps) | CPU OpenMP | 170 ms (169 ms–170 ms) | 5.90e9 | 2.59e11 (–) |

Measured on:

- [NVIDIA H100 80GB HBM3](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-nvidia-h100-80gb-hbm3)
- [NVIDIA L4](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-nvidia-l4)
- [NVIDIA A100-SXM4-40GB](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-nvidia-a100-sxm4-40gb)
- [Tesla T4](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-tesla-t4)
- [Quadro P2000](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-quadro-p2000)
- [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://amasat01.github.io/eagle/content/performance.html#perf-card-gpu-nvidia-rtx-pro-6000-blackwell-server-edition)
- [AMD EPYC 9B45](https://amasat01.github.io/eagle/content/performance.html#perf-card-cpu-amd-epyc-9b45)
- [AMD EPYC 7B13](https://amasat01.github.io/eagle/content/performance.html#perf-card-cpu-amd-epyc-7b13)
- [Google Axion (Arm Neoverse-V2)](https://amasat01.github.io/eagle/content/performance.html#perf-card-cpu-aarch64)
- [Intel Xeon Platinum 8581C](https://amasat01.github.io/eagle/content/performance.html#perf-card-cpu-intel-xeon-platinum-8581c)
<!-- cards:end -->

<!-- cpu-cards:begin -->
**On a CPU alone: AMD EPYC 9B45, 8 cores, card of 2026-10-13**

| Arm (N = 1,000,000) | spread S=1000 wall | uniform S=1000 wall | vs eagle 8 threads (spread) |
|---|---:|---:|---:|
| eagle host, termination loop, up to 8 threads | 142 ms | 187 ms | 1.0× |
| Numba prange, 8 threads | 328 ms | 1.53 s | 2.3× |
| JAX sharded over 8 CPU devices | 1.86 s | 1.94 s | 13× |
| PyTorch masked, 8 threads | 10.2 s | 10.2 s | 72× |
| NumPy masked | 15.3 s | 13.4 s | 100× |

**On a CPU alone: AMD EPYC 7B13, 8 cores, card of 2026-10-13**

| Arm (N = 1,000,000) | spread S=1000 wall | uniform S=1000 wall | vs eagle 8 threads (spread) |
|---|---:|---:|---:|
| eagle host, termination loop, up to 8 threads | 400 ms | 693 ms | 1.0× |
| Numba prange, 8 threads | 398 ms | 1.85 s | 0.99× |
| JAX sharded over 8 CPU devices | 2.15 s | 2.14 s | 5.3× |
| PyTorch masked, 8 threads | 8.19 s | 8.37 s | 20× |
| NumPy masked | 21.6 s | 19.6 s | 54× |

**On a CPU alone: Google Axion (Arm Neoverse-V2), 16 cores, card of 2026-10-13**

| Arm (N = 1,000,000) | spread S=1000 wall | uniform S=1000 wall | vs eagle 8 threads (spread) |
|---|---:|---:|---:|
| eagle host, termination loop, up to 8 threads | 330 ms | 648 ms | 1.0× |
| Numba prange, 8 threads | 388 ms | 1.8 s | 1.1× |
| JAX sharded over 8 CPU devices | 410 ms | 411 ms | 1.2× |
| PyTorch masked, 8 threads | 3.02 s | 2.76 s | 9.1× |
| NumPy masked | 16.4 s | 15.6 s | 49× |

**On a CPU alone: Intel Xeon Platinum 8581C, 8 cores, card of 2026-10-13**

| Arm (N = 1,000,000) | spread S=1000 wall | uniform S=1000 wall | vs eagle 8 threads (spread) |
|---|---:|---:|---:|
| eagle host, termination loop, up to 8 threads | 162 ms | 123 ms | 1.0× |
| Numba prange, 8 threads | 335 ms | 1.59 s | 2.0× |
| JAX sharded over 8 CPU devices | 1.03 s | 1.03 s | 6.3× |
| PyTorch masked, 8 threads | 7.39 s | 7.24 s | 45× |
| NumPy masked | 29.1 s | 28.4 s | 170× |
<!-- cpu-cards:end -->

Figures are written by `tools/sync_readme_cards.py` from the card JSON (the
same numbers `card_nvidia-h100-80gb-hbm3.md` renders): wall time and rates rounded to 3
significant figures, percent-of-peak to 2; a dash means that arm has no
FP64-peak baseline recorded (CPU). Re-run the script after a card is re-run or
a new `perf_card` device is added; `python/tests/test_readme_cards.py` fails
if the committed block has drifted from a fresh render.

**Where each tool fits.** On a dense batch, where every sample runs all 1000 steps, NVIDIA Warp's per-thread
kernel is a few percent ahead of `eagle.simulate` from N = 100,000 up (<!--c wall arm=warp_kernel n=100000 dist=uniform-->245 µs<!--/c--> vs <!--c wall arm=eagle_simulate n=100000 dist=uniform-->260 µs<!--/c--> at N = 100,000, <!--c pctgap arm=eagle_simulate den_arm=warp_kernel n=100000 dist=uniform-->6%<!--/c-->;
<!--c wall arm=warp_kernel dist=uniform-->1.71 ms<!--/c--> vs <!--c wall arm=eagle_simulate dist=uniform-->1.75 ms<!--/c--> at N = 1,000,000, <!--c pctgap arm=eagle_simulate den_arm=warp_kernel dist=uniform-->2.5%<!--/c-->), so it is the better fit for a batch of that shape. On a batch whose samples stop at different steps, `eagle.simulate` is faster
than Warp's per-thread kernel at N = 1,000,000 (<!--c wall arm=eagle_simulate-->613 µs<!--/c--> against Warp's <!--c wall arm=warp_kernel-->1.54 ms<!--/c-->, <!--c x arm=warp_kernel den_arm=eagle_simulate-->2.5×<!--/c-->). JAX compiles cold faster than hawk + eagle
(about <!--c comp key=jax:1000 which=cold-->242 ms<!--/c-->, against `eagle.simulate`'s <!--c comp key=eagle:eagle_simulate which=cold-->875 ms<!--/c-->; NVIDIA Warp's cold compile is <!--c comp key=warp which=cold-->574 ms<!--/c-->), and
JAX's warm recompile (<!--c comp key=jax:1000 which=warm-->40 ms<!--/c-->) is shorter than hawk + eagle's own (<!--c comp key=eagle:eagle_simulate which=warm-->70 ms<!--/c-->). These are
complementary tools: Warp and hawk + eagle both skip nothing, and the difference is who writes the per-thread loop and
who decides when to compact. The CPU OpenMP arm needs no GPU, which makes it the tool to reach for on a machine
without one; where a GPU is present, the GPU arms are far ahead on this card (<!--c wall arm=cpu_openmp-->199 ms<!--/c--> against
`eagle.simulate`'s <!--c wall arm=eagle_simulate-->613 µs<!--/c--> on spread, <!--c wall arm=cpu_openmp dist=uniform-->170 ms<!--/c--> against <!--c wall arm=eagle_simulate dist=uniform-->1.75 ms<!--/c--> on uniform, at N = 1,000,000).

Compaction's advantage tracks how early the batch thins, and the device matters. On this H100 the plain graph runs the whole
batch so quickly that active-set bookkeeping has little to recover: the plain graph is faster than the compaction arm in every
cell of the card, <!--c wall arm=eagle_graph-->2.13 ms<!--/c--> against <!--c wall arm=eagle_graph_compact-->16 ms<!--/c--> on the spread distribution at N = 1,000,000 and <!--c wall arm=eagle_graph dist=uniform-->2.32 ms<!--/c--> against
<!--c wall arm=eagle_graph_compact dist=uniform-->22.9 ms<!--/c--> on the uniform one, so plain `hawk + eagle graph` is the better fit here. The P2000 tab on the performance page
shows a card where compaction pays on a thinning batch: <!--c wall dev=p2000 arm=eagle_graph-->586 ms<!--/c--> → <!--c wall dev=p2000 arm=eagle_graph_compact-->374 ms<!--/c--> at N = 1,000,000 on spread
(<!--c x dev=p2000 arm=eagle_graph den_arm=eagle_graph_compact-->1.5×<!--/c-->). The eager launch loop takes <!--c x arm=eager_loop den_arm=eagle_graph-->15×<!--/c--> (spread) and <!--c x arm=eager_loop dist=uniform den_arm=eagle_graph-->16×<!--/c--> (uniform) the plain graph's
time at this N (<!--c wall arm=eager_loop-->32.3 ms<!--/c--> / <!--c wall arm=eager_loop dist=uniform-->39.1 ms<!--/c--> vs <!--c wall arm=eagle_graph-->2.13 ms<!--/c--> / <!--c wall arm=eagle_graph dist=uniform-->2.32 ms<!--/c-->), which is what the CUDA graph is for: it keeps the step loop on the device.
CuPy masked is the array-style option (<!--c wall arm=cupy_masked-->368 ms<!--/c--> on spread, <!--c pct field=fraction_of_peak_dram arm=cupy_masked dist=uniform-->74%<!--/c--> of peak memory bandwidth on uniform, recorded in the card).

The reorder arm is compaction's index map plus an occasional physical reorder, opt-in through
`eagle.ActiveSet(mask, reorder=theta)`. On the H100 it is faster than compaction alone in one cell of the card, N = 1,000,000 on
the spread distribution (<!--c wall arm=eagle_graph_compact-->16 ms<!--/c--> → <!--c wall arm=eagle_graph_reorder-->10.7 ms<!--/c-->, <!--c x arm=eagle_graph_compact den_arm=eagle_graph_reorder-->1.4×<!--/c-->), and takes longer in the others, for example <!--c wall arm=eagle_graph_reorder dist=uniform-->23.3 ms<!--/c--> against
<!--c wall arm=eagle_graph_compact dist=uniform-->22.9 ms<!--/c--> on uniform at N = 1,000,000.

CPU, no GPU at all: the CPU block above shows each CPU card measured; on the AMD EPYC 9B45 the host
termination loop on 8 threads takes <!--c wall card=rk4cpu dev=amd-epyc-9b45 arm=eagle_term8-->142 ms<!--/c--> (spread) / <!--c wall card=rk4cpu dev=amd-epyc-9b45 arm=eagle_term8 dist=uniform-->187 ms<!--/c--> (uniform) at N = 1,000,000, against Numba's
<!--c wall card=rk4cpu dev=amd-epyc-9b45 arm=numba_prange-->328 ms<!--/c--> / <!--c wall card=rk4cpu dev=amd-epyc-9b45 arm=numba_prange dist=uniform-->1.53 s<!--/c-->. On the harder RK7(8) family (adaptive orbits, heavier per-step work) the GPU still leads clearly at
N = 1,000,000: <!--c wall card=rk78 arm=eagle_simulate-->43.6 ms<!--/c--> for `eagle.simulate` on the H100, <!--c wall card=rk78cpu dev=amd-epyc-9b45 arm=eagle_cpu8-->3.76 s<!--/c--> for the eagle CPU path on 8 EPYC threads, and
<!--c wall card=rk78 arm=eagle_cpu-->6.51 s<!--/c--> for it on the GPU card's own host. See
[where RAPTOR fits](https://amasat01.github.io/where_raptor_fits.html) for that comparison.

**Hardware caveat:** the tables and text above are measured on a <!--c spec field=device.name-->NVIDIA H100 80GB HBM3<!--/c-->, a data-centre card (<!--c spec field=device.sm_count-->132<!--/c--> SMs,
<!--c spec field=peak.fp64_flops-->3.35e13<!--/c--> FLOP/s FP64 peak at its rated clock, from the card). The method
carries over unchanged to other GPUs; the absolute numbers, and the crossover points between arms, differ from device to
device, and the performance page has a tab for each one measured.

Full write-up, method and the per-N tables: [the performance
page](https://amasat01.github.io/eagle/content/performance.html). Raw
numbers:
[`card_nvidia-h100-80gb-hbm3.json`](https://github.com/amasat01/eagle/blob/main/benchmarks/perf_card/card_nvidia-h100-80gb-hbm3.json) /
[`card_nvidia-h100-80gb-hbm3.md`](https://github.com/amasat01/eagle/blob/main/benchmarks/perf_card/card_nvidia-h100-80gb-hbm3.md).

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
