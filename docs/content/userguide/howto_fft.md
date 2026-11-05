# Fourier transforms in a graph

`eagle.fft` makes a transform a graph step. It runs on the CPU (pocketfft) or the GPU (cuFFT), allocates nothing when it runs, and can sit inside a captured graph beside your kernels. Each function returns a node for `Graph.add_node`.

## Data layout

eagle has no complex dtype. Complex data is a `float32` or `float64` array whose last axis has extent 2 (real part, imaginary part). `axes` count the data axes only; the trailing 2 is not one. A NumPy complex array is refused by name: pass a float view of (re, im) pairs.

`norm` is NumPy's: `"backward"` (the default: forward unscaled, inverse scaled by `1/n`), `"forward"` or `"ortho"`.

## Complex to complex

```python
import numpy as np
import eagle
import eagle.fft as F
from eagle import arrays

x = np.random.default_rng(0).uniform(-1, 1, (4, 8, 2))   # 4 signals of 8 points, (re, im)
xa = arrays.asarray(x)
ya = arrays.zeros((4, 8, 2), "float64")

g = eagle.Graph().add_node(F.c2c(xa, ya, axes=(1,))).build()
g.launch()
spectrum = np.asarray(ya)          # same as np.fft.fft(x[..., 0] + 1j * x[..., 1], axis=1)
```

The leading axis is the batch: every row is its own transform. `forward=False` runs the inverse. `out` is written in place in your buffer, and a read-only `out` is refused by name. Pass `arrays.asarray(..., device="cuda:0")` (or a CuPy, torch or DLPack array) to run on the GPU; `x` and `out` must be on the same device.

## Real signals

`r2c` takes a real array and writes the half spectrum, `n // 2 + 1` points along the last of `axes`. `c2r` goes back and leaves its input unchanged.

```python
r = arrays.asarray(np.random.default_rng(1).uniform(-1, 1, (4, 8)))
half = arrays.zeros((4, 5, 2), "float64")
back = arrays.zeros((4, 8), "float64")

g = (eagle.Graph()
     .add_node(F.r2c(r, half, axes=(1,)))
     .add_node(F.c2r(half, back, axes=(1,)))
     .build())
g.launch()
```

## Derivatives

A transform is linear, so `F.jvp(node, tangent, out)` is the same transform of the tangent. `F.vjp(node, cotangent, out)` is the opposite-direction transform with `"backward"` and `"forward"` swapped, so the vjp of the default forward is the unscaled inverse. A real transform also weights the interior of the half spectrum, as the adjoint needs. Both are graph steps, so one graph can hold a forward and its backward:

```python
node = F.r2c(r, half, axes=(1,))
cot = arrays.asarray(np.ones((4, 5, 2)))
bar = arrays.zeros((4, 8), "float64")

g = eagle.Graph().add_node(node).add_node(F.vjp(node, cot, bar)).build()
g.launch()
```

`eagle.diff.Reverse` pulls an `fft` node back through the same rule. See [Gradients through a graph](tutorials/10_gradients_through_a_graph).

## On the GPU, inside a captured graph

The transform plan is built when you call `F.c2c`, `F.r2c` or `F.c2r`, never when the graph runs. Building a node while a stream capture is open is refused by name (`a plan is built outside a stream capture`): create the nodes first, build and capture after. Once built, a replay allocates nothing and gives the same bits each time.

## cuFFT is an optional extra

The CPU path needs nothing extra. The GPU path opens cuFFT the first time a plan is built. Install it with the extra that matches your CUDA:

```bash
pip install 'aether-dsc[cufft12]'    # CUDA 12
pip install 'aether-dsc[cufft13]'    # CUDA 13
```

Without it, a GPU transform raises `aether.FftUnavailable`, and the message names the extra to install.

The API is on the [Python API](../api/api_python) page.
