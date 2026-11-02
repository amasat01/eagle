# Matrix products in a graph

`eagle.gemm` runs a matrix product (`out = op(a) @ op(b)`) as cuBLAS work that is legal inside a CUDA graph capture, with a plain host path for the CPU. Use it when a product has to sit between your kernels in a replayed graph.

## One product, on the host or the device

`matmul_step` builds one step. Operands are row-major `eagle.Array`s; `transpose_a`, `transpose_b` and `transpose_out` are flags of the product, so an orientation costs nothing.

```python
import numpy as np
from eagle import arrays
from eagle.gemm.plan import matmul_step

a = np.random.default_rng(0).standard_normal((3, 4))
b = np.random.default_rng(1).standard_normal((4, 5))
bufs = {"a": arrays.asarray(a), "b": arrays.asarray(b), "c": arrays.zeros((3, 5), "float64")}

matmul_step("c", a="a", b="b", out="c").run(bufs, "host")
c = np.asarray(bufs["c"])           # equals a @ b
```

`run(bufs, "device")` enqueues the same product on cuBLAS, on eagle's current stream. The host path takes float32 or float64.

## Capture it

A captured product is float32 only. cuBLAS makes its handle on first use, which allocates, so the handle is made and warmed outside the capture and only `sgemm` runs inside it. `sgemm_params_for_arrays` reads the layout off your three arrays.

```python
from eagle import cuda
from eagle.gemm import CublasCaptureHandle, sgemm_params_for_arrays

a = arrays.asarray(a.astype("float32"), device="cuda:0")
b = arrays.asarray(b.astype("float32"), device="cuda:0")
out = arrays.zeros((3, 5), "float32", "cuda:0")
operands = {"a": a, "b": b, "out": out}

stream = cuda.Stream(non_blocking=True)
handle = CublasCaptureHandle(stream.ptr())
params = sgemm_params_for_arrays(a, b, out)
try:
    handle.warm(params, operands)            # once per shape, before the capture
    out.fill(0.0, stream.ptr())              # warm() wrote `out`: clear it
    cap = cuda.StreamCapturer(stream.ptr())
    cap.begin()
    try:
        handle.sgemm(params, operands)       # recorded, not run
    finally:
        captured = cap.end()
    launcher = cuda.Graph.from_captured(captured).launcher()
    launcher.stream(stream.ptr())
    launcher.launch()                        # replay as often as you like
    launcher.synchronize()
finally:
    handle.destroy()
```

Rules that matter:

- `alpha` and `beta` are baked into the recorded node: change them and you record again.
- Destroy the graph first, then the handle, then the memory pool. The handle must outlive the graph.
- A float64 operand is refused by name (`GemmMappingUnavailable`), as is a shape one `Sgemm` cannot address. No cuBLAS means `GemmCaptureUnavailable`; nothing falls back silently.

## From a contraction to a plan

If a producer hands you a recognised contraction (a `RecognizedContraction` record), `lower_dense_gemm(record, direction="forward"|"adjoint", dtype=...)` builds the steps and the buffers it needs (`plan.required`). Call the plan with `plan("host" | "device", buffers)`. To capture it, `captured_gemm_calls(plan, buffers, operand_class="gemm")` gives one `SgemmParams` and operand map per step, ready for `handle.sgemm`. The `"gemv"` class and non-float32 buffers are refused for capture.

## Opt a kernel in

A hawk kernel that is one dense two-operand contraction (a sum over one loop of `w.at(...) * f.at(...)`, one output per lane) can run as this product instead of its own loop. Build it under hawk's policy:

```python
from hawk.emit import policy

with policy.using(contraction="gemm"):
    plan = eagle.plan.auto(contract)
plan.run(w=w, f=f, y=y)          # one GEMM, written straight into the lane plane y
```

`run`, `bind` and graphs take the plan as before. The values are within a few ulp of the kernel's loop (cuBLAS sums in its own order), not the same bits, so nothing changes unless you opt in; two runs on one device and one library build give the same bits. A kernel the route does not fit (two outputs, a transformed term, lanes that are not the product in row-major order) runs its own loop.

`eagle.diff.reverse` finds the route's adjoint on its own: the two gradient products add into the gradient planes, so a table read by several launches gets the sum.

The API is on the [Python API](../api/api_python) page.
