# Put eagle's work on your stream

With nothing set, eagle issues its GPU work on the legacy default stream. `eagle.stream` puts it on yours for the length of a `with` block. This is how eagle's work lands in a capture you opened, or beside a framework's own work.

```python
import cupy as cp
import eagle

s = cp.cuda.Stream(non_blocking=True)
with eagle.stream(s):
    graph.launch()          # eagle's launches inside the block run on s
```

`s` can be a raw integer handle, an object with `__cuda_stream__`, a PyTorch or Warp stream (`.cuda_stream`), or a CuPy stream (`.ptr`). Anything else raises `TypeError`. The handle is translated once, at the block's edge.

## Inside a capture

A launch on a raw handle joins the capture that handle is in. So a block opened inside your own capture records eagle's work into it, and it runs on every replay, not at launch:

```python
with s:
    s.begin_capture()
    with eagle.stream(s):
        graph.launch()      # recorded into your capture
    captured = s.end_capture()
captured.launch(s)
s.synchronize()
```

## Rules

- Scopes nest; the innermost one wins. They unwind on an exception.
- The stack is per thread: another thread starts with no scope.
- `eagle.stream(s, device=n)` limits the scope to work on GPU `n`; the default covers all.
- eagle's own capturers open a scope on their stream, so work built inside a build or run goes on the stream being captured.

For streams next to PyTorch, CuPy or Warp, see [Use eagle with NumPy and CuPy](../interop/use_eagle_with_cupy_numpy).
