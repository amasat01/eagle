# Best practices

Habits that keep eagle runs fast and predictable. Each point is a rule and a small example.

## Build once, then launch

- Load the kernel, build the graph and bind the arrays once, before the loop.
- In the loop, only launch. Do not allocate arrays, pass keyword arguments or copy data on each call.
- Read results from the arrays you bound; the launch wrote them in place.

```python
import eagle

x = eagle.arrays.asarray(x0)                      # allocate once
y = eagle.arrays.zeros(x.shape, "float64", x.device)
bound = plan.bind(x=x, y=y)                       # bind once

for _ in range(steps):
    bound.launch()                                # no allocation, no kwargs, no copy
    use(y)                                        # y holds the result
```

Change what is bound with `bound.rebind(x=new)` rather than binding again. See [graph model](graph_model) and [zero-copy interop](examples/zero_copy_interop).

## Threads for small and large batches

- For a small batch stepped many times (a per-step hook, say), run on one thread: set `OMP_NUM_THREADS=1` and do not open `g.parallel()`.
- Use more host threads only for a big batch, where each tile has enough work to pay for the team.
- Measure on a quiet machine; a loaded one hides the difference.

```python
import eagle
eagle.exec.set_host_threads(1)        # same as OMP_NUM_THREADS=1, for this process
```

A launch of one sample starts no team at all. See [threading](threading).

## Choose the device explicitly

- Give every eagle array its device: `eagle.arrays.zeros(shape, dtype, "cuda:0")`.
- Do not infer the device from the library an input array came from. An array made without a device lives on the CPU.

```python
a = eagle.arrays.asarray(host_values, device="cuda:0")
out = eagle.arrays.zeros(a.shape, "float64", "cuda:0")
```

See [several GPUs](howto_several_gpus).
