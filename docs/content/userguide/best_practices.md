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

## Threads: leave the choice to eagle

- Do not tune threads per call. eagle sizes each stage's team from its measured cost: a stage too small to share runs on the calling thread, and a big batch spreads over the cores this process may use.
- Keep independent branches as `g.parallel()`: they run side by side only when each has enough work.
- To give eagle fewer cores (other work on the machine, one process per core group), cap the budget once:

```python
import eagle
eagle.exec.set_host_threads(4)        # or OMP_NUM_THREADS=4 before eagle starts
```

`graph.host_lanes` reports how many threads each stage used. See [threading](threading).

## Choose the device explicitly

- Give every eagle array its device: `eagle.arrays.zeros(shape, dtype, "cuda:0")`.
- Do not infer the device from the library an input array came from. An array made without a device lives on the CPU.

```python
a = eagle.arrays.asarray(host_values, device="cuda:0")
out = eagle.arrays.zeros(a.shape, "float64", "cuda:0")
```

See [several GPUs](howto_several_gpus).
