# Concepts

The ideas behind eagle, one page each. Read [Foundations](foundations) first;
the rest can be read in any order. The [glossary](../glossary) defines the
words.

```{toctree}
:maxdepth: 1
:hidden:

foundations
parallel_execution
cuda_graphs
graph_model
graph_forms
graph_across_devices
under_the_hood
graph_view
autodiff
determinism
threading
../interop
../interop/torch
../interop/jax
../interop/cupy_numpy
../interop/warp
```

- [Foundations](foundations): the words the documentation uses, with an example each.
- [Parallel execution](parallel_execution): one body of code run across many items, on a GPU or a CPU.
- [CUDA graphs](cuda_graphs): record once, replay many times; the execution model.
- [Graph model](graph_model): a step as nodes, and how eagle works out what depends on what.
- [Every form of a graph builder](graph_forms): each form, one line each, executed.
- [Graphs across the CPU and the GPU](graph_across_devices): host nodes, `eagle.copy` and `eagle.pinned` in one captured graph.
- [What happens under the hood](under_the_hood): from the decorator to a replay.
- [Graph view](graph_view): draw any graph as ASCII, Mermaid or DOT.
- [Gradients through a graph](autodiff): reverse mode over a whole graph, on the CPU or the GPU.
- {doc}`Same bits, every time <determinism>`: the fixed-order sum behind reproducible reductions.
- [Interoperability](../interop): numpy, cupy and PyTorch arrays in and out through DLPack.
- [eagle in your PyTorch program](../interop/torch): one changed line in a training loop.
- [eagle in your JAX program](../interop/jax): one changed line in a jitted training step.
- [eagle in your NumPy and CuPy pipeline](../interop/cupy_numpy): the same call on either, with streams and the memory pool.
- [eagle in your NVIDIA Warp program](../interop/warp): swap one `wp.launch`, keep the rest.

The library's modules are toured on the [Library structure](../modules/modules) page,
and the measured speed-ups are on the [Performance](../performance) page.
