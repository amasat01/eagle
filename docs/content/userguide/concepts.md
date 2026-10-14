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
graph_view
autodiff
determinism
threading
../interop
```

- [Foundations](foundations): the words the documentation uses, with an example each.
- [Parallel execution](parallel_execution): one body of code run across many items, on a GPU or a CPU.
- [CUDA graphs](cuda_graphs): record once, replay many times; the execution model.
- [Graph model](graph_model): a step as nodes, and how eagle works out what depends on what.
- [Graph view](graph_view): draw any graph as ASCII, Mermaid or DOT.
- [Gradients through a graph](autodiff): reverse mode over a whole graph, on the CPU or the GPU.
- {doc}`Same bits, every time <determinism>`: the fixed-order sum behind reproducible reductions.
- [Interoperability](../interop): numpy, cupy and PyTorch arrays in and out through DLPack.

The library's modules are toured on the [Library structure](../modules/modules) page,
and the measured speed-ups are on the [Performance](../performance) page.
