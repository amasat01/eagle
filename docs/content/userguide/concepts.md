# Explanation

The ideas behind eagle, grouped by topic. Read [Foundations](foundations) first; the rest can be read in any order. The [glossary](../glossary) defines the words.

## Build and control graphs

How a step becomes a graph, how it replays, and how to look at one. To write one, see [Build a graph programmatically](graph_forms).

```{toctree}
:maxdepth: 1
:hidden:

cuda_graphs
graph_model
graph_across_devices
graph_view
```

- [CUDA graphs](cuda_graphs): record once, replay many times; the execution model.
- [Graph model](graph_model): a step as nodes, and how eagle works out what depends on what; the builder forms are in [Build a graph programmatically](graph_forms).
- [Graphs across the CPU and the GPU](graph_across_devices): host nodes, `eagle.copy` and `eagle.pinned` in one captured graph.
- [Graph view](graph_view): draw any graph as ASCII, Mermaid or DOT.

## Run kernels

What a launch is, what runs under it, and why a run gives the same bits every time.

```{toctree}
:maxdepth: 1
:hidden:

foundations
parallel_execution
under_the_hood
determinism
threading
why_infrastructure
```

- [Foundations](foundations): the words the documentation uses, with an example each.
- [Parallel execution](parallel_execution): one body of code run across many items, on a GPU or a CPU.
- [How eagle runs a graph](under_the_hood): from the decorator to a replay.
- {doc}`Same bits, every time <determinism>`: the fixed-order sum behind reproducible reductions.
- [Using eagle from several threads](threading): what is safe, and how it is checked. The contract is [raptor's threading page](https://amasat01.github.io/raptor/content/threading.html).
- [Why scheduling is infrastructure](why_infrastructure): what eagle owns so applications do not.

## Derivatives

Reverse mode over a whole graph.

```{toctree}
:maxdepth: 1
:hidden:

autodiff
```

- [Gradients through a graph](autodiff): reverse mode over a whole graph, on the CPU or the GPU.

## Library structure

The modules of the library.

```{toctree}
:maxdepth: 1
:hidden:

../modules/modules
```

- [Library structure](../modules/modules): a tour of each module.
