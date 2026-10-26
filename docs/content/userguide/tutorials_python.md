# Tutorials

A learning path through eagle from Python, grouped by topic. Start with the [Python quickstart](quickstart_python), then take a topic. Within a topic the notebooks go from first use to advanced. Each runs on the CPU and picks up the GPU when one is visible. Cases from other fields, such as heat and fluids, are in the [How-to guides](howto); the domain pages are in [hawk](https://amasat01.github.io/hawk/content/domains.html).

## Build and control graphs

Record a step as a graph, chain kernels, branch, loop and stop, and add parallel branches.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/02_cuda_graphs_python
tutorials/04_multi_kernel_workflow
tutorials/09_graph_model
tutorials/11_conditionals_budgets_events
extra/fork_join_graph
```

- [What a graph is](tutorials/02_cuda_graphs_python): one launch replays a whole run.
- [A multi-kernel step](tutorials/04_multi_kernel_workflow): propagate, diagnose and stop, declaratively.
- [Build a step as a graph](tutorials/09_graph_model): `eagle.Graph`, built, inspected and run.
- [Conditionals, budgets and events](tutorials/11_conditionals_budgets_events): reverse through a skipped step, a capped memory budget, and a stop between two steps.
- [A graph with parallel branches](extra/fork_join_graph): fork, join and reduce, optional reading.

## Run kernels

Launch a batch, fuse steps, let finished samples leave, choose where it runs and see where the time went. The C++ track is the last entry.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/05_steps_per_launch
tutorials/06_compaction_and_reorder
tutorials/07_host_or_device
tutorials/08_profile_your_run
tutorials
```

- [Steps per launch](tutorials/05_steps_per_launch): fuse steps to cut host-device chatter.
- [Finished samples](tutorials/06_compaction_and_reorder): compaction, reorder, and when not to.
- [Host or device?](tutorials/07_host_or_device): let the data decide where it runs.
- [Profile your run](tutorials/08_profile_your_run): where the time went, in three numbers.
- [C++ tutorials](tutorials): the same patterns as small, compiled C++ programs.

## Derivatives

Put a kernel inside a training loop and differentiate a whole graph. hawk's [derivatives reference](https://amasat01.github.io/hawk/content/derivatives.html) names `step_vjp` and `jump=`.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/03_torch_training
tutorials/10_gradients_through_a_graph
tutorials/train_through_a_branch
```

- [Gradients through a physics kernel](tutorials/03_torch_training): a hawk kernel inside `loss.backward()`.
- [Gradients through a graph](tutorials/10_gradients_through_a_graph): `eagle.diff`, checked against finite differences.
- [Train through a branch](tutorials/train_through_a_branch): `jump=` gives a learnable threshold its gradient, and descent finds it.
