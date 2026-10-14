# Python tutorials

A learning path through eagle from Python. Start with the
[Python quickstart](quickstart_python), then follow the groups in order. Each
tutorial is a short notebook that runs on the CPU and picks up the GPU
automatically when one is visible. The C++ path is on the
[C++ tutorials](tutorials) page.

## Launching and graphs

What a graph is, how many steps to fuse into one launch, how finished samples
leave the run, and where the work runs.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/02_cuda_graphs_python
tutorials/05_steps_per_launch
tutorials/06_compaction_and_reorder
tutorials/04_multi_kernel_workflow
tutorials/07_host_or_device
```

- [What a graph is](tutorials/02_cuda_graphs_python): one launch replays a whole run.
- [Steps per launch](tutorials/05_steps_per_launch): fuse steps to cut host-device chatter.
- [Finished samples](tutorials/06_compaction_and_reorder): compaction, reorder, and when not to.
- [A multi-kernel step](tutorials/04_multi_kernel_workflow): propagate, diagnose and stop, declaratively.
- [Host or device?](tutorials/07_host_or_device): let the data decide where it runs.

## Training and gradients

Putting a kernel inside a training loop, and differentiating a whole graph.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/03_torch_training
tutorials/09_graph_model
tutorials/10_gradients_through_a_graph
tutorials/11_conditionals_budgets_events
```

- [Gradients through a physics kernel](tutorials/03_torch_training): a hawk kernel inside `loss.backward()`.
- [Build a step as a graph](tutorials/09_graph_model): `eagle.Graph`, built, inspected and run.
- [Gradients through a graph](tutorials/10_gradients_through_a_graph): `eagle.diff`, checked against finite differences.
- [Conditionals, budgets and events](tutorials/11_conditionals_budgets_events): reverse through a skipped step, a capped memory budget, and a stop between two steps.

## Seeing what ran

```{toctree}
:maxdepth: 1
:hidden:

tutorials/08_profile_your_run
```

- [Profile your run](tutorials/08_profile_your_run): where the time went, in three numbers.

## Further

```{toctree}
:maxdepth: 1
:hidden:

extra/fork_join_graph
```

- [A graph with parallel branches](extra/fork_join_graph): fork, join and reduce, optional reading.
