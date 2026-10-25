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
tutorials/21_monte_carlo_early_termination
tutorials/22_heat_energy_check
tutorials/23_fluids_pinned_frames
tutorials/24_fem_solve_early_stop
tutorials/25_particle_binning_stream
tutorials/04_multi_kernel_workflow
tutorials/07_host_or_device
```

- [What a graph is](tutorials/02_cuda_graphs_python): one launch replays a whole run.
- [Steps per launch](tutorials/05_steps_per_launch): fuse steps to cut host-device chatter.
- [Finished samples](tutorials/06_compaction_and_reorder): compaction, reorder, and when not to.
- [Trajectories that stop early](tutorials/21_monte_carlo_early_termination): a Monte Carlo batch thins out, and `ActiveSet(group=)` gathers the live samples, animated.
- [Heat diffusion with an energy check](tutorials/22_heat_energy_check): the steps as one graph, the total heat checked on the host every few steps.
- [Shallow water: frames come home while the steps run](tutorials/23_fluids_pinned_frames): a copy to pinned host memory beside the GPU steps, with the schedule drawn.
- [A finite-element solve that stops when it has converged](tutorials/24_fem_solve_early_stop): assemble, then iterate in one graph with a device loop that leaves when the residual is small.
- [Particle binning, streamed from the host](tutorials/25_particle_binning_stream): a host loader and a host-to-device copy beside the GPU binning, one replay per batch.
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
tutorials/19_train_through_a_branch
tutorials/12_jax_training_step
tutorials/13_jax_vmap_and_jacobians
tutorials/14_jax_simulation_loop
tutorials/15_a_learned_correction_per_step
tutorials/16_a_warp_kernel_in_a_graph
tutorials/17_a_jax_model_between_eagle_launches
tutorials/18_the_eagle_batch_inside_jit
tutorials/20_simulated_batches_for_jax_training
```

- [Gradients through a physics kernel](tutorials/03_torch_training): a hawk kernel inside `loss.backward()`.
- [Build a step as a graph](tutorials/09_graph_model): `eagle.Graph`, built, inspected and run.
- [Gradients through a graph](tutorials/10_gradients_through_a_graph): `eagle.diff`, checked against finite differences.
- [Conditionals, budgets and events](tutorials/11_conditionals_budgets_events): reverse through a skipped step, a capped memory budget, and a stop between two steps.
- [Train through a branch](tutorials/19_train_through_a_branch): `jump=` gives a learnable threshold its gradient, and descent finds it.
- [A hawk kernel inside your jitted JAX step](tutorials/12_jax_training_step): fit a drag coefficient with `jax.jit` and `grad`.
- [Batches and Jacobians](tutorials/13_jax_vmap_and_jacobians): `vmap`, `jacfwd`, `hessian`, and which cost one launch.
- [A stepping loop in `lax.scan`](tutorials/14_jax_simulation_loop): balls that land and stop, on the CPU and the GPU.
- [Train a torch net inside your simulation loop](tutorials/15_a_learned_correction_per_step): a net runs before every step; the gradient goes through the whole run.
- [A Warp kernel in the middle of an eagle graph](tutorials/16_a_warp_kernel_in_a_graph): your kernel between two eagle steps, with its gradient.
- [A JAX model between eagle launches](tutorials/17_a_jax_model_between_eagle_launches): your network writes the correction, eagle steps the batch.
- [The eagle batch inside `jit`](tutorials/18_the_eagle_batch_inside_jit): `while_loop` to run, `scan` to differentiate, no `where`.
- [Simulated batches for a JAX training loop](tutorials/20_simulated_batches_for_jax_training): eagle makes fresh data every step, one process per GPU for a bigger set.

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
