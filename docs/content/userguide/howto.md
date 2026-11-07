# How-to guides

Short, task-shaped pages: you know what you want to do, here is how. Grouped by topic, first use to advanced.

## Build and control graphs

Build a graph in code, in any of its forms. The ideas are in [Explanation](concepts); capped device loops with `if c: <assignments>; break` are shown in the graph builder page and in [Gradients through conditionals](tutorials/11_conditionals_budgets_events).

```{toctree}
:maxdepth: 1
:hidden:

graph_forms
```

- [Build a graph in code](graph_forms): every form of `@eagle.Graph.builder`, one line each, executed.

## Run kernels

Load a kernel by name, run it until every sample is done, and move arrays in and out without a copy.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/01_launch_and_registry
best_practices
examples/early_termination
examples/zero_copy_interop
examples
```

- [Load a compiled kernel](tutorials/01_launch_and_registry): load a compiled kernel by name, then call, stream or graph it.
- [Best practices](best_practices): build once then launch, threads for small and large batches, choose the device.
- [Early termination](examples/early_termination): a batch that finishes at different times, with a conditional graph node that skips the work once every sample is done.
- [Zero-copy interop](examples/zero_copy_interop): one kernel fed from and handed back into NumPy, CuPy and PyTorch arrays with no copy.
- [C++ examples](examples): complete, tested C++ programs and how their gate runs.

## Transforms, products and streams

Graph steps for Fourier transforms and matrix products, and how to put eagle's work on your own stream.

```{toctree}
:maxdepth: 1
:hidden:

howto_fft
howto_gemm
howto_streams
```

- [Fourier transforms in a graph](howto_fft): `eagle.fft` on the CPU and GPU, with derivatives, replayed in a captured graph.
- [Matrix products in a graph](howto_gemm): `eagle.gemm` host products and a cuBLAS product recorded in a capture.
- [Put eagle's work on your stream](howto_streams): `with eagle.stream(s):`, including inside your own capture.

## Keep the state at every step

`eagle.simulate(model, ..., max_steps=N, history=True)` keeps it: `res.history["x"]` has `N + 1` rows (the initial state, then after each step), and `history=k` keeps every k-th step (`N // k + 1` rows). A sample that finished repeats its final state. The final state is the same as without `history`. It runs one launch per step, so the kernel must advance one step per launch, and the rows take `rows * n * planes` of memory.

## Direct calls and bound planes

- A direct kernel call takes `returns="numpy"` (or `"cupy"`, `"torch"`, `"jax"`), as `run` does.
- A number or 0-d value for a per-lane `Scalar` beside a batch is that value on every lane. A `Param` takes a number or 0-d array; an array for it is refused by name.
- A kernel parameter declared `Param[Matrix[2, 2]]` (or `Vector`, `Table`) is one value shared by every lane. `run`, `bind` and a direct call take it in its own shape and refuse another by name.
- An `Index` plane has the width the kernel declares (`int32` unless `Index[int64]`). `bind` and `rebind` refuse another width by name and say which to pass; `run` converts on each call.
- Opaque notes a kernel's author put on a plane (`hawk.Meta`) are on the loaded kernel as `plane_meta`, by plane name.

## Run on several GPUs

```{toctree}
:maxdepth: 1
:hidden:

howto_several_gpus
```

- [Run on several GPUs](howto_several_gpus): `RankPartition` over MPI ranks, the same answer as one rank.

## Failures

A device out-of-memory error says which kernel needed what: the largest local memory per thread among the graph's kernels, and the GPU's free and total memory at the moment of failure.

## Deploy

Take a compiled kernel out of Python.

```{toctree}
:maxdepth: 1
:hidden:

examples/01_device_properties
cpp/c4_embed_a_kernel
```

- [Device properties](examples/01_device_properties): read the device, then combine two compiled force plugins from one manifest.
- [Run a deployed kernel on the CPU](cpp/c4_embed_a_kernel): load plugins (float64 or float32) from a manifest and run them from C++ with no CUDA toolchain.

## Use eagle with other frameworks

Keep your PyTorch, JAX, Warp or NumPy and CuPy program and swap in eagle for the stepping. Start with the overview.

```{toctree}
:maxdepth: 1
:hidden:

../interop
../interop/use_eagle_with_torch
../interop/use_eagle_with_jax
../interop/use_eagle_with_warp
../interop/use_eagle_with_cupy_numpy
tutorials/jax_training_step
tutorials/jax_vmap_and_jacobians
tutorials/jax_simulation_loop
tutorials/torch_learned_correction_per_step
tutorials/warp_kernel_in_a_graph
tutorials/jax_model_between_eagle_launches
tutorials/jax_eagle_batch_inside_jit
tutorials/jax_simulated_batches_for_training
```

- [Use eagle with other frameworks](../interop): numpy, cupy and PyTorch arrays in and out through DLPack.
- [Use eagle with PyTorch](../interop/use_eagle_with_torch): one changed line in a training loop.
- [Use eagle with JAX](../interop/use_eagle_with_jax): one changed line in a jitted training step.
- [Use eagle with NVIDIA Warp](../interop/use_eagle_with_warp): swap one `wp.launch`, keep the rest.
- [Use eagle with NumPy and CuPy](../interop/use_eagle_with_cupy_numpy): the same call on either, with streams and the memory pool.
- [A hawk kernel inside your jitted JAX step](tutorials/jax_training_step): fit a drag coefficient with `jax.jit` and `grad`.
- [Batches and Jacobians](tutorials/jax_vmap_and_jacobians): `vmap`, `jacfwd`, `hessian`, and which cost one launch.
- [A stepping loop with scan](tutorials/jax_simulation_loop): balls that land and stop, on the CPU and the GPU.
- [Train a net in the loop](tutorials/torch_learned_correction_per_step): a net runs before every step; the gradient goes through the whole run.
- [A Warp kernel in a graph](tutorials/warp_kernel_in_a_graph): your kernel between two eagle steps, with its gradient.
- [A JAX model between launches](tutorials/jax_model_between_eagle_launches): your network writes the correction, eagle steps the batch.
- [The batch inside jit](tutorials/jax_eagle_batch_inside_jit): `while_loop` to run, `scan` to differentiate, no `where`.
- [Simulated training batches](tutorials/jax_simulated_batches_for_training): eagle makes fresh data every step, one process per GPU for a bigger set.

## Domain cases

Five cases seen from eagle's side: how the launch is orchestrated. The physics and the kernel for each are on hawk's [Domains](https://amasat01.github.io/hawk/content/domains.html) pages (Monte Carlo, Heat diffusion, Fluids, Finite elements, Particle binning).

```{toctree}
:maxdepth: 1
:hidden:

tutorials/21_monte_carlo_early_termination
tutorials/22_heat_energy_check
tutorials/23_fluids_pinned_frames
tutorials/24_fem_solve_early_stop
tutorials/25_particle_binning_stream
```

- [Monte Carlo](tutorials/21_monte_carlo_early_termination): the batch thins out and `ActiveSet(group=)` gathers the live samples.
- [Heat diffusion](tutorials/22_heat_energy_check): the steps are one graph, the total heat checked every few steps.
- [Fluids](tutorials/23_fluids_pinned_frames): frames come home to pinned host memory while the steps run.
- [Finite elements](tutorials/24_fem_solve_early_stop): assemble, then iterate in one graph with a device loop that leaves when the residual is small.
- [Particle binning](tutorials/25_particle_binning_stream): a host loader and a host-to-device copy beside the GPU binning, one replay per batch.
