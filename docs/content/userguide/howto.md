# How-to guides

Short, task-shaped pages: you know what you want to do, here is how.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/01_launch_and_registry
examples
examples/01_device_properties
examples/early_termination
examples/zero_copy_interop
```

- [Launch a kernel and use the registry](tutorials/01_launch_and_registry): load a compiled kernel by name, then call, stream or graph it.
- [C++ examples](examples): complete, tested C++ programs and how their gate runs, including embedding a compiled kernel in your own program.
- [Device properties and a two-plugin manifest](examples/01_device_properties): read the device, then combine two compiled force plugins from one manifest.
- [Early termination](examples/early_termination): a batch that finishes at different times, with a conditional graph node that skips the work once every sample is done.
- [Zero-copy interop](examples/zero_copy_interop): one kernel fed from and handed back into NumPy, CuPy and PyTorch arrays with no copy.
- [Run a deployed kernel on the CPU](cpp/c4_embed_a_kernel): load plugins (float64 or float32) from a manifest and run them from C++ with no CUDA toolchain.
- [Choose a solver](../performance/rk78): an embedded pair is a tableau you pick; the 8(7) and the 5(4) pairs compared on the same orbits.
- [A JAX array in, an answer out of `until_done`](tutorials/14_jax_simulation_loop): run a whole batch to the end from your own arrays.
