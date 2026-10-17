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
