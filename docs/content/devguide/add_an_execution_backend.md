# Add an execution backend

An execution backend is the library that lets eagle run on one kind of device: its streams, graphs, conditional nodes
and events. There is one file per kind, `libeagle_<kind>`, and eagle's Python core names no kind in its logic. This
page is what you write to add one (HIP, Metal, OpenCL), and what eagle does with it.

## The seam: `eagle-backend/1`

The seam is a C header, `python/src/seam/eagle_backend.h`: plain C, structs of plain data, nothing from C++
crosses it. A backend exports functions with the `eagle_backend_` prefix.

**Seven symbols are required** of every backend:

| Symbol | It answers |
|---|---|
| `eagle_backend_version` | the seam version it was built from (`(major << 16) \| minor`) |
| `eagle_backend_last_error` | the message of the last failed call on this thread |
| `eagle_backend_build_info` | its kind, the eagle release it was built from and the device code it carries |
| `eagle_backend_probe` | whether a usable device is present |
| `eagle_backend_free` | frees a buffer the backend allocated |
| `eagle_backend_capabilities` | which optional groups it provides, one bit each |
| `eagle_backend_device_types` | the DLPack device types it serves |

**The rest are optional, in groups.** A backend sets one capability bit per group it provides, and a set bit promises
that group's symbols: a missing one is a load refusal that names it. A group the backend leaves out makes its calls
raise `eagle.BackendUnavailable("... does not provide <group>")`, so a backend can start small.

| Group | What it gives eagle |
|---|---|
| `stream` | create, synchronize and release a stream |
| `capturer`, `captured` | record the work on a stream into a replayable graph |
| `fork` | branches that run at the same time inside a capture |
| `conditional` | `if`, `while` and `switch` nodes decided on the device |
| `attribution` | which captured node belongs to which launch |
| `graph`, `launcher` | build a graph node by node, then replay it |
| `exec` | launch a compiled kernel over a range of lanes |
| `device` | device properties |
| `interop` | fences for DLPack hand-offs, memory copies |
| `filtering` | compaction and reorder for `ActiveSet` |

The header holds the exact symbol list of every group. Within major version 1 the seam only grows: a new symbol is
bound by itself, and its presence is the capability. A new symbol bumps `EAGLE_BACKEND_VERSION_MINOR`.

Minor 6 adds four optional symbols, bound one by one: `eagle_backend_memcpy` (host to device, device to host, device to
device and host to host, on a stream), `eagle_backend_host_alloc` and `eagle_backend_host_free` (page-locked host memory)
and `eagle_backend_host_func` (run a host function in stream order, which a captured graph keeps as a host node). A
backend without them still loads; `eagle.copy`, `eagle.pinned` and host nodes then raise `BackendUnavailable` naming the
missing symbol. A backend on hardware that shares one memory between host and device may make every copy a no-op: that
choice stays inside the backend.

## The kind and the file

A kind is an execution model (`cuda`, `hip`, `metal`), never a toolkit generation: a CUDA 13 build is still kind
`cuda`. The backend of kind `K` is the file `libeagle_K.so` (`.dylib` on macOS, `.dll` on Windows) beside the eagle
package. `$EAGLE_BACKEND_K` names an exact file to load instead, and a failure to load it is an error, not a fall-back
to another file.

## How eagle finds the backend

An array tells eagle where it lives. The kind comes from its DLPack device type: host memory is the host, type 2 is
`cuda`, 8 is `metal`, 10 is `hip`. So an array from CuPy, torch or JAX picks its backend without eagle knowing
the framework.

`backend="gpu"` in `Graph.build` means the one GPU kind installed on the machine. If two are installed, eagle refuses by
name and asks you to say which:

```text
eagle: backend='gpu' is ambiguous, cuda, hip are installed; name one of them as the backend
```

Pass `backend="hip"` (or any kind) to name it. The default, when you pass nothing, is unchanged: the backend is
inferred from where the planes live.

## The C++ graph: `ExecGraph`

In C++ the graph executor is `eagle::Graph`, an alias for the one of this build: `eagle::cpu::Graph` under
`EAGLE_CPU_ONLY`, `eagle::cuda::Graph` otherwise (`eagle/ExecGraph.h`). Both satisfy the `ExecGraph` concept: move-only,
a scratch arena, the native-node dependency table, and `finalize()` to commit the arena. Code written against
`ExecGraph` builds for either, at no runtime cost, and a new executor (`hip::Graph`) is added by satisfying the same
concept; a `static_assert` beside the alias checks it.

## Checklist

1. Write the seven required symbols and set the capability bits for the groups you provide.
2. Build `libeagle_<kind>` and put it beside the package, or point `$EAGLE_BACKEND_<KIND>` at it.
3. Add the kind to the DLPack table in `python/eagle/_device_kind.py` if it is not there.
4. Pin the symbols the core binds in `python/tests/expected_backend_symbols.txt`, and run the backend wall test.
5. Add the target on the hawk side: [Add a target](https://amasat01.github.io/hawk/content/contributing/add_a_target.html).
   The format and entry kind the target writes are in raptor's manifest vocabulary.

The core loads one kind per process today: the binding's default is a single named constant, and
`_core.cuda_backend(kind=...)` takes another. Taking the kind per call comes when a second GPU kind exists.
