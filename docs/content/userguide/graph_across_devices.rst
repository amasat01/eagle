Graphs across the CPU and the GPU
=================================

.. contents:: On this page
   :local:
   :depth: 2

Some steps have a part that belongs on the CPU: a check that reads a few
values, a loader that fills the next batch, your own library call. Others want
a result copied home while the GPU carries on. This page shows how to put
that in the same :class:`eagle.Graph` as your GPU kernels, so the whole step
is one object that is captured once and replayed.

Two words first. A *plane* is any array you pass to a node: a NumPy array, a
CuPy array, a torch tensor. An *opaque* node is a node made of your own
Python function, ``eagle.opaque(fn, reads=..., writes=...)``, where you name the
planes it reads and writes so eagle can order it.

The one rule
------------

**A node runs where its planes live.** Device arrays make a GPU node. Host
arrays make a host node. One node may not mix the two, and a host node may not
touch a device array: its function runs on a helper thread of the CUDA
runtime, which cannot call back into the device.

Moving data is a node of its own, ``eagle.copy(src, into=dst)``. Its direction
follows from where ``src`` and ``dst`` live: host to device, device to host,
device to device, or host to host. You never write the direction.

.. code-block:: python

   import numpy as np
   import cupy as cp
   import eagle
   import hawk

   # propagate on the GPU, check events on the host, propagate again
   @eagle.Graph.builder
   def propagate_with_events(state, h_probe, h_flags, flags):
       g = eagle.Graph()
       state = propagate(state)                    # device planes: a GPU node
       probe = sample(state)                       # a few values per sample, on the device
       eagle.copy(probe, into=h_probe)             # device -> host
       check_events(h_probe, buffer=h_flags)       # the same @hawk.kernel; host planes: a host node
       eagle.copy(h_flags, into=flags)             # host -> device
       g.result = propagate(state, flags)
       return g

   state, flags = cp.zeros((6, n)), cp.zeros(n, np.uint32)
   h_probe = eagle.pinned((3, n))
   h_flags = eagle.pinned(n, np.uint32)
   g = propagate_with_events(state, h_probe, h_flags, flags)
   g.launch(10)                                    # ten steps; one captured graph

``propagate``, ``sample`` and ``check_events`` stand for your own
``@hawk.kernel`` functions. The graph is captured once on the GPU, host node
included, and ``g.launch(10)`` replays it ten times without the host stepping
in between.

When a host array is safe to read
---------------------------------

``g.launch(n)`` replays and waits. When it returns, every host array a node
wrote holds its final value. ``g.enqueue(n)`` replays without waiting, and
then ``g.synchronize()`` is the point after which the host arrays are safe to
read. Reading one before that races with the copy.

A host node's own work runs between the copy before it and the copy after it,
in replay order. An exception inside it cannot unwind through the CUDA
thread: eagle keeps it and raises it from the next ``launch`` or
``synchronize``.

Pinned memory
-------------

``eagle.pinned(shape, dtype)`` returns a NumPy array in page-locked memory. The
GPU copies to and from it by DMA, straight from the array, and the copy can
overlap with kernels. An ordinary (pageable) array is staged through a hidden
buffer first, so the copy is slower and waits for it.

Pinning is yours to decide. eagle never pins or allocates a host array behind
your back: ``into=`` always names an array you made. A pageable array still
works for a one-off copy. A copy in a loop or a branch body needs a pinned
one, and the error says so.

If you reach for torch's ``x.cpu()``: ``eagle.copy(x, into=eagle.pinned(...))``
is the same move with the buffer in your hands. It is the same buffer on every
replay, which is what makes the graph replayable.

A copy between two distinct NumPy arrays is a plain memory copy, so the same
builder runs unchanged on a machine with no GPU.

Overlap: the next copy while this one computes
----------------------------------------------

Two branches of a fork run at the same time, so a copy in one branch overlaps
a kernel in the other. A common shape is the double buffer: the copy
reads one array while the steps that only read it run beside it, and the step that
overwrites it waits for the copy.

.. code-block:: python

   @eagle.Graph.builder
   def round_(a, b, c, h_frame):
       g = eagle.Graph()
       with g.parallel() as p:
           with p.branch():
               eagle.copy(a, into=h_frame)          # the frame leaves on the copy engine ...
           with p.branch() as w:
               b = step(a)                          # ... while the next steps run
               c = step(b)
               w.wait(h_frame)                      # the step that overwrites `a` waits for the copy
               a = step(c, buffer=a)
       return g

``a``, ``b`` and ``c`` are three arrays of one shape; ``step`` is a kernel.
The first steps only read ``a``, so they start with the copy; only the last one waits.
:doc:`tutorials/23_fluids_pinned_frames` runs this shape and draws its
schedule. To see whether the overlap happened on your device, run the program under
Nsight Systems (``nsys profile python run.py``, as in :doc:`tutorials/08_profile_your_run`) and look at the timeline: the
copy and the kernels sit on different streams, and they overlap in time when it did.

What it costs, and what it buys
-------------------------------

A host node in a captured graph adds about 24 µs per replay when the host
function is C++ and about 100 µs when it is Python, which pays for the
interpreter lock (a small probe on a Quadro P2000, 64 floats each way). Do not
read that as a speed-up: on a serial chain of nodes, an eager loop with
a device sync is as fast or faster. What you get is one replayable object, a
launch that does not block the host, and overlap with the branches next to
it. At 8 MiB each way the copy is limited by the PCIe link and both forms
take the same time.

If the same storage is both ends, ``eagle.copy(x, into=x)`` is a node that does
nothing. On a machine whose host and device share memory, a backend may make
every copy do nothing; eagle leaves that decision to the backend.

What is refused
---------------

Each message names the fix.

- **A node with planes on both sides.** ``layer(h_batch, w)`` with ``h_batch``
  on the host and ``w`` on the device: add ``eagle.copy(h_batch, into=<a device
  array>)`` before it.
- **A host node inside a device ``if`` or ``for`` body.** CUDA does not allow
  host functions inside a conditional body. A copy there is fine, from
  pinned memory.
- **A host function that calls the device.** A captured host node cannot. For
  host code that reads a device array, such as a torch op with a read-back, use
  ``eagle.opaque(fn, ..., eager=True)``: the graph is cut around it and the
  device is synchronised before and after.

Without a GPU
-------------

Make every plane a NumPy array and call the builder the same way. A copy is an
assignment, ``copy(x, into=x)`` does nothing, and the host nodes run in a plain
host schedule. Page-locked memory is a GPU matter: where there is no GPU, an
ordinary NumPy array does the same job as ``eagle.pinned``.

See also
--------

- :doc:`tutorials/22_heat_energy_check`: a GPU time-stepper with an energy
  check copied home every few steps.
- :doc:`tutorials/23_fluids_pinned_frames`: frames copied to pinned memory
  while the GPU keeps stepping.
- :doc:`tutorials/24_fem_solve_early_stop`: a solve on the device that decides for itself when to stop.
- :doc:`tutorials/25_particle_binning_stream`: a host loader streaming batches to the GPU with overlap.
- :doc:`graph_model`: the graph model itself; :doc:`graph_forms`: every form of
  the builder.
