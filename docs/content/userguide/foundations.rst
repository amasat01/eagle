Foundations — the words this documentation uses
=================================================

.. contents:: On this page
   :local:
   :depth: 2

Read this page once, before the quickstart. It walks through every term the
rest of the documentation assumes you already know, in the order you
actually need them — each one introduced with a concrete example first, a
plain definition second.

.. note::

   New to parallel or GPU programming? :doc:`parallel_execution` covers the
   one idea everything below builds on — one body of code, run across many
   items at once, on a GPU or a CPU — before diving into EAGLE's specific
   names for it.

What EAGLE does, in one sentence
---------------------------------

You have a kernel launch — or a sequence of them — that you want to run
thousands of times in an inner loop, without paying the CPU-side overhead of
re-issuing the same launch every time. EAGLE records the launch once into a
**graph**, then replays that graph as many times as you like. Everything
below is either part of that record-once/replay-many idea, or its
CPU-side (non-CUDA) counterpart.

The four objects you'll meet in every example
------------------------------------------------

Here is the smallest possible EAGLE program — capture one kernel launch and
replay it:

.. code-block:: cpp

   #include <eagle/eagle.h>

   eagle::cuda::Stream stream;
   eagle::cuda::Graph  graph;
   graph.stream(stream.cuda());

   eagle::cuda::StreamCapturer capturer(stream.cuda());
   capturer.begin();
   addOne<<<1, N, 0, stream.cuda()>>>(dBuf, N);
   graph.addNode(capturer.end());

   eagle::cuda::Launcher launcher = graph.launcher();
   launcher.launch();
   launcher.synchronize();

Four names appear, each doing one job:

``Stream``
  A thin RAII wrapper around a ``cudaStream_t`` — the ordered queue of GPU
  work a kernel launch goes onto. You create one and hand its raw handle
  (``.cuda()``) to whatever needs it.

``StreamCapturer``
  Turns "record what I launch next" on and off. ``begin()`` starts
  recording every kernel you launch on the stream; ``end()`` stops and hands
  you back a ``cudaGraph_t`` — a recording of exactly those launches, not yet
  runnable on its own.

``Graph``
  Collects one or more captured recordings (plus optional CPU callbacks via
  ``addHostNode``) into a single dependency graph, and turns that graph into
  something you can actually run — a ``Launcher`` — via ``graph.launcher()``.

``Launcher``
  The instantiated, replayable graph. ``launch()`` submits the whole
  recorded sequence again; ``synchronize()`` waits for it to finish. This is
  the object you call repeatedly in your inner loop — capturing only
  happens once.

The point of all four together: capture the launch sequence *once* at setup
time, then every later iteration is just ``launcher.launch()`` — no
per-launch CPU overhead, no rebuilding. See :doc:`cuda_graphs` for the full
execution model, including re-tuning launch geometry between replays without
re-capturing.

Dual-mode: the same shape without CUDA
------------------------------------------

Every project built on EAGLE compiles two ways: with a CUDA toolchain
(everything above), or as pure C++23 with OpenMP, no GPU required. The
**dual-mode** promise is that your calling code barely changes between the
two — you write against a Host launch API on the C++ side that mirrors the
CUDA ``Launcher``'s "just call it" shape:

.. code-block:: cpp

   #include <eagle/eagle.h>

   eagle::cpu::Host::launch(nSamples, [&](eagle::SampleIndex i) {
       // per-sample work, one sample per call
   });

``cpu::Host`` is the C++/OpenMP counterpart of the CUDA graph launchers —
where a CUDA kernel runs one GPU thread per sample, ``Host::launch`` runs
this lambda once per sample across your CPU cores. ``eagle::SampleIndex`` is
just the per-sample index type — EAGLE builds on **aether**, its sibling
array library, for this and every other per-sample data type you'll see
below.

Packets, lanes, and masks: how the CPU side reaches GPU-like throughput
------------------------------------------------------------------------

A single CPU core can't match one GPU thread per sample — instead,
``cpu::Host`` processes several samples at once per core using SIMD
(single-instruction-multiple-data) hardware. The vocabulary for this:

- A **packet** is a fixed-size group of ``W`` samples processed together by
  one SIMD instruction stream (``W`` is the SIMD width — how many samples
  fit in one vector register).
- A **lane** is one sample's slot within a packet.
- A **mask** (``PacketMask``) marks which lanes in a packet are currently
  active — used by the flag-aware launchers below to skip inactive lanes (or
  skip an entire packet when none of its lanes are active).

You'll see this vocabulary in ``Host::launchIf`` / ``launchIfNot`` (run only
where — or skip where — a per-sample flag is set, e.g. skipping terminated
samples) and ``Host::packetLaunch`` (hand the kernel the whole packet, for
code that wants to do its own SIMD work via ``packetLoad`` / ``packetStore``).
See :doc:`../modules/module_host` for the full API.

These packet helpers work over any per-sample element type, not only
``double`` — most code uses ``double`` since that runs fastest on today's
hardware, but a debug/oracle build can swap in ``aether::SoftDouble``, a
double built entirely out of ordinary integer/float arithmetic, to
cross-check a result on hardware whose double-precision throughput is
limited. That swap always runs one sample per packet: the software type has
no SIMD-native form to batch.

Stream compaction: shrinking a per-sample array down to only what's active
---------------------------------------------------------------------------

Some algorithms need more than "skip inactive lanes" — they need to
physically shrink an array down to just the samples that are still active,
so later kernels launch over a smaller ``N``. Concretely: 8 samples, flags
``[1,0,1,1,0,0,1,0]`` say which are active, and the operation you want turns
that into a packed 4-element array holding just samples 0, 2, 3, 6 — nothing
else touches memory it doesn't need to. Getting there is two steps: scan the
flags to compute each active sample's new, packed position, then scatter
each one there. That two-step operation is called **stream compaction**.
EAGLE's ``Scanner`` / ``RefScanner`` (in the filtering module) implement it
as a prefix scan over the flags followed by a scatter of the active
elements' offsets. See :doc:`../modules/module_filtering`.

Where to go next
------------------

You now have every term the rest of the documentation uses. From here:

- :doc:`quickstart` — the same capture-and-replay example, built and run end
  to end.
- :doc:`cuda_graphs` — the full CUDA-graph execution model: re-tuning launch
  geometry between replays, host callbacks, cross-stream synchronisation.
- :doc:`../modules/modules` — a tour of every module, once you know the
  vocabulary above.
- :doc:`../devguide/plugin_schema` — advanced: the on-disk contract for a
  *deployed* kernel (a separate, later concern — you don't need it to start).
