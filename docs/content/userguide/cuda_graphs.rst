CUDA graphs
===========

.. contents:: On this page
   :local:
   :depth: 2

.. note::

   This page is the C++ class/method reference for the graph engine, with
   no kernel code. Working from Python? :doc:`tutorials/02_cuda_graphs_python`
   is the same record-once/replay-many idea, worked end to end with
   :func:`eagle.until_done` and :class:`~eagle.GraphPipeline`.

EAGLE is built around **CUDA graphs**: record a stream of kernel launches once,
assemble them into a dependency graph, and replay the instantiated graph to
amortise per-launch overhead across the inner loop of a batched algorithm.

Execution model
---------------

There are three cooperating pieces:

``StreamCapturer``
  Wraps ``cudaStreamBeginCapture`` / ``cudaStreamEndCapture``.  Call
  ``begin()`` before launching kernels on the associated stream and ``end()``
  afterwards to obtain a ``cudaGraph_t``.

``Graph``
  Collects captured graphs and host callbacks into a single dependency graph
  (``addNode``, ``addKernelNode``, ``addHostNode``) and instantiates it into a
  ``Launcher`` via ``launcher()``.

``Launcher``
  The instantiated, replayable graph.  ``launch()`` submits it,
  ``synchronize()`` waits, and ``setLogicalSize()`` changes the **logical
  size** — how many samples the next replay actually processes — without
  rebuilding the graph. Concretely: for each kernel node it re-derives the
  grid/block launch dimensions for the new sample count and patches them
  directly into the already-instantiated graph (no re-capture, no new
  ``cudaGraph_t``) — so the next ``launch()`` runs exactly as many threads
  as the new size needs, no more. A typical reason to shrink it: a batched
  loop where some samples finish (**terminate**) before others and no
  longer need processing, so later replays only need to cover the samples
  still active.

Direct graph usage
------------------

.. code-block:: cpp

   #include <eagle/eagle.h>

   eagle::cuda::Stream stream;
   eagle::cuda::Graph  graph;
   graph.stream(stream.cuda());

   // Capture a kernel into the graph
   eagle::cuda::StreamCapturer cap(stream.cuda());
   cap.begin();
   myKernel<<<grid, block, 0, stream.cuda()>>>(args);
   graph.addNode(cap.end());

   // Add a host callback (runs on the CPU as a graph node)
   graph.addHostNode([&result]() { /* post-process on CPU */ });

   // Instantiate and launch
   eagle::cuda::Launcher launcher = graph.launcher();
   launcher.launch();
   launcher.synchronize();

Stream management
-----------------

``eagle::cuda::Stream`` wraps a ``cudaStream_t`` with RAII lifetime management; call
``.cuda()`` to obtain the raw handle:

.. code-block:: cpp

   #include <eagle/eagle.h>

   eagle::cuda::Stream myStream;      // creates a new CUDA stream
   graph.stream(myStream.cuda());      // hand the raw handle to the graph
   myStream.synchronize();

``eagle::cuda::Event`` similarly wraps ``cudaEvent_t`` for cross-stream
synchronisation — make one stream wait for work already queued on a
*different* stream, without a full host-side ``synchronize()`` on either:

.. code-block:: cpp

   eagle::cuda::Stream streamA, streamB;
   eagle::cuda::Event doneOnA;

   // ... launch work on streamA ...
   doneOnA.record(streamA);        // mark this point in streamA's queue

   // streamB won't start its next queued work until streamA reaches
   // the recorded point — no host-side wait, purely device-side ordering
   streamB.waitFor(doneOnA);
   // ... launch work on streamB that depends on streamA's result ...

.. note::

   eagle defines ``CUDA_API_PER_THREAD_DEFAULT_STREAM=1`` at compile time, so
   stream ``0`` maps to each thread's private default stream rather than the
   legacy NULL stream.

Conditional graph nodes: skippable regions
-------------------------------------------

**Needs CUDA ≥ 12.3.** Every node in a plain graph always runs on replay.
A **conditional node** lets one region decide, on the device and with no
host round trip, whether it runs at all this replay — useful for a region
that is only sometimes needed, like an event's follow-up work.
``eagle::conditional::ConditionalGroup`` is the native-node wrapper for it:
populate its own body graph, then add the whole group into a parent
``Graph`` exactly like any other native node.

.. code-block:: cpp

   #include <eagle/conditional.h>

   // *livePtr is a device word the caller updates between replays.
   eagle::CountGuard guard{ livePtr, /*baseline=*/nullptr };  // runs iff *livePtr != 0
   eagle::conditional::ConditionalGroup group(guard);

   cudaKernelNodeParams kp = {};
   void* args[]    = { (void*)&someBuffer };
   kp.func         = (void*)onlyWhenLiveKernel;
   kp.gridDim      = { 1, 1, 1 };
   kp.blockDim     = { 1, 1, 1 };
   kp.kernelParams = args;
   group.body().addKernelNode(kp, {});

   eagle::cuda::Graph g;
   g.addNative(std::move(group), {});
   g.finalizeNatives();

   eagle::cuda::Launcher launcher = g.launcher();
   launcher.launch();   // the body fires iff *livePtr != 0, read fresh every replay

``CountGuard`` compares a device word (``count``) against a baseline
(``0`` when ``baseline`` is ``nullptr``); change only that word between
replays and the body fires or is skipped accordingly, with no rebuild and
no re-capture. ``eagle/conditional.h`` is the umbrella header: it also pulls
in the loop-building sibling (``eagle::cuda::CaptureConditional``, for
weaving a conditional directly into stream capture) and documents the
nesting and single-launcher limits inline.

.. seealso::

   :doc:`../modules/module_graph` — full ``cuda`` module description.
   :doc:`../api/api_graph` — ``Graph``, ``Launcher``, ``StreamCapturer`` API.
