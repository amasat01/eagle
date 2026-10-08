Module: cuda
============

The ``cuda`` module is the heart of EAGLE: it records a stream of kernel
launches once, assembles them into a CUDA dependency graph, and replays the
instantiated graph with microsecond-class per-launch re-tuning.  It also
provides RAII wrappers for the underlying CUDA stream / event primitives.

Headers
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Header
     - Contents
   * - ``eagle/cuda/StreamCapturer.h``
     - ``StreamCapturer`` — records kernel launches into a ``cudaGraph_t``
       via stream capture (``begin()`` / ``end()``).
   * - ``eagle/cuda/Graph.h``
     - ``Graph`` — CUDA graph builder and dependency manager
       (``addNode``, ``addKernelNode``, ``addHostNode``, ``launcher``).
   * - ``eagle/cuda/Launcher.h``
     - ``Launcher`` — instantiates and executes a compiled graph
       (``launch``, ``synchronize``, ``setLogicalSize``).
   * - ``eagle/cuda/CapturedGraph.h``
     - ``CapturedGraph`` — owning wrapper around a captured ``cudaGraph_t``.
   * - ``eagle/cuda/ComputeBlocks.h``
     - Grid / thread-block sizing helper for kernel launches.
   * - ``eagle/cuda/stream/Stream.h``
     - ``Stream`` — RAII ``cudaStream_t`` wrapper.
   * - ``eagle/cuda/stream/Event.h``
     - ``Event`` — RAII ``cudaEvent_t`` wrapper.
   * - ``eagle/cuda/stream/HostCallback.h``
     - ``HostCallback`` — host-callback graph node wrapper.
   * - ``eagle/cuda/traits.h``
     - Graph type traits.

Capture, build, launch
----------------------

The canonical pattern captures a kernel launch, adds it as a graph node,
instantiates a ``Launcher``, and replays it:

.. code-block:: cpp

   #include <eagle/eagle.h>

   eagle::cuda::Stream stream;
   eagle::cuda::Graph  graph;
   graph.stream(stream.cuda());

   // Record a kernel launch into the graph
   eagle::cuda::StreamCapturer capturer(stream.cuda());
   capturer.begin();
   myKernel<<<grid, block, 0, stream.cuda()>>>(args);
   graph.addNode(capturer.end());

   // Add a host callback that runs on the CPU
   graph.addHostNode([&]() { /* post-process on CPU */ });

   // Instantiate and replay
   eagle::cuda::Launcher launcher = graph.launcher();
   launcher.launch();
   launcher.synchronize();

Re-tuning without re-capture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``setLogicalSize`` re-tunes the launch geometry of an instantiated graph
without rebuilding it, for the microsecond-class replay of a batch whose
active-sample count changes between iterations.

.. note::

   EAGLE defines ``CUDA_API_PER_THREAD_DEFAULT_STREAM=1`` at compile time, so
   stream ``0`` maps to each thread's private default stream rather than the
   legacy NULL stream.

.. seealso::

   :doc:`../userguide/cuda_graphs` — end-to-end CUDA-graph usage guide.
   :doc:`../api/api_graph` — full ``cuda`` API reference.
