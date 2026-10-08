Library structure
=================

eagle is organised as a single header-only library under the ``eagle/``
directory.  The public entry point is ``eagle/eagle.h``, which pulls in all
modules.

For most users, including the top-level header is sufficient:

.. code-block:: cpp

   #include <eagle/eagle.h>

Individual module headers may be included independently when compile-time
isolation is required.

Modules
-------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: :octicon:`git-merge` Graph
      :link: module_graph
      :link-type: doc

      CUDA-graph capture and launch: ``StreamCapturer``, ``Graph``,
      ``Launcher``, plus RAII ``Stream`` / ``Event`` / host-callback wrappers.

      :bdg-secondary:`→ API reference` :doc:`../api/api_graph`

   .. grid-item-card:: :octicon:`filter` Filtering
      :link: module_filtering
      :link-type: doc

      Prefix-scan, scatter, and array slicing for stream compaction.

      :bdg-secondary:`→ API reference` :doc:`../api/api_filtering`

   .. grid-item-card:: :octicon:`graph` Reduce
      :link: module_reduce
      :link-type: doc

      GPU (CUDA) and CPU (OpenMP) parallel reductions over aether scalar arrays.

      :bdg-secondary:`→ API reference` :doc:`../api/api_reduce`

   .. grid-item-card:: :octicon:`cpu` Host
      :link: module_host
      :link-type: doc

      OpenMP + SIMD packet-batched host dispatch with masked / flag-aware
      launch variants for the pure-C++ path.

      :bdg-secondary:`→ API reference` :doc:`../api/api_host`

   .. grid-item-card:: :octicon:`tools` Utilities
      :link: module_util
      :link-type: doc

      Global typedefs, logging, error handling, observer-pattern primitives,
      array slicing, and OpenMP thread management.

      :bdg-secondary:`→ API reference` :doc:`../api/api_util`

Directory layout
----------------

.. code-block:: text

   eagle/
   ├── eagle.h                   ← single top-level include (umbrella)
   ├── typedefs.h                ← Real, idx_t, uint_t, Vec3R, Vec6R, VecNT, …
   ├── cuda.h                    ← shortcut: cuda (graph) module
   ├── filtering.h               ← shortcut: filtering module
   ├── reduce.h                  ← shortcut: reduce module
   ├── ScratchArena.h            ← device-neutral graph scratch arena
   │
   ├── cuda/
   │   ├── Graph.h               ← CUDA graph builder + dependency manager
   │   ├── Launcher.h            ← instantiated graph executor
   │   ├── StreamCapturer.h      ← records launches into a CUDA graph
   │   ├── CapturedGraph.h       ← owning captured-graph wrapper
   │   ├── ComputeBlocks.h       ← grid / block sizing helper
   │   ├── NativeNode.h          ← native graph-node concept (addNative)
   │   ├── Reduction.h           ← cuda::Reduction — GPU reduction engine
   │   ├── Scan.h                ← cuda::Scan — GPU prefix-scan engine
   │   ├── traits.h              ← graph type traits
   │   ├── Stream.h              ← shortcut: stream primitives
   │   ├── detail.h              ← graph implementation detail
   │   ├── detail/Holder.h       ← graph node holder
   │   └── stream/
   │       ├── Stream.h          ← RAII cudaStream_t wrapper
   │       ├── Event.h           ← RAII cudaEvent_t wrapper
   │       └── HostCallback.h    ← host-callback graph node
   │
   ├── cpu/
   │   ├── Graph.h               ← cpu::Graph — dependency-ordered host executor
   │   ├── Host.h                ← cpu::Host packet-batched dispatcher
   │   ├── Reduction.h           ← cpu::Reduction — OpenMP reduction engine
   │   └── Scan.h                ← cpu::Scan — OpenMP prefix-scan engine
   │
   ├── filtering/
   │   ├── Scanner.h             ← prefix-scan container (Scanner / RefScanner)
   │   ├── Slice.h               ← FilteringSlice compaction helper
   │   ├── ScanNode.h            ← native scan graph node
   │   ├── FilteringSliceNode.h  ← native compaction graph node
   │   └── scan_detail.h         ← low-level scan / scatter kernels
   │
   ├── reduce/
   │   ├── ReductionNode.h       ← native reduction graph node
   │   └── detail.h              ← warp / block reduction kernels
   │
   └── util/
       ├── log.h                 ← EAGLE_TRACE/DEBUG/INFO/WARN/ERROR
       ├── throw.h               ← EAGLE_THROW, EAGLE_ASSERT
       ├── DeviceError.h         ← CUDA error tracking + EagleErrorTypes
       ├── Observable.h          ← CRTP observer-pattern base
       ├── Observer.h            ← observer interface
       ├── ObservableArray.h     ← observable array wrapper
       ├── Slice.h               ← general-purpose array slicing
       └── Threads.h             ← OpenMP thread-count management
