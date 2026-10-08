Module: reduce
==============

The ``reduce`` module provides parallel reduction over aether scalar arrays,
on both GPU (CUDA) via ``cuda::Reduction`` and CPU (OpenMP) via ``cpu::Reduction``.

Headers
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Header
     - Contents
   * - ``eagle/cuda/Reduction.h`` / ``eagle/cpu/Reduction.h``
     - ``cuda::Reduction`` (multi-level GPU) and ``cpu::Reduction<T,OP>``
       (cache-padded OpenMP).
   * - ``eagle/reduce/detail.h``
     - Internal warp-shuffle and block-reduction kernels (not public API).

``cuda::Reduction`` — GPU reduction
-----------------------------------

Blocking: the call synchronises the stream and returns the result on the host.

.. code-block:: cpp

   #include <eagle/eagle.h>

   // Sum of all elements (allocates a temporary work buffer)
   double total = eagle::cuda::Reduction::reduceBlocking<double, aether::SumOp<double>>(
       arr, /* init= */ 0.0, stream);

   // Reuse a pre-allocated work buffer (more efficient for repeated reductions)
   auto buf = eagle::cuda::Reduction::makeBuffer(arr);
   buf.upload(stream);
   double total2 = eagle::cuda::Reduction::reduceBlocking<double, aether::SumOp<double>>(
       arr, buf, 0.0, stream);

   // Check if all booleans are true
   bool allDone = eagle::cuda::Reduction::reduceBlocking<bool, aether::LogicalAndOp>(
       terminated, /* init= */ true, stream);

Supported ``OP`` types (from ``aether``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Operator
     - Effect
   * - ``aether::SumOp<T>``
     - Arithmetic sum.
   * - ``aether::MaxOp<T>``
     - Maximum value.
   * - ``aether::LogicalAndOp``
     - Logical AND (``bool`` only).

Graph-based reduction (non-blocking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For integration into a CUDA graph, ``reduceBlocking`` also accepts a
``Graph&`` overload that *adds* the reduction as a node instead of running it
immediately: ``result`` (a plain host pointer, not yet valid) is only filled
once you build and run the graph — ``graph.launcher().launch()`` then
``.synchronize()`` — same two-step capture/replay pattern as
:doc:`../userguide/cuda_graphs`:

.. code-block:: cpp

   double result;
   eagle::cuda::Graph graph;
   eagle::cuda::Reduction::reduceBlocking<double, aether::SumOp<double>>(
       graph, &result, arr.deviceRef(), buf, /* init= */ 0.0, stream);

   auto launcher = graph.launcher();
   launcher.launch();
   launcher.synchronize();
   // only now is 'result' valid

``cpu::Reduction`` — CPU/OpenMP reduction
-------------------------------------------

Each OpenMP thread accumulates into its own slot of a per-thread buffer;
``PaddedT`` is that slot's type, padded to a full cache line so one thread's
writes don't invalidate another thread's cache line for the same buffer (a
slowdown called **false sharing** — more on this in the note below).

.. code-block:: cpp

   #include <eagle/eagle.h>

   // Allocate buffer once and reuse
   std::vector<eagle::cpu::Reduction<double, aether::SumOp<double>>::PaddedT>
       buf(omp_get_max_threads(), {0.0});

   double total = eagle::cpu::Reduction<double, aether::SumOp<double>>::reduce(
       hostRef, buf, 0.0);

   // Or allocate buffer on the fly
   double total2 = eagle::cpu::Reduction<double, aether::SumOp<double>>::reduce(
       hostRef, 0.0);

.. note::

   Cache-line padding (``alignas(64)`` on ``PaddedT``) prevents false sharing
   across OpenMP threads in the per-thread accumulator buffer. 64 bytes is
   the standard CPU cache-line size on x86-64 hardware — padding each
   thread's slot to that size guarantees no two threads' slots ever share
   one cache line.

.. seealso::

   :doc:`../api/api_reduce` — full ``cuda::Reduction`` / ``cpu::Reduction`` API reference.
