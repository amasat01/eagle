API reference
=============

This section provides the complete Doxygen-generated C++ API for eagle,
rendered through Breathe and split by module. For the Python package
(``eagle``, imported from ``pip install -e ./python``), see
:doc:`api_python`.

.. note::

   The API reference is generated from Doxygen XML.  Run ``make doxygen``
   in the ``docs/`` directory before building the Sphinx docs, or use
   ``make doxygen html`` to build both in one step.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Contents
   * - :doc:`api_graph`
     - ``Graph``, ``Launcher``, ``StreamCapturer``, ``CapturedGraph`` (see
       :doc:`../modules/module_graph`); ``Stream``, ``Event`` (RAII wrappers,
       see :doc:`../userguide/cuda_graphs`); ``HostCallback`` (a CPU function
       run as a graph node, see :doc:`../modules/modules`); three free
       functions used internally for launch-geometry tuning —
       ``currentSMCount`` (the current GPU's streaming-multiprocessor count),
       ``currentFp64PerfRatio`` (its FP32:FP64 throughput ratio, used to pick
       between native and emulated double-precision kernels), and
       ``computeBlocks`` (derives a block-size/block-count pair for an
       N-sample launch from those two numbers).
   * - :doc:`api_filtering`
     - ``Scanner``, ``FilteringSlice``, ``RefScanner``, ``RefSlice``,
       ``cuda::Scan``, ``cpu::Scan``.
   * - :doc:`api_reduce`
     - ``cuda::Reduction``, ``cpu::Reduction``.
   * - :doc:`api_host`
     - ``eagle::cpu::Host`` — OpenMP + SIMD host dispatcher.
   * - :doc:`api_util`
     - Global typedefs, logging, error handling, ``Observable`` /
       ``Observer``, ``Slice``, and thread management.
