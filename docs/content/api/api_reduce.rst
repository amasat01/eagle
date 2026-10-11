API reference — reduce
======================

GPU (CUDA) and CPU (OpenMP) parallel reductions.  The two engines live in the
device namespaces as ``eagle::cuda::Reduction`` and ``eagle::cpu::Reduction``;
``eagle::reduce`` keeps the shared kernels and the native graph node.

``eagle::cuda::Reduction``
--------------------------

.. doxygenstruct:: eagle::cuda::Reduction
   :project: EAGLE
   :members:

``eagle::cpu::Reduction``
-------------------------

.. doxygenstruct:: eagle::cpu::Reduction
   :project: EAGLE
   :members:

Canonical tree — ``eagle::reduce::canonical``
---------------------------------------------

The fixed-order tree (leaf blocks of 256, then pairwise levels) with a host and a
device face; see :doc:`../userguide/determinism`. Its members are listed under the
``eagle::reduce`` namespace below.

Namespace ``eagle::reduce``
---------------------------

.. doxygennamespace:: eagle::reduce
   :project: EAGLE
   :members:
