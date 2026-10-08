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

Namespace ``eagle::reduce``
---------------------------

.. doxygennamespace:: eagle::reduce
   :project: EAGLE
   :members:
