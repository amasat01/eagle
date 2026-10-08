API reference — cuda
====================

CUDA-graph capture and launch machinery, the GPU scan / reduction engines,
plus the RAII stream primitives.

Namespace ``eagle::cuda``
-------------------------

.. note::

   The native graph-node protocol is a C++20 concept, which the documentation
   toolchain cannot render; it is summarised here and documented inline in
   ``eagle/cuda/NativeNode.h``. A type ``N`` satisfies ``eagle::cuda::NativeNode``
   when it provides ``reserveScratch(ScratchArena&)`` and
   ``buildInto(Graph&, deps) -> idx_t``. ``Graph::addNative(node, deps)``
   (constrained on the concept, hence also not listed below) registers such a
   node; ``Graph::finalizeNatives()`` commits the shared scratch arena and
   builds every registered node in add-order.

.. doxygennamespace:: eagle::cuda
   :project: EAGLE
   :members:
