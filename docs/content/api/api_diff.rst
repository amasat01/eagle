API reference — reverse of a graph
==================================

The transposition behind :mod:`eagle.diff`: given a graph ``Description`` and
the planes each node's vjp reads and writes, ``eagle::diff::transpose`` returns
the reverse pass (the order of the derivative steps, where each step's incoming
adjoint comes from, which adjoints are summed and in what order, and which
forward planes must be copied before they are overwritten).
``eagle::diff::orderedSum`` is the fixed-order elementwise sum the joins use.
Header-only and CUDA-free; the guide is :doc:`../userguide/autodiff`.

Namespace ``eagle::diff``
-------------------------

.. doxygennamespace:: eagle::diff
   :project: EAGLE
   :members:
