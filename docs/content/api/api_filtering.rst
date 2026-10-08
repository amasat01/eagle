API reference — filtering
=========================

Prefix-scan, scatter, and array-slicing for sample compaction.  The two scan
engines live in the device namespaces as ``eagle::cuda::Scan`` and
``eagle::cpu::Scan``; ``eagle::filtering`` keeps the containers, the
compaction helper, and the native graph nodes.

``eagle::cuda::Scan``
---------------------

.. doxygenstruct:: eagle::cuda::Scan
   :project: EAGLE
   :members:

``eagle::cpu::Scan``
--------------------

.. doxygenstruct:: eagle::cpu::Scan
   :project: EAGLE
   :members:

Namespace ``eagle::filtering``
------------------------------

.. doxygennamespace:: eagle::filtering
   :project: EAGLE
   :members:
