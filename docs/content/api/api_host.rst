API reference — host
====================

OpenMP + SIMD host dispatch: packet-batched launchers with masked and
flag-aware variants for the pure-C++ path.

``eagle::cpu::Host``
---------------------

.. doxygenstruct:: eagle::cpu::Host
   :project: EAGLE
   :members: launch, launchIf, launchIfNot, packetLaunch, packetLaunchIfNot, dispatchMasked, applyTail

.. note::

   The low-level SIMD helpers ``packetLoad`` / ``packetStore`` / ``loadMask``
   carry C++20 ``requires``-constrained overloads that the Sphinx C++ domain
   cannot render; they are described in :doc:`../modules/module_host` and
   documented inline in ``eagle/cpu/Host.h``.

Host schedules
--------------

How a graph stage runs on the host: ``work_share`` spreads each node over the
whole team, ``branch_tasks`` runs the stage's independent branches at once,
one per thread. :func:`eagle::exec::resolve` is the pure rule that picks one; the
guide is :doc:`../userguide/graph_model`.

.. doxygenenum:: eagle::exec::Schedule
   :project: EAGLE

.. doxygenvariable:: eagle::exec::BRANCH_TASKS_MAX_SAMPLES
   :project: EAGLE

.. doxygenfunction:: eagle::exec::resolve
   :project: EAGLE

.. doxygenstruct:: eagle::exec::HostBranchNode
   :project: EAGLE
   :members:

.. doxygenfunction:: eagle::exec::runHostBranches
   :project: EAGLE
