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
