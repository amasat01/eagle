Module: cpu
===========

The ``cpu`` module is the pure-C++ (OpenMP + SIMD) counterpart of the CUDA
graph launchers.  ``cpu::Host`` provides packet-batched launch wrappers: each
call processes a **packet** of ``W`` samples together in one SIMD
instruction stream, where each sample's slot within the packet is a
**lane** and a **mask** (``PacketMask``) marks which lanes are currently
active. ``W`` is a compile-time constant (a template parameter), fixed by
the build's target SIMD width — not something you choose or query at
runtime. See :doc:`../userguide/foundations` for the full walkthrough of this
vocabulary. The masked / flag-aware variants below skip an entire packet
when none of its lanes are active.

Headers
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Header
     - Contents
   * - ``eagle/cpu/Host.h``
     - ``cpu::Host`` — SIMD packet load/store helpers and the ``launch`` /
       ``launchIf`` / ``launchIfNot`` / ``packetLaunch`` family.

Scalar launches
---------------

``Host::launch`` runs a scalar kernel over ``size`` samples, packet-batching
the iteration and work-sharing **tiles** — contiguous chunks of packets, each
sized to fit one OpenMP thread's slice of work comfortably in its CPU core's
L2 cache — across OpenMP threads when called inside
a parallel region:

.. code-block:: cpp

   #include <eagle/eagle.h>

   eagle::cpu::Host::launch(nSamples, [&](eagle::SampleIndex i) {
       // per-sample work
   });

Flag-aware launches
-------------------

``launchIfNot`` skips samples whose boolean flag is set (e.g. terminated
samples); ``launchIf`` runs only where the flag is set.  Both load ``W`` flags
per packet into a ``PacketMask`` and skip whole packets with no active lanes:

.. code-block:: cpp

   // Run only on samples that have NOT terminated
   eagle::cpu::Host::launchIfNot(nSamples, terminated.hostRef().handle(),
       [&](eagle::SampleIndex i) {
           // per-active-sample work
       });

Packet-aware launches
---------------------

``launch`` (above) writes each sample's SIMD packing for you — your kernel body
sees one plain sample at a time, and ``Host::launch`` handles the packet
batching internally. ``packetLaunch`` passes the ``PacketIndex<W>`` directly
to the kernel instead, so *you* write the explicit SIMD code that operates on
``W`` samples at once via ``Host::packetLoad`` / ``Host::packetStore`` — more
work to write, but the right choice when your per-sample logic can be
vectorized more efficiently than the generic per-scalar path ``launch`` gives
you:

.. code-block:: cpp

   eagle::cpu::Host::packetLaunch(nSamples, [&](const auto& pi) {
       auto x = eagle::cpu::Host::packetLoad<Real, decltype(pi)::width>(xArr, pi);
       // ... SIMD work on the packet ...
       eagle::cpu::Host::packetStore<Real, decltype(pi)::width>(xArr, pi, x);
   });

``packetLoad`` / ``packetStore`` / ``loadMask`` above are generic over the
per-sample element type (a ``DataT`` template parameter), not hard-wired to
``Real``. Production code always passes ``Real`` (double) at its native SIMD
width. ``aether::SoftDouble`` — a double built entirely from ordinary
integer/float instructions — can stand in as a dev/debug/oracle type
instead, always at one sample per packet (``W = 1``), since it has no
SIMD-native form to batch.

.. note::

   The optional ``bytesPerSample`` argument drives runtime L2-derived tile
   sizing.  Pass the same value across every host kernel call in a step so
   each thread's tile slice stays L2-resident across kernels.

.. seealso::

   :doc:`../api/api_host` — full ``cpu::Host`` API reference.
