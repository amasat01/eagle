C2 — Run one loop on the CPU or the GPU, unchanged
==================================================

    The same source file compiles against CUDA and against plain
    C++/OpenMP -- one dispatch call, two backends, no ``#ifdef`` in the body.

**Time:** ~3 min · **Runs on:** CUDA GPU or CPU (OpenMP) -- pick the build
command · **You need:** C1 (the capture/replay vocabulary), a C++23 compiler

``eagle::cpu::Host::launch`` runs one task per index through eagle's OpenMP +
SIMD host launcher: consecutive indices are batched into aether SIMD packets and
spread across threads. The *same source* compiles and runs in CUDA mode
(``02_host_dispatch.cu``) and pure-C++ mode (``02_host_dispatch.cpp``) -- that is
eagle's dual-mode promise.

.. literalinclude:: ../../../examples/02_host_dispatch.cu
   :language: cpp
   :start-after: // [cell:launch]
   :end-before: // [cell:launch:end]

What it printed
^^^^^^^^^^^^^^^

Both builds print the same line (the prefix sums of the first *N* odd numbers
are *N*\ :sup:`2`):

.. literalinclude:: ../../../examples/02_host_dispatch.cpp.expected.txt
   :language: text

What just happened
^^^^^^^^^^^^^^^^^^^

- The loop body is one line, written once; ``Host::launch`` is what changes
  between the CUDA build (a device kernel over the same body) and the pure-C++
  build (an OpenMP + SIMD host loop).
- N=1024 lanes each wrote their own prefix sum; the host and device builds
  agree on every element, not just the final total.
- Nothing in ``02_host_dispatch.cu``/``.cpp`` names ``CUDA`` or ``OpenMP``
  directly -- the backend is a build flag (``EAGLE_CPU_ONLY``), not a code
  branch.

Try this
^^^^^^^^

Change ``N`` from 1024 to 1 and rebuild both modes: the printed sum becomes 1
either way -- a batch of one is not a special case.

Next
^^^^

C3 (:doc:`c3_reduce`) -- the same dual-mode idea applied to a reduction instead of a
per-element write.
