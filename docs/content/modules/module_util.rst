Module: util
============

The ``util`` module provides supporting infrastructure: global typedefs,
logging, exception helpers, CUDA device-error tracking, observer-pattern
primitives, array slicing, and OpenMP thread-count management.

Headers
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Header
     - Contents
   * - ``eagle/typedefs.h``
     - ``Real``, ``idx_t``, ``uint_t``, ``dims_t``, ``Vec3R``, ``Vec6R``,
       ``VecNT`` and their aether array aliases.
   * - ``eagle/util/log.h``
     - ``Level`` enum; ``setLogLevel`` / ``disableLogging``;
       ``EAGLE_TRACE/DEBUG/INFO/WARN/ERROR`` macros.
   * - ``eagle/util/throw.h``
     - ``EAGLE_THROW``, ``EAGLE_ASSERT``.
   * - ``eagle/util/DeviceError.h``
     - ``EAGLE_CHECK``, ``EAGLE_KERNEL_PRE/POST``, ``EAGLE_GPU_THROW/ASSERT``
       macros; ``eagle::err::EagleErrorTypes`` enum.
   * - ``eagle/util/Observable.h``
     - ``Observable`` — CRTP base for the observer pattern.
   * - ``eagle/util/Observer.h``
     - ``Observer`` — observer interface for ``Observable``.
   * - ``eagle/util/ObservableArray.h``
     - Observable array wrapper that notifies observers on writes.
   * - ``eagle/util/Slice.h``
     - General-purpose array slicing (``Slice`` / ``RefSlice``).
   * - ``eagle/util/Threads.h``
     - OpenMP thread-count management.

Logging
-------

.. code-block:: cpp

   #include <eagle/util/log.h>

   eagle::util::setLogLevel(eagle::util::Level::DEBUG);
   eagle::util::disableLogging();

   EAGLE_TRACE("val = %f", x);
   EAGLE_DEBUG("step %d", n);
   EAGLE_INFO("launch complete");
   EAGLE_WARN("small batch detected");
   EAGLE_ERROR("failed to launch");

Log levels (``eagle::util::Level``):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Description
   * - ``OFF``
     - Suppress all output.
   * - ``ERROR``
     - Error conditions only.
   * - ``WARN``
     - Warnings and errors.
   * - ``INFO``
     - Informational messages (default).
   * - ``DEBUG``
     - Detailed diagnostic output.
   * - ``TRACE``
     - Per-step trace output; very verbose.
   * - ``ALL``
     - Alias for ``TRACE``.

Error handling
--------------

Host-side exceptions:

.. code-block:: cpp

   #include <eagle/util/throw.h>

   EAGLE_THROW("bad input: %d", val);          // throws std::runtime_error
   EAGLE_ASSERT(condition, "msg: %d", val);     // throws on false

Host-side CUDA API check (unconditional, defined identically in both build
modes):

.. code-block:: cpp

   #include <eagle/util/DeviceError.h>

   EAGLE_CHECK_ALWAYS(cudaMalloc(&ptr, size));   // wraps CUDA API calls

Device-side error tracking (requires ``EAGLE_DEBUG_MODE``):

.. code-block:: cpp

   #include <eagle/util/DeviceError.h>

   EAGLE_KERNEL_PRE();
   myKernel<<<g, b, 0, stream>>>(args);
   EAGLE_KERNEL_POST();                   // checks for device errors after kernel

   // Inside a kernel (no-op unless EAGLE_DEBUG_MODE):
   EAGLE_GPU_ASSERT(condition, eagle::err::FAILED_WARP_LEADER);

Without ``EAGLE_DEBUG_MODE``, the ``KERNEL_*``/``GPU_*`` device-side macros
expand to no-ops; ``EAGLE_CHECK_ALWAYS`` always checks, in both modes.

Observer pattern
----------------

``Observable`` (CRTP base) notifies registered ``Observer`` instances when its
state changes; ``ObservableArray`` wraps an aether array so writes propagate to
observers — used to keep derived caches in sync.

.. seealso::

   :doc:`../api/api_util` — full ``util`` API reference.
