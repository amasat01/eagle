Build options
=============

.. _build-options:

eagle exposes the following CMake options.  All are ``OFF`` or a sensible
default out of the box; no flag is required for a standard CUDA build.

.. note::

   Platform scope: Linux and macOS (CPU only on macOS).  Windows is not tested.
   More platforms may be added in future releases.

CMake option reference
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Effect
   * - ``EAGLE_BUILD_TESTS``
     - ``OFF``
     - Build the GoogleTest suite (v1.14.0 is fetched automatically).
   * - ``EAGLE_DEBUG_MODE``
     - ``OFF``
     - Enable CUDA error checks (``EAGLE_CHECK``, ``EAGLE_KERNEL_PRE/POST``)
       and device-side assertions.
   * - ``EAGLE_CPP_MODE``
     - ``OFF``
     - CPU-only build: disables all CUDA code paths.  Requires an
       ``AETHER_CPP_MODE`` aether installation.
   * - ``EAGLE_WARN_AS_ERR``
     - ``OFF``
     - Promote all compiler warnings to errors (``-Werror``).  Used in CI.
   * - ``EAGLE_BLOCKSIZE``
     - ``256``
     - CUDA thread-block size (compile-time constant).  Must be a multiple
       of 32; tune for your GPU.
   * - ``EAGLE_BUILD_SANITIZE``
     - ``OFF``
     - Register the ``sanitize`` target (valgrind / compute-sanitizer
       wrapper); requires ``EAGLE_BUILD_TESTS=ON``.

CUDA architectures
------------------

``EAGLE_CUDA_ARCHS`` picks the architectures when ``CMAKE_CUDA_ARCHITECTURES``
is not given:

- ``native`` (the default): the GPUs in this machine only, the fastest build.
  On a machine with no GPU it builds PTX for the oldest architecture the
  ``nvcc`` in use compiles (at least ``sm_60``), which the driver compiles for
  any newer GPU at load.
- ``all``: every one of ``61;70;75;80;86;89;90;100;120`` (Pascal through
  Blackwell) that the ``nvcc`` in use still compiles. A CUDA 12.x toolkit keeps
  the Pascal targets; CUDA 13 drops the architectures below Turing (``75``) and
  adds the Blackwell ones.
- one architecture or a list, such as ``86`` or ``"75;86"`` (``sm_86`` is
  accepted too): exactly those; ``86-virtual`` builds PTX only.

.. code-block:: bash

   cmake -DEAGLE_CUDA_ARCHS=all -B build .
   cmake -DEAGLE_CUDA_ARCHS="80;90" -B build .

The published wheel is built with CUDA 12.9 and no override, so it carries SASS
for every architecture that ``nvcc`` lists from ``sm_60`` up: ``60 61 62 70 72
75 80 86 87 89 90 100 101 103 120 121``, plus PTX at ``100`` and ``121``. A CUDA 13 ``nvcc``
with the same default lists ``75`` up and ``110``, drops ``101``, and ends in
the same two PTX targets. The list a plugin was built for is reported by its ``archs``
field, SASS first and PTX after a slash (e.g. ``61,70,90/90``), and CI checks that the wheel
lists SASS ``100`` and ``120`` and PTX ``100``. The PTX at ``100`` is what lets a CUDA 13 driver
run on Jetson Thor (sm_110), which has no SASS here.

.. list-table:: Driver requirements (Linux; the driver, not the installed toolkit, decides, because the plugin uses the driver API only)
   :header-rows: 1

   * - Code used
     - GPUs
     - Minimum driver
   * - Precompiled SASS (every listed architecture)
     - Any GPU the wheel lists
     - Any driver that supports the GPU: ``>= 525.60.13`` for CUDA 12.x GPUs; Blackwell needs ``>= 570.26`` for the GPU itself
   * - PTX fallback (``compute_100``, ``compute_121``)
     - GPUs with no SASS in the wheel: Jetson Thor under a CUDA 13 driver (sm_110), future GPUs
     - ``>= 575.51.03`` (the CUDA 12.9 PTX ISA); the driver compiles the PTX at first load and caches it in ``~/.nv/ComputeCache``, near-SASS speed after a one-off JIT delay

``CMAKE_CUDA_ARCHITECTURES``, when given, still wins. aether takes the same
switch as ``AETHER_CUDA_ARCHS``.

Platform support
----------------

Pure C++ (host) execution is built and measured on x86-64 Linux (AVX-512
and AVX2-only CPUs) and on aarch64 Linux; the performance page lists the CPUs.
On aarch64 the step is not vectorised and there is no vector-math profile.
The aether headers eagle builds on are compiled with ``-march=x86-64-v3``
(AVX2 and FMA) by default on x86-64, so the host path needs an x86-64 CPU of
that level or newer; the ``AETHER_MARCH`` CMake variable overrides it.

On the device side, eagle is built with CUDA 12.6 or newer; the register gate
pins the toolkits it exercises, and each performance card records the toolkit
it was measured with. Later CUDA 12 releases share the same APIs and are
expected to work; releases before 12.6 are not supported.

eagle asks the driver for its entry points at the CUDA 12.0 interface level, so
a build made with a CUDA 13 toolkit also runs on any CUDA 13 driver, and a
CUDA 12 build on any CUDA 12 or 13 driver.

CPU-only build
--------------

.. _build-cpp:

To build without any CUDA dependency:

1. Install aether in CPU-only mode (see :doc:`installation`).
2. Configure eagle with ``EAGLE_CPP_MODE=ON``:

.. code-block:: bash

   cmake -B build \
         -DEAGLE_CPP_MODE=ON \
         -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
         -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
         .
   cmake --install build

In this mode the CUDA graph launchers are unavailable; use the OpenMP + SIMD
``eagle::cpu::Host`` dispatcher instead.

Debug build
-----------

Enable device-side error checking during development:

.. code-block:: bash

   cmake -B build \
         -DEAGLE_DEBUG_MODE=ON \
         -DEAGLE_BUILD_TESTS=ON \
         -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
         .

.. seealso::

   :doc:`installation` — full installation instructions.
