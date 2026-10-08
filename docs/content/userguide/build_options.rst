Build options
=============

.. _build-options:

eagle exposes the following CMake options.  All are ``OFF`` or a sensible
default out of the box; no flag is required for a standard CUDA build.

.. note::

   Platform scope: Linux only.  Windows and macOS are not tested.
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

``CMAKE_CUDA_ARCHITECTURES``, when given, still wins. aether takes the same
switch as ``AETHER_CUDA_ARCHS``.

Platform support
----------------

Pure C++ (host) execution is built and measured today on x86-64 Linux; the
reference machine is a Xeon W-2125 with AVX-512.  The aether headers eagle
builds on are compiled with ``-march=x86-64-v3`` (AVX2 and FMA) by default, so
the host path needs an x86-64 CPU of that level or newer; the ``AETHER_MARCH``
CMake variable overrides it.  Extending the host path to ARM (aarch64) and to
x86 CPUs with other vector widths (for example AVX2-only parts) is planned;
until then those targets are not built or measured.

On the device side, eagle is tested with CUDA 12.6 (a Quadro P2000, and a Tesla
T4 on Kaggle) and CUDA 13.0 (a Tesla T4 on Kaggle: build, test suites and a
correctness check against Warp); the published performance cards are measured
with CUDA 12.6. Later CUDA 12 releases share the same APIs and are expected to
work; releases before 12.6 are not supported.

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
