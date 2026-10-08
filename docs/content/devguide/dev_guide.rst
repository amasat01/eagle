Developer guide
===============

.. contents:: On this page
   :local:
   :depth: 2

This guide covers building eagle for development, running the test suite,
coding conventions, and the contribution workflow.

Development build
-----------------

.. code-block:: bash

   cd eagle
   cmake -B build \
         -DEAGLE_BUILD_TESTS=ON \
         -DEAGLE_DEBUG_MODE=ON \
         -DEAGLE_WARN_AS_ERR=ON \
         -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
         .
   cmake --build build -- -j$(nproc)

Running tests
-------------

.. code-block:: bash

   ./build/tests/eagle_tests

Tests live in ``tests/`` and are built with GoogleTest (v1.14.0, fetched
automatically).  They require ``EAGLE_BUILD_TESTS=ON``.  Run the binary
directly rather than through ``ctest`` — eagle deliberately has no ctest
wiring, and ``ctest --test-dir build`` reports ``Total Tests: 0`` and exits 0
here rather than actually running anything. CI drives the same binary through
``tests/check_gate.sh``, which additionally pins the exact set of tests that
ran against a committed manifest; see :doc:`test-map` for what each test
covers.

Coding conventions
------------------

Language
  C++23 and CUDA.  All headers are self-contained (``#pragma once``).

Naming
  * Classes: ``PascalCase``
  * Methods and free functions: ``camelCase``
  * Private data members: trailing underscore (``member_``)
  * Template parameters: ``PascalCase`` with descriptive suffix
    (``DataT``, ``KernelFunc``, ``MaskT``).

Formatting
  The repository ships a ``.clang-format`` file.  Run ``clang-format`` before
  committing:

  .. code-block:: bash

     find eagle -name '*.h' -print0 | xargs -0 clang-format -i

  (a plain ``eagle/**/*.h`` glob silently misses headers directly under
  ``eagle/`` and ones nested more than one directory deep, unless your shell
  has recursive globbing enabled — ``find`` picks up all of them regardless
  of shell).

Docstrings
  Every public symbol must have a Doxygen ``/** @brief … */`` docstring.
  See :doc:`docs_guide` for the full style guide.

CUDA device code
  * Use ``DEVICEHOST()`` for functions callable from both host and device.
  * Use ``KERNEL()`` for ``__global__`` kernels.
  * Device-only helpers that are not part of the public API should be placed in
    a ``detail`` namespace / ``detail/`` sub-directory.
  * Wrap such internal-only code with ``/// @cond INTERNAL`` / ``/// @endcond``.
  * This hides it from Breathe (the Sphinx/Doxygen bridge that generates the
    API reference pages; see :doc:`docs_guide`) during extraction.

Dual-mode discipline
  eagle compiles as CUDA or pure C++23 (OpenMP), selected by ``EAGLE_CPP_MODE``
  (which sets ``EAGLE_CPU_ONLY``).  Guard every CUDA-only construct behind
  ``#ifndef EAGLE_CPU_ONLY`` and provide an OpenMP + SIMD counterpart so the
  public API stays identical across both modes.

Adding a new error code
------------------------

1. Add a new enumerator to ``eagle::err::EagleErrorTypes`` in
   ``eagle/util/DeviceError.h`` (bit-flag, power of two).
2. Add a corresponding entry to ``eagle::err::errorMessages``.
3. Use ``EAGLE_GPU_THROW(MY_ERROR)`` or
   ``EAGLE_GPU_ASSERT(condition, MY_ERROR)`` inside device code.

Contribution workflow
----------------------

1. Branch off ``main``: ``git checkout -b feature/my-feature``.
2. Implement, add tests, update docs.
3. Run ``clang-format``, ``./build/tests/eagle_tests`` (or
   ``tests/check_gate.sh``), and ``make doxygen html`` in ``docs/``.
4. Sign off each commit (``git commit -s``) and open a pull request on GitHub;
   the CI workflows build, test and check the docs.
