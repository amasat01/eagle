Installation
============

Python (pip)
------------

.. code-block:: bash

   pip install "raptor-eagle[cuda12]"      # or [cuda13]; add [torch] for PyTorch interop

The package imports as ``eagle``. It needs Python 3.9 or newer (CPython
3.9-3.14) and an NVIDIA driver at run time; no ``nvcc`` or CUDA toolkit. Free-threaded builds (3.13t, 3.14t) are supported as wheels but do not yet declare GIL-free support: CPython re-enables the GIL when eagle is imported and prints a RuntimeWarning, so results are correct but not run in parallel. On 3.13t the ``[cuda12]`` / ``[cuda13]`` extras do not resolve (CuPy 14 ships no 3.13t wheel), so eagle runs on the CPU route there; 3.14t has no such limit. The
wheel ships GPU code for Pascal, Volta, Ampere and Hopper (sm_61/70/80/90) plus
PTX for newer GPUs. To write kernels for it with hawk, install
``"raptor-hawk[cuda12]"`` alongside (``[cuda13]`` on both for CUDA 13).

.. important::

   **NVIDIA packages come only through the extras.** ``raptor-eagle[cuda12]`` /
   ``[cuda13]`` pull CuPy (``cupy-cuda12x`` / ``cupy-cuda13x``, with the CUDA headers CuPy compiles against);
   ``raptor-hawk[cuda12]`` pulls ``cuda-bindings`` 12,
   ``nvidia-cuda-nvrtc-cu12`` and ``nvidia-cuda-cccl-cu12``, and
   ``raptor-hawk[cuda13]`` pulls ``cuda-bindings`` 13, ``nvidia-cuda-nvrtc`` 13
   and ``nvidia-cuda-cccl`` 13 (CUDA 13's wheels have no ``-cu13`` suffix).
   Without an extra pip installs no NVIDIA package: you get the CPU route, or the GPU route through a CUDA setup you already have. Pick the extra matching the CUDA
   version your driver reports (``nvidia-smi``, top right).

**Platforms:** built and tested on Linux x86_64 only so far (CPython 3.9–3.14, including free-threaded 3.13t and 3.14t), on NVIDIA GPUs from Pascal (Quadro P2000) and Turing (Tesla T4). There are no wheels for macOS, Windows or ARM yet, and WSL2 is untested. ``raptor-core`` and ``aether-dsc`` are pure Python and install anywhere.

C++ library
-----------

eagle is a header-only library.  Installation copies the headers and a CMake
config file so that downstream projects can consume it with
``find_package(eagle CONFIG REQUIRED)``.

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Dependency
     - Version
     - Notes
   * - aether
     - ≥ 0.2
     - Required; must be installed first (see below).
   * - CUDA Toolkit
     - 12.6 or newer (tested with 12.6 and 13.0)
     - Required for GPU mode; not needed for ``EAGLE_CPP_MODE``.
   * - OpenMP
     - any
     - Required for the ``eagle::cpu::Host`` dispatcher.
   * - GoogleTest
     - 1.14.0
     - Test-only; fetched automatically by CMake when ``EAGLE_BUILD_TESTS=ON``.
   * - CMake
     - ≥ 3.20
     - Build system.
   * - C++ compiler
     - C++23
     - GCC ≥ 12 or Clang ≥ 16; for GPU mode, also no newer than your CUDA
       Toolkit's own supported host-compiler ceiling (nvcc rejects a newer one).

Installing aether first
------------------------

eagle depends only on aether, which must be installed into the same prefix:

.. code-block:: bash

   git clone https://github.com/amasat01/aether.git
   cd aether
   cmake -DAETHER_DEBUG_MODE=OFF \
         -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
         -B build .
   cmake --install build

For a CPU-only aether build (required for ``EAGLE_CPP_MODE``):

.. code-block:: bash

   cmake -DAETHER_DEBUG_MODE=OFF \
         -DAETHER_CPP_MODE=ON \
         -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
         -B build .
   cmake --install build

Building and installing eagle
-----------------------------

.. code-block:: bash

   cd eagle
   cmake -B build \
         -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
         -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
         .
   cmake --install build

See :doc:`build_options` for all available CMake options.

Using eagle in your project
---------------------------

After installation, add to your ``CMakeLists.txt``:

.. code-block:: cmake

   find_package(eagle CONFIG REQUIRED)
   target_link_libraries(your_target PRIVATE eagle::eagle)

Then include the library with a single header:

.. code-block:: cpp

   #include <eagle/eagle.h>

Building the Python package from source
----------------------------------------

For contributors, or to build against your own toolchain. The Python package (``raptor-eagle``, imported as ``eagle``) has two compiled
parts. ``eagle._core`` is plain C++ and needs no CUDA toolkit, runtime or
driver. ``eagle/libeagle_cuda.so``, the CUDA backend the core loads on first
use, is built with ``nvcc`` (pass ``-DEAGLE_PYTHON_CUDA_PLUGIN=OFF`` to build
the core alone). At run time the backend needs only the NVIDIA driver
(``libcuda.so.1``): it bundles no CUDA runtime and requires no CUDA package. The
build needs ``aether`` installed in CUDA mode (without ``-DAETHER_CPP_MODE=ON``)
plus eagle's own headers, both in ``PREFIX``. It also depends on `raptor
<https://github.com/amasat01/raptor>`__, whose schema constants it
re-exports (``pip install raptor-core``, or a clone next to this one). The Python
package needs Python 3.9 or newer (CPython 3.9–3.14). The examples that use CuPy
arrays need CuPy for your CUDA major version, which the ``[cuda12]`` / ``[cuda13]``
extra installs.

.. code-block:: bash

   cmake -DCMAKE_PREFIX_PATH=${PREFIX} -B build-hdr . && cmake --install build-hdr --prefix ${PREFIX}
   pip install raptor-core         # or a clone of the raptor repository: pip install ../raptor
   export CMAKE_PREFIX_PATH=${PREFIX}
   # append ;-DCMAKE_CUDA_HOST_COMPILER=<g++-13> if your default compiler is newer than nvcc accepts
   export SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=<your GPU's arch, e.g. 70>"
   pip install -e ./python          # or: pip install ./python

Verify the install:

.. code-block:: bash

   python -c "import eagle; print(eagle.ABI_VERSION)"

A GPU and its driver are needed only to *run* device code (``eagle.loaded``,
``eagle.pipeline``) — the package imports and the host (CPU) launch path work
without them, and a device call then raises ``eagle.BackendUnavailable``. ``python/tests`` is the test suite:
``pytest python/tests -m "not gpu and not interop_matrix"`` drops the GPU-only
rows on a machine without a GPU, cupy or torch.

.. seealso::

   :doc:`build_options` — all CMake flags and their defaults.
   :doc:`quickstart` — a minimal working C++ capture-and-launch example.
   :doc:`quickstart_python` — a minimal working Python example.
