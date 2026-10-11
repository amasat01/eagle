Tutorials
=========

.. contents:: On this page
   :local:
   :depth: 1

These tutorials build up eagle's core patterns as small, complete C++ programs
-- the C++ track alongside the Python tutorials that start at
:doc:`quickstart_python`. Every code block below is ``literalinclude``\ d from a
real source file under ``docs/examples/`` that is **compiled and run** by
``docs/examples/run_examples.sh`` (the ``exampletest`` gate) in both CUDA and
pure-C++ / OpenMP mode. The code you read is exactly the code that produces the
output shown -- there are no illustrative-but-untested snippets here.

C1 is CUDA-only -- graph capture has no pure-C++ analogue. C2 and C3 are
**dual-mode**: the same source file compiles both against CUDA and against
plain C++/OpenMP, with no changes (see :doc:`foundations` for the full
picture). They also work with **SoA** (structure-of-arrays) data: instead of
one array of per-sample structs, each field (position, velocity, ...) lives in
its own contiguous array, which is what lets both the GPU and the OpenMP/SIMD
backend read and write it efficiently.

.. note::

   Build any example directly, with ``eagle`` and ``aether`` installed into your
   environment (``$CONDA_PREFIX``):

   .. code-block:: bash

      # CUDA mode
      nvcc -std=c++20 -arch=sm_61 -I$CONDA_PREFIX/include \
           -DEAGLE_BLOCKSIZE=256 -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 \
           -Xcompiler -fPIE -Xcompiler -fopenmp \
           docs/examples/01_capture_replay.cu -o 01_capture_replay

      # pure-C++ mode (dual-mode examples)
      g++ -std=c++23 -fopenmp -O2 -I$CONDA_PREFIX/include \
          -DEAGLE_CPU_ONLY=1 -DEAGLE_BLOCKSIZE=256 \
          docs/examples/02_host_dispatch.cpp -o 02_host_dispatch

The path, in order. Each page is a small program you can build and run:

.. toctree::
   :maxdepth: 1
   :hidden:

   cpp/c1_capture_replay
   cpp/c2_cpu_or_gpu
   cpp/c3_reduce
   cpp/c5_print_a_graph

- :doc:`cpp/c1_capture_replay` -- record a launch once and replay it (CUDA only).
- :doc:`cpp/c2_cpu_or_gpu` -- one loop, run on the CPU or the GPU unchanged.
- :doc:`cpp/c3_reduce` -- reduce an array the same way on either backend.
- :doc:`cpp/c5_print_a_graph` -- draw any eagle graph as a picture.

Embedding a compiled kernel in your own C++ program (the fourth C++ page) is
a how-to: :doc:`cpp/c4_embed_a_kernel`.

Python: composing several kernels into one step
-------------------------------------------------

The Python tutorials under ``tutorials/`` (starting at :doc:`quickstart_python`)
build the same core patterns from kernels loaded and compiled through hawk
and eagle's Python face, one kernel at a time.
:doc:`tutorials/04_multi_kernel_workflow` is the one to read for a *step*
built from several kernels -- a propagation, a diagnostic and a termination
event -- composed explicitly through ``eagle.plan.plan``,
:class:`~eagle.GraphPipeline` and :class:`~eagle.ActiveSet`, the same
declarative style as the single-kernel :func:`~eagle.run_until_done`
convenience, one level below it.
