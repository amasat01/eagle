Tutorials
=========

.. contents:: On this page
   :local:
   :depth: 1

These tutorials build up EAGLE's core patterns as small, complete C++ programs
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

C1 — Capture a launch once, replay it many times
---------------------------------------------------

    Record a stream of kernel launches once, then replay the recorded graph
    as many times as you like -- with no per-launch API overhead.

**Time:** ~3 min · **Runs on:** CUDA GPU only (graph capture has no pure-C++
analogue) · **You need:** a CUDA toolchain, the build command above

The kernel we want to run:

.. literalinclude:: ../../examples/01_capture_replay.cu
   :language: cuda
   :start-after: // [cell:kernel]
   :end-before: // [cell:kernel:end]

Record the launch into a graph. Everything issued on the stream between
``begin()`` and ``end()`` is captured rather than executed eagerly, then handed
to the ``Graph`` as a node:

.. literalinclude:: ../../examples/01_capture_replay.cu
   :language: cuda
   :start-after: // [cell:capture]
   :end-before: // [cell:capture:end]

Instantiate the graph once and replay it -- each ``launch()`` is a single
``cudaGraphLaunch``:

.. literalinclude:: ../../examples/01_capture_replay.cu
   :language: cuda
   :start-after: // [cell:replay]
   :end-before: // [cell:replay:end]

What it printed
^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/01_capture_replay.cuda.expected.txt
   :language: text

What just happened
^^^^^^^^^^^^^^^^^^^

- ``StreamCapturer`` recorded ``addOne``'s launch instead of running it: the
  ``cudaGraph_t`` it produced between ``begin()``/``end()`` became one node on
  ``graph``.
- ``graph.launcher()`` instantiated that graph exactly once into a
  ``Launcher``; every later ``launch()`` replays the same instantiated graph
  as a single driver call, not a fresh launch of ``addOne``.
- 10 replays of a kernel that adds 1 left every element at 10 -- the proof
  that a replay really re-issues the captured work and not a no-op.

Try this
^^^^^^^^

Raise ``REPLAYS`` to 1000 and rebuild: the loop issuing ``launcher.launch()``
still costs one driver call per iteration, but nothing about the capture
above changes -- the graph itself does not grow.

Next
^^^^

C2, below -- the same replay idea, but for work that also has a pure-C++ /
OpenMP side with no CUDA at all.

C2 — Run one loop on the CPU or the GPU, unchanged
------------------------------------------------------

    The same source file compiles against CUDA and against plain
    C++/OpenMP -- one dispatch call, two backends, no ``#ifdef`` in the body.

**Time:** ~3 min · **Runs on:** CUDA GPU or CPU (OpenMP) -- pick the build
command · **You need:** C1 (the capture/replay vocabulary), a C++23 compiler

``eagle::cpu::Host::launch`` runs one task per index through EAGLE's OpenMP +
SIMD host launcher: consecutive indices are batched into aether SIMD packets and
spread across threads. The *same source* compiles and runs in CUDA mode
(``02_host_dispatch.cu``) and pure-C++ mode (``02_host_dispatch.cpp``) -- that is
EAGLE's dual-mode promise.

.. literalinclude:: ../../examples/02_host_dispatch.cu
   :language: cpp
   :start-after: // [cell:launch]
   :end-before: // [cell:launch:end]

What it printed
^^^^^^^^^^^^^^^

Both builds print the same line (the prefix sums of the first *N* odd numbers
are *N*\ :sup:`2`):

.. literalinclude:: ../../examples/02_host_dispatch.cpp.expected.txt
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

C3, below -- the same dual-mode idea applied to a reduction instead of a
per-element write.

C3 — Reduce an array, the same way on either backend
---------------------------------------------------------

    Sum an aether SoA array on the GPU with a multi-level CUDA reduction, or
    on the CPU with a cache-padded OpenMP one -- same call, same answer.

**Time:** ~2 min · **Runs on:** CUDA GPU or CPU (OpenMP) · **You need:** C2

In CUDA mode, ``eagle::cuda::Reduction`` runs a multi-level GPU reduction over
the uploaded array:

.. literalinclude:: ../../examples/03_reduction.cu
   :language: cuda
   :start-after: // [cell:reduce]
   :end-before: // [cell:reduce:end]

In pure-C++ mode, ``eagle::cpu::Reduction`` runs the cache-padded OpenMP
reduction over the host-resident array -- same operator, same result:

.. literalinclude:: ../../examples/03_reduction.cpp
   :language: cpp
   :start-after: // [cell:reduce]
   :end-before: // [cell:reduce:end]

What it printed
^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/03_reduction.cuda.expected.txt
   :language: text

What just happened
^^^^^^^^^^^^^^^^^^^

- ``Reduction`` took the same aether array and the same combine operator on
  both backends; only the class name (``eagle::cuda::Reduction`` vs.
  ``eagle::cpu::Reduction``) differs between the two source files.
- The GPU build reduces in multiple levels (block, then grid); the CPU build
  pads each OpenMP thread's partial sum to its own cache line to avoid false
  sharing -- two different implementations reaching the same ``sum(1..1000)``.
- Neither build re-sums on the host afterwards: the reduction itself is the
  answer, not a seed for a second pass.

Try this
^^^^^^^^

Change the summed range from 1..1000 to 1..1 and rebuild: the expected total
becomes 1 on both backends, the smallest possible reduction.

Next
^^^^

C4 -- embedding a compiled plugin in your own C++ program, in
:doc:`examples` -- the other side of EAGLE: a binary ABI a C++ host can load
with no aether headers, no Python, and no code generator at build or run
time. deeper: :doc:`cuda_graphs` for the concepts behind capture/replay, and
the :doc:`../modules/module_reduce` / :doc:`../modules/module_host` module
guides for the reduction and host APIs.

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
