C3 — Reduce an array, the same way on either backend
====================================================

    Sum an aether SoA array on the GPU with a multi-level CUDA reduction, or
    on the CPU with a cache-padded OpenMP one -- same call, same answer.

**Time:** ~2 min · **Runs on:** CUDA GPU or CPU (OpenMP) · **You need:** C2

In CUDA mode, ``eagle::cuda::Reduction`` runs a multi-level GPU reduction over
the uploaded array:

.. literalinclude:: ../../../examples/03_reduction.cu
   :language: cuda
   :start-after: // [cell:reduce]
   :end-before: // [cell:reduce:end]

In pure-C++ mode, ``eagle::cpu::Reduction`` runs the cache-padded OpenMP
reduction over the host-resident array -- same operator, same result:

.. literalinclude:: ../../../examples/03_reduction.cpp
   :language: cpp
   :start-after: // [cell:reduce]
   :end-before: // [cell:reduce:end]

What it printed
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/03_reduction.cuda.expected.txt
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
:doc:`c4_embed_a_kernel` -- the other side of eagle: a binary ABI a C++ host can load
with no aether headers, no Python, and no code generator at build or run
time. deeper: :doc:`../cuda_graphs` for the concepts behind capture/replay, and
the :doc:`../../modules/module_reduce` / :doc:`../../modules/module_host` module
guides for the reduction and host APIs.
