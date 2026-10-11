C1 — Capture a launch once, replay it many times
================================================

    Record a stream of kernel launches once, then replay the recorded graph
    as many times as you like -- with no per-launch API overhead.

**Time:** ~3 min · **Runs on:** CUDA GPU only (graph capture has no pure-C++
analogue) · **You need:** a CUDA toolchain, the build command above

The kernel we want to run:

.. literalinclude:: ../../../examples/01_capture_replay.cu
   :language: cuda
   :start-after: // [cell:kernel]
   :end-before: // [cell:kernel:end]

Record the launch into a graph. Everything issued on the stream between
``begin()`` and ``end()`` is captured rather than executed eagerly, then handed
to the ``Graph`` as a node:

.. literalinclude:: ../../../examples/01_capture_replay.cu
   :language: cuda
   :start-after: // [cell:capture]
   :end-before: // [cell:capture:end]

Instantiate the graph once and replay it -- each ``launch()`` is a single
``cudaGraphLaunch``:

.. literalinclude:: ../../../examples/01_capture_replay.cu
   :language: cuda
   :start-after: // [cell:replay]
   :end-before: // [cell:replay:end]

What it printed
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../examples/01_capture_replay.cuda.expected.txt
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

:doc:`c2_cpu_or_gpu` -- the same replay idea, but for work that also has a pure-C++ /
OpenMP side with no CUDA at all.
