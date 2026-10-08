Examples
========

A gallery of complete, self-contained EAGLE programs. Each lives under
``docs/examples/`` and is compiled and run by the ``exampletest`` gate
(``docs/examples/run_examples.sh``) in every mode it supports, so the sources
here are guaranteed to build and produce the output the :doc:`tutorials` show.

.. list-table::
   :header-rows: 1
   :widths: 26 16 22 36

   * - Example
     - Mode
     - Modules
     - What it shows
   * - ``01_capture_replay.cu``
     - CUDA
     - graph
     - Capture a launch, build a graph, instantiate a ``Launcher``, replay it.
   * - ``02_host_dispatch.{cu,cpp}``
     - CUDA + C++
     - host
     - One task per index through the OpenMP + SIMD ``cpu::Host`` launcher.
   * - ``03_reduction.{cu,cpp}``
     - CUDA + C++
     - reduce
     - Sum an aether SoA array with the GPU or the OpenMP reduction.

Full sources
------------

.. dropdown:: 01_capture_replay.cu
   :icon: code

   .. literalinclude:: ../../examples/01_capture_replay.cu
      :language: cuda
      :linenos:

.. dropdown:: 02_host_dispatch.cu  (CUDA-mode twin)
   :icon: code

   .. literalinclude:: ../../examples/02_host_dispatch.cu
      :language: cpp
      :linenos:

.. dropdown:: 03_reduction.cu  /  03_reduction.cpp
   :icon: code

   .. literalinclude:: ../../examples/03_reduction.cu
      :language: cuda
      :linenos:

   .. literalinclude:: ../../examples/03_reduction.cpp
      :language: cpp
      :linenos:

Running the gate
----------------

.. code-block:: bash

   # from the repo root, with eagle + aether installed into $CONDA_PREFIX
   cd docs && make exampletest        # or: CONDA_PREFIX=<env> ./examples/run_examples.sh

The gate compiles each ``*.cu`` with ``nvcc`` (CUDA mode) and each ``*.cpp``
with ``g++`` (pure-C++ / OpenMP mode), runs the resulting programs, and diffs
their stdout against the committed ``*.expected.txt`` files. Adding an example
is just dropping a new ``NN_topic.cu`` (and, for a dual-mode example, its
``.cpp`` twin) plus the matching ``.expected.txt`` — the gate and this gallery
pick it up by its numeric prefix.

C4 — Embed a compiled kernel in your own C++ program
------------------------------------------------------

    Load a compiled plugin manifest straight into a C++/CUDA host through
    EAGLE's binary ABI — no aether headers, no Python, no code generator at
    build or run time.

**Time:** ~5 min · **Runs on:** CUDA GPU only · **You need:** a CUDA
toolchain, CMake (its own standalone build, separate from C1–C3's
``exampletest`` gate above)

The examples above show EAGLE's own core facilities (graph, host, reduce). A
separate, larger example shows the *other* side of EAGLE: **embedding** a
plugin deployment in a C++/CUDA host through nothing but EAGLE's binary ABI
(``plugin/*.h``) — no aether headers, no Python, no code generator at build
or run time. It lives under ``examples/cpp_embed_plugin/`` (its
own standalone CMake project, not part of the numbered ``exampletest`` gate
above, because it stages a multi-artifact manifest + fixture PTX rather than
compiling one ``NN_topic.cu``) and is the stable C++ embedding entry point
for deploying a plugin this way.

``embedding_demo.cu`` loads a 3-plugin manifest through
``eagle::cuda::PluginRegistry`` and captures all three as nodes in one CUDA
graph opening, proving two capabilities together:

* a matrix (``mat_in``) input, bound through the device registry's
  ``bind_matrix`` (``plugin/plugin_registry/registry.h``) — a batch
  ``out[i] = trace(M[i])`` over 3x3 matrices;
* a derivative (VJP) artifact carrying the optional ``derivative`` sidecar
  block (``plugin/sidecar.h``) — a primal ``e = p * exp(-0.5*|x|^2)`` and its
  custom VJP, checked against a host **central-difference** reference (a
  check independent of the analytic formula the VJP kernel itself
  evaluates), the same shape of check
  the code generator's own test suite runs on the Python producer/consumer
  side.

See ``examples/cpp_embed_plugin/README.rst`` for the full
walkthrough (why it is a separate directory from ``plugin/``'s three
Driver-API demos, build/run instructions, expected output) and
:doc:`../devguide/plugin_schema` for the ``mat_in`` role and the
``derivative`` block's schema.

.. dropdown:: embedding_demo.cu
   :icon: code

   .. literalinclude:: ../../../examples/cpp_embed_plugin/embedding_demo.cu
      :language: cuda
      :linenos:

What just happened
^^^^^^^^^^^^^^^^^^^

- ``eagle::cuda::PluginRegistry`` loaded three independently compiled
  artifacts from one manifest and captured all three as nodes in a single
  CUDA graph opening — the host program never recompiles or links against
  the kernels themselves.
- The matrix input (``mat_in``) crossed the ABI boundary through the
  registry's own ``bind_matrix``, with no aether header on the C++ side.
- The VJP artifact's output matched a central-difference reference built
  from the host's own closed form — a check independent of the analytic
  formula the deployed kernel evaluates, not a second copy of the same
  formula.

Try this
^^^^^^^^

Pass a different ``<artifact_dir>`` on the command line (the README shows
the baked-in default): the same host program loads whatever manifest it
finds there, with no rebuild.

Next
^^^^

:doc:`tutorials` — the three smaller C++ patterns (capture/replay, host
dispatch, reduction) this example builds on. deeper:
:doc:`../devguide/plugin_schema` — the full manifest/sidecar schema
``embedding_demo.cu`` reads.
