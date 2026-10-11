Examples
========

A gallery of complete, self-contained eagle programs. Each lives under
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

Embedding a compiled kernel
---------------------------

The examples above are eagle's own core facilities. The other side -- loading
a compiled plugin into your own C++/CUDA program through eagle's binary ABI --
is a separate, larger example: :doc:`cpp/c4_embed_a_kernel`.

.. toctree::
   :maxdepth: 1
   :hidden:

   cpp/c4_embed_a_kernel
