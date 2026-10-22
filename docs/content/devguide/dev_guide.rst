Developer guide
===============

How to build eagle, run its tests, find your way around the source and add the three things contributors add most:
an error code, a Python feature, a docs notebook. The pull-request rules are in :doc:`../contributing`.

Build and gate
--------------

The commands are in :doc:`../contributing`. Two points worth knowing before you run them:

* The C++ suite runs through ``tests/check_gate.sh <cpu|cuda> <binary>``, never the bare binary: an empty run
  exits 0. The script compares the test listing and the run against the committed name manifest
  ``tests/expected_tests_<mode>.txt``. Added or removed a test? Re-mint the manifest from your build and commit the
  diff with the test.
* The Python suite is ``pytest python/tests -m "not gpu and not interop_matrix"`` on a machine without a GPU.

Where things live
-----------------

``eagle/`` is the header-only C++ core, one folder per idea: ``cuda/`` (stream, capture, graph, launcher),
``cpu/`` (the host twin of ``cuda/``), ``graph/`` (the picture ``print(graph)`` draws), ``exec/`` (launch geometry,
the host team), ``filtering/`` (compaction, reorder, scans), ``reduce/``, ``conditional/``, ``diff/``, ``events/``,
``policy/`` (host or device) and ``util/`` (errors, threads).

``python/eagle/`` is the Python package, grouped by file family: ``graph.py`` and ``_graph_*.py`` (``eagle.Graph``),
``plan.py`` and ``_plan_*.py`` (deployed plans), ``_simulate.py`` and ``_until_done.py`` (the runners),
``registry.py``, ``loaded.py`` and ``sidecar.py`` (plugins), ``diff/``, ``frameworks/`` and ``gemm/``.
``python/src/`` holds the nanobind binding; its seam is versioned (see :doc:`../contributing`).

``plugin/`` is the C++ plugin registry. ``tests/`` holds the C++ tests, ``python/tests/`` the Python ones.

Tests
-----

The test files are many and change often, so this guide does not list them; the manifests under ``tests/`` are the
record. Two things hold the design together:

* **The gate** (above) makes a missing or swapped test a failure instead of a smaller number.
* **The conformance corpus**, ``tests/conformance/``, is a set of language-neutral fixtures, each with an expectation
  file. Two drivers read it, ``tests/test_ConformanceCorpus.cpp`` and ``python/tests/test_conformance_corpus.py``, and
  must give the same outcome, so the C++ and Python validators cannot drift apart.

Host and plugin parity between the CUDA and CPU paths is tracked cell by cell in :doc:`host-plugin-parity`, and
``python/tests/test_pipeline_parity.py`` keeps the closed cells closed.

Add a thing
-----------

**An error code.** Add an enumerator (a power of two) to ``eagle::err::EagleErrorTypes`` and its message to
``eagle::err::errorMessages`` in ``eagle/util/DeviceError.h``. Raise it with ``EAGLE_GPU_THROW`` or
``EAGLE_GPU_ASSERT``.

**A Python feature.** The behaviour goes into the C++ core first, with a test; the Python side only exposes it. If the
feature validates a manifest or sidecar, add a fixture and its expectation file to ``tests/conformance/`` so both
drivers check it. A new Python feature also gets an ``ft`` test (run with ``pytest -m ft`` on a free-threaded Python).
If it needs a new backend symbol, bump ``EAGLE_BACKEND_VERSION_MINOR`` and add a ``CHANGELOG.md`` line.

**A docs notebook.** Put it under ``docs/content/``, give it a title, a time and a "runs on" line, run ``make nbexec``
to store its outputs, then ``make nbcheck``. The notebook rules are in :doc:`docs_guide`. For a page about how eagle
works, :doc:`graph_capture_replay` is the model: one printed result per stage, the same text a test compares.
