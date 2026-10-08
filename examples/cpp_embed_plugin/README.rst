C++ embedding reference
========================

This directory is the **stable entry point** for embedding a generated plugin
deployment in a C++/CUDA host through nothing but eagle's binary ABI
(``plugin/*.h``) -- no nvcc-at-deploy-time knowledge, no
Python, no code generator dependency. It is cross-referenced from the userguide examples gallery
(``eagle/docs/content/userguide/examples.rst``) and from the code generator's extension
cookbook as *the* C++ embedding walkthrough, so its name and layout are meant
to stay put.

.. note::

   This README is a plain repo-browsing doc, not part of the Sphinx build (it
   lives outside ``docs/``). The Sphinx-rendered write-up of this example is
   in ``eagle/docs/content/userguide/examples.rst``.

``embedding_demo.cu`` is a standalone target (its own ``CMakeLists.txt``, its own
``fixtures/``) that loads a 3-plugin manifest through
``eagle::cuda::PluginRegistry`` and captures all three as nodes in one CUDA
graph opening -- the same independent-registry idiom the demos under
``plugin/`` use, but built to close two gaps nothing else in the repo
exercised:

1. **A matrix (``mat_in``) input**, bound through the device registry's
   ``bind_matrix`` (``plugin/plugin_registry/registry.h``) -- added
   alongside this example. ``fixtures/mattrace.cu`` is
   ``out[i] = trace(M[i])`` over a batch of 3x3 matrices, the device twin of
   the CPU-side ``fixtures/host_plugin_mattrace.cpp`` fixture
   (``eagle/tests/``).
2. **A derivative (VJP) artifact.** ``fixtures/toy_energy.cu`` is the primal
   ``e = p * exp(-0.5 * |x|^2)``; ``fixtures/toy_energy_vjp.cu`` is its custom
   VJP, carrying the optional ``derivative`` sidecar block
   (``plugin/sidecar.h``: ``{kind: "vjp", primal: "toy_energy", wrt: ["x",
   "p"], residual_policy: "recompute"}``). Both are hand-written
   reimplementations of the closed form the code generator's own downstream
   extension tests use to prove the same claim on the Python
   producer/consumer side -- reimplemented here so this example never
   imports the code generator. The demo checks the deployed VJP's output
   against a **central-difference** reference of the host closed form (a reference
   independent of the analytic formula the CUDA kernel itself evaluates), not
   just against a copy of the same formula.

Why this lives here and not in ``plugin/``
-------------------------------------------

``plugin/`` already holds three demos --
``plugin_host.cpp`` (a pure Driver-API loader, no nvcc needed to build it),
``plugin/graph_inject/inject_demo.cu`` (multi-plugin injection into a
host-owned graph opening), and
``plugin/pure_inject/pure_inject_demo.cu`` (the pure/Mutable RMW twin) -- all
built exclusively by ``eagle/python/tests/cpp_demo.py::build()``, which every
pytest in ``test_cpp_host.py`` / ``test_plugin_registry.py`` /
``test_gref_layout.py`` / ``test_handle_layout.py`` depends on for a fresh
``cmake -S/-B`` per run, and which ``test_reserved_words.py`` documents by
path. Relocating those three into ``examples/`` would have meant updating
every one of those references (plus ``tests/CMakeLists.txt``'s own
``plugin/plugin_registry/registry.h`` / ``plugin/host_registry.h`` includes and
the generated Doxygen cross-refs) for a cosmetic move, with real risk of
breaking that live test infrastructure for no functional gain. So this
directory **packages new content here and cross-references the existing three without
moving them** -- this directory adds capability (matrix + derivative), it does
not relocate or duplicate what ``plugin/`` already demonstrates.

Building and running
---------------------

Standalone, like ``plugin/graph_inject`` and ``plugin/pure_inject``:

.. code-block:: bash

   export PATH=/usr/local/cuda-12.6/bin:$PATH   # nvcc must resolve to the
                                                  # driver-supported toolchain
   cmake -S examples/cpp_embed_plugin -B /tmp/embed_demo_build \
         -DCMAKE_BUILD_TYPE=Release
   cmake --build /tmp/embed_demo_build -j4
   /tmp/embed_demo_build/embedding_demo            # uses the baked-in fixtures/
   # or: /tmp/embed_demo_build/embedding_demo <artifact_dir> <N>

The build compiles the three ``fixtures/*.cu`` kernels to raw PTX (loaded at
run time via ``cuModuleLoadData``, exactly like a real generated-plugin deploy) and links
``embedding_demo.cu`` against the CUDA driver stub, mirroring
``plugin/pure_inject/CMakeLists.txt``.

Expected output
----------------

::

   N_PLUGINS=3
   ACTIVE=3
   INJECTED=3
   TOTAL_NODES=3
   MATTRACE_MAX_REL=0.000e+00
   PRIMAL_MAX_REL=<< 1e-12
   VJP_X_MAX_REL_VS_FD=<< 1e-5
   VJP_P_MAX_REL_VS_FD=<< 1e-5
   PASS

``MATTRACE_MAX_REL`` / ``PRIMAL_MAX_REL`` compare the device kernels' output
against exact host closed-form arithmetic (tolerance ``1e-12``);
``VJP_*_MAX_REL_VS_FD`` compares the custom VJP's output against an
independent central-difference reference (``h = 1e-6``, tolerance ``1e-5`` --
the same house tolerance ``test_downstream_toy_extension_deploy.py`` uses for
its numpy finite-difference check).

See also
--------

* ``docs/content/devguide/plugin_schema.rst`` -- the full schema-v1
  ``arg_spec`` role vocabulary and the C++ reference-consumer scope boundary
  (what the device/host registries do and do not read from a sidecar).
* ``plugin/plugin_registry/registry.h`` -- ``eagle::cuda::PluginRegistry``,
  including ``bind_matrix``.
* ``plugin/host_registry.h`` -- the CPU twin (``bind_matrix`` landed there
  first; this example proves the device path now matches it).
