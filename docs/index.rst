.. eagle documentation master file

.. raw:: html

   <div class="raptor-hero">
     <img class="raptor-reveal dark-light" src="_static/brand/wide_family_eagle.svg" alt="eagle">
   </div>

eagle
=====

**One loop, recorded once, replayed until every sample is done.** eagle
runs a kernel over any number of samples — one or a million — without a
launch per sample: write the step once, and eagle captures it into a
single replayable launch, a CUDA graph on the GPU or an OpenMP + SIMD host
loop on the CPU, picked automatically from the data you pass in. A PyTorch
bridge lets that same kernel sit inside ``loss.backward()`` like any other
layer. (The name stands for *Extensible Adaptive Graph Launch Engine*, if
you're curious.) A header-only C++ core; a Python package over the same
engine.

.. raw:: html
   :file: _static/ecosystem/ecosystem_cards_eagle.html

Thirty seconds: ``eagle.simulate``
----------------------------------

Write one sample's step as a `hawk <https://amasat01.github.io/hawk/>`__
kernel, hand it to :func:`eagle.simulate`, and it runs every sample until
it is done — on the CPU or the GPU, no loop of your own:

.. code-block:: python

   @hawk.kernel
   def oscillator(omega: Scalar, t_end: Param, dt: Param, terminated: Terminated,
                  x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
       x0, v0 = x, v
       x = x0 + dt * v0
       v = v0 - dt * omega * omega * x0
       t += dt
       terminated = t >= t_end

   result = eagle.simulate(oscillator, omega=omegas, t_end=1.0, dt=1e-3,
                           x=x0, v=v0, t=t0, max_steps=10_000)

Pass the kernel's own arguments as you would call it: an array gives each
sample its own value, a plain number is shared by every sample.

The full, runnable version is the :doc:`Python quickstart
<content/userguide/quickstart_python>` — start there.

Which call do I use?
---------------------

.. list-table::
   :header-rows: 0
   :widths: 60 40

   * - One kernel (or a step's worth), run until every sample finishes
     - :func:`eagle.simulate`
   * - A plan already paced by its own finish kernel, one call
     - :func:`eagle.until_done`
   * - Build a kernel into a plan that runs where its data lives
     - :func:`eagle.deploy` / :mod:`eagle.plan`
   * - Resolve a launchable by name, then call/stream/graph it
     - :class:`eagle.KernelRegistry`

This is ``eagle``'s own module docstring — the same four lines ``help(eagle)``
or ``eagle.__dir__()`` leads with. The full reference, every other call
eagle owns, is :doc:`content/api/api_python`.

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: :octicon:`terminal` C++ users start here
      :link: content/userguide/foundations
      :link-type: doc

      eagle is a header-only C++23 library first: capture a CUDA graph,
      dispatch dual-mode CPU/GPU work, and more, with no Python runtime at
      all. :doc:`Foundations <content/userguide/foundations>` defines the
      vocabulary (``Stream``, ``Graph``, ``Launcher``); the
      :doc:`C++ quickstart <content/userguide/quickstart>` and
      :doc:`tutorials <content/userguide/tutorials>` build up from there —
      record a launch once, assemble it into a CUDA graph, and replay it,
      the same pattern ``eagle.simulate`` builds on in Python.

Where eagle sits
-----------------

eagle is the execution layer of the RAPTOR family — it launches what
`aether <https://amasat01.github.io/aether/>`__ lays out in memory, and what
`raptor <https://amasat01.github.io/raptor/>`__'s manifest schema describes:

.. code-block:: text

   aether (array layout)  --->  eagle (THIS: launch, graphs, host dispatch)  <---  raptor (manifest / interop contracts)
                                          ^
                                          |
                                        hawk (kernel compiler, consumes eagle to run what it compiles)

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: :octicon:`desktop-download` Install
      :link: content/userguide/installation
      :link-type: doc

      C++ (header-only, CMake) and Python (``pip install "raptor-eagle[cuda12]"``, or ``[cuda13]``).

   .. grid-item-card:: :octicon:`rocket` Python quickstart
      :link: content/userguide/quickstart_python
      :link-type: doc

      ``eagle.simulate``, end to end, in under a minute. Start here.

   .. grid-item-card:: :octicon:`mortar-board` Tutorials
      :link: content/userguide/tutorials
      :link-type: doc

      Build up eagle's core patterns step by step: graphs, launch cadence,
      compaction, multi-kernel steps, training through a kernel — in
      Python, with a C++ track alongside.

   .. grid-item-card:: :octicon:`code-square` API reference
      :link: content/api/api_reference
      :link-type: doc

      Complete C++ (Doxygen/breathe) and Python (autosummary) API reference.

.. toctree::
   :maxdepth: 1
   :caption: Start here
   :hidden:

   content/userguide/installation
   content/userguide/quickstart_python

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   content/userguide/tutorials/02_cuda_graphs_python
   content/userguide/tutorials/05_steps_per_launch
   content/userguide/tutorials/06_compaction_and_reorder
   content/userguide/tutorials/04_multi_kernel_workflow
   content/userguide/tutorials/07_host_or_device
   content/userguide/tutorials/03_torch_training
   content/userguide/tutorials/08_profile_your_run
   C++ quickstart <content/userguide/quickstart>
   C++ tutorials <content/userguide/tutorials>

.. toctree::
   :caption: Extra tutorials
   :hidden:

   Fork/join graph <content/userguide/extra/fork_join_graph>

.. toctree::
   :maxdepth: 1
   :caption: How-to guides
   :hidden:

   content/userguide/tutorials/01_launch_and_registry
   content/userguide/examples
   content/userguide/examples/01_device_properties

.. toctree::
   :maxdepth: 1
   :caption: Interoperability
   :hidden:

   content/interop
   content/interop_contract

.. toctree::
   :maxdepth: 1
   :caption: Explanation
   :hidden:

   content/userguide/foundations
   content/userguide/parallel_execution
   content/userguide/cuda_graphs
   content/userguide/build_options
   content/modules/modules
   content/modules/module_graph
   content/modules/module_filtering
   content/modules/module_reduce
   content/modules/module_host
   content/modules/module_util
   content/performance

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   content/api/api_reference
   content/api/api_graph
   content/api/api_filtering
   content/api/api_reduce
   content/api/api_host
   content/api/api_util
   content/api/api_python

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :hidden:

   content/contributing
   content/devguide/dev_guide
   content/devguide/plugin_schema
   content/devguide/host-plugin-parity
   content/devguide/test-map
   content/devguide/roadmap
   content/devguide/docs_guide

The RAPTOR family
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Repository
     - Role
   * - `raptor <https://github.com/amasat01/raptor>`__
     - The shared contracts: manifest schema, engine/launcher/provider
       contracts, the interop certification matrix. No hard dependencies.
   * - `aether <https://github.com/amasat01/aether>`__
     - Zero-cost array layout and GPU memory management (SoA containers).
   * - **eagle** (this repository)
     - GPU execution layer: kernel launch, graph capture, plugin registry,
       host/device dual mode.
   * - `hawk <https://github.com/amasat01/hawk>`__
     - Symbolic/numeric DSL and kernel compiler: hawk compiles kernels
       eagle runs; neither imports the other.
