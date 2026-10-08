Documentation guide
===================

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

eagle uses a two-layer documentation system:

* **Doxygen** — parses C++/CUDA headers and generates XML from Doxygen
  docstrings (``/** … */`` blocks).
* **Sphinx** — builds the HTML site from ``.rst`` files, pulling in the
  Doxygen XML via the **Breathe** extension.

The documentation source lives in ``docs/``.  HTML output goes to
``docs/_build/html/``.

Building the docs
-----------------

.. code-block:: bash

   cd docs/
   make doxygen html    # Doxygen XML, then the full Sphinx build
   make strict          # same, with -W --keep-going (0 warnings required)
   make nbexec          # execute every tutorial/example notebook IN PLACE (needs a GPU)
   make nbcheck         # refuse the build unless every notebook cell has run
   make linkcheck        # crawl every link 
   make livehtml         # watch + auto-rebuild (requires watchdog)
   make clean            # wipe _build/ and _doxybuild/

``make linkcheck`` crawls both internal and external links.

Open ``_build/html/index.html`` to view the result. Notebooks under
``content/userguide/tutorials/``, ``content/userguide/examples/`` and
``content/interop.ipynb`` are **executed locally** (``make nbexec``) and
committed with their outputs — the Sphinx build itself never re-executes
them (``nb_execution_mode = "off"`` in ``conf.py``); ``make nbcheck`` is the
gate that catches a notebook committed without having been (re-)run.

Dependencies
------------

A lightweight conda environment ships as ``environment-light.yml``:

.. code-block:: bash

   micromamba create -p .envs/eagle_docs -f environment-light.yml -y

Or install manually:

.. code-block:: bash

   pip install sphinx sphinx-book-theme sphinx-design sphinx-copybutton \
       sphinx-autodoc-typehints myst-nb breathe nbformat nbclient

Doxygen must also be on ``PATH``. Building the Python API reference needs the
``eagle`` package importable (``pip install -e ./python``, see
:doc:`../userguide/installation`); executing the notebooks additionally needs
a CUDA GPU and ``cupy``.

Writing docstrings
------------------

The project uses ``/** … */`` for documented public symbols. The one sanctioned
exception is ``/// @cond INTERNAL`` / ``/// @endcond``, used to exclude
internal-only code from Breathe's extraction — see :doc:`dev_guide` for where
that pattern applies.

**Short (single-line) form:**

.. code-block:: cpp

   /** @brief Return the raw cudaStream_t handle. */
   const cudaStream_t& cuda() const;

**Long form (detail block):**

.. code-block:: cpp

   /**
    * @brief Re-tune the launch geometry of the instantiated graph.
    *
    * @param size  New logical size (active-sample count) for the replay.
    * @note Does not rebuild the graph; only updates node launch parameters.
    */
   void setLogicalSize(const idx_t& size);

Tag order: ``@brief`` → ``@tparam`` → ``@param`` → ``@return`` →
``@throws`` → ``@pre`` / ``@post`` → ``@note`` → ``@see``.

Hide internal helpers from Breathe with ``/// @cond INTERNAL`` /
``/// @endcond``, or place them under a ``detail/`` sub-directory (excluded in
``Doxyfile.in`` via ``EXCLUDE_PATTERNS``).

Adding a new page
-----------------

1. Create ``docs/content/<subfolder>/mypage.rst``.
2. Add to the appropriate toctree in ``docs/index.rst`` or the
   section landing page.
3. Use relative ``:doc:`` paths for cross-folder references:

   .. code-block:: rst

      :doc:`../api/api_graph`    ← from modules/ to api/

4. Add a ``.. seealso::`` footer if the page has natural neighbours.
5. Add a ``.. contents::`` local TOC at the top if the page is long.
6. Run ``make doxygen html`` and check for Sphinx warnings.

Breathe directives
------------------

.. code-block:: rst

   .. doxygennamespace:: eagle::cuda
      :project: EAGLE
      :members:

   .. doxygenfile:: typedefs.h
      :project: EAGLE

   .. doxygenclass:: eagle::cuda::Graph
      :project: EAGLE
      :members:
