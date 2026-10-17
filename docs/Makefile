# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Minimal makefile for the eagle Sphinx + Doxygen documentation.

# `python -m sphinx`, not the bare `sphinx-build` script: in a layered env
# (e.g. a lightweight venv over a conda env) the `sphinx-build` console
# script on PATH can resolve to a DIFFERENT interpreter than the one that
# has `eagle` (and its `raptor` dependency) importable, which breaks the
# Python API autosummary/autodoc build silently-until-it-isn't. `python -m`
# always runs under the interpreter PATH resolves `python` to.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build
DOXYBUILDDIR  = _doxybuild

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile doxygen clean livehtml exampletest strict linkcheck nbexec nbcheck nbcheck-hosted nbcheck-floor nbcheck-hosted-gpu

doxygen:
	@EAGLE_VERSION=$$(python3 -c "import re, sys; m = re.search(r'project\s*\(\s*eagle\s+VERSION\s+(\d+\.\d+\.\d+)', open('../CMakeLists.txt').read(), re.IGNORECASE); print(m.group(1)) if m else sys.exit('doxygen: could not read eagle version from ../CMakeLists.txt')") && \
	EAGLE_VERSION="$$EAGLE_VERSION" doxygen Doxyfile.in

# Compile and run every docs example in its supported mode(s) and diff the
# output against the committed expectations. This is the "tested C++ cells"
# gate; the tutorials/examples pages literalinclude these same sources. Needs a
# CUDA toolchain (+ GPU) for the .cu builds; run before publishing docs.
exampletest:
	CONDA_PREFIX=$${CONDA_PREFIX:?set CONDA_PREFIX to the env with eagle+aether} \
	    ./examples/run_examples.sh

# The CI/local integrity gate: 0 Sphinx warnings allowed (-W), but keep going
# past the first one so a single run reports every warning. Builds the
# Doxygen XML first (standalone-usable: `make strict` alone is enough, no
# separate `make doxygen` step needed) so every `breathe` directive and every
# Python `autosummary` import resolves.
strict: doxygen
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Crawl every link (internal + external). External links under
# amasat01.github.io/* 404 until the family's GitHub Pages flip — expected,
# not a failure; see docs/content/devguide/docs_guide.rst.
linkcheck: doxygen
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

# Execute every tutorial/example/interop notebook IN PLACE (outputs kept) —
# run on a GPU before committing a notebook change. The filled notebooks are
# what gets committed; the Sphinx build never re-executes them
# (`nb_execution_mode = "off"` in conf.py).
NB_PATHS = content/userguide/quickstart_python.ipynb content/userguide/tutorials \
           content/userguide/examples content/userguide/extra content/interop.ipynb content/interop

nbexec:
	python _tools/nb_execute.py $(NB_PATHS)

# Refuse to proceed unless every code cell in every notebook above has
# actually run (an execution_count, not merely static source).
#
# nbcheck ALSO executes every notebook, in memory: on the CPU always, and again
# on the GPU when one is present (_tools/nb_run_gate.py; a notebook declares
# "gate" metadata for what it needs, and a GPU-only one skips cleanly on the CPU
# leg). A committed notebook can no longer outlive the code that produced it.
nbcheck:
	python _tools/nb_colab_setup.py --check
	python _tools/nb_check_filled.py $(NB_PATHS)
	python _tools/nb_run_gate.py $(NB_PATHS)

clean:
	rm -rf $(BUILDDIR) $(DOXYBUILDDIR)

# Watch for changes and auto-recompile (requires watchdog: pip install watchdog)
livehtml: Makefile
	watchmedo shell-command -p "*.rst;*.md;*.py" -R \
	    -c "make html" \
	    -i "$(BUILDDIR)/*;$(DOXYBUILDDIR)/*" \
	    --debug-force-polling .

# Catch-all target: route unknown targets to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Run every docs notebook as a hosted (Colab) user does: wheels built from the checkouts, a pinned environment,
# the setup cell included (_tools/nb_hosted.py). Needs uv, a Python 3.13 and a C++23 compiler (CXX).
NBH_PYTHON ?= python3.13
NBH_ARGS   ?=
nbcheck-hosted:
	python _tools/nb_hosted.py --python $(NBH_PYTHON) --lock colab --repos eagle --eagle $(abspath ..) $(NBH_ARGS)

# The same against the oldest numpy and torch that have Python 3.13 wheels (_tools/hosted_env/floor-cpu.txt).
nbcheck-floor:
	python _tools/nb_hosted.py --python $(NBH_PYTHON) --lock floor --repos eagle --eagle $(abspath ..) $(NBH_ARGS)

# The GPU runtime's lock (CUDA 12 builds: see the header of _tools/hosted_env/colab-gpu.txt): eagle's CUDA plugin is built for
# NBH_GPU_ARCH, the GPU-only notebooks run too, on GPU NBH_GPU_INDEX, under the flock file NBH_GPU_LOCK if set. Needs ~5 GB free.
NBH_GPU_ARCH  ?= 61
NBH_GPU_INDEX ?= 1
NBH_GPU_LOCK  ?=
nbcheck-hosted-gpu:
	python _tools/nb_hosted.py --python $(NBH_PYTHON) --lock colab-gpu --gpu --cuda-arch $(NBH_GPU_ARCH) --gpu-index $(NBH_GPU_INDEX) $(if $(NBH_GPU_LOCK),--gpu-lock $(NBH_GPU_LOCK)) --repos eagle --eagle $(abspath ..) $(NBH_ARGS)
