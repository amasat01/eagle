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

.PHONY: help Makefile doxygen clean livehtml exampletest strict linkcheck nbexec nbcheck

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
           content/userguide/examples content/interop.ipynb

nbexec:
	python _tools/nb_execute.py $(NB_PATHS)

# Refuse to proceed unless every code cell in every notebook above has
# actually run (an execution_count, not merely static source).
nbcheck:
	python _tools/nb_check_filled.py $(NB_PATHS)

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
