# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys

# -- Make the `eagle` Python package importable for autosummary + notebooks.
sys.path.insert(0, os.path.abspath('../python'))
# -- Make `_tools/gen_perf_pages.py` importable as a Sphinx extension (below).
sys.path.insert(0, os.path.abspath('_tools'))

# -- Project information -----------------------------------------------------

project = 'eagle'
copyright = '2026, Alessandro Masat'
author = 'Alessandro Masat'

# Read version from CMakeLists.txt
_version = "unknown"
_cmake = os.path.join(os.path.dirname(__file__), '..', 'CMakeLists.txt')
_pattern = re.compile(r'project\s*\(\s*eagle\s+VERSION\s+(\d+\.\d+\.\d+)', re.IGNORECASE)
try:
    with open(_cmake) as _f:
        for _line in _f:
            _m = _pattern.search(_line)
            if _m:
                _version = _m.group(1)
                break
except FileNotFoundError:
    pass

version = _version
release = _version

# -- General configuration ---------------------------------------------------

# The torch bridge imports torch at module level; the docs build does not install it.
autodoc_mock_imports = ["torch"]

extensions = [
    'sphinx_book_theme',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'myst_nb',             # markdown pages + executed notebooks (supersedes myst_parser)
    'breathe',              # Doxygen integration (C++ API)
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'raptor_docs.sphinx',   # family skin: font, css, favicons, theme defaults, llms.txt
    'gen_perf_pages',       # regenerates content/_generated/*.md from the performance
                            # cards on builder-inited, so performance.md's includes are
                            # never stale (see docs/_tools/gen_perf_pages.py)
]

# MyST options — enable math and rich fencing
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Notebook execution (myst_nb) ---------------------------------------------
# Tutorials/examples are executed LOCALLY on a GPU (`make nbexec`) and
# committed WITH their outputs; the docs build itself never re-executes them
# (`nb_execution_mode = "off"`) — `make nbcheck` is the gate that refuses an
# unfilled notebook before it can reach a publish.
nb_execution_mode = "off"
nb_execution_timeout = 3600
nb_merge_streams = True
nb_output_stderr = "show"
nb_render_markdown_format = "myst"

# -----------------------------------------------------------------------------
# Python API reference (autosummary/autodoc)
# -----------------------------------------------------------------------------
autosummary_generate = True
# eagle's public surface is a pure re-export (`eagle/__init__.py` is entirely
# `from .module import Name`), so excluding "imported" members would leave
# the top-level `eagle` page and every re-exporting submodule page empty.
autosummary_imported_members = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
    'exclude-members': '__weakref__',
}

# -----------------------------------------------------------------------------
# Intersphinx.
# -----------------------------------------------------------------------------
# Sibling sites (raptor, aether, hawk) are deliberately NOT mapped here: their
# `objects.inv` is unreachable until the family's GitHub Pages flip, and an
# unreachable inventory is a `make strict` (-W) failure, not merely a
# linkcheck one. Cross-site references use plain absolute URLs instead (see
# content/interop.md) — linkcheck lists those as expected-404 until the flip.
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Breathe configuration — point to the Doxygen XML output
breathe_projects = {
    "EAGLE": os.path.abspath(os.path.join(os.path.dirname(__file__), '_doxybuild', 'xml'))
}
breathe_default_project = "EAGLE"
breathe_default_members = ('members',)

# Templates and exclusions
templates_path = ['_templates']
# content/_generated/*.md (gen_perf_pages.py's output) are {include}-d by
# performance.md, never toctree'd on their own -- excluded so a strict build
# does not warn about an orphan document.
exclude_patterns = ['_templates', '_build', '_doxybuild', '_tools', 'README.md',
                     'Thumbs.db', '.DS_Store', 'content/_generated']

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_extra_path = ['_redirects']   # stubs for tutorial URLs that were renumbered
# The family skin (raptor-tokens/theme/reveal.css, font) comes from raptor_docs.sphinx; this site adds its accent first
# and its own rules after.
html_css_files = ['site-accent.css', 'brand.css', 'perf-chart.css']

html_theme_options = {
    "repository_url": "https://github.com/amasat01/eagle",
    # One transparent logo file works on light AND dark pages (RAPTOR brand kit).
    "logo": {
        "image_light": "_static/brand/family_eagle.svg",
        "image_dark": "_static/brand/family_eagle.svg",
        "alt_text": "eagle",
    },
}

html_title = f"eagle — run a million per-sample kernels on CPU or GPU ({version})"

suppress_warnings = [
    "cpp.parse",
    "ref.python",
    "duplicate_declaration.cpp",
]
