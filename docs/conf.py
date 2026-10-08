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
# raptor-tokens.css (kit) -> site-accent.css (this site's --accent) ->
# raptor-theme.css (shared skin, derives everything from --accent) ->
# raptor-reveal.css (kit) -> brand.css (skeleton rules that CONSUME the vars
# raptor-theme.css sets; no longer sets them itself, see its own header).
html_css_files = ['raptor-tokens.css', 'site-accent.css', 'raptor-theme.css',
                   'raptor-reveal.css', 'brand.css']

html_theme_options = {
    "repository_url": "https://github.com/amasat01/eagle",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "collapse_navigation": True,
    "navigation_with_keys": True,
    # Colab + download launch buttons on notebook pages (no Binder: not configured).
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "classic",
    },
    # Harmonised code themes for light/dark (pydata-sphinx-theme keys).
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    # One transparent logo file works on light AND dark pages (RAPTOR brand kit).
    "logo": {
        "image_light": "_static/brand/family_eagle.svg",
        "image_dark": "_static/brand/family_eagle.svg",
        "alt_text": "eagle",
    },
    # Family frame: purple "part of RAPTOR" chip in the footer, every site
    # (raptor-theme.css's .raptor-family-chip; theme's own extension point).
    "extra_footer": (
        '<div class="raptor-family-chip">part of '
        '<a href="https://amasat01.github.io/">RAPTOR</a></div>'
    ),
}

html_title = f"eagle — run a million per-sample kernels on CPU or GPU ({version})"

suppress_warnings = [
    "cpp.parse",
    "ref.python",
    "duplicate_declaration.cpp",
]

# RAPTOR brand: favicons + home-screen icon. Leave html_favicon unset: the
# hook below writes the icon links itself (SVG where supported, favicon.ico
# for Safari/older tools, 180 px icon for iOS home screens).
_RAPTOR_ICONS = [
    ("icon", "brand/favicon.ico", 'sizes="any"'),
    ("icon", "brand/favicon.svg", 'type="image/svg+xml"'),
    ("apple-touch-icon", "brand/app_icon_180.png", ""),
]


def _raptor_icons(app, pagename, templatename, context, doctree):
    pathto = context.get("pathto")
    if pathto is None:
        return
    links = "".join(
        f'<link rel="{rel}" href="{pathto("_static/" + path, 1)}" {extra}>\n'
        for rel, path, extra in _RAPTOR_ICONS
    )
    context["metatags"] = context.get("metatags", "") + links


def setup(app):
    app.connect("html-page-context", _raptor_icons)
