# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every docs notebook opens on Colab with its packages installed and its helper files in place.

The Colab launch button opens ONE notebook file in a fresh runtime. ``docs/_tools/nb_colab_setup.py`` writes
the setup cell (tag ``colab-setup``) that, on Colab only, ``pip install``s what the notebook imports and
downloads the helper files it needs. This test fails when

* a docs notebook lacks the current generated cell (re-run the generator, do not hand-edit it);
* a notebook, or a helper it uses, imports a third-party module the cell does not install;
* the cell names a repository file that does not exist.

RED: delete the cell from a notebook, add ``import scipy`` to one, or rename ``nb_helpers.py`` and the matching
test fails naming the notebook.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_GEN = _REPO / "docs" / "_tools" / "nb_colab_setup.py"
if not _GEN.is_file():  # the suite also runs from a copy of tests/ outside the repository
    pytest.skip("needs the repository's docs scripts", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("nb_colab_setup", _GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

_NOTEBOOKS = sorted(p for p in (_REPO / gen.CONTENT).rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
_ids = [p.relative_to(_REPO).as_posix() for p in _NOTEBOOKS]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _setup_cell(nb):
    cells = [c for c in nb["cells"] if gen.TAG in c.get("metadata", {}).get("tags", []) and c["cell_type"] == "code"]
    return "".join(cells[0]["source"]) if len(cells) == 1 else None


def test_notebooks_are_found():
    assert _NOTEBOOKS, "no docs notebooks found: the test would pass vacuously"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_ids)
def test_setup_cell_is_current(path):
    nb = _load(path)
    assert _setup_cell(nb) is not None, f"{path.name}: no Colab setup cell; run python docs/_tools/nb_colab_setup.py"
    assert not gen.apply(nb, path), f"{path.name}: Colab setup cell is stale; run python docs/_tools/nb_colab_setup.py"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_ids)
def test_every_third_party_import_is_installed_by_the_cell(path):
    nb = _load(path)
    cell = _setup_cell(nb) or ""
    helpers = gen.helper_modules()
    imported = set()
    for src in gen.code_sources(nb):
        imported |= gen._imports(src, "all")
    for mod in sorted(imported & set(helpers)):  # a helper's own third-party needs count too
        rel = helpers[mod]
        imported |= gen._imports((_REPO / rel).read_text(encoding="utf-8"), "top") | set(gen.HELPER_NEEDS.get(rel, ()))
    missing = []
    for mod in sorted(imported):
        if mod in gen.STDLIB or mod in gen.PREINSTALLED or mod in helpers:
            continue
        dist = gen.PYPI.get(mod)
        if dist is None or f'"{dist}' not in cell.replace('f"', '"'):
            missing.append(f"{mod} ({dist or 'no PyPI name known'})")
    assert not missing, f"{path.name}: the setup cell does not install {missing}"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_ids)
def test_every_helper_the_notebook_imports_is_downloaded_and_exists(path):
    nb = _load(path)
    cell = _setup_cell(nb) or ""
    downloaded = re.findall(r'^\s+"([\w./-]+)",$', cell, re.M)
    for rel in downloaded:
        assert (_REPO / rel).is_file(), f"{path.name}: the setup cell names {rel}, which is not in the repository"
    imported = set()
    for src in gen.code_sources(nb):
        imported |= gen._imports(src, "all")
    for mod, rel in gen.helper_modules().items():
        if mod in imported:
            assert rel in downloaded, f"{path.name}: imports helper {mod} but the setup cell does not download {rel}"
    uses_fixtures = any(gen.FIXTURE_DIRNAME in src for src in gen.code_sources(nb))
    if uses_fixtures:
        assert any(f"/{gen.FIXTURE_DIRNAME}/" in rel for rel in downloaded), f"{path.name}: uses fixtures, none downloaded"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_ids)
def test_the_cell_matches_the_cuda_major_and_says_plainly_when_it_runs_on_the_cpu(path):
    cell = _setup_cell(_load(path)) or ""
    assert "Running on the CPU (no GPU in this runtime). For a GPU: " in cell, f"{path.name}: no plain CPU line"
    assert "Runtime \u2192 Change runtime type \u2192 T4 GPU." in cell and "Settings \u2192 Accelerator \u2192 GPU T4 x2." in cell
    assert 'extra = "[cuda12]"' not in cell, f"{path.name}: the extra is fixed to CUDA 12"
    if "{extra}" in cell:  # a notebook that installs eagle or hawk picks the extra by the runtime
        for needle in ('"cupy-cuda{m}x"', "torch.version.cuda", "CUDA Version: (", 'extra = f"[cuda{major}]"'):
            assert needle.replace("{m}", "{m}") in cell, f"{path.name}: the CUDA-major selection lacks {needle}"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_ids)
def test_the_cell_has_the_release_candidate_and_branch_switches(path):
    cell = _setup_cell(_load(path)) or ""
    assert 'os.environ.get("RAPTOR_PRE")' in cell and '*pre' in cell, f"{path.name}: no RAPTOR_PRE switch"
    assert 'os.environ.get("RAPTOR_REF", "main")' in cell, f"{path.name}: no RAPTOR_REF switch"
    assert "Pre-release test: installing with --pre, helper files from {ref}." in cell
    if "urlretrieve" in cell:
        assert "/{ref}/" in cell and "/main/" not in cell, f"{path.name}: helper files are not fetched from the ref"
        assert 'os.environ.get("RAPTOR_RAW_BASE", "https://raw.githubusercontent.com")' in cell, f"{path.name}: no RAPTOR_RAW_BASE switch"
        assert 'os.environ.get("RAPTOR_WORKDIR")' in cell and '"/content"' in cell, f"{path.name}: no RAPTOR_WORKDIR switch"


def test_the_generated_cell_is_valid_python():
    for path in _NOTEBOOKS:
        ast.parse(_setup_cell(_load(path)) or "")
