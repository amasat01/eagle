# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle's OWN validator path accepts raptor's 5 golden fixtures.

Closes the "constructed wrappers" flag:
raptor's own ``validate_manifest`` is a MINIMAL shape validator by design (raptor's
``schema/__init__.py`` docstring), never eagle's full ``validate_sidecar`` /
``load_manifest``. The three byte-exact-harvested sidecar goldens are run through
:func:`eagle.sidecar.validate_sidecar` directly (GPU-free); the two CONSTRUCTED
manifest wrappers (``pure/manifest.json``, ``vector/manifest.json`` — hand-built
around the harvested sidecars, never previously run through eagle's own reader) are
run through :func:`eagle.registry.load_manifest`, using the same "reaches the
artifact-load step" acceptance idiom ``test_neural_manifest.py`` already establishes
for a schema-only fixture with no compiled ``.ptx`` artifact on disk: a ``ValueError``
means eagle's validator REJECTED something it should have accepted (test failure); any
other exception (missing artifact file, or cupy unavailable) means every validation
clause accepted the wrapper and only the artifact load — which raptor's goldens never
promise to satisfy — failed, exactly as expected.

raptor is a SIBLING checkout (see conftest.py's ``RAPTOR_ROOT``); SKIPs loudly if not
found, mirroring raptor's own ``_eagle_helpers.eagle_root()`` in the other direction.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from eagle.registry import load_manifest
from eagle.sidecar import validate_sidecar

try:
    import raptor

    GOLDENS = pathlib.Path(raptor.__file__).resolve().parent.parent / "goldens"
    _raptor_import_err = None
except Exception as exc:  # pragma: no cover - environment-dependent skip
    GOLDENS = None
    _raptor_import_err = exc

pytestmark = pytest.mark.skipif(
    GOLDENS is None or not GOLDENS.is_dir(),
    reason=(
        f"raptor checkout/goldens not found (set $RAPTOR_ROOT): {_raptor_import_err}"
    ),
)


def _assert_manifest_wrapper_accepted(path):
    """The STUB-reach criterion (mirrors test_neural_manifest.py's
    ``_assert_accept_reaches_stub``): every eagle validation clause must accept the
    wrapper, so the ONLY acceptable failure is the artifact load itself."""
    try:
        load_manifest(path)
    except ValueError as exc:
        pytest.fail(
            f"{path}: eagle's own reader REJECTED a raptor manifest golden it should "
            f"have accepted (closing the 'constructed wrappers' flag): {exc}"
        )
    except Exception:
        return  # expected: no real .ptx artifact on disk in raptor's goldens dir


def test_pure_sidecar_golden_accepted_by_validate_sidecar():
    doc = json.loads((GOLDENS / "pure" / "bump.json").read_text())
    validate_sidecar(doc, name="raptor goldens/pure/bump.json")  # must not raise


def test_vector_sidecar_golden_accepted_by_validate_sidecar():
    doc = json.loads((GOLDENS / "vector" / "gravity.json").read_text())
    validate_sidecar(doc, name="raptor goldens/vector/gravity.json")  # must not raise


def test_neural_block_sidecar_golden_accepted_by_validate_sidecar():
    doc = json.loads((GOLDENS / "row01_neural_block_golden.sidecar.json").read_text())
    validate_sidecar(
        doc, name="raptor goldens/row01_neural_block_golden.sidecar.json"
    )  # must not raise


def test_pure_manifest_wrapper_accepted_by_load_manifest():
    _assert_manifest_wrapper_accepted(GOLDENS / "pure" / "manifest.json")


def test_vector_manifest_wrapper_accepted_by_load_manifest():
    _assert_manifest_wrapper_accepted(GOLDENS / "vector" / "manifest.json")
