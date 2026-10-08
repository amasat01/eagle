# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""RED-first tests for eagle's manifest load path under schema v2.

Companion to ``test_schema_hardening.py``'s compat-matrix tests (which cover
``schema_version`` alone); this file exercises the NEW execution axis that
arrives with schema v2 — ``exec_targets``/``exec_access``/``exec_op`` — through
the one path wired for it today, :func:`eagle.registry.load_manifest`. Before
this file's tests were added, nothing in eagle's Python load path called
``raptor.schema.manifest.check_execution_axis`` at all: a v2 manifest missing
the execution axis loaded as if it were legacy v1 (whole-view, single-device)
instead of being refused ("absence = load refused") — a PRODUCT gap,
now closed by wiring ``eagle.roles.check_execution_axis`` (a re-export, see
``eagle/roles.py``) into ``load_manifest`` ahead of the by-value ABI check
(see the ORDER comment in ``eagle/registry.py``).

eagle's OWN execution structures (``eagle.exec``, ``eagle.plan``) are
NOT implemented yet (banked separately, see
``test_exec_contract_rows.py``). So a manifest that gets PAST the
schema+execution-axis gate still cannot fully LOAD under v2 today: eagle's
by-value ABI tag (:data:`eagle.abi.ABI_VERSION`) is still pinned to schema
v1's ``"aether-abi/1"`` (the aether-abi/2 tag+layout self-check is
separate). That downstream limit is banked below as an
``xfail(strict=True)``, named for what unbanks it — never a ``skip``.
"""

from __future__ import annotations

import json

import pytest

from eagle.registry import load_manifest
from eagle.roles import MAX_SCHEMA_VERSION


def test_max_schema_version_is_two():
    """Sanity anchor: every v2 fixture below hardcodes ``schema_version: 2`` —
    correctly, since v2 is a SPECIFIC, named schema (the execution
    axis, aether-abi/2), not "whatever the ceiling currently is". If a later
    wave bumps :data:`MAX_SCHEMA_VERSION` again (introducing v3), this test
    goes red FIRST, pointing here rather than leaving the v2 fixtures below
    silently misrepresenting v2."""
    assert MAX_SCHEMA_VERSION == 2


def _write_manifest(tmp_path, extra):
    manifest = {
        "schema_version": 2,
        "pattern": "vector",
        "plugins": [],
        **extra,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def test_v2_manifest_without_execution_axis_is_refused(tmp_path):
    """(a) A schema_version=2 manifest carrying NEITHER ``exec_targets`` nor
    ``exec_access`` is rejected — naming the missing-axis rule, not
    eagle's still-v1-only ABI tag. No ``aether_abi`` key is set here at all,
    so ``check_aether_abi`` (which this gate now runs AFTER the
    execution-axis check — see registry.py's ORDER comment) never gets a
    chance to misattribute the reject to the wrong rule."""
    manifest = _write_manifest(tmp_path, {})
    with pytest.raises(ValueError, match="requires execution key"):
        load_manifest(manifest)


def test_v2_manifest_with_partial_execution_axis_is_refused(tmp_path):
    """(a), the partial case: ``exec_targets`` alone (no ``exec_access``) is
    still an execution-axis violation, same rule (the ``missing`` list names
    only the still-absent key)."""
    manifest = _write_manifest(tmp_path, {"exec_targets": ["device"]})
    with pytest.raises(ValueError, match="requires execution key"):
        load_manifest(manifest)


def test_v2_manifest_with_valid_execution_axis_passes_schema_and_axis_validation(
    tmp_path,
):
    """(b) A VALID v2 manifest (exec keys present, ``aether_abi='aether-abi/2'``)
    passes eagle's schema_version + execution-axis validation ON ITS OWN
    TERMS — checked directly against the shape validators
    ``eagle.registry.load_manifest`` itself calls (raptor's
    ``check_schema_version`` / ``check_execution_axis``), so this stays a
    validation-ONLY assertion independent of anything downstream.

    UPDATE: ``eagle.abi.check_aether_abi`` was originally v1-only,
    so this test used to prove "axis validation passed" INDIRECTLY, via the
    raise coming from ``check_aether_abi`` rather than from the axis check.
    ``check_aether_abi`` was later widened to accept BOTH ABI generations,
    so that indirect proof no longer exists: the SAME manifest now fully
    loads end to end (see the sibling test below) rather than raising at
    all. This test now asserts the schema+axis validators directly instead."""
    from raptor.schema.manifest import check_execution_axis, check_schema_version

    manifest = _write_manifest(
        tmp_path,
        {
            "aether_abi": "aether-abi/2",
            "exec_targets": ["device"],
            "exec_access": "sample_local",
        },
    )
    meta = json.loads(manifest.read_text())
    version = check_schema_version(
        meta, name=manifest.name, allow_legacy_version_key=True
    )
    check_execution_axis(meta, version, name=manifest.name)  # must not raise


def test_v2_manifest_with_valid_execution_axis_fully_loads(tmp_path):
    """(b) eagle's aether-abi/2 loader now accepts the ABI tag,
    so a fully valid v2 manifest LOADS (returns a registry), not merely
    validates its shape."""
    manifest = _write_manifest(
        tmp_path,
        {
            "aether_abi": "aether-abi/2",
            "exec_targets": ["device"],
            "exec_access": "sample_local",
        },
    )
    reg = load_manifest(manifest)
    assert reg is not None


def test_v1_manifest_still_rejects_execution_axis_keys(tmp_path):
    """v1's legacy bridge is whole-view/single-device by the axis's
    ABSENCE — a v1 document naming an exec key is a strict-key violation,
    never a silent upgrade. Guards the registry.py reordering: this must
    still be caught by the axis check today, exactly as it would have been
    before this axis check was added (the axis check is new to this call
    site, but the underlying rule — v1 + any exec key = reject — is unconditional on
    call order)."""
    manifest = _write_manifest(
        tmp_path, {"schema_version": 1, "exec_targets": ["device"]}
    )
    with pytest.raises(ValueError, match="must not carry execution key"):
        load_manifest(manifest)
