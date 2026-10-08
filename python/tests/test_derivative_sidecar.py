# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The optional ``derivative`` sidecar block a VJP/JVP artifact carries.

A derivative kernel — generated or custom, structurally indistinguishable — ships an
additive ``derivative`` block (``{kind, primal, wrt, residual_policy, residuals}``)
alongside the ordinary pure-kernel sidecar. It is metadata only: the launch ignores it.
:func:`eagle.roles.parse_derivative` validates its shape at load (kind ∈ {vjp, jvp};
Phase A is recompute-only, so a populated ``residuals`` list is rejected, naming the
schema-bump rule) and :class:`eagle.loaded.LoadedKernel` exposes the parsed block on
``self.derivative`` — the consumer-side metadata surface. The unit tests below cover
the parser directly (CPU); the GPU tests prove the exposure through ``LoadedPure``.
"""

import json
import pathlib
import shutil

import pytest

import eagle
from eagle.roles import parse_derivative

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"

_BLOCK = {
    "kind": "vjp",
    "primal": "poly",
    "wrt": ["x", "s"],
    "residual_policy": "recompute",
    "residuals": [],
}


# --------------------------------------------------------------------------- #
# parse_derivative — the parse + Phase-A validation, CPU-only.
# --------------------------------------------------------------------------- #
def test_absent_block_returns_none():
    assert parse_derivative({"kernel": "k"}, name="k") is None


def test_well_formed_block_round_trips():
    got = parse_derivative({"derivative": dict(_BLOCK)}, name="poly_vjp")
    assert got == _BLOCK


def test_jvp_kind_is_accepted():
    block = {**_BLOCK, "kind": "jvp"}
    assert parse_derivative({"derivative": block}, name="poly_jvp")["kind"] == "jvp"


def test_unknown_kind_is_rejected():
    block = {**_BLOCK, "kind": "grad"}
    with pytest.raises(ValueError, match="not one of"):
        parse_derivative({"derivative": block}, name="poly_x")


def test_populated_residuals_is_rejected_naming_the_schema_bump():
    block = {**_BLOCK, "residuals": [{"name": "r0"}]}
    with pytest.raises(ValueError, match="schema_version bump"):
        parse_derivative({"derivative": block}, name="poly_vjp")


# --------------------------------------------------------------------------- #
# LoadedPure metadata exposure (GPU — needs a real module load).
# --------------------------------------------------------------------------- #
def _fixture_with_derivative(tmp_path, block):
    """Copy the committed ``bump`` pure fixture and splice a ``derivative`` block into
    its sidecar (the block is metadata; the PTX still launches as an ordinary pure
    kernel), returning the .ptx path a ``LoadedPure`` reads."""
    shutil.copy(FIX / "bump.ptx", tmp_path / "bump.ptx")
    meta = json.loads((FIX / "bump.json").read_text())
    meta["derivative"] = block
    (tmp_path / "bump.json").write_text(json.dumps(meta, indent=2))
    return tmp_path / "bump.ptx"


@pytest.mark.gpu
def test_loaded_pure_exposes_the_derivative_block(tmp_path):
    loaded = eagle.LoadedPure(_fixture_with_derivative(tmp_path, dict(_BLOCK)))
    assert loaded.derivative == _BLOCK


@pytest.mark.gpu
def test_loaded_pure_without_a_block_has_none(tmp_path):
    shutil.copy(FIX / "bump.ptx", tmp_path / "bump.ptx")
    shutil.copy(FIX / "bump.json", tmp_path / "bump.json")
    loaded = eagle.LoadedPure(tmp_path / "bump.ptx")
    assert loaded.derivative is None


@pytest.mark.gpu
def test_loaded_pure_rejects_populated_residuals(tmp_path):
    block = {**_BLOCK, "residuals": [{"name": "r0"}]}
    ptx = _fixture_with_derivative(tmp_path, block)
    with pytest.raises(ValueError, match="schema_version bump"):
        eagle.LoadedPure(ptx)
