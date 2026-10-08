# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The DEVICE half of the
wide-buffer coercions -- ``eagle.marshal.coerce_wide_inputs``/
``coerce_wide_outputs`` (mirroring ``coerce_vec_inputs``/``coerce_mutable``)
and ``eagle.launch.pure_prepare`` threading them through to
``eagle.launch.assemble_args`` (already wide-aware --
``test_wide_arg_classification.py`` covers that half).

A STUB kernel FUNCTION stands in for ``fn`` in the ``pure_prepare`` test --
``pure_prepare`` only requires ``fn`` be callable with ``(grid, block, args)``,
so this isolates the coercion/threading/return-merge logic from an unrelated
compile step, matching this repo's Gate-A convention one level down the call
stack, at the point real cupy device arrays first appear (device arrays ARE
needed here -- ``coerce_wide_inputs``/``coerce_wide_outputs`` import cupy
internally, like every other marshal.py coercion -- hence ``pytest.mark.gpu``,
unlike the cupy-free ``test_wide_arg_classification.py``)."""

from __future__ import annotations

import json
import pathlib
import shutil
from collections import namedtuple

import pytest

from eagle.launch import pure_prepare
from eagle.loaded import LoadedPure
from eagle.marshal import coerce_wide_inputs, coerce_wide_outputs

pytestmark = pytest.mark.gpu

_MDecl = namedtuple("_MDecl", ("name", "dtype", "width", "shape"), defaults=(None,))
_FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


# ============================================================================
# LoadedPure sidecar reading (`LoadedPure.__init__`) -- meta["wide_inputs"] /
# ["wide_outputs"] populate self.wide_inputs/self.wide_outputs and extend
# self._allowed. Built from the committed "bump" pure fixture (no code-generator
# import, matching this suite's fixture-only convention, test_graph_plugin.py)
# with a HAND-EDITED sidecar carrying additive wide_inputs/wide_outputs keys
# -- validate_sidecar has no cross-check against arg_spec for these (an
# informational, pass-through pair, like matrix_inputs/mat_shapes), so this
# is a legitimate, GPU-real (LoadedKernel.__init__ driver-loads the PTX
# unconditionally) attribute-reading proof that needs no wide-capable
# compiled kernel of its own -- only the sidecar's JSON shape matters here;
# this test never launches.
# ============================================================================
def test_loaded_pure_reads_wide_fields_from_sidecar(tmp_path):
    shutil.copy(_FIX / "bump.ptx", tmp_path / "bump.ptx")
    meta = json.loads((_FIX / "bump.json").read_text())
    meta["wide_inputs"] = ["x"]
    meta["wide_outputs"] = ["x_bar"]
    (tmp_path / "bump.json").write_text(json.dumps(meta))

    plugin = LoadedPure(tmp_path / "bump.ptx")
    assert plugin.wide_inputs == ("x",)
    assert plugin.wide_outputs == ("x_bar",)
    assert {"x", "x_bar"} <= plugin._allowed


def test_loaded_pure_wide_fields_default_empty_when_sidecar_omits_them():
    """A wide-free artifact (every EXISTING fixture) must read exactly as
    before -- the additive-only contract."""
    plugin = LoadedPure(_FIX / "bump.ptx")
    assert plugin.wide_inputs == ()
    assert plugin.wide_outputs == ()


# ============================================================================
# coerce_wide_inputs / coerce_wide_outputs (marshal.py) -- mirror
# coerce_vec_inputs' shape-check + N-reconcile pattern, coerce_mutable's
# import/cast pattern.
# ============================================================================
def test_coerce_wide_inputs_shape_and_n():
    import cupy as cp

    kw = {"x": cp.arange(24, dtype=cp.float64).reshape(6, 4)}
    out, n = coerce_wide_inputs(kw, ("x",))
    assert n == 4
    assert out["x"].shape == (6, 4)


def test_coerce_wide_inputs_reconciles_n_across_names():
    import cupy as cp

    kw = {
        "x": cp.zeros((6, 5), dtype=cp.float64),
        "theta": cp.zeros((9, 5), dtype=cp.float64),
    }
    out, n = coerce_wide_inputs(kw, ("x", "theta"))
    assert n == 5
    assert set(out) == {"x", "theta"}


def test_coerce_wide_inputs_rejects_1d():
    import cupy as cp

    kw = {"x": cp.zeros(4, dtype=cp.float64)}
    with pytest.raises(ValueError, match="2-D"):
        coerce_wide_inputs(kw, ("x",))


def test_coerce_wide_inputs_missing_raises_typeerror():
    with pytest.raises(TypeError, match="missing required wide input"):
        coerce_wide_inputs({}, ("x",))


def test_coerce_wide_inputs_empty_names_is_a_noop():
    out, n = coerce_wide_inputs({}, ())
    assert out == {} and n is None


def test_coerce_wide_outputs_shape_and_n():
    import cupy as cp

    kw = {"x_bar": cp.zeros((6, 4), dtype=cp.float64)}
    out, n = coerce_wide_outputs(kw, ("x_bar",))
    assert n == 4
    assert out["x_bar"].shape == (6, 4)


def test_coerce_wide_outputs_missing_raises_actionable_message():
    with pytest.raises(TypeError, match="never auto-allocated"):
        coerce_wide_outputs({}, ("x_bar",))


def test_coerce_wide_outputs_rejects_1d():
    import cupy as cp

    kw = {"x_bar": cp.zeros(4, dtype=cp.float64)}
    with pytest.raises(ValueError, match="2-D"):
        coerce_wide_outputs(kw, ("x_bar",))


# ============================================================================
# The row-indexed-destination ``exempt`` name -- an atomic
# Accum output's degenerate (rows, 1) buffer, which carries no batch-sized
# axis and so must not join N-reconciliation. ``_core`` md5 stays unchanged
# by this deliverable (python-only; the C++ vocabulary is untouched).
# ============================================================================
def test_coerce_wide_outputs_exempt_name_accepts_rows1_beside_n_sized_sources():
    """That probe's own finding, exercised through the coercion itself: a
    declared-exempt (rows, 1) destination is accepted and its width never
    reconciles against a co-bound N-sized wide input/output."""
    import cupy as cp

    kw = {
        "x": cp.zeros((6, 513), dtype=cp.float64),  # a genuine N=513 wide input
        "dest": cp.zeros((4, 1), dtype=cp.float64),  # the atomic arm's row-indexed dest
    }
    _, n = coerce_wide_inputs(kw, ("x",))
    out, n = coerce_wide_outputs(kw, ("dest",), n, exempt=("dest",))
    assert n == 513  # unaffected by the exempt name
    assert out["dest"].shape == (4, 1)


def test_coerce_wide_outputs_rejects_an_undeclared_mismatch_verbatim():
    """The exemption is OPT-IN per name, never a blanket relaxation: a name
    NOT listed in ``exempt`` still reconciles exactly as before, and an
    actual mismatch still raises today's verbatim message."""
    import cupy as cp

    kw = {
        "x": cp.zeros((6, 513), dtype=cp.float64),
        "x_bar": cp.zeros((6, 4), dtype=cp.float64),  # N=4, disagrees with x's N=513
    }
    _, n = coerce_wide_inputs(kw, ("x",))
    with pytest.raises(
        ValueError, match="inconsistent batch size N across per-sample inputs"
    ):
        coerce_wide_outputs(kw, ("x_bar",), n)  # exempt defaults to () -- unexempted


# ============================================================================
# pure_prepare (launch.py) -- wide_inputs/wide_outputs threading +
# wide_out merged into the returned dict (mirrors
# the code generator's own "out = dict(wide_out); ..."
# convention on the host side).
# ============================================================================
class _StubFn:
    """Records the exact (grid, block, args) a launch call receives -- no
    real compiled kernel needed; ``pure_prepare`` only requires ``fn`` be
    callable with those three positionals (``_resolved_block``'s
    ``_kernel_attrs`` gracefully degrades to ``{}`` on a plain object with no
    ``.attributes``)."""

    def __init__(self):
        self.calls = []

    def __call__(self, grid, block, args):
        self.calls.append((grid, block, args))


def test_pure_prepare_threads_wide_in_and_out_and_merges_return():
    import cupy as cp

    fn = _StubFn()
    kw = {
        "x": cp.arange(24, dtype=cp.float64).reshape(6, 4),
        "x_bar": cp.zeros((6, 4), dtype=cp.float64),
        "y": cp.zeros(4, dtype=cp.float64),
    }
    result = pure_prepare(
        fn,
        [("wide_in", "x"), ("wide_out", "x_bar"), ("mutable", "y")],
        vector_inputs=(),
        per_sample=(),
        params=(),
        mutables_decl=[_MDecl("y", "float", 1)],
        mutable_defaults={"y": None},
        lookup_counts={},
        kw=kw,
        wide_inputs=("x",),
        wide_outputs=("x_bar",),
    )
    assert len(fn.calls) == 1, "the kernel must be launched exactly once"
    _grid, _block, args = fn.calls[0]
    # wide_in, wide_out, mutable(HANDLE) -- one per arg_spec entry
    assert len(args) == 3
    # the wide gradient output is merged into the SAME returned dict as the
    # Mutable handoff (never a separate return value a caller must know to
    # ask for).
    assert "x_bar" in result and result["x_bar"].shape == (6, 4)
    assert "y" in result and result["y"].shape == (4,)


def test_pure_prepare_wide_free_call_is_unaffected():
    """A wide-free kernel (wide_inputs/wide_outputs omitted entirely, the
    byte-identical-when-additive contract) must launch exactly as before
    -- the return dict carries only the Mutable, no stray wide keys."""
    import cupy as cp

    fn = _StubFn()
    kw = {"y": cp.zeros(4, dtype=cp.float64)}
    result = pure_prepare(
        fn,
        [("mutable", "y")],
        vector_inputs=(),
        per_sample=(),
        params=(),
        mutables_decl=[_MDecl("y", "float", 1)],
        mutable_defaults={"y": None},
        lookup_counts={},
        kw=kw,
    )
    assert len(fn.calls) == 1
    assert set(result) == {"y"}


def test_pure_prepare_threads_wide_out_exempt_through_to_coercion():
    """A genuinely N-sized per-sample source (``x``, N=6)
    launches successfully beside a declared-exempt (rows, 1) ``dest`` —
    exactly the shape a prior probe found the FULL ``pure_prepare`` chain
    refusing before this exemption (its own recorded refusal, verbatim:
    "inconsistent batch size N across per-sample inputs")."""
    import cupy as cp

    fn = _StubFn()
    kw = {
        "x": cp.arange(6, dtype=cp.float64),
        "dest": cp.zeros((4, 1), dtype=cp.float64),
        "y": cp.zeros(6, dtype=cp.float64),
    }
    result = pure_prepare(
        fn,
        [("per_sample", "x"), ("wide_out", "dest"), ("mutable", "y")],
        vector_inputs=(),
        per_sample=("x",),
        params=(),
        mutables_decl=[_MDecl("y", "float", 1)],
        mutable_defaults={"y": None},
        lookup_counts={},
        kw=kw,
        wide_outputs=("dest",),
        wide_out_exempt=("dest",),
    )
    assert len(fn.calls) == 1
    assert result["dest"].shape == (4, 1)
    assert result["y"].shape == (6,)


def test_pure_prepare_missing_wide_input_raises_typeerror():
    import cupy as cp

    fn = _StubFn()
    kw = {"y": cp.zeros(4, dtype=cp.float64)}
    with pytest.raises(TypeError, match="missing required wide input"):
        pure_prepare(
            fn,
            [("wide_in", "x"), ("mutable", "y")],
            vector_inputs=(),
            per_sample=(),
            params=(),
            mutables_decl=[_MDecl("y", "float", 1)],
            mutable_defaults={"y": None},
            lookup_counts={},
            kw=kw,
            wide_inputs=("x",),
        )
