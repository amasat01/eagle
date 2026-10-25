# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A sidecar's ``launch.block`` is the exact threads per block of every device
launch of that artifact: the one-shot plan, a bound plan's launch, and the same
launch captured into a graph (read back off the kernel node, kept across a replay
over new values and a rebind). An explicit ``block=`` that disagrees is refused
naming both. A kernel emitted for a larger compaction group refuses a smaller or
non-matching ``ActiveSet`` at bind, naming both groups.
"""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest

from eagle import ActiveSet, _active_set, sidecar

elaunch = importlib.import_module("eagle.launch")

BLOCK = 96                  # a multiple of 32 the policy would not choose here
ROW = 64


def test_a_declared_block_is_returned_verbatim_and_a_contradiction_refused():
    assert elaunch._resolved_block(None, 10_000, None, declared=BLOCK) == BLOCK
    assert elaunch._resolved_block(None, 10_000, BLOCK, declared=BLOCK) == BLOCK
    with pytest.raises(ValueError, match=r"block=32 contradicts .*launch\.block 96"):
        elaunch._resolved_block(None, 10_000, 32, declared=BLOCK)
    assert elaunch._resolved_block(None, 10_000, 48) == 48


def test_the_sidecar_reader_takes_the_block_and_refuses_a_malformed_one():
    assert sidecar.read_block({"launch": {"block": BLOCK}}, name="k") == BLOCK
    assert sidecar.read_block({"launch": {}}, name="k") is None
    for bad in (0, -1, "96", 96.5, True):
        with pytest.raises(ValueError, match=r"launch\.block"):
            sidecar.read_block({"launch": {"block": bad}}, name="k")


# --------------------------------------------------------------------------- #
# The group a block-staged kernel was emitted for
# --------------------------------------------------------------------------- #
def _staged(group):
    return types.SimpleNamespace(
        name="k", exec_access="cross_sample_read", arg_spec=(), block=128,
        lane_space=sidecar.LaneSpace(((None, None),), group))


@pytest.mark.parametrize("group", [64, 192, 1])
def test_a_set_whose_group_is_smaller_or_not_a_multiple_of_the_kernels_is_refused(group):
    aset = ActiveSet(np.zeros(1024, dtype=bool), group=group)
    with pytest.raises(ValueError, match=rf"multiple of 128 lanes.*group {group};"):
        _active_set.check_group(_staged(128), aset.count)


def test_a_set_at_the_kernels_group_binds():
    aset = ActiveSet(np.zeros(1024, dtype=bool), group=128)
    _active_set.check_group(_staged(128), aset.count)


# --------------------------------------------------------------------------- #
# A real artifact whose sidecar states a block
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    hawk = pytest.importorskip("hawk")
    import hawk.math as m
    from hawk import Mutable, Scalar, Table
    from hawk.artifact import build_bundle, plugins
    from eagle.registry import load_manifest

    @hawk.kernel
    def rows_sum(x: Table[Scalar], y: Mutable[Scalar]):
        j, s = m.split_index(ROW)
        y = x.at(s) * m.real(j + 1)

    mp = pytest.MonkeyPatch()
    mp.setenv("HAWK_BOX", "0")
    bundle = build_bundle([rows_sum], tmp_path_factory.mktemp("declared"), targets=("cuda",),
                          host_profile="native")
    mp.undo()
    free = plugins(bundle, device_loader=load_manifest)["rows_sum"]
    for art in bundle.artifacts:
        if art.name == "rows_sum":
            art.sidecar["launch"]["block"] = BLOCK
    return free, plugins(bundle, device_loader=load_manifest)["rows_sum"]


def _args(cp, rows, seed):
    x = cp.asarray(np.random.default_rng(seed).standard_normal(ROW))
    return dict(x=x, y=cp.full(rows * ROW, cp.nan))


@pytest.mark.gpu
def test_a_declared_block_survives_capture_replay_and_rebind(staged):
    cp = pytest.importorskip("cupy")
    from test_launch_policy_gate_b import _node_grid_block_pairs, _verbose_dot_capture

    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle.pipeline import GraphPipeline

    free, declared = staged
    assert declared.block == BLOCK and getattr(free, "block", None) is None
    rows = 40

    def reference(seed):
        ref = _args(cp, rows, seed)
        eplan.plan(free, structure=eexec.DeviceKernel).bind(**ref).launch()
        cp.cuda.runtime.deviceSynchronize()
        return cp.asnumpy(ref["y"])

    arrays = _args(cp, rows, 1)
    bound = eplan.plan(declared, structure=eexec.DeviceKernel).bind(**arrays)
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(arrays["y"]), reference(1))

    with pytest.raises(ValueError, match=r"block=32 contradicts .*launch\.block 96"):
        bound.launch(block=32)

    with _verbose_dot_capture():
        pipe = GraphPipeline()
        pipe.add(lambda: bound.launch())
        pipe.build()
    assert [b for _, b in _node_grid_block_pairs(pipe._dot(), expected_node_count=1)] == [BLOCK]

    arrays["x"][...] = _args(cp, rows, 2)["x"]      # replay over new values in the same planes
    pipe.launch()
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(arrays["y"]), reference(2))

    eager = eplan.plan(declared, structure=eexec.DeviceKernel).bind(**_args(cp, rows, 3))
    fresh = _args(cp, rows, 4)
    eager.rebind(**fresh)                            # a rebound plan keeps the block
    eager.launch()
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(fresh["y"]), reference(4))
