# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel emitted with a thread box (sidecar ``launch.box``: ``T`` lanes
``stride`` apart per thread) launches threads, not lanes.

Rows: the plan's partitions count ``ceil(n / (T * stride)) * stride`` threads,
each still over ``n`` lanes, split evenly under ``npartitions``; explicit lane
partitions are refused; a box kernel built by hawk runs to the same bytes as
its box-free build on the host (one and three partitions, rows that fill the
boxes, end in a partial one, or are fewer than ``T``) and the device; an
``ActiveSet`` whose group would split a box is refused at bind, naming ``T``,
the group and the fix, and that refusal is the check's alone.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import eagle.exec as eexec
from eagle import ActiveSet, _active_set, _lane_bind, sidecar
from eagle import plan as eplan

pytestmark = pytest.mark.usefixtures("_exact_host_profile")

BOX = {"axis": 0, "T": 4, "stride": 8}
SPAN = BOX["T"] * BOX["stride"]
ROW = 64                    # the fastest axis of the built kernel


def _plugin(**kw):
    return types.SimpleNamespace(exec_access="cross_sample_read", arg_spec=(), **kw)


# --------------------------------------------------------------------------- #
# Count as threads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n, threads", [(SPAN * 5, 8 * 5), (SPAN * 5 + 8, 8 * 6), (8, 8)])
def test_a_box_kernel_launches_its_threads_over_every_lane(n, threads):
    assert _lane_bind.threads(_plugin(box=BOX), n) == threads
    (part,) = eplan.plan(_plugin(box=BOX), structure=eexec.HostTeam).partitions_for(n)
    assert (part.base, part.count, part.n_samples) == (0, threads, n)
    (part,) = eplan.plan(_plugin(), structure=eexec.HostTeam).partitions_for(n)
    assert (part.base, part.count, part.n_samples) == (0, n, n)


def test_partitions_split_the_threads():
    n = SPAN * 5 + 8
    parts = eplan.plan(_plugin(box=BOX), structure=eexec.HostTeam,
                       npartitions=4).partitions_for(n)
    assert [(p.base, p.count) for p in parts] == [(0, 12), (12, 12), (24, 12), (36, 12)]
    assert {p.n_samples for p in parts} == {n}


def test_explicit_lane_partitions_are_refused():
    with pytest.raises(ValueError, match="count threads, not lanes.*HAWK_BOX=0"):
        eplan.plan(_plugin(box=BOX), structure=eexec.HostTeam,
                   partitions=[(0, 64, 128), (64, 64, 128)])


def test_the_sidecar_reader_takes_the_box_and_refuses_a_malformed_one():
    assert sidecar.read_box({"launch": {"box": BOX}}, name="k") == BOX
    assert sidecar.read_box({"launch": {}}, name="k") is None
    for bad in ({"axis": 0, "T": 1, "stride": 8}, {"axis": 0, "T": 4},
                {"axis": 0, "T": 4.0, "stride": 8}, [4]):
        with pytest.raises(ValueError, match="launch.box"):
            sidecar.read_box({"launch": {"box": bad}}, name="k")


# --------------------------------------------------------------------------- #
# A real box kernel, against its box-free build
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    hawk = pytest.importorskip("hawk")
    import hawk.math as m
    from hawk import Index, Mutable, Scalar, Table
    from hawk.artifact import build_bundle, plugins

    @hawk.kernel
    def rows_mix(lane: Index, x: Table[Scalar], w: Table[Scalar], y: Mutable[Scalar]):
        j, s = m.split_index(ROW)
        acc = 0.0 * m.real(lane)
        for e in range(4):
            acc = acc + w.at(j * 4 + e) * x.at(e * ROW + s) + x.at(s)
        y = acc

    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("box")
    out = {}

    def build(tag, targets):
        bundle = build_bundle([rows_mix], root / tag, targets=targets, host_profile="native")
        return bundle

    for target in ("host", "cuda"):
        mp.delenv("HAWK_BOX", raising=False)
        out[target] = [build(target, (target,))]
        mp.setenv("HAWK_BOX", "0")
        out[target].append(build(target + "_free", (target,)))
    mp.undo()
    return out, plugins


def _plan(bundle, plugins, target, npartitions=1):
    if target == "host":
        return eplan.plan(plugins(bundle)["rows_mix"], structure=eexec.HostTeam,
                          npartitions=npartitions)
    from eagle.registry import load_manifest

    return eplan.plan(plugins(bundle, device_loader=load_manifest)["rows_mix"],
                      structure=eexec.DeviceKernel, npartitions=npartitions)


def _run(plan, xp, rows):
    n = rows * ROW
    rng = np.random.default_rng(rows)
    x = xp.asarray(rng.standard_normal(4 * ROW))
    w = xp.asarray(rng.standard_normal(4 * rows))
    y = xp.full(n, np.nan)
    plan.bind(lane=xp.arange(n, dtype=np.int64), x=x, w=w, y=y).launch()
    return np.asarray(y.get() if hasattr(y, "get") else y)


ROWS = (32, 21, 3)


@pytest.mark.parametrize("npartitions", [1, 3])
@pytest.mark.parametrize("rows", ROWS)
def test_host_box_kernel_equals_its_box_free_build(built, rows, npartitions):
    out, plugins = built
    boxed, free = out["host"]
    assert plugins(boxed)["rows_mix"].box["T"] > 1
    assert getattr(plugins(free)["rows_mix"], "box", None) is None
    got = _run(_plan(boxed, plugins, "host", npartitions), np, rows)
    assert not np.isnan(got).any()
    assert np.array_equal(got, _run(_plan(free, plugins, "host"), np, rows))


@pytest.mark.gpu
@pytest.mark.parametrize("rows", ROWS)
def test_device_box_kernel_equals_its_box_free_build(built, rows):
    cp = pytest.importorskip("cupy")
    out, plugins = built
    boxed, free = out["cuda"]
    got = _run(_plan(boxed, plugins, "cuda", 2), cp, rows)
    assert not np.isnan(got).any()
    assert np.array_equal(got, _run(_plan(free, plugins, "cuda"), cp, rows))


# --------------------------------------------------------------------------- #
# The group check: a box lies inside one compaction group
# --------------------------------------------------------------------------- #
def _boxed(group):
    return _plugin(name="k", box=BOX, lane_space=sidecar.LaneSpace(((None, None),), group))


def test_an_active_set_of_lanes_is_refused_naming_T_the_group_and_the_fix():
    lanes = ActiveSet(np.zeros(SPAN * 4, dtype=bool))
    with pytest.raises(ValueError, match=r"T=4 lanes per thread \(a box of 32 lanes\).*"
                                         r"group 1.*ActiveSet\(mask, group=32\).*HAWK_BOX=0"):
        _active_set.check_group(_boxed(1), lanes.count)


def test_a_group_that_splits_a_box_is_refused():
    half = ActiveSet(np.zeros(SPAN * 4, dtype=bool), group=SPAN // 2)
    with pytest.raises(ValueError, match=r"T=4 .*group 16"):
        _active_set.check_group(_boxed(SPAN // 2), half.count)


def test_a_group_of_whole_boxes_binds():
    whole = ActiveSet(np.zeros(SPAN * 4, dtype=bool), group=SPAN)
    _active_set.check_group(_boxed(SPAN), whole.count)
