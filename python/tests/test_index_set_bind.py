# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The bind side of a caller-built lane set (``Kind(..., lanes=IndexSet())``): the host builder, the width
(int32, or int64 for a 64-bit unit), the checks that refuse by name, ``ActiveSet.planes()`` feeding the same kernel,
and a captured graph that replays with the map and count rewritten in place."""

from __future__ import annotations

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
ext = pytest.importorskip("hawk.ext")
if not hasattr(ext, "IndexSet"):
    pytest.skip("this hawk has no lane sets (lanes=IndexSet())", allow_module_level=True)

import eagle  # noqa: E402
import eagle.plan  # noqa: E402
import hawk.math as m  # noqa: E402
from hawk import Grid, Mutable, Scalar, Table  # noqa: E402

N = 40
PAIRS = 12
NPAIR = PAIRS * (PAIRS - 1) // 2
LIVE = [7, 2, 5, 31, 12]


def _flat(x: Scalar, y: Mutable[Scalar]):
    y = x * x + 1.0


def _pair(lane: Grid["pair"], x: Table["node"], d: Mutable[Scalar], rs: Mutable["node"]):
    i, j = m.pair_index()
    t = x.at(i) * x.at(j) * x.at(j) * x.at(i) + 0.25 * x.at(j)
    d = t
    hawk.scatter(rs)[j] += t


SET32 = ext.Kind("lane_set", lanes=ext.IndexSet())
SET64 = ext.Kind("lane_set64", lanes=ext.IndexSet(width=64))


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("cache"))
    out = {}
    for tag, kind in (("dense", None), ("set32", SET32), ("set64", SET64)):
        k = hawk.kernel(_flat) if kind is None else hawk.kernel(_flat, kind=kind)
        (out[tag],) = eagle.plan.auto([k], targets=("host",), cache_dir=cache)
    return out


def _x():
    return np.random.default_rng(3).uniform(-2.0, 2.0, N)


def _same(got, want):
    """Equal to rounding (the kernel contracts ``x*x + 1`` to an FMA, numpy does not) and exactly zero off the set."""
    got, want = np.asarray(got), np.asarray(want)
    assert np.array_equal(got == 0.0, want == 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0)


def _dense_oracle(live):
    want = np.zeros(N)
    want[list(live)] = (_x() ** 2 + 1.0)[list(live)]
    return want


# -- host: the builder, int32 and int64 ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag, word", [("set32", "int32"), ("set64", "int64")])
def test_a_host_map_and_count_bind_without_an_activeset_in_both_widths(plans, tag, word):
    width = 64 if tag == "set64" else 32
    planes = eagle.IndexSet.planes(LIVE, N, width=width)
    assert str(planes["active_map"].dtype) == word and planes["active_count"].tolist() == [len(LIVE)]
    got = plans[tag].run(x=_x(), y=np.zeros(N), **planes)
    _same(np.asarray(got), _dense_oracle(LIVE))


def test_the_builder_takes_a_mask_a_capacity_and_picks_the_width_from_the_extent():
    mask = np.zeros(N, dtype=bool)
    mask[LIVE] = True
    p = eagle.IndexSet.planes(mask, N, capacity=16)
    assert p["active_map"].tolist()[:5] == sorted(LIVE) and len(p["active_map"]) == 16 and p["active_count"].tolist() == [5]
    assert str(eagle.IndexSet.planes([3, 2**31 + 5], 2**31 + 9)["active_map"].dtype) == "int64"
    assert str(eagle.IndexSet.planes([3], 2**31 - 1)["active_map"].dtype) == "int32"
    with pytest.raises(ValueError, match="width=32 cannot address"):
        eagle.IndexSet.planes([3], 2**31 + 9, width=32)


# -- the checks refuse by name ------------------------------------------------------------------------------------
def test_checks_refuse_by_name_at_bind(plans):
    x, y = _x(), np.zeros(N)
    one = np.array([1], np.uint32)
    with pytest.raises(ValueError, match=r"active_map\[1\] = 40 is outside the dense lanes \[0, 40\)"):
        plans["set32"].run(x=x, y=y, active_map=np.array([3, 40], np.int32), active_count=np.array([2], np.uint32))
    with pytest.raises(ValueError, match=r"active_map\[2\] = 3 repeats active_map\[0\]"):
        plans["set32"].run(x=x, y=y, active_map=np.array([3, 4, 3], np.int32), active_count=np.array([3], np.uint32))
    with pytest.raises(ValueError, match="active_count is 5, but active_map holds 2"):
        plans["set32"].run(x=x, y=y, active_map=np.array([3, 4], np.int32), active_count=np.array([5], np.uint32))
    with pytest.raises(ValueError, match="cast the map to int32, or build the unit with IndexSet"):
        plans["set32"].run(x=x, y=y, active_map=np.array([3], np.int64), active_count=one)
    # entries past the count are not the set's: they may hold anything
    ok = plans["set32"].run(x=x, y=y, active_map=np.array([3, 999], np.int32), active_count=one)
    assert np.count_nonzero(np.asarray(ok)) == 1


def test_a_host_int32_map_is_widened_for_a_64_bit_unit(plans):
    got = plans["set64"].run(x=_x(), y=np.zeros(N), active_map=np.array(LIVE, np.int32),
                             active_count=np.array([len(LIVE)], np.uint32))
    _same(np.asarray(got), _dense_oracle(LIVE))


def test_check_index_set_is_public_and_names_the_first_bad_position():
    ok = eagle.IndexSet.planes(LIVE, N)
    eagle.check_index_set(ok["active_map"], ok["active_count"], N)
    with pytest.raises(ValueError, match=r"active_map\[0\] = 7 is outside the dense lanes \[0, 5\)"):
        eagle.check_index_set(ok["active_map"], ok["active_count"], 5)
    with pytest.raises(ValueError, match="64-bit unit, but active_map holds 32-bit words"):
        eagle.check_index_set(ok["active_map"], ok["active_count"], N, width=64)


# -- ActiveSet.planes() feeds the same kernel ---------------------------------------------------------------------
def test_activeset_planes_feed_the_same_kernel(plans):
    term = np.zeros(N, dtype=bool)
    term[[0, 1, 3, 4, 6]] = True
    aset = eagle.ActiveSet(term)
    aset.compact()
    got = plans["set32"].run(x=_x(), y=np.zeros(N), **aset.planes())
    live = [i for i in range(N) if not term[i]]
    _same(np.asarray(got), _dense_oracle(live))


# -- GPU: the pair-grid screen, and the in-place rewrite between replays ------------------------------------------
@pytest.fixture(scope="module")
def device_plans(tmp_path_factory):
    pytest.importorskip("cupy")
    cache = str(tmp_path_factory.mktemp("dcache"))
    out = {}
    for tag, fn, kind in (("flat", _flat, SET32), ("pair", _pair, SET32), ("pair_dense", _pair, None)):
        k = hawk.kernel(fn) if kind is None else hawk.kernel(fn, kind=kind)
        try:
            (out[tag],) = eagle.plan.auto([k], targets=("cuda",), cache_dir=cache)
        except Exception as exc:        # noqa: BLE001 -- no CUDA toolchain or device
            pytest.skip(f"no device plan: {exc}")
    return out


@pytest.mark.gpu
def test_the_pair_grid_screen_on_the_device_equals_the_dense_launch_on_the_listed_pairs(device_plans):
    import cupy as cp

    live = np.array([0, 4, 17, 44, 65], dtype=np.int32)
    x = cp.asarray(np.random.default_rng(7).uniform(0.5, 2.0, PAIRS))

    def run(plan, **extra):
        d, rs = cp.zeros(NPAIR), cp.zeros(PAIRS)
        plan.run(x=x, d=d, rs=rs, pair=NPAIR, **extra)
        cp.cuda.Device().synchronize()
        return d.get(), rs.get()

    dense, _ = run(device_plans["pair_dense"])
    planes = eagle.IndexSet.planes(live, NPAIR, device="cuda:0")
    d, _rs = run(device_plans["pair"], **planes)
    want = np.zeros(NPAIR)
    want[live] = dense[live]
    assert np.array_equal(d, want)


@pytest.mark.gpu
def test_a_captured_graph_replays_with_the_map_and_count_rewritten_in_place(device_plans):
    import cupy as cp

    from eagle.pipeline import GraphPipeline

    capacity = 16
    x, y = cp.asarray(_x()), cp.zeros(N)
    planes = eagle.IndexSet.planes([1], N, capacity=capacity, device="cuda:0")
    lane_map, count = planes["active_map"], planes["active_count"]
    bound = device_plans["flat"].bind(x=x, y=y, **planes)
    pipe = GraphPipeline().add(bound.launch).build()                       # captured once, with these buffers
    for members in ([1], [7, 2, 5, 31, 12], [39, 0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 8, 9, 10], [3, 4]):
        assert len(members) <= capacity
        padded = list(members) + [0] * (capacity - len(members))
        lane_map[...] = eagle.arrays.asarray(padded, eagle.arrays.int32)   # the same buffers, new contents
        count[...] = eagle.arrays.asarray([len(members)], eagle.arrays.uint32)
        eagle.check_index_set(lane_map, count, N)                          # a replay checks nothing: the caller may
        y[...] = 0.0
        pipe.launch()
        cp.cuda.Device().synchronize()
        _same(y.get(), _dense_oracle(members))
    with pytest.raises(ValueError, match="outside the dense lanes"):       # and the explicit check catches a bad rewrite
        lane_map[...] = eagle.arrays.asarray([N + 3] + [0] * (capacity - 1), eagle.arrays.int32)
        eagle.check_index_set(lane_map, count, N)
