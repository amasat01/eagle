# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Compaction by groups of lanes: ``ActiveSet(mask, group=G)``.

Rows: a group is live while any of its lanes is, the map holds group indices
and ``count`` the lanes launched (live groups times ``G``), host and device,
for an ``n`` that is no multiple of ``G``; a terminating workload run through
the map as a group-emitted kernel reads it (``map[t / G] * G + t % G``, lanes
past ``n`` skipped) gives bytes identical to the lane-by-lane set; ``group=``
does not combine with ``reorder=``; and the bind-time group check refuses a
kernel emitted for another group, naming both groups (and is the only thing
that does: the same bind goes through with the check removed).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import eagle
import eagle.exec as eexec
from eagle import ActiveSet, _active_set, sidecar
from eagle import plan as eplan

pytestmark = pytest.mark.usefixtures("_exact_host_profile")

G = 32
N = 1000                    # not a multiple of G: the last group is partial


def _lanes(aset, xp):
    """The lanes a group-emitted kernel runs for threads ``0 .. count``."""
    t = xp.arange(int(aset.count[0]))
    lane = aset.map[t // aset.group] * aset.group + t % aset.group
    return lane[lane < aset.n]


# --------------------------------------------------------------------------- #
# The set itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 31, 32, 33, 1000, 4096])
def test_host_a_group_is_live_while_any_lane_is(n):
    rng = np.random.default_rng([1, n])
    mask = rng.random(n) < 0.97          # sparse live lanes: groups stay live
    a = ActiveSet(mask, group=G)
    assert a.live == -(-n // G) * G
    a.compact()
    ng = -(-n // G)
    padded = np.concatenate([mask, np.ones(ng * G - n, dtype=bool)])
    live = np.flatnonzero(~padded.reshape(ng, G).all(axis=1))
    assert a.live == live.size * G
    assert np.array_equal(a.map[: live.size], live)
    mask[:] = True
    a.compact()
    assert a.live == 0
    a.reset()
    assert a.live == ng * G and np.array_equal(a.map[:ng], np.arange(ng))


def test_the_group_is_checked():
    for bad in (0, -3, 1.5, True, "32"):
        with pytest.raises(ValueError, match="group is an int >= 1"):
            ActiveSet(np.zeros(8, dtype=bool), group=bad)


def test_group_does_not_combine_with_reorder():
    with pytest.raises(ValueError, match="group=32.*reorder"):
        ActiveSet(np.zeros(8192, dtype=bool), group=G, reorder=0.5)
    ActiveSet(np.zeros(8192, dtype=bool), group=1, reorder=0.5)


def test_repr_names_the_group():
    assert "group=32" in repr(ActiveSet(np.zeros(64, dtype=bool), group=G))
    assert "group" not in repr(ActiveSet(np.zeros(64, dtype=bool)))


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 31, 33, 1000, 4096, 100_003])
def test_device_matches_the_host_set(n):
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng([2, n])
    mask = rng.random(n) < 0.97
    host = ActiveSet(mask.copy(), group=G)
    dev = ActiveSet(cp.asarray(mask), group=G)
    assert dev.live == host.live
    host.compact()
    dev.compact()
    assert dev.live == host.live
    ng = host.live // G
    assert np.array_equal(dev.map.get()[:ng], host.map[:ng])
    dev.reset()
    assert dev.live == -(-n // G) * G


# --------------------------------------------------------------------------- #
# A terminating workload: group G equals group 1, byte for byte
# --------------------------------------------------------------------------- #
def _terminating(xp, group, steps=60):
    """Each lane counts up to its own stop step; the loop runs only the lanes
    the set's map names, as a kernel emitted for ``group`` does."""
    rng = np.random.default_rng(3)
    stop = xp.asarray(np.ceil(np.exp(rng.uniform(0.0, np.log(steps), N))))
    x = xp.zeros(N)
    k = xp.zeros(N)
    term = xp.zeros(N, dtype=xp.bool_)
    aset = ActiveSet(term, group=group)
    for step in range(steps):
        lane = _lanes(aset, xp)
        live = lane[~term[lane]]
        x[live] += 0.5 * (live + 1.0)
        k[live] += 1.0
        term |= k >= stop
        if step % 4 == 3:
            aset.compact()
    return x, k, aset.live


def test_host_group_gives_group_one_bytes():
    ref = _terminating(np, 1)
    got = _terminating(np, G)
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
    assert got[2] < -(-N // G) * G, "the set never shrank"
    assert got[2] % G == 0 and ref[2] <= got[2]


@pytest.mark.gpu
def test_device_group_gives_group_one_bytes():
    cp = pytest.importorskip("cupy")
    ref = _terminating(cp, 1)
    got = _terminating(cp, G)
    assert cp.array_equal(ref[0], got[0]) and cp.array_equal(ref[1], got[1])
    assert got[2] < -(-N // G) * G and got[2] % G == 0
    host = _terminating(np, G)
    assert np.array_equal(got[1].get(), host[1]) and got[2] == host[2]


# --------------------------------------------------------------------------- #
# The group check at bind
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def host_plan(tmp_path_factory):
    """The step of a counting kernel built with the active map, planned for the host."""
    hawk = pytest.importorskip("hawk")
    from hawk import Mutable, Param, Scalar, Terminated
    from hawk.ext import Guard, Kind

    def counter_step(dt: Param, terminated: Terminated, k: Mutable[Scalar]):
        k = k + dt

    kern = hawk.kernel(counter_step, kind=Kind("counter", guard=Guard(active_set=True)))
    return eagle.deploy(kern, targets=("host",),
                        cache_dir=str(tmp_path_factory.mktemp("hawk"))).host


def _stamped(plan, group):
    plugin = types.SimpleNamespace(**{**vars(plan.plugin),
                                      "lane_space": sidecar.LaneSpace(((None, None),), group)})
    return eplan.plan(plugin, structure=eexec.HostTeam)


def _bind(plan, aset, n=N):
    return plan.bind(dt=1.0, terminated=aset.mask, k=np.zeros(n), **aset.planes())


def test_bind_refuses_a_set_that_is_not_a_multiple_of_the_kernels_group(host_plan):
    plan = _stamped(host_plan, G)
    with pytest.raises(ValueError,
                       match=r"needs compaction groups of a multiple of 32 lanes.*group 1"):
        _bind(plan, ActiveSet(np.zeros(N, dtype=bool)))
    with pytest.raises(ValueError, match=r"needs compaction groups of a multiple of 32.*group 8"):
        _bind(plan, ActiveSet(np.zeros(N, dtype=bool), group=8))


def test_bind_refuses_a_set_at_another_group_than_the_kernel_reads(host_plan):
    with pytest.raises(ValueError, match=r"reads its active map at group 32.*group 64"):
        _bind(_stamped(host_plan, G), ActiveSet(np.zeros(N, dtype=bool), group=2 * G))
    with pytest.raises(ValueError, match=r"reads its active map at group 1.*group 32"):
        _bind(host_plan, ActiveSet(np.zeros(N, dtype=bool), group=G))


def test_bind_takes_the_matching_group(host_plan):
    _bind(_stamped(host_plan, G), ActiveSet(np.zeros(N, dtype=bool), group=G)).launch()
    _bind(host_plan, ActiveSet(np.zeros(N, dtype=bool))).launch()


def test_the_refusal_is_the_check_and_nothing_else(host_plan, monkeypatch):
    """With the check gone the same bind goes through, to a kernel that would read
    the map in the wrong form: the refusals above are the guard's, not a side effect."""
    plan = _stamped(host_plan, G)
    aset = ActiveSet(np.zeros(N, dtype=bool))
    with pytest.raises(ValueError, match="group"):
        _bind(plan, aset)
    monkeypatch.setattr(_active_set, "check_group", lambda plugin, count: None)
    _bind(plan, aset)
