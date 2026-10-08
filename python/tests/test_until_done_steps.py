# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.until_done`` over a fused kernel (``hawk.steps(kernel, K)``, ``K``
steps per launch), with and without compaction.

What is pinned:

* the cadence: ``every`` counts STEPS, so one loop iteration runs
  ``every / K`` launches then compacts; ``every`` defaults to the smallest
  multiple of ``K`` that is at least 16 (``K`` itself above 16: one compaction
  per launch); an ``every`` that is not a multiple of ``K`` is refused naming
  the nearest multiples; the ``MIN_EVERY`` floor is in steps, so a fused
  kernel may compact after every launch;
* bit identity: a fused plain or active-set artifact run through the one call
  leaves ``x``, ``v``, ``k`` bit-equal, on a spread and a uniform batch (with
  compactions on the spread one; with ``reorder=0.5`` too, sample order
  restored), to the one-step explicit oracle on the host and to the fused
  plain plan driven by hand on the device (the device compile may contract
  the K-step loop's arithmetic into FMAs differently from the single step's,
  so the one-step oracle is a host row);
* the accounting: ``steps == launches * every`` and the loop cap is
  ``ceil(max_steps / every)`` iterations.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_until_done import (
    _DT,
    _build,
    _explicit,
    _get,
    _have_gpu,
    _inputs,
    _oscillator,
    _planes,
    _xp,
)

import eagle
from eagle import compaction_body
from eagle._active_set import MIN_EVERY, MIN_REORDER_SPAN, ActiveSet

_KS = (4, 32)
_TARGETS = ["host", pytest.param("device", marks=pytest.mark.gpu)]


@pytest.fixture(scope="module", autouse=True)
def _exact_host_profile():
    """Every host build in this module under hawk's EXACT ``native`` profile
    (FMA contraction off): the rows compare host runs of separate compiles bit
    for bit, which only the exact profiles promise. hawk's default x86-64
    profile contracts ``a*b + c`` into FMAs and is gated by ULP bounds in
    hawk's own suite instead."""
    mp = pytest.MonkeyPatch()
    mp.setenv("HAWK_HOST_PROFILE", "native")
    yield
    mp.undo()


@pytest.fixture(scope="module")
def fused(tmp_path_factory):
    """``fused[(label, K, target)]``: label plain/map (finishing kernels), K in
    {1} + _KS; ``("oracle", 1)`` is the one-step kernel without finish."""
    hawk = pytest.importorskip("hawk")
    from hawk.ext import Guard, Kind

    targets = ("host", "cuda") if _have_gpu() else ("host",)
    out = {}
    oracle = _build(_oscillator(None, False), tmp_path_factory.mktemp("oracle"),
                    targets)
    out.update({("oracle", 1, t): p for t, p in oracle.items()})
    for label, guard in (("plain", None), ("map", Guard(active_set=True))):
        for k in (1, *_KS):
            kind = None if guard is None else Kind(f"uds_{label}_{k}", guard=guard)
            base = _oscillator(kind, True)
            kern = base if k == 1 else hawk.steps(base, k)
            built = _build(kern, tmp_path_factory.mktemp(f"{label}{k}"), targets)
            out.update({(label, k, t): p for t, p in built.items()})
    return out


def _need(fused, key):
    if key not in fused:
        pytest.skip("no CUDA device")


def _oracle(fused, target, inp, steps):
    plans = {("plain", target): fused[("oracle", 1, target)]}
    return _explicit(plans, target, inp, steps)


def _fused_explicit(fused, k, target, inp, steps):
    """The fused plain plan driven by hand: launch until the counter reads n
    (no runner, no compaction) -- the oracle of a fused run on the device,
    where the K-step loop and the single step are separate compiles."""
    xp = _xp(target)
    g = _planes(inp, xp)
    n = inp["x"].shape[0]
    term = xp.zeros(n, dtype=xp.bool_)
    counter = xp.zeros(1, dtype=xp.uint32)
    bound = fused[("plain", k, target)].bind(
        dt=_DT, terminated=term, finished_count=counter.view(xp.int32), **g)
    for _ in range(-(-steps // k)):
        if int(counter[0]) == n:
            break
        bound.launch()
    return {key: _get(g[key]) for key in ("x", "v", "k")}, _get(term)


def _reference(fused, k, target, inp, steps):
    """The one-step oracle on the host (``-ffp-contract=off``: the K-step
    loop is bit-identical to one step there); on the device the fused plan
    driven by hand, since the device compile contracts FMAs per compile."""
    if target == "host":
        return _oracle(fused, target, inp, steps)
    return _fused_explicit(fused, k, target, inp, steps)


# --------------------------------------------------------------------------- #
# The cadence (host plans; no launch)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k, every, lpc", [(1, 16, 16), (4, 16, 4), (32, 32, 1)])
def test_the_default_cadence_is_the_smallest_multiple_of_k_from_16(fused, k, every,
                                                                   lpc):
    """Every defaults to K * ceil(16 / K) steps; the cap is in compaction blocks."""
    g = _planes(_inputs(256, "spread", 20), np)
    runner = eagle.until_done(fused[("map", k, "host")], max_steps=1000, dt=_DT, **g)
    assert runner.steps_per_launch == k
    assert runner.every == every and runner.launches_per_compaction == lpc
    assert runner.loop.max_iters == -(-1000 // every)


def test_an_every_of_one_launch_is_accepted_for_a_fused_kernel(fused):
    """Every == K: one launch per compaction, allowed (4 steps >= MIN_EVERY)."""
    g = _planes(_inputs(256, "spread", 20), np)
    runner = eagle.until_done(fused[("map", 4, "host")], max_steps=100, every=4,
                              dt=_DT, **g)
    assert runner.launches_per_compaction == 1 and runner.every == 4


@pytest.mark.parametrize("every, match", [(6, "every=4 or every=8"),
                                          (2, "every=4"), (33, "every=32 or every=64")])
def test_an_every_not_a_multiple_of_k_is_refused_naming_the_nearest(fused, every,
                                                                   match):
    """A cadence that is not whole launches is refused, naming the multiples."""
    k = 32 if every == 33 else 4
    g = _planes(_inputs(256, "spread", 20), np)
    with pytest.raises(ValueError, match=rf"{k} steps per launch.*use {match}"):
        eagle.until_done(fused[("map", k, "host")], max_steps=100, every=every,
                         dt=_DT, **g)


def test_the_min_every_floor_is_in_steps(fused):
    """MIN_EVERY counts steps: every * steps_per_call."""
    g = _planes(_inputs(256, "spread", 20), np)
    with pytest.raises(ValueError, match=f"below {MIN_EVERY} steps"):
        eagle.until_done(fused[("map", 1, "host")], max_steps=100, every=2,
                         dt=_DT, **g)
    a = ActiveSet(np.zeros(64, dtype=bool))
    assert len(compaction_body(lambda: None, a, every=1, steps_per_call=4)) == 2
    with pytest.raises(ValueError, match=f"below {MIN_EVERY} steps"):
        compaction_body(lambda: None, a, every=1, steps_per_call=2)
    with pytest.raises(ValueError, match="steps_per_call"):
        compaction_body(lambda: None, a, every=4, steps_per_call=0)


# --------------------------------------------------------------------------- #
# Bit identity with the one-step oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("distribution", ["spread", "uniform"])
@pytest.mark.parametrize("k", _KS)
@pytest.mark.parametrize("label", ["plain", "map"])
def test_fused_runs_are_bit_identical_to_one_step(fused, target, distribution, k,
                                                  label):
    """A fused run through the one call, compacting or not, is bit-equal."""
    _need(fused, (label, k, target))
    n, steps = 5000, 200
    inp = _inputs(n, distribution, steps)
    ref, ref_term = _reference(fused, k, target, inp, steps)
    g = _planes(inp, _xp(target))
    runner = eagle.until_done(fused[(label, k, target)], max_steps=steps, dt=_DT, **g)
    rep = runner.run()
    for key in ("x", "v", "k"):
        assert np.array_equal(ref[key], _get(g[key])), key
    assert np.array_equal(ref_term, _get(runner.terminated))
    assert rep.done and rep.finished == n
    assert rep.steps == rep.launches * runner.every
    s = int(inp["nstop"].max())
    assert rep.launches <= -(-s // runner.every) + 1
    if label == "map" and distribution == "spread":
        assert rep.compactions >= 1 and runner.active.live < n


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("k", _KS)
def test_fused_reorder_runs_and_restores_sample_order(fused, target, k):
    """Fused + compaction + reorder: bit-equal and back in sample order."""
    _need(fused, ("map", k, target))
    n, steps = max(MIN_REORDER_SPAN, 100_000), 200
    inp = _inputs(n, "spread", steps)
    ref, _ = _reference(fused, k, target, inp, steps)
    g = _planes(inp, _xp(target))
    runner = eagle.until_done(fused[("map", k, target)], max_steps=steps, every=k,
                              reorder=0.5, dt=_DT, **g)
    rep = runner.run()
    assert rep.compactions >= 1 and rep.done
    a = runner.active
    ident = np.arange(n, dtype=np.int32)
    assert np.array_equal(_get(a.perm)[_get(a.inv)], ident)
    for key in ("x", "v", "k"):
        assert np.array_equal(ref[key], _get(g[key])), key
