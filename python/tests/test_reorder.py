# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The occasional physical reorder of an :class:`eagle.ActiveSet` (``reorder=theta``).

Rows (host and device faces; the end-to-end hawk rows are in
``test_active_set.py``):

* the reorder is an explicit opt-in (off by default, ``reorder=True`` = 0.5);
  ``theta`` is accepted in ``[0.25, 0.75]`` and refused outside;
* the trigger: a random live set fires, the same live fraction grouped at the
  front does not, an all-live batch does not, and the two faces agree on
  ``live32``/``fire``;
* a reorder moves every owned plane (1-D and ``(D, n)``) so ``plane[inv[i]]``
  is sample ``i``, the live samples fill ``[0, count)`` in sample order, ``perm``
  and ``inv`` are inverse, the map is the identity and ``span == count``; a
  ``fire`` of 0 leaves everything as it was; the device face equals the host
  face to the bit;
* the exits: :meth:`~eagle.ActiveSet.in_sample_order` while permuted,
  :meth:`~eagle.ActiveSet.restore` back to sample order, eagle's export doors
  refuse an owned plane while permuted and accept it after the restore, a
  device hook reads ``plane[inv[i]]``, :meth:`~eagle.ActiveSet.reset` gives the
  identity;
* the refusals: a plane owned twice or overlapping an owned one, an element
  size outside {1, 2, 4, 8, 16}, ``own()`` once the set is bound, ``restore()``
  under capture, the reorder API on a set that does not reorder;
* captured: a graph holding compaction + conditional reorder replays over masks
  changed between replays, equal to the host face every time.
"""

from __future__ import annotations

import numpy as np
import pytest

import eagle
from eagle import ActiveSet, compaction_body


def _xp(device):
    if device:
        return pytest.importorskip("cupy")
    return np


def _host(a):
    get = getattr(a, "get", None)
    return get() if get is not None else np.asarray(a)


FACES = [pytest.param(False, id="host"), pytest.param(True, id="device", marks=pytest.mark.gpu)]


# --------------------------------------------------------------------------- #
# theta
# --------------------------------------------------------------------------- #
def test_theta_bounds():
    m = np.zeros(64, dtype=bool)
    for ok in (0.25, 0.5, 0.75, eagle._active_set.DEFAULT_THETA):
        assert ActiveSet(m.copy(), reorder=ok).theta == ok
    for bad in (0.2, 0.76, 1.0, 0.0):
        with pytest.raises(ValueError, match=r"outside \[0.25, 0.75\]"):
            ActiveSet(m.copy(), reorder=bad)
    with pytest.raises(TypeError, match="theta"):
        ActiveSet(m.copy(), reorder="0.5")
    # an explicit opt-in: off by default, True = the default theta
    assert ActiveSet(m.copy()).theta is None
    assert ActiveSet(m.copy(), reorder=False).theta is None
    assert ActiveSet(m.copy(), reorder=True).theta == eagle._active_set.DEFAULT_THETA
    n = eagle._active_set.MIN_REORDER_SPAN
    assert len(compaction_body(lambda: None, ActiveSet(np.zeros(n, dtype=bool)))) == 2
    assert len(compaction_body(lambda: None, ActiveSet(np.zeros(n, dtype=bool), reorder=True))) == 3


# --------------------------------------------------------------------------- #
# The trigger
# --------------------------------------------------------------------------- #
def _fire(xp, drop, theta=0.5):
    m = xp.asarray(drop)
    a = ActiveSet(m, reorder=theta)
    a.compact()
    return int(_host(a.fire)[0]), int(_host(a.live32)[0]), a.live


@pytest.mark.parametrize("device", FACES)
def test_trigger_fires_on_random_order_and_not_on_grouped_order(device):
    xp = _xp(device)
    n = 100_000
    rng = np.random.default_rng(5)
    random = rng.random(n) < 0.8
    grouped = np.ones(n, dtype=bool)
    grouped[: n // 5] = False
    fire, live32, _ = _fire(xp, random)
    assert fire == 1 and live32 > 0.99 * (n // 32)
    fire, live32, live = _fire(xp, grouped)
    assert fire == 0 and live32 == -(-live // 32)
    assert _fire(xp, np.zeros(n, dtype=bool))[0] == 0
    # a span under the floor never fires: the moves would cost more than they save
    small = random[: eagle._active_set.MIN_REORDER_SPAN - 1]
    assert _fire(xp, small)[0] == 0
    assert _fire(xp, random[: eagle._active_set.MIN_REORDER_SPAN])[0] == 1
    for theta in (0.25, 0.5, 0.75):
        h = _fire(np, random, theta)
        assert _fire(xp, random, theta) == h


# --------------------------------------------------------------------------- #
# The reorder itself
# --------------------------------------------------------------------------- #
def _batch(xp, n, seed=0):
    rng = np.random.default_rng([seed, n])
    planes = dict(
        x=xp.asarray(rng.random(n)),
        k=xp.asarray(rng.integers(0, 2**31, n).astype(np.int32)),
        h=xp.asarray(rng.random(n).astype(np.float16)),
        c=xp.asarray((rng.random(n) + 1j * rng.random(n)).astype(np.complex128)),
        w=xp.asarray(rng.random((3, n)).astype(np.float32)),
    )
    return planes, {key: _host(v).copy() for key, v in planes.items()}


def _check_reordered(a, planes, ref, drop):
    inv, perm = _host(a.inv), _host(a.perm)
    n = a.n
    assert np.array_equal(perm[inv], np.arange(n)) and np.array_equal(inv[perm], np.arange(n))
    for key, v in planes.items():
        assert np.array_equal(_host(v)[..., inv], ref[key]), key
    assert np.array_equal(_host(a.mask)[inv], drop)
    count = a.live
    assert count == int((~drop).sum())
    assert int(_host(a.span)[0]) == count and int(_host(a.fire)[0]) == 0
    assert np.array_equal(_host(a.map)[:count], np.arange(count))
    assert not _host(a.mask)[:count].any()
    assert np.all(np.diff(perm[:count]) > 0), "the live samples keep their order"


@pytest.mark.parametrize("device", FACES)
@pytest.mark.parametrize("n", [1, 7, 257, 100_003])
def test_reorder_moves_every_owned_plane_and_restore_undoes_it(device, n):
    xp = _xp(device)
    planes, ref = _batch(xp, n)
    mask = xp.zeros(n, dtype=bool)
    a = ActiveSet(mask, reorder=0.5).own(*planes.values())
    rng = np.random.default_rng(n)
    drop = rng.random(n) < 0.6
    mask[...] = xp.asarray(drop)
    a.fire[0] = 1  # unconditional for the row (the trigger has its own rows)
    a.reorder()
    _check_reordered(a, planes, ref, drop)
    # thin further (monotone, in sample order) and reorder again
    drop2 = drop | (rng.random(n) < 0.5)
    inv = _host(a.inv)
    m = _host(mask).copy()
    m[inv[drop2]] = True
    mask[...] = xp.asarray(m)
    a.fire[0] = 1
    a.reorder()
    _check_reordered(a, planes, ref, drop2)
    assert int(_host(a.reorders)[0]) == 2 and a.permuted
    for key, v in planes.items():
        assert np.array_equal(_host(a.in_sample_order(v)), ref[key]), key
    a.restore()
    for key, v in planes.items():
        assert np.array_equal(_host(v), ref[key]), key
    assert np.array_equal(_host(mask), drop2)
    assert np.array_equal(_host(a.perm), np.arange(n)) and not a.permuted
    assert int(_host(a.span)[0]) == n
    assert np.array_equal(_host(a.map)[: a.live], np.flatnonzero(~drop2))


@pytest.mark.parametrize("device", FACES)
def test_clear_fire_is_a_no_op(device):
    xp = _xp(device)
    n = 1000
    planes, ref = _batch(xp, n)
    mask = xp.asarray(np.random.default_rng(1).random(n) < 0.5)
    a = ActiveSet(mask, reorder=0.5).own(*planes.values())
    a.reorder()  # fire == 0
    for key, v in planes.items():
        assert np.array_equal(_host(v), ref[key])
    assert int(_host(a.span)[0]) == n and not a.permuted


@pytest.mark.gpu
def test_device_face_equals_host_face_to_the_bit():
    cp = pytest.importorskip("cupy")
    n = 50_000
    out = {}
    for xp in (np, cp):
        planes, _ = _batch(xp, n, seed=3)
        mask = xp.zeros(n, dtype=bool)
        a = ActiveSet(mask, reorder=0.5).own(*planes.values())
        rng = np.random.default_rng(9)
        for frac in (0.5, 0.8, 0.95):
            drop = np.random.default_rng(int(frac * 100)).random(n) < frac
            m = _host(mask).copy()
            m[_host(a.inv)[drop]] = True
            mask[...] = xp.asarray(m)
            a.compact()
            a.reorder()
        del rng
        out[xp.__name__] = ({k: _host(v) for k, v in planes.items()}, _host(a.perm),
                            _host(a.inv), _host(a.map)[: a.live], int(_host(a.reorders)[0]))
    h, d = out["numpy"], out["cupy"]
    for key in h[0]:
        assert h[0][key].tobytes() == d[0][key].tobytes(), key
    for i in (1, 2, 3):
        assert np.array_equal(h[i], d[i])
    assert h[4] == d[4] >= 2


# --------------------------------------------------------------------------- #
# The exits
# --------------------------------------------------------------------------- #
def _permuted_set(xp, n=4096):
    planes, ref = _batch(xp, n)
    mask = xp.zeros(n, dtype=bool)
    a = ActiveSet(mask, reorder=0.5).own(*planes.values())
    mask[...] = xp.asarray(np.random.default_rng(2).random(n) < 0.8)
    a.compact()
    assert int(_host(a.fire)[0]) == 1
    a.reorder()
    assert a.permuted
    return a, planes, ref


@pytest.mark.parametrize("device", FACES)
def test_exports_of_an_owned_plane_are_refused_while_permuted(device):
    xp = _xp(device)
    a, planes, _ = _permuted_set(xp)
    from eagle import interop

    for door in (interop.import_buffer, interop.to_cupy if device else None):
        if door is None:
            continue
        for arr in (planes["x"], planes["w"][1], a.mask):
            with pytest.raises(ValueError, match=r"restore\(\)"):
                door(arr)
    if device:
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            with pytest.raises(ValueError, match=r"restore\(\)"):
                interop._TorchAdapter(True).from_cupy(planes["x"])
    # a copy in sample order is never refused; an unrelated array neither
    interop.import_buffer(a.in_sample_order(planes["x"]))
    interop.import_buffer(xp.zeros(8))
    a.restore()
    interop.import_buffer(planes["x"])


@pytest.mark.gpu
def test_device_hook_reads_a_sample_through_inv():
    cp = pytest.importorskip("cupy")
    a, planes, ref = _permuted_set(cp)
    hook = cp.RawKernel(r"""
    extern "C" __global__ void hook(const double* x, const int* inv, double* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = x[inv[i]];
    }""", "hook")
    out = cp.empty(a.n)
    hook(((a.n + 255) // 256,), (256,), (planes["x"], a.inv, out, np.int32(a.n)))
    assert np.array_equal(out.get(), ref["x"])


@pytest.mark.parametrize("device", FACES)
def test_reset_gives_the_identity(device):
    xp = _xp(device)
    a, _, _ = _permuted_set(xp)
    a.reset()
    n = a.n
    for arr in (a.perm, a.inv, a.map):
        assert np.array_equal(_host(arr), np.arange(n))
    assert int(_host(a.span)[0]) == n and a.live == n and not a.permuted


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_refusals():
    n = 64
    mask = np.zeros(n, dtype=bool)
    x = np.zeros(n)
    a = ActiveSet(mask, reorder=0.5).own(x)
    with pytest.raises(ValueError, match="already owns"):
        a.own(x)
    with pytest.raises(ValueError, match="another ActiveSet"):
        ActiveSet(np.zeros(n, dtype=bool), reorder=0.5).own(x)
    with pytest.raises(ValueError, match="another ActiveSet"):
        ActiveSet(mask, reorder=0.5)  # the mask is owned too
    with pytest.raises(ValueError, match="not movable"):
        a.own(np.zeros(n, dtype=np.dtype((np.void, 3))))
    with pytest.raises(ValueError, match=r"\(64,\) or \(D, 64\)"):
        a.own(np.zeros(n + 1))
    with pytest.raises(ValueError, match="C-contiguous"):
        a.own(np.zeros((n, 2)).T)
    plain = ActiveSet(np.zeros(n, dtype=bool))
    for call in (lambda: plain.own(np.zeros(n)), plain.reorder, plain.reorder_if_degraded,
                 lambda: plain.in_sample_order(np.zeros(n))):
        with pytest.raises(ValueError, match="does not reorder"):
            call()
    plain.restore()  # nothing to restore: a no-op
    with pytest.raises(ValueError, match="reorder=True"):
        compaction_body(lambda: None, plain, reorder=True)
    assert len(compaction_body(lambda: None, a)) == 2  # 64 < the span floor: never fires
    assert len(compaction_body(lambda: None, a, reorder=True)) == 3
    assert len(compaction_body(lambda: None, a, reorder=False)) == 2
    a.reorder_if_degraded()  # fixes the planes
    with pytest.raises(ValueError, match="fixed"):
        a.own(np.zeros(n))


@pytest.mark.gpu
def test_restore_is_refused_under_capture():
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    a, _, _ = _permuted_set(cp, 8192)
    with pytest.raises(RuntimeError, match="refused under stream capture"):
        GraphPipeline().add(a.restore).build()


# --------------------------------------------------------------------------- #
# Captured
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_captured_compaction_and_conditional_reorder_follow_the_mask():
    """RED: a reorder recorded unconditionally (or with its decision baked in at
    capture) would move the planes on the replay whose mask does not fire."""
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    n = 20_000
    dplanes, _ = _batch(cp, n, seed=4)
    hplanes, _ = _batch(np, n, seed=4)
    dmask, hmask = cp.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    d = ActiveSet(dmask, reorder=0.5).own(*dplanes.values())
    h = ActiveSet(hmask, reorder=0.5).own(*hplanes.values())
    pipe = GraphPipeline().add(d.compact).add(d.reorder_if_degraded()).build()
    fires = []
    for frac in (0.1, 0.5, 0.9, 0.97):  # thinning: the live set shrinks every replay
        drop = np.random.default_rng(int(frac * 100)).random(n) < frac
        m = hmask.copy()
        m[h.inv[drop]] = True
        hmask[...] = m
        dmask.set(m)
        h.compact()
        h.reorder_if_degraded()()
        pipe.launch()
        pipe.synchronize()
        fires.append(int(_host(d.reorders)[0]))
        for key in hplanes:
            assert _host(dplanes[key]).tobytes() == hplanes[key].tobytes(), (frac, key)
        assert np.array_equal(_host(d.perm), h.perm)
    assert fires[0] == 0 and fires[-1] >= 1, fires
