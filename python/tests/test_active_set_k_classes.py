# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""K-class :class:`eagle.ActiveSet` (``classes=K``): the stable counting sort by
``(live, class)``, the ``offsets`` class prefix, the ``mixed32`` trigger, the
pairing planes, the refusals. Host face; the device face rows are marked gpu.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import eagle
from eagle import ActiveSet, _active_set


def _make(n, k, *, seed=0, p_dead=0.3, epochs=False):
    rng = np.random.default_rng(seed)
    mask = (rng.random(n) < p_dead).astype(np.uint8)
    cls = rng.integers(0, k, n).astype(np.uint8)
    payload = np.arange(n, dtype=np.int64) * 7 + 3  # plane[i] identifies sample i
    a = ActiveSet(mask, reorder=0.5, classes=k, class_plane=cls, epochs=epochs)
    a.own(payload)
    return a, mask.copy(), cls.copy(), payload


def _oracle(mask, cls, k):
    """Serial oracle: samples ordered by (dead, class), ties by sample id."""
    n = len(mask)
    return sorted(range(n), key=lambda i: (k if mask[i] else int(cls[i]), i))


@pytest.mark.parametrize("k", [2, 4, 7])
def test_class_sort_matches_serial_oracle(k):
    n = 5000
    a, mask0, cls0, payload = _make(n, k)
    a.compact()
    assert a.class_sort()
    want = _oracle(mask0, cls0, k)
    assert list(a.perm) == want
    # perm is a permutation and inv is its inverse
    assert sorted(a.perm.tolist()) == list(range(n))
    assert np.array_equal(a.inv[a.perm], np.arange(n))
    # owned planes follow perm: payload at slot s is sample perm[s]'s
    assert np.array_equal(payload, np.asarray(want) * 7 + 3)
    assert np.array_equal(a.class_plane, cls0[want])
    live = int((mask0 == 0).sum())
    assert int(a.count[0]) == live and int(a.span[0]) == live
    assert np.array_equal(a.map[:live], np.arange(live))
    assert int(a.class_sorts[0]) == 1 and int(a.reorders[0]) == 0


@pytest.mark.parametrize("k", [2, 4])
def test_offsets_equal_bincount(k):
    a, mask0, cls0, _ = _make(4097, k, seed=3)
    a.class_sort()
    counts = np.bincount(cls0[mask0 == 0], minlength=k)
    off = np.asarray(a.offsets)
    assert off[0] == 0 and off[k] == counts.sum() == int(a.count[0])
    assert np.array_equal(np.diff(off.astype(np.int64)), counts)
    for c in range(k):  # the prefix IS the class: slots [off[c], off[c+1]) hold class c
        assert (a.class_plane[off[c]:off[c + 1]] == c).all()


def test_stable_within_class():
    a, mask0, cls0, _ = _make(3000, 3, seed=5)
    a.class_sort()
    live = int(a.count[0])
    perm = np.asarray(a.perm[:live])
    off = np.asarray(a.offsets)
    for c in range(3):
        seg = perm[off[c]:off[c + 1]]
        assert (np.diff(seg) > 0).all()


def test_within_live_sorts_prefix_only():
    a, mask0, cls0, payload = _make(2000, 4, seed=7)
    a.class_sort()  # live first
    live = int(a.count[0])
    dead_perm = np.asarray(a.perm[live:]).copy()
    # scramble classes inside the prefix, then re-sort within it
    rng = np.random.default_rng(1)
    a.class_plane[:live] = rng.integers(0, 4, live).astype(np.uint8)
    cls_now = np.asarray(a.class_plane[:live]).copy()
    perm_before = np.asarray(a.perm[:live]).copy()
    assert a.class_sort(within_live=True)
    assert int(a.count[0]) == live and int(a.span[0]) == live
    assert np.array_equal(a.perm[live:], dead_perm)  # beyond the prefix untouched
    order = np.argsort(cls_now, kind="stable")
    assert np.array_equal(a.perm[:live], perm_before[order])
    assert int(a.class_sorts[0]) == 2 and int(a.reorders[0]) == 0
    # injected sample-id plane equals perm after the sort (pairing)
    assert np.array_equal((payload - 3) // 7, a.perm)


def test_restore_returns_sample_order():
    a, mask0, cls0, payload = _make(4500, 4, seed=9)
    a.class_sort()
    assert a.permuted
    a.restore()
    assert not a.permuted
    assert np.array_equal(payload, np.arange(4500) * 7 + 3)
    assert np.array_equal(a.class_plane, cls0)
    assert np.array_equal(a.perm, np.arange(4500))


def test_in_sample_order_while_permuted():
    a, mask0, cls0, payload = _make(3000, 4, seed=11)
    a.class_sort()
    assert np.array_equal(a.in_sample_order(payload), np.arange(3000) * 7 + 3)


def test_mixed32_counts_groups_with_two_classes():
    n = 8192
    mask = np.zeros(n, dtype=np.uint8)
    cls = np.zeros(n, dtype=np.uint8)
    cls[(np.arange(n) // 32) % 2 == 0] = 1  # whole groups of one class: not mixed
    a = ActiveSet(mask, reorder=0.5, classes=2, class_plane=cls)
    a.compact()
    assert int(a.mixed32[0]) == 0 and int(a.class_fire[0]) == 0
    cls[::2] = 1 - cls[::2]  # every group now holds both classes
    a.compact()
    assert int(a.mixed32[0]) == n // 32
    assert int(a.class_fire[0]) == 1
    assert a.class_sort(if_fired=True)
    a.compact()
    assert int(a.mixed32[0]) <= 1  # grouped: at most the one boundary group
    assert int(a.class_fire[0]) == 0 or int(a.mixed32[0]) >= 0.5 * int(a.live32[0])
    assert a.class_sort(if_fired=True) is False


def test_trigger_floor_span():
    n = 1024  # below MIN_CLASS_SORT_SPAN: never fires
    mask = np.zeros(n, dtype=np.uint8)
    cls = (np.arange(n) % 2).astype(np.uint8)
    a = ActiveSet(mask, reorder=0.5, classes=2, class_plane=cls)
    a.compact()
    assert int(a.mixed32[0]) == n // 32 and int(a.class_fire[0]) == 0


def test_epoch_table_records_perm_snapshots():
    a, *_ = _make(4200, 3, seed=2, epochs=True)
    assert a.epochs == ()
    a.class_sort()
    a.class_plane[: int(a.count[0])] = 0
    a.class_sort(within_live=True)
    e = a.epochs
    assert len(e) == 2 and np.array_equal(e[1], a.perm) and not np.array_equal(e[0], np.arange(4200))
    a.perm[0] = a.perm[0]  # the snapshot is a copy
    assert e[0] is not a.perm
    a.reset()
    assert a.epochs == ()


def test_liveness_reorder_is_the_class_sort():
    """One permutation: with classes=K a fired liveness reorder is the (live, class) sort."""
    a, mask0, cls0, _ = _make(8192, 3, seed=4, p_dead=0.8)
    a.compact()
    assert int(a.fire[0]) == 1 and int(a.class_fire[0]) == 1  # the liveness term is OR-ed in
    a.reorder()
    assert int(a.reorders[0]) == 0 and int(a.class_sorts[0]) == 1
    assert list(a.perm) == _oracle(mask0, cls0, 3)
    a.compact()
    a.reorder()  # nothing fired: no second sort
    assert int(a.class_sorts[0]) == 1


def test_stale_trigger_is_exact_and_independent():
    n = 4096
    mask = np.zeros(n, dtype=np.uint8)
    cls = (np.arange(n) // 2048).astype(np.uint8)  # grouped: no mixed group
    a = ActiveSet(mask, reorder=0.5, classes=2, class_plane=cls)
    a.compact()
    assert int(a.class_fire[0]) == 0
    limit = int(_active_set.STALE_MAX * n)  # 204.8 -> 204 is not enough
    a.stale_count[0] = limit
    a.compact()
    assert int(a.class_fire[0]) == 0
    a.stale_count[0] = limit + 1
    a.compact()
    assert int(a.class_fire[0]) == 1
    # below the span floor the stale term still fires (it has no floor)
    m = np.zeros(64, dtype=np.uint8)
    b = ActiveSet(m, reorder=0.5, classes=2, class_plane=np.zeros(64, np.uint8))
    b.stale_count[0] = 4
    b.compact()
    assert int(b.class_fire[0]) == 1


def test_named_constants():
    assert _active_set.CLASS_THETA == _active_set.DEFAULT_THETA == 0.5
    assert _active_set.STALE_MAX == 0.05
    a = ActiveSet(np.zeros(64, np.uint8), reorder=0.5, classes=2,
                  class_plane=np.zeros(64, np.uint8), class_theta=0.3)
    assert a.class_theta == 0.3
    with pytest.raises(ValueError, match="outside"):
        ActiveSet(np.zeros(64, np.uint8), reorder=0.5, classes=2,
                  class_plane=np.zeros(64, np.uint8), class_theta=0.9)


# --- refusals -------------------------------------------------------------- #
def test_classes_need_reordering_set():
    with pytest.raises(ValueError, match="needs a reordering set"):
        ActiveSet(np.zeros(64, np.uint8), classes=2, class_plane=np.zeros(64, np.uint8))


def test_classes_need_class_plane_and_range():
    m = np.zeros(64, np.uint8)
    with pytest.raises(ValueError, match="needs class_plane"):
        ActiveSet(m, reorder=0.5, classes=2)
    with pytest.raises(ValueError, match="classes is an int"):
        ActiveSet(m, reorder=0.5, classes=1, class_plane=np.zeros(64, np.uint8))
    with pytest.raises(ValueError, match="class_plane needs classes"):
        ActiveSet(m, reorder=0.5, class_plane=np.zeros(64, np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        ActiveSet(m, reorder=0.5, classes=2, class_plane=np.zeros(64, np.int32))


def test_class_api_on_plain_set_refused():
    a = ActiveSet(np.zeros(64, np.uint8), reorder=0.5)
    with pytest.raises(ValueError, match="has no classes"):
        a.class_sort()


def test_class_out_of_range_refused():
    m = np.zeros(64, np.uint8)
    a = ActiveSet(m, reorder=0.5, classes=2, class_plane=np.full(64, 5, np.uint8))
    with pytest.raises(ValueError, match="outside \\[0, 2\\)"):
        a.class_sort()


def test_class_plane_cannot_be_owned_twice():
    a, *_ = _make(64, 2)
    with pytest.raises(ValueError, match="already owns"):
        a.own(a.class_plane)


def test_foreign_set_plane_refused_at_bind():
    from eagle._active_set import check_bound_planes

    a, *_ = _make(64, 2, seed=1)
    b, *_ = _make(64, 2, seed=2)
    adj = np.zeros(64)
    b.own(adj)
    with pytest.raises(ValueError, match="owned by a different ActiveSet"):
        check_bound_planes(a.map, [("mutable", "bar_x", adj)])


# --- thread safety --------------------------------------------------------- #
def test_distinct_sets_sort_concurrently():
    sets = [_make(4200, 4, seed=s)[0] for s in range(8)]
    errs = []

    def work(a):
        try:
            for _ in range(3):
                a.class_sort()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    th = [threading.Thread(target=work, args=(a,)) for a in sets]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs
    assert all(int(a.class_sorts[0]) == 3 for a in sets)


def test_distinct_sets_sorted_concurrently_equal_the_serial_permutation():
    seeds = range(8)
    want = []
    for s in seeds:
        a = _make(4200, 4, seed=s)[0]
        a.compact()
        a.class_sort()
        want.append((a.perm.copy(), a.inv.copy(), a.offsets.copy()))
    sets = [_make(4200, 4, seed=s)[0] for s in seeds]
    start = threading.Barrier(len(sets))
    errs = []

    def work(a):
        try:
            start.wait()
            a.compact()
            a.class_sort()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    th = [threading.Thread(target=work, args=(a,)) for a in sets]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs
    for a, (perm, inv, offsets) in zip(sets, want):
        assert np.array_equal(a.perm, perm) and np.array_equal(a.inv, inv)
        assert np.array_equal(a.offsets, offsets)


def test_one_set_shared_across_threads_is_serialised():
    a, *_ = _make(4200, 4, seed=1)
    errs = []

    def work():
        try:
            for _ in range(5):
                a.class_sort()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    th = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs and int(a.class_sorts[0]) == 30
    assert sorted(a.perm.tolist()) == list(range(4200))


# --- seam ------------------------------------------------------------------ #
def _core():
    from eagle import _core
    return _core


def test_backend_has_reports_both_symbols():
    core = _core()
    assert core.backend_has("eagle_backend_no_such_symbol") is False
    if core.cuda_backend()["loaded"]:
        for sym in ("eagle_backend_compact_classes_device", "eagle_backend_reorder_classes_device"):
            assert core.backend_has(sym) is True


def test_backend_without_class_symbols_refused_naming_backend_and_symbol(monkeypatch):
    core = _core()
    monkeypatch.setattr(core, "backend_has", lambda sym, kind="cuda": False)
    with pytest.raises(RuntimeError, match=r"compact_classes_device.*cuda backend"):
        _active_set._require_class_symbols()
    monkeypatch.setattr(core, "backend_has",
                        lambda sym, kind="cuda": sym.endswith("compact_classes_device"))
    with pytest.raises(RuntimeError, match=r"reorder_classes_device"):
        _active_set._require_class_symbols()


def test_unknown_flag_bit_is_unsupported_never_ignored():
    """Raw descs through the binding: an unknown flag bit is refused (UNSUPPORTED) before any device work."""
    core = _core()
    if not core.backend_has("eagle_backend_compact_classes_device"):
        pytest.skip("no cuda backend with the class symbols")
    with pytest.raises(Exception, match="(?i)flag bits"):
        core.compact_classes_device(0, 0, 0, 0, 8, 0, 4, -1, 0, 0, 0, 0.5, 0, 2, 0.5, 0)
    with pytest.raises(Exception, match="(?i)flag bits"):
        core.reorder_classes_device([(0, 1)], 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 1 << 2, -1, 0, 2, 0, 0)


# --- device faces (host == device, serial oracle) ----------------------------- #
gpu = pytest.mark.gpu


def _pair(n, k, seed, p_dead=0.3):
    cupy = pytest.importorskip("cupy")
    rng = np.random.default_rng(seed)
    mask = (rng.random(n) < p_dead).astype(np.uint8)
    cls = rng.integers(0, k, n).astype(np.uint8)
    pay = np.arange(n, dtype=np.int64) * 7 + 3
    h = ActiveSet(mask.copy(), reorder=0.5, classes=k, class_plane=cls.copy())
    hp = pay.copy()
    h.own(hp)
    dm, dc, dp = cupy.asarray(mask), cupy.asarray(cls), cupy.asarray(pay)
    d = ActiveSet(dm, reorder=0.5, classes=k, class_plane=dc)
    d.own(dp)
    return h, hp, d, dp


def _get(a):
    g = getattr(a, "get", None)
    return g() if g is not None else np.asarray(a)


@gpu
@pytest.mark.parametrize("k", [2, 4])
def test_device_class_sort_equals_host(k):
    h, hp, d, dp = _pair(6000, k, seed=k)
    for s in (h, d):
        s.compact()
        s.class_sort()
    assert np.array_equal(_get(d.perm), h.perm)
    assert np.array_equal(_get(d.offsets), h.offsets)
    assert np.array_equal(_get(dp), hp)
    assert int(_get(d.count)[0]) == int(h.count[0]) == int(_get(d.span)[0])
    assert int(_get(d.class_sorts)[0]) == 1 == int(h.class_sorts[0])
    assert np.array_equal(_get(d.map)[:int(h.count[0])], h.map[:int(h.count[0])])


@gpu
def test_device_within_live_equals_host():
    h, hp, d, dp = _pair(5000, 3, seed=5)
    for s in (h, d):
        s.compact()
        s.class_sort()
    rng = np.random.default_rng(3)
    live = int(h.count[0])
    new = rng.integers(0, 3, live).astype(np.uint8)
    h.class_plane[:live] = new
    d.class_plane[:live] = pytest.importorskip("cupy").asarray(new)
    for s in (h, d):
        s.class_sort(within_live=True)
    assert np.array_equal(_get(d.perm), h.perm)
    assert np.array_equal(_get(d.offsets), h.offsets)
    assert int(_get(d.count)[0]) == live and int(_get(d.span)[0]) == live


@gpu
def test_device_trigger_equals_host():
    for seed, p_dead in ((1, 0.0), (2, 0.5), (3, 0.9)):
        h, hp, d, dp = _pair(8192, 4, seed=seed, p_dead=p_dead)
        for stale in (0, 100, 1000):
            h.stale_count[0] = stale
            d.stale_count[0] = stale
            h.compact()
            d.compact()
            assert int(_get(d.mixed32)[0]) == int(h.mixed32[0])
            assert int(_get(d.class_fire)[0]) == int(h.class_fire[0])
            assert int(_get(d.live32)[0]) == int(h.live32[0])


@gpu
def test_device_restore_and_pairing():
    h, hp, d, dp = _pair(5000, 4, seed=8)
    d.compact()
    d.class_sort()
    assert d.permuted
    assert np.array_equal(_get(dp), (d.perm.get() * 7 + 3))
    d.restore()
    assert np.array_equal(_get(dp), np.arange(5000) * 7 + 3)


@gpu
def test_device_distinct_sets_sorted_concurrently_equal_the_host_permutation():
    cupy = pytest.importorskip("cupy")
    pairs = [_pair(6000, 4, seed=s) for s in range(4)]
    want = []
    for h, _hp, _d, _dp in pairs:
        h.compact()
        h.class_sort()
        want.append((h.perm.copy(), h.offsets.copy()))
    start = threading.Barrier(len(pairs))
    errs = []

    def work(d):
        try:
            with cupy.cuda.Device(cupy.cuda.runtime.getDevice()):
                start.wait()
                d.compact()
                d.class_sort()
                cupy.cuda.get_current_stream().synchronize()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    th = [threading.Thread(target=work, args=(p[2],)) for p in pairs]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs
    for (_h, _hp, d, _dp), (perm, offsets) in zip(pairs, want):
        assert np.array_equal(_get(d.perm), perm) and np.array_equal(_get(d.offsets), offsets)
