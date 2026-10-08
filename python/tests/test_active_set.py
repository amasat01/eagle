# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Active-set compaction (:class:`eagle.ActiveSet`, the ``filtering`` seam group).

Rows:

* the map is ``flatnonzero(~mask)`` and the count its size -- device (the
  plugin's ``eagle_backend_compact_device``) and host (NumPy), over the sizes
  that hit the scan's edges (empty, one sample, one block, one past a block,
  multi-level); a keep-mask run of the raw entry; the scratch query and the
  2^31 refusal;
* the compaction CAPTURES: recorded once into a :class:`eagle.GraphPipeline`,
  each replay follows a mask changed between replays (a replay that baked the
  first answer in would be red);
* a sequence body of :func:`eagle.repeat_while`: the plain part runs every
  iteration and the skippable part only when its guard holds (two witnesses);
  :meth:`ActiveSet.when_finished` compacts only when the finished counter moved;
* the cadence helper refuses ``every < 4``;
* end to end: a hawk kernel built with ``Guard(active_set=True)`` integrates a
  thinning batch bit-identically to the same kernel without the map, on the
  device (graph-captured loop with compaction) and on the host (eager loop),
  for spread and uniform stop steps; a multi-partition bind is refused;
* the physical reorder end to end (``ActiveSet(reorder=0.5)``, the set owning
  every per-sample plane): bit-identical to the map-only run and to the
  map-free run, host and device, for spread, uniform and grouped stop steps
  (grouped = the spread stops sorted so the live set stays contiguous), at
  N = 1e3 and 1e5; sample ``i`` read through ``inv`` right after a fire equals
  the map-free run at the same step; the reorder count stays within its bound
  (spread fires between 3 and ceil(log2 100) + 1 times, grouped and uniform
  never, the permuted spans sum to at most N / (1 - theta)); a reorder that
  leaves one owned plane behind turns these rows red; bind refuses a per-sample
  plane the set neither owns nor declares indirect, and ``own()`` after bind.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import eagle
from eagle import ActiveSet, SkipGuard, compaction_body, repeat_while, skippable

_SIZES = (0, 1, 7, 256, 257, 4096, 100_003)


def _mask(n, seed, p=0.6):
    return np.random.default_rng([seed, n]).random(n) < p


# --------------------------------------------------------------------------- #
# The map itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", _SIZES)
def test_host_map_is_the_live_indices(n):
    m = _mask(n, 1)
    a = ActiveSet(m)
    assert a.live == n and np.array_equal(a.map, np.arange(n, dtype=np.int32))
    a.compact()
    want = np.flatnonzero(~m)
    assert a.live == want.size
    assert np.array_equal(a.map[: a.live], want)


@pytest.mark.gpu
@pytest.mark.parametrize("n", _SIZES)
def test_device_map_is_the_live_indices(n):
    cp = pytest.importorskip("cupy")
    m = _mask(n, 2)
    a = ActiveSet(cp.asarray(m))
    a.compact()
    want = np.flatnonzero(~m)
    assert a.live == want.size
    assert np.array_equal(a.map.get()[: a.live], want)
    a.reset()
    assert a.live == n and np.array_equal(a.map.get(), np.arange(n))


@pytest.mark.gpu
def test_device_keep_mask_and_scratch_query():
    cp = pytest.importorskip("cupy")
    from eagle import _core

    n = 1000
    keep = cp.asarray(_mask(n, 3).astype(np.uint8))
    nbytes = _core.compact_device(0, 0, 0, 0, n)
    assert nbytes >= 3 * n * 4
    scratch = cp.empty(nbytes, dtype=cp.uint8)
    idx = cp.full(n, -1, dtype=cp.int32)
    count = cp.zeros(1, dtype=cp.uint32)
    _core.compact_device(int(keep.data.ptr), int(idx.data.ptr), int(count.data.ptr),
                         int(scratch.data.ptr), n,
                         int(cp.cuda.get_current_stream().ptr), 0)  # keep-mask
    want = np.flatnonzero(keep.get())
    assert int(count.get()[0]) == want.size
    assert np.array_equal(idx.get()[: want.size], want)
    assert (idx.get()[want.size:] == -1).all(), "slots past the count are untouched"


@pytest.mark.gpu
def test_device_refuses_more_than_int32_samples():
    from eagle import _core

    with pytest.raises(Exception, match="2\\^31"):
        _core.compact_device(8, 8, 8, 8, 2**31)


def test_mask_is_checked():
    with pytest.raises(ValueError, match="bool or uint8"):
        ActiveSet(np.zeros(4, dtype=np.int32))
    with pytest.raises(ValueError, match="one-dimensional"):
        ActiveSet(np.zeros((2, 2), dtype=bool))


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_captured_compaction_follows_the_mask_on_every_replay():
    """RED: a compaction whose answer were baked in at capture (or computed on
    the host) would return the first mask's map on the later replays."""
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    n = 5000
    mask = cp.zeros(n, dtype=cp.bool_)
    a = ActiveSet(mask)
    pipe = GraphPipeline().add(a.compact).build()
    for seed in range(3):
        m = _mask(n, 10 + seed, p=0.3 + 0.2 * seed)
        mask.set(m)
        pipe.launch()
        pipe.synchronize()
        want = np.flatnonzero(~m)
        assert a.live == want.size
        assert np.array_equal(a.map.get()[: a.live], want)


_SRC = r"""
extern "C" __global__ void as_tracer(int* counter) { atomicAdd(counter, 1); }
extern "C" __global__ void as_finish_one(bool* mask, unsigned int* finished, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int k = *finished;
        if (k < (unsigned int)n) { mask[k] = true; *finished = k + 1; }
    }
}
"""


@pytest.mark.gpu
def test_sequence_body_runs_parts_in_order_and_guards_the_skippable_one():
    """RED: drop the inner IF weave of a sequence part and the guarded tracer
    counts every iteration; drop the plain part and its tracer counts zero."""
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    tracer = cp.RawKernel(_SRC, "as_tracer")
    plain_hits = cp.zeros(1, dtype=cp.int32)
    guarded_hits = cp.zeros(1, dtype=cp.int32)
    flag = cp.zeros(1, dtype=cp.uint32)  # the skippable part's guard: never set
    live = cp.asarray([5], dtype=cp.uint32)

    def plain():
        tracer((1,), (1,), (plain_hits,))
        cp.subtract(live, cp.uint32(1), out=live)

    body = (plain, skippable(lambda: tracer((1,), (1,), (guarded_hits,)),
                             SkipGuard.nonzero(flag)))
    loop = repeat_while(body, SkipGuard.nonzero(live), 50)
    GraphPipeline().add(loop).build().launch()
    cp.cuda.Device().synchronize()
    assert loop.iterations() == 5
    assert int(plain_hits.get()[0]) == 5
    assert int(guarded_hits.get()[0]) == 0


@pytest.mark.gpu
def test_when_finished_compacts_only_when_the_counter_moved():
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    finish = cp.RawKernel(_SRC, "as_finish_one")
    n = 64
    mask = cp.zeros(n, dtype=cp.bool_)
    finished = cp.zeros(1, dtype=cp.uint32)
    a = ActiveSet(mask)
    pipe = GraphPipeline().add(a.when_finished(finished)).build()
    pipe.launch()  # nothing finished: the map stays the identity
    pipe.synchronize()
    assert a.live == n
    finish((1,), (1,), (mask, finished, np.int32(n)))
    finish((1,), (1,), (mask, finished, np.int32(n)))
    pipe.launch()
    pipe.synchronize()
    assert a.live == n - 2
    assert np.array_equal(a.map.get()[: a.live], np.arange(2, n))
    # the counter did not move: the guarded compaction must not run, so a
    # doctored count survives the replay
    a.count.fill(7)
    pipe.launch()
    pipe.synchronize()
    assert a.live == 7


def test_compaction_body_cadence():
    a = ActiveSet(np.zeros(8, dtype=bool))
    with pytest.raises(ValueError, match="below 4"):
        compaction_body(lambda: None, a, every=3)
    _steps, tail = compaction_body(lambda: None, a)
    assert tail == a.compact
    assert eagle._active_set.DEFAULT_EVERY == 16
    calls = []
    steps, _ = compaction_body(lambda: calls.append(1), a, every=5)
    steps()
    assert len(calls) == 5


# --------------------------------------------------------------------------- #
# End to end with a hawk kernel
# --------------------------------------------------------------------------- #
_DT = 1e-3


def _oscillator(kind):
    hawk = pytest.importorskip("hawk")
    from hawk import Mutable, Param, Scalar, Terminated

    def oscillator_step(omega: Scalar, zeta: Scalar, dt: Param,
                        terminated: Terminated, x: Mutable[Scalar],
                        v: Mutable[Scalar], k: Mutable[Scalar]):
        x0 = x
        v0 = v
        a = -(omega * omega) * x0 - 2.0 * zeta * omega * v0
        x = x0 + dt * v0
        v = v0 + dt * a
        k = k + 1.0

    return hawk.kernel(oscillator_step, kind=kind)


def _have_gpu():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    """(map-free, map) x (device, host) plans of the same oscillator step."""
    pytest.importorskip("hawk")
    import ctypes
    from types import SimpleNamespace

    from hawk import _core as hcore
    from hawk.artifact import build_bundle, plan_view
    from hawk.ext import Guard, Kind

    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle.registry import load_manifest

    gpu = _have_gpu()
    targets = ("host", "cuda") if gpu else ("host",)
    out = {}
    for label, kind in (("plain", None),
                        ("map", Kind("active_osc", guard=Guard(active_set=True)))):
        kern = _oscillator(kind)
        work = tmp_path_factory.mktemp(label)
        bundle = build_bundle([kern], work, targets=targets)
        sidecar = next(a.sidecar for a in bundle.artifacts if a.name == kern.name)
        view = plan_view(sidecar)
        host_entry = view.pop("host_entry")
        so = work / f"{kern.name}.so"
        lib = hcore.HostLibrary(str(so))
        cdll = ctypes.CDLL(str(so))
        host = SimpleNamespace(
            host_entry=ctypes.cast(getattr(cdll, host_entry), ctypes.c_void_p).value,
            _keepalive=(lib, cdll), **view)
        out[(label, "host")] = eplan.plan(host, structure=eexec.HostTeam)
        if gpu:
            registry = load_manifest(work / "manifest.json")
            loaded = registry[kern.name]
            dev = SimpleNamespace(device_function=loaded.fn.kernel.ptr,
                                  _keepalive=(registry, loaded), **view)
            out[(label, "device")] = eplan.plan(dev, structure=eexec.DeviceKernel)
    return out


def _inputs(n, distribution, steps):
    rng = np.random.default_rng([7, n, steps, distribution == "spread"])
    d = dict(omega=rng.uniform(1.0, 10.0, n), zeta=rng.uniform(0.01, 0.2, n),
             x=rng.uniform(-1.0, 1.0, n), v=rng.uniform(-1.0, 1.0, n))
    if distribution == "spread":
        lo, hi = math.log(steps / 100), math.log(steps)
        nstop = np.clip(np.ceil(np.exp(rng.uniform(lo, hi, n))), 1, steps)
    else:
        nstop = np.full(n, float(steps))
    d["nstop"] = nstop.astype(np.float64)
    return d


def _host_run(plan, inp, n, steps, active):
    x, v = inp["x"].copy(), inp["v"].copy()
    k = np.zeros(n)
    term = np.zeros(n, dtype=bool)
    extra, aset = {}, None
    if active:
        aset = ActiveSet(term)
        extra = aset.planes()
    bound = plan.bind(omega=inp["omega"], zeta=inp["zeta"],
                      dt=_DT, terminated=term, x=x, v=v, k=k, **extra)
    for s in range(steps):
        bound.launch()
        term |= k >= inp["nstop"]
        if aset is not None and s % 4 == 3:
            aset.compact()
    return x, v, k, (aset.live if aset is not None else None)


@pytest.mark.parametrize("distribution", ["spread", "uniform"])
def test_host_map_kernel_is_bit_identical(plans, distribution):
    n, steps = 3000, 200
    inp = _inputs(n, distribution, steps)
    ref = _host_run(plans[("plain", "host")], inp, n, steps, active=False)
    got = _host_run(plans[("map", "host")], inp, n, steps, active=True)
    for r, g in zip(ref[:3], got[:3]):
        assert np.array_equal(r, g)
    assert np.array_equal(got[2], inp["nstop"])
    if distribution == "spread":
        assert got[3] < n, "the host compaction never shrank the live set"


_MARK = r"""
extern "C" __global__ void as_mark(const double* k, const double* nstop, bool* term,
                                   unsigned int* done, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !term[i] && k[i] >= nstop[i]) { term[i] = true; atomicAdd(done, 1u); }
}
"""


def _device_run(plan, inp, n, steps, active, every=16):
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    mark = cp.RawKernel(_MARK, "as_mark")
    g = {key: cp.asarray(inp[key]) for key in ("omega", "zeta", "nstop", "x", "v")}
    k = cp.zeros(n)
    term = cp.zeros(n, dtype=cp.bool_)
    done = cp.zeros(1, dtype=cp.uint32)
    total = cp.asarray([n], dtype=cp.uint32)
    extra, aset = {}, None
    if active:
        aset = ActiveSet(term)
        extra = aset.planes()
    bound = plan.bind(omega=g["omega"], zeta=g["zeta"], dt=_DT,
                      terminated=term, x=g["x"], v=g["v"], k=k, **extra)
    grid = ((n + 255) // 256,)

    def step():
        bound.launch()
        mark(grid, (256,), (k, g["nstop"], term, done, cp.int64(n)))

    if active:
        body = compaction_body(step, aset, every=every, finished=done)
        loop = repeat_while(body, SkipGuard(done, 0, total, 0), -(-steps // every))
    else:
        loop = repeat_while(step, SkipGuard(done, 0, total, 0), steps)
    GraphPipeline().add(loop).build().launch()
    cp.cuda.Device().synchronize()
    live_left = aset.live if aset is not None else None
    return g["x"].get(), g["v"].get(), k.get(), live_left


@pytest.mark.gpu
@pytest.mark.parametrize("distribution", ["spread", "uniform"])
def test_device_map_kernel_is_bit_identical_in_a_captured_loop(plans, distribution):
    """RED: a prologue that ignored the map's count (or read sample t, not
    map[t]) would integrate the wrong samples past the first compaction."""
    if ("map", "device") not in plans:
        pytest.skip("no CUDA device")
    n, steps = 20_000, 300
    inp = _inputs(n, distribution, steps)
    ref = _device_run(plans[("plain", "device")], inp, n, steps, active=False)
    got = _device_run(plans[("map", "device")], inp, n, steps, active=True)
    for r, gg in zip(ref[:3], got[:3]):
        assert np.array_equal(r, gg)
    assert np.array_equal(got[2], inp["nstop"])
    if distribution == "spread":
        assert got[3] < n, "the captured compaction never shrank the live set"


@pytest.mark.gpu
def test_map_kernel_refuses_a_split_batch(plans):
    if ("map", "device") not in plans:
        pytest.skip("no CUDA device")
    cp = pytest.importorskip("cupy")
    import eagle.exec as eexec
    from eagle import plan as eplan

    p = plans[("map", "device")]
    split = eplan.plan(p.plugin, structure=eexec.DeviceKernel, npartitions=2)
    n = 64
    term = cp.zeros(n, dtype=cp.bool_)
    a = ActiveSet(term)
    z = cp.zeros(n)
    with pytest.raises(ValueError, match="single partition"):
        split.bind(omega=z, zeta=z, dt=_DT, terminated=term, x=z.copy(),
                   v=z.copy(), k=z.copy(), **a.planes())


# --------------------------------------------------------------------------- #
# The physical reorder, end to end
# --------------------------------------------------------------------------- #
_THETA = 0.5


def _inputs_dist(n, distribution, steps):
    if distribution != "grouped":
        return _inputs(n, distribution, steps)
    inp = _inputs(n, "spread", steps)
    inp["nstop"] = np.sort(inp["nstop"])[::-1].copy()  # live set always contiguous
    return inp


def _host_plain(plan, inp, n, steps, record=()):
    x, v = inp["x"].copy(), inp["v"].copy()
    k = np.zeros(n)
    term = np.zeros(n, dtype=bool)
    bound = plan.bind(omega=inp["omega"], zeta=inp["zeta"], dt=_DT, terminated=term,
                      x=x, v=v, k=k)
    snaps = {}
    for s in range(steps):
        bound.launch()
        term |= k >= inp["nstop"]
        if s in record:
            snaps[s] = (x.copy(), v.copy())
    return x, v, k, snaps


def _host_reorder(plan, inp, n, steps, every=4, broken=False):
    """The map kernel with the reorder on the host face; returns the final
    planes after restore() plus the run's reorder statistics and the first
    post-fire snapshot read through ``inv``."""
    own = {key: inp[key].copy() for key in ("omega", "zeta", "x", "v", "nstop")}
    k = np.zeros(n)
    term = np.zeros(n, dtype=bool)
    aset = ActiveSet(term, reorder=_THETA).own(*own.values(), k)
    bound = plan.bind(omega=own["omega"], zeta=own["zeta"], dt=_DT, terminated=term,
                      x=own["x"], v=own["v"], k=k, **aset.planes())
    if broken:  # a reorder that leaves v behind
        aset._owned = [p for p in aset._owned if p is not own["v"]]
    spans, snap = 0, None
    for s in range(steps):
        bound.launch()
        term |= k >= own["nstop"]
        if s % every == every - 1:
            aset.compact()
            if int(aset.fire[0]):
                spans += int(aset.span[0])
                aset.reorder()
                if snap is None:
                    snap = (s, own["x"][aset.inv].copy(), own["v"][aset.inv].copy())
    in_order = aset.in_sample_order(own["x"])
    fires = int(aset.reorders[0])
    aset.restore()
    assert np.array_equal(in_order, own["x"])
    return own["x"], own["v"], k, dict(fires=fires, spans=spans, snap=snap)


def _fire_bounds(distribution, n, fires):
    if n < 100_000:
        return  # a small batch's last few samples can fire on their own
    if distribution == "spread":
        assert 3 <= fires <= math.ceil(math.log2(100)) + 1, fires
    else:
        assert fires == 0, (distribution, fires)


@pytest.mark.parametrize("n", [1000, 100_000])
@pytest.mark.parametrize("distribution", ["spread", "uniform", "grouped"])
def test_host_reorder_is_bit_identical(plans, distribution, n):
    steps = 200
    inp = _inputs_dist(n, distribution, steps)
    probe = _host_reorder(plans[("map", "host")], inp, n, steps)[3]
    record = () if probe["snap"] is None else (probe["snap"][0],)
    ref = _host_plain(plans[("plain", "host")], inp, n, steps, record=record)
    step1 = _host_run(plans[("map", "host")], inp, n, steps, active=True)
    got = _host_reorder(plans[("map", "host")], inp, n, steps)
    for r, s1, g in zip(ref[:3], step1[:3], got[:3]):
        assert np.array_equal(r, g) and np.array_equal(s1, g)
    stats = got[3]
    _fire_bounds(distribution, n, stats["fires"])
    assert stats["spans"] <= n / (1 - _THETA)
    if stats["snap"] is not None:
        s, x_snap, v_snap = stats["snap"]
        assert np.array_equal(x_snap, ref[3][s][0]) and np.array_equal(v_snap, ref[3][s][1])


def test_host_broken_reorder_turns_the_identity_rows_red(plans):
    n, steps = 100_000, 200
    inp = _inputs(n, "spread", steps)
    ref = _host_plain(plans[("plain", "host")], inp, n, steps)
    got = _host_reorder(plans[("map", "host")], inp, n, steps, broken=True)
    assert got[3]["fires"] >= 1
    assert np.array_equal(ref[0], got[0]) is False or np.array_equal(ref[1], got[1]) is False


def _device_reorder(plan, inp, n, steps, every=16, broken=False):
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    mark = cp.RawKernel(_MARK, "as_mark")
    g = {key: cp.asarray(inp[key]) for key in ("omega", "zeta", "nstop", "x", "v")}
    k = cp.zeros(n)
    term = cp.zeros(n, dtype=cp.bool_)
    done = cp.zeros(1, dtype=cp.uint32)
    total = cp.asarray([n], dtype=cp.uint32)
    aset = ActiveSet(term, reorder=_THETA).own(*g.values(), k)
    bound = plan.bind(omega=g["omega"], zeta=g["zeta"], dt=_DT, terminated=term,
                      x=g["x"], v=g["v"], k=k, **aset.planes())
    if broken:
        aset._owned = [p for p in aset._owned if p is not g["v"]]
    grid = ((n + 255) // 256,)

    def step():
        bound.launch()
        mark(grid, (256,), (k, g["nstop"], term, done, cp.int64(n)))

    body = compaction_body(step, aset, every=every, finished=done)
    assert len(body) == (3 if n >= eagle._active_set.MIN_REORDER_SPAN else 2)
    loop = repeat_while(body, SkipGuard(done, 0, total, 0), -(-steps // every))
    GraphPipeline().add(loop).build().launch()
    cp.cuda.Device().synchronize()
    fires = int(aset.reorders.get()[0])
    in_order = aset.in_sample_order(g["x"]).get()
    aset.restore()
    assert np.array_equal(in_order, g["x"].get())
    return g["x"].get(), g["v"].get(), k.get(), fires


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1000, 100_000])
@pytest.mark.parametrize("distribution", ["spread", "uniform", "grouped"])
def test_device_reorder_is_bit_identical_in_a_captured_loop(plans, distribution, n):
    """RED: a reorder that moved the planes but not the stop steps (or read the
    map as sample ids after it) integrates the wrong samples from the first
    fire on."""
    if ("map", "device") not in plans:
        pytest.skip("no CUDA device")
    steps = 300
    inp = _inputs_dist(n, distribution, steps)
    ref = _device_run(plans[("plain", "device")], inp, n, steps, active=False)
    step1 = _device_run(plans[("map", "device")], inp, n, steps, active=True)
    got = _device_reorder(plans[("map", "device")], inp, n, steps)
    for r, s1, gg in zip(ref[:3], step1[:3], got[:3]):
        assert np.array_equal(r, gg) and np.array_equal(s1, gg)
    assert np.array_equal(got[2], inp["nstop"])
    _fire_bounds(distribution, n, got[3])


@pytest.mark.gpu
def test_device_snapshot_after_a_fire_and_broken_reorder(plans):
    """Sample i read through inv at the step right after a fire equals the
    map-free run at that step; a reorder that leaves v behind turns the
    captured identity row red."""
    if ("map", "device") not in plans:
        pytest.skip("no CUDA device")
    cp = pytest.importorskip("cupy")
    n, steps, every = 100_000, 300, 16
    inp = _inputs(n, "spread", steps)
    broken = _device_reorder(plans[("map", "device")], inp, n, steps, broken=True)
    ref = _device_run(plans[("plain", "device")], inp, n, steps, active=False)
    assert broken[3] >= 1
    assert not (np.array_equal(ref[0], broken[0]) and np.array_equal(ref[1], broken[1]))

    # eager device loop: compaction + reorder by hand, snapshot after the first fire
    mark = cp.RawKernel(_MARK, "as_mark")
    g = {key: cp.asarray(inp[key]) for key in ("omega", "zeta", "nstop", "x", "v")}
    k, term = cp.zeros(n), cp.zeros(n, dtype=cp.bool_)
    done = cp.zeros(1, dtype=cp.uint32)
    aset = ActiveSet(term, reorder=_THETA).own(*g.values(), k)
    bound = plans[("map", "device")].bind(omega=g["omega"], zeta=g["zeta"], dt=_DT,
                                          terminated=term, x=g["x"], v=g["v"], k=k,
                                          **aset.planes())
    grid = ((n + 255) // 256,)
    fired_at = None
    for s in range(steps):
        bound.launch()
        mark(grid, (256,), (k, g["nstop"], term, done, cp.int64(n)))
        if s % every == every - 1:
            aset.compact()
            if int(aset.fire.get()[0]):
                aset.reorder()
                fired_at = s
                snap = (g["x"][aset.inv].get(), g["v"][aset.inv].get())
                break
    assert fired_at is not None
    # the map-free device run, stopped at the same step
    r = {key: cp.asarray(inp[key]) for key in ("omega", "zeta", "nstop", "x", "v")}
    rk, rterm = cp.zeros(n), cp.zeros(n, dtype=cp.bool_)
    rdone = cp.zeros(1, dtype=cp.uint32)
    rbound = plans[("plain", "device")].bind(omega=r["omega"], zeta=r["zeta"], dt=_DT,
                                             terminated=rterm, x=r["x"], v=r["v"], k=rk)
    for _ in range(fired_at + 1):
        rbound.launch()
        mark(grid, (256,), (rk, r["nstop"], rterm, rdone, cp.int64(n)))
    assert np.array_equal(snap[0], r["x"].get())
    assert np.array_equal(snap[1], r["v"].get())


def test_bind_refuses_a_plane_the_reordering_set_does_not_cover(plans):
    n = 64
    term = np.zeros(n, dtype=bool)
    z = {key: np.zeros(n) for key in ("omega", "zeta", "x", "v", "k")}
    a = ActiveSet(term, reorder=_THETA).own(z["omega"], z["zeta"], z["x"], z["v"])
    p = plans[("map", "host")]
    with pytest.raises(ValueError, match=r"neither owns it nor declares it indirect"):
        p.bind(dt=_DT, terminated=term, **z, **a.planes())
    a.indirect(z["k"])
    p.bind(dt=_DT, terminated=term, **z, **a.planes())
    with pytest.raises(ValueError, match="fixed"):
        a.own(np.zeros(n))


@pytest.mark.gpu
@pytest.mark.parametrize("reorder", [None, 0.5])
def test_device_reset_allocates_nothing(reorder):
    """The identity map (and a reordering set's perm/inv) is written in
    place: a batch-sized temporary would stay in cupy's pool after it is
    freed and count as the run's device memory."""
    cp = pytest.importorskip("cupy")
    from cupy.cuda import memory_hook

    class Count(memory_hook.MemoryHook):
        name = "count"
        allocated = 0

        def malloc_preprocess(self, **kw):  # every pool request, served from the pool or not
            Count.allocated += kw["size"]

    s = ActiveSet(cp.zeros(1 << 20, dtype=cp.bool_), reorder=reorder)
    s.map.fill(7)
    with Count():
        s.reset()
    cp.cuda.Device().synchronize()
    assert Count.allocated == 0
    assert bool((s.map == cp.arange(1 << 20, dtype=cp.int32)).all())
