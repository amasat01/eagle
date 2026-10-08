# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.until_done`` / ``eagle.run_until_done``: one call runs a finishing
kernel's batch until every sample is done.

What is pinned:

* the ``finish`` sidecar reader (:func:`eagle.sidecar.read_finish`): absent is
  ``None``, every malformed shape is a ``ValueError`` naming the artifact;
* the refusals (no ``finish``, a reserved plane passed, ``every``/``reorder``
  on a plain artifact, a bad ``max_steps``);
* the oracle: the runner's planes are bit-equal to today's explicit path
  (a non-finishing step kernel plus a separate stop rule in a
  ``repeat_while``), the mask is all set, ``finished == n``, ``done``;
* non-vacuity: a batch whose longest sample stops at ``S`` steps runs at
  most ``S`` loop iterations (``S / every + 1`` with compaction) and reports
  them exactly; the kernel emitted without its epilogue's count turns the
  row red (the loop runs to its cap and ``done`` is false);
* monotonicity: samples set before the run keep their planes; ``reset()``
  then runs them;
* compaction and reorder: an artifact built with
  ``Guard(active_set=True)`` through the same call, bit-equal to the oracle,
  with ``compactions >= 1`` on a spread batch and, with ``reorder=0.5`` at
  ``n >= MIN_REORDER_SPAN``, ``reorders >= 1`` and sample order restored.

The finishing kernel is a hawk step kernel that assigns its mask
(``terminated = k >= nstop``); the explicit path's oracle is the same
step without the assignment, followed by a separate stop-rule kernel.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import eagle
from eagle import SkipGuard, compaction_body, repeat_while
from eagle._active_set import MIN_REORDER_SPAN
from eagle.sidecar import read_finish

_DT = 1e-3
_FINISH = {"mask": "terminated", "counter": "finished_count", "steps": 1}


# --------------------------------------------------------------------------- #
# The sidecar reader
# --------------------------------------------------------------------------- #
def test_read_finish_absent_is_none():
    assert read_finish({"arg_spec": []}, name="k") is None


def test_read_finish_returns_the_declaration():
    meta = {"finish": dict(_FINISH, steps=4), "terminated_readonly": False}
    assert read_finish(meta, name="k") == dict(_FINISH, steps=4)


@pytest.mark.parametrize("bad", [
    [],
    {"mask": "terminated", "counter": "finished_count"},
    dict(_FINISH, extra=1),
    dict(_FINISH, mask=""),
    dict(_FINISH, counter="done"),
    dict(_FINISH, steps=0),
    dict(_FINISH, steps=True),
    dict(_FINISH, steps=2.0),
])
def test_read_finish_refuses_a_malformed_declaration(bad):
    with pytest.raises(ValueError, match="my_kernel"):
        read_finish({"finish": bad}, name="my_kernel")


def test_read_finish_refuses_a_readonly_mask_beside_it():
    with pytest.raises(ValueError, match="terminated_readonly"):
        read_finish({"finish": _FINISH, "terminated_readonly": True}, name="k")


def test_the_counter_plane_name_is_the_sidecar_counter():
    assert eagle.FINISHED_PLANE == "finished_count" == _FINISH["counter"]


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def _oscillator(kind, finishing, steps=1):
    hawk = pytest.importorskip("hawk")
    from hawk import Mutable, Param, Scalar, Terminated
    from hawk import math as m

    # `min(1, nstop)` is exactly 1 (nstop >= 1): it keeps nstop a bound plane of
    # the non-finishing kernel too, so the reorder moves it with the rest.
    if finishing:
        def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                            terminated: Terminated, x: Mutable[Scalar],
                            v: Mutable[Scalar], k: Mutable[Scalar]):
            x0 = x
            v0 = v
            a = -(omega * omega) * x0 - 2.0 * zeta * omega * v0
            x = x0 + dt * v0
            v = v0 + dt * a
            k1 = k + m.min(1.0, nstop)
            k = k1
            terminated = k1 >= nstop
    else:
        def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                            terminated: Terminated, x: Mutable[Scalar],
                            v: Mutable[Scalar], k: Mutable[Scalar]):
            x0 = x
            v0 = v
            a = -(omega * omega) * x0 - 2.0 * zeta * omega * v0
            x = x0 + dt * v0
            v = v0 + dt * a
            k = k + m.min(1.0, nstop)

    return hawk.kernel(oscillator_step, kind=kind, steps=steps)


def _have_gpu():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _build(kern, work, targets, cache_dir=None, defines=()):
    import ctypes

    from hawk import _core as hcore
    from hawk.artifact import build_bundle, plan_view

    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle.registry import load_manifest

    kw = {} if cache_dir is None else {"cache_dir": cache_dir}
    bundle = build_bundle([kern], work, targets=targets, defines=tuple(defines), **kw)
    sidecar = next(a.sidecar for a in bundle.artifacts if a.name == kern.name)
    view = plan_view(sidecar)
    host_entry = view.pop("host_entry")
    out = {}
    so = work / f"{kern.name}.so"
    lib = hcore.HostLibrary(str(so))
    cdll = ctypes.CDLL(str(so))
    host = SimpleNamespace(
        host_entry=ctypes.cast(getattr(cdll, host_entry), ctypes.c_void_p).value,
        name=kern.name, _keepalive=(lib, cdll), **view)
    out["host"] = eplan.plan(host, structure=eexec.HostTeam)
    if "cuda" in targets:
        registry = load_manifest(work / "manifest.json")
        loaded = registry[kern.name]
        dev = SimpleNamespace(device_function=loaded.fn.kernel.ptr, name=kern.name,
                              _keepalive=(registry, loaded), **view)
        out["device"] = eplan.plan(dev, structure=eexec.DeviceKernel)
    return out


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    """``plans[(label, target)]`` for label in plain/map (the explicit path's
    non-finishing kernel) and fplain/fmap (the finishing kernel)."""
    pytest.importorskip("hawk")
    from hawk.ext import Guard, Kind

    targets = ("host", "cuda") if _have_gpu() else ("host",)
    out = {}
    for label, guard in (("plain", None), ("map", Guard(active_set=True))):
        for prefix, finishing in (("", False), ("f", True)):
            kind = (None if guard is None
                    else Kind(f"ud_{prefix}{label}", guard=guard))
            built = _build(_oscillator(kind, finishing),
                           tmp_path_factory.mktemp(prefix + label), targets)
            out.update({(prefix + label, t): p for t, p in built.items()})
    return out


def _with_finish(p):
    import dataclasses

    plugin = SimpleNamespace(**vars(p.plugin), finish=dict(_FINISH))
    return dataclasses.replace(p, plugin=plugin)


# --------------------------------------------------------------------------- #
# The explicit path's stop rule (today's spelling, the oracle)
# --------------------------------------------------------------------------- #
_MARK = r"""
extern "C" __global__ void ud_mark(const double* k, const double* nstop, bool* term,
                                   unsigned int* done, long long n, int count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !term[i] && k[i] >= nstop[i]) {
        term[i] = true;
        if (count) atomicAdd(done, 1u);
    }
}
"""


def _runner(plans, label, target, **kw):
    """``eagle.until_done`` over the finishing plan."""
    return eagle.until_done(plans[(label, target)], **kw)


def _inputs(n, distribution, steps, seed=7):
    rng = np.random.default_rng([seed, n, steps, distribution == "spread"])
    d = dict(omega=rng.uniform(1.0, 10.0, n), zeta=rng.uniform(0.01, 0.2, n),
             x=rng.uniform(-1.0, 1.0, n), v=rng.uniform(-1.0, 1.0, n))
    if distribution == "spread":
        lo, hi = math.log(steps / 100), math.log(steps)
        nstop = np.clip(np.ceil(np.exp(rng.uniform(lo, hi, n))), 1, steps)
    else:
        nstop = np.full(n, float(steps))
    d["nstop"] = nstop.astype(np.float64)
    return d


def _planes(inp, xp):
    """Fresh caller planes in ``xp`` (numpy or cupy)."""
    n = inp["x"].shape[0]
    out = {key: xp.asarray(inp[key]).copy() for key in ("omega", "zeta", "nstop",
                                                       "x", "v")}
    out["k"] = xp.zeros(n)
    return out


def _xp(target):
    if target == "device":
        return pytest.importorskip("cupy")
    return np


def _need(plans, key):
    if key not in plans:
        pytest.skip("no CUDA device")


def _get(a):
    return a.get() if hasattr(a, "get") else a


def _explicit(plans, target, inp, max_steps):
    """Today's explicit path: the non-finishing kernel, the stop rule after
    each launch, a repeat_while on `done != total`."""
    xp = _xp(target)
    g = _planes(inp, xp)
    n = inp["x"].shape[0]
    term = xp.zeros(n, dtype=xp.bool_)
    done = xp.zeros(1, dtype=xp.uint32)
    total = xp.asarray([n], dtype=xp.uint32)
    bound = plans[("plain", target)].bind(dt=_DT, terminated=term, **g)
    if target == "device":
        import cupy as cp

        from eagle.pipeline import GraphPipeline

        mark = cp.RawKernel(_MARK, "ud_mark")

        def step():
            bound.launch()
            mark(((n + 255) // 256,), (256,),
                 (g["k"], g["nstop"], term, done, cp.int64(n), np.int32(1)))

        loop = repeat_while(step, SkipGuard(done, 0, total, 0), max_steps)
        GraphPipeline().add(loop).build().launch()
    else:
        def step():
            bound.launch()
            new = ~term & (g["k"] >= g["nstop"])
            term[...] |= new
            done[0] += np.uint32(np.count_nonzero(new))

        loop = repeat_while(step, SkipGuard(done, 0, total, 0), max_steps)
        loop()
    return {key: _get(g[key]) for key in ("x", "v", "k")}, _get(term)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_a_kernel_without_finish_is_refused_naming_it(plans):
    p = plans[("plain", "host")]
    g = _planes(_inputs(64, "uniform", 10), np)
    with pytest.raises(ValueError, match=p.plugin.name + ".*explicit path"):
        eagle.until_done(p, max_steps=10, dt=_DT, **g)


@pytest.mark.parametrize("reserved", ["finished_count", "active_map", "active_count"])
def test_reserved_planes_are_refused(plans, reserved):
    g = _planes(_inputs(64, "uniform", 10), np)
    g[reserved] = np.zeros(1, dtype=np.int32)
    with pytest.raises(ValueError, match=reserved):
        _runner(plans, "fplain", "host", max_steps=10, dt=_DT, **g)


@pytest.mark.parametrize("kw", [{"every": 8}, {"reorder": 0.5}])
def test_every_and_reorder_on_a_plain_artifact_are_refused(plans, kw):
    g = _planes(_inputs(64, "uniform", 10), np)
    with pytest.raises(ValueError, match="plain artifact"):
        _runner(plans, "fplain", "host", max_steps=10, dt=_DT, **g, **kw)


@pytest.mark.parametrize("bad, exc", [(0, ValueError), (True, TypeError),
                                      (10.0, TypeError)])
def test_max_steps_is_a_positive_int(plans, bad, exc):
    g = _planes(_inputs(64, "uniform", 10), np)
    with pytest.raises(exc, match="max_steps"):
        _runner(plans, "fplain", "host", max_steps=bad, dt=_DT, **g)


def test_a_finishing_kernel_without_a_counter_slot_is_refused(plans):
    p = _with_finish(plans[("plain", "host")])
    g = _planes(_inputs(64, "uniform", 10), np)
    with pytest.raises(ValueError, match="finished_count"):
        eagle.until_done(p, max_steps=10, dt=_DT, **g)


# --------------------------------------------------------------------------- #
# Oracle, non-vacuity, monotone set/reset, compaction + reorder
# --------------------------------------------------------------------------- #
_TARGETS = ["host", pytest.param("device", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("distribution", ["spread", "uniform"])
def test_run_until_done_matches_the_explicit_path(plans, target,
                                                  distribution):
    _need(plans, ("fplain", target))
    n, steps = 5000, 200
    inp = _inputs(n, distribution, steps)
    ref, ref_term = _explicit(plans, target, inp, steps)
    g = _planes(inp, _xp(target))
    runner = _runner(plans, "fplain", target, max_steps=steps,
                     dt=_DT, **g)
    rep = runner.run()
    for key in ("x", "v", "k"):
        assert np.array_equal(ref[key], _get(g[key])), key
    assert ref_term.all() and _get(runner.terminated).all()
    assert rep.finished == n and rep.done and rep.n == n
    assert np.array_equal(_get(g["k"]), inp["nstop"])
    assert rep.compactions == 0 and rep.reorders == 0
    # a second run continues from the finished state: nothing launches
    again = runner.run()
    assert again.launches == 0 and again.done
    assert np.array_equal(ref["x"], _get(g["x"]))


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("label, every", [("fplain", 1), ("fmap", 8)])
def test_launches_follow_the_longest_sample(plans, target, label, every):
    """The longest sample stops at S << max_steps; the loop stops there."""
    _need(plans, (label, target))
    n, s_max, max_steps = 3000, 40, 100_000
    inp = _inputs(n, "spread", s_max)
    s = int(inp["nstop"].max())
    g = _planes(inp, _xp(target))
    kw = {} if label == "fplain" else {"every": every}
    runner = _runner(plans, label, target, max_steps=max_steps,
                     dt=_DT, **g, **kw)
    rep = runner.run()
    assert rep.finished == n and rep.done
    assert rep.launches <= -(-s // every) + 1 and rep.launches * every <= s + every
    assert runner.loop.iterations() == rep.launches
    assert rep.steps == rep.launches * every
    assert runner.loop.max_iters == -(-max_steps // every)


@pytest.mark.parametrize("target", _TARGETS)
def test_a_kernel_without_the_count_runs_to_the_cap(plans, tmp_path, monkeypatch,
                                                    target):
    """Red case: the same kernel emitted with the epilogue's count dropped; the
    guard never fires and the loop runs to its cap."""
    _need(plans, ("fplain", target))
    from hawk.emit import aether

    original = aether._Renderer._emit

    def no_count(self, line):
        original(self, line.replace(" hawk_abi::finish_count(lut_finished_count);",
                                    ""))

    monkeypatch.setattr(aether._Renderer, "_emit", no_count)
    targets = ("host", "cuda") if target == "device" else ("host",)
    stripped = _build(_oscillator(None, True), tmp_path, targets,
                      cache_dir=str(tmp_path / "cache"))[target]
    monkeypatch.undo()
    n, cap = 500, 60
    inp = _inputs(n, "spread", 20)
    g = _planes(inp, _xp(target))
    rep = eagle.run_until_done(stripped, max_steps=cap, dt=_DT, **g)
    assert rep.launches == cap and not rep.done and rep.finished == 0
    # the same batch through the real kernel stops at its longest sample
    g = _planes(inp, _xp(target))
    rep = eagle.run_until_done(plans[("fplain", target)], max_steps=cap, dt=_DT, **g)
    assert rep.done and rep.launches == int(inp["nstop"].max())


@pytest.mark.parametrize("target", _TARGETS)
def test_preset_samples_are_untouched_then_reset_runs_them(plans,
                                                           target):
    """Monotone set-only mask; #36 skips samples terminated on entry."""
    _need(plans, ("fplain", target))
    xp = _xp(target)
    n, steps = 2000, 50
    inp = _inputs(n, "spread", steps)
    g = _planes(inp, xp)
    preset = np.zeros(n, dtype=bool)
    preset[::7] = True
    term = xp.asarray(preset.copy())
    runner = _runner(plans, "fplain", target, max_steps=steps, dt=_DT,
                     terminated=term, **g)
    rep = runner.run()
    assert rep.done and rep.finished == n and _get(term).all()
    for key in ("x", "v"):
        assert np.array_equal(_get(g[key])[preset], inp[key][preset]), key
    assert not _get(g["k"])[preset].any()
    assert np.array_equal(_get(g["k"])[~preset], inp["nstop"][~preset])
    runner.reset()
    assert not _get(term).any() and int(_get(runner.finished)[0]) == 0
    rep2 = runner.run()
    assert rep2.done
    assert np.array_equal(_get(g["k"])[preset], inp["nstop"][preset])


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("distribution", ["spread", "uniform"])
def test_the_active_set_artifact_compacts_through_the_same_call(
        plans, target, distribution):
    """Compaction is inferred from the artifact; bit-equal to the oracle."""
    _need(plans, ("fmap", target))
    n, steps = 5000, 200
    inp = _inputs(n, distribution, steps)
    ref, _ = _explicit(plans, target, inp, steps)
    g = _planes(inp, _xp(target))
    runner = _runner(plans, "fmap", target, max_steps=steps, every=4,
                     dt=_DT, **g)
    assert runner.active is not None and runner.active.theta is None
    rep = runner.run()
    for key in ("x", "v", "k"):
        assert np.array_equal(ref[key], _get(g[key])), key
    assert rep.done and rep.finished == n
    if distribution == "spread":
        assert rep.compactions >= 1 and runner.active.live < n
    assert rep.steps == rep.launches * 4


@pytest.mark.parametrize("target", _TARGETS)
def test_reorder_runs_and_restores_sample_order(plans, target):
    """reorder=0.5 at n >= MIN_REORDER_SPAN fires and the caller sees sample
    order at the end (perm and inv back to the identity)."""
    _need(plans, ("fmap", target))
    n, steps = max(MIN_REORDER_SPAN, 100_000), 200
    inp = _inputs(n, "spread", steps)
    ref, _ = _explicit(plans, target, inp, steps)
    g = _planes(inp, _xp(target))
    runner = _runner(plans, "fmap", target, max_steps=steps,
                     reorder=0.5, every=4, dt=_DT, **g)
    rep = runner.run()
    assert rep.reorders >= 1 and rep.compactions >= 1 and rep.done
    a = runner.active
    ident = np.arange(n, dtype=np.int32)
    assert np.array_equal(_get(a.perm)[_get(a.inv)], ident)
    assert np.array_equal(_get(a.in_sample_order(g["x"])), _get(g["x"]))
    for key in ("x", "v", "k"):
        assert np.array_equal(ref[key], _get(g[key])), key


@pytest.mark.gpu
def test_the_runner_loop_composes_into_a_pipeline(plans):
    """The explicit door stays open: runner.loop inside a caller's pipeline."""
    _need(plans, ("fplain", "device"))
    cp = pytest.importorskip("cupy")
    from eagle.pipeline import GraphPipeline

    n, steps = 1000, 30
    inp = _inputs(n, "spread", steps)
    g = _planes(inp, cp)
    runner = _runner(plans, "fplain", "device", max_steps=steps,
                     dt=_DT, **g)
    pre = []
    GraphPipeline().add(lambda: pre.append(1)).add(runner.loop).build().launch()
    assert int(runner.finished.get()[0]) == n
    assert np.array_equal(g["k"].get(), inp["nstop"])


def test_compaction_body_is_what_the_runner_runs(plans):
    """One implementation, two doors: the runner's loop body is the
    compaction_body tuple (steps, counted compaction)."""
    g = _planes(_inputs(256, "spread", 20), np)
    runner = _runner(plans, "fmap", "host", max_steps=20, every=4,
                     dt=_DT, **g)
    ref = compaction_body(lambda: None, runner.active, every=4)
    assert len(runner.loop.parts) == len(ref) == 2
    assert isinstance(runner.loop.parts[1], eagle.Skippable)


# --------------------------------------------------------------------------- #
# One packed counter readback per run (eagle._until_done._packed_read).
# --------------------------------------------------------------------------- #
def test_packed_read_host_reads_each_cell_directly():
    """The host (non-device) branch: no sync exists to fold, so each cell is
    just read, in order, off plain numpy arrays."""
    from eagle._until_done import _packed_read

    a = np.asarray([7], dtype=np.uint32)
    b = np.asarray([0, 3], dtype=np.uint32)
    assert _packed_read(np, False, [(a, 0), (b, 1)]) == (7, 3)


@pytest.mark.gpu
def test_packed_read_device_matches_direct_reads_in_one_transfer():
    """The device branch: the SAME values as reading each cell directly
    (``.get()``), combined into ONE staging buffer and ONE ``.get()`` --
    verified by counting ``RawArray.get`` calls through a spy, so this is a
    structural check on sync COUNT, not just the returned values."""
    cp = pytest.importorskip("cupy")
    from eagle._until_done import _packed_read

    finished = cp.asarray([5], dtype=cp.uint32)
    compactions = cp.asarray([2], dtype=cp.uint32)
    got = _packed_read(cp, True, [(finished, 0), (compactions, 0)])
    assert got == (5, 2)

    gets = []
    real_get = cp.ndarray.get

    def spy_get(self, *a, **kw):
        gets.append(1)
        return real_get(self, *a, **kw)

    cp.ndarray.get = spy_get
    try:
        _packed_read(cp, True, [(finished, 0), (compactions, 0)])
    finally:
        cp.ndarray.get = real_get
    assert gets == [1], f"expected exactly ONE .get() for two packed cells, got {gets}"


@pytest.mark.gpu
def test_non_compacting_run_reports_zero_compactions_with_no_pre_read(plans):
    """The packed readback's other half: a plain (non active-set) run's
    ``_compactions`` cell can only ever be 0 (nothing ever bumps it without
    an ActiveSet), so :meth:`Runner.run` skips its pre-launch packed read
    entirely in this case (``pre_cells`` stays empty) rather than reading a
    cell that cannot change -- checked here via the public report, and via
    :func:`eagle._until_done._packed_read` staying unreached (spied)."""
    cp = pytest.importorskip("cupy")
    from eagle import _until_done as ud

    calls = []
    real = ud._packed_read

    def spy(xp, device, cells):
        calls.append(len(cells))
        return real(xp, device, cells)

    g = _planes(_inputs(256, "spread", 20), cp)
    runner = _runner(plans, "fplain", "device", max_steps=20, dt=_DT, **g)
    assert runner.active is None
    ud._packed_read = spy
    try:
        report = runner.run()
    finally:
        ud._packed_read = real
    assert report.compactions == 0
    # exactly one packed call (the post-launch snapshot: iterations + finished),
    # none for a pre-launch snapshot that would only ever read a stuck 0.
    assert calls == [2], f"expected one packed call of 2 cells, got {calls}"
