# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.until_done`` over an automatic kernel (``hawk.steps(kernel,
"auto")``: the steps per launch are a device word the runner's policy sets).

What is pinned (each row names what it catches):

* the sidecar and the binding: ``read_finish`` accepts ``steps: "auto"`` only
  with an int ``steps_max`` and refuses ``steps_max`` beside an int; the auto
  artifact binds both reserved lookups; ``fused_steps=`` from the caller is
  refused; ``every=`` under auto is refused;
* bit identity: plain-kind and active-set auto runs leave ``x``, ``v``, ``k``
  bit-equal to the one-step explicit oracle on the host, and on the device to
  the same auto artifact driven by hand with the k sequence an independent
  re-statement of the policy picks (nvcc may contract the fused loop's
  arithmetic differently from the single step's);
* the exact cap: samples that never finish stop at exactly ``max_steps``
  (``report.steps`` exact) where the fixed ``steps=16`` artifact overshoots;
* the band: a uniform batch climbs to 64 and never visits 8, a batch whose half
  stops at step 1 halves to 8 once, ``sum(k * launches) == steps``, an
  all-finished batch launches nothing; a constant-K policy turns it red;
* host == device: one batch yields the same ``launches_by_k``,
  ``compactions`` and ``steps`` on both faces when the host runs the band
  (``host_policy = "band"``, which every band row above uses on the host);
* the host default resolves per entry: ``"tiled"`` for an artifact whose
  fused side is the tiled loop (it exports its tile), ``"measured"`` for one
  that is not (the active-set kind keeps the per-sample loop); the tiled
  policy asks ``steps_max`` per launch from the first launch, cuts the last
  to the cap (exact), and is bit-equal to the one-step oracle;
* the measured host policy: a uniform batch sweeps one step per
  launch and never probes; on a thinning batch the scripted walls decide --
  fused launches cheaper per step than a sweep and the run turns fused after
  its first probe, dearer and it only sweeps and probes (each live-count
  halving at most once); the planes are bit-equal to the one-step oracle
  either way, with an active set too;
* ``steps`` is the steps the loop ran: within one launch of the longest
  sample's own count, and equal to it whenever ``exact_steps`` says so (a
  band run of a uniform batch overshoots inside its last launch and is not
  exact; the measured host run of the same batch is).
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
from eagle import _until_done as ud
from eagle._active_set import MIN_REORDER_SPAN
from eagle.sidecar import read_finish

_TARGETS = ["host", pytest.param("device", marks=pytest.mark.gpu)]
_AUTO = {"mask": "terminated", "counter": "finished_count", "steps": "auto",
         "steps_max": 64}
_S = 300          # longest sample of the spread / uniform batches
_NEVER = 1e9      # a stop step no run reaches


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
def auto(tmp_path_factory):
    """``auto[(label, target)]``: ``plain`` / ``map`` (the auto kernel, plain
    kind / active-set kind), ``fixed16`` (``steps=16``, plain kind),
    ``oracle`` (the one-step kernel without finish), ``default`` (the plain
    decorator, no ``steps=``: the auto kernel under the author's name) and
    ``one`` (``steps=1``: the plain finishing kernel)."""
    hawk = pytest.importorskip("hawk")
    import hawk.trace.steps as hsteps

    if not hasattr(hsteps, "AUTO_K_MAX"):
        pytest.skip("this hawk has no steps='auto'")
    from hawk.ext import Guard, Kind

    targets = ("host", "cuda") if _have_gpu() else ("host",)
    out = {}
    kernels = {
        "oracle": _oscillator(None, False),
        "plain": hawk.steps(_oscillator(None, True), "auto"),
        "map": hawk.steps(_oscillator(Kind("uda_map", guard=Guard(active_set=True)),
                                      True), "auto"),
        "fixed16": hawk.steps(_oscillator(None, True), 16),
        # the plain decorator (no steps=): the automatic kernel, own name
        "default": _oscillator(None, True, steps=None),
        "one": _oscillator(None, True),
    }
    # a private cache: a kernel another module built in this process (the
    # oracle is test_until_done's plain kernel) is then built here afresh
    cache = tmp_path_factory.mktemp("cache")
    # the automatic kernels as eagle.deploy builds them: with hawk's fast
    # entries (the range entry, which reports exact steps)
    from hawk.emit import FAST_ENTRIES_MACRO

    fast = (f"{FAST_ENTRIES_MACRO}=1",)
    for label, kern in kernels.items():
        built = _build(kern, tmp_path_factory.mktemp(label), targets, cache_dir=cache,
                       defines=fast if label == "plain" else ())
        out.update({(label, t): p for t, p in built.items()})
    return out


def _need(auto, key):
    if key not in auto:
        pytest.skip("no CUDA device")


def _batch(n, which, seed=11):
    """Inputs: ``spread`` / ``uniform`` (``_S`` steps), ``grouped`` (stops at
    5, 60 or ``_S``), ``half`` (half stop at step 1, the rest at ``_S``),
    ``never`` (no sample finishes)."""
    if which in ("spread", "uniform"):
        return _inputs(n, which, _S, seed)
    inp = _inputs(n, "uniform", _S, seed)
    rng = np.random.default_rng([seed, n])
    if which == "grouped":
        inp["nstop"] = rng.choice([5.0, 60.0, float(_S)], n)
    elif which == "half":
        inp["nstop"] = np.where(np.arange(n) % 2 == 0, 1.0, float(_S))
    elif which == "never":
        inp["nstop"] = np.full(n, _NEVER)
    return inp


def _run(auto, label, target, inp, max_steps, policy="band", **kw):
    """One ``until_done`` run; a host run follows ``policy`` (the device's
    band by default, so a row's ``launches_by_k`` is the device's)."""
    xp = _xp(target)
    g = _planes(inp, xp)
    runner = eagle.until_done(auto[(label, target)], max_steps=max_steps, dt=_DT,
                              **g, **kw)
    runner.host_policy = policy
    report = runner.run()
    return runner, report, {key: _get(g[key]) for key in ("x", "v", "k")}


def _policy_sequence(finished_after, n, max_steps):
    """The k sequence, re-stated from the documented rule (independent of eagle's
    code): ``finished_after(k)`` launches ``k`` steps and returns the counter."""
    seq, k, fin_at, done = [], ud.AUTO_K0, finished_after(None), 0
    word = min(k, max_steps)
    while fin_at != n and done < max_steps:
        seq.append(word)
        fin = finished_after(word)
        live, newly = n - fin, fin - fin_at
        done += word
        fin_at = fin
        if newly * 16 < live:
            k = min(2 * k, 64)
        elif newly * 4 > live:
            k = max(k // 2, ud.AUTO_K_MIN)
        word = min(k, max_steps - done)
    return seq


def _pinned_sequence(finished_after, n, max_steps, k_max):
    """The k sequence under the no-active-set-rebuild rule's small-N branch, re-stated
    from the documented rule (independent of eagle's code): a batch that
    fits inside one resident wave is never compacted, and K never leaves
    ``k_max`` (the band's floor and ceiling pinned to the same value make
    both its arms a no-op) -- every launch takes ``k_max`` steps except the
    last, cut to what is left."""
    seq, fin_at, done = [], finished_after(None), 0
    word = min(k_max, max_steps)
    while fin_at != n and done < max_steps:
        seq.append(word)
        fin = finished_after(word)
        done += word
        fin_at = fin
        word = min(k_max, max_steps - done)
    return seq


def _capacity_probe(auto, label, target):
    """A fresh one-sample :class:`~eagle.plan.BoundPlan` of ``auto[(label,
    target)]``, for the device-only probes below -- ``None`` for a host
    target. Capacity/the fast-mode decision do not depend on the batch
    actually bound, only on the kernel's registers and the device's
    properties. Binds the reserved active-set lookups too when the plugin
    declares them (the "map" kind), using :class:`eagle.ActiveSet`'s own
    identity planes so this has no hand-rolled twin of its plane shapes."""
    if target != "device":
        return None
    xp = _xp(target)
    g = _planes(_batch(1, "uniform"), xp)
    term = xp.zeros(1, dtype=xp.bool_)
    counter = xp.zeros(1, dtype=xp.uint32)
    word = xp.zeros(1, dtype=xp.int64)
    kw = dict(dt=_DT, terminated=term, fused_steps=word,
             finished_count=counter.view(xp.int32), **g)
    plugin = auto[(label, target)].plugin
    declared = {name for _role, name in plugin.arg_spec}
    if "active_map" in declared and "active_count" in declared:
        from eagle import ActiveSet

        kw.update(ActiveSet(term).planes())
    return auto[(label, target)].bind(**kw)


def _device_capacity(auto, label, target):
    """``BoundPlan.device_capacity()`` for ``auto[(label, target)]`` --
    ``None`` for a host target (the no-active-set-rebuild rule is device-only)."""
    bound = _capacity_probe(auto, label, target)
    return None if bound is None else bound.device_capacity()


def _above(auto):
    """A batch comfortably above the plain kernel's resident-wave capacity on
    this GPU (5000 on a small card): what the entry-choice rows run, since only
    above capacity are both fast entries in play."""
    return max(5000, 2 * (_device_capacity(auto, "plain", "device") or 0))


def _runner_fast_mode(auto, target, label, n):
    """``eagle._until_done._fast_mode_for`` this ``(label, target, n)``
    would pick -- the production decision itself (not a hand-rolled twin),
    bound through :func:`_capacity_probe`. ``None`` on host (fast mode is
    device-only) or for the "map" kind (it always carries an active set,
    so :meth:`eagle._until_done.Runner._build_auto`'s own
    ``self.active is None`` gate means fast mode never applies to it)."""
    if label == "map":
        return None
    bound = _capacity_probe(auto, label, target)
    return None if bound is None else ud._fast_mode_for(bound, n)


def _map_skip(auto, target, label, n):
    """The ``k_max`` the no-active-set-rebuild rule pins K at for this ``(label, target,
    n)``, or ``None`` when the normal ramped band applies instead: only the
    "map" (active-set) kind, only on the device, only when ``n`` fits inside
    the kernel's resident-wave capacity (0/``None`` capacity -- unavailable
    device/kernel properties -- never counts as fitting)."""
    if label != "map" or target != "device":
        return None
    cap = _device_capacity(auto, label, target)
    return _AUTO["steps_max"] if cap and n <= cap else None


def _by_hand(auto, target, inp, max_steps, *, pinned_k_max=None):
    """The plain-kind auto artifact driven launch by launch with the policy's
    k sequence: the planes and the sequence. ``pinned_k_max`` (device rule
    (iii)) switches the SEQUENCE oracle to :func:`_pinned_sequence` -- the
    finished-counter dynamics it drives are identical for the plain and the
    map kind (compaction never changes WHICH sample finishes when, only how
    fast the device gets there), so driving the plain kernel still gives the
    right sequence for a map-kind, capacity-skipping run."""
    xp = _xp(target)
    g = _planes(inp, xp)
    n = inp["x"].shape[0]
    term = xp.zeros(n, dtype=xp.bool_)
    counter = xp.zeros(1, dtype=xp.uint32)
    word = xp.zeros(1, dtype=xp.int64)
    bound = auto[("plain", target)].bind(dt=_DT, terminated=term, fused_steps=word,
                                         finished_count=counter.view(xp.int32), **g)

    def finished_after(k):
        if k is not None:
            word[0] = k
            bound.launch()
        return int(_get(counter)[0])

    if pinned_k_max is not None:
        seq = _pinned_sequence(finished_after, n, max_steps, pinned_k_max)
    else:
        seq = _policy_sequence(finished_after, n, max_steps)
    return {key: _get(g[key]) for key in ("x", "v", "k")}, seq


def _hist(seq):
    out = {}
    for k in seq:
        out[k] = out.get(k, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# The sidecar, the binding and the options
# --------------------------------------------------------------------------- #
def test_read_finish_accepts_auto_with_steps_max():
    assert read_finish({"finish": dict(_AUTO)}, name="k") == _AUTO


@pytest.mark.parametrize("bad, match", [
    ({"steps": 16, "steps_max": 64}, "steps_max"),
    ({"steps": "auto", "steps_max": None}, "steps_max"),
    ({"steps": "Auto"}, "auto"),
    ({"steps": "fast"}, "auto"),
    ({"steps": "auto", "steps_max": 0}, "steps_max"),
    ({"steps": "auto", "steps_max": True}, "steps_max"),
])
def test_read_finish_refuses_a_malformed_auto(bad, match):
    value = {k: v for k, v in dict(_AUTO, **bad).items() if v is not None}
    if bad.get("steps") in ("Auto", "fast"):
        value.pop("steps_max", None)
    with pytest.raises(ValueError, match=f"my_kernel.*{match}"):
        read_finish({"finish": value}, name="my_kernel")


def test_the_auto_artifact_binds_both_reserved_lookups(auto):
    for label in ("plain", "map"):
        p = auto[(label, "host")].plugin
        spec = {tuple(pair) for pair in p.arg_spec}
        assert {("lookup", "finished_count"), ("lookup", "fused_steps")} <= spec
        assert dict(p.finish) == _AUTO
    assert eagle.FUSED_STEPS_PLANE == "fused_steps"


def test_fused_steps_from_the_caller_is_refused(auto):
    g = _planes(_batch(64, "uniform"), np)
    with pytest.raises(ValueError, match="fused_steps"):
        eagle.until_done(auto[("plain", "host")], max_steps=10, dt=_DT,
                         fused_steps=np.zeros(1, dtype=np.int64), **g)


@pytest.mark.parametrize("label", ["plain", "map"])
def test_every_under_auto_is_refused(auto, label):
    g = _planes(_batch(64, "uniform"), np)
    with pytest.raises(ValueError, match="auto.*every"):
        eagle.until_done(auto[(label, "host")], max_steps=10, every=64, dt=_DT, **g)


def test_the_default_auto_run_does_not_reorder(auto):
    runner, report, _ = _run(auto, "map", "host", _batch(256, "spread"), _S)
    assert runner.active.theta is None and report.reorders == 0
    assert runner.steps_per_launch == "auto" and runner.every is None


# --------------------------------------------------------------------------- #
# Bit identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target, n", [
    ("host", 1000),
    pytest.param("device", 1000, marks=pytest.mark.gpu),
    pytest.param("device", 100_000, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("which", ["spread", "uniform", "grouped"])
@pytest.mark.parametrize("label", ["plain", "map"])
def test_auto_runs_are_bit_identical(auto, label, which, target, n):
    _need(auto, (label, target))
    inp = _batch(n, which)
    _, report, got = _run(auto, label, target, inp, 10 * _S)
    assert report.done and report.finished == n
    _steps_bound(report, got)
    if target == "host":
        want, _ = _explicit({("plain", "host"): auto[("oracle", "host")]}, "host",
                            inp, 10 * _S)
        seq = None
    else:
        fast = _runner_fast_mode(auto, target, label, n)
        want, seq = _by_hand(auto, target, inp, 10 * _S,
                             pinned_k_max=_map_skip(auto, target, label, n))
        if fast is not None:
            # One launch of the whole budget -- no band, no policy kernel,
            # so _by_hand's ramped/pinned k-SEQUENCE oracle does not apply
            # (the VALUES it drives out still do -- same finished-counter
            # dynamics regardless of launch shape).
            assert report.launches_by_k == {10 * _S: 1}
        else:
            assert report.launches_by_k == _hist(seq)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


# --------------------------------------------------------------------------- #
# The internal fast path (``_fast_mode``, forced past whatever
# ``_fast_mode_for`` would pick): one launch, no graph, no policy kernel --
# eagle.until_done's forced persist/fused_one arm (perf_card.py, rk78_card.py).
# Bit identity against the one-step oracle, the already-finished and
# continuing-run shortcuts, the lane-utilisation opt-in switch, and the
# per-run sync budget.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fused_one", "persist"])
@pytest.mark.parametrize("n", [1, 33, 1000, 2048, 2049, 10_000])
@pytest.mark.parametrize("which", ["spread", "uniform"])
def test_fast_path_is_bit_identical_to_the_oracle(auto, mode, n, which):
    _need(auto, ("plain", "device"))
    inp = _batch(n, which)
    # the DEVICE oracle (_by_hand drives the same auto kernel launch by
    # launch): nvcc's fused-loop contraction need not match the host's.
    want, _ = _by_hand(auto, "device", inp, 10 * _S)
    _, report, got = _run(auto, "plain", "device", inp, 10 * _S, _fast_mode=mode)
    assert report.done and report.finished == n and report.launches == 1
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fused_one", "persist"])
def test_fast_path_budget_hit_is_bit_identical_to_the_oracle(auto, mode):
    """``max_steps`` below the longest sample's own stop: nobody finishes,
    ``steps`` is exact (the budget WAS the bottleneck for every sample)."""
    _need(auto, ("plain", "device"))
    n, cap = 2049, _S // 2
    inp = _batch(n, "uniform")
    want, _ = _by_hand(auto, "device", inp, cap)
    _, report, got = _run(auto, "plain", "device", inp, cap, _fast_mode=mode)
    assert not report.done and report.finished == 0
    assert report.launches == 1 and report.steps == cap and report.exact_steps
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


@pytest.mark.gpu
def test_a_fast_path_runner_never_allocates_compaction_scratch(auto, tmp_path):
    """An active-set runner that takes a no-compaction fast path holds no
    device compaction scratch (GPU memory parity with a plain kernel); the
    same runner forced onto the compaction loop allocates it."""
    import hawk
    from hawk.ext import Guard, Kind

    _need(auto, ("map", "device"))
    # as eagle.deploy builds it: with the fast entries (the fixture's map is not)
    plan = eagle.deploy(hawk.steps(_oscillator(Kind("uda_map", guard=Guard(active_set=True)),
                                               True), "auto"), cache_dir=tmp_path)
    one = {("deployed", "device"): plan}
    inp = _batch(4096, "spread")
    fast, report, _ = _run(one, "deployed", "device", inp, 10 * _S)
    assert report.mode in ("fused_one", "persist") and report.done
    assert fast.active._scratch is None
    band, report, _ = _run(one, "deployed", "device", inp, 10 * _S, _fast_mode=False)
    assert report.mode == "band" and report.done
    assert band.active._scratch is not None



def _deployed_map(cache_dir, **kw):
    import hawk
    from hawk.ext import Guard, Kind

    return eagle.deploy(hawk.steps(_oscillator(Kind("uda_map", guard=Guard(active_set=True)),
                                               True), "auto"), cache_dir=cache_dir, **kw)


def test_a_default_deploy_builds_the_host_side_in_the_background(auto, tmp_path):
    """With a GPU, a default deploy returns with the device side built and
    the host side building on its own thread; the first host run waits for
    it and runs the same bits as an explicit host build."""
    _need(auto, ("map", "device"))
    plan = _deployed_map(tmp_path)
    assert getattr(plan.plugin, "host_entry", None) is None and plan.plugin.device_function
    assert "host" in repr(plan)
    inp = _batch(256, "spread")
    one = {("deployed", "host"): plan,
           ("explicit", "host"): _deployed_map(tmp_path, targets=("host",))}
    _, report, got = _run(one, "deployed", "host", inp, 10 * _S)
    assert report.done and plan.host is plan.host and plan.host.plugin.host_entry
    _, _, want = _run(one, "explicit", "host", inp, 10 * _S)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


def test_a_failed_background_host_build_surfaces_at_the_first_host_use(auto, tmp_path,
                                                                       monkeypatch):
    """The deploy itself succeeds (a GPU-only caller never meets the host
    build's error); the first host plan raises it."""
    from eagle import _plan_deploy

    _need(auto, ("map", "device"))
    real = _plan_deploy._hawk_plugins

    def host_fails(kernels, targets, *a, **kw):
        if tuple(targets) == ("host",):
            raise RuntimeError("no host compiler here")
        return real(kernels, targets, *a, **kw)

    monkeypatch.setattr(_plan_deploy, "_hawk_plugins", host_fails)
    plan = _deployed_map(tmp_path)
    assert plan.device is not None
    with pytest.raises(RuntimeError, match="no host compiler here"):
        plan.host


def test_builds_of_one_bundle_folder_never_overlap(auto, tmp_path, monkeypatch):
    """hawk writes a bundle folder in place, so a background host build and
    repeat deploys of the same kernel take turns on it."""
    import threading
    import time

    import hawk.artifact

    _need(auto, ("map", "device"))
    real = hawk.artifact.build_bundle
    lock = threading.Lock()
    inside, most = {}, {}

    def watched(kernels, where, **kw):
        with lock:
            inside[where] = inside.get(where, 0) + 1
            most[where] = max(most.get(where, 0), inside[where])
        try:
            if tuple(kw.get("targets", ())) == ("host",):
                time.sleep(2.0)  # outlast the next deploy, so unserialized builds overlap
            return real(kernels, where, **kw)
        finally:
            with lock:
                inside[where] -= 1

    monkeypatch.setattr(hawk.artifact, "build_bundle", watched)
    plans = [_deployed_map(tmp_path), _deployed_map(tmp_path),
             _deployed_map(tmp_path, targets=("host",))]
    for p in plans[:2]:
        assert p.host.plugin.host_entry
    assert most and max(most.values()) == 1, most

def test_a_child_forked_during_the_host_build_builds_its_own(auto, tmp_path, monkeypatch):
    """A forked child inherits no running thread (and no released lock): it
    builds the host side itself, and a repeat host deploy there completes."""
    import os
    import signal
    import time

    from eagle import _plan_deploy

    _need(auto, ("map", "device"))
    real = _plan_deploy._hawk_plugins

    def slow_host(kernels, targets, *a, **kw):
        if tuple(targets) == ("host",) and os.getpid() == parent:
            time.sleep(3.0)  # the parent's build is still running at the fork
        return real(kernels, targets, *a, **kw)

    parent = os.getpid()
    monkeypatch.setattr(_plan_deploy, "_hawk_plugins", slow_host)
    plan = _deployed_map(tmp_path)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child reports through its exit code
        code = 1
        try:
            signal.alarm(120)
            ok = bool(plan.host.plugin.host_entry)
            ok = ok and bool(_deployed_map(tmp_path, targets=("host",)).host.plugin.host_entry)
            code = 0 if ok else 2
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert plan.host.plugin.host_entry

_FIRST_RUN_COMPILES = """
import sys
from cupy.cuda import compiler
sys.path.insert(0, sys.argv[2])
from test_until_done import _DT, _inputs, _planes
from test_until_done_auto import _deployed_map
import cupy, eagle
seen = [0]
real = compiler._compile_module_with_cache
def counted(*a, **kw):
    seen[0] += 1
    return real(*a, **kw)
compiler._compile_module_with_cache = counted
plan = _deployed_map(sys.argv[1])
at_deploy = seen[0]
planes = _planes(_inputs(64, "spread", 300, 11), cupy)  # the inputs' own copies
seen[0] = 0
runner = eagle.until_done(plan, max_steps=3000, dt=_DT, **planes)
done = runner.run().done
print("COMPILES", at_deploy, seen[0], done)
"""


@pytest.mark.gpu
def test_the_first_device_run_after_a_deploy_compiles_nothing(auto, tmp_path):
    """The loop's own small cupy kernels compile during the deploy (beside
    its device build), so the first run compiles none; a fresh process and
    an empty cupy cache, or an earlier test's compiles would hide it."""
    import os
    import pathlib
    import subprocess
    import sys

    _need(auto, ("map", "device"))
    env = dict(os.environ, CUPY_CACHE_DIR=str(tmp_path / "cupy"))
    out = subprocess.run([sys.executable, "-c", _FIRST_RUN_COMPILES, str(tmp_path / "hawk"),
                          str(pathlib.Path(__file__).parent)],
                         env=env, capture_output=True, text=True, check=True).stdout
    at_deploy, at_run, done = out.split("COMPILES", 1)[1].split()
    assert int(at_deploy) >= 5 and int(at_run) == 0 and done == "True", out

@pytest.mark.parametrize("bad", ["persistent", "fused", True, 1])
def test_fast_mode_rejects_an_unknown_value(auto, bad):
    """A mistyped ``_fast_mode`` fails at construction with the accepted
    values named, not later inside a launch with no grid."""
    _need(auto, ("plain", "host"))
    with pytest.raises(ValueError, match="_fast_mode must be"):
        _run(auto, "plain", "host", _batch(8, "uniform"), _S, _fast_mode=bad)


@pytest.mark.gpu
def test_fast_path_lane_utilisation_is_off_unless_asked(auto):
    """The persist entry's lane-utilisation counter (2 same-address
    64-bit atomics per warp per step) stays off -- ``RunReport.lane_utilisation
    is None`` -- unless the internal ``_lane_utilisation`` switch asks for it
    (the same convention ``_fast_mode`` already uses)."""
    _need(auto, ("plain", "device"))
    inp = _batch(2000, "uniform")
    _, off, _ = _run(auto, "plain", "device", inp, 10 * _S, _fast_mode="persist")
    assert off.lane_utilisation is None
    _, on, _ = _run(auto, "plain", "device", inp, 10 * _S, _fast_mode="persist",
                    _lane_utilisation=True)
    assert on.lane_utilisation is not None and 0 < on.lane_utilisation <= 1
    # fused_one has no persist entry to count lanes on: the switch is a no-op
    _, fused, _ = _run(auto, "plain", "device", inp, 10 * _S, _fast_mode="fused_one",
                       _lane_utilisation=True)
    assert fused.lane_utilisation is None


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fused_one", "persist"])
def test_fast_path_already_finished_batch_launches_nothing(auto, mode):
    """A batch the caller hands over already fully terminated: the run still
    launches (every call re-syncs ``finished`` to the mask fresh, no cache
    of "already done" across calls -- see ``Runner._run_fast``), but it is a
    no-op (``steps=0``/``exact_steps=True``: the kernel skips every
    terminated lane) and reports ``done``/``finished == n``."""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")
    n = 256
    inp = _batch(n, "uniform")
    g = _planes(inp, cp)
    term = cp.ones(n, dtype=cp.bool_)
    report = eagle.run_until_done(auto[("plain", "device")], max_steps=_S, dt=_DT,
                                  terminated=term, _fast_mode=mode, **g)
    assert report.launches == 1 and report.steps == 0 and report.exact_steps
    assert report.done and report.finished == n


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fused_one", "persist"])
def test_fast_path_second_run_continues_a_partly_finished_batch(auto, mode):
    """Two calls, each under the longest sample's own stop: the first leaves
    it mid-flight, the second finishes it -- bit-identical to running the
    whole budget in one call; a third call (now already finished) still
    launches (no "already done" cache across calls any more -- every call
    re-syncs ``finished`` to the mask fresh) but is a no-op."""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")
    n, half = 500, _S // 2
    inp = _batch(n, "uniform")  # every sample stops at exactly _S
    g = _planes(inp, cp)
    runner = eagle.until_done(auto[("plain", "device")], max_steps=half, dt=_DT,
                              _fast_mode=mode, **g)
    report1 = runner.run()
    assert not report1.done and report1.finished == 0 and report1.launches == 1
    report2 = runner.run()
    assert report2.done and report2.finished == n and report2.launches == 1
    got = {key: _get(g[key]) for key in ("x", "v", "k")}
    want, _ = _by_hand(auto, "device", inp, 2 * half)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
    report3 = runner.run()
    assert report3.launches == 1 and report3.steps == 0 and report3.exact_steps
    assert report3.done and report3.finished == n


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fused_one", "persist"])
def test_fast_path_run_does_one_sync(auto, mode):
    """A continuing (not-first, not-done) fast-path run makes exactly ONE
    ``eagle._until_done._packed_read`` call -- the post-run packed readback,
    the same sync-count spy
    ``test_non_compacting_run_reports_zero_compactions_with_no_pre_read``
    (test_until_done.py) uses for the band path."""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")
    n, step_budget = 500, _S // 3
    inp = _batch(n, "uniform")
    g = _planes(inp, cp)
    runner = eagle.until_done(auto[("plain", "device")], max_steps=step_budget,
                              dt=_DT, _fast_mode=mode, **g)
    runner.run()  # first call: seeds (its own, unspied, 2-cell packed read)
    calls = []
    real = ud._packed_read

    def spy(xp, device, cells):
        calls.append(len(cells))
        return real(xp, device, cells)

    ud._packed_read = spy
    try:
        report = runner.run()
    finally:
        ud._packed_read = real
    assert not report.done and report.finished == 0
    # one call: the exact step count and the finished count, adjacent words
    assert calls == [2], f"expected exactly one packed call of 2 cells, got {calls}"


# --------------------------------------------------------------------------- #
# The fast path is routed to ACTIVE-SET automatic kernels too (no reorder
# requested) -- eagle.deploy builds them with hawk's fast entries
# (HAWK_FAST_ENTRIES=1) by default, so the ordinary deploy/until_done path
# picks fused_one/persist exactly like a plain automatic kernel, launching
# <name>_range/<name>_persist with NO active-set machinery (map, rebuilds,
# policy kernel) ever built. every=/reorder= still take the band path.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def active_auto():
    """A fresh active-set (map-kind) automatic oscillator kernel, deployed
    the ordinary way (``eagle.deploy``): the fast entries ride along by
    default, no caller opt-in. ``None`` (skip) with no CUDA device."""
    if not _have_gpu():
        return None
    hawk = pytest.importorskip("hawk")
    from hawk.ext import Guard, Kind

    kind = Kind("uda_map_f3_test", guard=Guard(active_set=True))
    kern = hawk.steps(_oscillator(kind, True), "auto")
    return eagle.deploy(kern)


def _need_active(active_auto):
    if active_auto is None:
        pytest.skip("no CUDA device")


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 33, 1000, 2048, 2049, 10_000])
@pytest.mark.parametrize("which", ["spread", "uniform"])
def test_active_set_auto_default_deploy_picks_the_fast_path(active_auto, n, which):
    """eagle.deploy's default build carries the fast entries: the ordinary
    (no override) eligibility decision picks fused_one/persist for an
    active-set auto kernel too, bit-identical to the band path on n/finished/
    done (steps/exact_steps/launches are the chosen MODE's own bookkeeping,
    not shared across modes -- see test_active_set_fast_and_band_agree_on_the_budget_hit
    for the case where they DO coincide)."""
    _need_active(active_auto)
    cp = pytest.importorskip("cupy")
    inp = _batch(n, which)
    g1 = _planes(inp, cp)
    fast = eagle.until_done(active_auto, max_steps=10 * _S, dt=_DT, **g1)
    assert fast.active is not None and fast._fast_mode in ("fused_one", "persist")
    rep1 = fast.run()
    g2 = _planes(inp, cp)
    band = eagle.until_done(active_auto, max_steps=10 * _S, dt=_DT, _fast_mode=False, **g2)
    assert band._fast_mode is None and band.active is not None
    rep2 = band.run()
    assert (rep1.n, rep1.finished, rep1.done) == (rep2.n, rep2.finished, rep2.done) == (n, n, True)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(_get(g1[key]), _get(g2[key]), err_msg=key)


@pytest.mark.gpu
def test_active_set_fast_and_band_agree_on_the_budget_hit():
    """``max_steps`` below every sample's own stop: nobody finishes, the
    budget IS the bottleneck for both paths -- here ``steps``/``exact_steps``
    (not just n/finished/done) are ALSO bit-identical, since both modes
    report the same thing (the whole offered budget, exact)."""
    hawk = pytest.importorskip("hawk")
    cp = pytest.importorskip("cupy")
    from hawk.ext import Guard, Kind

    kind = Kind("uda_map_f3_budget", guard=Guard(active_set=True))
    kern = hawk.steps(_oscillator(kind, True), "auto")
    plan = eagle.deploy(kern)
    n, cap = 2049, _S // 2
    inp = _batch(n, "uniform")
    g1 = _planes(inp, cp)
    fast = eagle.until_done(plan, max_steps=cap, dt=_DT, **g1)
    rep1 = fast.run()
    g2 = _planes(inp, cp)
    band = eagle.until_done(plan, max_steps=cap, dt=_DT, _fast_mode=False, **g2)
    rep2 = band.run()
    assert not rep1.done and not rep2.done
    assert (rep1.steps, rep1.exact_steps) == (rep2.steps, rep2.exact_steps) == (cap, True)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(_get(g1[key]), _get(g2[key]), err_msg=key)


@pytest.mark.gpu
def test_active_set_fast_path_already_finished_and_continuing():
    """Already-finished (a run handed an all-terminated mask still
    launches -- no cross-call cache any more -- but is a no-op) and
    continuing a partial batch (two calls) -- the SAME conventions
    :func:`test_fast_path_already_finished_batch_launches_nothing`/
    :func:`test_fast_path_second_run_continues_a_partly_finished_batch` pin
    for a plain automatic kernel, now exercised through an active-set one."""
    hawk = pytest.importorskip("hawk")
    cp = pytest.importorskip("cupy")
    from hawk.ext import Guard, Kind

    kind = Kind("uda_map_f3_cont", guard=Guard(active_set=True))
    kern = hawk.steps(_oscillator(kind, True), "auto")
    plan = eagle.deploy(kern)
    n = 256
    term = cp.ones(n, dtype=cp.bool_)
    g = _planes(_batch(n, "uniform"), cp)
    already = eagle.run_until_done(plan, max_steps=_S, dt=_DT, terminated=term, **g)
    assert already.launches == 1 and already.steps == 0 and already.exact_steps
    assert already.done and already.finished == n

    n2, half = 500, _S // 2
    inp = _batch(n2, "uniform")
    g2 = _planes(inp, cp)
    runner = eagle.until_done(plan, max_steps=half, dt=_DT, **g2)
    rep1 = runner.run()
    assert not rep1.done and rep1.finished == 0
    rep2 = runner.run()
    assert rep2.done and rep2.finished == n2
    rep3 = runner.run()
    assert rep3.launches == 1 and rep3.steps == 0 and rep3.exact_steps
    assert rep3.done and rep3.finished == n2


@pytest.mark.gpu
def test_active_set_every_and_reorder_keep_the_band_path(active_auto):
    """Explicit compaction/reorder still refuses the fast path entirely:
    ``every=`` is refused for ANY auto kernel (unrelated to this package),
    and ``reorder=theta`` keeps today's band path (no fast mode, an
    active set with ``theta is not None``)."""
    _need_active(active_auto)
    cp = pytest.importorskip("cupy")
    g = _planes(_batch(5000, "spread"), cp)
    with pytest.raises(ValueError, match="auto.*every"):
        eagle.until_done(active_auto, max_steps=10 * _S, every=64, dt=_DT, **g)
    g2 = _planes(_batch(5000, "spread"), cp)
    runner = eagle.until_done(active_auto, max_steps=10 * _S, dt=_DT,
                              reorder=0.5, **g2)
    assert runner._fast_mode is None and runner.active.theta is not None
    rep = runner.run()
    assert rep.done and rep.finished == 5000


@pytest.mark.parametrize("target", _TARGETS)
def test_auto_reorder_runs_and_restores_sample_order(auto, target):
    _need(auto, ("map", target))
    # Comfortably ABOVE the no-active-set-rebuild rule's capacity: this test exercises
    # reordering, which only runs as part of compaction, so it needs a batch
    # the capacity decision does NOT skip compaction for (reorder's own
    # MIN_REORDER_SPAN floor still applies too).
    cap = _device_capacity(auto, "map", target) or 0
    n = max(MIN_REORDER_SPAN * 2, cap + MIN_REORDER_SPAN)
    inp = _batch(n, "spread")
    runner, report, got = _run(auto, "map", target, inp, 10 * _S, reorder=0.5)
    assert report.done and report.reorders >= 1 and report.compactions >= 1
    if target == "host":
        want, _ = _explicit({("plain", "host"): auto[("oracle", "host")]}, "host",
                            inp, 10 * _S)
    else:
        want, _ = _by_hand(auto, target, inp, 10 * _S)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


@pytest.mark.gpu
def test_auto_reorder_inside_capacity_keeps_compaction(auto):
    """Inside the capacity the rule skips compaction -- unless the caller asked
    for a reorder, which only runs as part of compaction (whether one fires
    is then the set's own decision: its MIN_REORDER_SPAN floor may exceed a
    small device's capacity)."""
    _need(auto, ("map", "device"))
    n = min(_device_capacity(auto, "map", "device") or 0, 1000)
    if n < 64:
        pytest.skip("device capacity unavailable")
    inp = _batch(n, "spread")
    _, plain, _ = _run(auto, "map", "device", inp, 10 * _S)
    _, report, got = _run(auto, "map", "device", inp, 10 * _S, reorder=0.5)
    assert plain.compactions == 0
    assert report.done and report.compactions >= 1
    want, _ = _by_hand(auto, "device", inp, 10 * _S)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


def _steps_bound(report, got):
    """``steps`` (the loop's) is within one launch of the longest sample's own
    step count (the oscillator's ``k`` plane), and equal to it when
    ``exact_steps``."""
    longest = int(np.max(got["k"]))
    assert longest <= report.steps < longest + max(report.launches_by_k, default=1)
    if report.exact_steps:
        assert report.steps == longest


@pytest.mark.parametrize("target", _TARGETS)
def test_steps_is_the_loop_s_and_exact_only_when_it_says_so(auto, target):
    """A uniform batch under the band climbs past S inside its last launch
    (16 + 32 + 64 * 4 = 304 > 300): ``steps`` overshoots the longest sample
    and is NOT exact. The measured host run of the same batch sweeps one step
    per launch: ``steps == 300``, exact."""
    _need(auto, ("plain", target))
    inp = _batch(1000, "uniform")
    _, band, got = _run(auto, "plain", target, inp, 10 * _S)
    if _runner_fast_mode(auto, target, "plain", 1000) is not None:
        # One launch of the whole budget, not the ramped band: n=1000 fits
        # inside this kernel/device's latency-regime capacity; the range
        # entry reports the longest sample's own steps, exactly.
        assert band.done and band.steps == _S and band.exact_steps
        assert band.launches_by_k == {10 * _S: 1}
    else:
        assert band.done and band.steps == 304 and not band.exact_steps
    assert int(np.max(got["k"])) == _S
    _steps_bound(band, got)
    if target == "host":
        _, swept, got = _run(auto, "plain", "host", inp, 10 * _S, policy="measured")
        assert swept.done and swept.steps == _S and swept.exact_steps
        assert swept.launches_by_k == {1: _S}
        _steps_bound(swept, got)


# --------------------------------------------------------------------------- #
# The host default and the tiled host policy
# --------------------------------------------------------------------------- #
def test_the_host_default_is_tiled_exactly_for_a_tiled_entry(auto):
    """The plain-kind auto artifact's host TU is the tiled loop (its entry's
    shared object exports ``<entry>_tile`` > 0); the active-set kind's is not
    (a gathered map is not a contiguous tile), nor is a kernel without fused
    steps. ``"auto"`` resolves to ``"tiled"`` / ``"measured"`` accordingly."""
    from eagle import _host_loop

    want = {"plain": True, "default": True, "map": False, "one": False}
    for label, tiled in want.items():
        g = _planes(_batch(64, "uniform"), np)
        runner = eagle.until_done(auto[(label, "host")], max_steps=10, dt=_DT, **g)
        assert (_host_loop.host_tile(runner.step._entry) > 0) == tiled, label
        assert runner.host_policy == "auto"
        if runner.auto:
            assert _host_loop.host_policy(runner) == ("tiled" if tiled else "measured")
    assert _host_loop.host_tile(0) == 0


@pytest.mark.parametrize("label", ["plain", "default"])
def test_the_tiled_host_policy_launches_steps_max_and_matches_the_oracle(auto, label):
    """A spread batch: every launch but the last asks for 64, no sweep, no
    probe; the planes equal the one-step oracle bit for bit."""
    inp = _batch(2000, "spread")
    want, _ = _explicit({("plain", "host"): auto[("oracle", "host")]}, "host",
                        inp, 10 * _S)
    g = _planes(inp, np)
    runner = eagle.until_done(auto[(label, "host")], max_steps=10 * _S, dt=_DT, **g)
    report = runner.run()
    got = {key: g[key].copy() for key in ("x", "v", "k")}
    assert report.done and set(report.launches_by_k) == {64}
    assert report.launches == -(-_S // 64)
    assert sum(k * c for k, c in report.launches_by_k.items()) == report.steps
    _steps_bound(report, got)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


@pytest.mark.parametrize("cap,launches", [(37, {37: 1}), (150, {64: 2, 22: 1})])
def test_the_tiled_host_policy_caps_exactly(auto, cap, launches):
    """Samples that never finish stop at exactly the cap: the last launch is
    cut to what is left (the fixed ``steps=16`` artifact would overshoot)."""
    _, report, got = _run(auto, "plain", "host", _batch(512, "never"), cap,
                          policy="tiled")
    assert (got["k"] == cap).all() and report.launches_by_k == launches
    assert report.steps == cap and report.exact_steps and not report.done


# --------------------------------------------------------------------------- #
# The measured host policy
# --------------------------------------------------------------------------- #
def _scripted(monkeypatch, runner_box, sweep_s, fused_step_s):
    """Replace the host loop's clock by a scripted one: each team launch adds
    ``sweep_s`` (a one-step launch) or ``k * fused_step_s`` (a launch of
    ``k`` steps) to it."""
    from eagle import _host_loop
    from eagle import exec as eexec

    now = [0.0]
    real = eexec.HostTeam.run

    def run(self, entry, params, partition, **kw):
        out = real(entry, params, partition, **kw)
        k = int(runner_box[0].fused_steps[0])
        now[0] += sweep_s if k == 1 else k * fused_step_s
        return out

    # the CLASS attribute (an instance attribute would outlive the undo as a
    # stale bound method shadowing every later patch of the class)
    monkeypatch.setattr(type(eexec.HostTeam), "run", run)
    monkeypatch.setattr(_host_loop, "clock", lambda: now[0])


def _measured(auto, label, inp, monkeypatch, sweep_s=None, fused_step_s=None):
    g = _planes(inp, np)
    runner = eagle.until_done(auto[(label, "host")], max_steps=10 * _S, dt=_DT, **g)
    if sweep_s is not None:
        _scripted(monkeypatch, [runner], sweep_s, fused_step_s)
    runner.host_policy = "measured"
    report = runner.run()
    return report, {key: g[key].copy() for key in ("x", "v", "k")}


@pytest.mark.parametrize("label", ["plain", "map"])
def test_the_measured_host_policy_sweeps_a_uniform_batch(auto, label):
    """Nothing halves, nothing probes: one step per launch, ``S`` launches."""
    inp = _batch(1000, "uniform")
    report, got = _measured(auto, label, inp, None)
    assert report.done and report.launches_by_k == {1: _S} and report.exact_steps
    want, _ = _explicit({("plain", "host"): auto[("oracle", "host")]}, "host",
                        inp, 10 * _S)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


@pytest.mark.parametrize("label", ["plain", "map"])
def test_the_measured_host_policy_follows_the_walls(auto, label, monkeypatch):
    """A spread batch: fused launches scripted cheaper per step than a sweep ->
    one probe of AUTO_K_MIN at the first halving, then the band (launches of
    16 and more); scripted dearer -> sweeps and failed probes of AUTO_K_MIN
    only, at most one per halving. Bit-equal to the one-step oracle in both
    (the non-vacuity pair: the walls alone separate the two paths)."""
    n = 2000
    inp = _batch(n, "spread")
    want, _ = _explicit({("plain", "host"): auto[("oracle", "host")]}, "host",
                        inp, 10 * _S)
    cheap, got_c = _measured(auto, label, inp, monkeypatch, 1.0, 0.1)
    assert cheap.done and cheap.launches_by_k.get(1, 0) >= 1
    assert cheap.launches_by_k.get(ud.AUTO_K_MIN) >= 1
    assert any(k >= ud.AUTO_K0 for k in cheap.launches_by_k)
    monkeypatch.undo()
    dear, got_d = _measured(auto, label, inp, monkeypatch, 1.0, 10.0)
    probes = {k: c for k, c in dear.launches_by_k.items() if k != 1}
    assert dear.done and dear.launches_by_k[1] > 0
    assert all(k <= ud.AUTO_K_MIN for k in probes)
    assert sum(probes.values()) <= int(np.log2(n)) + 2
    assert cheap.launches_by_k != dear.launches_by_k
    for rep, got in ((cheap, got_c), (dear, got_d)):
        assert sum(k * c for k, c in rep.launches_by_k.items()) == rep.steps
        _steps_bound(rep, got)
        for key in ("x", "v", "k"):
            np.testing.assert_array_equal(got[key], want[key], err_msg=key)


# --------------------------------------------------------------------------- #
# The exact cap (a non-vacuity pair)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("label", ["plain", "map"])
def test_auto_caps_exactly_where_the_fixed_artifact_overshoots(auto, label, target):
    _need(auto, (label, target))
    inp = _batch(512, "never")
    _, report, got = _run(auto, label, target, inp, 37)
    assert (got["k"] == 37).all()
    assert report.steps == 37 and report.exact_steps and not report.done
    if target == "host":
        _, swept, got = _run(auto, label, target, inp, 37, policy="measured")
        assert (got["k"] == 37).all() and swept.launches_by_k == {1: 37}
        assert swept.steps == 37 and swept.exact_steps and not swept.done
    assert sum(k * c for k, c in report.launches_by_k.items()) == 37
    _, fixed, fgot = _run(auto, "fixed16", target, inp, 37)
    assert (fgot["k"] == 48).all() and fixed.steps == 48
    assert fixed.launches_by_k == {16: 3} and not fixed.exact_steps


# --------------------------------------------------------------------------- #
# The band (with its non-vacuity)
# --------------------------------------------------------------------------- #
def _band_rows(auto, target):
    # At n=1000, a "map" (active-set) run on the DEVICE
    # skips compaction and pins K at k_max once n fits inside the kernel's
    # resident-wave capacity -- the ramped-band assertions below only hold
    # when that skip does NOT apply (the host band, untouched, or a device
    # whose kernel/capacity does not cover n=1000).
    skip = _map_skip(auto, target, "map", 1000) is not None
    _, uni, _ = _run(auto, "map", target, _batch(1000, "uniform"), 10 * _S)
    assert uni.done
    if skip:
        assert uni.compactions == 0
        assert 64 in uni.launches_by_k and not ({8, 16, 32} & set(uni.launches_by_k))
    else:
        assert uni.launches_by_k.get(64, 0) >= 2 and 8 not in uni.launches_by_k
    assert sum(k * c for k, c in uni.launches_by_k.items()) == uni.steps
    runner, half, _ = _run(auto, "map", target, _batch(1000, "half"), 10 * _S)
    assert half.done
    if skip:
        assert half.compactions == 0
        assert 64 in half.launches_by_k and not ({8, 16, 32} & set(half.launches_by_k))
    else:
        assert half.launches_by_k.get(8) == 1
        assert max(half.launches_by_k) == 64
    assert sum(k * c for k, c in half.launches_by_k.items()) == half.steps
    assert runner.policy["steps_done"] == half.steps and runner.policy["go"] == 0
    xp = _xp(target)
    g = _planes(_batch(256, "uniform"), xp)
    term = xp.ones(256, dtype=xp.bool_)
    done = eagle.run_until_done(auto[("map", target)], max_steps=_S, dt=_DT,
                                terminated=term, **g)
    assert done.launches == 0 and done.launches_by_k == {} and done.steps == 0


@pytest.mark.parametrize("target", _TARGETS)
def test_the_band_climbs_halves_and_stops(auto, target):
    _need(auto, ("map", target))
    _band_rows(auto, target)


@pytest.mark.parametrize("target", _TARGETS)
def test_a_constant_k_policy_turns_the_band_red(auto, target, monkeypatch):
    _need(auto, ("map", target))
    monkeypatch.setattr(
        ud, "_next_k",
        lambda k, newly, live, k_max, k_min=ud.AUTO_K_MIN: ud.AUTO_K0)
    monkeypatch.setattr(ud, "_BAND_SRC", f"k_next = {ud.AUTO_K0};")
    with pytest.raises(AssertionError):
        _band_rows(auto, target)


# --------------------------------------------------------------------------- #
# Host == device
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("which", ["spread", "half", "grouped"])
def test_host_and_device_take_the_same_path(auto, which):
    _need(auto, ("map", "device"))
    # Comfortably ABOVE the no-active-set-rebuild rule's capacity: below it the device
    # intentionally diverges from the host (skips compaction, pins K), which
    # is this item's whole point, not a bug this parity check should catch.
    cap = _device_capacity(auto, "map", "device") or 0
    n = max(2000, cap + 1000)
    inp = _batch(n, which)
    _, host, _ = _run(auto, "map", "host", inp, 10 * _S)
    _, dev, _ = _run(auto, "map", "device", inp, 10 * _S)
    assert host.launches_by_k == dev.launches_by_k
    assert (host.compactions, host.steps, host.launches) == (
        dev.compactions, dev.steps, dev.launches)
    assert host.compactions >= 1


# --------------------------------------------------------------------------- #
# The default: the plain decorator builds the auto kernel; a launch outside
# until_done takes exactly one step
# --------------------------------------------------------------------------- #
def test_the_plain_decorator_builds_the_auto_kernel_under_its_name(auto):
    p = auto[("default", "host")].plugin
    assert p.name == auto[("one", "host")].plugin.name == "oscillator_step"
    assert dict(p.finish) == _AUTO
    assert ("lookup", FUSED_STEPS) in {tuple(pair) for pair in p.arg_spec}
    assert dict(auto[("one", "host")].plugin.finish)["steps"] == 1


FUSED_STEPS = "fused_steps"


def _launches(plan, target, inp, door, launches, word=None):
    """``launches`` launches of ``plan`` through ``door`` (``run`` / ``bind``),
    the word bound only when ``word`` is given; the planes after."""
    xp = _xp(target)
    g = _planes(inp, xp)
    n = inp["x"].shape[0]
    term = xp.zeros(n, dtype=xp.bool_)
    counter = xp.zeros(1, dtype=xp.uint32)
    kw = dict(dt=_DT, terminated=term, finished_count=counter.view(xp.int32), **g)
    if word is not None:
        kw[FUSED_STEPS] = xp.full(1, word, dtype=xp.int64)
    if door == "bind":
        bound = plan.bind(**kw)
        for _ in range(launches):
            bound.launch()
    else:
        for _ in range(launches):
            plan.run(**kw)
    return {key: _get(g[key]).copy() for key in ("x", "v", "k")}


@pytest.mark.parametrize("door", ["run", "bind"])
@pytest.mark.parametrize("target", _TARGETS)
def test_a_launch_outside_until_done_takes_exactly_one_step(auto, target, door):
    """Plan.run / Plan.bind of the default kernel without the word: each
    launch is ONE step (k counts them). Host: bit-equal to the ``steps=1``
    kernel; device: equal to the same artifact with the word bound to 1 (FMA
    contraction may differ from the plain kernel by an ULP). Non-vacuity: a
    word of 5 takes five steps."""
    _need(auto, ("default", target))
    inp = _batch(256, "uniform")
    got = _launches(auto[("default", target)], target, inp, door, 3)
    np.testing.assert_array_equal(got["k"], 3.0)
    ref_label, word = ("one", None) if target == "host" else ("default", 1)
    want = _launches(auto[(ref_label, target)], target, inp, door, 3, word=word)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
    five = _launches(auto[("default", target)], target, inp, door, 3, word=5)
    np.testing.assert_array_equal(five["k"], 15.0)
    assert not np.array_equal(five["x"], got["x"])


@pytest.mark.parametrize("door", ["run", "bind"])
def test_without_the_one_step_word_a_launch_refuses(auto, door, monkeypatch):
    """Non-vacuity of the default word: with ``one_step_word`` disabled, the
    same launch refuses naming the unbound ``fused_steps``."""
    import eagle._plan_binding as binding
    import eagle.plan as eplan

    for mod in (eplan, binding):
        monkeypatch.setattr(mod, "one_step_word", lambda spec, planes, *a: planes)
    with pytest.raises(ValueError, match=FUSED_STEPS):
        _launches(auto[("default", "host")], "host", _batch(64, "uniform"), door, 1)


@pytest.mark.gpu
def test_device_plan_run_puts_the_one_step_word_on_the_device(auto, monkeypatch):
    """A device Plan.run builds its one-step word on the device (no upload per
    run), and that word does not count as a device input: host planes in,
    host planes out, one step taken."""
    import eagle.plan as eplan

    _need(auto, ("default", "device"))
    seen = []
    real = eplan.one_step_word

    def spy(spec, planes, *a):
        out = real(spec, planes, *a)
        seen.append(type(out.get(FUSED_STEPS)).__module__.split(".")[0])
        return out

    monkeypatch.setattr(eplan, "one_step_word", spy)
    g = _planes(_batch(256, "uniform"), np)
    n = g["x"].shape[0]
    auto[("default", "device")].run(
        dt=_DT, terminated=np.zeros(n, dtype=bool),
        finished_count=np.zeros(1, dtype=np.int32), **g)
    assert seen == ["cupy"]
    assert isinstance(g["k"], np.ndarray)
    np.testing.assert_array_equal(g["k"], 1.0)


def test_until_done_paces_the_default_kernel_like_the_explicit_auto(auto):
    """The default kernel IS the auto kernel to until_done: the same policy
    (launches_by_k), the same planes as ``hawk.steps(kernel, "auto")``; and
    ``every=`` on an auto active-set kernel is refused naming the opt-out."""
    inp = _batch(2000, "spread")
    _, rep_d, got = _run(auto, "default", "host", inp, _S)
    _, rep_a, want = _run(auto, "plain", "host", inp, _S)
    assert rep_d.launches_by_k == rep_a.launches_by_k and rep_d.done
    assert set(rep_d.launches_by_k) != {1}
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
    g = _planes(_batch(64, "uniform"), np)
    with pytest.raises(ValueError, match="steps=1"):
        eagle.until_done(auto[("map", "host")], max_steps=10, every=64,
                         dt=_DT, **g)



# --------------------------------------------------------------------------- #
# The fast paths report EXACT steps: the range and persist entries max each
# sample's steps taken this run into a run-level word read back with the
# finished count, so ``steps`` is the longest sample's own count this run.
# --------------------------------------------------------------------------- #
def _taken(before, after):
    """The longest per-sample step count of one run, off the ``k`` plane."""
    d = np.asarray(after["k"], dtype=np.int64) - np.asarray(before["k"], dtype=np.int64)
    return int(d.max()) if d.size else 0


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fused_one", "persist"])
@pytest.mark.parametrize("label", ["plain", "map"])
@pytest.mark.parametrize("n", [1, 33, 1000, 2049, 10_000])
@pytest.mark.parametrize("which", ["spread", "uniform"])
def test_fast_path_steps_are_exact(auto, active_auto, mode, label, n, which):
    """All finish, budget hit, already finished, and a run continuing a
    partly-run batch: ``steps`` is the longest sample's steps of that run,
    ``exact_steps``; the planes stay bit-identical to the device oracle and
    to the band path's, and ``steps`` equals the band's report wherever the
    band's is exact."""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")
    # the map kind as eagle.deploy builds it (with the fast entries)
    plan = auto[("plain", "device")] if label == "plain" else active_auto

    def _run(_auto, _label, _target, inp, max_steps, **kw):
        g = _planes(inp, cp)
        runner = eagle.until_done(plan, max_steps=max_steps, dt=_DT, **g, **kw)
        return runner, runner.run(), {key: _get(g[key]) for key in ("x", "v", "k")}

    inp = _batch(n, which)
    zero = {"k": np.zeros(n)}
    # all finish
    _, rep, got = _run(auto, label, "device", inp, 10 * _S, _fast_mode=mode)
    want, _ = _by_hand(auto, "device", inp, 10 * _S)
    assert rep.done and rep.exact_steps and rep.steps == _taken(zero, got) > 0
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
    _, band, bgot = _run(auto, label, "device", inp, 10 * _S, _fast_mode=False)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(bgot[key], want[key], err_msg=f"band {key}")
    if band.exact_steps:
        assert band.steps == rep.steps
    # budget hit, then the run continuing the partly-run batch
    cap = _S // 3
    g = _planes(inp, cp)
    runner = eagle.until_done(plan, max_steps=cap, dt=_DT, _fast_mode=mode, **g)
    first = runner.run()
    mid = {"k": _get(g["k"])}
    assert first.exact_steps and first.steps == _taken(zero, mid) <= cap
    second = runner.run()
    end = {"k": _get(g["k"])}
    assert second.exact_steps and second.steps == _taken(mid, end)
    want2, _ = _by_hand(auto, "device", inp, 2 * cap)
    for key in ("x", "v", "k"):
        np.testing.assert_array_equal(_get(g[key]), want2[key], err_msg=f"continuing {key}")
    # already finished on entry: still launches (no cross-call "done" cache
    # any more), but it is a no-op
    g = _planes(inp, cp)
    term = cp.ones(n, dtype=cp.bool_)
    done = eagle.run_until_done(plan, max_steps=_S, dt=_DT,
                                terminated=term, _fast_mode=mode, **g)
    assert done.launches == 1 and done.steps == 0 and done.exact_steps
    assert done.done and done.finished == n


# --------------------------------------------------------------------------- #
# The fast path used to seed `finished` from the mask
# ONCE and cache "already done" across calls (`_fast_known_done`), so a
# caller who edited `terminated` between runs without `reset()` got a stale,
# wrong report and no re-launch. `run()`'s own contract ("the counter is
# re-synced to the mask first, so a run continues from the current state")
# must hold on EVERY call, fast path or band, for a plain and an active-set
# kernel.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("n", [1000, 5000])
def test_second_run_resyncs_to_an_edited_mask_without_reset(auto, active_auto, n):
    """Un-terminate half the batch between two ``run()`` calls, with no
    ``reset()`` in between -- a caller re-running a subset by hand (``k``
    left as the finished run left it, so the restored samples cross their
    own stop step again on the very next step, same as the reviewer's
    probe). The old cache would have returned a stale
    ``launches=0``/``steps=0`` no-op (no device touch at all) still
    claiming ``finished == n``; the fix must actually launch, and the
    report it returns must equal the mask's own true count once that run
    ends. Checked on eagle's own auto pick (which is ``fused_one`` at
    n=1000, below this kernel/device's capacity, and ``persist`` at
    n=5000, above it) and on the explicit band (``_fast_mode=False``),
    which must agree bit for bit; and for a plain and an active-set
    (compacting) automatic kernel."""
    _need(auto, ("plain", "device"))
    _need_active(active_auto)
    cp = pytest.importorskip("cupy")
    inp = _batch(n, "uniform")  # every sample stops at exactly _S
    half = n // 2
    for plan in (auto[("plain", "device")], active_auto):
        results = {}
        for force in (None, False):  # None: eagle's own auto pick; False: the band
            g = _planes(inp, cp)
            term = cp.zeros(n, dtype=cp.bool_)
            kw = {} if force is None else {"_fast_mode": force}
            runner = eagle.until_done(plan, max_steps=_S, dt=_DT, terminated=term,
                                      **g, **kw)
            report1 = runner.run()
            assert report1.done and report1.finished == n and int(term.sum()) == n
            # the caller edits terminated directly, no reset(): the old
            # cache would have skipped the very next launch entirely.
            term[:half] = False
            report2 = runner.run()
            assert report2.launches > 0  # it re-ran them, not the old no-op skip
            assert report2.finished == int(term.sum())
            assert report2.done and report2.finished == n
            results[force] = {key: _get(g[key]) for key in ("x", "v", "k")}
        for key in ("x", "v", "k"):
            np.testing.assert_array_equal(results[None][key], results[False][key],
                                          err_msg=key)


@pytest.mark.gpu
def test_device_run_never_invokes_cupy_s_count_nonzero(auto, monkeypatch):
    """eagle's own mask-count kernel (:func:`eagle._until_done._mask_count`)
    replaces cupy's ``count_nonzero`` on the device path, fast or band:
    that reduction's cub kernel JIT-compiles slowly on a cold
    ``CUPY_CACHE_DIR`` (~2.9 s, measured).
    A spy that raises if ``cupy.count_nonzero`` is ever called must survive
    a FRESH runner's first run (seeding the finished count from a mask the
    caller may have handed over partly terminated) and a second,
    continuing run -- in both the fast path and the band. (The sync count
    itself -- still exactly one host<->device round trip per run -- is
    pinned separately by :func:`test_fast_path_run_does_one_sync`.)"""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")

    def _boom(*a, **k):
        raise AssertionError("cupy.count_nonzero was called on the device path")

    monkeypatch.setattr(cp, "count_nonzero", _boom)
    n, half = 500, _S // 2
    for force in (None, False):  # None: eagle's own auto pick; False: the band
        inp = _batch(n, "uniform")
        g = _planes(inp, cp)
        term = cp.zeros(n, dtype=cp.bool_)
        term[:50] = True  # some samples already terminated at the first run
        kw = {} if force is None else {"_fast_mode": force}
        runner = eagle.until_done(auto[("plain", "device")], max_steps=half, dt=_DT,
                                  terminated=term, **g, **kw)
        report1 = runner.run()
        assert report1.finished == 50 and not report1.done
        report2 = runner.run()
        assert report2.done and report2.finished == n


# --------------------------------------------------------------------------- #
# The measured entry pick: above capacity, with both fast entries present, the
# first run takes the size pick, the second the other entry, and later runs
# the one whose launch wall was smaller.
# --------------------------------------------------------------------------- #
def _probe_runner(auto, n, **kw):
    """A device runner over ``n`` > capacity samples (a uniform batch: a full
    step fill, so the fused entry is worth timing) plus a ``rerun()`` that
    restores the inputs, resets, runs and returns ``(report, planes)``."""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")
    inp = _batch(n, "uniform")
    g = _planes(inp, cp)
    runner = eagle.until_done(auto[("plain", "device")], max_steps=10 * _S, dt=_DT,
                              **g, **kw)

    def rerun():
        fresh = _planes(inp, cp)
        for key in g:
            g[key][...] = fresh[key]
        runner.reset()
        report = runner.run()
        return report, {key: _get(g[key]) for key in ("x", "v", "k")}

    return runner, rerun


def _slow_entry(monkeypatch, runner, slow, seconds=0.03):
    """Make the ``slow`` entry's launch cost ``seconds`` more host time (the
    launch events bracket it, so the measured wall carries it)."""
    import time

    name = "launch_range" if slow == "fused_one" else "launch_persist"
    cls = type(runner.step)  # BoundPlan is slotted: patch the class
    real = getattr(cls, name)

    def delayed(self, *a, **k):
        time.sleep(seconds)
        return real(self, *a, **k)

    monkeypatch.setattr(cls, name, delayed)


@pytest.mark.gpu
@pytest.mark.parametrize("slow", ["fused_one", "persist"])
def test_fast_entry_probe_picks_the_faster_entry(auto, monkeypatch, slow):
    runner, rerun = _probe_runner(auto, _above(auto))
    first = runner._fast_mode
    other = "persist" if first == "fused_one" else "fused_one"
    assert first == "persist"  # above capacity the size rule picks persist
    _slow_entry(monkeypatch, runner, slow)
    reports = [rerun()[0] for _ in range(5)]
    fast = "persist" if slow == "fused_one" else "fused_one"
    assert [r.mode for r in reports] == [first, other, fast, fast, fast]
    assert [r.probe for r in reports] == [False, True, False, False, False]
    assert all(r.done for r in reports)


@pytest.mark.gpu
def test_fast_entry_probe_is_bit_identical_across_switches(auto, monkeypatch):
    n = _above(auto)
    runner, rerun = _probe_runner(auto, n)
    _slow_entry(monkeypatch, runner, "persist")  # the pick moves to fused_one
    oracles = {}
    for mode in ("persist", "fused_one"):
        _, _, oracles[mode] = _run(auto, "plain", "device", _batch(n, "uniform"),
                                   10 * _S, _fast_mode=mode)
    seen = set()
    for _ in range(5):
        report, got = rerun()
        seen.add(report.mode)
        for key in ("x", "v", "k"):
            np.testing.assert_array_equal(got[key], oracles[report.mode][key],
                                          err_msg=f"{report.mode}:{key}")
    assert seen == {"persist", "fused_one"}


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["persist", "fused_one"])
def test_forced_fast_mode_never_probes(auto, mode):
    runner, rerun = _probe_runner(auto, _above(auto), _fast_mode=mode)
    reports = [rerun()[0] for _ in range(5)]
    assert [r.mode for r in reports] == [mode] * 5
    assert not any(r.probe for r in reports)
    assert runner._fast_pick is None and not runner._fast_wall


@pytest.mark.gpu
def test_fast_entry_probe_reprobes_on_regime_change(auto, monkeypatch):
    n = _above(auto)
    runner, rerun = _probe_runner(auto, n)
    _slow_entry(monkeypatch, runner, "persist")
    modes = [rerun()[0].mode for _ in range(4)]
    assert modes[2:] == ["fused_one"] * 2
    # a run that ends differently (no sample can finish within 3 steps) is a
    # new regime: the loser is timed again within two runs
    runner._max_steps = 3
    runner.fused_steps[0] = 3
    after = [rerun() for _ in range(3)]
    assert any(r.probe for r, _ in after[:2])
    assert {r.mode for r, _ in after[:2]} == {"persist", "fused_one"}


@pytest.mark.gpu
def test_fast_entry_probe_below_capacity_is_off(auto):
    runner, rerun = _probe_runner(auto, 1000)
    reports = [rerun()[0] for _ in range(4)]
    assert [r.mode for r in reports] == ["fused_one"] * 4
    assert not any(r.probe for r in reports)
    assert runner._fast_pick is None


@pytest.mark.gpu
def test_fast_entry_probe_pins_persist_without_a_range_entry(auto, monkeypatch):
    _need(auto, ("plain", "device"))
    from eagle._plan_binding import BoundPlan

    monkeypatch.setattr(BoundPlan, "range_entry", lambda self: None)
    runner, rerun = _probe_runner(auto, _above(auto))
    reports = [rerun()[0] for _ in range(4)]
    assert [r.mode for r in reports] == ["persist"] * 4
    assert not any(r.probe for r in reports)
    assert runner._fast_pick is None


# --------------------------------------------------------------------------- #
# The step-fill prior: the first persist run counts every lane's executed steps
# (utilisation counter on for that run only), f = sum(steps) / (n * longest);
# below FUSED_MIN_STEP_RATIO the fused entry is never timed.
# --------------------------------------------------------------------------- #
def _prior_runner(auto, which, n=None, **kw):
    """A device runner over ``n`` > capacity samples of batch ``which`` plus a
    ``rerun(nstop=None)`` that restores the inputs (optionally under a new
    ``nstop``), resets, runs and returns the report."""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")
    n = n or _above(auto)
    inp = _batch(n, which)
    g = _planes(inp, cp)
    runner = eagle.until_done(auto[("plain", "device")], max_steps=10 * _S, dt=_DT,
                              **g, **kw)

    def rerun(nstop=None):
        fresh = _planes(inp, cp)
        for key in g:
            g[key][...] = fresh[key]
        if nstop is not None:
            g["nstop"][...] = cp.asarray(nstop)
        runner.reset()
        return runner.run()

    return runner, inp, rerun


def _spy_util(monkeypatch, runner):
    seen = []
    cls = type(runner.step)
    real = cls.launch_persist

    def spy(self, *a, **k):
        seen.append(k.get("util") is not None)
        return real(self, *a, **k)

    monkeypatch.setattr(cls, "launch_persist", spy)
    return seen


@pytest.mark.gpu
def test_entry_prior_is_the_step_sum(auto):
    runner, inp, rerun = _prior_runner(auto, "spread")
    report = rerun()
    assert report.mode == "persist" and not report.probe
    total = int(runner._fast_views[2].get()[0])
    assert total == int(np.minimum(inp["nstop"], 10 * _S).sum())
    assert runner._fast_prior == total / (runner.n * report.steps)


@pytest.mark.gpu
def test_entry_prior_pins_persist_on_a_spread_batch(auto, monkeypatch):
    runner, _, rerun = _prior_runner(auto, "spread")
    seen = _spy_util(monkeypatch, runner)
    reports = [rerun() for _ in range(4)]
    assert [r.mode for r in reports] == ["persist"] * 4
    assert not any(r.probe for r in reports)
    assert set(runner._fast_wall) == {"persist"}
    assert runner._fast_prior < ud.FUSED_MIN_STEP_RATIO
    assert seen == [False] * 4  # util is never on unless asked


@pytest.mark.gpu
def test_entry_prior_still_probes_a_uniform_batch(auto):
    runner, _, rerun = _prior_runner(auto, "uniform")
    reports = [rerun() for _ in range(3)]
    assert runner._fast_prior == 1.0
    assert [r.mode for r in reports][:2] == ["persist", "fused_one"]
    assert [r.probe for r in reports][:2] == [False, True]
    assert set(runner._fast_wall) == {"persist", "fused_one"}


@pytest.mark.gpu
def test_entry_prior_reevaluates_on_a_new_readback(auto):
    runner, inp, rerun = _prior_runner(auto, "uniform")
    modes = [rerun().mode for _ in range(2)]
    assert modes == ["persist", "fused_one"] and runner._fast_prior == 1.0
    # a spread batch with a shorter longest sample: a new readback, so the
    # prior is re-measured and now pins persist
    spread = np.minimum(_batch(runner.n, "spread")["nstop"], _S // 2)
    reports = [rerun(spread) for _ in range(4)]
    assert runner._fast_prior < ud.FUSED_MIN_STEP_RATIO
    assert [r.mode for r in reports[-2:]] == ["persist"] * 2
    assert not any(r.probe for r in reports[-2:])
    assert set(runner._fast_wall) == {"persist"}
    # and back: the uniform readback flips f to 1.0 and the probe returns
    back = [rerun(inp["nstop"]) for _ in range(4)]
    assert runner._fast_prior == 1.0
    assert any(r.probe and r.mode == "fused_one" for r in back)


@pytest.mark.gpu
def test_entry_prior_keeps_a_heavy_adaptive_kernel_on_persist(auto):
    # half the samples stop at step 1, the rest run _S: f ~ 0.5
    runner, _, rerun = _prior_runner(auto, "half")
    reports = [rerun() for _ in range(4)]
    assert 0.4 < runner._fast_prior < ud.FUSED_MIN_STEP_RATIO
    assert [r.mode for r in reports] == ["persist"] * 4
    assert not any(r.probe for r in reports)


@pytest.mark.gpu
def test_entry_prior_keeps_the_utilisation_report_opt_in(auto, monkeypatch):
    runner, _, rerun = _prior_runner(auto, "spread")
    seen = _spy_util(monkeypatch, runner)
    reports = [rerun() for _ in range(3)]
    assert all(r.lane_utilisation is None for r in reports)
    assert not runner._fast_util_on and seen == [False] * 3
    # asked for, the report is filled on every run and the counter stays on
    asked, _, again = _prior_runner(auto, "spread", _lane_utilisation=True)
    reports = [again() for _ in range(3)]
    assert all(r.lane_utilisation is not None and 0 < r.lane_utilisation <= 1
               for r in reports)
    assert asked._fast_util_on


@pytest.mark.gpu
def test_entry_pick_settled_runs_are_untimed(auto, monkeypatch):
    runner, _, rerun = _prior_runner(auto, "uniform")
    for _ in range(4):
        rerun()
    assert set(runner._fast_wall) == {"persist", "fused_one"}
    wall = dict(runner._fast_wall)
    records = []
    ev_cls = type(runner._fast_ev[0])
    real = ev_cls.record

    def spy(self, *a, **k):
        records.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(ev_cls, "record", spy)
    for _ in range(10):
        rerun()
    assert runner._fast_wall == wall and not records


@pytest.mark.gpu
def test_settled_runs_sum_their_steps_one_run_in_step_fill_every(auto, monkeypatch):
    """A settled runner passes the step-sum cell on one run in
    ``STEP_FILL_EVERY`` and none on the others (the device adds nothing
    there); the pick and the step fill stay as they were."""
    from eagle._until_done import STEP_FILL_EVERY

    runner, _, rerun = _prior_runner(auto, "uniform")
    for _ in range(4):
        rerun()
    pick, prior = runner._fast_pick, runner._fast_prior
    # the summing launches: the prepared variants bound WITH the step-sum cell
    # (the range branch passes only prepared=, the persist one stepsum= too)
    summing = [runner._fast_prep[key] for key in ("range", "persist", "persist_util")
               if key in runner._fast_prep]
    sums = []
    for name in ("launch_range", "launch_persist"):
        real = getattr(type(runner.step), name)

        def spy(self, *a, _real=real, **k):
            sums.append(any(k.get("prepared") is p for p in summing))
            return _real(self, *a, **k)

        monkeypatch.setattr(type(runner.step), name, spy)
    for _ in range(3 * STEP_FILL_EVERY):
        rerun()
    assert sum(sums) == 3 and len(sums) == 3 * STEP_FILL_EVERY
    assert runner._fast_pick == pick and runner._fast_prior == prior


@pytest.mark.gpu
def test_device_reset_launches_no_fill_kernel(auto, monkeypatch):
    """reset() clears the mask and the finished count by async memset on the
    current stream: neither array's ``fill`` (a cupy kernel launch) runs."""
    runner, _, rerun = _prior_runner(auto, "uniform")
    report = rerun()
    assert report.finished == runner.n

    def no_fill(self, *a, **k):
        raise AssertionError("reset() launched a fill kernel")

    cp = pytest.importorskip("cupy")
    monkeypatch.setattr(cp.ndarray, "fill", no_fill, raising=False)
    runner.reset()
    cp.cuda.Device().synchronize()
    assert not bool(runner.terminated.any()) and int(runner.finished[0]) == 0


def _ops_runner(auto, monkeypatch, step_ops):
    """A uniform ``_prior_runner`` whose plugin reports ``step_ops`` (``None``:
    absent), as ``(runner, rerun, n)``."""
    _need(auto, ("plain", "device"))
    plugin = auto[("plain", "device")].plugin
    if step_ops is None:
        monkeypatch.delattr(plugin, "step_ops", raising=False)
    else:
        monkeypatch.setattr(plugin, "step_ops", step_ops, raising=False)
    runner, inp, rerun = _prior_runner(auto, "uniform")
    return runner, rerun, len(inp["nstop"])


@pytest.mark.gpu
@pytest.mark.parametrize("step_ops, bound", [(None, 0.6), (40, 0.7), (1203, 1 - 12 / 1203)])
def test_entry_prior_bound_follows_step_ops(auto, monkeypatch, step_ops, bound):
    runner, _, _ = _ops_runner(auto, monkeypatch, step_ops)
    assert runner._fast_ratio == pytest.approx(bound)


@pytest.mark.gpu
def test_entry_prior_heavy_step_skips_the_probe_at_f_09(auto, monkeypatch):
    # 20% of the samples stop at half the budget: f = 0.9
    runner, rerun, n = _ops_runner(auto, monkeypatch, 1203)
    nstop = np.where(np.arange(n) % 5 == 0, _S // 2, _S).astype(float)
    reports = [rerun(nstop) for _ in range(4)]
    assert 0.85 < runner._fast_prior < 0.95
    assert [r.mode for r in reports] == ["persist"] * 4
    assert not any(r.probe for r in reports)
    assert set(runner._fast_wall) == {"persist"}


@pytest.mark.gpu
def test_entry_prior_light_step_still_probes_at_f_09(auto, monkeypatch):
    runner, rerun, n = _ops_runner(auto, monkeypatch, 40)
    nstop = np.where(np.arange(n) % 5 == 0, _S // 2, _S).astype(float)
    reports = [rerun(nstop) for _ in range(3)]
    assert [r.mode for r in reports][:2] == ["persist", "fused_one"]
    assert reports[1].probe


@pytest.mark.gpu
def test_fast_path_runner_allocates_no_active_map(active_auto):
    """The active-set map is never read on the fast path, so a fast runner
    holds a one-word stand-in; the band path (and a forced band) gets the
    real, identity map, bound before its first launch."""
    _need_active(active_auto)
    cp = pytest.importorskip("cupy")
    n = 5000
    inp = _batch(n, "spread")
    fast = eagle.until_done(active_auto, max_steps=10 * _S, dt=_DT, **_planes(inp, cp))
    assert fast._fast_mode is not None and fast.active.map.size == 1
    fast.run()
    fast.reset()
    assert fast.active.map.size == 1 and fast.run().done
    band = eagle.until_done(active_auto, max_steps=10 * _S, dt=_DT, _fast_mode=False,
                            **_planes(inp, cp))
    assert band.active.map.size == n and band.run().done


@pytest.mark.gpu
def test_entry_prior_is_known_after_a_fused_one_run(auto):
    runner, _, rerun = _prior_runner(auto, "uniform")
    reports = [rerun() for _ in range(2)]
    assert reports[1].mode == "fused_one" and runner._fast_prior == 1.0
    runner._fast_prior = None  # forget it: the fused run alone must restore it
    runner._fast_pick = "fused_one"
    runner._fast_wall.pop("persist")
    assert rerun().mode == "fused_one" and runner._fast_prior == 1.0


@pytest.mark.gpu
def test_entry_prior_stale_after_a_same_readback_change_moves_to_persist(auto):
    runner, inp, rerun = _prior_runner(auto, "uniform")
    for _ in range(4):
        rerun()
    assert set(runner._fast_wall) == {"persist", "fused_one"}
    # half the samples stop at step 1: the same longest lane and finished
    # count, so only f changes
    # count, so only f changes -- seen on the next run that sums its steps,
    # within STEP_FILL_EVERY runs, and persist with no probe from then on
    half = np.where(np.arange(runner.n) % 2 == 0, 1.0, float(_S))
    reports, priors = [], []
    for _ in range(2 * ud.STEP_FILL_EVERY):
        reports.append(rerun(half))
        priors.append(runner._fast_prior)
    seen = next(i for i, f in enumerate(priors) if f < ud.FUSED_MIN_STEP_RATIO)
    assert seen < ud.STEP_FILL_EVERY
    assert all(r.mode == "persist" and not r.probe for r in reports[seen + 1:])


@pytest.mark.gpu
def test_fast_run_takes_one_device_to_host_transfer(auto, monkeypatch):
    runner, _, rerun = _prior_runner(auto, "uniform")
    for _ in range(3):
        rerun()
    cp = pytest.importorskip("cupy")
    gets = []
    real = cp.ndarray.get

    def counting(self, *a, **k):
        gets.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(cp.ndarray, "get", counting)
    for _ in range(4):
        before = len(gets)
        rerun()
        assert len(gets) - before == 1


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fused_one", "persist"])
def test_fast_entries_count_the_mask_so_the_run_launches_no_mask_count(auto, monkeypatch, mode):
    """An artifact whose fast entries count the samples finished on entry
    (``entry_counts_finished``) gets a zeroed counter, not eagle's mask-count
    kernel, and still reports the batch's finished count -- here after the
    caller pre-finished every third sample."""
    _need(auto, ("plain", "device"))
    cp = pytest.importorskip("cupy")
    import eagle._until_done as ud

    assert getattr(auto[("plain", "device")].plugin, "entry_counts_finished", False)
    calls = []
    real = ud._mask_count
    monkeypatch.setattr(ud, "_mask_count", lambda *a, **k: calls.append(1) or real(*a, **k))
    n = 2000
    g = _planes(_batch(n, "spread"), cp)
    runner = eagle.until_done(auto[("plain", "device")], max_steps=10 * _S, dt=_DT,
                              _fast_mode=mode, **g)
    runner.terminated[::3] = True
    rt_calls = []

    class _Runtime:
        # counts the run's memsets and async copies, forwards everything
        def __getattr__(self, name):
            fn = getattr(real_rt, name)
            if name in ("memsetAsync", "memcpyAsync"):
                return lambda *a: rt_calls.append(name) or fn(*a)
            return fn

    real_rt = cp.cuda.runtime
    monkeypatch.setattr(cp.cuda, "runtime", _Runtime())
    report = runner.run()
    monkeypatch.setattr(cp.cuda, "runtime", real_rt)
    assert not calls
    # one memset zeroes every counter word, the finished count among them;
    # no device-to-device copy gathers the count for the readback
    assert rt_calls == ["memsetAsync"]
    assert report.done and report.finished == n
