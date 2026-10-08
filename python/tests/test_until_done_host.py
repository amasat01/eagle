# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host branch of ``eagle.until_done`` (``eagle._host_loop``): the loop
launches the bound entry on the host team directly, one team launch per step.

What is pinned (the ruling's rows, host side):

* The oracle: the runner's planes are bit-equal to today's explicit path (the
  non-finishing kernel plus the NumPy stop rule in a ``repeat_while``) AND to
  the serial oracle (the same finishing entry through
  ``HostTeam.run_serial`` until every sample is done); mask all set,
  ``finished == n``, ``done``; for the RK4 oscillator and, where the checkout
  carries it, the RK7(8) card's attempt kernel;
* Non-vacuity: launches follow the longest sample (``<= S + every``) and
  equal ``loop.iterations()``; an epilogue that does not count turns the row
  red (the loop runs to its cap, ``done`` is false);
* Monotone set/reset (#36): pre-set samples keep their planes; ``reset()`` runs them;
* The ``Guard(active_set=True)`` artifact through the same call, bit-equal
  to the oracle above, compacting on a spread batch, and reordering (``reorder=0.5``) with
  sample order restored;
* the loop is the direct one (the step is launched on the team, not through
  the loop's host arm), and a step it cannot see through runs the host arm.
"""

from __future__ import annotations

import ctypes
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_FINISH = {"mask": "terminated", "counter": "finished_count", "steps": 1}


# --------------------------------------------------------------------------- #
# A finishing HOST kernel. Until hawk emits them itself, the fixture takes
# hawk's own host TU for the non-finishing step and adds, by text, what the
# finishing epilogue adds: a writable mask, the
# `finished_count` counter appended to the arg_spec, and as the LAST text of
# the guarded scope `if (cond) { mask[i] = 1; atomic_ref(count) += 1; }`.
# Once hawk's sidecar carries `finish`, `_finishing_host_plan` builds through
# hawk unchanged and the patch is not used.
# --------------------------------------------------------------------------- #
_GUARD_OPEN = "if (!trm_terminated[i].eval()) {"


def _patch_finish(src: str, *, cond: str, mask_slot: int, counter_slot: int,
                  count: bool = True) -> str:
    """Add the finish epilogue to hawk's host TU ``src`` (``count=False``: the
    epilogue marks the sample but never counts it -- the non-vacuity red case)."""
    decl = re.search(r"\n( *)\(void\)trm_terminated;\n", src)
    assert decl, "hawk's host TU no longer declares trm_terminated"
    pad = decl.group(1)
    handle = "static_cast<const eagle::plugin::ScalarHandle*>(params[{}])->data"
    handles = (
        f"{pad}unsigned char* const hawk_mask_w = static_cast<unsigned char*>("
        f"{handle.format(mask_slot)});\n"
        f"{pad}std::uint32_t* const hawk_finished = static_cast<std::uint32_t*>("
        f"{handle.format(counter_slot)});\n")
    src = src[:decl.end()] + handles + src[decl.end():]
    assert src.count(_GUARD_OPEN) == 1, "expected exactly one guarded scope"
    at = src.index(_GUARD_OPEN)
    indent = src[src.rindex("\n", 0, at) + 1:at]
    close = src.index("\n" + indent + "}\n", at)
    tally = ("std::atomic_ref<std::uint32_t>(*hawk_finished)"
             ".fetch_add(1u, std::memory_order_relaxed);" if count
             else "(void)hawk_finished;")
    epilogue = (
        f"\n{indent}    if ({cond}) {{\n"
        f"{indent}        hawk_mask_w[i.global()] = 1;\n"
        f"{indent}        {tally}\n"
        f"{indent}    }}")
    return src[:close] + epilogue + src[close:]


def _load_host(so, entry_name, view, name, extra_keepalive=()):
    from hawk import _core as hcore

    import eagle.exec as eexec
    from eagle import plan as eplan

    lib = hcore.HostLibrary(str(so))
    cdll = ctypes.CDLL(str(so))
    plugin = SimpleNamespace(
        host_entry=ctypes.cast(getattr(cdll, entry_name), ctypes.c_void_p).value,
        name=name, _keepalive=(lib, cdll, *extra_keepalive), **view)
    return eplan.plan(plugin, structure=eexec.HostTeam)


def finishing_host_plan(kern, work, *, cond, kind=None, finishing=True, count=True):
    """``(plan, real)``: the host plan of ``kern`` with a ``finish`` declaration
    (``finishing=False``: the plain plan). ``real`` says hawk emitted the finish
    itself; otherwise ``cond`` (C++ over the TU's names, e.g.
    ``"mut_k[i].get()() >= psc_nstop[i].eval()"``) is the patched epilogue's
    finish condition."""
    from hawk.artifact import build_bundle, plan_view

    kw = {} if kind is None else {"kind": kind}
    bundle = build_bundle([kern], work, targets=("host",), **kw)
    art = next(a for a in bundle.artifacts if a.name == kern.name)
    sidecar, where = art.sidecar, Path(art.directory)
    view = plan_view(sidecar)
    entry = view.pop("host_entry")
    real = "finish" in sidecar
    if not finishing or real:
        if real:
            view["sidecar"] = sidecar
        return _load_host(where / f"{kern.name}.so", entry, view, kern.name), real

    from hawk.compile.drivers import HOST, CompileOptions, compile_source

    arg_spec = tuple(tuple(p) for p in sidecar["arg_spec"])
    mask_slot = arg_spec.index(("terminated", _FINISH["mask"]))
    src = _patch_finish((where / f"{kern.name}.cpp").read_text(), cond=cond,
                        mask_slot=mask_slot, counter_slot=len(arg_spec), count=count)
    import os

    res = compile_source(src, f"{kern.name}_finishing{'' if count else '_nocount'}",
                         CompileOptions(backend=HOST,
                                        cache_dir=os.environ.get("HAWK_CACHE_DIR")))
    view["arg_spec"] = arg_spec + (("lookup", _FINISH["counter"]),)
    view["sidecar"] = dict(sidecar, arg_spec=[list(p) for p in view["arg_spec"]],
                           terminated_readonly=False, finish=dict(_FINISH))
    return _load_host(res.artifact, entry, view, kern.name), False


# --------------------------------------------------------------------------- #
# Kernels and plans
# --------------------------------------------------------------------------- #
_DT = 0.01
_COND = "mut_k[i].get()() >= psc_nstop[i].eval()"
_PLANES = ("x", "v", "k")


def _oscillator(kind=None, finishing=False):
    """One RK4 step of the damped oscillator; ``k`` counts steps and the
    sample finishes when ``k >= nstop`` (``min(1, nstop)`` is exactly 1: it
    keeps ``nstop`` a bound plane of the non-finishing kernel too).
    ``finishing`` spells the finish in the kernel (hawk with I-A only)."""
    hawk = pytest.importorskip("hawk")
    from hawk import Mutable, Param, Scalar, Terminated
    from hawk import math as m

    def rk4(omega, zeta, dt, x0, v0):
        w2 = omega * omega
        c = 2.0 * zeta * omega
        h2 = 0.5 * dt
        h6 = dt * 0.16666666666666666
        a1 = -w2 * x0 - c * v0
        x2 = x0 + h2 * v0
        v2 = v0 + h2 * a1
        a2 = -w2 * x2 - c * v2
        x3 = x0 + h2 * v2
        v3 = v0 + h2 * a2
        a3 = -w2 * x3 - c * v3
        x4 = x0 + dt * v3
        v4 = v0 + dt * a3
        a4 = -w2 * x4 - c * v4
        return (x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4),
                v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4))

    if finishing:
        def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                            terminated: Terminated, x: Mutable[Scalar],
                            v: Mutable[Scalar], k: Mutable[Scalar]):
            x1, v1 = rk4(omega, zeta, dt, x, v)
            x = x1
            v = v1
            k1 = k + m.min(1.0, nstop)
            k = k1
            terminated = k1 >= nstop
    else:
        def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                            terminated: Terminated, x: Mutable[Scalar],
                            v: Mutable[Scalar], k: Mutable[Scalar]):
            x1, v1 = rk4(omega, zeta, dt, x, v)
            x = x1
            v = v1
            k = k + m.min(1.0, nstop)

    return hawk.kernel(oscillator_step, kind=kind, steps=1)


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
def plans(tmp_path_factory):
    """``plans[label]``: ``plain`` (the explicit path's non-finishing kernel),
    ``fplain``/``fmap`` (finishing, plain and active-set) and ``nocount`` (a
    finishing kernel whose epilogue marks but never counts; only where the
    epilogue is the fixture's)."""
    pytest.importorskip("hawk")
    from hawk.ext import Guard, Kind

    def work(label):
        return tmp_path_factory.mktemp(label)

    out = {"plain": finishing_host_plan(_oscillator(), work("plain"), cond=_COND,
                                        finishing=False)[0]}
    try:
        out["fplain"], real = finishing_host_plan(
            _oscillator(finishing=True), work("fplain"), cond=_COND)
    except Exception:  # this hawk refuses `terminated = cond` (no I-A)
        real = False
    if not real:
        out["fplain"], _ = finishing_host_plan(_oscillator(), work("fplain2"),
                                               cond=_COND)
        out["nocount"], _ = finishing_host_plan(
            _oscillator(), work("nocount"), cond=_COND, count=False)
    kind = Kind("udh_map", guard=Guard(active_set=True))
    out["fmap"], _ = finishing_host_plan(_oscillator(kind, finishing=real),
                                         work("fmap"), cond=_COND, kind=kind)
    out["real"] = real
    return out


def _inputs(n, distribution, steps, seed=11):
    rng = np.random.default_rng([seed, n, steps, distribution == "spread"])
    d = dict(omega=rng.uniform(1.0, 10.0, n), zeta=rng.uniform(0.01, 0.2, n),
             x=rng.uniform(-1.0, 1.0, n), v=rng.uniform(-1.0, 1.0, n))
    if distribution == "spread":
        lo, hi = np.log(steps / 100), np.log(steps)
        nstop = np.clip(np.ceil(np.exp(rng.uniform(lo, hi, n))), 1, steps)
    else:
        nstop = np.full(n, float(steps))
    d["nstop"] = nstop.astype(np.float64)
    return d


def _planes(inp):
    out = {key: inp[key].copy() for key in ("omega", "zeta", "nstop", "x", "v")}
    out["k"] = np.zeros(inp["x"].shape[0])
    return out


def _explicit(plans, inp, max_steps):
    """Today's explicit path: the non-finishing kernel, the NumPy stop rule
    after each launch, a repeat_while on ``done != total``."""
    from eagle import SkipGuard, repeat_while

    g = _planes(inp)
    n = inp["x"].shape[0]
    term = np.zeros(n, dtype=bool)
    done = np.zeros(1, dtype=np.uint32)
    total = np.array([n], dtype=np.uint32)
    bound = plans["plain"].bind(dt=_DT, terminated=term, **g)

    def step():
        bound.launch()
        new = ~term & (g["k"] >= g["nstop"])
        term[...] |= new
        done[0] += np.uint32(np.count_nonzero(new))

    repeat_while(step, SkipGuard(done, 0, total, 0), max_steps)()
    assert term.all()
    return g


def _serial(plan, max_steps, **planes):
    """The serial oracle: the finishing entry over the whole batch through
    ``HostTeam.run_serial``, one call per step, until every sample is done.
    ``n`` comes off a scalar plane (``x``) when there is one, else off the
    last axis of a vector plane (``s``: shape ``(dim, n)``)."""
    import eagle.exec as eexec

    n = planes["x"].shape[0] if "x" in planes else planes["s"].shape[-1]
    finished = np.zeros(1, dtype=np.uint32)
    bound = plan.bind(terminated=np.zeros(n, dtype=bool),
                      finished_count=finished.view(np.int32), **planes)
    whole = eexec.Partition(0, n, n)
    for _ in range(max_steps):
        if int(finished[0]) == n:
            break
        eexec.HostTeam.run_serial(bound._entry, bound._addrs, whole)
    assert int(finished[0]) == n
    return planes


def _assert_planes_equal(ref, got, names=_PLANES):
    for key in names:
        assert np.array_equal(ref[key], got[key]), key


# --------------------------------------------------------------------------- #
# The loop is the direct one
# --------------------------------------------------------------------------- #
def test_the_host_branch_launches_the_team_directly(plans, monkeypatch):
    import eagle
    import eagle.exec as eexec
    from eagle import _host_loop

    g = _planes(_inputs(1000, "spread", 30))
    runner = eagle.until_done(plans["fplain"], max_steps=30, dt=_DT, **g)
    assert _host_loop._direct(runner.step)
    calls = []
    real = eexec.HostTeam.run
    monkeypatch.setattr(type(eexec.HostTeam), "run",
                        lambda self, *a, **kw: calls.append(a[2]) or real(*a, **kw))
    monkeypatch.setattr(type(runner.loop), "__call__",
                        lambda self: pytest.fail("the host arm ran"))
    rep = runner.run()
    assert rep.done and len(calls) == rep.launches
    assert all((p.base, p.count, p.n_samples) == (0, 1000, 1000) for p in calls)


def test_a_step_it_cannot_see_through_runs_the_host_arm(plans):
    import eagle
    from eagle import _host_loop

    inp = _inputs(800, "spread", 30)
    g = _planes(inp)
    runner = eagle.until_done(plans["fplain"], max_steps=30, dt=_DT, **g)
    bound = runner.step
    runner.step = SimpleNamespace(launch=bound.launch, n=bound.n,
                                  planes=bound.planes)
    assert not _host_loop._direct(runner.step)
    rep = runner.run()
    assert rep.done and rep.launches == runner.loop.iterations()
    _assert_planes_equal(_explicit(plans, inp, 30), g)


# --------------------------------------------------------------------------- #
# The oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distribution", ["spread", "uniform"])
def test_run_until_done_matches_the_explicit_path_and_the_serial_oracle(
        plans, distribution):
    import eagle

    n, steps = 20_000, 200
    inp = _inputs(n, distribution, steps)
    ref = _explicit(plans, inp, steps)
    serial = _serial(plans["fplain"], steps, dt=_DT, **_planes(inp))
    g = _planes(inp)
    runner = eagle.until_done(plans["fplain"], max_steps=steps, dt=_DT, **g)
    rep = runner.run()
    _assert_planes_equal(ref, g)
    _assert_planes_equal(serial, g)
    assert runner.terminated.all() and rep.finished == n and rep.done
    assert np.array_equal(g["k"], inp["nstop"])
    assert rep.launches == int(inp["nstop"].max())
    assert rep.compactions == 0 and rep.reorders == 0 and rep.build_s == 0.0
    again = runner.run()
    assert again.launches == 0 and again.done
    _assert_planes_equal(ref, g)


def _rk78_card():
    """The RK7(8) card module of this checkout, or a skip."""
    pytest.importorskip("hawk")  # the card builds its kernel with hawk
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[2]
            / "benchmarks" / "rk78_card" / "rk78_card.py")
    if not path.is_file():
        pytest.skip("this checkout carries no RK7(8) card")
    spec = importlib.util.spec_from_file_location("eagle_rk78_card_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rk78_attempt_kernel_matches_the_explicit_path_and_the_serial_oracle(
        tmp_path):
    """The oracle, on the RK7(8) card's adaptive attempt kernel: hawk's native
    in-kernel finish (``terminated = t1 >= t_final``, state as one
    ``s: Mutable[Vector[6]]`` plane), built the way the card itself builds
    its host plan (``card._build(..., targets=("host",))``), not a patched
    kernel."""
    import eagle

    card = _rk78_card()
    _, host_plan, _ = card._build(tmp_path / "kernel", targets=("host",))
    n = 2000
    inp = card._inputs(n)

    def fresh():
        g = {"s": np.stack([inp[k] for k in card.STATE_NAMES])}
        g["t"] = np.zeros(n)
        g["h"] = np.full(n, card.H0)
        g["n_acc"] = np.zeros(n)
        g["n_rej"] = np.zeros(n)
        return g

    def expand(g):
        # the card's own "s" (6, n) -> STATE_NAMES unpacking (_CpuArm.download)
        out = {k: g["s"][d].copy() for d, k in enumerate(card.STATE_NAMES)}
        out.update({k: g[k].copy() for k in card.PLANE_NAMES[6:]})
        return out

    arm = card._CpuArm(host_plan, n)    # the card's explicit eagle_cpu arm
    arm.upload(inp)
    arm.run()
    ref = arm.download()
    serial = expand(_serial(host_plan, card.MAX_ATTEMPTS, t_final=card.T_FINAL,
                            **fresh()))
    g = fresh()
    rep = eagle.run_until_done(host_plan, max_steps=card.MAX_ATTEMPTS,
                               t_final=card.T_FINAL, **g)
    assert rep.done and rep.finished == n
    _assert_planes_equal(ref, expand(g), card.PLANE_NAMES)
    _assert_planes_equal(serial, expand(g), card.PLANE_NAMES)
    attempts = int((g["n_acc"] + g["n_rej"]).max())
    if set(rep.launches_by_k) != {1}:
        # the card's plain decorator builds the automatic kernel: the policy
        # picks k per launch, and the steps run cover the longest sample's
        # attempts, exceeding them by less than one launch
        assert sum(k * c for k, c in rep.launches_by_k.items()) == rep.steps
        assert attempts <= rep.steps < attempts + max(rep.launches_by_k)
        assert rep.launches < attempts
    else:
        # one attempt of every live sample per launch: the longest sample's
        # attempt count is the loop's
        assert rep.launches == attempts


# --------------------------------------------------------------------------- #
# Non-vacuity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label, every", [("fplain", None), ("fmap", 8)])
def test_launches_follow_the_longest_sample(plans, label, every):
    import eagle

    n, s_max, max_steps = 5000, 60, 100_000
    inp = _inputs(n, "spread", s_max)
    s = int(inp["nstop"].max())
    g = _planes(inp)
    kw = {} if every is None else {"every": every}
    runner = eagle.until_done(plans[label], max_steps=max_steps, dt=_DT, **g, **kw)
    rep = runner.run()
    e = every or 1
    assert rep.finished == n and rep.done
    assert rep.launches * e <= s + e and rep.launches == -(-s // e)
    assert runner.loop.iterations() == rep.launches
    assert rep.steps == rep.launches * e
    assert runner.loop.max_iters == -(-max_steps // e)


def test_an_epilogue_that_never_counts_runs_to_the_cap(plans):
    """Red case: the epilogue marks but does not count, so the guard never fires."""
    import eagle

    if "nocount" not in plans:
        pytest.skip("the epilogue red is hawk's (I-A) once kernels finish natively")
    n, cap = 500, 60
    g = _planes(_inputs(n, "spread", 20))
    rep = eagle.run_until_done(plans["nocount"], max_steps=cap, dt=_DT, **g)
    assert rep.launches == cap and not rep.done and rep.finished == 0


# --------------------------------------------------------------------------- #
# Monotone set/reset (#36)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", ["fplain", "fmap"])
def test_preset_samples_are_untouched_then_reset_runs_them(plans, label):
    import eagle

    n, steps = 3000, 50
    inp = _inputs(n, "spread", steps)
    g = _planes(inp)
    preset = np.zeros(n, dtype=bool)
    preset[::7] = True
    term = preset.copy()
    runner = eagle.until_done(plans[label], max_steps=steps, dt=_DT,
                              terminated=term, **g)
    rep = runner.run()
    assert rep.done and rep.finished == n and term.all()
    for key in ("x", "v"):
        assert np.array_equal(g[key][preset], inp[key][preset]), key
    assert not g["k"][preset].any()
    assert np.array_equal(g["k"][~preset], inp["nstop"][~preset])
    runner.reset()
    assert not term.any() and int(runner.finished[0]) == 0
    rep2 = runner.run()
    assert rep2.done and rep2.finished == n
    assert np.array_equal(g["k"][preset], inp["nstop"][preset])


# --------------------------------------------------------------------------- #
# Compaction + reorder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distribution", ["spread", "uniform"])
def test_the_active_set_artifact_compacts_through_the_same_call(plans,
                                                                distribution):
    import eagle

    n, steps = 20_000, 200
    inp = _inputs(n, distribution, steps)
    ref = _explicit(plans, inp, steps)
    g = _planes(inp)
    runner = eagle.until_done(plans["fmap"], max_steps=steps, every=4, dt=_DT, **g)
    assert runner.active is not None and runner.active.theta is None
    rep = runner.run()
    _assert_planes_equal(ref, g)
    assert rep.done and rep.finished == n and runner.terminated.all()
    if distribution == "spread":
        assert rep.compactions >= 1 and runner.active.live < n
    assert rep.steps == rep.launches * 4


def test_reorder_runs_and_restores_sample_order(plans):
    import eagle
    from eagle._active_set import MIN_REORDER_SPAN

    n, steps = max(MIN_REORDER_SPAN, 100_000), 200
    inp = _inputs(n, "spread", steps)
    ref = _explicit(plans, inp, steps)
    g = _planes(inp)
    runner = eagle.until_done(plans["fmap"], max_steps=steps, reorder=0.5,
                              every=4, dt=_DT, **g)
    rep = runner.run()
    assert rep.reorders >= 1 and rep.compactions >= 1 and rep.done
    a = runner.active
    ident = np.arange(n, dtype=np.int32)
    assert np.array_equal(a.perm[a.inv], ident)
    assert np.array_equal(a.in_sample_order(g["x"]), g["x"])
    _assert_planes_equal(ref, g)
