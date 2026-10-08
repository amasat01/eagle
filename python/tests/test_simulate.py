# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.simulate``: the problem-level door over ``until_done`` and the
composed multi-kernel loop.

What each group of rows proves:

* ONE KERNEL: ``simulate(kernel)`` is ``run_until_done(deploy(kernel))``:
  every plane bit-equal on a spread batch (host and device), the reports
  equal field by field, and ``sim.runner`` the very runner (its loop a
  ``RepeatWhile``). Catches a second loop.
* PIPELINE (the non-vacuity pair): tutorial 4's three kernels through
  ``simulate([...])`` bit-equal to tutorial 4's hand-composed cell (copied
  below), with equal iteration and settled counts and ``energy`` allocated;
  the same call with the ``diagnostic`` launch dropped from the step must
  turn the comparison RED.
* ``until=``: ``simulate([propagate, diagnostic], until=event)`` equals the
  pipeline case above.
* REFUSALS: the exact texts, each in the caller's words.
* ONE SAMPLE: floats in, head-shaped arrays out, equal to row ``i`` of a
  pairwise-distinct batch bit for bit on the host and different from row
  ``j``.
* RESIDENCY: numpy runs the ``HostTeam`` plan, cupy the ``DeviceKernel``
  plan, a mix is refused.
* CAP: never-finishing samples end at ``max_steps`` as a result; a
  compacting several-kernel step checks the cap once per ``every`` steps.
* SEAM: the door is Python above the runner: the kernel's emitted source
  and sidecar are byte-equal to hawk's own build, and the door's module
  loads no native code.
* EASE: the quickstart's lead example (at most 12 lines after the kernel)
  runs as written, and tutorial 4's explicit composed cell is still there,
  unchanged.
* The call shape: the kernel's own arguments bind exactly like a call to the
  kernel function -- positionally, by name, or mixed, to the same result as
  a fully-keyword call -- and a missing / unknown / duplicated argument, a
  passed ``Terminated``, a kernel parameter colliding with one of eagle's
  own options, and the removed ``state=``/``params=`` keywords each raise a
  message naming the kernel and the parameter.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import replace

import numpy as np
import pytest

import eagle
import eagle.exec as eexec
from eagle import ActiveSet, SkipGuard, compaction_body, repeat_while

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated, Vector  # noqa: E402
from hawk import math as hm  # noqa: E402
from hawk.ext import Guard, Kind  # noqa: E402
import eagle._plan_deploy as plan_deploy  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "content" / "userguide"
TUTORIAL_04 = DOCS / "tutorials" / "04_multi_kernel_workflow.ipynb"
QUICKSTART = DOCS / "quickstart_python.ipynb"

_TARGETS = ["host", pytest.param("device", marks=pytest.mark.gpu)]


def _xp(target):
    if target == "device":
        return pytest.importorskip("cupy")
    return np


def _host(a):
    get = getattr(a, "get", None)
    return np.asarray(get() if get is not None else a)


def _bits_equal(a, b):
    a, b = _host(a), _host(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@hawk.kernel
def sim_osc(omega: Scalar, t_end: Scalar, dt: Param, terminated: Terminated,
            x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    x0, v0 = x, v
    x = x0 + dt * v0
    v = v0 - dt * omega * omega * x0
    t1 = t + dt
    t = t1
    terminated = t1 >= t_end


@hawk.kernel
def sim_never(dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x1 = x + dt
    x = x1
    terminated = x1 < -1.0e300


def propagate(omega: Param, zeta: Param, dt: Param, terminated: Terminated,
              x: Mutable[Scalar], v: Mutable[Scalar]):
    """One semi-implicit Euler step of a damped harmonic oscillator."""
    x0, v0 = x, v
    a = -(omega * omega) * x0 - 2.0 * zeta * omega * v0
    v_new = v0 + dt * a
    v = v_new
    x = x0 + dt * v_new


def diagnostic(x: Scalar, v: Scalar, omega: Param, terminated: Terminated,
               energy: Mutable[Scalar]):
    """The sample's mechanical energy -- frozen once it settles."""
    energy = 0.5 * v * v + 0.5 * omega * omega * x * x


def event(energy: Scalar, eps: Param, terminated: Terminated):
    """The settling event: terminate once the energy decays below eps."""
    terminated = energy < eps


def diagnostic_vec(x: Vector[3], terminated: Terminated, energy: Mutable[Scalar]):
    energy = hm.dot(x, x)


def diagnostic_param(x: Param, terminated: Terminated, energy: Mutable[Scalar]):
    energy = x


STEP = Kind("ensemble_step", guard=Guard(active_set=True))
K_PROP, K_DIAG, K_EVENT = (hawk.kernel(fn, kind=STEP)
                           for fn in (propagate, diagnostic, event))
K_DIAG_PLAIN = hawk.kernel(diagnostic)


# --------------------------------------------------------------------------- #
# One kernel is run_until_done
# --------------------------------------------------------------------------- #
def _osc_batch(xp, n=1500):
    rng = np.random.default_rng(11)
    omega = rng.uniform(1.0, 3.0, n)
    t_end = rng.uniform(0.05, 1.0, n)          # spread: samples finish apart
    x0 = rng.uniform(-1.0, 1.0, n)
    return {k: xp.asarray(a) for k, a in
            dict(omega=omega, t_end=t_end, x=x0, v=np.zeros(n), t=np.zeros(n)).items()}


@pytest.mark.parametrize("target", _TARGETS)
def test_one_kernel_is_run_until_done(target):
    xp = _xp(target)
    a, b = _osc_batch(xp), _osc_batch(xp)
    sim = eagle.simulation(sim_osc, omega=a["omega"], t_end=a["t_end"], dt=1e-3,
                           x=a["x"], v=a["v"], t=a["t"], max_steps=5000)
    result = sim.run()
    mask = xp.zeros(b["x"].shape[-1], dtype=xp.bool_)
    report = eagle.run_until_done(eagle.deploy(sim_osc), max_steps=5000, dt=1e-3,
                                  terminated=mask, **b)
    for name in ("x", "v", "t"):
        _bits_equal(result[name], b[name])
        assert result[name] is a[name]           # the caller's array, in place
    _bits_equal(result.finished, mask)
    # every field but the one-time build wall (a timing)
    assert replace(result.report, build_s=0.0) == replace(report, build_s=0.0)
    assert isinstance(sim.runner, eagle.Runner)
    if sim.runner._fast_mode is None:
        # the device no-graph fast mode has no
        # loop at all -- one launch IS the run -- which only applies when
        # this batch fits the launch mode's own threshold; above it (or on
        # host, where fast mode never applies) the WHILE loop is still here.
        assert isinstance(sim.runner.loop, eagle.RepeatWhile)
        assert sim.loop is sim.runner.loop
    else:
        assert sim.runner.loop is None
    assert result.status == "finished" and result.done and result.steps == report.steps
    assert result.allocated == ()


# --------------------------------------------------------------------------- #
# The pipeline is tutorial 4's loop (the hand cell is copied verbatim)
# --------------------------------------------------------------------------- #
N2 = 2000
OMEGA, ZETA, DT, EPS = 3.0, 0.15, 0.01, 0.01


def _x0(xp):
    x0 = np.linspace(-1.0, 1.0, N2 + 2)[1:-1]       # pairwise distinct, no zero
    return xp.asarray(np.random.default_rng(3).permutation(x0))


def _hand_composed(xp):
    """Tutorial 4's composed path, cell for cell."""
    n = N2
    omega, zeta, dt, eps = OMEGA, ZETA, DT, EPS
    propagate_plan, diagnostic_plan, event_plan = eagle.deploy([K_PROP, K_DIAG, K_EVENT])  # noqa: E501

    x = _x0(xp)
    v = xp.zeros(n)
    energy = xp.zeros(n)
    terminated = xp.zeros(n, dtype=xp.bool_)
    finished = xp.zeros(1, dtype=xp.uint32)
    total = xp.asarray([n], dtype=xp.uint32)

    active = ActiveSet(terminated)
    propagate_run = propagate_plan.bind(omega=omega, zeta=zeta, dt=dt, terminated=terminated,  # noqa: E501
                                        x=x, v=v, **active.planes())
    diagnostic_run = diagnostic_plan.bind(x=x, v=v, omega=omega, terminated=terminated,
                                          energy=energy, **active.planes())
    event_run = event_plan.bind(energy=energy, eps=eps, terminated=terminated,
                                finished_count=finished, **active.planes())

    def one_step():
        propagate_run.launch()
        diagnostic_run.launch()
        event_run.launch()

    every, max_steps = 16, 600
    body = compaction_body(one_step, active, every=every, finished=finished)
    loop = repeat_while(body, SkipGuard(finished, 0, total, 0), -(-max_steps // every))

    if xp is not np:
        eagle.GraphPipeline().add(loop).build().launch()
        xp.cuda.Device().synchronize()
    else:
        loop()
    return dict(x=x, v=v, energy=energy, terminated=terminated,
                iterations=loop.iterations(), settled=n - active.live)


def _a2_call(xp, *, until=False):
    model = [K_PROP, K_DIAG] if until else [K_PROP, K_DIAG, K_EVENT]
    sim = eagle.simulation(model, until=K_EVENT if until else None,
                           omega=OMEGA, zeta=ZETA, dt=DT, eps=EPS,
                           x=_x0(xp), v=xp.zeros(N2),
                           max_steps=600, every=16)
    return sim, sim.run()


def _check_a2(hand, sim, result):
    for name in ("x", "v", "energy"):
        _bits_equal(result[name], hand[name])
    _bits_equal(result.finished, hand["terminated"])
    assert sim.loop.iterations() == hand["iterations"]
    assert result.report.launches == hand["iterations"]
    assert int(_host(result.finished).sum()) == hand["settled"]
    assert result.allocated == ("energy",)


@pytest.mark.parametrize("target", _TARGETS)
def test_pipeline_is_tutorial_04(target):
    xp = _xp(target)
    hand = _hand_composed(xp)
    sim, result = _a2_call(xp)
    _check_a2(hand, sim, result)
    assert isinstance(sim.runner, eagle.Runner) and sim.loop is sim.runner.loop
    assert isinstance(sim.runner.step, tuple) and len(sim.runner.step) == 3
    assert 0 < hand["settled"] <= N2
    assert result.report.steps == 16 * hand["iterations"]
    assert result.report.launches_by_k == {1: result.report.steps}


@pytest.mark.parametrize("target", _TARGETS)
def test_mutant_without_diagnostic_turns_red(target, monkeypatch):
    """The same call with the diagnostic launch dropped from the step: energy
    is never written (it stays 0), so the comparison with the hand cell must
    fail."""
    import eagle._simulate as door

    xp = _xp(target)
    hand = _hand_composed(xp)
    real = door.until_done

    def drop_diagnostic(plans, **kw):
        keep = [p for p in plans if p.plugin.name != K_DIAG.name]
        names = {nm for p in keep for _role, nm in p.plugin.arg_spec}
        return real(keep, **{k: v for k, v in kw.items()
                             if k in names or k in ("max_steps", "every", "reorder")})

    monkeypatch.setattr(door, "until_done", drop_diagnostic)
    sim, result = _a2_call(xp)
    assert not _host(result.energy).any()        # energy stays 0
    with pytest.raises(AssertionError):
        _check_a2(hand, sim, result)


# --------------------------------------------------------------------------- #
# until= is the last kernel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", _TARGETS)
def test_until_places_the_finishing_kernel_last(target):
    xp = _xp(target)
    hand = _hand_composed(xp)
    sim, result = _a2_call(xp, until=True)
    _check_a2(hand, sim, result)


# --------------------------------------------------------------------------- #
# Refusals, in the caller's words
# --------------------------------------------------------------------------- #
def _osc_kw(n=8, **over):
    kw = dict(omega=np.full(n, 2.0), t_end=np.full(n, 0.1), dt=1e-2,
              x=np.ones(n), v=np.zeros(n), t=np.zeros(n), max_steps=100)
    kw.update(over)
    return kw


def _refused(text, model, *args, **kw):
    with pytest.raises(ValueError, match=re.escape(f"eagle.simulate: {text}")):
        eagle.simulate(model, *args, **kw)


def test_array_for_a_param():
    kw = _osc_kw(dt=np.full(8, 1e-2))
    _refused("'dt' is declared Param (one value shared by all samples) but you passed "
             "8 values; declare it `dt: Scalar` to give each sample its own",
             sim_osc, **kw)


def test_number_for_a_per_sample_plane():
    kw = _osc_kw(n=1000, omega=3.0)
    _refused("'omega' needs one value per sample (n = 1000): np.full(n, 3.0), or "
             "declare it `omega: Param` to share one value", sim_osc, **kw)


@pytest.mark.parametrize("name", ["finished_count", "active_map", "active_count",
                                  "fused_steps"])
def test_reserved_names(name):
    kw = _osc_kw(**{name: np.zeros(8)})
    _refused(f"{name!r} is allocated and bound by eagle.simulate (the door owns it); "
             "drop it from the call", sim_osc, **kw)


def test_no_finishing_kernel():
    _refused("no kernel of the model finishes its samples, so nothing would stop them; "
             "add the stop rule to a kernel (terminated = t >= t_end in "
             f"{K_DIAG.name}, say) or pass the kernel that carries it: "
             "eagle.simulate(model, until=event, ...)",
             [K_PROP, K_DIAG], x=np.ones(4), v=np.zeros(4),
             omega=1.0, zeta=0.1, dt=0.1, max_steps=10)


@pytest.mark.parametrize("fn, text", [
    (diagnostic_vec, "'x' is 1 wide (float64) in {a} but 3 wide (float64) in {b}"),
    (diagnostic_param, "'x' is declared per-sample in {a} but Param in {b}"),
])
def test_two_declarations_of_one_name(fn, text):
    other = hawk.kernel(fn, kind=STEP)
    text = text.format(a=K_PROP.name, b=other.name)
    _refused(text + "; one name is one plane across the kernels of a step: declare it "
             "the same way in both", [K_PROP, other, K_EVENT],
             x=np.ones(4), v=np.zeros(4),
             omega=1.0, zeta=0.1, dt=0.1, eps=0.1, max_steps=10)


def test_differing_guards():
    _refused(f"{K_PROP.name} reads the active set (Guard(active_set=True)) but "
             f"{K_DIAG_PLAIN.name} does not; the kernels of one step share one guard: "
             "build them under one Kind, as eagle's tutorial 4 does",
             [K_PROP, K_DIAG_PLAIN, K_EVENT], x=np.ones(4), v=np.zeros(4),
             omega=1.0, zeta=0.1, dt=0.1, eps=0.1, max_steps=10)


@pytest.mark.parametrize("until, what", [
    ("settled", "the string 'settled'"),
    (1.0, "a float"),
    (lambda s: s.t >= 1.0, "a Python function"),
])
def test_until_is_a_kernel(until, what):
    _refused(f"until= is the kernel that finishes your samples, not {what}. A stop "
             "rule runs on the device, once per sample, so it is a line of a kernel: "
             "terminated = t >= t_end; pass that kernel as until= (or as the last "
             "kernel of the model)", sim_osc, until=until, **_osc_kw())


def test_state_plane_named_like_a_result_field():
    @hawk.kernel
    def sim_counted(dt: Param, terminated: Terminated, steps: Mutable[Scalar]):
        s1 = steps + dt
        steps = s1
        terminated = s1 >= 1.0

    _refused("the state plane 'steps' would shadow the result's own field of that "
             "name (the result's fields: state, finished, done, steps, status, report, "
             "n, wall_s, allocated); rename the plane in the kernel",
             sim_counted, steps=np.zeros(4), dt=0.5, max_steps=10)


def test_prior_read_state_needs_a_value():
    kw = _osc_kw()
    del kw["x"]
    _refused("'x' is updated from its previous value, so it needs an initial value: "
             "pass x=...", sim_osc, **kw)


# --------------------------------------------------------------------------- #
# One sample is one row of the batch
# --------------------------------------------------------------------------- #
def test_one_sample_is_a_batch_row():
    omegas = np.array([1.1, 1.7, 2.3, 2.9, 1.3, 2.1, 2.7])
    t_ends = np.array([0.31, 0.47, 0.23, 0.59, 0.41, 0.37, 0.53])
    x0s = np.array([0.9, -0.4, 0.7, -0.8, 0.2, 0.6, -0.3])
    batch = eagle.simulate(sim_osc, omega=omegas, t_end=t_ends, dt=1e-3,
                           x=x0s.copy(), v=np.zeros(7), t=np.zeros(7),
                           max_steps=10_000)
    i = 3
    one = eagle.simulate(sim_osc, omega=float(omegas[i]), t_end=float(t_ends[i]),
                         dt=1e-3, x=float(x0s[i]), v=0.0, t=0.0, max_steps=10_000)
    for name in ("x", "v", "t"):
        assert one[name].shape == () and isinstance(one[name], np.ndarray)
        _bits_equal(one[name], batch[name][i])
        assert all(_host(one[name]).tobytes() != _host(batch[name][j]).tobytes()
                   for j in range(7) if j != i)
    assert one.finished.shape == () and bool(one.finished)
    assert one.done and one.n == 1 and isinstance(float(one.x), float)


# --------------------------------------------------------------------------- #
# Residency picks the plan
# --------------------------------------------------------------------------- #
def test_numpy_runs_on_the_host_team():
    sim = eagle.simulation(sim_osc, **_osc_kw())
    assert sim.runner.step.plan.structure is eexec.HostTeam


@pytest.mark.gpu
def test_cupy_runs_on_the_device_and_a_mix_is_refused():
    cp = pytest.importorskip("cupy")
    kw = {k: (cp.asarray(v) if hasattr(v, "shape") else v)
          for k, v in _osc_kw().items()}
    sim = eagle.simulation(sim_osc, **kw)
    assert sim.runner.step.plan.structure is eexec.DeviceKernel
    mixed = _osc_kw()
    mixed["x"] = cp.asarray(mixed["x"])
    _refused("'x' is on the device but 'omega' is on the host; the data decides where the "
             "model runs, and it runs in one place: move one side with cp.asarray / "
             ".get()", sim_osc, **mixed)


# --------------------------------------------------------------------------- #
# The cap is a result
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", _TARGETS)
def test_cap_is_a_result(target):
    xp = _xp(target)
    result = eagle.simulate(sim_never, x=xp.zeros(50), dt=1.0, max_steps=37)
    assert result.status == "max_steps"
    assert result.done is False
    assert bool(_host(result.finished).any()) is False
    assert result.steps == 37 and result.report.exact_steps
    np.testing.assert_array_equal(_host(result.x), np.full(50, 37.0))


@pytest.mark.parametrize("target", _TARGETS)
def test_several_kernels_check_the_cap_once_per_compaction(target):
    """A compacting several-kernel step checks the cap once per ``every``
    steps, as the docs say: ``max_steps=20`` with ``every=16`` runs two rounds
    (32 steps) and the report counts every one of them."""
    xp = _xp(target)
    result = eagle.simulate([K_PROP, K_DIAG, K_EVENT],
                            omega=OMEGA, zeta=ZETA, dt=DT, eps=EPS,
                            x=_x0(xp), v=xp.zeros(N2),
                            max_steps=20, every=16)
    assert result.status == "max_steps"
    assert result.report.launches == 2 and result.steps == 32


# --------------------------------------------------------------------------- #
# The door is Python above the runner
# --------------------------------------------------------------------------- #
def _artifact_digests(directory: pathlib.Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir())
            if p.suffix in (".cu", ".cpp", ".json") and p.name != "manifest.json"}


def test_the_artifact_is_hawks_own(tmp_path):
    from hawk.artifact import build_bundle
    from hawk.compile import default_cache_dir

    from eagle import plan as eplan

    eagle.simulate(sim_osc, **_osc_kw())
    from hawk.emit import FAST_ENTRIES_MACRO

    targets = plan_deploy._eager_targets()
    # the define eagle builds an automatic kernel with (its fast entries)
    defines = ((f"{FAST_ENTRIES_MACRO}=1",)
               if plan_deploy._wants_fast_entries([sim_osc], targets) else ())
    used = default_cache_dir() / "bundles" / plan_deploy._bundle_key([sim_osc], targets,
                                                                     defines)
    ref = tmp_path / "ref"
    build_bundle([sim_osc], ref, targets=targets, defines=defines)
    got, want = _artifact_digests(used), _artifact_digests(ref)
    assert want and got == want


def test_the_door_loads_no_native_code():
    import eagle._simulate as door

    source = pathlib.Path(door.__file__).read_text()
    for needle in ("ctypes", "_core", "_backend", "eagle_backend_", "open(", "sidecar"):
        assert needle not in source, needle


# --------------------------------------------------------------------------- #
# The lead example and tutorial 4's explicit cell
# --------------------------------------------------------------------------- #
def _cells(path):
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]
            if c["cell_type"] == "code"]


def _lead_example():
    cells = [c for c in _cells(QUICKSTART) if "eagle.simulate(" in c]
    assert len(cells) == 1, "the quickstart carries one simulate example"
    return cells[0]


def test_lead_example_is_short():
    source = _lead_example()
    lines = source.splitlines()
    body_start = max(i for i, ln in enumerate(lines) if "terminated = " in ln)
    after = [ln for ln in lines[body_start + 1:] if ln.strip()]
    assert 0 < len(after) <= 12, after


@pytest.mark.gpu
def test_lead_example_runs_as_written(tmp_path):
    import runpy

    pytest.importorskip("cupy")
    # a file of its own: hawk reads the kernel's source back to trace it
    script = tmp_path / "lead_example.py"
    # The quickstart opens with the family's device switch (DEVICE, xp); the
    # example runs after it, exactly as the notebook does.
    switch = [c for c in _cells(QUICKSTART) if "DEVICE = " in c]
    assert len(switch) == 1, "the quickstart opens with one device-switch cell"
    shared = DOCS.parent / "_shared"
    header = (f"import sys\nsys.path.insert(0, {str(shared)!r})\n"
              "from nb_helpers import gpu_available, xp_for\n")
    script.write_text(header + switch[0] + "\n\n" + _lead_example() + "\n")
    ns = runpy.run_path(str(script))
    result = ns["result"]
    assert result.done and result.status == "finished" and result.n == ns["n"]


def test_tutorial_04_keeps_its_explicit_cell():
    """The hand-built half of tutorial 4 still writes `one_step`, the
    compaction cadence and the bounded loop out itself, across their own
    cells now -- it is not hidden behind one convenience call."""
    cells = _cells(TUTORIAL_04)
    loop_cells = [c for c in cells
                 if "loop = repeat_while(body, SkipGuard(finished, 0, total, 0), "
                 "-(-max_steps // every))" in c]
    assert len(loop_cells) == 1
    assert any("def one_step():" in c for c in cells)
    assert any("every, max_steps = 16, 600" in c for c in cells)


def test_simulate_works_in_a_fresh_process_that_imported_nothing_else(tmp_path):
    """The first call into eagle a user makes is often ``eagle.simulate``:
    nothing has imported ``eagle.plan`` yet, so the deploy module loads
    first. That order must not hit an import cycle."""
    import subprocess
    import sys

    script = tmp_path / "fresh.py"
    script.write_text(
        "import numpy as np, eagle, hawk\n"
        "from hawk import Mutable, Param, Scalar, Terminated\n"
        "@hawk.kernel\n"
        "def tick(t_end: Param, dt: Param, terminated: Terminated,\n"
        "         t: Mutable[Scalar]):\n"
        "    t_start = t\n"
        "    t = t_start + dt\n"
        "    terminated = t_start + dt >= t_end\n"
        "r = eagle.simulate(tick, t=np.zeros(3), t_end=0.004, dt=1e-3, max_steps=50)\n"
        "print(r.status)\n")
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("finished")


# --------------------------------------------------------------------------- #
# The call shape: bind like a call to the kernel
# --------------------------------------------------------------------------- #
def test_call_shape_positional_equals_keyword():
    kw = _osc_kw(n=8)
    keyword = eagle.simulate(sim_osc, omega=kw["omega"], t_end=kw["t_end"], dt=kw["dt"],
                             x=kw["x"].copy(), v=kw["v"].copy(), t=kw["t"].copy(),
                             max_steps=kw["max_steps"])
    positional = eagle.simulate(sim_osc, kw["omega"], kw["t_end"], kw["dt"],
                                kw["x"].copy(), kw["v"].copy(), kw["t"].copy(),
                                max_steps=kw["max_steps"])
    for name in ("x", "v", "t"):
        _bits_equal(keyword[name], positional[name])


def test_call_shape_mixed_equals_keyword():
    kw = _osc_kw(n=8)
    keyword = eagle.simulate(sim_osc, omega=kw["omega"], t_end=kw["t_end"], dt=kw["dt"],
                             x=kw["x"].copy(), v=kw["v"].copy(), t=kw["t"].copy(),
                             max_steps=kw["max_steps"])
    mixed = eagle.simulate(sim_osc, kw["omega"], kw["t_end"], dt=kw["dt"],
                           x=kw["x"].copy(), v=kw["v"].copy(), t=kw["t"].copy(),
                           max_steps=kw["max_steps"])
    for name in ("x", "v", "t"):
        _bits_equal(keyword[name], mixed[name])


def test_call_shape_missing_argument():
    kw = _osc_kw()
    del kw["dt"]
    _refused(f"{sim_osc.name}() missing a required argument: 'dt'", sim_osc, **kw)


def test_call_shape_unknown_argument():
    kw = _osc_kw(zzz=1.0)
    _refused(f"{sim_osc.name}() got an unexpected keyword argument 'zzz'",
             sim_osc, **kw)


def test_call_shape_duplicate_argument():
    kw = _osc_kw()
    omega, t_end, dt = kw.pop("omega"), kw.pop("t_end"), kw.pop("dt")
    _refused(f"{sim_osc.name}() multiple values for argument 'omega'",
             sim_osc, omega, t_end, dt, omega=omega, **kw)


def test_call_shape_too_many_positional_arguments():
    kw = _osc_kw()
    args = [kw.pop(k) for k in ("omega", "t_end", "dt", "x", "v", "t")]
    _refused(f"{sim_osc.name}() too many positional arguments",
             sim_osc, *args, 999.0, **kw)


def test_call_shape_terminated_is_never_passed():
    kw = _osc_kw(terminated=np.zeros(8, dtype=bool))
    _refused("'terminated' is the Terminated mask; every sample starts running, so "
             "eagle.simulate drives it itself -- it is never passed by the caller",
             sim_osc, **kw)


def test_call_shape_option_name_collision():
    @hawk.kernel
    def sim_bad(max_steps: Param, terminated: Terminated, x: Mutable[Scalar]):
        x1 = x + max_steps
        x = x1
        terminated = x1 >= 1.0

    _refused(f"{sim_bad.name} declares a parameter named 'max_steps', which is "
             "eagle.simulate's own max_steps= option; rename the kernel's "
             "parameter 'max_steps'",
             sim_bad, x=np.zeros(4), max_steps=10)


def test_call_shape_old_keywords_refused():
    order = ", ".join(f"{nm}=..." for nm in
                      ("omega", "t_end", "dt", "x", "v", "t"))
    prefix = ("state=/params= were removed: call it the way you would call the "
             f"kernel itself, e.g. eagle.simulate({sim_osc.name}, {order}, "
             "max_steps=...)")
    kw = _osc_kw()
    _refused(prefix, sim_osc, state=dict(x=kw["x"]), **{k: v for k, v in kw.items()
                                                        if k != "x"})
    _refused(prefix, sim_osc, params=dict(dt=kw["dt"]), **{k: v for k, v in kw.items()
                                                           if k != "dt"})


@pytest.mark.parametrize("target", ["host", pytest.param("device", marks=pytest.mark.gpu)])
def test_scalar_type_float32_runs_end_to_end(target):
    """``scalar_type="float32"`` builds the kernel in single precision: the
    run finishes, the state stays float32 and tracks the float64 run within
    single-precision error."""
    xp = np if target == "host" else pytest.importorskip("cupy")
    n = 64
    res = {}
    for st in ("float64", "float32"):
        omega = xp.asarray(np.linspace(1.0, 3.0, n), dtype=st)
        t_end = xp.asarray(np.linspace(0.1, 0.5, n), dtype=st)
        res[st] = eagle.simulate(sim_osc, omega=omega, t_end=t_end, dt=1e-3,
                                 x=xp.ones(n, dtype=st), v=xp.zeros(n, dtype=st),
                                 t=xp.zeros(n, dtype=st), max_steps=5000,
                                 scalar_type=None if st == "float64" else st)
    single, double = res["float32"], res["float64"]
    assert single.status == "finished" and single.x.dtype == np.float32
    got = single.x.get() if hasattr(single.x, "get") else single.x
    want = double.x.get() if hasattr(double.x, "get") else double.x
    np.testing.assert_allclose(got, want, atol=1e-2)


def test_scalar_type_is_checked():
    """An unknown precision, or one given for an already-built plan, is
    refused naming the fix."""
    with pytest.raises(ValueError, match="scalar_type must be one of"):
        eagle.deploy(sim_osc, scalar_type="float16")
    plan = eagle.deploy(sim_osc)
    with pytest.raises(ValueError, match="scalar_type"):
        eagle.deploy(plan.plugin if hasattr(plan, "plugin") else plan, scalar_type="float32")
    with pytest.raises(ValueError, match="scalar_type"):
        eagle.simulate(plan, omega=1.0, t_end=0.1, dt=1e-3, x=1.0, v=0.0, t=0.0,
                       max_steps=10, scalar_type="float32")
