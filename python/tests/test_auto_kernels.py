# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.deploy`` (the same function as ``eagle.plan.auto``) takes hawk
kernels directly: ``deploy(kernel)`` and ``deploy([k1, k2, k3])``.

What each group of rows proves:

* EQUALITY: ``auto(kernel)``'s host run equals the plan of the plugin
  assembled by hand (``build_bundle`` -> ``plan_view`` -> host library + entry
  address, the recipe every caller used to write) bit for bit; its device run
  equals the hand-assembled device plan bit for bit and is within one ULP of
  the host per multiply-add the device may contract. Inputs are pairwise
  distinct so a wrong row cannot pass.
* ORDER: ``auto([k1, k2, k3])`` is a tuple of plans in the kernels' order,
  all built into ONE bundle.
* CACHE: a second call compiles nothing. In the same process the publisher's
  memo answers (``unit_stats``); in a NEW process (same ``cache_dir``) the
  unit's stamp answers, the compile cache has no miss and no published file's
  mtime moves. Asserted on counters and mtimes, never on timing.
* DERIVED KERNELS: a ``jvp`` kernel and a ``hawk.steps(kernel, K)`` kernel are
  inputs like any other, and ``run_until_done`` takes the plan of the latter.
* SEVERANCE: ``import eagle`` and ``auto(plugin)`` work with hawk masked
  (a subprocess with ``sys.modules["hawk"] = None``); a hawk kernel there is
  refused naming hawk.
* ONE DOOR: ``eagle.deploy is eagle.plan.auto``, and ``import eagle`` imports
  neither ``eagle.plan`` nor hawk until ``eagle.deploy`` is first touched.
* REFUSALS: an empty list, a list mixing a kernel and a plugin, ``targets=``
  on a built plugin, and a side the build lacks (the existing text).
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

import eagle
import eagle.exec as eexec
from eagle import plan as eplan

hawk = pytest.importorskip("hawk")
from hawk import Kernel, Mutable, Param, Scalar, Terminated, Vector  # noqa: E402
from hawk import math as hm  # noqa: E402
from hawk.diff import jvp  # noqa: E402
import eagle._plan_deploy as plan_deploy  # noqa: E402

N = 9


@hawk.kernel
def ak_affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def ak_energy(v: Vector[3], s: Scalar, e: Mutable[Scalar]):
    e = 0.5 * hm.dot(v, v) * s


@hawk.kernel
def ak_osc(omega: Scalar, nstop: Scalar, dt: Param, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], k: Mutable[Scalar]):
    x0 = x
    v0 = v
    x = x0 + dt * v0
    v = v0 - dt * (omega * omega) * x0
    k1 = k + 1.0
    k = k1
    terminated = k1 >= nstop


ak_energy_jvp = Kernel("ak_energy_jvp", jvp(ak_energy, wrt=("v",)))


def _have_gpu():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _hand_plugins(kernels, work, targets=("host",)):
    """The plugins every caller used to assemble by hand, ``{name: plugin}``."""
    from hawk import _core as hcore
    from hawk.artifact import build_bundle, plan_view

    bundle = build_bundle(kernels, work, targets=targets)
    registry = None
    if "cuda" in targets:
        from eagle.registry import load_manifest

        registry = load_manifest(work / "manifest.json")
    out = {}
    for art in bundle.artifacts:
        view = plan_view(art.sidecar)
        so = work / f"{art.name}.so"
        lib, cdll = hcore.HostLibrary(str(so)), ctypes.CDLL(str(so))
        extra = {"host_entry": ctypes.cast(getattr(cdll, view.pop("host_entry")),
                                           ctypes.c_void_p).value}
        keep = [lib, cdll]
        if registry is not None:
            loaded = registry[art.name]
            extra["device_function"] = loaded.fn.kernel.ptr
            keep += [registry, loaded]
        out[art.name] = SimpleNamespace(_keepalive=keep, **extra, **view)
    return out


def _inputs(name):
    rng = np.random.default_rng(21)
    if name == "ak_affine":
        return {"x": rng.normal(size=N), "a": 1.25, "b": -0.5}
    if name == "ak_positive":
        return {"x": rng.uniform(0.5, 2.0, N), "a": 1.25, "b": 0.5}
    if name == "ak_energy":
        return {"v": rng.normal(size=(3, N)), "s": rng.uniform(0.5, 2.0, N)}
    return {"v": rng.normal(size=(3, N)), "s": rng.uniform(0.5, 2.0, N),
            "dot_v": rng.normal(size=(3, N))}


def _bytes(result):
    result = result.values() if isinstance(result, dict) else (result,)
    return [np.asarray(r).tobytes() for r in result]


def _ulp(a, b) -> int:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return int(np.max(np.abs(a.view(np.int64) - b.view(np.int64)))) if a.size else 0


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("auto_cache")


# --------------------------------------------------------------------------- #
# Equality with the hand-assembled plan
# --------------------------------------------------------------------------- #
#: ``(id, kernels, index)``: plan ``kernels[index]`` out of ``auto(kernels)``.
#: A ``jvp`` kernel names its primal, which must share its bundle: it is
#: built in a list with it (alone, hawk refuses it naming ``primal_unit``).
_EQUALITY = [("affine", [ak_affine], 0), ("energy", [ak_energy], 0),
             ("energy_jvp", [ak_energy, ak_energy_jvp], 1)]


@pytest.mark.parametrize("kernels,index", [c[1:] for c in _EQUALITY],
                         ids=[c[0] for c in _EQUALITY])
def test_auto_kernel_runs_bit_identical_to_the_hand_assembled_plan(
        kernels, index, cache, tmp_path):
    built = eplan.auto(kernels if len(kernels) > 1 else kernels[0], cache_dir=cache)
    ap = built[index] if isinstance(built, tuple) else built
    assert isinstance(ap, eplan.AutoPlan)
    name = kernels[index].name
    inputs = _inputs(name)
    want = eplan.plan(_hand_plugins(kernels, tmp_path)[name],
                      structure=eexec.HostTeam).run(**inputs)
    got = ap.run(**inputs)
    assert ap.select(inputs).structure is eexec.HostTeam
    assert _bytes(got) == _bytes(want)
    first = np.asarray(next(iter(got.values())) if isinstance(got, dict) else got)
    assert len({first[..., i].tobytes() for i in range(N)}) == N   # distinct rows


@hawk.kernel
def ak_positive(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


#: The multiply-adds the device may contract per kernel, each moving a result
#: by at most one ULP (positive terms, no cancellation).
_CONTRACTIONS = {"ak_positive": 1, "ak_energy": 4}


@pytest.mark.parametrize("kernel", [ak_positive, ak_energy], ids=lambda k: k.name)
def test_auto_kernel_on_the_device_matches_the_hand_plan_and_the_host(
        kernel, cache, tmp_path):
    """Bit for bit against the hand-assembled DEVICE plan (the same compiled
    bytes), and within one ULP of the host per multiply-add the device may
    contract (no cancellation in these inputs)."""
    if not _have_gpu():
        pytest.skip("no CUDA device")
    import cupy as cp

    ap = eplan.auto(kernel, cache_dir=cache)
    inputs = _inputs(kernel.name)
    on_device = {k: cp.asarray(v) if isinstance(v, np.ndarray) else v
                 for k, v in inputs.items()}
    host = ap.run(**inputs)
    dev = ap.run(**on_device)
    assert ap.select(on_device).structure is eexec.DeviceKernel
    assert isinstance(dev, cp.ndarray)
    hand = _hand_plugins([kernel], tmp_path, targets=("host", "cuda"))[kernel.name]
    want = eplan.plan(hand, structure=eexec.DeviceKernel).run(**on_device)
    assert dev.get().tobytes() == want.get().tobytes()
    assert _ulp(dev.get(), host) <= _CONTRACTIONS[kernel.name]


# --------------------------------------------------------------------------- #
# A list is one bundle, in order
# --------------------------------------------------------------------------- #
def test_a_list_gives_a_tuple_of_plans_in_order_from_one_bundle(cache):
    plans = eplan.auto([ak_energy, ak_affine, ak_osc], cache_dir=cache)
    assert isinstance(plans, tuple) and len(plans) == 3
    assert [p.plugin.name for p in plans] == ["ak_energy", "ak_affine", "ak_osc"]
    plans[0].host  # a default GPU deploy builds its host side in the background
    sides = 2 if plan_deploy._eager_targets() == ("cuda",) else 1
    manifests = [json.loads(m.read_text())
                 for m in cache.glob("bundles/*/manifest.json")]
    assert [[e["id"] for e in m["plugins"]] for m in manifests].count(
        ["ak_energy", "ak_affine", "ak_osc"]) == sides      # ONE bundle per side holds all three
    e, a, _ = plans
    assert _bytes(a.run(**_inputs("ak_affine"))) == \
        _bytes(eplan.auto(ak_affine, cache_dir=cache).run(**_inputs("ak_affine")))
    assert np.asarray(e.run(**_inputs("ak_energy"))).shape == (N,)


def test_targets_default_and_explicit(cache):
    plain = eplan.auto(ak_affine, cache_dir=cache)
    assert (getattr(plain.plugin, "device_function", None) is not None) \
        == ("cuda" in plan_deploy._default_targets())
    host_only = eplan.auto(ak_affine, targets=("host",), cache_dir=cache)
    assert getattr(host_only.plugin, "device_function", None) is None
    assert host_only.plugin.host_entry


# --------------------------------------------------------------------------- #
# The cache: no rebuild, in-process and in a new process
# --------------------------------------------------------------------------- #
def test_a_second_call_in_the_same_process_compiles_nothing(tmp_path):
    from hawk.artifact import unit_stats

    eplan.auto([ak_affine, ak_energy], targets=("host",), cache_dir=tmp_path)
    before = unit_stats()
    eplan.auto([ak_affine, ak_energy], targets=("host",), cache_dir=tmp_path)
    after = unit_stats()
    assert after["published"] == before["published"]
    assert after["memo_hits"] == before["memo_hits"] + 1


_NEW_PROCESS = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {tests!r})
    import hawk.artifact, hawk.compile
    from eagle import plan as eplan
    from test_auto_kernels import ak_affine, ak_energy
    hawk.compile.reset_cache_stats()
    a, e = eplan.auto([ak_affine, ak_energy], targets=("host",),
                      cache_dir={cache!r})
    print(json.dumps({{"units": hawk.artifact.unit_stats(),
                      "compile": hawk.compile.cache_stats()}}))
""")


def _published_mtimes(root):
    return {p: p.stat().st_mtime_ns for p in sorted(root.rglob("bundles/*/*"))}


def test_a_second_call_in_a_new_process_hits_the_cache(tmp_path):
    code = _NEW_PROCESS.format(tests=os.path.dirname(__file__), cache=str(tmp_path))
    env = dict(os.environ)
    first = subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                           capture_output=True, text=True, timeout=600)
    assert first.returncode == 0, first.stderr
    mtimes = _published_mtimes(tmp_path)
    assert mtimes and json.loads(first.stdout.splitlines()[-1])["units"][
        "published"] == 1
    second = subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                            capture_output=True, text=True, timeout=600)
    assert second.returncode == 0, second.stderr
    report = json.loads(second.stdout.splitlines()[-1])
    assert report["units"]["published"] == 0
    assert report["units"]["stamp_hits"] == 1
    assert report["compile"]["misses"] == 0
    assert _published_mtimes(tmp_path) == mtimes


def test_the_bundle_key_folds_in_the_opt_level(monkeypatch):
    """Mirrors the host profile's own fold-in: two opt levels of the same
    kernels/targets must never land in the same cache folder, since the
    level changes the device (nvcc) build too, not just the host one — an
    ``O0`` folder served an ``O3`` .so (or vice versa) would be silently
    wrong, not just a slow rebuild."""
    monkeypatch.delenv("HAWK_OPT_LEVEL", raising=False)
    default_key = plan_deploy._bundle_key([ak_affine], ("host",))
    monkeypatch.setenv("HAWK_OPT_LEVEL", "O3")
    o3_key = plan_deploy._bundle_key([ak_affine], ("host",))
    assert o3_key == default_key, "O3 is hawk's own default opt level"
    monkeypatch.setenv("HAWK_OPT_LEVEL", "O0")
    o0_key = plan_deploy._bundle_key([ak_affine], ("host",))
    assert o0_key != o3_key
    monkeypatch.setenv("HAWK_OPT_LEVEL", "fastest")
    with pytest.raises(hawk.HawkError, match="unknown opt level"):
        plan_deploy._bundle_key([ak_affine], ("host",))


# --------------------------------------------------------------------------- #
# Derived kernels
# --------------------------------------------------------------------------- #
#: ``ak_osc`` finishes its own samples, so ``@hawk.kernel`` built it as a
#: default automatic kernel; ``.step`` is its single step and
#: ``hawk.steps(ak_osc, 4)`` takes four steps per launch.
_FINISHING = [("default_auto", lambda: ak_osc), ("step", lambda: ak_osc.step),
              ("steps4", lambda: hawk.steps(ak_osc, 4))]


@pytest.mark.parametrize("make", [c[1] for c in _FINISHING],
                         ids=[c[0] for c in _FINISHING])
def test_a_finishing_kernel_runs_through_run_until_done(make, cache, tmp_path):
    kernel = make()
    assert kernel is not None
    ap = eplan.auto(kernel, cache_dir=cache)
    hand = eplan.plan(_hand_plugins([kernel], tmp_path)[kernel.name],
                      structure=eexec.HostTeam)
    rng = np.random.default_rng(4)
    omega, nstop = rng.uniform(1.0, 3.0, N), np.arange(1.0, N + 1.0) * 3.0
    states = []
    for p in (ap, hand):
        x, v, k = np.ones(N), np.zeros(N), np.zeros(N)
        report = eagle.run_until_done(p, max_steps=1000, omega=omega, nstop=nstop,
                                      dt=1e-2, x=x, v=v, k=k)
        assert report.done
        states.append((x.tobytes(), v.tobytes(), k.tobytes()))
    assert states[0] == states[1]
    assert np.array_equal(np.frombuffer(states[0][2]), nstop)


# --------------------------------------------------------------------------- #
# Severance: eagle without hawk
# --------------------------------------------------------------------------- #
_MASKED = textwrap.dedent("""
    import sys
    sys.modules["hawk"] = None                     # hawk masked: import fails
    import eagle
    from types import SimpleNamespace
    from eagle import plan as eplan
    plugin = SimpleNamespace(arg_spec=(), exec_access="sample_local",
                             host_entry=1)
    assert isinstance(eplan.auto(plugin), eplan.AutoPlan)
    Fake = type("Kernel", (), {"__module__": "hawk.trace.kernel",
                               "walk": None, "sinks": (), "name": "k"})
    try:
        eplan.auto(Fake())
    except ImportError as exc:
        print("REFUSED", exc)
    assert "hawk" not in [m for m in sys.modules if sys.modules[m] is not None]
""")


def test_eagle_imports_and_plans_a_plugin_with_hawk_absent():
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", _MASKED],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith(
        "REFUSED eagle.plan.auto: building a hawk kernel needs hawk, which does "
        "not import here"), out.stdout
    assert out.stdout.rstrip().endswith("install hawk, or pass a built plugin")


_LAZY = textwrap.dedent("""
    import sys
    import eagle
    assert "eagle.plan" not in sys.modules and "hawk" not in sys.modules
    from eagle import deploy
    import eagle.plan
    assert deploy is eagle.plan.auto and eagle.deploy is deploy
    print("OK")
""")


def test_eagle_deploy_is_plan_auto_bound_on_first_use(cache):
    assert eagle.deploy is eplan.auto
    assert "deploy" in eagle.__all__
    a, e = eagle.deploy([ak_affine, ak_energy], targets=("host",), cache_dir=cache)
    assert _bytes(a.run(**_inputs("ak_affine"))) == _bytes(
        eplan.auto(ak_affine, targets=("host",), cache_dir=cache).run(
            **_inputs("ak_affine")))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", _LAZY],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip() == "OK", out.stderr


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_an_empty_list_is_refused():
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.auto: the list is empty; pass one hawk kernel or a "
            r"list of them, built into one bundle$")):
        eplan.auto([])


def test_a_list_mixing_a_kernel_and_a_plugin_is_refused(cache):
    plugin = eplan.auto(ak_affine, targets=("host",), cache_dir=cache).plugin
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.auto: a list is built into one bundle, so every item "
            r"must be a hawk kernel, but item 1 is a SimpleNamespace; pass a built "
            r"plugin on its own: eagle\.plan\.auto\(plugin\)$")):
        eplan.auto([ak_affine, plugin])


def test_build_keywords_on_a_built_plugin_are_refused(cache):
    plugin = eplan.auto(ak_affine, targets=("host",), cache_dir=cache).plugin
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.auto: targets= builds a hawk kernel, but this plugin "
            r"is already built: pass the kernel, or drop the keyword$")):
        eplan.auto(plugin, targets=("host",))


def test_a_missing_target_is_refused_with_the_existing_text(cache):
    ap = eplan.auto(ak_affine, targets=("host",), cache_dir=cache)
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.auto: the data is on the device, but this plugin "
            r"carries no device_function \(its exec_targets: \['host'\]\); build "
            r"it with targets=\(\"host\", \"cuda\"\) to run it on either side$")):
        ap.device                                     # noqa: B018
