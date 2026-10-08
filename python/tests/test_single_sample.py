# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One sample through the same kernel: the head rule, head-shaped results,
and :func:`eagle.plan.auto` (residency picks the target).

A value whose shape IS a plane's per-sample head (a number or 0-d array for a
scalar plane, ``(w,)`` for a vector, ``(r, c)`` for a matrix) is ONE sample:
it runs as a batch of one through its zero-copy view and comes back in its
head shape. What each group of rows proves:

* BIT IDENTITY (the non-vacuity pair): on a batch of pairwise-distinct
  samples, the single-sample call on row ``i`` equals ``batch[..., i]`` bit for
  bit on the host, for a Vector/Scalar/Matrix kernel, its ``jvp``, and a
  finishing kernel through ``run_until_done`` (whose per-sample step count
  must match too); and it DIFFERS from every other row, so a mutant that
  returned any row would fail. Device: within the documented 1-ULP rule.
* RESULTS: numpy scalar / ``(w,)`` / ``(r, c)`` results; cupy 0-d for device
  inputs; supplied head-shaped outputs written in place and returned by
  identity; a batch of one still returns batch shapes.
* REFUSALS: the exact texts for a mix of one sample and a batch, a number
  under an output name, a number for a ``Mutable`` in ``until_done``, an auto
  plan given data on both sides, and an auto plan missing the selected target.
* ROUTING: ``auto`` selects ``HostTeam`` for numpy and numbers and
  ``DeviceKernel`` for cupy, at N = 1 and N = 1000 (asserted on the selected
  plan's structure, never on timing).
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

import eagle
import eagle.exec as eexec
from eagle import plan as eplan

hawk = pytest.importorskip("hawk")
from hawk import Kernel, Matrix, Mutable, Param, Scalar, Terminated, Vector  # noqa: E402
from hawk import math as hm  # noqa: E402
from hawk.diff import jvp  # noqa: E402

N = 7


@hawk.kernel
def ss_mix(x: Vector[3], s: Scalar, m: Matrix[2, 3], a: Param,
           y: Mutable[Vector[2]], z: Mutable[Scalar], w: Mutable[Matrix[2, 3]]):
    y = (m @ x) * s
    z = hm.dot(x, x) * s + a
    w = m * s


ss_mix_jvp = Kernel("ss_mix_jvp", jvp(ss_mix, wrt=("x", "s")))


@hawk.kernel
def ss_osc(omega: Scalar, nstop: Scalar, dt: Param, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], k: Mutable[Scalar]):
    x0 = x
    v0 = v
    x = x0 + dt * v0
    v = v0 - dt * (omega * omega) * x0
    k1 = k + hm.min(1.0, nstop)
    k = k1
    terminated = k1 >= nstop


def _have_gpu():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _plugins(kernels, work):
    """``{name: plugin}``: each plugin carries the host entry and, when a GPU
    is present, the device function — the one object ``eagle.plan.auto``
    takes."""
    from hawk import _core as hcore
    from hawk.artifact import build_bundle, plan_view

    targets = ("host", "cuda") if _have_gpu() else ("host",)
    bundle = build_bundle(kernels, work, targets=targets)
    registry = None
    if "cuda" in targets:
        from eagle.registry import load_manifest

        registry = load_manifest(work / "manifest.json")
    out = {}
    for art in bundle.artifacts:
        view = plan_view(art.sidecar)
        host_entry = view.pop("host_entry")
        so = work / f"{art.name}.so"
        lib, cdll = hcore.HostLibrary(str(so)), ctypes.CDLL(str(so))
        extra = {"host_entry": ctypes.cast(getattr(cdll, host_entry),
                                           ctypes.c_void_p).value}
        keep = [lib, cdll]
        if registry is not None:
            loaded = registry[art.name]
            extra["device_function"] = loaded.fn.kernel.ptr
            keep += [registry, loaded]
        out[art.name] = SimpleNamespace(name=art.name, _keepalive=keep,
                                        **extra, **view)
    return out


@pytest.fixture(scope="module")
def mix(tmp_path_factory):
    return _plugins([ss_mix, ss_mix_jvp], tmp_path_factory.mktemp("ss_mix"))


@pytest.fixture(scope="module")
def osc(tmp_path_factory):
    return _plugins([ss_osc], tmp_path_factory.mktemp("ss_osc"))["ss_osc"]


def _need_gpu():
    if not _have_gpu():
        pytest.skip("no CUDA device")


def _batch_inputs():
    rng = np.random.default_rng(11)
    return {"x": rng.normal(size=(3, N)), "s": rng.uniform(0.5, 2.0, N),
            "m": rng.normal(size=(6, N))}


def _tangents():
    rng = np.random.default_rng(12)
    return {"dot_x": rng.normal(size=(3, N)), "dot_s": rng.normal(size=N)}


def _row(planes, i, heads):
    """Sample ``i`` of every batch plane, in its head shape."""
    return {k: (v[..., i].reshape(heads[k]) if heads.get(k) != () else v[i])
            for k, v in planes.items()}


def _ulp(a, b) -> int:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ia, ib = a.view(np.int64), b.view(np.int64)
    return int(np.max(np.abs(ia - ib))) if a.size else 0


def _named(plugin, result):
    """``{output name: value}`` for a ``Plan.run`` result: already a dict
    for several outputs, a bare single output wrapped under its own name."""
    if isinstance(result, dict):
        return dict(result)
    names = [nm for role, nm in plugin.arg_spec if role == "mutable"]
    return {names[0]: result}


_HEADS = {"x": (3,), "s": (), "m": (2, 3), "dot_x": (3,), "dot_s": ()}


def _distinct(cols):
    """Pairwise distinctness, asserted first (the mutant row's premise)."""
    keys = [np.asarray(c).tobytes() for c in cols]
    assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------- #
# Bit identity, host
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["ss_mix", "ss_mix_jvp"])
def test_one_sample_equals_its_batch_column_on_the_host(mix, name):
    p = eplan.plan(mix[name], structure=eexec.HostTeam)
    planes = _batch_inputs()
    if name == "ss_mix_jvp":
        planes.update(_tangents())
    batch = p.run(a=0.25, **planes)
    batch = tuple(batch.values()) if isinstance(batch, dict) else (batch,)
    out_names = [nm for role, nm in mix[name].arg_spec if role == "mutable"]
    for plane in batch:
        _distinct(np.asarray(plane).reshape(-1, N).T)
    for i in range(N):
        one = p.run(a=0.25, **_row(planes, i, _HEADS))
        one = tuple(one.values()) if isinstance(one, dict) else (one,)
        for nm, b, o in zip(out_names, batch, one):
            col = b[..., i]
            assert np.array_equal(np.asarray(o).ravel(), col.ravel()), (nm, i)
            for j in range(N):
                if j != i:
                    assert not np.array_equal(np.asarray(o).ravel(),
                                              b[..., j].ravel()), (nm, i, j)


def _osc_inputs():
    rng = np.random.default_rng(5)
    return {"omega": rng.uniform(1.0, 5.0, N),
            "nstop": np.arange(3.0, 3.0 + 4 * N, 4.0),
            "x": rng.uniform(-1.0, 1.0, N), "v": rng.uniform(-1.0, 1.0, N)}


def _osc_run(p, inp, xp, i=None):
    take = (lambda a: xp.asarray(a).copy()) if i is None else \
        (lambda a: xp.asarray(np.array(a[i])))
    planes = {k: take(v) for k, v in inp.items()}
    planes["k"] = xp.zeros(()) if i is not None else xp.zeros(N)
    report = eagle.run_until_done(p, max_steps=500, dt=1e-2, **planes)
    assert report.done
    return planes


def test_one_trajectory_equals_its_batch_column_through_run_until_done(osc):
    p = eplan.plan(osc, structure=eexec.HostTeam)
    inp = _osc_inputs()
    batch = _osc_run(p, inp, np)
    _distinct(batch["x"])
    _distinct(batch["k"])
    for i in range(N):
        one = _osc_run(p, inp, np, i)
        assert one["x"].shape == ()
        for key in ("x", "v", "k"):                   # k: the step count
            assert one[key] == batch[key][i], (key, i)
        assert all(one["x"] != batch["x"][j] for j in range(N) if j != i)


# --------------------------------------------------------------------------- #
# Device (within the documented 1-ULP rule; max ULP recorded)
# --------------------------------------------------------------------------- #
def test_one_sample_on_the_device_matches_its_batch_column(mix, record_property):
    _need_gpu()
    import cupy as cp

    p = eplan.plan(mix["ss_mix"], structure=eexec.DeviceKernel)
    planes = {k: cp.asarray(v) for k, v in _batch_inputs().items()}
    batch = p.run(a=0.25, **planes)
    batch = tuple(batch.values()) if isinstance(batch, dict) else (batch,)
    worst = 0
    for i in range(N):
        row = {k: cp.ascontiguousarray(v[..., i]).reshape(_HEADS[k])
               for k, v in planes.items()}
        one = p.run(a=0.25, **row)
        one = tuple(one.values()) if isinstance(one, dict) else (one,)
        for b, o in zip(batch, one):
            assert isinstance(o, cp.ndarray)
            worst = max(worst, _ulp(o.get(), b[..., i].get()))
            for j in range(N):
                if j != i:
                    assert _ulp(o.get(), b[..., j].get()) > 1
    record_property("max_ulp_device_single", worst)
    assert worst <= 1


def test_one_trajectory_on_the_device_matches_its_batch_column(osc):
    _need_gpu()
    import cupy as cp

    p = eplan.plan(osc, structure=eexec.DeviceKernel)
    inp = _osc_inputs()
    batch = {k: v.get() for k, v in _osc_run(p, inp, cp).items()}
    for i in (0, N - 1):
        one = _osc_run(p, inp, cp, i)
        assert one["x"].shape == ()
        assert float(one["k"]) == batch["k"][i]
        assert _ulp(one["x"].get(), batch["x"][i]) <= 1


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def test_results_take_the_head_shape_in_the_callers_framework(mix):
    p = eplan.plan(mix["ss_mix"], structure=eexec.HostTeam)
    r = _named(mix["ss_mix"], p.run(x=np.array([1.0, 2.0, 3.0]), s=2.0,
                                    m=np.ones((2, 3)), a=0.5))
    y, z, w = r["y"], r["z"], r["w"]
    assert isinstance(z, np.float64) and isinstance(z, float) and z.shape == ()
    assert float(z) == 28.5
    assert isinstance(y, np.ndarray) and y.shape == (2,)
    assert isinstance(w, np.ndarray) and w.shape == (2, 3)
    np.testing.assert_array_equal(y, [12.0, 12.0])


def test_supplied_head_shaped_outputs_are_written_in_place_and_returned(mix):
    p = eplan.plan(mix["ss_mix"], structure=eexec.HostTeam)
    y, z, w = np.zeros(2), np.zeros(()), np.zeros((2, 3))
    got = _named(mix["ss_mix"], p.run(x=np.array([1.0, 2.0, 3.0]), s=np.array(2.0),
                                      m=np.ones((2, 3)), a=0.5, y=y, z=z, w=w))
    assert got["y"] is y and got["z"] is z and got["w"] is w
    assert float(z) == 28.5
    np.testing.assert_array_equal(y, [12.0, 12.0])


def test_a_batch_of_one_still_returns_batch_shapes(mix):
    p = eplan.plan(mix["ss_mix"], structure=eexec.HostTeam)
    r = _named(mix["ss_mix"], p.run(x=np.ones((3, 1)), s=np.ones(1),
                                    m=np.ones((6, 1)), a=0.0))
    y, z, w = r["y"], r["z"], r["w"]
    assert y.shape == (2, 1) and z.shape == (1,) and w.shape == (6, 1)


def test_device_inputs_give_cupy_head_shaped_results(mix):
    _need_gpu()
    import cupy as cp

    p = eplan.plan(mix["ss_mix"], structure=eexec.DeviceKernel)
    r = _named(mix["ss_mix"], p.run(x=cp.asarray([1.0, 2.0, 3.0]), s=cp.asarray(2.0),
                                    m=cp.ones((2, 3)), a=0.5))
    y, z, w = r["y"], r["z"], r["w"]
    assert isinstance(z, cp.ndarray) and z.shape == ()
    assert y.shape == (2,) and w.shape == (2, 3)
    assert float(z) == 28.5


# --------------------------------------------------------------------------- #
# Refusals (exact texts)
# --------------------------------------------------------------------------- #
def test_a_mix_of_one_sample_and_a_batch_is_refused_naming_both(mix):
    p = eplan.plan(mix["ss_mix"], structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan: 'x' is one sample \(shape \(3,\)\) but 'm' is a batch "
            r"of 1000 \(shape \(6, 1000\)\); a call is one sample or one batch, never "
            r"both\. A value shared by every sample is a Param \(scalar\) or a "
            r"Table \(vector\); or tile it once: "
            r"np\.repeat\(x\.reshape\(-1, 1\), 1000, axis=1\)$")):
        p.run(x=np.ones(3), s=np.ones(1000), m=np.ones((6, 1000)), a=0.0)


def test_a_number_under_an_output_name_is_refused(mix):
    p = eplan.plan(mix["ss_mix"], structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.run: mutable 'z' was given a Python float, which cannot "
            r"receive a write; pass a writable 0-d array \(np\.zeros\(\(\)\)\) or "
            r"leave it out to have it returned$")):
        p.run(x=np.ones(3), s=1.0, m=np.ones((2, 3)), a=0.0, z=0.0)


def test_a_number_for_a_mutable_in_until_done_is_refused(osc):
    p = eplan.plan(osc, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.bind: mutable 'x' was given a Python float, which "
            r"cannot receive a write; pass a writable 0-d array \(np\.array\(x\)\)$")):
        eagle.until_done(p, max_steps=10, dt=1e-2, omega=np.array(1.0),
                         nstop=np.array(3.0), x=0.5, v=np.array(0.0),
                         k=np.zeros(()))


def test_auto_refuses_data_on_both_sides(mix):
    _need_gpu()
    import cupy as cp

    ap = eplan.auto(mix["ss_mix"])
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.auto: 'x' is on the device but 's' is on the host; the "
            r"data decides where the kernel runs, and it runs in one place: move "
            r"one side with cp\.asarray / \.get\(\)$")):
        ap.run(x=cp.ones(3), s=np.array(1.0), m=cp.ones((2, 3)), a=0.0)


def test_auto_refuses_a_target_the_plugin_lacks(mix):
    host_only = SimpleNamespace(**{k: v for k, v in vars(mix["ss_mix"]).items()
                                   if k != "device_function"})
    ap = eplan.auto(host_only)
    with pytest.raises(ValueError, match=(
            r"^eagle\.plan\.auto: the data is on the device, but this plugin "
            r"carries no device_function \(its exec_targets: \['host'\]\); build "
            r"it with targets=\(\"host\", \"cuda\"\) to run it on either side$")):
        ap.device                                     # noqa: B018
    assert ap.select({"x": np.ones(3)}).structure is eexec.HostTeam


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 1000])
def test_auto_routes_by_residency_never_by_count(mix, n):
    ap = eplan.auto(mix["ss_mix"])
    if n == 1:
        host = {"x": np.ones(3), "s": 1.0, "m": np.ones((2, 3))}
    else:
        host = {"x": np.ones((3, n)), "s": np.ones(n), "m": np.ones((6, n))}
    assert ap.select(host).structure is eexec.HostTeam
    out = _named(mix["ss_mix"], ap.run(a=0.0, **host))
    assert out["y"].shape == ((2,) if n == 1 else (2, n))
    if not _have_gpu():
        return
    import cupy as cp

    dev = {k: (cp.asarray(v) if not isinstance(v, float) else cp.asarray(v))
           for k, v in host.items()}
    assert ap.select(dev).structure is eexec.DeviceKernel
    assert isinstance(_named(mix["ss_mix"], ap.run(a=0.0, **dev))["y"], cp.ndarray)


def test_until_done_takes_an_auto_plan(osc):
    ap = eplan.auto(osc)
    inp = _osc_inputs()
    batch = _osc_run(ap.host, inp, np)
    one = _osc_run(ap, inp, np, 2)
    assert one["k"] == batch["k"][2] and one["x"] == batch["x"][2]
