# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Which GPU a run uses: the one its arrays live on, whatever is current.

Two-GPU rows (marker ``gpu2``, skipped below two devices). A kernel loaded or
a framework left on one device must not decide where a plan, a bound launch,
a graph or a reduction runs; mixed planes and a wrong-device stream are
refused by name before anything launches.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("hawk")

import eagle  # noqa: E402
import eagle.exec as eexec  # noqa: E402
import importlib  # noqa: E402

from eagle import _core  # noqa: E402
from eagle import plan as eplan  # noqa: E402
from eagle.interop import BufferRefused  # noqa: E402
from test_plan_bind_launch import (  # noqa: E402
    N, W, _bundle, _device_subject, _x_plane, e4_vec3_scale)

HERE = Path(__file__).resolve().parent
_launch_mod = importlib.import_module("eagle.launch")


def _device_count() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


pytestmark = [
    pytest.mark.gpu2,
    pytest.mark.skipif(_device_count() < 2,
                       reason="needs two CUDA devices (skipped, never passed, on one)"),
]


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    return _bundle(e4_vec3_scale, tmp_path_factory, "device_sel")


@pytest.fixture(autouse=True)
def _device_zero_after():
    yield
    cp.cuda.Device(0).use()


def _subject(unit, device):
    """The plan-able plugin, its kernel freshly loaded under ``device``."""
    from eagle.registry import _clear_plugin_memo

    _clear_plugin_memo()
    with cp.cuda.Device(device):
        return _device_subject(unit, "e4_vec3_scale")


def _planes(device):
    with cp.cuda.Device(device):
        return cp.asarray(_x_plane()), cp.zeros((W, N))


def _host(arr):
    with cp.cuda.Device(arr.device.id):
        cp.cuda.Device().synchronize()
        return cp.asnumpy(arr)


def _plan(plugin):
    return eplan.plan(plugin, structure=eexec.DeviceKernel)


def test_r1_plan_runs_on_the_planes_device_whatever_is_current(unit):
    plugin = _subject(unit, 1)
    x, _ = _planes(1)
    cp.cuda.Device(0).use()
    out = _plan(plugin).run(x=x, a=2.0, returns="cupy")
    assert out.device.id == 1
    np.testing.assert_array_equal(_host(out), 2.0 * _host(x))


def test_r2_a_kernel_loaded_on_one_device_runs_on_the_other(unit):
    plugin = _subject(unit, 0)
    x, _ = _planes(1)
    cp.cuda.Device(0).use()
    out = _plan(plugin).run(x=x, a=2.0, returns="cupy")
    assert out.device.id == 1
    np.testing.assert_array_equal(_host(out), 2.0 * _host(x))


def test_r3_bind_and_launch_follow_the_planes(unit):
    plugin = _subject(unit, 0)
    x, y = _planes(1)
    cp.cuda.Device(0).use()
    bound = _plan(plugin).bind(x=x, y=y, a=2.0)
    bound.launch()
    np.testing.assert_array_equal(_host(y), 2.0 * _host(x))
    with cp.cuda.Device(1):
        y[...] = 0.0
        s1 = cp.cuda.Stream(non_blocking=True)
    bound.launch(stream=s1)
    s1.synchronize()
    np.testing.assert_array_equal(_host(y), 2.0 * _host(x))
    loaded = plugin._keepalive[-1]
    assert loaded.function(1) is loaded.function(1)


def _mixed_planes_probe(directory: str, manifest: str) -> None:
    """Child process: planes on two devices; prints what the door did."""
    from types import SimpleNamespace

    from eagle.registry import _clear_plugin_memo

    _clear_plugin_memo()
    unit = SimpleNamespace(directory=Path(directory), manifest_path=Path(manifest))
    plugin = _device_subject(unit, "e4_vec3_scale")
    with cp.cuda.Device(0):
        x = cp.asarray(_x_plane())
    with cp.cuda.Device(1):
        y = cp.zeros((W, N))
    try:
        _plan(plugin).run(x=x, y=y, a=2.0)
    except BufferRefused as e:
        untouched = bool((_host(y) == 0).all())
        print("REFUSED", untouched, e)
    except Exception as e:  # noqa: BLE001
        print("RAW", type(e).__name__, e)
    else:
        print("RAN")


def test_r4_mixed_planes_are_refused_by_name(unit):
    code = ("import sys; sys.path.insert(0, %r); import test_device_selection as t; "
            "t._mixed_planes_probe(*sys.argv[1:3])" % str(HERE))
    done = subprocess.run([sys.executable, "-c", code, str(unit.directory), str(unit.manifest_path)],
                          capture_output=True, text=True, timeout=300)
    line = [ln for ln in done.stdout.splitlines()
            if ln.startswith(("REFUSED", "RAW", "RAN"))]
    assert line and line[-1].startswith("REFUSED True"), done.stdout + done.stderr
    assert ("'x'" in line[-1] or "'y'" in line[-1])
    assert "device 1" in line[-1] and "device 0" in line[-1]


def test_r5_plan_graph_runs_on_the_planes_device(unit):
    plugin = _subject(unit, 1)
    x, y = _planes(1)
    bound = _plan(plugin).bind(x=x, y=y, a=2.0)
    graph = eagle.Graph().add(bound).build()
    for _ in range(2):
        cp.cuda.Device(0).use()
        graph.launch()
    np.testing.assert_array_equal(_host(y), 2.0 * _host(x))


def test_r6_a_pipeline_refuses_a_plan_of_another_device(unit):
    plugin = _subject(unit, 1)
    x, y = _planes(1)
    bound = _plan(plugin).bind(x=x, y=y, a=2.0)
    cp.cuda.Device(0).use()
    pipe = eagle.GraphPipeline()
    with pytest.raises(ValueError, match=r"runs on device 0, the plan's planes live on device 1"):
        pipe.add(bound.launch)


def test_r7_reduce_follows_the_planes():
    with cp.cuda.Device(1):
        x = cp.arange(8, dtype=cp.float64)
        out = cp.zeros(1)
    cp.cuda.Device(0).use()
    eagle.Graph().add_node(eagle.reduce(x, out)).build().launch()
    assert float(_host(out)[0]) == 28.0


def test_r8_run_device_takes_a_device_and_names_a_stream_mismatch(unit):
    plugin = _subject(unit, 1)
    x, y = _planes(1)
    bound = _plan(plugin).bind(x=x, y=y, a=2.0)
    addrs, part = list(bound._addrs), bound.partitions[0]
    cp.cuda.Device(0).use()
    assert _core.run_device(plugin.device_function, addrs, part, stream=0, device=1) == 1
    np.testing.assert_array_equal(_host(y), 2.0 * _host(x))
    s0 = cp.cuda.Stream(non_blocking=True)
    with pytest.raises(Exception, match=r"(?s)device 0.*device 1"):
        _core.run_device(plugin.device_function, addrs, part, stream=s0.ptr, device=1)


def test_r10_device_props_are_per_ordinal(monkeypatch):
    assert _launch_mod._device_props(0) is not _launch_mod._device_props(1)
    seen = []
    real = _core.device_props
    monkeypatch.setattr(_core, "device_props", lambda d: (seen.append(d), real(d))[1])
    with cp.cuda.Device(1):
        eagle.device_props()
    assert seen == [-1]   # the current device, resolved in C++ (no pinned 0)


def test_r13_a_device_keyword_matches_the_arrays_or_refuses(unit):
    plugin = _subject(unit, 1)
    x, y = _planes(1)
    cp.cuda.Device(0).use()
    p = _plan(plugin)
    out = p.run(x=x, a=2.0, device=1, returns="cupy")
    np.testing.assert_array_equal(_host(out), 2.0 * _host(x))
    with pytest.raises(BufferRefused, match=r"device=0 was asked, the arrays live on device 1"):
        p.run(x=x, y=y, a=2.0, device=0)
    assert not _host(y).any()
    p.bind(x=x, y=y, a=2.0, device=1).launch()
    np.testing.assert_array_equal(_host(y), 2.0 * _host(x))
    with pytest.raises(BufferRefused, match=r"device=0 was asked, the arrays live on device 1"):
        p.bind(x=x, y=y, a=2.0, device=0)


def test_r14_a_runner_reads_the_props_of_its_planes_device(unit, monkeypatch):
    from eagle._until_done import Runner

    plugin = _subject(unit, 1)
    x, y = _planes(1)
    bound = _plan(plugin).bind(x=x, y=y, a=2.0)
    cp.cuda.Device(0).use()
    runner = object.__new__(Runner)
    runner.step = bound
    asked = []
    real = _launch_mod._device_props
    monkeypatch.setattr(_launch_mod, "_device_props", lambda d=-1: (asked.append(d), real(d))[1])
    assert runner._run_device() == 1
    runner._device_and_ops()
    assert asked == [1]


def test_r15_a_gemm_handle_follows_its_device(unit):
    from eagle.gemm.capture import CublasCaptureHandle, sgemm_params_for_arrays

    with cp.cuda.Device(1):
        a = cp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))
        b = cp.asarray(np.arange(8, dtype=np.float32).reshape(4, 2))
        out = cp.zeros((3, 2), dtype=cp.float32)
        stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.Device(0).use()
    handle = CublasCaptureHandle(stream.ptr, device=1)
    try:
        params = sgemm_params_for_arrays(a, b, out)
        handle.sgemm(params, {"a": a, "b": b, "out": out})
        stream.synchronize()
        np.testing.assert_allclose(_host(out), _host(a) @ _host(b))
        other = cp.zeros((3, 2), dtype=cp.float32)      # on device 0
        with pytest.raises(ValueError, match=r"operand 'out' lives on device 0, the handle on device 1"):
            handle.sgemm(params, {"a": a, "b": b, "out": other})
    finally:
        handle.destroy()


def test_r16_a_composer_refuses_a_member_of_another_device(unit):
    from eagle.compose import GraphComposer

    plugin = _subject(unit, 1)
    x, y = _planes(1)
    bound = _plan(plugin).bind(x=x, y=y, a=2.0)
    cp.cuda.Device(0).use()
    composer = GraphComposer(None, mode="sequenced")
    with pytest.raises(ValueError, match=r"composer runs on device 0, the member's planes live on device 1"):
        composer.register(bound.launch)


def test_r17_a_parameter_named_device_is_refused_at_plan_build(unit):
    from types import SimpleNamespace

    plugin = _subject(unit, 0)
    spec = [(role, "device" if name == "a" else name) for role, name in plugin.arg_spec]
    clash = SimpleNamespace(**{**vars(plugin), "arg_spec": spec})
    with pytest.raises(ValueError, match=r"parameter named 'device' clashes with the device= keyword"):
        _plan(clash)
