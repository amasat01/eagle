# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0


"""``AutoPlan.__call__``: a direct kernel call binds like ``eagle.simulate``,
launches once where the arrays live, and returns the kernel's tree."""

import types

import numpy as np
import pytest

import eagle
from eagle import _recording

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar  # noqa: E402


@hawk.kernel
def scale(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def kick(x: Scalar, v: Scalar, dt: Param, x_next: Mutable[Scalar]):
    x_next = x + v * dt


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return eagle.deploy(scale, targets=("host",),
                        cache_dir=str(tmp_path_factory.mktemp("hawk")))


X = np.linspace(0.0, 1.0, 8)


def test_autoplan_call_binds_like_simulate(plan):
    want = 2.0 * X + 0.5
    np.testing.assert_allclose(plan(X, 2.0, 0.5), want)
    np.testing.assert_allclose(plan(X, 2.0, b=0.5), want)
    np.testing.assert_allclose(plan(x=X, b=0.5, a=2.0), want)
    y = np.zeros(8)
    out = plan(X, 2.0, 0.5, y=y)          # a Mutable is the caller's array
    np.testing.assert_allclose(y, want)
    np.testing.assert_allclose(out, want)


def test_autoplan_call_names_the_missing_argument(plan):
    with pytest.raises(ValueError, match=r"scale\(\) missing a required argument: 'a'"):
        plan(X)
    with pytest.raises(ValueError, match="multiple values"):
        plan(X, 2.0, 0.5, x=X)


def test_autoplan_call_returns_tree(plan, monkeypatch):
    # hawk sets `returns` on the kernel; a stand-in carries it here
    k = plan.kernel
    for tree in ({"y": "y"}, ("y", ["y"])):
        monkeypatch.setattr(plan, "kernel", types.SimpleNamespace(
            planes=k.planes, name=k.name, walk=k.walk, returns=tree))
        got = plan(X, 2.0, 0.5)
        want = 2.0 * X + 0.5
        if isinstance(tree, dict):
            assert list(got) == ["y"]
            np.testing.assert_allclose(got["y"], want)
        else:
            a, (b,) = got
            np.testing.assert_allclose(a, want)
            np.testing.assert_allclose(b, want)


def test_autoplan_call_prebuilt_plugin_binds_by_name_only(plan):
    prebuilt = eagle.deploy(plan.plugin)
    np.testing.assert_allclose(prebuilt(x=X, a=2.0, b=0.5), 2.0 * X + 0.5)
    with pytest.raises(ValueError, match="by name"):
        prebuilt(X, 2.0, 0.5)


def test_autoplan_call_hands_off_to_the_recorder(plan):
    seen = []

    class Recorder:
        def record_call(self, p, args, kw):
            seen.append((p, args, kw, p.bind_args(args, kw)))
            return "recorded"

    assert _recording.current() is None
    with _recording.recording(Recorder()):
        assert plan(X, 2.0, b=0.5) == "recorded"
    assert _recording.current() is None
    (p, args, kw, bound), = seen
    assert p is plan and args == (X, 2.0) and kw == {"b": 0.5}
    assert list(bound) == ["x", "a", "b"]


def test_autoplan_call_reads_a_layout_marker(plan):
    np.testing.assert_allclose(plan(eagle.samples_last(X), 2.0, 0.5), 2.0 * X + 0.5)


BANDED_ON_CPU = "banded runs on device execution paths; on a CPU use float64"


def test_banded_execution_on_the_cpu_is_refused_by_name_at_deploy(tmp_path):
    with pytest.raises(ValueError, match=BANDED_ON_CPU):
        eagle.plan.auto(scale, targets=("host",), cache_dir=str(tmp_path), scalar_type="banded")


def test_a_banded_plugin_is_refused_a_host_team_plan():
    plugin = types.SimpleNamespace(scalar_type="banded", arg_spec=[("input", "x")], exec_access="sample_local")
    with pytest.raises(ValueError, match=BANDED_ON_CPU):
        eagle.plan.plan(plugin, structure=eagle.exec.HostTeam)


@pytest.mark.gpu
def test_autoplan_builds_and_runs_a_banded_kernel_on_the_device(tmp_path):
    from eagle import arrays

    p = eagle.plan.auto(scale, cache_dir=str(tmp_path), scalar_type="banded")
    x, y = (arrays.asarray(eagle.banded_pack(arrays.asarray(v)), device="cuda:0") for v in (X, np.zeros(8)))
    p.run(x=x, a=2.0, b=0.5, y=y)
    host = arrays.asarray(y, device="cpu")
    np.testing.assert_allclose(eagle.banded_unpack(host).tolist(), 2.0 * X + 0.5, rtol=0, atol=1e-15)
