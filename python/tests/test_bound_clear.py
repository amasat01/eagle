# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``BoundPlan.clear``: zero a sink and the compensation companion eagle owns.

A keyed sum under the ``auto`` policy is compensated: after one run the
hidden companion holds the rounding residual. A caller who only re-zeroes the
target leaves that residual to be folded into the next run. The subject is a
rounding sum (values ``[2**53, 1, 1, 1]`` into one key, f64), so the
companion is nonzero after one run and a stale one changes the next result.
"""

from __future__ import annotations

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
hawk_trace = pytest.importorskip("hawk.trace")
hawk_artifact = pytest.importorskip("hawk.artifact")


@hawk_trace.kernel
def clear_scatter(x: hawk_trace.Scalar, key: hawk_trace.Index,
                  acc: hawk_trace.Mutable[hawk_trace.Scalar]):
    hawk.scatter(acc)[key] += x


@pytest.fixture(scope="module")
def host_plugin(tmp_path_factory):
    import ctypes
    from types import SimpleNamespace

    import json

    import eagle.exec as eexec  # noqa: F401

    unit = hawk_artifact.build_bundle(
        [clear_scatter], tmp_path_factory.mktemp("clear"), targets=("host",),
        device_arch=hawk_artifact.arch())
    sidecar = json.loads((unit.directory / "clear_scatter.json").read_text())
    lib = ctypes.CDLL(str(unit.directory / "clear_scatter.so"))
    view = hawk_artifact.plan_view(sidecar)
    entry = getattr(lib, view.pop("host_entry"))
    return SimpleNamespace(host_entry=ctypes.cast(entry, ctypes.c_void_p).value,
                           **hawk_artifact.fold_entries(lib, sidecar),
                           _keepalive=lib, _meta=sidecar, **view)


def _bind(plugin):
    import eagle.exec as eexec
    from eagle import plan as eplan

    x = np.array([2.0**53, 1.0, 1.0, 1.0])
    key = np.zeros(4, dtype=np.int32)
    acc = np.zeros(1)
    bound = eplan.plan(plugin, structure=eexec.HostTeam).bind(x=x, key=key, acc=acc)
    return bound, acc


def test_clear_zeroes_the_target_and_its_owned_companion(host_plugin):
    bound, acc = _bind(host_plugin)
    bound.launch()
    clean = acc.copy()
    comp = bound._planes["acc_c"]
    assert _host(comp).any(), "the subject must leave a nonzero companion"

    bound.clear("acc")
    assert not _host(acc).any() and not _host(comp).any()

    bound.launch()
    np.testing.assert_array_equal(acc, clean)
    np.testing.assert_array_equal(comp, bound._planes["acc_c"])


def test_zeroing_only_the_target_is_not_enough(host_plugin):
    """The control: the failure `clear` exists to prevent."""
    bound, acc = _bind(host_plugin)
    bound.launch()
    clean = acc.copy()
    acc.fill(0)
    bound.launch()
    assert not np.array_equal(acc, clean)


def test_clear_refuses_a_name_that_is_not_an_output(host_plugin):
    bound, _ = _bind(host_plugin)
    with pytest.raises(ValueError, match=r"valid: \['acc'\]"):
        bound.clear("nope")
    with pytest.raises(ValueError, match="acc_c"):
        bound.clear("acc_c")


def test_rebinding_the_target_renews_its_owned_companion(host_plugin):
    """A rebound target starts from a zero companion: the residual of the old
    target's run is not folded into the new one."""
    bound, acc = _bind(host_plugin)
    bound.launch()
    clean = acc.copy()
    assert _host(bound._planes["acc_c"]).any(), "the subject must leave a nonzero companion"

    fresh = np.zeros(1)
    bound.rebind(acc=fresh)
    bound.launch()
    np.testing.assert_array_equal(fresh, clean)


def test_rebinding_the_target_to_another_shape_makes_a_companion_of_that_shape(host_plugin):
    bound, acc = _bind(host_plugin)
    bound.launch()
    clean = acc.copy()

    wider = np.zeros(3)
    bound.rebind(acc=wider)
    assert bound._planes["acc_c"].shape == (3,)
    bound.launch()
    np.testing.assert_array_equal(wider, [clean[0], 0.0, 0.0])


# --- SplitBound.clear: every stage that binds the name, and the keyed read-back --- #
@hawk_trace.kernel
def split_acc(x: hawk_trace.Scalar, key: hawk_trace.Index,
              acc: hawk_trace.Mutable[hawk_trace.Scalar],
              z: hawk_trace.Mutable[hawk_trace.Scalar]):
    hawk.scatter(acc)[key] += x
    z = x / hawk.sum(x)


@hawk_trace.kernel
def split_keyed_share(lane: hawk_trace.Grid["row", "col":8], x: hawk_trace.Scalar,
                      acc: hawk_trace.Mutable[hawk_trace.Scalar],
                      key: hawk_trace.Index) -> hawk_trace.Scalar:
    hawk.scatter(acc)[key] += x
    return x / hawk.sum(x, over="col", policy="compensated")


def _xp(framework):
    if framework == "numpy":
        return np
    return pytest.importorskip("cupy")


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _split_bind(tmp_path, framework):
    import eagle.plan

    xp = _xp(framework)
    target = "host" if framework == "numpy" else "cuda"
    (p,) = eagle.plan.auto([split_acc], targets=(target,), cache_dir=str(tmp_path))
    x = xp.asarray(np.array([2.0**53, 1.0, 1.0, 1.0]))
    key = xp.zeros(4, dtype=xp.int32)
    acc, z = xp.zeros(1), xp.zeros(4)
    return p.bind(x=x, key=key, acc=acc, z=z), acc, z


FRAMEWORKS = [pytest.param("numpy"), pytest.param("cupy", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_split_clear_zeroes_the_output_and_its_owned_companion(tmp_path, framework):
    bound, acc, z = _split_bind(tmp_path, framework)
    bound.launch()
    clean_acc, clean_z = _host(acc).copy(), _host(z).copy()
    comps = [b._planes["acc_c"] for b in bound.stages if "acc_c" in b._planes]
    assert comps and _host(comps[0]).any(), "the subject must leave a nonzero companion"

    bound.clear("acc", "z")
    assert not _host(acc).any() and not _host(z).any()
    assert not any(_host(c).any() for c in comps)

    bound.launch()                      # equals a fresh first launch, bit for bit
    np.testing.assert_array_equal(_host(acc), clean_acc)
    np.testing.assert_array_equal(_host(z), clean_z)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_split_clear_of_the_target_alone_is_not_enough(tmp_path, framework):
    """The control: zeroing the target by hand leaves the residual."""
    bound, acc, _z = _split_bind(tmp_path, framework)
    bound.launch()
    clean = _host(acc).copy()
    acc.fill(0)
    bound.launch()
    assert not np.array_equal(_host(acc), clean)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_split_clear_refuses_what_is_not_a_plain_output(tmp_path, framework):
    bound, _acc, _z = _split_bind(tmp_path, framework)
    for bad in ("nope", "acc_c", "cut0", "x"):
        with pytest.raises(ValueError, match="are not bound planes|not bound output"):
            bound.clear(bad)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_split_clear_after_a_keyed_read_back_repeats_the_first_launch(tmp_path, framework):
    import eagle.plan

    xp = _xp(framework)
    target = "host" if framework == "numpy" else "cuda"
    (p,) = eagle.plan.auto([split_keyed_share], targets=(target,), cache_dir=str(tmp_path))
    rng = np.random.default_rng(4)
    x = xp.asarray(rng.uniform(0.5, 2.0, (4, 8)))
    key = xp.asarray(np.zeros(32, dtype=np.int32))
    acc, out = xp.zeros(1), xp.zeros((4, 8))
    bound = p.bind(x=x, key=key, acc=acc, out=out)
    bound.launch()
    first_acc, first_out = _host(acc).copy(), _host(out).copy()
    bound.clear("acc", "out")
    assert not _host(acc).any() and not _host(out).any()
    bound.launch()
    np.testing.assert_array_equal(_host(acc), first_acc)
    np.testing.assert_array_equal(_host(out), first_out)


@pytest.mark.gpu
def test_bound_clear_on_the_device_zeroes_the_target_and_its_companion(tmp_path):
    cp = pytest.importorskip("cupy")
    import eagle.exec as eexec
    from eagle import plan as eplan
    import eagle.plan

    (p,) = eagle.plan.auto([clear_scatter], targets=("cuda",), cache_dir=str(tmp_path))
    x = cp.asarray(np.array([2.0**53, 1.0, 1.0, 1.0]))
    acc = cp.zeros(1)
    bound = eplan.plan(p.plugin, structure=eexec.DeviceKernel).bind(
        x=x, key=cp.zeros(4, dtype=cp.int32), acc=acc)
    bound.launch()
    clean = acc.get().copy()
    comp = bound._planes["acc_c"]
    bound.clear("acc")
    assert not _host(acc).any() and not _host(comp).any()
    bound.launch()
    np.testing.assert_array_equal(acc.get(), clean)
    with pytest.raises(ValueError, match="acc_c"):
        bound.clear("acc_c")
