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
    key = np.zeros(4, dtype=np.int64)
    acc = np.zeros(1)
    bound = eplan.plan(plugin, structure=eexec.HostTeam).bind(x=x, key=key, acc=acc)
    return bound, acc


def test_clear_zeroes_the_target_and_its_owned_companion(host_plugin):
    bound, acc = _bind(host_plugin)
    bound.launch()
    clean = acc.copy()
    comp = bound._planes["acc_c"]
    assert comp.any(), "the subject must leave a nonzero companion"

    bound.clear("acc")
    assert not acc.any() and not comp.any()

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
    assert bound._planes["acc_c"].any(), "the subject must leave a nonzero companion"

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
