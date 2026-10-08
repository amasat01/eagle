# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Two public-API checks on ``eagle.simulate``/``eagle.deploy``, kept in a
file of their own rather than folded into ``test_simulate.py``.

* **The eagle arch twin.** ``eagle.deploy``/``eagle.simulate`` build hawk
  kernels through ``hawk.artifact.build_bundle`` with no ``device_arch=`` of
  their own (``eagle._plan_deploy._hawk_plugins``), so they must resolve the
  SAME device arch hawk's own ``build``/``build_bundle`` would: the running
  GPU (probed through ``hawk.compile.nvrtc.current_device_arch``), not a
  blind ``sm_61`` default.
* **The positional-binding refusal.** A model built from a PREBUILT plan (no
  raw hawk kernel to read a call signature from) refuses positional kernel
  arguments outright, naming the keyword call to use instead -- binding by
  keyword always works, and a raw-kernel model still binds positionally
  (unchanged, covered by ``test_simulate.py``'s own call-shape rows).
"""

from __future__ import annotations

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
from hawk.compile import nvrtc as hnvrtc  # noqa: E402
from hawk.compile import toolchain as tc


@pytest.fixture(autouse=True)
def _fresh_probe_memo():
    """Isolate the per-process device-arch probe memo between rows."""
    tc.reset_device_arch_memo()
    yield
    tc.reset_device_arch_memo()


@hawk.kernel
def _deploy_arch_probe(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    """The smallest possible kernel: one Scalar in, one Scalar out."""
    y = x


def test_deploy_builds_for_the_running_device(monkeypatch, tmp_path):
    """``eagle.deploy(kernel, targets=("host", "cuda"))`` passes no
    ``device_arch=`` of its own (``_hawk_plugins`` -> ``build_bundle``), so
    with no ``$HAWK_CUDA_ARCH`` pin it must resolve through the SAME probe
    hawk.artifact.arch uses -- spied at ``toolchain.device_flags``, the real
    function underneath the AOT nvcc device compile.

    Only the COMPILE step's arch resolution is under test here, not loading
    the result onto a device, so the registry loader is stubbed: this proves
    the resolution itself on a GPU-less sandbox, the same way the hawk-side
    twin (``hawk/tests/test_build_arch_default.py``) does."""
    import collections
    from types import SimpleNamespace

    import eagle
    import eagle.registry as eregistry

    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    monkeypatch.setattr(hnvrtc, "current_device_arch", lambda: "sm_80")
    real_device_flags = tc.device_flags
    seen = []

    def _spy(*, arch="", **kw):
        seen.append(arch)
        return real_device_flags(arch=arch, **kw)

    monkeypatch.setattr(tc, "device_flags", _spy)
    fake_loaded = SimpleNamespace(fn=SimpleNamespace(kernel=SimpleNamespace(ptr=0)))
    monkeypatch.setattr(
        eregistry, "load_manifest",
        lambda manifest_path: collections.defaultdict(lambda: fake_loaded))

    eagle.deploy(_deploy_arch_probe, cache_dir=str(tmp_path), targets=("host", "cuda"))

    assert seen and all(a == "sm_80" for a in seen), (
        f"eagle.deploy must resolve the device arch through the running-GPU "
        f"probe like hawk.artifact.arch does; saw {seen}"
    )


@hawk.kernel
def _one_shot_finisher(x: hawk.Mutable[hawk.Scalar], terminated: hawk.Terminated):
    """The only bindable plane is ``x``; every sample finishes at once (the
    condition is trivially true, spelled rather than a bare ``= True``), so
    one call to :func:`eagle.simulate` is enough to exercise the bind."""
    x = x + 1.0
    terminated = x > -1e300


def test_positional_refused_for_prebuilt_plans(tmp_path):
    """A model built from ``eagle.deploy(...)`` (a prebuilt plan, no raw hawk
    kernel for the door to read a call signature from) refuses a positional
    kernel argument, naming the keyword fix; the identical call by keyword
    runs."""
    import eagle

    plan = eagle.deploy(_one_shot_finisher, cache_dir=str(tmp_path), targets=("host",))
    x = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="x="):
        eagle.simulate(plan, x, max_steps=1)

    result = eagle.simulate(plan, x=x.copy(), max_steps=1)
    np.testing.assert_array_equal(result.x, x + 1.0)
    assert result.done


def test_positional_still_binds_for_a_raw_kernel(tmp_path):
    """The companion, non-vacuity half: a model that IS a raw hawk kernel
    (no prebuilt plan involved) still binds positionally -- the refusal is
    specific to the prebuilt-plan hazard, not a blanket ban."""
    import eagle

    x0 = np.array([0.0, 1.0, 2.0])
    result = eagle.simulate(_one_shot_finisher, x0.copy(), max_steps=1)
    np.testing.assert_array_equal(result.x, x0 + 1.0)
    assert result.done
