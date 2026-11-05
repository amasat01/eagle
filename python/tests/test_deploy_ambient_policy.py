# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A deploy reads the caller's ``hawk.emit.policy.using`` block: two settings sharing one ``cache_dir`` build two bundles."""

import pytest

import eagle

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Scalar  # noqa: E402
from hawk.emit import policy  # noqa: E402


@hawk.kernel
def square(x: Scalar, y: Mutable[Scalar]):
    y = x * x + 1.0


def _bundle_dirs(root):
    return sorted(p.name for p in (root / "bundles").iterdir())


def test_ambient_launch_bounds_key_the_bundle_and_reach_the_build(tmp_path):
    eagle.deploy(square, targets=("host",), cache_dir=str(tmp_path))
    with policy.using(launch_bounds=(128, 2)):
        eagle.deploy(square, targets=("host",), cache_dir=str(tmp_path))
    with policy.using(wide_form="thread"):
        eagle.deploy(square, targets=("host",), cache_dir=str(tmp_path))
    assert len(_bundle_dirs(tmp_path)) == 3


def test_a_default_deploy_keeps_its_key(tmp_path):
    from eagle import _plan_deploy as d

    assert d._ambient_policy() == dict(form="auto", linalg_form="auto", wide_form="auto", contraction="direct",
                                       launch_bounds=None)
    a = d._bundle_key([square], ("host",))
    with policy.using(launch_bounds=(128, 2)):
        b = d._bundle_key([square], ("host",))
    assert a != b and a == d._bundle_key([square], ("host",))


def test_the_host_side_of_a_default_deploy_carries_the_policy(tmp_path):
    from eagle import _plan_deploy as d

    with policy.using(launch_bounds=(128, 2)):
        side = d._HostSide([square], str(tmp_path), None)
    side.plugin(0)
    assert side._ambient["launch_bounds"] == (128, 2) and len(_bundle_dirs(tmp_path)) == 1
