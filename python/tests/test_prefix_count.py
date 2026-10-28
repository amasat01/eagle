# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A run-time sample count: ``bind(..., capacity=n_max)`` of a prefix kernel (``IndexSet(prefix=True)``) and
``set_count(m)``. One bind and one captured graph run any count ``m <= n_max``; only the one-element count
plane changes, by a stream-ordered write, so nothing is rebuilt, rebound or recaptured. Lanes at or past the
count keep their outputs; a sum adds the live lanes only; a count above the capacity is refused by name."""

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
hawk_ext = pytest.importorskip("hawk.ext")
if "prefix" not in getattr(getattr(hawk_ext, "IndexSet", None), "__dataclass_fields__", {}):
    pytest.skip("this hawk has no IndexSet(prefix=True)", allow_module_level=True)
import eagle  # noqa: E402
import eagle.exec as eexec  # noqa: E402
import eagle.plan  # noqa: E402
from eagle import _split  # noqa: E402
from hawk import Kernel, Mutable, Scalar  # noqa: E402
from hawk.artifact import build_bundle, plugins  # noqa: E402
from hawk.diff import vjp  # noqa: E402

PRE = hawk_ext.Kind("prefix", lanes=hawk_ext.IndexSet(prefix=True))
N = 200
COUNTS = (N, 77, 1, 0)
TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


def _square(x: Scalar, y: Mutable[Scalar]):
    y = x * x + 1.0


def _loss(x: Scalar, total: Mutable[Scalar]):
    total = hawk.sum(x * x)


def _share(x: Scalar, total: Mutable[Scalar], share: Mutable[Scalar]):
    s = hawk.sum(x * x)         # read back below, so the kernel runs in two stages with the sum between
    total = s
    share = x * x / s


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp("prefix_cache"))


def _xp(target):
    return np if target == "host" else pytest.importorskip("cupy")


def _host(a):
    return np.array(a.get() if hasattr(a, "get") else a)


def _plan(kernels, tmp_path, cache, target, name):
    bundle = build_bundle(kernels, tmp_path, targets=("cuda",) if target == "cuda" else ("host",), cache_dir=cache,
                          host_profile="native", kind=PRE)
    by = plugins(bundle, device_loader=eagle.registry.load_manifest if target == "cuda" else None)
    _split.link(by, bundle)
    return eagle.plan.plan(by[name], structure=eexec.DeviceKernel if target == "cuda" else eexec.HostTeam)


def _fresh(plan_, planes, m):
    """The planes after one launch of a bind of their own at count ``m``."""
    xp = np if isinstance(planes["x"], np.ndarray) else pytest.importorskip("cupy")
    own = {k: v.copy() for k, v in planes.items()}
    bound = plan_.bind(active_count=xp.asarray(np.array([m], np.uint32)), **own)
    bound.launch()
    return own


@pytest.mark.parametrize("target", TARGETS)
def test_one_bind_and_one_graph_run_every_count_equal_to_a_fresh_run(tmp_path, cache, target):
    plan_ = _plan([hawk.kernel(_square, kind=PRE)], tmp_path, cache, target, "_square")
    xp = _xp(target)
    x, prior = np.random.default_rng(1).normal(size=N), np.linspace(-5, -4, N)
    planes = {"x": xp.asarray(x), "y": xp.asarray(prior)}
    bound = plan_.bind(capacity=N, **planes)
    graph = eagle.Graph().add(bound, name="square").build()
    for m in COUNTS:
        bound.set_count(m)
        planes["y"][...] = xp.asarray(prior)
        graph.launch()
        got = _host(planes["y"])
        want = _host(_fresh(plan_, {"x": xp.asarray(x), "y": xp.asarray(prior)}, m)["y"])
        assert got.tobytes() == want.tobytes()
        np.testing.assert_array_equal(got[m:], prior[m:])               # lanes past the count are untouched
        np.testing.assert_allclose(got[:m], x[:m] * x[:m] + 1.0, rtol=1e-15)


@pytest.mark.parametrize("target", TARGETS)
def test_a_sum_adds_the_live_lanes_only_at_every_count(tmp_path, cache, target):
    plan_ = _plan([hawk.kernel(_share, kind=PRE)], tmp_path, cache, target, "_share")
    xp = _xp(target)
    x = np.random.default_rng(2).normal(size=N)
    planes = {"x": xp.asarray(x), "total": xp.zeros(1), "share": xp.full(N, 7.0)}
    bound = plan_.bind(capacity=N, **planes)
    graph = eagle.Graph().add(bound, name="loss").build()
    for m in COUNTS:
        bound.set_count(m)
        planes["total"][...] = 0
        planes["share"][...] = 7.0
        graph.launch()
        np.testing.assert_allclose(_host(planes["total"])[0], float(x[:m] @ x[:m]), rtol=1e-13, atol=0)
        if m:
            np.testing.assert_allclose(_host(planes["share"])[:m], x[:m] ** 2 / (x[:m] @ x[:m]), rtol=1e-12)
        assert (_host(planes["share"])[m:] == 7.0).all()


@pytest.mark.parametrize("target", TARGETS)
def test_the_vjp_at_a_count_covers_the_live_lanes_only(tmp_path, cache, target):
    loss = hawk.kernel(_loss, kind=PRE)
    grad = Kernel("loss_grad", vjp(loss, wrt=("x",)), {}, kind=PRE)
    plan_ = _plan([loss, grad], tmp_path, cache, target, "loss_grad")
    xp = _xp(target)
    x = np.random.default_rng(3).normal(size=N)
    planes = {"x": xp.asarray(x), "bar_total": 1.5, "bar_x": xp.zeros(N)}
    bound = plan_.bind(capacity=N, **planes)
    for m in COUNTS:
        bound.set_count(m)
        planes["bar_x"][...] = 7.0
        bound.launch()
        got = _host(planes["bar_x"])
        np.testing.assert_allclose(got[:m], 2 * 1.5 * x[:m], rtol=1e-14)
        assert (got[m:] == 7.0).all()


@pytest.mark.parametrize("target", TARGETS)
def test_a_count_above_the_capacity_is_refused_by_name(tmp_path, cache, target):
    plan_ = _plan([hawk.kernel(_square, kind=PRE)], tmp_path, cache, target, "_square")
    xp = _xp(target)
    planes = {"x": xp.zeros(N), "y": xp.zeros(N)}
    bound = plan_.bind(capacity=120, **planes)
    bound.set_count(120)
    with pytest.raises(ValueError, match=r"count 121 is above this bind's capacity 120"):
        bound.set_count(121)
    with pytest.raises(ValueError, match=r"capacity=300 is above the 200 lanes"):
        plan_.bind(capacity=300, **planes)
    plain = plan_.bind(active_count=xp.asarray(np.array([5], np.uint32)), **planes)
    with pytest.raises(ValueError, match=r"not bound with capacity="):
        plain.set_count(1)


def test_capacity_is_for_a_prefix_kernel_only(tmp_path, cache):
    @hawk.kernel
    def plain(x: Scalar, y: Mutable[Scalar]):
        y = x + 1.0

    bundle = build_bundle([plain], tmp_path, targets=("host",), cache_dir=cache, host_profile="native")
    by = plugins(bundle)
    with pytest.raises(ValueError, match=r"capacity= is for a kernel built with IndexSet\(prefix=True\)"):
        eagle.plan.plan(by["plain"], structure=eexec.HostTeam).bind(capacity=4, x=np.zeros(4), y=np.zeros(4))
