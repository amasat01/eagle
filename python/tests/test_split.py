# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A kernel split in stages (it reads a reduction over samples after computing it):
every eagle door that can sequence the stages runs them through one helper, and
a door that launches one kernel per step refuses the kernel by name.

The bundles here are assembled by hand: each stage is a kernel of its own, the
finished cell of one is the one-cell ``lookup`` plane of the next, and the
``stages`` entry is the one hawk writes into the sidecar of a split kernel."""

import numpy as np
import pytest

pytest.importorskip("hawk")
import hawk.diff  # noqa: E402
import eagle  # noqa: E402
import eagle.exec as eexec  # noqa: E402
import eagle.diff  # noqa: E402
import eagle.plan  # noqa: E402
from eagle import _cell_rows, _split  # noqa: E402
from hawk import Mutable, Param, Scalar  # noqa: E402
from hawk.artifact import build_bundle, plugins  # noqa: E402

SIZES = (1, 31, 33, 1000, 65536)


@hawk.kernel
def one_sum(x: Scalar, t: Mutable[Scalar]):
    t = hawk.sum(x)


@hawk.kernel
def divide(x: Scalar, t: Param, z: Mutable[Scalar]):
    z = x / t


@hawk.kernel
def weighted_sum(x: Scalar, t: Param, u: Mutable[Scalar]):
    u = hawk.sum(x * t)


@hawk.kernel
def divide_u(x: Scalar, u: Param, z: Mutable[Scalar]):
    z = x / u


def _bundle(tmp_path, mode, stages, targets=("host",), internal=()):
    """``stages``: ``[(kernel, cells_in, cells_out)]`` -> ``(plugin of stage 1, by name)``;
    ``internal``: the cells the first sidecar marks as the kernel's own."""
    names = ["split"] + [f"split_{k + 1}" for k in range(1, len(stages))]
    kernels = [hawk.Kernel(nm, (hawk.diff.param_planes(k) if cin else k).sinks)
               for nm, (k, cin, _out) in zip(names, stages)]
    cache = str(tmp_path / "cache")
    bundle = build_bundle(kernels, tmp_path / "bundle", targets=targets, mode=mode,
                          cache_dir=cache, host_profile="native")
    by = plugins(bundle, device_loader=eagle.registry.load_manifest if "cuda" in targets else None)
    for art in bundle.artifacts:
        if art.name == "split":
            art.sidecar["stages"] = [{"kernel": nm, "cells_in": list(cin), "cells_out": list(out)}
                                     for nm, (_k, cin, out) in zip(names, stages)]
            art.sidecar["cells"] = {c: {**spec, **({"internal": True} if c in internal else {})}
                                    for c, spec in art.sidecar["cells"].items()}
    _split.link(by, bundle)
    return by["split"], by


def _host(a):
    """``a`` as a numpy array (a cupy one copied home)."""
    return np.asarray(a.get() if hasattr(a, "get") and not isinstance(a, dict) else a)


def _x(n, mode):
    return np.random.default_rng(n).uniform(1.0, 2.0, n).astype(mode)


def _tol(mode):
    return 1e-12 if mode == "float64" else 5e-5


@pytest.fixture(scope="module", params=["float64", "float32"])
def two(request, tmp_path_factory):
    plugin, _ = _bundle(tmp_path_factory.mktemp(request.param), request.param,
                        [(one_sum, (), ("t",)), (divide, ("t",), ())])
    return plugin, request.param


@pytest.fixture(scope="module", params=["float64", "float32"])
def three(request, tmp_path_factory):
    plugin, _ = _bundle(tmp_path_factory.mktemp(request.param), request.param,
                        [(one_sum, (), ("t",)), (weighted_sum, ("t",), ("u",)),
                         (divide_u, ("u",), ())])
    return plugin, request.param


@pytest.mark.parametrize("n", SIZES)
def test_two_stages_divide_by_the_total_through_plan_run(two, n):
    plugin, mode = two
    assert len(_split.stages_of(plugin)) == 2
    x = _x(n, mode)
    got = eagle.plan.plan(plugin, structure=eexec.HostTeam).run(x=x)
    np.testing.assert_allclose(got["z"], x / x.astype(np.float64).sum(), rtol=_tol(mode))
    assert got["t"].shape == (1,)


@pytest.mark.parametrize("n", SIZES)
def test_three_stages_chain_two_cells(three, n):
    plugin, mode = three
    assert len(_split.stages_of(plugin)) == 3
    x = _x(n, mode).astype(np.float64)
    x = x.astype(mode)
    t = x.astype(np.float64).sum()
    u = (x.astype(np.float64) * t).sum()
    got = eagle.plan.plan(plugin, structure=eexec.HostTeam).run(x=x)
    np.testing.assert_allclose(got["z"], x / u, rtol=10 * _tol(mode))


def test_scratch_of_every_stage_is_made_once(two):
    plugin, mode = two
    p = eagle.plan.plan(plugin, structure=eexec.HostTeam)
    p.run(x=_x(100, mode))
    before = _cell_rows.made
    for _ in range(3):
        p.run(x=_x(100, mode))
    assert _cell_rows.made == before


def test_bind_runs_every_stage_and_finishes_between(two):
    plugin, mode = two
    p = eagle.plan.plan(plugin, structure=eexec.HostTeam)
    x, z, t = _x(100, mode), np.zeros(100, mode), np.zeros(1, mode)
    bound = p.bind(x=x, z=z, t=t)
    assert len(bound.stages) == 2 and set(bound.names) == {"x", "z", "t"}
    bound.launch()
    np.testing.assert_allclose(z, x / x.astype(np.float64).sum(), rtol=_tol(mode))
    assert t[0] == p.run(x=x)["t"][0]       # the caller's own cell, finished in place


def test_a_cell_the_sidecar_marks_internal_is_not_handed_back(tmp_path):
    plugin, _ = _bundle(tmp_path, "float64", [(one_sum, (), ("t",)), (divide, ("t",), ())],
                        internal=("t",))
    p = eagle.plan.plan(plugin, structure=eexec.HostTeam)
    x = _x(40, "float64")
    np.testing.assert_allclose(p.run(x=x), x / x.sum(), rtol=1e-12)
    assert p.bind(x=x, z=np.zeros(40)).names == ("x", "z")


def test_doors_that_launch_one_kernel_entry_refuse_a_split_plan_by_name(two):
    from eagle import _dispatch_door

    plugin, mode = two
    bound = eagle.plan.plan(plugin, structure=eexec.HostTeam).bind(
        x=_x(10, mode), z=np.zeros(10, mode), t=np.zeros(1, mode))
    want = r"'split' runs in 2 stages; eagle dispatch timing runs single-stage kernels"
    with pytest.raises(ValueError, match=want):
        _dispatch_door.time_launch(bound, 1)
    with pytest.raises(ValueError, match="'split' runs in 2 stages"):
        bound._entry


def test_bind_refuses_the_planes_eagle_owns(two):
    plugin, mode = two
    p = eagle.plan.plan(plugin, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=r"\['t_comp'\] are not planes of kernel 'split'"):
        p.bind(x=_x(10, mode), z=np.zeros(10, mode), t_comp=np.zeros(1, mode))


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 33, 65536])
def test_two_stages_on_cuda(tmp_path, n):
    cp = pytest.importorskip("cupy")
    plugin, _ = _bundle(tmp_path, "float64", [(one_sum, (), ("t",)), (divide, ("t",), ())],
                        targets=("cuda",))
    x = _x(n, "float64")
    got = eagle.plan.plan(plugin, structure=eexec.DeviceKernel).run(x=cp.asarray(x))
    np.testing.assert_allclose(_host(got["z"]), x / x.sum(), rtol=1e-12)


# --- the kernels hawk splits itself: the same doors over its own bundles --- #
@hawk.kernel
def over_total(x: Scalar, z: Mutable[Scalar]):
    z = x / hawk.sum(x)


@hawk.kernel
def chain(x: Scalar, z: Mutable[Scalar]):
    s = hawk.sum(x)
    u = hawk.sum(x * s)
    z = x / u


@pytest.mark.parametrize("mode", ["float64", "float32"])
@pytest.mark.parametrize("n", SIZES)
def test_hawk_split_one_cut_through_plan_run(tmp_path, mode, n):
    (p,) = eagle.plan.auto([over_total], targets=("host",), cache_dir=str(tmp_path), scalar_type=mode)
    assert len(_split.stages_of(p.plugin)) == 2
    x = _x(n, mode)
    got = p.run(x=x)
    assert not isinstance(got, dict)    # the cell hawk made for the cut is not handed back
    assert p.plugin._internal == {"cut0"}   # read from the sidecar's marker
    np.testing.assert_allclose(got,
                               x / x.astype(np.float64).sum(), rtol=_tol(mode))


@pytest.mark.parametrize("mode", ["float64", "float32"])
@pytest.mark.parametrize("n", SIZES)
def test_hawk_split_three_stages_through_plan_run(tmp_path, mode, n):
    (p,) = eagle.plan.auto([chain], targets=("host",), cache_dir=str(tmp_path), scalar_type=mode)
    assert len(_split.stages_of(p.plugin)) == 3
    x = _x(n, mode)
    s = x.astype(np.float64).sum()
    got = p.run(x=x)
    np.testing.assert_allclose(got["z"] if isinstance(got, dict) else got,
                               x / (x.astype(np.float64) * s).sum(), rtol=10 * _tol(mode))


@pytest.mark.parametrize("mode", ["float64", "float32"])
@pytest.mark.parametrize("n", SIZES)
def test_hawk_split_one_cut_through_bind(tmp_path, mode, n):
    (p,) = eagle.plan.auto([over_total], targets=("host",), cache_dir=str(tmp_path), scalar_type=mode)
    x, z = _x(n, mode), np.zeros(n, mode)
    bound = p.bind(x=x, z=z)
    assert bound.names == ("x", "z")        # the cell between the stages is eagle's
    before = _cell_rows.made
    for _ in range(2):                      # a launch allocates nothing and repeats exactly
        bound.launch()
        np.testing.assert_allclose(z, x / x.astype(np.float64).sum(), rtol=_tol(mode))
    assert _cell_rows.made == before
    np.testing.assert_array_equal(z, p.run(x=x))


def test_rebind_reaches_every_stage_that_reads_the_plane(tmp_path):
    (p,) = eagle.plan.auto([chain], targets=("host",), cache_dir=str(tmp_path))
    x, z = _x(100, "float64"), np.zeros(100)
    bound = p.bind(x=np.zeros(100), z=z)
    bound.rebind(x=x).launch()
    np.testing.assert_allclose(z, x / (x * x.sum()).sum(), rtol=1e-12)


def test_hawk_split_three_stages_in_a_graph(tmp_path):
    (p,) = eagle.plan.auto([chain], targets=("host",), cache_dir=str(tmp_path))
    x, z = _x(1000, "float64"), np.zeros(1000)
    eagle.Graph().add(p.bind(x=x, z=z), name="chain").build().launch()
    np.testing.assert_allclose(z, x / (x * x.sum()).sum(), rtol=1e-11)


@hawk.kernel
def weighted(x: Scalar, w: Param, z: Mutable[Scalar]):
    z = w * x / hawk.sum(x * x)


def _reverse_vs_fd(tmp_path, targets, xp):
    n, w = 200, 0.7
    x = _x(n, "float64")
    seed = np.random.default_rng(5).uniform(-1.0, 1.0, n)
    (p,) = eagle.plan.auto([weighted], targets=targets, cache_dir=str(tmp_path))
    g = eagle.Graph().add(p.bind(x=xp.asarray(x), w=w, z=xp.zeros(n)), name="s")
    g = g.build(record=True)
    g.launch()
    rev = eagle.diff.Reverse(g)
    assert len(_split.stages_of(rev._vjp[0].plugin)) >= 2     # the vjp is split too
    grads = rev(g.record, {"z": xp.asarray(seed)})

    def loss(wv, xv):
        return seed @ (wv * xv / (xv * xv).sum())

    h = 1e-6
    fd_w = (loss(w + h, x) - loss(w - h, x)) / (2 * h)
    fd_x = np.array([(loss(w, x + h * e) - loss(w, x - h * e)) / (2 * h) for e in np.eye(n)])
    np.testing.assert_allclose(float(_host(grads["w"]).reshape(-1)[0]), fd_w, rtol=1e-6)
    np.testing.assert_allclose(_host(grads["x"]), fd_x, rtol=1e-6, atol=1e-12)


def test_reverse_runs_the_split_vjp_and_matches_central_differences(tmp_path):
    _reverse_vs_fd(tmp_path, ("host",), np)


@pytest.mark.gpu
def test_reverse_runs_the_split_vjp_on_cuda(tmp_path):
    cp = pytest.importorskip("cupy")
    _reverse_vs_fd(tmp_path, ("cuda",), cp)


def test_hawk_split_kernel_is_refused_by_the_doors_that_step(tmp_path):
    # hawk refuses a cut in a stepping kernel, so these doors meet a split kernel only
    # as a prebuilt plan; their step loop launches one kernel per step, so they refuse it
    (p,) = eagle.plan.auto([over_total], targets=("host",), cache_dir=str(tmp_path))
    x = _x(8, "float64")
    with pytest.raises(ValueError, match=r"eagle\.until_done: kernel 'over_total' is split in 2"):
        eagle.until_done(p, max_steps=2, x=x, z=np.zeros(8))
    with pytest.raises(ValueError, match=r"eagle\.simulate: kernel 'over_total' is split in 2"):
        eagle.simulate(p, x=x, max_steps=1)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 33, 65536])
def test_hawk_split_chain_on_cuda(tmp_path, n):
    cp = pytest.importorskip("cupy")
    (p,) = eagle.plan.auto([chain], targets=("cuda",), cache_dir=str(tmp_path))
    x = _x(n, "float64")
    got = p.run(x=cp.asarray(x))
    np.testing.assert_allclose(_host(got["z"] if isinstance(got, dict) else got),
                               x / (x * x.sum()).sum(), rtol=1e-11)
    z = cp.zeros(n)
    eagle.Graph().add(p.bind(x=cp.asarray(x), z=z)).build().launch()
    np.testing.assert_allclose(cp.asnumpy(z), x / (x * x.sum()).sum(), rtol=1e-11)


def test_torch_bridge_runs_the_split_kernel_on_the_host(tmp_path):
    torch = pytest.importorskip("torch")
    from eagle.frameworks import torch as eagle_torch

    K = eagle_torch.function(over_total, cache_dir=tmp_path)
    x = torch.tensor(_x(33, "float64"))
    np.testing.assert_allclose(K(x).detach().numpy(), (x / x.sum()).numpy(), rtol=1e-12)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_jax_bridge_chains_the_stages(tmp_path, jit):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from eagle.frameworks import jax as eagle_jax

    jax.config.update("jax_enable_x64", True)
    K = eagle_jax.function(over_total, wrt=(), cache_dir=tmp_path)
    x = _x(33, "float64")
    got = (jax.jit(K) if jit else K)(jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(got), x / x.sum(), rtol=1e-12)


@hawk.kernel
def total(x: Scalar, t: Mutable[Scalar]):
    t = hawk.sum(x * x)


@hawk.kernel
def weighted_total(x: Scalar, w: Param, z: Mutable[Scalar]):
    z = w * x / hawk.sum(x * x)


@pytest.fixture
def jax_x64():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_jax_grad_through_a_cell_output(tmp_path, jax_x64, jit):
    # the vjp reads only the cell's cotangent: it is told the sample count
    jax, jnp = jax_x64, pytest.importorskip("jax.numpy")
    from eagle.frameworks import jax as eagle_jax

    K = eagle_jax.function(total, cache_dir=tmp_path)
    grad = jax.grad(lambda a: 3.0 * K(a)[0])
    x = _x(65, "float64")
    np.testing.assert_allclose(np.asarray((jax.jit(grad) if jit else grad)(jnp.asarray(x))),
                               6.0 * x, rtol=1e-12)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_jax_grad_through_a_split_kernel_matches_central_differences(tmp_path, jax_x64, jit):
    jax, jnp = jax_x64, pytest.importorskip("jax.numpy")
    from eagle.frameworks import jax as eagle_jax

    n, w = 65, 0.7
    x, c = _x(n, "float64"), np.random.default_rng(9).uniform(-1.0, 1.0, n)
    K = eagle_jax.function(weighted_total, cache_dir=tmp_path)
    grad = jax.grad(lambda a, wv: jnp.sum(K(a, wv) * c), argnums=(0, 1))
    gx, gw = (jax.jit(grad) if jit else grad)(jnp.asarray(x), w)

    def loss(xv, wv):
        return c @ (wv * xv / (xv * xv).sum())

    h = 1e-6
    fd_x = np.array([(loss(x + h * e, w) - loss(x - h * e, w)) / (2 * h) for e in np.eye(n)])
    np.testing.assert_allclose(np.asarray(gx), fd_x, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(float(gw), (loss(x, w + h) - loss(x, w - h)) / (2 * h), rtol=1e-6)


@hawk.kernel(jump="auto")
def switch(x: Scalar, t: Param, a: Param, b: Param, y: Mutable[Scalar]):
    if x < t:
        y = a
    else:
        y = b


def test_jax_grad_through_a_jump_auto_switch(tmp_path, jax_x64):
    import math

    jax, jnp = jax_x64, pytest.importorskip("jax.numpy")
    from eagle.frameworks import jax as eagle_jax

    n, A, B, T = 257, -0.5, 0.8, 0.4
    x = np.random.default_rng(1).uniform(-1.0, 1.0, n)
    c = np.random.default_rng(2).uniform(-1.0, 1.0, n)
    K = eagle_jax.function(switch, cache_dir=tmp_path)
    gt, ga = jax.grad(lambda t, a: jnp.sum(K(jnp.asarray(x), t, a, B) * c), argnums=(0, 1))(T, A)
    width = math.sqrt(5.0) * 0.02 * math.sqrt(12.0) * np.std(x - T)
    kern = np.where(np.abs(x - T) < width, 0.75 / width * (1.0 - ((x - T) / width) ** 2), 0.0)
    np.testing.assert_allclose(float(gt), (c * (A - B) * kern).sum(), rtol=1e-10)
    np.testing.assert_allclose(float(ga), c[x < T].sum(), rtol=1e-12)


def test_warp_bridge_runs_the_split_kernel_on_the_host(tmp_path):
    wp = pytest.importorskip("warp")
    from eagle.frameworks import warp as eagle_warp

    W = eagle_warp.function(over_total, cache_dir=tmp_path)
    x = _x(33, "float64")
    got = W(wp.array(x, dtype=wp.float64, device="cpu"))
    np.testing.assert_allclose(got.numpy(), x / x.sum(), rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("bridge", ["torch", "jax", "warp"])
def test_bridges_run_the_split_kernel_on_cuda(tmp_path, bridge):
    x = _x(1000, "float64")
    want = x / x.sum()
    if bridge == "torch":
        torch = pytest.importorskip("torch")
        from eagle.frameworks import torch as eagle_torch

        got = eagle_torch.function(over_total, cache_dir=tmp_path)(torch.tensor(x, device="cuda"))
        got = got.detach().cpu().numpy()
    elif bridge == "jax":
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        from eagle.frameworks import jax as eagle_jax

        try:
            gpu = jax.devices("gpu")[0]
        except RuntimeError:
            pytest.skip("this jax has no CUDA platform")
        jax.config.update("jax_enable_x64", True)
        K = eagle_jax.function(over_total, wrt=(), cache_dir=tmp_path)
        got = np.asarray(jax.jit(K)(jax.device_put(jnp.asarray(x), gpu)))
    else:
        wp = pytest.importorskip("warp")
        from eagle.frameworks import warp as eagle_warp

        got = eagle_warp.function(over_total, cache_dir=tmp_path)(
            wp.array(x, dtype=wp.float64, device="cuda:0")).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_a_placement_only_stage_passes_the_stepping_doors_and_a_real_split_does_not():
    # a kernel whose only cut is its wide-value scratch is one launch: the stepping doors run it
    from types import SimpleNamespace

    from eagle import _split

    def plugin(stages, team_planes):
        return SimpleNamespace(name="k", _stages=stages, _team_planes=team_planes)

    placed = _split.Stage(plugin=None, cells_in=(), cells_out=())
    _split.refuse(plugin((placed,), {"hwd0": {}}), "eagle.simulate")
    for stages, planes in (((placed, placed), {"hwd0": {}}),
                           ((_split.Stage(plugin=None, cells_in=("c",), cells_out=()),), {"hwd0": {}}),
                           ((_split.Stage(plugin=None, cells_in=(), cells_out=(), team={"n": 32}),), {"hwd0": {}}),
                           ((placed,), {})):
        with pytest.raises(ValueError, match=r"eagle\.simulate: kernel 'k' is split in"):
            _split.refuse(plugin(stages, planes), "eagle.simulate")
