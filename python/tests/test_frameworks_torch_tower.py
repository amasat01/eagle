# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.frameworks.torch`` beyond first order: the generic launch op,
``torch.compile``, derivatives of derivatives, ``torch.func``, float32,
Params read on the device, threads, and CUDA-graph capture.

Every row runs on the CPU and, marked ``gpu``/``torch_gpu``, on CUDA. The
subject is the same drag law as ``test_frameworks_torch.py``; values are
judged against the law written in plain torch (autograd as the independent
reference) and against central differences of the bridge's own first
derivative.
"""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch",
    reason="torch is not installed here (torch has free-threaded CPU wheels for 3.14t only (none for 3.13t))")
hawk = pytest.importorskip("hawk")

from hawk import Mutable, Param, Scalar  # noqa: E402
from test_frameworks_torch import (  # noqa: E402
    DEVICES,
    _device,
    _inputs,
    _reference,
    drag,
    step_and_finish,
    torch_internal_jit,
)

from eagle.frameworks import torch as eagle_torch  # noqa: E402

HAS_OP = hasattr(torch.library, "custom_op")
needs_op = pytest.mark.skipif(not HAS_OP, reason="the generic op needs torch 2.4")
MODES = ["float64", "float32"]
DTYPE = {"float64": torch.float64, "float32": torch.float32}
TOL = {"float64": dict(rtol=1e-10, atol=1e-12), "float32": dict(rtol=2e-4, atol=1e-5)}


@hawk.kernel
def spin(x: Scalar, g: Param, y: Mutable[Scalar]):
    """A Param read inside a loop body."""
    s = x
    for _k in range(4):
        s = hawk.math.sin(g * s) + s * 1.2
    y = s  # noqa: F841 (DSL output)


@pytest.fixture(scope="module")
def op(tmp_path_factory):
    return eagle_torch.function(drag, cache_dir=tmp_path_factory.mktemp("tower_fn"))


@pytest.fixture(scope="module")
def op_loop(tmp_path_factory):
    return eagle_torch.function(spin, cache_dir=tmp_path_factory.mktemp("tower_loop"))


def _case(device, mode, grad=True):
    v, cd, terminated = _inputs(device)
    dtype = DTYPE[mode]
    v, cd = v.to(dtype), cd.to(dtype)
    if grad:
        v.requires_grad_()
        cd.requires_grad_()
    return v, cd, terminated


def _loss(fn):
    return lambda v, cd, t: (fn(v, cd, t) ** 2).sum()


# --------------------------------------------------------------------------- #
# the generic op
# --------------------------------------------------------------------------- #
@needs_op
def test_one_generic_op_serves_every_kernel(op, tmp_path):
    first = torch.ops.eagle.launch
    other = eagle_torch.function(step_and_finish, cache_dir=tmp_path)
    assert op is not other and torch.ops.eagle.launch is first
    assert [
        n for n in dir(torch.ops.eagle) if n not in ("name",) and not n.startswith("_")
    ] == ["launch"]


@needs_op
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("mode", MODES)
def test_fake_shapes_match_the_real_launch(op, dev, mode):
    from torch._subclasses.fake_tensor import FakeTensorMode

    device = _device(dev)
    v, cd, terminated = _case(device, mode, grad=False)
    cd = cd.reshape(1)
    real = torch.ops.eagle.launch(op._root.digest(mode), [v, cd, terminated])
    with FakeTensorMode() as fake:
        planes = [fake.from_tensor(t) for t in (v, cd, terminated)]
        out = torch.ops.eagle.launch(op._root.digest(mode), planes)
    assert [(tuple(o.shape), o.dtype, o.device.type) for o in out] == [
        (tuple(o.shape), o.dtype, o.device.type) for o in real
    ]


@torch_internal_jit
@needs_op
@pytest.mark.parametrize("dev", DEVICES)
def test_compile_forward_has_no_graph_break(op, dev):
    import torch._dynamo as dynamo

    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    explained = dynamo.explain(lambda v, cd: op(v, cd, terminated))(v, cd)
    assert explained.graph_break_count == 0, explained.break_reasons


@torch_internal_jit
@needs_op
@pytest.mark.parametrize("dev", DEVICES)
def test_compile_fullgraph_forward_and_backward_match_reference(op, dev):
    device = _device(dev)
    v, cd, terminated = _case(device, "float64")
    # inductor needs triton (sm70+); dynamo + AOTAutograd is the part eagle's op meets
    backend = "inductor" if dev == "cpu" else "aot_eager"
    compiled = torch.compile(
        lambda v, cd: _loss(op)(v, cd, terminated), fullgraph=True, backend=backend
    )
    value = compiled(v, cd)
    value.backward()
    v_ref, cd_ref = (
        v.detach().clone().requires_grad_(),
        cd.detach().clone().requires_grad_(),
    )
    expect = _loss(_reference)(v_ref, cd_ref, terminated)
    expect.backward()
    torch.testing.assert_close(value, expect, rtol=1e-12, atol=0)
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-11, atol=0)
    torch.testing.assert_close(cd.grad, cd_ref.grad, rtol=1e-11, atol=0)


@pytest.mark.skipif(HAS_OP, reason="below torch 2.4 only")
def test_compile_is_refused_by_name_below_torch_2_4(op, monkeypatch):
    """No ``custom_op`` below 2.4: the bridge runs its launches directly, and a
    call it finds under a compile trace is refused by name (torch 2.2 on Python
    3.12 cannot compile at all, so the trace is stood in for)."""
    v, cd, terminated = _case(torch.device("cpu"), "float64", grad=False)
    monkeypatch.setattr(eagle_torch, "_is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match=r"torch\.compile support needs torch 2\.4"):
        op(v, cd, terminated)


# --------------------------------------------------------------------------- #
# derivatives of derivatives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dev", DEVICES)
def test_hessian_matches_the_reference_law(op, dev):
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    got = torch.autograd.functional.hessian(
        lambda v, cd: _loss(op)(v, cd, terminated), (v, cd)
    )
    expect = torch.autograd.functional.hessian(
        lambda v, cd: _loss(_reference)(v, cd, terminated), (v, cd)
    )
    for i in range(2):
        for j in range(2):
            torch.testing.assert_close(got[i][j], expect[i][j], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dev", DEVICES)
def test_create_graph_double_backward_matches_the_reference_law(op, dev):
    v, cd, terminated = _case(_device(dev), "float64")
    (gv,) = torch.autograd.grad(_loss(op)(v, cd, terminated), v, create_graph=True)
    (gcd,) = torch.autograd.grad((gv**2).sum(), cd)
    v_ref, cd_ref = (
        v.detach().clone().requires_grad_(),
        cd.detach().clone().requires_grad_(),
    )
    (rv,) = torch.autograd.grad(
        _loss(_reference)(v_ref, cd_ref, terminated), v_ref, create_graph=True
    )
    (rcd,) = torch.autograd.grad((rv**2).sum(), cd_ref)
    torch.testing.assert_close(gv, rv, rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(gcd, rcd, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradgradcheck(op, dev):
    v, cd, terminated = _case(_device(dev), "float64")
    assert torch.autograd.gradgradcheck(
        lambda v, cd: op(v, cd, terminated), (v, cd), check_batched_grad=False
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_third_derivative_matches_the_reference_law(op, dev):
    v, cd, terminated = _case(_device(dev), "float64")

    def third(fn, v, cd):
        (g1,) = torch.autograd.grad(_loss(fn)(v, cd, terminated), cd, create_graph=True)
        (g2,) = torch.autograd.grad(g1, cd, create_graph=True)
        (g3,) = torch.autograd.grad(g2, cd)
        return g3

    torch.testing.assert_close(
        third(op, v, cd), third(_reference, v, cd), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_second_derivative_matches_a_central_difference_of_the_first(op, dev):
    """The reference here is the bridge's own FIRST derivative, differenced."""
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    v = v.detach()
    h = 1e-6

    def first(c):
        c = c.clone().requires_grad_()
        (g,) = torch.autograd.grad(_loss(op)(v, c, terminated), c)
        return g

    second = torch.autograd.functional.hessian(
        lambda c: _loss(op)(v, c, terminated), cd
    ).reshape(1)
    difference = (first(cd + h) - first(cd - h)) / (2 * h)
    torch.testing.assert_close(second, difference.reshape(1), rtol=1e-6, atol=1e-8)


@torch_internal_jit
@pytest.mark.parametrize("dev", DEVICES)
def test_hessian_vector_product_forward_over_reverse(op, dev):
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    direction = torch.ones_like(v)

    def grad_v(fn, v):
        return torch.func.grad(lambda v: _loss(fn)(v, cd, terminated))(v)

    got = torch.func.jvp(lambda v: grad_v(op, v), (v,), (direction,))[1]
    expect = torch.func.jvp(lambda v: grad_v(_reference, v), (v,), (direction,))[1]
    torch.testing.assert_close(got, expect, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dev", DEVICES)
def test_finishing_kernel_second_derivative(tmp_path, dev):
    """A kernel that finishes its own mask differentiates twice too (its
    second derivative is zero: the step is linear)."""
    device = _device(dev)
    finishing = eagle_torch.function(step_and_finish, cache_dir=tmp_path)
    k = torch.arange(4, dtype=torch.float64, device=device).requires_grad_()
    nstop = torch.tensor(9.0, dtype=torch.float64, device=device)
    mask = torch.zeros(4, dtype=torch.bool, device=device)
    (g,) = torch.autograd.grad(
        (finishing(k, nstop, mask)[0] ** 2).sum(), k, create_graph=True
    )
    (gg,) = torch.autograd.grad(g.sum(), k)
    torch.testing.assert_close(g, 2 * (k.detach() + 1))
    torch.testing.assert_close(gg, torch.full_like(k, 2.0))


# --------------------------------------------------------------------------- #
# forward AD and torch.func
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dev", DEVICES)
def test_func_grad_matches_the_reference_law(op, dev):
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    got = torch.func.grad(_loss(op), argnums=(0, 1))(v, cd, terminated)
    expect = torch.func.grad(_loss(_reference), argnums=(0, 1))(v, cd, terminated)
    torch.testing.assert_close(got, expect, rtol=1e-11, atol=1e-13)


@torch_internal_jit
@pytest.mark.parametrize("dev", DEVICES)
def test_func_jvp_matches_the_reference_law(op, dev):
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    direction = torch.ones_like(v)
    got = torch.func.jvp(lambda v: op(v, cd, terminated), (v,), (direction,))
    expect = torch.func.jvp(lambda v: _reference(v, cd, terminated), (v,), (direction,))
    torch.testing.assert_close(got, expect, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("dev", DEVICES)
def test_func_vmap_folds_the_batch_into_the_sample_axis(op, dev):
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    batch = torch.stack([v, 2 * v, -v])
    got = torch.func.vmap(lambda v: op(v, cd, terminated))(batch)
    expect = torch.stack([_reference(b, cd, terminated) for b in batch])
    torch.testing.assert_close(got, expect, rtol=1e-12, atol=0)
    # the batch on the last axis, and a batched mask
    last = torch.func.vmap(lambda v: op(v, cd, terminated), in_dims=-1, out_dims=-1)(
        batch.movedim(0, -1)
    )
    torch.testing.assert_close(last, expect.movedim(0, -1), rtol=1e-12, atol=0)


@pytest.mark.parametrize("dev", DEVICES)
def test_func_vmap_over_a_batched_param_loops(op, dev):
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    cds = torch.stack([cd, 2 * cd, 3 * cd])
    got = torch.func.vmap(lambda c: op(v, c, terminated))(cds)
    expect = torch.stack([_reference(v, c, terminated) for c in cds])
    torch.testing.assert_close(got, expect, rtol=1e-12, atol=0)


@torch_internal_jit
@pytest.mark.parametrize("dev", DEVICES)
def test_func_vmap_of_grad_and_hessian(op, dev):
    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    batch = torch.stack([v, 2 * v])
    got = torch.func.vmap(torch.func.grad(lambda v: _loss(op)(v, cd, terminated)))(
        batch
    )
    expect = torch.func.vmap(
        torch.func.grad(lambda v: _loss(_reference)(v, cd, terminated))
    )(batch)
    torch.testing.assert_close(got, expect, rtol=1e-11, atol=1e-13)
    hess = torch.func.hessian(lambda c: _loss(op)(v, c, terminated))(cd)
    ref = torch.func.hessian(lambda c: _loss(_reference)(v, c, terminated))(cd)
    torch.testing.assert_close(hess, ref, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# float32
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dev", DEVICES)
def test_a_float32_call_selects_the_float32_build(tmp_path, monkeypatch, dev):
    import hawk.artifact

    seen = []
    real = hawk.artifact.build_bundle

    def spy(kernels, directory, **kw):
        seen.append(kw.get("mode", "float64"))
        return real(kernels, directory, **kw)

    monkeypatch.setattr(hawk.artifact, "build_bundle", spy)
    fresh = eagle_torch.function(drag, cache_dir=tmp_path)
    v, cd, terminated = _case(_device(dev), "float32", grad=False)
    out = fresh(v, cd, terminated)
    assert out.dtype == torch.float32 and seen == ["float32"]


@pytest.mark.parametrize("dev", DEVICES)
def test_float32_values_and_derivatives_match_float64(op, dev):
    device = _device(dev)
    v64, cd64, terminated = _case(device, "float64")
    v32, cd32, _ = _case(device, "float32")
    a64, a32 = op(v64, cd64, terminated), op(v32, cd32, terminated)
    assert a32.dtype == torch.float32
    torch.testing.assert_close(a32.double(), a64, **TOL["float32"])
    g64 = torch.autograd.grad(
        _loss(op)(v64, cd64, terminated), (v64, cd64), create_graph=True
    )
    g32 = torch.autograd.grad(
        _loss(op)(v32, cd32, terminated), (v32, cd32), create_graph=True
    )
    for x, y in zip(g32, g64):
        assert x.dtype == torch.float32
        torch.testing.assert_close(x.double(), y, **TOL["float32"])
    h64 = torch.autograd.grad(g64[1], v64)[0]
    h32 = torch.autograd.grad(g32[1], v32)[0]
    assert h32.dtype == torch.float32
    torch.testing.assert_close(h32.double(), h64, **TOL["float32"])


@pytest.mark.parametrize("dev", DEVICES)
def test_mixed_dtypes_promote_to_the_widest(op, dev):
    from eagle import PerformanceWarning

    v, cd, terminated = _case(_device(dev), "float64", grad=False)
    want = op(v, cd, terminated)
    for args in ((v.float(), cd, terminated), (v, cd.float(), terminated)):
        with pytest.warns(PerformanceWarning, match="promoted to float64"):
            got = op(*args)
        assert got.dtype == torch.float64
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_half_precision_is_refused_by_name(op):
    v, cd, terminated = _case(torch.device("cpu"), "float64", grad=False)
    with pytest.raises(ValueError, match="not supported"):
        op(v.half(), cd.half(), terminated)


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #
def test_every_kernel_reads_its_param_from_a_device_plane(op, op_loop):
    assert op.promoted is not None
    assert op_loop.promoted is not None  # a Param read inside a loop body too


@pytest.mark.parametrize("dev", DEVICES)
def test_a_promoted_param_is_bound_as_a_plane(op, dev, monkeypatch):
    device = _device(dev)
    v, cd, terminated = _case(device, "float64", grad=False)
    bound = {}
    if dev == "cpu":
        import hawk.runtime

        real = hawk.runtime.run

        def spy(kernel, **arrays):
            bound.update({k: getattr(a, "shape", None) for k, a in arrays.items()})
            return real(kernel, **arrays)

        monkeypatch.setattr(hawk.runtime, "run", spy)
    else:
        from eagle.plan import Plan

        real = Plan.bind

        def spy(self, /, **planes):
            bound.update({k: getattr(a, "shape", None) for k, a in planes.items()})
            return real(self, **planes)

        monkeypatch.setattr(Plan, "bind", spy)
    op(v, cd, terminated)
    assert bound["cd"] == (1,)


@pytest.mark.gpu
@pytest.mark.torch_gpu
def test_a_promoted_param_launches_and_differentiates_without_a_host_read(op):
    device = _device("cuda")
    v, cd, terminated = _case(device, "float64")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any host read raises
    try:
        out = op(v, cd, terminated)
        (gv, gcd) = torch.autograd.grad((out**2).sum(), (v, cd), create_graph=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    expect = torch.autograd.grad(_loss(_reference)(v, cd, terminated), (v, cd))
    torch.testing.assert_close(gv, expect[0], rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(gcd, expect[1], rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("dev", DEVICES)
def test_a_param_inside_a_loop_is_read_from_a_device_plane(op_loop, dev):
    device = _device(dev)
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.float64, device=device).requires_grad_()
    g = torch.tensor(0.7, dtype=torch.float64, device=device).requires_grad_()

    def law(x, g):
        s = x
        for _ in range(4):
            s = torch.sin(g * s) + s * 1.2
        return s

    out = op_loop(x, g)
    torch.testing.assert_close(out, law(x, g), rtol=1e-12, atol=0)
    got = torch.autograd.grad((out**2).sum(), (x, g))
    expect = torch.autograd.grad((law(x, g) ** 2).sum(), (x, g))
    torch.testing.assert_close(got, expect, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# threads: the first call of a fresh kernel, and many calls
# --------------------------------------------------------------------------- #
def _in_threads(count, work):
    barrier = threading.Barrier(count)
    errors, results = [], [None] * count

    def run(i):
        try:
            barrier.wait()
            results[i] = work(i)
        except BaseException as why:  # noqa: BLE001 - surfaced below
            errors.append(why)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    return results


@pytest.mark.ft
@pytest.mark.parametrize("dev", DEVICES)
def test_the_first_call_from_eight_threads_makes_one_level_each(tmp_path, dev):
    """Thread safety: eight threads make the FIRST (and first derivative)
    call of one fresh kernel at a barrier; one tower, no error, right values."""
    device = _device(dev)
    fresh = eagle_torch.function(drag, cache_dir=tmp_path)
    cases = [_case(device, "float64") for _ in range(8)]

    def work(i):
        v, cd, terminated = cases[i]
        out = fresh(v, cd, terminated)
        (g,) = torch.autograd.grad((out**2).sum(), cd, create_graph=True)
        (gg,) = torch.autograd.grad(g, cd)
        return out.detach(), g.detach(), gg.detach()

    results = _in_threads(8, work)
    for (v, cd, terminated), (out, g, gg) in zip(cases, results):
        v_ref, cd_ref = v.detach().clone(), cd.detach().clone().requires_grad_()
        ref = _reference(v_ref, cd_ref, terminated)
        (rg,) = torch.autograd.grad((ref**2).sum(), cd_ref, create_graph=True)
        (rgg,) = torch.autograd.grad(rg, cd_ref)
        torch.testing.assert_close(out, ref.detach(), rtol=1e-12, atol=0)
        torch.testing.assert_close(g, rg.detach(), rtol=1e-10, atol=1e-13)
        torch.testing.assert_close(gg, rgg.detach(), rtol=1e-10, atol=1e-13)
    assert set(fresh._root._children) == {"vjp", "jvp"}
    assert set(fresh._root.child("vjp")._children) == {"vjp"}


@pytest.mark.ft
@pytest.mark.parametrize("dev", DEVICES)
def test_fifty_calls_each_from_eight_threads(op, dev):
    device = _device(dev)

    def work(i):
        bad = 0
        for j in range(50):
            v, cd, terminated = _inputs(device, n=6, seed=1000 * i + j)
            bad += not torch.allclose(
                op(v, cd, terminated), _reference(v, cd, terminated), rtol=1e-12, atol=0
            )
        return bad

    assert _in_threads(8, work) == [0] * 8


# --------------------------------------------------------------------------- #
# CUDA graph capture
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.torch_gpu
def test_cuda_graph_capture_replays_the_bridge(op):
    device = _device("cuda")
    v, cd, terminated = _case(device, "float64", grad=False)
    op(v, cd, terminated)  # warm: build, load, allocator
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(v, cd, terminated)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(v, cd, terminated)
    for scale in (1.0, 2.5):
        v.copy_(_inputs(device)[0] * scale)
        cd.fill_(0.3 * scale)  # a Param changes with a memcpy, no re-capture
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, _reference(v, cd, terminated), rtol=1e-13, atol=0
        )


@pytest.mark.gpu
@pytest.mark.torch_gpu
def test_cuda_graph_capture_replays_a_gradient(op):
    device = _device("cuda")
    v, cd, terminated = _case(device, "float64")
    (op(v, cd, terminated) ** 2).sum().backward()  # warm every level
    v.grad = cd.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (op(v, cd, terminated) ** 2).sum().backward()
    for scale in (1.0, 2.0):
        with torch.no_grad():
            v.copy_(_inputs(device)[0] * scale)
            cd.fill_(0.3 * scale)
        graph.replay()
        torch.cuda.synchronize()
        v_ref, cd_ref = (
            v.detach().clone().requires_grad_(),
            cd.detach().clone().requires_grad_(),
        )
        _loss(_reference)(v_ref, cd_ref, terminated).backward()
        torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-11, atol=0)
        torch.testing.assert_close(cd.grad, cd_ref.grad, rtol=1e-11, atol=0)


@pytest.mark.gpu
@pytest.mark.torch_gpu
def test_cuda_graph_capture_replays_a_loop_kernel(op_loop):
    device = _device("cuda")
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.float64, device=device)
    g = torch.tensor(0.7, dtype=torch.float64, device=device)
    op_loop(x, g)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op_loop(x, g)
    g.fill_(0.4)
    graph.replay()
    torch.cuda.synchronize()
    expect = x.clone()
    for _ in range(4):
        expect = torch.sin(0.4 * expect) + expect * 1.2
    torch.testing.assert_close(out, expect, rtol=1e-12, atol=0)
