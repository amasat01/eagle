# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A skippable reverses under the guard value its launch saw, a repeat_while
under the iterations it ran; neither guard is evaluated in the reverse."""

import threading

import numpy as np
import pytest

import eagle
import eagle.plan
from eagle import SkipGuard, repeat_while, skippable
from _reverse_fixtures import central_difference, xp_for

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar  # noqa: E402
from hawk import math as m  # noqa: E402

N = 12


@hawk.kernel
def cv_square(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x * x


@hawk.kernel
def cv_sin(x: Scalar, y: Mutable[Scalar]):
    y = m.sin(x)


@hawk.kernel
def cv_step(x: Scalar, a: Param, c: Scalar, y: Mutable[Scalar], d: Mutable[Scalar]):
    y = x + 0.3 * a * m.sin(x)
    d = c - 1.0


def _plans(tmp_path, backend):
    targets = ("host", "cuda") if backend == "gpu" else ("host",)
    return eagle.plan.auto([cv_square, cv_sin, cv_step], targets=targets,
                           cache_dir=str(tmp_path / "hawk_cache"))


class Counting(SkipGuard):
    """A guard that counts how often anything reads it."""

    def __init__(self, *args):
        super().__init__(*args)
        self.calls = 0

    def evaluate(self):
        self.calls += 1
        return super().evaluate()


def _skip_graph(ps, x, a, *, flag, record=False, guard=None):
    xp = xp_for("gpu" if type(x).__module__.startswith("cupy") else "cpu")
    sq, sn, _ = ps
    y, z = xp.zeros_like(x), xp.zeros_like(x)
    guard = guard or SkipGuard(flag, 0)
    maybe = skippable(sq.bind(x=x, a=a, y=y).launch, guard)
    g = (eagle.Graph().add_node(maybe, name="maybe")
         .add(sn.bind(x=y, y=z), name="sin").build(record=record))
    return g, z


def _flag(backend, value):
    return xp_for(backend).asarray([value], dtype="uint32")


def _fd(ps, x0, a, flag, seed):
    return central_difference(lambda p, x_, a_: _skip_graph(p, x_, a_, flag=flag), ps, x0, a, seed)


def _check(grads, ps, x0, a, flag, seed):
    dx, da = _fd(ps, x0, a, flag, seed)
    np.testing.assert_allclose(grads["x"], dx, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(float(grads["a"]), da, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("on", [1, 0])
def test_skippable_matches_central_differences(tmp_path, backend, on):
    ps, xp = _plans(tmp_path, backend), xp_for(backend)
    x0, seed = np.linspace(-1.1, 0.9, N), np.linspace(1.0, 2.0, N)
    flag = _flag(backend, on)
    g, _z = _skip_graph(ps, xp.asarray(x0), 0.8, flag=flag, record=True)
    g.launch()
    grads = eagle.diff.reverse(g.record, {"sin.y": xp.asarray(seed)})
    _check({k: v.get() if hasattr(v, "get") else v for k, v in grads.items()}, ps, x0, 0.8,
           _flag("cpu", on), seed)
    if not on:
        assert not np.any(np.asarray(grads["x"].get() if hasattr(grads["x"], "get") else grads["x"]))


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_a_guard_that_flips_between_replays_and_is_never_called_in_the_reverse(tmp_path, backend):
    ps, xp = _plans(tmp_path, backend), xp_for(backend)
    x0, seed = np.linspace(-1.1, 0.9, N), np.linspace(1.0, 2.0, N)
    flag = _flag(backend, 1)
    guard = Counting(flag, 0)
    g, _z = _skip_graph(ps, xp.asarray(x0), 0.8, flag=flag, record=True, guard=guard)
    seeds = {"sin.y": xp.asarray(seed)}
    rev = eagle.diff.Reverse(g)
    for on in (1, 0, 1):
        flag[0] = on
        g.launch()
        seen = guard.calls
        grads = rev(g.record, seeds)
        assert guard.calls == seen
        _check({k: v.get() if hasattr(v, "get") else v for k, v in grads.items()}, ps, x0,
               0.8, _flag("cpu", on), seed)


def test_two_threads_reverse_two_records_of_one_graph(tmp_path):
    ps = _plans(tmp_path, "cpu")
    x0, seed = np.linspace(-1.1, 0.9, N), np.linspace(1.0, 2.0, N)
    flag = _flag("cpu", 1)
    g, _z = _skip_graph(ps, x0.copy(), 0.8, flag=flag, record=True)
    kept = {}
    for on in (1, 0):
        flag[0] = on
        g.launch()
        kept[on] = g.record.keep()
    out = {}

    def work(on):
        out[on] = [eagle.diff.Reverse(g)(kept[on], {"sin.y": seed}) for _ in range(5)]

    threads = [threading.Thread(target=work, args=(on,)) for on in (1, 0)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    for on in (1, 0):
        for grads in out[on]:
            _check(grads, ps, x0, 0.8, _flag("cpu", on), seed)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_n_reverses_of_one_conditional_record_agree_bitwise(tmp_path, backend):
    ps, xp = _plans(tmp_path, backend), xp_for(backend)
    x0, seed = np.linspace(-1.1, 0.9, N), np.linspace(1.0, 2.0, N)
    flag = _flag(backend, 1)
    g, _z = _skip_graph(ps, xp.asarray(x0), 0.8, flag=flag, record=True)
    kept = {}
    for on in (1, 0):
        flag[0] = on
        g.launch()
        kept[on] = g.record.keep()
    seeds = {"sin.y": xp.asarray(seed)}

    def bytes_of(grads):
        return {k: np.asarray(v.get() if hasattr(v, "get") else v).tobytes()
                for k, v in grads.items()}

    want = {on: bytes_of(eagle.diff.Reverse(g)(kept[on], seeds)) for on in (1, 0)}
    threads_n = 8
    out, errors = [None] * threads_n, []
    start = threading.Barrier(threads_n)

    def work(i):
        try:
            rev = eagle.diff.Reverse(g)
            start.wait()
            for _ in range(5):
                out[i] = bytes_of(rev(kept[i % 2], seeds))
        except BaseException as exc:      # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(threads_n)]
    [t.start() for t in threads]
    [t.join(120) for t in threads]
    assert not errors, errors
    assert out == [want[i % 2] for i in range(threads_n)]


def test_one_reverse_of_a_conditional_record_shared_by_two_threads_is_busy(tmp_path):
    from _reverse_block import Gate, block

    sq = _plans(tmp_path, "cpu")[0]
    x, y, w = np.linspace(0, 1, N), np.zeros(N), np.zeros(N)
    flag, gate = _flag("cpu", 1), Gate()
    maybe = skippable(sq.bind(x=x, a=0.8, y=y).launch, SkipGuard(flag, 0))
    g = eagle.Graph().add_node(maybe, name="maybe").add_node(block(y, w, gate)).build(record=True)
    g.launch()
    rev = eagle.diff.Reverse(g)
    t = threading.Thread(target=lambda: rev(g.record, {"block.y": np.ones(N)}))
    t.start()
    try:
        assert gate.entered.wait(30)
        with pytest.raises(eagle.Busy, match="eagle.diff.Reverse.__call__"):
            rev(g.record, {"block.y": np.ones(N)})
    finally:
        gate.go.set()
        t.join(30)
    assert not t.is_alive()


def test_a_forward_without_record_saves_nothing(tmp_path):
    ps = _plans(tmp_path, "cpu")
    flag = _flag("cpu", 1)
    g, _z = _skip_graph(ps, np.linspace(0, 1, N), 0.8, flag=flag)
    g.launch()
    assert g.record is None and g._recorder is None
    gr, _z = _skip_graph(ps, np.linspace(0, 1, N), 0.8, flag=flag, record=True)
    assert len(gr._recorder.recorded) == 1        # one guard cell, copied only when recording


# the guard reads the high word of the float counter: zero exactly when the counter is 0.0
def _loop_graph(ps, x, a, c, *, cap, record=False, guard=None):
    step = ps[2]
    body = step.bind(x=x, a=a, c=c, y=x, d=c)
    g = eagle.Graph().add_node(
        repeat_while(body.launch, guard or SkipGuard(c.view("uint32"), 1), cap), name="loop").build(record=record)
    return g, x


def _unrolled(ps, x, a, c, *, n, record=False):
    g = eagle.Graph()
    for k in range(n):
        g.add(ps[2].bind(x=x, a=a, c=c, y=x, d=c), name=f"s{k}")
    return g.build(record=record), x


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("n", [0, 1, 7, 20])
def test_repeat_while_matches_central_differences_and_the_unrolled_chain(tmp_path, backend, n):
    ps, xp = _plans(tmp_path, backend), xp_for(backend)
    x0, seed = np.linspace(-0.9, 0.9, N), np.linspace(1.0, 2.0, N)
    cap, a = 40, 0.8      # 40 > the 16-iteration window: n = 20 crosses a boundary

    def make(p, x_, a_):
        return _loop_graph(p, x_, a_, np.full(N, float(n)), cap=cap)       # on the host
    g, _x = _loop_graph(ps, xp.asarray(x0.copy()), a, xp.full(N, float(n)), cap=cap, record=True)
    g.launch()
    loop = g.record
    grads = eagle.diff.reverse(loop, {"y": xp.asarray(seed)})
    host = {k: np.asarray(v.get() if hasattr(v, "get") else v) for k, v in grads.items()}
    dx, da = central_difference(
        lambda p, x_, a_: make(p, x_, a_), ps, x0, a, seed)
    np.testing.assert_allclose(host["x"], dx, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(float(host["a"]), da, rtol=1e-6, atol=1e-12)
    if n:
        u, _ = _unrolled(ps, xp.asarray(x0.copy()), a, xp.full(N, float(n)), n=n, record=True)
        u.launch()
        ref = eagle.diff.reverse(u.record, {"y": xp.asarray(seed)})
        pull = lambda v: np.asarray(v.get() if hasattr(v, "get") else v)
        assert np.array_equal(host["x"], pull(ref["x"]))
        np.testing.assert_allclose(float(host["a"]), sum(float(pull(ref[f"s{k}.a" if n > 1 else "a"])) for k in range(n)),
                                   rtol=1e-12)
    else:
        np.testing.assert_array_equal(host["x"], seed)


def _carry_graph(ps, x, a, flag, *, record=False):
    """``y = a x^2``, then ``y = sin x`` only while the flag holds, then ``z = sin y``:
    skipped, the first writer's ``y`` stands."""
    xp = xp_for("gpu" if type(x).__module__.startswith("cupy") else "cpu")
    sq, sn, _ = ps
    y, z = xp.zeros_like(x), xp.zeros_like(x)
    maybe = skippable(sn.bind(x=x, y=y).launch, SkipGuard(flag, 0))
    g = (eagle.Graph().add(sq.bind(x=x, a=a, y=y), name="first")
         .add_node(maybe, name="maybe").add(sn.bind(x=y, y=z), name="last").build(record=record))
    return g, z


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("on", [1, 0])
def test_a_skipped_member_passes_the_adjoint_to_the_plane_s_earlier_writer(tmp_path, backend, on):
    ps, xp = _plans(tmp_path, backend), xp_for(backend)
    x0, seed = np.linspace(-1.1, 0.9, N), np.linspace(1.0, 2.0, N)
    g, _z = _carry_graph(ps, xp.asarray(x0), 0.8, _flag(backend, on), record=True)
    g.launch()
    grads = eagle.diff.reverse(g.record, {"last.y": xp.asarray(seed)})
    dx, da = central_difference(
        lambda p, x_, a_: _carry_graph(p, x_, a_, _flag("cpu", on)), ps, x0, 0.8, seed)
    pull = lambda v: np.asarray(v.get() if hasattr(v, "get") else v)
    np.testing.assert_allclose(pull(grads["x"]), dx, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(float(pull(grads["a"])), da, rtol=1e-6, atol=1e-12)
    assert (da == 0) == bool(on)       # taken, the first writer's value is overwritten
