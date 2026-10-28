# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A kernel split in team stages: eagle launches each team stage with one block of
``stages[k].team.threads`` threads per work item (a lane, a key, a scatter chunk or a target
cell), over the scratch planes it makes once at bind, inside the same launch order, so a Graph
captures the whole chain and replays it bit for bit.

Rows run on the host (a team of one, OpenMP across the blocks) and, marked ``gpu``, on the device.
They need a hawk that lowers team stages and skip on one that does not."""

import numpy as np
import pytest

pytest.importorskip("hawk.ir.team")
import hawk  # noqa: E402
import hawk.math as m  # noqa: E402
import eagle  # noqa: E402
import eagle.exec as eexec  # noqa: E402
import eagle.plan  # noqa: E402
from eagle import _split  # noqa: E402
from hawk import Grid, Index, Kernel, Matrix, Mutable, Scalar, Terminated, Vector, kernel  # noqa: E402
from hawk.artifact import build_bundle, plugins  # noqa: E402
from hawk.diff import vjp  # noqa: E402
from hawk.emit import policy  # noqa: E402

TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]
EPS = np.finfo(np.float64).eps


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp("team_cache"))


def _xp(target):
    if target == "host":
        return np
    return pytest.importorskip("cupy")


def _host(a):
    return np.array(a.get() if hasattr(a, "get") else a)


def _plan(kernels, tmp_path, cache, target, name=None):
    """``(plan of kernel `name`, bundle)``: built for ``target`` and linked, the door `eagle.plan.auto` takes."""
    bundle = build_bundle(kernels, tmp_path, targets=("cuda",) if target == "cuda" else ("host",),
                          cache_dir=cache, host_profile="native")
    by = plugins(bundle, device_loader=eagle.registry.load_manifest if target == "cuda" else None)
    _split.link(by, bundle)
    structure = eexec.DeviceKernel if target == "cuda" else eexec.HostTeam
    return eagle.plan.plan(by[name or kernels[0].name], structure=structure), bundle


def _fresh_and_replayed(plan_, planes, out, target):
    """The plane ``out`` after one fresh launch, then after three replays of a captured graph (each from
    a cleared output): ``(fresh, [replay, replay, replay])``."""
    bound = plan_.bind(**planes)
    bound.launch()
    fresh = _host(planes[out])
    graph = eagle.Graph().add(bound, name="team").build()
    seen = []
    for _ in range(3):
        planes[out][...] = 0
        graph.launch()
        seen.append(_host(planes[out]))
    return fresh, seen


def _same_bits(fresh, seen):
    assert all(s.tobytes() == fresh.tobytes() for s in seen)


# ---- a keyed block sum (a Gram matrix) -------------------------------------------------------------

P, ROWS, KEYS = 71, 8, 3


@pytest.mark.parametrize("target", TARGETS)
def test_a_keyed_block_sum_is_launched_per_key_and_replays_bitwise(tmp_path, cache, target):
    with policy.using(linalg_form="auto"):
        @kernel
        def gram(lane: Grid["key", "row":ROWS], j: Vector[P]) -> Matrix[P, P]:
            return hawk.sum(m.outer(j, j), over="row")

    plan_, _ = _plan([gram], tmp_path, cache, target)
    stages = _split.stages_of(plan_.plugin)
    assert [bool(s.team) for s in stages] == [False, True] and stages[1].team["grid"] == "keys"
    xp = _xp(target)
    j = np.random.default_rng(1).standard_normal((P, KEYS * ROWS))
    planes = {"j": xp.asarray(j), "out": xp.zeros((P, P, KEYS))}
    fresh, seen = _fresh_and_replayed(plan_, planes, "out", target)
    want = np.stack([j[:, k * ROWS:(k + 1) * ROWS] @ j[:, k * ROWS:(k + 1) * ROWS].T for k in range(KEYS)], -1)
    assert np.abs(fresh - want).max() <= 8 * ROWS * EPS * np.abs(want).max()
    _same_bits(fresh, seen)


# ---- a team LU solve at n = 71 -----------------------------------------------------------------------

N, LANES = 71, 5


@pytest.mark.parametrize("target", TARGETS)
def test_a_team_lu_solve_runs_one_block_per_lane_and_replays_bitwise(tmp_path, cache, target):
    @kernel
    def solve(A: Matrix[N, N], b: Vector[N], x: Mutable[Vector[N]]):
        x, _info = m.solve(A, b)

    plan_, _ = _plan([solve], tmp_path, cache, target)
    stages = _split.stages_of(plan_.plugin)
    assert any(s.team and s.team["threads"] > 1 for s in stages)
    rng = np.random.default_rng(2)
    a = rng.standard_normal((LANES, N, N)) / np.sqrt(N) + 0.5 * np.eye(N)
    b = rng.standard_normal((LANES, N))
    xp = _xp(target)
    planes = {"A": xp.asarray(np.ascontiguousarray(a.reshape(LANES, -1).T)),
              "b": xp.asarray(np.ascontiguousarray(b.T)), "x": xp.zeros((N, LANES))}
    fresh, seen = _fresh_and_replayed(plan_, planes, "x", target)
    for lane in range(LANES):
        want = np.linalg.solve(a[lane], b[lane])
        assert np.abs(fresh[:, lane] - want).max() <= 50 * N * EPS * np.abs(want).max() * np.linalg.cond(a[lane])
    _same_bits(fresh, seen)


# ---- a damped step: per-row stores, block sums, a Cholesky team stage, per-lane updates ---------------

SP, SROWS, SKEYS = 33, 40, 2


@pytest.mark.parametrize("target", TARGETS)
def test_a_step_chain_in_one_graph_replays_three_times_equal_to_three_fresh_runs(tmp_path, cache, target):
    @kernel
    def step(lane: Grid["key", "row":SROWS], j: Vector[SP], r: Scalar, x: Mutable[Vector[SP]]):
        gram = hawk.sum(m.outer(j, j), over="row")
        grad = hawk.sum(j * r, over="row")
        x = m.cho_solve(m.cholesky(gram)[0], grad)

    plan_, _ = _plan([step], tmp_path, cache, target)
    n = SKEYS * SROWS
    rng = np.random.default_rng(3)
    j, r = rng.standard_normal((n, SP)), rng.standard_normal(n)
    xp = _xp(target)

    def planes():       # lane-shaped: (width, keys, rows), the form a two-axis Grid reads unambiguously
        return {"j": xp.asarray(np.ascontiguousarray(j.T).reshape(SP, SKEYS, SROWS)),
                "r": xp.asarray(r), "x": xp.zeros((SP, SKEYS, SROWS))}

    runs = []
    for _ in range(3):                              # three fresh binds and launches
        fresh = planes()
        plan_.bind(**fresh).launch()
        runs.append(_host(fresh["x"]))
    assert runs[0].tobytes() == runs[1].tobytes() == runs[2].tobytes()
    first, seen = _fresh_and_replayed(plan_, planes(), "x", target)
    assert first.tobytes() == runs[0].tobytes()
    _same_bits(runs[0], seen)
    for k in range(SKEYS):
        jk, rk = j[k * SROWS:(k + 1) * SROWS], r[k * SROWS:(k + 1) * SROWS]
        want = np.linalg.solve(jk.T @ jk, jk.T @ rk)
        assert np.abs(first[:, k, 0] - want).max() <= 100 * SP * EPS * np.abs(want).max() * np.linalg.cond(jk.T @ jk)


# ---- an ordered scatter with lanes that add nothing ------------------------------------------------------

@kernel
def add_term(key: Index, x: Scalar, terminated: Terminated, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x


@pytest.mark.parametrize("target", TARGETS)
def test_an_ordered_scatter_skips_terminated_lanes_and_replays_bitwise(tmp_path, cache, target):
    plan_, _ = _plan([add_term], tmp_path, cache, target)
    n, cells = 6000, 9
    rng = np.random.default_rng(5)
    key, x = rng.integers(0, cells, n).astype(np.int64), rng.standard_normal(n)
    xp = _xp(target)
    first, second = rng.random(n) < 0.5, rng.random(n) < 0.3
    for term in (first, second, first):              # a lane live in one run and terminated in the next
        planes = {"key": xp.asarray(key), "x": xp.asarray(x), "terminated": xp.asarray(term),
                  "acc": xp.zeros(cells)}
        fresh, seen = _fresh_and_replayed(plan_, planes, "acc", target)
        want = np.zeros(cells)
        np.add.at(want, key[~term], x[~term])
        np.testing.assert_allclose(fresh, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
        _same_bits(fresh, seen)


@pytest.mark.parametrize("target", TARGETS)
def test_the_bind_records_its_chunks_and_makes_the_key_log_filled_with_minus_one(tmp_path, cache, target):
    plan_, _ = _plan([add_term], tmp_path, cache, target)
    xp = _xp(target)
    for n, cells, chunks in ((6000, 9, 256), (50, 200, 1), (1000, 1000, 1)):
        planes = {"key": xp.zeros(n, dtype=np.int64), "x": xp.zeros(n), "terminated": xp.zeros(n, dtype=bool),
                  "acc": xp.zeros(cells)}
        bound = plan_.bind(**planes)
        assert bound.chunks == {"acc": chunks}
        log = _host(bound._team["tms_acc_k"])
        assert log.dtype == np.int32 and (log == -1).all()
        bound.launch()
        assert (_host(bound._team["tms_acc_k"]) == -1).all()


# ---- the ordered net-axis gradient --------------------------------------------------------------------------

H, NP, NETS, S = 8, 24, 64, 40


def _net_loss():
    @kernel
    def net_loss(lane: Grid["net", "sample":S], W: hawk.Table["net", "p":NP], mask: hawk.Table["net", "h":H],
                 x: Scalar, y: Scalar) -> Scalar:
        pred = 0.0 * x
        for h in range(H):
            a = W.at(net=lane.net, p=3 * h)
            b = W.at(net=lane.net, p=3 * h + 1)
            c = W.at(net=lane.net, p=3 * h + 2)
            pred = pred + mask.at(net=lane.net, h=h) * c * m.tanh(a * x + b)
        return hawk.sum((pred - y) * (pred - y), over="sample", policy="ordered")
    return Kernel("net_loss_ordered", net_loss.sinks, net_loss.planes)


@pytest.mark.parametrize("target", TARGETS)
def test_the_ordered_net_gradient_is_launched_by_label_and_replays_bitwise(tmp_path, cache, target):
    loss = _net_loss()
    primal = build_bundle([loss], tmp_path / "loss", targets=("cuda", "host"), cache_dir=cache,
                          host_profile="native")
    grad = Kernel("net_grad_ordered", vjp(loss, wrt=("W",), primal_unit=primal.digest))
    plan_, _ = _plan([grad], tmp_path / "grad", cache, target)
    rng = np.random.default_rng(7)
    w, x, y = rng.normal(size=(NETS, NP)) * 0.7, rng.normal(size=(NETS, S)), rng.normal(size=(NETS, S))
    mask = (np.arange(H)[None, :] < np.resize(np.array([8, 3, 5, 1, 6]), NETS)[:, None]).astype(np.float64)
    xp = _xp(target)
    planes = {"W": xp.asarray(w.ravel()), "mask": xp.asarray(mask.ravel()), "x": xp.asarray(x.ravel()),
              "y": xp.asarray(y.ravel()), "bar_out": xp.ones(NETS), "bar_W": xp.zeros(NETS * NP)}
    bound = plan_.bind(net=NETS, **planes)
    bound.launch()
    fresh = _host(planes["bar_W"])
    replay = bound.launch                             # a derivative is no Graph step: capture its launch itself
    if target == "cuda":
        stream = _xp(target).cuda.Stream(non_blocking=True)
        stream.begin_capture()
        bound.launch(stream.ptr)
        graph = stream.end_capture()
        replay = lambda: (graph.launch(stream), stream.synchronize())  # noqa: E731
    for _ in range(3):
        planes["bar_W"][...] = 0
        replay()
        assert _host(planes["bar_W"]).tobytes() == fresh.tobytes()
    for net in (0, 17, 63):                           # the loop over nets, one by one
        a, b, c = w[net, 0::3], w[net, 1::3], w[net, 2::3]
        t = np.tanh(a[:, None] * x[net] + b[:, None])
        r = (mask[net] * c) @ t - y[net]
        want = np.empty(NP)
        want[0::3] = ((2 * mask[net] * c)[:, None] * (1 - t * t) * x[net] * r).sum(axis=1)
        want[1::3] = ((2 * mask[net] * c)[:, None] * (1 - t * t) * r).sum(axis=1)
        want[2::3] = ((2 * mask[net])[:, None] * t * r).sum(axis=1)
        np.testing.assert_allclose(fresh.reshape(NETS, NP)[net], want, rtol=1e-10, atol=1e-10)


# ---- the scratch refusal ----------------------------------------------------------------------------------------

@pytest.mark.gpu
def test_scratch_larger_than_the_free_device_memory_is_refused_by_name_with_the_byte_sum(tmp_path, cache, monkeypatch):
    @kernel
    def solve(A: Matrix[N, N], b: Vector[N], x: Mutable[Vector[N]]):
        x, _info = m.solve(A, b)

    plan_, _ = _plan([solve], tmp_path, cache, "cuda")
    cp = pytest.importorskip("cupy")
    planes = {"A": cp.zeros((N * N, LANES)), "b": cp.zeros((N, LANES)), "x": cp.zeros((N, LANES))}
    specs = plan_.plugin._team_planes
    total = sum(spec.get("per_lane", 1) * LANES * (4 if spec["dtype"] == "int32" else 8) for spec in specs.values())
    monkeypatch.setattr(_split, "_free_bytes", lambda device: total - 1)
    with pytest.raises(ValueError, match=r"needs [\d,]+ bytes of team scratch .* at 5 lanes, but the device has "
                                         r"[\d,]+ bytes free; stage 'solve_\d+' alone needs") as err:
        plan_.bind(**planes)
    assert f"{total:,} bytes" in str(err.value) and "'solve'" in str(err.value)
    monkeypatch.setattr(_split, "_free_bytes", lambda device: total)
    plan_.bind(**planes)                              # exactly enough binds


# ---- the key log is refilled every launch; the once plane; the scatter status --------------------------------------

@pytest.mark.parametrize("target", TARGETS)
def test_a_launch_stopped_after_the_first_stage_leaves_no_trace_in_the_next_replay(tmp_path, cache, target):
    plan_, _ = _plan([add_term], tmp_path, cache, target)
    n, cells = 6000, 9
    rng = np.random.default_rng(11)
    key, x = rng.integers(0, cells, n).astype(np.int64), rng.standard_normal(n)
    term = rng.random(n) < 0.5
    xp = _xp(target)
    planes = {"key": xp.asarray(key), "x": xp.asarray(x), "terminated": xp.zeros(n, dtype=bool), "acc": xp.zeros(cells)}
    bound = plan_.bind(**planes)
    bound.stages[0].launch()                          # every lane live: the key log now holds all keys
    planes["terminated"][...] = xp.asarray(term)
    graph = eagle.Graph().add(bound, name="scatter").build()
    graph.launch()
    want = np.zeros(cells)
    np.add.at(want, key[~term], x[~term])
    np.testing.assert_allclose(_host(planes["acc"]), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("target", TARGETS)
def test_the_once_plane_is_one_copy_whatever_the_lane_count(tmp_path, cache, target):
    plan_, _ = _plan([add_term], tmp_path, cache, target)
    xp = _xp(target)
    sizes = set()
    for n in (50, 6000):
        planes = {"key": xp.zeros(n, dtype=np.int64), "x": xp.zeros(n), "terminated": xp.zeros(n, dtype=bool),
                  "acc": xp.zeros(9)}
        sizes.add(plan_.bind(**planes)._team["tms_acc_m"].shape)
    assert sizes == {(257,)}


OUTSIDE = r"hawk.scatter\(acc, policy='ordered'\) got a key outside \[0, 9\)"


def _out_of_range_case(tmp_path, cache, target):
    """``(plan, bound, planes, key, bad key, x)``: the output plane is an eagle.Array, whose read-back is
    one of the points where eagle already waits."""
    plan_, _ = _plan([add_term], tmp_path, cache, target)
    n, cells = 6000, 9
    rng = np.random.default_rng(13)
    key, x = rng.integers(0, cells, n).astype(np.int32), rng.standard_normal(n)
    bad = key.copy()
    bad[4321] = cells + 3
    xp = _xp(target)
    acc = eagle.arrays.zeros((cells,), "float64", "cuda:0" if target == "cuda" else "cpu")
    planes = {"key": xp.asarray(bad), "x": xp.asarray(x), "terminated": xp.zeros(n, dtype=bool), "acc": acc}
    return plan_, plan_.bind(**planes), planes, key, bad, x


def _clean_run(bound, planes, key, x, xp):
    planes["key"][...] = xp.asarray(key)
    planes["acc"].fill(0)
    bound.launch()
    want = np.zeros(9)
    np.add.at(want, key, x)
    np.testing.assert_allclose(np.asarray(planes["acc"].get()), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("target", TARGETS)
def test_a_key_out_of_range_is_reported_where_eagle_next_waits_and_the_next_run_is_clean(tmp_path, cache, target):
    _plan_, bound, planes, key, bad, x = _out_of_range_case(tmp_path, cache, target)
    xp = _xp(target)
    if target == "host":                               # a host launch is complete when it returns
        with pytest.raises(Exception, match=OUTSIDE):
            bound.launch()
    else:                                              # the device launch waits for nothing; the read-back does
        bound.launch()
        assert bound in _split._PENDING
        with pytest.raises(Exception, match=OUTSIDE):
            planes["acc"].get()
        assert bound not in _split._PENDING
    _clean_run(bound, planes, key, x, xp)              # the next run is clean
    if target == "cuda":                               # the next launch of the same bind reports a pending one
        planes["key"][...] = xp.asarray(bad)
        bound.launch()
        with pytest.raises(Exception, match=OUTSIDE):
            bound.launch()
        _clean_run(bound, planes, key, x, xp)


@pytest.mark.gpu
def test_a_replay_reports_at_the_synchronise_and_the_next_replay_is_clean(tmp_path, cache):
    _plan_, bound, planes, key, bad, x = _out_of_range_case(tmp_path, cache, "cuda")
    graph = eagle.Graph().add(bound, name="scatter").build()
    with pytest.raises(Exception, match=OUTSIDE):
        graph.launch()                                 # launch waits already, so it reports there
    graph.enqueue()
    assert bound in _split._PENDING
    with pytest.raises(Exception, match=OUTSIDE):
        graph.synchronize()
    planes["key"][...] = _xp("cuda").asarray(key)
    graph.launch()
    graph.synchronize()
    assert not _split._PENDING


@pytest.mark.parametrize("target", TARGETS)
def test_a_launch_without_a_scatter_records_nothing(tmp_path, cache, target):
    with policy.using(linalg_form="auto"):
        @kernel
        def gram(lane: Grid["key", "row":ROWS], j: Vector[P]) -> Matrix[P, P]:
            return hawk.sum(m.outer(j, j), over="row")

    plan_, _ = _plan([gram], tmp_path, cache, target)
    xp = _xp(target)
    bound = plan_.bind(j=xp.zeros((P, KEYS * ROWS)), out=xp.zeros((P, P, KEYS)))
    bound.launch()
    assert not _split._PENDING


@pytest.mark.gpu
def test_a_synchronise_on_one_stream_does_not_wait_for_a_scatter_on_another(tmp_path, cache):
    from eagle import _core

    _plan_, bound, planes, key, bad, x = _out_of_range_case(tmp_path, cache, "cuda")
    stream_a, stream_b = _core.Stream(non_blocking=True), _core.Stream(non_blocking=True)
    bound.launch(stream_a.ptr())
    assert bound in _split._PENDING and bound._event._stream == stream_a.ptr()
    stream_b.synchronize()                             # not the stream of the scatter: nothing read, nothing waited on
    assert bound in _split._PENDING
    with pytest.raises(Exception, match=OUTSIDE):
        stream_a.synchronize()
    assert bound not in _split._PENDING
    del bound, planes                                  # arrays go before the streams they were launched on
    _split._PENDING.clear()


@pytest.mark.gpu
def test_a_pipeline_replay_reports_the_scatter_at_its_synchronise(tmp_path, cache):
    from eagle.pipeline import GraphPipeline

    _plan_, bound, planes, key, bad, x = _out_of_range_case(tmp_path, cache, "cuda")
    pipe = GraphPipeline().add(bound.launch).build()   # until_done, Graph and Composer replay through this door
    pipe.enqueue()
    assert bound in _split._PENDING
    with pytest.raises(Exception, match=OUTSIDE):
        pipe.synchronize()
    planes["key"][...] = _xp("cuda").asarray(key)
    pipe.launch()
    assert not _split._PENDING


# ---- Plan.run of a team-staged kernel is the bind path ------------------------------------------------------------

@pytest.mark.parametrize("target", TARGETS)
def test_plan_run_of_a_team_kernel_equals_the_bind_path(tmp_path, cache, target):
    @kernel
    def solve(A: Matrix[N, N], b: Vector[N], x: Mutable[Vector[N]]):
        x, _info = m.solve(A, b)

    plan_, _ = _plan([solve], tmp_path, cache, target)
    rng = np.random.default_rng(21)
    a = rng.standard_normal((LANES, N, N)) / np.sqrt(N) + 0.5 * np.eye(N)
    b = rng.standard_normal((LANES, N))
    xp = _xp(target)

    def inputs():
        return {"A": xp.asarray(np.ascontiguousarray(a.reshape(LANES, -1).T)), "b": xp.asarray(np.ascontiguousarray(b.T))}

    bound_planes = {**inputs(), "x": xp.zeros((N, LANES))}
    plan_.bind(**bound_planes).launch()
    ran = plan_.run(**inputs())
    assert _host(ran).tobytes() == _host(bound_planes["x"]).tobytes()
    given = {**inputs(), "x": xp.zeros((N, LANES))}                       # a supplied output is written in place
    plan_.run(**given)
    assert _host(given["x"]).tobytes() == _host(ran).tobytes()


@pytest.mark.parametrize("target", TARGETS)
def test_plan_run_of_a_keyed_team_kernel_equals_the_bind_path(tmp_path, cache, target):
    with policy.using(linalg_form="auto"):
        @kernel
        def gram(lane: Grid["key", "row":ROWS], j: Vector[P]) -> Matrix[P, P]:
            return hawk.sum(m.outer(j, j), over="row")

    plan_, _ = _plan([gram], tmp_path, cache, target)
    xp = _xp(target)
    j = np.random.default_rng(22).standard_normal((P, KEYS * ROWS))
    bound_planes = {"j": xp.asarray(j), "out": xp.zeros((P, P, KEYS))}
    plan_.bind(**bound_planes).launch()
    given = {"j": xp.asarray(j), "out": xp.zeros((P, P, KEYS))}
    ran = plan_.run(**given)
    assert _host(given["out"]).tobytes() == _host(bound_planes["out"]).tobytes()
    assert _host(ran).tobytes() == _host(given["out"]).tobytes()
    free = plan_.run(j=xp.asarray(j))                  # no output supplied: sized by the key count and payload
    assert _host(free).tobytes() == _host(ran).tobytes() and tuple(free.shape) == (P, P, KEYS)


@pytest.mark.parametrize("target", TARGETS)
def test_a_bare_bind_of_a_sum_kernel_makes_its_rows_and_equals_plan_run(tmp_path, cache, target):
    @kernel
    def loss(x: Scalar, total: Mutable[Scalar]):
        total = hawk.sum(x * x)

    plan_, _ = _plan([loss], tmp_path, cache, target)
    xp = _xp(target)
    x = np.random.default_rng(23).normal(size=1000)
    total = xp.zeros(1)
    plan_.bind(x=xp.asarray(x), total=total).launch()
    if target == "cuda":
        _xp(target).cuda.get_current_stream().synchronize()
    ran = plan_.run(x=xp.asarray(x))
    assert _host(total).tobytes() == _host(ran).tobytes()
    np.testing.assert_allclose(_host(total)[0], x @ x, rtol=1e-12)
