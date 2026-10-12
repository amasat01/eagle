# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``dispatch=`` door: bitwise-equal results fused, split and automatic,
the class histogram, the static-selector rule and its one warning, the
refusals, the 16-step cadence with the hysteresis band, and two threads."""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
import hawk.math as M  # noqa: E402
from hawk import Index, Mutable, Scalar, Terminated  # noqa: E402

import eagle  # noqa: E402
from eagle import _dispatch_door as door  # noqa: E402
from eagle import _dispatch_policy as policy  # noqa: E402

N = 2000


@hawk.kernel
def walker(k: Index, x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    x = x + M.dispatch(k, [1.0, M.exp(x * 0.001), M.sin(x) * 2.0], policy="auto")
    t = t + 1.0
    terminated = t >= 40.0


@hawk.kernel
def long_walker(k: Index, x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    x = x + M.dispatch(k, [1.0, M.exp(x * 0.001), M.sin(x) * 2.0], policy="auto")
    t = t + 1.0
    terminated = t >= 100.0


@hawk.kernel
def drifting(k: Mutable[Index], x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    x = x + M.dispatch(k, [1.0, M.exp(x * 0.001), M.sin(x) * 2.0], policy="auto")
    k = M.select(k > 1, k - 2, k + 1)
    t = t + 1.0
    terminated = t >= 12.0


@hawk.kernel
def declared(k: Index, x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    x = x + M.dispatch(k, [1.0, M.sin(x)], policy="switch")
    t = t + 1.0
    terminated = t >= 5.0


@hawk.kernel
def plain(x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    x = x + 1.0
    t = t + 1.0
    terminated = t >= 5.0


def rare_classes():
    """Class 2 for 2% of the samples, class 0 for the rest."""
    return (np.arange(N) * 7919 % 100 > 97).astype(np.int64) * 2


def run(kernel, k, **options):
    return eagle.simulate(kernel, k=k.copy(), x=np.zeros(N), t=np.zeros(N), max_steps=200,
                          **options)


@pytest.fixture(scope="module")
def reference():
    return run(walker, rare_classes(), dispatch="switch")


# --- results and the class histogram --- #
@pytest.mark.parametrize("dispatch", [None, "segmented", "predicated", {"dispatch0": "segmented"}])
def test_every_policy_gives_the_fused_result_bit_for_bit(reference, dispatch):
    got = run(walker, rare_classes(), dispatch=dispatch)
    assert got.status == "finished"
    np.testing.assert_array_equal(got.x, reference.x)
    np.testing.assert_array_equal(got.t, reference.t)


def test_the_reported_mix_is_the_bincount_of_the_classes(reference):
    k = rare_classes()
    want = tuple(int(c) for c in np.bincount(k, minlength=3))
    for dispatch in (None, "segmented"):
        policy_in_force, mix, gain = run(walker, k, dispatch=dispatch).report.dispatch["dispatch0"]
        assert mix == want
        assert policy_in_force in ("switch", "segmented") and gain > 0


def test_a_split_run_counts_its_class_sorts():
    report = run(walker, rare_classes(), dispatch="segmented").report
    assert report.class_sorts >= 1
    assert run(walker, rare_classes(), dispatch="switch").report.class_sorts == 0


def test_the_default_keeps_a_library_math_group_fused_under_an_exact_host_profile(monkeypatch):
    # exp and sin stay scalar under libm bit-identity, so the fused loop is no
    # lockstep over the branches and splitting the rare ones is no gain
    monkeypatch.setenv("HAWK_HOST_PROFILE", "native")
    assert run(walker, rare_classes()).report.class_sorts == 0


def test_the_default_splits_a_library_math_group_under_the_vector_math_profile(monkeypatch):
    from hawk.compile import toolchain

    if toolchain.default_host_profile() != "native-vector-math":
        pytest.skip("no vector-math host profile on this machine")
    monkeypatch.setenv("HAWK_HOST_PROFILE", "native-vector-math")
    assert run(walker, rare_classes()).report.class_sorts >= 1


# --- the static-selector rule --- #
def test_a_selector_the_step_writes_is_never_split_by_the_door():
    k0 = (np.arange(N) % 3).astype(np.int64)
    ref = run(drifting, k0, dispatch="switch")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = run(drifting, k0)
    assert got.report.dispatch is None          # no group was left to measure
    np.testing.assert_array_equal(got.x, ref.x)


def test_forcing_a_split_on_such_a_selector_warns_once_and_stays_correct():
    k0 = (np.arange(N) % 3).astype(np.int64)
    ref = run(drifting, k0, dispatch="switch")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = run(drifting, k0, dispatch="segmented")
    assert [w.category for w in caught] == [eagle.PerformanceWarning]
    assert "slower" in str(caught[0].message)
    np.testing.assert_array_equal(got.x, ref.x)
    np.testing.assert_array_equal(got.t, ref.t)
    assert got.report.class_sorts > 1          # the batch went stale and was sorted again


def test_the_warning_filters_on_its_own():
    assert issubclass(eagle.PerformanceWarning, UserWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", eagle.PerformanceWarning)
        run(drifting, (np.arange(N) % 3).astype(np.int64), dispatch="segmented")


def test_static_selector_over_the_whole_step():
    assert door._policy.static_selector([("per_sample", "k")], {"x", "t"})
    assert not door._policy.static_selector([("mutable", "k")], {"k"})


# --- refusals --- #
def test_naming_a_group_the_author_declared_is_refused():
    with pytest.raises(ValueError, match=r"declared policy='switch'.*only 'auto' groups"):
        eagle.simulate(declared, k=np.zeros(4, dtype=np.int64), x=np.zeros(4), t=np.zeros(4),
                       max_steps=9, dispatch={"dispatch0": "segmented"})


def test_a_policy_for_a_model_with_no_auto_group_is_refused():
    with pytest.raises(ValueError, match="has none"):
        eagle.simulate(plain, x=np.zeros(4), t=np.zeros(4), max_steps=9, dispatch="segmented")


def test_an_unknown_policy_names_the_valid_ones():
    with pytest.raises(ValueError, match=r"'switch', 'segmented', 'predicated'"):
        run(walker, rare_classes(), dispatch="fastest")


def test_an_unknown_group_names_the_groups():
    with pytest.raises(ValueError, match=r"not a dispatch group.*\['dispatch0'\]"):
        run(walker, rare_classes(), dispatch={"dispatch9": "switch"})


def test_dispatch_of_the_wrong_type_is_refused():
    with pytest.raises(ValueError, match="None, a policy name or a"):
        run(walker, rare_classes(), dispatch=3)


# --- the cadence and the band --- #
def test_a_scripted_gain_flips_the_policy_only_when_it_leaves_the_band(monkeypatch, reference):
    ref = run(long_walker, rare_classes(), dispatch="switch")
    gains = iter([1.10, 1.16, 1.10, 1.04, 1.10])
    looks = []
    steps = {"split": 0, "fused runs": []}

    def scripted(cost, counts, **kwargs):
        looks.append(steps["split"])
        return next(gains)

    split, plain = door.DispatchRunner._step_segmented, door.DispatchRunner._run_plain

    def count_split(self):
        steps["split"] += 1
        split(self)

    def count_plain(self, n):
        steps["fused runs"].append(n)
        return plain(self, n)

    monkeypatch.setattr(policy, "predicted_gain", scripted)
    monkeypatch.setattr(door.DispatchRunner, "_step_segmented", count_split)
    monkeypatch.setattr(door.DispatchRunner, "_run_plain", count_plain)
    got = run(long_walker, rare_classes())          # in band from switch: stays fused
    assert looks == [0] and steps == {"split": 0, "fused runs": [200]}
    sim = eagle.simulation(long_walker, k=rare_classes(), x=np.zeros(N), t=np.zeros(N),
                           max_steps=200)
    looks.clear()
    got = sim.run()                                  # 1.16 enters; 1.10 keeps; 1.04 leaves
    assert looks == [0, 16, 32]                      # on entry, then every 16 split steps
    # split from the first look to the third (step 32), then one fused run for the rest
    assert steps == {"split": 32, "fused runs": [200, 168]}
    np.testing.assert_array_equal(got.x, ref.x)          # flipping changed no result


# --- zero cost when unused --- #
def test_a_model_without_an_auto_group_never_touches_the_door(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the dispatch door ran")

    monkeypatch.setattr(door, "resolve_kernels", boom)
    monkeypatch.setattr(door.DispatchRunner, "__init__", boom)
    for kernel, extra in ((plain, {}), (declared, {"k": np.zeros(4, dtype=np.int64)})):
        got = eagle.simulate(kernel, x=np.zeros(4), t=np.zeros(4), max_steps=9, **extra)
        assert got.status == "finished" and got.report.dispatch is None


def test_a_forced_switch_is_a_plain_kernel(monkeypatch, reference):
    monkeypatch.setattr(door.DispatchRunner, "__init__", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a forced switch built a dispatch runner")))
    got = run(walker, rare_classes(), dispatch="switch")
    np.testing.assert_array_equal(got.x, reference.x)


# --- threads --- #
def test_two_threads_with_two_simulations_each_get_the_right_answer(reference):
    out, errors = {}, []

    def work(name, dispatch):
        try:
            out[name] = run(walker, rare_classes(), dispatch=dispatch)
        except Exception as exc:        # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i, d)) for i, d in enumerate((None, "segmented"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for got in out.values():
        np.testing.assert_array_equal(got.x, reference.x)


@pytest.mark.parametrize("dispatch", [None, "segmented"])
def test_n_threads_n_simulations_equal_the_serial_run_bit_for_bit(dispatch, reference):
    threads_n = 6
    out, errors = [None] * threads_n, []
    start = threading.Barrier(threads_n)

    def work(i):
        try:
            start.wait()
            out[i] = run(walker, rare_classes(), dispatch=dispatch)
        except Exception as exc:        # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors
    for got in out:
        np.testing.assert_array_equal(got.x, reference.x)
        np.testing.assert_array_equal(got.t, reference.t)


def test_the_calibration_timer_runs_on_distinct_bound_plans_in_many_threads():
    from eagle.plan import auto

    plan = auto(plain).select({"x": np.zeros(64)})
    spec = plan.plugin.arg_spec
    threads_n, reps = 6, 5
    bound, xs, errors = [], [], []
    for _ in range(threads_n):
        x, t, done = np.zeros(64), np.zeros(64), np.zeros(64, dtype=bool)
        extra = {nm: np.zeros(1, dtype=np.int64 if nm == "fused_steps" else np.int32)
                 for role, nm in spec if role == "lookup"}
        bound.append(plan.bind(x=x, t=t, terminated=done, **extra))
        xs.append(x)
    start = threading.Barrier(threads_n)

    def work(b):
        try:
            start.wait()
            assert len(door.time_launch(b, reps)) == reps
        except Exception as exc:        # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(b,)) for b in bound]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors
    for x in xs:
        np.testing.assert_array_equal(x, np.full(64, float(reps)))


# --- the split step --- #
def test_a_static_split_step_launches_each_branch_over_its_own_run_only(monkeypatch, reference):
    counts = []
    launch = door.DispatchRunner._launch

    def spy(self, bound, count, block=256):
        counts.append(count)
        launch(self, bound, count, block)

    monkeypatch.setattr(door.DispatchRunner, "_launch", spy)
    got = run(walker, rare_classes(), dispatch="segmented")
    np.testing.assert_array_equal(got.x, reference.x)
    classes = np.bincount(rare_classes(), minlength=3)
    assert counts[0] == N                                # the one classification, over the batch
    runs = sorted(set(counts[1:]))
    assert runs == sorted(int(c) for c in classes if c)  # then one launch per non-empty run
    assert len(counts[1:]) == got.report.steps * int(np.count_nonzero(classes))


def test_the_calibration_timer_keeps_every_repetition():
    from eagle.plan import auto

    plan = auto(plain).select({"x": np.zeros(64)})
    x, t, done = np.zeros(64), np.zeros(64), np.zeros(64, dtype=bool)
    spec = plan.plugin.arg_spec
    extra = {nm: np.zeros(1, dtype=np.int64 if nm == "fused_steps" else np.uint32)
             for role, nm in spec if role == "lookup"}
    bound = plan.bind(x=x, t=t, terminated=done, **extra)
    seconds = door.time_launch(bound, 4)
    assert len(seconds) == 4 and all(s >= 0.0 for s in seconds)
    np.testing.assert_array_equal(x, np.full(64, 4.0))   # every repetition ran


# --- the device (run by the gate on a GPU) --- #
def _device_run(kernel, k, **options):
    import cupy as cp

    return eagle.simulate(kernel, k=cp.asarray(k), x=cp.zeros(N), t=cp.zeros(N), max_steps=200,
                          **options)


@pytest.mark.gpu
@pytest.mark.parametrize("dispatch", [None, "segmented", {"dispatch0": "segmented"}])
def test_device_every_policy_gives_the_fused_result_bit_for_bit(dispatch):
    cp = pytest.importorskip("cupy")
    ref = _device_run(walker, rare_classes(), dispatch="switch")
    got = _device_run(walker, rare_classes(), dispatch=dispatch)
    np.testing.assert_array_equal(cp.asnumpy(got.x), cp.asnumpy(ref.x))
    np.testing.assert_array_equal(cp.asnumpy(got.t), cp.asnumpy(ref.t))
    _, mix, _ = got.report.dispatch["dispatch0"]
    assert mix == tuple(int(c) for c in np.bincount(rare_classes(), minlength=3))


@pytest.mark.gpu
def test_device_split_on_a_state_dependent_selector_warns_once_and_is_correct():
    cp = pytest.importorskip("cupy")
    k0 = (np.arange(N) % 3).astype(np.int64)
    ref = _device_run(drifting, k0, dispatch="switch")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = _device_run(drifting, k0, dispatch="segmented")
    assert [w.category for w in caught] == [eagle.PerformanceWarning]
    np.testing.assert_array_equal(cp.asnumpy(got.x), cp.asnumpy(ref.x))
    assert got.report.class_sorts > 1


def test_the_band_alone_decides_at_its_edges():
    assert policy.decide(policy.DISPATCH_ENTER_GAIN, "switch") == "segmented"
    assert policy.decide(np.nextafter(policy.DISPATCH_ENTER_GAIN, 0), "switch") == "switch"
    assert policy.decide(policy.DISPATCH_LEAVE_GAIN, "segmented") == "segmented"
    assert policy.decide(np.nextafter(policy.DISPATCH_LEAVE_GAIN, 0), "segmented") == "switch"
    with pytest.raises(ValueError, match="'switch' or 'segmented'"):
        policy.decide(1.0, "auto")


# --- Graph.build(dispatch=) --- #
def _graph_with(kernel):
    from eagle._plan_deploy import auto

    planes = dict(k=rare_classes(), x=np.zeros(N), t=np.zeros(N),
                  terminated=np.zeros(N, dtype=bool), dispatch0_mix=np.zeros(3, dtype=np.uint32),
                  finished_count=np.zeros(1, dtype=np.uint32))
    step = kernel.step if kernel.step is not None else kernel
    bound = auto(step).host.bind(**planes)
    return eagle.Graph().add(bound), planes


@pytest.mark.parametrize("dispatch", [None, "switch", "predicated", {"dispatch0": "switch"}])
def test_graph_build_lowers_auto_groups_to_the_fused_result(dispatch):
    ref = run(walker, rare_classes(), dispatch="switch")
    graph, planes = _graph_with(walker)
    graph.build(dispatch=dispatch)
    expected = dispatch if isinstance(dispatch, str) else "switch"
    assert graph.dispatch == {"walker": {"dispatch0": expected}}
    for _ in range(40):
        graph.launch()
    np.testing.assert_array_equal(planes["x"], ref.x)


def test_graph_build_refuses_what_simulate_refuses():
    graph, _ = _graph_with(walker)
    with pytest.raises(ValueError, match="built graph cannot do"):
        graph.build(dispatch="segmented")
    graph, _ = _graph_with(walker)
    with pytest.raises(ValueError, match="unknown"):
        graph.build(dispatch="fastest")
    graph = eagle.Graph()
    from eagle._plan_deploy import auto
    graph.add(auto(plain).host.bind(x=np.zeros(4), t=np.zeros(4), terminated=np.zeros(4, dtype=bool),
                                    finished_count=np.zeros(1, dtype=np.uint32)))
    with pytest.raises(ValueError, match="no plan of this graph has one"):
        graph.build(dispatch="switch")


def test_graph_build_without_dispatch_leaves_a_plain_graph_alone():
    graph = eagle.Graph()
    from eagle._plan_deploy import auto
    graph.add(auto(plain).host.bind(x=np.zeros(4), t=np.zeros(4), terminated=np.zeros(4, dtype=bool),
                                    finished_count=np.zeros(1, dtype=np.uint32)))
    graph.build()
    assert graph.dispatch == {}


@pytest.mark.gpu
def test_device_calibration_timer_keeps_every_repetition():
    cp = pytest.importorskip("cupy")
    from eagle.plan import auto

    x, t, done = cp.zeros(64), cp.zeros(64), cp.zeros(64, dtype=cp.bool_)
    plan = auto(plain).select({"x": x})
    extra = {nm: cp.zeros(1, dtype=cp.int64 if nm == "fused_steps" else cp.int32)
             for role, nm in plan.plugin.arg_spec if role == "lookup"}
    seconds = door.time_launch(plan.bind(x=x, t=t, terminated=done, **extra), 4)
    assert len(seconds) == 4 and all(s > 0.0 for s in seconds)
    np.testing.assert_array_equal(cp.asnumpy(x), np.full(64, 4.0))


@pytest.mark.gpu
def test_device_two_threads_with_two_simulations_each_get_the_right_answer():
    cp = pytest.importorskip("cupy")
    ref = _device_run(walker, rare_classes(), dispatch="switch")
    out, errors = {}, []

    def work(name, dispatch):
        try:
            out[name] = _device_run(walker, rare_classes(), dispatch=dispatch)
        except Exception as exc:        # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i, d)) for i, d in enumerate((None, "segmented"))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    for got in out.values():
        np.testing.assert_array_equal(cp.asnumpy(got.x), cp.asnumpy(ref.x))


@pytest.mark.gpu
def test_device_calibration_timer_runs_on_distinct_bound_plans_in_many_threads():
    cp = pytest.importorskip("cupy")
    from eagle.plan import auto

    plan = auto(plain).select({"x": cp.zeros(64)})
    threads_n, reps = 4, 5
    bound, xs, errors = [], [], []
    for _ in range(threads_n):
        x, t, done = cp.zeros(64), cp.zeros(64), cp.zeros(64, dtype=cp.bool_)
        extra = {nm: cp.zeros(1, dtype=cp.int64 if nm == "fused_steps" else cp.int32)
                 for role, nm in plan.plugin.arg_spec if role == "lookup"}
        bound.append(plan.bind(x=x, t=t, terminated=done, **extra))
        xs.append(x)
    start = threading.Barrier(threads_n)

    def work(b):
        try:
            start.wait()
            assert len(door.time_launch(b, reps)) == reps
        except Exception as exc:        # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(b,)) for b in bound]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors
    for x in xs:
        np.testing.assert_array_equal(cp.asnumpy(x), np.full(64, float(reps)))


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_one_dispatch_simulation_run_by_two_threads_at_once_is_busy(monkeypatch, backend):
    xp = pytest.importorskip("cupy") if backend == "gpu" else np
    entered, release = threading.Event(), threading.Event()
    real = door.DispatchRunner._run

    def gated(self):
        entered.set()
        assert release.wait(30)
        return real(self)

    monkeypatch.setattr(door.DispatchRunner, "_run", gated)
    sim = eagle.simulation(walker, k=xp.asarray(rare_classes()), x=xp.zeros(N), t=xp.zeros(N),
                           max_steps=200, dispatch="segmented")
    t = threading.Thread(target=sim.run)
    t.start()
    try:
        assert entered.wait(30)
        with pytest.raises(eagle.Busy, match=r"eagle\.Simulation\.run: another call is still running"):
            sim.run()
        with pytest.raises(eagle.Busy, match=r"eagle\.Runner\.run: another call is still running"):
            sim.runner.run()
    finally:
        release.set()
        t.join(60)
    assert not t.is_alive()
