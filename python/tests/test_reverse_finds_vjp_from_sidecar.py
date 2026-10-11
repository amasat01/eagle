# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Where each node's vjp comes from: derived by eagle beside a plan made by
eagle.plan.auto, or passed as vjps= (checked against the kernel it
differentiates); a vjp is never a step of a forward graph."""

import numpy as np
import pytest

import eagle
import eagle.plan
from eagle.exec import HostTeam
from _reverse_fixtures import chain, plans, rv_sin, rv_square

hawk = pytest.importorskip("hawk")
from hawk import Kernel  # noqa: E402
from hawk.artifact import build_bundle, plugins  # noqa: E402
from hawk.diff import param_planes, vjp  # noqa: E402

N = 32


@pytest.fixture
def hand_built(tmp_path):
    """Host plugins of a bundle built by hand: the kernels a recorded graph
    runs (``a`` read from a one-cell plane) and their vjps."""
    square = param_planes(rv_square)
    kernels = [square, Kernel("rv_square_params_vjp", vjp(square)),
               rv_sin, Kernel("rv_sin_vjp", vjp(rv_sin))]
    return plugins(build_bundle(kernels, tmp_path / "bundle", targets=("host",)))


def _raw_chain(by_name, x, a):
    y, z = np.zeros_like(x), np.zeros_like(x)
    sq = eagle.plan.plan(by_name["rv_square_params"], structure=HostTeam)
    sn = eagle.plan.plan(by_name["rv_sin"], structure=HostTeam)
    g = (eagle.Graph().add(sq.bind(x=x, a=np.full(1, a), y=y), name="square")
         .add(sn.bind(x=y, y=z), name="sin").build(record=True))
    return g.launch()


def _vjp_plans(by_name):
    return {"square": eagle.plan.plan(by_name["rv_square_params_vjp"], structure=HostTeam),
            "sin": eagle.plan.plan(by_name["rv_sin_vjp"], structure=HostTeam)}


def test_an_auto_plan_finds_its_vjp_and_keeps_it(tmp_path):
    ps = plans(tmp_path, "cpu")
    x = np.linspace(-1, 1, N)
    g, _z = chain(ps, x, 0.5, record=True)
    g.launch()
    grads = eagle.diff.reverse(g.record, {"sin.y": np.ones(N)})
    found = ps[0].plugin._params._vjp      # the vjp of the parameter-plane kernel it runs
    assert found.derivative["kind"] == "vjp" and found.derivative["primal"] == "rv_square_params"
    np.testing.assert_allclose(grads["x"], np.cos(0.5 * x * x) * x, rtol=1e-12)


def test_vjps_override_the_found_ones_with_the_same_answer(tmp_path, hand_built):
    ps = plans(tmp_path, "cpu")
    x = np.linspace(-1, 1, N)
    g, _z = chain(ps, x, 0.5, record=True)
    g.launch()
    found = eagle.diff.Reverse(g)(g.record, {"sin.y": np.ones(N)})
    given = eagle.diff.Reverse(g, vjps=_vjp_plans(hand_built))(g.record, {"sin.y": np.ones(N)})
    np.testing.assert_array_equal(found["x"], given["x"])
    assert float(found["a"]) == float(given["a"])


def test_a_raw_plugin_needs_vjps(hand_built):
    g = _raw_chain(hand_built, np.linspace(-1, 1, N), 0.5)
    with pytest.raises(ValueError, match=(
            r"eagle.diff.Reverse: node 'square' has no vjp kernel and none was passed: "
            r"eagle derives one only for a plan made by eagle.plan.auto\(kernel\). "
            r"Build hawk.diff.vjp\(<kernel 'rv_square_params'>\) and pass "
            r"vjps=\{'square': plan\}.")):
        eagle.diff.Reverse(g)
    grads = eagle.diff.Reverse(g, vjps=_vjp_plans(hand_built))(g.record, {"sin.y": np.ones(N)})
    x = np.linspace(-1, 1, N)
    np.testing.assert_allclose(grads["x"], np.cos(0.5 * x * x) * x, rtol=1e-12)


def test_a_mismatched_vjp_is_refused_naming_both_kernels(hand_built):
    g = _raw_chain(hand_built, np.linspace(-1, 1, N), 0.5)
    wrong = _vjp_plans(hand_built)
    wrong["square"] = wrong["sin"]
    with pytest.raises(ValueError, match=(
            r"vjps\['square'\] differentiates kernel 'rv_sin', but node 'square' "
            r"launches 'rv_square_params'; pass the vjp of 'rv_square_params'.")):
        eagle.diff.Reverse(g, vjps=wrong)


def test_a_primal_passed_as_a_vjp_is_refused(hand_built):
    g = _raw_chain(hand_built, np.linspace(-1, 1, N), 0.5)
    vjps = _vjp_plans(hand_built)
    vjps["square"] = eagle.plan.plan(hand_built["rv_square_params"], structure=HostTeam)
    with pytest.raises(ValueError, match=r"vjps\['square'\] is not a vjp kernel"):
        eagle.diff.Reverse(g, vjps=vjps)


def test_vjps_naming_no_node_is_refused(hand_built):
    g = _raw_chain(hand_built, np.linspace(-1, 1, N), 0.5)
    with pytest.raises(ValueError, match="vjps names 'nope', which is not a node"):
        eagle.diff.Reverse(g, vjps={"nope": None})


def test_a_raw_plugin_with_by_value_parameters_is_refused_with_the_fix(tmp_path):
    (sq,) = plugins(build_bundle([rv_square], tmp_path / "b", targets=("host",))).values()
    x = np.zeros(N)
    g = eagle.Graph().add(eagle.plan.plan(sq, structure=HostTeam).bind(x=x, a=1.0, y=x.copy()),
                          name="square")
    with pytest.raises(ValueError, match=(
            r"node 'square' takes \['a'\] by value, and a recorded graph holds its "
            r"parameters in one-cell planes .* make the plan with eagle.plan.auto\(kernel\), "
            r"or bind a kernel made with hawk.diff.param_planes\(kernel\)")):
        g.build(record=True)


def test_a_prebuilt_vjp_plugin_runs_alone_but_is_refused_in_a_graph(hand_built):
    vp = hand_built["rv_square_params_vjp"]
    eagle.plan.auto(vp)   # a plan of its own
    x = np.zeros(N)
    bound = eagle.plan.plan(vp, structure=HostTeam).bind(
        x=x, a=np.ones(1), bar_y=x.copy(), bar_x=x.copy(), bar_a=x.copy())
    with pytest.raises(ValueError, match="is a vjp derivative kernel of 'rv_square_params'"
                                         ".*enters those only through eagle.diff"):
        eagle.Graph().add(bound)
