# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A derivative (VJP/JVP) kernel runs on its own as a plan, and is refused,
with the message pinned, at every door that composes steps; a ``prior_read`` plane is refused at the reverse door's
shared helper; the conditional reserved names bind only while a recording,
event-refining or reverse run is on. CPU-only: the refusals fire before any
build or launch."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

import eagle
from eagle import _until_done as ud
from eagle import plan as plan_mod
from eagle.sidecar import derived_tag, refuse_derived, refuse_prior_read

_BLOCK = {"kind": "vjp", "primal": "poly", "wrt": ["x"], "residuals": []}

#: The pinned text every door shares, after its own ``<door>: <what> is a``.
_TAIL = ("derivative kernel of 'poly'; a derivative runs on its own as a plan, "
         "but it is not a step of a pipeline, Graph, until_done or simulate -- it "
         "enters those only through eagle.diff, which pairs it with its primal")


def _artifact():
    return SimpleNamespace(derivative=dict(_BLOCK))


def _traced():
    sinks = SimpleNamespace(kind="jvp", primal="poly")
    return SimpleNamespace(sinks=sinks, walk=object(), name="poly_jvp")


def _plan(plugin):
    return SimpleNamespace(plugin=plugin, bind=lambda **kw: None)


def test_detection_sees_through_wrappers():
    art = _artifact()
    assert derived_tag(art) == ("vjp", "poly")
    assert derived_tag(_traced()) == ("jvp", "poly")
    assert derived_tag(_plan(art)) == ("vjp", "poly")
    assert derived_tag(functools.partial(print, art)) == ("vjp", "poly")

    class Holder:
        def launch(self):
            pass
        derivative = _BLOCK

    assert derived_tag(Holder().launch) == ("vjp", "poly")
    assert derived_tag(SimpleNamespace(derivative=None)) is None
    assert derived_tag(lambda: None) is None
    assert derived_tag(object()) is None


def test_refuse_derived_message_is_pinned():
    with pytest.raises(ValueError) as exc:
        refuse_derived(_artifact(), door="eagle.X", what="the step")
    assert str(exc.value) == f"eagle.X: the step is a vjp {_TAIL}"


def test_until_done_refuses_a_derived_step():
    with pytest.raises(ValueError) as exc:
        ud.until_done(_plan(_artifact()), max_steps=3)
    assert str(exc.value) == f"eagle.until_done: step 0 is a vjp {_TAIL}"
    with pytest.raises(ValueError, match=r"step 1 is a vjp"):
        ud.until_done([_plan(SimpleNamespace(derivative=None)), _plan(_artifact())],
                      max_steps=3)


def test_simulate_refuses_a_derived_model_item():
    with pytest.raises(ValueError) as exc:
        eagle.simulate(_traced(), max_steps=3)
    assert str(exc.value) == f"eagle.simulate: item 0 of the model is a jvp {_TAIL}"
    with pytest.raises(ValueError, match=r"item 1 of the model is a vjp"):
        eagle.simulate([object(), _plan(_artifact())], max_steps=3)


@pytest.mark.gpu
def test_pipeline_doors_refuse_a_derived_step():
    pytest.importorskip("cupy")
    pipe = eagle.GraphPipeline()
    with pytest.raises(ValueError) as exc:
        pipe.add(functools.partial(print, _artifact()))
    assert str(exc.value) == f"eagle.GraphPipeline.add: the step is a vjp {_TAIL}"
    with pytest.raises(ValueError, match=r"add_concurrent: member 1 is a vjp"):
        pipe.add_concurrent([lambda: None, functools.partial(print, _artifact())])


def test_prior_read_refusal_is_hawks_message():
    with pytest.raises(ValueError) as exc:
        refuse_prior_read({"y", "z"}, door="eagle.diff.reverse")
    assert str(exc.value) == (
        "eagle.diff.reverse: 'y' is read before it is assigned: a plane's "
        "launch-start value is a recurrence across launches; its derivative "
        "needs the value the primal launch overwrites, and eagle keeps no tape "
        "for it -- differentiate the per-launch term instead")
    refuse_prior_read((), door="eagle.diff.reverse")   # none: silent


def test_conditional_reserved_names_only_under_their_modes():
    assert ud._RESERVED == (ud.FINISHED_PLANE, ud.MAP_PLANE, ud.COUNT_PLANE,
                            ud.FUSED_STEPS_PLANE)
    extra = {"applied_steps", "t_stop", "tape", "event_status"}
    assert not extra & set(ud._reserved_names())
    assert not extra & set(ud._reserved_names(events=None))
    for kw in ({"record": True}, {"events": "refine"}, {"reverse": True}):
        assert extra <= set(ud._reserved_names(**kw))
        assert set(ud._RESERVED) <= set(ud._reserved_names(**kw))


def test_eagle_diff_is_the_reverse_not_a_stub():
    assert callable(eagle.diff.reverse) and callable(eagle.diff.Reverse)


def test_plan_auto_runs_a_derivative_kernel_on_its_own(tmp_path):
    """Deployed next to its primal, a jvp kernel is a plan of its own and gives
    the directional derivative."""
    hawk = pytest.importorskip("hawk")
    import numpy as np
    from hawk.diff import jvp
    from hawk.trace.kernel import Kernel

    @hawk.kernel
    def poly2(x: hawk.Scalar, v: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x * x + v

    _, fwd = plan_mod.auto([poly2, Kernel("poly2_jvp", jvp(poly2, wrt=("x",)))],
                           cache_dir=str(tmp_path), targets=("host",))
    x = np.linspace(0.5, 2.0, 8)
    got = fwd.run(x=x, v=np.zeros(8), dot_x=np.ones(8))
    np.testing.assert_allclose(np.asarray(got), 2 * x)
