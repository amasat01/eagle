# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lane space a kernel declares (sidecar ``launch.lane_space``), at bind.

Rows: the sidecar reader (a good block, an undeclared sidecar, each bad shape
named by its key, ``launch.block``); a declared kernel takes its first-axis
extent by label or from a lane-shaped plane (``(64, 8192)``, ``(W, 64, 8192)``),
returns lane-shaped outputs, refuses ``n=``, a contradicting shape and an extent
on an all-static space; and a launch that binds only Tables is refused whether
or not the lane space is declared (a Table's length is never the lane count).
The kernels are real hawk builds; the lane space is stamped on the plugin the
way the sidecar reader hands it over.
"""

import copy
import types

import numpy as np
import pytest

import eagle
import eagle.exec as eexec
from eagle import _lane_bind, sidecar
from eagle.plan import plan as eplan
from eagle._plan_deploy import AutoPlan

hawk = pytest.importorskip("hawk")
import hawk.math as m  # noqa: E402
from hawk import Grid, Mutable, Param, Scalar, Table, Vector  # noqa: E402


@hawk.kernel
def scale(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x


@hawk.kernel
def swirl(v: Vector[2], a: Param, w: Mutable[Vector[2]]):
    w = a * v


@hawk.kernel
def first(t: Table[Scalar], y: Mutable[Scalar]):
    y = t.at(0) * 2.0


BINS = 16


@hawk.kernel
def hist(lane: Grid["i"], x: Scalar, bins: Mutable["bin":BINS]):
    hawk.scatter(bins)[m.index(x * BINS, BINS)] += 1.0


@hawk.kernel
def rowsum(lane: Grid["channel", "sample":8], x: Scalar) -> Scalar:
    return hawk.sum(x, over="sample")


@hawk.kernel
def vsum(lane: Grid["channel", "sample":2], x: Scalar) -> Vector[3]:
    return hawk.sum(m.vec(x, 2.0 * x, 3.0 * x), over="sample")


def _space(*axes, group=1):
    return sidecar.LaneSpace(tuple(axes), group)


CHANNEL_BY_SAMPLE = _space(("channel", None), ("sample", 16))
ALL_STATIC = _space(("channel", 4), ("sample", 16))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("hawk"))
    return {k.name: eagle.deploy(k, targets=("host",), cache_dir=cache)
            for k in (scale, swirl, first)}


def declared(built, name, space):
    """``name`` as an auto plan whose plugin declares ``space``."""
    base = built[name]
    plugin = types.SimpleNamespace(**{**vars(base.plugin), "lane_space": space})
    return AutoPlan(plugin, {}, None, base.kernel)


# --------------------------------------------------------------------------- #
# The sidecar reader
# --------------------------------------------------------------------------- #
GOOD = {"axes": [{"label": "channel", "extent": None},
                 {"label": None, "extent": 16}], "group": 32}


def test_reader_gives_axes_as_tuples_and_the_group():
    got = sidecar.read_lane_space({"launch": {"lane_space": GOOD}}, name="k")
    assert got == sidecar.LaneSpace((("channel", None), (None, 16)), 32)


def test_reader_undeclared_is_one_unlabeled_axis_of_group_one():
    assert sidecar.read_lane_space({}, name="k") == sidecar.UNDECLARED_LANE_SPACE
    assert sidecar.UNDECLARED_LANE_SPACE == sidecar.LaneSpace(((None, None),), 1)


@pytest.mark.parametrize("bad", [
    [],
    {"axes": GOOD["axes"]},
    {**GOOD, "extra": 1},
    {**GOOD, "group": 0},
    {**GOOD, "axes": []},
    {**GOOD, "axes": [{"label": "a", "extent": 4}, {"label": "b", "extent": None}]},
    {**GOOD, "axes": [{"label": "a", "extent": None}, {"label": "a", "extent": 2}]},
    {**GOOD, "axes": [{"label": "a"}]},
])
def test_reader_refuses_a_bad_block_naming_the_key(bad):
    meta = {"launch": {"lane_space": bad}}
    with pytest.raises(ValueError, match=r"launch\.lane_space"):
        sidecar.read_lane_space(meta, name="k")
    with pytest.raises(ValueError, match=r"launch\.lane_space"):
        sidecar.validate_sidecar({"arg_spec": [], **meta}, name="k")


def test_reader_block():
    assert sidecar.read_block({}, name="k") is None
    assert sidecar.read_block({"launch": {"block": 128}}, name="k") == 128
    for bad in (0, -1, 1.5, "128", True):
        with pytest.raises(ValueError, match=r"launch\.block"):
            sidecar.read_block({"launch": {"block": bad}}, name="k")


# --------------------------------------------------------------------------- #
# Extent by label, lane-shaped planes
# --------------------------------------------------------------------------- #
def test_a_lane_shaped_plane_sets_the_extent_and_the_output_comes_back_lane_shaped(built):
    k = declared(built, "scale", CHANNEL_BY_SAMPLE)
    x = np.arange(5 * 16, dtype=np.float64).reshape(5, 16)
    y = k(x, 3.0)
    assert y.shape == (5, 16)
    np.testing.assert_array_equal(y, 3.0 * x)
    # the plan door, run: the same
    plan = eplan(k.plugin, structure=eexec.HostTeam)
    out = plan.run(x=x, a=2.0)
    assert out.shape == (5, 16)
    np.testing.assert_array_equal(out, 2.0 * x)


def test_the_extent_by_label_sizes_a_launch_without_a_plane_of_that_shape(built):
    k = declared(built, "first", CHANNEL_BY_SAMPLE)
    t = np.array([1.5, 0.0])
    y = k(t, channel=7)
    assert y.shape == (7, 16) and np.all(y == 3.0)
    assert eplan(k.plugin, structure=eexec.HostTeam).run(t=t, channel=3).shape == (3, 16)


def test_a_given_output_plane_is_filled_in_place(built):
    k = declared(built, "scale", CHANNEL_BY_SAMPLE)
    x = np.ones((3, 16))
    y = np.zeros((3, 16))
    k(x, 4.0, y=y)
    assert np.all(y == 4.0)


def test_the_label_and_a_plane_must_agree(built):
    k = declared(built, "scale", CHANNEL_BY_SAMPLE)
    with pytest.raises(ValueError, match=r"channel=4 gives 64 lanes.*hold 48"):
        k(np.ones((3, 16)), 1.0, channel=4)


def test_a_vector_plane_takes_head_then_lane_shape(built):
    k = declared(built, "swirl", CHANNEL_BY_SAMPLE)
    v = np.random.default_rng(1).random((2, 6, 16))
    w = k(v, 2.0)
    assert w.shape == (2, 6, 16)
    np.testing.assert_array_equal(w, 2.0 * v)
    # the flat component-major form (W, N) is still the flat form
    flat = np.random.default_rng(2).random((2, 40))
    np.testing.assert_array_equal(k(flat, 2.0), 2.0 * flat)


def test_an_all_static_space_needs_no_extent(built):
    k = declared(built, "first", ALL_STATIC)
    y = k(np.array([1.0]))
    assert y.shape == (4, 16) and np.all(y == 2.0)
    with pytest.raises(ValueError, match="every axis.*static"):
        k(np.array([1.0]), channel=4)


def test_a_contradicting_lane_shape_is_refused(built):
    k = declared(built, "scale", CHANNEL_BY_SAMPLE)
    with pytest.raises(ValueError, match=r"lane shape \(3, 8\).*Grid\[\"channel\", \"sample\":16\]"):
        k(np.ones((3, 8)), 1.0)
    with pytest.raises(ValueError, match="C-contiguous"):
        k(np.ones((16, 3)).T, 1.0)


def test_n_is_refused_on_a_declared_kernel_naming_the_label(built):
    k = declared(built, "first", CHANNEL_BY_SAMPLE)
    with pytest.raises(ValueError, match=r"n= is not an argument.*pass channel=<extent>"):
        k(np.array([1.0, 0.0]), n=7)
    plan = eplan(k.plugin, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=r"pass channel=<extent>"):
        plan.run(t=np.array([1.0, 0.0]), n=7)


def test_a_static_axis_label_is_not_an_extent(built):
    k = declared(built, "first", CHANNEL_BY_SAMPLE)
    with pytest.raises(ValueError, match="axis 'sample'.*static"):
        k(np.array([1.0]), channel=2, sample=16)


# --------------------------------------------------------------------------- #
# A Table's length is never the lane count
# --------------------------------------------------------------------------- #
def test_a_tables_only_launch_is_refused_declared_or_not(built):
    t = np.arange(8.0)
    plain = built["first"]
    with pytest.raises(ValueError, match="never the lane count"):
        plain.run(t=t)
    with pytest.raises(ValueError, match="never the lane count"):
        plain(t)
    with pytest.raises(ValueError, match=r"pass channel=<extent>.*never the lane count"):
        declared(built, "first", CHANNEL_BY_SAMPLE)(t)
    # the output plane is a per-sample plane: it sets the count
    y = np.zeros(5)
    plain(t, y=y)
    assert np.all(y == t[0] * 2.0)


def test_bind_takes_the_same_door(built):
    k = declared(built, "first", CHANNEL_BY_SAMPLE)
    plan = eplan(k.plugin, structure=eexec.HostTeam)
    t = np.array([2.0])
    with pytest.raises(ValueError, match="never the lane count"):
        plan.bind(t=t)
    y = np.zeros((3, 16))
    bound = plan.bind(t=t, y=y)
    assert bound.n == 48
    bound.launch()
    assert np.all(y == 4.0)
    y2 = np.zeros((2, 16))
    plan.bind(t=t, y=y2, channel=2).launch()
    assert np.all(y2 == 4.0)


def test_a_plugin_may_carry_the_sidecar_block_as_is(built):
    base = built["scale"].plugin
    assert _lane_bind.declared_axes(base) == ()
    dup = copy.copy(base)
    dup.lane_space = {"axes": [{"label": "channel", "extent": None}], "group": 1}
    assert _lane_bind.declared_axes(dup) == (("channel", None),)


@pytest.mark.gpu
def test_device_arm_lane_shaped_planes_and_extent_by_label(tmp_path):
    cp = pytest.importorskip("cupy")
    auto = eagle.deploy(scale, cache_dir=str(tmp_path))
    plugin = types.SimpleNamespace(**{**vars(auto.plugin), "lane_space": CHANNEL_BY_SAMPLE})
    k = AutoPlan(plugin, {}, None, auto.kernel)
    x = cp.arange(5 * 16, dtype=cp.float64).reshape(5, 16)
    y = k(x, 3.0)
    assert isinstance(y, eagle.Array) and y.shape == (5, 16)
    assert cp.array_equal(cp.asarray(y), 3.0 * x)
    bound = k.bind(x=x, a=2.0, y=cp.zeros((5, 16)))
    assert bound.n == 80
    with pytest.raises(ValueError, match=r"pass channel=<extent>"):
        k(x, 3.0, n=5)


# --------------------------------------------------------------------------- #
# Sinks with named axes and reduced outputs: allocated in their own shape
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sinks(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("hawk_sinks"))
    return {k.name: eagle.deploy(k, targets=("host",), cache_dir=cache)
            for k in (hist, rowsum, vsum)}


def _counts(x):
    return np.bincount((x * BINS).astype(int), minlength=BINS).astype(float)


def test_the_output_shape_rule(sinks):
    plugin = sinks["vsum"].plugin
    assert plugin.plane_axes["out"]["payload"] == [3]
    assert _lane_bind.output_shape(plugin, "out", 1, 8) == (3, 4)
    assert _lane_bind.output_shape(sinks["hist"].plugin, "bins", 1, 4) == (BINS,)
    assert _lane_bind.output_shape(sinks["rowsum"].plugin, "out", 1, 40) == (5,)
    assert _lane_bind.output_shape(sinks["rowsum"].plugin, "x", 1, 40) == (40,)


def test_an_omitted_axis_sink_is_allocated_in_its_own_shape_and_zeroed(sinks):
    """n = 4 lanes, 16 bins: a ``(n,)`` plane was the wrong shape and too short."""
    x = np.array([0.01, 0.01, 0.5, 0.99])
    bins = sinks["hist"](x=x)
    assert bins.shape == (BINS,)
    np.testing.assert_array_equal(bins, _counts(x))


def test_a_passed_axis_sink_keeps_its_contents(sinks):
    x = np.array([0.01, 0.01, 0.5, 0.99])
    bins = np.arange(BINS, dtype=np.float64)
    out = sinks["hist"](x=x, bins=bins)
    np.testing.assert_array_equal(bins, np.arange(BINS) + _counts(x))
    assert out is bins or np.shares_memory(out, bins)


def test_a_passed_axis_sink_of_another_shape_is_refused_by_name(sinks):
    x = np.zeros(4)
    with pytest.raises(ValueError, match=r"'bins' is declared with the axes \(bin=16\).*\(16,\).*got \(4,\)"):
        sinks["hist"](x=x, bins=np.zeros(4))
    plan = eplan(sinks["hist"].plugin, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=r"eagle\.plan\.bind: 'bins' is declared"):
        plan.bind(x=x, bins=np.zeros(BINS + 1))
    bound = plan.bind(x=x, bins=np.zeros(BINS))
    with pytest.raises(ValueError, match=r"eagle\.plan\.rebind: 'bins' is declared"):
        bound.rebind(bins=np.zeros(3))


def test_a_reduced_output_is_the_kept_axes_not_the_lane_count(sinks):
    x = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    out = sinks["rowsum"](x=x)
    assert out.shape == (5,)
    np.testing.assert_allclose(out, x.sum(axis=1))
    out = sinks["rowsum"](channel=3, x=np.ones((3, 8)))
    np.testing.assert_array_equal(out, np.full(3, 8.0))


def test_payload_times_kept_axis_longer_than_the_lane_count(sinks):
    """3 components x 4 rows = 12 cells over 8 lanes: the buffer an ``(n,)``
    allocation gave was overrun."""
    x = np.arange(8, dtype=np.float64).reshape(4, 2)
    out = sinks["vsum"](x=x)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0])[:, None] * x.sum(axis=1))


def test_the_bound_door_takes_the_sink_it_is_given(sinks):
    x = np.array([0.01, 0.5])
    bins = np.zeros(BINS)
    eplan(sinks["hist"].plugin, structure=eexec.HostTeam).bind(x=x, bins=bins).launch()
    np.testing.assert_array_equal(bins, _counts(x))


@pytest.mark.gpu
def test_device_arm_sinks_and_reduced_outputs(tmp_path):
    cp = pytest.importorskip("cupy")
    run = {k.name: eagle.deploy(k, cache_dir=str(tmp_path)) for k in (hist, rowsum, vsum)}
    x = cp.asarray([0.01, 0.01, 0.5, 0.99])
    bins = run["hist"](x=x)
    assert bins.shape == (BINS,)
    np.testing.assert_array_equal(cp.asnumpy(bins), _counts(cp.asnumpy(x)))
    keep = cp.arange(BINS, dtype=cp.float64)
    run["hist"](x=x, bins=keep)
    np.testing.assert_array_equal(cp.asnumpy(keep), np.arange(BINS) + _counts(cp.asnumpy(x)))
    with pytest.raises(ValueError, match=r"'bins' is declared with the axes \(bin=16\)"):
        run["hist"](x=x, bins=cp.zeros(4))
    bound = run["hist"].bind(x=x, bins=cp.zeros(BINS))
    with pytest.raises(ValueError, match=r"rebind: 'bins' is declared"):
        bound.rebind(bins=cp.zeros(3))
    xs = cp.arange(8, dtype=cp.float64).reshape(4, 2)
    out = run["vsum"](x=xs)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(cp.asnumpy(out),
                               np.array([1.0, 2.0, 3.0])[:, None] * cp.asnumpy(xs).sum(axis=1))
    assert run["rowsum"](x=cp.ones((5, 8))).shape == (5,)


# --------------------------------------------------------------------------- #
# HostPluginLibrary reads the launch declaration like the device loader
# --------------------------------------------------------------------------- #
def test_the_host_plugin_library_carries_the_launch_declaration(tmp_path):
    import pathlib

    from eagle.host_launch import HostPluginLibrary
    from hawk.artifact import build_bundle

    bundle = build_bundle([hist], pathlib.Path(tmp_path) / "b", targets=("host",),
                          host_profile="native")
    art = bundle.artifacts[0]
    meta = {**art.sidecar, "launch": {**art.sidecar["launch"], "block": 128}}
    lib = HostPluginLibrary(art.entries["host"].artifact, meta)
    assert lib.lane_space == sidecar.LaneSpace((("i", None),), 1)
    assert lib.block == 128
    assert lib.plane_axes == meta["plane_axes"]
    assert lib.sorted_inputs == ()
    assert _lane_bind.declared_axes(lib) == (("i", None),)
    assert sidecar.read_launch(meta, name="k") == (lib.lane_space, lib.block, lib.plane_axes,
                                                   lib.sorted_inputs, lib.box)


def test_a_malformed_plane_axes_is_refused_naming_the_key():
    meta = {"arg_spec": [["out", "o"]], "plane_axes": {"o": {"payload": [0], "axes": []}}}
    with pytest.raises(ValueError, match="plane_axes"):
        sidecar.read_plane_axes(meta, name="k")
    with pytest.raises(ValueError, match="plane_axes"):
        sidecar.read_plane_axes({**meta, "plane_axes": {"nope": {"payload": [], "axes": []}}},
                                name="k")


def test_rebind_takes_a_lane_shaped_plane_as_a_view(built):
    k = declared(built, "scale", CHANNEL_BY_SAMPLE)
    x = np.arange(3 * 16, dtype=np.float64).reshape(3, 16)
    y = np.zeros((3, 16))
    bound = eplan(k.plugin, structure=eexec.HostTeam).bind(x=x, a=2.0, y=y)
    x2 = x[::-1].copy()
    bound.rebind(x=x2).launch()
    np.testing.assert_array_equal(y, 2.0 * x2)
    with pytest.raises(ValueError, match="contradicts the kernel's declared"):
        bound.rebind(x=np.zeros((3, 8)))
    with pytest.raises(ValueError, match="n=48"):
        bound.rebind(x=np.zeros((2, 16)))


def test_positional_args_skip_the_grid_declaration(sinks):
    """``lane: Grid[...]`` is a declaration, not a plane: positional arguments
    start at the first plane, as the keyword call does."""
    x = np.array([0.01, 0.01, 0.5, 0.99])
    keep = np.arange(BINS, dtype=np.float64)
    want = sinks["hist"](x=x, bins=keep.copy())
    np.testing.assert_array_equal(sinks["hist"](x), want - np.arange(BINS))
    mine = keep.copy()
    sinks["hist"](x, mine)
    np.testing.assert_array_equal(mine, want)
    rows = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    np.testing.assert_array_equal(sinks["rowsum"](rows), sinks["rowsum"](x=rows))
    with pytest.raises(ValueError, match="too many positional arguments"):
        sinks["hist"](x, mine, mine)


def test_a_compensated_sums_companion_is_bound_by_name_only(sinks):
    """The companion of ``bins`` is eagle's and takes no position; a caller
    that passes its own by name keeps it: the counts land in ``bins`` plus
    ``bins_c``, which eagle neither folds nor replaces."""
    x = np.array([0.01, 0.01, 0.5, 0.99])
    bins, comp = np.zeros(BINS), np.zeros(BINS)
    sinks["hist"](x, bins, bins_c=comp)
    np.testing.assert_array_equal(bins + comp, _counts(x))
    np.testing.assert_array_equal(comp, np.zeros(BINS))
    comp[3] = 0.5      # the caller's own state, never folded into the target
    sinks["hist"](x, bins, bins_c=comp)
    assert comp[3] == 0.5
    np.testing.assert_array_equal(bins, 2 * _counts(x))


@pytest.mark.gpu
def test_device_arm_positional_args_skip_the_grid_declaration(tmp_path):
    cp = pytest.importorskip("cupy")
    k = eagle.deploy(hist, cache_dir=str(tmp_path))
    x = cp.asarray([0.01, 0.01, 0.5, 0.99])
    keep = cp.arange(BINS, dtype=cp.float64)
    k(x, keep)
    np.testing.assert_array_equal(cp.asnumpy(keep), np.arange(BINS) + _counts(cp.asnumpy(x)))
    np.testing.assert_array_equal(cp.asnumpy(k(x)), _counts(cp.asnumpy(x)))


# --------------------------------------------------------------------------- #
# The graph builder takes the same extent keywords as a direct call
# --------------------------------------------------------------------------- #


def _first_in_a_builder(k, t, **extent):
    @eagle.Graph.builder
    def tables_only(t):
        g = eagle.Graph()
        g.result = k(t, **extent)
        return g

    return tables_only(t)


def test_a_builder_binds_a_tables_only_grid_kernel_by_label_like_the_direct_call(built):
    k = declared(built, "first", CHANNEL_BY_SAMPLE)
    t = np.array([1.5, 0.0])
    g = _first_in_a_builder(k, t, channel=7)
    assert g.backend == "cpu"
    np.testing.assert_array_equal(np.asarray(g()).ravel(), k(t, channel=7).ravel())


def test_a_builder_refuses_a_wrong_label_and_n_naming_the_declared_ones(built):
    k = declared(built, "first", CHANNEL_BY_SAMPLE)
    t = np.array([1.5, 0.0])
    with pytest.raises(ValueError, match=r"n= is not an argument.*pass channel=<extent>"):
        _first_in_a_builder(k, t, n=7)
    with pytest.raises(ValueError, match=r"axis 'sample'.*static"):
        _first_in_a_builder(k, t, channel=2, sample=16)
    with pytest.raises(ValueError, match=r"binds no per-sample plane.*pass channel=<extent>"):
        _first_in_a_builder(k, t)
    with pytest.raises(ValueError, match=r"unexpected keyword 'row'.*Grid\[\"channel\", \"sample\":16\].*channel=<extent>"):
        _first_in_a_builder(k, t, row=7)
    with pytest.raises(ValueError, match=r"unexpected keyword 'row'.*channel=<extent>"):
        k(np.array([1.5, 0.0]), row=7)             # the direct call says the same


@pytest.mark.gpu
def test_a_builder_binds_a_tables_only_grid_kernel_by_label_on_the_gpu(tmp_path):
    cp = pytest.importorskip("cupy")
    base = eagle.deploy(first, cache_dir=str(tmp_path))
    k = declared({"first": base}, "first", CHANNEL_BY_SAMPLE)
    t = cp.asarray([1.5, 0.0])
    g = _first_in_a_builder(k, t, channel=7)
    assert g.backend == "gpu"
    np.testing.assert_array_equal(g().get().ravel(), k(t, channel=7).get().ravel())
