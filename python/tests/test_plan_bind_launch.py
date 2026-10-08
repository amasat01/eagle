# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The capture-legal, by-name, caller-supplied-plane launch door.

WHAT WAS MISSING. eagle offered two doors to a compiled kernel and neither was
the shape a capture-recording consumer has. The v1 door
(``eagle.loaded.LoadedPure.launch``) is capture-legal and packs the v1
parameter block; handed an ``aether-abi/2`` entry — whose parameters are the
by-value mirrors FOLLOWED by the partition triple — it built an argument
array of the wrong length and the outcome was undefined rather than diagnosed
(measured on this box: SIGSEGV, and ``CUDA_ERROR_INVALID_VALUE`` for a
differently-shaped body). The v2 door (``eagle.plan.Plan.run``) speaks the
triple correctly and is not capture-legal: it uploads its inputs, allocates
its output planes, ``deviceSynchronize()``s and copies the answers home with
``cp.asnumpy`` — and its sample-count probe called ``np.asarray`` on a bound
plane, which cupy refuses outright, so it could not even be HANDED a device
plane. Those six facts are executed as rows in a downstream sibling's
test suite; this file is the other side of them.

WHAT LANDED. ``Plan.bind(**planes) -> BoundPlan`` (packs the block ONCE over
the caller's own planes, checking every one and refusing by NAME) and
``BoundPlan.launch(stream=None)`` (exactly the structure's launches for the
plan's partitions, and nothing else), plus ``BoundPlan.rebind(**changed)``,
and a LOUD refusal at the v1 door for a v2 artifact.

THE SUBJECT IS A REAL HAWK ARTIFACT, not a hand-mirrored fixture. A fixture
that re-declares the v2 ABI in a local ``struct`` pins the copy, not the
contract, and the whole reason this door exists is a consumer holding a real
emission. So the module builds HAWK bundles with HAWK's own publisher
(``hawk.artifact.build_bundle``), loads them through eagle's own manifest door
(``eagle.registry.load_manifest``), verifies the LOADED module's exported
``eagle_abi_tag`` / ``eagle_layout_sizes`` against this build, and
names the published unit's digest — a gate builds its own inputs, and the
verdict names the artifact it was reached on.

FOUR SUBJECTS, one per shape the door must serve, and four bundles because the
execution axis is a MANIFEST-level declaration and these four disagree
about it: ``e4_vec3_scale`` (``sample_local``, a ``Vector[3]`` input and a
``Mutable[Vector[3]]`` output — the 40-byte mirror on both sides, plus a
uniform), ``e4_gather`` (``cross_sample_read``, the ``Table`` read whose lookup
plane the v1 door could not even NAME), ``e4_energy`` (``mapreduce``), and
``e4_scatter`` (``cross_sample_write``, whose ``Accum`` plane is
read-modify-written — the one shape that can show a replay accumulating).

RED HISTORY (measured with this file's own source in place and eagle's two
changed modules reverted):

* every ``bind``/``launch``/``rebind`` row — ``AttributeError: 'Plan' object
  has no attribute 'bind'``. A new door has no subtler red than its absence;
  what makes each row more than that is the assertion AFTER it (bit-identity
  against ``Plan.run``, the launch COUNT, the replay arithmetic), each of which
  is checked below on the door that now exists.
* the v1-door row — the child interpreter died of ``Signals.SIGSEGV`` (rc
  ``-11``) after printing ``LOADED``; it now exits 0 having printed ``RAISED
  ValueError``.
* ``test_bind_reads_the_sample_count_off_a_device_plane`` — the earlier probe
  raised ``TypeError: Implicit conversion to a NumPy array is not allowed.
  Please use `.get()` to construct a NumPy array explicitly.`` The row keeps
  that measured refusal as its own control: ``np.asarray`` must STILL raise on
  the plane that ``bind`` now reads happily, or the row would be proving
  nothing about how the extent was obtained.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from _device_file import device_artifact
import eagle._plan_pack as plan_pack  # noqa: E402

hawk_trace = pytest.importorskip("hawk.trace")
hawk_artifact = pytest.importorskip("hawk.artifact")
hawk_math = pytest.importorskip("hawk.math")

#: The device architecture the fixture bundles are emitted for: the GPU this
#: process runs on (hawk's own resolver), so the bundles load on any card.
DEVICE_ARCH = hawk_artifact.arch()

#: How many entries the layout array carries (``plugin/gref_abi.h``).
LAYOUT_FIELDS = 5

W = 3
N = 8


# --------------------------------------------------------------------------- #
# The four subjects. Bodies are exact in binary floating point (a scaling by 2,
# a doubling, a permuted sum of exactly-representable values), so every
# comparison below is bit-for-bit and no wrong answer can hide in a tolerance.
# --------------------------------------------------------------------------- #
@hawk_trace.kernel
def e4_vec3_scale(x: hawk_trace.Vector[W], a: hawk_trace.Param,
                  y: hawk_trace.Mutable[hawk_trace.Vector[W]]):
    """``sample_local``: the mirror on both sides plus a by-value uniform."""
    y = a * x


@hawk_trace.kernel
def e4_gather(table: hawk_trace.Table[hawk_trace.Scalar], where: hawk_trace.Index,
              y: hawk_trace.Mutable[hawk_trace.Scalar]):
    """``cross_sample_read``: the ``lookup`` plane, bound BY NAME."""
    y = table.at(where)


@hawk_trace.kernel
def e4_energy(v: hawk_trace.Vector[W], total: hawk_trace.Reduce("sum")):
    """``mapreduce``: the per-sample partial plane an ``accum_out`` role names."""
    total.contribute(hawk_math.dot(v, v))


@hawk_trace.kernel
def e4_scatter(x: hawk_trace.Scalar, lane: hawk_trace.Index,
               acc: hawk_trace.Accum[hawk_trace.Scalar]):
    """``cross_sample_write``: an ``Accum`` plane that is read-modify-written, so
    launching it twice is visibly twice, which is what a replay row needs."""
    acc.add(x, at=lane)


def _x_plane(n=N, w=W, dtype=np.float64) -> np.ndarray:
    """A ``(w, n)`` plane whose every entry differs, so a row comparing the
    wrong samples cannot pass by coincidence."""
    return np.arange(w * n, dtype=dtype).reshape(w, n) * 0.25 + 1.0


def _lane(n=N) -> np.ndarray:
    """A PERMUTATION of ``[0, n)``: every lane has exactly one writer, so the
    scatter is exact rather than a race (the fixture convention HAWK's own
    deployment set uses, and for the same reason)."""
    return ((np.arange(n) * 3 + 1) % n).astype(np.int64)


# --------------------------------------------------------------------------- #
# Bundles + subjects. The bundles are module-scoped: publishing is
# content-addressed, so a rebuild would be free, but the compile behind a MISS
# is not and there is no reason to pay it per row.
# --------------------------------------------------------------------------- #
def _bundle(kernel, tmp_path_factory, slug, targets=("cuda",)):
    return hawk_artifact.build_bundle(
        [kernel], tmp_path_factory.mktemp(f"e4_{slug}"), targets=targets,
        device_arch=DEVICE_ARCH,
    )


@pytest.fixture(scope="module")
def vec3_unit(tmp_path_factory):
    """The ``sample_local`` unit, on BOTH targets — the device rows and the
    host row bind the same emission, which is what makes them a pair."""
    return _bundle(e4_vec3_scale, tmp_path_factory, "vec3", ("cuda", "host"))


@pytest.fixture(scope="module")
def gather_unit(tmp_path_factory):
    return _bundle(e4_gather, tmp_path_factory, "gather")


@pytest.fixture(scope="module")
def energy_unit(tmp_path_factory):
    return _bundle(e4_energy, tmp_path_factory, "energy")


@pytest.fixture(scope="module")
def scatter_unit(tmp_path_factory):
    return _bundle(e4_scatter, tmp_path_factory, "scatter")


def _sidecar(unit, name) -> dict:
    return json.loads((unit.directory / f"{name}.json").read_text())


def _stamp(unit) -> dict:
    """The unit's own publish stamp: the digest it published under, and the
    digest of every file it published."""
    return json.loads((unit.directory / hawk_artifact.STAMP_NAME).read_text())


def _digest_of(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _device_subject(unit, name):
    """The plan-able view of a bundle's DEVICE entry, SELF-CHECKED.

    Everything here is a consumer act, in the order a consumer must do it:
    eagle's manifest door loads and validates the unit; the LOADED module's own
    exported ``eagle_abi_tag`` and ``eagle_layout_sizes`` are read back off the
    device and checked against this build (a tag alone cannot catch a
    layout mismatch); and only then is the sidecar projected into the
    declaration ``eagle.plan`` reads, with the loaded entry point attached.
    HAWK's ``plan_view`` does that projection and imports no eagle to do it."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle.registry import load_manifest

    reg = load_manifest(unit.manifest_path)
    loaded = reg[name]

    sizes = [int(v) for v in cp.ndarray(
        (LAYOUT_FIELDS,), dtype=cp.uint64,
        memptr=loaded.module.get_global("eagle_layout_sizes")).get()]
    eexec.check_layout_sizes(sizes)
    raw = bytes(cp.ndarray((len(eexec.ABI_TAG_V2) + 1,), dtype=cp.uint8,
                           memptr=loaded.module.get_global("eagle_abi_tag")).get())
    assert raw.split(b"\0", 1)[0].decode() == eexec.ABI_TAG_V2

    view = hawk_artifact.plan_view(_sidecar(unit, name))
    view.pop("host_entry", None)
    return SimpleNamespace(device_function=loaded.fn.kernel.ptr,
                           _keepalive=(reg, loaded), **view)


def _host_subject(unit, name):
    """The plan-able view of a bundle's HOST entry, self-checked the same way.

    ``dlsym`` rather than ``cuModuleGetGlobal``, and no cupy anywhere: this is
    the arm that binds numpy planes, and it must be reachable on a box with no
    GPU at all."""
    import eagle.exec as eexec

    so = unit.directory / f"{name}.so"
    lib = ctypes.CDLL(str(so))
    sizes = (ctypes.c_uint64 * LAYOUT_FIELDS).in_dll(lib, "eagle_layout_sizes")
    eexec.check_layout_sizes(list(sizes))
    tag = ctypes.cast(
        ctypes.addressof(ctypes.c_char.in_dll(lib, "eagle_abi_tag")),
        ctypes.c_char_p,
    ).value.decode()
    assert tag == eexec.ABI_TAG_V2

    view = hawk_artifact.plan_view(_sidecar(unit, name))
    entry = getattr(lib, view.pop("host_entry"))
    return SimpleNamespace(host_entry=ctypes.cast(entry, ctypes.c_void_p).value,
                           _keepalive=lib, **view)


# --------------------------------------------------------------------------- #
# Row 0 — the subject, named by its digest.
# --------------------------------------------------------------------------- #
def test_the_subject_is_the_published_hawk_unit_named_by_its_digest(vec3_unit):
    """The bytes this file's rows drive are the bytes HAWK published.

    Three things have to be the same thing for a verdict below to be about the
    artifact it claims: what the publisher recorded, what is on disk, and what
    the loader will read. So the row re-digests every published file and
    compares it against the unit's own stamp, and reports the unit digest in
    its message — a verdict names the artifact it was reached on, and a digest
    is the only name an emitted artifact has.

    Not marked ``gpu``: publishing needs a compiler, not a device."""
    stamp = _stamp(vec3_unit)
    published = stamp["files"]
    assert published, "the unit published no files"
    for name, digest in sorted(published.items()):
        on_disk = _digest_of(vec3_unit.directory / name)
        assert on_disk == digest, (
            f"unit {stamp['unit']}: {name} on disk digests {on_disk}, but the "
            f"unit was published with {digest} — the tree moved under the gate"
        )
    assert stamp["aether_abi"] == "aether-abi/2", (
        f"unit {stamp['unit']} is not the v2 subject this file exists for"
    )
    # The two files every device row actually loads, named explicitly. The
    # device file's extension follows driver vs toolkit (``.ptx`` or
    # ``.cubin``), so it is resolved rather than hard-coded.
    assert device_artifact(vec3_unit.directory, "e4_vec3_scale").name in published, (
        sorted(published)
    )
    assert "e4_vec3_scale.json" in published, sorted(published)


# --------------------------------------------------------------------------- #
# Rows 1-3 — bind by name, one per role shape, against Plan.run's own answer.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_a_bound_sample_local_launch_equals_plan_run_bit_for_bit(vec3_unit):
    """The identity that makes ``bind`` a second DOOR and not a second engine.

    Same artifact, same structure, same partitioning, same block — the only
    difference is who owns the planes. ``Plan.run`` uploads a host array and
    allocates its output; ``bind`` is handed both. Bit-for-bit equality is the
    claim, and it is available here because the body is a scaling by 2.0 (exact
    in binary floating point) so neither arm can be right-ish."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    x = _x_plane()
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)

    through_run = p.run(x=x, a=2.0)

    y_dev = cp.zeros((W, N), dtype=cp.float64)
    bound = p.bind(x=cp.asarray(x), y=y_dev, a=2.0)
    assert bound.n == N
    assert len(bound.partitions) == 1
    assert bound.launch() is None, "the answers are in the caller's planes"
    cp.cuda.runtime.deviceSynchronize()

    np.testing.assert_array_equal(cp.asnumpy(y_dev), through_run)
    np.testing.assert_array_equal(cp.asnumpy(y_dev), 2.0 * x)


@pytest.mark.gpu
def test_a_bound_table_read_binds_the_lookup_plane_by_name(gather_unit):
    """The ``Table`` shape: a ``lookup`` plane bound BY NAME.

    This is the shape eagle's v1 door could not even name — a HAWK lookup
    declares no compile-time ``count``, so the v1 loader (which reads the
    length out of ``buffers[]``) never learned the name existed and refused
    ``source=``/``table=`` as an unexpected argument. Here the length comes off
    the PLANE, which is the only thing that knows it."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(gather_unit, "e4_gather")
    # DELIBERATELY longer than the run: a lookup plane's length is not a sample
    # count, and a table exactly n long would let a packer that confused the two
    # pass this row by coincidence.
    table = np.arange(2 * N, dtype=np.float64) * 3.0
    where = ((np.arange(N) * 5 + 2) % (2 * N)).astype(np.int64)
    assert where.max() >= N, "the row reaches past the sample count, or proves less"
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)

    through_run = p.run(table=table, where=where)

    y_dev = cp.zeros(N, dtype=cp.float64)
    bound = p.bind(table=cp.asarray(table), where=cp.asarray(where), y=y_dev)
    assert bound.n == N, "n comes off the per-sample plane, never off the table"
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()

    np.testing.assert_array_equal(cp.asnumpy(y_dev), through_run)
    np.testing.assert_array_equal(cp.asnumpy(y_dev), table[where])


@pytest.mark.gpu
def test_a_lookup_handle_carries_the_planes_own_length_not_the_sample_count():
    """The extent a buffer role's 32-byte handle crosses with.

    A ``lookup`` table's / ``wide_in`` buffer's length is a RUNTIME quantity —
    no sidecar declares it, which is precisely why the v1 door could not bind a
    HAWK ``Table`` at all — so the bound plane is the only thing that knows it,
    and the handle says what the plane says. Every per-sample role keeps the
    run's ``n``, and both are asserted here, on the packed BYTES rather than on
    an answer: a wrong extent in a ``View`` is invisible to a body that never
    reads it, and would surface later as somebody else's bug.

    Unit-level and GPU-free on purpose: this is the ONE packing rule that changed
    for BOTH doors, so it is pinned where both doors reach it."""
    import ctypes as ct

    from eagle import plan as eplan
    from eagle.abi import DEVICE_CPU
    from eagle.host_launch import ScalarHandle

    spec = [("lookup", "table"), ("per_sample", "where"), ("mutable", "y")]
    values = {"table": np.zeros(21), "where": np.zeros(N), "y": np.zeros(N)}
    boxes, addrs = eplan._pack_args(spec, values, {}, N, DEVICE_CPU)

    table_h = ScalarHandle.from_address(addrs[0])
    where_h = ScalarHandle.from_address(addrs[1])
    assert ct.sizeof(boxes[0]) == 32
    assert table_h.samples == 21, "the table's own length, not the sample count"
    assert where_h.samples == N, "a per-sample plane still spans the run"
    assert table_h.deviceType == DEVICE_CPU and table_h.stride == 1


@pytest.mark.gpu
def test_a_bound_mapreduce_partial_plane_binds_by_name(energy_unit):
    """The ``mapreduce`` shape: the ``accum_out`` partial plane, bound by name.

    Its declaration carries no width (``arg_widths`` covers what the artifact
    actually declares, and this plane's extent is a runtime quantity), so the
    plane the caller brings is the only statement of how long it is."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(energy_unit, "e4_energy")
    assert plugin.exec_access == "mapreduce" and plugin.exec_op == "sum"
    v = _x_plane()
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)

    through_run = p.run(v=v)

    total = cp.zeros(N, dtype=cp.float64)
    p.bind(v=cp.asarray(v), total=total).launch()
    cp.cuda.runtime.deviceSynchronize()

    np.testing.assert_array_equal(cp.asnumpy(total), through_run)
    np.testing.assert_array_equal(cp.asnumpy(total), (v * v).sum(axis=0))


def test_a_bound_host_team_launch_equals_plan_run_bit_for_bit(vec3_unit):
    """The HOST arm of the identity row: numpy planes, the same emission.

    ``bind`` picks the framework off the plan's STRUCTURE, not off the values
    it is handed, so this arm is the one that proves the choice is a decision
    and not an accident of what happened to be passed. No ``gpu`` marker: this
    reaches no device at all."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _host_subject(vec3_unit, "e4_vec3_scale")
    x = _x_plane()
    p = eplan.plan(plugin, structure=eexec.HostTeam)

    through_run = p.run(x=x, a=2.0)

    y = np.zeros((W, N), dtype=np.float64)
    p.bind(x=x, y=y, a=2.0).launch()

    np.testing.assert_array_equal(y, through_run)
    np.testing.assert_array_equal(y, 2.0 * x)


# --------------------------------------------------------------------------- #
# Row 4 — the block is packed ONCE, and the launch packs nothing.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_bind_packs_the_block_once_and_launch_packs_nothing(vec3_unit,
                                                            monkeypatch):
    """"Packed ONCE" is the property the whole door rests on, so it is counted.

    A launch that re-packed would be doing per-call Python work inside what a
    consumer records as a graph node — and worse, would be free to observe a
    plane that changed since bind, which is exactly the surprise a captured
    replay cannot express."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    packs, slots = [], []
    real_pack_args, real_pack_one = eplan._pack_args, plan_pack._pack_one
    import eagle._plan_binding as binding
    import eagle._plan_pack as pack

    for mod in (eplan, binding):
        monkeypatch.setattr(mod, "_pack_args",
                            lambda *a, **k: (packs.append(1), real_pack_args(*a, **k))[1])
    for mod in (binding, pack):
        monkeypatch.setattr(mod, "_pack_one",
                            lambda *a, **k: (slots.append(a[1]), real_pack_one(*a, **k))[1])

    p = eplan.plan(plugin, structure=eexec.DeviceKernel)
    bound = p.bind(x=cp.asarray(_x_plane()), y=cp.zeros((W, N)), a=2.0)

    assert len(packs) == 1, f"the block was packed {len(packs)} times"
    assert slots == ["y", "x", "a"], "one box per arg_spec slot, in arg_spec order"

    packs.clear()
    slots.clear()
    for _ in range(3):
        bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    assert packs == [] and slots == [], (
        f"launch re-packed: {len(packs)} block(s), slots {slots}"
    )


# --------------------------------------------------------------------------- #
# Row 5 — the sample count comes off the plane's metadata, never through numpy.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_bind_reads_the_sample_count_off_a_device_plane(vec3_unit):
    """The measured defect, and its own control.

    ``Plan.run``'s sample-count probe called ``np.asarray`` on whatever it was
    handed, which cupy refuses outright — so the v2 door could not be HANDED a
    device plane at all, never mind capture one. The control matters as much as
    the fix: ``np.asarray`` must STILL raise on this very plane, or the row
    would be showing that cupy got more permissive rather than that eagle
    stopped asking it the wrong question."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    x = cp.asarray(_x_plane())

    with pytest.raises(TypeError, match="Implicit conversion to a NumPy array"):
        np.asarray(x)

    assert plan_pack._trailing_extent(x) == N
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        x=x, y=cp.zeros((W, N)), a=2.0)
    assert bound.n == N


# --------------------------------------------------------------------------- #
# Row 6 — npartitions=k issues exactly k launches, for the same answer.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_k_partitions_issue_exactly_k_launches_for_the_same_answer(vec3_unit,
                                                                   monkeypatch):
    """The partitions are the plan's, and the launch does not invent or
    merge any. Counted at the crossing into ``eagle.exec``, not inferred from
    the answer — a body that wrote the whole plane from one launch would give
    the same numbers and hide the miscount."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    x = cp.asarray(_x_plane())

    issued = []
    real_run = eexec.DeviceKernel.run

    def counting_run(function, params, partition, **kw):
        issued.append((partition.base, partition.count, partition.n_samples))
        return real_run(function, params, partition, **kw)

    monkeypatch.setattr(eexec.DeviceKernel, "run", counting_run)

    whole = cp.zeros((W, N), dtype=cp.float64)
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        x=x, y=whole, a=2.0).launch()
    assert issued == [(0, N, N)]

    issued.clear()
    split = cp.zeros((W, N), dtype=cp.float64)
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel, npartitions=2).bind(
        x=x, y=split, a=2.0)
    assert len(bound.partitions) == 2
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()

    assert issued == [(0, N // 2, N), (N // 2, N // 2, N)]
    np.testing.assert_array_equal(cp.asnumpy(split), cp.asnumpy(whole))


# --------------------------------------------------------------------------- #
# Rows 7-8 — CAPTURE. The property the door exists for.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_captured_bound_launches_replay_identically_to_the_uncaptured_twin(
        vec3_unit):
    """Three bound launches recorded into a CUDA graph, replayed, against three
    uncaptured ones — the twin claim, and the instrument for every "and nothing
    else" in the door's contract.

    WHAT THIS ROW CERTIFIES, precisely: that ``launch`` neither synchronizes
    (injecting a ``deviceSynchronize`` into it reddens this row with
    ``cudaErrorStreamCaptureUnsupported`` — measured) nor issues on the wrong
    stream (reading the stream at BIND time instead of at launch time would put
    the launches on the legacy default stream, which cannot be captured at
    all), and that what the graph replays is the launches themselves.

    WHAT IT DOES NOT CERTIFY, stated because the omission is easy to assume
    away: an ALLOCATION. Injecting ``cp.zeros(1)`` into ``launch`` left this
    row green — cupy's pool serves a small allocation from a block it already
    holds and never reaches ``cudaMalloc``, the only thing a capture region
    objects to. That half is asserted by
    ``test_launch_touches_none_of_the_calls_plan_run_needs``, which poisons the
    calls directly."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    x = cp.asarray(_x_plane())
    y = cp.zeros((W, N), dtype=cp.float64)
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(x=x, y=y, a=2.0)

    replays = 3
    for _ in range(replays):
        bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    uncaptured = cp.asnumpy(y).copy()

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        bound.launch(stream=stream)   # warm this stream's first launch OUTSIDE
        stream.synchronize()
        y.fill(0.0)
        stream.begin_capture()
        for _ in range(replays):
            bound.launch()            # stream=None -> the CURRENT stream: this one
        graph = stream.end_capture()
        graph.launch(stream=stream)
    stream.synchronize()

    np.testing.assert_array_equal(cp.asnumpy(y), uncaptured)
    np.testing.assert_array_equal(cp.asnumpy(y), 2.0 * cp.asnumpy(x))


@pytest.mark.gpu
def test_a_captured_accum_plane_accumulates_across_replays_as_uncaptured(
        scatter_unit):
    """The stronger capture row: a plane that is read-modify-written.

    ``e4_vec3_scale`` ASSIGNS, so three of its launches look exactly like one
    and a graph that quietly recorded a single node would pass the row above.
    An ``Accum`` plane cannot hide that: three recorded launches must land
    three times, and a second replay must land three more — into the SAME
    device memory the caller still holds, which is the "no copy-home" half of
    the contract stated arithmetically."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(scatter_unit, "e4_scatter")
    xs = _x_plane(w=1)[0]
    lane = _lane()
    x_dev, lane_dev = cp.asarray(xs), cp.asarray(lane)
    acc = cp.zeros(N, dtype=cp.float64)
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        x=x_dev, lane=lane_dev, acc=acc)

    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    once = cp.asnumpy(acc).copy()
    expected = np.zeros(N)
    expected[lane] = xs
    np.testing.assert_array_equal(once, expected)

    replays = 3
    acc.fill(0.0)
    for _ in range(replays):
        bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    uncaptured = cp.asnumpy(acc).copy()
    np.testing.assert_array_equal(uncaptured, replays * once)

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        bound.launch(stream=stream)
        stream.synchronize()
        acc.fill(0.0)
        stream.begin_capture()
        for _ in range(replays):
            bound.launch()
        graph = stream.end_capture()
        graph.launch(stream=stream)
        stream.synchronize()
        first = cp.asnumpy(acc).copy()
        graph.launch(stream=stream)
    stream.synchronize()

    np.testing.assert_array_equal(first, uncaptured)
    np.testing.assert_array_equal(cp.asnumpy(acc), 2 * replays * once)


@pytest.mark.gpu
def test_launch_touches_none_of_the_calls_plan_run_needs(vec3_unit, monkeypatch):
    """"And NOTHING else", certified directly rather than inferred from capture.

    The capture rows above are a real instrument for a SYNCHRONIZE (injecting
    one into ``launch`` reddens them with ``cudaErrorStreamCaptureUnsupported``
    — measured), but they are NOT one for an allocation: injecting
    ``cp.zeros(1)`` into ``launch`` left them BOTH GREEN, because cupy's memory
    pool serves a small allocation out of a block it already owns and never
    reaches ``cudaMalloc``, which is the only thing a capture region objects
    to. An instrument that cannot fail on half its claim must not be cited for
    that half, so the allocation/upload/copy-home half is asserted here
    instead, and asserted the only way that cannot drift: by POISONING the
    exact calls and running the launch through them.

    The control is the same poison in the same test: it must KILL ``Plan.run``,
    whose whole convenience is that it does every one of these things. A poison
    that killed nothing would prove nothing about what survived it."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)
    x_host = _x_plane()
    x, y = cp.asarray(x_host), cp.zeros((W, N), dtype=cp.float64)
    bound = p.bind(x=x, y=y, a=2.0)

    tripped = []

    def poison(what):
        def _poisoned(*a, **kw):
            tripped.append(what)
            raise AssertionError(f"the launch path called {what}")
        return _poisoned

    for attr in ("asarray", "ascontiguousarray", "zeros", "empty", "asnumpy"):
        monkeypatch.setattr(cp, attr, poison(f"cp.{attr}"))
    monkeypatch.setattr(cp.cuda.runtime, "deviceSynchronize",
                        poison("cudaDeviceSynchronize"))

    for _ in range(3):
        bound.launch()
    assert tripped == [], f"BoundPlan.launch reached {tripped}"

    with pytest.raises(AssertionError, match=r"the launch path called cp\."):
        p.run(x=x_host, a=2.0)
    assert tripped, "the poison killed nothing, so it certified nothing"

    monkeypatch.undo()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(cp.asnumpy(y), 2.0 * x_host)


# --------------------------------------------------------------------------- #
# Row 9 — rebind touches the named slot and nothing else.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_rebind_repacks_only_the_named_slot(vec3_unit, monkeypatch):
    """One name in, one box out — and the new pointer is the one that is read.

    Both halves are asserted, because either alone is passable by an
    implementation that is wrong in the other direction: a rebind that re-packed
    everything would still compute the right answer, and a rebind that packed
    exactly one box but kept the old address would still count as one."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    x1 = cp.asarray(_x_plane())
    x2 = cp.asarray(_x_plane() * -4.0)
    y = cp.zeros((W, N), dtype=cp.float64)
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(x=x1, y=y, a=2.0)
    before = list(bound._addrs)

    slots = []
    packs = []
    real_pack_args, real_pack_one = eplan._pack_args, plan_pack._pack_one
    import eagle._plan_binding as binding
    import eagle._plan_pack as pack

    for mod in (eplan, binding):
        monkeypatch.setattr(mod, "_pack_args",
                            lambda *a, **k: (packs.append(1), real_pack_args(*a, **k))[1])
    for mod in (binding, pack):
        monkeypatch.setattr(mod, "_pack_one",
                            lambda *a, **k: (slots.append(a[1]), real_pack_one(*a, **k))[1])

    assert bound.rebind(x=x2) is bound
    assert slots == ["x"], f"rebind packed {slots}, not just the named slot"
    assert packs == [], "rebind re-packed the whole block"

    after = list(bound._addrs)
    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    assert changed == [1], (
        f"slots {changed} moved; only the rebound 'x' (index 1) may"
    )

    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(cp.asnumpy(y), 2.0 * cp.asnumpy(x2))


@pytest.mark.gpu
def test_rebind_refuses_a_plane_of_a_different_length(vec3_unit):
    """The frozen-``n`` rule, stated as a refusal: the partitions were resolved
    against the bound sample count, so a shorter plane is a different run and
    must be bound afresh rather than smuggled into an existing block whose
    triples still describe the old one.

    RE-OWNED (RED before this edit, against the unfixed row, on
    HAWK's sidecar carrying the widened ``arg_widths``):
    ``eagle.plan._check_plane`` checks the DECLARED-width branch (``if width
    is not None``) before the generic per-sample-length one (``elif role in
    _SAMPLE_PLANE_ROLES``), and this change widened ``arg_widths`` to cover an
    INPUT ``vec_in`` plane too — 'x' is now declared 3-wide, where it used to
    carry no declared width at all. So a shorter 'x' at rebind is still
    refused, still naming 'x', but from the WIDTH branch now, not the
    sample-count one: the message reads "declared 3 component(s) wide, so at
    n=8 its plane is (3, 8); got (3, 4)" rather than the old "is a per-sample
    plane, so its last axis is the run's sample count n=8". Observed verbatim
    against the unfixed row: ``AssertionError: Regex pattern did not match``,
    the actual message being exactly that declared-width sentence. The
    row's INTENT — a plane rebound at the wrong length is refused, naming the
    slot — is unchanged; only WHICH check catches it first, and what it says,
    changed with the declaration."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        x=cp.asarray(_x_plane()), y=cp.zeros((W, N)), a=2.0)

    with pytest.raises(ValueError, match=r"'x' is declared 3 component\(s\) wide.*n=8"):
        bound.rebind(x=cp.zeros((W, N // 2)))
    with pytest.raises(ValueError, match="not a bound name"):
        bound.rebind(nope=cp.zeros((W, N)))


# --------------------------------------------------------------------------- #
# The refusals. Each names the field.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_bind_refuses_a_plane_of_the_wrong_dtype(vec3_unit):
    """A float32 plane under a float64 artifact: the body reads and writes
    8-byte elements, so binding it would walk off the end of a buffer half the
    size it assumed. Refused at bind, naming the plane and both dtypes —
    ``launch`` cannot cast, and would not be allowed to if it could."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)
    with pytest.raises(ValueError, match=r"'x' has dtype float32"):
        p.bind(x=cp.asarray(_x_plane(dtype=np.float32)),
               y=cp.zeros((W, N)), a=2.0)


@pytest.mark.gpu
def test_bind_refuses_a_plane_of_the_wrong_declared_width(vec3_unit):
    """The output plane's width IS declared (``arg_widths``), so it is checked
    exactly.

    RE-OWNED (RED before this edit): ``arg_widths`` used to be an
    OUTPUT-only projection, so the row's own rest-on assertion read
    ``{"y": 3}`` and an input's width was "not declared by a v2 sidecar,
    deliberately not guessed". This change widened ``arg_widths`` to cover every
    STATICALLY-shaped plane-bound role, inputs included — 'x' (``vec_in``) is
    now declared 3-wide too, same as 'y' — so the rest-on assertion is now
    ``{"y": 3, "x": 3}``; observed verbatim against the unfixed row:
    ``AssertionError: assert {'y': 3, 'x': 3} == {'y': 3}``. This row's own
    SUBJECT is unaffected — it still binds 'x' at its correct width and 'y'
    at the wrong one, and the refusal still fires on 'y', at bind, naming it,
    exactly as before. A RUNTIME-length role's width is still never declared
    (this amendment, ``hawk/artifact/sidecar.py``), and still not
    guessed here — only the STATIC roles moved from undeclared to declared."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    assert plugin.arg_widths == {"y": 3, "x": 3}, "the declaration this row rests on"
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)
    with pytest.raises(ValueError, match=r"'y' is declared 3 component"):
        p.bind(x=cp.asarray(_x_plane()), y=cp.zeros((2, N)), a=2.0)


@pytest.mark.gpu
def test_bind_refuses_a_non_contiguous_plane(vec3_unit):
    """A strided view. The mirror can carry strides; a 32-byte scalar handle
    carries a UNIT stride by construction, and ``launch`` may not repack — so a
    strided plane is a plane some role would decode at the wrong addresses.
    Refused for every role rather than for the ones that would notice, because
    "which roles notice" is an implementation detail of the packer and this is
    a contract."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    strided = cp.zeros((W, 2 * N), dtype=cp.float64)[:, ::2]
    assert not strided.flags.c_contiguous and strided.shape == (W, N)
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)
    with pytest.raises(ValueError, match=r"'x' is not C-contiguous"):
        p.bind(x=strided, y=cp.zeros((W, N)), a=2.0)


@pytest.mark.gpu
def test_bind_refuses_a_host_array_on_a_device_plan(vec3_unit):
    """The framework is the STRUCTURE's, not the caller's. An upload here would
    be an allocation and a copy — the two things this door exists not to do —
    so a host array is refused rather than quietly promoted."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    p = eplan.plan(plugin, structure=eexec.DeviceKernel)
    with pytest.raises(ValueError, match=r"'x' is a HOST array"):
        p.bind(x=_x_plane(), y=cp.zeros((W, N)), a=2.0)


@pytest.mark.gpu
def test_bind_refuses_a_device_array_on_a_host_plan(vec3_unit):
    """The mirror image, and the reason the check is a decision rather than a
    fallback: a host entry dereferences the pointer it is handed, so a device
    pointer bound here is a segfault the moment the team runs."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _host_subject(vec3_unit, "e4_vec3_scale")
    p = eplan.plan(plugin, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=r"'x' is a DEVICE array"):
        p.bind(x=cp.asarray(_x_plane()), y=np.zeros((W, N)), a=2.0)


def test_bind_refuses_an_unknown_name(vec3_unit):
    """A stray keyword is a typo far more often than a courtesy, and ignoring
    one launches the body over whatever the correctly-spelled plane last held.
    GPU-free: the refusal happens before anything is packed or launched."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _host_subject(vec3_unit, "e4_vec3_scale")
    p = eplan.plan(plugin, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=r"'why' is not a name"):
        p.bind(x=_x_plane(), y=np.zeros((W, N)), a=2.0, why=1.0)


def test_bind_refuses_a_missing_name(vec3_unit):
    """Nothing is defaulted here — not even an output plane, which
    ``Plan.run`` would have allocated. A plane this door allocated would have
    to outlive the graph that writes it while the caller held no handle on it,
    so the caller brings every one."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _host_subject(vec3_unit, "e4_vec3_scale")
    p = eplan.plan(plugin, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=r"'mutable' named 'y'"):
        p.bind(x=_x_plane(), a=2.0)
    with pytest.raises(ValueError, match=r"'uniform' named 'a'"):
        p.bind(x=_x_plane(), y=np.zeros((W, N)))


def test_bind_refuses_a_grid_and_a_host_stream(vec3_unit):
    """Two refusals a consumer's launch descriptor depends on.

    The grid is EAGLE's, derived from each partition's own count, and a
    recording consumer passes ``grid=None, block=None`` precisely so that the
    captured launch and its uncaptured twin resolve the SAME configuration —
    accepting a caller's grid would make the twin claim about two different
    launches. And a host_team run has no stream: taking one and ignoring it
    would let a caller believe an ordering that does not exist."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _host_subject(vec3_unit, "e4_vec3_scale")
    bound = eplan.plan(plugin, structure=eexec.HostTeam).bind(
        x=_x_plane(), y=np.zeros((W, N)), a=2.0)

    with pytest.raises(ValueError, match="takes no grid"):
        bound.launch(grid=(4,))
    with pytest.raises(ValueError, match="host_team run has no stream"):
        bound.launch(stream=7)


# --------------------------------------------------------------------------- #
# Row 16 — the v1 door refuses a v2 entry, LOUDLY, in a child interpreter.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_the_v1_door_refuses_a_v2_artifact_instead_of_segfaulting(vec3_unit):
    """The refusal that replaces a SIGSEGV — asserted IN A SUBPROCESS.

    It runs in a child not because the refusal needs one, but because the
    DEFECT did: before this fix, this exact call killed the interpreter (measured on
    this box: stdout ``LOADED``, returncode ``-11``), and a row that ran it
    in-process would have taken the suite down with it. Keeping the child is
    what lets the row keep asserting the severity it was written about — the
    difference between "raises" and "the process is gone" is invisible to
    ``pytest.raises``.

    The child prints ``RAISED <type>`` for an exception and ``LAUNCHED-OK`` if
    the v1 packer ever drives a v2 entry to completion, which would mean the
    door stopped refusing."""
    ptx = device_artifact(vec3_unit.directory, "e4_vec3_scale")
    script = f"""
import numpy as np, cupy as cp
from eagle.loaded import LoadedPure
lp = LoadedPure({str(ptx)!r})
print("LOADED", flush=True)  # flushed: a SIGSEGV takes the buffer with it
n = {N}
x = cp.asarray(np.arange({W} * n, dtype=np.float64).reshape({W}, n))
y = cp.zeros(({W}, n))
try:
    lp.launch(grid=None, block=None, x=x, y=y, a=2.0,
              terminated=cp.zeros(n, dtype=bool))
    cp.cuda.runtime.deviceSynchronize()
    print("LAUNCHED-OK", flush=True)
except Exception as exc:
    print("RAISED", type(exc).__name__, flush=True)
    print("MESSAGE", str(exc).replace(chr(10), " "), flush=True)
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=600)
    assert "LOADED" in proc.stdout, (
        "the child never loaded the artifact, so nothing below is about the "
        f"launch: rc={proc.returncode}, stderr={proc.stderr.strip()[-400:]!r}"
    )
    assert proc.returncode == 0, (
        "the v1 door did not survive a v2 artifact — a refusal must be an "
        f"exception, not a signal: rc={proc.returncode}, "
        f"stdout={proc.stdout.strip()!r}"
    )
    assert "LAUNCHED-OK" not in proc.stdout, (
        "the v1 packer drove an aether-abi/2 entry to completion; the parameter "
        "block it builds carries no partition triple, so whatever ran was not "
        f"this artifact's contract (stdout: {proc.stdout.strip()!r})"
    )
    assert "RAISED ValueError" in proc.stdout, (
        f"expected a ValueError refusal; got {proc.stdout.strip()!r}"
    )
    message = proc.stdout.split("MESSAGE", 1)[1]
    assert "aether-abi/2" in message, (
        f"the refusal must name the ABI tag it refused: {message!r}"
    )
    assert "bind" in message, (
        f"the refusal must point at the door that does serve it: {message!r}"
    )


@pytest.mark.gpu
def test_the_v1_door_names_the_tag_and_the_v2_door_in_process(vec3_unit):
    """The same refusal, reachable without a subprocess, so the MESSAGE can be
    read directly rather than through a child's stdout — and so the guard is
    covered on the ``__call__`` door too, which is the other way a consumer
    reaches the v1 packer (and would crash identically)."""
    import eagle.exec as eexec
    from eagle.loaded import LoadedPure

    loaded = LoadedPure(device_artifact(vec3_unit.directory, "e4_vec3_scale"))
    assert loaded.abi_tag == eexec.ABI_TAG_V2

    for door in (lambda: loaded.launch(grid=None, block=None),
                 lambda: loaded(x=_x_plane())):
        with pytest.raises(ValueError) as excinfo:
            door()
        message = str(excinfo.value)
        assert eexec.ABI_TAG_V2 in message
        assert eexec.ABI_TAG_V1 in message
        assert "eagle.plan.plan" in message and "bind" in message


# --------------------------------------------------------------------------- #
# The launch policy on the v2 door.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_a_bound_launch_consults_the_launch_policy_with_the_entrys_own_properties(
        vec3_unit, monkeypatch):
    """RED before this consultation was wired in: ``BoundPlan.launch(block=None)``
    issued at a door-local default and never asked ``eagle.launch.resolve_block`` — a
    sibling group replayed at 256 whatever the policy said, and a downstream
    block-invariance twin found "the resolver was never called". The resolver
    is consulted PER CALL through ``eagle.launch`` (a consumer's monkeypatch
    sees it), with the ambient sibling count and the entry's DRIVER-QUERIED
    properties (a raw ``CUfunction`` handle has no ``.attributes``; an empty
    dict would have "consulted" the policy into the fallback every time), and
    its answer is what crosses into ``eagle.exec``. An explicit ``block``
    bypasses the policy untouched."""
    import importlib

    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle._launch_policy import sibling_context

    # by module name: the package exports a `launch` FUNCTION that shadows it
    elaunch = importlib.import_module("eagle.launch")

    plugin = _device_subject(vec3_unit, "e4_vec3_scale")
    x = cp.asarray(_x_plane())
    y = cp.zeros((W, N), dtype=cp.float64)
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(x=x, y=y, a=2.0)

    seen = []
    real_resolve = elaunch.resolve_block

    def recording_resolve(n, siblings, dev, kern):
        seen.append((n, siblings, kern))
        return real_resolve(n, siblings, dev, kern)

    issued = []
    real_run = eexec.DeviceKernel.run

    def counting_run(function, params, partition, **kw):
        issued.append(kw.get("block"))
        return real_run(function, params, partition, **kw)

    monkeypatch.setattr(elaunch, "resolve_block", recording_resolve)
    monkeypatch.setattr(eexec.DeviceKernel, "run", counting_run)

    with sibling_context(4):
        bound.launch()
    cp.cuda.runtime.deviceSynchronize()

    assert len(seen) == 1, seen
    n, siblings, kern = seen[0]
    assert (n, siblings) == (N, 4)
    assert kern.get("num_regs", 0) > 0 and kern.get("max_threads_per_block", 0) > 0, (
        f"the entry's own properties must reach the policy; got {kern!r}")
    assert issued == [real_resolve(N, 4, elaunch._device_props(), kern)]

    seen.clear()
    issued.clear()
    bound.launch(block=64)
    cp.cuda.runtime.deviceSynchronize()
    assert seen == [] and issued == [64]
    np.testing.assert_array_equal(cp.asnumpy(y), 2.0 * _x_plane())


@pytest.mark.parametrize("count, refused", [(2**32 - 2 * 256, False), (2**32 - 256 + 1, True),
                                            (2**32, True)])
def test_launch_persist_refuses_a_count_that_would_wrap_its_counter(monkeypatch, count, refused):
    """The persist entry's fetch counter is a uint32 the lanes keep fetching
    from until they pass ``count``: ``count`` plus one overshooting fetch per
    lane must stay below 2**32, or the counter wraps and the launch never
    ends. Refused before anything is touched on the device."""
    from eagle._plan_binding import BoundPlan

    bound = object.__new__(BoundPlan)
    bound._parts = (SimpleNamespace(base=0, count=count, n_samples=count),)
    monkeypatch.setattr(BoundPlan, "persist_entry", lambda self: object())
    if refused:
        with pytest.raises(ValueError, match="fewer than 2\\*\\*32"):
            bound.launch_persist(grid=1, block=256, counter=None)
    else:
        # past the guard the stub has no device state to launch with
        with pytest.raises(AttributeError):
            bound.launch_persist(grid=1, block=256, counter=None)


def _stub_bound(monkeypatch, parts):
    """A BoundPlan with no device state whose driver launch records the
    packed words (each argument read through its address) and the geometry."""
    import ctypes

    from eagle._plan_binding import BoundPlan

    issued = []

    def launch_kernel(ptr, gx, gy, gz, bx, by, bz, shm, stream, params, extra):
        arr = ctypes.cast(params, ctypes.POINTER(ctypes.c_void_p))
        words = [ctypes.c_longlong.from_address(arr[i]).value
                 for i in range(len(bound._addrs) + tail[0])]
        issued.append(((gx, bx, stream), words))

    bound = object.__new__(BoundPlan)
    tail = [7]  # trailing words read: persist 7, range 5
    boxes = [ctypes.c_longlong(11), ctypes.c_longlong(22)]
    bound._boxes = boxes
    bound._addrs = [ctypes.addressof(b) for b in boxes]
    bound._parts = parts
    bound._gen = 0
    bound._entry = object()
    bound._cupy = SimpleNamespace(cuda=SimpleNamespace(
        driver=SimpleNamespace(launchKernel=launch_kernel),
        get_current_stream=lambda: SimpleNamespace(ptr=5)))
    fn = SimpleNamespace(kernel=SimpleNamespace(ptr=7))
    monkeypatch.setattr(BoundPlan, "persist_entry", lambda self: fn)
    monkeypatch.setattr(BoundPlan, "range_entry", lambda self: fn)
    return bound, issued, tail


def test_prepared_launch_arguments_equal_a_fresh_marshal(monkeypatch):
    """prepare_persist/prepare_range pack the argument block once: every call
    of the prepared launch issues exactly the words (and geometry) a fresh
    launch_persist/launch_range packs, and a rebind is seen by the next call."""
    import ctypes

    part = SimpleNamespace(base=3, count=1000, n_samples=1000)
    bound, issued, tail_cell = _stub_bound(monkeypatch, (part,))
    cell = lambda v: SimpleNamespace(data=SimpleNamespace(ptr=v))  # noqa: E731
    counter, util, steps, stepsum = cell(0x100), cell(0x200), cell(0x300), cell(0x400)

    for with_util in (False, True):
        u = util if with_util else None
        kw = dict(grid=4, counter=counter, util=u, block=128, steps=steps,
                  stepsum=stepsum)
        prepared = bound.prepare_persist(**kw)
        del issued[:]
        bound.launch_persist(**kw)
        prepared()
        prepared()
        fresh, first, second = issued[:3]
        assert first == second and fresh[0] == first[0]
        assert first[1][:2] == [11, 22]
        assert first[1][2:5] == [3, 1000, 1000]
        assert first[1][5:9] == [0x100, 0x200 if with_util else 0, 0x300, 0x400]
        del issued[:]
        bound.launch_persist(grid=1, block=1, counter=counter, prepared=prepared)
        assert issued[0][0] == first[0] and issued[0][1] == first[1]

    prepared = bound.prepare_range(block=64, steps=steps, stepsum=stepsum)
    tail_cell[0] = 5
    del issued[:]
    bound.launch_range(block=64, steps=steps, stepsum=stepsum)
    prepared()
    assert issued[0] == issued[1]
    assert issued[0][1][2:7] == [3, 1000, 1000, 0x300, 0x400]
    assert issued[0][0] == (16, 64, 5)

    # a rebind re-packs: the prepared launch follows it
    box = ctypes.c_longlong(99)
    bound._boxes[0] = box
    bound._addrs[0] = ctypes.addressof(box)
    bound._gen += 1
    del issued[:]
    prepared()
    assert issued[0][1][0] == 99
