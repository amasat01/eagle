# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ARTIFACT-DRIVEN Python rank bed for :data:`eagle.exec.RankPartition`
(mechanism only — no CUDA-aware MPI, no performance claims).

The Python twin of ``tests/mpi/test_MpiBed.cpp``, and deliberately its mirror
image. Read that file first: the reasoning behind every choice repeated here —
why every row runs on BOTH ranks, why the oracle is local rather than remote, why
an identity row must also assert its PRE-GATHER state — is written out there and
is not re-argued line by line below.

EVERY ROW HERE RUNS ON EVERY RANK. The collectives inside ``RankPartition`` are
blocking, so a row that returned early on one rank would hang the others; there is
deliberately no ``if rank == 0`` anywhere in this file, and no ``pytest.skip``
either (a rank that skips a row its peers enter is the same deadlock wearing a
green hat). The gate (``tests/mpi/check_rank_bed.sh``) independently compares the
ranks' JUnit verdicts, so a row that quietly asserted nothing on rank 1 would
still be caught.

WHAT MAKES THESE ROWS NON-VACUOUS. A distributed identity row passes trivially if
every rank simply ran the WHOLE view and the gather then overwrote everything with
the same numbers. So the identity rows run a second, PRE-GATHER arm
(``eagle.plan.plan(..., _gather=False)``) and assert that outside this rank's own
sub-partition the output plane still holds its sentinel. The mapreduce row asserts
that dropping one rank's partial FAILS the band, so the band is not so wide that
it certifies anything.

THE ARTIFACT IS AN INPUT, NOT A FIXTURE. ``$EAGLE_RANK_BED_ARTIFACT`` names a
deployed aether-abi/2 artifact root (``tests/mpi/check_rank_bed.sh`` builds the
default one before it launches any rank — a gate builds its own inputs), and a caller
can point this same bed at a real deployed emission. So NO row here names a kernel: each
asks :func:`v2_artifact.unit_for` for "the unit DECLARING this access class" and
drives whatever it gets, with the ``arg_spec``, entry symbols, scalar type and
reduction op all read off the manifest and its sidecars.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import v2_artifact

import eagle.exec as eexec
from eagle import plan as eplan

#: A value no body in any artifact writes, so "still the sentinel" is unambiguous
#: evidence that nothing touched that sample on this rank.
SENTINEL = -12345.0


# --------------------------------------------------------------------------- #
# The artifact + the world.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def units():
    """Every unit of the artifact under ``$EAGLE_RANK_BED_ARTIFACT``.

    Raises — never skips — when the variable is unset: this bed exists to certify
    a real artifact, and a run that quietly certified nothing is the exact defect
    the whole gate is built around."""
    root = os.environ.get("EAGLE_RANK_BED_ARTIFACT")
    if not root:
        raise RuntimeError(
            "$EAGLE_RANK_BED_ARTIFACT is unset: the rank bed drives a deployed "
            "aether-abi/2 artifact root. Run it through tests/mpi/check_rank_bed.sh, "
            "which builds the default artifact and exports the variable."
        )
    return v2_artifact.load(pathlib.Path(root))


def _world():
    """``(rank, size)`` — read through the structure under test, which is also the
    only place MPI is initialised."""
    return eexec.RankPartition.world()


def _expected_slice(n: int):
    """THIS RANK's expected ``(base, count)``, stated from the CONTRACT rather
    than read back from :meth:`eagle.exec.RankPartition.local`.

    Asking the function under test what to expect is how the first version of the
    C++ identity row survived an injected defect that gave EVERY rank the whole
    view: the row's own idea of "mine" moved with the defect, and the sentinel
    check then compared the wrong thing to itself. ``q``/``rem`` here are the rule
    restated — contiguous, the first ``rem`` ranks take one extra — not a call."""
    rank, size = _world()
    q, rem = divmod(n, size)
    count = q + (1 if rank < rem else 0)
    base = rank * q + (rank if rank < rem else rem)
    return base, count


def _all_ranks(values) -> np.ndarray:
    """Every rank's copy of ``values``, as a ``(size, len(values))`` plane whose
    row ``r`` is rank ``r``'s.

    Built out of the gather already under test rather than a new binding: over a
    world of ``size``, ``Partition.whole(size)`` gives each rank exactly one
    sample, so an ``allgather_plane`` with ``elems_per_sample = len(values)``
    fills row ``r`` from rank ``r``. "Identical on every rank" then means what it
    should mean operationally — a value every rank agrees on TO THE LAST BIT, not
    one every rank finds plausible."""
    _, size = _world()
    flat = np.ascontiguousarray(np.atleast_1d(np.asarray(values, dtype=np.float64)))
    rank, _ = _world()
    plane = np.zeros((size, flat.size), dtype=np.float64)
    plane[rank, :] = flat
    eexec.RankPartition.allgather_plane(
        plane.ctypes.data, eexec.Partition.whole(size), int(flat.size)
    )
    return plane


def _assert_identical_on_all_ranks(values, what: str) -> None:
    plane = _all_ranks(values)
    for r in range(1, plane.shape[0]):
        assert plane[r].tobytes() == plane[0].tobytes(), (
            f"{what}: rank {r}'s copy differs BITWISE from rank 0's "
            f"({plane[r]} vs {plane[0]})"
        )


# --------------------------------------------------------------------------- #
# The RULED band (fork F-d): S x 2 x eps(float64), S = the number of elements
# combined, applied RELATIVELY. Derived from ONE anchor, never fitted to an
# observed difference, and identical to the C++ bed's — the whole point of a band
# is that it is one rule, not one per test file.
# --------------------------------------------------------------------------- #
def _band(S: int, a: float, b: float) -> float:
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return S * 2.0 * np.finfo(np.float64).eps * scale


def _within_band(a: float, b: float, S: int) -> bool:
    return abs(float(a) - float(b)) <= _band(S, a, b)


# --------------------------------------------------------------------------- #
# The oracle: the WHOLE run, computed locally from the replicated inputs.
# --------------------------------------------------------------------------- #
def _serial_oracle(plugin, **kw):
    """The whole run through :meth:`eagle.exec.HostTeam.run_serial` — one call,
    one triple: the correctness oracle.

    Reached through ``eagle.plan``'s own host packer with the launch swapped,
    because the ORACLE must be the SERIAL arm and the plan surface deliberately
    has no serial structure to name (F-c rules the serial path a reference, not a
    structure). Packing it a second time here would risk the oracle and the
    subject disagreeing about the argument block rather than about the answer."""
    arg_spec = plugin.arg_spec
    n = eplan._n_from_kw(arg_spec, kw)
    return eplan._run_host_plan(
        plugin,
        arg_spec,
        kw,
        n,
        (eexec.Partition.whole(n),),
        launch=lambda entry, addrs, part: eexec.HostTeam.run_serial(
            entry, addrs, part
        ),
    )


def _x(n: int) -> np.ndarray:
    """A sample-dependent input with no repeated values, so a row that compared
    the wrong samples could not pass by coincidence."""
    return 1.0 / (np.arange(n, dtype=np.float64) + 1.0)


def _uniforms(plugin) -> dict:
    """A value for every ``uniform`` the artifact declares, keyed by its declared
    NAME — the bed binds what the arg_spec asks for and knows nothing else about
    it."""
    return {
        name: 3.25 - 0.75 * i
        for i, (role, name) in enumerate(plugin.arg_spec)
        if role == "uniform"
    }


def _sample_input_name(plugin) -> str:
    """The one ``per_sample`` input this artifact's body reads, by its DECLARED
    name."""
    names = [name for role, name in plugin.arg_spec if role == "per_sample"]
    assert len(names) == 1, (
        f"the rank bed's rows bind exactly one per_sample input; this unit "
        f"declares {names}"
    )
    return names[0]


def _output_name(plugin) -> str:
    """The one output plane this artifact's body writes, by its DECLARED name."""
    names = [
        name for role, name in plugin.arg_spec
        if role in ("out", "mutable", "wide_out", "accum_out")
    ]
    assert len(names) == 1, (
        f"the rank bed's rows read exactly one output plane; this unit declares "
        f"{names}"
    )
    return names[0]


# --------------------------------------------------------------------------- #
# The bed is a MULTI-rank bed, and the split is the contract's
# --------------------------------------------------------------------------- #
def test_world_is_a_multi_rank_world():
    """The bed certifies a DISTRIBUTED structure, so a world of one would let
    every row below pass having exercised nothing distributed at all."""
    rank, size = _world()
    expected = int(os.environ.get("EAGLE_MPI_BED_RANKS", "2"))
    assert size == expected, (
        f"the rank bed runs under `mpirun -np {expected}`; this world has "
        f"{size} rank(s)"
    )
    assert size >= 2, "a one-rank world exercises nothing this bed exists to check"
    assert 0 <= rank < size


def test_local_split_is_contiguous_and_covers_every_sample():
    """The rank cut is contiguous, tolerates an uneven split, keeps the TRUE
    ``nSamples``, and — gathered across the world — covers every sample exactly
    once. Checked at a sample count the world size does NOT divide, because an
    even split is the case that hides an off-by-one."""
    _, size = _world()
    n = 1001
    whole = eexec.Partition.whole(n)
    mine = eexec.RankPartition.local(whole)

    exp_base, exp_count = _expected_slice(n)
    assert (mine.base, mine.count) == (exp_base, exp_count)
    assert mine.n_samples == n, "nSamples must stay the TRUE total"

    # Every rank's share, reassembled from the world: contiguous, no gap, no
    # overlap, and never unbalanced by more than one sample.
    bases = _all_ranks([float(mine.base)])[:, 0]
    counts = _all_ranks([float(mine.count)])[:, 0]
    assert counts.sum() == n, "the ranks' shares do not cover [0, n)"
    cursor = 0
    for r in range(size):
        assert bases[r] == cursor, f"rank {r}'s share is not contiguous"
        cursor += counts[r]
    assert cursor == n
    assert counts.max() - counts.min() <= 1


def test_artifact_declares_every_access_class_the_bed_certifies(units):
    """The bed's INPUT must be complete. An artifact missing an access class
    cannot be certified for it, and this row says so by NAME rather than letting
    the rows for the missing classes disappear — which is what a skip would
    do."""
    declared = {u.access for u in units}
    missing = [a for a in v2_artifact.REQUIRED_ACCESS_CLASSES if a not in declared]
    assert not missing, (
        f"the artifact at $EAGLE_RANK_BED_ARTIFACT declares {sorted(declared)} and "
        f"is missing {missing}; the rank bed certifies every class in "
        f"{list(v2_artifact.REQUIRED_ACCESS_CLASSES)}"
    )
    for unit in units:
        assert "host" in unit.targets or "device" in unit.targets
        assert unit.plugin.abi_tag == eexec.ABI_TAG_V2


# --------------------------------------------------------------------------- #
# (a) sample_local under a HOST inner is BIT-identical whole vs across ranks
# --------------------------------------------------------------------------- #
def test_sample_local_host_is_bit_exact_versus_the_whole_run(units):
    """A ``sample_local`` body run WHOLE must be BIT-identical to the same
    body run one contiguous sub-partition per rank and gathered — no rank ever
    recomputes another rank's samples, so there is nothing to round differently.

    The PRE-GATHER arm is what makes that non-vacuous: before the gather this
    rank's plane must hold its own answers INSIDE its sub-partition and the
    untouched sentinel everywhere else."""
    plugin = v2_artifact.unit_for(units, "sample_local").plugin
    n = 1000
    xname, yname = _sample_input_name(plugin), _output_name(plugin)
    kw = {xname: _x(n), **_uniforms(plugin)}
    base, count = _expected_slice(n)

    # PRE-GATHER: this rank touched its OWN sub-partition and nothing else.
    pre = eplan.plan(
        plugin, structure=eexec.RankPartition, inner=eexec.HostTeam, _gather=False
    ).run(**kw, **{yname: np.full(n, SENTINEL)})
    mine = np.zeros(n, dtype=bool)
    mine[base : base + count] = True
    assert not np.any(pre[mine] == SENTINEL), (
        "this rank left samples of its OWN sub-partition unwritten"
    )
    assert np.all(pre[~mine] == SENTINEL), (
        "this rank wrote samples belonging to another rank — the ranks are not "
        "running disjoint sub-partitions, and the gather would hide it"
    )

    got = eplan.plan(
        plugin, structure=eexec.RankPartition, inner=eexec.HostTeam
    ).run(**kw, **{yname: np.full(n, SENTINEL)})
    oracle = _serial_oracle(plugin, **kw)

    assert got.tobytes() == oracle.tobytes(), (
        "the rank-partitioned run differs from the whole run"
    )
    _assert_identical_on_all_ranks(got, "the gathered sample_local plane")
    assert got[0] != got[1], "the body did nothing sample-dependent"


# --------------------------------------------------------------------------- #
# (b) sample_local under a DEVICE inner
# --------------------------------------------------------------------------- #
def test_sample_local_device_matches_the_whole_device_run_and_the_host_band(units):
    """The device arm — deliberately UNMARKED. The C++ bed's device rows are
    compiled into every CUDA build and always run; the ``gpu`` marker the main
    suite uses would turn this row into a SKIP on a GPU-less box, and a skipped
    row inside a pinned manifest is green having certified nothing. A rank bed
    without a GPU is a bed that cannot do its job, and it should say so loudly.

    Each rank launches its own ``cuLaunchKernel`` over its own
    sub-partition (both ranks on GPU 0, each in its own context — a single-node
    bed, and the mechanism is the same one on two GPUs as on one), and the gather
    is host-staged because CUDA-aware MPI was ruled out.

    Two comparisons, because they fail for different reasons: BIT-identical
    against the whole DEVICE run (sample_local is elementwise, so
    partitioning cannot move a bit), and within the RULED band against the SERIAL
    HOST oracle (a host/device difference is a different claim
    entirely)."""
    plugin = v2_artifact.unit_for(units, "sample_local").plugin
    n = 1000
    xname, yname = _sample_input_name(plugin), _output_name(plugin)
    kw = {xname: _x(n), **_uniforms(plugin)}
    base, count = _expected_slice(n)

    pre = eplan.plan(
        plugin, structure=eexec.RankPartition, inner=eexec.DeviceKernel, _gather=False
    ).run(**kw, **{yname: np.full(n, SENTINEL)})
    mine = np.zeros(n, dtype=bool)
    mine[base : base + count] = True
    assert not np.any(pre[mine] == SENTINEL)
    assert np.all(pre[~mine] == SENTINEL), (
        "this rank's kernel wrote samples belonging to another rank"
    )

    got = eplan.plan(
        plugin, structure=eexec.RankPartition, inner=eexec.DeviceKernel
    ).run(**kw, **{yname: np.full(n, SENTINEL)})
    whole_device = eplan.plan(plugin, structure=eexec.DeviceKernel).run(**kw)
    host_oracle = _serial_oracle(plugin, **kw)

    assert got.tobytes() == whole_device.tobytes(), (
        "the rank-partitioned device run differs from the whole device run"
    )
    _assert_identical_on_all_ranks(got, "the gathered device plane")
    worst = float(np.max(np.abs(got - host_oracle)))
    assert worst <= _band(n, float(np.max(np.abs(got))), 0.0), (
        f"host/device difference {worst} exceeds the ruled band S*2*eps*scale"
    )


# --------------------------------------------------------------------------- #
# (c) mapreduce: partials folded in RANK ORDER, within band
# --------------------------------------------------------------------------- #
def test_mapreduce_partials_fold_in_rank_order_within_the_band(units):
    """The body stays ELEMENTWISE and eagle owns the COMBINE. Each rank folds
    its own partials, the R partials cross as one ``MPI_Allgather``, and every
    rank folds THEM in a fixed RANK ORDER — never ``MPI_Allreduce``, whose combine
    order is the implementation's business.

    Band-gated rather than bit-exact BECAUSE the combine order differs between a
    whole run and a partitioned one; and the band is proven NOT vacuous
    by requiring that a result missing an entire rank's partial fails it."""
    unit = v2_artifact.unit_for(units, "mapreduce")
    plugin, op = unit.plugin, unit.op
    assert op, "a mapreduce unit must declare exec_op"
    n = 4096
    xname = _sample_input_name(plugin)
    kw = {xname: _x(n), **_uniforms(plugin)}

    partials = eplan.plan(
        plugin, structure=eexec.RankPartition, inner=eexec.HostTeam
    ).run(**kw)
    whole_value = eexec.fold(op, _serial_oracle(plugin, **kw))

    base, count = _expected_slice(n)
    mine = eexec.fold(op, partials[base : base + count])
    split = eexec.RankPartition.allgather_fold(op, mine)

    assert _within_band(whole_value, split, n), (
        f"|{whole_value} - {split}| exceeds the ruled band "
        f"{_band(n, whole_value, split)}"
    )
    _assert_identical_on_all_ranks(
        [split], "the rank-ordered fold (the combine order is not fixed)"
    )
    assert not _within_band(whole_value, mine, n), (
        "the band accepts a result missing an entire rank's partial — it is so "
        "wide that it certifies nothing"
    )


# --------------------------------------------------------------------------- #
# (d) the placement table for the rank structure
# --------------------------------------------------------------------------- #
def test_cross_sample_write_is_refused_at_plan_time_naming_the_fork(units):
    """Two ranks accumulating into one shared target would need a
    partial-accum combine that is not specified, so the placement is ILLEGAL —
    refused at PLAN time, before anything is packed or launched, and NAMING the
    rule. Refused at EVERY world size including one: a placement that is legal at
    ``-np 1`` and illegal at ``-np 2`` is a trap."""
    plugin = v2_artifact.unit_for(units, "cross_sample_write").plugin
    with pytest.raises(ValueError) as excinfo:
        eplan.plan(plugin, structure=eexec.RankPartition, inner=eexec.HostTeam)
    message = str(excinfo.value)
    for fragment in ("cross_sample_write", "rank_partition", "F-e"):
        assert fragment in message, f"the refusal does not name {fragment!r}: {message}"


def test_cross_sample_read_plans_and_is_bit_exact_because_inputs_are_replicated(units):
    """``cross_sample_read`` is LEGAL across ranks BECAUSE inputs are REPLICATED, so the
    lookup / Staged tables a body reads are whole on every rank — the placement table's
    stated condition, met by construction. The evidence is the answer itself:
    bit-identical to the whole run, which it could not be if any rank held a sliced
    table."""
    plugin = v2_artifact.unit_for(units, "cross_sample_read").plugin
    # A prime sample count, so the split is uneven and reads cross the rank
    # boundary in both directions rather than staying inside one rank's slice.
    n = 997
    tname, yname = _sample_input_name(plugin), _output_name(plugin)
    table = _x(n) + 0.125 * np.arange(n, dtype=np.float64)
    kw = {tname: table, **_uniforms(plugin)}

    rank_plan = eplan.plan(
        plugin, structure=eexec.RankPartition, inner=eexec.HostTeam
    )
    got = rank_plan.run(**kw, **{yname: np.full(n, SENTINEL)})
    oracle = _serial_oracle(plugin, **kw)

    assert got.tobytes() == oracle.tobytes(), (
        "a cross-sample read differs from the whole run — the table was not whole "
        "on this rank"
    )
    _assert_identical_on_all_ranks(got, "the gathered cross_sample_read plane")


# --------------------------------------------------------------------------- #
# (e) the wire spelling resolves to the real structure
# --------------------------------------------------------------------------- #
def test_structure_from_name_resolves_the_rank_structure():
    """``rank_partition`` is an IMPLEMENTED structure now, not a named
    placeholder: the wire spelling every manifest carries must resolve to the
    singleton the plan layer dispatches on, and that singleton must actually
    run."""
    assert eexec._structure_from_name("rank_partition") is eexec.RankPartition
    assert eexec.RankPartition.name == "rank_partition"
    assert eexec.RankPartition.available, (
        "eagle._mpi is not importable inside the bed — the gate ran the bed "
        "against a build that has no rank structure to certify"
    )
    # The placement table for this structure, through the same check_placement the plan
    # layer uses, at the LIVE world size rather than a literal.
    _, size = _world()
    for access in ("sample_local", "cross_sample_read", "mapreduce"):
        eexec.check_placement(access, eexec.RankPartition, size)
    with pytest.raises(ValueError, match="F-e"):
        eexec.check_placement("cross_sample_write", eexec.RankPartition, size)
