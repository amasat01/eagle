# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The single-process face of :data:`eagle.exec.RankPartition`
(MECHANISM only).

The DISTRIBUTED certification of the rank structure is the two-rank bed
(``tests/mpi/test_rank_bed.py``, driven by ``tests/mpi/check_rank_bed.sh``); this
file is what an ordinary ``pytest`` covers, and it deliberately covers the things
a world of one can still certify honestly:

* the structure is REAL — the wire spelling resolves to it, and the placement
  table it now obeys is the table, not "refused for everything" (which is what
  every one of these rows pinned before this was implemented);
* the SINGLETON WORLD works: rank 0 of 1 gets the whole partition, and the run
  through ``RankPartition`` is bit-identical to the ordinary single-process run.
  A structure that only functions under ``mpirun`` would be one nobody could
  develop against, and the singleton path is also the one downstream consumers reach
  first. Those checks run in a CHILD process (``tests/rank_singleton_cases.py``)
  — read that file's header: ``MPI_Init`` inside this long CUDA/cupy/torch
  session crashed the interpreter in about one run in three, AFTER the summary
  line, which is a verdict that never reaches the caller's exit code;
* ``eagle._core`` does NOT link ``libmpi`` — the whole reason the rank surface is
  a SECOND extension. Asserted by reading the built object's own ``ldd``, because
  a promise in a CMake comment is not a check.

The MPI-dependent rows carry the ``mpi`` marker, gated in ``conftest.py`` exactly
as the existing ``gpu``/``torch`` markers are: ``eagle._mpi`` is optional by
design (``-DEAGLE_PYTHON_MPI``), and a build without an MPI legitimately has no
rank structure to exercise. What is NOT conditional is the REFUSAL: a build
without ``_mpi`` must say so by naming the build switch, and that row runs
everywhere.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import rank_singleton_cases
import v2_artifact

import eagle.exec as eexec
from eagle import plan as eplan

PY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES_SCRIPT = PY_ROOT / "tests" / "rank_singleton_cases.py"


@pytest.fixture(scope="module")
def bed_artifact(tmp_path_factory):
    """A freshly built default aether-abi/2 artifact root.

    Built here rather than committed: its bodies are the repo's own v2 fixture
    TUs, and a committed binary would go stale against them silently. Raises
    rather than skips when the toolchain is missing — the same choice
    ``test_exec_contract_rows.py`` makes for its own compiled fixture."""
    return v2_artifact.build(tmp_path_factory.mktemp("rank_artifact"))


# --------------------------------------------------------------------------- #
# The structure is real (no MPI needed).
# --------------------------------------------------------------------------- #
def test_rank_partition_is_an_implemented_structure():
    """``rank_partition`` resolves to a runnable structure, not to the shared
    "not implemented" placeholder it once was — and ``device_group``, which
    not yet implemented, still is one (NCCL last)."""
    assert eexec._structure_from_name("rank_partition") is eexec.RankPartition
    assert eexec.RankPartition.name == "rank_partition"
    assert type(eexec.RankPartition) is not type(eexec.DeviceGroup)
    with pytest.raises(ValueError, match="NCCL not implemented"):
        eexec.check_placement("sample_local", eexec.DeviceGroup, 1)


def test_placement_table_for_the_rank_structure_follows_l3():
    """The placement table itself: ``sample_local``, ``cross_sample_read`` (legal
    HERE because inputs are REPLICATED, so a body's lookup/Staged tables are
    whole on every rank — the table's stated condition) and ``mapreduce`` are legal at
    ANY world size; ``cross_sample_write`` is refused at EVERY world size
    INCLUDING one, because a placement that is legal at ``-np 1`` and illegal at
    ``-np 2`` is a trap."""
    for npartitions in (1, 2, 8):
        for access in ("sample_local", "cross_sample_read", "mapreduce"):
            eexec.check_placement(access, eexec.RankPartition, npartitions)
        with pytest.raises(ValueError) as excinfo:
            eexec.check_placement(
                "cross_sample_write", eexec.RankPartition, npartitions
            )
        message = str(excinfo.value)
        for fragment in ("cross_sample_write", "rank_partition", "F-e"):
            assert fragment in message


def test_plan_refuses_cross_sample_write_on_the_rank_structure_naming_the_fork():
    """At PLAN time: the refusal reaches the caller from :func:`eagle.plan`,
    before anything is packed or launched, and names the rule."""

    class _Body:
        arg_spec = (("wide_out", "acc"), ("per_sample", "x"), ("nsamples", "n"))
        exec_access = "cross_sample_write"
        exec_op = None

    with pytest.raises(ValueError, match="F-e"):
        eplan.plan(_Body(), structure=eexec.RankPartition, inner=eexec.HostTeam)


def test_inner_and_gather_are_refused_for_a_structure_that_has_neither():
    """``inner=`` names the structure each RANK's share is driven through and
    ``gather=False`` suppresses the rank gather; neither means anything under a
    single-process structure, so both are refused by NAME rather than silently
    ignored — a silently ignored ``inner=`` would let a caller believe a
    ``HostTeam`` plan was distributed."""

    class _Body:
        arg_spec = (("mutable", "y"), ("per_sample", "x"), ("nsamples", "n"))
        exec_access = "sample_local"
        exec_op = None

    with pytest.raises(ValueError, match="rank_partition"):
        eplan.plan(_Body(), structure=eexec.HostTeam, inner=eexec.DeviceKernel)
    with pytest.raises(ValueError, match="PRE-GATHER"):
        eplan.plan(_Body(), structure=eexec.HostTeam, _gather=False)


def test_missing_mpi_module_refuses_by_naming_the_build_switch(monkeypatch):
    """A build configured without an MPI has no ``eagle._mpi``, and the caller
    must be told THAT — with the switch that fixes it — rather than being handed
    a bare ``ImportError`` indistinguishable from a broken install.

    The absence is SIMULATED (the import is forced to fail) so the row runs on
    every build, including this one, where the module is present."""
    import builtins

    real_import = builtins.__import__

    def _no_mpi(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("_mpi") or "_mpi" in (fromlist or ()):
            raise ImportError("simulated: eagle._mpi was not built")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_mpi)
    assert eexec.RankPartition.available is False
    with pytest.raises(RuntimeError, match="EAGLE_PYTHON_MPI=ON"):
        eexec.RankPartition.world()


# --------------------------------------------------------------------------- #
# The audit: `_core` never links libmpi.
# --------------------------------------------------------------------------- #
def test_core_extension_does_not_link_libmpi():
    """The reason the rank surface is a SECOND extension: ``eagle._core`` is
    imported by every eagle user, including the ones with no MPI installed, so it
    must not carry ``libmpi`` as a dependency.

    Read off the BUILT object with ``ldd``, and the matcher is proven on a known
    answer first — ``eagle._mpi`` MUST show a hit — because a scan that reports
    "no hits" is evidence only once it has been shown it can hit."""
    from eagle import _core

    def _needs_mpi(path):
        out = subprocess.run(
            ["ldd", str(path)], capture_output=True, text=True, check=True
        ).stdout
        return [line.strip() for line in out.splitlines() if "libmpi" in line]

    if eexec.RankPartition.available:
        from eagle import _mpi

        assert _needs_mpi(_mpi.__file__), (
            "the ldd matcher found no libmpi in eagle._mpi, which links it — the "
            "audit below cannot fail and certifies nothing"
        )
    assert not _needs_mpi(_core.__file__), (
        "eagle._core links libmpi; every eagle user would then need an MPI "
        "installed just to `import eagle`"
    )


# --------------------------------------------------------------------------- #
# The singleton world — driven in a CHILD process.
# --------------------------------------------------------------------------- #
@pytest.mark.mpi
@pytest.mark.parametrize("case", sorted(rank_singleton_cases.CASES))
def test_singleton_world_cases_pass_in_their_own_process(case, bed_artifact):
    """Every single-process ``RankPartition`` check, one child per case.

    THE CHILD IS THE POINT, not an implementation detail: ``MPI_Init`` installs
    process-wide memory hooks that do not coexist with a long-lived CUDA/cupy/torch
    session (see ``tests/rank_singleton_cases.py`` for the measurement), so this
    suite never initialises MPI in its own process. The verdict crosses back as an
    EXIT CODE, and is cross-checked against the child's own ``OK <case>`` marker —
    a child that died before reaching its assertions exits non-zero AND prints no
    marker, so neither channel alone has to be trusted."""
    proc = subprocess.run(
        [sys.executable, str(CASES_SCRIPT), case, str(bed_artifact)],
        capture_output=True,
        text=True,
        cwd=str(PY_ROOT),
    )
    assert proc.returncode == 0, (
        f"the {case!r} case failed in its child process (rc {proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert f"OK {case}" in proc.stdout, (
        f"the {case!r} child exited 0 without reporting it reached its assertions:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.mpi
def test_the_singleton_case_runner_can_fail(bed_artifact):
    """The instrument must be able to fail. A case name the runner does not
    implement must come back NON-ZERO and without an ``OK`` marker — otherwise the
    row above would report success for a child that never ran a check."""
    proc = subprocess.run(
        [sys.executable, str(CASES_SCRIPT), "no_such_case", str(bed_artifact)],
        capture_output=True,
        text=True,
        cwd=str(PY_ROOT),
    )
    assert proc.returncode != 0
    assert "OK " not in proc.stdout
