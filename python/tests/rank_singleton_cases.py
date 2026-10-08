# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The single-process :data:`eagle.exec.RankPartition` checks, run in a CHILD
process — one case per invocation, the verdict carried by the exit code.

WHY A CHILD PROCESS AND NOT AN ORDINARY TEST ROW. ``MPI_Init`` is not a passive
call: Open MPI installs process-wide memory-management hooks that do not coexist
with a long-lived process already running the CUDA driver, cupy and torch
allocators. Measured on eagle's own Python suite, with MPI initialised inside the
pytest process, the run crashed in roughly one attempt in three — inside Open
MPI's own teardown (exit 139 AFTER pytest had printed "506 passed": a verdict
reaching the terminal but not the caller's exit code) or, with finalize
suppressed, deeper still inside ``libcuda``'s ``cuModuleGetFunction``. The same
suite with these rows removed was clean every time. So MPI gets a process of its
own here, exactly as it gets an ``mpirun`` world of its own in ``tests/mpi/``.

The cases are ORDINARY ASSERTIONS, not pytest rows: this file is executed, never
collected (its name carries no ``test_`` prefix, and ``tests/`` is on the path
only so ``v2_artifact`` resolves). A failing assertion raises, the traceback goes
to stderr and the exit code is non-zero; ``tests/test_rank_partition.py`` asserts
on BOTH the exit code and the ``OK <case>`` marker, so a child that died before
reaching its assertions cannot be mistaken for one that passed them.

Usage (what the parent row runs)::

    python tests/rank_singleton_cases.py <case> <artifact-root>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import v2_artifact

import eagle.exec as eexec
from eagle import plan as eplan


def _sample_local(artifact):
    return v2_artifact.unit_for(v2_artifact.load(artifact), "sample_local").plugin


def _kwargs(plugin, n):
    """Bind every role the artifact declares, by its DECLARED name — nothing here
    knows a kernel's name, exactly as in the two-rank bed."""
    kw = {
        name: 1.0 / (np.arange(n, dtype=np.float64) + 1.0)
        for role, name in plugin.arg_spec
        if role == "per_sample"
    }
    kw.update(
        {
            name: 3.25 - 0.75 * i
            for i, (role, name) in enumerate(plugin.arg_spec)
            if role == "uniform"
        }
    )
    return kw


def _output_name(plugin) -> str:
    return next(
        name for role, name in plugin.arg_spec
        if role in ("out", "mutable", "wide_out", "accum_out")
    )


def _raises(exc_type, match: str, fn, *a, **kw):
    """Assert ``fn`` refuses with ``exc_type`` NAMING ``match`` — the child's
    stand-in for ``pytest.raises``, since this file is not a pytest module."""
    try:
        fn(*a, **kw)
    except exc_type as e:
        assert match in str(e), f"the refusal does not name {match!r}: {e}"
        return
    raise AssertionError(
        f"expected {exc_type.__name__} naming {match!r}; nothing raised"
    )


def case_world(artifact):
    """A world of one: rank 0 of 1, whose share of any partition is the whole of
    it. The rank cut is the same function at every world size, so this is the
    ``size == 1`` row of the rule the two-rank bed checks at two."""
    rank, size = eexec.RankPartition.world()
    assert (rank, size) == (0, 1), f"expected a singleton world, got {(rank, size)}"
    mine = eexec.RankPartition.local(eexec.Partition.whole(7))
    assert (mine.base, mine.count, mine.n_samples) == (0, 7, 7)
    assert eexec.RankPartition.allgather_fold("sum", 2.5) == 2.5


def case_identity(artifact):
    """At a world size of one: driving a real aether-abi/2 body through
    ``RankPartition`` must produce exactly what the ordinary single-process
    ``HostTeam`` run produces. The gather is a no-op over one rank, so any
    difference here is the rank path mis-packing or mis-slicing rather than
    anything distributed."""
    plugin = _sample_local(artifact)
    n = 256
    kw = _kwargs(plugin, n)
    rank_run = eplan.plan(
        plugin, structure=eexec.RankPartition, inner=eexec.HostTeam
    ).run(**kw)
    whole_run = eplan.plan(plugin, structure=eexec.HostTeam).run(**kw)
    assert rank_run.tobytes() == whole_run.tobytes(), (
        "the rank-partitioned run differs from the whole host run"
    )
    assert rank_run[0] != rank_run[1], "the body did nothing sample-dependent"


def case_dtype_refused(artifact):
    """The gather's wire type is ``MPI_DOUBLE``, so a plane of any other element
    type must be REFUSED rather than gathered as if it were float64 — checked
    against a caller-supplied ``float32`` output plane, the shape a
    ``scalar_type='float32'`` artifact would produce."""
    plugin = _sample_local(artifact)
    n = 64
    kw = _kwargs(plugin, n)
    kw[_output_name(plugin)] = np.zeros(n, dtype=np.float32)
    _raises(
        ValueError, "MPI_DOUBLE",
        eplan.plan(plugin, structure=eexec.RankPartition, inner=eexec.HostTeam).run,
        **kw,
    )


def case_inner_refused(artifact):
    """``RankPartition``'s inner is one of the two single-process structures; the
    unimplemented ``device_group`` is refused by NAME rather than dispatched into
    an inner that does not exist."""
    plugin = _sample_local(artifact)
    _raises(
        ValueError, "host_team or device_kernel",
        eplan.plan(plugin, structure=eexec.RankPartition, inner=eexec.DeviceGroup).run,
        **_kwargs(plugin, 32),
    )


def case_unknown_inner(artifact):
    """The compiled surface's own guard, reached directly: an inner spelling it
    does not implement is refused naming the two it does, never dispatched on a
    fallback."""
    from eagle import _mpi

    _raises(
        RuntimeError, "host_team, device_kernel",
        _mpi.run_rank_partition, 1, [], (0, 1, 1), "telepathy",
    )


CASES = {
    "world": case_world,
    "identity": case_identity,
    "dtype_refused": case_dtype_refused,
    "inner_refused": case_inner_refused,
    "unknown_inner": case_unknown_inner,
}


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in CASES:
        print(
            f"usage: {argv[0]} <{'|'.join(sorted(CASES))}> <artifact-root>",
            file=sys.stderr,
        )
        return 2
    CASES[argv[1]](pathlib.Path(argv[2]))
    print(f"OK {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
