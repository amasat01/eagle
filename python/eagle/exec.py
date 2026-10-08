# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle's execution structures: the thin Python face over the compiled
``eagle._core`` aether-abi/2 primitives (``Partition``, ``run_device`` /
``run_host`` / ``run_host_serial``, ``check_placement``, ``fold``, and the
layout self-check).

This is the EXECUTION half; :mod:`eagle.plan` marshals a plugin's
``arg_spec`` into by-value ABI structs and imports this module rather than
duplicating it.

:data:`RankPartition` alone lives in a second, optional compiled extension,
``eagle._mpi`` (built only with ``-DEAGLE_PYTHON_MPI``), imported lazily so
an MPI-less install never pays for it and its absence is reported as a
:class:`RuntimeError` naming the build switch, never a bare ``ImportError``.
"""

from __future__ import annotations

from . import _core


def _call(fn, *args, **kw):
    """Call a compiled ``_core`` entry point, translating its
    ``std::runtime_error`` into :class:`ValueError` (every refusal here is a
    rejected input, never an internal fault). Message text passes through
    unchanged."""
    try:
        return fn(*args, **kw)
    except RuntimeError as e:
        raise ValueError(str(e)) from e


#: The launch partition triple ``{base, count, nSamples}``, re-exported from
#: the compiled binding.
Partition = _core.Partition

#: The two ABI generations' wire tags (byte-identical to
#: :data:`eagle.abi.ABI_TAG_V1` / :data:`eagle.abi.ABI_TAG_V2`).
ABI_TAG_V1 = _core.ABI_TAG_V1
ABI_TAG_V2 = _core.ABI_TAG_V2

#: The exported symbol name a v2 plugin's layout self-check carries.
LAYOUT_SYMBOL = _core.LAYOUT_SYMBOL


class Structure:
    """One of eagle's execution structures: ``device_kernel`` / ``host_team``
    / ``rank_partition`` / ``device_group``. ``.name`` is the wire spelling.
    Instances are the singletons below -- never construct one directly."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"eagle.exec.{self.name}"

    def __eq__(self, other):
        return isinstance(other, Structure) and self.name == other.name

    def __hash__(self):
        return hash(("eagle.exec.Structure", self.name))


class _DeviceKernel(Structure):
    """The DEVICE execution structure: one ``cuLaunchKernel`` per partition."""

    def __init__(self):
        super().__init__("device_kernel")

    def run(
        self, function, params, partition, *, stream: int = 0, block: int = 256
    ) -> int:
        """Launch ``function`` (a ``CUfunction`` handle, as an int) once
        over ``partition``. ``params`` is a sequence of int addresses, one
        per already-packed by-value argument; the partition triple is
        appended by the binding. Returns 1 if launched, 0 for an empty
        partition."""
        return _call(
            _core.run_device,
            function,
            list(params),
            partition,
            stream=stream,
            block=block,
        )

    def grid(self, count: int, block: int = 256) -> int:
        """The launch grid eagle derives from ``count``."""
        return _call(_core.device_grid, count, block)


class _HostTeam(Structure):
    """The HOST execution structure: OpenMP over contiguous tiles of one
    partition."""

    def __init__(self):
        super().__init__("host_team")

    def run(self, entry, params, partition, *, bytes_per_sample: int = 0) -> int:
        """Run ``entry`` (a host function pointer, as an int) over
        ``partition`` as a team of tiles. ``params`` addresses each
        already-packed argument directly (``void* const*``). Returns the
        number of tiles run."""
        return _call(
            _core.run_host,
            entry,
            list(params),
            partition,
            bytes_per_sample=bytes_per_sample,
        )

    def run_serial(self, entry, params, partition) -> None:
        """Run ``entry`` serially over the whole of ``partition`` -- the
        correctness oracle the host/device band is judged against."""
        _call(_core.run_host_serial, entry, list(params), partition)

    def tile_size(self, bytes_per_sample: int = 0) -> int:
        """The samples per tile this team cuts for ``bytes_per_sample``."""
        return int(_core.host_tile_size(bytes_per_sample))

    def tile_count(self, partition, bytes_per_sample: int = 0) -> int:
        """How many tiles ``partition`` would be cut into."""
        return _core.host_tile_count(partition, bytes_per_sample)


#: The refusal shown when the optional ``eagle._mpi`` extension is not
#: built, naming the build switch (never a bare ``ImportError``).
_NO_MPI_MODULE = (
    "eagle._mpi is not built: configure with -DEAGLE_PYTHON_MPI=ON (needs an MPI)"
)


class _RankPartition(Structure):
    """The MULTI-PROCESS execution structure (plain MPI, mechanism only).

    Rank ``r`` of ``R`` runs its contiguous sub-partition through an inner
    structure (:data:`HostTeam`, default, or :data:`DeviceKernel`); inputs
    are replicated and outputs gathered host-staged, so every rank ends with
    the whole plane, bit-identical to a single run.

    Delegates to the optional ``eagle._mpi`` extension (eagle itself never
    links ``libmpi``); every method refuses with a :class:`RuntimeError`
    naming :data:`_NO_MPI_MODULE` where it is not built. Imported lazily, so
    merely importing :mod:`eagle.exec` costs an MPI-less install nothing."""

    def __init__(self):
        super().__init__("rank_partition")

    @staticmethod
    def _mpi():
        """The compiled rank surface, or :class:`RuntimeError` naming the
        build switch (never a bare :class:`ImportError`)."""
        try:
            from . import _mpi
        except ImportError as e:
            raise RuntimeError(_NO_MPI_MODULE) from e
        return _mpi

    @property
    def available(self) -> bool:
        """Whether the optional ``eagle._mpi`` extension is built."""
        try:
            self._mpi()
        except RuntimeError:
            return False
        return True

    def world(self) -> tuple:
        """This process's ``(rank, size)`` in ``MPI_COMM_WORLD``; a
        singleton world (no ``mpirun``) reports ``(0, 1)``."""
        return self._mpi().world()

    def local(self, partition):
        """This rank's contiguous share of ``partition``, as a
        :class:`Partition`. Uneven splits give the first ``n % size`` ranks
        one extra sample; ``n_samples`` is unchanged."""
        base, count, n_samples = self._mpi().local_partition(
            (partition.base, partition.count, partition.n_samples)
        )
        return Partition(base, count, n_samples)

    def run(
        self,
        entry,
        params,
        partition,
        *,
        inner=None,
        bytes_per_sample: int = 0,
        stream: int = 0,
        block: int = 256,
    ) -> int:
        """Run this rank's share of ``partition`` through ``inner``
        (:data:`HostTeam` by default). ``partition`` is the whole partition
        the world covers; the rank cut happens inside. Returns the inner
        structure's own count."""
        inner = inner if inner is not None else HostTeam
        name = inner.name if isinstance(inner, Structure) else str(inner)
        return _call(
            self._mpi().run_rank_partition,
            entry,
            list(params),
            (partition.base, partition.count, partition.n_samples),
            name,
            bytes_per_sample=bytes_per_sample,
            stream=stream,
            block=block,
        )

    def allgather_plane(
        self, address: int, partition, elems_per_sample: int = 1
    ) -> None:
        """Gather a host per-sample float64 plane in place
        (``MPI_Allgatherv``) so every rank ends with the whole of it.
        ``address`` is a global-sample-indexed buffer of ``elems_per_sample``
        contiguous doubles per sample."""
        _call(
            self._mpi().allgather_plane,
            int(address),
            (partition.base, partition.count, partition.n_samples),
            elems_per_sample=int(elems_per_sample),
        )

    def allgather_fold(self, op: str, partial: float) -> float:
        """Combine the per-rank ``mapreduce`` partials under ``op`` in a
        fixed rank order (``MPI_Allgather`` + :func:`fold`, never
        ``MPI_Allreduce``), returning the same value on every rank."""
        return _call(self._mpi().allgather_fold, op, float(partial))


class _Unimplemented(Structure):
    """``device_group`` -- not implemented in this build (NCCL last)."""

    def run(self, *a, **kw):
        check_placement("sample_local", self, 1)
        raise AssertionError(  # pragma: no cover - check_placement always raises above
            f"eagle.exec.{self.name}.run: unreachable (NCCL not implemented)"
        )


#: The four execution structures. ``DeviceGroup`` is a named placeholder
#: whose ``.run`` refuses "NCCL not implemented".
DeviceKernel = _DeviceKernel()
HostTeam = _HostTeam()
RankPartition = _RankPartition()
DeviceGroup = _Unimplemented("device_group")

_BY_NAME = {s.name: s for s in (DeviceKernel, HostTeam, RankPartition, DeviceGroup)}


def _structure_from_name(name: str) -> Structure:
    """Resolve a wire-spelled structure name to its singleton :class:`Structure`."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"execution structure {name!r} is not one of {sorted(_BY_NAME)}"
        ) from None


def check_placement(access: str, structure, npartitions: int = 1) -> None:
    """Raise :class:`ValueError` naming the rule if running a body of
    declared ``access`` under ``structure`` over ``npartitions`` is illegal.
    ``structure`` is a :class:`Structure` singleton or its wire name."""
    name = structure.name if isinstance(structure, Structure) else structure
    _call(_core.check_placement, access, name, npartitions)


def fold(op: str, values) -> float:
    """Combine ``values`` under ``op`` (``sum``/``times``/``max``/``land``)
    in eagle's fixed ascending order -- never atomics, so the result is
    reproducible regardless of how the run was partitioned."""
    return _call(_core.fold, op, list(values))


def layout_sizes() -> list[int]:
    """This build's layout self-check sizes, in :data:`LAYOUT_SYMBOL` order."""
    return _core.layout_sizes()


_LAYOUT_FIELD_NAMES: tuple[str, ...] | None = None


def _layout_field_names() -> tuple[str, ...]:
    """The layout fields' short names, cached after the first call."""
    global _LAYOUT_FIELD_NAMES
    if _LAYOUT_FIELD_NAMES is None:
        names = []
        for i in range(len(_core.layout_sizes())):
            raw = _core.layout_field_name(i)
            if raw.startswith("sizeof(") and raw.endswith(")"):
                raw = raw[len("sizeof(") : -1]
            names.append(raw)
        _LAYOUT_FIELD_NAMES = tuple(names)
    return _LAYOUT_FIELD_NAMES


def check_layout_sizes(sizes) -> None:
    """Refuse a v2 plugin's layout self-check if it disagrees with this
    build's own :func:`layout_sizes`, naming the field.

    A tag alone cannot catch a layout mismatch (a wrong-arch or
    stale-binding artifact can carry a correct ``aether_abi`` tag); this is
    what catches it instead.

    ``sizes`` is either a partial ``{field_name: byte_size}`` dict (only the
    given keys are checked) or the raw positional sequence a real plugin's
    exported :data:`LAYOUT_SYMBOL` carries (every position checked).

    Raises :class:`ValueError` naming the disagreeing field, or a length
    mismatch for the positional form."""
    expected = _core.layout_sizes()
    names = _layout_field_names()
    if isinstance(sizes, dict):
        pairs = list(sizes.items())
    else:
        seq = list(sizes)
        if len(seq) != len(expected):
            raise ValueError(
                f"layout self-check failed: exported {LAYOUT_SYMBOL!r} has "
                f"{len(seq)} entries, but this build expects {len(expected)} "
                f"({', '.join(names)})"
            )
        pairs = list(zip(names, seq))
    for field_name, got in pairs:
        try:
            idx = names.index(field_name)
        except ValueError:
            raise ValueError(
                f"layout self-check: unknown layout field {field_name!r} "
                f"(known fields: {list(names)})"
            ) from None
        want = expected[idx]
        if int(got) != want:
            raise ValueError(
                f"layout self-check failed: field {field_name!r} reports "
                f"size {got}, but this build expects {want} (the "
                "plugin was built against a different aether, or a stale/"
                "wrong-arch binding is earlier on the import path); rebuild "
                "the plugin"
            )
