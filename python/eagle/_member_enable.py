# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Member-enable surface: individually addressable pipeline members,
toggleable between replays with no recapture (composition mode ``"enabled"``).

Differs from :mod:`eagle._conditional`'s ``SkipGuard``/``Skippable`` (mode
``"conditional"``), which bakes a device-evaluated IF node into graph
structure and re-reads its predicate every replay. This mechanism captures
all members, then flips a driver-level enabled bit on an
already-instantiated exec graph -- zero per-replay cost, microsecond-scale
toggles.

Member boundaries are recorded inside
:meth:`~eagle.pipeline.GraphPipeline.build`: it snapshots the
in-construction graph's node set (:func:`eagle._core.capture_snapshot_nodes`)
before and after each member's launches and records the set difference. A
member whose node set contains anything other than a kernel, memcpy or
memset node (most pertinently a :class:`~eagle._conditional.Skippable`'s
conditional node) is recorded non-toggleable, and a later toggle raises
:class:`NonToggleableMemberError`.

:class:`RecordedSchedule` is the host/numpy-mode twin: the same
registration surface, replayed by plain Python calls with a disabled
member's callable simply skipped. No cupy, no CUDA, no GPU.

CUDA floor: requires CUDA >= 12.3 (``cudaStreamGetCaptureInfo_v3``), same
as the conditional facility already imposes on ``eagle._core``.
"""

from __future__ import annotations


class UnknownMemberError(LookupError):
    """No member is registered under the given name, or an index is out of
    range.

    Subclasses :class:`LookupError`, the common base of ``KeyError`` and
    ``IndexError``, since :meth:`_MemberRegistry.resolve` accepts either a
    name or a plain index."""


class NonToggleableMemberError(TypeError):
    """A :meth:`~eagle.pipeline.GraphPipeline.set_member_enabled` /
    :meth:`RecordedSchedule.set_member_enabled` call targeted a member whose
    captured node set is not entirely kernel/memcpy/memset -- most commonly
    one containing a :class:`~eagle._conditional.Skippable`'s conditional
    node. Raised at the first toggle attempt, never by a raw driver
    rejection."""


class MemberHandle:
    """Opaque, stable reference to one individually-addressable pipeline
    member, returned by registration or by ``.member(name_or_index)``
    lookup.

    Immutable; compares and hashes by :attr:`index` alone. :attr:`name` is
    metadata for error messages and lookup, not part of the handle's
    identity.
    """

    __slots__ = ("index", "name")

    def __init__(self, index: int, name: str | None = None):
        self.index = int(index)
        self.name = name

    def __repr__(self) -> str:
        tail = f", {self.name!r}" if self.name is not None else ""
        return f"MemberHandle({self.index}{tail})"

    def __eq__(self, other) -> bool:
        return isinstance(other, MemberHandle) and self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)


class _MemberRegistry:
    """Shared registration-order bookkeeping used by both
    :class:`~eagle.pipeline.GraphPipeline` and :class:`RecordedSchedule`, so
    the two faces share one addressing scheme and one set of typed errors.

    Every registered unit gets the next integer index, in registration
    order (a group's members are consecutive indices). An optional name
    aliases that index; :meth:`resolve` accepts either an index, a
    :class:`MemberHandle`, or a name.
    """

    __slots__ = ("_index_by_name", "_name_by_index", "_count")

    def __init__(self):
        self._index_by_name: dict[str, int] = {}
        self._name_by_index: dict[int, str] = {}
        self._count = 0

    def register(self, name: str | None = None) -> MemberHandle:
        idx = self._count
        self._count += 1
        if name is not None:
            if name in self._index_by_name:
                raise ValueError(f"duplicate member name {name!r}")
            self._index_by_name[name] = idx
            self._name_by_index[idx] = name
        return MemberHandle(idx, name)

    def resolve(self, handle_or_index) -> int:
        """Resolve a :class:`MemberHandle`, a plain index, or a name (a bare
        ``str``) to an index."""
        if isinstance(handle_or_index, MemberHandle):
            idx = handle_or_index.index
        elif isinstance(handle_or_index, str):
            idx = self._index_by_name.get(handle_or_index)
            if idx is None:
                raise UnknownMemberError(
                    f"no member registered under name {handle_or_index!r}"
                )
        else:
            idx = int(handle_or_index)
        if not (0 <= idx < self._count):
            raise UnknownMemberError(
                f"member index {idx} out of range (0..{self._count - 1})"
            )
        return idx

    def name_of(self, index: int) -> str | None:
        return self._name_by_index.get(index)

    def label(self, index: int) -> str:
        """Human-readable ``"3"`` or ``"3 ('expert2')"`` for error messages."""
        name = self.name_of(index)
        return f"{index}" if name is None else f"{index} ({name!r})"

    def __len__(self) -> int:
        return self._count


class RecordedSchedule:
    """Host/numpy-mode twin of :class:`~eagle.pipeline.GraphPipeline`'s
    member-enable surface: the same registration surface, replayed by
    calling this object (``schedule()``). No cupy, no CUDA graph, no GPU.

    A disabled member's callable is simply skipped, never a structural
    change. Every member here is unconditionally toggleable --
    :meth:`set_member_enabled` never raises
    :class:`NonToggleableMemberError` on this class.

    ``add_concurrent``'s members always run serially, in registration
    order, which :meth:`GraphPipeline.add_concurrent`'s own contract
    already permits.
    """

    __slots__ = ("_steps", "_registry", "_enabled")

    def __init__(self):
        self._steps: list = []  # flat, registration order
        self._registry = _MemberRegistry()
        self._enabled: list[bool] = []

    def add(self, step, *, name: str | None = None) -> MemberHandle:
        """Register one zero-arg callable. Enabled by default. Returns
        this member's :class:`MemberHandle`."""
        handle = self._registry.register(name)
        self._steps.append(step)
        self._enabled.append(True)
        return handle

    def add_concurrent(
        self, members, *, names: list[str | None] | None = None
    ) -> list[MemberHandle]:
        """Register a group of >=2 members, run serially in registration
        order (permitted by :meth:`GraphPipeline.add_concurrent`'s
        contract). ``names`` (optional) aliases each member; must match
        ``len(members)`` if given."""
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"add_concurrent needs at least 2 members, got {len(members)}"
            )
        if names is not None and len(names) != len(members):
            raise ValueError(
                f"names has {len(names)} entries but members has {len(members)}"
            )
        return [
            self.add(m, name=(names[i] if names is not None else None))
            for i, m in enumerate(members)
        ]

    def member(self, name_or_index) -> MemberHandle:
        """Resolve a name or index to this schedule's stable
        :class:`MemberHandle`. Available any time after registration."""
        idx = self._registry.resolve(name_or_index)
        return MemberHandle(idx, self._registry.name_of(idx))

    def set_member_enabled(self, handle_or_index, enabled: bool) -> RecordedSchedule:
        """Enable/disable one registered member's callable for subsequent
        ``schedule()`` calls. Takes effect immediately."""
        idx = self._registry.resolve(handle_or_index)
        self._enabled[idx] = bool(enabled)
        return self

    def __len__(self) -> int:
        return len(self._registry)

    def __call__(self) -> None:
        """Run every ENABLED member's callable, in registration order,
        skipping disabled ones -- the host-mode replay."""
        for step, enabled in zip(self._steps, self._enabled):
            if enabled:
                step()
