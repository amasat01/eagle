# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Fused or split? Picks how a ``dispatch(..., policy="auto")`` group runs.

hawk counts what each branch costs from the kernel's IR
(:func:`hawk.ir.dispatch_costs`); a running simulation measures how many
samples took each branch; the rule that compares the two is C++
(``eagle/policy/Dispatch.h``) and this module only gathers its inputs and
applies the one condition that is about the kernel step rather than the
numbers: a branch mix may be split only when the selector cannot change
between sorts.

The answer is ``"switch"`` (one fused kernel) or ``"segmented"`` (one kernel
per branch over a sorted batch). A policy a user declared is never routed
through here.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence

from . import _core

__all__ = ["DISPATCH_ENTER_GAIN", "DISPATCH_LEAVE_GAIN", "DISPATCH_EVERY", "DispatchState",
           "decide", "predicted_gain", "resolve", "static_selector"]

#: Predicted gain (fused time / split time) at which a group starts being split.
DISPATCH_ENTER_GAIN = _core.DISPATCH_ENTER_GAIN
#: Predicted gain under which a split group goes back to one fused kernel.
DISPATCH_LEAVE_GAIN = _core.DISPATCH_LEAVE_GAIN
#: Steps between two looks at the branch mix.
DISPATCH_EVERY = _core.DISPATCH_EVERY

_PROPS_FIELDS = ("peak_flops_sp", "peak_bytes_per_s", "fp64_ratio")


def static_selector(selector_reads: Iterable[Sequence[str]], written: Iterable[str]) -> bool:
    """Can the selector's value change between two class sorts?

    A branch class is stable only when no plane the selector reads is written
    by any kernel of the step. Splitting a group whose class drifts is correct
    but slower than the fused kernel, so the rule only considers static ones.

    Parameters
    ----------
    selector_reads
        The ``(role, name)`` slots the group's selector reads (hawk's
        ``DispatchCost.selector_reads``, or a sidecar's ``selector_reads``).
    written
        The names of every plane any kernel of the step writes.

    Returns
    -------
    bool
        True when no read plane is written.

    Example
    -------
    >>> static_selector([("per_sample", "kind")], {"state", "t"})
    True
    >>> static_selector([("mutable", "mode")], {"mode"})
    False
    """
    written = set(written)
    return not any(read[1] in written for read in selector_reads)


def decide(gain: float, current: str) -> str:
    """The hysteresis band alone: ``"segmented"`` at a gain of
    :data:`DISPATCH_ENTER_GAIN` or more, ``"switch"`` under
    :data:`DISPATCH_LEAVE_GAIN`, ``current`` in between.

    Example
    -------
    >>> decide(1.10, "switch"), decide(1.10, "segmented"), decide(1.2, "switch")
    ('switch', 'segmented', 'segmented')
    """
    return _core.dispatch_decide(gain, current)


def _props(props, target: str) -> Mapping[str, float]:
    """The device facts the rule reads, from a mapping or a ``DeviceProps`` view."""
    if target not in ("host", "device"):
        raise ValueError(f"target must be 'host' or 'device', got {target!r}")
    if props is None:
        if target == "device":
            raise ValueError("a device target needs its device properties: pass "
                             "props=eagle._device_props.device_props()")
        return _core.cpu_device_props()
    get = props.get if isinstance(props, Mapping) else lambda f: getattr(props, f)
    return {f: get(f) for f in _PROPS_FIELDS}


def _rows(cost) -> list[tuple[float, ...]]:
    return [b.as_tuple() for b in cost.branches]


def _vectorised(cost) -> bool:
    return bool(getattr(cost, "fused_vectorised", False))


def predicted_gain(cost, counts: Sequence[int], *, target: str = "host", props=None,
                   fused_steps: int = 1, remaining_steps: int = DISPATCH_EVERY,
                   mixed32: float | None = None) -> float:
    """Predicted time of the fused kernel over the split one for this branch mix.

    Parameters
    ----------
    cost
        hawk's :class:`~hawk.ir.DispatchCost` for the group.
    counts
        Samples that took each branch (exact integers, one per branch).
    target
        ``"host"`` or ``"device"``.
    props
        Device properties (an ``eagle._device_props.DeviceProps`` or a mapping
        with ``peak_flops_sp``, ``peak_bytes_per_s``, ``fp64_ratio``); the host
        defaults to its documented values, a device must supply them.
    fused_steps
        Steps one fused launch covers; the split path launches every step.
    remaining_steps
        Steps left in the run, over which one class sort is paid.
    mixed32
        Measured share of 32-lane groups holding two or more branches (the
        device's divergence); ``None`` uses the independent-lanes estimate.

    Returns
    -------
    float
        Above 1 the split wins; 1 for an empty mix.

    Raises
    ------
    ValueError
        For a ``target`` other than the two above, a device without ``props``,
        or a group that does not have 2 to 16 branches.
    """
    return _core.dispatch_gain(_rows(cost), list(counts), _props(props, target), target,
                               fused_steps, _vectorised(cost), remaining_steps,
                               -1.0 if mixed32 is None else mixed32)


class DispatchState:
    """What the rule remembers between two looks: the policy in force and the
    steps since the last look. Safe to share between threads.

    Example
    -------
    >>> state = DispatchState()
    >>> state.policy
    'switch'
    """

    def __init__(self, policy: str = "switch") -> None:
        if policy not in ("switch", "segmented"):
            raise ValueError(f"policy must be 'switch' or 'segmented', got {policy!r}")
        self._lock = threading.Lock()
        self._policy = policy
        self._since = 0

    @property
    def policy(self) -> str:
        """The policy in force."""
        return self._policy

    def due(self, steps: int = 1) -> bool:
        """Count ``steps`` more steps; True every :data:`DISPATCH_EVERY` steps,
        when the mix should be looked at again."""
        with self._lock:
            look, self._since = _core.dispatch_due(self._since, steps)
            return look

    def apply(self, gain: float) -> str:
        """Run a measured ``gain`` through the band; keep and return the policy."""
        with self._lock:
            self._policy = decide(gain, self._policy)
            return self._policy

    def update(self, cost, counts: Sequence[int], *, static: bool = True, **kwargs) -> str:
        """Look at the mix now; keep and return the resulting policy.

        ``static`` is :func:`static_selector` for the group: a group whose
        selector can change between sorts stays ``"switch"``. The other keywords
        are :func:`resolve`'s."""
        policy = resolve(cost, counts, current=self._policy, static=static, **kwargs)
        with self._lock:
            self._policy = policy
        return policy


def resolve(cost, counts: Sequence[int], *, target: str = "host", props=None,
            current: str = "switch", static: bool = True, fused_steps: int = 1,
            remaining_steps: int = DISPATCH_EVERY, mixed32: float | None = None) -> str:
    """The policy for a group under this branch mix: ``"segmented"`` once the
    predicted gain reaches :data:`DISPATCH_ENTER_GAIN`, ``"switch"`` once it
    falls under :data:`DISPATCH_LEAVE_GAIN`, ``current`` in between (and for
    an empty mix). A group with a non-static selector is always ``"switch"``.

    Parameters
    ----------
    cost, counts, target, props, fused_steps, remaining_steps, mixed32
        As in :func:`predicted_gain`.
    current
        The policy in force now (the band keeps it).
    static
        :func:`static_selector` for the group.

    Returns
    -------
    str
        ``"switch"`` or ``"segmented"``.

    Raises
    ------
    ValueError
        As :func:`predicted_gain`, and for a ``current`` that is neither policy.

    Example
    -------
    >>> resolve(cost, [1_000_000, 10_000], static=True)  # doctest: +SKIP
    'segmented'
    """
    if not static:
        return "switch"
    return _core.resolve_dispatch_policy(_rows(cost), list(counts), _props(props, target),
                                         target, current, fused_steps, _vectorised(cost),
                                         remaining_steps, -1.0 if mixed32 is None else mixed32)
