# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle._device_props`` — the Python VIEW over ``eagle::DeviceProps``.

Every field here is a straight pass-through of ``eagle._core.device_props()``'s
dict — NO Python-side computation, ever. The formulas (``peak_bytes_per_s``,
``peak_flops_sp``/``peak_flops_dp``, ``fp64_ratio``, the two ridge points) are
the SINGLE source of truth in C++ (``eagle/DeviceProps.h``); this module only
reshapes that dict into a typed, frozen, dot-addressable object.
``ridge_flops_per_byte(dtype)`` is the one method here, and it is a plain
SELECT between the two precomputed ``ridge_flops_per_byte_sp``/``_dp`` dict
entries (a flat dict cannot carry a dtype-parametric field, so the binding
precomputes both) — dispatch, not arithmetic.

:func:`device_props` is the constructor a caller reaches for; the dataclass
itself is not meant to be built by hand outside this module."""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = ["DeviceProps", "device_props"]


@dataclass(frozen=True)
class DeviceProps:
    """Frozen view of one CUDA device's raw + derived properties.

    Field set mirrors ``eagle::DeviceProps`` (``eagle/DeviceProps.h``)
    exactly — see that header for every formula's one-line derivation."""

    name: str
    cc_major: int
    cc_minor: int
    sm_count: int
    clock_rate_khz: int
    memory_clock_rate_khz: int
    memory_bus_width_bits: int
    shared_mem_per_block: int
    shared_mem_per_sm: int
    regs_per_block: int
    regs_per_sm: int
    warp_size: int
    peak_bytes_per_s: float
    peak_flops_sp: float
    peak_flops_dp: float
    fp64_ratio: float
    ridge_flops_per_byte_sp: float
    ridge_flops_per_byte_dp: float

    def ridge_flops_per_byte(self, dtype: str) -> float:
        """flops/byte this device balances at for ``dtype`` ("float32" or
        "float64") — a SELECT between the two C++-precomputed fields above,
        the same dtype convention ``eagle::DeviceProps::ridgeFlopsPerByte``
        uses. Zero arithmetic on the Python side."""
        if dtype == "float32":
            return self.ridge_flops_per_byte_sp
        return self.ridge_flops_per_byte_dp


#: Field names, in declaration order — the exact keys ``device_props()``
#: expects to find in ``eagle._core.device_props()``'s dict.
_FIELD_NAMES = tuple(f.name for f in fields(DeviceProps))


def device_props(device: int = 0) -> DeviceProps:
    """Query device ``device``'s properties via ``eagle._core.device_props()``
    and wrap the resulting dict in a frozen :class:`DeviceProps` — no
    Python-side computation, only a dict-to-dataclass reshape.

    ``eagle._core`` (the compiled extension) is imported LOCALLY, here,
    never at this module's top — importing ``eagle`` on a cuda-free box must
    not force-load it (the same discipline :meth:`eagle.pipeline.
    GraphPipeline.__init__` follows for the same reason)."""
    from . import _core

    raw = _core.device_props(device)
    return DeviceProps(**{name: raw[name] for name in _FIELD_NAMES})
