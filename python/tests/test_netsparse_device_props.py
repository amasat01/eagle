# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle._device_props``'s Python VIEW over
``eagle::DeviceProps``. Two tiers, the same split
``test_netsparse_member_enable.py`` documents:

  (a) HOST-MODE -- no GPU, no ``eagle._core``. Field-set/frozen-ness of the
      dataclass, exercised against a HAND-BUILT dict shaped exactly like
      ``eagle._core.device_props()``'s real return (never imports
      ``eagle._core``).

  (b) CUDA-ARM -- ``pytest.mark.gpu``: :func:`eagle._device_props.
      device_props` against the REAL compiled ``eagle._core.device_props()``
      binding -- every field present, and identical to the raw dict
      ``eagle._core.device_props()`` itself returns (no Python-side
      computation anywhere in the reshape)."""

from __future__ import annotations

import dataclasses

import pytest

from eagle._device_props import DeviceProps, device_props

#: Shaped exactly like eagle._core.device_props()'s real dict (eagle_core.cu's
#: `device_props` binding) -- every DeviceProps field, once each.
_FAKE_RAW = {
    "name": "Fake GPU",
    "cc_major": 6,
    "cc_minor": 1,
    "sm_count": 8,
    "clock_rate_khz": 1_000_000,
    "memory_clock_rate_khz": 3_500_000,
    "memory_bus_width_bits": 128,
    "shared_mem_per_block": 49152,
    "shared_mem_per_sm": 98304,
    "regs_per_block": 65536,
    "regs_per_sm": 65536,
    "warp_size": 32,
    "peak_bytes_per_s": 1.0e11,
    "peak_flops_sp": 1.0e12,
    "peak_flops_dp": 1.0e12 / 32,
    "fp64_ratio": 32.0,
    "ridge_flops_per_byte_sp": 10.0,
    "ridge_flops_per_byte_dp": 10.0 / 32,
}


# ============================================================================
# (a) HOST-MODE -- no GPU, no eagle._core.
# ============================================================================


def test_device_props_fields_match_the_core_dict_shape():
    """Every field name eagle._core.device_props() is documented to return
    (eagle_core.cu's device_props binding) has a same-named DeviceProps
    field, and construction from that exact dict round-trips field-for-
    field -- the "no Python-side computation" contract, checked
    structurally."""
    props = DeviceProps(**_FAKE_RAW)
    for name, value in _FAKE_RAW.items():
        assert getattr(props, name) == value


def test_device_props_is_frozen():
    props = DeviceProps(**_FAKE_RAW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        props.sm_count = 999


def test_ridge_flops_per_byte_selects_by_dtype_no_arithmetic():
    """A plain select between the two precomputed fields -- "float32" picks
    ``_sp``, anything else ("float64") picks ``_dp``, exactly like
    ``eagle::DeviceProps::ridgeFlopsPerByte``'s own dtype convention."""
    props = DeviceProps(**_FAKE_RAW)
    assert props.ridge_flops_per_byte("float32") == props.ridge_flops_per_byte_sp
    assert props.ridge_flops_per_byte("float64") == props.ridge_flops_per_byte_dp


# ============================================================================
# (b) CUDA-ARM -- pytest.mark.gpu, the real compiled eagle._core.device_props().
# ============================================================================


@pytest.mark.gpu
def test_device_props_values_identical_to_the_raw_core_dict():
    from eagle import _core

    raw = _core.device_props(0)
    props = device_props(0)
    for name, value in raw.items():
        assert getattr(props, name) == value


@pytest.mark.gpu
def test_device_props_default_device_is_zero():
    assert device_props() == device_props(0)
