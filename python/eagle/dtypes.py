# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Scalar-type tags of the launch/artifact contract.

A compiled or deployed kernel carries a resolved ``scalar_type`` tag, one of
:data:`SCALAR_TYPES` in sidecar spelling. :func:`dtype_of` gives the :class:`eagle.DType` of its
Real-typed arrays and :func:`dtype_name` the spelling of any dtype-like value.
"""

from __future__ import annotations

from types import MappingProxyType

from raptor.schema.dtypes import SCALAR_TYPES

from . import _core

__all__ = ["SCALAR_TYPES", "dtype_name", "dtype_of", "scratch_dtype"]

_REAL = {"float64": 7, "float32": 6, "banded": 13}


def dtype_of(scalar_type: str):
    """The :class:`eagle.DType` of Real-typed arrays for a resolved ``scalar_type``.

    banded: an 8-byte cell dtype (eagle.DType code 13, kind 'V'); user-facing arrays
    are float64; conversion is the explicit pack/unpack below
    (:func:`eagle.banded_pack`, :func:`eagle.banded_unpack`).
    """
    return _core.array_dtype(_REAL[scalar_type])


_SCRATCH = MappingProxyType({"band_partial": 14})


def scratch_dtype(name: str):
    """The :class:`eagle.DType` of a scratch element, one that is never a user dtype (the name is refused by
    ``eagle.arrays.empty(..., dtype="band_partial")``): ``"band_partial"`` is the 16-byte banded partial (normalised
    carrier plus error lane) of hawk's block reduction. Its plane is made with ``eagle.arrays.empty`` and zeroed with
    ``fill(0)`` (16 zero bytes per element, the zero partial)."""
    try:
        return _core.array_dtype(_SCRATCH[name])
    except KeyError:
        raise ValueError(f"eagle: {name!r} is not a scratch element (known: {sorted(_SCRATCH)})") from None


def dtype_name(dtype) -> str:
    """The spelling (``"float64"``) of a DType, a string, an object with ``.name`` or a class named like its dtype."""
    if isinstance(dtype, str):
        return dtype
    return getattr(dtype, "name", None) or dtype.__name__
