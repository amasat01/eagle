# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Scalar-type tags of the launch/artifact contract.

A compiled or deployed kernel carries a resolved ``scalar_type`` tag, one of
:data:`SCALAR_TYPES` in sidecar spelling. :func:`dtype_of` gives the :class:`eagle.DType` of its
Real-typed arrays and :func:`dtype_name` the spelling of any dtype-like value.
"""

from __future__ import annotations

from raptor.schema.dtypes import SCALAR_TYPES

from . import _core

__all__ = ["SCALAR_TYPES", "dtype_name", "dtype_of"]

_REAL = {"float64": 7, "float32": 6, "softdouble": 7}


def dtype_of(scalar_type: str):
    """The :class:`eagle.DType` of Real-typed arrays for a resolved ``scalar_type`` (SoftDouble is float64)."""
    return _core.array_dtype(_REAL[scalar_type])


def dtype_name(dtype) -> str:
    """The spelling (``"float64"``) of a DType, a string, an object with ``.name`` or a class named like its dtype."""
    if isinstance(dtype, str):
        return dtype
    return getattr(dtype, "name", None) or dtype.__name__
