# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Scalar-type tags of the launch/artifact contract.

A compiled or deployed kernel carries a resolved ``scalar_type`` tag — one of
:data:`SCALAR_TYPES`, in sidecar spelling — and :func:`np_dtype` gives the numpy
dtype of its Real-typed arrays. Both are the schema's own
(:mod:`raptor.schema.dtypes`), re-exported here.
"""

from __future__ import annotations

from raptor.schema.dtypes import SCALAR_TYPES, np_dtype

__all__ = ["SCALAR_TYPES", "np_dtype"]
