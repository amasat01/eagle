# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Framework-free DLPack detection + origin-selection logic.

The parts of :mod:`eagle.interop` that classify a tensor's framework and pick the
output framework purely from ``type(x).__module__`` and DLPack attributes — no cupy,
no framework, no GPU. The zero-copy import / round-trip legs (which need cupy + a real
or shim framework tensor) live with the producer's compiled-kernel suite; eagle
verifies only the framework-agnostic routing here.
"""

import numpy as np
import pytest

import eagle
from eagle import interop


def test_detect_numpy_list_and_scalar():
    assert interop.detect(np.ones(3)).name == "numpy"
    assert interop.detect([1.0, 2.0, 3.0]) is None  # a python list is not a tensor
    assert interop.detect(3.0) is None


def test_public_helpers_exported():
    # eagle surfaces the import helper and the two probes on its public API.
    assert eagle.to_cupy is interop.to_cupy
    assert eagle.detect is interop.detect
    assert eagle.origin_adapter is interop.origin_adapter


def test_origin_adapter_backward_compatible_defaults():
    assert interop.origin_adapter([]).name == "numpy"  # nothing -> host
    assert interop.origin_adapter([np.ones(3)]).name == "numpy"  # numpy -> numpy out


def test_origin_adapter_mixed_strong_frameworks_raise():
    # Module-tagged fakes: detect() classifies by __module__ WITHOUT importing
    # torch/jax, so this ambiguity check runs with neither framework installed.
    class _T:
        pass

    class _J:
        pass

    _T.__module__ = "torch"
    _J.__module__ = "jax"
    assert interop.detect(_T()).name == "torch"
    assert interop.detect(_J()).name == "jax"
    with pytest.raises(TypeError, match="mixed tensor frameworks"):
        interop.origin_adapter([_T(), _J()])
