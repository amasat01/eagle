# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The device kind comes from DLPack; ``backend="gpu"`` names the installed GPU kind."""

import numpy as np
import pytest

import eagle
from eagle import _device_kind
from eagle.graph import opaque

from _graph_fixtures import host_chain


class _Dlpack:
    def __init__(self, code):
        self.code = code

    def __dlpack_device__(self):
        return (self.code, 0)


@pytest.mark.parametrize("code, kind", [(1, "host"), (3, "host"), (2, "cuda"), (8, "metal"), (10, "hip")])
def test_the_kind_is_read_from_the_dlpack_device_type(code, kind):
    assert _device_kind.device_kind(_Dlpack(code)) == kind


def test_a_numpy_array_is_host_and_an_unknown_device_type_is_refused():
    assert _device_kind.device_kind(np.zeros(1)) == "host"
    assert _device_kind.device_kind(object()) == "host"
    with pytest.raises(ValueError, match="DLPack device type 4"):
        _device_kind.device_kind(_Dlpack(4))


def test_gpu_resolves_to_the_one_installed_kind_and_refuses_two(monkeypatch):
    monkeypatch.setattr(_device_kind, "installed_gpu_kinds", lambda: ())
    assert _device_kind.gpu_kind() == "cuda"
    monkeypatch.setattr(_device_kind, "installed_gpu_kinds", lambda: ("hip",))
    assert _device_kind.gpu_kind() == "hip"
    monkeypatch.setattr(_device_kind, "installed_gpu_kinds", lambda: ("cuda", "hip"))
    with pytest.raises(ValueError, match="cuda, hip are installed"):
        _device_kind.gpu_kind()


def test_a_host_graph_refuses_a_gpu_backend_by_name(tmp_path):
    b1, *_ = host_chain(tmp_path)
    for backend in ("gpu", "cuda"):
        with pytest.raises(ValueError, match=f"backend='{backend}' but the planes live on the cpu"):
            eagle.Graph().add(b1).build(backend=backend)


def test_an_explicit_cpu_backend_builds_without_planes():
    g = eagle.Graph().add_node(opaque(lambda: None)).build(backend="cpu")
    assert g.backend == "cpu"
