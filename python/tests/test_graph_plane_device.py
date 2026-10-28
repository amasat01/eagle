# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A graph runs on the one CUDA device its planes live on: a plane on another device is refused at build,
by name (the plane, its device, the graph's device), before any kernel is reached.

The check is one function, ``eagle.graph._plane_device``; the first group covers it with stand-in planes
(no GPU needed), the others build real graphs: one GPU asks for ``device=1``, two GPUs mix planes."""

from types import SimpleNamespace

import pytest

import eagle
from eagle import graph as G
from eagle.graph import opaque


class _Plane:
    """A stand-in device plane on CUDA ordinal ``ordinal``: what ``device_kind`` and ``array_device`` read."""

    def __init__(self, ordinal):
        self.data = SimpleNamespace(device_id=ordinal)
        self._ordinal = ordinal

    def __dlpack_device__(self):
        return (2, self._ordinal)


class _HostPlane:
    pass


def _access(reads=(), writes=()):
    return SimpleNamespace(reads=tuple((n, a, "r") for n, a in reads),
                           writes=tuple((n, a, "w") for n, a in writes))


def test_planes_on_one_device_give_that_device():
    acc = [_access(reads=[("x", _Plane(1))], writes=[("y", _Plane(1))])]
    assert G._plane_device(acc) == 1
    assert G._plane_device(acc, 1) == 1
    assert G._plane_device([_access(reads=[("h", _HostPlane())])], 3) == 3


def test_a_plane_on_another_device_is_refused_by_name():
    acc = [_access(reads=[("x", _Plane(0))]), _access(writes=[("y", _Plane(1))])]
    with pytest.raises(ValueError, match=r"plane 'y' lives on device 1, the graph runs on device 0 "
                                         r"\(where plane 'x' is\)"):
        G._plane_device(acc)


def test_an_asked_device_a_plane_is_not_on_is_refused_by_name():
    acc = [_access(reads=[("x", _Plane(1))], writes=[("y", _Plane(1))])]
    with pytest.raises(ValueError, match=r"plane 'x' lives on device 1, the graph runs on device 0 "
                                         r"\(device=0 was asked\)"):
        G._plane_device(acc, 0)


def _counted():
    """An opaque node that counts its runs: a refusal at build must leave it at zero."""
    calls = []
    return calls, (lambda x, y: opaque(lambda: calls.append(1), reads={"x": x}, writes={"y": y}))


@pytest.mark.gpu
def test_build_refuses_a_device_the_planes_are_not_on():
    import cupy as cp

    x, y = cp.zeros(4), cp.zeros(4)
    here = x.device.id
    calls, node = _counted()
    with pytest.raises(ValueError, match=rf"plane 'x' lives on device {here}, the graph runs on "
                                         rf"device {here + 1} \(device={here + 1} was asked\)"):
        eagle.Graph().add_node(node(x, y)).build(device=here + 1)
    assert calls == []


@pytest.mark.gpu2
def test_build_refuses_planes_on_two_devices_by_name():
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two GPUs")
    x = cp.zeros(4)
    with cp.cuda.Device(1):
        y, z = cp.zeros(4), cp.zeros(4)
    try:
        calls, node = _counted()
        with pytest.raises(ValueError, match=r"plane 'y' lives on device 1, the graph runs on device 0 "
                                             r"\(where plane 'x' is\)"):
            eagle.Graph().add_node(node(x, y)).build()
        with pytest.raises(ValueError, match=r"plane 'x' lives on device 1, the graph runs on device 0 "
                                             r"\(device=0 was asked\)"):
            eagle.Graph().add_node(node(y, z)).build(device=0)
        assert calls == []
    finally:
        cp.cuda.Device(0).use()
