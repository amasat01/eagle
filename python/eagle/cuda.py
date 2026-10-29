# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle.cuda — the CUDA backend surface (torch.cuda-familiar).

Mirrors the C++ ``eagle::cuda`` namespace 1:1 in Python: the nanobind-bound
CUDA-graph machinery from the compiled :mod:`eagle._core` extension — the
capturable graph (:class:`Graph`), the stream-capture recorder
(:class:`StreamCapturer`) and its owned result (:class:`CapturedGraph`), the
CUDA stream wrapper (:class:`Stream`, the ``cudaStream_t`` interop backbone), and
the instantiated/replayable executable handle (:class:`Launcher`).

Importing this submodule loads the compiled ``_core`` extension, so it is kept
out of the top-level :mod:`eagle` import (which must stay usable in a
pure-Python / no-GPU install). Reach these as ``eagle.cuda.Graph`` etc.

It also carries the capture-introspection trio a graph recorder needs: the
forked-stream scope (:class:`CaptureFork`), the node snapshot of a capture
(:func:`capture_snapshot_nodes`) and whether a captured node can be toggled
(:func:`is_node_toggleable`).

The CPU backend (``eagle::cpu`` — the host graph executor + OMP Scan/Reduction)
is currently C++-only.
"""

from ._core import (
    CaptureFork,
    CapturedGraph,
    Graph,
    Launcher,
    Stream,
    StreamCapturer,
    capture_snapshot_nodes,
    is_node_toggleable,
)

__all__ = ["CaptureFork", "CapturedGraph", "Graph", "Launcher", "Stream",
           "StreamCapturer", "capture_snapshot_nodes", "is_node_toggleable"]


def _selftest() -> int:
    """``python -m eagle.cuda --selftest``: a short report of what this install can do on this machine.

    Steps: load the plugin, find a driver and a device, make a stream, capture an empty graph. Prints one line
    per step. Exit 0 when every step passes, 2 when there is no driver or no device (the plugin itself loaded or
    not, the report says), 1 when a step fails on a machine that has one.
    """
    from . import BackendUnavailable
    from . import _core

    info = _core.cuda_backend()
    print(f"plugin: {'loaded' if info['loaded'] else 'NOT loaded'}  path={info['path'] or '-'}  "
          f"version={info['version'] or '-'}  error={info['error'] or '-'}")
    if not info["loaded"]:
        return 1
    try:
        stream = _core.Stream(True)
    except BackendUnavailable as exc:
        print(f"driver: none ({exc})")
        return 2
    try:
        props = _core.device_props(0)
        print(f"device: {props['name']}  sm_{props['cc_major']}{props['cc_minor']}  {props['sm_count']} SMs")
        stream.synchronize()
        print("stream: created and synchronised")
        capturer = _core.StreamCapturer(stream.ptr())
        capturer.begin()
        graph = capturer.end()
        print(f"capture: empty graph recorded ({'non-empty handle' if graph else 'empty handle'})")
    except Exception as exc:  # report, then fail: the user pastes this back
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    print("selftest: pass")
    return 0


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--selftest"]:
        sys.exit("usage: python -m eagle.cuda --selftest")
    sys.exit(_selftest())
