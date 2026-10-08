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
