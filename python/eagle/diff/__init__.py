# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.diff``: gradients through a recorded :class:`eagle.Graph`, by
reverse mode.

Build the graph with ``record=True`` and launch it; then
``eagle.diff.reverse(g.record, seeds)`` returns the gradient of your
objective ``J`` with respect to every input plane, given its gradient with
respect to the outputs you name (the SEEDS, ``∂J/∂output``, one value per
sample). Each node's derivative kernel is its vjp (``hawk.diff.vjp``); eagle
runs those backwards through the graph, sums adjoints in one fixed order, and
reduces parameter gradients in sample order, so a gradient is the same bits
on every run and every thread count.
"""

from __future__ import annotations

from .. import graph as _graph
from ._record import Record, Recorder
from ._reverse import Gradients, Report, Reverse, broadcast_adjoint, reverse
from .events import phase

__all__ = ["Gradients", "Record", "Report", "Reverse", "broadcast_adjoint", "phase",
           "reverse"]

_graph._install_record_factory(Recorder)
