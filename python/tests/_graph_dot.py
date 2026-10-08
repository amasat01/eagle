# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Reading a captured graph's ``cudaGraphDebugDotPrint`` output across driver
versions: older drivers label each conditional node ``Type: IF`` /
``Type: WHILE`` in the default print, newer ones (580) print no node type at
all. The child graphs are drawn as their own clusters on both, so the cluster
count is the structure every driver shows."""

from __future__ import annotations

import re


def conditional_count(dot: str, kind: str) -> int | None:
    """How many ``kind`` (``"IF"``/``"WHILE"``) conditional nodes ``dot``
    labels, or ``None`` when this driver's print labels no node types."""
    if "Type:" not in dot:
        return None
    return dot.count(f"Type: {kind}")


def clusters(dot: str) -> list:
    """The bodies of the dot's graph clusters: the outer graph first, then one
    per conditional child graph."""
    return re.findall(r"subgraph cluster_\d+ \{(.*?)\n\}", dot, re.S)
