#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the figure on the eagle.Graph page from the same graph the page
describes. CPU only; needs the optional plot extra (matplotlib).

    python _tools/make_graph_figure.py
"""
import pathlib

import numpy as np

import eagle
from eagle.graph import opaque

OUT = pathlib.Path(__file__).resolve().parent.parent / "content/userguide/_figures"


def example_graph() -> eagle.Graph:
    x, left, right, out = (np.zeros(8) for _ in range(4))
    flag = np.ones(1, dtype=np.uint32)
    nop = lambda: None  # noqa: E731 -- the shape is the point, not the work
    g = eagle.Graph()
    g.add_node(opaque(nop, writes={"x": x}), name="seed")
    g.add_group([opaque(nop, reads={"x": x}, writes={"left_out": left}),
                 opaque(nop, reads={"x": x}, writes={"right_out": right})],
                names=["left", "right"])
    g.add_node(eagle.skippable(
        opaque(nop, reads={"left_out": left, "right_out": right}, writes={"out": out}),
        eagle.SkipGuard.nonzero(flag)), name="combine")
    return g.build()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    g = example_graph()
    g.plot().savefig(OUT / "graph_fork_join.png", dpi=110, bbox_inches="tight")
    print(g)
