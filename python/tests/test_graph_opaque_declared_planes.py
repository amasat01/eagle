# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""An opaque callable is allowed with DECLARED planes: declared overlap is
refused, an undeclared write is NOT detected (documented limit)."""

import numpy as np
import pytest

import eagle
from eagle.graph import opaque


def test_declared_sibling_overlap_is_refused_naming_both():
    shared = np.zeros(4)
    g = eagle.Graph().add_group([
        opaque(lambda: None, writes={"left": shared}),
        opaque(lambda: None, reads={"right": shared[1:]}),
    ])
    with pytest.raises(ValueError) as exc:
        g.build()
    msg = str(exc.value)
    assert "siblings of one fork/join group" in msg
    assert "'left'" in msg and "'right'" in msg


def test_undeclared_write_that_aliases_a_sibling_is_not_detected():
    shared = np.zeros(4)
    hidden = shared                     # the closure writes it; nothing declares it
    g = eagle.Graph().add_group([
        opaque(lambda: hidden.__setitem__(slice(None), 1.0)),
        opaque(lambda: None, reads=[shared]),
    ])
    g.build()                           # documented limit: no refusal
    assert g.backend == "cpu"


def test_opaque_runs_eagerly_on_the_cpu_in_add_order():
    log = []
    a = np.zeros(2)
    g = (eagle.Graph().add_node(opaque(lambda: log.append("a"), writes=[a]), name="first")
         .add_node(opaque(lambda: log.append("b"), reads=[a]), name="second"))
    g.build().launch(2)
    assert log == ["a", "b", "a", "b"]
    assert g.node_labels() == ["first", "second"]
    assert g.schedule == ("work_share", "work_share")


def test_opaque_needs_a_callable():
    with pytest.raises(TypeError):
        opaque(5)
