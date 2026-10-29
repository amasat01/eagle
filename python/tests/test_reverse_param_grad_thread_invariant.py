# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A parameter's gradient (summed over samples) is the same bits for every
host thread count: per-sample terms are thread-independent, and the sum is
the canonical fixed-order tree."""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("hawk")

_CHILD = textwrap.dedent('''
    import sys
    import numpy as np
    import hawk
    from hawk import Mutable, Param, Scalar
    from hawk import math as m
    import eagle, eagle.plan

    @hawk.kernel
    def ti_step(x: Scalar, a: Param, y: Mutable[Scalar]):
        y = m.sin(a * x) * x

    (p,) = eagle.plan.auto([ti_step], targets=("host",), cache_dir=sys.argv[1])
    rng = np.random.default_rng(5)
    n = 100003
    x = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 7, n)
    y = np.zeros(n)
    g = eagle.Graph().add(p.bind(x=x, a=0.3, y=y), name="step").build(record=True)
    g.launch()
    grads = eagle.diff.reverse(g.record, {"y": rng.standard_normal(n)})
    sys.stdout.write(np.asarray(grads["a"]).tobytes().hex() + " " + np.asarray(grads["x"]).tobytes().hex()[:4096])
''')


def test_parameter_gradient_is_bitwise_thread_invariant(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    got = set()
    for threads in (1, 2, 4, 8):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        r = subprocess.run([sys.executable, str(script), str(tmp_path / "cache")], env=env,
                           capture_output=True, text=True, check=True)
        got.add(r.stdout.strip())
    assert len(got) == 1
