# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A graph with parallel branches: fork, join, reduce -- on the GPU and the CPU.

    x --+-- branch_a --+
        +-- branch_b --+-- join --> s --> sum(s)
        +-- branch_c --+

Three independent branches read the same input and write their own output
(the fork), one kernel combines them per sample (the join), and a final sum
collapses the ensemble to one number (the reduce). The GPU route records the
fork/join as one CUDA graph; the CPU route runs the same bound launches on the
host. The script checks the two agree.

Run:  python fork_join_graph.py
"""
import numpy as np

import hawk
from hawk import Mutable, Param, Scalar, Terminated

from eagle import GraphPipeline, deploy


# [cell:kernels]
def branch_a(x: Scalar, k: Param, terminated: Terminated, a: Mutable[Scalar]):
    a = x * x + k


def branch_b(x: Scalar, k: Param, terminated: Terminated, b: Mutable[Scalar]):
    b = 3.0 * x - k


def branch_c(x: Scalar, terminated: Terminated, c: Mutable[Scalar]):
    c = 0.5 * x + 1.0


def join(a: Scalar, b: Scalar, c: Scalar, terminated: Terminated, s: Mutable[Scalar]):
    s = a * b + c


kernels = [hawk.kernel(fn) for fn in (branch_a, branch_b, branch_c, join)]
plan_a, plan_b, plan_c, plan_join = deploy(kernels)
# [cell:kernels:end]


# [cell:build]
def bind_all(xp, n=100_000, k=0.5):
    """Allocate the ensemble's arrays and pack each kernel's arguments once."""
    x = xp.asarray(np.linspace(-1.0, 1.0, n))
    a, b, c, s = (xp.zeros(n) for _ in range(4))
    dead = xp.zeros(n, dtype=xp.bool_)  # the `terminated` mask: nobody is finished
    runs = [
        plan_a.bind(x=x, k=k, terminated=dead, a=a),
        plan_b.bind(x=x, k=k, terminated=dead, b=b),
        plan_c.bind(x=x, terminated=dead, c=c),
        plan_join.bind(a=a, b=b, c=c, terminated=dead, s=s),
    ]
    return runs, s


def run_gpu():
    import cupy as cp

    (ra, rb, rc, rj), s = bind_all(cp)
    pipe = GraphPipeline()
    pipe.add_concurrent([ra.launch, rb.launch, rc.launch], names=["a", "b", "c"])  # fork
    pipe.add(rj.launch, name="join")                                                # join
    pipe.build()
    pipe.launch()
    cp.cuda.Device().synchronize()
    return cp.asnumpy(s), float(cp.sum(s))                                          # reduce


def run_cpu():
    (ra, rb, rc, rj), s = bind_all(np)
    for run in (ra, rb, rc, rj):  # same launches, host team, in order
        run.launch()
    return s, float(np.sum(s))
# [cell:build:end]


if __name__ == "__main__":
    s_cpu, total_cpu = run_cpu()
    try:
        s_gpu, total_gpu = run_gpu()
    except Exception as exc:  # no usable GPU
        print("GPU route unavailable:", type(exc).__name__)
        print(f"cpu total = {total_cpu:.9f}")
        raise SystemExit(0)
    print(f"gpu total = {total_gpu:.9f}")
    print(f"cpu total = {total_cpu:.9f}")
    print(f"max |gpu - cpu| over {s_cpu.size} samples = {np.abs(s_gpu - s_cpu).max():.3e}")
