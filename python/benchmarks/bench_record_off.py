# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Paired wall time of a simulate run with record=False (the forward must be
unchanged by the record feature) and, for scale, record=True plus its
reverse. Best of N repeats, one JSON line per arm; run it on two trees (or
two builds) and compare the record=False lines.

    python benchmarks/bench_record_off.py [--n 100000] [--steps 10000] [--repeats 5] [--gpu]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import eagle
import eagle.diff as eagle_diff
import hawk
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel(steps=1)
def bench_osc(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
              x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -x - c * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def _arm(xp, n, steps, record, repeats):
    dt = 1.0 / steps
    tf = xp.asarray(np.linspace(0.5, 1.0, n))
    x, v, t = xp.ones(n), xp.zeros(n), xp.zeros(n)
    kw = {"record": True} if record else {}
    sim = eagle.simulation(bench_osc, dt=dt, c=0.1, t_final=tf, x=x, v=v, t=t,
                           max_steps=steps, **kw)
    best = float("inf")
    for _ in range(repeats):
        x[...], v[...], t[...] = 1.0, 0.0, 0.0
        sim.reset()
        t0 = time.perf_counter()
        sim.run()
        if record:
            eagle_diff.reverse(sim.record, {"x": xp.ones(n)})
        if xp is not np:
            xp.cuda.Device().synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--off-only", action="store_true",
                    help="time only record=False (for a tree without record=)")
    args = ap.parse_args()
    if args.gpu:
        import cupy as xp
    else:
        xp = np
    for record in ((False,) if args.off_only else (False, True)):
        wall = _arm(xp, args.n, args.steps, record, args.repeats)
        print(json.dumps({"backend": "gpu" if args.gpu else "cpu", "record": record,
                          "n": args.n, "steps": args.steps, "best_s": wall,
                          "eagle": getattr(eagle, "__version__", "?")}))


if __name__ == "__main__":
    main()
