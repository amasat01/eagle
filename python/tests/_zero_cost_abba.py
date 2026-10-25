# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Per-call ``build()`` and ``launch()`` of graphs that cross no host/device boundary, this
tree against a reference tree, interleaved A B B A in fresh process pairs.

    ABBA_TREES='{"A": "<reference>/python", "B": "<this>/python"}' python _zero_cost_abba.py N ROUNDS PAIRS

A process carries a bias of its own (allocator layout; it shows for the same tree against
itself), so the verdict is the median of the per-pair B - A differences against 2 x the MAD of
those differences. Prints one json line per phase."""

import json
import os
import subprocess
import sys
import time

import numpy as np


def _graphs():
    import eagle
    from eagle.graph import opaque

    def chain(xp):
        a, b = xp.zeros(256), xp.zeros(256)
        g = eagle.Graph()
        for i in range(4):
            x, y = (a, b) if i % 2 == 0 else (b, a)
            g.add_node(opaque(lambda x=x, y=y: xp.add(x, 1.0, out=y), reads=[x], writes=[y], name=f"n{i}"))
        return g

    legs = [("cpu", lambda: chain(np), lambda: None)]
    try:
        import cupy
        cupy.zeros(1)
        legs.insert(0, ("gpu", lambda: chain(cupy), cupy.cuda.Device().synchronize))
    except Exception:
        pass
    return legs


def _child():
    legs = _graphs()

    def run(n):
        out = {}
        for tag, make, sync in legs:
            built, launched = [], []
            for _ in range(n):
                g = make()
                t0 = time.perf_counter()
                g.build()
                t1 = time.perf_counter()
                g.launch()
                sync()
                built.append(t1 - t0)
                launched.append(time.perf_counter() - t1)
            out[tag + "_build"], out[tag + "_launch"] = built, launched
        return out

    run(5)
    print("ready", flush=True)
    for line in sys.stdin:
        if not line.startswith("run"):
            break
        print(json.dumps(run(int(line.split()[1]))), flush=True)


def _pair(trees, n, rounds):
    procs = {}
    for tag, tree in trees.items():
        env = dict(os.environ, PYTHONPATH=tree + os.pathsep + os.environ.get("PYTHONPATH", ""))
        procs[tag] = subprocess.Popen([sys.executable, "-u", __file__, "--child"], env=env, cwd="/tmp",
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for p in procs.values():
        p.stdout.readline()
    blocks = {"A": [], "B": []}
    for _ in range(rounds):
        for tag in ("A", "B", "B", "A"):
            p = procs[tag]
            p.stdin.write(f"run {n}\n")
            p.stdin.flush()
            blocks[tag].append({k: float(np.median(v)) * 1e6 for k, v in json.loads(p.stdout.readline()).items()})
    for p in procs.values():
        p.stdin.write("q\n")
        p.stdin.flush()
        p.wait()
        p.stdin.close()
        p.stdout.close()
    return {t: {k: float(np.median([b[k] for b in bs])) for k in bs[0]} for t, bs in blocks.items()}


def measure(trees, n=200, rounds=3, pairs=12):
    """``{phase: {A_us, B_us, diff_us, mad_us, ok}}``; ``ok`` is |median diff| <= 2 x MAD."""
    runs = [_pair(trees, n, rounds) for _ in range(pairs)]
    out = {}
    for k in runs[0]["A"]:
        a, b = np.array([r["A"][k] for r in runs]), np.array([r["B"][k] for r in runs])
        d = b - a
        med = float(np.median(d))
        mad = float(np.median(abs(d - med)))
        out[k] = dict(A_us=float(np.median(a)), B_us=float(np.median(b)), diff_us=med, mad_us=mad,
                      ok=bool(abs(med) <= 2 * mad))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        _child()
    else:
        for phase, row in measure(json.loads(os.environ["ABBA_TREES"]), *map(int, sys.argv[1:4])).items():
            print(json.dumps({phase: row}))
