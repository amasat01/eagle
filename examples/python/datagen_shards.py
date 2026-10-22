# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Simulated training data: a batch of thrown balls with air drag, and sharded generation.

``simulate`` runs the flight of every ball in a batch to its landing and returns the
landing distance, which makes a training target. The same call serves one process or
many: ``main`` makes one shard of a dataset, and one process per GPU each make their own.

    for r in 0 1; do
        CUDA_VISIBLE_DEVICES=$r python examples/python/datagen_shards.py --rank $r --world 2 --out shards &
    done; wait

With no GPU the processes run on the CPU. Needs hawk next to eagle.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

import hawk
from hawk import Mutable, Param, Scalar, Terminated
from hawk.math import sqrt

import eagle
import eagle.plan

DT, G, MAX_STEPS = 0.02, 9.81, 600


@hawk.kernel(steps=1)
def flight(dt: Param, g: Param, cd: Scalar, terminated: Terminated,
           x: Mutable[Scalar], z: Mutable[Scalar], vx: Mutable[Scalar], vz: Mutable[Scalar]):
    s = sqrt(vx * vx + vz * vz)
    vxn = vx - dt * cd * s * vx
    vzn = vz - dt * (g + cd * s * vz)
    x = x + dt * vxn
    zn = z + dt * vzn
    z = zn
    vx = vxn
    vz = vzn
    terminated = zn < 0.0


def array_module():
    """CuPy when a GPU is visible, else NumPy: where the planes live."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:  # no CuPy, or no CUDA here
        pass
    return np


_plans = {}


def simulate(speed, angle, cd, xp):
    """Landing distance of each ball thrown at ``speed`` (m/s) and ``angle`` (rad) with drag ``cd``.

    The inputs are ``xp`` arrays (NumPy or CuPy); the result is a new ``xp`` array."""
    target = "cuda" if xp is not np else "host"
    if target not in _plans:
        (_plans[target],) = eagle.plan.auto([flight], targets=(target,))
    n = speed.shape[0]
    planes = dict(x=xp.zeros(n), z=xp.zeros(n), vx=speed * xp.cos(angle), vz=speed * xp.sin(angle))
    eagle.run_until_done(_plans[target], max_steps=MAX_STEPS, dt=DT, g=G, cd=xp.asarray(cd),
                         terminated=xp.zeros(n, bool), **planes)
    return planes["x"]


def make_shard(n, seed, xp):
    """``n`` random throws and their landing distances, as NumPy arrays."""
    rng = np.random.default_rng(seed)
    speed, angle, cd = rng.uniform(5, 30, n), rng.uniform(0.1, 1.4, n), rng.uniform(0.0, 0.05, n)
    out = simulate(xp.asarray(speed), xp.asarray(angle), xp.asarray(cd), xp)
    return dict(speed=speed, angle=angle, cd=cd, range=np.asarray(out.get() if xp is not np else out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1, help="how many processes share the dataset")
    ap.add_argument("--n", type=int, default=8192, help="samples in the whole dataset")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("shards"))
    a = ap.parse_args()
    xp = array_module()
    seed = np.random.SeedSequence(2026).spawn(a.world)[a.rank]  # one independent stream per rank
    shard = make_shard(a.n // a.world, seed, xp)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / f"shard_{a.rank}.npz", **shard)
    print(f"rank {a.rank}/{a.world}: {len(shard['range'])} samples on {'GPU' if xp is not np else 'CPU'}")


if __name__ == "__main__":
    main()
