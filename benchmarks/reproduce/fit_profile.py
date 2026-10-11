#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Fit the dispatch cost model's weights to a calibration CSV.

    python fit_profile.py calibration-cpu-*.csv [--json weights.json]

Reads the raw repetitions ``calibrate.py`` writes, takes each case's median
warm wall, and fits the isolated groups (A class slopes, B launch and state
traffic, C split overhead and class sort) by least squares on the relative
error, predicting through ``eagle._core.dispatch_seconds`` -- the same C++ the
policy runs. Groups D (device divergence) and E (paired fused vs split) are
held out and only checked. A fit whose relative RMS exceeds ``--max-residual``
(default 0.20) is refused: the defaults stay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

from eagle import _core

#: Weights fitted, with their starting values taken from the defaults.
FITTED = ("lane_rate", "arith32", "special32", "special64", "memory_rate", "cache_ratio",
          "launch_seconds", "split_seconds")


def read(path):
    """``(header dict, {case_id: (first row, median warm wall)})`` of one CSV."""
    header, lines = {}, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition("=")
            header[key] = value
        else:
            lines.append(line)
    cases = {}
    for row in csv.DictReader(lines):
        if row["warm"] == "1" and row["status"] == "ok":
            cases.setdefault(row["case_id"], ([], row))[0].append(float(row["wall_s"]))
    return header, {cid: (row, statistics.median(walls)) for cid, (walls, row) in cases.items()}


def props_of(header):
    """The device facts the rule reads, from the CSV header (or the host's)."""
    if header["target"] == "cpu":
        return _core.cpu_device_props()
    return {f: float(header[f"dev_{f}"]) for f in ("peak_flops_sp", "peak_bytes_per_s", "fp64_ratio")}


def branches_of(row):
    keys = [k for k in ("cost_b0", "cost_b1", "cost_b2", "cost_b3") if row[k]]
    return [tuple(float(x) for x in row[k].split(";")) for k in keys]


def predict(row, target, props, weights):
    """Predicted wall of one case under ``weights``."""
    weights = _core.default_dispatch_weights(target) if weights is None else weights
    n, steps = int(row["n"]), int(row["steps"])
    branches = branches_of(row)
    kind = row["kind"]
    vectorised = row.get("fused_vectorised", "0") == "1"
    if kind == "sort":
        return n * weights["sort_seconds"]
    if kind in ("unit", "launch"):
        zero = (0.0,) * 6
        fused, _ = _core.dispatch_seconds([branches[0], zero], [n, 0], props, target,
                                          int(row["fused_steps"]), fused_vectorised=vectorised,
                                          weights=weights)
        return fused * steps
    counts = [int(c) for c in row["counts"].split(";")]
    fused, split = _core.dispatch_seconds(branches, counts, props, target, int(row["fused_steps"]),
                                          fused_vectorised=vectorised, remaining_steps=10 ** 12,
                                          weights=weights)
    return {"split": split, "fused": fused}[kind]


def rel_rmse(cases, target, props, weights, groups):
    errs = [(predict(row, target, props, weights) - wall) / wall
            for row, wall in cases.values() if row["group"] in groups]
    return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else 0.0


def nelder_mead(f, x0, steps=300):
    """Minimise ``f`` from ``x0`` (a plain Nelder-Mead; the problem is 7-dimensional)."""
    n = len(x0)
    simplex = [np.array(x0, float)] + [np.array(x0, float) + 0.5 * np.eye(n)[i] for i in range(n)]
    values = [f(x) for x in simplex]
    for _ in range(steps * n):
        order = np.argsort(values)
        simplex, values = [simplex[i] for i in order], [values[i] for i in order]
        centre = np.mean(simplex[:-1], axis=0)
        reflect = centre + (centre - simplex[-1])
        fr = f(reflect)
        if fr < values[0]:
            expand = centre + 2 * (centre - simplex[-1])
            fe = f(expand)
            simplex[-1], values[-1] = (expand, fe) if fe < fr else (reflect, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflect, fr
        else:
            contract = centre + 0.5 * (simplex[-1] - centre)
            fc = f(contract)
            if fc < values[-1]:
                simplex[-1], values[-1] = contract, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (p - simplex[0]) for p in simplex[1:]]
                values = [values[0]] + [f(p) for p in simplex[1:]]
    best = int(np.argmin(values))
    return simplex[best], values[best]


def fit(cases, header, groups="ABC", cache_bytes=0.0):
    """``(weights dict, relative RMS over the fitted groups)``."""
    target = "host" if header["target"] == "cpu" else "device"
    props = props_of(header)
    sorts = [wall / int(row["n"]) for row, wall in cases.values() if row["kind"] == "sort"]
    sort_seconds = math.exp(statistics.fmean(math.log(s) for s in sorts)) if sorts else 0.0
    start = {"lane_rate": 15e9 if target == "host" else 0.5 * props["peak_flops_sp"],
             "arith32": 1.0, "special32": 20.0, "special64": 90.0,
             "memory_rate": 38e9 if target == "host" else props["peak_bytes_per_s"],
             "cache_ratio": 2.0, "launch_seconds": 2e-6 if target == "host" else 5e-6,
             "split_seconds": 0.2e-9}

    def weights_of(x):
        w = {k: math.exp(v) for k, v in zip(FITTED, x)}
        w["sort_seconds"] = sort_seconds
        w["cache_bytes"] = cache_bytes
        return w

    def loss(x):
        w = weights_of(x)
        return rel_rmse(cases, target, props, w, groups) ** 2

    x0 = [math.log(start[k]) for k in FITTED]
    rng = np.random.default_rng(0)
    best_x, best = None, math.inf
    for attempt in range(10):
        x, value = nelder_mead(loss, np.array(x0) + (0 if attempt == 0 else rng.normal(0, 0.4, len(x0))))
        if value < best:
            best_x, best = x, value
    return weights_of(best_x), math.sqrt(best)


def winners(cases, header, weights):
    """Held-out pairs (group E): ``(agree, total)`` of the predicted faster path."""
    target = "host" if header["target"] == "cpu" else "device"
    props = props_of(header)
    agree = total = 0
    ids = list(cases)
    for i, cid in enumerate(ids[:-1]):
        row, wall = cases[cid]
        split_row, split_wall = cases[ids[i + 1]]
        if row["group"] == "E" and row["kind"] == "fused" and split_row["kind"] == "split":
            agree += ((predict(row, target, props, weights) < predict(split_row, target, props, weights))
                      == (wall < split_wall))
            total += 1
    return agree, total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--max-residual", type=float, default=0.20)
    ap.add_argument("--cache-bytes", type=float, default=0.0,
                    help="last-level cache that holds a gathered plane (0: every gather is memory)")
    ap.add_argument("--json", help="write the fitted weights here")
    args = ap.parse_args(argv)
    status, out = 0, {}
    for path in args.csv:
        header, cases = read(path)
        weights, rms = fit(cases, header, cache_bytes=args.cache_bytes)
        held = rel_rmse(cases, "host" if header["target"] == "cpu" else "device",
                        props_of(header), weights, "D")
        agree, total = winners(cases, header, weights)
        verdict = "refused (defaults stay)" if rms > args.max_residual else "accepted"
        print(f"{Path(path).name}: relative RMS {rms:.3f} on A-C ({verdict}); "
              f"divergence group D {held:.3f}; held-out winners {agree}/{total}")
        for key, value in weights.items():
            print(f"  {key} = {value:.4g}")
        out[header["target"]] = weights
        status |= rms > args.max_residual
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
