#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Fit the dispatch cost model's weights to a calibration CSV.

    python fit_profile.py calibration-cpu-*.csv [--json weights.json]

Reads the raw repetitions ``calibrate.py`` writes, takes each case's median
warm wall, and fits the isolated groups (A class slopes and the 1-chain / 4-chain
pair, B launch and state traffic, C split overhead and class sort) by least
squares on the relative error, predicting through
``eagle._core.dispatch_seconds`` -- the same C++ the policy runs. Groups D
(device divergence) and E (paired fused vs split) are held out and only checked.

A fit is refused (the defaults stay) unless every check passes: relative RMS over
A-C within ``--max-residual`` (default 0.20), every A case within 0.35, the
1-chain / 4-chain ratio within 20 %, and (host) every E pair deciding as the
measurement does with its gain within 0.15. Every weight is held within 4x of its
default (the dependent-op time within [0.1, 0.6] ns, the 32-bit op within
[0.8, 1.25]). CSVs of different host profiles are never merged.
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
          "launch_seconds", "split_seconds", "chain_seconds", "vector_width")
#: Hard bounds on a fitted weight, in the weight's own unit; the others stay within 4x of the default.
BOUNDS = {"chain_seconds": (0.10e-9, 0.60e-9), "arith32": (0.8, 1.25)}
#: Acceptance bars.
MAX_A_CASE, MAX_ILP, MAX_GAIN_ERROR, ENTER_GAIN = 0.35, 0.20, 0.15, 1.15
SCHEMA = 2


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
    """``(rows of 6 counts, 4 chain lengths and, when the cell has one, the plane elements
    written; per-branch vectorised flags)``."""
    keys = [k for k in ("cost_b0", "cost_b1", "cost_b2", "cost_b3") if row[k]]
    cells = [[float(x) for x in row[k].split(";")] for k in keys]
    # a schema-1 cell has the six counts only: no chains, every branch takes the fused flag
    fused = row.get("fused_vectorised", "0") == "1"
    return ([tuple(c[:10]) + tuple(c[11:12]) if len(c) > 6 else tuple(c) + (0.0,) * 4 for c in cells],
            [bool(c[10]) if len(c) > 10 else fused for c in cells])


def predict(row, target, props, weights):
    """Predicted wall of one case under ``weights``."""
    weights = _core.default_dispatch_weights(target) if weights is None else weights
    n, steps = int(row["n"]), int(row["steps"])
    branches, flags = branches_of(row)
    kind = row["kind"]
    vectorised = row.get("fused_vectorised", "0") == "1"
    if kind == "sort":
        return n * weights["sort_seconds"]
    if kind in ("unit", "launch"):
        zero = (0.0,) * 10
        fused, _ = _core.dispatch_seconds([branches[0], zero], [n, 0], props, target,
                                          int(row["fused_steps"]), fused_vectorised=vectorised,
                                          weights=weights, branch_vectorised=[flags[0]] * 2)
        return fused * steps
    counts = [int(c) for c in row["counts"].split(";")]
    fused, split = _core.dispatch_seconds(branches, counts, props, target, int(row["fused_steps"]),
                                          fused_vectorised=vectorised, remaining_steps=10 ** 12,
                                          weights=weights, branch_vectorised=flags)
    return {"split": split, "fused": fused}[kind]


def rel_error(row, wall, target, props, weights):
    return (predict(row, target, props, weights) - wall) / wall


def rel_rmse(cases, target, props, weights, groups):
    errs = [rel_error(row, wall, target, props, weights)
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
    prior = _core.default_dispatch_weights(target)
    start = {"lane_rate": 15e9 if target == "host" else 0.5 * props["peak_flops_sp"],
             "arith32": 1.0, "special32": 20.0, "special64": 90.0,
             "memory_rate": 38e9 if target == "host" else props["peak_bytes_per_s"],
             "cache_ratio": 2.0, "launch_seconds": 2e-6 if target == "host" else 5e-6,
             "split_seconds": 0.2e-9, "chain_seconds": prior["chain_seconds"],
             "vector_width": prior["vector_width"]}
    # the device has no chain term or vector width: they stay at their defaults
    free = [k for k in FITTED if target == "host" or k not in ("chain_seconds", "vector_width")]
    box = {k: BOUNDS.get(k, (prior[k] / 4, prior[k] * 4)) if prior.get(k, 0) > 0 else (0.0, math.inf)
           for k in free}
    for k in ("lane_rate", "memory_rate", "cache_ratio"):      # device priors are derived, not defaults
        if prior[k] <= 0 or target != "host":
            box[k] = (start[k] / 4, start[k] * 4)

    def weights_of(x):
        w = {k: min(max(math.exp(v), box[k][0]), box[k][1]) for k, v in zip(free, x)}
        w.update({k: prior[k] for k in FITTED if k not in free})
        w["sort_seconds"] = sort_seconds
        w["cache_bytes"] = cache_bytes
        return w

    def loss(x):
        w = weights_of(x)
        return rel_rmse(cases, target, props, w, groups) ** 2

    x0 = [math.log(start[k]) for k in free]
    rng = np.random.default_rng(0)
    best_x, best = None, math.inf
    for attempt in range(10):
        x, value = nelder_mead(loss, np.array(x0) + (0 if attempt == 0 else rng.normal(0, 0.4, len(x0))))
        if value < best:
            best_x, best = x, value
    return weights_of(best_x), math.sqrt(best)


def a_cases(cases, target, props, weights):
    """``[(case id, relative error)]`` of the group-A cases."""
    return [(cid, rel_error(row, wall, target, props, weights))
            for cid, (row, wall) in cases.items() if row["group"] == "A"]


def ilp_ratios(cases, target, props, weights):
    """``[(m, predicted, measured)]``: wall of the 1-chain kernel over the 4-chain one, per m."""
    by = {}
    for row, wall in cases.values():
        if row["group"] == "A" and row.get("chains", "0") in ("1", "4"):
            by.setdefault(row["m"], {})[row["chains"]] = (row, wall)
    out = []
    for m, pair in sorted(by.items(), key=lambda kv: int(kv[0])):
        if len(pair) == 2:
            (r1, w1), (r4, w4) = pair["1"], pair["4"]
            out.append((int(m), predict(r1, target, props, weights) / predict(r4, target, props, weights),
                        w1 / w4))
    return out


def e_pairs(cases, target, props, weights):
    """Held-out pairs (group E): ``[(predicted gain, measured gain)]`` of fused time over split time."""
    out = []
    ids = list(cases)
    for cid, nxt in zip(ids, ids[1:]):
        (row, wall), (split_row, split_wall) = cases[cid], cases[nxt]
        if row["group"] == "E" and row["kind"] == "fused" and split_row["kind"] == "split":
            out.append((predict(row, target, props, weights) / predict(split_row, target, props, weights),
                        wall / split_wall))
    return out


def decides_alike(pred, meas):
    return _core.dispatch_decide(pred, "switch") == _core.dispatch_decide(meas, "switch")


def checks(cases, header, weights, rms, max_residual):
    """``[(name, passed, detail)]``: the acceptance rows for these weights."""
    target = "host" if header["target"] == "cpu" else "device"
    props = props_of(header)
    worst_a = max(((abs(e), cid) for cid, e in a_cases(cases, target, props, weights)), default=(0.0, "-"))
    ilp = ilp_ratios(cases, target, props, weights)
    worst_ilp = max((abs(p / m - 1) for _, p, m in ilp), default=0.0)
    out = [("A-C relative RMS", rms <= max_residual, f"{rms:.3f} <= {max_residual}"),
           ("every A case", worst_a[0] <= MAX_A_CASE, f"worst {worst_a[1]} {worst_a[0]:.3f} <= {MAX_A_CASE}"),
           ("1-chain / 4-chain ratio", worst_ilp <= MAX_ILP and bool(ilp),
            "; ".join(f"m={m} pred {p:.2f} meas {q:.2f}" for m, p, q in ilp) + f" (worst {worst_ilp:.2f} <= {MAX_ILP})")]
    if target == "host":
        pairs = e_pairs(cases, target, props, weights)
        agree = sum(decides_alike(p, m) for p, m in pairs)
        worst_gain = max((abs(p - m) for p, m in pairs), default=0.0)
        decisive = [(p, m) for p, m in pairs if abs(m - 1) >= MAX_GAIN_ERROR]
        winners = sum((p > 1) == (m > 1) for p, m in decisive)
        out += [("E decide agreement", agree == len(pairs), f"{agree}/{len(pairs)}"),
                ("E gain error", worst_gain <= MAX_GAIN_ERROR, f"worst {worst_gain:.3f} <= {MAX_GAIN_ERROR}"),
                ("E winners (decisive pairs)", winners == len(decisive), f"{winners}/{len(decisive)}")]
    return out


def host_profile_of(path, header):
    if header.get("schema") != str(SCHEMA) or not header.get("host_profile"):
        raise SystemExit(f"{path}: schema {header.get('schema')} has no host_profile; "
                         f"rerun calibrate.py (schema {SCHEMA})")
    return header["host_profile"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--max-residual", type=float, default=0.20)
    ap.add_argument("--cache-bytes", type=float, default=0.0,
                    help="last-level cache that holds a gathered plane (0: every gather is memory)")
    ap.add_argument("--json", help="write the fitted weights here")
    args = ap.parse_args(argv)
    status, out, profiles = 0, {}, {}
    for path in args.csv:
        header, cases = read(path)
        profile = host_profile_of(path, header)
        if profiles.setdefault(header["target"], (profile, path))[0] != profile:
            first = profiles[header["target"]]
            raise SystemExit(f"refusing to merge {first[1]} ({first[0]}) with {path} ({profile}): "
                             "fit one host profile at a time")
        weights, rms = fit(cases, header, cache_bytes=args.cache_bytes)
        target = "host" if header["target"] == "cpu" else "device"
        held = rel_rmse(cases, target, props_of(header), weights, "D")
        rows = checks(cases, header, weights, rms, args.max_residual)
        ok = all(passed for _, passed, _ in rows)
        print(f"{Path(path).name} [{profile}]: " + ("accepted" if ok else "refused (defaults stay)")
              + f"; divergence group D {held:.3f}")
        for name, passed, detail in rows:
            print(f"  {'ok  ' if passed else 'FAIL'} {name}: {detail}")
        for key, value in weights.items():
            print(f"  {key} = {value:.4g}")
        out[header["target"]] = weights
        status |= not ok
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
