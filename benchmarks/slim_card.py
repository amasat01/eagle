#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Keep in git only the part of a performance card that the docs, the site and the tests read.

A card run writes every raw row (per-sample timings, clocks, counters). The pages
and the site read the summaries, the fits and the metadata; the rest is published
as a release asset, not committed. After a card run::

    python benchmarks/slim_card.py --raw-out <dir> benchmarks/perf_card/card_<slug>.json

copies the full card to ``<dir>/<parent>/<name>`` and rewrites the committed card
in place with ``KEEP`` applied and ``slim_of_md5`` (the md5 of the full file) added.
Tests that need the raw rows read them from ``$EAGLE_CARD_RAW_DIR`` (same layout)
through ``raw_card`` and skip, by name, when it is not set. When it is set (CI), a missing
or mismatched file fails the test by name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

#: The paths kept: ``True`` keeps the whole subtree; a dict keeps those keys (applied
#: to every element of a list); ``"*"`` stands for any key (arm names).
KEEP = {
    "arm_labels": {"*": True}, "arms": True, "arms_not_run": True,
    "code": {"arms": {"*": {"lines": True}}},
    "code_lines": {"arm_blocks": {"*": True}, "arms": {"*": True}},
    "compaction_sweep": {"rows": {"distribution": True, "k": True, "max_steps": True, "n": True,
                                  "variant": True, "wall_s": {"mean": True}}},
    "compile_time": {"none": True, "shared": True,
                     "rows": {"arm": True, "arms": True, "key": True,
                              "cold_s": {"median": True}, "warm_s": {"median": True}}},
    "configs": {"distribution": True, "max_steps": True},
    "cpu": {"cores": True, "logical_cpus": True, "model": True}, "cpu_slug": True,
    "device": {"compute_capability": True, "cuda_driver_api": True, "cuda_runtime": True,
               "driver_version": True, "name": True, "nvidia_smi": {"clocks_max_sm": True},
               "sm_count": True},
    "device_slug": True, "fit": {"arms": {"*": True}},
    "fixed_overhead": {"rows": {"arm": True, "distribution": True, "max_steps": True, "n": True,
                                "wall_s": {"mean": True}}},
    "generated_utc": True,
    "memory": {"rows": {"arm": True, "device_factor": True, "device_peak_bytes": True,
                        "distribution": True, "max_steps": True, "minimum_bytes": True, "n": True,
                        "peak_bytes": True, "rss_baseline_bytes": True}},
    "method": {"cpu_threads": True, "statistic": True}, "ns": True, "peak": {"fp64_flops": True},
    "platform": True,
    "results": {"arm": True, "cpu_mhz_at_timer_start": True, "distribution": True,
                "fraction_of_peak_dram": True, "fraction_of_peak_fp64": True,
                "issued_bytes_per_s": True, "kernel_only_s": True, "max_steps": True, "n": True,
                "sample_steps_per_s": True, "useful_flops_per_s": True,
                "wall_s": {"hi": True, "iqr": True, "lo": True, "mean": True, "median": True,
                           "samples": True}},
    "schema": True, "script": True, "script_md5": True,
    "software": {"cpu": True, "cupy": True, "jax": True, "numpy": True, "torch": True, "warp": True},
    "toolchain": {"host_compiler": True, "python": True}, "workload": {"problem": True},
}


def _keep(obj, spec):
    if spec is True:
        return obj
    if isinstance(obj, list):
        return [_keep(x, spec) for x in obj]
    if isinstance(obj, dict):
        if "*" in spec:
            return {k: _keep(v, spec["*"]) for k, v in obj.items()}
        return {k: _keep(v, spec[k]) for k, v in obj.items() if k in spec}
    return obj


def slim(card: dict, full_md5: str) -> dict:
    out = _keep(card, KEEP)
    clocks = [m for r in card.get("results", []) for m in r.get("wall_s", {}).get("sm_mhz", [])]
    if clocks:
        out["max_sm_mhz"] = max(clocks)  # the one clock the speed-of-light specification needs
    out["slim_of_md5"] = full_md5
    return out


def raw_card(card_path: pathlib.Path) -> dict | None:
    """The full card behind a committed one, from ``$EAGLE_CARD_RAW_DIR``; None when the
    variable is unset. With it set, a file that is absent or whose md5 is not ``slim_of_md5``
    raises, naming the file."""
    root = os.environ.get("EAGLE_CARD_RAW_DIR")
    if not root:
        return None
    path = pathlib.Path(root) / card_path.parent.name / card_path.name
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the full card is missing from $EAGLE_CARD_RAW_DIR")
    want = json.loads(card_path.read_text()).get("slim_of_md5")
    if want and hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest() != want:
        raise ValueError(f"{path}: not the full card this slim one was cut from")
    return json.loads(path.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-out", type=pathlib.Path, required=True)
    ap.add_argument("cards", nargs="+", type=pathlib.Path)
    a = ap.parse_args(argv)
    for path in a.cards:
        raw = path.read_bytes()
        card = json.loads(raw)
        if "slim_of_md5" in card:
            sys.exit(f"{path}: already slim")
        dest = a.raw_out / path.resolve().parent.name / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        path.write_text(json.dumps(slim(card, hashlib.md5(raw, usedforsecurity=False).hexdigest()), indent=1) + "\n")
        print(f"{path}: {len(raw)} -> {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
