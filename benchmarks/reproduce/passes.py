#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Run a GPU card one pass at a time, so a card fits sessions with a time limit.

Each invocation runs ONE chunk of one pass with the card script's own pass
function and saves its rows under ``--parts``; a chunk already saved is
skipped, so an interrupted run resumes where it stopped. ``final`` runs the
timing pass and assembles the card with the script's own ``main()`` from the
saved passes (a pass with no saved chunks is skipped, as ``--no-<pass>``
would). The card written is the same schema the monolithic run writes.

Usage::

    passes.py perf|rk78 count nsys|memory|compile     # number of chunks
    passes.py perf|rk78 nsys|memory|compile --index I --parts DIR [--ns ...]
    passes.py perf|rk78 final --parts DIR --out-dir DIR [--ns ...] [--reps R]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile

BENCH = pathlib.Path(__file__).resolve().parent.parent
CHUNK = {"perf": {"compile": 4}, "rk78": {"memory": 5, "compile": 3}}


def _load_card(card):
    sub, name = ("perf_card", "perf_card") if card == "perf" else ("rk78_card", "rk78_card")
    sys.path.insert(0, str(BENCH / sub))
    return __import__(name)


def _chunks(mod, card, what, ns):
    if what == "nsys":
        return len(mod.CONFIGS) if card == "perf" else len(mod.ARMS)
    if what == "memory":
        return len(mod.CONFIGS) if card == "perf" else math.ceil(len(mod.ARMS) / CHUNK[card]["memory"])
    if what == "compile":
        return math.ceil(len(mod._compile_keys(ns)) / CHUNK[card]["compile"])
    raise SystemExit(f"unknown pass {what}")


def _save(parts, name, obj):
    tmp = parts / f".{name}.json.tmp"
    tmp.write_text(json.dumps(obj))
    tmp.replace(parts / f"{name}.json")


def _load(parts, prefix):
    return [json.loads(p.read_text()) for p in sorted(parts.glob(f"{prefix}_*.json"))]


def _check_meta(parts, ns):
    meta = parts / "meta.json"
    if meta.exists():
        saved = json.loads(meta.read_text())["ns"]
        if saved != ns:
            raise SystemExit(f"{parts} holds passes for --ns {saved}, not {ns}; "
                             "use another --parts directory")
    else:
        meta.write_text(json.dumps({"ns": ns}))


def run_chunk(mod, card, what, idx, ns, parts):
    name = f"{what}_{idx:02d}"
    if (parts / f"{name}.json").exists():
        print(f"{card} {name}: already saved, skipped", flush=True)
        return
    _, _, _, bus = mod.cc.device_facts()
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"{card}_{what}_"))
    if card == "perf":
        cfgs = mod.CONFIGS
        if what == "nsys":
            mod.CONFIGS = (cfgs[idx],)
            out, note = mod._nsys_kernel_times(ns, work)
            obj = {"note": note, "out": [[list(k), v] for k, v in (out or {}).items()]}
        elif what == "memory":
            mod.CONFIGS = (cfgs[idx],)
            obj = mod._run_memory_pass(ns, bus)
        else:
            keys = mod._compile_keys(ns)
            lo = idx * CHUNK[card]["compile"]
            mod._compile_keys = lambda ns, arms=None: keys[lo:lo + CHUNK[card]["compile"]]
            obj = mod._run_compile_pass(ns)
    else:
        arms = mod.ARMS
        if what == "nsys":
            out, note = mod._nsys_kernel_times(ns, work, (arms[idx],))
            obj = {"note": note, "out": [[list(k), v] for k, v in (out or {}).items()]}
        elif what == "memory":
            lo = idx * CHUNK[card]["memory"]
            obj = mod._run_memory_pass(ns, bus, arms[lo:lo + CHUNK[card]["memory"]])
        else:
            keys = mod._compile_keys(ns)
            lo = idx * CHUNK[card]["compile"]
            mod._compile_keys = lambda ns: keys[lo:lo + CHUNK[card]["compile"]]
            obj = mod._run_compile_pass(ns)
    _save(parts, name, obj)
    print(f"{card} {name}: saved", flush=True)


def final(mod, card, ns, parts, rest):
    flags = []
    nsys = _load(parts, "nsys")
    if nsys:
        out = {tuple(k): v for part in nsys for k, v in part["out"]}
        notes = list(dict.fromkeys(p["note"] for p in nsys))
        note = "; ".join(notes)
        if card == "perf":
            mod._nsys_kernel_times = lambda ns, work: (out, note)
        else:
            mod._nsys_kernel_times = lambda ns, work, names: (out, note)
    else:
        flags.append("--no-nsys")
    mem = [r for part in _load(parts, "memory") for r in part]
    if mem:
        if card == "perf":
            mod._run_memory_pass = lambda ns, bus_id: mem
        else:
            mod._run_memory_pass = lambda ns, bus_id, names: mem
    else:
        flags.append("--no-memory")
    comp = [r for part in _load(parts, "compile") for r in part]
    if comp:
        mod._run_compile_pass = lambda ns: comp
    else:
        flags.append("--no-compile")
    return mod.main(["--ns", *map(str, ns), *flags, *rest])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("card", choices=["perf", "rk78"])
    ap.add_argument("what", choices=["count", "nsys", "memory", "compile", "final"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--parts", type=pathlib.Path)
    ap.add_argument("--ns", type=int, nargs="+")
    args, rest = ap.parse_known_args(argv)
    mod = _load_card(args.card)
    ns = args.ns or list(mod.NS)
    if args.what == "count":
        if not rest or rest[0] not in ("nsys", "memory", "compile"):
            ap.error("count wants a pass: nsys, memory or compile")
        print(_chunks(mod, args.card, rest[0], ns))
        return 0
    if args.parts is None:
        ap.error("--parts is required")
    args.parts.mkdir(parents=True, exist_ok=True)
    _check_meta(args.parts, ns)
    if args.what == "final":
        return final(mod, args.card, ns, args.parts, rest)
    if args.index is None:
        ap.error("--index is required")
    run_chunk(mod, args.card, args.what, args.index, ns, args.parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
