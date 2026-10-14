#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Copy one session's large-N compare evidence into benchmarks/compare_card/.

usage: ingest_compare.py <session-dir> --slug <device-slug> --kit-commit <sha> [--apply]

Dry by default. The session dir holds exactly one ``compare*/compare.csv``
(the output of ``benchmarks/reproduce/gpu_profile.py --stage compare``). The
CSV is copied byte for byte to ``compare_card/<slug>.csv`` and a sidecar
``<slug>.json`` records where it came from. The slug is the device slug of the
perf cards (``card_<slug>.json``). Refuses a CSV without the expected header
and a session whose eagle/hawk commits cannot be found.

Commits come from the session log's bootstrap summary (``eagle:  <sha> (``),
else ``commits.txt``, else the ``software`` block of a card the same session
produced. ``--kit-commit`` is the profile-kit branch commit the session ran
``gpu_profile.py`` from; its md5 is read from this repository.
Adding a device is a copy: no code change.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DEST = REPO / "benchmarks" / "compare_card"
HEADER = (
    "kernel,n,steps,dist,arm,status,median_ms,best_ms,mode,threads_per_sm_launched,"
    "slot_fill,bitwise_vs_auto,max_diff_vs_auto,sm_mhz,clock_normalised_ms,modes_per_run,"
    "probe_walls,launch_fields,precision,note"
)
SCRIPT = "benchmarks/reproduce/gpu_profile.py"


class IngestError(Exception):
    pass


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _git(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def find_csv(session: pathlib.Path) -> pathlib.Path:
    hits = sorted(session.glob("compare*/compare.csv"))
    if len(hits) != 1:
        raise IngestError(f"{session}: expected one compare*/compare.csv, found {len(hits)}")
    return hits[0]


def session_commits(session: pathlib.Path) -> tuple[dict, str]:
    """({'eagle': sha, 'hawk': sha}, where they were read) or IngestError."""
    pat = re.compile(r"^(eagle|hawk):\s+([0-9a-f]{7,40}) \(")
    for log in sorted(session.glob("*.log")):
        got = {}
        for line in log.read_text(errors="replace").splitlines():
            m = pat.match(line)
            if m:
                got[m.group(1)] = m.group(2)
        if {"eagle", "hawk"} <= got.keys():
            return got, f"session log {log.name}"
    f = session / "commits.txt"
    if f.is_file():
        got = {w[0]: w[1] for w in (ln.split() for ln in f.read_text().splitlines()) if len(w) >= 2}
        if {"eagle", "hawk"} <= got.keys():
            return {k: got[k] for k in ("eagle", "hawk")}, "commits.txt"
    for card in sorted(session.rglob("card_*.json")):
        try:
            sw = json.loads(card.read_text()).get("software", {})
        except ValueError:
            continue
        if sw.get("eagle_commit") and sw.get("hawk_commit"):
            return {"eagle": sw["eagle_commit"], "hawk": sw["hawk_commit"]}, f"card {card.name}"
    raise IngestError(f"{session}: no eagle/hawk commits in a session log, commits.txt or a card")


def script_md5(commit: str) -> str:
    data = _git("show", f"{commit}:{SCRIPT}")
    if data is None:
        raise IngestError(f"{SCRIPT} is missing at kit commit {commit}")
    return md5_bytes(data)


def build(session: pathlib.Path, slug: str, device: str | None, kit_commit: str):
    csv = find_csv(session)
    raw = csv.read_bytes()
    first = raw.split(b"\n", 1)[0].decode().strip()
    if first != HEADER:
        raise IngestError(f"{csv}: unexpected header {first!r}")
    commits, source = session_commits(session)
    full = {}
    for name, sha in commits.items():
        out = _git("rev-parse", "--verify", "-q", sha + "^{commit}") if name == "eagle" else None
        full[name] = out.decode().strip() if out else sha
    kit_full = _git("rev-parse", "--verify", "-q", kit_commit + "^{commit}")
    if not kit_full:
        raise IngestError(f"kit commit {kit_commit} is not a commit of this repository")
    kit_full = kit_full.decode().strip()
    md = script_md5(kit_full)
    side = {
        "schema": "eagle-compare-evidence/1",
        "slug": slug,
        "device": device or slug,
        "session": session.name,
        "eagle_commit": full["eagle"],
        "hawk_commit": full["hawk"],
        "commits_source": source,
        "csv_md5": md5_bytes(raw),
        "kit_branch": "profile-kit",
        "kit_commit": kit_full,
        "gpu_profile_md5": md,
        "generated_utc": datetime.datetime.fromtimestamp(
            csv.stat().st_mtime, datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return csv, raw, side


def device_name(session: pathlib.Path):
    for card in sorted(session.rglob("card_*.json")):
        try:
            return json.loads(card.read_text()).get("device", {}).get("name")
        except ValueError:
            continue
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=pathlib.Path)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--kit-commit", required=True,
                    help="commit of the profile-kit branch the session ran gpu_profile.py from")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    try:
        csv, raw, side = build(a.session.resolve(), a.slug, device_name(a.session), a.kit_commit)
    except IngestError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    rows = raw.decode().count("\n") - 1
    dest = DEST / f"{a.slug}.csv"
    verb = "REPLACES" if dest.exists() else "new"
    print(f"{csv}\n  -> {dest.relative_to(REPO)} [{verb}] rows={rows} eagle={side['eagle_commit'][:10]} "
          f"hawk={side['hawk_commit'][:10]} (from {side['commits_source']}) md5={side['csv_md5']}")
    if not a.apply:
        print("dry run (use --apply)")
        return 0
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv, dest)
    (DEST / f"{a.slug}.json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"copied {dest.name} and {a.slug}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
