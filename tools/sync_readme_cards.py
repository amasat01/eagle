#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regenerate README.md's Performance table from the committed perf_card cards.

The headline table (N = 1,000,000, five arms, both distributions) used to be
hand-copied from ``card_quadro-p2000.md``; it now lives between
``<!-- cards:begin -->`` / ``<!-- cards:end -->`` markers in README.md, and
this script -- read-only on the cards -- is what writes it, together with the
"Measured on" list right after it (every device with a committed
``benchmarks/perf_card/`` card, linked to its docs anchor).

Run it after a card is re-run (same device: its rows change) or a new
``perf_card`` device is added (it gets its own "Measured on" line, with no
other edit to this file). ``tests/test_readme_cards.py`` fails if the
committed block has drifted from what this script renders today -- re-run it
and commit the diff instead of hand-editing either.

Usage::

    python tools/sync_readme_cards.py           # write README.md in place
    python tools/sync_readme_cards.py --check   # exit 1 on a diff, write nothing
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
for _d in ("benchmarks", pathlib.Path("docs") / "_tools"):
    _p = str(_REPO / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _card_common as cc    # noqa: E402
import gen_perf_pages as gp  # noqa: E402

README = _REPO / "README.md"
BEGIN, END = "<!-- cards:begin -->", "<!-- cards:end -->"
REFERENCE_CARD = _REPO / "benchmarks" / "perf_card" / "card_quadro-p2000.json"
_DOCS_BASE = "https://amasat01.github.io/eagle/content/performance.html"

#: The five arms and the N = 1,000,000 / max_steps = 1000 cell chosen
#: for the headline table (the three graph arms, the masked CuPy arm, the CPU
#: arm) -- kept in lockstep with
#: ``python/tests/test_perf_card_table.py``'s ``_README_ROWS``.
ARMS = ("eagle_graph", "eagle_graph_compact", "eagle_graph_reorder",
        "cupy_masked", "cpu_openmp")
DISTRIBUTIONS = ("spread", "uniform")
N, MAX_STEPS = 1_000_000, 1000

_DIST_LABEL = {
    "spread": f"Spread (log-uniform stop steps, up to {MAX_STEPS})",
    "uniform": f"Uniform (every sample runs {MAX_STEPS} steps)",
}


def _wall_cell(w: dict) -> str:
    """The trimmed mean with the range of the kept runs, else median (IQR)."""
    if "mean" in w:
        return f"{cc.fmt_time(w['mean'])} ({cc.fmt_time(w['lo'])}\u2013{cc.fmt_time(w['hi'])})"
    return f"{cc.fmt_time(w['median'])} ({cc.fmt_time(w['iqr'])})"


def render_table(card: dict) -> str:
    labels = card["arms"]
    by_key = {(r["distribution"], r["max_steps"], r["n"], r["arm"]): r
              for r in card["results"]}
    trimmed = "mean" in by_key[(DISTRIBUTIONS[0], MAX_STEPS, N, ARMS[0])]["wall_s"]
    spread = "range" if trimmed else "IQR"
    lines = [f"| Distribution (N = 1,000,000) | Arm | wall ({spread}) | sample·steps/s "
             "| useful FLOP/s (% of peak) |",
             "|---|---|---:|---:|---:|"]
    for dist in DISTRIBUTIONS:
        for arm in ARMS:
            r = by_key[(dist, MAX_STEPS, N, arm)]
            w = r["wall_s"]
            lines.append(
                f"| {_DIST_LABEL[dist]} | {labels[arm]} "
                f"| {_wall_cell(w)} "
                f"| {cc.fmt_rate(r['sample_steps_per_s'])} "
                f"| {cc.fmt_rate(r['useful_flops_per_s'])} "
                f"({cc.fmt_pct(r['fraction_of_peak_fp64'])}) |")
    return "\n".join(lines)


def render_measured_on(eagle_root: pathlib.Path) -> str:
    lines = ["Measured on:", ""]
    for key in ("perf_card_gpu", "perf_card_cpu"):
        spec = next(f for f in gp.FAMILIES if f["key"] == key)
        for e in gp.load_family(eagle_root, spec):
            # Sphinx/docutils normalizes a MyST target label's underscores to
            # hyphens when it emits the built page's HTML id (gen_perf_pages.py
            # writes `({spec['key']}-{slug})=` verbatim; the internal
            # cross-reference MyST rewrites to match is not available to this
            # literal external link, so it must already be the normalized
            # form -- else the link 404s on the anchor once built).
            anchor = f"{spec['key']}-{e['slug']}".replace("_", "-")
            lines.append(f"- [{e['name']}]({_DOCS_BASE}#{anchor})")
    return "\n".join(lines)


def render_block(eagle_root: pathlib.Path) -> str:
    card = json.loads(REFERENCE_CARD.read_text())
    return render_table(card) + "\n\n" + render_measured_on(eagle_root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                     help="exit 1 on a diff instead of writing README.md")
    args = ap.parse_args(argv)

    text = README.read_text()
    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END, start)
    rendered = render_block(_REPO)
    new_text = f"{text[:start]}\n{rendered}\n{text[end:]}"

    if args.check:
        if new_text != text:
            print("README.md's cards block is stale; run tools/sync_readme_cards.py")
            return 1
        print("README.md's cards block is up to date")
        return 0
    README.write_text(new_text)
    print("wrote README.md's cards block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
