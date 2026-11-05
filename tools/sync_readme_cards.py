#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regenerate README.md's Performance table from the committed perf_card cards.

A second block, between ``<!-- cpu-cards:begin -->`` / ``<!-- cpu-cards:end -->``
right after it, shows each CPU card (every ``cpu_card_*.json``) the same way.

The headline table (N = 1,000,000, five arms, both distributions) used to be
hand-copied from ``card_nvidia-h100-80gb-hbm3.md``; it now lives between
``<!-- cards:begin -->`` / ``<!-- cards:end -->`` markers in README.md, and
this script -- read-only on the cards -- is what writes it, together with the
"Measured on" list right after it (every device with a committed
``benchmarks/perf_card/`` card, linked to its docs anchor).

Run it after a card is re-run (same device: its rows change) or a new
``perf_card`` device is added (it gets its own "Measured on" line, with no
other edit to this file). ``tests/test_readme_cards.py`` fails if the
committed block has drifted from what this script renders today -- re-run it
and commit the diff instead of hand-editing either.

The README headline (between ``<!-- hero:begin -->`` / ``<!-- hero:end -->``)
is the configured headline device's card at its larger N with the smaller N
beside it (``prose_claims.HEADLINE`` is the one line to change). Every other
measured number in the README, the performance docs and the tutorials is a
claim span re-rendered by ``tools/prose_claims.py`` (its docstring states the
one rounding rule and the span syntax); ``PROSE_FILES`` lists those pages.

When the headline device has no committed card this script stops with an
error naming the missing file and writes nothing; it never falls back to
another device. ``--without-headline`` regenerates everything except the
headline block.

Usage::

    python tools/sync_readme_cards.py           # write the pages in place
    python tools/sync_readme_cards.py --check   # write nothing; exit 1 on a diff, 3 on a missing card
    python tools/sync_readme_cards.py --without-headline   # leave the hero block alone
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
_t = str(_REPO / "tools")
if _t not in sys.path:
    sys.path.insert(0, _t)
import _card_common as cc    # noqa: E402
import gen_perf_pages as gp  # noqa: E402
import prose_claims as pcl   # noqa: E402

README = _REPO / "README.md"
BEGIN, END = "<!-- cards:begin -->", "<!-- cards:end -->"
CPU_BEGIN, CPU_END = "<!-- cpu-cards:begin -->", "<!-- cpu-cards:end -->"
HERO_BEGIN, HERO_END = "<!-- hero:begin -->", "<!-- hero:end -->"
#: Pages whose measured numbers are claim spans (README.md also carries the
#: generated blocks above). Notebooks: their markdown cells.
PROSE_FILES = (
    "README.md",
    "docs/content/performance/reading.md",
    "docs/content/performance/mechanism.md",
)
REFERENCE_CARD = _REPO / "benchmarks" / "perf_card" / "card_nvidia-h100-nvl.json"
_DOCS_BASE = "https://amasat01.github.io/eagle/content/performance.html"
NO_CARD = "No committed ensemble card yet."

#: The five GPU arms and the N = 1,000,000 cells chosen for the headline table.
ARMS = ("eagle_graph", "eagle_graph_persist", "eagle_simulate", "cupy_masked", "warp_kernel")
DISTRIBUTIONS = ("spread", "uniform")
N = 1_000_000

_DIST_LABEL = {
    "spread": "Spread (each sample stops at its own time)",
    "uniform": "Uniform (every sample runs the same orbit)",
}


def _wall_cell(w: dict) -> str:
    """The trimmed mean with the range of the kept runs, else median (IQR)."""
    if "mean" in w:
        return f"{cc.fmt_time(w['mean'])} ({cc.fmt_time(w['lo'])}\u2013{cc.fmt_time(w['hi'])})"
    return f"{cc.fmt_time(w['median'])} ({cc.fmt_time(w['iqr'])})"


def _row_cells(row):
    """(wall, attempt-samples/s, efficiency) cells of one result row, or its label when it has no time."""
    label = None if row is None else pcl.row_label(row)
    if row is None or label:
        return label or "not run", "–", "–"
    return (_wall_cell(row["wall_s"]), cc.fmt_rate_si(row["attempt_samples_per_s"]),
            row.get("eff_sol_note") or ("–" if row.get("eff_sol") is None else pcl.fmt_eff(row["eff_sol"])))


def render_table(card: dict) -> str:
    labels = card["arms"]
    by_key = {(r["distribution"], r["n"], r["arm"]): r for r in card["results"]}
    lines = [f"| Distribution (N = {N:,}) | Arm | wall (IQR) | attempts·samples/s | % of the device's top FP64 rate |",
             "|---|---|---:|---:|---:|"]
    for dist in DISTRIBUTIONS:
        for arm in ARMS:
            wall, rate, eff = _row_cells(by_key.get((dist, N, arm)))
            lines.append(f"| {_DIST_LABEL[dist]} | {labels.get(arm, arm)} | {wall} | {rate} | {eff} |")
    return "\n".join(lines)


#: The CPU block's arms: the same five tools the landing page charts (every
#: other arm lives on the full card). Kept in lockstep with
#: ``python/tests/test_perf_card_table.py``.
CPU_ARMS = ("eagle_cpu8", "eagle_cpu1", "numba_prange", "jax_vmap")
CPU_BASE_ARM = "eagle_cpu8"
CPU_SPEC_KEY = "perf_card_cpu"


def render_cpu_table(card: dict) -> str:
    by_key = {(r["distribution"], r["n"], r["arm"]): r for r in card["results"]}
    labels = card["arms"]
    lines = [f"| Arm (N = {N:,}) | spread wall | uniform wall | vs {labels.get(CPU_BASE_ARM, CPU_BASE_ARM)} (spread) |",
             "|---|---:|---:|---:|"]
    base = by_key.get(("spread", N, CPU_BASE_ARM))
    for arm in CPU_ARMS:
        sp, un = by_key.get(("spread", N, arm)), by_key.get(("uniform", N, arm))
        ratio = "–" if not (sp and base and not pcl.row_label(sp) and not pcl.row_label(base)) else \
            pcl.fmt_ratio(cc.center(sp["wall_s"]) / cc.center(base["wall_s"]))
        lines.append(f"| {labels.get(arm, arm)} | {_row_cells(sp)[0]} | {_row_cells(un)[0]} | {ratio} |")
    return "\n".join(lines)


def render_cpu_heading(card: dict, path: pathlib.Path) -> str:
    return (f"**On a CPU alone: {cc.cpu_display_name(path, card['cpu'])}, {card['logical_cpus']} logical CPUs, "
            f"card of {card['generated_utc'][:10]}**")


def render_cpu_block(eagle_root: pathlib.Path) -> str:
    spec = next(f for f in gp.FAMILIES if f["key"] == CPU_SPEC_KEY)
    parts = []
    for e in gp.load_family(eagle_root, spec):
        parts.append(render_cpu_heading(e["card"], e["path"]) + "\n\n" + render_cpu_table(e["card"]))
    return "\n\n".join(parts) or NO_CARD


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
    if not REFERENCE_CARD.is_file():
        return NO_CARD
    card = json.loads(REFERENCE_CARD.read_text())
    if card.get("device_fma_per_s"):
        card = cc.rescore_card(card)       # eff_sol derived from the raw fields; a value over 1 reads pending
    lead = f"**{card['device']['name']}, card of {card['generated_utc'][:10]}**"
    return lead + "\n\n" + render_table(card) + "\n\n" + render_measured_on(eagle_root)


def render_hero_block(eagle_root: pathlib.Path, headline=None) -> str:
    """The headline strip; ``ClaimError`` if the headline device has no card."""
    return pcl.render_hero(pcl.headline_facts(pcl.Cards(eagle_root, headline)))


def process_file(rel: str, eagle_root: pathlib.Path, cards) -> str:
    """One prose page's text with every claim span re-rendered."""
    path = eagle_root / rel
    text = path.read_text()
    if rel.endswith(".ipynb"):
        nb = json.loads(text)
        ctx_text = "\n@@CELL@@\n".join("".join(c["source"]) for c in nb["cells"]
                                       if c["cell_type"] == "markdown")
        parts = pcl.process(ctx_text, cards, rel).split("\n@@CELL@@\n")
        it = iter(parts)
        for c in nb["cells"]:
            if c["cell_type"] == "markdown":
                c["source"] = next(it).splitlines(keepends=True)
        return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    return pcl.process(text, cards, rel)


def main(argv=None):
    """Exit 0 fresh, 1 stale, 3 a card or cell a claim cites is missing, 2 usage."""
    try:
        return _main(argv)
    except pcl.ClaimError as exc:
        print(f"missing card or cell: {exc}", file=sys.stderr)
        return pcl.EXIT_MISSING_CARD


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                     help="exit 1 on a diff instead of writing")
    ap.add_argument("--without-headline", action="store_true",
                     help="regenerate everything except the hero block")
    ap.add_argument("--root", type=pathlib.Path, default=_REPO, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = args.root
    cards = pcl.Cards(root)

    readme = (root / "README.md").read_text()
    start = readme.index(BEGIN) + len(BEGIN)
    end = readme.index(END, start)
    new_readme = f"{readme[:start]}\n{render_block(root)}\n{readme[end:]}"
    cstart = new_readme.index(CPU_BEGIN) + len(CPU_BEGIN)
    cend = new_readme.index(CPU_END, cstart)
    new_readme = f"{new_readme[:cstart]}\n{render_cpu_block(root)}\n{new_readme[cend:]}"
    if not args.without_headline:
        try:
            hero = render_hero_block(root)
        except pcl.ClaimError as exc:
            print(f"headline card missing, nothing written: {exc}", file=sys.stderr)
            return pcl.EXIT_MISSING_CARD
        hstart = new_readme.index(HERO_BEGIN) + len(HERO_BEGIN)
        hend = new_readme.index(HERO_END, hstart)
        new_readme = f"{new_readme[:hstart]}\n{hero}\n{new_readme[hend:]}"

    outputs = {}
    for rel in PROSE_FILES:
        if rel == "README.md":
            outputs[rel] = pcl.process(new_readme, cards, rel)
        else:
            outputs[rel] = process_file(rel, root, cards)

    stale = [rel for rel, text in outputs.items() if text != (root / rel).read_text()]
    if args.check:
        if stale:
            print("stale (run tools/sync_readme_cards.py): " + ", ".join(stale))
            return pcl.EXIT_STALE
        print("README.md's cards blocks and every prose claim are up to date")
        return 0
    for rel in stale:
        (root / rel).write_text(outputs[rel])
    print("wrote " + (", ".join(stale) if stale else "nothing (already up to date)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
