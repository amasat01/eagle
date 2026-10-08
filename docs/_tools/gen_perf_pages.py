#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Generate the performance page's per-family MyST includes from the committed cards.

``benchmarks/{perf_card,rk78_card}/`` each hold a GPU card (``card_<slug>.json``)
and a CPU card (``cpu_card_<slug>.json``); a *family* here is one of those four
(perf GPU, perf CPU, rk78 GPU, rk78 CPU). Dropping a new device's card next to
the existing ones (same prefix, new slug) is enough for it to appear everywhere
this script writes to, with no hand edit: this is what keeps
``docs/content/performance.md`` from drifting when a card is re-run (same
device: its rows replace themselves) or added (a new device: a new section).

For each family this writes ``<out-dir>/<family>.md``, a MyST fragment with:

- a "Measured on" table, one row per device (name, compute capability or CPU
  model, FP64 class if the card records one, date from ``generated_utc``, a
  link to the device's own section);
- one section per device, reference device first then the rest by name, each
  under a stable anchor (``{family}-{slug}``) and including that device's own
  committed ``card_<slug>.md`` table;
- for every device OTHER than the reference one, machine-derived lines only
  (never hand-written prose): the fastest arm at each configuration and N
  (``_card_common.fastest_lines``), and, where the card has both arms,
  ``eagle.simulate`` against its hand-built twin (``eagle_graph_auto`` for the
  RK4 families, ``eagle_graph`` for the RK7(8) ones).

Hooked from ``conf.py`` on Sphinx's ``builder-inited`` event, so ``make html``
(or ``make strict``) regenerates these before the build reads them; also
runnable by hand (see ``main`` below) to inspect the output without a build.

Reads ``_card_common.py`` (stdlib only at import time) for the number
formatting and the "fastest arm" lines, so the wording matches the cards'
own tables exactly. Everything else here is self-contained: this script never
imports a card script (those may need GPU libraries at import time) and never
writes under ``benchmarks/``.
"""

from __future__ import annotations

import pathlib
import re
import sys

DOCS_DIR = pathlib.Path(__file__).resolve().parent.parent
EAGLE_ROOT = DOCS_DIR.parent

_BENCH_DIR = EAGLE_ROOT / "benchmarks"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
import _card_common as cc  # noqa: E402

#: One entry per generated family. ``twin``/``simulate`` name the RK7(8)-or-RK4
#: pair this script reports a ratio between on every non-reference device;
#: CPU cards run neither arm, so they stay unset.
FAMILIES = [
    {"key": "perf_card_gpu", "dir": "benchmarks/perf_card", "prefix": "card_",
     "reference_slug": "quadro-p2000", "heading": "GPU", "has_config": True,
     "simulate": "eagle_simulate", "twin": "eagle_graph_auto"},
    {"key": "perf_card_cpu", "dir": "benchmarks/perf_card", "prefix": "cpu_card_",
     "reference_slug": "intel-xeon-w-2125", "heading": "CPU", "has_config": True,
     "simulate": None, "twin": None},
    {"key": "rk78_card_gpu", "dir": "benchmarks/rk78_card", "prefix": "card_",
     "reference_slug": "quadro-p2000", "heading": "GPU", "has_config": False,
     "simulate": "eagle_simulate", "twin": "eagle_graph"},
    {"key": "rk78_card_cpu", "dir": "benchmarks/rk78_card", "prefix": "cpu_card_",
     "reference_slug": "intel-r-xeon-r-w-2125-cpu-4-00ghz", "heading": "CPU",
     "has_config": False, "simulate": None, "twin": None},
]


# --------------------------------------------------------------------------- #
# Card loading
# --------------------------------------------------------------------------- #
def _slug(card: dict, path: pathlib.Path, prefix: str) -> str:
    """``device_slug``/``cpu_slug`` if the card records one, else the slug the
    filename itself carries (``cpu_card_<slug>.json`` -> ``<slug>``)."""
    return card.get("device_slug") or card.get("cpu_slug") or path.stem[len(prefix):]


def _arm_keys(card: dict) -> list[str]:
    arms = card.get("arms")
    return list(arms.keys()) if isinstance(arms, dict) else list(arms or [])


def _arm_labels(card: dict) -> dict[str, str]:
    arms = card.get("arms")
    if isinstance(arms, dict):
        return dict(arms)
    labels = card.get("arm_labels", {})
    return {a: labels.get(a, a) for a in (arms or [])}


def _clean_cpu_name(model: str) -> str:
    """Drop the ``(R)``/``(TM)`` trademark glyphs lscpu/``/proc/cpuinfo`` carry,
    for a heading; the raw ``model`` string is kept in the table's second
    column, so nothing is lost."""
    return re.sub(r"\s+", " ", model.replace("(R)", "").replace("(TM)", "")).strip()


def _device_name_and_class(card: dict) -> tuple[str, str]:
    """(display name, compute capability or CPU model) of the card's device."""
    if card.get("device"):
        d = card["device"]
        return d["name"], d.get("compute_capability", "–")
    cpu = card.get("cpu")
    model = cpu.get("model", "unknown CPU") if isinstance(cpu, dict) else cpu
    if isinstance(model, str) and model:
        return _clean_cpu_name(model), model
    return "unknown device", "–"


def load_family(eagle_root: pathlib.Path, spec: dict) -> list[dict]:
    """[{"slug", "path", "card", "is_reference"}, ...] for one family, the
    reference device first, the rest sorted by display name."""
    card_dir = eagle_root / spec["dir"]
    if not card_dir.is_dir():
        return []
    entries = []
    for path in sorted(card_dir.glob(f"{spec['prefix']}*.json")):
        card = _load_json(path)
        slug = _slug(card, path, spec["prefix"])
        name, _ = _device_name_and_class(card)
        entries.append({"slug": slug, "path": path, "card": card,
                        "name": name, "is_reference": slug == spec["reference_slug"]})
    entries.sort(key=lambda e: (not e["is_reference"], e["name"].lower()))
    return entries


def _load_json(path: pathlib.Path) -> dict:
    import json

    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Machine-derived lines for a non-reference device (never hand-written prose)
# --------------------------------------------------------------------------- #
def _config_label(cfg: dict) -> str:
    dist, steps = cfg.get("distribution"), cfg.get("max_steps")
    if dist is None:
        return "all configurations"
    return f"{dist} (up to {steps} steps)" if steps is not None else dist


def _fastest_arm_lines(card: dict, has_config: bool) -> list[str]:
    labels = _arm_labels(card)
    results = card.get("results", [])
    lines: list[str] = []
    if has_config and card.get("configs"):
        for cfg in card["configs"]:
            rows = [r for r in results if r.get("distribution") == cfg.get("distribution")
                    and r.get("max_steps") == cfg.get("max_steps")]
            if not rows:
                continue
            lines.append(f"**{_config_label(cfg)}**")
            lines.append("")
            lines += cc.fastest_lines(rows, labels)
            lines.append("")
    elif results:
        lines += cc.fastest_lines(results, labels)
        lines.append("")
    return lines


def _simulate_vs_twin_lines(card: dict, simulate_arm: str | None, twin_arm: str | None) -> list[str]:
    if not simulate_arm or not twin_arm:
        return []
    arms = _arm_keys(card)
    if simulate_arm not in arms or twin_arm not in arms:
        return []

    def key(r):
        return (r.get("distribution"), r.get("max_steps"), r["n"])

    by_arm: dict[str, dict] = {}
    for r in card.get("results", []):
        by_arm.setdefault(r["arm"], {})[key(r)] = r
    rows = []
    for k, r in by_arm.get(simulate_arm, {}).items():
        t = by_arm.get(twin_arm, {}).get(k)
        if t is None:
            continue
        wall = cc.center(r["wall_s"]) / cc.center(t["wall_s"])
        kernel = None
        if r.get("kernel_only_s") and t.get("kernel_only_s"):
            kernel = r["kernel_only_s"] / t["kernel_only_s"]
        rows.append((k, wall, kernel))
    if not rows:
        return []
    lines = [f"`{simulate_arm}` against its hand-built twin `{twin_arm}`:", ""]
    for (dist, steps, n), wall, kernel in sorted(rows, key=lambda x: (str(x[0][0]), x[0][1] or 0, x[0][2])):
        cfg_part = f"{_config_label({'distribution': dist, 'max_steps': steps})}, " if dist else ""
        kernel_part = f", kernel-only {kernel:.3g}×" if kernel is not None else ""
        lines.append(f"- {cfg_part}N = {n:,}: wall {wall:.3g}×{kernel_part} the twin's time")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _measured_on_table(spec: dict, entries: list[dict]) -> list[str]:
    lines = ["### Measured on", "",
             "| device | compute capability / CPU model | date | section |",
             "|---|---|---|---|"]
    for e in entries:
        card = e["card"]
        name, cls = _device_name_and_class(card)
        date = (card.get("generated_utc") or "–").split("T")[0]
        anchor = f"{spec['key']}-{e['slug']}"
        lines.append(f"| {name} | {cls} | {date} | [{name}](#{anchor}) |")
    lines.append("")
    return lines


def render_family(eagle_root: pathlib.Path, spec: dict, entries: list[dict],
                   out_dir: pathlib.Path) -> str:
    lines = [f"<!-- Generated by docs/_tools/gen_perf_pages.py from "
             f"{spec['dir']}/{spec['prefix']}*.json; do not edit. -->", ""]
    if not entries:
        lines.append(f"No committed `{spec['prefix']}*.json` under `{spec['dir']}/` yet.")
        return "\n".join(lines) + "\n"
    lines += _measured_on_table(spec, entries)
    for e in entries:
        anchor = f"{spec['key']}-{e['slug']}"
        # Docs-root-relative ("/..."): a nested {include} would otherwise
        # resolve against the INCLUDING page, not this generated file.
        rel = pathlib.Path("/" + _relpath(e["path"].with_suffix(".md"),
                                          out_dir.parent.parent))
        lines.append(f"({anchor})=")
        lines.append(f"#### {spec['heading']} ({e['name']})")
        lines.append("")
        lines.append(f"```{{include}} {rel.as_posix()}")
        lines.append("```")
        lines.append("")
        if not e["is_reference"]:
            lines += _fastest_arm_lines(e["card"], spec["has_config"])
            lines += _simulate_vs_twin_lines(e["card"], spec["simulate"], spec["twin"])
    return "\n".join(lines).rstrip("\n") + "\n"


def _relpath(path: pathlib.Path, start: pathlib.Path) -> str:
    import os

    return os.path.relpath(path, start)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def generate(eagle_root: pathlib.Path, out_dir: pathlib.Path) -> dict[str, int]:
    """Writes ``<out-dir>/<family>.md`` for every family; returns {family:
    device count} for a caller that wants to log or assert on the result."""
    eagle_root = pathlib.Path(eagle_root)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for spec in FAMILIES:
        entries = load_family(eagle_root, spec)
        text = render_family(eagle_root, spec, entries, out_dir)
        (out_dir / f"{spec['key']}.md").write_text(text)
        counts[spec["key"]] = len(entries)
    return counts


def setup(app):
    """Sphinx extension hook: regenerate before the build reads any page."""

    def _build(app):
        generate(EAGLE_ROOT, DOCS_DIR / "content" / "_generated")

    app.connect("builder-inited", _build)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eagle-root", type=pathlib.Path, default=EAGLE_ROOT)
    ap.add_argument("--out-dir", type=pathlib.Path,
                     default=DOCS_DIR / "content" / "_generated")
    args = ap.parse_args(argv)
    counts = generate(args.eagle_root, args.out_dir)
    for key, n in counts.items():
        print(f"{key}: {n} device(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
