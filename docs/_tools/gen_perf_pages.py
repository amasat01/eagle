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
  link to the device's own section), also written alone as ``<family>_devices.md``
  for pages that only list the devices;
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

import html
import pathlib
import re
import shutil
import sys

DOCS_DIR = pathlib.Path(__file__).resolve().parent.parent
EAGLE_ROOT = DOCS_DIR.parent

_BENCH_DIR = EAGLE_ROOT / "benchmarks"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
import _card_common as cc  # noqa: E402

_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import perf_charts as charts  # noqa: E402

#: One entry per generated family. ``twin``/``simulate`` name the RK7(8)-or-RK4
#: pair this script reports a ratio between on every non-reference device;
#: CPU cards run neither arm, so they stay unset.
FAMILIES = [
    {"key": "perf_card_gpu", "page": "rk4", "dir": "benchmarks/perf_card", "prefix": "card_",
     "reference_slug": "nvidia-h100-80gb-hbm3", "heading": "GPU", "has_config": True,
     "simulate": "eagle_simulate", "twin": "eagle_graph_auto"},
    {"key": "perf_card_cpu", "page": "rk4", "dir": "benchmarks/perf_card", "prefix": "cpu_card_",
     "reference_slug": "amd-epyc-9b45", "heading": "CPU", "has_config": True,
     "simulate": None, "twin": None},
    {"key": "rk78_card_gpu", "page": "rk78", "dir": "benchmarks/rk78_card", "prefix": "card_",
     "reference_slug": "nvidia-h100-80gb-hbm3", "heading": "GPU", "has_config": False,
     "simulate": "eagle_simulate", "twin": "eagle_graph"},
    {"key": "rk78_card_cpu", "page": "rk78", "dir": "benchmarks/rk78_card", "prefix": "cpu_card_",
     "reference_slug": "amd-epyc-9b45", "heading": "CPU",
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


def _device_name_and_class(card: dict, path: pathlib.Path | None = None) -> tuple[str, str]:
    """(display name, compute capability or CPU model) of the card's device. A CPU
    card whose file stem is in ``benchmarks/device_names.json`` shows that name."""
    if card.get("device"):
        d = card["device"]
        return d["name"], d.get("compute_capability", "–")
    cpu = card.get("cpu")
    model = cpu.get("model", "unknown CPU") if isinstance(cpu, dict) else cpu
    if isinstance(model, str) and model:
        name = _clean_cpu_name(model)
        return (cc.cpu_display_name(path, name) if path else name), model
    return "unknown device", "–"


def load_family(eagle_root: pathlib.Path, spec: dict) -> list[dict]:
    """[{"slug", "path", "card", "name", "is_reference"}, ...] for one family, the
    default tab's device first (``perf_charts.GPU_PREFERENCE`` /
    ``CPU_PREFERENCE``), the rest alphabetically."""
    card_dir = eagle_root / spec["dir"]
    if not card_dir.is_dir():
        return []
    entries = []
    for path in sorted(card_dir.glob(f"{spec['prefix']}*.json")):
        card = _load_json(path)
        slug = _slug(card, path, spec["prefix"])
        name, _ = _device_name_and_class(card, path)
        entries.append({"slug": slug, "path": path, "card": card,
                        "name": name, "is_reference": slug == spec["reference_slug"]})
    pref = charts.GPU_PREFERENCE if spec["heading"] == "GPU" else charts.CPU_PREFERENCE
    return charts.in_tab_order(entries, pref)


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
def _measured_on_table(entries: list[dict]) -> list[str]:
    lines = ["| device | compute capability / CPU model | date |", "|---|---|---|"]
    for e in entries:
        name, cls = _device_name_and_class(e["card"], e["path"])
        date = (e["card"].get("generated_utc") or "–").split("T")[0]
        lines.append(f"| {name} | {cls} | {date} |")
    lines.append("")
    return lines


def _chart_for(spec: dict, e: dict, cfg: dict | None, name_config: bool = True) -> str | None:
    """One chart's HTML (the card's tools at its largest N), or None."""
    tools = charts.GPU_TOOLS if spec["heading"] == "GPU" else charts.CPU_TOOLS
    chart_bars, n = charts.bars(e["card"], tools, cfg)
    if not chart_bars:
        return None
    title = f"{_config_label(cfg)}, N = {n:,}" if cfg and name_config else f"N = {n:,}"
    return (f'<p class="chart-title">{title}</p>\n'
            + charts.chart_html(chart_bars, e["name"], ""))


def _configs(spec: dict, e: dict) -> list[dict | None]:
    return (e["card"].get("configs") if spec["has_config"] else None) or [None]


def headline_config(spec: dict, e: dict) -> dict | None:
    """The workload the overview shows: the README's, else the card's first."""
    configs = _configs(spec, e)
    return README_CONFIG if README_CONFIG in configs else configs[0]


def _slug_key(cfg: dict | None) -> str:
    return "all" if not cfg else f"{cfg.get('distribution')}-{cfg.get('max_steps')}"


def _raw_html(html_text: str) -> list[str]:
    return ["```{raw} html", *html_text.split("\n"), "```", ""]


def card_overview(spec: dict, e: dict, page: str) -> str:
    """Fragment: the device's headline chart, a one-line description, a link to
    the results page. Carries the anchor external links point at."""
    name, cls = _device_name_and_class(e["card"], e["path"])
    date = (e["card"].get("generated_utc") or "").split("T")[0]
    chart = _chart_for(spec, e, headline_config(spec, e))
    detail = ([f"compute capability {cls}"] if spec["heading"] == "GPU" else []) + (
        [f"card of {date}"] if date else [])
    lines = [f"({spec['key']}-{e['slug']})=",
             f"**{name}**{' (' + ', '.join(detail) + ')' if detail else ''}.", ""]
    if chart:
        lines += _raw_html(chart)
    lines += [f"[Every workload, every arm and every N for {name}](/content/performance/{page}.md)", ""]
    return "\n".join(lines)


#: The driver accounts device memory in pages of about 2 MiB.
DRIVER_PAGE_BYTES = 2 * 2**20
#: A table row's leading N and its device-memory cell ("50 MiB (1.31×)" or
#: "50 MiB (1.31× min.)" -- the host cell carries no factor).
_ROW_N = re.compile(r"\|\s*([\d,]+)\s*\|")
_DEVICE_CELL = re.compile(r"\| ([\d.e+]+) MiB \([\d.e+]+×(?: min\.)?\) \|")
WITHHELD_CELL = "not measured"
#: Marks a device-memory cell borrowed from another card of the family.
BORROWED_MARK = "†"


def _valid_reading(row: dict) -> bool:
    """False when a memory row's device reading is below the workload's
    minimum state, less one driver page: no run fits in less."""
    peak, minimum = row.get("device_peak_bytes"), row.get("minimum_bytes")
    return peak is None or not minimum or peak + DRIVER_PAGE_BYTES >= minimum


def borrowed_readings(e: dict, entries) -> dict:
    """{(arm, n): (peak_bytes, minimum_bytes, [device names])} for every
    reading of `e`'s card that `_valid_reading` rejects and that the family's
    other cards measured validly with ONE value at that (arm, n) across every
    configuration -- an allocator's pool that does not depend on the card.
    A (arm, n) whose other cards disagree is left out (it stays withheld)."""
    bad = {(r["arm"], r["n"]) for r in (e["card"].get("memory") or {}).get("rows", [])
           if not _valid_reading(r)}
    found: dict = {}
    for other in entries:
        if other is e:
            continue
        for r in (other["card"].get("memory") or {}).get("rows", []):
            key = (r["arm"], r["n"])
            if key in bad and r.get("device_peak_bytes") is not None and _valid_reading(r):
                found.setdefault(key, {}).setdefault(
                    (r["device_peak_bytes"], r["minimum_bytes"]), set()).add(other["name"])
    return {key: (*next(iter(vals)), sorted(next(iter(vals.values()))))
            for key, vals in found.items() if len(vals) == 1}


def withhold_impossible_readings(line: str, minimum_by_n: dict, labels: dict | None = None,
                                 borrowed: dict | None = None) -> str:
    """`line` with an impossible device-memory cell replaced: a reading below
    the minimum state for the row's N, less one driver page, missed memory held
    before the baseline. The replacement is the family's other cards' reading
    for the same arm and N, marked `BORROWED_MARK` (`borrowed`, keyed by the
    arm whose label is `labels`[row label]), else `WITHHELD_CELL`. Any other
    line comes back unchanged."""
    n_match = _ROW_N.match(line)
    cell = _DEVICE_CELL.search(line)
    if not n_match or not cell:
        return line
    n = int(n_match.group(1).replace(",", ""))
    minimum = minimum_by_n.get(n)
    if not minimum or float(cell.group(1)) * 2**20 + DRIVER_PAGE_BYTES >= minimum:
        return line
    label = line.split("|")[2].strip()
    hit = (borrowed or {}).get(((labels or {}).get(label), n))
    if hit:
        peak, need, _names = hit
        suffix = " min." if cell.group(0).endswith(" min.) |") else ""
        text = f"{peak / 2**20:.3g} MiB ({peak / need:.3g}×{suffix}){BORROWED_MARK}"
    else:
        text = WITHHELD_CELL
    return line[:cell.start()] + f"| {text} |" + line[cell.end():]


def _withheld_note(borrowed_used: dict, any_withheld: bool) -> list[str]:
    lines = []
    if borrowed_used:
        names = sorted({name for _p, _m, ns in borrowed_used.values() for name in ns})
        lines.append(
            f"{BORROWED_MARK} Measured on {' and '.join(names)}: on this card the reading fell below "
            "the state the workload needs, because the library reserved its memory pool during "
            "the warm-up, before the measurement's baseline (JAX does). The pool's size does not "
            "depend on the card, so the other cards' reading stands in until this card is "
            "measured again with the baseline taken before any library allocates.")
        lines.append("")
    if any_withheld:
        lines.append(
            f"\"{WITHHELD_CELL}\": the reading fell below the state the workload needs (less one "
            "driver page), so it missed memory the library reserved before the measurement's "
            "baseline, and no other card measured that arm at that N.")
        lines.append("")
    return lines


def _table_fragment(spec: dict, e: dict, entries=()) -> str:
    """The card's own table with its headings turned into bold lines (a heading
    cannot sit inside a collapsed block) and its impossible device-memory
    readings replaced (`withhold_impossible_readings`, borrowing from the
    family's other `entries`), then the machine-derived lines."""
    minimum_by_n = {r["n"]: r.get("minimum_bytes")
                    for r in (e["card"].get("memory") or {}).get("rows", [])}
    arms = e["card"].get("arms")
    labels = {label: arm for arm, label in arms.items()} if isinstance(arms, dict) else {}
    borrowed = borrowed_readings(e, entries)
    used, withheld = {}, False
    out, fenced = [], False
    for line in e["path"].with_suffix(".md").read_text().splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        m = None if fenced else re.match(r"#{1,6}\s+(.*)", line)
        if not fenced and not m:
            kept, line = line, withhold_impossible_readings(line, minimum_by_n, labels, borrowed)
            if line != kept:
                n = int(_ROW_N.match(line).group(1).replace(",", ""))
                key = (labels.get(line.split("|")[2].strip()), n)
                if key in borrowed:
                    used[key] = borrowed[key]
                else:
                    withheld = True
        out.append(f"**{m.group(1)}**" if m else line)
    out.append("")
    out += _withheld_note(used, withheld)
    if not e["is_reference"]:
        out += _fastest_arm_lines(e["card"], spec["has_config"])
        out += _simulate_vs_twin_lines(e["card"], spec["simulate"], spec["twin"])
    return "\n".join(out).rstrip("\n") + "\n"


def card_results(spec: dict, e: dict, table_rel: str) -> str:
    """Fragment: one tab per workload configuration with its chart, then the
    detailed table collapsed underneath."""
    configs = [(c, _chart_for(spec, e, c, name_config=False)) for c in _configs(spec, e)]
    configs = [(c, h) for c, h in configs if h]
    lines: list[str] = []
    if len(configs) == 1:
        lines += _raw_html(configs[0][1])
    elif configs:
        lines += ["::::{tab-set}", ":sync-group: workload", ""]
        for cfg, html_text in configs:
            lines += [f":::{{tab-item}} {_config_label(cfg)}", f":sync: {_slug_key(cfg)}", ""]
            lines += _raw_html(html_text)
            lines += [":::"]
        lines += ["::::", ""]
    lines += [":::{dropdown} Table of every arm and every N", "",
              f"```{{include}} {table_rel}", "```", ":::", ""]
    return "\n".join(lines)


def _tab_set(spec: dict, entries: list[dict], fragment_rel) -> list[str]:
    """Device tabs (first = default), synced across pages by slug."""
    lines = ["::::{tab-set}", f":sync-group: {spec['heading'].lower()}", ""]
    for e in entries:
        lines += [f":::{{tab-item}} {e['name']}", f":sync: {e['slug']}", "",
                  f"```{{include}} {fragment_rel(e)}", "```", ":::", ""]
    lines += ["::::", ""]
    return lines


def render_family(spec: dict, entries: list[dict], out_dir: pathlib.Path, view: str) -> str:
    """The family's assembled fragment for ``view`` ("results" or "overview")."""
    lines = [f"<!-- Generated by docs/_tools/gen_perf_pages.py from "
             f"{spec['dir']}/{spec['prefix']}*.json; do not edit. -->", ""]
    if not entries:
        lines.append(f"No committed `{spec['prefix']}*.json` under `{spec['dir']}/` yet.")
        return "\n".join(lines) + "\n"
    root = out_dir.parent.parent
    lines += _tab_set(spec, entries, lambda e: "/" + _relpath(
        out_dir / spec["key"] / f"{e['slug']}_{view}.md", root))
    if view == "results":
        lines += _measured_on_table(entries)
    return "\n".join(lines).rstrip("\n") + "\n"


def _relpath(path: pathlib.Path, start: pathlib.Path) -> str:
    import os

    return os.path.relpath(path, start)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
#: The workload the README charts show: the landing page's own.
README_CONFIG = {"distribution": "spread", "max_steps": 1000}


def write_readme_svgs(eagle_root: pathlib.Path, entries_by_family: dict[str, list[dict]]) -> list[pathlib.Path]:
    """``docs/_static/perf/readme_{gpu,cpu}.svg``: the default tab's chart of the
    RK4 family as a standalone SVG (GitHub runs no script, so the READMEs embed
    this file and link to the page that has every tab)."""
    out_dir = eagle_root / "docs" / "_static" / "perf"
    written = []
    for key, tools, kind in (("perf_card_gpu", charts.GPU_TOOLS, "gpu"),
                              ("perf_card_cpu", charts.CPU_TOOLS, "cpu")):
        entries = entries_by_family.get(key) or []
        if not entries:
            continue
        card = entries[0]["card"]
        cfg = README_CONFIG if README_CONFIG in (card.get("configs") or []) else (card.get("configs") or [None])[0]
        chart_bars, n = charts.bars(card, tools, cfg)
        if not chart_bars:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"RK4 oscillators, {n:,} samples" + (f", {_config_label(cfg)}" if cfg else "")
        path = out_dir / f"readme_{kind}.svg"
        path.write_text(charts.chart_svg(chart_bars, title, entries[0]["name"]))
        written.append(path)
    return written


def write_glimpse(out_dir: pathlib.Path, entries_by_family: dict[str, list[dict]]) -> None:
    """``perf_glimpse.html``: the home page's performance glimpse, the same
    chart the README embeds (the default GPU card), with the numbers in its alt."""
    entries = entries_by_family.get("perf_card_gpu") or []
    target = out_dir / "perf_glimpse.html"
    target.unlink(missing_ok=True)
    if not entries:
        return
    e = entries[0]
    cfg = headline_config(FAMILIES[0], e)
    chart_bars, n = charts.bars(e["card"], charts.GPU_TOOLS, cfg)
    if not chart_bars:
        return
    alt = html.escape(charts.alt_text(chart_bars, e["name"]), quote=True)
    target.write_text(
        '<figure class="perf-glimpse">\n'
        '<a href="content/performance.html"><img src="_static/perf/readme_gpu.svg" '
        f'alt="{alt}" style="max-width:100%;height:auto"></a>\n'
        f'<figcaption>{n:,} samples, each stopping at its own step, on {html.escape(e["name"])}: '
        'wall time per tool, log scale. <a href="content/performance.html">Every device, workload '
        'and arm</a>.</figcaption>\n</figure>\n')


def generate(eagle_root: pathlib.Path, out_dir: pathlib.Path) -> dict[str, int]:
    """Writes ``<out-dir>/<family>.md`` for every family; returns {family:
    device count} for a caller that wants to log or assert on the result."""
    eagle_root = pathlib.Path(eagle_root)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    loaded = {}
    root = out_dir.parent.parent
    for spec in FAMILIES:
        entries = loaded[spec["key"]] = load_family(eagle_root, spec)
        card_dir = out_dir / spec["key"]
        shutil.rmtree(card_dir, ignore_errors=True)   # a removed card leaves no fragment
        card_dir.mkdir(parents=True, exist_ok=True)
        for e in entries:
            table = card_dir / f"{e['slug']}_table.md"
            table.write_text(_table_fragment(spec, e, entries))
            (card_dir / f"{e['slug']}_results.md").write_text(
                card_results(spec, e, "/" + _relpath(table, root)))
            (card_dir / f"{e['slug']}_overview.md").write_text(
                card_overview(spec, e, spec["page"]))
        for view in ("results", "overview"):
            name = spec["key"] if view == "results" else f"{spec['key']}_overview"
            (out_dir / f"{name}.md").write_text(render_family(spec, entries, out_dir, view))
        (out_dir / f"{spec['key']}_devices.md").write_text(
            "\n".join(_measured_on_table(entries)))
        counts[spec["key"]] = len(entries)
    write_readme_svgs(eagle_root, loaded)
    write_glimpse(out_dir, loaded)
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
