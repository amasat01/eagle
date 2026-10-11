# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Wall-time bar charts from the performance cards: the HTML rows the docs
pages show (styled by ``_static/perf-chart.css``) and the SVG the READMEs embed.

Both draw the same chart: one bar per tool at the card's largest N, log scale,
eagle's bar in the family purple, a colour chip per tool. Stdlib only; the
caller passes cards already loaded as dicts.
"""

from __future__ import annotations

import html
import math
import re

#: Substrings of the lower-cased device name, best first. A device that matches
#: none follows the ones that do, alphabetically. The first card present is the
#: tab shown by default.
GPU_PREFERENCE = ("h100", "a100", "l4", "t4", "p2000")
#: Newest server CPUs first (by name or by model-number family), then
#: workstation parts.
CPU_PREFERENCE = ("sapphire rapids", "platinum 84", "genoa", "epyc 9", "ice lake",
                  "platinum 83", "xeon w-", "threadripper", "ryzen")

#: (arm key, plain tool name, colour variable) per kind of card, in display
#: order. An arm a card did not run is left out of its chart.
GPU_TOOLS = (("eagle_simulate", "eagle.simulate", "--mark-eagle"),
             ("warp_kernel", "NVIDIA Warp", "--tool-warp"),
             ("jax_vmap", "JAX", "--tool-jax"),
             ("cupy_masked", "CuPy", "--tool-cupy"),
             ("torch_masked", "PyTorch", "--tool-pytorch"))
CPU_TOOLS = (("eagle_term8", "eagle, 8 threads", "--mark-eagle"),
             ("eagle_cpu8", "eagle, 8 threads", "--mark-eagle"),
             ("numba_prange", "Numba", "--tool-numba"),
             ("jax_shard8", "JAX", "--tool-jax"),
             ("jax_vmap", "JAX", "--tool-jax"),
             ("torch_masked", "PyTorch", "--tool-pytorch"),
             ("numpy_masked", "NumPy", "--tool-numpy"))

# Light and dark values of the colour variables, for the standalone SVG.
_LIGHT = {"--mark-eagle": "#cf416b", "--tool-warp": "#306101", "--tool-jax": "#1a9a8c",
          "--tool-cupy": "#518d05", "--tool-pytorch": "#ee4c2c", "--tool-numba": "#00a3e0",
          "--tool-numpy": "#4d77cf"}
_DARK = {"--tool-warp": "#88e97e", "--tool-cupy": "#76b900", "--tool-numpy": "#7ea3f2"}


def _rank(name: str, preference) -> int:
    low = re.sub(r"\((r|tm)\)", "", name.lower())
    for i, key in enumerate(preference):
        if key in low:
            return i
    return len(preference)


def in_tab_order(entries, preference):
    """Entries (dicts with a ``name``) sorted by preference, then by name."""
    return sorted(entries, key=lambda e: (_rank(e["name"], preference), e["name"].lower()))


def _time(seconds: float, sig: int = 3) -> str:
    """Wall time with a fixed number of significant figures (7.40 s keeps its zero)."""
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        v = seconds / scale
        if v >= 1:
            before = max(1, math.floor(math.log10(v)) + 1)
            return f"{v:.{max(0, sig - before)}f} {unit}"
    return f"{seconds * 1e9:.{sig}g} ns"


def _center(stat: dict) -> float:
    return stat["mean"] if "mean" in stat else stat["median"]


def bars(card: dict, tools, config: dict | None = None):
    """``(bars, n)``: ``[(label, colour var, wall_s, is_eagle), ...]`` fastest
    first, for the tools the card ran at its largest N and ``config``
    (``{"distribution", "max_steps"}``; ``None`` for a card without configs)."""
    if not card.get("ns"):
        return [], None
    n = max(card["ns"])
    cfg = config or {}
    rows = {}
    for r in card.get("results", []):
        if (r["n"] == n and r.get("distribution") == cfg.get("distribution")
                and r.get("max_steps") == cfg.get("max_steps")):
            rows[r["arm"]] = r
    out, seen = [], set()
    for arm, label, var in tools:
        if arm in rows and label not in seen:
            seen.add(label)
            out.append((label, var, _center(rows[arm]["wall_s"]), arm.startswith("eagle")))
    out.sort(key=lambda b: b[2])
    return out, n


def _positions(chart_bars):
    lo = min(b[2] for b in chart_bars) / 1.6
    hi = max(b[2] for b in chart_bars) * 1.6
    span = math.log10(hi) - math.log10(lo)
    return lo, hi, [100 * (math.log10(b[2]) - math.log10(lo)) / span for b in chart_bars]


def alt_text(chart_bars, where: str) -> str:
    return f"Wall time on {where}: " + ", ".join(f"{b[0]} {_time(b[2])}" for b in chart_bars) + "."


def chart_html(chart_bars, where: str, note: str) -> str:
    """The landing page's bar markup; ``note`` is an HTML sentence after the
    log-scale range."""
    lo, hi, pct = _positions(chart_bars)
    rows = []
    for (label, var, wall, us), p in zip(chart_bars, pct):
        rows.append(f'<div class="row{" us" if us else ""}"><div class="who">'
                    f'<span class="chip" style="background:var({var})"></span>{html.escape(label)}</div>'
                    f'<div class="track"><div class="fill" style="width:{p:.1f}%"></div>'
                    f'<span class="val" style="left:{p:.1f}%">{_time(wall)}</span></div></div>')
    return (f'<div class="chart" role="img" aria-label="{html.escape(alt_text(chart_bars, where))}">\n'
            + "\n".join(rows) + "\n</div>\n"
            f'<p class="axis-note">Log scale, from {_time(lo)} to {_time(hi)}. {note}</p>')


def chart_svg(chart_bars, title: str, where: str) -> str:
    """A standalone SVG of the same chart for the READMEs: no script, light and
    dark by ``prefers-color-scheme``, the numbers in ``<desc>``."""
    lo, hi, pct = _positions(chart_bars)
    w, label_w, track_w, row_h, top = 700, 150, 480, 30, 56
    h = top + row_h * (len(chart_bars) - 1) + 24
    light = "\n".join(f".c{i}{{fill:{_LIGHT[b[1]]}}}" for i, b in enumerate(chart_bars))
    dark = "\n".join(f".c{i}{{fill:{_DARK[b[1]]}}}" for i, b in enumerate(chart_bars) if b[1] in _DARK)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'role="img" aria-labelledby="t d" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
             f'<title id="t">{html.escape(title)}</title>',
             f'<desc id="d">{html.escape(alt_text(chart_bars, where))}</desc>',
             "<style>.ink{fill:#0b0b0b}.sub{fill:#52514e}.bar{fill:rgba(127,126,120,.5)}.us{fill:#7d53dc}",
             light, "@media (prefers-color-scheme: dark){.ink{fill:#f2f2ee}.sub{fill:#c3c2b7}", dark, "}</style>",
             f'<text class="ink" x="0" y="22" font-size="16" font-weight="600">{html.escape(title)}</text>',
             f'<text class="sub" x="0" y="42" font-size="12.5">{html.escape(where)}. Log scale, '
             f'{_time(lo)} to {_time(hi)}.</text>']
    for i, ((label, _var, wall, us), p) in enumerate(zip(chart_bars, pct)):
        y = top + i * row_h
        bw = max(2.0, track_w * p / 100)
        parts.append(f'<circle class="c{i}" cx="5" cy="{y + 10}" r="4.5"/>'
                     f'<text class="ink" x="18" y="{y + 15}" font-size="14">{html.escape(label)}</text>'
                     f'<rect class="{"us" if us else "bar"}" x="{label_w}" y="{y + 3}" width="{bw:.1f}" '
                     f'height="14" rx="3"/>'
                     f'<text class="ink" x="{label_w + bw + 8:.1f}" y="{y + 15}" font-size="13">{_time(wall)}</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)
