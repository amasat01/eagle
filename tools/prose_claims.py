# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Performance numbers in prose, generated from the committed cards.

Every measured number a reader sees in a README, a docs page or a tutorial is
written by this module from ``benchmarks/{perf_card,rk78_card}/*.json``; none
is typed by hand. A page marks each number with a claim span::

    <!--ctx dev=p2000 n=1e6 dist=spread steps=1000-->
    ... takes <!--c wall arm=eagle_simulate-->156 ms<!--/c--> ...

``ctx`` sets defaults for the spans after it (cumulative; a span's own
``key=value`` overrides them). A span names its claim completely: kind, card
family, device, arm, N, distribution and steps. The text between ``c`` and
``/c`` is rewritten from the card; ``--check`` mode only compares.

ROUNDING RULE (the one rule, applied to every claim kind below)
  * times:   3 significant figures, in s / ms / µs (``_card_common.fmt_time``);
  * ratios:  2 significant figures, truncated towards zero (never overstated),
             as ``4.0×`` / ``41×`` / ``0.99×``;
  * memory:  whole MiB;
  * percent, rates: 2 and 3 significant figures.
  The cards' own tables keep the full 3-figure record; prose and headlines use
  this rule, so a ratio has exactly one spelling wherever it appears.

Claim kinds (``kind`` is the first word of a span)
  wall   a wall time          x     a ratio of two walls (num=arm, den_arm=arm)
  spec   a device field of the card (``field=peak.fp64_flops``, ``field=device.name``)
  pctgap how much longer one wall is than another, as a percentage
  mib    device memory        xmem  a ratio of two device-memory peaks
  xmin   factor over the minimum state   pct   a percentage of a card field
  rate   a rate card field    comp  a cold/warm compile time (``which=``)
  cmp   large-N compare evidence (``benchmarks/compare_card/<slug>.csv``): a
        ratio of two arms' walls when ``den_arm`` is given, else one arm's wall;
        selectors ``dev``, ``n``, ``dist``, ``steps``, ``arm`` and optionally
        ``kernel`` (default ``osc``)::

          <!--c cmp dev=h100 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->3.2×<!--/c-->
Selectors: ``card`` (rk4 | rk4cpu | rk78 | rk78cpu), ``dev`` (a device alias
below, or ``headline``), ``arm``, ``n``, ``dist``, ``steps``; ``src``
(results | fixed | sweep) with ``variant``/``k`` for the cadence sweep. A ratio
takes its denominator's selectors as ``den_<selector>``.

A claim whose card is not committed (the headline device before its card is
run) raises ``ClaimError`` naming the missing file; it never falls back to
another device.

Reads the cards only; imports no GPU library.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import math
import pathlib
import re
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO / "benchmarks"))
import _card_common as cc  # noqa: E402

#: The README headline: ONE line to change the device / sizes it quotes.
#: ``ns`` is (the headline N, the N shown beside it).
HEADLINE = {
    "dev": "h100",
    "ns": (1_000_000, 100_000),
    "dist": "spread",
    "steps": 1000,
}

#: Device alias -> glob of the card slug (the file's ``card_<slug>.json``).
DEVICES = {
    "p2000": "quadro-p2000",
    "l4": "nvidia-l4",
    "a100": "nvidia-a100*",
    "t4": "tesla-t4",
    "h100": "nvidia-h100*",
}

#: Card family -> (directory, filename prefix).
FAMILIES = {
    "rk4": ("benchmarks/perf_card", "card_"),
    "rk4cpu": ("benchmarks/perf_card", "cpu_card_"),
    "rk78": ("benchmarks/rk78_card", "card_"),
    "rk78cpu": ("benchmarks/rk78_card", "cpu_card_"),
}

#: Defaults a page starts from (before its first ``ctx``).
BASE_CTX = {
    "card": "rk4",
    "dev": "p2000",
    "n": "1e6",
    "dist": "spread",
    "steps": "1000",
}


class ClaimError(Exception):
    """A claim cannot be rendered (missing card, cell or malformed span)."""


# --------------------------------------------------------------------------- #
# The rounding rule
# --------------------------------------------------------------------------- #
def floor_sig(value, sig=2):
    """`value` truncated towards zero to `sig` significant figures."""
    if value is None or value <= 0:
        return None
    exp = math.floor(math.log10(value))
    quantum = 10.0 ** (exp - sig + 1)
    count = math.floor(value / quantum + 1e-9)
    return round(count * quantum, max(0, sig - 1 - exp))


def fmt_ratio(value):
    """A ratio as ``4.0×`` / ``41×``: floored to 2 significant figures."""
    r = floor_sig(value)
    if r is None:
        return cc.DASH
    exp = math.floor(math.log10(r))
    return f"{r:.{max(0, 1 - exp)}f}×"


def fmt_time(seconds):
    """A time at 3 significant figures (the cards' own format)."""
    return cc.fmt_time(seconds)


def fmt_mib(nbytes):
    """Device memory as a whole number of MiB."""
    return cc.DASH if nbytes is None else f"{round(nbytes / 2**20)} MiB"


def fmt_pct(fraction):
    """A percentage, 2 significant figures (``87%``, ``1.1%``, ``100%``)."""
    if fraction is None:
        return cc.DASH
    return f"{float(f'{100.0 * fraction:.2g}'):g}%"


# --------------------------------------------------------------------------- #
# Card store
# --------------------------------------------------------------------------- #
class Cards:
    """Lazy reader of the committed cards under one eagle root."""

    def __init__(self, root, headline=None):
        self.root = pathlib.Path(root)
        self.headline = dict(headline or HEADLINE)
        self._cache = {}

    def path(self, card, dev):
        if card not in FAMILIES:
            raise ClaimError(
                f"unknown card family {card!r} (one of {sorted(FAMILIES)})"
            )
        if dev == "headline":
            dev = self.headline["dev"]
        pattern = DEVICES.get(dev, dev)
        folder, prefix = FAMILIES[card]
        hits = sorted((self.root / folder).glob(f"{prefix}{pattern}.json"))
        # a GPU glob must not swallow the CPU prefix ("card_" vs "cpu_card_")
        hits = [h for h in hits if fnmatch.fnmatch(h.name[len(prefix) : -5], pattern)]
        if not hits:
            raise ClaimError(
                f"missing card: {folder}/{prefix}{pattern}.json (device {dev!r}, family "
                f"{card!r}); the headline is configured in tools/prose_claims.py HEADLINE "
                "and is never replaced by another device"
            )
        if len(hits) > 1:
            raise ClaimError(
                f"{pattern!r} matches several cards: {[h.name for h in hits]}"
            )
        return hits[0]

    def compare_rows(self, dev):
        """The rows of ``benchmarks/compare_card/<slug>.csv`` for a device alias."""
        if dev == "headline":
            dev = self.headline["dev"]
        pattern = DEVICES.get(dev, dev)
        folder = self.root / "benchmarks" / "compare_card"
        hits = [h for h in sorted(folder.glob("*.csv")) if fnmatch.fnmatch(h.stem, pattern)]
        if not hits:
            raise ClaimError(
                f"missing compare evidence: benchmarks/compare_card/{pattern}.csv (device {dev!r})"
            )
        if len(hits) > 1:
            raise ClaimError(f"{pattern!r} matches several compare files: {[h.name for h in hits]}")
        if hits[0] not in self._cache:
            with open(hits[0], newline="") as fh:
                self._cache[hits[0]] = list(csv.DictReader(fh))
        return self._cache[hits[0]]

    def load(self, card, dev):
        p = self.path(card, dev)
        if p not in self._cache:
            self._cache[p] = json.loads(p.read_text())
        return self._cache[p]


def _num(text):
    return int(float(text))


def _select(rows, sel):
    """Rows whose fields equal the selectors (a field the row lacks is ignored)."""
    out = []
    for r in rows:
        ok = True
        for key, field, conv in (
            ("arm", "arm", str),
            ("n", "n", _num),
            ("dist", "distribution", str),
            ("steps", "max_steps", _num),
        ):
            if key in sel and field in r and r[field] != conv(sel[key]):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _one(rows, sel, what):
    got = _select(rows, sel)
    if len(got) != 1:
        raise ClaimError(f"{what}: {len(got)} rows match {sel}")
    return got[0]


def _resolve(cards, sel):
    """The (card, row) a selector set names, honouring ``src``."""
    card = cards.load(sel["card"], sel["dev"])
    src = sel.get("src", "results")
    if src == "results":
        return card, _one(card["results"], sel, f"{sel['dev']}/{sel['card']} results")
    if src == "fixed":
        return card, _one(card["fixed_overhead"]["rows"], sel, "fixed_overhead")
    if src == "sweep":
        rows = [
            r
            for r in card["compaction_sweep"]["rows"]
            if r["variant"] == sel["variant"] and r["k"] == _num(sel["k"])
        ]
        return card, _one(rows, sel, "compaction_sweep")
    if src == "memory":
        return card, _one(card["memory"]["rows"], sel, "memory")
    raise ClaimError(f"unknown src {src!r}")


def _split(sel):
    """(numerator selectors, denominator selectors) of a ratio span."""
    num = {k: v for k, v in sel.items() if not k.startswith("den_")}
    den = dict(num)
    for k, v in sel.items():
        if k.startswith("den_"):
            den[k[4:]] = v
    return num, den


def _cmp_wall(cards, sel):
    """One compare row's wall in seconds. Reads ``clock_normalised_ms`` when the
    row has it (the wall rescaled to the device's reference SM clock), else
    ``median_ms``. A row whose status is not ``ok`` is refused."""
    want = {
        "kernel": sel.get("kernel", "osc"),
        "n": str(_num(sel["n"])),
        "steps": str(_num(sel["steps"])),
        "dist": sel["dist"],
        "arm": sel["arm"],
    }
    rows = [r for r in cards.compare_rows(sel["dev"]) if all(r[k] == v for k, v in want.items())]
    if len(rows) != 1:
        raise ClaimError(f"compare {sel['dev']}: {len(rows)} rows match {want}")
    row = rows[0]
    if row["status"] != "ok":
        raise ClaimError(f"compare {sel['dev']}: row {want} has status {row['status']!r}, not ok")
    ms = row["clock_normalised_ms"] or row["median_ms"]
    return float(ms) / 1000.0


def render(cards, kind, sel):
    """The text one claim stands for."""
    if kind == "cmp":
        num, den = _split(sel)
        if "den_arm" not in sel:
            return fmt_time(_cmp_wall(cards, num))
        return fmt_ratio(_cmp_wall(cards, num) / _cmp_wall(cards, den))
    if kind == "wall":
        _c, row = _resolve(cards, sel)
        return fmt_time(cc.center(row["wall_s"]))
    if kind == "x":
        num, den = _split(sel)
        (_a, rn), (_b, rd) = _resolve(cards, num), _resolve(cards, den)
        return fmt_ratio(cc.center(rn["wall_s"]) / cc.center(rd["wall_s"]))
    if kind in ("mib", "xmem", "xmin"):
        if kind == "xmem":
            num, den = _split(dict(sel, src="memory"))
            (_a, rn), (_b, rd) = _resolve(cards, num), _resolve(cards, den)
            return fmt_ratio(rn["device_peak_bytes"] / rd["device_peak_bytes"])
        _c, row = _resolve(cards, dict(sel, src="memory"))
        if kind == "mib":
            return fmt_mib(row["device_peak_bytes"])
        return fmt_ratio(row["device_factor"])
    if kind == "pct":
        _c, row = _resolve(cards, sel)
        return fmt_pct(row[sel["field"]])
    if kind == "spec":
        value = cards.load(sel["card"], sel["dev"])
        for part in sel["field"].split("."):
            value = value[part]
        return cc.fmt_rate(value) if isinstance(value, float) else str(value)
    if kind == "pctgap":
        num, den = _split(sel)
        (_a, rn), (_b, rd) = _resolve(cards, num), _resolve(cards, den)
        return fmt_pct(cc.center(rn["wall_s"]) / cc.center(rd["wall_s"]) - 1.0)
    if kind == "rate":
        _c, row = _resolve(cards, sel)
        fmt = cc.fmt_rate_si if sel["card"].startswith("rk78") else cc.fmt_rate
        return fmt(row[sel["field"]])
    if kind == "comp":
        card = cards.load(sel["card"], sel["dev"])
        rows = [r for r in card["compile_time"]["rows"] if r["key"] == sel["key"]]
        if len(rows) != 1:
            raise ClaimError(f"compile_time: {len(rows)} rows for key {sel['key']!r}")
        value = rows[0][f"{sel['which']}_s"]
        return fmt_time(value["median"] if isinstance(value, dict) else value)
    raise ClaimError(f"unknown claim kind {kind!r}")


# --------------------------------------------------------------------------- #
# Page processing
# --------------------------------------------------------------------------- #
_SPAN = re.compile(
    r"<!--ctx\s+(?P<ctx>[^>]*?)\s*-->"
    r"|<!--c\s+(?P<spec>[^>]*?)\s*-->(?P<body>.*?)<!--/c-->",
    re.S,
)
_RST = re.compile(
    r"^\.\. c: (?P<spec>.+)\n(?P<def>\.\. \|(?P<name>[\w-]+)\| replace:: )(?P<body>.*)$",
    re.M,
)


def _kv(text):
    out = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _headline_value(sel, headline):
    """``dev=headline`` / ``n=headline1|headline2`` resolve through HEADLINE."""
    sel = dict(sel)
    if sel.get("dev") == "headline":
        sel["dev"] = headline["dev"]
    for key in ("n", "den_n"):
        if sel.get(key) in ("headline1", "headline2"):
            sel[key] = str(headline["ns"][int(sel[key][-1]) - 1])
    return sel


def _claim(cards, spec, ctx):
    words = spec.split()
    kind = words[0]
    sel = dict(ctx)
    sel.update(_kv(spec))
    sel = _headline_value(sel, cards.headline)
    return render(cards, kind, sel)


def process(text, cards, name="<page>"):
    """`text` with every claim span (and rst ``.. c:`` definition) re-rendered."""
    ctx = dict(BASE_CTX)

    def sub(m):
        if m.group("ctx") is not None:
            ctx.update(_kv(m.group("ctx")))
            return m.group(0)
        line = text.count("\n", 0, m.start()) + 1
        try:
            value = _claim(cards, m.group("spec"), ctx)
        except (ClaimError, KeyError) as exc:
            raise ClaimError(
                f"{name}:{line}: <!--c {m.group('spec')}-->: {exc}"
            ) from None
        return f"<!--c {m.group('spec')}-->{value}<!--/c-->"

    def rst(m):
        line = text.count("\n", 0, m.start()) + 1
        try:
            value = _claim(cards, m.group("spec"), ctx)
        except (ClaimError, KeyError) as exc:
            raise ClaimError(f"{name}:{line}: .. c: {m.group('spec')}: {exc}") from None
        return f".. c: {m.group('spec')}\n{m.group('def')}{value}"

    text = _SPAN.sub(sub, text)
    return _RST.sub(rst, text)


def count_claims(text):
    """How many claim spans (and rst definitions) a page carries."""
    return len(re.findall(r"<!--c\s", text)) + len(re.findall(r"^\.\. c: ", text, re.M))


# --------------------------------------------------------------------------- #
# The headline: HEADLINE's device, its larger N with the smaller N beside it
# --------------------------------------------------------------------------- #
#: Tools the headline names, in this order (never resorted by ratio).
HEAD_TOOLS = (
    ("warp_kernel", "NVIDIA Warp"),
    ("jax_vmap", "JAX"),
    ("cupy_masked", "CuPy"),
    ("torch_masked", "PyTorch"),
)
HEAD_MEM_TOOLS = ("cupy_masked", "torch_masked")
BASE_ARM = "eagle_simulate"


def _head_cell(card, n, dist, steps):
    return {
        r["arm"]: r
        for r in card["results"]
        if r["n"] == n and r["distribution"] == dist and r["max_steps"] == steps
    }


def _head_mem(card, arm, n, dist, steps):
    for r in card["memory"]["rows"]:
        if (r["arm"], r["n"], r["distribution"], r["max_steps"]) == (
            arm,
            n,
            dist,
            steps,
        ):
            return r
    raise ClaimError(f"headline card has no memory row for {arm} at N={n:,}")


def headline_facts(cards):
    """Every number the headline shows, rendered under the rounding rule.

    Raises ``ClaimError`` naming the missing card when the configured headline
    device has none (no other device is ever substituted)."""
    hl = cards.headline
    card = cards.load("rk4", hl["dev"])
    dist, steps = hl["dist"], hl["steps"]
    facts = {
        "device": card["device"]["name"],
        "date": card["generated_utc"][:10],
        "dist": dist,
        "steps": steps,
        "ns": [],
    }
    for n in hl["ns"]:
        cell = _head_cell(card, n, dist, steps)
        if BASE_ARM not in cell:
            raise ClaimError(
                f"headline card lacks {BASE_ARM} at N={n:,} ({dist}, {steps} steps)"
            )
        base = cc.center(cell[BASE_ARM]["wall_s"])
        speed = []
        for arm, name in HEAD_TOOLS:
            if arm in cell:
                speed.append((name, floor_sig(cc.center(cell[arm]["wall_s"]) / base)))
        base_mem = _head_mem(card, BASE_ARM, n, dist, steps)
        mem = [
            floor_sig(
                _head_mem(card, a, n, dist, steps)["device_peak_bytes"]
                / base_mem["device_peak_bytes"]
            )
            for a in HEAD_MEM_TOOLS
        ]
        facts["ns"].append(
            {
                "n": n,
                "wall": fmt_time(base),
                "speed": speed,
                "mem": mem,
                "mib": fmt_mib(base_mem["device_peak_bytes"]),
                "xmin": fmt_ratio(base_mem["device_factor"]),
            }
        )
    # the dense batch and how often Warp is the fastest GPU arm
    n_dense = hl["ns"][0]
    dense = _head_cell(card, n_dense, "uniform", steps)
    gpu = [a for a in card["arms"] if a != "cpu_openmp"]
    cells, warp_first = {}, 0
    for r in card["results"]:
        cells.setdefault((r["n"], r["distribution"], r["max_steps"]), {})[r["arm"]] = (
            cc.center(r["wall_s"])
        )
    for walls in cells.values():
        ours = [w for a, w in walls.items() if a in gpu]
        if "warp_kernel" in walls and ours and walls["warp_kernel"] == min(ours):
            warp_first += 1
    facts["warp_first"], facts["cells"] = warp_first, len(cells)
    if "warp_kernel" in dense and BASE_ARM in dense:
        facts["dense"] = (
            n_dense,
            fmt_time(cc.center(dense["warp_kernel"]["wall_s"])),
            fmt_time(cc.center(dense[BASE_ARM]["wall_s"])),
        )
    return facts


def _span(speed):
    vals = [v for _n, v in speed if v is not None]
    lo, hi = min(vals), max(vals)
    return fmt_ratio(lo) if lo == hi else f"{fmt_ratio(lo)}–{fmt_ratio(hi)}"


def _caption(speed):
    groups = []
    for name, v in speed:
        if groups and groups[-1][0] == v:
            groups[-1][1].append(name)
        else:
            groups.append((v, [name]))
    return ", ".join(f"{' and '.join(names)} ({fmt_ratio(v)})" for v, names in groups)


def render_hero(facts):
    """The README's three-figure strip and the note under it."""
    a, b = facts["ns"]

    def mspan(f):
        vals = [m for m in f["mem"] if m is not None]
        lo, hi = min(vals), max(vals)
        return fmt_ratio(lo) if lo == hi else f"{fmt_ratio(lo)}\u2013{fmt_ratio(hi)}"

    steps = facts["steps"]
    lines = [
        f"| **{a['wall']}** at N = {a['n']:,}, {b['wall']} at N = {b['n']:,} "
        f"| **{_span(a['speed'])} faster** at N = {a['n']:,}, {_span(b['speed'])} at N = {b['n']:,} "
        f"| **{mspan(a)} less GPU memory** at N = {a['n']:,}, {mspan(b)} at N = {b['n']:,} |",
        "|:---:|:---:|:---:|",
        f"| RK4 oscillators, each stopping at its own step, up to {steps} steps "
        f"| than {_caption(a['speed'])} at N = {a['n']:,} "
        f"| than CuPy and PyTorch: {a['mib']} for N = {a['n']:,}, {a['xmin']} the bare minimum state |",
        "",
        f"<sub>{facts['device']} (card of {facts['date']}). NVIDIA Warp's per-thread kernel is the "
        f"fastest GPU arm in {facts['warp_first']} of {facts['cells']} RK4 cells",
    ]
    if "dense" in facts:
        n, warp, sim = facts["dense"]
        lines[-1] += (
            f"; on the dense batch (every sample running all {steps} steps) at N = {n:,} "
            f"it takes {warp} and `eagle.simulate` {sim}"
        )
    lines[-1] += ". See [every number, every arm](#performance).</sub>"
    return "\n".join(lines)


def render_hero_sentence(facts, subject="`eagle.simulate`"):
    """The same headline as one sentence, for a README that only quotes it."""
    a, b = facts["ns"]
    return (
        f"{subject} integrates {a['n']:,} RK4 oscillators, each stopping at its own step, "
        f"in {a['wall']} on a {facts['device']} ({b['wall']} for {b['n']:,}), "
        f"{_span(a['speed'])} faster than {_caption(a['speed'])} at N = {a['n']:,}"
        f" ({_span(b['speed'])} at N = {b['n']:,}), in {a['mib']} of GPU memory"
    )
