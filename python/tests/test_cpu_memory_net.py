# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""CPU peak memory is published net of the benchmark harness's own copies.

The RK4 CPU card's eagle arms hold 36 bytes per sample (separate omega and zeta
planes, and the copies their download makes) that eagle itself does not need;
the docs tables show the peak less those bytes, and the card keeps the raw
reading. The rule lives once, in ``benchmarks/_card_common.py``
(:func:`harness_copy_bytes`, :func:`net_peak_bytes`).

RED: make ``harness_copy_bytes`` return 0 and the rendered-cell test fails.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
if not (_REPO / "tools" / "prose_claims.py").is_file():  # the suite copied outside the repository
    pytest.skip("repository files are not present", allow_module_level=True)
_CARD = _REPO / "benchmarks" / "perf_card" / "cpu_card_intel-xeon-platinum-8581c.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cc = _load("eagle_card_common_net", _REPO / "benchmarks" / "_card_common.py")


def test_eagle_arm_is_net_and_numba_is_raw():
    n = 10**6
    peak = 100 * 2**20
    eagle = {"arm": "eagle_term8", "n": n, "peak_bytes": peak}
    numba = {"arm": "numba_prange", "n": n, "peak_bytes": peak}
    assert _cc.net_peak_bytes(eagle, "rk4_cpu") == peak - 36 * n
    assert _cc.net_peak_bytes(numba, "rk4_cpu") == peak
    for arm in ("eagle_compact1", "eagle_reorder8"):
        assert _cc.harness_copy_bytes(arm, n, "rk4_cpu") == 36 * n
    # another card kind stays raw
    assert _cc.net_peak_bytes(eagle, "rk78_cpu") == peak
    assert _cc.net_peak_bytes(eagle, "gpu") == peak


def test_a_negative_result_shows_the_measured_value():
    row = {"arm": "eagle_term8", "n": 10**6, "peak_bytes": 20 * 10**6}
    assert _cc.net_peak_bytes(row, "rk4_cpu") == 20 * 10**6
    assert _cc.net_peak_bytes({"arm": "eagle_term8", "n": 1000, "peak_bytes": 0}, "rk4_cpu") == 0


def _row(arm, n, peak, base):
    return {"arm": arm, "n": n, "peak_bytes": peak, "rss_baseline_bytes": base, "minimum_bytes": 36 * n}


def test_a_small_n_floor_row_and_a_1e6_row_near_the_floor_are_unresolved():
    mib = 2**20
    rows = [_row("eagle_term8", 1000, 40 * mib, 60 * mib), _row("eagle_term8", 10**4, 39 * mib, 60 * mib),
            _row("eagle_term8", 10**6, 42 * mib, 60 * mib)]
    floor = _cc.floor_estimate(rows, "eagle_term8")
    assert floor == 100 * mib
    assert not _cc.is_resolved(rows[0], floor)
    assert not _cc.is_resolved(rows[2], floor)
    assert _cc.published_peak_bytes(rows[2], rows, "rk4_cpu") is None


def test_a_1e6_row_above_the_floor_is_resolved_and_netted():
    mib = 2**20
    rows = [_row("eagle_term8", 1000, 40 * mib, 60 * mib), _row("eagle_term8", 10**6, 96 * mib, 60 * mib),
            _row("numba_prange", 1000, 0, 220 * mib), _row("numba_prange", 10**6, 68 * mib, 220 * mib)]
    assert _cc.published_peak_bytes(rows[1], rows, "rk4_cpu") == 96 * mib - 36 * 10**6
    assert _cc.published_peak_bytes(rows[3], rows, "rk4_cpu") == 68 * mib  # raw: no harness copies


def test_rendered_cpu_table_cell_is_the_net_value_read_from_the_card():
    gen = _load("eagle_gen_perf_pages_net", _REPO / "docs" / "_tools" / "gen_perf_pages.py")
    spec = next(s for s in gen.FAMILIES if s["key"] == "perf_card_cpu")
    entry = next(e for e in gen.load_family(_REPO, spec) if e["slug"].startswith("intel-xeon"))
    card = json.loads(_CARD.read_text())
    cfg = card["configs"][1]  # the second "###" section
    rows = {(r["arm"], r["n"]): r for r in card["memory"]["rows"]
            if r["distribution"] == cfg["distribution"] and r["max_steps"] == cfg["max_steps"]}
    fragment = gen._table_fragment(spec, entry, [entry])
    sections = re.split(r"^\*\*", fragment, flags=re.M)
    section = next(s for s in sections if f"up to {cfg['max_steps']} steps" in s.splitlines()[0])
    labels = card["arm_labels"]
    n = max(card["ns"])
    for arm in ("eagle_term8", "numba_prange"):
        row = rows[(arm, n)]
        net = _cc.published_peak_bytes(row, card["memory"]["rows"], "rk4_cpu")
        assert net is not None
        line = next(ln for ln in section.splitlines()
                    if ln.startswith(f"| {n:,} | {labels[arm]} |"))
        cell = line.split("|")[-2].strip()
        assert cell == f"{_cc.fmt_mib(net)} ({_cc.fmt_x(net / row['minimum_bytes'])})"
    assert _cc.net_peak_bytes(rows[("eagle_term8", n)], "rk4_cpu") < rows[("eagle_term8", n)]["peak_bytes"]
    assert gen.CPU_MEMORY_NOTE in fragment
    small = next(ln for ln in section.splitlines() if ln.startswith(f"| 1,000 | {labels['eagle_term8']} |"))
    assert small.split("|")[-2].strip() == gen.UNRESOLVED_CELL
