# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The performance page's bar charts: one tab per card, default tab from one
preference list, a new card dropped in appears with no code change, and the
README SVG carries the numbers as text.

``repo_local``: reads the repository's docs generator and committed cards.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "docs" / "_tools"


def _generator():
    spec = importlib.util.spec_from_file_location("eagle_gen_perf_pages", _TOOLS / "gen_perf_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _charts():
    return _generator().charts


def test_gpu_default_tab_follows_the_preference_list():
    charts = _charts()
    names = ["Zed GPU", "Quadro P2000", "Tesla T4", "NVIDIA H100 80GB", "Alpha GPU"]
    got = [e["name"] for e in charts.in_tab_order([{"name": n} for n in names], charts.GPU_PREFERENCE)]
    assert got == ["NVIDIA H100 80GB", "Tesla T4", "Quadro P2000", "Alpha GPU", "Zed GPU"]


def test_cpu_server_parts_precede_workstation_parts():
    charts = _charts()
    names = ["Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz", "Intel(R) Xeon(R) Platinum 8480+", "Acme Z9"]
    got = [e["name"] for e in charts.in_tab_order([{"name": n} for n in names], charts.CPU_PREFERENCE)]
    assert got == [names[1], names[0], names[2]]


def test_dropping_a_card_adds_a_tab_with_no_code_change(tmp_path):
    gen = _generator()
    root = tmp_path / "eagle"
    shutil.copytree(_REPO / "benchmarks" / "perf_card", root / "benchmarks" / "perf_card",
                    ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    shutil.copytree(_REPO / "benchmarks" / "rk78_card", root / "benchmarks" / "rk78_card",
                    ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    shutil.copy(_REPO / "benchmarks" / "_card_common.py", root / "benchmarks" / "_card_common.py")
    out = root / "docs" / "content" / "_generated"
    gen.generate(root, out)
    before = (out / "perf_card_gpu.md").read_text().count(":::{tab-item}")
    src = root / "benchmarks" / "perf_card" / "card_tesla-t4.json"
    (root / "benchmarks" / "perf_card" / "card_extra-gpu.json").write_text(
        src.read_text().replace("Tesla T4", "Extra GPU").replace("tesla-t4", "extra-gpu"))
    (root / "benchmarks" / "perf_card" / "card_extra-gpu.md").write_text(
        (root / "benchmarks" / "perf_card" / "card_tesla-t4.md").read_text())
    gen.generate(root, out)
    text = (out / "perf_card_gpu.md").read_text()
    assert text.count(":::{tab-item}") == before + 1
    assert ":::{tab-item} Extra GPU" in text


def test_readme_svg_names_the_tools_and_the_machine():
    svg = (_REPO / "docs" / "_static" / "perf" / "readme_gpu.svg").read_text()
    assert "<desc" in svg and "eagle.simulate" in svg and "NVIDIA Warp" in svg
    assert "prefers-color-scheme: dark" in svg
