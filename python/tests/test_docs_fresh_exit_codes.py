# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""`tools/sync_readme_cards.py --check` says which failure it is, by exit code.

0 fresh, 1 stale, 3 a card the docs cite is missing, 4 a tool or companion
checkout is missing, 2 usage only. Each case runs the script on a temporary
copy of the cards and pages (``--root``). A grep test keeps the workflows from
reporting every failure as "out of date".

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "tools" / "sync_readme_cards.py"
_HEADLINE_CARD = "benchmarks/perf_card/card_nvidia-h100-80gb-hbm3.json"


def _run(*args, root=None, cwd=_REPO):
    cmd = [sys.executable, str(_SCRIPT), "--check", *args]
    if root is not None:
        cmd += ["--root", str(root)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


@pytest.fixture()
def tree(tmp_path):
    """A copy of the cards and every page the generator reads."""
    sys.path.insert(0, str(_REPO / "tools"))
    try:
        import sync_readme_cards as src
    finally:
        sys.path.pop(0)
    shutil.copytree(_REPO / "benchmarks", tmp_path / "benchmarks",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for rel in src.PROSE_FILES:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO / rel, dest)
    return tmp_path


def test_fresh_tree_exits_0(tree):
    assert _run(root=tree).returncode == 0


def test_stale_page_exits_1_and_names_the_file(tree):
    readme = tree / "README.md"
    text = readme.read_text()
    begin = text.index("<!-- cards:begin -->")
    readme.write_text(text[:begin] + "<!-- cards:begin -->\nhand edit\n"
                      + text[text.index("<!-- cards:end -->"):])
    res = _run(root=tree)
    assert res.returncode == 1
    assert "README.md" in res.stdout


def test_missing_card_exits_3_and_names_the_card(tree):
    (tree / _HEADLINE_CARD).unlink()
    res = _run(root=tree)
    assert res.returncode == 3
    assert "benchmarks/perf_card/card_nvidia-h100" in res.stderr


def test_missing_card_with_hero_skipped_still_exits_3(tree):
    (tree / _HEADLINE_CARD).unlink()
    res = _run("--without-headline", root=tree)
    assert res.returncode == 3
    assert "missing card" in res.stderr


def test_usage_error_exits_2():
    assert _run("--no-such-flag").returncode == 2


def test_script_reports_a_missing_site_tool_as_a_missing_companion(tmp_path):
    """docs_fresh.sh with no site checkout: one annotation naming the tool, non-zero exit."""
    res = subprocess.run(["bash", str(_REPO / "tools" / "docs_fresh.sh"), str(tmp_path / "nosite")],
                         capture_output=True, text=True, cwd=_REPO)
    assert res.returncode != 0
    assert "generator or companion missing" in res.stdout
    assert "sync_family_visuals.py" in res.stdout


def test_workflows_say_out_of_date_only_in_the_stale_branch():
    for wf in sorted((_REPO / ".github" / "workflows").glob("*.yml")):
        assert "out of date" not in wf.read_text(), wf.name
    script = (_REPO / "tools" / "docs_fresh.sh").read_text().splitlines()
    for i, line in enumerate(script):
        if "out of date" in line:
            assert line.lstrip().startswith("1)"), line
