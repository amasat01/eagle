# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The large-N compare evidence under ``benchmarks/compare_card/``.

Each device has ``<slug>.csv`` (a byte copy of the compare stage's output) and
``<slug>.json`` (its provenance). Prose quotes the CSVs through the ``cmp``
claim span of ``tools/prose_claims.py``; these tests pin the pairing, the span
and the check that covers pages using it.

RED: edit one byte of a committed CSV and ``test_sidecar_md5_matches_its_csv``
fails naming it; drop a sidecar and ``test_every_csv_has_a_sidecar`` fails.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import shutil

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
if not (_REPO / "tools" / "prose_claims.py").is_file():  # the suite copied outside the repository
    pytest.skip("repository files are not present", allow_module_level=True)
_DIR = _REPO / "benchmarks" / "compare_card"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"eagle_{name}", _REPO / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load("sync_readme_cards")
ingest = _load("ingest_compare")
pcl = sync.pcl

_HEADER = ingest.HEADER


def _row(arm, ms, status="ok", norm="", n=10_000_000, dist="spread"):
    cols = [
        "osc", str(n), "1000", dist, arm, status, str(ms), str(ms), "persist", "-", "",
        "1", "0", "1980", str(norm), "", "", "", "fp64", "",
    ]
    return ",".join(cols)


def _tree(tmp_path, rows):
    root = tmp_path / "root"
    (root / "benchmarks" / "compare_card").mkdir(parents=True)
    (root / "benchmarks" / "compare_card" / "quadro-p2000.csv").write_text(
        "\n".join([_HEADER, *rows]) + "\n"
    )
    return root


def _csvs():
    return sorted(_DIR.glob("*.csv"))


def test_there_is_evidence_for_every_final_device():
    assert {p.stem for p in _csvs()} >= {
        "quadro-p2000", "tesla-t4", "nvidia-a100-sxm4-40gb", "nvidia-h100-80gb-hbm3",
    }


@pytest.mark.parametrize("csv", _csvs(), ids=lambda p: p.stem)
def test_sidecar_md5_matches_its_csv(csv):
    import json

    side = json.loads(csv.with_suffix(".json").read_text())
    assert side["csv_md5"] == hashlib.md5(csv.read_bytes(), usedforsecurity=False).hexdigest()
    assert side["slug"] == csv.stem
    assert side["eagle_commit"] and side["hawk_commit"]
    assert side["kit_branch"] == "profile-kit" and side["kit_commit"]
    assert len(side["gpu_profile_md5"]) == 32
    assert csv.read_text().splitlines()[0] == _HEADER


def test_every_csv_has_a_sidecar():
    assert _csvs()
    for csv in _csvs():
        assert csv.with_suffix(".json").is_file(), f"{csv.name} has no sidecar"
    for js in _DIR.glob("*.json"):
        assert js.with_suffix(".csv").is_file(), f"{js.name} has no csv"


def test_the_span_renders_from_a_synthetic_csv(tmp_path):
    root = _tree(tmp_path, [_row("warp", 3300.0), _row("auto", 1000.0)])
    cards = pcl.Cards(root)
    page = (
        "<!--ctx dev=p2000 n=1e7 dist=spread steps=1000-->\n"
        "ratio <!--c cmp arm=warp den_arm=auto-->0×<!--/c--> "
        "wall <!--c cmp arm=auto-->0 s<!--/c-->\n"
    )
    out = pcl.process(page, cards, "x.md")
    assert "<!--c cmp arm=warp den_arm=auto-->3.3×<!--/c-->" in out
    assert "<!--c cmp arm=auto-->1 s<!--/c-->" in out


def test_clock_normalised_wall_is_preferred(tmp_path):
    root = _tree(tmp_path, [_row("auto", 1000.0, norm=500.0)])
    out = pcl.process(
        "<!--c cmp dev=p2000 n=1e7 dist=spread steps=1000 arm=auto-->x<!--/c-->",
        pcl.Cards(root),
    )
    assert ">500 ms<" in out


def test_a_non_ok_row_is_refused(tmp_path):
    root = _tree(tmp_path, [_row("warp", 1.0, status="skipped"), _row("auto", 1.0)])
    with pytest.raises(pcl.ClaimError, match="status 'skipped'"):
        pcl.process(
            "<!--c cmp dev=p2000 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->x<!--/c-->",
            pcl.Cards(root),
            "x.md",
        )


def test_a_missing_row_or_device_is_refused(tmp_path):
    root = _tree(tmp_path, [_row("auto", 1.0)])
    with pytest.raises(pcl.ClaimError, match="0 rows match"):
        pcl.process(
            "<!--c cmp dev=p2000 n=1e7 dist=spread steps=1000 arm=nope-->x<!--/c-->",
            pcl.Cards(root),
        )
    with pytest.raises(pcl.ClaimError, match="missing compare evidence"):
        pcl.process(
            "<!--c cmp dev=t4 n=1e7 dist=spread steps=1000 arm=auto-->x<!--/c-->",
            pcl.Cards(root),
        )


def test_the_sync_check_covers_pages_using_the_span(tmp_path):
    """A tree with the prose pages, the cards and one compare CSV: a README
    carrying a stale ``cmp`` span makes ``sync_readme_cards --check`` fail."""
    root = tmp_path / "root"
    shutil.copytree(_REPO / "benchmarks", root / "benchmarks",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for rel in sync.PROSE_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO / rel, root / rel)
    span = "<!--ctx dev=p2000 n=1e7 dist=spread steps=1000-->\n<!--c cmp arm=warp den_arm=auto-->{}<!--/c-->\n"
    readme = root / "README.md"
    base = readme.read_text()
    good = pcl.process(span.format("?"), pcl.Cards(root)).split("<!--c ")[1].split("-->")[1].split("<")[0]
    readme.write_text(base + span.format(good))
    assert sync.main(["--check", "--root", str(root)]) == 0
    readme.write_text(base + span.format("9.9×"))
    assert sync.main(["--check", "--root", str(root)]) == 1
