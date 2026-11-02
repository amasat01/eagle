# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every measured number in prose is ``tools/prose_claims.py``'s own output.

The README, the performance reading page, the filtering page and the tutorials
mark each number with a claim span (see the module's docstring); this test
re-renders every span from the committed cards through
``tools/sync_readme_cards.py`` and requires the committed text to match, so a
hand-edited number, or a card re-run whose prose was not regenerated, fails
here instead of drifting.

RED: change one number inside a ``<!--c ...-->...<!--/c-->`` span, or re-run a
card without ``python tools/sync_readme_cards.py``, and
``test_prose_pages_match_a_fresh_render`` fails naming the page.

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


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"eagle_{name}", _REPO / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytest.importorskip("raptor_docs")  # tools/sync_readme_cards reads the shared docs tokens (pip install ./site)
sync = _load("sync_readme_cards")
pcl = sync.pcl


def test_one_rounding_rule():
    # the same ratio has one spelling: truncated to 2 significant figures
    assert pcl.fmt_ratio(55.7) == pcl.fmt_ratio(55.1) == "55×"
    assert pcl.fmt_ratio(53.3) == "53×"
    assert pcl.fmt_ratio(7.98) == "7.9×"
    assert pcl.fmt_ratio(3.99) == "3.9×"
    assert pcl.fmt_ratio(0.995) == "0.99×"
    assert pcl.fmt_ratio(41.0) == "41×"
    assert pcl.fmt_time(0.1556) == "156 ms"
    assert pcl.fmt_mib(50.4 * 2**20) == "50 MiB"
    assert pcl.fmt_pct(1.0023) == "100%"


@pytest.mark.parametrize("rel", sync.PROSE_FILES)
def test_prose_pages_match_a_fresh_render(rel):
    cards = pcl.Cards(_REPO)
    committed = (_REPO / rel).read_text()
    fresh = (
        pcl.process(committed, cards, rel)
        if rel == "README.md"
        else sync.process_file(rel, _REPO, cards)
    )
    assert fresh == committed, (
        f"{rel}: a claim span differs from its card; run tools/sync_readme_cards.py "
        "and commit the diff instead of editing a number by hand"
    )


def test_the_pages_keep_their_spans():
    total = 0
    for rel in sync.PROSE_FILES:
        total += pcl.count_claims((_REPO / rel).read_text())
    assert total >= 30, f"only {total} claim spans left across the prose pages"


def test_a_hand_edited_number_is_caught():
    cards = pcl.Cards(_REPO)
    text = (_REPO / "docs/content/performance/reading.md").read_text()
    # the first generated wall time on the page, whatever the committed card says
    m = re.search(r"(<!--c wall [^>]*-->)([^<]+)(<!--/c-->)", text)
    assert m, "reading.md has no generated wall time"
    edited = text.replace(m.group(0), m.group(1) + "1" + m.group(2) + m.group(3), 1)
    assert pcl.process(edited, cards, "reading.md") != edited


def test_a_card_without_the_cell_names_it(tmp_path):
    _synthetic(tmp_path)
    with pytest.raises(pcl.ClaimError, match="0 rows match"):
        pcl.process("<!--c wall arm=no_such_arm-->1 ms<!--/c-->", pcl.Cards(tmp_path), "x.md")


def _headline_card_committed():
    return bool(
        list(
            (_REPO / "benchmarks" / "perf_card").glob(
                pcl.FAMILIES["ensemble"][1] + pcl.DEVICES[pcl.HEADLINE["dev"]] + ".json"
            )
        )
    )


def test_missing_headline_card_fails_loudly_and_writes_nothing(tmp_path, capsys):
    if _headline_card_committed():
        pytest.skip("the headline card is committed")
    with pytest.raises(
        pcl.ClaimError,
        match=r"missing card: benchmarks/perf_card/card_nvidia-a100\*\.json",
    ):
        sync.render_hero_block(_REPO)
    before = (_REPO / "README.md").read_bytes()
    assert sync.main(["--check"]) == pcl.EXIT_MISSING_CARD
    assert "headline card missing" in capsys.readouterr().err
    assert (_REPO / "README.md").read_bytes() == before


def test_committed_hero_matches_the_headline_card():
    if not _headline_card_committed():
        pytest.skip(
            f"headline card for device {pcl.HEADLINE['dev']!r} is not committed yet"
        )
    text = (_REPO / "README.md").read_text()
    start = text.index(sync.HERO_BEGIN) + len(sync.HERO_BEGIN)
    end = text.index(sync.HERO_END, start)
    assert text[start:end].strip("\n") == sync.render_hero_block(_REPO)


def _card(name, slug, scale=1.0, eff=0.5):
    """A small ensemble card: five GPU arms at two sizes, spread and uniform, with memory rows."""
    arms = {"eagle_simulate": "eagle simulate", "warp_kernel": "NVIDIA Warp", "jax_vmap": "JAX", "cupy_masked": "CuPy",
            "torch_graphed": "PyTorch with CUDA graphs"}
    results, memory = [], []
    for n in (1_000_000, 10_000_000):
        for dist in ("spread", "uniform"):
            for k, arm in enumerate(arms):
                wall = scale * n * 1e-9 * (1 + k) * (2 if dist == "uniform" else 1)
                results.append({"n": n, "distribution": dist, "arm": arm, "status": "ok",
                                "wall_s": {"median": wall, "iqr": wall / 100}, "eff_sol": eff / (1 + k),
                                "attempt_samples_per_s": n / wall})
        for arm in arms:
            memory.append({"n": n, "arm": arm, "device_peak_bytes": n * 80, "minimum_bytes": n * 72})
    return {"device": {"name": name, "sm_count": 108}, "device_slug": slug, "generated_utc": "2026-11-01T00:00:00Z",
            "arms": arms, "results": results, "memory": {"rows": memory}}


def _synthetic(root: pathlib.Path):
    dst = root / "benchmarks" / "perf_card"
    dst.mkdir(parents=True)
    (dst / "card_nvidia-h100-fixture.json").write_text(json.dumps(_card("NVIDIA H100 (fixture)", "nvidia-h100-fixture")))
    (dst / "card_quadro-p2000.json").write_text(json.dumps(_card("Quadro P2000", "quadro-p2000", 10.0)))


def test_headline_renders_from_a_configured_device(tmp_path):
    _synthetic(tmp_path)
    pcl.DEVICES["h100fix"] = "nvidia-h100-fixture"
    try:
        head = {"dev": "h100fix", "ns": (10_000_000, 1_000_000), "dist": "spread"}
        hero = sync.render_hero_block(tmp_path, head)
        walls = [f["wall"] for f in pcl.headline_facts(pcl.Cards(tmp_path, head))["ns"]]
        # the other device is selectable with the same one-line switch
        p2000 = sync.render_hero_block(tmp_path, {"dev": "p2000", "ns": (1_000_000, 10_000_000), "dist": "spread"})
    finally:
        del pcl.DEVICES["h100fix"]
    assert "N = 10,000,000" in hero and "N = 1,000,000" in hero
    assert "NVIDIA H100 (fixture)" in hero and all(w in hero for w in walls)
    assert "Quadro P2000" in p2000 and hero != p2000


def test_a_row_that_did_not_run_shows_its_label(tmp_path):
    _synthetic(tmp_path)
    path = tmp_path / "benchmarks" / "perf_card" / "card_quadro-p2000.json"
    card = json.loads(path.read_text())
    for row in card["results"]:
        if row["arm"] == "cupy_masked" and row["n"] == 10_000_000:
            row.update(status="out_of_memory")
        if row["arm"] == "torch_graphed" and row["n"] == 10_000_000:
            row.update(status="not_run: budget")
    path.write_text(json.dumps(card))
    cards = pcl.Cards(tmp_path)
    text = pcl.process("<!--c wall arm=cupy_masked n=1e7-->x<!--/c--> / <!--c wall arm=torch_graphed n=1e7-->x<!--/c-->", cards)
    assert "out of memory" in text and "> 120 s per run" in text


def test_a_row_above_the_speed_of_light_blocks_an_efficiency_claim(tmp_path):
    _synthetic(tmp_path)
    path = tmp_path / "benchmarks" / "perf_card" / "card_quadro-p2000.json"
    card = json.loads(path.read_text())
    assert pcl.process("<!--c eff arm=eagle_simulate-->x<!--/c-->", pcl.Cards(tmp_path)).endswith("50%<!--/c-->")
    card["results"][0]["eff_sol"] = 1.2
    path.write_text(json.dumps(card))
    with pytest.raises(pcl.ClaimError, match="speed of light"):
        pcl.process("<!--c eff arm=eagle_simulate-->x<!--/c-->", pcl.Cards(tmp_path))


_UNIT_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)? ?(?:µs|ms|×|MiB)(?![\w])")


def _prose_outside_spans(rel):
    """A page's text without claim spans and generated blocks."""
    text = (_REPO / rel).read_text()
    text = re.sub(r"<!--c .*?<!--/c-->", " ", text, flags=re.DOTALL)
    for begin, end in (
        (sync.BEGIN, sync.END),
        (sync.CPU_BEGIN, sync.CPU_END),
        (sync.HERO_BEGIN, sync.HERO_END),
    ):
        text = re.sub(
            re.escape(begin) + ".*?" + re.escape(end), " ", text, flags=re.DOTALL
        )
    return text


@pytest.mark.parametrize(
    "rel", ["README.md", "docs/content/performance/reading.md"]
)
def test_no_hand_typed_time_or_ratio_is_left(rel):
    left = _UNIT_NUMBER.findall(_prose_outside_spans(rel))
    assert not left, f"{rel}: measured numbers outside a claim span: {left}"
