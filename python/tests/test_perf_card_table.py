# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The performance page's table is the committed card, re-rendered.

``benchmarks/perf_card/perf_card.py`` writes ``card_<device>.json`` (every
number) and ``card_<device>.md`` (the table the docs page includes), both from
one run. This test re-renders every committed card JSON with the script's own
:func:`render_markdown` and requires the committed Markdown to match it byte
for byte, so the published table cannot drift from the numbers, and a hand
edit to either file fails here. It also checks the docs page includes each
card table and that each card was produced by the committed script (its
recorded md5 matches ``perf_card.py``).

It also checks each card covers every arm of its script, with a memory row
per result row and every arm in the compile-time table (or named as having no
compile step), that every arm's code snippet matches its script's
markers, that the K-steps, automatic and ``eagle.simulate`` arms are wired
through every table of the script, that the eagle arms' snippets show the
user-facing doors (``eagle.deploy``/``eagle.simulate``), that no card script
imports a private eagle or hawk name, and that the CPU card's compile flags
came from hawk itself.

RED: edit one number in a committed ``card_*.md`` (or ``card_*.json``) and
this fails naming the file.

``repo_local``: reads repository files (the benchmark and the docs source),
never shipped in the wheel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import tempfile

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CARD_DIR = _REPO / "benchmarks" / "perf_card"
_GEN_SCRIPT = _REPO / "docs" / "_tools" / "gen_perf_pages.py"
#: Every card script and the helpers they share.
_CARD_SCRIPTS = (_CARD_DIR / "perf_card.py", _CARD_DIR / "cpu_card.py",
                 _REPO / "benchmarks" / "rk78_card" / "rk78_card.py",
                 _REPO / "benchmarks" / "rk78_card" / "cpu_rk78_card.py",
                 _REPO / "benchmarks" / "_card_common.py")
_PAGE = _REPO / "docs" / "content" / "performance.md"
_README = _REPO / "README.md"

#: The README's headline table: N = 1,000,000, the five arms shown
#: (the three graph arms -- plain, compacted, compacted + reorder -- the
#: masked CuPy arm, the CPU arm), both distributions run up to 1000 steps.
#: Kept in lockstep with the README by hand; this test is what catches the
#: two drifting apart.
_README_ROWS = [(dist, 1000, arm)
                for dist in ("spread", "uniform")
                for arm in ("eagle_graph", "eagle_graph_compact", "eagle_graph_reorder",
                            "cupy_masked", "cpu_openmp")]


def _script():
    spec = importlib.util.spec_from_file_location(
        "eagle_perf_card", _CARD_DIR / "perf_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cards():
    cards = sorted(_CARD_DIR.glob("card_*.json"))
    assert cards, f"no committed card_*.json under {_CARD_DIR}"
    return cards


def test_committed_tables_match_their_cards():
    render = _script().render_markdown
    for card_path in _cards():
        card = json.loads(card_path.read_text())
        table_path = card_path.with_suffix(".md")
        assert table_path.is_file(), f"{card_path.name}: no rendered {table_path.name}"
        assert table_path.read_text() == render(card), (
            f"{table_path.name} is not the rendering of {card_path.name}: "
            "re-run benchmarks/perf_card/perf_card.py instead of editing either file")


def test_cards_name_their_script_and_are_complete():
    for card_path in _cards():
        card = json.loads(card_path.read_text())
        assert card["script"] == "benchmarks/perf_card/perf_card.py"
        script_md5 = hashlib.md5((_CARD_DIR / "perf_card.py").read_bytes()).hexdigest()
        assert card["script_md5"] == script_md5, (
            f"{card_path.name} was produced by a different perf_card.py: re-run "
            "the script so the card and its producer match")
        expected = {(c["distribution"], c["max_steps"], n, arm)
                    for c in card["configs"] for n in card["ns"]
                    for arm in card["arms"]}
        got = {(r["distribution"], r["max_steps"], r["n"], r["arm"])
               for r in card["results"]}
        assert got == expected, f"{card_path.name}: result rows do not cover the matrix"
        assert card["method"]["repetitions"] >= 5


def test_cards_cover_every_arm_memory_and_compile_step():
    """Every arm of the script is on the card; every result row has a memory
    row (peak device memory for the GPU arms, peak host memory for all); and
    every arm is either in the compile-time table or named as having no
    compile step.

    RED: drop an arm from ``ARMS``' run, a memory row or a compile row from a
    committed card and this fails naming the card."""
    module = _script()
    for card_path in _cards():
        card = json.loads(card_path.read_text())
        run = set(card["arms"])
        not_run = set(card["arms_not_run"])
        assert set(module.ARMS) <= run, f"{card_path.name}: arms missing"
        assert run | not_run == set(module.ARMS) | set(module.OPTIONAL_ARMS), (
            f"{card_path.name}: every optional arm is either run or named as not run")
        assert not run & not_run
        got = {(r["distribution"], r["max_steps"], r["n"], r["arm"])
               for r in card["results"]}
        mem = {(r["distribution"], r["max_steps"], r["n"], r["arm"]): r
               for r in card["memory"]["rows"]}
        assert set(mem) == got, (
            f"{card_path.name}: memory rows do not cover the result rows")
        for key, r in mem.items():
            gpu = key[3] in module.GPU_ARMS
            assert (r["device_peak_bytes"] is not None) == gpu, (
                f"{card_path.name}: device memory missing or spurious for {key}")
            assert r["host_peak_bytes"] is not None, (
                f"{card_path.name}: no host memory for {key}")
        comp = card["compile_time"]
        covered = {a for r in comp["rows"] for a in r["arms"]} | set(comp["none"])
        assert covered == run, (
            f"{card_path.name}: compile-time table does not cover every arm "
            f"(missing {sorted(run - covered)})")
        keys = {r["key"] for r in comp["rows"]}
        want = {k for k, _, _ in module._compile_keys(card["ns"], tuple(run))}
        assert keys == want, (
            f"{card_path.name}: compile-time rows differ from the script's keys")
        assert set(card["fit"]["arms"]) == run
        for extra in card["per_config"]:
            for key in ("auto_schedule_matches_run", "graph_schedule_matches_run"):
                assert extra.get(key, True), (
                    f"{card_path.name}: {key} is false at {extra['distribution']} "
                    f"S={extra['max_steps']} N={extra['n']}: the replayed launches "
                    "differ from the run's, so the issued bytes would be wrong")
        _check_code_record(card, card_path, {"perf_card.py": module._code_blocks(
            _CARD_DIR / "perf_card.py")}, prefixed=False)


def _check_code_record(card, card_path, blocks_by_file, prefixed):
    """Every arm on the card has a code snippet made of marked blocks of its
    script (``# >>> code:<name>`` ... ``# <<< code:<name>``), with the line
    counts the script's markers give."""
    if prefixed:
        blocks = {f"{f}:{k}": v for f, bs in blocks_by_file.items()
                  for k, v in bs.items()}
    else:
        blocks = next(iter(blocks_by_file.values()))
    code = card["code"]
    assert set(code["arms"]) == set(card["arms"]), (
        f"{card_path.name}: arms without a snippet")
    assert code["blocks"] == blocks, (
        f"{card_path.name}: code blocks differ from the script's markers")
    for arm, rec in code["arms"].items():
        assert rec["lines"] == sum(blocks[b]["lines"] for b in rec["blocks"]) > 0, (
            f"{card_path.name}: {arm}'s snippet line count is stale")


def test_the_k_steps_arms_are_wired_through_every_table():
    """The K-steps arms (``@hawk.kernel(steps=K)``) have a label, a loop row,
    a kernel-pass entry, a compile row, a code snippet built from the
    decorator form, and a fit note that says what fusion does and that K is
    chosen by hand -- checked on the script, so it holds before a card is
    re-run.

    RED: drop either arm from ``ARMS`` or one of these tables, or spell the
    kernel without the decorator argument, and this fails naming the arm."""
    pytest.importorskip("torch")  # the script asks torch which optional arms run
    module = _script()
    blocks = module._code_blocks(_CARD_DIR / "perf_card.py")
    source = (_CARD_DIR / "perf_card.py").read_text().splitlines()
    first = blocks["hawk_steps"]["first_line"]
    assert source[first - 1].strip() == "@hawk.kernel(steps=STEPS_PER_LAUNCH)", (
        "the K-steps kernel is not declared with the decorator form")
    keys = {a for _, arms, _ in module._compile_keys([1_000]) for a in arms}
    assert isinstance(module.STEPS_PER_LAUNCH, int) and module.STEPS_PER_LAUNCH > 1
    card = {"arms": list(module.ARMS), "ns": [1_000], "configs": [], "results": [],
            "torch_graph_block": module.GRAPH_BLOCK,
            "workload": {"compaction_every_steps": module.COMPACT_EVERY,
                         "steps_per_launch": module.STEPS_PER_LAUNCH}}
    fit = module._fit_notes(card)["arms"]
    assert module.ARM_CODE["eagle_graph_steps"][0] == "hawk_steps"
    compact = blocks["eagle_graph_steps_compact"]
    compact_src = "\n".join(source[compact["first_line"] - 1:compact["last_line"]])
    assert "hawk.steps(oscillator_step, STEPS_PER_LAUNCH)" in compact_src, (
        "the K-steps compaction arm does not derive its kernel with hawk.steps")
    for arm in module.STEPS_ARMS:
        assert arm in module.ARMS and arm in module.EAGLE_GPU_ARMS, arm
        assert arm in module.ARM_LABELS and arm in module.ARM_LOOP, arm
        assert module.KERNEL_PASS_RUNS[arm] >= 1, arm
        assert arm in keys, f"{arm}: no compile-time row"
        assert all(b in blocks for b in module.ARM_CODE[arm]), arm
        launches = -(-1000 // module.STEPS_PER_LAUNCH)
        assert module._expected_kernels(arm, 1000) == launches
    steps_note = fit["eagle_graph_steps"]
    assert f"@hawk.kernel(steps={module.STEPS_PER_LAUNCH})" in steps_note
    assert "registers" in steps_note and "Warp" in steps_note
    assert "eagle_graph_auto arm picks it at run time" in steps_note
    assert module.ALL_ARMS.index("torch_compiled") == (
        module.ALL_ARMS.index("torch_graphed") + 1)
    assert module.STEPS_COMPACT_EVERY % module.STEPS_PER_LAUNCH == 0


def test_the_auto_arm_is_wired_through_every_table():
    """The automatic arm (``@hawk.kernel(steps="auto")``) has a label, a loop
    row, a kernel-pass entry, a compile row, a code snippet built from the
    decorator form and a fit note that says it picks the steps per launch
    itself -- checked on the script, so it holds before a card is re-run.

    RED: drop the arm from ``ARMS`` or one of these tables, or spell the
    kernel without ``steps="auto"``, and this fails naming the arm."""
    pytest.importorskip("torch")  # the script asks torch which optional arms run
    module = _script()
    arm = module.AUTO_ARM
    blocks = module._code_blocks(_CARD_DIR / "perf_card.py")
    source = (_CARD_DIR / "perf_card.py").read_text().splitlines()
    first = blocks["hawk_auto"]["first_line"]
    assert source[first - 1].strip() == '@hawk.kernel(steps="auto", kind=ACTIVE_SET)', (
        "the automatic kernel is not declared with the decorator form")
    assert arm in module.ARMS and arm in module.EAGLE_GPU_ARMS
    assert arm in module.ARM_LABELS and arm in module.ARM_LOOP
    assert module.KERNEL_PASS_RUNS[arm] >= 1
    keys = {a for _, arms, _ in module._compile_keys([1_000]) for a in arms}
    assert arm in keys, f"{arm}: no compile-time row"
    assert module.ARM_CODE[arm] == ("hawk_kind", "hawk_auto", arm)
    assert all(b in blocks for b in module.ARM_CODE[arm])
    assert "every=" not in "\n".join(
        source[blocks[arm]["first_line"] - 1:][:blocks[arm]["lines"]])
    card = {"arms": list(module.ARMS), "ns": [1_000], "configs": [], "results": [],
            "torch_graph_block": module.GRAPH_BLOCK,
            "workload": {"compaction_every_steps": module.COMPACT_EVERY,
                         "steps_per_launch": module.STEPS_PER_LAUNCH}}
    note = module._fit_notes(card)["arms"][arm]
    assert 'steps="auto"' in note and "picks the steps per launch itself" in note
    assert "Warp" in note and "compacted" in note


def test_the_simulate_arm_is_wired_through_every_table():
    """The ``eagle.simulate`` arm runs the automatic kernel through
    ``eagle.simulation`` and has a label, a loop row, a kernel-pass entry, a
    compile row, a snippet and a fit note -- checked on the script.

    RED: drop the arm from ``ARMS`` or one of these tables, or build its loop
    by hand instead of through ``eagle.simulation``, and this fails."""
    pytest.importorskip("torch")  # the script asks torch which optional arms run
    module = _script()
    arm = module.SIMULATE_ARM
    blocks = module._code_blocks(_CARD_DIR / "perf_card.py")
    source = (_CARD_DIR / "perf_card.py").read_text().splitlines()
    assert arm in module.ARMS and arm in module.EAGLE_GPU_ARMS
    assert arm in module.ARM_LABELS and arm in module.ARM_LOOP
    assert module.KERNEL_PASS_RUNS[arm] >= 1
    assert arm in {a for _, arms, _ in module._compile_keys([1_000]) for a in arms}
    assert module.ARM_CODE[arm] == ("hawk_kind", "hawk_auto", arm)
    text = "\n".join(source[blocks[arm]["first_line"] - 1:blocks[arm]["last_line"]])
    assert "eagle.simulation(oscillator_step" in text
    card = {"arms": list(module.ARMS), "ns": [1_000], "configs": [], "results": [],
            "torch_graph_block": module.GRAPH_BLOCK,
            "workload": {"compaction_every_steps": module.COMPACT_EVERY,
                         "steps_per_launch": module.STEPS_PER_LAUNCH}}
    assert "eagle.simulate" in module._fit_notes(card)["arms"][arm]


def test_eagle_snippets_use_the_user_facing_doors():
    """Every eagle arm's own snippet builds its plan with ``eagle.deploy`` (or
    runs through ``eagle.simulation``) from the hawk kernel it names: no plugin
    assembly, no prebuilt plan passed in.

    RED: hand an eagle arm a plan built elsewhere and this fails naming it."""
    for path, arm_blocks in ((_CARD_DIR / "perf_card.py",
                              {a: [a] for a in _script().EAGLE_GPU_ARMS}),
                             (_CARD_DIR / "perf_card.py",
                              {"cpu_openmp": ["cpu_openmp_deploy"]}),
                             (_CARD_DIR / "cpu_card.py",
                              {a: [a] for a in ("eagle_term", "eagle_compact",
                                                "eagle_reorder")})):
        module = _script()
        source = path.read_text().splitlines()
        blocks = module._code_blocks(path)
        for arm, names in arm_blocks.items():
            text = "\n".join("\n".join(source[blocks[b]["first_line"] - 1:
                                               blocks[b]["last_line"]]) for b in names)
            assert "eagle.deploy(" in text or "eagle.simulation(" in text, (
                f"{path.name}: {arm}'s snippet does not deploy its kernel")


def test_card_scripts_import_no_private_eagle_or_hawk_name():
    """The cards run on eagle's and hawk's public surface only: no
    ``eagle._x``/``hawk._x`` module, no ``_x`` name imported from either.

    RED: import ``eagle._until_done`` or ``hawk._core`` in a card and this
    fails naming the file and the line."""
    import ast

    for path in _CARD_SCRIPTS:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                names = [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                parts, names = None, [a.name for a in node.names]
            else:
                continue
            if parts is not None:
                if parts[0] not in ("eagle", "hawk"):
                    continue
                bad = [p for p in parts[1:] if p.startswith("_")]
                bad += [n for n in names if n.startswith("_")]
            else:
                bad = [n for n in names if n.split(".")[0] in ("eagle", "hawk")
                       and any(p.startswith("_") for p in n.split(".")[1:])]
            assert not bad, f"{path.name}:{node.lineno} imports a private name: {bad}"


def test_the_auto_issued_bytes_follow_eagle_s_policy():
    """The automatic arm's analytic bytes replay eagle's own policy: a uniform
    batch of n samples over S = 1000 steps launches 16, 32, then 64s (the last
    one the remainder), compacts once at the end, and issues the mapped count
    per launch.

    RED: count the automatic arm per step, or replay a constant k, and this
    fails."""
    import numpy as np

    module = _script()
    n, s = 1000, 1000
    running = np.full(s, n)
    sched = module._auto_schedule(running, n, s)
    ks = [k for _, k in sched]
    assert ks[:3] == [16, 32, 64] and sum(ks) == s and max(ks) == 64
    per = module.STEP_BYTES_ALL + module.MAP_BYTES + module.STEP_BYTES_ACTIVE
    want = per * n * len(ks) + n + module.COMPACT_BYTES_PER_SAMPLE * n
    assert module._auto_issued_bytes(running, n, n, s) == want


def test_k_steps_issued_bytes_count_per_launch():
    """The K-steps arm's analytic bytes are the step kernel's count over the
    launches' first steps: a uniform batch of n samples over S steps issues
    (33 + 40) B per sample per launch plus 1 B per finishing sample.

    RED: count the K-steps arm per step instead of per launch and this fails."""
    import numpy as np

    module = _script()
    n, s, k = 1000, 1000, module.STEPS_PER_LAUNCH
    running = np.full(s, n)
    launches = -(-s // k)
    got = module._kernel_issued_bytes(running[::k], n, n)
    assert got == (module.STEP_BYTES_ALL + module.STEP_BYTES_ACTIVE) * n * launches + n


def _gen_module():
    spec = importlib.util.spec_from_file_location("eagle_gen_perf_pages", _GEN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docs_page_includes_every_generated_family():
    """performance.md includes each family's generated fragment
    (``_generated/<family>.md``, written by ``docs/_tools/gen_perf_pages.py``
    from the cards on disk), not a raw per-device card path: the generated
    fragment is what {include}s the device's own ``card_<slug>.md``, so a new
    device needs no edit to this page.

    RED: drop one of the four {include} lines from performance.md and this
    fails naming the family."""
    page = _PAGE.read_text()
    module = _gen_module()
    for spec in module.FAMILIES:
        rel = f"_generated/{spec['key']}.md"
        assert rel in page, f"{_PAGE.name} does not include {rel}"


def test_generated_families_cover_every_committed_card():
    """Every committed ``card_*.json``/``cpu_card_*.json`` under
    ``benchmarks/{perf_card,rk78_card}/`` is reachable from performance.md
    through its family's generated fragment (run on the real cards, into a
    scratch directory -- the generated files themselves are never committed).

    RED: a card whose device the generator does not pick up (e.g. a slug the
    glob misses) fails here naming the family and the missing table."""
    module = _gen_module()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = pathlib.Path(tmp)
        module.generate(_REPO, out_dir)
        for spec in module.FAMILIES:
            card_dir = _REPO / spec["dir"]
            wanted = sorted(p.with_suffix(".md").name
                            for p in card_dir.glob(f"{spec['prefix']}*.json"))
            assert wanted, f"{spec['key']}: no committed cards under {card_dir}"
            text = (out_dir / f"{spec['key']}.md").read_text()
            for name in wanted:
                assert name in text, (
                    f"{spec['key']}.md does not include {name}")


# --------------------------------------------------------------------------- #
# Stale-prose gate: "Reading the results" / "Reading the RK7(8) results" cite
# numbers from the reference cards by hand; a ``stale-prose-gate`` HTML
# comment right under each heading records the md5 of every reference card it
# was written against, so a re-run of that card (same device, new numbers)
# cannot drift from the prose silently. A card recorded as ``PENDING``
# (the CPU cards, mid re-run as this gate was added) is deliberately not
# checked -- never treated as a silent pass.
# --------------------------------------------------------------------------- #
_STALE_GATE_RE = re.compile(r"<!-- stale-prose-gate:\n(.*?)-->", re.S)
_STALE_LINE_RE = re.compile(r"^(\S+\.json):\s*(\S+)\s*$", re.M)


def _stale_prose_sections():
    """{heading line: {card relpath: recorded md5-or-"PENDING"}} for every
    ``stale-prose-gate`` comment in performance.md, keyed by the nearest
    Markdown heading above it (for a readable failure message)."""
    text = _PAGE.read_text()
    sections = {}
    for m in _STALE_GATE_RE.finditer(text):
        before = text[:m.start()].splitlines()
        heading = next((ln for ln in reversed(before) if ln.startswith("#")), "?")
        sections[heading] = dict(_STALE_LINE_RE.findall(m.group(1)))
    return sections


def test_stale_prose_gate_present():
    sections = _stale_prose_sections()
    assert sections, f"{_PAGE.name} has no stale-prose-gate comment"
    for heading, entries in sections.items():
        assert entries, f"{heading}: stale-prose-gate comment has no card lines"


def test_stale_prose_gate_reference_cards_match():
    """A reference card's md5, recorded in a ``stale-prose-gate`` comment,
    must match the committed JSON: the hand-written prose under that heading
    was checked against exactly that card.

    RED: edit one byte of a committed reference card (e.g.
    ``card_quadro-p2000.json``) without updating the prose and its
    stale-prose-gate comment, and this fails naming the section and the
    card."""
    for heading, entries in _stale_prose_sections().items():
        for relpath, recorded in entries.items():
            if recorded == "PENDING":
                continue
            path = _REPO / relpath
            assert path.is_file(), f"{heading}: {relpath} is not a committed card"
            actual = hashlib.md5(path.read_bytes()).hexdigest()
            assert actual == recorded, (
                f"{heading}: {relpath} has changed (md5 {actual}, comment says "
                f"{recorded}) -- re-check the prose under this heading against "
                "the new card and update its stale-prose-gate comment")


def test_readme_performance_numbers_match_card():
    """The README's Performance table is copied from the quadro-p2000 card.

    Re-renders each headline cell with the script's own ``_fmt_time``/
    ``_fmt_rate``/``_fmt_pct`` (the same helpers ``render_markdown`` uses for
    ``card_quadro-p2000.md``) and requires that exact text to appear in
    ``README.md``, so a stale number left behind by a re-run of the benchmark
    is caught here rather than drifting silently.

    RED: edit one number in the README's Performance table, or re-run the
    benchmark without updating the README, and this fails naming the cell.
    """
    module = _script()
    card_path = _CARD_DIR / "card_quadro-p2000.json"
    assert card_path.is_file(), f"no committed {card_path.name} for the README table"
    card = json.loads(card_path.read_text())
    rows = {(r["distribution"], r["max_steps"], r["arm"]): r
            for r in card["results"] if r["n"] == 1_000_000}

    readme = _README.read_text()
    assert "## Performance" in readme, "README.md has no Performance section"
    section = readme.split("## Performance", 1)[1].split("## The RAPTOR family", 1)[0]

    for key in _README_ROWS:
        assert key in rows, f"card_quadro-p2000.json has no N=1,000,000 row for {key}"
        r = rows[key]
        fmt_time = module._fmt_time
        w = r["wall_s"]
        if "mean" in w:
            wall_cell = (f"{fmt_time(w['mean'])} "
                         f"({fmt_time(w['lo'])}\u2013{fmt_time(w['hi'])})")
        else:
            wall_cell = f"{fmt_time(w['median'])} ({fmt_time(w['iqr'])})"
        rate_cell = module._fmt_rate(r["sample_steps_per_s"])
        flops_cell = (f"{module._fmt_rate(r['useful_flops_per_s'])} "
                      f"({module._fmt_pct(r['fraction_of_peak_fp64'])})")
        for cell in (wall_cell, rate_cell, flops_cell):
            assert cell in section, (
                f"README.md's Performance section is missing {cell!r} for "
                f"{key} (re-derived from card_quadro-p2000.json) -- "
                "the table has drifted from the card")


# --------------------------------------------------------------------------- #
# The CPU card (benchmarks/perf_card/cpu_card.py -> cpu_card_<cpu>.{json,md})
# --------------------------------------------------------------------------- #
def _cpu_script():
    spec = importlib.util.spec_from_file_location(
        "eagle_cpu_card", _CARD_DIR / "cpu_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpu_cards():
    cards = sorted(_CARD_DIR.glob("cpu_card_*.json"))
    assert cards, f"no committed cpu_card_*.json under {_CARD_DIR}"
    return cards


def test_committed_cpu_tables_match_their_cards():
    """RED: edit one number in a committed ``cpu_card_*.md`` and this fails."""
    render = _cpu_script().render_markdown
    for card_path in _cpu_cards():
        card = json.loads(card_path.read_text())
        table_path = card_path.with_suffix(".md")
        assert table_path.is_file(), f"{card_path.name}: no rendered {table_path.name}"
        assert table_path.read_text() == render(card), (
            f"{table_path.name} is not the rendering of {card_path.name}: "
            "re-run benchmarks/perf_card/cpu_card.py instead of editing either file")


def test_cpu_cards_time_full_blocks():
    """Every timed repetition of a short run spans at least half MIN_TIMED_S
    (its run count comes from a calibration block, and every block starts
    after its own RAMP_S of untimed runs, so a half-length block already sits
    on a ramped clock; a block of a few runs without the ramp read slow). The
    busy cores during each settle pause and the clock at each timer start are
    recorded per row as diagnostics."""
    for card_path in _cpu_cards():
        card = json.loads(card_path.read_text())
        for r in card["results"]:
            inner = r["inner_runs"]
            if inner > 1:
                w = r["wall_s"]
                span = inner * (w["mean"] if "mean" in w else w["median"])
                assert span >= 0.5 * _cpu_script().MIN_TIMED_S, (
                    f"{card_path.name}: {r['arm']} at {r['distribution']}/{r['max_steps']}/{r['n']} "
                    f"timed {inner} runs = {span:.3g} s per repetition, under MIN_TIMED_S")
            assert "busy_cores_during_settle" in r and "cpu_mhz_at_timer_start" in r


@pytest.mark.skip(reason="the committed CPU card was measured on a shared 4-core host whose 8-thread "
                         "arms use every hardware thread; re-enabled with the dedicated-hardware card")
def test_cpu_cards_agree_with_their_reruns():
    """The reference arm's re-runs across arm groups agree within 1.10x."""
    for card_path in _cpu_cards():
        card = json.loads(card_path.read_text())
        for rr in card.get("reference_reruns") or []:
            assert rr["max_over_min"] <= 1.10, (
                f"{card_path.name}: the reference arm's re-runs differ {rr['max_over_min']:.3g}x "
                f"at {rr['distribution']}/{rr['max_steps']}/{rr['n']}")


def test_cpu_cards_name_their_scripts_and_are_complete():
    """Each CPU card was produced by the committed cpu_card.py on the committed
    perf_card.py workload, and every (configuration, N, arm) cell is either a
    result row with a memory row or a dropped cell with its stated reason."""
    module = _cpu_script()
    for card_path in _cpu_cards():
        card = json.loads(card_path.read_text())
        assert card["script"] == "benchmarks/perf_card/cpu_card.py"
        for key, name in (("script_md5", "cpu_card.py"),
                          ("workload_script_md5", "perf_card.py")):
            md5 = hashlib.md5((_CARD_DIR / name).read_bytes()).hexdigest()
            assert card[key] == md5, (
                f"{card_path.name} was produced by a different {name}: re-run "
                "benchmarks/perf_card/cpu_card.py so the card and its producer match")
        assert card["method"]["repetitions"] >= 5
        assert set(card["arms"]) == set(module.ARMS), f"{card_path.name}: arms differ"
        assert card["configs"] == [{"distribution": d, "max_steps": s}
                                   for d, s in module.pc.CONFIGS]
        assert card["ns"] == list(module.pc.NS)
        expected = {(c["distribution"], c["max_steps"], n, arm)
                    for c in card["configs"] for n in card["ns"]
                    for arm in card["arms"]}
        got = {(r["distribution"], r["max_steps"], r["n"], r["arm"])
               for r in card["results"]}
        dropped = {(d["distribution"], d["max_steps"], d["n"], d["arm"])
                   for d in card["dropped"]}
        assert not got & dropped, f"{card_path.name}: a cell is both run and dropped"
        assert got | dropped == expected, (
            f"{card_path.name}: result rows plus dropped cells do not cover the matrix")
        for d in card["dropped"]:
            assert d["reason"] == module._included(
                d["arm"], d["n"], d["distribution"], d["max_steps"]), (
                f"{card_path.name}: dropped cell {d} does not match the script's rule")
        mem = {(r["distribution"], r["max_steps"], r["n"], r["arm"])
               for r in card["memory"]["rows"]}
        assert mem == got, f"{card_path.name}: memory rows do not cover the result rows"
        for r in card["results"]:
            assert r["max_abs_diff_vs_reference"] <= module.pc.ATOL
        comp = card["compile_time"]
        covered = ({r["arm"] for r in comp["rows"]} | set(comp["none"])
                   | set(comp.get("shared", {})))
        assert covered == set(module.ARMS), (
            f"{card_path.name}: compile-time table does not cover every arm "
            f"(missing {sorted(set(module.ARMS) - covered)})")


def test_cpu_cards_record_their_code_snippets():
    module = _cpu_script()
    for card_path in _cpu_cards():
        card = json.loads(card_path.read_text())
        _check_code_record(card, card_path, {
            name: module.pc._code_blocks(_CARD_DIR / name)
            for name in ("perf_card.py", "cpu_card.py")}, prefixed=True)


def test_cpu_cards_record_hawk_flags_from_hawk():
    """The eagle arms' compile flags are the ones hawk reported (its host
    profile's code-generation flags lead the recorded recipe), not a copy.

    RED: hardcode a flag list in ``cpu_card.py``'s facts and this fails."""
    for card_path in _cpu_cards():
        card = json.loads(card_path.read_text())
        for arm in ("eagle_term8", "eagle_compact8"):
            facts = card["arm_info"][arm]["facts"]
            prof = facts["host_profile"]
            assert prof["profile"] in prof["profiles"], (
                f"{card_path.name}: {arm} profile")
            codegen = prof["codegen_flags"]
            assert facts["compile_flags"][:len(codegen)] == codegen, (
                f"{card_path.name}: {arm}'s recorded flags do not start with hawk's "
                f"{prof['profile']} code-generation flags")


def _card_common():
    spec = importlib.util.spec_from_file_location(
        "eagle_card_common_under_test", _REPO / "benchmarks" / "_card_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trimmed_statistic_and_center_accessor():
    """``trimmed`` drops the single highest and lowest of 12 runs and averages
    the middle 10; ``center`` reads the mean, falling back to the median.

    RED: keep all 12 in the mean (or drop two per side) and the mean, lo or
    hi assertion below fails."""
    cc = _card_common()
    values = [5.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 100.0]
    t = cc.trimmed(values)
    assert t["mean"] == pytest.approx(65 / 10)
    assert t["mean"] == pytest.approx(6.5)
    assert (t["lo"], t["hi"]) == (2.0, 11.0)
    assert t["median"] == pytest.approx(6.5)
    assert t["samples"] == values
    assert cc.center(t) == t["mean"]
    assert cc.center({"median": 3.0, "iqr": 1.0}) == 3.0


# ---- cool-down gate and clock record (fake NVML, no GPU) -------------------- #
class _FakeClock:
    """A clock that only moves when the gate sleeps."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def _gate_script(monkeypatch, temps, util=0):
    pf = _script()
    clk = _FakeClock()
    seq = iter(temps)
    last = [temps[-1]]

    def temp():
        last[0] = next(seq, last[0])
        return last[0]

    monkeypatch.setattr(pf, "_now", clk.now)
    monkeypatch.setattr(pf, "_sleep", clk.sleep)
    monkeypatch.setattr(pf, "_nvml_temp", temp)
    monkeypatch.setattr(pf, "_nvml_util", lambda: util)
    monkeypatch.setitem(pf._NVML, "handle", object())
    monkeypatch.setitem(pf._NVML, "baseline_c", 40.0)
    return pf, clk


def test_gate_waits_until_the_temperature_falls(monkeypatch):
    pf, clk = _gate_script(monkeypatch, [60.0, 55.0, 50.0, 44.0, 43.0, 42.0])
    g = pf._cool_down_gate()
    assert g["gate_timed_out"] is False
    assert g["temp_c"] == 43.0  # baseline 40 + 3
    assert g["waited_s"] == pytest.approx(4 * pf.GATE_POLL_S)


def test_gate_waits_for_zero_utilisation(monkeypatch):
    pf, _ = _gate_script(monkeypatch, [40.0], util=35)
    g = pf._cool_down_gate()
    assert g["gate_timed_out"] is True


def test_gate_times_out_at_gate_max_s(monkeypatch):
    pf, clk = _gate_script(monkeypatch, [70.0])
    g = pf._cool_down_gate()
    assert g["gate_timed_out"] is True
    assert g["temp_c"] == 70.0
    assert g["waited_s"] == pytest.approx(pf.GATE_MAX_S, abs=pf.GATE_POLL_S)


def test_clock_and_throttle_are_read_after_the_timer_stops(monkeypatch):
    pf, _ = _gate_script(monkeypatch, [40.0])
    events = []
    clocks = iter([1500.0, 585.0, 1590.0, 1000.0, 900.0])

    def fake_timed(arm, inp, is_gpu):
        events.append("timer_stop")
        return 0.002, 0.003, None

    def fake_clock():
        events.append("clock")
        return next(clocks)

    masks = iter([0, 0x4, 0x4 | 0x20, 0x1, 0x8])
    monkeypatch.setattr(pf, "_timed", fake_timed)
    monkeypatch.setattr(pf, "_nvml_sm_mhz", fake_clock)
    monkeypatch.setattr(pf, "_nvml_throttle_mask", lambda: next(masks))
    monkeypatch.setattr(pf, "RAMP_S", 0.0)
    meta = {}
    walls, _ = pf._time_arm_block({}, {}, True, 5, meta=meta)
    # one ramp run, then each timed run is followed by its clock read
    assert events == ["timer_stop"] + ["timer_stop", "clock"] * 5
    assert meta["sm_mhz"] == [1500.0, 585.0, 1590.0, 1000.0, 900.0]
    assert meta["throttle"][1] == ["power cap"]
    assert meta["throttle"][2] == ["power cap", "SW thermal slowdown"]
    assert meta["gate"]["gate_timed_out"] is False
    stat = pf._arm_stat(walls, meta)
    per_run = sorted(0.002 * m * 1e6 for m in meta["sm_mhz"])
    assert stat["cycles"]["mean"] == pytest.approx(sum(per_run[1:-1]) / 3)
    assert stat["wall_s" if False else "mean"] == pytest.approx(0.002)
    assert pf._clock_range(stat) == "585–1590"


def _synthetic_card(pf, with_clock):
    card = json.loads(next(iter(_cards())).read_text())
    for row in card["results"]:
        row["wall_s"].pop("sm_mhz", None)
    fp = {a: {"mean": 1e-3, "lo": 9e-4, "hi": 1.1e-3, "median": 1e-3, "iqr": 1e-4,
              "samples": [1e-3] * 5} for a in pf.FP32_ARMS}
    if with_clock:
        for a, mhz in zip(pf.FP32_ARMS, (600.0, 1500.0, 1500.0)):
            fp[a]["sm_mhz"] = [mhz] * 5
            fp[a]["throttle"] = [["power cap"]] * 5
            c = 1e-3 * mhz * 1e6
            fp[a]["cycles"] = {"mean": c, "lo": c, "hi": c, "median": c, "iqr": 0.0,
                               "samples": [c] * 5}
    card["fp32"] = {
        "max_steps": 1000, "method": "m", "cap_fixture": None,
        "rows": [{"distribution": "uniform", "n": 10000, "max_steps": 1000,
                  "wall_s": fp, "eagle_run_mode": "persist"}]}
    return card


def test_fp32_table_has_the_cycles_column_only_with_clock_data():
    pf = _script()
    with_clock = pf.render_markdown(_synthetic_card(pf, True))
    assert "eagle / Warp (cycles)" in with_clock
    assert "600–600 MHz" in with_clock
    assert "0.4× |" in with_clock  # 600 / 1500 cycles ratio
    without = pf.render_markdown(_synthetic_card(pf, False))
    assert "(cycles)" not in without and "MHz" not in without.split("### FP32")[1]


def test_no_nvml_skips_the_gate_and_the_clock(monkeypatch):
    # the script imports pynvml for its constants before it checks the handle
    pytest.importorskip("pynvml")
    pf = _script()
    monkeypatch.setattr(pf.cc, "_nvml_handle", lambda bus: None)
    rec = pf._gate_init("0000:00:00.0")
    assert rec["skipped"] and rec["idle_baseline_c"] is None
    assert pf._cool_down_gate() is None and pf._clock_record() is None
    monkeypatch.setattr(pf, "_timed", lambda *a: (0.001, 0.002, None))
    monkeypatch.setattr(pf, "RAMP_S", 0.0)
    meta = {}
    walls, _ = pf._time_arm_block({}, {}, True, 5, meta=meta)
    assert meta == {}
    stat = pf._arm_stat(walls, meta)
    assert "cycles" not in stat and "sm_mhz" not in stat
    assert pf._gate_sentence() == ""


# --------------------------------------------------------------------------- #
# The README's CPU block (<!-- cpu-cards:begin --> ... <!-- cpu-cards:end -->):
# one heading and one compact table per cpu_card_*.json, pinned the way the
# GPU rows above are.
# --------------------------------------------------------------------------- #
#: arm -> N = 1,000,000 rows shown; kept in lockstep with
#: ``tools/sync_readme_cards.py``'s CPU_ARMS by hand (this test catches drift).
_README_CPU_ARMS = ("eagle_term8", "numba_prange", "jax_shard8", "torch_masked",
                    "numpy_masked")


def _readme_cpu_block():
    text = _README.read_text()
    begin, end = "<!-- cpu-cards:begin -->", "<!-- cpu-cards:end -->"
    assert begin in text and end in text, "README.md has no cpu-cards block"
    start = text.index(begin) + len(begin)
    return text[start:text.index(end, start)]


def test_readme_cpu_block_matches_every_cpu_card():
    """Each CPU card is named by a heading (model, cores, card date) and its
    rows (spread / uniform S=1000 wall at the largest N, and the ratio to
    eagle on 8 threads) are re-derived here from the card JSON.

    RED: edit a number in the cpu-cards block, or re-run a CPU card without
    re-running ``tools/sync_readme_cards.py``, and this names the cell."""
    block = _readme_cpu_block()
    cards = _cpu_cards()
    assert cards, "no committed cpu_card_*.json"
    module = _card_common()
    for path in cards:
        card = json.loads(path.read_text())
        cpu = card["cpu"]
        heading = (f"On a CPU alone: {cpu['model']}, {cpu['cores']} cores, "
                   f"card of {card['generated_utc'][:10]}")
        assert heading in block, f"README cpu block is missing {heading!r}"
        n = max(card["ns"])
        rows = {(r["distribution"], r["max_steps"], r["arm"]): r
                for r in card["results"] if r["n"] == n}
        base = module.center(rows[("spread", 1000, "eagle_term8")]["wall_s"])
        for arm in _README_CPU_ARMS:
            sp = module.center(rows[("spread", 1000, arm)]["wall_s"])
            un = module.center(rows[("uniform", 1000, arm)]["wall_s"])
            row = (f"| {card['arm_labels'][arm]} | {module.fmt_time(sp)} "
                   f"| {module.fmt_time(un)} | {sp / base:.3g}× |")
            assert row in block, (
                f"README cpu block is missing {row!r} (re-derived from {path.name}) "
                "-- the table has drifted from the card")


def test_readme_stale_prose_gate_lists_the_cpu_cards():
    """The README's stale-prose-gate comment carries every committed CPU card's
    md5 (its CPU block and the prose around it were checked against them)."""
    m = _STALE_GATE_RE.search(_README.read_text())
    assert m, "README.md has no stale-prose-gate comment"
    entries = dict(_STALE_LINE_RE.findall(m.group(1)))
    for path in _cpu_cards():
        rel = f"benchmarks/perf_card/{path.name}"
        assert rel in entries, f"stale-prose-gate comment does not list {rel}"
        assert entries[rel] == hashlib.md5(path.read_bytes()).hexdigest(), (
            f"{rel} changed since the README's gate comment was written")
