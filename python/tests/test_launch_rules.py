# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The launch rules the runner applies: the compaction cadence on both faces,
the crossover probe that no step fill suppresses, the chunk and the ELF reader
behind the bound-on-demand code-size test. The rules themselves are pinned in
C++ (``tests/test_Launch.cpp``); these rows pin the Python faces and the
runner's use of them."""

from __future__ import annotations

import subprocess
import threading

import numpy as np
import pytest

from eagle import _bounded, _core
from eagle import _until_done as ud

# h = newly / (live * k) = 0.0022 at k = 100
_LIVE, _K = 1_000_000, 100
_NEWLY = 220_000


@pytest.mark.parametrize("rate, want", [(1.3, {32}), (8.4, {64, 128}), (44.0, {256})])
def test_next_k_host_face_rows(rate, want):
    assert ud._next_k(_K, _NEWLY, _LIVE, 1000, ud.AUTO_K_MIN, rate) in want


def test_next_k_holds_its_bounds_and_runs_long_when_nothing_finishes():
    assert ud._next_k(16, 0, 1000, 64, 8, 8.0) == 64
    assert ud._next_k(16, 0, 1000, 64, 64, 8.0) == 64      # skipped compaction: pinned
    assert ud._next_k(_K, _NEWLY, _LIVE, 100, 8, 44.0) == 100
    assert ud._next_k(16, 5, 0, 64, 8, 8.0) == 8


@pytest.mark.parametrize("rate, want", [(1.3, {32}), (8.4, {64, 128}), (44.0, {256})])
def test_host_policy_cells_take_the_same_rows(rate, want):
    """One host policy step over numpy cells: ``k`` follows R in the rate cell."""
    cells = np.zeros(5, dtype=np.int64)
    cells[ud._K], cells[ud._CAP] = _K, 10 ** 6
    cells[ud._R] = round(rate * ud.RATE_SCALE)
    finished, total = np.array([_NEWLY], dtype=np.uint32), np.array([_LIVE + _NEWLY], dtype=np.uint32)
    word, go = np.array([_K], dtype=np.int64), np.zeros(1, dtype=np.uint32)
    hist = np.zeros(1001, dtype=np.int64)
    # newly = fin - finished_at_launch = _NEWLY; live = total - fin = _LIVE
    ud._host_policy(cells, finished, total, word, go, hist, 1000)
    assert int(cells[ud._K]) in want and go[0] == 1


@pytest.mark.gpu
@pytest.mark.parametrize("rate", [1.3, 8.4, 44.0, 0.5, 200.0])
@pytest.mark.parametrize("newly", [0, 3_000, 220_000, 900_000])
def test_device_policy_kernel_matches_the_host_face(rate, newly):
    """The device kernel and ``_next_k`` round the same way."""
    cp = pytest.importorskip("cupy")
    kernel = ud._policy_kernel(1000, ud.AUTO_K_MIN)
    cells = cp.zeros(5, dtype=cp.int64)
    cells[ud._K], cells[ud._CAP] = _K, 10 ** 6
    cells[ud._R] = round(rate * ud.RATE_SCALE)
    finished = cp.asarray([newly], dtype=cp.uint32)
    total = cp.asarray([_LIVE + newly], dtype=cp.uint32)
    word, go = cp.asarray([_K], dtype=cp.int64), cp.zeros(1, dtype=cp.uint32)
    hist = cp.zeros(1001, dtype=cp.int64)
    kernel((1,), (1,), (finished, total, cells, word, go, hist))
    want = ud._next_k(_K, newly, _LIVE, 1000, ud.AUTO_K_MIN, round(rate * ud.RATE_SCALE) / ud.RATE_SCALE)
    assert int(cells[ud._K].get()) == want


def test_the_cost_model_prices_a_compaction_in_steps():
    a100 = (8, 0, 108, 2, 1_410_000)
    rate = _core.band_rate(*a100, True, 1_000_000, 33.0)
    assert 5.0 < rate < 20.0
    assert _core.band_rate(8, 0, 108, 2, 0, True, 1_000_000, 33.0) == 0.0


def test_the_chunk_follows_the_mean_steps():
    assert [_core.persist_chunk(m) for m in (1, 23, 100, 130, 500)] == [4, 8, 32, 32, 32]


# --------------------------------------------------------------------------- #
# The crossover probe: no step fill suppresses it, and it settles in three runs
# --------------------------------------------------------------------------- #
def _runner(pick="persist"):
    runner = ud.Runner.__new__(ud.Runner)
    runner._fast_pick, runner._fast_mode = pick, pick
    runner._fast_wall, runner._fast_runs = {}, 0
    runner._fast_last = runner._fast_prior = None
    runner._fast_prep = {}
    return runner


def _simulate(runner, walls, fill, runs=3):
    """Drive ``_pick_entry``/``_note_run`` with simulated launch walls."""
    seen = []
    for _ in range(runs):
        mode, probe = runner._pick_entry()
        seen.append((mode, probe))
        runner._note_run(mode, walls[mode], 1000, 100_000, fill)
    return seen


@pytest.mark.parametrize("fill", [0.05, 0.23, 0.59, 0.6, 1.0])
def test_a_low_step_fill_no_longer_suppresses_the_probe(fill):
    runner = _runner()
    seen = _simulate(runner, {"persist": 2.0, "fused_one": 1.0}, fill)
    assert seen[:2] == [("persist", False), ("fused_one", True)]
    assert set(runner._fast_wall) == {"persist", "fused_one"}


@pytest.mark.parametrize("walls, winner", [({"persist": 2.0, "fused_one": 1.0}, "fused_one"),
                                           ({"persist": 1.0, "fused_one": 2.0}, "persist")])
def test_the_pick_settles_on_the_faster_entry_by_the_third_run(walls, winner):
    runner = _runner()
    seen = _simulate(runner, walls, 0.23)
    assert seen[2] == (winner, False) and runner._fast_pick == winner


def test_a_new_readback_times_both_entries_again():
    runner = _runner()
    _simulate(runner, {"persist": 2.0, "fused_one": 1.0}, 0.23)
    runner._note_run("fused_one", None, 400, 100_000, 0.23)   # a settled run, new steps
    assert runner._fast_wall == {}
    assert runner._pick_entry() == ("persist", True)           # the other entry is timed first


# --------------------------------------------------------------------------- #
# The code-size reader the bound test uses
# --------------------------------------------------------------------------- #
def test_elf_section_size_reads_a_real_object(tmp_path):
    gcc = subprocess.run(["which", "gcc"], capture_output=True, text=True).stdout.strip()
    if not gcc:
        pytest.skip("no C compiler to make an ELF object")
    src = tmp_path / "k.c"
    src.write_text("int k(int x) { return x * 3 + 1; }\nint other(int x) { return x - 2; }\n")
    obj = tmp_path / "k.o"
    subprocess.run([gcc, "-O1", "-ffunction-sections", "-c", str(src), "-o", str(obj)], check=True)
    image = obj.read_bytes()
    assert _bounded.elf_section_size(image, ".text.k") > 0
    assert _bounded.elf_section_size(image, ".text.other") > 0
    assert _bounded.elf_section_size(image, ".text.absent") == 0


def test_elf_section_size_is_zero_for_anything_that_is_not_elf64():
    assert _bounded.elf_section_size(b"", ".text.k") == 0
    assert _bounded.elf_section_size(b".version 7.0\n.target sm_80\n", ".text.k") == 0
    assert _bounded.elf_section_size(b"\x7fELF" + b"\x01" * 80, ".text.k") == 0


def test_the_rules_are_safe_to_call_from_many_threads():
    got, errors = [], []

    def worker():
        try:
            for _ in range(200):
                got.append((ud._next_k(_K, _NEWLY, _LIVE, 1000, 8, 8.4), _core.persist_chunk(23.0)))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(set(got)) == 1

