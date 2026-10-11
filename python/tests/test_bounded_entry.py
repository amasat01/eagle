# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The register bound: kept only when the compiler proves it spill-free.

The compiler's answers are synthetic here (a stand-in image loader), so every
edge and every refusal is exercised without a GPU; the GPU row that asks the
real compiler is in ``test_until_done_auto.py``.
"""

from __future__ import annotations

import importlib
import threading
from types import SimpleNamespace

import pytest

from eagle import _bounded

launch = importlib.import_module("eagle.launch")

A100 = {"multiProcessorCount": 108, "major": 8, "minor": 0, "regsPerMultiprocessor": 65536,
        "maxThreadsPerMultiProcessor": 2048}


class _Fn:
    def __init__(self, regs, local):
        self.attributes = {"num_regs": regs, "local_size_bytes": local}


def _decide(monkeypatch, plain, bounded, loaded=None):
    """``_bounded._decide`` over an A100 with the given plain entry and the
    compiler's answer ``bounded`` ((regs, local), or ``None`` for no recompile)."""
    monkeypatch.setattr(launch, "_device_props", lambda: A100)
    asked = []

    def load(loaded_, name, cap):
        asked.append(cap)
        return None if bounded is None else _Fn(*bounded)

    monkeypatch.setattr(_bounded, "_load_bounded", load)
    fn, report = _bounded._decide(loaded or SimpleNamespace(), "k_range", _Fn(*plain))
    return fn, report, asked


@pytest.mark.parametrize("plain,edge", [((46, 0), 40), ((41, 0), 40), ((48, 0), 40),
                                        ((174, 0), 168), ((169, 0), 168), ((176, 0), 168),
                                        ((174, 24), 168)])
def test_kept_when_the_compiler_fits_the_edge_with_the_same_local_size(monkeypatch, plain, edge):
    fn, report, asked = _decide(monkeypatch, plain, (edge, plain[1]))
    assert asked == [edge]
    assert report.bounded and report.edge == edge and report.note == ""
    assert report.regs == edge and report.local_bytes == plain[1]
    assert isinstance(fn, _Fn) and fn.attributes["num_regs"] == edge


@pytest.mark.parametrize("plain,edge", [((46, 0), 40), ((174, 0), 168), ((174, 24), 168)])
def test_refused_when_the_bound_adds_any_local_memory(monkeypatch, plain, edge):
    for extra in (8, 16):  # one spilled double is already too many
        fn, report, _ = _decide(monkeypatch, plain, (edge, plain[1] + extra))
        assert not report.bounded and report.edge == edge
        assert fn.attributes["num_regs"] == plain[0] and "B local" in report.note


@pytest.mark.parametrize("plain,edge", [((46, 0), 40), ((174, 0), 168)])
def test_refused_when_the_compiler_cannot_reach_the_edge(monkeypatch, plain, edge):
    fn, report, _ = _decide(monkeypatch, plain, (edge + 2, plain[1]))
    assert not report.bounded and report.regs == plain[0]
    assert fn.attributes["num_regs"] == plain[0]


def test_nothing_is_recompiled_out_of_reach_of_an_edge(monkeypatch):
    for plain in ((186, 0), (188, 0), (128, 0), (255, 0)):
        _, report, asked = _decide(monkeypatch, plain, (168, 0))
        assert asked == [] and not report.bounded and report.edge == 0, plain


def test_a_cubin_image_is_left_alone_and_says_so(monkeypatch):
    fn, report, asked = _decide(monkeypatch, (46, 0), None)
    assert asked == [40] and not report.bounded and "cubin" in report.note
    assert fn.attributes["num_regs"] == 46


def test_report_carries_the_residency_the_entry_leaves(monkeypatch):
    _, plain_report, _ = _decide(monkeypatch, (46, 0), (46, 8))
    assert plain_report.resident_threads == 1280
    _, kept_report, _ = _decide(monkeypatch, (46, 0), (40, 0))
    assert kept_report.resident_threads == 1536
    _, rk78_report, _ = _decide(monkeypatch, (188, 0), None)
    assert rk78_report.resident_threads == 256


def test_decided_once_per_entry_under_concurrent_use(monkeypatch):
    monkeypatch.setattr(launch, "_device_props", lambda: A100)
    calls = []

    def load(loaded_, name, cap):
        calls.append(name)
        return _Fn(40, 0)

    monkeypatch.setattr(_bounded, "_load_bounded", load)
    loaded = SimpleNamespace()
    plain = _Fn(46, 0)
    got = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        got.append(_bounded.entry(loaded, "k_persist", plain))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["k_persist"] and len({id(f) for f in got}) == 1
