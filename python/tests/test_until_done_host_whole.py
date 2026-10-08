"""The whole-run host path: an automatic kernel's run as ONE team pass.

When the tiled host policy's entry exports its run-to-completion twin
(``<entry>_run``, :func:`eagle._host_loop.host_run`) and the run neither
compacts nor maps, :func:`eagle._host_loop.run_host` makes one team pass over
the batch (or one serial call for a small batch) instead of one team launch
per ``steps_max`` steps. What is pinned:

* **Bit identity with the per-launch tiled loop**: the planes, the mask and
  the counter, and the report (launches, ``launches_by_k``, steps, exact
  steps, finished), for N = 1, 7, 8, 9, 1000 and 100k, uniform and spread
  stops, and a batch the cap stops (``max_steps`` hit, the last round cut);
  the RK7(8) attempt kernel on a mixed-eccentricity batch, finishing and
  capped. The per-launch arm is the same artifact with the twin hidden.
* **One pass**: exactly one team call per run, serial below
  ``threads * RUN_LANES * RUN_TILES_PER_THREAD`` samples; a compacting or
  mapped artifact keeps the per-launch loop.
* **The schedule rule** (:func:`eagle._host_loop.whole_run_schedule`).

Every build here pins hawk's EXACT ``native`` host profile (the fixture
imported from ``test_until_done_auto``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_until_done import _DT, _planes
from test_until_done_auto import _S, _batch, _exact_host_profile, auto  # noqa: F401

import eagle
from eagle import _host_loop
from eagle import exec as eexec


def _per_launch(monkeypatch):
    """Hide the run-to-completion twin: the tiled loop launches per round."""
    monkeypatch.setattr(_host_loop, "host_run", lambda entry: None)


def _go(plan, inp, max_steps):
    g = _planes(inp, np)
    n = g["x"].shape[0]
    term = np.zeros(n, dtype=bool)
    runner = eagle.until_done(plan, max_steps=max_steps, dt=_DT, terminated=term, **g)
    report = runner.run()
    return report, {**{key: g[key].copy() for key in ("x", "v", "k")},
                    "terminated": term.copy(), "finished": int(runner.finished[0])}


def _same(a, b):
    (ra, ga), (rb, gb) = a, b
    for key in ("x", "v", "k", "terminated"):
        assert ga[key].tobytes() == gb[key].tobytes(), key
    assert ga["finished"] == gb["finished"]
    for field in ("n", "finished", "done", "launches", "steps", "launches_by_k",
                  "exact_steps", "compactions"):
        assert getattr(ra, field) == getattr(rb, field), field


@pytest.mark.parametrize("n", [1, 7, 8, 9, 1000, 100_000])
@pytest.mark.parametrize("which,cap", [("uniform", 10 * _S), ("spread", 10 * _S),
                                       ("spread", 150), ("never", 150)])
@pytest.mark.parametrize("label", ["plain", "default"])
def test_the_whole_run_is_bit_identical_to_the_per_launch_loop(
        auto, label, which, cap, n, monkeypatch):  # noqa: F811
    plan = auto[(label, "host")]
    inp = _batch(n, which)
    whole = _go(plan, inp, cap)
    with monkeypatch.context() as mp:
        _per_launch(mp)
        launched = _go(plan, inp, cap)
    _same(whole, launched)
    report = whole[0]
    assert (whole[1]["k"] <= cap).all() and report.steps <= cap
    if which == "never":
        assert not report.done and (whole[1]["k"] == cap).all() and report.steps == cap
    elif cap >= _S:
        assert report.done


def test_the_whole_run_is_one_team_pass(auto, monkeypatch):  # noqa: F811
    """One team call per run (serial for a small batch), against
    ``ceil(_S / 64)`` launches of the per-launch loop."""
    calls = []
    real_run, real_serial = eexec._HostTeam.run, eexec._HostTeam.run_serial

    def run(self, *a, **kw):
        calls.append("team")
        return real_run(self, *a, **kw)

    def serial(self, *a, **kw):
        calls.append("serial")
        return real_serial(self, *a, **kw)

    monkeypatch.setattr(eexec._HostTeam, "run", run)
    monkeypatch.setattr(eexec._HostTeam, "run_serial", serial)
    monkeypatch.setattr(_host_loop, "_team", lambda runner: 8)  # the team, whatever its policy
    plan = auto[("plain", "host")]
    small = 8 * _host_loop.RUN_LANES * _host_loop.RUN_TILES_PER_THREAD - 1
    for n, want in ((small, ["serial"]), (small + 1, ["team"])):
        calls.clear()
        report, _ = _go(plan, _batch(n, "uniform"), 10 * _S)
        assert calls == want, n
        assert report.launches == -(-_S // 64)
    calls.clear()
    _per_launch(monkeypatch)
    report, _ = _go(plan, _batch(small + 1, "uniform"), 10 * _S)
    assert calls == ["team"] * report.launches and report.launches == -(-_S // 64)


def test_a_mapped_artifact_keeps_the_per_launch_loop(auto, monkeypatch):  # noqa: F811
    seen = []
    monkeypatch.setattr(_host_loop, "_run_whole",
                        lambda *a, **k: seen.append(1) or 0)
    plan = auto[("map", "host")]
    runner = eagle.until_done(plan, max_steps=10 * _S, dt=_DT,
                              **_planes(_batch(300, "spread"), np))
    runner.run()
    assert not seen


def test_the_whole_run_schedule():
    """Serial below ``threads * RUN_LANES * RUN_TILES_PER_THREAD``; above it,
    tiles of about ``n / (threads * RUN_TILES_PER_THREAD)`` samples (never
    below the team's floor, never above its default)."""
    size = eexec.HostTeam.tile_size
    lanes, per = _host_loop.RUN_LANES, _host_loop.RUN_TILES_PER_THREAD
    small = 8 * lanes * per
    assert _host_loop.whole_run_schedule(small - 1, 8) == (True, 0)
    for n in (small, 4000, 20_000):
        serial, nbytes = _host_loop.whole_run_schedule(n, 8)
        assert not serial and size(nbytes) <= max(n // (8 * per), size(1 << 30)), n
    assert _host_loop.whole_run_schedule(1_000_000, 8) == (False, 0)


# --------------------------------------------------------------------------- #
# RK7(8)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rk78(tmp_path_factory):
    from test_until_done_host import _rk78_card

    card = _rk78_card()
    _, host_plan, _ = card._build(tmp_path_factory.mktemp("rk78"), targets=("host",))
    return card, host_plan


@pytest.mark.parametrize("n", [1, 9, 1000])
@pytest.mark.parametrize("cap", [None, 40])
def test_rk78_whole_run_is_bit_identical(rk78, n, cap, monkeypatch):
    card, plan = rk78
    inp = card._inputs(n)
    cap = card.MAX_ATTEMPTS if cap is None else cap

    def go():
        g = {"s": np.stack([inp[k] for k in card.STATE_NAMES]), "t": np.zeros(n),
             "h": np.full(n, card.H0), "n_acc": np.zeros(n), "n_rej": np.zeros(n)}
        term = np.zeros(n, dtype=bool)
        report = eagle.run_until_done(plan, max_steps=cap, t_final=card.T_FINAL,
                                      terminated=term, **g)
        return report, g, term

    rw, gw, tw = go()
    with monkeypatch.context() as mp:
        _per_launch(mp)
        rl, gl, tl = go()
    for key in gw:
        assert gw[key].tobytes() == gl[key].tobytes(), key
    assert tw.tobytes() == tl.tobytes()
    # the launch counts are the host policy's own choices, made from measured
    # walls, so they match only when the two runs' walls do; the results must
    # match whatever the policy chose
    for field in ("finished", "done", "exact_steps"):
        assert getattr(rw, field) == getattr(rl, field), field
    if cap == 40:
        assert not rw.done and (gw["n_acc"] + gw["n_rej"]).max() == 40


def test_entry_memos_stay_bounded(monkeypatch):
    """The per-entry tile and run-twin answers are cleared when full, so a
    process that loads many kernels does not grow them forever."""
    monkeypatch.setattr(_host_loop, "_TILES", {})
    monkeypatch.setattr(_host_loop, "_RUNS", {})
    for entry in range(1, 3 * _host_loop._MEMO_LIMIT):
        _host_loop.host_tile(entry)
        _host_loop.host_run(entry)
    assert 0 < len(_host_loop._TILES) <= _host_loop._MEMO_LIMIT
    assert 0 < len(_host_loop._RUNS) <= _host_loop._MEMO_LIMIT


def test_host_team_size_follows_the_work(monkeypatch):
    """A short run uses one thread per physical core, a long one every
    logical CPU; with no hyper-threading the two agree."""
    assert 1 <= _host_loop.physical_cores() <= (os.cpu_count() or 1)
    monkeypatch.setattr(_host_loop, "physical_cores", lambda: 4)
    small = _host_loop.SMT_MIN_WORK // 100 - 1
    assert _host_loop.host_team_size(small, 100, threads=8) == 4
    assert _host_loop.host_team_size(_host_loop.SMT_MIN_WORK, 1, threads=8) == 8
    assert _host_loop.host_team_size(small, 100, threads=2) == 2


def test_host_run_sets_the_team_and_restores_the_callers(auto, monkeypatch):
    """A host run sets its team for the run's length only; ``host_threads``
    forces it; the caller's OpenMP setting is back afterwards."""
    omp = _host_loop._omp_runtime()
    if omp is None:
        pytest.skip("no OpenMP runtime to steer in this build")
    get, put = omp
    seen = []
    real = _host_loop._run_host_loop
    monkeypatch.setattr(_host_loop, "_run_host_loop",
                        lambda runner, threads: (seen.append((threads, get())),
                                                 real(runner, threads)))
    put(7)
    g = _planes(_batch(64, "uniform"), np)
    runner = eagle.until_done(auto[("plain", "host")], max_steps=_S, dt=_DT, **g)
    runner.run()
    runner.reset()
    runner.host_threads = 3
    runner.run()
    want = _host_loop.host_team_size(64, _S, threads=_host_loop._threads())
    assert seen == [(want, want), (3, 3)]
    assert get() == 7


def test_a_second_concurrent_host_run_of_one_artifact_is_refused(auto, monkeypatch):  # noqa: F811
    """The rounds cell is one global per artifact: a host run started while
    another is inside its team pass (here: from inside it) is refused, and the
    next run after it finishes goes through."""
    real_serial = eexec._HostTeam.run_serial
    plan = auto[("plain", "host")]
    caught = []

    def serial(self, *a, **kw):
        if not caught:
            with pytest.raises(RuntimeError, match="one host run per artifact"):
                _go(plan, _batch(8, "uniform"), 10 * _S)
            caught.append(True)
        return real_serial(self, *a, **kw)

    monkeypatch.setattr(eexec._HostTeam, "run_serial", serial)
    report, _ = _go(plan, _batch(8, "uniform"), 10 * _S)
    assert caught and report.done
    report, _ = _go(plan, _batch(8, "uniform"), 10 * _S)
    assert report.done


def test_dladdr_is_looked_up_once_per_process(monkeypatch):
    """``ctypes.util.find_library`` scans the system libraries (a few ms per
    call): the C library's ``dladdr`` is found once and every later host
    entry lookup reuses it, so a first run does not pay the scan twice."""
    import ctypes.util

    getattr(_host_loop._dladdr, "cache_clear", lambda: None)()
    calls = []
    real = ctypes.util.find_library
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: calls.append(name) or real(name))
    first = _host_loop._dladdr()
    assert _host_loop._dladdr() is first
    assert len(calls) <= 1
