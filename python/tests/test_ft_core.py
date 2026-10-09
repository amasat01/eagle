# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading rows for the compiled core (``eagle._core``, ``eagle._mpi``).

Row classes FT-1, FT-2, FT-5 and FT-6 (capture) of the family's free-threading
acceptance, registered against the conformance harness that ``raptor-core``
hosts (a TEST dependency only: nothing under ``eagle/`` imports it).

On a GIL build every row that needs real parallelism SKIPS with a reason; the
name manifest below keeps "skipped" from degrading into "silently not
collected". No row ever sets ``-X gil=0`` or ``PYTHON_GIL``: the harness
refuses to run when either is in force.

The shared-object and capture rows run their threads in a CHILD process so that
a crash is a red verdict (non-zero exit, faulthandler traceback) and not the end
of the pytest session.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from raptor.conformance import freethreading as ft
from raptor.conformance.freethreading import declare_ft_row, register_ft_row

pytestmark = pytest.mark.ft

declare_ft_row("FT-1-GIL-FREE-EAGLE-CORE", "eagle",
               "importing eagle._core leaves the GIL disabled")
declare_ft_row("FT-1-GIL-FREE-EAGLE-MPI", "eagle",
               "importing eagle._mpi leaves the GIL disabled (when built)")
declare_ft_row("FT-2-CANARY-EAGLE-CORE", "eagle",
               "the planted unsynchronised counter in eagle._core loses >= 10% of updates")
declare_ft_row("FT-5-EAGLE-LAUNCHER-SHARED", "eagle",
               "one Launcher replayed from many threads gives the exact result")
declare_ft_row("FT-5-EAGLE-COMPOSER-SHARED", "eagle",
               "one GraphComposer launched from many threads fires every step exactly")
declare_ft_row("FT-5-EAGLE-CONSUME-TWICE", "eagle",
               "a handle consumed by two threads at once is consumed exactly once")
declare_ft_row("FT-6-EAGLE-CAPTURE-CONCURRENT", "eagle",
               "two threads capture concurrently and both graphs replay correctly")
declare_ft_row("FT-6-EAGLE-CAPTURE-NESTED", "eagle",
               "a second capture on a thread that already has one open raises")
declare_ft_row("FT-6-EAGLE-CAPTURE-OTHER-THREAD", "eagle",
               "allocation and free on another thread during a capture are harmless")
declare_ft_row("FT-LINT-EAGLE-NO-GIL-RELEASE", "eagle",
               "the bindings never detach (no gil_scoped_release / call_guard)")

EXPECTED_TESTS = frozenset(
    {
        "test_manifest_all_tests_collected",
        "test_rows_declared_and_collected",
        "test_core_is_gil_free",
        "test_mpi_is_gil_free",
        "test_core_canary_loses_updates",
        "test_shared_launcher_replays_exactly",
        "test_shared_composer_fires_exactly",
        "test_consume_twice_consumes_once",
        "test_two_threads_capture_concurrently",
        "test_nested_capture_on_one_thread_raises",
        "test_other_thread_alloc_free_during_capture",
        "test_bindings_never_release_the_gil",
    }
)

_SRC = Path(__file__).resolve().parent.parent / "src"
_HERE = str(Path(__file__).resolve().parent.parent)  # the tree holding the eagle package


def test_manifest_all_tests_collected(request):
    here = {
        i.name for i in request.session.items if i.module is sys.modules[__name__]
    }
    # a marked row (gpu, repo_local) may be deselected by the leg's -m
    # expression; every unmarked row must be collected
    required = {
        n for n in EXPECTED_TESTS
        if not getattr(globals().get(n), "pytestmark", None)
    }
    assert required <= here, sorted(required - here)
    assert {n for n in here if "[" not in n} <= EXPECTED_TESTS


def test_rows_declared_and_collected():
    ft.assert_ft_rows_complete("eagle")


@register_ft_row("FT-1-GIL-FREE-EAGLE-CORE")
def test_core_is_gil_free():
    ft.require_free_threaded()
    ft.assert_gil_free("eagle._core")


@register_ft_row("FT-1-GIL-FREE-EAGLE-MPI")
def test_mpi_is_gil_free():
    ft.require_free_threaded()
    try:
        import eagle._mpi  # noqa: F401
    except ImportError:
        pytest.skip("eagle._mpi is not built here (no MPI); its CMake line declares "
                    "FREE_THREADED like _core")
    ft.assert_gil_free("eagle._mpi")


@register_ft_row("FT-2-CANARY-EAGLE-CORE")
def test_core_canary_loses_updates():
    ft.require_free_threaded()
    from eagle import _core

    res = ft.require_not_vacuous(
        ft.measure_loss(_core._unsynchronised_bump, _core._unsynchronised_count)
    )
    print(f"eagle._core canary lost {res.lost_fraction:.1%} of {res.expected}")


# --------------------------------------------------------------------------- #
# Child-process runner: a crash is a red verdict, not the end of the session.
# --------------------------------------------------------------------------- #
_PRELUDE = r"""
import faulthandler, json, sys, threading
import cupy as cp
from eagle import _core

faulthandler.enable()
faulthandler.dump_traceback_later(90, exit=True)

def capture_add(stream, a, value, times):
    cap = _core.StreamCapturer(stream.ptr())
    with cp.cuda.ExternalStream(stream.ptr()):
        cap.begin()
        for _ in range(times):
            a += value
        return cap.end()

def build_launcher(stream, a, value, times):
    launcher = _core.Graph.from_captured(capture_add(stream, a, value, times)).launcher()
    launcher.stream(stream.ptr())
    return launcher

def warm(a):                       # JIT outside every capture
    a += 1.0
    cp.cuda.Device().synchronize()
    a[...] = 0.0
    cp.cuda.Device().synchronize()

def run_threads(fn, n):
    barrier = threading.Barrier(n)
    out = [None] * n
    def work(k):
        barrier.wait()
        try:
            out[k] = ("ok", fn(k))
        except BaseException as exc:
            out[k] = ("err", type(exc).__name__ + ": " + str(exc)[:160])
    pool = [threading.Thread(target=work, args=(k,)) for k in range(n)]
    for t in pool: t.start()
    for t in pool: t.join()
    return out
"""


def _run_child(body: str, timeout: float = 240.0) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PRELUDE + body],
        capture_output=True, text=True, timeout=timeout, check=False, cwd=_HERE,
    )
    assert proc.returncode == 0, (
        f"child died (rc={proc.returncode}):\n{proc.stderr[-2500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_LAUNCHER_CHILD = r"""
T, N = 8, 250
a = cp.zeros(1, dtype=cp.float64)
warm(a)
s = _core.Stream(non_blocking=True)
launcher = build_launcher(s, a, 1.0, 1)
cp.cuda.Device().synchronize()
def hammer(k):
    for _ in range(N):
        launcher.launch()
res = run_threads(hammer, T)
launcher.synchronize()
print(json.dumps({"results": [r[0] for r in res], "errors": [r[1] for r in res if r[0] == "err"],
                  "value": float(a[0]), "expected": float(T * N)}))
"""


@register_ft_row("FT-5-EAGLE-LAUNCHER-SHARED")
@pytest.mark.gpu
def test_shared_launcher_replays_exactly():
    ft.require_free_threaded()
    out = _run_child(_LAUNCHER_CHILD)
    assert out["errors"] == [], out["errors"]
    assert out["value"] == out["expected"] == 2000.0, out


_COMPOSER_CHILD = r"""
T, N = 8, 250
a = cp.zeros(1, dtype=cp.float64)
warm(a)
s = _core.Stream(non_blocking=True)
launcher = build_launcher(s, a, 1.0, 1)
steps = [0]
def step(stream):                      # plain int: exact only if the composer's calls serialise
    steps[0] += 1
c = _core.GraphComposer("sequenced")
c.register_launcher(launcher, name="member")
c.register_callable(step, name="counter")
c.build()
cp.cuda.Device().synchronize()
def hammer(k):
    for _ in range(N):
        c.launch()
res = run_threads(hammer, T)
cp.cuda.Device().synchronize()
print(json.dumps({"errors": [r[1] for r in res if r[0] == "err"], "steps": steps[0],
                  "value": float(a[0]), "fired": len(c.fired_history()), "members": c.num_members()}))
"""


@register_ft_row("FT-5-EAGLE-COMPOSER-SHARED")
@pytest.mark.gpu
def test_shared_composer_fires_exactly():
    ft.require_free_threaded()
    out = _run_child(_COMPOSER_CHILD)
    assert out["errors"] == [], out["errors"]
    assert out["members"] == 2
    assert out["steps"] == 2000, out
    assert out["value"] == 2000.0, out
    assert out["fired"] == 2000, out


_CONSUME_CHILD = r"""
REPS, T = 40, 8
stream = _core.Stream(non_blocking=True)
a = cp.zeros(1, dtype=cp.float64)
warm(a)
bad = []
wins = {"from_captured": 0, "register_launcher": 0, "add_node": 0}
errs = set()
for rep in range(REPS):
    # (1) one CapturedGraph adopted by T threads
    captured = capture_add(stream, a, 1.0, 1)
    res = run_threads(lambda k: _core.Graph.from_captured(captured), T)
    ok = sum(1 for r in res if r[0] == "ok")
    wins["from_captured"] += ok
    if ok != 1: bad.append(("from_captured", rep, ok))
    errs.update(r[1].split(":")[0] for r in res if r[0] == "err")
    # (2) one Launcher registered into T composers
    launcher = build_launcher(stream, a, 1.0, 1)
    comps = [_core.GraphComposer("sequenced") for _ in range(T)]
    res = run_threads(lambda k: comps[k].register_launcher(launcher), T)
    ok = sum(1 for r in res if r[0] == "ok")
    wins["register_launcher"] += ok
    if ok != 1: bad.append(("register_launcher", rep, ok))
    errs.update(r[1].split(":")[0] for r in res if r[0] == "err")
    # (3) one CapturedGraph folded into T Graphs
    captured = capture_add(stream, a, 1.0, 1)
    graphs = [_core.Graph() for _ in range(T)]
    res = run_threads(lambda k: graphs[k].add_node(captured), T)
    ok = sum(1 for r in res if r[0] == "ok")
    wins["add_node"] += ok
    if ok != 1: bad.append(("add_node", rep, ok))
    errs.update(r[1].split(":")[0] for r in res if r[0] == "err")
print(json.dumps({"bad": bad, "wins": wins, "errors": sorted(errs), "reps": REPS}))
"""


@register_ft_row("FT-5-EAGLE-CONSUME-TWICE")
@pytest.mark.gpu
def test_consume_twice_consumes_once():
    ft.require_free_threaded()
    out = _run_child(_CONSUME_CHILD)
    assert out["bad"] == [], out["bad"]
    assert out["wins"] == {k: out["reps"] for k in out["wins"]}, out
    # the documented outcomes: RuntimeError (consumed) / ValueError (empty graph)
    assert set(out["errors"]) <= {"RuntimeError", "ValueError"}, out


# --------------------------------------------------------------------------- #
# Capture mode (amendment A1): ThreadLocal + a per-thread capture guard.
# --------------------------------------------------------------------------- #
_CONCURRENT_CAPTURE_CHILD = r"""
REPS = 5
out = {"fail": [], "reps": REPS}
for rep in range(REPS):
    arrays = [cp.zeros(1, dtype=cp.float64) for _ in range(2)]
    streams = [_core.Stream(non_blocking=True) for _ in range(2)]
    for a in arrays: warm(a)
    barrier = threading.Barrier(2, timeout=30)
    def one(k):
        a, s = arrays[k], streams[k]
        cap = _core.StreamCapturer(s.ptr())
        with cp.cuda.ExternalStream(s.ptr()):
            cap.begin()
            barrier.wait()                       # both captures are open at once
            for _ in range(20):
                a += float(k + 1)
            captured = cap.end()
        launcher = _core.Graph.from_captured(captured).launcher()
        launcher.stream(s.ptr())
        for _ in range(3):
            launcher.launch()
        launcher.synchronize()
        return float(a[0])
    res = run_threads(one, 2)
    want = [3 * 20 * 1.0, 3 * 20 * 2.0]
    got = [r[1] for r in res]
    if got != want:
        out["fail"].append((rep, got, want))
out["depth_after"] = _core.capture_guard_depth()
print(json.dumps(out))
"""


@register_ft_row("FT-6-EAGLE-CAPTURE-CONCURRENT")
@pytest.mark.gpu
def test_two_threads_capture_concurrently():
    out = _run_child(_CONCURRENT_CAPTURE_CHILD)
    assert out["fail"] == [], out["fail"]
    assert out["depth_after"] == 0, out


_NESTED_CHILD = r"""
a = cp.zeros(1, dtype=cp.float64)
warm(a)
s1 = _core.Stream(non_blocking=True)
s2 = _core.Stream(non_blocking=True)
c1 = _core.StreamCapturer(s1.ptr())
c2 = _core.StreamCapturer(s2.ptr())
raised = None
with cp.cuda.ExternalStream(s1.ptr()):
    c1.begin()
    depth_open = _core.capture_guard_depth()
    try:
        c2.begin()
    except RuntimeError as exc:
        raised = str(exc)
    a += 1.0
    captured = c1.end()
# the refused begin left the first session intact: its graph replays
launcher = _core.Graph.from_captured(captured).launcher()
launcher.stream(s1.ptr())
launcher.launch()
launcher.synchronize()
print(json.dumps({"raised": raised, "depth_open": depth_open,
                  "depth_after": _core.capture_guard_depth(), "value": float(a[0])}))
"""


@register_ft_row("FT-6-EAGLE-CAPTURE-NESTED")
@pytest.mark.gpu
def test_nested_capture_on_one_thread_raises():
    out = _run_child(_NESTED_CHILD)
    assert out["raised"] is not None and "already open on this thread" in out["raised"], out
    assert out["depth_open"] == 1 and out["depth_after"] == 0, out
    assert out["value"] == 1.0, out


_OTHER_THREAD_CHILD = r"""
a = cp.zeros(1, dtype=cp.float64)
warm(a)
s = _core.Stream(non_blocking=True)
opened, done = threading.Event(), threading.Event()
other = {}
def churn():
    opened.wait(30)
    try:
        for _ in range(50):
            buf = cp.empty(1 << 16, dtype=cp.float64)    # allocation on another thread
            del buf                                       # and its free
        other["status"] = "ok"
    except BaseException as exc:
        other["status"] = type(exc).__name__ + ": " + str(exc)[:160]
    finally:
        done.set()
t = threading.Thread(target=churn); t.start()
cap = _core.StreamCapturer(s.ptr())
with cp.cuda.ExternalStream(s.ptr()):
    cap.begin()
    opened.set()
    a += 1.0
    done.wait(30)
    captured = cap.end()
t.join()
launcher = _core.Graph.from_captured(captured).launcher()
launcher.stream(s.ptr())
launcher.launch()
launcher.synchronize()
print(json.dumps({"other": other.get("status"), "value": float(a[0])}))
"""


@register_ft_row("FT-6-EAGLE-CAPTURE-OTHER-THREAD")
@pytest.mark.gpu
def test_other_thread_alloc_free_during_capture():
    out = _run_child(_OTHER_THREAD_CHILD)
    assert out == {"other": "ok", "value": 1.0}, out


# --------------------------------------------------------------------------- #
# A2 lint: the bindings never detach.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
@register_ft_row("FT-LINT-EAGLE-NO-GIL-RELEASE")
def test_bindings_never_release_the_gil():
    # nb::lock_self is a critical section that CPython suspends whenever the
    # thread detaches, so a binding that releases the GIL needs a per-object
    # busy state instead (and an allow-list entry naming it). None does.
    pattern = re.compile(r"gil_scoped_release|call_guard|Py_BEGIN_ALLOW_THREADS")
    hits = [
        f"{p.relative_to(_SRC)}:{n}: {line.strip()}"
        for p in sorted(_SRC.rglob("*"))
        if p.suffix in {".cpp", ".h", ".cu"}
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("//")
    ]
    assert hits == [], hits
