# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A host node that needs the GIL must not deadlock a thread waiting on its stream.

The wait (``Stream.synchronize``, ``Launcher.synchronize``) used to hold the GIL, so a
Python function the stream was about to call could never start. Each row runs in a child
process under a timeout, so a deadlock is a failure instead of a hung suite. The stream
is held back by a sleeping host function on a side stream until the main thread is
already waiting, so the function under test cannot run early.

The function under test is enqueued two ways: through cupy's ``launchHostFunc`` (any
Python host function behaves the same) and through ``eagle._core.HostFunc``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("cupy")

TIMEOUT_S = 10

CHILD = textwrap.dedent(
    """
    import sys, time
    import cupy as cp
    from eagle import _core

    source, sink = sys.argv[1], sys.argv[2]
    hits = []

    def subject(*_):
        hits.append(1)

    s = _core.Stream(non_blocking=True)
    es = cp.cuda.ExternalStream(s.ptr())
    side = cp.cuda.Stream(non_blocking=True)
    gate = cp.cuda.Event()
    x = cp.zeros(4)

    def enqueue():
        if source == "cupy":
            cp.cuda.runtime.launchHostFunc(s.ptr(), subject, 0)
        else:
            enqueue.keep = _core.HostFunc(subject)
            enqueue.keep.enqueue(s.ptr())

    side.launch_host_func(lambda _: time.sleep(0.4), None)
    gate.record(side)

    if sink == "stream":
        es.wait_event(gate)
        enqueue()
        s.synchronize()
    else:
        cap = _core.StreamCapturer(s.ptr())
        with es:
            cap.begin()
            x += 1
            enqueue()
            captured = cap.end()
        launcher = _core.Graph.from_captured(captured).launcher()
        launcher.stream(s.ptr())
        es.wait_event(gate)
        launcher.launch()
        launcher.synchronize()
    print("done", len(hits))
    """
)


@pytest.mark.gpu
@pytest.mark.parametrize("sink", ["stream", "launcher"])
@pytest.mark.parametrize("source", ["cupy", "seam"])
def test_waiting_on_a_stream_with_a_python_host_node_does_not_deadlock(source, sink):
    try:
        run = subprocess.run(
            [sys.executable, "-c", CHILD, source, sink],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"deadlock: {sink}.synchronize() held the GIL while a {source} host node "
            f"waited for it ({TIMEOUT_S} s)"
        )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["done", "1"], run.stdout + run.stderr
