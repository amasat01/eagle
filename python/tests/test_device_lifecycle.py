# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Two lifecycle rows on a real device: no context on the calling thread, and a stream that dies first."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.gpu

_DEAD_STREAM_CHILD = """
from eagle import _core, arrays
s = _core.Stream(non_blocking=True)
a = arrays.full((1 << 20,), 7, arrays.int32, "cuda:0", s.ptr())
s.synchronize()
print("ready")
"""


def test_device_props_on_a_thread_with_no_context():
    from eagle import _core

    found = {}

    def worker():
        try:
            found["props"] = _core.device_props(-1)
            found["memory"] = _core.device_mem_info(-1)   # reads cuMemGetInfo, which needs a context
        except Exception as error:   # the row reports it
            found["error"] = error

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert "error" not in found, found.get("error")
    total, free = found["memory"]
    assert total > 0 and 0 < free <= total


def test_an_array_that_outlives_its_stream_frees_cleanly_at_exit():
    # `s` is a module global destroyed before `a`: the array's free runs after its stream's owner is gone
    done = subprocess.run([sys.executable, "-I", "-c", _DEAD_STREAM_CHILD], capture_output=True, text=True,
                          env=dict(os.environ), timeout=120)
    assert done.returncode == 0, (done.returncode, done.stderr[-400:])
    assert done.stdout.strip() == "ready"


def test_a_free_inside_a_capture_waits_and_drains_after_it():
    from eagle import _core, arrays

    s = _core.Stream(non_blocking=True)
    a = arrays.full((1 << 16,), 3, arrays.int32, "cuda:0", s.ptr())
    s.synchronize()
    cap = _core.StreamCapturer(s.ptr())
    cap.begin()
    del a                      # stream-ordered free on a capturing stream: queued, never issued inside it
    assert _core.mem_drain() == 1
    cap.end()
    assert _core.mem_drain() == 0
