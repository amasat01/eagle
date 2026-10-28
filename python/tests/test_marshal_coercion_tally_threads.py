# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The fast/slow coercion tallies of ``marshal._as_dtype`` lose no increment
when many threads coerce at once (a bare ``+= 1`` on a dict slot is a
read-modify-write, on a GIL build as well as on a free-threaded one)."""

import sys
import threading
import types

from eagle import marshal

THREADS, PER = 8, 50000


def test_the_fast_coercion_tally_is_exact_under_threads(monkeypatch):
    # The tally is host-side; a stand-in cupy lets the row run on every runner,
    # free-threaded ones included, where no cupy wheel is installed.
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(marshal, "_fast_path_ok", lambda value, dt: True)
    marshal._reset_coercion_stats()
    start = threading.Barrier(THREADS)

    def work():
        start.wait()
        for _ in range(PER):
            marshal._as_dtype(None)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)             # make a GIL build interleave like a free-threaded one
    try:
        threads = [threading.Thread(target=work) for _ in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
    finally:
        sys.setswitchinterval(interval)
    stats = marshal._coercion_stats()
    marshal._reset_coercion_stats()
    assert stats == {"fast": THREADS * PER, "slow": 0}, stats
