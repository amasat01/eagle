# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The import checks of CI's wheel jobs, run from the installed package (outside the repo).
usage: ci_import_check.py nodriver|sdist"""
import platform
import sys
from importlib.metadata import version

import eagle
import eagle._core
import eagle.exec

assert "site-packages" in eagle._core.__file__, eagle._core.__file__
assert eagle.exec.fold("sum", [1.0, 2.0]) == 3.0
if sys.argv[1] == "nodriver":
    d = eagle._core.cuda_backend()
    assert not d["loaded"], d
    try:
        eagle._core.Stream()
        raise SystemExit("Stream() did not refuse")
    except eagle.BackendUnavailable as e:
        print("typed refusal:", e)
    maps = open("/proc/self/maps").read()
    assert "libcuda" not in maps and "libcudart" not in maps
else:
    print(f"OK: raptor-eagle {version('raptor-eagle')} from its sdist on {platform.machine()}")
