# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``import eagle`` needs none of the front-end pieces: eagle loads built kernels where only raptor and the runtime are installed."""

import subprocess
import sys

_BLOCK = ("lowering", "hawk", "hawk_c", "hawk_cuda", "osprey_cpu", "simt", "aether.wings")

_PROGRAM = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in %r):
            raise ImportError(f"blocked: {name}")

sys.meta_path.insert(0, Block())
import eagle
print("ok")
""" % (_BLOCK,)


def test_import_eagle_with_the_front_end_pieces_blocked():
    done = subprocess.run([sys.executable, "-c", _PROGRAM], capture_output=True, text=True)
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr[-1500:]
