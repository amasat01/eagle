# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle's completeness gate + no-skip self-test + V1-ABSENT-IS-RED.

Deliberately SEPARATE from ``test_interop_matrix.py`` (the pure row
declaration file): these meta-tests are not matrix rows themselves, and
V1 below spawns a subprocess that re-runs the matrix module by path — keeping
that module free of anything but the 8 registered rows means the subprocess
run can never recurse into itself.

* **The completeness check** — ``test_all_eagle_rows_are_collected``
  (``assert_rows_complete``) and ``test_matrix_module_has_no_skip_tokens``
  (``assert_no_skip_machinery``,
  DISCOVERY-based over ``eagle/python/tests`` — no hand-passed
  module list to fall out of sync with reality) close eagle's completeness
  gate. Legitimate only because this suite owns all eight eagle-declared rows.
* **V1-ABSENT-IS-RED** — a subprocess with
  ``torch`` (or ``jax``) blocked via a ``sys.meta_path`` finder runs the
  matrix module (or the jax module); the rows that need the blocked framework
  each call ``certified_framework`` and ``pytest.fail`` (never skip), so the
  run must exit NONZERO, report failures, and report ZERO skipped. Proves the
  machinery cannot skip its way to green — parsed from the pytest summary
  line, not merely the exit code.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

from raptor.conformance.interop import assert_no_skip_machinery, assert_rows_complete

_MATRIX_MODULE = pathlib.Path(__file__).resolve().parent / "test_interop_matrix.py"
_JAX_MODULE = pathlib.Path(__file__).resolve().parent / "test_interop_jax.py"


@pytest.mark.interop_matrix
def test_all_eagle_rows_are_collected():
    assert_rows_complete("eagle")


def test_matrix_module_has_no_skip_tokens():
    assert_no_skip_machinery(pathlib.Path(__file__).resolve().parent)


_V1_SCRIPT = r"""
import sys


class _Block:
    # Blocks a framework regardless of sys.path/site-packages (it is genuinely
    # installed in this env, so omitting it from PYTHONPATH is not enough).
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == sys.argv[2]:
            raise ImportError("blocked " + sys.argv[2] + " for V1-ABSENT-IS-RED")
        return None


sys.meta_path.insert(0, _Block())

import pytest

rc = pytest.main(["-q", "-rs", sys.argv[1]])
sys.exit(rc)
"""


def _run_with_blocked(module, framework):
    proc = subprocess.run(
        [sys.executable, "-c", _V1_SCRIPT, str(module), framework],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode != 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    failed = re.search(r"(\d+) failed", proc.stdout)
    skipped = re.search(r"(\d+) skipped", proc.stdout)
    assert failed is not None and int(failed.group(1)) > 0, proc.stdout
    assert skipped is None or int(skipped.group(1)) == 0, proc.stdout


def test_absent_torch_is_red():
    """V1-ABSENT-IS-RED: with torch blocked, the matrix module's torch-owning
    rows (T-IN-CUDA-ALIAS, T-OUT-CUDA-ALIAS, T-IN-CPU-COPY,
    STREAM-TORCH-PRODUCER-ORDER, STREAM-TORCH-CONSUMER-ORDER, STREAM-IDENTITY) each
    hit ``certified_framework("torch")`` and FAIL loudly; the cupy-only rows
    (CP-CAPTURE-REJECT, STREAM-CUPY-PRODUCER-ORDER) still collect and pass. The
    run must exit nonzero, report at least one failure, and report zero
    skipped -- proof the machinery cannot skip its way to green."""
    _run_with_blocked(_MATRIX_MODULE, "torch")


def test_absent_jax_is_red():
    """The same proof for jax: with jax blocked, all five jax rows hit
    ``certified_framework("jax")`` and FAIL loudly, with nothing skipped."""
    _run_with_blocked(_JAX_MODULE, "jax")
