# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Pytest config: make ``eagle`` importable and gate GPU-only tests.

Mirrors the reference suite's conftest mechanics, trimmed to what eagle's ported
tests use: the ``gpu`` marker (needs cupy + a CUDA device) and the ``torch`` marker
(needs torch + cupy). eagle's own import surface is producer-free, so nothing here
pulls in a compiler.
"""

import os
import pathlib
import sys

import pytest  # does not need the path setup below; grouped with the stdlib imports

ROOT = pathlib.Path(__file__).parent  # eagle/python (holds the `eagle` package)
# Sibling checkouts follow this tree's own directory suffix (eagle -> raptor), so an
# aether-branch worktree resolves its aether-branch siblings and the main tree resolves
# mains.
SFX = ROOT.parent.name.removeprefix("eagle")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))  # so tests import the cpp_demo helper module

# raptor: eagle.roles re-exports its schema constants, and the golden conformance test
# reads its goldens directly. Overridable via $RAPTOR_ROOT (mirrors the code generator's
# conftest $EAGLE_PYTHON); the default assumes raptor is a sibling checkout.
RAPTOR_ROOT = os.environ.get("RAPTOR_ROOT", str(ROOT.parent.parent / f"raptor{SFX}"))
sys.path.insert(0, RAPTOR_ROOT)

os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")


def _cupy_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _mpi_available() -> bool:
    """Whether the OPTIONAL ``eagle._mpi`` extension is part of this install.

    Import-only: the module initialises MPI lazily, on the first CALL, so probing
    for it here does not start a world in a session that never uses one."""
    try:
        import eagle._mpi  # noqa: F401

        return True
    except Exception:
        return False


def _torch_available() -> bool:
    """Whether torch imports (CPU or CUDA) — enough for detection / host round-trips."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


CUPY_AVAILABLE = _cupy_available()
TORCH_AVAILABLE = _torch_available()
MPI_AVAILABLE = _mpi_available()


#: The rank bed (``tests/mpi/``) is a DISTRIBUTED suite: every row in it runs on
#: every rank of an ``mpirun`` world, drives a deployed artifact its gate builds,
#: and initialises MPI — which is not a passive act in a long CUDA/cupy/torch
#: session (see ``tests/rank_singleton_cases.py``). So an ordinary ``pytest``
#: must never wander into it. It is collected only when its own gate says so:
#: ``tests/mpi/check_rank_bed.sh`` exports this variable, builds the artifact,
#: and launches the ranks.
#:
#: SCOPE OF THE GUARD, stated honestly: ``pytest_ignore_collect`` governs
#: RECURSIVE collection, so a plain ``pytest`` (which reaches the directory by
#: walking ``testpaths``) never enters it. A path named DIRECTLY on the command
#: line is an initial argument and bypasses this hook by pytest's own design, so
#: ``pytest tests/mpi`` still collects — and then fails loudly in a singleton
#: world ("the rank bed runs under `mpirun -np 2`; this world has 1 rank(s)")
#: rather than hanging, because nothing here blocks at a world size of one.
_BED_ENV = "EAGLE_RANK_BED"


def pytest_ignore_collect(collection_path, config):
    """Keep the rank bed out of every recursive collection but its own gate's."""
    if collection_path == ROOT / "tests" / "mpi" and not os.environ.get(_BED_ENV):
        return True
    return None


def pytest_collection_modifyitems(config, items):
    """Skip framework-gated tests whose backend is unavailable (so the rest run)."""
    gates = {
        "gpu": (CUPY_AVAILABLE, "cupy + CUDA GPU not available"),
        "gpu2": (CUPY_AVAILABLE, "cupy + CUDA GPU not available"),
        "torch": (
            CUPY_AVAILABLE and TORCH_AVAILABLE,
            "torch (+ cupy GPU) not available",
        ),
        "mpi": (
            MPI_AVAILABLE,
            "eagle._mpi not built (configure with -DEAGLE_PYTHON_MPI=ON)",
        ),
    }
    for item in items:
        for marker, (available, reason) in gates.items():
            if marker in item.keywords and not available:
                item.add_marker(pytest.mark.skip(reason=reason))
