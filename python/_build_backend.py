# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Build backend: scikit-build-core, plus the files a source distribution needs.

The C++ headers ``eagle._core`` compiles (eagle's own ``eagle/`` and
``plugin/`` trees and aether's ``aether/``) and the README live one directory
above this package in the repository. A source distribution carries nothing
above itself, so before the sdist is written they are copied in:

* ``README.md``: copied from ``../README.md`` before every hook, because the
  project metadata of every build (wheel, editable, sdist) names it.
* ``_sdist_include/``: the header trees, copied for the sdist only and removed
  again afterwards. ``CMakeLists.txt`` compiles them only when the live
  repository tree above this directory is absent, so a checkout always builds
  against its own live headers.

Both are git-ignored and listed in ``sdist.include``. aether's headers come from
``$EAGLE_SDIST_AETHER_INCLUDE`` (a directory holding ``aether/aether.h``), else
the sibling ``aether`` checkout next to the eagle repository, else the running
environment's ``include/``. Every hook then runs scikit-build-core unchanged.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path

from scikit_build_core import build as _backend

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_INCLUDE = _HERE / "_sdist_include"


def _stage_readme() -> None:
    source = _REPO / "README.md"
    target = _HERE / "README.md"
    if source.is_file() and (
        not target.is_file() or target.read_bytes() != source.read_bytes()
    ):
        shutil.copyfile(source, target)


def _aether_root() -> Path:
    candidates = [
        os.environ.get("EAGLE_SDIST_AETHER_INCLUDE"),
        _REPO.parent / "aether",
        os.environ.get("CONDA_PREFIX") and Path(os.environ["CONDA_PREFIX"]) / "include",
        Path(sys.prefix) / "include",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "aether" / "aether.h").is_file():
            return Path(candidate)
    tried = [str(c) for c in candidates if c]
    raise RuntimeError(
        f"cannot find aether's headers (aether/aether.h) to put in the sdist: "
        f"tried {tried}. Set $EAGLE_SDIST_AETHER_INCLUDE."
    )


def _stage_headers() -> None:
    if not (_REPO / "eagle" / "cuda" / "CaptureFork.h").is_file():
        raise RuntimeError(
            f"the sdist is built from the eagle repository; no eagle headers "
            f"at {_REPO / 'eagle'}"
        )
    shutil.rmtree(_INCLUDE, ignore_errors=True)
    headers_only = shutil.ignore_patterns("*.cpp", "*.cu", "*.txt", "*.cmake")
    for root, name in ((_REPO, "eagle"), (_REPO, "plugin"), (_aether_root(), "aether")):
        shutil.copytree(root / name, _INCLUDE / name, ignore=headers_only)


def _with_readme(hook):
    @functools.wraps(hook)
    def run(*args, **kwargs):
        _stage_readme()
        return hook(*args, **kwargs)

    return run


for _name in _backend.__all__:
    globals()[_name] = _with_readme(getattr(_backend, _name))


def build_sdist(sdist_directory, config_settings=None):
    """Write the sdist with the README and the header trees copied in."""
    _stage_readme()
    _stage_headers()
    try:
        return _backend.build_sdist(sdist_directory, config_settings)
    finally:
        shutil.rmtree(_INCLUDE, ignore_errors=True)


__all__ = list(_backend.__all__)
