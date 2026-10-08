# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every quoted ``#include`` in eagle's installed headers resolves from the
INSTALLED layout, not just from the source tree.

``cmake --install`` copies ``eagle/`` to ``include/eagle/`` and ``plugin/`` to
``include/eagle/plugin/`` (CMakeLists.txt), so the two trees sit at different
depths once installed. A relative spelling such as ``../../eagle/exec/X.h`` from
``plugin/plugin_registry/`` resolves in the source tree and points nowhere after
install: every downstream consumer fails to compile while in-tree builds stay
green. This test replays the compiler's quoted-include lookup (the including
file's directory, then the include root) over both layouts.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.repo_local

REPO = pathlib.Path(__file__).resolve().parents[2]
#: source directory -> its path under the installed ``include/`` root
INSTALLED_AS = {"eagle": pathlib.PurePosixPath("eagle"),
                "plugin": pathlib.PurePosixPath("eagle/plugin")}
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
SUFFIXES = {".h", ".hpp", ".cuh", ".inl"}


def _installed_tree():
    """Map each installed path (relative to ``include/``) to its source file."""
    tree = {}
    for top, dest in INSTALLED_AS.items():
        for src in (REPO / top).rglob("*"):
            if src.is_file():
                tree[dest / src.relative_to(REPO / top).as_posix()] = src
    return tree


def _resolve(including_dir, spelling, exists):
    for base in (including_dir, pathlib.PurePosixPath(".")):
        parts = []
        for part in (base / spelling).parts:
            if part == "..":
                if not parts:
                    break
                parts.pop()
            elif part != ".":
                parts.append(part)
        else:
            candidate = pathlib.PurePosixPath(*parts)
            if exists(candidate):
                return candidate
    return None


def test_every_installed_header_include_resolves_from_the_install_prefix():
    tree = _installed_tree()
    source_exists = lambda p: (REPO / p).is_file()  # noqa: E731
    broken = []
    for installed, src in sorted(tree.items()):
        if src.suffix not in SUFFIXES:
            continue
        src_dir = pathlib.PurePosixPath(src.relative_to(REPO).as_posix()).parent
        for spelling in INCLUDE.findall(src.read_text(errors="replace")):
            if _resolve(src_dir, spelling, source_exists) is None:
                continue  # external (aether, CUDA, ...): not this layout's claim
            if _resolve(installed.parent, spelling, tree.__contains__) is None:
                broken.append(f"{src.relative_to(REPO)}: #include \"{spelling}\"")
    assert not broken, (
        "includes that only resolve in the source tree:\n" + "\n".join(broken))
