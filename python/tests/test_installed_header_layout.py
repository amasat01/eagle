# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every quoted ``#include`` in eagle's installed headers resolves from the
INSTALLED layout, not just from the source tree.

``cmake --install`` copies ``eagle/`` (without ``eagle/door/``) to ``include/eagle/``
and ``plugin/`` to ``include/eagle/plugin/`` (CMakeLists.txt), and the door install
adds the door's two public headers (cmake/EagleDoor.cmake), so the trees sit at different
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
#: the door's own install: only its two public C headers, installed beside its library
DOOR_PUBLIC = ("eg_door.h", "eg_unit_meta.h")
DOOR_CMAKE = REPO / "cmake" / "EagleDoor.cmake"


def _installed_tree():
    """Map each installed path (relative to ``include/``) to its source file."""
    tree = {}
    for top, dest in INSTALLED_AS.items():
        for src in (REPO / top).rglob("*"):
            relative = src.relative_to(REPO / top)
            if src.is_file() and relative.parts[:1] != ("door",):   # the door's install is its own (DOOR_PUBLIC)
                tree[dest / relative.as_posix()] = src
    for name in DOOR_PUBLIC:
        tree[pathlib.PurePosixPath("eagle/door") / name] = REPO / "eagle" / "door" / name
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


def test_the_door_install_names_its_two_public_headers_and_no_other_door_file():
    cmake = DOOR_CMAKE.read_text()
    install = re.search(r"install\(FILES([^)]*)\)", cmake).group(1)
    assert sorted(re.findall(r"/(eg_[a-z_]+\.h)", install)) == sorted(DOOR_PUBLIC)
    assert "eagle/door" in install
    tree = _installed_tree()
    door = sorted(p.as_posix() for p in tree if p.parts[:2] == ("eagle", "door"))
    assert door == sorted(f"eagle/door/{name}" for name in DOOR_PUBLIC)
