# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The declared lower bounds of numpy and torch are the versions the floor lock tests.

The floor lock (``docs/_tools/hosted_env/floor-cpu.txt``) runs the notebooks with the oldest numpy and torch a
user may install. A bound raised in ``pyproject.toml`` without the lock, or the other way round, would publish a
floor nobody ran.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.repo_local

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _declared(name):
    text = (ROOT / "python" / "pyproject.toml").read_text()
    found = re.findall(rf'"{name}>=([0-9.]+)', text)
    assert len(found) == 1, f"expected one lower bound for {name} in pyproject.toml, found {found}"
    return found[0]


def _pinned(name):
    text = (ROOT / "docs" / "_tools" / "hosted_env" / "floor-cpu.txt").read_text()
    found = re.findall(rf"^{name}==([0-9.]+)", text, re.MULTILINE)
    assert len(found) == 1, f"expected one {name} pin in floor-cpu.txt, found {found}"
    return found[0]


@pytest.mark.parametrize("name", ["numpy", "torch"])
def test_floor_lock_pins_the_declared_lower_bound(name):
    declared = _declared(name).split(".")
    pinned = _pinned(name).split(".")
    assert pinned[: len(declared)] == declared, f"{name}: pyproject says >={'.'.join(declared)}, the floor lock pins {'.'.join(pinned)}"
