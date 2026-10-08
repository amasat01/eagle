# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A card dict must never carry a local absolute path.

``benchmarks/perf_card/cpu_card.py``'s "compiler" field once recorded
``hawk.compile.host_compiler()``'s absolute discovery path verbatim (e.g.
``/home/<user>/.conda/envs/.../g++``) -- a leak of the build box's
directory layout into a committed, published card. ``_hawk_host_flags``
already strips ``-I``/``-isystem``/``-include`` for the identical reason (an
absolute include path would leak the build box the same way);
``_card_common.tool_name`` gives the "compiler" field the same treatment, and
``_card_common.find_absolute_paths`` is the generic half of this gate -- it
walks any nested dict/list a card builds and names every absolute path it
finds, so a future field that starts recording one has the same net.

RED: hand-revert ``cpu_card.py``'s "compiler" line back to
``hc.host_compiler()`` (or plant any other field that stores an absolute
path) and the matching test below fails naming it.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CARD_DIR = _REPO / "benchmarks" / "perf_card"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpu_card():
    """Loading ``cpu_card.py`` this way (mirrors test_perf_card_table.py's
    own ``_script()``) runs its own ``sys.path`` setup, so ``module.cc`` is
    the same ``_card_common`` instance the real card writer uses."""
    return _load("eagle_cpu_card_abspath_test", _CARD_DIR / "cpu_card.py")


def test_tool_name_strips_an_absolute_discovery_path():
    cc = _cpu_card().cc
    assert cc.tool_name(
        "/opt/conda/envs/dev/bin/g++") == "g++"
    assert cc.tool_name("g++") == "g++"
    assert cc.tool_name(None) is None


def test_find_absolute_paths_catches_a_planted_one():
    """Non-vacuity: the scanner must actually fire on a known-bad input,
    nested the way a real card dict is (arm -> field -> value), and must
    stay silent on an already-clean one."""
    cc = _cpu_card().cc

    clean = {"arms": {"eagle_term": {"compiler": "g++", "flags": ["-O3"]}}}
    assert cc.find_absolute_paths(clean) == []

    planted = {"arms": {"eagle_term": {
        "compiler": "/opt/conda/envs/dev/bin/g++",
        "flags": ["-O3"]}}}
    hits = cc.find_absolute_paths(planted)
    assert hits, "the scanner did not catch a planted absolute path"
    assert any("compiler" in key for key, _value in hits)


def test_compiler_field_is_sanitized_the_same_way_facts_does_it():
    """Exercises the real line in ``_EagleHostArm.facts()`` (via
    ``hc.host_compiler()``, cheap -- a PATH lookup, no compilation) rather
    than re-deriving the sanitization here."""
    module = _cpu_card()
    hc = pytest.importorskip("hawk.compile")

    discovered = hc.host_compiler()
    assert os.path.isabs(discovered), (
        "host_compiler() is expected to resolve to an absolute path here; "
        "this test's premise (the card must not record that verbatim) "
        "does not apply if that changes")
    recorded = module.cc.tool_name(discovered)
    assert not os.path.isabs(recorded), (
        f"cpu_card.py's 'compiler' field would still be absolute: {recorded!r}")
    assert module.cc.find_absolute_paths({"compiler": recorded}) == []
