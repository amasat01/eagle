# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether-ABI version stamp: it stays in sync with the C++ host and a mismatch is
rejected at load.

The by-value GRef/HandleT POD layout has no version field of its own; the
``static_assert``s in ``plugin/gref_abi.h`` catch a layout drift at *compile* time, but
a shipped PTX/cubin/fatbin carries no other tie to the ABI it was built against. The
``aether_abi`` stamp closes that artifact-vs-loader gap at *load* time — eagle's
:class:`~eagle.loaded.LoadedKernel` validates it before binding any by-value struct.
"""

import json
import pathlib
import re
import shutil

import pytest

import eagle

HOST = pathlib.Path(__file__).resolve().parent.parent.parent / "plugin"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


@pytest.mark.repo_local
def test_abi_version_matches_cpp_header():
    """``eagle.ABI_VERSION`` and the C++ ``EAGLE_AETHER_ABI`` #define must be
    identical — they are the two ends of the same artifact-vs-loader contract, bumped
    together by hand when the POD layout changes."""
    text = (HOST / "gref_abi.h").read_text()
    m = re.search(r'#define\s+EAGLE_AETHER_ABI\s+"([^"]+)"', text)
    assert m, "EAGLE_AETHER_ABI #define not found in plugin/gref_abi.h"
    assert m.group(1) == eagle.ABI_VERSION


def test_version_is_nonempty_string():
    assert isinstance(eagle.ABI_VERSION, str)
    assert eagle.ABI_VERSION


@pytest.mark.gpu
def test_loaded_plugin_accepts_valid_abi():
    # the committed fixture is stamped with the current ABI -> it loads cleanly.
    eagle.LoadedVector(FIX / "gravity.ptx")


@pytest.mark.gpu
def test_loaded_plugin_rejects_abi_mismatch(tmp_path):
    # Copy the fixture, doctor its sidecar to a stale ABI tag -> the loader must refuse
    # it (before binding any by-value struct).
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    meta = json.loads((FIX / "gravity.json").read_text())
    meta["aether_abi"] = "aether-abi/0-stale"
    (tmp_path / "gravity.json").write_text(json.dumps(meta, indent=2))
    with pytest.raises(ValueError, match="AETHER ABI"):
        eagle.LoadedVector(tmp_path / "gravity.ptx")


# --------------------------------------------------------------------------- #
# eagle.abi now speaks BOTH ABI
# generations; pin its acceptance + its vocabulary against the compiled
# eagle._core extension (the loaded binding is the artifact-vs-loader
# contract's other half — a drift here is exactly the "a proxy measures
# itself" failure class, so this checks the REAL compiled surface, not just
# the pure-Python literals).
# --------------------------------------------------------------------------- #
def test_check_aether_abi_accepts_both_generations():
    from eagle.abi import ABI_TAG_V1, ABI_TAG_V2, check_aether_abi

    check_aether_abi({"aether_abi": ABI_TAG_V1}, kind="plugin manifest", name="m")
    check_aether_abi({"aether_abi": ABI_TAG_V2}, kind="plugin manifest", name="m")


def test_check_aether_abi_still_rejects_absent_and_stale_tags():
    from eagle.abi import check_aether_abi

    with pytest.raises(ValueError, match="AETHER ABI"):
        check_aether_abi({}, kind="plugin manifest", name="m")
    with pytest.raises(ValueError, match="AETHER ABI"):
        check_aether_abi(
            {"aether_abi": "aether-abi/0-stale"}, kind="plugin manifest", name="m"
        )


def test_abi_version_of_matches_generations():
    from eagle.abi import ABI_TAG_V1, ABI_TAG_V2, abi_version_of

    assert abi_version_of(ABI_TAG_V1) == 1
    assert abi_version_of(ABI_TAG_V2) == 2
    assert abi_version_of("aether-abi/0-stale") == 0
    assert abi_version_of(None) == 0


@pytest.mark.gpu
def test_abi_tags_match_the_compiled_core_extension():
    """The Python literals and the compiled ``eagle._core`` binding (built from
    the SAME C++ ``EAGLE_AETHER_ABI``/``EAGLE_AETHER_ABI_V2`` #defines) must
    name the identical two tags — a drift here would mean eagle.abi accepts a
    tag the loaded binding does not actually speak, or vice versa."""
    from eagle import _core
    from eagle.abi import ABI_TAG_V1, ABI_TAG_V2

    assert _core.ABI_TAG_V1 == ABI_TAG_V1
    assert _core.ABI_TAG_V2 == ABI_TAG_V2
    assert _core.abi_version_of(ABI_TAG_V1) == 1
    assert _core.abi_version_of(ABI_TAG_V2) == 2


@pytest.mark.gpu
def test_max_schema_version_matches_the_compiled_core_extension():
    """A ``MAX_SCHEMA_VERSION`` pin test python <-> ``_core`` —
    raptor's schema ceiling (``eagle.roles.MAX_SCHEMA_VERSION``, re-exported
    from ``raptor.schema.manifest``) and the compiled C++ loader's own ceiling
    (``kPluginMaxSchemaVersion``, bound as ``eagle._core.MAX_SCHEMA_VERSION``)
    must agree, or a manifest one side accepts the other silently refuses."""
    from eagle import _core
    from eagle.roles import MAX_SCHEMA_VERSION, SCHEMA_VERSION

    assert _core.MAX_SCHEMA_VERSION == MAX_SCHEMA_VERSION
    assert _core.SCHEMA_VERSION == SCHEMA_VERSION
