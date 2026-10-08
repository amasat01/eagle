# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``load_manifest`` loads byte-identical plugin CONTENT once per process.

WHAT THE MEMO IS FOR. ``LoadedKernel.__init__`` ends in
``cp.RawModule(path=...)``, and that call JITs the artifact's PTX through the
driver. It is the dominant cost of loading a deployed unit, and it is paid per
PATH: a producer that publishes the same kernel into a fresh deployment
directory — which is what an AOT toolchain does every time it re-authors a
kernel in one process — pays a full driver JIT for bytes this process has
already compiled. The measured shape ("warm, same process,
fresh kernel object") was ~7 ms of ``load_manifest`` against a whole re-author
budget of ~14 ms, versus a JIT toolchain that simply hits its in-memory module
cache.

WHAT THE KEY IS, AND WHY. CONTENT, never a path and never an mtime: the digest
of the artifact's bytes and of its sidecar's bytes, beside the manifest ENTRY
that names them (its id, filename, format and enabled flag) and the loader class
that will read them. Path is excluded because publishing the same bytes
elsewhere is the whole case the memo exists for; mtime is excluded because a
re-published copy has a new one and identical content — the same rule HAWK's own
content-closure cache states, for the same reason. The manifest's
OTHER fields are not in the key and are not memoised at all: ``schema_version``,
the execution axis and the by-value ABI tag are re-checked on EVERY call, so a
manifest that got worse still refuses even when its plugins are unchanged.

The entry's ``id`` is in the key because the loaded object CARRIES it
(``LoadedKernel.id``, read by a consumer building a descriptor). Two artifacts
with identical bytes under different ids are two different loaded objects, and
sharing one between them would hand the second the first one's name.

RED HISTORY: two planted defects against an earlier eagle revision, both removed.

(a) the memo LOOKUP disabled (``hit = None``) — i.e. the behaviour before this
memo existed::

    AssertionError: two load_manifest calls over byte-identical plugin content
    performed 2 driver module loads, not 1 - the second re-JITed bytes this
    process had already compiled
    assert 2 == 1

(b) the key changed from the artifact's BYTES to its file NAME — the shape a
memo takes when it is written for speed rather than for identity. The two
discrimination rows fired at once, which is what makes them the non-vacuity
half rather than decoration::

    AssertionError: a changed artifact was served from the memo
    AssertionError: a changed sidecar was served from the memo
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from eagle.abi import ABI_VERSION
from eagle.registry import _clear_plugin_memo, load_manifest

pytestmark = pytest.mark.gpu

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


class _Loads:
    """A counter around ``cupy.RawModule`` — the driver load itself.

    Counting the DRIVER call rather than timing the load is the point: a memo
    that merely got faster would still be a memo that loads twice, and a wall
    time cannot tell those apart on a machine with other work on it."""

    def __init__(self, monkeypatch) -> None:
        import cupy as cp

        self.n = 0
        real = cp.RawModule

        def counted(*args, **kwargs):
            self.n += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(cp, "RawModule", counted)


def _unit(root, name, *, stem="gravity", plugin_id="gravity", indent=2,
          enabled=True, extra_ptx=""):
    """Publish the committed fixture plugin into its own deployment directory.

    Each unit is a SEPARATE directory holding the same bytes, which is exactly
    the shape a producer that re-publishes per build creates."""
    directory = pathlib.Path(root) / name
    directory.mkdir(parents=True)
    ptx = directory / f"{plugin_id}.ptx"
    shutil.copy(FIX / f"{stem}.ptx", ptx)
    if extra_ptx:
        ptx.write_text(ptx.read_text() + extra_ptx)
    sidecar = json.loads((FIX / f"{stem}.json").read_text())
    (directory / f"{plugin_id}.json").write_text(json.dumps(sidecar, indent=indent))
    (directory / "manifest.json").write_text(json.dumps({
        "version": 1,
        "pattern": "vector",
        "aether_abi": ABI_VERSION,
        "plugins": [{"id": plugin_id, "order": 0, "enabled": enabled,
                     "artifact": ptx.name, "sidecar": f"{plugin_id}.json",
                     "format": "ptx"}],
    }, indent=2))
    return directory / "manifest.json"


@pytest.fixture(autouse=True)
def _fresh_memo():
    """Every row starts from an empty memo and leaves one behind.

    The memo is per-PROCESS state, so a row that inherited another's entries
    would be measuring the suite's order rather than the memo."""
    _clear_plugin_memo()
    yield
    _clear_plugin_memo()


def test_byte_identical_content_loads_once_and_returns_the_same_object(
        tmp_path, monkeypatch):
    """The row the memo exists for: two directories, one driver load."""
    counter = _Loads(monkeypatch)
    a = load_manifest(_unit(tmp_path, "a"))
    b = load_manifest(_unit(tmp_path, "b"))
    assert counter.n == 1, (
        f"two load_manifest calls over byte-identical plugin content performed "
        f"{counter.n} driver module loads, not 1 — the second re-JITed bytes "
        "this process had already compiled")
    assert a["gravity"] is b["gravity"], (
        "byte-identical plugin content did not return the same loaded object")


def test_each_call_still_gets_its_own_registry(tmp_path):
    """The REGISTRY is not shared, only the launchables inside it.

    A registry is a mutable by-name map; handing two callers the same one would
    make one caller's ``register`` visible to the other."""
    a = load_manifest(_unit(tmp_path, "a"))
    b = load_manifest(_unit(tmp_path, "b"))
    assert a is not b
    a.register(a["gravity"], name="alias")
    assert "alias" not in b, "the two calls shared one registry object"


def test_changed_artifact_bytes_load_again(tmp_path, monkeypatch):
    """A changed BINARY is a different plugin, however it is published."""
    counter = _Loads(monkeypatch)
    a = load_manifest(_unit(tmp_path, "a"))
    # a trailing PTX comment: different bytes, same program, so the row turns on
    # the digest and not on whether the module still loads.
    b = load_manifest(_unit(tmp_path, "b", extra_ptx="\n// republished\n"))
    assert counter.n == 2, "a changed artifact was served from the memo"
    assert a["gravity"] is not b["gravity"]


def test_changed_sidecar_bytes_load_again(tmp_path, monkeypatch):
    """A changed SIDECAR is a different plugin: it is what the loader READS.

    The two sidecars here carry the same JSON with different indentation, so the
    row fails unless the key is the sidecar's BYTES."""
    counter = _Loads(monkeypatch)
    a = load_manifest(_unit(tmp_path, "a"))
    b = load_manifest(_unit(tmp_path, "b", indent=4))
    assert counter.n == 2, "a changed sidecar was served from the memo"
    assert a["gravity"] is not b["gravity"]


def test_a_different_plugin_id_is_a_different_object(tmp_path, monkeypatch):
    """Identical bytes under two ids are two plugins.

    ``LoadedKernel.id`` is read off the artifact's own stem and travels with the
    object (a consumer names the artifact it wraps by it), so sharing one across
    ids would give the second plugin the first one's name."""
    counter = _Loads(monkeypatch)
    a = load_manifest(_unit(tmp_path, "a"))
    b = load_manifest(_unit(tmp_path, "b", plugin_id="gravity_two"))
    assert counter.n == 2
    assert a["gravity"].id == "gravity"
    assert b["gravity_two"].id == "gravity_two"


def test_a_disabled_entry_is_still_never_loaded(tmp_path, monkeypatch):
    """The memo must not change WHICH entries load."""
    counter = _Loads(monkeypatch)
    reg = load_manifest(_unit(tmp_path, "a", enabled=False))
    assert counter.n == 0 and len(reg) == 0


def test_a_manifest_that_got_worse_is_still_refused(tmp_path):
    """The manifest's own gates are re-run on every call, memo or not.

    Only the loaded PLUGIN is memoised; ``schema_version``, the execution axis
    and the by-value ABI tag are re-checked each time, so a second call over the
    same plugin bytes under a broken manifest refuses instead of being served."""
    load_manifest(_unit(tmp_path, "a"))
    bad = _unit(tmp_path, "b")
    doc = json.loads(bad.read_text())
    doc["aether_abi"] = "aether-abi/999"
    bad.write_text(json.dumps(doc, indent=2))
    with pytest.raises(ValueError):
        load_manifest(bad)


def test_clearing_the_memo_forces_a_reload(tmp_path, monkeypatch):
    """The memo is a per-process shortcut a caller can always opt out of."""
    counter = _Loads(monkeypatch)
    a = load_manifest(_unit(tmp_path, "a"))
    _clear_plugin_memo()
    b = load_manifest(_unit(tmp_path, "b"))
    assert counter.n == 2
    assert a["gravity"] is not b["gravity"]
