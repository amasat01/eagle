# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Artifact schema hardening, Python side.

Three things, mirroring ``eagle/tests/test_SchemaHardening.cpp``:

1. Ignore-unknown: a sidecar/manifest carrying an unrecognized top-level key,
   or an unrecognized ``launch`` subkey, loads cleanly and the extra content
   is ignored. The JSON Schema files declare ``additionalProperties: true``
   and ``plugin_schema.rst`` asserts this in prose, but nothing exercised it
   before this file.
2. The ``schema_version`` compat matrix. Most of the *function-level* matrix
   (absent -> v1, legacy ``version`` fallback, current ok, future rejected)
   is already pinned directly against :func:`eagle.roles.check_schema_version`
   in ``test_roles_vocab.py`` — this file does not repeat those, only adds
   the full-*load*-path companions (through ``LoadedVector`` /
   ``load_manifest``, not the bare function) plus the "v1 + unknown key"
   case, which is exactly the ignore-unknown scenario above.
3. The frozen forward-strict field list (acceptance design item (c)).
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

import eagle
from eagle.abi import ABI_VERSION, check_aether_abi
from eagle.roles import (
    MAX_SCHEMA_VERSION,
    NEURAL_REQUIRED_FIELDS,
    SCHEMA_VERSION,
    check_schema_version,
    validate_roles,
)
from eagle.sidecar import validate_sidecar

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


def _doctor_sidecar(tmp_path, stem, patch):
    """Copy the committed ``<stem>`` fixture and merge ``patch`` into its sidecar
    (top-level keys only); returns the ``.ptx`` path a loader reads."""
    shutil.copy(FIX / f"{stem}.ptx", tmp_path / f"{stem}.ptx")
    meta = json.loads((FIX / f"{stem}.json").read_text())
    meta.update(patch)
    (tmp_path / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return tmp_path / f"{stem}.ptx"


def _write_manifest(tmp_path, plugins, *, pattern="vector", extra=None):
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pattern": pattern,
        "aether_abi": ABI_VERSION,
        "plugins": plugins,
        **(extra or {}),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return tmp_path / "manifest.json"


def _one_entry(stem):
    return {
        "id": stem,
        "order": 0,
        "enabled": True,
        "artifact": f"{stem}.ptx",
        "sidecar": f"{stem}.json",
        "format": "ptx",
    }


# --------------------------------------------------------------------------- #
# Item 1 — ignore-unknown.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_loaded_vector_ignores_unknown_top_level_key(tmp_path):
    ptx = _doctor_sidecar(
        tmp_path, "gravity",
        {"a_future_producer_key_this_loader_has_never_seen": {"x": 1}},
    )
    loaded = eagle.LoadedVector(ptx)  # must not raise
    assert loaded.kernel_name == "raptor_kernel"


@pytest.mark.gpu
def test_loaded_vector_ignores_unrecognized_launch_subkey(tmp_path):
    # `launch` is reserved-and-unpopulated in v1 (plugin_schema.rst "Reserved
    # capability fields"); a future minor revision may add subkeys without a
    # schema_version bump, so an unrecognized one must not fail this loader.
    ptx = _doctor_sidecar(
        tmp_path, "gravity",
        {"launch": {"block": 128, "a_hint_from_a_future_minor_revision": 7}},
    )
    loaded = eagle.LoadedVector(ptx)  # must not raise
    assert loaded.kernel_name == "raptor_kernel"


@pytest.mark.gpu
def test_loaded_pure_ignores_unknown_top_level_key(tmp_path):
    ptx = _doctor_sidecar(
        tmp_path, "bump",
        {"a_future_producer_key_this_loader_has_never_seen": [1, 2, 3]},
    )
    loaded = eagle.LoadedPure(ptx)  # must not raise
    assert loaded.kernel_name == "raptor_kernel"


@pytest.mark.gpu
def test_load_manifest_ignores_unknown_top_level_key(tmp_path):
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    shutil.copy(FIX / "gravity.json", tmp_path / "gravity.json")
    manifest = _write_manifest(
        tmp_path,
        [_one_entry("gravity")],
        extra={"a_future_manifest_key_this_loader_has_never_seen": True},
    )
    reg = eagle.load_manifest(manifest)  # must not raise
    assert "gravity" in reg


# --------------------------------------------------------------------------- #
# Item 2 — compat matrix (full-load-path companions to test_roles_vocab.py's
# function-level checks; see this file's module docstring for what already
# exists and is intentionally not repeated here).
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_loaded_vector_with_current_schema_version_loads(tmp_path):
    ptx = _doctor_sidecar(tmp_path, "gravity", {"schema_version": SCHEMA_VERSION})
    eagle.LoadedVector(ptx)  # must not raise


@pytest.mark.gpu
def test_loaded_vector_rejects_future_schema_version_naming_upgrade(tmp_path):
    # "Future" = beyond MAX_SCHEMA_VERSION (raptor's ceiling, the transition
    # bridge), not SCHEMA_VERSION + 1: this split the two constants, and
    # SCHEMA_VERSION + 1 (schema v2) is now an ACCEPTED version.
    ptx = _doctor_sidecar(
        tmp_path, "gravity", {"schema_version": MAX_SCHEMA_VERSION + 1}
    )
    with pytest.raises(ValueError, match="upgrade eagle"):
        eagle.LoadedVector(ptx)


def test_load_manifest_rejects_future_schema_version_naming_upgrade(tmp_path):
    # check_schema_version runs before load_manifest ever imports cupy (it's
    # ahead of the LoadedVector/LoadedPure construction loop) -> CPU-only.
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    shutil.copy(FIX / "gravity.json", tmp_path / "gravity.json")
    manifest = _write_manifest(
        tmp_path, [_one_entry("gravity")],
        extra={"schema_version": MAX_SCHEMA_VERSION + 1},
    )
    with pytest.raises(ValueError, match="upgrade eagle"):
        eagle.load_manifest(manifest)


# --------------------------------------------------------------------------- #
# Item 3 — the frozen forward-strict field list (acceptance design (c)).
#
# KEEP IN SYNC with the C++ mirror,
# eagle/tests/test_SchemaHardening.cpp::kFrozenStrictFields.
# --------------------------------------------------------------------------- #

#: The schema-v1 fields validated forward-strict by at least one loader (an
#: unrecognized value, or an unrecognized structural violation, is REJECTED
#: at load rather than silently accepted) — mapped to its CLASS
#: (restated here per the normative table): ``"gate"`` (aether_abi, schema_version — a
#: compatibility/version certification, not a schema-variant selector) |
#: ``"discriminant"`` (SELECTS
#: which schema VARIANT an artifact is; single-sourced vocabulary in
#: ``plugin/roles.h`` + :mod:`eagle.roles`, ONE check per field in the shared
#: validation layer — no loader may implement a private literal comparison for
#: a classed field outside it) | ``"structural"`` (a required-key /
#: well-formedness check, not a variant selector) | ``"optional-additive"`` (a
#: purely additive OPTIONAL field — absent is always fine). Most additive hints
#: (e.g. the reserved ``launch`` block) are NOT forward-strict and so never belong
#: in this list at all; ``vjp_exec`` / ``jvp_exec`` are the exception that earns the
#: class an entry here, because they are optional but strictly SHAPED when present.
#:
#: `pattern` here is the MANIFEST-level field: it is forward-strict in
#: Python's registry.py::load_manifest, AND — since a later ruling overturned an
#: earlier "deliberate asymmetry" decision — in both C++ registries too
#: (plugin_registry/registry.h::from_manifest, host_registry.h::load). The
#: C++ manifest PARSER (plugin/plugin_registry/manifest.h) still never
#: validates it — parsers stay validation-free by design; the check lives at
#: registry level in both languages (see plugin_schema.rst's C++/Python
#: asymmetry note for the narrower absence-strictness asymmetry this does
#: NOT close).
#:
#: The three trailing entries are later additions: `buffer kind`
#: (closes the silent `kind == "lookup"` dispatch filter),
#: `derivative.kind` (strict already — restated here, not newly added
#: behaviour), and `format` (the manifest-entry artifact container, lands
#: alongside it). An earlier pass added the buffer-`kind` CHECK without touching
#: this list — proof the frozen guard could not mechanically detect a
#: strict-field addition; restating membership here is what makes it able to.
#: The six trailing entries are the atomic-widening additions:
#: the `neural_block` descriptor's own strict surface. `exec_ref kind` and
#: `scatter_policy` are exactly the TWO reserved discriminant slots the comment
#: above left open — the register's v1 discriminant set is now closed at seven.
#: `neural_block required fields` is ONE entry standing for the whole
#: `eagle.roles.NEURAL_REQUIRED_FIELDS` constant, which both validators ITERATE:
#: naming the list here rather than its eight members is what keeps this register a
#: register instead of a second spelling of the vocabulary.
FROZEN_STRICT_FIELDS = {
    "schema_version": "gate",
    "kernel": "structural",
    "pattern": "discriminant",
    "aether_abi": "gate",
    "arg_spec role": "structural",
    "scalar_type": "discriminant",
    "buffer kind": "discriminant",
    "derivative.kind": "discriminant",
    "format": "discriminant",
    "forward_exec": "structural",
    "vjp_exec": "optional-additive",
    "jvp_exec": "optional-additive",
    "exec_ref kind": "discriminant",
    "scatter_policy": "discriminant",
    "neural_block required fields": "structural",
}

#: The CLASS values this register defines — every FROZEN_STRICT_FIELDS value must be
#: one.
FROZEN_FIELD_CLASSES = frozenset(
    {"gate", "discriminant", "structural", "optional-additive"}
)

_VERSION_BUMP_RULE = (
    "FROZEN_STRICT_FIELDS changed. Before editing it: does this change the "
    "MEANING or REQUIREDNESS of an EXISTING field, or add a NEW field a "
    "consumer must read to launch a kernel safely? If yes, bump "
    "schema_version (eagle.roles.SCHEMA_VERSION / "
    "plugin/roles.h::kPluginSchemaVersion, single-sourced) and update this "
    "list AND its behavioral pins below (and the C++ mirror). A purely "
    "additive OPTIONAL hint (e.g. the reserved `launch` block) does not "
    "need a bump. See eagle/docs/content/devguide/plugin_schema.rst "
    "'Compatibility policy'."
)


def test_frozen_strict_field_list_is_exactly_fifteen():
    assert set(FROZEN_STRICT_FIELDS) == {
        "schema_version", "kernel", "pattern", "aether_abi",
        "arg_spec role", "scalar_type",
        "buffer kind", "derivative.kind", "format",
        "forward_exec", "vjp_exec", "jvp_exec",
        "exec_ref kind", "scatter_policy", "neural_block required fields",
    }, _VERSION_BUMP_RULE
    assert len(FROZEN_STRICT_FIELDS) == 15, _VERSION_BUMP_RULE


def test_frozen_field_classes_are_all_recognized():
    for field, cls in FROZEN_STRICT_FIELDS.items():
        assert cls in FROZEN_FIELD_CLASSES, (
            f"{field!r} has unrecognized class {cls!r}; the CLASS column "
            f"is one of {sorted(FROZEN_FIELD_CLASSES)}"
        )


def test_frozen_field_classes_match_s9_d2():
    # The discriminant register is now CLOSED for v1 at seven: an earlier step
    # minted the two slots reserved for it — `exec_ref kind` and `scatter_policy`.
    # Everything else here is a gate, a structural check, or one of the
    # two optional-additive derivative exec references.
    assert {f for f, c in FROZEN_STRICT_FIELDS.items() if c == "discriminant"} == {
        "pattern", "scalar_type", "buffer kind", "derivative.kind", "format",
        "exec_ref kind", "scatter_policy",
    }
    assert {f for f, c in FROZEN_STRICT_FIELDS.items() if c == "gate"} == {
        "schema_version", "aether_abi",
    }
    assert {f for f, c in FROZEN_STRICT_FIELDS.items() if c == "structural"} == {
        "kernel", "arg_spec role", "forward_exec", "neural_block required fields",
    }
    assert {
        f for f, c in FROZEN_STRICT_FIELDS.items() if c == "optional-additive"
    } == {"vjp_exec", "jvp_exec"}


def test_frozen_field_schema_version_rejects_newer():
    assert "schema_version" in FROZEN_STRICT_FIELDS
    # beyond MAX_SCHEMA_VERSION, not SCHEMA_VERSION + 1 (now an accepted v2).
    with pytest.raises(ValueError, match="upgrade eagle"):
        check_schema_version({"schema_version": MAX_SCHEMA_VERSION + 1}, name="k")


def test_frozen_field_arg_spec_role_rejects_unknown():
    assert "arg_spec role" in FROZEN_STRICT_FIELDS
    with pytest.raises(ValueError, match="unknown arg role"):
        validate_roles([("bogus_role", "x")], name="k")


def test_frozen_field_aether_abi_rejects_mismatch():
    assert "aether_abi" in FROZEN_STRICT_FIELDS
    with pytest.raises(ValueError, match="AETHER ABI"):
        check_aether_abi(
            {"aether_abi": "aether-abi/0-stale"}, kind="plugin manifest", name="m"
        )


def test_frozen_field_pattern_rejects_unknown_at_manifest_level(tmp_path):
    assert "pattern" in FROZEN_STRICT_FIELDS
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    shutil.copy(FIX / "gravity.json", tmp_path / "gravity.json")
    manifest = _write_manifest(
        tmp_path, [_one_entry("gravity")], pattern="not_a_real_pattern",
    )
    with pytest.raises(ValueError, match="not a supported plugin family"):
        eagle.load_manifest(manifest)


@pytest.mark.gpu
def test_frozen_field_kernel_is_a_required_sidecar_key(tmp_path):
    assert "kernel" in FROZEN_STRICT_FIELDS
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    meta = json.loads((FIX / "gravity.json").read_text())
    del meta["kernel"]
    (tmp_path / "gravity.json").write_text(json.dumps(meta, indent=2))
    with pytest.raises(KeyError):
        eagle.LoadedVector(tmp_path / "gravity.ptx")


@pytest.mark.gpu
def test_frozen_field_scalar_type_rejects_unknown_value(tmp_path):
    assert "scalar_type" in FROZEN_STRICT_FIELDS
    ptx = _doctor_sidecar(
        tmp_path, "gravity", {"scalar_type": "int8_not_a_real_scalar_type"}
    )
    with pytest.raises(ValueError, match="unknown sidecar scalar_type"):
        eagle.LoadedVector(ptx)


# --------------------------------------------------------------------------- #
# The LoadedVector/LoadedPure sidecar-`pattern` absence asymmetry:
# `LoadedVector` DEFAULTS an absent `pattern` to "vector";
# `LoadedPure` has no default and REJECTS absence. Not a shared-conformance-corpus
# row: `_require_pattern` runs AFTER `LoadedKernel.__init__`'s `cupy.RawModule`
# device-module load (both the accept AND the reject case need a real device load
# to reach it), which would break the corpus's GPU-free / no-real-artifact-bytes
# design. Fixed here instead, against the committed gravity/bump PTX
# fixtures — the same GPU-marked style as this file's other full-load-path fixes.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_loaded_vector_defaults_absent_pattern_to_vector(tmp_path):
    """`LoadedVector`'s kind-DISPATCH default (`_require_pattern`, retained):
    an untagged (pre-freeze) sidecar with NO `pattern` key at
    all is accepted as a vector kernel by definition."""
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    meta = json.loads((FIX / "gravity.json").read_text())
    del meta["pattern"]
    (tmp_path / "gravity.json").write_text(json.dumps(meta, indent=2))
    loaded = eagle.LoadedVector(tmp_path / "gravity.ptx")  # must not raise
    assert loaded.kernel_name == "raptor_kernel"


@pytest.mark.gpu
def test_loaded_pure_rejects_absent_pattern(tmp_path):
    """The other half of the same locked asymmetry: `LoadedPure` has no
    absence default — an absent `pattern` is rejected (mismatch against
    "pure"), unlike `LoadedVector` above."""
    shutil.copy(FIX / "bump.ptx", tmp_path / "bump.ptx")
    meta = json.loads((FIX / "bump.json").read_text())
    del meta["pattern"]
    (tmp_path / "bump.json").write_text(json.dumps(meta, indent=2))
    with pytest.raises(ValueError, match="not a pure plugin"):
        eagle.LoadedPure(tmp_path / "bump.ptx")


# --------------------------------------------------------------------------- #
# Behavioural fixes for the six frozen entries the atomic-widening pass added.
# The register above records MEMBERSHIP; these prove the strictness
# is real — the earlier lesson (a CHECK landed without a register entry, so the frozen
# guard could not detect it) inverted: a register entry without a behavioural fix is
# just as mute. Pure dict-level, no GPU: the shared validator is the one gate, and
# nothing launches a descriptor, so there is no full-load-path companion to add.
# --------------------------------------------------------------------------- #
def _neural_golden(**patch):
    """The row01 conformance golden as a dict; ``patch`` merges/overrides top-level
    keys, and a value of ``None`` DELETES the key (to record a missing-field reject)."""
    meta = {
        "schema_version": SCHEMA_VERSION,
        "pattern": "neural_block",
        "scalar_type": "float64",
        "forward_exec": {"kind": "kernel", "kernel": "mlp_block_fwd"},
        "kernel": "mlp_block",
        "in_degree": 4,
        "out_degree": 2,
        "input_width": 8,
        "output_width": 3,
        "state_width": 6,
        "param_width": 24,
        "scatter_policy": "unique_write",
        "arg_spec": [],
    }
    for key, value in patch.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    return meta


def test_neural_golden_validates():
    """The counter-gate for every reject below: each one doctors exactly one field of
    THIS dict, so the only thing that can make it fail is the field under test."""
    validate_sidecar(_neural_golden(), name="descriptor")


def test_frozen_field_forward_exec_is_required():
    assert "forward_exec" in FROZEN_STRICT_FIELDS
    with pytest.raises(
        ValueError, match="missing required neural_block field 'forward_exec'"
    ):
        validate_sidecar(_neural_golden(forward_exec=None), name="descriptor")


def test_frozen_field_neural_required_fields_are_iterated():
    """ONE entry stands for the whole constant, so this test walks it: EVERY member
    must be individually required, or the register entry overstates the check."""
    assert "neural_block required fields" in FROZEN_STRICT_FIELDS
    for field in sorted(NEURAL_REQUIRED_FIELDS):
        with pytest.raises(
            ValueError, match=f"missing required neural_block field '{field}'"
        ):
            validate_sidecar(_neural_golden(**{field: None}), name="descriptor")


def test_frozen_field_exec_ref_kind_rejects_unknown():
    assert "exec_ref kind" in FROZEN_STRICT_FIELDS
    with pytest.raises(ValueError, match="forward_exec.kind"):
        validate_sidecar(
            _neural_golden(forward_exec={"kind": "plan_bundle", "kernel": "k"}),
            name="descriptor",
        )


def test_frozen_field_scatter_policy_rejects_unknown():
    assert "scatter_policy" in FROZEN_STRICT_FIELDS
    with pytest.raises(ValueError, match="scatter_policy"):
        validate_sidecar(
            _neural_golden(scatter_policy="unique_target"), name="descriptor"
        )


def test_frozen_field_vjp_and_jvp_exec_are_optional_but_strict():
    """optional-ADDITIVE: absent is fine (the golden carries neither), present is
    validated with the same strictness as ``forward_exec``."""
    assert FROZEN_STRICT_FIELDS["vjp_exec"] == "optional-additive"
    assert FROZEN_STRICT_FIELDS["jvp_exec"] == "optional-additive"
    validate_sidecar(_neural_golden(), name="descriptor")  # both absent -> accept
    for field in ("vjp_exec", "jvp_exec"):
        validate_sidecar(
            _neural_golden(**{field: {"kind": "kernel", "kernel": "d"}}),
            name="descriptor",
        )
        with pytest.raises(ValueError, match=f"{field}.kind"):
            validate_sidecar(
                _neural_golden(**{field: {"kind": "not_a_kind", "kernel": "d"}}),
                name="descriptor",
            )
