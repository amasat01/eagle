# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The plugin arg-spec role vocabulary is single-sourced, and the schema_version
compat policy is backward-lenient / forward-strict.

``plugin/roles.h`` (C++) and ``eagle/roles.py`` (Python) must enumerate the SAME set
of canonical roles, every committed fixture's ``arg_spec`` must use only those
roles, and the two ``schema_version`` gates must agree. This is the test that keeps
the C++ and Python loaders validating identically (the ``mat_in`` drift that existed
before the freeze would fail here). Pure-Python: no GPU, no nvcc, no cupy import.

Also, the vocabulary-parity extension requires: EVERY discriminant
vocabulary single-sourced in ``plugin/roles.h`` / ``eagle.roles`` (not just
``ROLES``) gets the same C++-header-vs-Python-constant set-equality check, plus
``eagle.registry.load_manifest``'s ``loaders``-dict keys checked == the Python
launch-certified set: the four newer
constants (``RECOGNIZED_PATTERNS`` / ``LAUNCH_CERTIFIED_PATTERNS`` /
``BUFFER_KINDS`` / ``MANIFEST_FORMATS``) could otherwise drift with zero test failure.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from eagle.roles import (
    BUFFER_KINDS,
    EXEC_REF_KINDS,
    LAUNCH_CERTIFIED_PATTERNS,
    MANIFEST_FORMATS,
    MAX_SCHEMA_VERSION,
    NEURAL_EXEC_REF_FIELDS,
    NEURAL_FORBIDDEN_FIELDS,
    NEURAL_REQUIRED_FIELDS,
    RECOGNIZED_PATTERNS,
    ROLES,
    SCATTER_POLICIES,
    SCHEMA_VERSION,
    check_schema_version,
    validate_roles,
)

HERE = pathlib.Path(__file__).parent
FIXTURES = HERE / "fixtures"
ROLES_H = HERE.parent.parent / "plugin" / "roles.h"
REGISTRY_PY = HERE.parent / "eagle" / "registry.py"


def _cpp_string_array(varname: str) -> set:
    """The string literals of a ``const char* const <varname>[] = {...}`` array
    declared in ``plugin/roles.h`` — the single-sourced C++ half of a vocabulary
    constant ("vocabularies single-sourced in plugin/roles.h + roles.py")."""
    text = ROLES_H.read_text()
    m = re.search(rf"{varname}\[\]\s*=\s*\{{(.*?)\}}", text, re.DOTALL)
    assert m, f"could not find the {varname} initializer in plugin/roles.h"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _cpp_roles() -> set:
    """The ``kPluginArgRoles`` string literals declared in ``plugin/roles.h``."""
    return _cpp_string_array("kPluginArgRoles")


@pytest.mark.repo_local
def test_cpp_and_python_role_vocab_identical():
    """The single source of truth: C++ and Python enumerate the same roles."""
    assert _cpp_roles() == set(ROLES), (
        "plugin/roles.h::kPluginArgRoles and eagle.roles.ROLES have drifted; they are "
        "the single source of truth for schema v1 and must match exactly"
    )


@pytest.mark.repo_local
def test_cpp_and_python_schema_version_identical():
    text = ROLES_H.read_text()
    m = re.search(r"kPluginSchemaVersion\s*=\s*(\d+)", text)
    assert m, "could not find kPluginSchemaVersion in plugin/roles.h"
    assert int(m.group(1)) == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# The vocabulary-parity extension: every discriminant vocabulary, not
# just ROLES, gets the C++-header-vs-Python-constant set-equality check. This
# closed a debt where the RECOGNIZED / LAUNCH-CERTIFIED pattern sets, and
# BUFFER_KINDS, landed with zero cross-language parity test — meaning the exact
# constants the atomicity clause's "one-line widening" will edit could drift
# invisibly until the neural golden lands. MANIFEST_FORMATS landed alongside
# it and gets the same check from day one.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_cpp_and_python_recognized_patterns_identical():
    assert _cpp_string_array("kRecognizedPatterns") == set(RECOGNIZED_PATTERNS), (
        "plugin/roles.h::kRecognizedPatterns and eagle.roles.RECOGNIZED_PATTERNS "
        "have drifted; both are the single source of truth for the RECOGNIZED "
        "set and must match exactly"
    )


@pytest.mark.repo_local
def test_cpp_and_python_launch_certified_patterns_identical():
    assert _cpp_string_array("kLaunchCertifiedPatterns") == set(
        LAUNCH_CERTIFIED_PATTERNS
    ), (
        "plugin/roles.h::kLaunchCertifiedPatterns and "
        "eagle.roles.LAUNCH_CERTIFIED_PATTERNS have drifted; both are the single "
        "source of truth for the LAUNCH-CERTIFIED set and must match exactly"
    )


def test_launch_certified_patterns_is_subset_of_recognized():
    """LAUNCH-CERTIFIED is a subset of RECOGNIZED, nested and both
    value-strict — never peers, never disjoint."""
    assert set(LAUNCH_CERTIFIED_PATTERNS) <= set(RECOGNIZED_PATTERNS)


@pytest.mark.repo_local
def test_cpp_and_python_buffer_kinds_identical():
    assert _cpp_string_array("kBufferKinds") == set(BUFFER_KINDS), (
        "plugin/roles.h::kBufferKinds and eagle.roles.BUFFER_KINDS have drifted; "
        "both are the single source of truth for the declared-buffer kind "
        "vocabulary and must match exactly"
    )


@pytest.mark.repo_local
def test_cpp_and_python_manifest_formats_identical():
    assert _cpp_string_array("kManifestFormats") == set(MANIFEST_FORMATS), (
        "plugin/roles.h::kManifestFormats and eagle.roles.MANIFEST_FORMATS have "
        "drifted; both are the single source of truth for the manifest-entry "
        "`format` vocabulary and must match exactly"
    )


# --------------------------------------------------------------------------- #
# The `neural_block` descriptor vocabularies. Same single-sourcing
# idiom as everything above: the constant pair IS the schema, so a one-sided edit is
# a mechanical failure rather than an inspection finding.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_cpp_and_python_exec_ref_kinds_identical():
    assert _cpp_string_array("kExecRefKinds") == set(EXEC_REF_KINDS), (
        "plugin/roles.h::kExecRefKinds and eagle.roles.EXEC_REF_KINDS have drifted; "
        "both are the single source of truth for the exec-reference `kind` "
        "discriminant and must match exactly"
    )


@pytest.mark.repo_local
def test_cpp_and_python_scatter_policies_identical():
    assert _cpp_string_array("kScatterPolicies") == set(SCATTER_POLICIES), (
        "plugin/roles.h::kScatterPolicies and eagle.roles.SCATTER_POLICIES have "
        "drifted; both are the single source of truth for the terminal-write "
        "contract vocabulary and must match exactly"
    )


@pytest.mark.repo_local
def test_cpp_and_python_neural_required_fields_identical():
    assert _cpp_string_array("kNeuralRequiredFields") == set(NEURAL_REQUIRED_FIELDS), (
        "plugin/roles.h::kNeuralRequiredFields and eagle.roles.NEURAL_REQUIRED_FIELDS "
        "have drifted; both validators ITERATE this list, so a one-sided edit makes "
        "one language require a field the other ignores"
    )


@pytest.mark.repo_local
def test_cpp_and_python_neural_forbidden_fields_identical():
    assert _cpp_string_array("kNeuralForbiddenFields") == set(
        NEURAL_FORBIDDEN_FIELDS
    ), (
        "plugin/roles.h::kNeuralForbiddenFields and "
        "eagle.roles.NEURAL_FORBIDDEN_FIELDS have drifted; both validators ITERATE "
        "this list, so a one-sided edit makes one language silently accept kernel "
        "machinery on a descriptor (the mis-read hazard)"
    )


@pytest.mark.repo_local
def test_cpp_and_python_neural_exec_ref_fields_identical():
    """The C++ parser BOUNDS these objects before scanning top-level keys, so a
    name present in one language and not the other is not merely a
    validation gap — it is an unmasked nested object and a mis-parsed `kernel`."""
    assert _cpp_string_array("kNeuralExecRefFields") == set(NEURAL_EXEC_REF_FIELDS), (
        "plugin/roles.h::kNeuralExecRefFields and eagle.roles.NEURAL_EXEC_REF_FIELDS "
        "have drifted; both are the single source of truth for which keys hold exec "
        "references and must match exactly"
    )


def test_scatter_policies_excludes_mechanism_and_dead_spellings():
    """`scatter_policy` names the terminal-write CONTRACT, never the
    MECHANISM — so `atomic_add`, `block_reduce_then_atomic` and `segmented_reduce`
    (all executors of the SAME accumulate contract) may never enter the vocabulary;
    promoting mechanism to the wire would make every performance experiment a
    schema event. `unique_target` is barred for a different reason: it is a
    dead synonym for `unique_write`, overturned before any code existed,
    and permanently fixed dead here and by conformance row 5.

    The C++ half is covered by the `kScatterPolicies` set-equality parity above, so
    this one assertion checks both languages. Amending it requires overturning
    that ruling, in the same change."""
    assert {
        "unique_target",
        "atomic_add",
        "block_reduce_then_atomic",
        "segmented_reduce",
    }.isdisjoint(SCATTER_POLICIES)


def test_scatter_policies_cardinality_pinned():
    """The lock: "no second scatter_policy value before the accumulate contract
    landed" was a CARDINALITY lock at single-valued; landing `accumulate` in the
    SAME change converted
    `test_layer_rejects_non_unanimous_scatter_policies`'s monkeypatched vocabulary to
    real values (its second value stopped being monkeypatch-only the moment this
    assertion widened). The lock now fixes the vocabulary at exactly TWO values — no
    third spelling may enter without amending this test, the same discipline the
    disjointness assertion above applies to the four named-dead mechanism spellings.
    The C++ half is already covered by the `kScatterPolicies` set-equality parity
    above."""
    assert SCATTER_POLICIES == frozenset({"unique_write", "accumulate"})


@pytest.mark.repo_local
def test_neural_block_is_recognized_but_never_launch_certified():
    """The atomicity clause, at its narrowest: a descriptor is structurally
    validatable and never launchable. If `neural_block` ever enters the certified
    set, some loader will bind a descriptor as a bare kernel."""
    assert "neural_block" in RECOGNIZED_PATTERNS
    assert "neural_block" not in LAUNCH_CERTIFIED_PATTERNS
    assert "neural_block" not in _cpp_string_array("kLaunchCertifiedPatterns")
    assert "neural_block" not in _registry_loader_pattern_keys()


def _registry_loader_pattern_keys() -> set:
    """The string keys of ``eagle.registry.load_manifest``'s ``loaders`` dict,
    extracted from SOURCE TEXT (mirroring how the C++ vocabularies above are
    parsed) rather than imported at runtime: ``registry.py`` keeps its
    ``LoadedVector``/``LoadedPure`` import function-local by design (deferred,
    not eagerly pulled in at package-import time), and this test must not change
    that."""
    text = REGISTRY_PY.read_text()
    m = re.search(r"loaders\s*=\s*\{(.*?)\}", text, re.DOTALL)
    assert m, "could not find the `loaders = {...}` dict in eagle/registry.py"
    return set(re.findall(r'"([^"]+)"\s*:', m.group(1)))


@pytest.mark.repo_local
def test_registry_loaders_dict_keys_match_launch_certified_patterns():
    """``registry.py``'s ``loaders`` dict is the
    de-facto Python manifest-level launch gate — check its keys == the Python
    launch-certified set (it never referenced the constant before now; the
    two could drift with zero test failure)."""
    assert _registry_loader_pattern_keys() == set(LAUNCH_CERTIFIED_PATTERNS), (
        "eagle.registry.load_manifest's `loaders` dict keys have drifted from "
        "LAUNCH_CERTIFIED_PATTERNS (the Python `loaders` dict's keys are "
        "the launch-certified set)"
    )


def test_all_fixture_roles_are_canonical():
    """Every committed fixture sidecar uses only canonical roles."""
    seen = 0
    for sc in sorted(FIXTURES.glob("*.json")):
        meta = json.loads(sc.read_text())
        spec = meta.get("arg_spec")
        if spec is None:  # the manifest fixture carries no arg_spec
            continue
        roles = {entry[0] for entry in spec}
        assert roles <= set(ROLES), (
            f"{sc.name}: non-canonical roles {roles - set(ROLES)}"
        )
        seen += 1
    assert seen > 0, "no arg_spec-bearing fixtures found"


def test_validate_roles_rejects_unknown():
    with pytest.raises(ValueError, match="unknown arg role"):
        validate_roles([("out", "out"), ("bogus_role", "x")], name="kern")


def test_validate_roles_accepts_mat_in():
    """mat_in is a canonical role (the pre-freeze drift is resolved)."""
    validate_roles([("mat_in", "A"), ("mutable", "state")], name="kern")


def test_schema_version_absent_is_v1():
    """Backward-lenient: an untagged (pre-freeze) artifact loads as v1."""
    assert check_schema_version({}, name="kern") == 1


def test_schema_version_legacy_version_fallback_scoped_to_manifests():
    """The legacy ``version`` key is a MANIFEST-only fallback: C++
    ``parse_manifest`` honors it, ``parse_sidecar`` never has. A SIDECAR's
    ``{"version": N}`` must therefore be ignored (schema_version absent -> v1,
    backward-lenient), not honored as a fallback — the opt-in keyword scopes it
    to :func:`~eagle.registry.load_manifest`'s call site.

    "Future" here means beyond :data:`MAX_SCHEMA_VERSION` (raptor's ceiling,
    the transition bridge), not ``SCHEMA_VERSION + 1``
    — this split the two constants, and ``SCHEMA_VERSION + 1`` (schema v2) is
    now an ACCEPTED version, not a future one."""
    # Manifest-shaped call (opted in): the legacy key is honored.
    assert (
        check_schema_version({"version": 1}, name="kern", allow_legacy_version_key=True)
        == 1
    )
    with pytest.raises(ValueError, match="upgrade eagle"):
        check_schema_version(
            {"version": MAX_SCHEMA_VERSION + 1},
            name="kern",
            allow_legacy_version_key=True,
        )
    # Sidecar-shaped call (default): the legacy key is NOT honored — a stray
    # ``version`` key is ignored, exactly like C++ ``parse_sidecar``, which never
    # reads it (absent ``schema_version`` -> v1 regardless of ``version``'s value).
    assert check_schema_version({"version": MAX_SCHEMA_VERSION + 1}, name="kern") == 1


def test_schema_version_current_ok():
    assert (
        check_schema_version({"schema_version": SCHEMA_VERSION}, name="kern")
        == SCHEMA_VERSION
    )


def test_schema_version_max_accepted_ok():
    """Schema v2 (:data:`MAX_SCHEMA_VERSION`) is accepted, not future — the
    transition bridge: v1 AND v2 both load today."""
    assert (
        check_schema_version({"schema_version": MAX_SCHEMA_VERSION}, name="kern")
        == MAX_SCHEMA_VERSION
    )


def test_schema_version_future_rejected():
    """Forward-strict: a version beyond :data:`MAX_SCHEMA_VERSION` is refused
    with an upgrade hint. Not ``SCHEMA_VERSION + 1``: that names schema v2,
    which this split made an ACCEPTED version (see
    :func:`test_schema_version_max_accepted_ok`)."""
    with pytest.raises(ValueError, match="upgrade eagle"):
        check_schema_version(
            {"schema_version": MAX_SCHEMA_VERSION + 1}, name="kern"
        )
