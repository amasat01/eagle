# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Canonical plugin arg-spec role vocabulary + schema version (the Python half).

The sidecar ``arg_spec`` is an ordered list of ``[role, name]`` pairs; ``role``
is one of the fixed strings below. The C++ half is ``plugin/roles.h``, and a
cross-check test (``tests/test_roles_vocab.py``) asserts the two lists are
identical. Every loader validates each arg_spec role against :data:`ROLES`
at load time (forward-strict), so the Python ``Loaded*`` and the C++
``PluginRegistry`` accept exactly the same vocabulary.

``mat_in`` is a valid schema-v1 role (a pure kernel's matrix input). A matrix
binds through the same 40-byte GRef mirror as a vector (a matrix GRef is a
width-``R*C`` vector GRef; the R x C shape is in-kernel flat indexing only),
so every loader packs it as that same layout, validating the bound shape
against the sidecar's ``mat_shapes`` / Mutable ``shape``. Only the cupy
device path calls ``eagle.abi.make_gref``; the C++ host ``PluginRegistry``
and the ctypes ``HostPluginLibrary`` each build their own byte-compatible
``GRefMirror``-shaped struct directly. All three constructions produce the
identical 40-byte ABI layout; only the source language and POD type differ.

:func:`classify_arg` is the single role -> ABI-shape classifier both launch
paths (``eagle.launch.assemble_args`` and ``eagle.host_launch.
HostPluginLibrary.run``) dispatch on, so an unrecognised role always raises
rather than silently falling through in only one path. The two paths still
build different artifacts (cupy objects vs ctypes structs); only the
classification decision is shared.
"""

from __future__ import annotations

from raptor.schema import blocks, manifest

#: The 12 canonical arg-spec roles (schema v1). Keep in sync with
#: ``plugin/roles.h::kPluginArgRoles`` (set-equality, order-independent).
#: ``accum_out`` names the cross-sample accumulate plane; it resolves to
#: the same ABI as ``wide_out`` (see :func:`classify_arg` / :data:`ARG_TAGS`).
ROLES = frozenset(
    {
        "out",
        "vec_in",
        "mat_in",
        "per_sample",
        "lookup",
        "mutable",
        "terminated",
        "uniform",
        "nsamples",
        "wide_in",
        "wide_out",
        "accum_out",
    }
)

#: The roles of an output plane: allocated when the caller supplies none, and
#: returned by :meth:`eagle.plan.Plan.run`.
OUTPUT_ROLES = frozenset({"out", "mutable", "wide_out", "accum_out"})

#: The roles of an input plane the caller supplies by name.
INPUT_ROLES = frozenset({"per_sample", "vec_in", "mat_in", "terminated", "lookup",
                         "wide_in"})

#: The roles whose plane holds one element (or one column) per sample.
PER_SAMPLE_ROLES = frozenset({"per_sample", "vec_in", "mat_in", "terminated", "out",
                              "mutable"})

#: The roles of a plane a stepping model updates (``eagle.simulate``'s ``state``).
STATE_ROLES = frozenset({"mutable", "accum_out", "wide_out"})

#: The current + maximum plugin-schema version this loader understands. A
#: sidecar/manifest tagged with a higher ``schema_version`` is rejected
#: (forward-strict); an untagged artifact is treated as v1
#: (backward-lenient). Orthogonal to the ``"aether-abi/1"`` ABI tag (see
#: :mod:`eagle.abi`). Single-sourced with
#: ``raptor.schema.manifest.SCHEMA_VERSION``; the C++ half
#: (``plugin/roles.h::kPluginSchemaVersion``) stays a source-level literal,
#: cross-checked by ``tests/test_roles_vocab.py``.
SCHEMA_VERSION = manifest.SCHEMA_VERSION

#: The highest ``schema_version`` :func:`check_schema_version` accepts
#: (v1 and v2 both load today, v3+ does not) — a separate constant from
#: :data:`SCHEMA_VERSION`, not a bump of it. Attribute-derived, so a
#: "future schema version" anywhere in tests is ``MAX_SCHEMA_VERSION + 1``,
#: never a hardcoded literal.
MAX_SCHEMA_VERSION = manifest.MAX_SCHEMA_VERSION

#: The plugin families the shared sidecar validator
#: (:func:`eagle.sidecar.validate_sidecar`) can structurally validate.
#: ``neural_block`` is recognized but not launched by any loader. Keep in
#: sync with ``plugin/roles.h::kRecognizedPatterns``; sourced from
#: ``raptor.schema.blocks.ALL_PATTERNS``.
RECOGNIZED_PATTERNS = blocks.ALL_PATTERNS

#: The plugin families a launching entry point will actually bind and
#: run — a subset of :data:`RECOGNIZED_PATTERNS`. ``neural_block`` is
#: recognized but never launched; every launching door uses the shared
#: message in :func:`check_launch_certified_pattern`. The Python manifest
#: door (:func:`eagle.registry.load_manifest`) dispatches on this set plus
#: ``neural_block`` itself (its sole descriptor-consuming branch). Keep in
#: sync with ``plugin/roles.h::kLaunchCertifiedPatterns``.
LAUNCH_CERTIFIED_PATTERNS = frozenset({"vector", "pure"})

#: The ``kind`` discriminant of an exec reference
#: (``{"kind": ..., "kernel": ...}``) on a ``neural_block`` descriptor.
#: One value in v1: the referenced artifact is a plain plugin kernel. A
#: future aggregate/plan-bundle kind is a meaning change, so a
#: ``schema_version`` bump. Keep in sync with
#: ``plugin/roles.h::kExecRefKinds``; sourced from
#: ``raptor.schema.manifest.EXEC_REF_KINDS``.
EXEC_REF_KINDS = manifest.EXEC_REF_KINDS

#: The declared terminal-write contract of a block's scatter.
#:
#: ``scatter_policy`` declares what the committed result MEANS, never the
#: mechanism eagle uses to commit it. ``unique_write`` (v1's sole value):
#: every ``(target, slot)`` is written by exactly one source per step, so
#: the commit is a plain store, deterministic with zero atomics (the
#: bit-exact gate mode applies). ``accumulate``: more than one source may
#: write the same ``(target, slot)`` per step, and the result is the
#: carried base plus an order-unspecified sum (band-gated, never
#: bit-exact). Keep in sync with ``plugin/roles.h::kScatterPolicies``;
#: sourced from ``raptor.schema.blocks.SCATTER_POLICIES``.
SCATTER_POLICIES = blocks.SCATTER_POLICIES

#: The fields a ``neural_block`` descriptor must carry, beyond the
#: general required set (``schema_version``, ``kernel``, ``pattern``,
#: ``scalar_type``, an empty ``arg_spec``). Keep in sync with
#: ``plugin/roles.h::kNeuralRequiredFields``; sourced from
#: ``raptor.schema.blocks.NEURAL_REQUIRED_FIELDS``.
NEURAL_REQUIRED_FIELDS = blocks.NEURAL_REQUIRED_FIELDS

#: The keys whose values are exec references. ``forward_exec`` is
#: required; the two derivative refs are optional-additive. Keep in sync
#: with ``plugin/roles.h::kNeuralExecRefFields``; sourced from
#: ``raptor.schema.blocks.NEURAL_EXEC_REF_FIELDS``.
NEURAL_EXEC_REF_FIELDS = blocks.NEURAL_EXEC_REF_FIELDS

#: Kernel-machinery fields a descriptor must not carry (each would
#: otherwise be silently ignored rather than fail loudly). ``aether_abi``,
#: ``derivative``, ``buffers``, ``mutables``, ``mat_shapes``,
#: ``host_entry`` — a descriptor binds nothing. Keep in sync with
#: ``plugin/roles.h::kNeuralForbiddenFields``; sourced from
#: ``raptor.schema.blocks.NEURAL_FORBIDDEN_FIELDS``.
NEURAL_FORBIDDEN_FIELDS = blocks.NEURAL_FORBIDDEN_FIELDS

#: The declared-buffer ``kind`` vocabulary (schema v1) — the only kinds a
#: loader can bind; an unrecognized kind is rejected up front rather
#: than silently dropped. Sourced from ``raptor.schema.blocks.BUFFER_KINDS``.
BUFFER_KINDS = blocks.BUFFER_KINDS

#: The manifest-entry ``format`` vocabulary (schema v1) — the artifact
#: container a manifest entry names; an unrecognized value is refused
#: rather than handed to a loader. Keep in sync with
#: ``plugin/roles.h::kManifestFormats``; sourced from
#: ``raptor.schema.manifest.MANIFEST_FORMATS``.
MANIFEST_FORMATS = manifest.MANIFEST_FORMATS


def validate_roles(arg_spec, *, name: str) -> None:
    """Reject an ``arg_spec`` carrying a role outside :data:`ROLES` (forward-strict).

    ``arg_spec`` is the sidecar's list of ``(role, name)`` pairs; ``name`` names the
    artifact in the error. Mirrors ``is_valid_role`` on the C++ side."""
    for entry in arg_spec:
        role = entry[0]
        if role not in ROLES:
            raise ValueError(
                f"{name!r}: unknown arg role {role!r}; schema-v1 roles are "
                f"{sorted(ROLES)}"
            )


#: The ABI-shape tags a role resolves to — the classification both launch
#: paths dispatch on, single-sourced so ``eagle.launch.assemble_args``
#: and ``eagle.host_launch.HostPluginLibrary.run`` can never diverge.
#:
#: ``GREF_VEC``/``GREF_MAT`` stay distinct (rather than one shared
#: ``GREF``) because the ctypes host path reads a "mutable" role's bound
#: value from one of two different caller-populated dicts (``self._vec``
#: vs ``self._mat``) depending on shape; the cupy device path's own
#: ``make_gref`` call is identical either way.
#:
#: ``WIDE_IN``/``WIDE_OUT`` and ``ACCUM_OUT`` are each kept distinct from
#: ``HANDLE``/``WIDE_OUT`` for the same reason: each reads from its own
#: caller-populated dict (``wide_in``/``wide_out``/``accum_out``), even
#: though construction is byte-identical to a plain scalar-handle pointer.
ARG_TAGS = frozenset(
    {
        "GREF_VEC",
        "GREF_MAT",
        "HANDLE",
        "NSAMPLES",
        "UNIFORM",
        "WIDE_IN",
        "WIDE_OUT",
        "ACCUM_OUT",
    }
)


def classify_arg(
    role: str,
    name: str,
    *,
    vec_mutables: frozenset = frozenset(),
    mat_mutables: frozenset = frozenset(),
) -> str:
    """The ABI-shape tag ``(role, name)`` resolves to — one of :data:`ARG_TAGS`.

    Extracts the classification only: both launch paths still build
    their own artifacts from whichever tag comes back (a cupy ``GRef``
    numpy-structured scalar vs a ctypes ``GRefMirror``/``ScalarHandle``
    POD) — that construction stays per-path.

    * ``"out"`` / ``"vec_in"`` -> ``GREF_VEC``; ``"mat_in"`` -> ``GREF_MAT``.
    * ``"mutable"`` resolves by context: ``GREF_MAT`` if ``name`` is in
      ``mat_mutables``, ``GREF_VEC`` if in ``vec_mutables``, else
      ``HANDLE`` (a scalar/int Mutable).
    * ``"per_sample"`` / ``"lookup"`` / ``"terminated"`` -> ``HANDLE``.
    * ``"wide_in"`` -> ``WIDE_IN``; ``"wide_out"`` -> ``WIDE_OUT``.
    * ``"accum_out"`` -> ``ACCUM_OUT`` (the cross-sample accumulate plane;
      same construction as ``WIDE_OUT``).
    * ``"nsamples"`` -> ``NSAMPLES``; ``"uniform"`` -> ``UNIFORM``.
    * Any other ``role`` -> ``ValueError`` (fail-loud; never a silent skip).
    """
    if role == "mat_in" or (role == "mutable" and name in mat_mutables):
        return "GREF_MAT"
    if role in ("out", "vec_in") or (role == "mutable" and name in vec_mutables):
        return "GREF_VEC"
    if role in ("per_sample", "lookup", "terminated", "mutable"):
        return "HANDLE"
    if role == "wide_in":
        return "WIDE_IN"
    if role == "wide_out":
        return "WIDE_OUT"
    if role == "accum_out":
        return "ACCUM_OUT"
    if role == "nsamples":
        return "NSAMPLES"
    if role == "uniform":
        return "UNIFORM"
    raise ValueError(
        f"classify_arg: unknown arg role {role!r} for {name!r}; schema-v1 roles "
        f"are {sorted(ROLES)}"
    )


def check_launch_certified_pattern(pattern, *, subject: str) -> None:
    """Refuse ``pattern`` at a launching door, with the two-branch message
    this module locks (mirrors the C++ ``check_launch_certified_pattern``
    in ``plugin/roles.h`` message for message).

    * ``pattern`` outside :data:`RECOGNIZED_PATTERNS` — this build has
      never heard of the family: the message names the supported set and
      says ``upgrade eagle``.
    * ``pattern`` recognized but not certified (e.g. a ``neural_block``
      descriptor, not runnable) — the message says ``recognized but not
      launchable by this loader``, with no upgrade suffix.

    ``subject`` names the artifact, followed directly by
    ``" pattern '<value>'"``. An absent/empty ``pattern`` is lenient (a
    pre-freeze artifact never stamped one)."""
    if not pattern or pattern in LAUNCH_CERTIFIED_PATTERNS:
        return
    supported = ", ".join(sorted(LAUNCH_CERTIFIED_PATTERNS))
    if pattern not in RECOGNIZED_PATTERNS:
        raise ValueError(
            f"{subject} pattern '{pattern}' is not a supported plugin family "
            f"(expected one of {supported}); upgrade eagle"
        )
    raise ValueError(
        f"{subject} pattern '{pattern}' is recognized but not launchable by this "
        f"loader (expected one of {supported})"
    )


#: Return the artifact's plugin-schema version, rejecting one we cannot
#: load.
#:
#: ``schema_version`` absent => v1 (backward-lenient). A version greater
#: than :data:`MAX_SCHEMA_VERSION` is rejected (forward-strict); v1 and
#: v2 both load today. Mirrors the C++ ``kPluginSchemaVersion`` gate
#: (still v1-only pending the twin-site bump).
#:
#: ``allow_legacy_version_key`` scopes the legacy ``version``-key
#: fallback to manifests only (pass ``True`` from
#: :func:`eagle.registry.load_manifest`); the default (``False``) matches
#: C++ sidecar behaviour, where an unrecognized ``version`` key is
#: ignored.
#:
#: Re-exported from raptor, which owns the version-compare wording.
check_schema_version = manifest.check_schema_version


#: Validate the schema-v2 execution axis, given the already-resolved
#: ``version`` (:func:`check_schema_version`'s return). v1 documents must
#: carry none of ``exec_targets``/``exec_access``/``exec_op``; v2
#: documents must carry ``exec_targets`` + ``exec_access`` (absence =
#: load refused) and validate their vocabulary + ``exec_op``'s
#: mapreduce-conditional requiredness.
#:
#: Re-exported from raptor, which owns the execution-axis shape. Wired
#: into :func:`eagle.registry.load_manifest`.
check_execution_axis = manifest.check_execution_axis


#: The sidecar ``params`` block's own wire version — the current +
#: maximum shape this build can read.
#:
#: * **1** (what an absent ``params_schema`` key means): ``"params":
#:   ["mu", "k"]`` — a bare name list, every uniform implicitly a ``Real``.
#: * **2**: ``"params": [{"name": "mu", "dtype": "float"}, ...]`` —
#:   decl-carrying, the same ``{name, dtype}`` shape ``mutables`` has
#:   always had, so an ``int`` uniform binds through the int binder
#:   instead of arriving widened through a double.
#:
#: Field-local rather than a ``schema_version`` bump: the C++ side never
#: reads ``params`` at all (it resolves uniforms by ``arg_spec`` role),
#: so no C++ reader can misread this block's new shape. Python-only by
#: construction, except the code generator's conformance-gated verbatim
#: copy, which must be bumped together with this one.
PARAMS_SCHEMA = 2


def param_decls(params):
    """Normalize a broadcast-param list to ``((name, dtype), …)``.

    One spelling for "what type is this uniform?", shared by every eagle
    consumer of a params list. Accepts:

    * a bare ``str`` — a v1 sidecar's name; defaults to ``float``.
    * a ``(name, dtype)`` pair.
    * anything with ``.name`` / ``.dtype`` (:class:`eagle.sidecar.ParamSpec`
      or a code generator's ``ParamDecl``) — so a generated
      ``TraceResult.params`` can be handed straight to a launch.
    * a ``{"name", "dtype"}`` dict — a v2 sidecar entry read raw.

    An unrecognised entry raises: a uniform whose type cannot be
    established is the exact silent-wrong-answer this normalization
    exists to prevent."""
    out = []
    for p in params:
        if isinstance(p, str):
            out.append((p, "float"))
        elif isinstance(p, dict):
            out.append((p["name"], p.get("dtype", "float")))
        elif isinstance(p, tuple) and len(p) == 2:
            out.append((p[0], p[1]))
        elif hasattr(p, "name") and hasattr(p, "dtype"):
            out.append((p.name, p.dtype))
        else:
            raise TypeError(
                f"broadcast param entry {p!r} carries neither a name nor a "
                "declared dtype; pass a name, a (name, dtype) pair, a "
                "{'name','dtype'} object, or an eagle.sidecar.ParamSpec"
            )
    return tuple(out)


def param_names(params) -> tuple[str, ...]:
    """Just the NAMES of a broadcast-param list (any of the shapes
    :func:`param_decls` accepts) — the binding-set / kwarg-key view."""
    return tuple(nm for nm, _ in param_decls(params))


def parse_derivative(meta: dict, *, name: str):
    """Return the sidecar's optional ``derivative`` block (or ``None``).

    A VJP/JVP derivative artifact carries an additive ``derivative`` block
    describing its role; an ordinary primal/kernel has none.
    Backward-lenient: an absent block returns ``None``. Validates the
    block's shape at load (mirroring the C++ ``validate_sidecar``):
    ``kind`` must be ``vjp``/``jvp``, and a populated ``residuals`` list
    is rejected (recompute-only for now). The returned dict is the
    loaded-kernel metadata surface (``LoadedKernel.derivative``)."""
    d = meta.get("derivative")
    if d is None:
        return None
    kind = d.get("kind")
    if kind not in ("vjp", "jvp"):
        raise ValueError(
            f"{name!r}: derivative.kind {kind!r} is not one of 'vjp' | 'jvp'"
        )
    if d.get("residuals"):
        raise ValueError(
            f"{name!r}: derivative.residuals is populated, but derivative "
            "artifacts are recompute-only (residuals must be []); a populated "
            "residuals list requires a schema_version bump"
        )
    return d
