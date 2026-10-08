# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ONE shared schema-v1 sidecar validator (the Python half of
``plugin/sidecar.h``).

Every Python loader that reads a sidecar -- :class:`eagle.loaded.LoadedKernel`
and :class:`eagle.host_launch.HostPluginLibrary` -- funnels through
:func:`validate_sidecar` before it loads anything (before ``cupy.RawModule``
or ``ctypes.CDLL``), so a rejected artifact never reaches a driver load and
every reject case is exercisable with stub bytes and no GPU.

Absence is lenient here for ``pattern`` and ``scalar_type`` (a pre-freeze
sidecar never stamped either); per-loader absence rules (e.g.
``LoadedVector`` defaulting an absent ``pattern`` to ``"vector"``) stay with
the loader. The ``neural_block`` clause is imported, not defined here
(:func:`raptor.schema.blocks.validate_neural_block_descriptor`): the spine
declares that wire contract, this module owns only the kernel-family checks
(``schema_version``, ``arg_spec`` roles, ``scalar_type``, ``pattern``,
``derivative``, buffer ``kind``).

**The optional ``terminated_readonly`` declaration.** A producer may stamp
``"terminated_readonly": true`` on a kernel sidecar to declare that its
kernel only ever reads the ``terminated`` mask, letting
:func:`eagle.marshal.coerce_terminated` serve an omitted mask from a shared
cache instead of allocating a fresh one. Opt-in only the producer can
assert; absent means "not declared". Read by
:func:`read_terminated_readonly`, not by :func:`validate_sidecar` -- it is a
Python-side launch hint the C++ registry has no use for.
"""

from __future__ import annotations

from dataclasses import dataclass

from raptor.schema.blocks import validate_neural_block_descriptor

from .roles import (
    BUFFER_KINDS,
    PARAMS_SCHEMA,
    RECOGNIZED_PATTERNS,
    check_schema_version,
    parse_derivative,
    validate_roles,
)

#: The ``neural_block`` family name. A literal here (rather than a constant)
#: is deliberate: this clause validates exactly one family.
_NEURAL_BLOCK = "neural_block"

#: The optional sidecar key a producer stamps to opt its kernel into the
#: shared all-false ``terminated`` mask. See this module's docstring.
TERMINATED_READONLY_KEY = "terminated_readonly"


def read_terminated_readonly(meta: dict, *, name: str) -> bool:
    """Whether this sidecar declares its ``terminated`` mask read-only;
    ``False`` when the key is absent. Present-but-not-a-bool raises
    ``ValueError`` -- a truthiness coercion would let a typo like
    ``"false"`` silently enable the shared buffer."""
    if TERMINATED_READONLY_KEY not in meta:
        return False
    value = meta[TERMINATED_READONLY_KEY]
    if not isinstance(value, bool):
        raise ValueError(
            f"{name}: sidecar '{TERMINATED_READONLY_KEY}' must be a JSON "
            f"boolean (true/false); got {value!r} of type "
            f"{type(value).__name__}. It declares that the kernel never WRITES "
            "the terminated mask, which is what allows the mask buffer to be "
            "shared between launches — it is never inferred from truthiness."
        )
    return value


#: The optional sidecar key a producer stamps on a kernel that finishes its
#: own samples. Read by :func:`read_finish`.
FINISH_KEY = "finish"
#: The one counter plane a finishing kernel counts into.
FINISH_COUNTER = "finished_count"
_FINISH_FIELDS = ("mask", "counter", "steps")
#: The ``finish.steps`` value of a kernel whose steps per launch are a
#: run-time word (``hawk.steps(kernel, "auto")``); such a kernel also
#: stamps ``finish.steps_max``.
FINISH_AUTO = "auto"
_FINISH_OPTIONAL = ("steps_max",)


def read_finish(meta: dict, *, name: str) -> dict | None:
    """The kernel's ``finish`` declaration, or ``None`` when it never
    finishes: ``{"mask": "<plane>", "counter": "finished_count", "steps":
    K}`` (``K`` an int ``>= 1``, or ``"auto"`` paired with an integer
    ``steps_max``). A malformed declaration raises ``ValueError`` naming the
    artifact."""
    if FINISH_KEY not in meta:
        return None
    value = meta[FINISH_KEY]
    where = f"{name}: sidecar '{FINISH_KEY}'"
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object, got {value!r}")
    missing = [k for k in _FINISH_FIELDS if k not in value]
    extra = sorted(set(value) - set(_FINISH_FIELDS) - set(_FINISH_OPTIONAL))
    if missing or extra:
        raise ValueError(
            f"{where} carries exactly {list(_FINISH_FIELDS)} (and 'steps_max' "
            f"with steps 'auto'); missing {missing}, unknown {extra}")
    mask, counter, steps = value["mask"], value["counter"], value["steps"]
    if not isinstance(mask, str) or not mask:
        raise ValueError(f"{where}: 'mask' names the terminated plane, got {mask!r}")
    if counter != FINISH_COUNTER:
        raise ValueError(
            f"{where}: 'counter' is the reserved plane {FINISH_COUNTER!r}, got "
            f"{counter!r}")
    auto = isinstance(steps, str) and steps == FINISH_AUTO
    if not auto and (isinstance(steps, bool) or not isinstance(steps, int)
                     or steps < 1):
        raise ValueError(
            f"{where}: 'steps' is an integer >= 1 or {FINISH_AUTO!r}, got {steps!r}")
    out = {"mask": mask, "counter": counter, "steps": steps}
    if auto:
        top = value.get("steps_max")
        if isinstance(top, bool) or not isinstance(top, int) or top < 1:
            raise ValueError(
                f"{where}: steps {FINISH_AUTO!r} needs 'steps_max', an integer "
                f">= 1 (the most steps one launch may take), got {top!r}")
        out["steps_max"] = top
    elif "steps_max" in value:
        raise ValueError(
            f"{where}: 'steps_max' goes with steps {FINISH_AUTO!r} only, not "
            f"beside the fixed steps={steps}")
    if meta.get(TERMINATED_READONLY_KEY) is True:
        raise ValueError(
            f"{where} declares a kernel that WRITES its mask, but the sidecar also "
            f"stamps '{TERMINATED_READONLY_KEY}': true")
    return out


# The decl-carrying ``params`` block (the int-uniform addition). Read here,
# not inside :func:`validate_sidecar`: the C++ side never reads ``params``
# at all (it resolves uniforms by ``arg_spec`` role), so a Python-only
# clause in the shared validator would split the two languages' sets.

#: The sidecar key carrying the ``params`` block's own wire version. Absent => 1.
PARAMS_SCHEMA_KEY = "params_schema"

#: The uniform element types a v2 ``params`` entry may declare: ``float`` (a
#: ``Real`` argument) or ``int`` (the 8-byte signed ``Int p_<name>`` slot).
PARAM_DTYPES = ("float", "int")


@dataclass(frozen=True)
class ParamSpec:
    """One declared broadcast (uniform) parameter: its ``name`` and
    ``dtype`` (``"float"`` | ``"int"``). A v1 sidecar's bare name yields
    ``ParamSpec(name, "float")``."""

    name: str
    dtype: str = "float"


def read_params(meta: dict, *, name: str, required: bool = True):
    """The sidecar's declared broadcast params as a tuple of
    :class:`ParamSpec`. Accepts both wire shapes, keyed on
    :data:`~eagle.roles.PARAMS_SCHEMA`: v1 (absent or ``1``) is bare names
    (``["mu", "k"]``, every uniform a ``Real``); v2 is decl-carrying
    (``[{"name": "mu", "dtype": "float"}, ...]``).

    Raises loudly on a ``params_schema`` newer than this build, a shape
    disagreeing with the declared version, or an unknown ``dtype`` -- an
    integer uniform silently bound through a ``double`` is wrong from 2^53
    up, so every ambiguity here fails instead. ``required`` mirrors the two
    calling conventions in the tree: the device path reads
    ``meta["params"]``, the host path ``meta.get("params", [])``."""
    raw = meta["params"] if required else meta.get("params", ())
    version = meta.get(PARAMS_SCHEMA_KEY, 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(
            f"{name}: sidecar '{PARAMS_SCHEMA_KEY}' must be an integer; got "
            f"{version!r} of type {type(version).__name__}"
        )
    if version > PARAMS_SCHEMA:
        raise ValueError(
            f"{name}: sidecar '{PARAMS_SCHEMA_KEY}' is {version}, but this build "
            f"reads at most {PARAMS_SCHEMA}; the params block was written by a "
            "newer producer and its entries cannot be bound without guessing a "
            "uniform's by-value type — upgrade eagle (and the code generator's "
            "own params-schema copy)"
        )
    if version < 1:
        raise ValueError(
            f"{name}: sidecar '{PARAMS_SCHEMA_KEY}' is {version}; the params "
            "block's first version is 1"
        )
    out = []
    for entry in raw:
        if version == 1:
            if not isinstance(entry, str):
                raise ValueError(
                    f"{name}: params entry {entry!r} is not a bare name, but the "
                    f"sidecar declares '{PARAMS_SCHEMA_KEY}' 1 (a bare NAME "
                    "list). A decl-carrying {name, dtype} entry needs "
                    f"'{PARAMS_SCHEMA_KEY}': 2 — the producer stamped the wrong "
                    "version, and binding it as v1 would drop the declared type"
                )
            out.append(ParamSpec(entry, "float"))
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"{name}: params entry {entry!r} is not a {{name, dtype}} object, "
                f"but the sidecar declares '{PARAMS_SCHEMA_KEY}' {version} "
                "(decl-carrying entries)"
            )
        pname = entry.get("name")
        if not isinstance(pname, str) or not pname:
            raise ValueError(
                f"{name}: params entry {entry!r} has no 'name' (a non-empty string)"
            )
        dtype = entry.get("dtype", "float")
        if dtype not in PARAM_DTYPES:
            raise ValueError(
                f"{name}: param {pname!r} declares unknown dtype {dtype!r} "
                f"(supported: {_joined(PARAM_DTYPES)}); upgrade eagle"
            )
        out.append(ParamSpec(pname, dtype))
    return tuple(out)


def _joined(vocab) -> str:
    """A stable "a, b" rendering of a vocabulary, for an error naming the valid set."""
    return ", ".join(sorted(vocab))


def validate_sidecar(meta: dict, *, name: str) -> dict:
    """Validate a parsed sidecar against the schema-v1 contract; raise on
    any breach. Mirrors the C++ ``eagle::plugin::validate_sidecar`` check
    for check, so an artifact accepted by one language is accepted by the
    other (pinned by the shared conformance corpus). ``name`` names the
    artifact in every error. Returns ``meta`` unchanged, for chaining.

    Checks, in the C++ order: ``schema_version`` (forward-strict, absent =>
    v1); ``arg_spec`` roles; ``scalar_type`` (absent/empty stays lenient);
    ``pattern`` (value-strict against
    :data:`~eagle.roles.RECOGNIZED_PATTERNS`, absent stays lenient); the
    ``neural_block`` clause (pattern-conditional, called from
    :func:`raptor.schema.blocks.validate_neural_block_descriptor`, before
    derivative/buffer since it forbids both); ``derivative``'s shape; buffer
    ``kind`` (value-strict against :data:`~eagle.roles.BUFFER_KINDS`).

    Not checked here: the presence of ``kernel`` / ``arg_spec`` -- those are
    required keys a loader reads directly (``KeyError`` naming the key), as
    the C++ ``parse_sidecar`` enforces at parse rather than validate. A
    ``neural_block`` descriptor is the exception, since no Python loader
    reads it: its clause requires both keys itself.
    """
    from .dtypes import SCALAR_TYPES

    check_schema_version(meta, name=name)
    validate_roles(meta.get("arg_spec", ()), name=name)

    scalar_type = meta.get("scalar_type", "")
    if scalar_type and scalar_type not in SCALAR_TYPES:
        raise ValueError(
            f"{name}: unknown sidecar scalar_type {scalar_type!r} "
            f"(supported: {SCALAR_TYPES})"
        )

    # The set checked here is RECOGNIZED, not launch-certified: this is the
    # shared structural validator, so it admits every family it can
    # validate. A launching loader gates additionally on
    # LAUNCH_CERTIFIED_PATTERNS.
    pattern = meta.get("pattern")
    if pattern is not None and pattern not in RECOGNIZED_PATTERNS:
        raise ValueError(
            f"{name}: unknown sidecar pattern '{pattern}' "
            f"(supported: {_joined(RECOGNIZED_PATTERNS)}); upgrade eagle"
        )

    # Runs before the derivative/buffer checks below because it forbids both
    # on a descriptor.
    if pattern == _NEURAL_BLOCK:
        validate_neural_block_descriptor(meta, name=name)

    parse_derivative(meta, name=name)

    for buf in meta.get("buffers", ()):
        kind = buf.get("kind")
        if kind not in BUFFER_KINDS:
            raise ValueError(
                f"{name}: buffer {buf.get('name')!r} has unknown kind '{kind}' "
                f"(supported: {_joined(BUFFER_KINDS)}); upgrade eagle"
            )

    return meta
