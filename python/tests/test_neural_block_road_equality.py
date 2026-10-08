# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Road equality between the raptor entry
point and ``eagle.sidecar.validate_sidecar``.

Probe C ran a raptor-only build of the moved
``neural_block`` clause against all 15 ``pattern == "neural_block"``
conformance fixtures and found 15/15 verdict-identical, 14/15
message-byte-identical — the one delta being the schema-version gate wording
This unifies. This file is that probe, productized as eagle's suite (eagle
knows raptor, downstream→upstream) against the
now-landed, unified code: both drivers below must be BYTE-identical on every
row, verdict and message.

Fixtures are discovered by SCANNING ``eagle/tests/conformance/`` for
``pattern == "neural_block"`` sidecars, never a hand-passed list (the "a
scanner fed its own expected answers finds nothing wrong" lesson) — a
separate test pins the discovered count at 15 so a fixture silently
added/removed is itself a signal, not silence.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from raptor.schema.blocks import validate_neural_block_descriptor

from eagle.sidecar import validate_sidecar

# Both rows read the repository's neural-block fixtures, which a wheel does not carry.
pytestmark = pytest.mark.repo_local

CORPUS = pathlib.Path(__file__).resolve().parent.parent.parent / "tests" / "conformance"


def _neural_block_sidecar_fixtures() -> list[pathlib.Path]:
    """Every ``*.sidecar.json`` fixture whose ``pattern`` is ``neural_block``,
    discovered by scanning the corpus directory."""
    hits = []
    for path in sorted(CORPUS.glob("*.sidecar.json")):
        meta = json.loads(path.read_text())
        if meta.get("pattern") == "neural_block":
            hits.append(path)
    return hits


_FIXTURES = _neural_block_sidecar_fixtures()
_PARAMS = [pytest.param(p, id=p.stem) for p in _FIXTURES] or [
    pytest.param(None, id="EMPTY-CORPUS")
]


def test_fifteen_neural_block_fixtures_discovered():
    """Pins the discovered population (the "instrument needs a known-answer
    population" lesson) — a corpus that silently lost or gained a
    ``neural_block`` fixture changes this count."""
    assert CORPUS.is_dir(), f"conformance corpus directory is missing: {CORPUS}"
    names = [p.name for p in _FIXTURES]
    assert len(_FIXTURES) == 15, (
        f"expected exactly 15 neural_block sidecar fixtures in {CORPUS}; "
        f"found {len(_FIXTURES)}: {names}"
    )


def _verdict_and_message(fn, meta: dict, name: str) -> tuple[str, str | None]:
    """Run ``fn(meta, name=name)``; return ``("accept", None)`` if it does not
    raise, ``("reject", <message>)`` if it raises ``ValueError``. Any other
    exception is let through uncaught -- an unexpected exception type is a
    genuine failure, not a verdict to normalize away."""
    try:
        fn(meta, name=name)
    except ValueError as exc:
        return "reject", str(exc)
    return "accept", None


@pytest.mark.parametrize("fixture_path", _PARAMS)
def test_raptor_entry_point_and_eagle_sidecar_agree(fixture_path):
    assert fixture_path is not None, (
        "the neural_block fixture scan collected ZERO fixtures — this "
        "parametrization ran on a sentinel. See "
        "test_fifteen_neural_block_fixtures_discovered."
    )
    meta = json.loads(fixture_path.read_text())
    name = fixture_path.stem

    # Each callee gets its OWN dict copy: validators are documented to return
    # their input unchanged, never to mutate it, but a shared mutable dict
    # passed to two independent call sites would make a hidden mutation on
    # one road silently corrupt the other's input -- the comparison must not
    # rely on that never happening.
    raptor_verdict, raptor_message = _verdict_and_message(
        validate_neural_block_descriptor, dict(meta), name
    )
    eagle_verdict, eagle_message = _verdict_and_message(
        validate_sidecar, dict(meta), name
    )

    assert raptor_verdict == eagle_verdict, (
        f"{name}: verdict mismatch — raptor={raptor_verdict!r} vs "
        f"eagle={eagle_verdict!r}"
    )
    assert raptor_message == eagle_message, (
        f"{name}: message mismatch — raptor={raptor_message!r} vs "
        f"eagle={eagle_message!r}"
    )
