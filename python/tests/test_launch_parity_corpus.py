# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""THE LAUNCH-PARITY CORPUS — a gate instrument.

This is the DEVICE-leg corpus: eagle's existing conformance corpus is
validation-tier (schema accept/reject, no launches), so nothing in the tree
could have shown that the new conforming FAST PATH produces the same NUMBERS as
the coercion triple it replaces. This file is that instrument.

**What it does.** Every case runs the SAME launch twice in the same process —
once with the fast path live, once with :func:`eagle.marshal._force_slow_coercion`
forcing the general coercion path — and asserts the results are BITWISE identical
(``tobytes()``, never a tolerance). It then asserts the case took the path it
was built to take, by reading the fast/slow counters.

**Why bitwise is the honest bar here.** For a conforming input the coercion
triple is provably a no-op chain — cupy's ``asarray`` and ``ascontiguousarray``
both return their argument when no copy is required — so the fast path returns
the SAME OBJECT the slow path would have. Anything less than bit-equality would
mean one of those three calls was not the no-op it is documented to be.

**The counter RED leg**: a deliberately STRIDED input, a
wrong-dtype input and a host (numpy) input must each take the SLOW path. If any
of them reached the fast path they would be bound with the wrong element stride
/ width — so the corpus asserts ``fast == 0`` for exactly those cases, and a
non-vacuity guard asserts the conforming cases really did take the fast path
(otherwise every leg would be comparing the slow path with itself).

Also here: the DEVICE leg of the opt-in zero mask, which needs a real
loader reading a real (synthetic, declaring) sidecar.

Device-marked throughout — run serialized on the device under test.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np
import pytest

from eagle import LoadedPure, LoadedVector, marshal
from eagle.sidecar import TERMINATED_READONLY_KEY

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
N = 2048
MU = 3.986004418e5

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _fresh_marshal_state():
    marshal._force_slow_coercion(False)
    marshal._reset_coercion_stats()
    marshal._reset_zero_mask_cache()
    yield
    marshal._force_slow_coercion(False)
    marshal._reset_coercion_stats()
    marshal._reset_zero_mask_cache()


def _positions(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(3, N))
    d /= np.linalg.norm(d, axis=0)
    return np.ascontiguousarray((d * rng.uniform(7e3, 4.2e4, N)).astype(np.float64))


# --------------------------------------------------------------------------- #
# The corpus: each entry builds ONE launch's inputs and says which path the
# inputs are supposed to take.
#
# The gravity fixture is the ideal probe: its only marshalled role is the
# ``position`` vector input (``mu`` is a by-value uniform, ``terminated`` is
# omitted, there are no tables or per-sample scalars), so ONE ``_as_dtype``
# call happens per launch and the counters are unambiguous.
# --------------------------------------------------------------------------- #
def _case_conforming_cupy():
    import cupy as cp

    P = _positions(seed=1)
    return {"position": cp.asarray(P)}, "fast"


def _case_strided_cupy():
    """A STRIDED view — the RED-leg input. Its element stride is not the
    packed one, so binding it without the contiguating copy would read the
    wrong doubles."""
    import cupy as cp

    P = _positions(seed=2)
    wide = cp.asarray(np.repeat(P, 2, axis=1))  # (3, 2N), C-contiguous
    view = wide[:, ::2]  # (3, N), NOT contiguous
    assert not view.flags.c_contiguous
    return {"position": view}, "slow"


def _case_float32_cupy():
    import cupy as cp

    P = _positions(seed=3)
    return {"position": cp.asarray(P.astype(np.float32))}, "slow"


def _case_numpy_host():
    return {"position": _positions(seed=4)}, "slow"


def _case_non_contiguous_transpose():
    """An F-ordered buffer reaching the launcher as ``(3, N)`` — contiguous in
    memory but not in the layout the GRef stride arithmetic assumes. Built
    entirely device-side so the strides are what this case claims regardless of
    how ``cp.asarray`` treats a host array's order."""
    import cupy as cp

    P = _positions(seed=5)
    f_order = cp.ascontiguousarray(cp.asarray(P).T).T  # (3, N), F-order strides
    assert f_order.shape == (3, N)
    assert not f_order.flags.c_contiguous
    return {"position": f_order}, "slow"


_CASES = {
    "conforming_cupy": _case_conforming_cupy,
    "strided_cupy": _case_strided_cupy,
    "float32_cupy": _case_float32_cupy,
    "numpy_host": _case_numpy_host,
    "non_contiguous_transpose": _case_non_contiguous_transpose,
}


def _to_host(x):
    """Whatever framework the launcher handed back -> a numpy array (the
    origin adapter returns numpy for a numpy input, cupy for a cupy one)."""
    if isinstance(x, np.ndarray):
        return x
    import cupy as cp

    return cp.asnumpy(x)


def _run_gravity(plugin, kw):
    return np.asarray(_to_host(plugin(mu=MU, **kw)))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_fast_and_forced_slow_paths_are_bitwise_identical(case):
    plugin = LoadedVector(FIX / "gravity.ptx")
    kw, _expected_path = _CASES[case]()

    marshal._force_slow_coercion(False)
    fast_result = _run_gravity(plugin, kw)
    marshal._force_slow_coercion(True)
    try:
        slow_result = _run_gravity(plugin, kw)
    finally:
        marshal._force_slow_coercion(False)

    assert fast_result.dtype == slow_result.dtype
    assert fast_result.shape == slow_result.shape
    assert fast_result.tobytes() == slow_result.tobytes(), (
        f"{case}: the fast path and the general coercion path "
        "produced DIFFERENT bytes for the same launch"
    )


@pytest.mark.parametrize("case", sorted(_CASES))
def test_each_case_takes_exactly_the_path_it_should(case):
    """fast-path-IFF-conforming. The strided / wrong-dtype / host cases are the
    The RED legs: each MUST take the slow path."""
    plugin = LoadedVector(FIX / "gravity.ptx")
    kw, expected_path = _CASES[case]()

    marshal._reset_coercion_stats()
    _run_gravity(plugin, kw)
    stats = marshal._coercion_stats()

    assert stats["fast"] + stats["slow"] == 1, (
        f"{case}: expected exactly one marshalled input for the gravity "
        f"fixture, saw {stats} — the corpus's counter probe is no longer "
        "unambiguous and this gate cannot be trusted"
    )
    if expected_path == "fast":
        assert stats == {"fast": 1, "slow": 0}, (
            f"{case}: a fully conforming cupy input did NOT take the fast path "
            f"({stats}) — every parity leg in this file would then be "
            "comparing the slow path with itself"
        )
    else:
        assert stats["fast"] == 0, (
            f"{case}: a NON-CONFORMING input reached the fast path ({stats}) — "
            "it would be bound without the coercion that makes it launchable"
        )


def test_the_forced_slow_arm_actually_bites():
    """NON-VACUITY GUARD for the parity legs: with the forced-slow switch on,
    even the fully conforming case must be counted slow. If this ever went
    green with ``fast == 1``, every parity assertion above would be vacuous."""
    plugin = LoadedVector(FIX / "gravity.ptx")
    kw, _ = _case_conforming_cupy()

    marshal._reset_coercion_stats()
    _run_gravity(plugin, kw)
    assert marshal._coercion_stats() == {"fast": 1, "slow": 0}

    marshal._force_slow_coercion(True)
    try:
        marshal._reset_coercion_stats()
        _run_gravity(plugin, kw)
        assert marshal._coercion_stats() == {"fast": 0, "slow": 1}
    finally:
        marshal._force_slow_coercion(False)


def test_the_fast_path_binds_the_callers_own_buffer_not_a_copy():
    """The property that makes bitwise parity structural: for a conforming
    input the coercion triple ALREADY returned the caller's own object, so
    the fast path changes the amount of work, never the buffer bound."""
    import cupy as cp

    conforming = cp.asarray(_positions(seed=6))
    marshal._force_slow_coercion(True)
    try:
        via_slow = marshal._as_dtype(conforming, np.float64)
    finally:
        marshal._force_slow_coercion(False)
    via_fast = marshal._as_dtype(conforming, np.float64)
    assert via_fast is conforming
    assert via_slow.data.ptr == conforming.data.ptr


# --------------------------------------------------------------------------- #
# The pure-kernel leg: the same parity claim through a DIFFERENT launch path
# (``pure_prepare``'s Mutable coercion rather than the vector-input one).
# --------------------------------------------------------------------------- #
def _bump_once(plugin, counter0, rate=2.5):
    import cupy as cp

    counter = cp.asarray(counter0)
    result = plugin(counter=counter, rate=rate)
    return np.asarray(_to_host(result["counter"]))


def test_pure_kernel_fast_and_forced_slow_are_bitwise_identical():
    plugin = LoadedPure(FIX / "bump.ptx")
    counter0 = np.linspace(-3.0, 11.0, 1024)

    fast = _bump_once(plugin, counter0)
    marshal._force_slow_coercion(True)
    try:
        slow = _bump_once(plugin, counter0)
    finally:
        marshal._force_slow_coercion(False)
    assert fast.tobytes() == slow.tobytes()
    assert np.allclose(fast, counter0 + 2.5)


def test_pure_kernel_conforming_mutable_takes_the_fast_path():
    plugin = LoadedPure(FIX / "bump.ptx")
    marshal._reset_coercion_stats()
    _bump_once(plugin, np.linspace(0.0, 1.0, 256))
    stats = marshal._coercion_stats()
    assert stats["fast"] >= 1 and stats["slow"] == 0, stats


# --------------------------------------------------------------------------- #
# The OPT-IN zero mask, through a REAL loader reading a synthetic
# DECLARING sidecar (the host units cover the predicate; this covers the
# threading through LoadedPure and the numerical parity of a shared mask).
# --------------------------------------------------------------------------- #
def _bump_fixture(tmp_path, *, declare):
    """The committed ``bump`` pure fixture, re-stamped with (or without) the
    ``terminated_readonly`` declaration. Stands in for what a generated
    sidecars will carry in a later window."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / "bump.ptx", tmp_path / "bump.ptx")
    meta = json.loads((FIX / "bump.json").read_text())
    if declare:
        meta[TERMINATED_READONLY_KEY] = True
    (tmp_path / "bump.json").write_text(json.dumps(meta, indent=2))
    return tmp_path / "bump.ptx"


def test_a_declaring_sidecar_reaches_the_shared_mask_and_an_undeclared_one_does_not(
    tmp_path,
):
    declaring = LoadedPure(_bump_fixture(tmp_path / "yes", declare=True))
    plain = LoadedPure(_bump_fixture(tmp_path / "no", declare=False))
    assert declaring.terminated_readonly is True
    assert plain.terminated_readonly is False

    counter0 = np.linspace(0.0, 4.0, 512)

    marshal._reset_zero_mask_cache()
    declared_result = None
    for _ in range(5):
        declared_result = _bump_once(declaring, counter0)
    assert marshal._zero_mask_stats() == {"hits": 4, "misses": 1, "bypassed": 0}, (
        "a DECLARING plugin did not reach the shared all-false mask"
    )

    marshal._reset_zero_mask_cache()
    plain_result = None
    for _ in range(5):
        plain_result = _bump_once(plain, counter0)
    assert marshal._zero_mask_stats() == {"hits": 0, "misses": 0, "bypassed": 0}, (
        "an UNDECLARED plugin reached the shared mask — the opt-in is not "
        "opt-in"
    )

    assert declared_result.tobytes() == plain_result.tobytes(), (
        "the shared mask changed the numbers"
    )
