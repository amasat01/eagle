# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A plane named ``p`` (or ``self``) shadows
``eagle.plan``'s OWN parameter of the same spelling.

**The finding.** ``Plan.run(**kw)`` forwards every bound plane straight to
``_run_plan(p: Plan, **kw)`` (``eagle/plan.py``) as ``_run_plan(self, **kw)``.
Nothing in ``eagle.roles``'s canonical vocabulary reserves the spelling ``p``
(or ``self``) — a HAWK-authored kernel is free to name a parameter
``p`` (a momentum plane, a pressure field, ...) exactly as freely as ``x`` or
``y`` — so a plugin declaring one and run through ``plan_obj.run(p=array)``
reaches ``_run_plan`` as a POSITIONAL ``self`` (bound by the method call) and
a KEYWORD ``p=array`` both landing on ``_run_plan``'s first parameter, which
happens to be spelled ``p`` too. Python refuses the call before a single line
of ``_run_plan`` runs, naming ITS OWN parameter rather than the caller's
plane. ``Plan.run``'s own ``self`` has exactly the same exposure one level up:
a plane named ``self`` collides with the bound-method's own ``self`` the
moment ``**kw`` is re-expanded as keywords, which is what ``Plan.run(**kw)``
does. ``Plan.bind``/``BoundPlan.rebind`` carry the ``self`` half of the same
exposure (their PRIVATE delegates already pass the caller's planes as one
plain ``dict`` argument, never re-expanded — see below), but not the ``p``
half, since neither delegate's parameter is spelled ``p``.

**RED, observed against the UNFIXED ``eagle/plan.py``** (``eagle`` at
an earlier version): building a plugin whose ``vec_in`` plane
is named ``p`` and running it::

    >>> eplan.plan(plugin, structure=eexec.HostTeam).run(p=x, a=2.0)
    TypeError: _run_plan() got multiple values for argument 'p'

and the same plugin with its plane renamed ``self``::

    >>> eplan.plan(plugin, structure=eexec.HostTeam).run(self=x, a=2.0)
    TypeError: Plan.run() got multiple values for argument 'self'

and ``Plan.bind`` for the ``self`` case (the ``p`` case does NOT reproduce
here — ``_bind_plan``'s ``planes`` argument is already a plain ``dict``, so a
plane literally named ``p`` is merely a dict key, never a positional
collision)::

    >>> eplan.plan(plugin_self, structure=eexec.HostTeam).bind(self=x, a=2.0)
    TypeError: Plan.bind() got multiple values for argument 'self'

A plane named ``kw`` was NEVER broken anywhere in this file (kept below as
the POSITIVE control that proves the ``/`` fix touches nothing else): ``**kw``
is a catch-all, and a catch-all's own spelling is never itself a reserved
keyword at the call site — ``f(**kw)`` called as ``f(kw=1)`` has always bound
``kw = {"kw": 1}`` with no collision, in every Python version this codebase
targets.

**The fix.** ``eagle/plan.py``'s ``_run_plan`` gained a positional-only marker
on its first parameter (``def _run_plan(p, /, **kw)``), and ``Plan.run``,
``Plan.bind`` and ``BoundPlan.rebind`` each gained the same marker on their
OWN ``self`` (``def run(self, /, **kw)`` and so on) — auditing every other
``**kw``-taking helper in plan.py: the two non-method delegates that take
the caller's planes as a plain ``dict`` (``_run_device_plan``/``_run_host_plan``,
and ``_bind_plan`` itself) were already confirmed safe by inspection and are
not touched.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

# The fixture compiles against the repository's plugin/ headers, which a wheel does
# not carry.
pytestmark = pytest.mark.repo_local

EAGLE_ROOT = pathlib.Path(__file__).resolve().parents[2]  # holds plugin/gref_abi.h

#: One host entry, ``y = a*x + c`` component-wise (``c`` the component index) —
#: deliberately the same shape ``test_plan_packer_e2.py``'s ``e2_vec3_host``
#: already exercises (a GRef-shaped ``out``/``vec_in`` pair plus a ``uniform``),
#: so this row's only NEW variable is the plane's NAME, never its packing.
_HOST_SRC = r"""
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

static constexpr std::uint64_t kWidth = 3;

extern "C" void shadow_vec3_host(void* const* params, std::int64_t base,
                                 std::int64_t count, std::int64_t /*nSamples*/)
{
    const GRefMirror* yv = static_cast<const GRefMirror*>(params[0]);
    const GRefMirror* xv = static_cast<const GRefMirror*>(params[1]);
    double* y            = reinterpret_cast<double*>(yv->data_);
    const double* x      = reinterpret_cast<const double*>(xv->data_);
    const double a       = *static_cast<const double*>(params[2]);
    for (std::int64_t i = base; i < base + count; ++i)
        for (std::uint64_t c = 0; c < kWidth; ++c)
            y[c * yv->compStride_ + i * yv->sampleStride_] =
                a * x[c * xv->compStride_ + i * xv->sampleStride_] + double(c);
}
"""


class _ShadowPlugin:
    """The minimal duck-typed v2 plugin ``eagle.plan`` drives — the SAME shape
    ``test_plan_packer_e2.py``'s ``_E2Plugin`` is, kept local so this file does
    not reach across test modules for fixtures."""

    exec_access = "sample_local"
    exec_op = None

    def __init__(self, arg_spec, *, host_entry, arg_widths=None):
        self.arg_spec = tuple(arg_spec)
        self.scalar_type = "float64"
        self.arg_widths = dict(arg_widths or {})
        self.host_entry = host_entry


def _gxx() -> str:
    found = shutil.which(os.environ.get("CXX", "")) or shutil.which("g++")
    if found is None and os.path.exists("/usr/bin/g++"):
        found = "/usr/bin/g++"
    if found is None:
        raise RuntimeError("this row needs g++ to build the host fixture")
    return found


@pytest.fixture(scope="module")
def shadow_host_lib(tmp_path_factory):
    """The one compiled host entry every row in this file shares — the plane
    NAME lives entirely on the Python side (``arg_spec``), so one binary
    serves every shadowed spelling below without recompiling."""
    tmp = tmp_path_factory.mktemp("shadow_kwarg_host")
    src = tmp / "shadow_host.cpp"
    src.write_text(_HOST_SRC)
    so = tmp / "shadow_host.so"
    proc = subprocess.run(
        [_gxx(), "-O2", "-std=c++17", "-shared", "-fPIC", f"-I{EAGLE_ROOT}",
         "-o", str(so), str(src)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"shadow host fixture compile failed:\n{proc.stderr}"
    return ctypes.CDLL(str(so))


def _plugin_named(lib, plane_name: str) -> _ShadowPlugin:
    """``_ShadowPlugin`` with its ``vec_in`` plane spelled ``plane_name`` — the
    ONE thing this whole file varies. Packing order is POSITIONAL (the C++
    body reads ``params[0]``/``params[1]``/``params[2]`` by index, never by
    name), so renaming the Python-side ``arg_spec`` entry is enough; nothing
    recompiles."""
    fn = getattr(lib, "shadow_vec3_host")
    arg_spec = (("out", "y"), ("vec_in", plane_name), ("uniform", "a"),
                ("nsamples", "n"))
    return _ShadowPlugin(arg_spec, host_entry=ctypes.cast(fn, ctypes.c_void_p).value,
                         arg_widths={"y": 3})


def _x(n: int) -> np.ndarray:
    return np.arange(3 * n, dtype=np.float64).reshape(3, n) * 0.5


def _expected(x: np.ndarray, a: float) -> np.ndarray:
    return a * x + np.arange(3, dtype=np.float64)[:, None]


#: The three spellings tested: the two collisions this closes
#: (``p`` against ``_run_plan``'s own parameter, ``self`` against every
#: method's own bound-``self``) and ``kw`` — the positive control that was
#: NEVER broken (see the module docstring) and must stay that way after the
#: fix.
_NAMES = ("p", "kw", "self")


@pytest.mark.parametrize("plane_name", _NAMES)
def test_a_plane_named_like_a_plan_parameter_runs_through_plan_run(
    shadow_host_lib, plane_name
):
    """RED today on ``p``/``self`` (see the module docstring for the two
    captured ``TypeError``s); ``kw`` passes on both sides of the fix and is
    kept here so a future regression in the OTHER direction — the ``/``
    somehow swallowing a legitimately-named plane — would show up as a new
    failure on this exact row rather than as a silent gap in coverage."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, 2.0
    x = _x(n)
    plugin = _plugin_named(shadow_host_lib, plane_name)

    y = eplan.plan(plugin, structure=eexec.HostTeam).run(**{plane_name: x, "a": a})

    np.testing.assert_array_equal(y, _expected(x, a))


@pytest.mark.parametrize("plane_name", _NAMES)
def test_a_plane_named_like_a_plan_parameter_binds_and_launches(
    shadow_host_lib, plane_name
):
    """The capture-legal door: RED today on ``self`` only (``_bind_plan``
    already takes the caller's planes as one plain ``dict``, so ``p`` was
    never broken at THIS door — see the module docstring). ``.bind()`` does
    not allocate outputs (unlike ``.run()``), so ``y`` is supplied here too,
    pre-zeroed, and ``.launch()`` must have written the real answer into it —
    a launch that silently no-opped would leave ``y`` all zero and this
    assertion would still catch it."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, 2.0
    x = _x(n)
    y = np.zeros((3, n), dtype=np.float64)
    plugin = _plugin_named(shadow_host_lib, plane_name)

    bound = eplan.plan(plugin, structure=eexec.HostTeam).bind(
        **{"y": y, plane_name: x, "a": a}
    )
    result = bound.launch()

    assert result is None
    np.testing.assert_array_equal(y, _expected(x, a))
