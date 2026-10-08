# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Plan.run``'s locked return shape: one written plane returns that
array; several return a ``dict`` keyed by
plane NAME -- never a positional tuple. ``arg_spec``'s own order is
ROLE-grouped then NAME-sorted (``hawk/ir/walk.py``), not the kernel's
authored parameter order, so a tuple return silently swaps two outputs
declared the other way around (``def k(v: Mutable, r: Mutable)`` would have
returned ``(r, v)``); a dict keyed by name cannot be misread that way.

Reuses ``test_plan_packer_e2``'s already-compiled multi-output fixture
(``e2_multi_host``: declares ``y`` (``out``) and ``w`` (``wide_out``), both
output roles) instead of re-deriving a second C++ TU -- this file's new
ground is the RETURN SHAPE, not the packing those rows already certify.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_plan_packer_e2 import (  # noqa: E402
    _MULTI_SPEC,
    _VEC3_SPEC,
    _host_plugin,
    _x,
    e2_host_lib,  # noqa: F401 -- reused as a pytest fixture, by parameter name
)

# The reused fixture compiles against the repository's plugin/gref_abi.h, as
# every test_plan_packer_e2 row that uses it does: repo_local like those rows.
pytestmark = pytest.mark.repo_local


def test_two_outputs_come_back_as_a_dict_keyed_by_name(e2_host_lib):
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 16
    x = _x(n)
    plugin = _host_plugin(e2_host_lib, "e2_multi_host", _MULTI_SPEC,
                          arg_widths={"y": 3})

    result = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x)

    assert isinstance(result, dict), f"expected a dict, got {type(result).__name__}"
    assert set(result) == {"y", "w"}
    np.testing.assert_array_equal(result["y"],
                                  x + np.arange(3, dtype=np.float64)[:, None])
    np.testing.assert_array_equal(result["w"], 10.0 * x[0])


def test_one_output_still_returns_the_bare_array_never_a_1_item_dict(e2_host_lib):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _host_plugin(e2_host_lib, "e2_vec3_host", _VEC3_SPEC,
                          arg_widths={"y": 3})
    result = eplan.plan(plugin, structure=eexec.HostTeam).run(x=_x(8), a=1.0)

    assert isinstance(result, np.ndarray), (
        f"a single-output plugin must return the bare plane, got "
        f"{type(result).__name__}"
    )


def test_a_supplied_output_is_returned_as_the_callers_own_object(e2_host_lib):
    """Among several outputs, a caller-supplied plane is written in place and
    handed back AS the caller's own array (``is``, not a copy); an output the
    caller left out is freshly allocated and returned under its own key."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 16
    x = _x(n)
    plugin = _host_plugin(e2_host_lib, "e2_multi_host", _MULTI_SPEC,
                          arg_widths={"y": 3})
    y = np.zeros((3, n))

    result = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, y=y)

    assert result["y"] is y, "the supplied output must come back as the SAME object"
    np.testing.assert_array_equal(y, x + np.arange(3, dtype=np.float64)[:, None])
    assert "w" in result and result["w"] is not y
    np.testing.assert_array_equal(result["w"], 10.0 * x[0])
