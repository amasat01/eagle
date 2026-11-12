# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The test-side gates over raptor's conformance checks: each fails (never skips) on the problem text the check returns."""

import pytest
from raptor.conformance import freethreading as ft
from raptor.conformance.interop import certified_framework_problem


def certified_framework(fw: str) -> None:
    """Fail when the certified framework ``fw`` is absent from this environment."""
    problem = certified_framework_problem(fw)
    if problem is not None:
        pytest.fail(f"{fw}: {problem}")


def require_free_threaded() -> None:
    """Fail on a build that cannot certify free-threading."""
    problem = ft.free_threaded_problem()
    if problem is not None:
        pytest.fail(problem)


def require_not_vacuous(result=None, min_lost: float = ft.MIN_LOST_FRACTION):
    """The canary result, failing when the run lost too few updates to mean anything."""
    result, problem = ft.not_vacuous_problem(result, min_lost)
    if problem is not None:
        pytest.fail(problem)
    return result
