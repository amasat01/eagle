# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The recorder a kernel call hands itself to instead of launching."""

from __future__ import annotations

import contextlib
import contextvars

__all__ = ["current", "recording"]

_RECORDER = contextvars.ContextVar("eagle_recorder", default=None)

#: The active recorder of this thread/context, or ``None``. A recorder has
#: ``record_call(plan, args, kw)``; :meth:`AutoPlan.__call__` hands it the call.
current = _RECORDER.get


@contextlib.contextmanager
def recording(recorder):
    """Make ``recorder`` the active one for the body of the ``with``."""
    token = _RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _RECORDER.reset(token)
