# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Small compatibility helpers for the Python versions eagle supports (3.9+)."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 10):

    def zip_strict(*iterables):
        """Return ``zip(*iterables, strict=True)`` (unequal lengths raise ValueError)."""
        return zip(*iterables, strict=True)

else:

    def zip_strict(*iterables):
        """Return a zip that raises ``ValueError`` on unequal lengths (3.9 stand-in)."""
        iters = [iter(it) for it in iterables]
        if not iters:
            return
        while True:
            items = []
            for pos, it in enumerate(iters):
                try:
                    items.append(next(it))
                except StopIteration:
                    if pos:
                        raise ValueError(
                            f"zip_strict() argument {pos + 1} is shorter than "
                            f"argument{'s 1-' if pos > 1 else ' '}{pos}"
                        ) from None
                    # First iterator is exhausted: every other one must be too.
                    for later, other in enumerate(iters[1:], start=2):
                        try:
                            next(other)
                        except StopIteration:
                            continue
                        raise ValueError(
                            f"zip_strict() argument {later} is longer than "
                            f"argument{'s 1-' if later > 2 else ' '}{later - 1}"
                        ) from None
                    return
            yield tuple(items)
