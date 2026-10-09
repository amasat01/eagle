# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""zip_strict: equal lengths zip; any mismatch raises ValueError."""

import pytest

from eagle._compat import zip_strict


def test_equal_lengths():
    assert list(zip_strict([1, 2], "ab", (3.0, 4.0))) == [(1, "a", 3.0), (2, "b", 4.0)]
    assert list(zip_strict([], [])) == []


@pytest.mark.parametrize(
    "args", [([1, 2], [1]), ([1], [1, 2]), ([1, 2], [1, 2], [1]), ([1], [1, 2], [1])]
)
def test_mismatch_raises(args):
    with pytest.raises(ValueError):
        list(zip_strict(*args))
