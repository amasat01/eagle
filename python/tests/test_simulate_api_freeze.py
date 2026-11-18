# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A public-API check on ``eagle.simulate``/``eagle.deploy``, kept in a
file of its own rather than folded into ``test_simulate.py``.

**The positional-binding refusal.** A model built from a PREBUILT plan (no
raw hawk kernel to read a call signature from) refuses positional kernel
arguments outright, naming the keyword call to use instead -- binding by
keyword always works, and a raw-kernel model still binds positionally
(unchanged, covered by ``test_simulate.py``'s own call-shape rows).
"""

from __future__ import annotations

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")


@hawk.kernel
def _one_shot_finisher(x: hawk.Mutable[hawk.Scalar], terminated: hawk.Terminated):
    """The only bindable plane is ``x``; every sample finishes at once (the
    condition is trivially true, spelled rather than a bare ``= True``), so
    one call to :func:`eagle.simulate` is enough to exercise the bind."""
    x = x + 1.0
    terminated = x > -1e300


def test_positional_refused_for_prebuilt_plans(tmp_path):
    """A model built from ``eagle.deploy(...)`` (a prebuilt plan, no raw hawk
    kernel for the door to read a call signature from) refuses a positional
    kernel argument, naming the keyword fix; the identical call by keyword
    runs."""
    import eagle

    plan = eagle.deploy(_one_shot_finisher, cache_dir=str(tmp_path), targets=("host",))
    x = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="x="):
        eagle.simulate(plan, x, max_steps=1)

    result = eagle.simulate(plan, x=x.copy(), max_steps=1)
    np.testing.assert_array_equal(result.x, x + 1.0)
    assert result.done


def test_positional_still_binds_for_a_raw_kernel(tmp_path):
    """The companion, non-vacuity half: a model that IS a raw hawk kernel
    (no prebuilt plan involved) still binds positionally -- the refusal is
    specific to the prebuilt-plan hazard, not a blanket ban."""
    import eagle

    x0 = np.array([0.0, 1.0, 2.0])
    result = eagle.simulate(_one_shot_finisher, x0.copy(), max_steps=1)
    np.testing.assert_array_equal(result.x, x0 + 1.0)
    assert result.done
