# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A second embedded pair on the same problem: Dormand-Prince 5(4).

``rk78_card.py`` fixes one pair, Fehlberg's RKF7(8). A pair is only a
tableau, so this file writes the attempt kernel once for any tableau
(:func:`attempt_kernel`) and gives it Dormand-Prince 5(4) (Dormand and Prince,
1980). The controller, the clip onto the final time, the accept/reject rule
and the ``Terminated`` mask are the RKF7(8) attempt's, and the problem and the
analytic truth are the card's own (``rk78_card._inputs`` and
``rk78_card.kepler_truth``).

The attempt below is the plain, non-FSAL form: all 7 stages every attempt,
the 5th-order solution is kept (local extrapolation), the 4th-order one only
estimates the error. No card is run here.
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # benchmarks/: the cards' shared helpers
_spec = importlib.util.spec_from_file_location("eagle_rk78_card", _HERE / "rk78_card.py")
rc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("eagle_rk78_card", rc)
_spec.loader.exec_module(rc)

# >>> code:dp54_tableau
# Dormand-Prince 5(4): the lower-triangular a (stages 2..7; the last row is
# the 5th-order weights), the 5th-order weights b and the error weights
# be = b - b_hat (b_hat the 4th-order weights).
DP_A = (
    (1. / 5,),
    (3. / 40, 9. / 40),
    (44. / 45, -56. / 15, 32. / 9),
    (19372. / 6561, -25360. / 2187, 64448. / 6561, -212. / 729),
    (9017. / 3168, -355. / 33, 46732. / 5247, 49. / 176, -5103. / 18656),
    (35. / 384, 0, 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84),
)
DP_B = (35. / 384, 0, 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84, 0)
DP_BE = (71. / 57600, 0, -71. / 16695, 71. / 1920, -17253. / 339200, 22. / 525, -1. / 40)
DP_ORDER = 5
# <<< code:dp54_tableau


def attempt_kernel(a, b, be, order, rtol, atol):
    """The hawk attempt kernel for the tableau (``a``, ``b``, ``be``) of a pair
    whose kept solution has order ``order``, at tolerances ``rtol``, ``atol``."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated, UInt32, Vector
    from hawk import math as m

    exp_reject, exp_accept = -1.0 / order, -1.0 / (order + 1)

    def stages(s0, h):
        k = [rc.kepler_rhs(s0, m.sqrt)]
        for row in a:
            si = [s0[d] + h * rc.sum_terms([c * kj[d] for c, kj in zip(row, k) if c != 0])
                  for d in range(6)]
            k.append(rc.kepler_rhs(si, m.sqrt))
        new = [s0[d] + h * rc.sum_terms([w * kj[d] for w, kj in zip(b, k) if w != 0])
               for d in range(6)]
        err = [h * rc.sum_terms([w * kj[d] for w, kj in zip(be, k) if w != 0])
               for d in range(6)]
        return new, err

    def error_ratio(s0, new, err):
        # the card's mixed max-norm, at this kernel's tolerances
        ratios = [abs(err[d]) / (atol + rtol * m.max(abs(new[d]), abs(s0[d])))
                  for d in range(6)]
        return functools.reduce(m.max, ratios)

    @hawk.kernel
    def attempt(t_final: Param, terminated: Terminated, s: Mutable[Vector[6]],
                t: Mutable[Scalar], h: Mutable[Scalar],
                n_acc: Mutable[UInt32], n_rej: Mutable[UInt32]):
        s0 = s
        t0 = t
        last = h >= t_final - t0
        step = m.where(last, t_final - t0, h)
        new, err = stages(s0, step)
        ratio = error_ratio(s0, new, err)
        accept = ratio < 1.0
        expo = m.where(accept, exp_accept, exp_reject)
        h = step * m.min(rc.MAX_SCALE, m.max(rc.MIN_SCALE, rc.SAFETY * ratio ** expo))
        s = m.where(accept, m.vec(new), s0)
        t1 = m.where(accept, m.where(last, t_final, t0 + step), t0)
        t = t1
        n_acc = n_acc + m.uint32(m.where(accept, 1.0, 0.0))
        n_rej = n_rej + m.uint32(m.where(accept, 0.0, 1.0))
        terminated = t1 >= t_final

    return attempt


def dp54_kernel(rtol=rc.RTOL, atol=rc.ATOL):
    """The Dormand-Prince 5(4) attempt kernel."""
    return attempt_kernel(DP_A, DP_B, DP_BE, DP_ORDER, rtol, atol)


def rkf78_kernel(rtol=rc.RTOL, atol=rc.ATOL):
    """The card's RKF7(8) attempt kernel, at the card's tolerances."""
    return attempt_kernel(rc.RK_A, rc.RK_B, rc.RK_BE, 8, rtol, atol)


def solve(kernel, inp, device, t_final=rc.T_FINAL, max_steps=rc.MAX_ATTEMPTS):
    """Integrate the card's orbits ``inp`` to ``t_final`` with ``kernel`` on the
    GPU (``device``) or the host threads: ``(state (6, n), n_acc, n_rej)``."""
    import numpy as np

    import eagle

    n = len(inp["x"])
    xp = np
    if device:
        import cupy as xp
    planes = {"s": xp.asarray(np.stack([inp[k] for k in rc.STATE_NAMES])),
              "t": xp.zeros(n), "h": xp.full(n, rc.H0),
              "n_acc": xp.zeros(n, dtype="uint32"), "n_rej": xp.zeros(n, dtype="uint32")}
    terminated = xp.zeros(n, dtype=bool)
    if device:
        eagle.until_done(eagle.deploy(kernel), max_steps=max_steps, t_final=t_final,
                         terminated=terminated, **planes).run()
    else:
        plan = eagle.deploy(kernel, targets=("host",)).host
        eagle.run_until_done(plan, max_steps=max_steps, t_final=t_final,
                             terminated=terminated, **planes)
    get = (lambda v: v.get()) if device else (lambda v: v)
    return (get(planes["s"]), get(planes["n_acc"]).astype(int),
            get(planes["n_rej"]).astype(int))
