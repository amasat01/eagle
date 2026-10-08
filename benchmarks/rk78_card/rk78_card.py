# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The eagle adaptive-step card: Kepler orbits under RKF7(8), several ways, one JSON.

The workload is a batch of independent two-body (Kepler) orbits, float64,
mu = 1, each started at periapsis with semi-major axis a = 1, eccentricity e
drawn uniformly in [0, 0.9] and a random orientation, and integrated to the
same final time with the adaptive Runge-Kutta-Fehlberg 7(8) pair. Every sample
picks its own step sizes and makes its own accept/reject decisions, so the
samples need very different numbers of attempts (a circular orbit few, an
e = 0.9 orbit many more, spent near periapsis) and the batch thins out as it
runs.

The scheme is Fehlberg's RKF7(8) pair (NASA Technical Report R-287, 1968)
with a mixed absolute/relative max-norm error control and a standard
step-size controller (a safety factor, a growth clamp and the usual
exponents),
written once as a hawk kernel (``rkf78_attempt``): ONE step attempt per sample
per launch -- 13 stages, the embedded error estimate, the controller, accept
or reject, the time and step-size update. A sample whose time reached the
final time finishes itself: the kernel assigns its ``Terminated`` mask, and
hawk counts each newly finished sample into the counter that ends eagle's
loop (``eagle.until_done``), exactly as the RK4 card's eagle arms stop. One choice, stated: a common alternative rejects an
attempt that would step past the end time and retries it shortened; here the
step is clipped to land on the final time before it is attempted (the brief's
rule; it saves that one wasted attempt per sample, identically in every arm).
No minimum-step termination is applied (no sample would come near a usual
1e-6 floor on this problem); every arm bounds the attempts by
``MAX_ATTEMPTS`` instead.

The arms (each written as a user of that tool would write it):

``eagle_graph``
    The hawk kernel deployed (``eagle.deploy``) and run in one call
    (``eagle.until_done``): the attempt kernel captured once as the body of a
    device-side loop (one CUDA graph with a WHILE node) that ends when every
    sample has finished itself. The plain ``@hawk.kernel`` decorator builds
    the automatic kernel: each launch runs the attempts eagle's policy picks.
``eagle_graph_compact``
    The same call on the single attempt traced under a kind whose guard
    reads the active-set map (hawk ``Guard(active_set=True)``): active-set
    compaction every ``COMPACT_EVERY`` attempts.
``eagle_graph_reorder``
    Compaction plus the occasional physical reorder
    (``eagle.ActiveSet(reorder=REORDER_THETA)``), restored at the end.
``eagle_simulate``
    The kernel of ``eagle_graph`` run by ``eagle.simulate``'s bound form
    (``eagle.simulation``): the state and the parameters by name; eagle
    deploys it and builds the same loop.
``cupy_masked`` / ``torch_masked``
    The whole batch advanced one attempt at a time with array expressions;
    the accept/reject decision, the step-size update and finished samples are
    masks (``where``); the host checks "any sample still running" after every
    attempt.
``torch_graphed``
    ``torch_masked``'s attempt over static buffers, ``GRAPH_BLOCK`` attempts
    captured once as a CUDA graph (``torch.cuda.CUDAGraph``) and replayed,
    the host check once per block. (``torch.compile``'s GPU backend needs
    compute capability 7.0 or newer; it is listed, not run, on older GPUs.)
``jax_vmap``
    ``jit(vmap(lax.while_loop))``: each sample's whole adaptive loop as a
    while loop, batched; float64.
``warp_kernel``
    A Warp kernel, one thread per sample running its whole adaptive loop.
``eagle_cpu``
    The same hawk kernel, the host side of the same ``eagle.deploy`` (NumPy
    planes select it), run through eagle's OpenMP host team
    (``eagle.run_until_done``).

Truth is the analytic Kepler solution (Kepler's equation solved by Newton's
method to machine precision). Every arm is checked against it and against
the others before any timing counts.

Repetitions are interleaved, the first run of each arm is a warm-up and is
excluded, every time is a median with its interquartile range. A MEMORY pass
and a COMPILE-TIME pass follow, each in fresh processes; an Nsight Systems
pass gives kernel-only time. The run writes ``card_<device-slug>.json`` and
``card_<device-slug>.md`` (rendered from the JSON by :func:`render_markdown`).

Reproduce (from the eagle repository root)::

    python benchmarks/rk78_card/rk78_card.py

``--quick`` runs N up to 1e4; ``--no-nsys``, ``--no-memory`` and
``--no-compile`` skip those passes (dry runs only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))  # benchmarks/: the cards' shared helpers
import _card_common as cc  # noqa: E402

SCRIPT_REL = "benchmarks/rk78_card/rk78_card.py"

# --------------------------------------------------------------------------- #
# The problem
# --------------------------------------------------------------------------- #
MU = 1.0
SEMI_MAJOR = 1.0
E_RANGE = (0.0, 0.9)
#: The final time: about 10 orbital periods of a = 1 (period 2 pi), chosen
#: off a whole number of periods so the truth needs Kepler's equation.
T_FINAL = 64.0
RTOL = 1e-10
ATOL = 1e-10
#: The initial step; the controller's
#: first rejections bring it down to the orbit's scale.
H0 = 1.0
#: The controller's constants (safety factor, minimum and maximum step scale)
#: and its exponents: -1/p on a rejection, -1/(p+1) on an
#: acceptance, p = 8 (RKF78's order).
SAFETY = 0.9
MIN_SCALE = 0.2
MAX_SCALE = 5.0
EXP_REJECT = -1.0 / 8.0
EXP_ACCEPT = -1.0 / 9.0
#: Upper bound on attempts per sample (every arm's loop bound); the e = 0.9
#: orbits need under a thousand at this tolerance.
MAX_ATTEMPTS = 4000
SEED = 20261002

ARMS = ("eagle_graph", "eagle_graph_compact", "eagle_graph_reorder",
        "eagle_graph_persist", "eagle_simulate",
        "cupy_masked", "torch_masked", "torch_graphed", "jax_vmap", "warp_kernel",
        "eagle_cpu")
GPU_ARMS = ARMS[:-1]
EAGLE_GPU_ARMS = ARMS[:5]
ARM_LABELS = {
    "eagle_graph": "eagle graph",
    "eagle_graph_compact": "eagle graph + compaction",
    "eagle_graph_reorder": "eagle graph + compaction + reorder",
    "eagle_graph_persist": "hawk + eagle persistent",
    "eagle_simulate": "eagle.simulate (eagle picks the launch mode)",
    "cupy_masked": "CuPy masked",
    "torch_masked": "PyTorch masked",
    "torch_graphed": "PyTorch + CUDA graph",
    "jax_vmap": "JAX jit(vmap(while_loop))",
    "warp_kernel": "Warp per-thread kernel",
    "eagle_cpu": "eagle CPU (OpenMP)",
}
ARM_LOOP = {
    "eagle_graph": "one CUDA graph: device WHILE loop of the attempt kernel, which "
                   "marks and counts its finished samples and runs the attempts per "
                   "launch eagle's policy picks; finished samples skipped by the "
                   "Terminated guard",
    "eagle_graph_compact": "as eagle graph, the attempt launched over the live samples "
                           "only (index map rebuilt every 16 attempts)",
    "eagle_graph_reorder": "as + compaction, live samples moved to the front of every "
                           "plane when spread thin; restored at the end",
    "eagle_graph_persist": "ONE launch, no graph, no map, no policy kernel: a grid "
                           "sized from the SM count and the kernel's occupancy, each "
                           "lane fetching base + atomicAdd(counter, 1) whenever its "
                           "sample is done",
    "eagle_simulate": "eagle.simulate takes the kernel, its state and its parameters "
                      "by name; eagle's automatic policy picks the launch mode from "
                      "the batch (one launch for a small batch, the persistent launch "
                      "above it); the row records the mode taken",
    "cupy_masked": "host loop: one masked whole-batch attempt (array expressions), "
                   "host check per attempt",
    "torch_masked": "host loop: one masked whole-batch attempt (tensor expressions), "
                    "host check per attempt",
    "torch_graphed": "host loop: a CUDA graph of 16 masked attempts replayed, host "
                     "check per block",
    "jax_vmap": "one compiled executable: per-sample while loop, batched (runs until "
                "the slowest sample finishes)",
    "warp_kernel": "one launch: each thread loops over its own sample's attempts to "
                   "the final time",
    "eagle_cpu": "host loop: one attempt over the batch (OpenMP), the kernel marks "
                 "and counts its finished samples",
}
NS = (1_000, 10_000, 100_000, 1_000_000)
REPS = 5
#: An arm whose warm-up run says ``reps`` repetitions would take longer than
#: ``SLOW_ARM_BUDGET_S`` runs ``SLOW_ARM_REPS`` instead (the masked baselines
#: at a million samples take minutes each; their spread is already tight).
#: Every row records the repetitions it ran.
SLOW_ARM_BUDGET_S = 60.0
SLOW_ARM_REPS = 3
COMPACT_EVERY = 16
REORDER_THETA = 0.5
GRAPH_BLOCK = 16
CPU_THREADS = 8
WARM_N = 64

# --------------------------------------------------------------------------- #
# The scheme: Fehlberg's RKF7(8) tableau (NASA TR R-287, 1968): c (stages 2..13),
# the lower-triangular a, the 8th-order weights b and the error weights be
# (b_hat - b; only |error| is used).
# --------------------------------------------------------------------------- #
RK_C = (2. / 27, 1. / 9, 1. / 6, 5. / 12, 1. / 2, 5. / 6, 1. / 6, 2. / 3,
        1. / 3, 1.0, 0, 1.0)
RK_A = (
    (2. / 27,),
    (1. / 36, 1. / 12),
    (1. / 24, 0, 1. / 8),
    (5. / 12, 0, -25. / 16, 25. / 16),
    (1. / 20, 0, 0, 1. / 4, 1. / 5),
    (-25. / 108, 0, 0, 125. / 108, -65. / 27, 125. / 54),
    (31. / 300, 0, 0, 0, 61. / 225, -2. / 9, 13. / 900),
    (2.0, 0, 0, -53. / 6, 704. / 45, -107. / 9, 67. / 90, 3.0),
    (-91. / 108, 0, 0, 23. / 108, -976. / 135, 311. / 54, -19. / 60, 17. / 6, -1. / 12),
    (2383. / 4100, 0, 0, -341. / 164, 4496. / 1025, -301. / 82, 2133. / 4100,
     45. / 82, 45. / 164, 18. / 41),
    (3. / 205, 0, 0, 0, 0, -6. / 41, -3. / 205, -3. / 41, 3. / 41, 6. / 41, 0),
    (-1777. / 4100, 0, 0, -341. / 164, 4496. / 1025, -289. / 82, 2193. / 4100,
     51. / 82, 33. / 164, 12. / 41, 0, 1.0),
)
RK_B = (0, 0, 0, 0, 0, 34. / 105, 9. / 35, 9. / 35, 9. / 280, 9. / 280, 0,
        41. / 840, 41. / 840)
RK_BE = (41. / 840, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41. / 840, -41. / 840, -41. / 840)
N_STAGES = 13


# >>> code:shared_scheme
def kepler_rhs(s, sqrt):
    """d/dt of the state s = [x, y, z, vx, vy, vz] under mu = 1 gravity."""
    x, y, z = s[0], s[1], s[2]
    r2 = x * x + y * y + z * z
    ir3 = 1.0 / (r2 * sqrt(r2))
    return [s[3], s[4], s[5], -x * ir3, -y * ir3, -z * ir3]


def rkf78_stages(s0, h, sqrt):
    """One RKF7(8) attempt of step h from s0: (8th-order state, error vector)."""
    k = [kepler_rhs(s0, sqrt)]
    for row in RK_A:
        si = [s0[d] + h * sum_terms([a * kj[d] for a, kj in zip(row, k) if a != 0])
              for d in range(6)]
        k.append(kepler_rhs(si, sqrt))
    s8 = [s0[d] + h * sum_terms([b * ks[d] for b, ks in zip(RK_B, k) if b != 0])
          for d in range(6)]
    err = [h * sum_terms([e * ks[d] for e, ks in zip(RK_BE, k) if e != 0])
           for d in range(6)]
    return s8, err


def error_ratio(s0, s8, err, absf, maxf):
    """The mixed max-norm: max_d |err_d| / (atol + rtol max(|s8_d|, |s0_d|))."""
    ratio = None
    for d in range(6):
        r = absf(err[d]) / (ATOL + RTOL * maxf(absf(s8[d]), absf(s0[d])))
        ratio = r if ratio is None else maxf(ratio, r)
    return ratio
# <<< code:shared_scheme


def sum_terms(terms):
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total


# --------------------------------------------------------------------------- #
# The kernel (hawk)
# --------------------------------------------------------------------------- #
def _define_kernel():
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated, Vector
    from hawk import math as m

    # >>> code:hawk_kernel
    @hawk.kernel
    def rkf78_attempt(t_final: Param, terminated: Terminated, s: Mutable[Vector[6]],
                      t: Mutable[Scalar], h: Mutable[Scalar],
                      n_acc: Mutable[Scalar], n_rej: Mutable[Scalar]):
        # One adaptive RKF7(8) attempt of this sample, with its step-size controller.
        s0 = s
        t0 = t
        last = h >= t_final - t0
        step = m.where(last, t_final - t0, h)  # clip onto the final time
        s8, err = rkf78_stages(s0, step, m.sqrt)
        ratio = error_ratio(s0, s8, err, abs, m.max)
        accept = ratio < 1.0
        expo = m.where(accept, EXP_ACCEPT, EXP_REJECT)
        h = step * m.min(MAX_SCALE, m.max(MIN_SCALE, SAFETY * ratio ** expo))
        s = m.where(accept, m.vec(s8), s0)
        t1 = m.where(accept, m.where(last, t_final, t0 + step), t0)
        t = t1
        n_acc = n_acc + m.where(accept, 1.0, 0.0)
        n_rej = n_rej + m.where(accept, 0.0, 1.0)
        terminated = t1 >= t_final  # the sample finishes itself
    # <<< code:hawk_kernel

    return rkf78_attempt


def _define_active_kernel():
    """The single attempt of :func:`_define_kernel` traced under the
    active-set kind: the kernel the compaction arms launch, one attempt per
    launch at their own ``every`` cadence."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated, Vector
    from hawk import math as m
    from hawk.ext import Guard, Kind

    # >>> code:hawk_kernel_active
    ACTIVE_SET = Kind("active_set", guard=Guard(active_set=True))

    @hawk.kernel(steps=1, kind=ACTIVE_SET)
    def rkf78_attempt(t_final: Param, terminated: Terminated, s: Mutable[Vector[6]],
                      t: Mutable[Scalar], h: Mutable[Scalar],
                      n_acc: Mutable[Scalar], n_rej: Mutable[Scalar]):
        # The attempt above, unchanged; its launch covers the active set.
        s0 = s
        t0 = t
        last = h >= t_final - t0
        step = m.where(last, t_final - t0, h)
        s8, err = rkf78_stages(s0, step, m.sqrt)
        ratio = error_ratio(s0, s8, err, abs, m.max)
        accept = ratio < 1.0
        expo = m.where(accept, EXP_ACCEPT, EXP_REJECT)
        h = step * m.min(MAX_SCALE, m.max(MIN_SCALE, SAFETY * ratio ** expo))
        s = m.where(accept, m.vec(s8), s0)
        t1 = m.where(accept, m.where(last, t_final, t0 + step), t0)
        t = t1
        n_acc = n_acc + m.where(accept, 1.0, 0.0)
        n_rej = n_rej + m.where(accept, 0.0, 1.0)
        terminated = t1 >= t_final
    # <<< code:hawk_kernel_active

    return rkf78_attempt


def _check_kernel_bodies():
    """The active-set attempt's body is the plain attempt's, op for op."""
    assert _define_active_kernel().walk.digest == _define_kernel().step.walk.digest, (
        "the active-set attempt's body drifted from the plain attempt's")


#: Each eagle arm's kernel, by the function that defines it.
ARM_KERNEL = {"eagle_graph": _define_kernel, "eagle_graph_compact": _define_active_kernel,
              "eagle_graph_reorder": _define_active_kernel,
              "eagle_graph_persist": _define_kernel,
              "eagle_simulate": _define_kernel, "eagle_cpu": _define_kernel}

STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
PLANE_NAMES = STATE_NAMES + ("t", "h", "n_acc", "n_rej")


def _build(work_dir, targets=("host", "cuda")):
    """The attempt kernel deployed for ``targets`` (``eagle.deploy``): (device
    plan, host plan, source md5s), a plan ``None`` when its target was not
    built. The eagle arms deploy the same way (a cache hit after this)."""
    import eagle

    targets = tuple(targets)
    plan = eagle.deploy(_define_kernel(), targets=targets)
    sources = _kernel_sources(work_dir, targets, kernels=(("", _define_kernel()),))
    return (plan.device if "cuda" in targets else None,
            plan.host if "host" in targets else None, sources)


def _kernel_sources(work_dir, targets=("host", "cuda"), kernels=None):
    """md5 of the kernel sources the eagle arms run, built by hawk's own
    bundle builder into ``work_dir`` (the content-addressed build
    ``eagle.deploy`` uses, so a cache hit recompiles nothing)."""
    from hawk.artifact import build_bundle

    sources = {}
    if kernels is None:
        kernels = (("", _define_kernel()), ("compact/", _define_active_kernel()))
    for prefix, kernel in kernels:
        d = work_dir / f"kernel_{prefix.strip('/') or 'plain'}"
        build_bundle([kernel], d, targets=tuple(targets))
        sources.update({f"{prefix}{p.name}": cc.md5(p) for p in sorted(d.iterdir())
                        if p.suffix in (".cu", ".cpp", ".ptx")})
    return sources


# --------------------------------------------------------------------------- #
# Inputs and the analytic truth
# --------------------------------------------------------------------------- #
def _rotations(rng, n):
    import numpy as np

    inc = np.arccos(rng.uniform(-1.0, 1.0, n))
    raan = rng.uniform(0.0, 2 * math.pi, n)
    argp = rng.uniform(0.0, 2 * math.pi, n)
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    # perifocal -> inertial, columns P and Q (3-1-3 rotation)
    p = np.stack([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si])
    q = np.stack([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si])
    return p, q


def _inputs(n):
    """Initial states at periapsis (a = 1, e uniform in E_RANGE, random
    orientation) and the orbit elements the truth needs."""
    import numpy as np

    rng = np.random.default_rng([SEED, n])
    e = rng.uniform(E_RANGE[0], E_RANGE[1], n)
    e[0] = E_RANGE[1]  # the batch always holds the most eccentric orbit
    p, q = _rotations(rng, n)
    rp = SEMI_MAJOR * (1.0 - e)
    vp = np.sqrt(MU / SEMI_MAJOR * (1.0 + e) / (1.0 - e))
    r0 = p * rp
    v0 = q * vp
    return dict(e=e, p=p, q=q, x=r0[0], y=r0[1], z=r0[2], vx=v0[0], vy=v0[1], vz=v0[2])


def kepler_truth(inp, t):
    """The analytic state at time t (from periapsis at t = 0): Kepler's equation
    M = E - e sin E by Newton's method until the update stops changing E."""
    import numpy as np

    e = inp["e"]
    nmean = math.sqrt(MU / SEMI_MAJOR ** 3)
    mean = math.fmod(nmean * t, 2 * math.pi)
    ecc = np.where(e > 0.8, math.pi, mean) + np.zeros_like(e)
    for _ in range(60):
        f = ecc - e * np.sin(ecc) - mean
        d = f / (1.0 - e * np.cos(ecc))
        ecc = ecc - d
        if np.all(np.abs(d) <= 4e-16 * np.maximum(1.0, np.abs(ecc))):
            break
    a = SEMI_MAJOR
    b = a * np.sqrt(1.0 - e * e)
    r = a * (1.0 - e * np.cos(ecc))
    xp, yp = a * (np.cos(ecc) - e), b * np.sin(ecc)
    vfac = math.sqrt(MU * a) / r
    vxp, vyp = -vfac * np.sin(ecc), vfac * np.sqrt(1.0 - e * e) * np.cos(ecc)
    pos = inp["p"] * xp + inp["q"] * yp
    vel = inp["p"] * vxp + inp["q"] * vyp
    return pos, vel


def truth_bound(inp, n_acc):
    """The stated, derived (not fitted) bound on each sample's position error.

    Every accepted step's error ESTIMATE is below the controller's budget
    atol + rtol max|state| <= tol (1 + v_p) (v_p, the periapsis speed, is
    the largest state component). Taking every step's local error at that
    budget, all aligned (no cancellation), and the worst amplification a
    local velocity error suffers on a Kepler orbit (a velocity error dv at
    speed <= v_p changes the semi-major axis by 2 a^2 v_p dv / mu, which
    drifts the position along track by 3 pi da per orbit, for the orbits
    the run spans): bound = n_acc tol (1 + v_p) (1 + 6 pi n_orbits v_p).
    The integrated solution is the 8th-order one (local extrapolation), whose
    local error sits well below the 7th-order estimate, so the bound is
    conservative by construction."""
    import numpy as np

    e = inp["e"]
    vp = np.sqrt((1.0 + e) / (1.0 - e))
    tol = max(RTOL, ATOL)
    orbits = T_FINAL / (2 * math.pi)
    return n_acc * tol * (1.0 + vp) * (1.0 + 6 * math.pi * orbits * vp)


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
class _GpuArm:
    """Device planes for one eagle GPU arm (uploaded in place, so captured
    pointers stay valid across repetitions): the state as one (6, N) plane."""

    def __init__(self, n):
        import cupy as cp

        self.n = n
        self.planes = {"s": cp.empty((6, n))}
        self.planes.update({name: cp.empty(n) for name in PLANE_NAMES[6:]})
        self.terminated = cp.zeros(n, dtype=cp.bool_)
        self.runner = None

    def upload(self, inp):
        import numpy as np

        self.planes["s"].set(np.stack([inp[name] for name in STATE_NAMES]))
        self.planes["t"].fill(0.0)
        self.planes["h"].fill(H0)
        self.planes["n_acc"].fill(0.0)
        self.planes["n_rej"].fill(0.0)
        if self.runner is not None:
            self.runner.reset()

    def download(self):
        s = self.planes["s"].get()
        out = {name: s[d].copy() for d, name in enumerate(STATE_NAMES)}
        out.update({name: self.planes[name].get() for name in PLANE_NAMES[6:]})
        return out


def _make_eagle_arms(n, names):
    import eagle
    import eagle.plan  # noqa: F401  (eagle.simulate first in a process imports it circularly)

    # eagle_graph/compact/reorder are the device-loop (band) rows: their
    # labels, ARM_LOOP description and (where modelled) issued-bytes/kernel-
    # count assume the band's own launch pattern, so each is forced off the
    # fast path explicitly -- eagle_simulate (below) is the one row that
    # keeps running until_done's own auto pick, recorded per run as
    # report.mode (eagle_graph_compact/reorder build
    # on a fixed steps=1 kernel, never auto-eligible today, forced anyway so
    # a future kernel change cannot drift them onto the fast path unnoticed).
    variants = {"eagle_graph": {"_fast_mode": False},
                "eagle_graph_compact": {"every": COMPACT_EVERY, "_fast_mode": False},
                "eagle_graph_reorder": {"every": COMPACT_EVERY, "reorder": REORDER_THETA,
                                        "_fast_mode": False},
                "eagle_graph_persist": {"_fast_mode": "persist",
                                         "_lane_utilisation": True}}
    arms = {}
    for name in (a for a in EAGLE_GPU_ARMS if a in names):
        rkf78_attempt = ARM_KERNEL[name]()
        options = variants.get(name)
        st = _GpuArm(n)
        if name == "eagle_simulate":
            # >>> code:eagle_simulate
            st.runner = eagle.simulation(rkf78_attempt, t_final=T_FINAL,
                                         max_steps=MAX_ATTEMPTS, **st.planes)
            # <<< code:eagle_simulate
        else:
            # >>> code:hawk_launch
            st.runner = eagle.until_done(eagle.deploy(rkf78_attempt), max_steps=MAX_ATTEMPTS,
                                         t_final=T_FINAL, terminated=st.terminated,
                                         **st.planes, **options)
            # <<< code:hawk_launch
        arm = {"state": st, "download": st.download}
        if name == "eagle_graph_reorder":
            arm["fired"] = []

        def run(st=st, arm=arm):
            report = st.runner.run()  # the planes come back in sample order
            report = getattr(report, "report", report)  # a simulation's result
            arm["build_s"] = report.build_s
            arm["report"] = report
            if "fired" in arm:
                arm["fired"].append(report.reorders)

        arm["run"] = run
        arms[name] = arm
    return arms


class _ArrayState:
    """Whole-batch arrays for the masked arms (CuPy or PyTorch)."""

    def __init__(self, xp_name, n):
        self.xp_name = xp_name
        self.n = n
        self.out = None
        if xp_name == "cupy":
            import cupy as cp

            self.planes = {name: cp.empty(n) for name in PLANE_NAMES}
        else:
            import torch

            self.planes = {name: torch.empty(n, dtype=torch.float64, device="cuda")
                           for name in PLANE_NAMES}

    def upload(self, inp):
        for name in STATE_NAMES:
            if self.xp_name == "cupy":
                self.planes[name].set(inp[name])
            else:
                import torch

                self.planes[name].copy_(torch.from_numpy(inp[name]))
        self.planes["t"].fill_(0.0) if self.xp_name == "torch" else self.planes["t"].fill(0.0)
        for name, val in (("h", H0), ("n_acc", 0.0), ("n_rej", 0.0)):
            if self.xp_name == "torch":
                self.planes[name].fill_(val)
            else:
                self.planes[name].fill(val)

    def download(self):
        src = self.out if self.out is not None else self.planes
        if self.xp_name == "cupy":
            return {k: v.get() for k, v in src.items()}
        return {k: v.cpu().numpy() for k, v in src.items()}


# >>> code:cupy_masked
def cupy_attempt(s, t, h, n_acc, n_rej):
    """One masked RKF7(8) attempt over the whole batch (CuPy arrays)."""
    import cupy as cp

    running = t < T_FINAL
    last = h >= T_FINAL - t
    step = cp.where(last, T_FINAL - t, h)
    s8, err = rkf78_stages(s, step, cp.sqrt)
    ratio = error_ratio(s, s8, err, cp.abs, cp.maximum)
    accept = ratio < 1.0
    take = running & accept
    expo = cp.where(accept, EXP_ACCEPT, EXP_REJECT)
    scale = cp.clip(SAFETY * ratio ** expo, MIN_SCALE, MAX_SCALE)
    s = [cp.where(take, s8[d], s[d]) for d in range(6)]
    t = cp.where(take, cp.where(last, T_FINAL, t + step), t)
    h = cp.where(running, step * scale, h)
    return s, t, h, n_acc + take, n_rej + (running & ~accept), running


def cupy_run(st):
    p = st.planes
    s, t, h = [p[k] for k in STATE_NAMES], p["t"], p["h"]
    n_acc, n_rej = p["n_acc"], p["n_rej"]
    for _ in range(MAX_ATTEMPTS):
        s, t, h, n_acc, n_rej, running = cupy_attempt(s, t, h, n_acc, n_rej)
        if not bool(running.any()):
            break
    return s, t, h, n_acc, n_rej
# <<< code:cupy_masked


# >>> code:torch_masked
def torch_attempt(s, t, h, n_acc, n_rej):
    """One masked RKF7(8) attempt over the whole batch (PyTorch tensors)."""
    import torch

    running = t < T_FINAL
    last = h >= T_FINAL - t
    step = torch.where(last, T_FINAL - t, h)
    s8, err = rkf78_stages(s, step, torch.sqrt)
    ratio = error_ratio(s, s8, err, torch.abs, torch.maximum)
    accept = ratio < 1.0
    take = running & accept
    expo = torch.where(accept, EXP_ACCEPT, EXP_REJECT)
    scale = torch.clamp(SAFETY * ratio ** expo, MIN_SCALE, MAX_SCALE)
    s = [torch.where(take, s8[d], s[d]) for d in range(6)]
    t = torch.where(take, torch.where(last, T_FINAL, t + step), t)
    h = torch.where(running, step * scale, h)
    return s, t, h, n_acc + take, n_rej + (running & ~accept), running


def torch_run(st):
    import torch

    with torch.inference_mode():
        p = st.planes
        s, t, h = [p[k] for k in STATE_NAMES], p["t"], p["h"]
        n_acc, n_rej = p["n_acc"], p["n_rej"]
        for _ in range(MAX_ATTEMPTS):
            s, t, h, n_acc, n_rej, running = torch_attempt(s, t, h, n_acc, n_rej)
            if not bool(running.any()):
                break
    return s, t, h, n_acc, n_rej
# <<< code:torch_masked


# >>> code:torch_graphed
def torch_capture(p, block):
    """Capture ``block`` masked attempts over the static tensors ``p`` as one
    CUDA graph; returns (graph, flag), flag = "any sample still running"."""
    import torch

    def attempts():
        s, t, h = [p[k] for k in STATE_NAMES], p["t"], p["h"]
        n_acc, n_rej = p["n_acc"], p["n_rej"]
        for _ in range(block):
            s, t, h, n_acc, n_rej, _ = torch_attempt(s, t, h, n_acc, n_rej)
        for k, v in zip(PLANE_NAMES, (*s, t, h, n_acc, n_rej)):
            p[k].copy_(v)
        return (p["t"] < T_FINAL).any()

    side = torch.cuda.Stream()  # warm-up on a side stream before capture
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(side):
        attempts()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        flag = attempts()
    return graph, flag


def torch_graphed_run(graph, flag, block):
    for _ in range(-(-MAX_ATTEMPTS // block)):
        graph.replay()
        if not bool(flag):
            break
# <<< code:torch_graphed


def _make_array_arms(n, names):
    import torch

    arms = {}
    if "cupy_masked" in names:
        c = _ArrayState("cupy", n)

        def run_cupy():
            s, t, h, a, r = cupy_run(c)
            c.out = dict(zip(PLANE_NAMES, (*s, t, h, a, r)))

        arms["cupy_masked"] = dict(state=c, run=run_cupy, download=c.download)
    if "torch_masked" in names:
        tm = _ArrayState("torch", n)

        def run_torch():
            s, t, h, a, r = torch_run(tm)
            tm.out = dict(zip(PLANE_NAMES, (*s, t, h, a, r)))

        arms["torch_masked"] = dict(state=tm, run=run_torch, download=tm.download)
    if "torch_graphed" in names:
        tg = _ArrayState("torch", n)
        tg.upload(_inputs(n))
        t0 = time.perf_counter()
        graph, flag = torch_capture(tg.planes, GRAPH_BLOCK)
        torch.cuda.synchronize()
        arms["torch_graphed"] = dict(state=tg, run=lambda: torch_graphed_run(
            graph, flag, GRAPH_BLOCK), download=tg.download,
            build_s=time.perf_counter() - t0)
    return arms


# --------------------------------------------------------------------------- #
# JAX
# --------------------------------------------------------------------------- #
# >>> code:jax_vmap
def jax_integrate_fn():
    """jit(vmap(per-sample while loop)) over the initial states."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    jax.config.update("jax_enable_x64", True)

    def one(s0):
        def cond(c):
            _, t, _, n_acc, n_rej = c
            return (t < T_FINAL) & (n_acc + n_rej < MAX_ATTEMPTS)

        def body(c):
            s, t, h, n_acc, n_rej = c
            last = h >= T_FINAL - t
            step = jnp.where(last, T_FINAL - t, h)
            s8, err = rkf78_stages(list(s), step, jnp.sqrt)
            ratio = error_ratio(list(s), s8, err, jnp.abs, jnp.maximum)
            accept = ratio < 1.0
            expo = jnp.where(accept, EXP_ACCEPT, EXP_REJECT)
            scale = jnp.clip(SAFETY * ratio ** expo, MIN_SCALE, MAX_SCALE)
            s = jnp.where(accept, jnp.stack(s8), s)
            t = jnp.where(accept, jnp.where(last, T_FINAL, t + step), t)
            return s, t, step * scale, n_acc + accept, n_rej + (1 - accept)

        init = (s0, jnp.float64(0.0), jnp.float64(H0), jnp.int64(0), jnp.int64(0))
        return lax.while_loop(cond, body, init)

    return jax.jit(jax.vmap(one))
# <<< code:jax_vmap


def _jax_compile(n):
    import jax
    import jax.numpy as jnp

    fn = jax_integrate_fn()
    return fn.lower(jax.ShapeDtypeStruct((n, 6), jnp.float64)).compile()


class _JaxState:
    def __init__(self, n, compiled):
        self.n = n
        self.compiled = compiled
        self.s0 = None
        self.out = None

    def upload(self, inp):
        import jax
        import numpy as np

        self.s0 = jax.device_put(np.stack([inp[k] for k in STATE_NAMES], axis=1))
        self.s0.block_until_ready()

    def run(self):
        self.out = self.compiled(self.s0)
        self.out[1].block_until_ready()

    def download(self):
        import numpy as np

        s, t, h, a, r = (np.asarray(v) for v in self.out)
        d = {k: s[:, i].copy() for i, k in enumerate(STATE_NAMES)}
        d.update(t=t, h=h, n_acc=a.astype(np.float64), n_rej=r.astype(np.float64))
        return d


def _make_jax_arm(n, compiled=None):
    t0 = time.perf_counter()
    compiled = compiled if compiled is not None else _jax_compile(n)
    st = _JaxState(n, compiled)
    return dict(state=st, run=st.run, download=st.download,
                compile_s=time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Warp
# --------------------------------------------------------------------------- #
_WARP = []


def _warp_kernel():
    if _WARP:
        return _WARP[0]
    import warp as wp

    cc.warp_init()
    f64 = wp.float64
    mat_a = wp.types.matrix(shape=(N_STAGES, N_STAGES), dtype=f64)
    vec_s = wp.types.vector(length=N_STAGES, dtype=f64)
    mat_k = wp.types.matrix(shape=(N_STAGES, 6), dtype=f64)
    vec6 = wp.types.vector(length=6, dtype=f64)
    flat_a = [0.0] * (N_STAGES * N_STAGES)
    for i, row in enumerate(RK_A):
        for j, a in enumerate(row):
            flat_a[(i + 1) * N_STAGES + j] = a
    # >>> code:warp_kernel
    A = wp.constant(mat_a(*flat_a))
    B = wp.constant(vec_s(*RK_B))
    BE = wp.constant(vec_s(*RK_BE))

    @wp.func
    def rhs(s: vec6):
        r2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2]
        ir3 = f64(1.0) / (r2 * wp.sqrt(r2))
        return vec6(s[3], s[4], s[5], -s[0] * ir3, -s[1] * ir3, -s[2] * ir3)

    @wp.kernel
    def rkf78(x: wp.array(dtype=f64), y: wp.array(dtype=f64), z: wp.array(dtype=f64),
              vx: wp.array(dtype=f64), vy: wp.array(dtype=f64), vz: wp.array(dtype=f64),
              t_out: wp.array(dtype=f64), h_out: wp.array(dtype=f64),
              acc_out: wp.array(dtype=f64), rej_out: wp.array(dtype=f64),
              t_final: f64, rtol: f64, atol: f64, h0: f64, max_attempts: int):
        # one thread per sample: its whole adaptive loop to the final time
        i = wp.tid()
        s = vec6(x[i], y[i], z[i], vx[i], vy[i], vz[i])
        t = f64(0.0)
        h = h0
        n_acc = int(0)
        n_rej = int(0)
        while t < t_final and n_acc + n_rej < max_attempts:
            last = h >= t_final - t
            step = h
            if last:
                step = t_final - t
            k = mat_k()
            k0 = rhs(s)
            for d in range(6):
                k[0, d] = k0[d]
            for st in range(1, 13):
                si = s
                for j in range(st):
                    if A[st, j] != f64(0.0):
                        for d in range(6):
                            si[d] = si[d] + step * A[st, j] * k[j, d]
                ks = rhs(si)
                for d in range(6):
                    k[st, d] = ks[d]
            ratio = f64(0.0)
            s8 = s
            for d in range(6):
                acc = f64(0.0)
                err = f64(0.0)
                for st in range(13):
                    acc = acc + B[st] * k[st, d]
                    err = err + BE[st] * k[st, d]
                s8[d] = s[d] + step * acc
                sc = atol + rtol * wp.max(wp.abs(s8[d]), wp.abs(s[d]))
                ratio = wp.max(ratio, wp.abs(step * err) / sc)
            expo = f64(-0.125)
            if ratio < f64(1.0):
                expo = f64(-1.0 / 9.0)
            scale = wp.min(f64(5.0), wp.max(f64(0.2), f64(0.9) * wp.pow(ratio, expo)))
            if ratio < f64(1.0):
                s = s8
                t = t + step
                if last:
                    t = t_final
                n_acc += 1
            else:
                n_rej += 1
            h = step * scale
        x[i] = s[0]
        y[i] = s[1]
        z[i] = s[2]
        vx[i] = s[3]
        vy[i] = s[4]
        vz[i] = s[5]
        t_out[i] = t
        h_out[i] = h
        acc_out[i] = f64(n_acc)
        rej_out[i] = f64(n_rej)
    # <<< code:warp_kernel

    _WARP.append(rkf78)
    return rkf78


class _WarpState:
    def __init__(self, n):
        import warp as wp

        self.wp = wp
        self.planes = {k: wp.empty(n, dtype=wp.float64, device="cuda:0")
                       for k in PLANE_NAMES}

    def upload(self, inp):
        for k in STATE_NAMES:
            self.planes[k].assign(inp[k])
        self.wp.synchronize_device("cuda:0")

    def download(self):
        return {k: v.numpy().copy() for k, v in self.planes.items()}


def _make_warp_arm(n):
    import warp as wp

    kernel = _warp_kernel()
    s = _WarpState(n)

    def run():
        # >>> code:warp_launch
        wp.launch(kernel, dim=n, inputs=[*s.planes.values(), T_FINAL, RTOL, ATOL, H0,
                                         MAX_ATTEMPTS], device="cuda:0")
        # <<< code:warp_launch

    return dict(state=s, run=run, download=s.download)


# --------------------------------------------------------------------------- #
# The CPU arm (the same hawk kernel through eagle's host team)
# --------------------------------------------------------------------------- #
class _CpuArm:
    def __init__(self, plan, n):
        import numpy as np

        self.plan = plan
        self.n = n
        self.planes = {"s": np.empty((6, n))}
        self.planes.update({k: np.empty(n) for k in PLANE_NAMES[6:]})
        self.terminated = np.zeros(n, dtype=bool)
        self.runner = None
        self.kernel_s = None  # the loop launches and checks in one call

    def upload(self, inp):
        import numpy as np

        np.copyto(self.planes["s"], np.stack([inp[k] for k in STATE_NAMES]))
        self.planes["t"].fill(0.0)
        self.planes["h"].fill(H0)
        self.planes["n_acc"].fill(0.0)
        self.planes["n_rej"].fill(0.0)
        self.terminated.fill(False)

    def run(self):
        import eagle

        # >>> code:eagle_cpu
        eagle.run_until_done(self.plan, max_steps=MAX_ATTEMPTS, t_final=T_FINAL,
                             terminated=self.terminated, **self.planes)
        # <<< code:eagle_cpu

    def download(self):
        out = {k: self.planes["s"][d].copy() for d, k in enumerate(STATE_NAMES)}
        out.update({k: self.planes[k].copy() for k in PLANE_NAMES[6:]})
        return out


def _make_arms(n, names):
    arms = {}
    if any(a in names for a in EAGLE_GPU_ARMS):
        arms.update(_make_eagle_arms(n, names))
    if any(a in names for a in ("cupy_masked", "torch_masked", "torch_graphed")):
        arms.update(_make_array_arms(n, names))
    if "jax_vmap" in names:
        arms["jax_vmap"] = _make_jax_arm(n)
    if "warp_kernel" in names:
        arms["warp_kernel"] = _make_warp_arm(n)
    if "eagle_cpu" in names:
        import eagle

        rkf78_attempt = _define_kernel()
        # >>> code:eagle_cpu_deploy
        plan = eagle.deploy(rkf78_attempt)  # host planes select the host build
        # <<< code:eagle_cpu_deploy
        c = _CpuArm(plan, n)
        arms["eagle_cpu"] = dict(state=c, run=c.run, download=c.download)
    return {a: arms[a] for a in names}


# --------------------------------------------------------------------------- #
# FLOP count (from the scheme code itself)
# --------------------------------------------------------------------------- #
class _Counter:
    """A number that counts the arithmetic done on it (+ - * / abs max sqrt,
    1 FLOP each; comparisons 0)."""

    count = 0

    def __init__(self, v=1.0):
        self.v = v

    def _op(self, other=None):
        _Counter.count += 1
        return _Counter()

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _op
    __truediv__ = __rtruediv__ = __abs__ = __neg__ = _op


def _count_flops():
    _Counter.count = 0
    s0 = [_Counter() for _ in range(6)]
    s8, err = rkf78_stages(s0, _Counter(), lambda v: v._op())
    error_ratio(s0, s8, err, lambda v: v._op(), lambda a, b: a._op())
    scheme = _Counter.count
    # the controller around it, as written in rkf78_attempt: t_final - t0 (the
    # clip), ratio ** expo, SAFETY *, the clamp (min, max), step * scale, t0 + step
    controller = 7
    return scheme, controller


FLOPS_SCHEME, FLOPS_CONTROLLER = _count_flops()
FLOPS_PER_ATTEMPT = FLOPS_SCHEME + FLOPS_CONTROLLER
_COUNTING_RULE = (
    "useful FLOPs = FLOPs per attempt x the attempts the samples actually make "
    "(accepted + rejected, summed over samples, as counted by the arm itself); "
    f"{FLOPS_PER_ATTEMPT} FLOP per attempt: {FLOPS_SCHEME} counted by running "
    "the scheme code (rkf78_stages, kepler_rhs, error_ratio) on a counting number "
    "(+, -, x, /, abs, max and sqrt one FLOP each; zero coefficients skipped, as "
    f"every arm skips them) plus {FLOPS_CONTROLLER} in the controller (the clip, "
    "the power, the safety factor, the two-sided clamp, the new step, the new "
    "time). The same count for every arm: work a masked arm spends on finished "
    "samples, and a batched loop spends past a sample's end, is not useful.")


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
_sync = cc.sync


def _sync_all(arm_name):
    if arm_name == "eagle_cpu":
        return
    _sync()
    if arm_name.startswith("torch"):
        import torch

        torch.cuda.synchronize()
    elif arm_name == "warp_kernel":
        import warp as wp

        wp.synchronize_device("cuda:0")


def _timed(name, arm, inp):
    t_a = time.perf_counter()
    arm["state"].upload(inp)
    _sync_all(name)
    t_b = time.perf_counter()
    arm["run"]()
    _sync_all(name)
    t_c = time.perf_counter()
    out = arm["download"]()
    t_d = time.perf_counter()
    if name in ("cupy_masked", "torch_masked"):
        # the masked arms' temporaries (about 1.2 KB per sample) go back to the
        # driver after each run, outside the timed span, so the arms sharing
        # the GPU in this process do not hold them all at once at N = 1e6
        arm["state"].out = None
        cc.trim_pools()
    return t_c - t_b, t_d - t_a, out


_median_iqr = cc.median_iqr


def _summary(values):
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), "p50": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max()),
            "mean": float(v.mean()), "sum": float(v.sum())}


def _verify(outs, inp, ref_arm):
    """Every arm against the analytic truth (the stated bound) and against the
    reference arm (step counts, final states)."""
    import numpy as np

    pos_t, vel_t = kepler_truth(inp, T_FINAL)
    ref = outs[ref_arm]
    bound = truth_bound(inp, ref["n_acc"])
    report = {}
    for arm, o in outs.items():
        assert np.all(o["t"] == T_FINAL), f"{arm}: a sample did not reach the final time"
        pos = np.stack([o["x"], o["y"], o["z"]])
        vel = np.stack([o["vx"], o["vy"], o["vz"]])
        perr = np.sqrt(((pos - pos_t) ** 2).sum(axis=0))
        verr = np.sqrt(((vel - vel_t) ** 2).sum(axis=0))
        over = perr > truth_bound(inp, o["n_acc"])
        assert not over.any(), (
            f"{arm}: {int(over.sum())} samples exceed the truth bound "
            f"(worst {float(perr[over].max()):.3e})")
        d_acc = o["n_acc"] - ref["n_acc"]
        d_rej = o["n_rej"] - ref["n_rej"]
        differ = (d_acc != 0) | (d_rej != 0)
        same = ~differ
        dpos = np.sqrt(((pos - np.stack([ref["x"], ref["y"], ref["z"]])) ** 2).sum(axis=0))
        report[arm] = {
            "max_position_error_vs_truth": float(perr.max()),
            "median_position_error_vs_truth": float(np.median(perr)),
            "max_velocity_error_vs_truth": float(verr.max()),
            "max_error_over_bound": float((perr / bound).max()),
            "samples_with_step_counts_differing_from_reference": int(differ.sum()),
            "max_abs_accepted_difference": int(np.abs(d_acc).max()),
            "max_abs_rejected_difference": int(np.abs(d_rej).max()),
            "max_position_difference_vs_reference_same_counts": (
                float(dpos[same].max()) if same.any() else None),
            "attempts_total": float((o["n_acc"] + o["n_rej"]).sum()),
        }
    steps = {"accepted": _summary(ref["n_acc"]), "rejected": _summary(ref["n_rej"])}
    return report, steps


def _run_n(n, reps, bus_id, names, kernel_pass=False):
    """All arms at one N: build, warm-up + verify, interleaved repetitions."""
    inp = _inputs(n)
    arms = _make_arms(n, names)
    order = list(arms)
    outs, reps_for = {}, {}
    for name in order:  # warm-up, also the verification run
        warm, _, outs[name] = _timed(name, arms[name], inp)
        reps_for[name] = (reps if warm * reps <= SLOW_ARM_BUDGET_S
                          else min(reps, SLOW_ARM_REPS))
    ref = "eagle_graph" if "eagle_graph" in outs else order[0]
    agreement, steps = _verify(outs, inp, ref)
    if kernel_pass:
        from cupy.cuda import nvtx

        for r in range(reps):
            for name in [a for a in order if r < reps_for[a]]:
                nvtx.RangePush(f"rk78|{n}|{name}|{r}")
                _timed(name, arms[name], inp)
                nvtx.RangePop()
        return None
    walls = {a: [] for a in order}
    e2e = {a: [] for a in order}
    kern = {a: [] for a in order}
    poller = cc.ClockPoller(bus_id)
    try:
        for r in range(reps):
            rot = order[r % len(order):] + order[:r % len(order)]
            for name in [a for a in rot if r < reps_for[a]]:
                w, e, _ = _timed(name, arms[name], inp)
                walls[name].append(w)
                e2e[name].append(e)
                if name == "eagle_cpu" and arms[name]["state"].kernel_s is not None:
                    kern[name].append(arms[name]["state"].kernel_s)
    finally:
        clocks = poller.stop()
    rows = []
    for name in order:
        attempts = agreement[name]["attempts_total"]
        wall = _median_iqr(walls[name])
        row = {
            "n": n, "arm": name, "wall_s": wall, "repetitions": len(walls[name]),
            "end_to_end_s": _median_iqr(e2e[name]),
            "attempts": attempts,
            "attempt_samples_per_s": attempts / wall["median"],
            "useful_flops_per_s": FLOPS_PER_ATTEMPT * attempts / wall["median"],
            "kernel_only_s": (statistics.median(kern[name]) if kern[name] else None),
            "build_s": arms[name].get("build_s"),
            "compile_s": arms[name].get("compile_s"),
            "lane_utilisation": getattr(arms[name].get("report"),
                                       "lane_utilisation", None),
            # the launch shape the runner actually took ("band", "fused_one"
            # or "persist"; None for a non-eagle arm) -- eagle_simulate
            # auto-routes past the band at/above capacity, so this is read off the run, never assumed.
            "run_mode": getattr(arms[name].get("report"), "mode", None),
        }
        if "fired" in arms[name]:
            row["reorders_fired"] = arms[name]["fired"][-1] if arms[name]["fired"] else None
        rows.append(row)
    extra = {"n": n, "reference_arm": ref, "agreement": agreement, "steps": steps,
             "gpu_state_during": clocks}
    return rows, extra


# --------------------------------------------------------------------------- #
# Memory pass (fresh process per (arm, N); NVML per-process device memory)
# --------------------------------------------------------------------------- #
_MEMORY_METHOD = (
    "A separate pass after the timing pass. Each (arm, N) runs in a fresh "
    "process: imports, the arm's kernels built and a warm-up run on a "
    f"{WARM_N}-sample batch (every compile happens here; JAX also compiles for "
    "shape N), every caching allocator trimmed (CuPy's pools, PyTorch's "
    "cache), then the baseline read; then the arm built for N (eagle's graph, "
    "PyTorch's captured graph) and one full run with upload and download; then "
    "the reading again, the allocators still holding their high-water "
    "reservation (the same measure for every arm, no sampling needed). Device: "
    "the memory NVML accounts to the process on the timed GPU, after minus "
    "baseline. Caveats: each allocator's rounding and growth policy count (JAX's "
    "allocator grows in regions that can exceed the request; PyTorch rounds "
    "blocks up and a captured CUDA graph keeps a private pool; Warp's "
    "stream-ordered pool reserves in chunks of tens of MiB, its release "
    "threshold set to 0 before the baseline); the driver accounts in pages of "
    "about 2 MiB, so small-N rows read 0 or one page. Host: the peak resident "
    "set (VmHWM, reset to the current RSS at the baseline) minus the baseline "
    "RSS. Minimum: the planes the workload needs, N x 10 (state, time, step, "
    "two counters) x 8 bytes.")


def _probe_arm(arm, n, compiled=None):
    """One arm for a probe process (an eagle arm deploys its kernel here)."""
    if arm == "jax_vmap":
        return _make_jax_arm(n, compiled)
    return _make_arms(n, (arm,))[arm]


def _memory_probe(arm, n, bus_id):
    import gc

    reply_fd = cc.probe_reply_fd()
    gpu = arm != "eagle_cpu"
    a = _probe_arm(arm, WARM_N)
    _timed(arm, a, _inputs(WARM_N))
    compiled = _jax_compile(n) if arm == "jax_vmap" else None
    del a
    gc.collect()
    cc.trim_pools()
    if gpu:
        _sync()
    dev_base, src = cc.process_device_bytes(bus_id) if gpu else (None, None)
    rss_base = cc.status_kib("VmRSS")
    cc.reset_host_peak()
    inp = _inputs(n)
    a = _probe_arm(arm, n, compiled)
    _, _, out = _timed(arm, a, inp)
    assert (out["t"] == T_FINAL).all(), f"memory probe {arm}: unfinished samples"
    dev_after, _ = cc.process_device_bytes(bus_id) if gpu else (None, None)
    cc.probe_reply(reply_fd, {
        "device_baseline_bytes": dev_base, "device_after_bytes": dev_after,
        "device_peak_bytes": None if not gpu else max(0, dev_after - dev_base),
        "device_source": src,
        "host_peak_bytes": max(0, cc.status_kib("VmHWM") - rss_base) * 1024})


def _run_memory_pass(ns, bus_id, names):
    rows = []
    for n in ns:
        for arm in names:
            r = cc.run_probe(__file__, ["--memory-probe", arm, str(n), bus_id],
                             timeout=1800, what=f"memory probe {arm} {n}")
            r.update(n=n, arm=arm, minimum_bytes=n * len(PLANE_NAMES) * 8)
            rows.append(r)
            print(f"memory {arm:20s} N={n:>8,} device "
                  f"{(r['device_peak_bytes'] or 0) / 2**20:.3g} MiB", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Compile-time pass (fresh process per measurement; cold = empty caches)
# --------------------------------------------------------------------------- #
COMPILE_REPS = 3
NO_COMPILE_ARMS = ("torch_masked", "torch_graphed")
_CACHE_ENV = {"HAWK_CACHE_DIR": "hawk", "CUPY_CACHE_DIR": "cupy",
              "CUDA_CACHE_PATH": "cuda", "RK78_JAX_CACHE": "jax",
              "WARP_CACHE_PATH": "warp"}
_COMPILE_METHOD = (
    "Its own pass, never part of any wall. Each measurement is a fresh process: "
    "imports and the GPU context run untimed, then the timer spans the first "
    "call to a ready kernel (trace + code generation + compile + load, or the "
    "cache lookup). Cold: every cache empty (HAWK_CACHE_DIR, CUPY_CACHE_DIR, "
    "CUDA_CACHE_PATH, WARP_CACHE_PATH and JAX's persistent compilation cache, "
    "each a fresh directory); warm: a new process on the caches the first cold "
    "run populated. eagle GPU: eagle.deploy of the hawk attempt kernel and of "
    "its active-set variant (host and device builds each), and one eagle graph "
    f"of each built and run on {WARM_N} samples (eagle.simulate deploys the "
    "first again, a cache hit). eagle CPU: the same eagle.deploy of the attempt "
    "kernel (host and device builds) and one host run. CuPy: one "
    f"masked attempt on {WARM_N} samples (every elementwise kernel compiled on "
    "first use). JAX: lower + compile from abstract shapes at N. Warp: the "
    "kernel module (code generation, compile, load) and one launch. Medians "
    "of 3 cold and 3 warm processes.")


def _compile_keys(ns):
    keys = [("hawk_cuda", list(EAGLE_GPU_ARMS),
             "eagle.deploy of the attempt kernel and its active-set variant"),
            ("hawk_host", ["eagle_cpu"], "eagle.deploy of the attempt kernel, host run"),
            ("cupy", ["cupy_masked"], "CuPy elementwise kernels")]
    keys += [(f"jax:{n}", ["jax_vmap"], f"XLA executable, N = {n:,}") for n in ns]
    keys.append(("warp", ["warp_kernel"], "Warp kernel module"))
    return keys


def _compile_probe(key):
    reply_fd = cc.probe_reply_fd()
    tiny = _inputs(WARM_N)
    if key.startswith("jax:"):
        import jax

        cc.jax_persistent_cache(os.environ["RK78_JAX_CACHE"])
        jax.devices("gpu")
        t0 = time.perf_counter()
        _jax_compile(int(key.split(":")[1]))
        seconds = time.perf_counter() - t0
    else:
        import cupy as cp

        import eagle  # noqa: F401  (imports untimed)
        import eagle.plan  # noqa: F401
        import hawk  # noqa: F401
        import hawk.artifact  # noqa: F401
        import hawk.compile  # noqa: F401
        if key == "warp":
            import warp  # noqa: F401
        cp.cuda.Device().synchronize()
        names = {"hawk_cuda": ("eagle_graph", "eagle_graph_compact"),
                 "hawk_host": ("eagle_cpu",), "cupy": ("cupy_masked",),
                 "warp": ("warp_kernel",)}[key]
        t0 = time.perf_counter()
        arms = _make_arms(WARM_N, names)
        if key == "cupy":
            st = arms["cupy_masked"]["state"]
            st.upload(tiny)
            p = st.planes
            cupy_attempt([p[k] for k in STATE_NAMES], p["t"], p["h"], p["n_acc"],
                         p["n_rej"])
            cp.cuda.Device().synchronize()
        else:
            for name, a in arms.items():
                _timed(name, a, tiny)
        seconds = time.perf_counter() - t0
    cc.probe_reply(reply_fd, {"seconds": seconds})


def _run_compile_pass(ns):
    rows = []
    for key, arms, what in _compile_keys(ns):
        cold, warm = cc.cold_warm(__file__, ["--compile-probe", key], _CACHE_ENV,
                                  reps=COMPILE_REPS, prefix="rk78_cache_", timeout=900,
                                  what=f"compile probe {key}")
        rows.append({"key": key, "arms": arms, "what": what,
                     "cold_s": _median_iqr(cold), "warm_s": _median_iqr(warm)})
        print(f"compile {key:12s} cold {statistics.median(cold):.3g}s "
              f"warm {statistics.median(warm):.3g}s", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Kernel-only pass (Nsight Systems)
# --------------------------------------------------------------------------- #
KERNEL_PASS_RUNS = 3
#: The masked arms' traced runs are long (every sample, every attempt, until
#: the slowest finishes): one traced run each, after the warm-up.
KERNEL_PASS_RUNS_MASKED = 1
MASKED_ARMS = ("cupy_masked", "torch_masked", "torch_graphed")


def _nsys_kernel_times(ns, work, names):
    nsys = cc.which("nsys")
    if nsys is None:
        return None, None, "nsys not found"
    gpu = [a for a in names if a != "eagle_cpu"]
    out, out_counts, failed = {}, {}, []
    for n in ns:
        for arm in gpu:
            try:
                ranges = cc.nsys_ranges(nsys, work / f"kp_{n}_{arm}", __file__,
                                        ["--kernel-pass", "--ns", str(n), "--arms", arm],
                                        f"rk78|{n}|{arm}|", timeout=3600)
                valid = [(seconds, memcpy, api) for _text, count, seconds, memcpy, api
                        in ranges if count > 0]
                out[(n, arm)] = (statistics.median(s for s, _, _ in valid)
                                 if valid else None)
                if valid:
                    memcpys = [m for _, m, _ in valid if m is not None]
                    apis = [a for _, _, a in valid if a is not None]
                    out_counts[(n, arm)] = {
                        "memcpy": statistics.median(memcpys) if memcpys else None,
                        "api_calls": statistics.median(apis) if apis else None}
                else:
                    out_counts[(n, arm)] = None
            except (subprocess.SubprocessError, sqlite3.Error, OSError):
                failed.append(f"{arm}@{n}")
                out[(n, arm)] = None
                out_counts[(n, arm)] = None
    note = (f"{cc.nsys_version(nsys)}: sum of kernel durations over one run, median of "
            f"{KERNEL_PASS_RUNS} traced runs ({KERNEL_PASS_RUNS_MASKED} for the masked "
            "arms; --cuda-graph-trace=node) after a warm-up, a separate "
            "process per (arm, N); includes the loop-control kernels "
            "and every library kernel")
    if failed:
        note += "; traced process failed for: " + ", ".join(failed)
    return out, out_counts, note


# --------------------------------------------------------------------------- #
# Code comparison (the marked blocks of this file)
# --------------------------------------------------------------------------- #
#: Which marked blocks make up each arm's code (the scheme block is shared by
#: every arm except Warp, which writes its stages in its own kernel).
ARM_CODE = {
    "eagle_graph": ("hawk_kernel", "hawk_launch"),
    "eagle_graph_compact": ("hawk_kernel_active", "hawk_launch"),
    "eagle_graph_reorder": ("hawk_kernel_active", "hawk_launch"),
    "eagle_graph_persist": ("hawk_kernel", "hawk_launch"),
    "eagle_simulate": ("hawk_kernel", "eagle_simulate"),
    "cupy_masked": ("cupy_masked",),
    "torch_masked": ("torch_masked",),
    "torch_graphed": ("torch_masked", "torch_graphed"),
    "jax_vmap": ("jax_vmap",),
    "warp_kernel": ("warp_kernel", "warp_launch"),
    "eagle_cpu": ("hawk_kernel", "eagle_cpu_deploy", "eagle_cpu"),
}


def code_blocks(path=None):
    """{block name: source lines} for every ``# >>> code:NAME`` ...
    ``# <<< code:NAME`` pair in this file (markers excluded)."""
    return cc.code_text(path or __file__)


def code_line_counts(path=None):
    """Non-blank, non-comment lines per block and per arm."""
    per_block = {k: cc.count_code_lines(v) for k, v in code_blocks(path).items()}
    per_arm = {a: sum(per_block[b] for b in bs) for a, bs in ARM_CODE.items()}
    return {"blocks": per_block, "arms": per_arm, "arm_blocks": ARM_CODE,
            "shared_scheme_lines": per_block["shared_scheme"],
            "rule": ("non-blank, non-comment lines between the `# >>> code:NAME` and "
                     "`# <<< code:NAME` markers of the script; every arm but Warp "
                     "also uses the shared scheme block (the RKF7(8) stages, the "
                     "right-hand side, the error norm), counted once in "
                     "shared_scheme_lines")}


# --------------------------------------------------------------------------- #
# Device facts and software
# --------------------------------------------------------------------------- #
_device_facts, _software, _version_of = cc.device_facts, cc.software, cc.version_of


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_fmt_time, _fmt_rate, _fmt_pct = cc.fmt_time_si, cc.fmt_rate_si, cc.fmt_pct_si
_fmt_mib = cc.fmt_mib


def _fit_notes(card):
    """'Where each tool fits': one neutral line per arm, every comparison
    carrying the card number it rests on (largest N)."""
    rows = {r["arm"]: r for r in card["results"] if r["n"] == max(card["ns"])}
    n = max(card["ns"])
    lines = card["code_lines"]["arms"]

    def w(a):
        return _fmt_time(rows[a]["wall_s"]["median"]) if a in rows else "–"

    def ratio(a, b):
        if a not in rows or b not in rows:
            return "–"
        return f"{rows[a]['wall_s']['median'] / rows[b]['wall_s']['median']:.3g}×"

    fastest = min((r for a, r in rows.items() if a != "eagle_cpu"),
                  key=lambda r: r["wall_s"]["median"], default=None)
    intro = (f"Each tool here solves the same {card['workload']['n_stages']}-stage "
             f"adaptive problem to the same tolerance and agrees with the analytic "
             f"orbit; they differ in how the per-sample control flow is written and "
             f"run. Numbers below are at N = {n:,}"
             + (f"; the fastest arm there is {ARM_LABELS[fastest['arm']]} "
                f"({_fmt_time(fastest['wall_s']['median'])})." if fastest else "."))
    notes = {
        "eagle_graph": (
            f"the attempt is per-sample scalar code ({lines['eagle_graph']} lines with "
            f"its launch; the kernel finishes its own samples), one kernel per attempt looped "
            f"on the device: {w('eagle_graph')}."),
        "eagle_graph_compact": (
            f"the same kernel launched over the live samples only: {w('eagle_graph_compact')}; "
            f"the plain graph takes {ratio('eagle_graph', 'eagle_graph_compact')} as long."),
        "eagle_graph_reorder": (
            f"adds the occasional reorder of the planes: {w('eagle_graph_reorder')}; "
            f"the compacted graph takes {ratio('eagle_graph_compact', 'eagle_graph_reorder')} "
            f"as long."),
        "eagle_graph_persist": (
            f"the plain (not active-set) attempt kernel, forced through hawk's "
            f"persist entry: one launch, no graph, no map, no policy kernel -- "
            f"a grid of SMs blocks of 256 lanes steals the next unfinished "
            f"sample off a counter when their own finishes: {w('eagle_graph_persist')}; "
            f"the eagle graph takes "
            f"{ratio('eagle_graph', 'eagle_graph_persist')} as long."),
        "eagle_simulate": (
            f"the same kernel through eagle.simulate, its state and parameters by name "
            f"({lines['eagle_simulate']} lines with the kernel): {w('eagle_simulate')}; "
            f"the eagle graph takes {ratio('eagle_graph', 'eagle_simulate')} as long."),
        "cupy_masked": (
            f"array code in the NumPy style ({lines['cupy_masked']} lines), the "
            f"control flow as masks over the batch: {w('cupy_masked')}."),
        "torch_masked": (
            f"the same array formulation in PyTorch ({lines['torch_masked']} lines), "
            f"running where a PyTorch model already runs: {w('torch_masked')}."),
        "torch_graphed": (
            f"records {card['workload']['graph_block']} attempts as a CUDA graph and "
            f"replays them, removing per-kernel launch cost: {w('torch_graphed')}; "
            f"eager PyTorch takes {ratio('torch_masked', 'torch_graphed')} as long."),
        "jax_vmap": (
            f"writes the per-sample loop directly ({lines['jax_vmap']} lines) and "
            f"compiles the whole batch into one executable: {w('jax_vmap')}."),
        "warp_kernel": (
            f"an explicit per-thread loop in a Python-embedded kernel language "
            f"({lines['warp_kernel']} lines, its own stages): {w('warp_kernel')}."),
        "eagle_cpu": (
            f"the same hawk kernel on the host's cores, no GPU needed: {w('eagle_cpu')}."),
    }
    return {"intro": intro, "arms": {a: notes[a] for a in card["arms"]}}


def _dev_cell(m):
    if m is None or m["device_peak_bytes"] is None:
        return "–"
    return (f"{_fmt_mib(m['device_peak_bytes'])} "
            f"({m['device_peak_bytes'] / m['minimum_bytes']:.3g}× min.)")


def render_markdown(card: dict) -> str:
    """The card's tables, as MyST Markdown; deterministic in ``card`` alone."""
    dev = card["device"]
    wl = card["workload"]
    lines = [
        f"<!-- Generated by {SCRIPT_REL} from card_{card['device_slug']}.json; "
        f"do not edit. -->",
        "",
        f"**Device:** {dev['name']} (compute capability {dev['compute_capability']}, "
        f"{dev['sm_count']} SMs). **Peak FP64:** {_fmt_rate(card['peak']['fp64_flops'])} "
        f"FLOP/s. Medians of {card['method']['repetitions']} interleaved repetitions"
        + (f" ({card['method']['slow_arm_reps']} for an arm whose warm-up says "
           f"{card['method']['repetitions']} would take over "
           f"{card['method']['slow_arm_budget_s']:g} s)"
           if 'slow_arm_reps' in card['method'] else "")
        + "; IQR in parentheses.",
        "",
        f"**Workload:** {wl['problem']} Tolerance rtol = atol = {wl['rtol']:g}; "
        f"final time {wl['t_final']:g} ({wl['t_final'] / (2 * math.pi):.3g} periods). "
        f"{wl['flops_per_attempt']} FLOP per attempt.",
        "",
        "| arm | loop structure | lines of code |",
        "|---|---|---:|",
    ]
    for arm in card["arms"]:
        lines.append(f"| {ARM_LABELS[arm]} | {ARM_LOOP[arm]} "
                     f"| {card['code_lines']['arms'][arm]} |")
    lines += ["", f"Lines of code: {card['code_lines']['rule']} "
              f"({card['code_lines']['shared_scheme_lines']} lines).", ""]
    mem = {(r["n"], r["arm"]): r for r in (card.get("memory") or {}).get("rows", [])}
    lines += [
        "| N | arm | wall (IQR) | kernel-only | end-to-end | attempts·samples/s "
        "| useful FLOP/s (% peak) | device memory (× minimum) | host memory |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    peak = card["peak"]["fp64_flops"]
    for row in sorted(card["results"], key=lambda r: (r["n"], ARMS.index(r["arm"]))):
        m = mem.get((row["n"], row["arm"]))
        pct = (None if row["arm"] == "eagle_cpu" or not peak
               else row["useful_flops_per_s"] / peak)
        lines.append(
            f"| {row['n']:,} | {ARM_LABELS[row['arm']]} "
            f"| {_fmt_time(row['wall_s']['median'])} ({_fmt_time(row['wall_s']['iqr'])}) "
            f"| {_fmt_time(row['kernel_only_s'])} "
            f"| {_fmt_time(row['end_to_end_s']['median'])} "
            f"| {_fmt_rate(row['attempt_samples_per_s'])} "
            f"| {_fmt_rate(row['useful_flops_per_s'])} ({_fmt_pct(pct)}) "
            f"| {_dev_cell(m)} "
            f"| {_fmt_mib(m['host_peak_bytes']) if m else '–'} |")
    lines.append("")
    lines += ["### Agreement with the analytic orbit", "",
              card["method"]["verification"], "",
              "| N | arm | max position error | max error / bound "
              "| samples with different step counts | max Δ accepted / rejected |",
              "|---:|---|---:|---:|---:|---:|"]
    for extra in card["per_n"]:
        for arm in card["arms"]:
            a = extra["agreement"].get(arm)
            if a is None:
                continue
            lines.append(
                f"| {extra['n']:,} | {ARM_LABELS[arm]} "
                f"| {a['max_position_error_vs_truth']:.2e} "
                f"| {a['max_error_over_bound']:.2e} "
                f"| {a['samples_with_step_counts_differing_from_reference']:,} "
                f"| {a['max_abs_accepted_difference']} / "
                f"{a['max_abs_rejected_difference']} |")
    lines.append("")
    top = max(card["per_n"], key=lambda e: e["n"])
    st = top["steps"]
    lines += [f"Steps per sample at N = {top['n']:,} (reference arm): accepted "
              f"min {st['accepted']['min']:.0f} / median {st['accepted']['p50']:.0f} / "
              f"p90 {st['accepted']['p90']:.0f} / max {st['accepted']['max']:.0f}; "
              f"rejected min {st['rejected']['min']:.0f} / median "
              f"{st['rejected']['p50']:.0f} / max {st['rejected']['max']:.0f}.", ""]
    lines += cc.fit_section(card, ARM_LABELS)
    lines += cc.compile_table(card.get("compile_time") or {}, ARM_LABELS, _fmt_time,
                              " (PyTorch's eager kernels ship prebuilt; the CUDA graph "
                              "is recorded, not compiled).")
    if (card.get("memory") or {}).get("rows"):
        lines += ["### Memory method", "", card["memory"]["method"], ""]
    if card.get("torch_compile_note"):
        lines += [card["torch_compile_note"], ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _torch_compile_note():
    import torch

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) >= (7, 0):
        return None
    return (f"torch.compile is not run: its GPU code generator (Inductor, through "
            f"Triton) needs compute capability 7.0 or newer, and this GPU is "
            f"{major}.{minor}. The PyTorch + CUDA graph arm removes the same "
            f"per-kernel launch cost without generating code.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-nsys", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="build, run every arm once per N, print agreement, no card")
    ap.add_argument("--kernel-pass", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--memory-probe", nargs=3, help=argparse.SUPPRESS)
    ap.add_argument("--compile-probe", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.memory_probe:
        a, n, bus = args.memory_probe
        _memory_probe(a, int(n), bus)
        return 0
    if args.compile_probe:
        _compile_probe(args.compile_probe)
        return 0
    if args.reps < 5 and not (args.check or args.kernel_pass):
        ap.error("--reps must be >= 5")
    if args.quick:
        args.ns = [n for n in args.ns if n <= 10_000]
    names = tuple(a for a in ARMS if a in args.arms)

    work = pathlib.Path(tempfile.mkdtemp(prefix="rk78_card_"))
    device, peak, slug, bus_id = _device_facts()

    if args.kernel_pass:
        runs = (KERNEL_PASS_RUNS_MASKED if set(names) <= set(MASKED_ARMS)
                else KERNEL_PASS_RUNS)
        for n in args.ns:
            _run_n(n, runs, bus_id, names, kernel_pass=True)
        shutil.rmtree(work, ignore_errors=True)
        return 0
    if args.check:
        for n in args.ns:
            inp = _inputs(n)
            arms = _make_arms(n, names)
            outs = {}
            for name, arm in arms.items():
                w, _, outs[name] = _timed(name, arm, inp)
                print(f"N={n:>8,} {name:20s} {w:9.4f}s", flush=True)
            report, steps = _verify(outs, inp, names[0])
            print(json.dumps({"steps": steps, "agreement": report}, indent=1), flush=True)
        shutil.rmtree(work, ignore_errors=True)
        return 0

    _check_kernel_bodies()
    sources = _kernel_sources(work)
    results, per_n = [], []
    for n in args.ns:
        t0 = time.perf_counter()
        rows, extra = _run_n(n, args.reps, bus_id, names)
        results += rows
        per_n.append(extra)
        best = min(rows, key=lambda r: r["wall_s"]["median"])
        print(f"N={n:>8,} fastest={best['arm']} ({time.perf_counter() - t0:.1f}s)",
              flush=True)
    ktimes, kcounts, kernel_note = None, None, "skipped (--no-nsys)"
    if not args.no_nsys:
        ktimes, kcounts, kernel_note = _nsys_kernel_times(args.ns, work, names)
        for row in results:
            if row["arm"] != "eagle_cpu" and ktimes is not None:
                row["kernel_only_s"] = ktimes.get((row["n"], row["arm"]))
    for row in results:
        row["fraction_of_peak_fp64"] = (
            None if row["arm"] == "eagle_cpu" or not peak["fp64_flops"]
            else row["useful_flops_per_s"] / peak["fp64_flops"])
        # Item 5: memcpy/runtime-API-call counts from the same nsys pass.
        row["nsys_counts"] = (kcounts.get((row["n"], row["arm"]))
                              if row["arm"] != "eagle_cpu" and kcounts is not None
                              else None)
        # Item 4: lane utilisation -- set already, per-row, from the persist
        # arm's own report (the persistent launch); every
        # other arm's row has none, so this only fills the gap.
        row.setdefault("lane_utilisation", None)
    # Item 1: the SM clock the repetitions ran at (ClockPoller: NVML where
    # available, else nvidia-smi) and the FP64 peak at that clock.
    clocks = [e["gpu_state_during"].get("sm_clock_mhz_median") for e in per_n]
    clock_sources = {e["gpu_state_during"].get("source") for e in per_n
                     if e["gpu_state_during"].get("source")}
    clocks = [c for c in clocks if c is not None]
    if clocks and peak["fp64_flops"]:
        observed_hz = statistics.median(clocks) * 1e6
        peak["observed_sm_clock_hz"] = observed_hz
        peak["fp64_flops_at_observed_sm_clock"] = (
            peak["fp64_flops"] * observed_hz / device["clock_rate_hz"])
        peak["observed_clock_source"] = (
            "median over N of the median per-repetition clock reading ("
            + "; ".join(sorted(clock_sources)) + ")")
        for row in results:
            row["fraction_of_peak_fp64_at_observed_clock"] = (
                None if row["fraction_of_peak_fp64"] is None
                else row["useful_flops_per_s"] / peak["fp64_flops_at_observed_sm_clock"])
    memory_rows = [] if args.no_memory else _run_memory_pass(args.ns, bus_id, names)
    compile_rows = [] if args.no_compile else _run_compile_pass(args.ns)
    card = {
        "schema": "eagle-rk78-card/2",
        "toolchain": cc.toolchain(),
        "device_slug": slug,
        "script": SCRIPT_REL,
        "script_md5": cc.md5(__file__),
        "common_md5": cc.md5(cc.__file__),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device, "peak": peak, "software": _software(),
        "kernel_sources_md5": sources,
        "workload": {
            "problem": ("two-body Kepler orbits (mu = 1, float64) from periapsis, "
                        f"a = {SEMI_MAJOR:g}, e uniform in [{E_RANGE[0]:g}, "
                        f"{E_RANGE[1]:g}], random orientation; adaptive RKF7(8) "
                        "(Fehlberg's tableau, a mixed absolute/relative max-norm error "
                        "control and a standard step-size controller)."),
            "scheme_source": "E. Fehlberg, NASA Technical Report R-287 (1968): "
                             "the RKF7(8) pair",
            "n_stages": N_STAGES, "rtol": RTOL, "atol": ATOL, "t_final": T_FINAL,
            "h0": H0, "safety": SAFETY, "min_scale": MIN_SCALE, "max_scale": MAX_SCALE,
            "exponents": {"reject": EXP_REJECT, "accept": EXP_ACCEPT},
            "max_attempts": MAX_ATTEMPTS, "seed": SEED,
            "last_step": "clipped to land on the final time (a common alternative rejects "
                         "an overshooting attempt and retries it shortened)",
            "flops_per_attempt": FLOPS_PER_ATTEMPT,
            "compaction_every_attempts": COMPACT_EVERY, "reorder_theta": REORDER_THETA,
            "graph_block": GRAPH_BLOCK,
        },
        "ns": list(args.ns),
        "arms": {a: ARM_LABELS[a] for a in names},
        "code_lines": code_line_counts(),
        "method": {
            "repetitions": args.reps,
            "slow_arm_budget_s": SLOW_ARM_BUDGET_S, "slow_arm_reps": SLOW_ARM_REPS,
            "statistic": "median, interquartile range (inclusive quartiles)",
            "warm_up": "one run per arm, excluded; it is also the verification run",
            "interleaving": "the arm order rotates by one every repetition",
            "wall": "the integration alone, inputs already on the device, to the "
                    "device synchronised",
            "end_to_end": "inputs uploaded, integration, outputs downloaded",
            "verification": (
                "Every arm, every sample: final time reached exactly; position "
                "error against the analytic Kepler state (Newton on Kepler's "
                "equation to machine precision) below the stated bound n_acc tol "
                "(1 + v_p)(1 + 6 pi n_orbits v_p), derived from the controller's "
                "per-step budget and the Kepler along-track amplification (see "
                "truth_bound; not fitted). Step counts are compared with the "
                "reference arm (eagle graph): any difference comes from a "
                "floating-point difference flipping one accept/reject decision "
                "(contraction into fused multiply-adds, pow and sqrt "
                "implementations differ between compilers and libraries); the "
                "table reports how many samples differ and by how much."),
            "counting_rule": _COUNTING_RULE,
            "kernel_only": kernel_note,
            "cpu_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "results": results,
        "per_n": per_n,
        "memory": {"method": _MEMORY_METHOD, "rows": memory_rows},
        "compile_time": {"method": _COMPILE_METHOD, "rows": compile_rows,
                         "none": [a for a in NO_COMPILE_ARMS if a in names]},
        "torch_compile_note": _torch_compile_note(),
    }
    card["fit"] = _fit_notes(card)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"card_{slug}.json"
    json_path.write_text(json.dumps(card, indent=1) + "\n")
    (args.out_dir / f"card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {json_path}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _threads = int(os.environ.get("OMP_NUM_THREADS") or CPU_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(min(_threads, CPU_THREADS))
    sys.exit(main())
