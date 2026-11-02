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

    python benchmarks/perf_card/perf_card.py

``--quick`` runs N up to 1e4; ``--no-nsys``, ``--no-memory`` and
``--no-compile`` skip those passes (dry runs only).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))  # benchmarks/: the cards' shared helpers
import _card_common as cc  # noqa: E402

# JAX reserves 75 % of the GPU when it starts; the arms of a cell share the device, so it takes what it uses
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SCRIPT_REL = "benchmarks/perf_card/perf_card.py"

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
        "cupy_masked", "torch_masked", "torch_graphed", "torch_compiled", "jax_vmap",
        "warp_kernel", "eagle_cpu")
GPU_ARMS = ARMS[:-1]
#: The optional library each arm needs: an arm whose library is not installed
#: in this profile is listed under ``arms_not_run`` and the run continues.
ARM_LIBRARY = {"cupy_masked": "cupy", "torch_masked": "torch", "torch_graphed": "torch", "torch_compiled": "torch",
               "jax_vmap": "jax", "warp_kernel": "warp"}


def _lib_installed(lib):
    """Whether ``lib`` can be imported here, decided WITHOUT importing it
    (``find_spec``); a library blocked through ``sys.modules[lib] = None``
    counts as not installed."""
    try:
        return importlib.util.find_spec(lib) is not None
    except (ImportError, ValueError):
        return False


def _select_arms(requested):
    """(the requested arms this machine runs, in table order, {arm: reason}
    for each requested arm whose library is not installed)."""
    not_run = {a: f"{ARM_LIBRARY[a]} not installed in this profile"
               for a in ARMS if a in requested and a in ARM_LIBRARY
               and not _lib_installed(ARM_LIBRARY[a])}
    if "torch_compiled" in requested and "torch_compiled" not in not_run:
        why = _compiled_unavailable()
        if why:
            not_run["torch_compiled"] = why
    return tuple(a for a in ARMS if a in requested and a not in not_run), not_run
def _compiled_unavailable():
    """Why torch.compile's Triton backend cannot run here (None when it can)."""
    if not _lib_installed("triton"):
        return "triton not installed in this profile"
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 0):
        return "torch.compile's Triton backend needs a GPU of compute capability 7.0 or newer"
    return None


EAGLE_GPU_ARMS = ARMS[:5]
ARM_LABELS = {
    "torch_compiled": "PyTorch torch.compile (reduce-overhead)",
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
    "torch_compiled": "host loop: a block of 16 masked attempts compiled by Inductor "
                      "(Triton) and replayed as a CUDA graph, host check per block",
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
NS = (1_000_000, 10_000_000)
#: ``spread``: e uniform in E_RANGE (divergent per-sample work); ``uniform``:
#: every sample the same orbit, e = UNIFORM_E (no divergence).
DISTRIBUTIONS = ("spread", "uniform", "spread-short")  # the cells' distributions
UNIFORM_E = 0.45
#: The arms of the committed card; the others stay CLI arms (``--arms``).
COMMITTED_ARMS = ("eagle_graph", "eagle_graph_persist", "eagle_simulate", "cupy_masked",
                  "torch_graphed", "torch_compiled", "jax_vmap", "warp_kernel")
#: The tail floor: the same kernel on this many samples (launch + tail cost).
TAIL_FLOOR_N = 32
OVERHEAD_N = 64
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
    from hawk import Mutable, Param, Scalar, Terminated, UInt32, Vector
    from hawk import math as m

    # >>> code:hawk_kernel
    @hawk.kernel
    def rkf78_attempt(t_final: Param, terminated: Terminated, s: Mutable[Vector[6]],
                      t: Mutable[Scalar], h: Mutable[Scalar],
                      n_acc: Mutable[UInt32], n_rej: Mutable[UInt32]):
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
        n_acc = n_acc + m.uint32(m.where(accept, 1.0, 0.0))
        n_rej = n_rej + m.uint32(m.where(accept, 0.0, 1.0))
        terminated = t1 >= t_final  # the sample finishes itself
    # <<< code:hawk_kernel

    return rkf78_attempt


def _define_active_kernel():
    """The single attempt of :func:`_define_kernel` traced under the
    active-set kind: the kernel the compaction arms launch, one attempt per
    launch at their own ``every`` cadence."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated, UInt32, Vector
    from hawk import math as m
    from hawk.ext import Guard, Kind

    # >>> code:hawk_kernel_active
    ACTIVE_SET = Kind("active_set", guard=Guard(active_set=True))

    @hawk.kernel(steps=1, kind=ACTIVE_SET)
    def rkf78_attempt(t_final: Param, terminated: Terminated, s: Mutable[Vector[6]],
                      t: Mutable[Scalar], h: Mutable[Scalar],
                      n_acc: Mutable[UInt32], n_rej: Mutable[UInt32]):
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
        n_acc = n_acc + m.uint32(m.where(accept, 1.0, 0.0))
        n_rej = n_rej + m.uint32(m.where(accept, 0.0, 1.0))
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
#: The eagle arms' counting planes are 4-byte unsigned integers (every other
#: plane is float64); the other arms keep their counters in float64.
COUNTER_DTYPES = {"n_acc": "uint32", "n_rej": "uint32"}
#: Bytes per sample of the workload's minimum footprint: the float64 planes
#: (state, t, h) at 8 B and the two counters at 4 B each.
MIN_BYTES_PER_SAMPLE = (len(PLANE_NAMES) - len(COUNTER_DTYPES)) * 8 + len(COUNTER_DTYPES) * 4


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


def _inputs(n, distribution="spread"):
    """Initial states at periapsis (a = 1, e uniform in E_RANGE or, for
    ``uniform``, UNIFORM_E, random orientation) and the orbit elements the truth needs."""
    import numpy as np

    rng = np.random.default_rng([SEED, n])
    e = rng.uniform(E_RANGE[0], E_RANGE[1], n)
    if distribution == "uniform":
        e[:] = UNIFORM_E
    else:
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
        self.planes = {"s": cp.empty((6, n), dtype=SCALAR)}
        self.planes.update({name: cp.empty(n, dtype=COUNTER_DTYPES.get(name, SCALAR))
                            for name in PLANE_NAMES[6:]})
        self.terminated = cp.zeros(n, dtype=cp.bool_)
        self.runner = None

    def upload(self, inp):
        import numpy as np

        self.planes["s"].set(np.stack([inp[name] for name in STATE_NAMES]).astype(SCALAR))
        self.planes["t"].fill(0.0)
        self.planes["h"].fill(H0)
        self.planes["n_acc"].fill(0)
        self.planes["n_rej"].fill(0)
        if self.runner is not None:
            self.runner.reset()

    def download(self):
        import numpy as np

        s = self.planes["s"].get()
        out = {name: s[d].astype(np.float64) for d, name in enumerate(STATE_NAMES)}
        out.update({name: self.planes[name].get().astype(np.float64)
                    for name in PLANE_NAMES[6:]})  # the counters compared as float64
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
                                         "_slot_fill": True}}
    arms = {}
    for name in (a for a in EAGLE_GPU_ARMS if a in names):
        rkf78_attempt = ARM_KERNEL[name]()
        options = variants.get(name)
        st = _GpuArm(n)
        if name == "eagle_simulate":
            # >>> code:eagle_simulate
            st.runner = eagle.simulation(rkf78_attempt, t_final=T_FINAL, scalar_type=SCALAR,
                                         max_steps=MAX_ATTEMPTS, **st.planes)
            # <<< code:eagle_simulate
        else:
            # >>> code:hawk_launch
            st.runner = eagle.until_done(eagle.deploy(rkf78_attempt, scalar_type=SCALAR), max_steps=MAX_ATTEMPTS,
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


# >>> code:torch_compiled
def _compiled_block(s, t, h, n_acc, n_rej):
    """GRAPH_BLOCK masked attempts; compiled once, replayed as a CUDA graph."""
    for _ in range(GRAPH_BLOCK):
        s, t, h, n_acc, n_rej, _ = torch_attempt(s, t, h, n_acc, n_rej)
    return s, t, h, n_acc, n_rej


def torch_compiled_run(st, block_fn):
    import torch

    with torch.inference_mode():
        p = st.planes
        s, t, h = [p[k].clone() for k in STATE_NAMES], p["t"].clone(), p["h"].clone()
        n_acc, n_rej = p["n_acc"].clone(), p["n_rej"].clone()
        for _ in range(-(-MAX_ATTEMPTS // GRAPH_BLOCK)):
            torch.compiler.cudagraph_mark_step_begin()
            s, t, h, n_acc, n_rej = block_fn(s, t, h, n_acc, n_rej)
            s = [x.clone() for x in s]  # the graph's outputs are reused by the next replay
            t, h, n_acc, n_rej = t.clone(), h.clone(), n_acc.clone(), n_rej.clone()
            if not bool((t < T_FINAL).any()):
                break
    return s, t, h, n_acc, n_rej
# <<< code:torch_compiled


def _make_array_arms(n, names):
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
    if "torch_compiled" in names:
        import torch

        tc = _ArrayState("torch", n)
        tc.upload(_inputs(n))
        block_fn = torch.compile(_compiled_block, mode="reduce-overhead")

        def run_compiled():
            s, t, h, a, r = torch_compiled_run(tc, block_fn)
            tc.out = dict(zip(PLANE_NAMES, (*s, t, h, a, r)))

        t0 = time.perf_counter()
        run_compiled()  # the compile happens here, outside the timed span
        torch.cuda.synchronize()
        arms["torch_compiled"] = dict(state=tc, run=run_compiled, download=tc.download,
                                      compile_s=time.perf_counter() - t0)
    if "torch_graphed" in names:
        import torch

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
    f64 = wp.float32 if SCALAR == "float32" else wp.float64  # the cell's plane type
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
        real = wp.float32 if SCALAR == "float32" else wp.float64
        self.planes = {k: wp.empty(n, dtype=real, device="cuda:0") for k in PLANE_NAMES}

    def upload(self, inp):
        for k in STATE_NAMES:
            self.planes[k].assign(inp[k].astype(SCALAR))
        self.wp.synchronize_device("cuda:0")

    def download(self):
        return {k: v.numpy().astype("float64") for k, v in self.planes.items()}


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
        self.planes.update({k: np.empty(n, dtype=COUNTER_DTYPES.get(k, "float64"))
                            for k in PLANE_NAMES[6:]})
        self.terminated = np.zeros(n, dtype=bool)
        self.runner = None
        self.kernel_s = None  # the loop launches and checks in one call

    def upload(self, inp):
        import numpy as np

        np.copyto(self.planes["s"], np.stack([inp[k] for k in STATE_NAMES]))
        self.planes["t"].fill(0.0)
        self.planes["h"].fill(H0)
        self.planes["n_acc"].fill(0)
        self.planes["n_rej"].fill(0)
        self.terminated.fill(False)

    def run(self):
        import eagle

        # >>> code:eagle_cpu
        eagle.run_until_done(self.plan, max_steps=MAX_ATTEMPTS, t_final=T_FINAL,
                             terminated=self.terminated, **self.planes)
        # <<< code:eagle_cpu

    def download(self):
        import numpy as np

        out = {k: self.planes["s"][d].copy() for d, k in enumerate(STATE_NAMES)}
        out.update({k: self.planes[k].astype(np.float64) for k in PLANE_NAMES[6:]})
        return out


def _current_device():
    """This thread's current CUDA device: where eagle's unscoped work (the legacy stream) goes."""
    import cupy as cp

    return int(cp.cuda.runtime.getDevice())


def _use_device(device):
    """Make the arms' device current again: JAX start-up leaves the last GPU current, and the CuPy and torch arms
    run on the current device."""
    if device is not None and _current_device() != device:
        import cupy as cp

        cp.cuda.runtime.setDevice(device)


def _make_arms(n, names):
    arms = {}
    device = _current_device() if any(a in GPU_ARMS for a in names) else None
    if any(a in names for a in EAGLE_GPU_ARMS):
        arms.update(_make_eagle_arms(n, names))
    if any(a in names for a in ("cupy_masked", "torch_masked", "torch_graphed",
                                "torch_compiled")):
        arms.update(_make_array_arms(n, names))
        _use_device(device)
    if "jax_vmap" in names:
        arms["jax_vmap"] = _make_jax_arm(n)
        _use_device(device)
    if "warp_kernel" in names:
        arms["warp_kernel"] = _make_warp_arm(n)
        _use_device(device)
    if "eagle_cpu" in names:
        import eagle

        rkf78_attempt = _define_kernel()
        # >>> code:eagle_cpu_deploy
        plan = eagle.deploy(rkf78_attempt)  # host planes select the host build
        # <<< code:eagle_cpu_deploy
        c = _CpuArm(plan, n)
        arms["eagle_cpu"] = dict(state=c, run=c.run, download=c.download)
    for arm in arms.values():
        arm["device"] = device
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
    _use_device(arm["device"])
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


# --------------------------------------------------------------------------- #
# Timing: one fresh process per (arm, N, distribution) on a clean device
# --------------------------------------------------------------------------- #
#: A warm-up run longer than this makes the arm ``not_run: budget`` at that N and larger.
ARM_RUN_BUDGET_S = 120.0
#: Timed rounds of one (arm, cell) stop at this many seconds (at least one round is always timed).
CELL_BUDGET_S = 300.0
ROUNDS_MIN, ROUNDS_MAX, IQR_STOP = 3, 9, 0.02
TIMING_PROBE_TIMEOUT_S = cc.BUILD_S + ARM_RUN_BUDGET_S + CELL_BUDGET_S + ARM_RUN_BUDGET_S   # build, warm-up, rounds, a last round
#: ``spread-short`` ends at T_FINAL / SHORT_DIVISOR: the short-lived-sample regime.
SHORT_DIVISOR = 16
T_FINAL_FULL = T_FINAL
CELLS = ((1_000_000, "spread"), (1_000_000, "uniform"), (10_000_000, "spread"),
         (10_000_000, "uniform"), (10_000_000, "spread-short"))
#: Test hook: ``EAGLE_CARD_FAULT=<oom|slow|abort>:<arm>[,...]`` makes that arm's timing child fail that way.
_FAULT_ENV = "EAGLE_CARD_FAULT"
_OOM_TEXT = _OUT_OF_MEMORY = ("out of memory", "OutOfMemory", "RESOURCE_EXHAUSTED", "cudaErrorMemoryAllocation",
                              "MemoryError")


#: The plane precision of this process; the ``fp32`` cell runs the spread inputs in float32 at this tolerance.
SCALAR = "float64"
FP32_TOL = 1e-6


def _set_cell(distribution):
    """Apply a cell's distribution to this process; returns the input distribution ('spread' or 'uniform')."""
    global T_FINAL, SCALAR, RTOL, ATOL
    T_FINAL = T_FINAL_FULL / SHORT_DIVISOR if distribution == "spread-short" else T_FINAL_FULL
    if distribution == "fp32":
        SCALAR, RTOL, ATOL = "float32", FP32_TOL, FP32_TOL
    return "uniform" if distribution == "uniform" else "spread"


def _fault(arm):
    for item in os.environ.get(_FAULT_ENV, "").split(","):
        kind, _, target = item.partition(":")
        if target == arm:
            return kind
    return None


def _warm_up(arm, a, inp, reply_fd, poller):
    """The warm-up run, stopped at the per-run budget: a timer replies ``not_run: budget`` and ends this process."""
    def stop():
        poller.stop()
        cc.probe_reply(reply_fd, {"status": "not_run: budget", "warm_s": ARM_RUN_BUDGET_S, "stopped": True})
        os._exit(0)

    timer = threading.Timer(ARM_RUN_BUDGET_S, stop)
    timer.daemon = True
    timer.start()
    try:
        return _timed(arm, a, inp)
    finally:
        timer.cancel()


def _time_probe(arm, n, distribution, bus_id, rounds_max, out_path):
    """Child body: build ``arm`` at N alone, warm up, time rounds 3..9 until IQR/median <= 2 %, reply with the walls,
    the agreement with the analytic truth and (in ``out_path``) the step counts the parent compares with the reference."""
    import numpy as np

    reply_fd = cc.probe_reply_fd()
    gpu = arm != "eagle_cpu"
    base = _set_cell(distribution)
    if gpu:
        _touch_device()
    inp = _inputs(n, base)
    poller = cc.ClockPoller(bus_id)
    try:
        if _fault(arm) == "abort":
            os.abort()
        if _fault(arm) == "oom":
            raise MemoryError("out of memory (forced by EAGLE_CARD_FAULT)")
        cc.heartbeat("build")
        a = _make_arms(n, (arm,))[arm]
        cc.heartbeat()
        warm, _, out = _warm_up(arm, a, inp, reply_fd, poller)
        cc.heartbeat()
    except BaseException as err:  # noqa: BLE001  (an allocation failure of any library is a row, not a crash)
        poller.stop()
        if any(k in f"{type(err).__name__} {err}" for k in _OOM_TEXT):
            cc.probe_reply(reply_fd, {"status": "out_of_memory"})
            return
        raise
    if _fault(arm) == "slow":
        warm = ARM_RUN_BUDGET_S + 1.0
    reply = {"build_s": a.get("build_s"), "compile_s": a.get("compile_s"), "warm_s": warm}
    if warm > ARM_RUN_BUDGET_S:
        poller.stop()
        cc.probe_reply(reply_fd, {**reply, "status": "not_run: budget"})
        return
    agreement, steps = _verify({arm: out}, inp, arm)
    np.savez(out_path, n_acc=out["n_acc"].astype(np.uint32), n_rej=out["n_rej"].astype(np.uint32))
    walls, e2e, kern, rule = [], [], [], "rounds_max"
    started = time.perf_counter()
    while len(walls) < rounds_max:
        poller.mark(arm)
        w, e, _ = _timed(arm, a, inp)
        cc.heartbeat()
        walls.append(w)
        e2e.append(e)
        if arm == "eagle_cpu" and a["state"].kernel_s is not None:
            kern.append(a["state"].kernel_s)
        if len(walls) >= min(ROUNDS_MIN, rounds_max):
            q1, med, q3 = np.percentile(walls, [25, 50, 75])
            if q3 - q1 <= IQR_STOP * med:
                rule = "iqr<=2%"
                break
        if time.perf_counter() - started > CELL_BUDGET_S:
            rule = "cell budget 300 s"
            break
    clocks = poller.stop()
    cc.probe_reply(reply_fd, {
        **reply, "status": "ok", "walls": walls, "e2e": e2e, "stop_rule": rule, "kernel_s": kern,
        "agreement": agreement[arm], "steps": steps, "gpu_state_during": clocks,
        "sm_clock_mhz": clocks.get("sm_clock_mhz_by_label", {}).get(arm),
        "slot_fill": getattr(a.get("report"), "slot_fill", None),
        "run_mode": getattr(a.get("report"), "mode", None),
        "reorders_fired": (a["fired"][-1] if a.get("fired") else None) if "fired" in a else None})


_ASSEMBLE = False   # --assemble: run nothing, write the card from the saved parts
# The order everything runs in, so a session deadline only ever drops the tail. Arms: eagle first (the claim), then the fast
# libraries, then the slow masked arms (cupy_masked, torch_graphed: minutes per run) last. Cells: the headline cell first
# (1e7 spread where the plan times 1e7, else 1e6 spread), the other cells after it, method cells (fp32, tail, overhead) last.
ARM_PRIORITY = ("eagle_graph_persist", "eagle_simulate", "eagle_graph", "eagle_graph_compact", "eagle_graph_reorder",
                "warp_kernel", "jax_vmap", "torch_compiled", "cupy_masked", "torch_graphed", "torch_masked", "eagle_cpu")


def _ordered_cells(cells):
    """``cells`` with the headline cell first, the rest in their order."""
    head = (10_000_000, "spread") if (10_000_000, "spread") in cells else (1_000_000, "spread")
    return sorted(cells, key=lambda c: c != head)


_PARTS = None   # --parts: every finished (arm, cell) is saved here and a re-run reuses it


def _time_cell_arm(arm, n, distribution, bus_id, rounds_max, work):
    """Parent: one timing child. A failure is a row: ``out_of_memory`` or ``failed: <last error line>``.
    With ``--parts`` a finished result is saved, and an earlier one is reused."""
    key = f"timing_{arm}_{n}_{distribution}"
    out_path = work / f"counts_{arm}_{n}_{distribution}.npz"
    saved = cc.part_load(_PARTS, key)
    if saved is not None:
        if (_PARTS / f"{key}.npz").exists():
            shutil.copy(_PARTS / f"{key}.npz", out_path)
        return {**saved, "counts_path": str(out_path)}
    if _ASSEMBLE or cc.deadline_passed():
        return _stub(arm, n, distribution, cc.SESSION_LIMIT, work)
    try:
        r = cc.run_probe(__file__, ["--timing-probe", arm, str(n), distribution, bus_id, str(rounds_max),
                                    str(out_path)], timeout=TIMING_PROBE_TIMEOUT_S,
                         what=f"timing probe {arm} {n} {distribution}", progress=out_path.with_suffix(".alive"))
    except subprocess.TimeoutExpired:
        r = {"status": f"failed: no result in {TIMING_PROBE_TIMEOUT_S:.0f} s"}
    except cc.Stalled:
        r = {"status": "failed: no progress"}
    except (RuntimeError, ValueError, IndexError) as err:
        text = str(err)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        r = {"status": "out_of_memory" if any(k in text for k in _OOM_TEXT)
             else "failed: " + (lines[-1][:300] if len(lines) > 1 else "the process ended without a result")}
    r.update(n=n, distribution=distribution, arm=arm, counts_path=str(out_path))
    if _PARTS is not None:
        if out_path.exists():
            _PARTS.mkdir(parents=True, exist_ok=True)
            shutil.copy(out_path, _PARTS / f"{key}.npz")
        cc.part_save(_PARTS, key, {k: v for k, v in r.items() if k != "counts_path"})
    return r


def _predicted_over_budget(walls, n):
    """The smaller N whose measured per-run wall, scaled linearly to ``n``, is over the per-run budget (else None):
    the arm is not warmed up at ``n``."""
    for small, wall in sorted(walls.items(), reverse=True):
        if small < n and wall * n / small > ARM_RUN_BUDGET_S:
            return small
    return None


def _row_of(r, fma_dev, ops_per_attempt, samples_per_lane):
    """One result row from a timing reply; a non-``ok`` reply is a row with its status and the rule that applied."""
    row = {"n": r["n"], "distribution": r["distribution"], "arm": r["arm"], "status": r["status"]}
    if r["status"] != "ok":
        budget = (f"> {ARM_RUN_BUDGET_S:g} s per run, predicted from {r['predicted_dist']} N={r['predicted_from']:,}"
                  if r.get("predicted_from") else
                  f"> {ARM_RUN_BUDGET_S:g} s per run (warm-up stopped at {ARM_RUN_BUDGET_S:g} s)" if r.get("stopped")
                  else f"> {ARM_RUN_BUDGET_S:g} s per run (warm-up {r.get('warm_s', 0):.0f} s)")
        row["rule"] = {"not_run: budget": budget}.get(r["status"], r["status"])
        return row
    wall = _median_iqr(r["walls"])
    attempts = r["agreement"]["attempts_total"]
    rate = attempts / wall["median"]
    row.update(
        wall_s=wall, rounds=len(r["walls"]), rule=r["stop_rule"],
        sm_clock_mhz=r.get("sm_clock_mhz"), end_to_end_s=_median_iqr(r["e2e"]),
        attempts=attempts, attempt_samples_per_s=rate,
        useful_flops_per_s=FLOPS_PER_ATTEMPT * attempts / wall["median"],
        **(cc.derive_eff_sol(rate, fma_dev["float32" if r["distribution"] == "fp32" else "float64"], ops_per_attempt)
           if fma_dev and r["arm"] != "eagle_cpu" else {"eff_sol": None, "eff_sol_note": None}),
        samples_per_lane=samples_per_lane, build_s=r.get("build_s"), compile_s=r.get("compile_s"),
        kernel_only_s=(statistics.median(r["kernel_s"]) if r.get("kernel_s") else None),
        slot_fill=r.get("slot_fill"), run_mode=r.get("run_mode"))
    if r.get("reorders_fired") is not None:
        row["reorders_fired"] = r["reorders_fired"]
    return row


def _compare_with_reference(replies):
    """The agreement of every ok reply with the truth (from the child) and the reference arm (step counts, here)."""
    import numpy as np

    ok = {r["arm"]: r for r in replies if r["status"] == "ok"}
    if not ok:
        return None, "none", {}
    ref = "eagle_graph" if "eagle_graph" in ok else next(iter(ok))
    with np.load(ok[ref]["counts_path"]) as z:
        ref_acc, ref_rej = z["n_acc"].astype(np.int64), z["n_rej"].astype(np.int64)
    agreement = {}
    for arm, r in ok.items():
        with np.load(r["counts_path"]) as z:
            d_acc, d_rej = z["n_acc"].astype(np.int64) - ref_acc, z["n_rej"].astype(np.int64) - ref_rej
        agreement[arm] = {**r["agreement"],
                          "samples_with_step_counts_differing_from_reference": int(((d_acc != 0) | (d_rej != 0)).sum()),
                          "max_abs_accepted_difference": int(np.abs(d_acc).max()),
                          "max_abs_rejected_difference": int(np.abs(d_rej).max())}
    for r in replies:
        pathlib.Path(r["counts_path"]).unlink(missing_ok=True)
    return agreement, ref, ok[ref]["steps"]


#: Register-only DFMA chains: the device's FP64 instruction rate, the speed of light eff_sol is a fraction of.
_FMA_PROBE_SRC = r'''
extern "C" __global__ void fma_chain(double* out, int iters) {
  double a0 = threadIdx.x * 1e-3 + 1.0, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const double b = 1.0000001, c = 1e-9;
  for (int i = 0; i < iters; ++i) {
    a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
    a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}'''
FMA_PROBE_ITERS = 20000
FMA_PER_ITER = 8


def _fma_probe(sm_count, dtype="float64"):
    """Thread-level FMA instructions per second the device sustains in ``dtype`` (best of three launches)."""
    import cupy as cp

    src = _FMA_PROBE_SRC
    if dtype == "float32":
        src = src.replace("double", "float").replace("1.0000001", "1.0000001f").replace("1e-9", "1e-9f")
    kernel = cp.RawKernel(src, "fma_chain")
    blocks, threads = sm_count * 8, 256
    out = cp.empty(blocks * threads, dtype=dtype)
    kernel((blocks,), (threads,), (out, FMA_PROBE_ITERS))
    cp.cuda.runtime.deviceSynchronize()
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        kernel((blocks,), (threads,), (out, FMA_PROBE_ITERS))
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return blocks * threads * FMA_PROBE_ITERS * FMA_PER_ITER / best


def _ops_per_attempt(work, arch):
    """(the workload constant of ``sol_reference.json``, the audit: DP instructions counted in the fused attempt's PTX)."""
    const = json.loads((HERE.parent / "sol_reference.json").read_text()).get("ops_per_attempt")
    audit = {}
    try:
        from hawk.artifact import build_bundle

        d = work / "ops_audit"
        build_bundle([_define_kernel()], d, targets=("cuda",))
        for ptx in sorted(d.glob("*.ptx"))[:1]:
            for a in sorted({arch, "sm_80"}):
                got = cc.sass_dp_instruction_count(ptx, a)
                audit[a] = None if got is None else got["count"]
    except Exception as err:  # noqa: BLE001  (the audit is a record; its absence is recorded)
        audit["error"] = f"{type(err).__name__}: {err}"[:200]
    return const, audit


# --------------------------------------------------------------------------- #
# Memory pass (fresh process per (arm, N); NVML per-process device memory)
# --------------------------------------------------------------------------- #
_MEMORY_METHOD = (
    "A separate pass after the timing pass. Each (arm, N) runs in a fresh "
    "process: imports, the CUDA context created by one tiny device operation, "
    "then the baseline read, before the arm builds or allocates anything; "
    "then the arm's kernels built and a warm-up run on a "
    f"{WARM_N}-sample batch (every compile happens here; JAX also compiles for "
    "shape N), so every library's pool reserved during the warm-up counts in "
    "the reading; CuPy's pools and PyTorch's cache trimmed after the warm-up; "
    "then the arm built for N (eagle's graph, "
    "PyTorch's captured graph) and one full run with upload and download, "
    "the device read every 2 ms throughout (the masked arms hand their "
    "temporaries back after each run, so a reading after it would miss them). "
    "Device: the most memory NVML accounted to the process on the timed GPU "
    "during the build and the run, minus the baseline. Caveats: each allocator's rounding and growth policy count (JAX's "
    "allocator grows in regions that can exceed the request; PyTorch rounds "
    "blocks up and a captured CUDA graph keeps a private pool; Warp's "
    "stream-ordered pool reserves in chunks of tens of MiB, its release "
    "threshold set to 0 after the warm-up); the driver accounts in pages of "
    "about 2 MiB, so small-N rows read 0 or one page. Host: the peak resident "
    "set (VmHWM, reset to the current RSS at the host baseline, taken after the "
    "warm-up) minus the baseline RSS. Minimum: the planes the workload needs, "
    "N x (8 float64 planes (state, time, step) x 8 bytes + 2 counters x 4 "
    "bytes) = N x 72 bytes.")


def _probe_arm(arm, n, compiled=None):
    """One arm for a probe process (an eagle arm deploys its kernel here)."""
    if arm == "jax_vmap":
        device = _current_device()
        built = _make_jax_arm(n, compiled)
        _use_device(device)
        built["device"] = device
        return built
    return _make_arms(n, (arm,))[arm]


def _touch_device():
    """Create the CUDA context with one tiny device operation, so a baseline
    read right after it sees the context and nothing else."""
    import cupy as cp

    cp.zeros(1, dtype=cp.uint8).sum()
    cp.cuda.Device().synchronize()


def _memory_probe(arm, n, bus_id):

    reply_fd = cc.probe_reply_fd()
    gpu = arm != "eagle_cpu"
    # the device baseline: the CUDA context exists (one tiny device op), nothing
    # of the arm is built or allocated yet, so whatever any library's pool
    # reserves during the warm-up counts in the reading
    if gpu:
        _touch_device()
    dev_base, src = cc.process_device_bytes(bus_id) if gpu else (None, None)
    a = _probe_arm(arm, WARM_N)
    _timed(arm, a, _inputs(WARM_N))
    compiled = _jax_compile(n) if arm == "jax_vmap" else None
    del a
    gc.collect()
    cc.trim_pools()
    if gpu:
        _sync()
    rss_base = cc.status_kib("VmRSS")
    cc.reset_host_peak()
    inp = _inputs(n)
    with contextlib.nullcontext() if not gpu else cc.DevicePeak(bus_id) as peak:
        a = _probe_arm(arm, n, compiled)
        _, _, out = _timed(arm, a, inp)
    assert (out["t"] == T_FINAL).all(), f"memory probe {arm}: unfinished samples"
    dev_after, _ = cc.process_device_bytes(bus_id) if gpu else (None, None)
    cc.probe_reply(reply_fd, {
        "device_baseline_bytes": dev_base, "device_after_bytes": dev_after,
        "device_peak_bytes": None if not gpu else max(0, max(peak.peak, dev_after) - dev_base),
        "device_source": src,
        "host_peak_bytes": max(0, cc.status_kib("VmHWM") - rss_base) * 1024})


#: A memory probe runs the arm once at N; one that takes longer is reported as not measured.
MEMORY_PROBE_TIMEOUT_S = 900
#: How each library reports a device allocation it could not make.
_OUT_OF_MEMORY = ("out of memory", "OutOfMemory", "RESOURCE_EXHAUSTED", "cudaErrorMemoryAllocation")


def _run_memory_pass(ns, bus_id, names):
    """One fresh process per (arm, N); an arm that does not fit the device is a row too ("out of memory")."""
    rows = []
    for n in ns:
        for arm in names:
            saved = cc.part_load(_PARTS, f"memory_{arm}_{n}")
            if saved is not None:
                rows.append(saved)
                continue
            if _ASSEMBLE or cc.deadline_passed():
                rows.append({"n": n, "arm": arm, "status": cc.SESSION_LIMIT, "device_peak_bytes": None, "host_peak_bytes": None,
                             "minimum_bytes": n * MIN_BYTES_PER_SAMPLE})
                continue
            try:
                r = cc.run_probe(__file__, ["--memory-probe", arm, str(n), bus_id],
                                 timeout=MEMORY_PROBE_TIMEOUT_S, what=f"memory probe {arm} {n}")
            except subprocess.TimeoutExpired:
                r = {"timed_out": True, "device_peak_bytes": None, "host_peak_bytes": None}
            except RuntimeError as err:
                if not any(k in str(err) for k in _OUT_OF_MEMORY):
                    raise
                r = {"out_of_memory": True, "device_peak_bytes": None, "host_peak_bytes": None}
            r.update(n=n, arm=arm, minimum_bytes=n * MIN_BYTES_PER_SAMPLE)
            cc.part_save(_PARTS, f"memory_{arm}_{n}", r)
            rows.append(r)
            print(f"memory {arm:20s} N={n:>8,} device "
                  + _dev_cell(r),
                  flush=True)
    return rows


ARM_CODE = {
    "eagle_graph": ("hawk_kernel", "hawk_launch"),
    "eagle_graph_compact": ("hawk_kernel_active", "hawk_launch"),
    "eagle_graph_reorder": ("hawk_kernel_active", "hawk_launch"),
    "eagle_graph_persist": ("hawk_kernel", "hawk_launch"),
    "eagle_simulate": ("hawk_kernel", "eagle_simulate"),
    "cupy_masked": ("cupy_masked",),
    "torch_masked": ("torch_masked",),
    "torch_graphed": ("torch_masked", "torch_graphed"),
    "torch_compiled": ("torch_masked", "torch_compiled"),
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
        "torch_compiled": (
            f"compiles a block of {card['workload']['graph_block']} masked attempts with "
            f"torch.compile (reduce-overhead): {w('torch_compiled')}; eager PyTorch takes "
            f"{ratio('torch_masked', 'torch_compiled')} as long."),
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
    if m is not None and m.get("status") == cc.SESSION_LIMIT:
        return cc.SESSION_LIMIT
    if m is not None and m.get("out_of_memory"):
        return "out of memory"
    if m is not None and m.get("timed_out"):
        return f"not measured (one run > {MEMORY_PROBE_TIMEOUT_S // 60} min)"
    if m is None or m["device_peak_bytes"] is None:
        return "–"
    return (f"{_fmt_mib(m['device_peak_bytes'])} "
            f"({m['device_peak_bytes'] / m['minimum_bytes']:.3g}× min.)")


def _fmt_mhz(mhz):
    """A row's median SM clock; '–' for a row too short to sample (or a card from before rows had one)."""
    return "–" if mhz is None else f"{mhz:.0f} MHz"


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
        f"FLOP/s. Medians of {min(card['method']['rounds']['min'], card['method']['rounds']['max'])} to {card['method']['rounds']['max']} rounds "
        f"per arm, each arm in its own process (rounds stop at IQR/median <= 2 %); IQR in parentheses.",
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
    for arm, why in (card.get("arms_not_run") or {}).items():
        lines += ["", f"{ARM_LABELS[arm]}: {why}."]
    lines += ["", f"Lines of code: {card['code_lines']['rule']} "
              f"({card['code_lines']['shared_scheme_lines']} lines).", ""]
    mem = {(r["n"], r["arm"]): r for r in (card.get("memory") or {}).get("rows", [])}
    lines += [
        "| N | distribution | arm | wall (IQR) | rounds | end-to-end | attempts·samples/s "
        "| useful FLOP/s (% peak) | eff_sol | SM clock | device memory (× minimum) | host memory |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    peak = card["peak"]["fp64_flops"]
    for row in sorted(card["results"], key=lambda r: (r["n"], r.get("distribution", ""), ARMS.index(r["arm"]))):
        m = mem.get((row["n"], row["arm"]))
        if row["status"] != "ok":
            lines.append(f"| {row['n']:,} | {row.get('distribution', 'spread')} | {ARM_LABELS[row['arm']]} "
                         f"| {row['status'].replace('_', ' ')} ({row['rule']}) | – | – | – | – | – | – "
                         f"| {_dev_cell(m)} | – |")
            continue
        pct = (None if row["arm"] == "eagle_cpu" or not peak
               else row["useful_flops_per_s"] / peak)
        lines.append(
            f"| {row['n']:,} | {row.get('distribution', 'spread')} | {ARM_LABELS[row['arm']]} "
            f"| {_fmt_time(row['wall_s']['median'])} ({_fmt_time(row['wall_s']['iqr'])}) "
            f"| {row['rounds']} "
            f"| {_fmt_time(row['end_to_end_s']['median'])} "
            f"| {_fmt_rate(row['attempt_samples_per_s'])} "
            f"| {_fmt_rate(row['useful_flops_per_s'])} ({_fmt_pct(pct)}) "
            f"| {row.get('eff_sol_note') or ('–' if row['eff_sol'] is None else format(row['eff_sol'], '.3f'))} "
            f"| {_fmt_mhz(row.get('sm_clock_mhz'))} "
            f"| {_dev_cell(m)} "
            f"| {_fmt_mib(m['host_peak_bytes']) if m and m['host_peak_bytes'] is not None else '–'} |")
    lines.append("")
    untimed = sorted({n for n, _ in mem} - {r["n"] for r in card["results"]})
    lines += [f"Device FP64 FMA rate (probe): {_fmt_rate(card['device_fma_per_s'])} instructions/s; "
              f"ops per attempt {card['workload']['ops_per_attempt']}.", ""] if card.get("device_fma_per_s") else []
    if untimed:  # memory-only sizes: too long to time on this device, but whether an arm fits is the point
        lines += ["### Device memory at sizes not timed here", "",
                  "| N | " + " | ".join(ARM_LABELS[a] for a in card["arms"] if a != "eagle_cpu") + " |",
                  "|---:|" + "---:|" * sum(a != "eagle_cpu" for a in card["arms"])]
        for n in untimed:
            lines.append(f"| {n:,} | " + " | ".join(_dev_cell(mem.get((n, a))) for a in card["arms"]
                                                     if a != "eagle_cpu") + " |")
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
                f"| {extra['n']:,} ({extra.get('distribution', 'spread')}) | {ARM_LABELS[arm]} "
                f"| {a['max_position_error_vs_truth']:.2e} "
                f"| {a['max_error_over_bound']:.2e} "
                f"| {a['samples_with_step_counts_differing_from_reference']:,} "
                f"| {a['max_abs_accepted_difference']} / "
                f"{a['max_abs_rejected_difference']} |")
    lines.append("")
    seen = [e for e in card["per_n"] if e["steps"]]
    top = max(seen, key=lambda e: (e["n"], e.get("distribution") == "spread")) if seen else None
    st = top["steps"] if top else None
    lines += [] if top is None else [f"Steps per sample at N = {top['n']:,} (reference arm): accepted "
              f"min {st['accepted']['min']:.0f} / median {st['accepted']['p50']:.0f} / "
              f"p90 {st['accepted']['p90']:.0f} / max {st['accepted']['max']:.0f}; "
              f"rejected min {st['rejected']['min']:.0f} / median "
              f"{st['rejected']['p50']:.0f} / max {st['rejected']['max']:.0f}.", ""]
    lines += cc.fit_section(card, ARM_LABELS)
    if (card.get("memory") or {}).get("rows"):
        lines += ["### Memory method", "", card["memory"]["method"], ""]
    if card.get("torch_compile_note"):
        lines += [card["torch_compile_note"], ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _stub(arm, n, dist, status, work, warm_s=None):
    return {"arm": arm, "n": n, "distribution": dist, "status": status, "warm_s": warm_s or 0.0,
            "counts_path": str(work / "missing.npz")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--reps", type=int, default=ROUNDS_MAX,
                    help=f"most timed rounds per (arm, cell); rounds stop at IQR/median <= {IQR_STOP:.0%} (default {ROUNDS_MAX})")
    ap.add_argument("--ns", type=int, nargs="+", help="sizes (default: the committed cells)")
    ap.add_argument("--arms", nargs="+", default=list(COMMITTED_ARMS))
    ap.add_argument("--distributions", nargs="+", choices=DISTRIBUTIONS,
                    help="with --ns: the distributions of each size (default: spread, uniform)")
    ap.add_argument("--smoke", action="store_true",
                    help="smallest N, one round, no memory pass, card written to a temporary directory")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--assemble", action="store_true",
                    help="run nothing: write the card from the saved parts (--parts), a missing arm reads not_run: step time limit")
    ap.add_argument("--parts", type=pathlib.Path,
                    help="save each finished (arm, cell) here and reuse what is there: a cap or a preemption costs one arm")
    ap.add_argument("--memory-ns", type=int, nargs="+",
                    help="sizes of the memory pass (default: the cells' sizes); a size too long to time can still show what fits")
    ap.add_argument("--memory-probe", nargs=3, help=argparse.SUPPRESS)
    ap.add_argument("--timing-probe", nargs=6, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.memory_probe:
        a, n, bus = args.memory_probe
        _memory_probe(a, int(n), bus)
        return 0
    if args.timing_probe:
        a, n, dist, bus, rounds, out = args.timing_probe
        _time_probe(a, int(n), dist, bus, int(rounds), out)
        return 0
    global _PARTS, _ASSEMBLE
    _PARTS, _ASSEMBLE = args.parts, args.assemble
    if args.smoke:
        args.ns, args.reps, args.distributions, args.no_memory = [1_000], 1, ["spread"], True
        if args.out_dir == HERE:
            args.out_dir = pathlib.Path(tempfile.mkdtemp(prefix="perf_card_smoke_"))
    if args.ns:
        cells = [(n, d) for n in args.ns for d in (args.distributions or ("spread", "uniform"))]
    else:
        cells = list(CELLS)
    ns = sorted({n for n, _ in cells})
    names, arms_not_run = _select_arms(args.arms)
    names = tuple(sorted(names, key=ARM_PRIORITY.index))
    cells = _ordered_cells(cells)
    for arm, why in arms_not_run.items():
        print(f"arm {arm} not run: {why}", flush=True)

    t_start = time.perf_counter()
    work = pathlib.Path(tempfile.mkdtemp(prefix="perf_card_"))
    device, peak, slug, bus_id = _device_facts()
    _check_kernel_bodies()
    sources = _kernel_sources(work)

    # the memory pass first: an arm out of memory at N is never built at N or larger
    memory_rows = [] if args.no_memory else _run_memory_pass(args.memory_ns or ns, bus_id, names)
    oom_from = {}
    for r in memory_rows:
        if r.get("out_of_memory"):
            oom_from[r["arm"]] = min(r["n"], oom_from.get(r["arm"], r["n"]))

    budget_from, replies, per_cell, wall_at = {}, [], [], {}
    for n, dist in cells:
        t0 = time.perf_counter()
        cell = []
        for arm in names:
            if arm in oom_from and n >= oom_from[arm]:
                r = _stub(arm, n, dist, "out_of_memory", work)
            elif arm in budget_from and n >= budget_from[arm][0]:
                r = {**_stub(arm, n, dist, "not_run: budget", work), "predicted_from": budget_from[arm][0],
                     "predicted_dist": budget_from[arm][1]}
            elif (pred := _predicted_over_budget(wall_at.get((arm, dist), {}), n)) is not None:
                r = {**_stub(arm, n, dist, "not_run: budget", work), "predicted_from": pred, "predicted_dist": dist}
            else:
                r = _time_cell_arm(arm, n, dist, bus_id, args.reps, work)
                if r["status"] == "not_run: budget":
                    budget_from[arm] = (n, dist)
                if r["status"] == "out_of_memory":
                    oom_from[arm] = n
            if r["status"] == "ok":
                wall_at.setdefault((arm, dist), {})[n] = statistics.median(r["walls"])
            cell.append(r)
            print(f"{dist} N={n:>9,} {arm:22s} {r['status']}", flush=True)
        agreement, ref, steps = _compare_with_reference(cell)
        per_cell.append({"n": n, "distribution": dist, "reference_arm": ref, "agreement": agreement or {},
                         "steps": steps})
        replies += cell
        print(f"{dist} N={n:>9,} done ({time.perf_counter() - t0:.1f}s)", flush=True)

    # the float32 method cell: the spread inputs at 1e6 (1e3 in a smoke run), eagle and Warp, float32 planes
    fp32_n = ns[0] if args.smoke or args.ns else 1_000_000
    fp32_arms = [a for a in names if a.startswith("eagle") or a == "warp_kernel"]
    if args.smoke:
        fp32_arms = [a for a in fp32_arms if a in ("eagle_graph", "warp_kernel")]
    cell = []
    for arm in fp32_arms:
        r = _time_cell_arm(arm, fp32_n, "fp32", bus_id, args.reps, work)
        cell.append(r)
        print(f"fp32 N={fp32_n:>9,} {arm:22s} {r['status']}", flush=True)
    if cell:
        agreement, ref, steps = _compare_with_reference(cell)
        per_cell.append({"n": fp32_n, "distribution": "fp32", "reference_arm": ref,
                         "agreement": agreement or {}, "steps": steps})
        replies += cell

    # method cells: the floor a launch plus a tail costs, and the fixed overhead of a tiny batch
    def method_cell(n, arm, dist="spread"):
        r = _time_cell_arm(arm, n, dist, bus_id, args.reps, work)
        pathlib.Path(r["counts_path"]).unlink(missing_ok=True)
        if r["status"] != "ok":
            return {"status": r["status"]}
        return {"status": "ok", "wall_s": _median_iqr(r["walls"]), "rounds": len(r["walls"])}

    floor_arm = "eagle_graph" if "eagle_graph" in names else names[0]
    tail = {d: method_cell(TAIL_FLOOR_N, floor_arm, d) for d in (args.distributions or ("spread", "uniform"))}
    overhead = {} if args.smoke else {a: method_cell(OVERHEAD_N, a) for a in names if a.startswith("eagle")}

    sm_count = device["sm_count"]
    try:
        fma_dev = {d: _fma_probe(sm_count, d) for d in ("float64", "float32")}
    except Exception as err:  # noqa: BLE001
        print(f"FMA probe failed: {err}", flush=True)
        fma_dev = None
    ops, audit = _ops_per_attempt(work, f"sm_{device['compute_capability'].replace('.', '')}")
    threads_per_sm = _threads_per_sm()
    results = [_row_of(r, fma_dev, ops, r["n"] / (sm_count * threads_per_sm)) for r in replies]
    for row in results:
        if row["status"] == "ok":
            row["fraction_of_peak_fp64"] = (None if row["arm"] == "eagle_cpu" or not peak["fp64_flops"]
                                            else row["useful_flops_per_s"] / peak["fp64_flops"])
    over = [r for r in results if r.get("eff_sol_note")]      # pending: the reference count would score them above 1
    memory_card = {"method": _MEMORY_METHOD, "rows": memory_rows}
    card = {
        "schema": "eagle-ensemble-card/1",
        "toolchain": cc.toolchain(), "device_slug": slug, "script": SCRIPT_REL,
        "script_md5": cc.md5(__file__), "common_md5": cc.md5(cc.__file__),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release": cc.release_string(),
        "device": device, "peak": peak, "software": _software(), "kernel_sources_md5": sources,
        "workload": {
            "problem": ("two-body Kepler orbits (mu = 1, float64) from periapsis, "
                        f"a = {SEMI_MAJOR:g}, e uniform in [{E_RANGE[0]:g}, {E_RANGE[1]:g}] "
                        f"(uniform: e = {UNIFORM_E:g}), random orientation; adaptive RKF7(8) "
                        "(Fehlberg's tableau, a mixed absolute/relative max-norm error "
                        "control and a standard step-size controller)."),
            "scheme_source": "E. Fehlberg, NASA Technical Report R-287 (1968): the RKF7(8) pair",
            "n_stages": N_STAGES, "rtol": RTOL, "atol": ATOL, "t_final": T_FINAL_FULL,
            "t_final_short": T_FINAL_FULL / SHORT_DIVISOR, "h0": H0, "safety": SAFETY,
            "min_scale": MIN_SCALE, "max_scale": MAX_SCALE,
            "exponents": {"reject": EXP_REJECT, "accept": EXP_ACCEPT},
            "max_attempts": MAX_ATTEMPTS, "seed": SEED,
            "last_step": "clipped to land on the final time",
            "flops_per_attempt": FLOPS_PER_ATTEMPT, "ops_per_attempt": ops,
            "ops_per_attempt_source": "hand count; A100 dynamic count pending",
            "ops_per_attempt_audit": audit, "graph_block": GRAPH_BLOCK,
        },
        "cells": [list(c) for c in cells], "ns": ns,
        "distributions": sorted({d for _, d in cells} | {"fp32"}),
        "arms": {a: ARM_LABELS[a] for a in names}, "arms_not_run": arms_not_run,
        "code_lines": code_line_counts(),
        "tail_floor_ms": {d: (None if v["status"] != "ok" else v["wall_s"]["median"] * 1e3) for d, v in tail.items()},
        "tail_floor": tail, "fixed_overhead_n": OVERHEAD_N, "fixed_overhead": overhead,
        "fp32": {"n": fp32_n, "arms": fp32_arms, "tolerance": FP32_TOL},
        "device_fma_per_s": None if fma_dev is None else fma_dev["float64"],
        "device_fma_per_s_float32": None if fma_dev is None else fma_dev["float32"], "samples_per_lane_threads_per_sm": threads_per_sm,
        "eff_sol_over_one": [[r["arm"], r["n"], r["distribution"]] for r in over],
        "method": {
            "rounds": {"min": ROUNDS_MIN, "max": args.reps, "stop": f"IQR/median <= {IQR_STOP:.0%}"},
            "isolation": "every (arm, N, cell) times in its own fresh process on a clean device; arms run one after another, "
                         "the SM clock of each row shows the drift between them; the memory pass runs first and an arm out "
                         "of memory at N is not built at N or larger",
            "budget": {"per_run_s": ARM_RUN_BUDGET_S, "per_cell_s": CELL_BUDGET_S,
                       "rule": "a warm-up run over the per-run budget makes the arm not_run: budget at that N and larger; "
                               "timed rounds stop at the per-cell budget (at least one round)"},
            "statistic": "median, interquartile range (inclusive quartiles)",
            "warm_up": "one run per arm, excluded; it is also the verification run",
            "wall": "the integration alone, inputs already on the device, to the device synchronised",
            "end_to_end": "inputs uploaded, integration, outputs downloaded",
            "eff_sol": "attempts per second x ops_per_attempt / device_fma_per_s (the register-only DFMA probe run in this "
                       "script after the arms); ops_per_attempt is the DP-pipe instruction count of one fused eagle attempt "
                       "(sol_reference.json), audited against cuobjdump in workload.ops_per_attempt_audit",
            "samples_per_lane": "N / (SMs x resident threads per SM)",
            "verification": (
                "Every arm, every sample: final time reached exactly; position error against the analytic Kepler state "
                "(Newton on Kepler's equation to machine precision) below the stated bound n_acc tol (1 + v_p)(1 + 6 pi "
                "n_orbits v_p), derived from the controller's per-step budget and the Kepler along-track amplification "
                "(see truth_bound; not fitted). The truth check runs in the arm's own process; the step counts it returns "
                "are compared with the reference arm (eagle graph) in the parent: any difference comes from a "
                "floating-point difference flipping one accept/reject decision."),
            "counting_rule": _COUNTING_RULE, "cpu_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "results": results, "per_n": per_cell, "memory": memory_card,
        "torch_compile_note": arms_not_run.get("torch_compiled"),
    }
    card["fit"] = _fit_notes({**card, "results": [r for r in results if r["status"] == "ok" and r["distribution"] == "spread"]})
    card["wall_s"] = time.perf_counter() - t_start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"card_{slug}.json"
    json_path.write_text(json.dumps(card, indent=1) + "\n")
    (args.out_dir / f"card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {json_path}")
    if over:
        print("eff_sol > 1 on " + ", ".join(f"{r['arm']}@{r['n']}" for r in over) + ": not publishable", flush=True)
    if not args.smoke and args.out_dir.resolve().is_relative_to(HERE.parent):
        cc.record_provenance(json_path)
    shutil.rmtree(work, ignore_errors=True)
    return 0


def _threads_per_sm():
    import cupy as cp

    return cp.cuda.Device().attributes["MaxThreadsPerMultiProcessor"]



if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _threads = int(os.environ.get("OMP_NUM_THREADS") or CPU_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(min(_threads, CPU_THREADS))
    sys.exit(main())
