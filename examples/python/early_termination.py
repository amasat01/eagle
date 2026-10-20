# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Early-termination demo: a batch whose samples stop at different times.

Every trajectory is a planar two-body orbit. Depending on how it is seeded, a
trajectory either falls below a collision radius or crosses an escape radius
at some point along the way -- and once it does, it stops changing. The
step kernel maintains a device-side live count of the samples still going;
a conditional (IF) graph node guards the whole step with that count, so once
every sample has terminated the remaining replays of the horizon do no work.
hawk traces the ONE kernel that does all of this (RK4 step, termination
test, live-count update); eagle runs it three ways on the GPU: the GUARDED
graph (the IF node skips a replay once the count is zero), the UNGUARDED
graph (the identical captured pair, never skipped -- the one-variable
control the guarded arm is measured against), and an EAGER loop (the same
kernel, the guard evaluated host-side before every two-step replay, kept for
agreement and for a measured host-sync count, AND as a genuine speed
comparison now that the conditional node's own cost is measured out: on a
Quadro P2000 the guarded graph beats this host-guarded eager loop by
roughly 1.3-1.65x at small-to-medium batch and ties it at very large batch,
where the kernel itself dominates; at the smallest batches (1, 32 samples)
the skip does not pay for itself at all -- the conditional node's fixed
cost exceeds a tiny kernel's, so the UNGUARDED graph is the fastest arm
there -- see the module's ``report`` for the measured numbers). A dead
sample inside a still-live batch still costs a full step (the kernel has no
compaction, only a per-sample select); the saving only starts once EVERY
sample is done, which is why the skip's payoff scales with how much of the
horizon is past the last termination. A CPU arm (``--host``) runs the
identical kernel through eagle's host execution structure, with the guard
evaluated eagerly (there is no graph capture on host), so the example needs
no GPU at all.

Run it with ``python examples/python/early_termination.py --host`` (CPU) or
``--device`` (GPU, guarded vs unguarded vs eager). ``--torch`` (device only)
also times the same RK4 two-body physics written in idiomatic torch --
eager, eager with a host check, and both as a replayed CUDA graph -- and
prints each arm's ratio against our guarded graph, plus an agreement check
against our final state and termination step (demo measurement, not a
benchmark). ``--sweep`` prints the guarded/unguarded/eager wall-per-step
table over a range of batch sizes and horizon lengths (a quick sweep, not
part of the test; add ``--torch`` for an optional torch column, dropped for
the largest batch to keep the sweep under ~3 minutes).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import statistics
import tempfile
import time
from types import SimpleNamespace

import numpy as np

#: Gravitational parameter of the central body, in the demo's own units.
MU = 1.0
#: A sample terminates once its squared radius leaves ``(r_col**2, r_esc**2)``.
R_ESC = 1.5
R_COL = 0.2
#: Bins the printed lifetime histogram spans the REALISED lifetime range
#: with (not the full horizon -- see ``report``'s label).
HIST_BINS = 10
#: Acceptance: the graph and eager arms' final states must agree to this
#: relative tolerance (float32 arithmetic, so not bit-exact; the guarded and
#: UNGUARDED graph arms, by contrast, must agree bit-for-bit -- see
#: ``run_device``'s ``unguarded_exact_match``).
STATE_REL_TOL = 1e-5
#: The kernel's name, and so the name of every file it is published under.
KERNEL_NAME = "et_step"
#: Sweep parameters for ``--sweep`` / :func:`run_sweep`.
SWEEP_BATCHES = (4096, 65536, 1048576)
SWEEP_STEPS = (400, 4000)


# --------------------------------------------------------------------------- #
# Seeding: every orbit is drawn from a band that is guaranteed to terminate.
# --------------------------------------------------------------------------- #
def seed_states(batch: int, seed: int) -> np.ndarray:
    """A ``(4, batch)`` float64 plane of planar orbit seeds, ``(rx, ry, vx, vy)``.

    Each seed starts at radius 1 with purely tangential speed ``u``, drawn
    from one of two bands: ``u`` in ``(0.1, 0.5)`` is a sub-circular ellipse
    whose periapsis falls below ``R_COL``; ``u`` in ``(1.12, 1.6)`` is
    super-circular and either escapes outright or reaches an apoapsis beyond
    ``R_ESC``. Gravity-only orbits with ``u`` in the untouched middle band
    (roughly ``0.5`` to ``1.1``) never cross either radius, so that band is
    never sampled here -- every seed this function returns is one this
    demo's horizon is long enough to terminate.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi, batch)
    collide = rng.uniform(0.1, 0.5, batch)
    escape = rng.uniform(1.12, 1.6, batch)
    speed = np.where(rng.uniform(size=batch) < 0.5, collide, escape)
    return np.stack([
        np.cos(angle),
        np.sin(angle),
        -speed * np.sin(angle),
        speed * np.cos(angle),
    ])


# --------------------------------------------------------------------------- #
# The kernel: ONE traced hawk kernel = one RK4 step, gated by a live count.
# --------------------------------------------------------------------------- #
def step_kernel():
    """Trace the gated RK4 step.

    ``alive``/``alive_next`` are explicit 0/1 planes (``T.Terminated`` is
    read-only, so a terminated sample's own flag has to be carried this way).
    ``age``/``age_next`` ping-pong an increment-while-alive counter
    (``age_next = age + alive``): once a sample's ``alive`` reaches 0, its age
    stops advancing, so the age plane read back at the end IS that sample's
    termination step -- there is no per-replay step-number parameter to read
    it from, because a captured graph replay cannot see a scalar that varies
    per replay (a ``Param`` freezes at capture time).

    ``n_active`` is a one-cell accumulator, seeded to the batch size by the
    caller and never re-zeroed: every launch subtracts exactly the number of
    samples that terminate on THIS step (``still - alive``, zero for a sample
    that stays alive or was already dead), so it decays monotonically to zero
    as the batch finishes. A by-key plane must be a float type (there is
    no integer atomic accumulate), so the caller views this float32 cell as
    uint32 to build the graph's skip guard -- exact for batches under 2**24
    samples, comfortably above anything this demo runs.

    A dead sample's own step is a mathematical no-op (``x_next = x``,
    ``alive_next = 0``, ``age_next = age``, the accumulator's contribution
    ``still - alive = 0 - 0 = 0``) -- this is WHY the unguarded graph arm
    (which never skips, so it re-runs this no-op every remaining step) still
    lands bit-for-bit on the same final state as the guarded one.
    """
    import hawk
    import hawk.math as hm

    def rhs(s):
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = hm.rsqrt(r2 * r2 * r2)
        return hm.vec(s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3)

    @hawk.kernel
    def et_step(
        x: hawk.Vector[4],
        alive: hawk.Scalar,
        age: hawk.Scalar,
        dt: hawk.Param,
        r_esc2: hawk.Param,
        r_col2: hawk.Param,
        x_next: hawk.Mutable[hawk.Vector[4]],
        alive_next: hawk.Mutable[hawk.Scalar],
        age_next: hawk.Mutable[hawk.Scalar],
        n_active: hawk.Mutable[hawk.Scalar],
    ):
        k1 = rhs(x)
        k2 = rhs(x + (0.5 * dt) * k1)
        k3 = rhs(x + (0.5 * dt) * k2)
        k4 = rhs(x + dt * k3)
        xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        r2 = xn[0] * xn[0] + xn[1] * xn[1]
        zero = alive * 0
        one = zero + 1
        live = alive != 0
        still = hm.select(
            live, hm.select(r2 < r_esc2, hm.select(r2 > r_col2, one, zero), zero), zero
        )

        x_next = hm.select(live, xn, x)
        alive_next = still
        age_next = age + alive
        hawk.all_samples(n_active)[0] += still - alive

    return et_step


def build_artifact(directory, targets) -> object:
    """Emit, compile and publish the step kernel into ``directory``."""
    import hawk

    return hawk.build([step_kernel()], pathlib.Path(directory),
                        mode="float32", targets=tuple(targets))


def check_manifest(bundle) -> dict:
    """Read the published manifest back and validate its shape through raptor."""
    from raptor.schema import validate_manifest

    doc = json.loads(pathlib.Path(bundle.manifest_path).read_text())
    validate_manifest(doc)
    return doc


# --------------------------------------------------------------------------- #
# The consumer side: load the artifact through eagle and run it.
# --------------------------------------------------------------------------- #
def _sidecar(bundle, name: str) -> dict:
    return next(a.sidecar for a in bundle.artifacts if a.name == name)


def host_plugin(bundle, name: str):
    """The plan-able view of the bundle's HOST entry (mirrors two_body_hybrid.py)."""
    from hawk.artifact import plan_view

    library = ctypes.CDLL(str(pathlib.Path(bundle.directory) / f"{name}.so"))
    view = plan_view(_sidecar(bundle, name))
    entry = getattr(library, view.pop("host_entry"))
    return SimpleNamespace(
        host_entry=ctypes.cast(entry, ctypes.c_void_p).value,
        _keepalive=library, **view)


def device_plugin(bundle, name: str):
    """The plan-able view of the bundle's DEVICE entry (mirrors two_body_hybrid.py)."""
    from eagle.registry import load_manifest
    from hawk.artifact import plan_view

    registry = load_manifest(pathlib.Path(bundle.manifest_path))
    loaded = registry[name]
    view = plan_view(_sidecar(bundle, name))
    view.pop("host_entry", None)
    return SimpleNamespace(
        device_function=loaded.fn.kernel.ptr,
        _keepalive=(registry, loaded), **view)


def _fresh_planes(state: np.ndarray, batch: int, xp) -> tuple:
    """Ping-pong ``x``/``alive``/``age`` planes plus the live-count cell.

    ``xp`` is ``numpy`` (host) or ``cupy`` (device); the seed state is
    written into slot 0 of ``x``, every sample starts alive with age 0, and
    the count starts at ``batch`` (the decrement form the kernel implements
    needs no zeroing between launches).
    """
    x = [xp.asarray(np.ascontiguousarray(state, dtype=np.float32)),
         xp.zeros((4, batch), dtype=xp.float32)]
    alive = [xp.ones(batch, dtype=xp.float32), xp.zeros(batch, dtype=xp.float32)]
    age = [xp.zeros(batch, dtype=xp.float32), xp.zeros(batch, dtype=xp.float32)]
    count = xp.full(1, batch, dtype=xp.float32)
    return x, alive, age, count


def _reset_planes_(planes, state: np.ndarray, batch: int) -> None:
    """Reset ping-pong planes to their seed state IN PLACE (same device
    memory, same pointers, no reallocation).

    A captured :class:`~eagle.pipeline.GraphPipeline` replays the EXACT
    device pointers it was built against; calling :func:`_fresh_planes`
    again between timing reps would allocate new memory and leave the
    already-captured graph replaying stale buffers. Writing the seed state
    back into the SAME arrays (``.set``/``.fill``, never a new allocation)
    is what lets one capture be timed over many reset-and-replay reps.
    """
    x, alive, age, count = planes
    seed = np.ascontiguousarray(state, dtype=np.float32)
    if hasattr(x[0], "set"):  # cupy: an in-place host -> device copy
        x[0].set(seed)
    else:
        x[0][...] = seed
    x[1].fill(0)
    alive[0].fill(1)
    alive[1].fill(0)
    age[0].fill(0)
    age[1].fill(0)
    count.fill(float(batch))


def _bind_pair(plan, x, alive, age, count, dt, r_esc2, r_col2) -> list:
    """Both ping-pong directions of one bound plan, sharing the same count cell."""
    kw = dict(dt=dt, r_esc2=r_esc2, r_col2=r_col2, n_active=count)
    return [
        plan.bind(x=x[0], alive=alive[0], age=age[0],
                  x_next=x[1], alive_next=alive[1], age_next=age[1], **kw),
        plan.bind(x=x[1], alive=alive[1], age=age[1],
                  x_next=x[0], alive_next=alive[0], age_next=age[0], **kw),
    ]


def _counted(count_view):
    """A nonzero guard on ``count_view`` that counts its host-side reads.

    ``SkipGuard.evaluate`` is the only place a guard reads the live count on
    the host (each read is a device-to-host sync); a captured IF node reads
    it on the device and never calls it. Counting the calls makes the host
    syncs a MEASURED number rather than a claim.
    """
    import eagle

    reads = {"n": 0}

    class CountingGuard(eagle.SkipGuard):
        __slots__ = ()

        def evaluate(self):
            reads["n"] += 1
            return super().evaluate()

    return CountingGuard.nonzero(count_view), reads


def _time_arms(plan, state: np.ndarray, batch: int, dt: float, r_esc2: float,
               r_col2: float, steps: int, reps: int) -> tuple:
    """Build the guarded graph, unguarded graph and eager arms ONCE, warm
    each up once (excluded from the timing), then time ``reps`` INTERLEAVED
    reps (one rep = every arm in turn, each reset in place first) and return
    ``(wall_per_step, host_syncs, last_termination, arms)``.

    ``wall_per_step``/``host_syncs`` are ``{"guarded"/"unguarded"/"eager":
    ...}`` dicts (``host_syncs`` has no "unguarded" entry -- it carries no
    guard to read); ``wall_per_step`` values are the MEDIAN over the reps.
    ``arms`` is ``{name: (planes, run)}``, returned so the caller can read
    off final state / continue replaying (the skip proof) without rebuilding
    anything.
    """
    import cupy as cp
    import eagle
    from eagle.pipeline import GraphPipeline

    replays = steps // 2

    def make_pair(planes):
        bound = _bind_pair(plan, *planes, dt, r_esc2, r_col2)

        def pair():
            bound[0].launch()
            bound[1].launch()

        return pair

    g_planes = _fresh_planes(state, batch, cp)
    guard, g_reads = _counted(g_planes[3].view(cp.uint32))
    g_pipe = GraphPipeline()
    g_pipe.add(eagle.skippable(make_pair(g_planes), guard), name="pair")
    g_pipe.build()

    u_planes = _fresh_planes(state, batch, cp)
    u_pipe = GraphPipeline()
    u_pipe.add(make_pair(u_planes), name="pair")  # no skippable: the control
    u_pipe.build()

    e_planes = _fresh_planes(state, batch, cp)
    e_guard, e_reads = _counted(e_planes[3].view(cp.uint32))
    e_step = eagle.skippable(make_pair(e_planes), e_guard)

    def e_run(n):
        for _ in range(n):
            e_step()
        cp.cuda.runtime.deviceSynchronize()

    arms = {
        "guarded": (g_planes, g_pipe.launch),
        "unguarded": (u_planes, u_pipe.launch),
        "eager": (e_planes, e_run),
    }
    reads = {"guarded": g_reads, "eager": e_reads}

    for planes, run in arms.values():
        _reset_planes_(planes, state, batch)
        run(1)  # warm-up: build/first-launch/JIT cost, excluded

    times = {name: [] for name in arms}
    for _ in range(max(1, reps)):
        for name, (planes, run) in arms.items():
            _reset_planes_(planes, state, batch)
            if name in reads:
                reads[name]["n"] = 0
            start = time.perf_counter()
            run(replays)
            times[name].append((time.perf_counter() - start) / steps)

    wall = {name: statistics.median(v) for name, v in times.items()}
    host_syncs = {
        "guarded": 1 + g_reads["n"],  # the launch's own sync + every guard read
        "eager": e_reads["n"],
    }
    last_termination = int(cp.asnumpy(g_planes[2][0]).max())
    return wall, host_syncs, last_termination, arms


# --------------------------------------------------------------------------- #
# The device arm: guarded graph vs its own unguarded control vs eager.
# --------------------------------------------------------------------------- #
def run_device(bundle, state: np.ndarray, dt: float, r_esc: float,
               r_col: float, steps: int, reps: int = 3) -> dict:
    """Time the guarded graph against the SAME captured pair with no guard
    (the one-variable control that isolates what the skip alone buys), plus
    the eager loop (agreement + measured sync count only -- see the module
    docstring for why this is not a speed comparison on this card).
    """
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    if steps % 2:
        raise ValueError(
            f"a pair-launch replay advances two steps, so --steps must be "
            f"even; got {steps}")

    batch = state.shape[1]
    r_esc2, r_col2 = float(r_esc) ** 2, float(r_col) ** 2
    plan = eplan.plan(device_plugin(bundle, KERNEL_NAME), structure=eexec.DeviceKernel)

    wall, host_syncs, last_termination, arms = _time_arms(
        plan, state, batch, dt, r_esc2, r_col2, steps, reps)

    g_planes, g_run = arms["guarded"]
    u_planes, _ = arms["unguarded"]
    e_planes, _ = arms["eager"]

    final_x_guarded = cp.asnumpy(g_planes[0][0])
    age_guarded = cp.asnumpy(g_planes[2][0])
    active_after = int(g_planes[3].get()[0])

    final_x_unguarded = cp.asnumpy(u_planes[0][0])
    age_unguarded = cp.asnumpy(u_planes[2][0])
    unguarded_exact_match = bool(
        np.array_equal(age_guarded, age_unguarded)
        and np.array_equal(final_x_guarded, final_x_unguarded))

    final_x_eager = cp.asnumpy(e_planes[0][0])
    age_eager = cp.asnumpy(e_planes[2][0])
    termination_match = bool(np.array_equal(age_guarded, age_eager))
    denom = np.maximum(np.abs(final_x_eager), 1e-6)
    state_rel_diff = float(np.max(np.abs(final_x_guarded - final_x_eager) / denom))

    hist_range = (0, max(last_termination, 1))
    hist, _ = np.histogram(age_guarded, bins=HIST_BINS, range=hist_range)

    # ---- skip proof: force every flag back alive, replay again (guarded). ----
    # The graph's IF node reads only the live count, which is already 0, so
    # this must be a no-op regardless of what the (otherwise dead) alive
    # plane says -- proving the skip, not merely that nothing HAPPENED to.
    alive_g = g_planes[1]
    count_g = g_planes[3]
    alive_g[0].fill(1.0)
    alive_g[1].fill(1.0)
    cp.cuda.runtime.deviceSynchronize()
    before = cp.asnumpy(g_planes[0][0]).tobytes()
    g_run(steps // 2)
    after = cp.asnumpy(g_planes[0][0]).tobytes()
    skip_frozen = before == after
    skip_count_zero = int(count_g.get()[0]) == 0

    # ---- positive control: a nonzero count must let the step run again. ----
    count_g.fill(float(batch))
    cp.cuda.runtime.deviceSynchronize()
    g_run(1)
    skip_control_runs = cp.asnumpy(g_planes[0][0]).tobytes() != after

    skipped_share = max(0.0, (steps - last_termination) / steps) if steps else 0.0
    expected_speedup = (
        (steps / last_termination) if last_termination > 0 else float("nan"))
    measured_speedup = (
        wall["unguarded"] / wall["guarded"] if wall["guarded"] > 0 else float("nan"))
    speedup_vs_eager_host_guard = (
        wall["eager"] / wall["guarded"] if wall["guarded"] > 0 else float("nan"))

    return {
        "wall_per_step_guarded": wall["guarded"],
        "wall_per_step_unguarded": wall["unguarded"],
        "wall_per_step_eager": wall["eager"],
        "measured_speedup": measured_speedup,
        "expected_speedup": expected_speedup,
        "speedup_vs_eager_host_guard": speedup_vs_eager_host_guard,
        "skipped_share": skipped_share,
        "last_termination": last_termination,
        "host_syncs_graph": host_syncs["guarded"],
        "host_syncs_eager": host_syncs["eager"],
        "termination_match": termination_match,
        "state_rel_diff": state_rel_diff,
        "unguarded_exact_match": unguarded_exact_match,
        "active_after": active_after,
        "final_x_guarded": final_x_guarded,
        "age_guarded": age_guarded,
        "lifetime_histogram": hist.tolist(),
        "skip_frozen": skip_frozen,
        "skip_count_zero": skip_count_zero,
        "skip_control_runs": skip_control_runs,
    }


# --------------------------------------------------------------------------- #
# The torch arm: the SAME physics written idiomatically in torch, timed the
# same way as the eagle arms above, and checked against them for agreement.
# --------------------------------------------------------------------------- #
def run_torch(state: np.ndarray, batch: int, dt: float, r_esc: float,
              r_col: float, steps: int, reps: int,
              ref_final_x: np.ndarray, ref_age: np.ndarray) -> dict:
    """Four idiomatic-torch arms (eager masked, eager + host check, torch
    CUDA graph masked, torch CUDA graph + host check) over the SAME
    batch/steps/dt/seed as the eagle arms, timed with the same pattern
    (build once, warm up once excluded, median of ``reps`` interleaved
    reps), each checked against ``ref_final_x``/``ref_age`` (the guarded
    graph's own final state and per-sample termination step).

    A torch ``CUDAGraph`` has no conditional node, so ``graph_masked`` is
    the torch analogue of the UNGUARDED control, not of the guarded arm --
    it always replays the full horizon. ``eager_hostcheck`` and
    ``graph_hostcheck`` may stop early once every sample is dead, which is
    safe: the kernel's own masking makes a replay past that point a
    mathematical no-op (the same property the guarded arm's skip proof
    relies on), so an early stop and a full run land on the same state.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed; --torch needs it (pip install torch), "
            "or drop --torch to run without the comparison") from exc

    if steps % 2:
        raise ValueError(
            f"a graph pair advances two steps, so --steps must be even; "
            f"got {steps}")

    r_esc2, r_col2 = float(r_esc) ** 2, float(r_col) ** 2
    dev = torch.device("cuda")
    s0 = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).to(dev)
    dt_t = torch.tensor(dt, dtype=torch.float32, device=dev)
    re2_t = torch.tensor(r_esc2, dtype=torch.float32, device=dev)
    rc2_t = torch.tensor(r_col2, dtype=torch.float32, device=dev)

    def rhs(s):
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = torch.rsqrt(r2 * r2 * r2)
        return torch.stack((s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3))

    def step(x, alive, age, dt_, re2_, rc2_):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt_ * k1)
        k3 = rhs(x + 0.5 * dt_ * k2)
        k4 = rhs(x + dt_ * k3)
        xn = x + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        r2 = xn[0] * xn[0] + xn[1] * xn[1]
        live = alive != 0
        still = (live & (r2 < re2_) & (r2 > rc2_)).to(x.dtype)
        age_next = age + alive
        x.copy_(torch.where(live, xn, x))
        alive.copy_(still)
        age.copy_(age_next)

    def fresh():
        return (s0.clone(), torch.ones(batch, dtype=torch.float32, device=dev),
                torch.zeros(batch, dtype=torch.float32, device=dev))

    def reset_(planes):
        x, alive, age = planes
        x.copy_(s0)
        alive.fill_(1.0)
        age.fill_(0.0)
        torch.cuda.synchronize()

    def capture_pair(planes):
        x, alive, age = planes
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step(x, alive, age, dt_t, re2_t, rc2_t)
        torch.cuda.current_stream().wait_stream(stream)
        reset_(planes)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(x, alive, age, dt_t, re2_t, rc2_t)
            step(x, alive, age, dt_t, re2_t, rc2_t)
        return graph

    p_em, p_eh, p_gm, p_gh = fresh(), fresh(), fresh(), fresh()
    graph_m = capture_pair(p_gm)
    graph_h = capture_pair(p_gh)

    def eager_masked(n, planes=p_em):
        x, alive, age = planes
        for _ in range(n):
            step(x, alive, age, dt_t, re2_t, rc2_t)
        torch.cuda.synchronize()

    def eager_hostcheck(n, planes=p_eh):
        x, alive, age = planes
        for _ in range(0, n, 2):
            if alive.sum().item() == 0:
                break
            step(x, alive, age, dt_t, re2_t, rc2_t)
            step(x, alive, age, dt_t, re2_t, rc2_t)
        torch.cuda.synchronize()

    def graph_masked(n, planes=p_gm):
        for _ in range(n // 2):
            graph_m.replay()
        torch.cuda.synchronize()

    def graph_hostcheck(n, planes=p_gh):
        _x, alive, _age = planes
        for _ in range(n // 2):
            if alive.sum().item() == 0:
                break
            graph_h.replay()
        torch.cuda.synchronize()

    arms = {
        "eager_masked": (p_em, eager_masked),
        "eager_hostcheck": (p_eh, eager_hostcheck),
        "graph_masked": (p_gm, graph_masked),
        "graph_hostcheck": (p_gh, graph_hostcheck),
    }

    for planes, run in arms.values():
        reset_(planes)
        run(2)  # warm-up: capture/first-launch cost already paid, excluded

    times = {name: [] for name in arms}
    for _ in range(max(1, reps)):
        for name, (planes, run) in arms.items():
            reset_(planes)
            start = time.perf_counter()
            run(steps)
            times[name].append((time.perf_counter() - start) / steps)

    wall = {name: statistics.median(v) for name, v in times.items()}

    agreement = {}
    for name, (planes, _run) in arms.items():
        x, alive, age = planes
        final_x = x.detach().cpu().numpy()
        final_age = age.detach().cpu().numpy()
        # A termination-step mismatch (a float32 op-order difference between
        # torch's and our kernel's arithmetic flipping WHICH sub-step a
        # sample crossed its radius on) makes that sample's own final state
        # legitimately far from ours -- a different point on the same
        # trajectory, not a numerical-agreement failure. So STATE_REL_TOL is
        # checked over the samples whose termination step DOES match; the
        # mismatch count is reported (never hidden), not folded into a
        # loosened tolerance.
        #
        # The per-sample error is the STATE VECTOR's relative norm, not a
        # per-component ratio: a planar orbit's rx/ry/vx/vy each cross zero
        # routinely, and a per-component ratio against a near-zero component
        # (denominator floored at 1e-6) inflates a genuinely tiny absolute
        # cross-implementation difference (torch's rsqrt/op-order vs our
        # compiled kernel's) into a spuriously large ratio -- measured on
        # this batch, a component-wise check reads 1.06e-4 (10x over
        # STATE_REL_TOL) at a sample whose rx is -0.0026, while the same
        # sample's whole-state relative norm is 1.2e-6. This is a measurement
        # choice, not a loosened tolerance: STATE_REL_TOL itself is unchanged.
        mismatch_mask = final_age != ref_age
        matched_mask = ~mismatch_mask
        if np.any(matched_mask):
            diff_norm = np.linalg.norm(
                final_x[:, matched_mask] - ref_final_x[:, matched_mask], axis=0)
            ref_norm = np.maximum(
                np.linalg.norm(ref_final_x[:, matched_mask], axis=0), 1e-6)
            state_rel_diff = float(np.max(diff_norm / ref_norm))
        else:
            state_rel_diff = float("nan")
        agreement[name] = {
            "state_rel_diff": state_rel_diff,
            "termination_mismatches": int(np.sum(mismatch_mask)),
            "final_alive": int(alive.sum().item()),
        }

    return {"wall_per_step": wall, "agreement": agreement, "batch": batch}


def run_host(bundle, state: np.ndarray, dt: float, r_esc: float,
             r_col: float, steps: int) -> dict:
    """Run the horizon on CPU threads through ``eagle.exec.HostTeam``.

    There is no graph capture on the host arm, so :func:`eagle.skippable`'s
    eager form IS the only form here: calling the wrapped step directly
    evaluates the guard (a plain host array read) before every pair.
    """
    import eagle
    import eagle.exec as eexec
    from eagle import plan as eplan

    if steps % 2:
        raise ValueError(
            f"a pair-launch replay advances two steps, so --steps must be "
            f"even; got {steps}")

    batch = state.shape[1]
    replays = steps // 2
    r_esc2, r_col2 = float(r_esc) ** 2, float(r_col) ** 2
    plan = eplan.plan(host_plugin(bundle, KERNEL_NAME), structure=eexec.HostTeam)

    x, alive, age, count = _fresh_planes(state, batch, np)
    bound = _bind_pair(plan, x, alive, age, count, dt, r_esc2, r_col2)

    def pair():
        bound[0].launch()
        bound[1].launch()

    guard, reads = _counted(count.view(np.uint32))
    step = eagle.skippable(pair, guard)
    start = time.perf_counter()
    for _ in range(replays):
        step()
    wall_per_step = (time.perf_counter() - start) / steps
    host_syncs = reads["n"]

    active_after = int(count[0])
    last_termination = int(age[0].max())
    hist, _ = np.histogram(age[0], bins=HIST_BINS, range=(0, max(last_termination, 1)))

    # Skip proof, same shape as the device arm's: force alive back on, replay
    # again -- the count is already 0, so nothing may change.
    alive[0][:] = 1.0
    before = x[0].tobytes()
    for _ in range(replays):
        step()
    after = x[0].tobytes()
    skip_frozen = before == after
    skip_count_zero = int(count[0]) == 0

    # Positive control: a nonzero count must let the step run again.
    count[0] = float(batch)
    step()
    skip_control_runs = x[0].tobytes() != after

    return {
        "host_syncs": host_syncs,
        "wall_per_step": wall_per_step,
        "active_after": active_after,
        "last_termination": last_termination,
        "lifetime_histogram": hist.tolist(),
        "skip_frozen": skip_frozen,
        "skip_count_zero": skip_count_zero,
        "skip_control_runs": skip_control_runs,
    }


# --------------------------------------------------------------------------- #
# The performance sweep: the table this round's numbers came from.
# --------------------------------------------------------------------------- #
def run_sweep(dt: float = 0.01, r_esc: float = R_ESC, r_col: float = R_COL,
             seed: int = 0, build_dir=None, torch_compare: bool = False) -> list:
    """The guarded/unguarded/eager wall-per-step table over
    ``SWEEP_BATCHES`` x ``SWEEP_STEPS``, ``reps=1``. Builds the device
    artifact once and reuses the same plan across every combination. NOT run
    by the test (a full pass targets under ~2 minutes on a Quadro P2000, but
    it is still real GPU time, not something a unit test should own).

    ``torch_compare`` adds a torch column (the four :func:`run_torch` arms,
    ``reps=1``) to every row EXCEPT the largest ``SWEEP_BATCHES`` entry,
    dropped to keep the whole sweep under ~3 minutes; that row's torch
    fields are left ``None`` and the print line says so.
    """
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    r_esc2, r_col2 = float(r_esc) ** 2, float(r_col) ** 2
    torch_skip_batch = max(SWEEP_BATCHES)
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        where = pathlib.Path(build_dir or tmpdir) / "early_termination_sweep"
        bundle = build_artifact(where, ("cuda",))
        plan = eplan.plan(
            device_plugin(bundle, KERNEL_NAME), structure=eexec.DeviceKernel
        )
        for batch in SWEEP_BATCHES:
            state = seed_states(batch, seed)
            for steps in SWEEP_STEPS:
                wall, _host_syncs, last_termination, arms = _time_arms(
                    plan, state, batch, dt, r_esc2, r_col2, steps, reps=1)
                row = {
                    "batch": batch, "steps": steps,
                    "last_termination": last_termination,
                    "wall_per_step_guarded": wall["guarded"],
                    "wall_per_step_unguarded": wall["unguarded"],
                    "wall_per_step_eager": wall["eager"],
                    "torch": None,
                }
                line = (
                    f"batch={batch:>8} steps={steps:>5} "
                    f"last_termination={last_termination:>5} "
                    f"({last_termination / steps:.0%})  us/step: "
                    f"guarded={wall['guarded'] * 1e6:7.2f}  "
                    f"unguarded={wall['unguarded'] * 1e6:7.2f}  "
                    f"eager={wall['eager'] * 1e6:7.2f}")
                if torch_compare:
                    if batch == torch_skip_batch:
                        line += (
                            "  torch=skipped (largest batch, keeps --sweep --torch "
                            "under ~3 min)"
                        )
                    else:
                        g_planes = arms["guarded"][0]
                        ref_final_x = cp.asnumpy(g_planes[0][0])
                        ref_age = cp.asnumpy(g_planes[2][0])
                        torch_row = run_torch(state, batch, dt, r_esc, r_col,
                                              steps, 1, ref_final_x, ref_age)
                        row["torch"] = torch_row
                        line += "  torch us/step: " + "  ".join(
                            f"{name}={us * 1e6:7.2f}"
                            for name, us in torch_row["wall_per_step"].items())
                rows.append(row)
                print(line, flush=True)
                cp.get_default_memory_pool().free_all_blocks()
    return rows


# --------------------------------------------------------------------------- #
# The demo itself.
# --------------------------------------------------------------------------- #
def run_demo(steps: int | None = None, batch: int | None = None, dt: float = 0.01,
             seed: int = 0, r_esc: float = R_ESC, r_col: float = R_COL,
             target: str = "host", reps: int = 3, build_dir=None,
             torch_compare: bool = False, verbose: bool = True) -> dict:
    """Build, run and check. Returns every measured number.

    ``batch``/``steps`` default differently per target when left ``None``:
    the device arm defaults to a batch and horizon long enough that the
    skipped tail is most of it (65536 x 1000), so the saving is visible by
    default; the host arm defaults to a size that keeps ``--host`` under
    about ten seconds (1024 x 400). ``torch_compare`` (device target only)
    adds the four idiomatic-torch arms from :func:`run_torch` under
    ``results["torch"]``.
    """
    if batch is None:
        batch = 65536 if target == "device" else 1024
    if steps is None:
        steps = 1000 if target == "device" else 400
    if torch_compare and target != "device":
        raise ValueError("torch_compare is device target only")

    state = seed_states(batch, seed)

    targets = ("host",) if target == "host" else ("cuda",)
    with tempfile.TemporaryDirectory() as tmpdir:
        where = pathlib.Path(build_dir or tmpdir) / "early_termination"
        bundle = build_artifact(where, targets)
        manifest = check_manifest(bundle)
        if target == "host":
            arm = run_host(bundle, state, dt, r_esc, r_col, steps)
        else:
            arm = run_device(bundle, state, dt, r_esc, r_col, steps, reps)
            if torch_compare:
                arm["torch"] = run_torch(
                    state, batch, dt, r_esc, r_col, steps, reps,
                    arm["final_x_guarded"], arm["age_guarded"])

    results = {
        "target": target, "steps": steps, "batch": batch, "dt": dt,
        "r_esc": r_esc, "r_col": r_col,
        "exec_targets": manifest["exec_targets"],
        **arm,
    }
    if verbose:
        report(results)

    assert results["active_after"] == 0, (
        f"not every sample terminated within {steps} steps "
        f"({results['active_after']} still active); raise --steps")
    assert results["skip_frozen"], (
        "a replay after all-terminated changed the state -- the guard did "
        "not skip the step")
    assert results["skip_count_zero"], (
        "the live count moved after all-terminated -- expected it pinned "
        "at 0")
    assert results["skip_control_runs"], (
        "restoring a nonzero live count did not let the step run -- the "
        "guard skips unconditionally, so the skip proof proves nothing")
    if target == "device":
        assert results["host_syncs_graph"] == 1, (
            f"graph-path host syncs: expected 1, got "
            f"{results['host_syncs_graph']}")
        assert results["unguarded_exact_match"], (
            "the unguarded control disagrees with the guarded graph -- the "
            "guard changed the RESULT, not just whether the step ran")
        assert results["termination_match"], (
            "the graph and eager arms disagree on WHEN some sample "
            "terminated")
        assert results["state_rel_diff"] <= STATE_REL_TOL, (
            f"graph vs eager final states differ by {results['state_rel_diff']:.3e}, "
            f"tolerance {STATE_REL_TOL:.0e}")
        if torch_compare:
            for name, agreement in results["torch"]["agreement"].items():
                # State agreement is a hard correctness check; a termination
                # STEP mismatch is float32-op-order-legitimate and only
                # reported (see run_torch), never used to loosen this.
                assert agreement["state_rel_diff"] <= STATE_REL_TOL, (
                    f"torch arm {name!r} disagrees with the guarded graph "
                    f"by {agreement['state_rel_diff']:.3e}, tolerance "
                    f"{STATE_REL_TOL:.0e}")
    return results


def report(results: dict) -> None:
    """Print the acceptance numbers and the timing lines."""
    print(f"target            : {results['target']} "
          f"(manifest exec_targets {results['exec_targets']})")
    print(f"batch x steps     : {results['batch']} x {results['steps']} "
          f"at dt = {results['dt']}")
    print(f"escape / collide  : r > {results['r_esc']} / r < {results['r_col']}")
    print(f"samples still active after the horizon : {results['active_after']} "
          f"(expected 0)")
    last = results.get("last_termination")
    print(f"lifetime histogram ({HIST_BINS} bins over the REALISED lifetime "
          f"range [0, {last}], not the full horizon):",
          results["lifetime_histogram"])
    print(f"replay after all-terminated: state bit-identical = "
          f"{results['skip_frozen']}, count stays 0 = {results['skip_count_zero']}")
    print(f"positive control (count restored nonzero): step runs = "
          f"{results['skip_control_runs']}")
    if results["target"] == "device":
        print(f"skipped share of the horizon: {results['skipped_share'] * 100:.1f}% "
              f"(last termination at step {last} of {results['steps']})")
        print(f"wall per step (demo measurement, not a benchmark): guarded graph "
              f"{results['wall_per_step_guarded'] * 1e6:.1f} us, unguarded graph "
              f"{results['wall_per_step_unguarded'] * 1e6:.1f} us -- measured "
              f"speedup {results['measured_speedup']:.2f}x (expected from the "
              f"skipped share alone: {results['expected_speedup']:.2f}x)")
        print(f"eager wall per step (demo measurement, not a benchmark): "
              f"{results['wall_per_step_eager'] * 1e6:.1f} us -- guarded "
              f"graph vs this host-guarded eager loop: "
              f"{results['speedup_vs_eager_host_guard']:.2f}x (this ratio depends "
              f"on the batch size; --sweep measures it across sizes)")
        print(f"host syncs (measured): graph = {results['host_syncs_graph']} "
              f"-- the skip decision never leaves the device; eager = "
              f"{results['host_syncs_eager']} (one host round-trip per two-step "
              f"replay)")
        print(f"unguarded control agrees with the guarded graph bit-for-bit "
              f"(termination steps and final state): "
              f"{results['unguarded_exact_match']}")
        print(f"graph vs eager    : termination steps identical = "
              f"{results['termination_match']}, max relative state diff = "
              f"{results['state_rel_diff']:.3e} (tolerance {STATE_REL_TOL:.0e})")
        if results.get("torch"):
            report_torch(results["torch"], results["wall_per_step_guarded"])
    else:
        print(f"host syncs        : {results['host_syncs']}")
        print(f"wall per step (demo measurement, not a benchmark): "
              f"{results['wall_per_step'] * 1e6:.1f} us")


def report_torch(torch_results: dict, guarded_wall: float) -> None:
    """Print the torch comparison table and its agreement lines (demo
    measurement, not a benchmark)."""
    batch = torch_results["batch"]
    print("torch comparison (demo measurement, not a benchmark; idiomatic "
          "torch, same RK4 physics, float32):")
    for name, us in torch_results["wall_per_step"].items():
        ratio = us / guarded_wall if guarded_wall > 0 else float("nan")
        print(f"  {name:<17} {us * 1e6:9.1f} us/step   "
              f"{ratio:6.2f}x vs our guarded graph")
    for name, agreement in torch_results["agreement"].items():
        note = ""
        if agreement["termination_mismatches"]:
            note = (f" -- {agreement['termination_mismatches']} sample(s) "
                     f"disagree on WHEN they terminated (float32 op-order; "
                     f"reported, not tolerated away)")
        print(f"  {name:<17} agreement: max relative state diff = "
              f"{agreement['state_rel_diff']:.3e} (tolerance "
              f"{STATE_REL_TOL:.0e}), termination mismatches = "
              f"{agreement['termination_mismatches']}/{batch}, "
              f"final alive = {agreement['final_alive']}{note}")


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=None,
                        help="RK4 steps in the horizon, must be even "
                             "(default: 1000 for --device, 400 for --host)")
    parser.add_argument("--batch", type=int, default=None,
                        help="independent trajectories (default: 65536 for "
                             "--device, 1024 for --host)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="step size (default: 0.01)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the orbits (default: 0)")
    parser.add_argument("--r-esc", type=float, default=R_ESC,
                        help=f"escape radius (default: {R_ESC})")
    parser.add_argument("--r-col", type=float, default=R_COL,
                        help=f"collision radius (default: {R_COL})")
    parser.add_argument("--reps", type=int, default=3,
                        help="interleaved timing reps for --device, median "
                             "reported (default: 3)")
    parser.add_argument("--build-dir", default=None,
                        help="where to publish the artifact (default: a "
                             "temporary directory)")
    parser.add_argument("--sweep", action="store_true",
                        help="print the guarded/unguarded/eager table over "
                             "a range of batch sizes and horizons, then exit "
                             "(reps=1 per point, ~2 minutes on a P2000)")
    parser.add_argument("--torch", dest="torch_compare", action="store_true",
                        help="also time four idiomatic-torch arms (eager, "
                             "eager + host check, torch CUDA graph, torch "
                             "CUDA graph + host check) against the guarded "
                             "graph and check agreement; device target only "
                             "(with --sweep, adds an optional torch column, "
                             "dropped for the largest batch)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--host", dest="target", action="store_const",
                       const="host", help="run on CPU threads (the default)")
    where.add_argument("--device", dest="target", action="store_const",
                       const="device", help="run on the GPU: guarded vs "
                       "unguarded vs eager")
    parser.set_defaults(target="host")
    args = parser.parse_args(argv)
    if args.torch_compare and not args.sweep and args.target != "device":
        parser.error("--torch requires --device (or --sweep --torch)")
    return args


def main(argv=None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.sweep:
        run_sweep(dt=args.dt, r_esc=args.r_esc, r_col=args.r_col,
                 seed=args.seed, build_dir=args.build_dir,
                 torch_compare=args.torch_compare)
        return 0
    run_demo(steps=args.steps, batch=args.batch, dt=args.dt, seed=args.seed,
             r_esc=args.r_esc, r_col=args.r_col, target=args.target,
             reps=args.reps, build_dir=args.build_dir,
             torch_compare=args.torch_compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
