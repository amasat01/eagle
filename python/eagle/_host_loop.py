# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host branch of :meth:`eagle.Runner.run`: the run-until-done loop on
eagle's OpenMP host team.

The loop is the runner's own :class:`eagle.RepeatWhile`, with its steps
issued straight to the team: each step is one :data:`eagle.exec.HostTeam`
launch over the live-count partition, and the guard reads the counter
cell the kernel counts its finished samples into. A finishing kernel
marks and counts its own samples inside the team, so a step costs one
team launch and nothing serial.

* **Plain artifact**: launches the step over the plan's one partition.
* **Active-set artifact** (``Guard(active_set=True)``): launches
  ``every`` steps over ``[0, active_count)``, stops early once every
  sample is done, then runs the loop's compaction (and reorder) tail.
* **Automatic artifact** (``steps="auto"``): launches the step once at
  the policy's ``k``, then runs the loop's remaining parts while ``go``
  holds. :attr:`eagle.Runner.host_policy` picks the pacing; its default
  ``"auto"`` resolves per entry:

  - ``"tiled"`` when the entry's fused side is the TILED loop (its shared
    object exports ``<entry>_tile`` > 0, :func:`host_tile`): each tile
    advances its samples together, vectorised, with their state in cache,
    and stops once its last sample finished. A fused launch then costs
    less than a sweep from the first launch, so every launch asks for the
    artifact's ``steps_max`` (the last cut to the cap): no sweep phase, no
    probe, and one team barrier per ``steps_max`` steps. When the entry
    also exports its run-to-completion twin (:func:`host_run`) and the run
    neither compacts nor maps, the whole run is ONE team pass instead
    (:func:`_run_whole`): each tile runs its samples to completion in the
    same rounds, under a dynamic schedule, or on one thread for a small
    batch (:func:`whole_run_schedule`).
  - ``"measured"`` otherwise: a ``k=1`` launch runs the vectorised
    straight-line step, a fused launch each sample's own scalar loop, so
    fusing only pays once early exit saves more than the vector lanes
    lose. The run SWEEPS and, each time the live count halves, PROBES one
    fused launch; once it beats the sweep's wall per step, it stays fused
    under the device's band (``"band"`` skips straight to it).

Tiles are the team's default (256 samples). Every launch is the team's
(:meth:`eagle.exec.HostTeam.run`); the same entry runs the same steps in
the same order as :meth:`eagle.exec.HostTeam.run_serial`, so the planes
are bit-equal to a serial run. A step the loop cannot see through runs
the loop's host arm unchanged (``runner.loop()``).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import threading
import time

from . import exec as eexec
from .plan import BoundPlan

__all__ = ["RUN_LANES", "RUN_TILES_PER_THREAD", "SWEEP_COMPACT_EVERY", "clock",
           "host_run", "host_tile", "run_host", "whole_run_schedule"]

#: The measured host policy's timer (seconds); a test replaces it to script
#: the walls the policy compares.
clock = time.perf_counter

#: Sweeps between two compactions while the measured host policy sweeps (a
#: compaction scans the whole batch, so compacting too often costs more).
SWEEP_COMPACT_EVERY = 16


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]


@functools.lru_cache(maxsize=1)
def _dladdr():
    """The C library's ``dladdr``, or ``None`` where there is none. Looked up
    once per process: ``ctypes.util.find_library`` scans the system libraries
    (a few ms) and the answer never changes."""
    # the process's own C library first; the library scan only if it lacks dladdr
    for name in (lambda: None, lambda: ctypes.util.find_library("dl")):
        try:
            fn = ctypes.CDLL(name()).dladdr
        except (OSError, AttributeError, TypeError):
            continue
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
        fn.restype = ctypes.c_int
        return fn
    return None


#: Per-entry answers of :func:`host_tile` / :func:`host_run`, cleared when
#: full so a process that loads many kernels does not grow them forever.
_TILES: dict = {}
_RUNS: dict = {}
_MEMO_LIMIT = 256
#: one lock per run entry's rounds cell: the cell is one global per artifact
_RUN_LOCKS: dict = {}


def _remember(memo: dict, entry, value):
    if len(memo) >= _MEMO_LIMIT:
        memo.clear()
    memo[entry] = value
    return value

#: The float64 lanes of one vector register the whole-run path sizes its
#: work by (AVX-512); a run of fewer than ``threads * RUN_LANES *
#: RUN_TILES_PER_THREAD`` samples runs on one thread, where a parallel
#: region would cost more than it shares out.
RUN_LANES = 8
#: The tiles per thread the whole-run path aims for, so the team's dynamic
#: schedule can even out samples that run for different step counts.
RUN_TILES_PER_THREAD = 4


def host_tile(entry) -> int:
    """The tile of ``entry``'s tiled fused-step loop, or ``0``.

    ``entry`` is a host entry's address. The answer is the
    ``<symbol>_tile`` int64 the shared object holding ``entry`` exports
    (HAWK's interchanged host loop exports its tile, and ``0`` when built
    without it); an object without that export, or a platform without
    ``dladdr``, answers ``0``. Read once per entry."""
    if not entry:
        return 0
    held = _TILES.get(entry)
    if held is not None:
        return held
    tile = 0
    dladdr = _dladdr()
    info = _DlInfo()
    try:
        if dladdr is not None and dladdr(entry, ctypes.byref(info)) \
                and info.dli_fname and info.dli_sname:
            lib = ctypes.CDLL(info.dli_fname.decode())
            symbol = info.dli_sname.decode() + "_tile"
            tile = int(ctypes.c_int64.in_dll(lib, symbol).value)
    except (OSError, ValueError, AttributeError):
        tile = 0
    return _remember(_TILES, entry, max(tile, 0))


def _symbol(entry):
    """``(library path, symbol name)`` of the host entry at ``entry``, or
    ``None`` where ``dladdr`` cannot say."""
    dladdr = _dladdr()
    info = _DlInfo()
    if dladdr is None or not dladdr(entry, ctypes.byref(info)) \
            or not info.dli_fname or not info.dli_sname:
        return None
    return info.dli_fname.decode(), info.dli_sname.decode()


def host_run(entry):
    """``(run entry, k, rounds cell)`` of ``entry``'s run-to-completion
    twin, or ``None``.

    The twin is the ``<symbol>_run`` function the shared object holding
    ``entry`` exports (HAWK's automatic host TUs do): same signature, the
    ``fused_steps`` word read as the steps LEFT in the run, each range run
    in rounds of at most ``<symbol>_run_k`` steps until its samples are
    finished, the most rounds any range ran max-folded into the int64
    ``<symbol>_run_rounds``. Read once per entry."""
    if not entry:
        return None
    if entry in _RUNS:
        return _RUNS[entry]
    found = None
    try:
        where = _symbol(entry)
        if where is not None:
            lib = ctypes.CDLL(where[0])
            name = where[1] + "_run"
            run = ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
            k = int(ctypes.c_int64.in_dll(lib, name + "_k").value)
            found = (run, k, ctypes.c_int64.in_dll(lib, name + "_rounds"))
    except (OSError, ValueError, AttributeError):
        found = None
    return _remember(_RUNS, entry, found)


#: Below this many sample-steps (samples x step budget) a host run uses one
#: thread per physical core: a run that short is stalled for a whole scheduler
#: slice when any thread of a team as wide as the logical CPUs waits behind
#: other work, and hyper-threading has nothing to win in it. At or above it the
#: team takes every logical CPU (15-45% faster on long runs with two threads
#: per core). A heuristic, like the device policy's constants.
SMT_MIN_WORK = 10_000_000


@functools.lru_cache(maxsize=1)
def physical_cores() -> int:
    """The physical cores among the CPUs this process may run on: logical
    CPUs grouped by the sibling list the kernel publishes (Linux sysfs);
    elsewhere, or if that is unreadable, the logical CPU count."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return max(os.cpu_count() or 1, 1)
    cores = set()
    for cpu in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                cores.add(f.read().strip())
        except OSError:
            return len(cpus)
    return max(len(cores), 1)


def host_team_size(n: int, max_steps: int, threads: int | None = None) -> int:
    """The team a host run of ``n`` samples with step budget ``max_steps``
    uses: ``threads`` (default :func:`_threads`) capped at the physical cores
    when ``n * max_steps`` is below :data:`SMT_MIN_WORK`."""
    threads = _threads() if threads is None else threads
    if n * max(max_steps, 1) < SMT_MIN_WORK:
        return max(min(threads, physical_cores()), 1)
    return threads


@functools.lru_cache(maxsize=1)
def _omp_runtime():
    """``(omp_get_max_threads, omp_set_num_threads)`` of the OpenMP runtime
    eagle's host code runs on (found among the loaded libraries), or
    ``None`` where there is none to steer."""
    from . import _core  # noqa: F401  (loads the runtime eagle links)

    try:
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if line.rstrip().endswith(".so")
                     or ".so." in line}
    except OSError:
        paths = set()
    for path in sorted(paths):
        name = os.path.basename(path)
        if name.startswith(("libgomp", "libomp", "libiomp")):
            try:
                lib = ctypes.CDLL(path)
                get, put = lib.omp_get_max_threads, lib.omp_set_num_threads
            except (OSError, AttributeError):
                continue
            get.restype, get.argtypes = ctypes.c_int, []
            put.restype, put.argtypes = None, [ctypes.c_int]
            return get, put
    return None


def _threads() -> int:
    """The threads the host team runs with: ``$OMP_NUM_THREADS``, else the
    CPUs this process may run on."""
    try:
        wanted = int(os.environ.get("OMP_NUM_THREADS", "").split(",")[0])
    except ValueError:
        wanted = 0
    if wanted > 0:
        return wanted
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(os.cpu_count() or 1, 1)


@functools.lru_cache(maxsize=256)
def whole_run_schedule(n: int, threads: int) -> tuple:
    """``(serial, bytes_per_sample)`` for one whole run over ``n`` samples:
    serial below ``threads * RUN_LANES * RUN_TILES_PER_THREAD`` samples,
    else the team's tile sized (through its ``bytes_per_sample`` estimate)
    to about ``RUN_TILES_PER_THREAD`` tiles per thread, never above the
    team's default."""
    if n < threads * RUN_LANES * RUN_TILES_PER_THREAD:
        return True, 0
    target = n // (threads * RUN_TILES_PER_THREAD)
    size = eexec.HostTeam.tile_size
    if size(0) <= target:
        return False, 0
    per = 64
    while size(per) > target and per < (1 << 30):
        per *= 2
    return False, per


def host_policy(runner) -> str:
    """The pacing a host run of ``runner`` follows: its
    :attr:`~eagle.Runner.host_policy`, with ``"auto"`` resolved to
    ``"tiled"`` when the step's entry is tiled (:func:`host_tile`), else
    ``"measured"``."""
    policy = runner.host_policy
    if policy != "auto":
        return policy
    step = runner.step
    return "tiled" if _direct(step) and host_tile(step._entry) else "measured"


def _direct(step) -> bool:
    """Whether ``step`` is a bound host plan this loop may launch
    directly: one partition, no staging copies."""
    return (type(step) is BoundPlan and not step._copies
            and len(step.partitions) == 1)


def _team(runner) -> int:
    """The team size of ``runner``'s host runs."""
    return runner.host_threads or host_team_size(runner.n, runner._max_steps)


def run_host(runner) -> None:
    """Run ``runner.loop`` on the host team until the guard fails or the
    cap is reached, writing the loop's ``[remaining, ran]`` cell. The team
    is :attr:`~eagle.Runner.host_threads` when set, else
    :func:`host_team_size`; the caller's OpenMP setting is restored after."""
    team = _team(runner)
    omp = _omp_runtime()
    if omp is None:
        _run_host_loop(runner, team)
        return
    get, put = omp
    before = get()
    put(team)
    try:
        _run_host_loop(runner, team)
    finally:
        put(before)


def _run_host_loop(runner, threads: int) -> None:
    loop = runner.loop
    step = runner.step
    if not _direct(step):
        loop()
        return
    entry, addrs, n = step._entry, step._addrs, step.n
    team = eexec.HostTeam
    guard = loop.guard
    finished, at = guard.count_arr, guard.count_idx
    base = guard.baseline_arr
    total = 0 if base is None else int(base[guard.baseline_idx])
    cap = loop.max_iters
    count = step._active_count
    ran = 0
    policy = host_policy(runner) if getattr(runner, "auto", False) else None
    if policy == "measured":
        ran = _run_measured(runner, entry, addrs, n, count)
    elif policy == "tiled":
        run = host_run(entry)
        whole_run = (run is not None and count is None and run[1] == runner._k_max
                     and not runner.loop.parts[1:-1])
        ran = (_run_whole(runner, run, addrs) if whole_run
               else _run_tiled(runner, entry, addrs, n, count))
    elif getattr(runner, "auto", False):
        tails = loop.parts[1:]
        whole = step.partitions[0]
        while ran < cap and int(finished[at]) != total:  # the go cell
            if count is None:
                team.run(entry, addrs, whole)
            else:
                live = min(int(count[0]), n)
                if live > 0:
                    team.run(entry, addrs, eexec.Partition(0, live, n))
            for tail in tails:
                tail()
            ran += 1
    elif count is None:
        whole = step.partitions[0]
        while ran < cap and int(finished[at]) != total:
            team.run(entry, addrs, whole)
            ran += 1
    else:
        every = runner.launches_per_compaction
        tails = loop.parts[1:]
        while ran < cap and int(finished[at]) != total:
            for _ in range(every):
                live = min(int(count[0]), n)
                if live <= 0 or int(finished[at]) == total:
                    break
                team.run(entry, addrs, eexec.Partition(0, live, n))
            for tail in tails:
                tail()
            ran += 1
    loop._cell[0] = max(cap - ran, 0)
    loop._cell[1] = ran


def _run_tiled(runner, entry, addrs, n, count) -> int:
    """The tiled host policy of an automatic artifact (see the module
    docstring): every launch asks for ``steps_max`` steps, the last cut to
    the cap, so no sample exceeds it. Leaves the runner's policy/``go``/
    word cells and ``launches_by_k`` histogram as the band does. Returns
    the loop iterations run."""
    from ._until_done import _CAP, _DONE, _FIN_AT, _K

    team = eexec.HostTeam
    cells, go, hist, word = runner._cells, runner._go, runner._hist, runner.fused_steps
    finished, total, k_max = runner.finished, runner.n, runner._k_max
    compact = runner.loop.parts[1:-1]   # compaction (+ reorder): never the policy
    whole = runner.step.partitions[0]
    if not go[0]:
        return 0
    hist[int(word[0])] -= 1             # the seed counted its first word
    fin = int(cells[_FIN_AT])
    done = int(cells[_DONE])
    left = int(cells[_CAP]) - done
    ran = 0
    k = int(word[0])
    while fin != total and left > 0:
        k = min(k_max, left)
        word[0] = k
        hist[k] += 1
        if count is None:
            team.run(entry, addrs, whole)
        else:
            live = min(int(count[0]), n)
            if live > 0:
                team.run(entry, addrs, eexec.Partition(0, live, n))
        for part in compact:
            part()
        fin = int(finished[0])
        done += k
        left -= k
        ran += 1
    cells[_FIN_AT], cells[_DONE], cells[_K] = fin, done, k
    go[0] = 0
    return ran


def _run_whole(runner, run, addrs) -> int:
    """The tiled policy as ONE pass of the team over the batch: the run
    entry (:func:`host_run`) gets the steps left in its word and runs each
    range to completion in the rounds the per-launch loop of
    :func:`_run_tiled` would have run, so every sample's results are the
    same bits, with one parallel region (or none, for a small batch:
    :func:`whole_run_schedule`) instead of one per launch. Leaves the
    policy/``go``/word cells, the ``launches_by_k`` histogram and the loop
    iterations exactly as :func:`_run_tiled` does, counting one launch per
    round the busiest range ran. Returns the loop iterations run."""
    from ._until_done import _CAP, _DONE, _FIN_AT, _K

    entry, k_max, rounds = run
    team = eexec.HostTeam
    cells, go, hist, word = runner._cells, runner._go, runner._hist, runner.fused_steps
    finished, total = runner.finished, runner.n
    whole = runner.step.partitions[0]
    if not go[0]:
        return 0
    hist[int(word[0])] -= 1             # the seed counted its first word
    fin = int(cells[_FIN_AT])
    done = int(cells[_DONE])
    left = int(cells[_CAP]) - done
    ran = 0
    k = int(word[0])
    if fin != total and left > 0:
        lock = _RUN_LOCKS.setdefault(ctypes.addressof(rounds), threading.Lock())
        if not lock.acquire(blocking=False):
            raise RuntimeError(
                "eagle.until_done: this artifact is already running a host run "
                "on another thread; one host run per artifact at a time")
        try:
            rounds.value = 0
            word[0] = left
            serial, per = whole_run_schedule(int(whole.count), _team(runner))
            if serial:
                team.run_serial(entry, addrs, whole)
            else:
                team.run(entry, addrs, whole, bytes_per_sample=per)
            most = int(rounds.value)
        finally:
            lock.release()
        for _ in range(most):
            k = min(k_max, left)
            hist[k] += 1
            done += k
            left -= k
            ran += 1
        word[0] = k
        fin = int(finished[0])
    cells[_FIN_AT], cells[_DONE], cells[_K] = fin, done, k
    go[0] = 0
    return ran


def _run_measured(runner, entry, addrs, n, count) -> int:
    """The measured host policy of an automatic artifact (see the module
    docstring): sweep at ``k=1``, probe one fused launch each time the
    live count halved, stay fused under the band once a probe wins.
    Leaves the runner's policy/``go``/word cells and ``launches_by_k``
    histogram as the band does. Returns the loop iterations run."""
    from ._until_done import _CAP, _DONE, _FIN_AT, _K, AUTO_K0, AUTO_K_MIN, _next_k

    team = eexec.HostTeam
    cells, go, hist, word = runner._cells, runner._go, runner._hist, runner.fused_steps
    finished, total, k_max = runner.finished, runner.n, runner._k_max
    compact = runner.loop.parts[1:-1]   # compaction (+ reorder): never the policy
    whole = runner.step.partitions[0]
    if not go[0]:
        return 0
    hist[int(word[0])] -= 1             # the seed counted its first word
    fin = int(cells[_FIN_AT])
    done = int(cells[_DONE])
    left = int(cells[_CAP]) - done
    look = total - fin                  # the live count at the last look
    ran = 0
    k = 1

    def launch():
        if count is None:
            team.run(entry, addrs, whole)
        else:
            live = min(int(count[0]), n)
            if live > 0:
                team.run(entry, addrs, eexec.Partition(0, live, n))

    def compaction():
        for part in compact:
            part()

    # sweep one step per launch until the live count halves, then compare
    # its wall per launch with one probing fused launch's wall per step
    sweep_s = float("inf")              # the last sweep phase's wall per launch
    while True:
        word[0] = 1
        limit = total - look // 2       # fin >= limit: the live count halved
        start, since = left, 0
        t0 = clock()
        if count is None:
            while fin < limit and left > 0:
                team.run(entry, addrs, whole)
                fin = int(finished[0])
                left -= 1
        else:
            while fin < limit and left > 0:
                live = min(int(count[0]), n)
                if live > 0:
                    team.run(entry, addrs, eexec.Partition(0, live, n))
                fin = int(finished[0])
                left -= 1
                since += 1
                if compact and since >= SWEEP_COMPACT_EVERY:
                    compaction()
                    since = 0
        sweeps = start - left
        if sweeps:
            sweep_s = (clock() - t0) / sweeps
        ran += sweeps
        done += sweeps
        hist[1] += sweeps
        if sweeps:
            k = 1
        if fin == total or left <= 0:
            break
        k = min(AUTO_K_MIN, left)
        word[0] = k
        hist[k] += 1
        look = total - fin
        t0 = clock()
        launch()
        wall = clock() - t0
        fin = int(finished[0])
        ran += 1
        done += k
        left -= k
        if compact:
            compaction()
        if wall / k < sweep_s:
            break                       # fused per step beats a sweep: stay fused
    cells[_FIN_AT], cells[_DONE], cells[_K] = fin, done, AUTO_K0
    go[0] = 1 if fin != total and left > 0 else 0
    if not go[0]:
        word[0] = k
        return ran
    # fused for good: the device's band from AUTO_K0, compacting every launch
    tails = runner.loop.parts[1:]
    k = min(AUTO_K0, left)
    word[0] = k
    hist[k] += 1
    while go[0]:
        launch()
        for tail in tails:              # compaction (+ reorder) and the band
            tail()
        ran += 1
    return ran
