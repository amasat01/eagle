# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Gate A — ``eagle._launch_policy.resolve_block`` value table.

This is Gate A: pure, no GPU, no cupy — it must
collect and pass with cupy entirely ABSENT from the environment, because
:mod:`eagle._launch_policy` itself never imports cupy. Gate B (a GPU end-to-end value
twin against a real captured graph's ``<<<grid,block,smem>>>`` label) and Gate C (the
nonvacuity ritual) are authored separately and are NOT run here — they need a CUDA
device.

Every expected value below is HAND-DERIVED in the comment beside it (plain
arithmetic, independent of :func:`resolve_block`'s own expression) — per this
gate's own lock: "Gate A compares the pure function to hand-pinned constants... the
two layers may never collapse into one recomputation of the implementation."

Shared pinned device (this suite's own row-table dev, used by every S>=2 row unless a
row says otherwise): 8 SMs, 65536 regs/SM, 98304 B shared mem/SM.
"""

from __future__ import annotations

import pytest

from eagle._launch_policy import resolve_block

#: The pinned dev used throughout this file: {nSMs 8, regsPerSM 65536, smemPerSM 98304}.
DEV = {
    "multiProcessorCount": 8,
    "regsPerMultiprocessor": 65536,
    "sharedMemPerMultiprocessor": 98304,
}

#: The pinned kern for the n=512/S=4, n=1e6/S=4, and n=1/S=2 rows:
#: {regs 64, smem 0, cap 1024}.
KERN = {
    "num_regs": 64,
    "shared_size_bytes": 0,
    "max_threads_per_block": 1024,
}


# --------------------------------------------------------------------------- #
# siblings <= 1: the permanent, non-concurrent identity, but now an
# N-DEPENDENT small-N geometry rather than a flat constant —
# it still converges to exactly 256 for any batch at or above the clamp's
# own upper arm (n >= 2*SMs*256 = 4096 on the pinned 8-SM DEV), so the OLD
# large-N identity is a special case of the new formula, not replaced by it.
# --------------------------------------------------------------------------- #
def test_siblings_one_small_n_uses_the_clamp_formula():
    """Pinned row: (n=512, siblings=1, DEV{8 SMs}, KERN{regs 64, cap 1024}).

    Hand-derivation (independent of the implementation expression):
      denom = 2*SMs*32 = 2*8*32 = 512
      ideal = 32*ceil(512/512) = 32*1 = 32              (already the floor)
      clamp range = [32, min(256, floor(1024/32)*32)] = [32, 256] -> 32 stays
      reg check @ bs=32 (siblings=1): 1*32*64 = 2048 <= 65536 (regsPerSM) -> OK
    -> 32 (NOT 256: n=512 is well below the 4096 convergence point).
    """
    assert resolve_block(512, 1, DEV, KERN) == 32


def test_siblings_zero_uses_the_same_small_n_formula_as_one():
    # siblings <= 1 covers 0 too (defensive; no real caller passes 0, but the
    # formula's own "<=1" phrasing is inclusive) -- same row as n=512/siblings=1
    # above, since the small-N formula does not read `siblings` at all.
    assert resolve_block(512, 0, DEV, KERN) == 32


def test_siblings_one_large_n_converges_to_the_old_256_identity():
    """Pinned row: (n=10**6, siblings=1) -> 256 (the clamp's upper arm, same
    value :func:`resolve_block` always returned for siblings<=1 before
    the small-N geometry change -- large batches are unaffected by it).

    Hand-derivation:
      denom = 512
      ideal = 32*ceil(1_000_000/512) = 32*1954 = 62528
      clamp = min(max(62528, 32), 256) = 256                 (upper clamp hit)
      reg check @ bs=256 (siblings=1): 1*256*64 = 16384 <= 65536 -> OK
    -> 256.
    """
    assert resolve_block(10**6, 1, DEV, KERN) == 256


def test_siblings_one_small_n_is_independent_of_kern_regs_below_the_reg_limit():
    """The small-N formula's `ideal` does not read `num_regs` (only the
    final register-feasibility narrowing does) -- these garbage-kern rows at
    n=512 give the SAME 32 as the plain-KERN row above, since 1*32*regs stays
    under regsPerSM for both."""
    pinned = [
        (
            512,
            DEV,
            {"num_regs": 600, "shared_size_bytes": 0, "max_threads_per_block": 1024},
        ),
        (
            512,
            DEV,
            {"num_regs": 64, "shared_size_bytes": 60000, "max_threads_per_block": 1024},
        ),
    ]
    for n, dev, kern in pinned:
        got = resolve_block(n, 1, dev, kern)
        assert got == 32, f"siblings=1 small-N row, got {got} for n={n}"


def test_siblings_one_reg_infeasible_falls_back_to_256():
    """A register-hungry kernel at siblings=1: even the formula's own floor
    (bs=32) cannot fit one block's registers in the SM, so no warp multiple
    is feasible and the saturation fallback (256) applies -- the same
    degradation :func:`resolve_block` uses everywhere else.

    Hand-derivation: ideal/bs = 32 (as above); reg check @ bs=32:
    1*32*2048 = 65536 <= 65536 -- OK actually, so pick regs higher still:
    num_regs=2049 -> 1*32*2049 = 65568 > 65536 (regsPerSM) -> FAIL, no smaller
    warp multiple exists (32 is the floor) -> fallback.
    """
    kern = {"num_regs": 2049, "shared_size_bytes": 0, "max_threads_per_block": 1024}
    assert resolve_block(512, 1, DEV, kern) == 256


# --------------------------------------------------------------------------- #
# S>=2 pinned rows (this file's minimum set).
# --------------------------------------------------------------------------- #
def test_reg_feasible_row_n512_s4():
    """Pinned row: (n=512, S=4, kern{regs 64, smem 0, cap 1024}) -> 64.

    Hand-derivation (independent of the implementation expression):
      share = max(1, nSMs*4 // S) = max(1, 8*4 // 4) = max(1, 8) = 8
      ideal = ceil(512 / 8) = 64                      (already a warp multiple)
      clamp range = [32, min(256, floor(1024/32)*32)] = [32, 256] -> 64 stays
      reg check @ bs=64: S*bs*regs = 4*64*64 = 16384 <= 65536 (regsPerSM) -> OK
    -> 64.
    """
    assert resolve_block(512, 4, DEV, KERN) == 64


def test_upper_clamp_row_n1e6_s4():
    """Pinned row: (n=10**6, S=4) -> 256 (clamp).

    Hand-derivation:
      share = max(1, 8*4 // 4) = 8
      ideal_raw = ceil(1_000_000 / 8) = 125_000
      warp_round_up(125_000) = ceil(125_000/32)*32 = 3907*32 = 125_024
      clamp upper = min(256, floor(1024/32)*32) = min(256, 1024) = 256
      bs = min(max(125_024, 32), 256) = 256                (upper clamp hit)
      reg check @ bs=256: 4*256*64 = 65536 <= 65536 (regsPerSM)          -> OK
    -> 256 (the clamp is the *reason*; it is coincidentally also reg-feasible).
    """
    assert resolve_block(10**6, 4, DEV, KERN) == 256


def test_lower_floor_row_n1_s2():
    """Pinned row: (n=1, S=2) -> 32 (floor).

    Hand-derivation:
      share = max(1, 8*4 // 2) = max(1, 16) = 16
      ideal_raw = ceil(1 / 16) = 1
      warp_round_up(1) = 32
      clamp = min(max(32, 32), 256) = 32                    (lower clamp hit)
      reg check @ bs=32: 2*32*64 = 4096 <= 65536 (regsPerSM)           -> OK
    -> 32.
    """
    assert resolve_block(1, 2, DEV, KERN) == 32


def test_reg_infeasible_row_falls_back_to_256():
    """Reg-infeasible fallback row (the "reg-infeasible row -> 256" case; the
    exact kern is this test's own construction — this suite pins the SHAPE of the
    row, not literal numbers here).

    dev unchanged (regsPerSM=65536); kern{regs=600, smem=0, cap=1024};
    n=512, S=4.

    Hand-derivation:
      share = max(1, 8*4 // 4) = 8
      ideal = warp_round_up(ceil(512/8)) = warp_round_up(64) = 64
      clamp -> bs = 64
      reg check @ bs=64: 4*64*600 = 153_600 > 65536                  -> FAIL
      reg check @ bs=32 (the floor, the last candidate tried):
                 4*32*600 = 76_800 > 65536                            -> FAIL
      no warp multiple >= 32 is feasible -> saturation fallback.
    -> 256.
    """
    kern = {"num_regs": 600, "shared_size_bytes": 0, "max_threads_per_block": 1024}
    assert resolve_block(512, 4, DEV, kern) == 256


def test_smem_infeasible_row_falls_back_to_256():
    """Shared-memory-infeasible fallback row (the "smem-infeasible row
    (S*smem > smemPerSM) -> 256"; this test's own numbers).

    dev unchanged (smemPerSM=98304); kern{regs=64, smem=60000, cap=1024};
    n=512, S=2.

    Hand-derivation:
      S * smem = 2 * 60_000 = 120_000 > 98_304 (smemPerSM)   -> co-residency
      impossible on shared memory ALONE, independent of n/regs/clamp.
    -> 256.
    """
    kern = {"num_regs": 64, "shared_size_bytes": 60000, "max_threads_per_block": 1024}
    assert resolve_block(512, 2, DEV, kern) == 256


def test_warp_multiple_property_row_n1000_s2():
    """Warp-multiple property row: a case whose raw ceil() is NOT already a
    warp multiple, to exercise the round-up — plus an explicit
    structural assertion that the result is always a multiple of 32.

    Hand-derivation:
      share = max(1, 8*4 // 2) = 16
      ideal_raw = ceil(1000 / 16) = ceil(62.5) = 63          (not a multiple of 32)
      warp_round_up(63) = ceil(63/32)*32 = 2*32 = 64
      clamp -> bs = 64
      reg check @ bs=64: 2*64*64 = 8192 <= 65536 (regsPerSM)            -> OK
    -> 64 (and 64 % 32 == 0, the warp-multiple property).
    """
    got = resolve_block(1000, 2, DEV, KERN)
    assert got == 64
    assert got % 32 == 0


def test_result_is_always_a_warp_multiple_or_the_fallback():
    """The warp-multiple property, checked structurally (not just the one
    pinned n=1000 case above) across a spread of S>=2 rows, including both
    fallback rows (256 is itself a multiple of 32, so the property holds
    universally, fallback or not)."""
    cases = [
        (512, 4, DEV, KERN),
        (10**6, 4, DEV, KERN),
        (1, 2, DEV, KERN),
        (1000, 2, DEV, KERN),
        (7, 3, DEV, KERN),
        (99999, 5, DEV, KERN),
    ]
    for n, siblings, dev, kern in cases:
        got = resolve_block(n, siblings, dev, kern)
        assert got % 32 == 0, (
            f"resolve_block({n}, {siblings}, ...) = {got} not a warp multiple"
        )


# --------------------------------------------------------------------------- #
# "Never raises; any missing/zero attribute degrades to the 256 fallback."
# --------------------------------------------------------------------------- #
def test_missing_dev_attribute_degrades_to_fallback():
    # regsPerMultiprocessor, sharedMemPerMultiprocessor absent
    dev = {"multiProcessorCount": 8}
    assert resolve_block(512, 4, dev, KERN) == 256


def test_missing_kern_attribute_degrades_to_fallback():
    kern = {"num_regs": 64}  # shared_size_bytes, max_threads_per_block absent
    assert resolve_block(512, 4, DEV, kern) == 256


def test_zero_valued_attribute_degrades_to_fallback():
    dev = dict(DEV, regsPerMultiprocessor=0)
    assert resolve_block(512, 4, dev, KERN) == 256


def test_none_dev_and_kern_degrade_to_fallback_without_raising():
    assert resolve_block(512, 4, None, None) == 256


def test_empty_dict_dev_and_kern_degrade_to_fallback_without_raising():
    assert resolve_block(512, 4, {}, {}) == 256


def test_wrong_typed_dev_and_kern_never_raise():
    # Not dict, not an object with the expected attributes at all — must
    # degrade, never raise (the "never raises" clause is unconditional).
    assert resolve_block(512, 4, "not-a-dev", 12345) == 256


def test_non_numeric_n_or_siblings_never_raise():
    assert resolve_block("not-a-number", 4, DEV, KERN) == 256
    assert resolve_block(512, "not-a-number", DEV, KERN) == 256


def test_siblings_one_degrades_to_fallback_without_raising_too():
    # The same degradation rows as above, now through the siblings<=1 small-N
    # branch (not just siblings>=2): missing dev/kern never raises.
    assert resolve_block(512, 1, None, None) == 256
    assert resolve_block(512, 1, {}, {}) == 256
    assert resolve_block("not-a-number", 1, DEV, KERN) == 256


# --------------------------------------------------------------------------- #
# The `max_threads_per_block` clamp (`_launch_policy.py`
# :135, `upper = min(_FALLBACK_BLOCK, _warp_floor(max_threads))`) is otherwise
# untested here -- every row above pins `cap=1024` (KERN), so `min(256, ...)`
# always picks the 256 side and the small-cap branch never fires. This row
# uses a cap SMALLER than 256 so the cap itself is the thing being clamped to.
# --------------------------------------------------------------------------- #
def test_max_threads_per_block_clamp_row_small_cap():
    """kern{regs 8, smem 0, cap 64}; n=10000, S=4; dev = the shared pinned DEV.

    Hand-derivation (independent of the implementation expression):
      share = max(1, nSMs*4 // S) = max(1, 8*4 // 4) = 8
      ideal_raw = ceil(10000 / 8) = 1250
      warp_round_up(1250) = ceil(1250/32)*32 = 40*32 = 1280
      clamp upper = min(256, floor(64/32)*32) = min(256, 64) = 64   (the
        small cap wins over the usual 256 ceiling -- this is the branch this
        row exists to exercise)
      bs = min(max(1280, 32), 64) = 64
      reg check @ bs=64: S*bs*regs = 4*64*8 = 2048 <= 65536 (regsPerSM) -> OK
    -> 64.

    (If the cap were ignored -- i.e. `upper` fell back to `_FALLBACK_BLOCK`
    unconditionally -- bs would clamp to 256 instead: 4*256*8=8192 <= 65536
    is also reg-feasible, so a cap-blind implementation would silently
    return 256 here instead of 64, and this assertion would catch it.)
    """
    kern = {"num_regs": 8, "shared_size_bytes": 0, "max_threads_per_block": 64}
    assert resolve_block(10000, 4, DEV, kern) == 64


def test_object_attribute_style_dev_and_kern_also_work():
    """``resolve_block`` reads mapping OR attribute style (the caching layer leaves the
    exact ``dev``/``kern`` shape to the caller — :mod:`eagle.launch` caches a
    cupy dict today, but the resolver itself must not assume that)."""

    class Obj:
        pass

    dev = Obj()
    dev.multiProcessorCount = 8
    dev.regsPerMultiprocessor = 65536
    dev.sharedMemPerMultiprocessor = 98304
    kern = Obj()
    kern.num_regs = 64
    kern.shared_size_bytes = 0
    kern.max_threads_per_block = 1024
    # Same as test_reg_feasible_row_n512_s4's hand-derivation -> 64.
    assert resolve_block(512, 4, dev, kern) == 64


# --------------------------------------------------------------------------- #
# The no-active-set-rebuild rule's capacity: C = SMs * resident threads at
# the kernel's registers (eagle._launch_policy.capacity / resident_threads_per_sm).
# --------------------------------------------------------------------------- #
def test_capacity_row_regs_limited():
    """regsPerSM=65536, num_regs=64 -> 65536//64=1024 threads/SM, no
    maxThreadsPerMultiProcessor given (0, degrades out of the min()); 8 SMs
    -> C = 8*1024 = 8192."""
    from eagle._launch_policy import capacity, resident_threads_per_sm

    assert resident_threads_per_sm(DEV, KERN) == 1024
    assert capacity(DEV, KERN) == 8 * 1024


def test_capacity_row_clamped_by_max_resident_threads():
    """The same regs budget (65536//64=1024/SM) but a device reporting a
    SMALLER maxThreadsPerMultiProcessor (512) clamps resident threads to
    that instead -- the P2000's real ratio (2048 max vs a looser reg bound)
    the other direction, exercised here with a tighter cap to isolate the
    clamp."""
    from eagle._launch_policy import capacity, resident_threads_per_sm

    dev = dict(DEV, maxThreadsPerMultiProcessor=512)
    assert resident_threads_per_sm(dev, KERN) == 512
    assert capacity(dev, KERN) == 8 * 512


def test_capacity_is_zero_when_properties_are_unavailable():
    from eagle._launch_policy import capacity, resident_threads_per_sm

    assert resident_threads_per_sm({}, {}) == 0
    assert capacity({}, {}) == 0
    assert capacity(None, None) == 0


def test_capacity_floors_to_a_warp_multiple():
    """num_regs chosen so regsPerSM // num_regs is NOT already a warp
    multiple (65536 // 100 = 655, not a multiple of 32): resident threads
    floors to 640 (20*32), not the raw 655."""
    from eagle._launch_policy import resident_threads_per_sm

    kern = dict(KERN, num_regs=100)
    assert resident_threads_per_sm(DEV, kern) == 640


# --------------------------------------------------------------------------- #
# latency_regime_capacity: the SAFE no-rebuild bound (min of residency and
# the issue-pipe knee) -- residency alone regressed a measured GPU row (see
# its docstring), which is why this is a separate, smaller bound.
# --------------------------------------------------------------------------- #
def test_latency_regime_capacity_is_the_p2000_row():
    """DEV/KERN pinned here reproduce the P2000 oscillator row that
    motivated this function: capacity()=8192 (DEV/KERN's regs-limited
    residency, see test_capacity_row_regs_limited), SMs*32*w_knee=8*32*8=2048
    -- the knee bound wins (2048 < 8192)."""
    from eagle._launch_policy import latency_regime_capacity

    assert latency_regime_capacity(DEV, KERN, w_knee=8) == 2048


def test_latency_regime_capacity_falls_back_to_residency_when_smaller():
    """A device/kernel pair whose residency ceiling is SMALLER than the
    issue-pipe knee bound (a tight regsPerSM: 2048 // 64 regs = 32
    threads/SM, capacity = 8*32 = 256 < the knee bound 2048): residency
    wins instead."""
    from eagle._launch_policy import latency_regime_capacity

    dev = dict(DEV, regsPerMultiprocessor=2048)
    assert latency_regime_capacity(dev, KERN, w_knee=8) == 8 * 32


def test_latency_regime_capacity_default_w_knee_matches_the_constant():
    from eagle._launch_policy import W_KNEE_DEFAULT, latency_regime_capacity

    assert W_KNEE_DEFAULT == 8
    assert latency_regime_capacity(DEV, KERN) == latency_regime_capacity(
        DEV, KERN, w_knee=W_KNEE_DEFAULT)


def test_latency_regime_capacity_is_zero_when_unavailable():
    from eagle._launch_policy import latency_regime_capacity

    assert latency_regime_capacity({}, {}) == 0
    assert latency_regime_capacity(None, None) == 0


# --------------------------------------------------------------------------- #
# Persistent launch: threads per SM
# --------------------------------------------------------------------------- #
P2000 = dict(DEV, major=6, minor=1, singleToDoublePrecisionPerfRatio=32,
             maxThreadsPerMultiProcessor=2048)
STEP64 = dict(KERN, num_regs=43)
STEP32 = dict(KERN, num_regs=30)


@pytest.mark.parametrize("scalar_type, kern, n, want", [
    # float64 on a 1:32 part: 8 warps/SM cover the chain; a large batch takes
    # one more warp per partition once each lane has >= 30 samples
    ("float64", STEP64, 10_000, 256), ("float64", STEP64, 60_000, 256),
    ("float64", STEP64, 100_000, 384), ("float64", STEP64, 1_000_000, 384),
    # float32 issues 32x faster: the chain needs 640, the fetch stalls 1408
    ("float32", STEP32, 10_000, 640), ("float32", STEP32, 30_000, 928),
    ("float32", STEP32, 1_000_000, 1408)])
def test_persistent_threads_per_sm_on_the_p2000(scalar_type, kern, n, want):
    """The rows measured on a Quadro P2000 (the rule's constants come from
    that card's float64 and float32 sweeps)."""
    from eagle._launch_policy import persistent_threads_per_sm

    assert persistent_threads_per_sm(P2000, kern, n, 1000, scalar_type) == want


def test_persistent_threads_follow_the_device_and_the_step_budget():
    """A wider FP64 part needs more warps for the same step; a short step
    budget fetches samples more often, so it needs more warps to hide it."""
    from eagle._launch_policy import persistent_threads_per_sm

    a100 = dict(P2000, multiProcessorCount=108, major=8, minor=0,
                singleToDoublePrecisionPerfRatio=2)
    big = 10 ** 9
    assert persistent_threads_per_sm(a100, STEP64, big, 1000, "float64") == 768
    assert persistent_threads_per_sm(a100, STEP32, big, 1000, "float32") == 896
    assert (persistent_threads_per_sm(P2000, STEP64, big, 100, "float64")
            > persistent_threads_per_sm(P2000, STEP64, big, 1000, "float64"))


def test_persistent_threads_stay_within_the_registers():
    from eagle._launch_policy import persistent_threads_per_sm, resident_threads_per_sm

    heavy = dict(STEP32, num_regs=128)
    got = persistent_threads_per_sm(P2000, heavy, 10 ** 9, 1000, "float32")
    assert got == resident_threads_per_sm(P2000, heavy) == 512


def test_persistent_launch_splits_into_blocks_of_at_most_512():
    from eagle._launch_policy import persistent_launch

    assert persistent_launch(P2000, STEP64, 1_000_000, 1000) == (384, 8)
    assert persistent_launch(P2000, STEP32, 1_000_000, 1000, "float32") == (352, 32)
    assert persistent_launch(P2000, STEP32, 10_000, 1000, "float32") == (320, 16)


def test_persistent_launch_falls_back_without_device_properties():
    from eagle._launch_policy import persistent_geometry, persistent_launch

    assert persistent_launch({}, {}, 10 ** 6, 1000) == persistent_geometry({}, {})
