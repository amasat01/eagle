"""The fast-path entry pick, driven on the host with scripted walls.

Above capacity the first run takes the fused entry, the second the persist
entry, and the pick then moves only when the other entry is 2 % faster. No
device is needed: the pick and its bookkeeping are plain Python.
"""
import eagle._until_done as ud


def _runner(pick="persist"):
    r = ud.Runner.__new__(ud.Runner)
    r._fast_pick = pick
    r._fast_mode = pick
    r._fast_wall = {}
    r._fast_prior = None
    r._fast_last = None
    r._fast_runs = 0
    r._fast_prep = {}
    r._fast_fit = None
    return r


def _step(r, walls, readback=(10, 5), f=1.0):
    """One run: pick, time it as the real run would, fold it in."""
    mode, probing = r._pick_entry()
    timed = r._carries_timing(probing)
    r._note_run(mode, walls[mode] if timed else None, *readback, f)
    return mode, probing, timed


def test_run_one_is_the_size_pick_and_the_other_entry_is_probed_at_run_two():
    for size_pick, other in (("persist", "fused_one"), ("fused_one", "persist")):
        r = _runner(size_pick)
        walls = {"fused_one": 2.0, "persist": 1.0}
        runs = [_step(r, walls) for _ in range(4)]
        assert runs[0] == (size_pick, False, True)  # real work: never a probe
        assert runs[1] == (other, True, True)
        assert set(r._fast_wall) == {"fused_one", "persist"}
        assert runs[-1][1:] == (False, False) and runs[-2][1:] == (False, False)


def test_settled_runs_carry_no_timing_on_an_unchanged_readback():
    r = _runner("fused_one")
    walls = {"fused_one": 1.0, "persist": 2.0}
    runs = [_step(r, walls) for _ in range(6)]
    assert all(not t for _, _, t in runs[2:]) and all(m == "fused_one" for m, _, _ in runs[2:])


def test_below_capacity_pick_is_unchanged():
    r = _runner(None)
    r._fast_mode = "persist"
    assert r._pick_entry() == ("persist", False)
    assert r._pick_entry() == ("persist", False)


def test_a_new_readback_times_both_again():
    r = _runner()
    walls = {"fused_one": 1.0, "persist": 2.0}
    for _ in range(3):
        _step(r, walls)
    runs = [_step(r, walls, readback=(7, 5)) for _ in range(4)]
    # the changed run only clears the walls; the next two time both entries
    assert [x[2] for x in runs] == [False, True, True, False]
    assert runs[1][1] and set(r._fast_wall) == {"fused_one", "persist"}


def test_tie_hysteresis():
    for ratio, moves in ((0.99, False), (0.97, True)):
        r = _runner("persist")
        walls = {"persist": 1.0, "fused_one": ratio}
        for _ in range(4):
            _step(r, walls)
        # the other entry (fused) must be under 0.98 x the pick's wall to take over
        assert (r._fast_pick == "fused_one") is moves
