# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The replay envelope.

Two mechanisms in ``eagle/pipeline.py`` are gated here:

* **Cached replay events.** ``GraphPipeline.launch`` reuses an
  INSTANCE-CACHED event pair (``eagle._stream.Event``) (two, on the common path) across
  calls, re-recording rather than reallocating them on the
  replay-critical path. This probe established that re-recording a cached event is legal
  (``cudaStreamWaitEvent`` SNAPSHOTS the event at the wait, so a later record
  cannot retroactively change an already-issued wait).
* **``enqueue()`` / ``synchronize()``.** ``launch()`` is exactly
  those two, in that order, and the pair is public so several pipelines can be
  in flight at once. ``enqueue`` documents the one-in-flight rule and the
  REVERSE hazard an unconditional sync would hide.

**How this file tests them WITHOUT a device.** ``GraphPipeline.enqueue`` /
``synchronize`` reach events and the caller's stream only through
``eagle._stream`` (``Event``, ``current``), and touch the outside world only through
``self._sptr`` (the pipeline's own stream) and ``self._launcher``. So a pipeline
built with ``object.__new__`` (no capture, no driver) plus stand-in ``Event`` and
``current`` in ``eagle._stream`` exercises the REAL methods and records the exact
sequence of event records, stream waits and launcher calls they issue. That is
the whole ordering contract, observable, on a host-only worker.

The device legs (real replay parity, real overlap) are marked ``gpu`` at the
bottom and need a CUDA device to run.

Every counter/ordering gate here carries an executable RED leg: the same
assertion helper is run against a DEFEATED reimplementation of the method under
test, and must reject it.
"""

from __future__ import annotations


import pytest

from eagle.compose import GraphComposer, _Member
from eagle.pipeline import GraphPipeline

# --------------------------------------------------------------------------- #
# The device-free stand-ins.
# --------------------------------------------------------------------------- #

_LEGACY_PTR = 1  # the legacy default stream, as eagle spells it

#: stub pipeline handle -> ``(log, tag)``: where a stand-in event's ``wait`` logs.
_STUB_STREAMS: dict = {}


def _event_class():
    """A fresh Event class per test, so its construction counter is not shared
    between tests (the counter IS the cached-event measurement)."""

    class _FakeEvent:
        constructed = 0

        def __init__(self, device=-1):
            _FakeEvent.constructed += 1
            self.uid = _FakeEvent.constructed
            self.records = []  # stream handles this event was recorded on

        def record(self, stream):
            self.records.append(int(stream))

        def wait(self, stream):
            log, tag = _STUB_STREAMS[stream]
            log.append(("wait", tag, self.uid))

    return _FakeEvent


class _FakeLauncher:
    def __init__(self, log, tag="ext"):
        self.log = log
        self.tag = tag

    def launch(self):
        self.log.append(("launch", self.tag))

    def synchronize(self):
        self.log.append(("sync", self.tag))


#: stub pipeline handle -> ``(log, tag)``: what a stand-in ``ExternalStream`` logs into.
_STUB_STREAMS: dict = {}


def _stub_pipeline(log, tag="ext"):
    """A ``GraphPipeline`` with exactly the state ``enqueue``/``synchronize``
    read — built without ``__init__`` so no stream, graph or driver is
    needed. The METHODS under test are the shipped ones."""
    pipe = GraphPipeline.__new__(GraphPipeline)
    handle = 4242 + len(_STUB_STREAMS)
    _STUB_STREAMS[handle] = (log, tag)
    pipe._sptr = handle
    pipe._launcher = _FakeLauncher(log, tag)
    pipe._seeds = []
    return pipe


def _pair(pipe):
    """``(seed_event, legacy_event)`` of a stub pipeline, ``None`` for one it never made."""
    seeds = pipe._seeds + [None, None]
    return seeds[0], seeds[1]


@pytest.fixture
def fake_cupy(monkeypatch):
    """Stand-ins for ``eagle._stream.Event`` and ``eagle._stream.current``; ``install(current)`` returns the
    Event class whose ``constructed`` counter the gates read. A scope opened on a stub pipeline's stream logs
    its enter and exit."""
    import contextlib

    from eagle import _core, _stream, pipeline

    def install(current_ptr=7777):
        event_cls = _event_class()
        monkeypatch.setattr(_stream, "Event", event_cls)
        monkeypatch.setattr(_stream, "current", lambda device=None: current_ptr)

        @contextlib.contextmanager
        def scoped(handle):
            log, tag = _STUB_STREAMS[handle]
            log.append(("stream_enter", tag))
            try:
                yield
            finally:
                log.append(("stream_exit", tag))

        monkeypatch.setattr(pipeline, "scoped_stream", scoped)
        monkeypatch.setattr(_core, "stream_is_capturing", lambda stream: False)   # no capture open
        return event_cls

    return install


# --------------------------------------------------------------------------- #
# The events are allocated ONCE per pipeline.
# --------------------------------------------------------------------------- #
def _assert_events_are_cached(event_cls, replays: int) -> None:
    """THE event-identity gate, factored out so the RED leg can be run through the
    SAME assertion. At most one seed event + one legacy event may ever be
    constructed by one pipeline, no matter how many times it replays."""
    assert event_cls.constructed <= 2, (
        f"{event_cls.constructed} cupy Events were constructed across {replays} "
        "replays of ONE pipeline — the event pair is not instance-cached "
        "(a per-call-allocating implementation allocates 2 per call)"
    )


def test_launch_allocates_its_event_pair_exactly_once(fake_cupy):
    event_cls = fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)

    for _ in range(5):
        pipe.launch(1)

    _assert_events_are_cached(event_cls, replays=5)
    assert event_cls.constructed == 2, (
        "the common path (current stream != legacy) uses BOTH events; both "
        "should have been created on the first launch"
    )
    # ...and the SAME objects are re-recorded every call.
    assert [len(e.records) for e in _pair(pipe)] == [5, 5]


def test_the_cached_events_are_the_same_objects_across_launches(fake_cupy):
    fake_cupy()
    pipe = _stub_pipeline([])
    pipe.launch(1)
    first_seed, first_legacy = _pair(pipe)
    pipe.launch(1)
    assert _pair(pipe) == (first_seed, first_legacy) and all(e is not None for e in _pair(pipe))


def test_legacy_event_is_never_created_when_the_caller_is_on_the_legacy_stream(
    fake_cupy,
):
    """The branch that skips the legacy event must also skip ALLOCATING it —
    a pipeline only ever pays for the events it actually uses."""
    event_cls = fake_cupy(current_ptr=_LEGACY_PTR)
    pipe = _stub_pipeline([])
    for _ in range(4):
        pipe.launch(1)
    assert event_cls.constructed == 1
    assert _pair(pipe)[1] is None


def test_red_leg_a_per_call_allocating_launch_fails_the_event_identity_gate(
    fake_cupy,
):
    """RED LEG, executable. The per-call-allocating body, replayed verbatim, must be
    REJECTED by the same assertion the gate above passes — otherwise that gate
    is vacuous and would have gone green before the change too."""
    event_cls = fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)

    def _defeated_launch(self, n=1):
        from eagle import _stream

        current = _stream.current()
        seed_event = _stream.Event()  # <- fresh, every call: the defect
        seed_event.record(current)
        seed_event.wait(self._sptr)
        if current != _stream.LEGACY:
            legacy_event = _stream.Event()
            legacy_event.record(_stream.LEGACY)
            legacy_event.wait(self._sptr)
        for _ in range(int(n)):
            self._launcher.launch()
        self._launcher.synchronize()

    for _ in range(5):
        _defeated_launch(pipe)

    with pytest.raises(AssertionError, match="not instance-cached"):
        _assert_events_are_cached(event_cls, replays=5)


# --------------------------------------------------------------------------- #
# The ordering contract of enqueue()/synchronize()/launch().
# --------------------------------------------------------------------------- #
def _assert_u1_ordering(
    log, *, tag="ext", n=1, with_hook=False, require_no_sync=True
) -> None:
    """THE ordering gate, factored out so the
    RED legs run through the SAME assertion.

    Required, in order: BOTH seed waits on this pipeline's own stream, then the
    optional ``pre_launch`` hook inside that stream's context, then exactly
    ``n`` launches. ``synchronize`` must NOT appear (this is ``enqueue``)."""
    waits = [e for e in log if e[0] == "wait" and e[1] == tag]
    assert len(waits) == 2, (
        f"expected BOTH the seed waits on stream {tag!r}, saw {len(waits)}: {log}"
    )
    launches = [i for i, e in enumerate(log) if e == ("launch", tag)]
    assert len(launches) == n, f"expected {n} launch(es), got {len(launches)}: {log}"
    last_wait = max(i for i, e in enumerate(log) if e[0] == "wait" and e[1] == tag)
    assert last_wait < launches[0], (
        f"a replay was enqueued BEFORE the seed waits were issued: {log}"
    )
    if with_hook:
        hook = log.index(("hook", tag))
        assert last_wait < hook < launches[0], (
            f"the pre_launch hook must run AFTER the seed waits and BEFORE the "
            f"replays: {log}"
        )
        enter = log.index(("stream_enter", tag))
        exit_ = log.index(("stream_exit", tag))
        assert enter < hook < exit_, (
            f"the pre_launch hook must run INSIDE this member's own stream "
            f"context: {log}"
        )
    if require_no_sync:
        assert ("sync", tag) not in log, (
            f"enqueue() must not synchronize — that is synchronize()'s job: {log}"
        )
    else:
        # The two-pass (compose) shape: a sync may appear, but only AFTER this
        # member's own replays have all been enqueued.
        assert all(
            i > launches[-1] for i, e in enumerate(log) if e == ("sync", tag)
        ), f"member {tag!r} was synchronized mid-enqueue: {log}"


def test_enqueue_orders_the_replay_after_both_seed_events_and_does_not_sync(
    fake_cupy,
):
    fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)
    pipe.enqueue(3)
    _assert_u1_ordering(log, n=3)


def test_enqueue_runs_the_pre_launch_hook_on_its_own_stream_between_wait_and_launch(
    fake_cupy,
):
    fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)
    pipe.enqueue(1, pre_launch=lambda: log.append(("hook", "ext")))
    _assert_u1_ordering(log, n=1, with_hook=True)


def test_enqueue_with_seed_events_records_nothing_of_its_own(fake_cupy):
    """The shared-event mode: given events, ``enqueue`` waits them and records
    NO event of its own (that is what lets one pair cover N pipelines)."""
    event_cls = fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)
    shared = [event_cls(), event_cls()]
    before = event_cls.constructed
    pipe.enqueue(1, seed_events=shared)
    assert event_cls.constructed == before, "enqueue recorded its own events anyway"
    assert _pair(pipe) == (None, None)
    assert [e for e in log if e[0] == "wait"] == [
        ("wait", "ext", shared[0].uid),
        ("wait", "ext", shared[1].uid),
    ]


def test_an_empty_seed_events_sequence_means_no_seed_waits(fake_cupy):
    """``seed_events=`` is meaningful and distinct from ``None``: the caller
    is taking the ordering on themselves. It must record nothing and
    wait nothing — never silently fall back to recording its own pair, which
    would make the opt-out look like it worked while doing the opposite."""
    event_cls = fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)
    pipe.enqueue(2, seed_events=())
    assert event_cls.constructed == 0
    assert [e for e in log if e[0] == "wait"] == []
    assert log == [("launch", "ext"), ("launch", "ext")]


def test_launch_is_enqueue_then_synchronize_verbatim(fake_cupy):
    """Two FRESH pipelines, same stand-in cupy: one driven through
    ``enqueue(2)``, one through ``launch(2)``. Event uids are normalized away
    (each pipeline gets its own pair, so the raw ids differ by construction);
    everything else must match exactly, with one trailing ``sync``."""

    def _shape(log):
        return [(e[0], e[1]) if e[0] == "wait" else e for e in log]

    fake_cupy()
    enq_log: list = []
    launch_log: list = []
    _stub_pipeline(enq_log).enqueue(2)
    _stub_pipeline(launch_log).launch(2)
    expected = _shape(enq_log) + [("sync", "ext")]
    assert _shape(launch_log) == expected, (
        "launch() must be exactly enqueue() followed by synchronize(): "
        f"{_shape(launch_log)} != {expected}"
    )


def test_launch_still_returns_self_and_synchronize_does_too(fake_cupy):
    fake_cupy()
    pipe = _stub_pipeline([])
    assert pipe.launch(1) is pipe
    assert pipe.enqueue(1) is pipe
    assert pipe.synchronize() is pipe


def test_enqueue_and_synchronize_refuse_before_build(fake_cupy):
    fake_cupy()
    pipe = GraphPipeline.__new__(GraphPipeline)
    pipe._launcher = None
    with pytest.raises(RuntimeError, match="call build"):
        pipe.enqueue()
    with pytest.raises(RuntimeError, match="call build"):
        pipe.synchronize()


def test_red_leg_dropping_a_seed_wait_fails_the_ordering_gate(fake_cupy):
    """RED LEG, executable: an ``enqueue`` that waits only the current-stream
    event (the legacy-stream wait dropped) must be
    REJECTED by the same ordering assertion."""
    fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)

    def _defeated_enqueue(self, n=1):
        from eagle import _stream

        events = _stream.seed_events(self._seeds)
        events[0].wait(self._sptr)  # <- the second wait is DROPPED
        for _ in range(int(n)):
            self._launcher.launch()

    _defeated_enqueue(pipe, 1)
    with pytest.raises(AssertionError, match="expected BOTH the seed waits"):
        _assert_u1_ordering(log, n=1)


def test_red_leg_launching_before_the_waits_fails_the_ordering_gate(fake_cupy):
    """RED LEG, executable: waits issued AFTER the replay (a reordering that
    still records and still waits, so a naive count-only gate would pass)."""
    fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)

    def _defeated_enqueue(self, n=1):
        from eagle import _stream

        events = _stream.seed_events(self._seeds)
        for _ in range(int(n)):
            self._launcher.launch()  # <- enqueued BEFORE the waits
        for event in events:
            event.wait(self._sptr)

    _defeated_enqueue(pipe, 1)
    with pytest.raises(AssertionError, match="BEFORE the seed waits"):
        _assert_u1_ordering(log, n=1)


# --------------------------------------------------------------------------- #
# The documented hazards, demonstrated executably.
# --------------------------------------------------------------------------- #
def test_the_documented_enqueue_then_write_discipline(fake_cupy):
    """DOC-TEST for ``enqueue``'s HAZARD 2 (the reverse write fence the
    unconditional sync used to provide, silently).

    ``launch()`` returned only after ``synchronize()``, so a caller could
    overwrite a device buffer the graph reads the instant it returned.
    ``enqueue()`` returns with the replay IN FLIGHT, and the seed events order
    replay-AFTER-seeds only — they say nothing about writes issued after the
    enqueue. This pins both halves of that: the UNSAFE interleaving really does
    put the write between the launch and the sync (no fence in between), and
    the documented discipline really does put it after the sync."""
    fake_cupy()

    unsafe: list = []
    pipe = _stub_pipeline(unsafe)
    pipe.enqueue(1)
    unsafe.append(("device_write", "ext"))  # HAZARD: replay still in flight
    pipe.synchronize()
    w = unsafe.index(("device_write", "ext"))
    assert unsafe.index(("launch", "ext")) < w < unsafe.index(("sync", "ext")), (
        "this leg is meant to DEMONSTRATE the hazardous interleaving"
    )
    assert not any(e[0] == "sync" for e in unsafe[:w]), (
        "nothing fenced the write against the in-flight replay — that is "
        "exactly the hazard enqueue()'s docstring documents"
    )

    safe: list = []
    pipe = _stub_pipeline(safe)
    pipe.enqueue(1)
    pipe.synchronize()
    safe.append(("device_write", "ext"))  # the documented discipline
    assert safe.index(("sync", "ext")) < safe.index(("device_write", "ext"))

    # ...and the pre_launch seam is the in-flight-safe way to stage new inputs:
    # it runs on the pipeline's OWN stream, ordered before its replays.
    hooked: list = []
    pipe = _stub_pipeline(hooked)
    pipe.enqueue(1, pre_launch=lambda: hooked.append(("hook", "ext")))
    _assert_u1_ordering(hooked, n=1, with_hook=True)


def test_one_in_flight_rule_re_records_the_same_events(fake_cupy):
    """``enqueue``'s HAZARD 1, pinned: a second enqueue before a synchronize
    RE-RECORDS the cached pair rather than allocating a new one. This is legal
    (``cudaStreamWaitEvent`` snapshots), and it is also exactly why the
    docstring says one in-flight enqueue per pipeline."""
    event_cls = fake_cupy()
    log: list = []
    pipe = _stub_pipeline(log)
    pipe.enqueue(1)
    first = _pair(pipe)[0]
    pipe.enqueue(1)  # second enqueue, nothing synchronized in between
    assert _pair(pipe)[0] is first
    assert event_cls.constructed == 2
    assert len(first.records) == 2


# --------------------------------------------------------------------------- #
# compose's use of the public split (host-side parity).
# --------------------------------------------------------------------------- #
def test_compose_overlap_group_drives_enqueue_then_synchronize_with_one_event_pair(
    fake_cupy,
):
    """``GraphComposer._async_launch_group`` is expressed VERBATIM
    through ``enqueue(seed_events=..., pre_launch=...)`` + ``synchronize``,
    rather than a separate implementation of the shared-event ordering
    pattern over ``_launch_handles``.
    The properties that must survive that migration, all asserted here:

    1. ONE shared event pair for the whole group (not one pair per member);
    2. every member waits BOTH shared events on its OWN stream;
    3. a member's ``pre_launch`` runs on that member's stream, after its waits
       and before its replays;
    4. EVERY member is enqueued before ANY member is synchronized (that is the
       overlap the method exists for);
    5. the synchronize pass runs in registration order.
    """
    event_cls = fake_cupy()
    log: list = []
    a = _stub_pipeline(log, tag="A")
    b = _stub_pipeline(log, tag="B")
    def _hook_a():
        log.append(("hook", "A"))

    members = [
        _Member(a, "a", kind="launchable", pre_launch=_hook_a),
        _Member(b, "b", kind="launchable", pre_launch=None),
    ]

    composer = GraphComposer.__new__(GraphComposer)
    composer._seeds = []
    composer._async_launch_group(members, 2)

    assert event_cls.constructed == 2, (
        f"the group must record ONE shared event pair, not one per member "
        f"(saw {event_cls.constructed} events for 2 members)"
    )
    assert _pair(a) == (None, None) and _pair(b) == (None, None), (
        "a member must not record its own events in shared-event mode"
    )
    uids = sorted({e[2] for e in log if e[0] == "wait"})
    for tag in ("A", "B"):
        assert sorted(e[2] for e in log if e[0] == "wait" and e[1] == tag) == uids, (
            f"member {tag} did not wait BOTH shared events"
        )
    _assert_u1_ordering(log, tag="A", n=2, with_hook=True, require_no_sync=False)
    _assert_u1_ordering(log, tag="B", n=2, require_no_sync=False)

    syncs = [i for i, e in enumerate(log) if e[0] == "sync"]
    last_launch = max(i for i, e in enumerate(log) if e[0] == "launch")
    assert min(syncs) > last_launch, (
        f"a member was synchronized before every member had been enqueued — "
        f"that serializes the group host-side, which is the bug the overlap "
        f"group exists to avoid: {log}"
    )
    assert [log[i][1] for i in syncs] == ["A", "B"], "sync pass out of order"


def test_red_leg_a_per_member_event_pair_fails_the_shared_pair_gate(fake_cupy):
    """RED LEG, executable: driving each member through plain ``launch()``
    (its own event pair, its own immediate sync) must fail BOTH the
    shared-pair count and the enqueue-before-any-sync property."""
    event_cls = fake_cupy()
    log: list = []
    a = _stub_pipeline(log, tag="A")
    b = _stub_pipeline(log, tag="B")
    for pipe in (a, b):
        pipe.launch(2)

    assert event_cls.constructed == 4  # 2 members x 2 events: NOT shared
    with pytest.raises(AssertionError):
        assert event_cls.constructed == 2, "the group must record ONE shared event pair"
    syncs = [i for i, e in enumerate(log) if e[0] == "sync"]
    last_launch = max(i for i, e in enumerate(log) if e[0] == "launch")
    assert min(syncs) < last_launch, (
        "this leg is meant to demonstrate the SERIALIZED shape"
    )


# --------------------------------------------------------------------------- #
# Device legs — serialized on the device under test.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_device_replay_parity_across_repeated_launches():
    """Replay parity on real hardware: a pipeline replayed many times
    through the CACHED event pair must produce bit-identical results to the
    same graph replayed once per freshly-built pipeline. If re-recording a
    cached event were not legal, this is where it would show."""
    import cupy as cp
    import numpy as np

    n = 4096
    src = cp.arange(n, dtype=cp.float64)
    kern = cp.ElementwiseKernel("float64 x", "float64 y", "y = x * 3.0 + 1.0", "ehp")

    def _build(out):
        kern(src, cp.empty_like(out))  # warm (compile) OUTSIDE capture
        pipe = GraphPipeline()
        pipe.add(lambda: kern(src, out))
        return pipe.build()

    out_shared = cp.zeros(n, dtype=cp.float64)
    pipe = _build(out_shared)
    digests = []
    for _ in range(8):
        out_shared.fill(0.0)
        pipe.launch(1)  # the cached-event path, repeatedly
        digests.append(cp.asnumpy(out_shared).tobytes())

    out_fresh = cp.zeros(n, dtype=cp.float64)
    _build(out_fresh).launch(1)
    reference = cp.asnumpy(out_fresh).tobytes()

    assert all(d == reference for d in digests), (
        "a replay through the cached event pair did not match a fresh "
        "pipeline's replay bit for bit"
    )
    assert pipe._seeds and pipe._seeds[0] is not None
    expected = np.asarray(cp.asnumpy(src)) * 3.0 + 1.0
    assert np.array_equal(cp.asnumpy(out_shared), expected)


@pytest.mark.gpu
def test_device_enqueue_synchronize_split_matches_launch():
    """On real hardware, ``enqueue`` + ``synchronize`` produces exactly
    what ``launch`` does, and two independent pipelines can be in flight at
    once (both results correct after the second pass)."""
    import cupy as cp
    import numpy as np

    n = 2048
    a = cp.arange(n, dtype=cp.float64)
    b = cp.arange(n, dtype=cp.float64) * -2.0
    kA = cp.ElementwiseKernel("float64 x", "float64 y", "y = x + 4.0", "ehp_A")
    kB = cp.ElementwiseKernel("float64 x", "float64 y", "y = x - 7.0", "ehp_B")
    outA = cp.zeros(n, dtype=cp.float64)
    outB = cp.zeros(n, dtype=cp.float64)
    kA(a, cp.empty_like(outA))
    kB(b, cp.empty_like(outB))

    pipeA = GraphPipeline().add(lambda: kA(a, outA)).build()
    pipeB = GraphPipeline().add(lambda: kB(b, outB)).build()

    pipeA.launch(1)
    ref_a = cp.asnumpy(outA).copy()
    outA.fill(0.0)

    # both in flight, then both synchronized (the compose shape, by hand)
    pipeA.enqueue(1)
    pipeB.enqueue(1)
    pipeA.synchronize()
    pipeB.synchronize()

    assert np.array_equal(cp.asnumpy(outA), ref_a)
    assert np.array_equal(cp.asnumpy(outB), cp.asnumpy(b) - 7.0)
