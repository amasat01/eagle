"""The graph builder over eagle's CPU Graph."""
import linecache
import numpy as np
import pytest
import eagle

hawk = pytest.importorskip("hawk")
from hawk import Grid, Index, Mutable, Param, Scalar, Table, UInt32
from hawk import math as m
import _builder_fixtures as bp

kernel = bp.kernel
U = np.uint32


@kernel
def inc(x: Scalar) -> Scalar: return x + 1.0
@kernel
def dbl(x: Scalar) -> Scalar: return 2.0 * x
@kernel
def half(x: Scalar) -> Scalar: return 0.5 * x
@kernel
def add(a: Scalar, b: Scalar) -> Scalar: return a + b
@kernel
def times10(x: Scalar) -> Scalar: return 10.0 * x
@kernel
def below(s: Scalar, lim: Param, flag: Mutable[UInt32]):
    flag = m.uint32(hawk.sum(s < lim))
@kernel
def mark(x: Scalar, c: Mutable[UInt32]):
    c = m.uint32(hawk.sum(x > 2.5))
@kernel
def clear_inc(x: Scalar, flag: Mutable[UInt32]) -> Scalar:     # an arm that clears the very cell it was decided on
    flag = m.uint32(hawk.sum(x > 1e9))
    return x + 1.0
@kernel
def count_ret(s: Scalar, flag: Mutable[UInt32]) -> Scalar:     # a reduction that also returns: not sample_local
    flag = m.uint32(hawk.sum(s > 1.0))
    return s * 1.0


def cell(v): return np.array([v], dtype=U)


def defn(code, ns=None):
    """A builder defined from text (its source on record), in this module's namespace."""
    ns = ns if ns is not None else dict(globals())
    name = f"<de1 {abs(hash(code))}>"
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)
    exec(compile(code, name, "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def doc():
    errors = {}
    ns = bp.run_doc(8, tolerate=(), errors=errors)
    ns["_errors"] = errors
    return ns


# ============================================================================ (a) the section-1 snippets
def ref_whole(x, flag, y):
    a = 0.5 * x; l = a + 1.0; l2 = 2.0 * l; r = l - 1.0; r2 = (a ** 2) ** 2
    s = x.copy(); y = y.copy()
    for _ in range(50):
        if not flag: break
        s = s - (l2 ** 2) ** 2
        flag = int((s > 1.0).sum())
    return s, y + s + r + r2


def test_a_all_eight_snippets_run(doc):
    # blocks 1-8 executed against the real CPU Graph; block 6's `reverse` is the diff stream's (see test below)
    assert all(k in doc for k in ("whole", "chain", "pair", "void", "g", "gr", "settle", "twice"))
    assert set(doc["_errors"]) <= {6}


def test_a_doc_reverse_gradient(doc):
    # snippet 6: gradients of J = sum(result) + sum(y) with respect to x, through the seed copy, the loop and the fork
    x = np.arange(8.0); y = np.zeros(8)
    g = doc["whole"](x, cell(1), y, record=True)
    g()
    gr = eagle.diff.reverse(g.record, {"result": np.ones(8), "y": np.ones(8)}, wrt=("x",))
    ds = 1 - 4 * (x + 2) ** 3
    np.testing.assert_allclose(gr["x"], 2 * ds + 0.5 + 0.5 * (0.5 * x) ** 3 * 4)


def test_a_whole_equals_python_and_x_unchanged(doc):
    whole = doc["whole"]
    x = np.arange(8.0); y = np.zeros(8); flag = cell(1); x0 = x.copy()
    s_ref, y_ref = ref_whole(x, 1, y)
    g = whole(x, flag, y)
    s = g()
    np.testing.assert_array_equal(s, s_ref); np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(x, x0)                      # the caller's array is never written
    assert s is not x and g.result is s
    assert g._bld.stats["seeds"] >= 1


def test_a_bind_equals_fresh_build_and_python(doc):
    whole = doc["whole"]
    x, x2 = np.arange(8.0), np.arange(8.0) + 1.0
    y = np.zeros(8); flag = cell(1)
    g = whole(x, flag, y)
    g()
    y[:] = 0; flag[0] = 1
    assert g.bind(x=x2) is g
    s = g()
    s_ref, y_ref = ref_whole(x2, 1, np.zeros(8))
    np.testing.assert_array_equal(s, s_ref); np.testing.assert_array_equal(y, y_ref)
    y2 = np.zeros(8); f2 = cell(1)
    s_fresh = whole(x2, f2, y2)()
    np.testing.assert_array_equal(s, s_fresh); np.testing.assert_array_equal(y, y2)


def test_a_result_forms(doc):
    chain, pair, void = doc["chain"], doc["pair"], doc["void"]
    x = np.arange(8.0)
    np.testing.assert_array_equal(chain(x)(), (2 * x + 1) ** 2)
    r = pair(x)()
    assert sorted(r) == ["sq", "v"]
    np.testing.assert_array_equal(r["v"], 2 * x + 1); np.testing.assert_array_equal(r["sq"][0], (2 * x + 1) ** 2)
    y = np.zeros(8); vg = void(x, y)
    assert vg() is None
    np.testing.assert_array_equal(y, 3 * x)


def test_a_g_n_replays(doc):
    void = doc["void"]
    x = np.arange(8.0); y = np.zeros(8)
    g = void(x, y)
    g(10)
    np.testing.assert_array_equal(y, 3 * x)
    assert g.launch(2) is g and g.enqueue(1).synchronize() is g
    np.testing.assert_array_equal(y, 3 * x)


# ============================================================================ builders for the rest
@eagle.Graph.builder
def b_first_hour(x, flag):
    g = eagle.Graph()
    while flag:
        x = inc(x)
        below(x, 3.0, flag)
    g.result = x
    return g


@eagle.Graph.builder
def b_if(x, flag):
    g = eagle.Graph()
    if flag:
        x = times10(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_if_else(x, flag):
    g = eagle.Graph()
    if flag:
        x = times10(x)
    else:
        x = half(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_elif(x, f1, f2):
    g = eagle.Graph()
    if f1:
        x = times10(x)
    elif f2:
        x = dbl(x)
    else:
        x = half(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_decide_once(x, flag):
    g = eagle.Graph()
    if flag:
        x = inc(x)
        below(x, -1.0, flag)            # the arm clears the very cell it was decided on
    else:
        x = half(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_not(x, flag):
    g = eagle.Graph()
    if not flag:
        x = inc(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_python_values(x, debug, n=3):
    g = eagle.Graph()
    if len(x) > 4:
        x = inc(x)
    else:
        x = dbl(x)
    if debug:
        x = dbl(x)
    for k in (inc, dbl, inc):
        x = k(x)
    for _ in range(n):
        x = inc(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_capped(x, flag, n):
    g = eagle.Graph()
    for _ in range(n):
        if not flag: break
        x = inc(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_break(x, flag, done):
    g = eagle.Graph()
    while flag:
        x = inc(x)
        mark(x, done)
        if done: break
        x = dbl(x)
        below(x, 50.0, flag)
    g.result = x
    return g


@eagle.Graph.builder
def b_continue(x, flag, skip):
    g = eagle.Graph()
    while flag:
        x = inc(x)
        below(x, 6.0, flag)
        mark(x, skip)
        if skip: continue
        x = dbl(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_local_temp(x, flag):
    g = eagle.Graph()
    for _ in range(5):
        if not flag: break
        t = half(x)                       # loop-local: fresh buffer, reused per iteration
        x = add(x, t)
        below(x, 20.0, flag)
    g.result = x
    return g


@eagle.Graph.builder
def b_nested(x, outer, inner):
    g = eagle.Graph()
    for _ in range(3):
        if not outer: break
        x = inc(x)
        while inner:
            x = dbl(x)
            below(x, 10.0, inner)
    g.result = x
    return g


@eagle.Graph.builder
def b_fork_in_loop(x, flag):
    g = eagle.Graph()
    for _ in range(3):
        if not flag: break
        with g.parallel() as p:
            with p.branch() as b:
                u = inc(x)
            with p.branch() as c:
                v = dbl(x)
        x = add(u, v)
    g.result = x
    return g


@eagle.Graph.builder
def b_buffer(x):
    g = eagle.Graph()
    keep = np.zeros(len(x))
    a = inc(x, buffer=keep)
    b = dbl(a, buffer=True)
    g.result = (a, b, keep)
    return g


@eagle.Graph.builder
def blk_twice(g: eagle.Graph, v):
    return inc(inc(v))


@eagle.Graph.builder
def b_block_forks(x):
    g = eagle.Graph()
    a = dbl(x)
    with g.parallel() as p:
        with p.branch() as b:
            u = blk_twice(b, a)
        with p.branch() as c:
            w = half(a)
            with c.parallel() as q:                 # parallel -> branch -> parallel
                with q.branch() as d:
                    w1 = inc(w)
                with q.branch() as e:
                    w2 = dbl(w)
    g.result = add(add(u, w1), w2)
    return g


@eagle.Graph.builder
def b_wait(x):
    g = eagle.Graph()
    a = dbl(x)
    with g.parallel() as p:
        with p.branch() as b:
            l = inc(a)
            l2 = dbl(l)
        with p.branch() as w:
            w.wait(l)
            r = add(l, a)                          # reads b's l: legal only through the wait
    g.result = (l2, r)
    return g


def test_a_python_references():
    x = np.zeros(4)
    for flag0, want in ((1, 3.0),):
        assert np.asarray(b_first_hour(x, cell(flag0))())[0] == want
    # python values decided once, loop over python data unrolled
    x = np.arange(8.0)
    for debug in (True, False):
        ref = x + 1.0
        ref = 2 * ref if debug else ref
        ref = ref + 1; ref = 2 * ref; ref = ref + 1; ref = ref + 3
        np.testing.assert_array_equal(b_python_values(x, debug)(), ref)
    np.testing.assert_array_equal(b_python_values(np.arange(3.0), False)(), (2 * np.arange(3.0) + 1) * 2 + 1 + 3)


def test_a_if_elif_else_every_arm():
    x = np.arange(4.0) + 1
    for f, want in ((1, 10 * x), (0, 0.5 * x)):
        np.testing.assert_array_equal(b_if_else(x, cell(f))(), want)
    for f1, f2, want in ((1, 1, 10 * x), (0, 1, 2 * x), (0, 0, 0.5 * x)):
        np.testing.assert_array_equal(b_elif(x, cell(f1), cell(f2))(), want)
    np.testing.assert_array_equal(b_not(x, cell(0))(), x + 1); np.testing.assert_array_equal(b_not(x, cell(1))(), x)
    f = cell(1)
    np.testing.assert_array_equal(b_decide_once(x, f)(), x + 1.0)       # the else arm must NOT see the cleared cell
    f = cell(0)
    np.testing.assert_array_equal(b_decide_once(x, f)(), 0.5 * x)


def test_a_replays_follow_the_cell():
    x = np.arange(4.0) + 1; f = cell(1)
    g = b_if_else(x, f)
    np.testing.assert_array_equal(g(), 10 * x)
    f[0] = 0
    np.testing.assert_array_equal(g(), 0.5 * x)                          # the guard is read per replay, no rebuild


def test_a_capped_for_runs_N_when_guard_never_clears():
    x = np.zeros(4)
    for n in (4, 1):
        g = b_capped(x, cell(1), n)
        np.testing.assert_array_equal(g(), np.full(4, float(n)))
        rw = [nd for nd in g._nodes if nd.kind == "repeat_while"][0].payload.wrapper
        assert rw.iterations() == n and rw.max_iters == n
    np.testing.assert_array_equal(b_capped(x, cell(0), 5)(), x)


def py_break(x, flag):
    x = x.copy()
    while flag:
        x = x + 1; done = int((x > 2.5).sum())
        if done: break
        x = 2 * x; flag = int((x < 50).sum())
    return x


def py_continue(x, flag):
    x = x.copy()
    while flag:
        x = x + 1; flag = int((x < 6).sum()); skip = int((x > 2.5).sum())
        if skip: continue
        x = 2 * x
    return x


def test_a_mid_body_break_and_continue():
    x = np.zeros(4)
    np.testing.assert_array_equal(b_break(x, cell(1), cell(0))(), py_break(x, 1))
    np.testing.assert_array_equal(b_continue(x, cell(1), cell(0))(), py_continue(x, 1))
    assert py_break(x, 1)[0] == 3.0 and py_continue(x, 1)[0] == 6.0


def test_a_local_temporaries_and_fork_and_nested_and_buffer():
    x = np.ones(4)
    ref = x.copy()
    for _ in range(5):
        ref = ref + 0.5 * ref
        if not (ref < 20).sum(): break
    np.testing.assert_array_equal(b_local_temp(x, cell(1))(), ref)
    ref = np.zeros(4)
    for _ in range(3):
        ref = (ref + 1) + 2 * ref
    np.testing.assert_array_equal(b_fork_in_loop(np.zeros(4), cell(1))(), ref)
    ref, inn = np.zeros(4), 1
    for _ in range(3):
        ref = ref + 1
        while inn:
            ref = 2 * ref; inn = int((ref < 10).sum())
    inner = cell(1)
    np.testing.assert_array_equal(b_nested(np.zeros(4), cell(1), inner)(), ref)
    keep_x = np.arange(4.0)
    a, b, keep = b_buffer(keep_x)()
    np.testing.assert_array_equal(a, keep_x + 1); np.testing.assert_array_equal(b, 2 * keep_x + 2)
    assert a is keep or np.shares_memory(a, keep)


def test_a_fork_nesting_block_in_branch_and_wait():
    x = np.arange(4.0)
    a = 2 * x
    g = b_block_forks(x)
    w = 0.5 * a
    np.testing.assert_array_equal(g(), (a + 2) + (w + 1) + 2 * w)
    l2, r = b_wait(x)()
    l = a + 1
    np.testing.assert_array_equal(l2, 2 * l); np.testing.assert_array_equal(r, l + a)


# ============================================================================ (b) the three review probes
def test_b_probe_rebind_three_cases():
    x = np.zeros(4)
    # case 1: while flag: j = step(j)  -> Python: 3 iterations, j == 3 (the recorded-once body replays)
    g = b_first_hour(x, cell(1))
    assert np.asarray(g())[0] == 3.0 and (x == 0).all()
    # case 2: if flag: j = correct(j) with the flag OFF -> j stays j0 (Python), NOT the unwritten slot
    j0 = np.ones(4)
    assert (np.asarray(b_if(j0, cell(0))()) == j0).all() and (j0 == 1).all()
    # case 3: if/else with the flag ON -> the IF arm's value, not the else arm's buffer
    assert (np.asarray(b_if_else(j0, cell(1))()) == 10.0).all()
    assert (np.asarray(b_if_else(j0, cell(0))()) == 0.5).all()


# ============================================================================ (c) refusals
B = "@eagle.Graph.builder\n"


def refused(code, call, exc, match):
    ns = defn(code)
    with pytest.raises(exc, match=match):
        call(ns)


def test_c_builder_return_and_graph_rules():
    refused(B + "def bad_return(x):\n    g = eagle.Graph(); return inc(x)\n", lambda n: n["bad_return"](np.ones(4)),
            TypeError, "a builder returns its graph: `return g`; declare the arrays with `g.result = ...`")
    refused(B + "def no_graph(x): return None\n", lambda n: n["no_graph"](np.ones(4)), TypeError, "a builder makes its graph")
    with pytest.raises(TypeError, match="a builder returns its Graph"):
        defn(B + "def f(x) -> float: pass\n")
    refused(B + "def badres(x):\n    g = eagle.Graph(); g.result = {1: x}; return g\n", lambda n: n["badres"](np.ones(4)),
            TypeError, "dict keys are strings")
    refused(B + "def two(x):\n    g = eagle.Graph(); h = eagle.Graph(); return g\n", lambda n: n["two"](np.ones(4)),
            RuntimeError, "ONE graph")


def test_c_result_bind_enqueue(doc):
    whole = doc["whole"]
    x = np.arange(8.0); g = whole(x, cell(1), np.zeros(8))
    with pytest.raises(RuntimeError, match="declared while building only"): g.result = None
    with pytest.raises(ValueError, match="changed shape/dtype: call the builder again"): g.bind(x=np.ones(3))
    with pytest.raises(ValueError, match="'q' is not a builder parameter"): g.bind(q=np.ones(8))
    with pytest.raises(ValueError, match="'record' is a Python value, decided at build: call the builder again"): g.bind(record=True)
    with pytest.raises(RuntimeError, match="a record=True graph runs with launch"): whole(x, cell(1), np.zeros(8), record=True).enqueue()


def test_c_uncapped_while_warns_and_record_refuses():
    with pytest.warns(UserWarning, match="uncapped device loop"):
        b_first_hour(np.zeros(4), cell(1))
    refused(B + "def unc(x, flag):\n    g = eagle.Graph(record=True)\n    while flag:\n        x = inc(x)\n    g.result = x\n    return g\n",
            lambda n: n["unc"](np.ones(4), cell(0)), ValueError, r"unc:\d+: an uncapped device loop cannot be recorded \(record=True\): cap it")


def test_c_in_place_by_non_sample_local():
    refused(B + "def red(x, flag):\n    g = eagle.Graph()\n    for _ in range(3):\n        if not flag: break\n        x = count_ret(x, flag)\n"
            "    g.result = x\n    return g\n", lambda n: n["red"](np.ones(4), cell(1)), ValueError,
            r"red:\d+: `x = count_ret\(\.\.\.\)` inside device control flow writes 'x' in place, and 'count_ret' is not sample_local")


def test_c_blocks():
    refused(B + "def blk(g: eagle.Graph, v):\n    h = eagle.Graph(); return v\n" + B + "def owner(x):\n    g = eagle.Graph(); blk(g, x); g.result = x; return g\n",
            lambda n: n["owner"](np.ones(4)), RuntimeError, "a block appends to the graph it is given")
    refused(B + "def inner(v):\n    g = eagle.Graph(); g.result = inc(v); return g\n" + B + "def outer2(x):\n    g = eagle.Graph(); inner(x); return g\n",
            lambda n: n["outer2"](np.ones(4)), RuntimeError, "a builder inside a builder; make it a block")
    refused(B + "def wrong(x):\n    g = eagle.Graph()\n    with g.parallel() as p:\n        with p.branch() as b:\n            blk_twice(g, x)\n    return g\n",
            lambda n: n["wrong"](np.ones(4)), RuntimeError, "pass the handle of the scope you are in")
    g = b_if(np.ones(4), cell(1))
    with pytest.raises(RuntimeError, match="'g' is built; a build is one instantiation"): blk_twice(g, np.ones(4))


def test_c_fork_and_wait():
    refused(B + "def outer(x):\n    g = eagle.Graph()\n    with g.parallel() as p:\n        with p.branch() as b:\n            with g.parallel(): pass\n    return g\n",
            lambda n: n["outer"](np.ones(4)), RuntimeError, "inside a branch, open the fork from the branch handle")
    refused(B + "def w0(x):\n    g = eagle.Graph(); g.wait(x); return g\n", lambda n: n["w0"](np.ones(4)), RuntimeError,
            "only inside a parallel region, on a branch handle")
    refused(B + "def w1(x):\n    g = eagle.Graph(); a = dbl(x)\n    with g.parallel() as p:\n        with p.branch() as b:\n            b.wait(a); u = inc(a)\n    return g\n",
            lambda n: n["w1"](np.ones(4)), ValueError, "before the fork everything is already ordered")
    refused(B + "def w2(x):\n    g = eagle.Graph(); a = dbl(x)\n    with g.parallel() as p:\n        with p.branch() as b:\n            u = inc(a); b.wait(u)\n    return g\n",
            lambda n: n["w2"](np.ones(4)), ValueError, "written by this branch, which is already in order")
    refused(B + "def w3(x):\n    g = eagle.Graph(); a = dbl(x)\n    with g.parallel() as p:\n        with p.branch() as b:\n            u = inc(a)\n"
            "        with p.branch() as c:\n            v = add(u, a)\n    return g\n", lambda n: n["w3"](np.ones(4)), ValueError,
            r"order them with <branch>.wait\(plane\)")


def test_c_carried_local_and_unbound_after():
    refused(B + "def car(x, flag):\n    g = eagle.Graph(); t = dbl(x)\n    for _ in range(3):\n        if not flag: break\n        t = add(t, x)\n"
            "    g.result = x\n    return g\n", lambda n: n["car"](np.ones(4), cell(1)), ValueError, "carried across iterations")
    refused(B + "def unb(x, flag):\n    g = eagle.Graph()\n    if flag:\n        a = inc(x)\n    else:\n        a = dbl(x)\n    g.result = a\n    return g\n",
            lambda n: n["unb"](np.ones(4), cell(1)), ValueError, "is not bound before it")
    refused(B + "def badc(x):\n    g = eagle.Graph()\n    if x:\n        x = inc(x)\n    return g\n", lambda n: n["badc"](np.ones(4)),
            ValueError, "a device condition is a 1-cell uint32 array")


# ============================================================================ (d) twin: same node kinds and edges
def edges_of(g): return [(e.src, e.dst, e.plane, e.kind) for e in g.edges()]


def test_d_twin_kinds_and_edges():
    @eagle.Graph.builder
    def tw(x, flag, cap):
        g = eagle.Graph()
        a = dbl(x)
        with g.parallel() as p:
            with p.branch() as b:
                u = inc(a)
            with p.branch() as c:
                v = dbl(a)
        w = add(u, v)
        if flag:
            w = inc(w)
        for _ in range(5):
            if not cap: break
            w = dbl(w)
            below(w, 100.0, cap)
        g.result = w
        return g

    x = np.arange(4.0); flag, cap = cell(1), cell(1)
    g = tw(x, flag, cap)
    a, u, v, w = (np.zeros(4) for _ in range(4))
    t = eagle.Graph()                              # the REAL eagle.Graph, hand-built
    t.add(dbl.bind(x=x, out=a))
    t.add_group([dbl.bind(x=a, out=v), inc.bind(x=a, out=u)][::-1])
    t.add(add.bind(a=u, b=v, out=w))
    t._if_else(eagle.SkipGuard(flag, 0), inc.bind(x=w, out=w).launch)
    t.add_node(eagle.repeat_while((dbl.bind(x=w, out=w).launch, below.bind(s=w, lim=100.0, flag=cap).launch),
                                  eagle.SkipGuard(cap, 0), 5))
    t.build()
    assert [n.kind for n in g._nodes] == [n.kind for n in t._nodes] == ["launch", "group", "launch", "copy", "skippable", "repeat_while"]
    assert g.node_labels()[:4] == ["a", "u", "v", "w"]       # call-site labels, not kernel names
    assert edges_of(g) == edges_of(t)
    assert g.schedule == t.schedule
    g.launch(); t.launch()
    np.testing.assert_array_equal(g.result, w)               # same numbers as the hand-built twin


def test_d_seed_adds_exactly_one_node_over_twin():
    g = b_if(np.ones(4), cell(1))
    assert [n.kind for n in g._nodes] == ["copy", "copy", "skippable"] and g.node_labels()[0] == "seed(x)"


# ============================================================================ (e) record-time cost
def test_e_record_time_cost(doc, capsys):
    whole = doc["whole"]
    x = np.arange(8.0); times = []
    for _ in range(5):
        g = whole(x, cell(1), np.zeros(8)); times.append(g._bld.timing)
    ms = lambda k: sorted(t[k] for t in times)[len(times) // 2]
    rw = whole.rewrite_ms or 0.0
    # per-kernel first-call compile is cached on disk; the rewrite is once per function
    line = (f"RECORD-TIME whole(): rewrite(once, per function) = {rw:.1f} ms, body-run+record = {ms('record_ms'):.2f} ms, "
           f"lower+build = {ms('lower_build_ms'):.2f} ms (median of 5; {g.num_nodes()} nodes)")
    with capsys.disabled():
        print("\n" + line)
        assert ms("record_ms") < 200


# ============================================================================ gradient keys
@kernel
def scale(x: Scalar, a: Param, b: Param) -> Scalar: return a * x + b
@kernel
def square(v: Scalar) -> Scalar: return v * v
@kernel
def put(s: Scalar, r: Scalar, y: Mutable[Scalar]):
    y = s + r


@eagle.Graph.builder
def gb(x, y):
    g = eagle.Graph(record=True)
    a = scale(x, 2.0, 1.0, buffer=True)
    s = square(a)
    put(s, a, y)
    g.result = {"s": s, "a": a}
    return g


def test_g_keys_result_mutable_argument_and_buffer():
    x = np.arange(4.0); y = np.zeros(4)
    g = gb(x, y)
    out = g()
    a = 2 * x + 1
    np.testing.assert_array_equal(out["s"], a * a); np.testing.assert_array_equal(y, a * a + a)
    ones = np.ones(4)
    gr = eagle.diff.reverse(g.record, {"result.s": ones, "y": ones}, wrt=("x",))
    np.testing.assert_allclose(gr["x"], 8 * a + 2)
    g = gb(x, y); g()
    gr = eagle.diff.reverse(g.record, {"a": ones}, wrt=("x",))                # a buffer= label as a seed
    np.testing.assert_allclose(gr["x"], 2 * ones)
    # wrt a buffer= label: the cotangent arriving at that buffer (J = sum(result.s)  ->  dJ/da = 2a)
    gr = eagle.diff.reverse(g.record, {"result.s": ones}, wrt=("a", "x"))
    np.testing.assert_allclose(gr["a"], 2 * a); np.testing.assert_allclose(gr["x"], 4 * a)


def test_g_refusals():
    x = np.arange(4.0); y = np.zeros(4); ones = np.ones(4)
    g = gb(x, y); g()
    with pytest.raises(ValueError, match=r"no default gradients: pass wrt=.*x, y"):
        eagle.diff.reverse(g.record, {"y": ones})
    with pytest.raises(KeyError, match="dedicated buffer"):
        eagle.diff.reverse(g.record, {"s": ones}, wrt=("x",))
    with pytest.raises(KeyError, match="dedicated buffer"):
        eagle.diff.reverse(g.record, {"y": ones}, wrt=("s",))
    with pytest.raises(KeyError, match="never writes: seed an output"):
        eagle.diff.reverse(g.record, {"x": ones}, wrt=("x",))
    with pytest.raises(KeyError, match="not in the result tree"):
        eagle.diff.reverse(g.record, {"result.pos": ones}, wrt=("x",))


def test_c_mixed_chain_and_decision_loop_under_record():
    refused(B + "def mix(x, f):\n    g = eagle.Graph()\n    if f:\n        x = inc(x)\n    elif len(x) > 1:\n        x = dbl(x)\n    g.result = x\n    return g\n",
            lambda n: n["mix"](np.ones(4), cell(1)), ValueError, "mixes a device cell and a Python value")
    refused(B + "def rl(x, flag, done):\n    g = eagle.Graph(record=True)\n    for _ in range(3):\n        if not flag: break\n"
            "        x = inc(x)\n        mark(x, done)\n        if done: break\n    g.result = x\n    return g\n",
            lambda n: n["rl"](np.ones(4), cell(1), cell(0)), ValueError, "holding a break, an if/elif/else or a match")


@eagle.Graph.builder
def b_count(x, n):
    g = eagle.Graph()
    for i in range(n):
        x = inc(x)
    g.result = x
    return g


def test_a_counted_loop_and_single_edge_wait():
    x = np.zeros(4)
    for n in (3, 0, 5):
        g = b_count(x, cell(n))
        np.testing.assert_array_equal(g(), np.full(4, float(n)))
    assert [nd.kind for nd in g._nodes][-1] == "repeat_while"
    g = b_wait(np.arange(4.0))
    assert [nd.kind for nd in g._nodes] == ["launch", "group"]                # no barrier stage: one group, one wait
    assert any(e.kind == "RAW" for e in g.edges() if g.node_labels()[e.dst] == "r" and g.node_labels()[e.src] == "l")


@eagle.Graph.builder
def b_match(x, sel):
    g = eagle.Graph()
    match sel:
        case 0:
            x = times10(x)
        case 2:
            x = dbl(x)
        case _:
            x = half(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_chain3(x, f1, f2, f3):
    g = eagle.Graph()
    if f1:
        x = times10(x)
        below(x, -1.0, f1)             # the arm clears its own guard
    elif f2:
        x = dbl(x)
    elif not f3:
        x = inc(x)
    else:
        x = half(x)
    g.result = x
    return g


def test_f_match_cells_and_python_values():
    x = np.arange(4.0) + 1
    for sel, want in ((0, 10 * x), (2, 2 * x), (1, 0.5 * x), (7, 0.5 * x)):
        np.testing.assert_array_equal(b_match(x, cell(sel))(), want)
    g = b_match(x, cell(0))
    np.testing.assert_array_equal(g(), 10 * x)
    g._bld.args["sel"][0] = 2
    np.testing.assert_array_equal(g(), 2 * x)                            # decided per replay
    np.testing.assert_array_equal(b_match(x, 2)(), 2 * x)                # a Python subject: decided once
    np.testing.assert_array_equal(b_match(x, 5)(), 0.5 * x)
    for code, msg in (("def m1(x, s):\n    g = eagle.Graph()\n    match s:\n        case [1, 2]:\n            x = inc(x)\n    g.result = x\n    return g\n", "MatchSequence"),
                      ("def m2(x, s):\n    g = eagle.Graph()\n    match s:\n        case 1 | 2:\n            x = inc(x)\n    g.result = x\n    return g\n", "MatchOr")):
        ns = defn(B + code)
        with pytest.raises(SyntaxError, match=msg):
            ns[code.split("(")[0][4:]](np.ones(4), cell(1))


def test_f_chain_of_three_conditions_and_self_clearing_arm():
    x = np.arange(4.0) + 1
    for f, want in (((1, 1, 1), 10 * x), ((0, 1, 1), 2 * x), ((0, 0, 0), x + 1), ((0, 0, 1), 0.5 * x)):
        np.testing.assert_array_equal(b_chain3(x, *map(cell, f))(), want)


@eagle.Graph.builder
def b_rec_if_in_loop(x, flag, sel):
    g = eagle.Graph(record=True)
    for _ in range(3):
        if not flag: break
        if sel:
            x = dbl(x)
        below(x, 1e30, flag)
    g.result = x
    return g


def test_f_decision_copy_inside_a_recorded_loop_is_refused():
    with pytest.raises(ValueError, match="holding a break, an if/elif/else or a match cannot be recorded"):
        b_rec_if_in_loop(np.ones(4), cell(1), cell(1))
    # the same loop without record=True runs
    @eagle.Graph.builder
    def plain(x, flag, sel):
        g = eagle.Graph()
        for _ in range(3):
            if not flag: break
            if sel:
                x = dbl(x)
            below(x, 1e30, flag)
        g.result = x
        return g
    np.testing.assert_array_equal(plain(np.ones(4), cell(1), cell(1))(), np.full(4, 8.0))


@eagle.Graph.builder
def b_deep_buffer(x, flag):
    g = eagle.Graph(record=True)
    for _ in range(2):
        if not flag: break
        x = dbl(x, buffer=True)
        below(x, 1e30, flag)
    g.result = x
    return g


def test_f_buffer_inside_a_body_is_not_a_key():
    x = np.arange(4.0) + 1
    g = b_deep_buffer(x, cell(1)); g()
    assert "dbl" in g._bld.deep or g._bld.deep
    name = sorted(g._bld.deep)[0]
    with pytest.raises(KeyError, match="inside a device loop or branch"):
        eagle.diff.reverse(g.record, {"result": np.ones(4)}, wrt=("x", name))


@pytest.mark.skipif(not bp.RETURNS, reason="needs hawk's return form")
def test_g_return_form_kernels_called_directly_keep_their_labels():
    @hawk.kernel
    def affine(x: Scalar, a: Param, b: Param) -> Scalar: return a * x + b

    @hawk.kernel
    def sq(v: Scalar) -> Scalar: return v * v

    @eagle.Graph.builder
    def fit(x):
        g = eagle.Graph(record=True)
        a = affine(x, 2.0, 1.0, buffer=True)
        g.result = sq(a)
        return g

    x = np.arange(4.0)
    g = fit(x); g()
    gr = eagle.diff.reverse(g.record, {"result": np.ones(4)}, wrt=("x", "a"))
    np.testing.assert_allclose(gr["x"], 4 * (2 * x + 1)); np.testing.assert_allclose(gr["a"], 2 * (2 * x + 1))


# ============================================================================ a Grid kernel in a builder
@hawk.kernel
def g_scale(lane: Grid["a", "b":4], x: Scalar) -> Scalar: return 2.0 * x * x
@hawk.kernel
def g_rows(lane: Grid["a", "b":4], x: Scalar, r: Mutable["a"]): hawk.scatter(r)[lane.a] += x
@hawk.kernel
def g_total(lane: Grid["a", "b":4], x: Scalar) -> Scalar: return hawk.sum(x, over="b")


def _grid_plan(k): return eagle.deploy(k, targets=("host",), cache_dir=bp.CACHE)


G = {k.name: _grid_plan(k) for k in (g_scale, g_rows, g_total)}
GX = np.arange(1.0, 13.0).reshape(3, 4)


@eagle.Graph.builder
def b_scale(x):
    g = eagle.Graph(); g.result = G["g_scale"](x, a=3); return g
@eagle.Graph.builder
def b_rows(x):
    g = eagle.Graph(); g.result = G["g_rows"](x, a=3); return g
@eagle.Graph.builder
def b_total(x):
    g = eagle.Graph(); g.result = G["g_total"](x, a=3); return g
@eagle.Graph.builder
def b_scale_rec(x):
    g = eagle.Graph(record=True); g.result = G["g_scale"](x, a=3); return g


@pytest.mark.parametrize("name, build", [("g_scale", b_scale), ("g_rows", b_rows), ("g_total", b_total)],
                         ids=["mutable", "axis_mutable_runtime_label", "sum_output"])
def test_g_a_builder_returns_what_a_direct_call_returns(name, build):
    direct = G[name](GX, a=3)
    graph = build(GX)
    got = graph()
    assert got.shape == direct.shape == graph.result.shape
    np.testing.assert_array_equal(got, direct)


def test_g_a_builder_gradient_comes_back_in_the_inputs_own_shape():
    from hawk.diff import vjp
    from hawk.trace.kernel import Kernel

    direct = Kernel("g_scale_vjp", vjp(g_scale, wrt=("x",)))(x=GX, bar_out=np.ones((3, 4)), a=3)
    graph = b_scale_rec(GX)
    graph()
    got = eagle.diff.reverse(graph.record, {"result": np.ones((3, 4))}, wrt=("x",))["x"]
    assert got.shape == direct.shape == (3, 4)
    np.testing.assert_allclose(got, direct, rtol=1e-14)


@hawk.kernel
def g_count(lane: Grid["e", "a":3], conn: Table[Index], b: Mutable["node"]):
    hawk.scatter(b)[conn.at(lane.flat)] += 1.0


def test_g_a_free_runtime_label_is_a_keyword_of_the_builder_call():
    plan, conn = _grid_plan(g_count), np.arange(6, dtype=np.int32) % 4

    @eagle.Graph.builder
    def b_count(c):
        g = eagle.Graph(); g.result = plan(c, e=2, node=4); return g

    direct = plan(conn, e=2, node=4)
    got = b_count(conn)()
    assert got.shape == direct.shape == (4,)
    np.testing.assert_array_equal(got, direct)
    with pytest.raises(ValueError, match="node"):
        plan(conn, b=np.zeros(5), e=2, node=4)
