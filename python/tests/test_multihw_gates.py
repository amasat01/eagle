# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Gates of the host/device graph: a mixed graph equals its single-device twin, one capture
replays and rebinds, the same builder runs over numpy planes with no GPU, a graph that crosses
nothing costs nothing, and eagle's orchestration calls no array
library's runtime."""

import ast
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import eagle
from eagle import _counters, _transfer
from eagle.graph import opaque

from _ulp import ulps

try:
    import cupy as cp
    _devices = cp.cuda.runtime.getDeviceCount()
except Exception:                      # no cupy, or no device: the GPU rows skip
    cp, _devices = None, 0
gpu = pytest.mark.gpu
needs_gpu = pytest.mark.skipif(not _devices, reason="needs a GPU")

#: Units in the last place a host evaluation may differ from the device's (libm sin, contraction).
KERNEL_ULPS = 4
N = 1 << 12


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _awkward(n, seed=0):
    """Random bit patterns: NaN payloads, infinities, subnormals and -0.0 included."""
    x = np.random.default_rng(seed).integers(0, 2**64, n, dtype=np.uint64).view(np.float64)
    x[:4] = [-0.0, np.inf, -np.inf, 5e-324]
    return x


try:
    import hawk
    import hawk.math
except ImportError:
    hawk = None


if hawk is not None:
    @hawk.kernel
    def tr_gate(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x * x + hawk.math.sin(x)


def _kernel(tmp):
    if hawk is None:
        pytest.skip("needs hawk")
    return eagle.deploy(tr_gate, targets=("host", "cuda"), cache_dir=str(tmp))


# --- 1. a mixed graph equals its single-device twin --- #
@gpu
@needs_gpu
def test_a_copy_chain_across_the_boundary_is_bitwise_the_device_copy_chain():
    x = _awkward(N)

    @eagle.Graph.builder
    def mixed(d_in, h_a, h_b, d_out):
        g = eagle.Graph()
        eagle.copy(d_in, into=h_a)                      # device -> pinned host
        eagle.copy(h_a, into=h_b)                       # host -> host
        eagle.copy(h_b, into=d_out)                     # host -> device
        return g

    @eagle.Graph.builder
    def twin(d_in, d_a, d_b, d_out):
        g = eagle.Graph()
        eagle.copy(d_in, into=d_a)
        eagle.copy(d_a, into=d_b)
        eagle.copy(d_b, into=d_out)
        return g

    d_in, out_m, out_t = cp.asarray(x), cp.zeros(N), cp.zeros(N)
    h_a, h_b = eagle.pinned(N), np.zeros(N)
    mixed(d_in, h_a, h_b, out_m).launch()
    twin(d_in, cp.zeros(N), cp.zeros(N), out_t).launch()
    assert (_bits(out_m.get()) == _bits(x)).all() and (_bits(out_t.get()) == _bits(x)).all()
    assert (_bits(h_a) == _bits(x)).all() and (_bits(h_b) == _bits(x)).all()


@gpu
@needs_gpu
def test_a_kernel_between_two_copies_matches_the_device_twin_and_the_numpy_build(tmp_path):
    twice = _kernel(tmp_path)
    x = np.random.default_rng(1).random(N)

    @eagle.Graph.builder
    def across(d_in, d_mid, h_mid, h_out, d_out):
        g = eagle.Graph()
        twice(d_in, d_mid)
        eagle.copy(d_mid, into=h_mid)
        twice(h_mid, h_out)                             # the same kernel over host planes
        eagle.copy(h_out, into=d_out)
        return g

    @eagle.Graph.builder
    def all_device(d_in, d_mid, d_mid2, d_out):
        g = eagle.Graph()
        twice(d_in, d_mid)
        eagle.copy(d_mid, into=d_mid2)
        twice(d_mid2, d_out)
        return g

    @eagle.Graph.builder
    def all_host(a, b, c):
        g = eagle.Graph()
        twice(a, b)
        twice(b, c)
        return g

    d_in, d_mid, d_out = cp.asarray(x), cp.zeros(N), cp.zeros(N)
    h_mid, h_out = eagle.pinned(N), eagle.pinned(N)
    g = across(d_in, d_mid, h_mid, h_out, d_out)
    assert g.backend == "gpu" and "eager_host" not in g.schedule
    g.launch()
    twin = cp.zeros(N)
    all_device(cp.asarray(x), cp.zeros(N), cp.zeros(N), twin).launch()
    b, c = np.zeros(N), np.zeros(N)
    h = all_host(x.copy(), b, c)
    assert h.backend == "cpu"
    h.launch()
    assert (_bits(h_mid) == _bits(d_mid.get())).all()                    # the copy moved the bytes
    assert (_bits(d_out.get()) == _bits(h_out)).all()
    assert ulps(d_out.get(), twin.get()).max() <= KERNEL_ULPS
    assert ulps(d_out.get(), c).max() <= KERNEL_ULPS


# --- 2. one capture, replay, rebind --- #
def _mixed_pipeline():
    d, h = cp.arange(8.0), eagle.pinned(8)
    g = eagle.Graph()
    g.add_node(opaque(lambda: cp.add(d, 1.0, out=d), reads=[d], writes=[d], name="bump"))
    g.add_node(eagle.copy(d, into=h))
    g.add_node(opaque(lambda: eagle.arrays.multiply(h, 2.0, out=h), reads=[h], writes=[h], name="host_scale"))
    g.add_node(eagle.copy(h, into=d))
    return d, g


@gpu
@needs_gpu
def test_one_capture_and_three_replays_equal_three_launches():
    before = _counters.snapshot()
    d, g = _mixed_pipeline()
    g.build()
    assert _counters.snapshot()["captures"] - before["captures"] == 1
    assert set(g.schedule) == {"capture"}
    g.launch(3)
    want = np.arange(8.0)
    for _ in range(3):
        want = (want + 1) * 2
    np.testing.assert_array_equal(d.get(), want)
    d2, g2 = _mixed_pipeline()
    g2.build()
    for _ in range(3):
        g2.launch()
    np.testing.assert_array_equal(d2.get(), want)
    assert _counters.snapshot()["captures"] - before["captures"] == 2     # replays captured nothing


@gpu
@needs_gpu
@pytest.mark.parametrize("pin", [True, False], ids=["pinned_source", "pageable_source"])
def test_a_copy_source_rebinds_with_no_recapture_and_replays_the_new_data(pin):
    make = eagle.pinned if pin else np.empty
    first, second = make(8), make(8)
    first[...], second[...] = np.arange(8.0), np.arange(8.0) + 100
    d = cp.zeros(8)
    g = eagle.Graph().add_node(eagle.copy(first, into=d)).build()
    g.launch(2)
    np.testing.assert_array_equal(d.get(), first)
    before = _counters.snapshot()
    g._rebind([(first, second)])
    g.launch(3)
    np.testing.assert_array_equal(d.get(), second)
    after = _counters.snapshot()
    assert after["captures"] == before["captures"] and after["graph_builds"] == before["graph_builds"]


# --- 3. the no-GPU leg --- #
def _builder_over_planes():
    @eagle.Graph.builder
    def step(h_in, h_mid, h_out):
        g = eagle.Graph()
        eagle.opaque(lambda: h_in.__imul__(2.0), reads=(h_in,), writes=(h_in,))
        eagle.copy(h_in, into=h_mid)
        eagle.copy(h_mid, into=h_mid)                   # the same storage: nothing to do
        eagle.copy(h_mid, into=h_out)
        return g
    return step


def test_the_builder_runs_over_numpy_planes():
    h_in, h_mid, h_out = np.arange(4.0), np.zeros(4), np.zeros(4)
    g = _builder_over_planes()(h_in, h_mid, h_out)
    assert g.backend == "cpu"
    g.launch(3)
    np.testing.assert_array_equal(h_out, np.arange(4.0) * 8)


def test_a_copy_onto_its_own_storage_does_no_work_on_either_arm(monkeypatch):
    class Tripwire(np.ndarray):
        def __setitem__(self, *_):
            raise AssertionError("a copy of an array onto itself assigned")

    a = np.arange(4.0).view(Tripwire)
    _transfer.copy_run_host([a, a])()
    calls = []
    monkeypatch.setattr(_transfer._core, "memcpy", lambda *args: calls.append(args), raising=False)
    _transfer.copy_run([a, a])()
    assert calls == []


def test_the_numpy_leg_runs_in_a_process_where_cupy_cannot_be_imported():
    code = (
        "import sys; sys.modules['cupy'] = None\n"
        "import numpy as np, eagle\n"
        "a, b = np.arange(4.0), np.zeros(4)\n"
        "g = eagle.Graph().add_node(eagle.copy(a, into=b)).add_node(eagle.copy(b, into=b)).build()\n"
        "g.launch(2)\n"
        "assert g.backend == 'cpu' and (b == a).all()\n"
        "assert (eagle.pinned(3).shape == (3,))\n")
    done = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, cwd="/tmp")
    assert done.returncode == 0, done.stderr


# --- 4. a graph that crosses nothing costs nothing --- #
def _zero_cost_rows():
    ref = os.environ.get("EAGLE_REFERENCE_TREE")
    if not ref:
        pytest.skip("set EAGLE_REFERENCE_TREE to the python/ directory of an eagle from before the "
                    "host/device graph (extension modules copied in)")
    import _zero_cost_abba as abba

    return abba.measure({"A": ref, "B": str(pathlib.Path(eagle.__file__).resolve().parents[1])})


@pytest.fixture(scope="module")
def zero_cost_rows():
    rows = _zero_cost_rows()
    print(json.dumps(rows, indent=1))
    return rows


def test_launch_of_a_graph_that_crosses_nothing_costs_nothing(zero_cost_rows):
    rows = {k: v for k, v in zero_cost_rows.items() if k.endswith("_launch")}
    assert all(v["ok"] for v in rows.values()), rows


def test_build_of_a_graph_that_crosses_nothing_costs_nothing(zero_cost_rows):
    rows = {k: v for k, v in zero_cost_rows.items() if k.endswith("_build")}
    assert all(v["ok"] for v in rows.values()), rows


# --- 5. eagle's orchestration calls no array library's runtime --- #
_ORCHESTRATION = ("graph.py", "_graph_kinds.py", "_transfer.py", "_builder.py", "_graph_edges.py",
                  "_conditional.py", "_recording.py", "_host_schedule.py", "_cell_rows.py",
                  "_plan_binding.py")
#: The array-library calls the orchestration makes: stream and event handles, device selection.
#: Copies, page-locked memory and host functions are not among them; they go through ``eagle._core``.
_ALLOWED: dict = {}
_DATA_MOVEMENT = ("memcpy", "launchHostFunc", "mallocHost", "hostAlloc", "pinned_memory", "PinnedMemory",
                  "empty_pinned", "zeros_pinned", "copy_to_host", "copyto")


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _cupy_calls(path):
    tree = ast.parse(path.read_text())
    calls, imports = [], []
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in inner:
            c = _chain(node)
            if c and c.split(".")[0] in ("cupy", "cupyx", "cp"):
                calls.append(c)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            imports += [n for n in names if n.split(".")[0] in ("cupyx", "cupy_backends")]
    return calls, imports


def test_no_array_library_runtime_in_the_orchestration_modules():
    root = pathlib.Path(eagle.__file__).parent
    for name in _ORCHESTRATION:
        path = root / name
        if not path.exists():
            continue
        calls, imports = _cupy_calls(path)
        assert not imports, f"{name} imports {imports}"
        allowed = _ALLOWED.get(name, set())
        for c in calls:
            assert not any(w.lower() in c.lower() for w in _DATA_MOVEMENT), f"{name}: {c}"
            assert any(c == a or c.startswith(a + ".") for a in allowed), f"{name}: {c} is not an allowed call"


def test_transfer_makes_no_array_library_call():
    calls, _ = _cupy_calls(pathlib.Path(_transfer.__file__))
    assert calls == []
