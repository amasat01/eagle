# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-launch envelope.

Host-only. Everything gated here is reachable without a device:

* **LaunchPlan** — the kernel-STATIC half of a launch (ABI tag per
  argument, the vector/matrix ``Mutable`` sets, the by-value POD boxes),
  resolved once per SIGNATURE instead of once per launch per argument.
  ``assemble_args`` only needs ``.data.ptr`` off its arrays, so a two-attribute
  stand-in exercises the real construction (the same trick
  ``test_wide_arg_classification.py`` already uses).
  **Block derivation is deliberately not in the plan** and this file
  asserts that too — ``resolve_block`` stays a per-call, patchable module
  global, because two live test spies (a downstream launch-policy twin) patch it by
  name.
* **The conforming fast path** — ``marshal._fast_path_ok`` decides
  whether an input can be bound with NO coercion. It takes the cupy module as
  an argument precisely so it can be exercised with a stand-in.
* **The opt-in zero mask** — a shared all-false ``terminated``
  buffer, reachable ONLY by a plugin whose sidecar declares the mask read-only.
* **The kernel-attrs cache** — ``_kernel_attrs_cache`` caps and refuses (never evicts).

Every counter gate carries an executable RED leg.
"""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest

from eagle import marshal as marshal_mod
from eagle.launch import (
    KERNEL_ATTRS_CACHE_CAP,
    KernelAttrsCacheFull,
    LaunchPlan,
    assemble_args,
    launch_plan,
    _launch_plan_stats,
    _reset_kernel_attrs_cache,
    _reset_launch_plan_cache,
)
from eagle.roles import classify_arg
from eagle.sidecar import TERMINATED_READONLY_KEY, read_terminated_readonly

# ``from eagle import launch`` binds the FUNCTION (``eagle/__init__.py`` does
# ``from .launch import launch``, overwriting the submodule attribute), so the
# module has to be reached through ``sys.modules`` — the same trap a downstream
# launch-policy twin documents at length. Monkeypatching a module global on the
# function object would be a silent no-op.
launch_mod = importlib.import_module("eagle.launch")


# --------------------------------------------------------------------------- #
# Device-free stand-ins.
# --------------------------------------------------------------------------- #
class _FakeArr:
    """A stand-in for a cupy array exposing only ``.data.ptr`` — all
    :func:`eagle.abi.make_gref` / :func:`eagle.abi.make_handle` ever read."""

    def __init__(self, ptr):
        self.data = types.SimpleNamespace(ptr=ptr)


class _MDecl:
    def __init__(self, name, dtype, width=1, shape=None):
        self.name, self.dtype, self.width, self.shape = name, dtype, width, shape


@pytest.fixture(autouse=True)
def _fresh_caches():
    _reset_launch_plan_cache()
    _reset_kernel_attrs_cache()
    marshal_mod._reset_coercion_stats()
    marshal_mod._reset_zero_mask_cache()
    marshal_mod._force_slow_coercion(False)
    yield
    _reset_launch_plan_cache()
    _reset_kernel_attrs_cache()
    marshal_mod._reset_coercion_stats()
    marshal_mod._reset_zero_mask_cache()
    marshal_mod._force_slow_coercion(False)


# --------------------------------------------------------------------------- #
# A small host-side signature corpus: every ABI tag, in several orders.
# --------------------------------------------------------------------------- #
_SIGNATURES = {
    "vector_kernel": (
        [("out", "out"), ("vec_in", "position"), ("terminated", "terminated"),
         ("uniform", "mu"), ("nsamples", "n")],
        (), (),
    ),
    "pure_scalar_mutable": (
        [("mutable", "counter"), ("terminated", "terminated"),
         ("uniform", "rate"), ("nsamples", "n")],
        (), (),
    ),
    "pure_vector_mutable": (
        [("mutable", "y"), ("vec_in", "x"), ("per_sample", "w"),
         ("lookup", "tab"), ("nsamples", "n")],
        ("y",), (),
    ),
    "pure_matrix_mutable": (
        [("mutable", "M"), ("mat_in", "A"), ("nsamples", "n")],
        (), ("M",),
    ),
    "wide_roles": (
        [("wide_in", "theta"), ("wide_out", "theta_bar"), ("mutable", "acc"),
         ("nsamples", "n")],
        (), (),
    ),
}


def _kw_for(arg_spec, vec_mut, mat_mut, *, base_ptr):
    """Build the per-call dicts ``assemble_args`` binds from, one distinct
    fake pointer per named argument."""
    out = vec = per_sample = tables = mutables = mats = wide_in = wide_out = None
    dicts = {
        "out": None, "vec": {}, "per_sample": {}, "tables": {}, "mutables": {},
        "mats": {}, "wide_in": {}, "wide_out": {}, "uniforms": {},
        "terminated": None,
    }
    for i, (role, name) in enumerate(arg_spec):
        ptr = base_ptr + 0x100 * (i + 1)
        if role == "out":
            dicts["out"] = _FakeArr(ptr)
        elif role == "vec_in":
            dicts["vec"][name] = _FakeArr(ptr)
        elif role == "mat_in":
            dicts["mats"][name] = _FakeArr(ptr)
        elif role == "per_sample":
            dicts["per_sample"][name] = _FakeArr(ptr)
        elif role == "lookup":
            dicts["tables"][name] = _FakeArr(ptr)
        elif role == "terminated":
            dicts["terminated"] = _FakeArr(ptr)
        elif role == "mutable":
            dicts["mutables"][name] = _FakeArr(ptr)
        elif role == "wide_in":
            dicts["wide_in"][name] = _FakeArr(ptr)
        elif role == "wide_out":
            dicts["wide_out"][name] = _FakeArr(ptr)
        elif role == "uniform":
            dicts["uniforms"][name] = np.float64(0.25 * (i + 1))
    del out, vec, per_sample, tables, mutables, mats, wide_in, wide_out
    return dicts


def _assemble(arg_spec, vec_mut, mat_mut, dicts, n, *, plan):
    return assemble_args(
        arg_spec,
        out=dicts["out"],
        vec=dicts["vec"],
        per_sample=dicts["per_sample"],
        terminated=dicts["terminated"],
        uniforms=dicts["uniforms"],
        n=n,
        tables=dicts["tables"],
        mutables=dicts["mutables"],
        mutable_vec=vec_mut,
        mutable_mat=mat_mut,
        mats=dicts["mats"],
        wide_in=dicts["wide_in"],
        wide_out=dicts["wide_out"],
        plan=plan,
    )


def _bytes_of(args):
    """The wire form of an argument list: the exact bytes each by-value POD
    would be packed from (a numpy scalar's ``tobytes``), so parity is asserted
    on CONTENT, never on object identity."""
    return [np.asarray(a).tobytes() for a in args]


# --------------------------------------------------------------------------- #
# The plan cache.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", sorted(_SIGNATURES))
def test_plan_tags_match_the_per_call_classifier_exactly(case):
    arg_spec, vec_mut, mat_mut = _SIGNATURES[case]
    plan = launch_plan(arg_spec, vec_mut, mat_mut)
    expected = tuple(
        classify_arg(r, nm, vec_mutables=set(vec_mut), mat_mutables=set(mat_mut))
        for r, nm in arg_spec
    )
    assert plan.tags == expected


def test_the_same_signature_returns_the_same_plan_object():
    arg_spec, vec_mut, mat_mut = _SIGNATURES["pure_vector_mutable"]
    first = launch_plan(arg_spec, vec_mut, mat_mut)
    for _ in range(5):
        assert launch_plan(list(arg_spec), tuple(vec_mut), tuple(mat_mut)) is first
    stats = _launch_plan_stats()
    assert stats == {"hits": 5, "misses": 1}, stats


def test_a_changed_signature_gets_its_own_plan():
    """Three ways a signature can differ — argument list, vector-Mutable set,
    matrix-Mutable set — must each key a DIFFERENT plan. A plan that survived
    any of these would bind the wrong ABI shape."""
    base_spec = [("mutable", "y"), ("vec_in", "x"), ("nsamples", "n")]
    plan = launch_plan(base_spec, ("y",), ())
    assert launch_plan(base_spec + [("uniform", "k")], ("y",), ()) is not plan
    assert launch_plan(base_spec, (), ()) is not plan
    assert launch_plan(base_spec, (), ("y",)) is not plan
    # ...and the differing plans really do disagree about the ABI shape.
    assert launch_plan(base_spec, ("y",), ()).tags[0] == "GREF_VEC"
    assert launch_plan(base_spec, (), ()).tags[0] == "HANDLE"
    assert launch_plan(base_spec, (), ("y",)).tags[0] == "GREF_MAT"


def test_red_leg_clearing_the_plan_cache_zeroes_the_hits():
    """RED LEG, executable: with the cache defeated (cleared between calls) the
    hit counter the gate above reads stays at zero."""
    arg_spec, vec_mut, mat_mut = _SIGNATURES["vector_kernel"]
    for _ in range(6):
        _reset_launch_plan_cache()
        launch_plan(arg_spec, vec_mut, mat_mut)
    assert _launch_plan_stats()["hits"] == 0


# --------------------------------------------------------------------------- #
# Planned vs unplanned assembly is byte-identical (the host-side
# half of the launch-parity claim; the device half is the corpus file).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", sorted(_SIGNATURES))
def test_planned_assembly_is_byte_identical_to_the_unplanned_path(case):
    arg_spec, vec_mut, mat_mut = _SIGNATURES[case]
    n = 137
    dicts = _kw_for(arg_spec, vec_mut, mat_mut, base_ptr=0x7F0000)

    unplanned = _assemble(arg_spec, set(vec_mut), set(mat_mut), dicts, n, plan=None)
    planned = _assemble(
        arg_spec, set(vec_mut), set(mat_mut), dicts, n,
        plan=launch_plan(arg_spec, vec_mut, mat_mut),
    )
    assert len(planned) == len(unplanned) == len(arg_spec)
    assert _bytes_of(planned) == _bytes_of(unplanned), (
        f"{case}: the LaunchPlan path produced different argument bytes than "
        "the unplanned path"
    )


@pytest.mark.parametrize("case", sorted(_SIGNATURES))
def test_a_second_planned_launch_rebinds_every_pointer(case):
    """The plan's POD boxes are REUSED and refilled in place. The property that
    makes that safe is that EVERY live field is rewritten on every call — a
    stale pointer surviving into the next launch would be a silent
    wrong-buffer bug, so it is pinned here against fully disjoint inputs."""
    arg_spec, vec_mut, mat_mut = _SIGNATURES[case]
    plan = launch_plan(arg_spec, vec_mut, mat_mut)

    first = _kw_for(arg_spec, vec_mut, mat_mut, base_ptr=0x100000)
    _assemble(arg_spec, set(vec_mut), set(mat_mut), first, 64, plan=plan)
    second = _kw_for(arg_spec, vec_mut, mat_mut, base_ptr=0x900000)
    args = _assemble(arg_spec, set(vec_mut), set(mat_mut), second, 512, plan=plan)

    reference = _assemble(arg_spec, set(vec_mut), set(mat_mut), second, 512, plan=None)
    assert _bytes_of(args) == _bytes_of(reference), (
        f"{case}: a reused plan box kept a value from the PREVIOUS launch"
    )


def test_planned_boxes_are_reused_not_reallocated():
    """The allocation the plan removes, observed: the SAME box objects come
    back from two planned assemblies of the same signature."""
    arg_spec, vec_mut, mat_mut = _SIGNATURES["vector_kernel"]
    plan = launch_plan(arg_spec, vec_mut, mat_mut)
    d = _kw_for(arg_spec, vec_mut, mat_mut, base_ptr=0x200000)
    a1 = _assemble(arg_spec, set(vec_mut), set(mat_mut), d, 8, plan=plan)
    a2 = _assemble(arg_spec, set(vec_mut), set(mat_mut), d, 8, plan=plan)
    struct_positions = [i for i, t in enumerate(plan.tags) if plan.boxes[i] is not None]
    assert struct_positions, "this signature has no by-value POD arguments"
    for i in struct_positions:
        assert a1[i] is a2[i] is plan.boxes[i]
    # ...whereas the unplanned path still allocates a fresh box per call.
    b1 = _assemble(arg_spec, set(vec_mut), set(mat_mut), d, 8, plan=None)
    b2 = _assemble(arg_spec, set(vec_mut), set(mat_mut), d, 8, plan=None)
    assert all(b1[i] is not b2[i] for i in struct_positions)


def test_assemble_args_refuses_a_plan_from_another_signature():
    arg_spec, vec_mut, mat_mut = _SIGNATURES["vector_kernel"]
    other = launch_plan([("nsamples", "n")], (), ())
    d = _kw_for(arg_spec, vec_mut, mat_mut, base_ptr=0x300000)
    with pytest.raises(ValueError, match="does not belong to this signature"):
        _assemble(arg_spec, set(vec_mut), set(mat_mut), d, 8, plan=other)


@pytest.mark.gpu
def test_launch_plan_carries_no_block_and_no_kernel(monkeypatch):
    """Asserted rather than asserted-in-prose. Block derivation is a
    PER-CALL function of ``n`` and the ambient sibling context, and two live
    spies (a downstream launch-policy twin) patch ``eagle.launch.resolve_block`` by
    name — so it must stay a module global read at call time, and no plan may
    carry a block, a kernel object or a device."""
    plan = launch_plan(*_SIGNATURES["vector_kernel"])
    assert set(LaunchPlan.__slots__) == {
        "arg_spec", "tags", "vec_mutables", "mat_mutables", "boxes"
    }
    assert not hasattr(plan, "block")

    seen = []
    monkeypatch.setattr(
        launch_mod, "resolve_block", lambda n, s, d, k: seen.append((n, s)) or 32
    )
    assert launch_mod._resolved_block(object(), 500, None) == 32
    assert launch_mod._resolved_block(object(), 900, None) == 32
    assert [n for n, _ in seen] == [500, 900], (
        "resolve_block was not consulted per call with the call's own n — the "
        "patchable-by-name, per-call contract is broken"
    )
    # the explicit override still bypasses the policy entirely
    assert launch_mod._resolved_block(object(), 500, 128) == 128


# --------------------------------------------------------------------------- #
# The conforming fast path.
# --------------------------------------------------------------------------- #
class _FakeCupyArray:
    def __init__(self, dtype=np.float64, c_contiguous=True, device_id=0):
        self.dtype = np.dtype(dtype)
        self.flags = types.SimpleNamespace(c_contiguous=c_contiguous)
        self.device = types.SimpleNamespace(id=device_id)


def _fake_cp(current_device=0):
    mod = types.ModuleType("cupy")
    mod.ndarray = _FakeCupyArray
    mod.bool_ = np.bool_
    mod.int64 = np.int64
    mod.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(getDevice=lambda: current_device)
    )
    mod.zeros = lambda n, dtype=None: _FakeCupyArray(dtype or np.float64)
    return mod


def test_a_conforming_cupy_array_takes_the_fast_path():
    cp = _fake_cp()
    assert marshal_mod._fast_path_ok(_FakeCupyArray(np.float64), np.float64, cp)
    assert marshal_mod._fast_path_ok(_FakeCupyArray(np.float32), np.float32, cp)
    assert marshal_mod._fast_path_ok(_FakeCupyArray(np.bool_), np.bool_, cp)
    assert marshal_mod._fast_path_ok(_FakeCupyArray(np.int64), np.int64, cp)


@pytest.mark.parametrize(
    "value, dt, why",
    [
        (_FakeCupyArray(np.float64, c_contiguous=False), np.float64, "strided"),
        (_FakeCupyArray(np.float32), np.float64, "wrong dtype"),
        (_FakeCupyArray(np.float64, device_id=1), np.float64, "wrong device"),
        (np.zeros(4), np.float64, "numpy, not cupy"),
        ([1.0, 2.0], np.float64, "a python list"),
        (3.5, np.float64, "a scalar"),
    ],
)
def test_a_non_conforming_input_never_takes_the_fast_path(value, dt, why):
    assert not marshal_mod._fast_path_ok(value, dt, _fake_cp()), (
        f"a {why} input was admitted to the fast path — it would be bound "
        "without the coercion that makes it launchable"
    )


def test_red_leg_a_strided_input_admitted_to_the_fast_path_fails_the_gate():
    """RED LEG, executable: a predicate that forgot the contiguity check (the
    single most damaging omission — a strided view has the wrong element
    stride and would be read as if packed) must be REJECTED by the same
    assertion the parametrized gate above uses."""
    cp = _fake_cp()
    strided = _FakeCupyArray(np.float64, c_contiguous=False)

    def _defeated_fast_path_ok(value, dt, cp):
        return isinstance(value, cp.ndarray) and value.dtype == dt

    assert _defeated_fast_path_ok(strided, np.float64, cp) is True
    with pytest.raises(AssertionError, match="admitted to the fast path"):
        assert not _defeated_fast_path_ok(strided, np.float64, cp), (
            "a strided input was admitted to the fast path"
        )


def test_force_slow_coercion_defeats_the_fast_path_for_every_input():
    """The launch-parity corpus's forced-slow arm — it must actually bite, or
    the corpus would compare the fast path against itself."""
    cp = _fake_cp()
    conforming = _FakeCupyArray(np.float64)
    assert marshal_mod._fast_path_ok(conforming, np.float64, cp)
    marshal_mod._force_slow_coercion(True)
    try:
        assert not marshal_mod._fast_path_ok(conforming, np.float64, cp)
    finally:
        marshal_mod._force_slow_coercion(False)
    assert marshal_mod._fast_path_ok(conforming, np.float64, cp)


def test_coercion_counters_move_with_the_path_taken(monkeypatch):
    import sys

    cp = _fake_cp()
    monkeypatch.setitem(sys.modules, "cupy", cp)
    marshal_mod._reset_coercion_stats()
    conforming = _FakeCupyArray(np.float64)
    assert marshal_mod._as_dtype(conforming, np.float64) is conforming
    assert marshal_mod._coercion_stats() == {"fast": 1, "slow": 0}


# --------------------------------------------------------------------------- #
# The opt-in zero mask.
# --------------------------------------------------------------------------- #
def test_undeclared_plugins_never_reach_the_zero_mask_cache(monkeypatch):
    """THE opt-in gate. Without a declaration, every omitted mask must still be
    a FRESH private buffer and the cache must record nothing at all — a plugin
    that writes its mask would silently corrupt every other plugin's if this
    ever defaulted on."""
    import sys

    monkeypatch.setitem(sys.modules, "cupy", _fake_cp())
    masks = [marshal_mod.coerce_terminated({}, 16) for _ in range(6)]
    assert marshal_mod._zero_mask_stats() == {"hits": 0, "misses": 0, "bypassed": 0}
    assert len({id(m) for m in masks}) == 6, "undeclared masks were shared"


def test_a_declaring_plugin_is_served_from_the_cache(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cupy", _fake_cp())
    masks = [
        marshal_mod.coerce_terminated({}, 16, readonly_mask=True) for _ in range(6)
    ]
    assert marshal_mod._zero_mask_stats() == {"hits": 5, "misses": 1, "bypassed": 0}
    assert all(m is masks[0] for m in masks)


def test_the_zero_mask_cache_keys_on_n_and_device(monkeypatch):
    import sys

    device = {"id": 0}
    cp = _fake_cp()
    cp.cuda.runtime.getDevice = lambda: device["id"]
    monkeypatch.setitem(sys.modules, "cupy", cp)

    m16 = marshal_mod.coerce_terminated({}, 16, readonly_mask=True)
    m32 = marshal_mod.coerce_terminated({}, 32, readonly_mask=True)
    assert m16 is not m32, "two batch sizes were served the same mask buffer"
    device["id"] = 1
    assert marshal_mod.coerce_terminated({}, 16, readonly_mask=True) is not m16, (
        "a mask allocated on another device was served — a device pointer is "
        "not portable across devices"
    )


def test_a_passed_mask_is_never_cached_under_either_setting(monkeypatch):
    """Only an OMITTED mask is cacheable. A caller-supplied mask is the
    caller's buffer and must be bound, checked and handed back as always."""
    import sys

    cp = _fake_cp()
    monkeypatch.setitem(sys.modules, "cupy", cp)
    supplied = _FakeCupyArray(np.bool_)
    supplied.shape = (16,)
    for flag in (False, True):
        got = marshal_mod.coerce_terminated(
            {"terminated": supplied}, 16, readonly_mask=flag
        )
        assert got is supplied
    assert marshal_mod._zero_mask_stats() == {"hits": 0, "misses": 0, "bypassed": 0}


def test_the_zero_mask_cache_bypasses_at_its_cap_instead_of_growing(monkeypatch):
    """Each retained mask pins ``n`` bytes of DEVICE memory for the life of the
    process, so a caller sweeping batch sizes must not grow it without bound.
    Past ``ZERO_MASK_CACHE_CAP`` the cache BYPASSES — a fresh private mask,
    exactly what an undeclared plugin gets — which is always correct here, so a
    silent fallback is the right answer at this seam (unlike the kernel-attrs
    cap, where a fallback would mean a driver query mid-replay)."""
    import sys

    monkeypatch.setitem(sys.modules, "cupy", _fake_cp())
    monkeypatch.setattr(marshal_mod, "ZERO_MASK_CACHE_CAP", 4)
    for n in range(1, 5):
        marshal_mod.coerce_terminated({}, n, readonly_mask=True)
    assert marshal_mod._zero_mask_stats()["misses"] == 4

    over = [
        marshal_mod.coerce_terminated({}, 99, readonly_mask=True) for _ in range(3)
    ]
    stats = marshal_mod._zero_mask_stats()
    assert stats["misses"] == 4, "the cache grew past its cap"
    assert stats["bypassed"] == 3
    assert len({id(m) for m in over}) == 3, "a bypassed mask was shared anyway"
    # ...and the entries already cached are still served.
    first = marshal_mod.coerce_terminated({}, 1, readonly_mask=True)
    assert marshal_mod.coerce_terminated({}, 1, readonly_mask=True) is first


def test_red_leg_a_default_on_mask_cache_fails_the_opt_in_gate(monkeypatch):
    """RED LEG, executable: a ``coerce_terminated`` whose cache is default-ON
    must be REJECTED by the opt-in assertion."""
    import sys

    cp = _fake_cp()
    monkeypatch.setitem(sys.modules, "cupy", cp)

    def _defeated_coerce_terminated(kw, n, *, readonly_mask=False):
        if kw.get("terminated") is None:
            return marshal_mod._cached_zero_mask(n, cp)  # <- ignores the flag
        raise AssertionError("unreachable in this leg")

    masks = [_defeated_coerce_terminated({}, 16) for _ in range(6)]
    with pytest.raises(AssertionError, match="undeclared masks were shared"):
        assert len({id(m) for m in masks}) == 6, "undeclared masks were shared"


# --------------------------------------------------------------------------- #
# The sidecar declaration's schema.
# --------------------------------------------------------------------------- #
def test_absent_declaration_reads_false():
    assert read_terminated_readonly({"kernel": "k"}, name="art") is False


@pytest.mark.parametrize("declared", [True, False])
def test_a_boolean_declaration_reads_through(declared):
    meta = {"kernel": "k", TERMINATED_READONLY_KEY: declared}
    assert read_terminated_readonly(meta, name="art") is declared


@pytest.mark.parametrize("bad", ["true", "false", 1, 0, "yes", [], {}])
def test_a_non_boolean_declaration_is_refused_loudly(bad):
    """Truthiness coercion is exactly the failure mode this declaration must
    not have: ``"false"`` is a non-empty string and would enable a SHARED
    buffer for a kernel whose producer meant the opposite."""
    meta = {"kernel": "k", TERMINATED_READONLY_KEY: bad}
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        read_terminated_readonly(meta, name="art.json")


def test_a_synthetic_declaring_sidecar_threads_the_flag_end_to_end(monkeypatch):
    """The synthetic DECLARING sidecar (the code generator's kernels will carry
    the same key in a later window): the loader reads it, and every layer down
    to ``coerce_terminated`` honors it. Threaded here without a driver load by
    driving the same functions the loader chains."""
    import sys

    monkeypatch.setitem(sys.modules, "cupy", _fake_cp())
    declaring = {
        "kernel": "raptor_kernel",
        "aether_abi": "aether-abi/1",
        "pattern": "pure",
        TERMINATED_READONLY_KEY: True,
        "arg_spec": [["mutable", "counter"], ["terminated", "terminated"]],
    }
    undeclaring = {k: v for k, v in declaring.items() if k != TERMINATED_READONLY_KEY}

    flag = read_terminated_readonly(declaring, name="declaring.json")
    for _ in range(4):
        marshal_mod.coerce_terminated({}, 8, readonly_mask=flag)
    assert marshal_mod._zero_mask_stats()["hits"] == 3

    marshal_mod._reset_zero_mask_cache()
    flag = read_terminated_readonly(undeclaring, name="plain.json")
    for _ in range(4):
        marshal_mod.coerce_terminated({}, 8, readonly_mask=flag)
    assert marshal_mod._zero_mask_stats() == {"hits": 0, "misses": 0, "bypassed": 0}


# --------------------------------------------------------------------------- #
# _kernel_attrs_cache caps and refuses (never evicts).
# --------------------------------------------------------------------------- #
class _CountingKernel:
    """A stand-in kernel whose ``.attributes`` counts DRIVER queries — the
    thing the cache exists to avoid on a replay-critical path."""

    def __init__(self, name="k"):
        self.name = name
        self.queries = 0

    @property
    def attributes(self):
        self.queries += 1
        return {"num_regs": 32, "max_threads_per_block": 1024}


def test_the_cap_is_a_named_generous_constant():
    assert isinstance(KERNEL_ATTRS_CACHE_CAP, int)
    assert KERNEL_ATTRS_CACHE_CAP >= 1024, (
        "the cap must be generous enough that reaching it means kernel churn, "
        "not that the cap is tight"
    )


def test_a_kernel_is_queried_once_and_served_from_the_cache_thereafter():
    k = _CountingKernel()
    for _ in range(5):
        assert launch_mod._kernel_attrs(k)["num_regs"] == 32
    assert k.queries == 1


def test_at_the_cap_a_new_kernel_is_refused_and_the_error_names_the_constant(
    monkeypatch,
):
    monkeypatch.setattr(launch_mod, "KERNEL_ATTRS_CACHE_CAP", 4)
    kernels = [_CountingKernel(f"k{i}") for i in range(4)]
    for k in kernels:
        launch_mod._kernel_attrs(k)

    overflow = _CountingKernel("overflow")
    with pytest.raises(KernelAttrsCacheFull) as excinfo:
        launch_mod._kernel_attrs(overflow)
    message = str(excinfo.value)
    assert "KERNEL_ATTRS_CACHE_CAP=4" in message
    assert "never evicts" in message
    assert overflow.queries == 0, "the refused kernel must not touch the driver"


def test_refusal_pins_refuse_semantics_not_eviction(monkeypatch):
    """The refuse-not-evict property, and the reason a lookup-after-overflow test is not
    enough on its own: an LRU-EVICTING implementation would also keep serving
    lookups after an overflow — it would just re-query the driver for the
    evicted kernel, mid-replay, which is precisely what this cache's contract
    forbids. So the gate pins REFUSAL: the overflow RAISES, and the
    FIRST-inserted kernel (an evictor's first victim) is still served from the
    cache with ZERO further driver queries."""
    monkeypatch.setattr(launch_mod, "KERNEL_ATTRS_CACHE_CAP", 4)
    kernels = [_CountingKernel(f"k{i}") for i in range(4)]
    for k in kernels:
        launch_mod._kernel_attrs(k)
    assert all(k.queries == 1 for k in kernels)

    with pytest.raises(KernelAttrsCacheFull):
        launch_mod._kernel_attrs(_CountingKernel("overflow"))

    for k in kernels:
        assert launch_mod._kernel_attrs(k)["num_regs"] == 32
    assert all(k.queries == 1 for k in kernels), (
        "an existing entry was re-queried from the driver after the overflow — "
        "the cache evicted instead of refusing"
    )


def test_red_leg_an_evicting_cache_does_not_refuse(monkeypatch):
    """RED LEG, executable: a bounded-EVICT implementation passes a
    lookup-after-overflow check and fails the refuse assertion — which is why
    the gate above is written the way it is."""
    monkeypatch.setattr(launch_mod, "KERNEL_ATTRS_CACHE_CAP", 4)
    evicting: dict = {}

    def _defeated_kernel_attrs(fn):
        if fn in evicting:
            return evicting[fn]
        if len(evicting) >= launch_mod.KERNEL_ATTRS_CACHE_CAP:
            evicting.pop(next(iter(evicting)))  # LRU-ish: drop the oldest
        evicting[fn] = fn.attributes
        return evicting[fn]

    kernels = [_CountingKernel(f"k{i}") for i in range(4)]
    for k in kernels:
        _defeated_kernel_attrs(k)
    _defeated_kernel_attrs(_CountingKernel("overflow"))  # no refusal at all

    with pytest.raises(Failed := type("Failed", (AssertionError,), {})):
        try:
            _defeated_kernel_attrs(_CountingKernel("overflow2"))
        except KernelAttrsCacheFull:  # pragma: no cover - the point is it does not
            raise AssertionError("unreachable") from None
        raise Failed("an evicting cache never refuses growth")

    _defeated_kernel_attrs(kernels[0])  # the evicted first entry
    assert kernels[0].queries == 2, (
        "this leg is meant to demonstrate the DRIVER RE-QUERY an evicting "
        "cache causes on a replay-critical path"
    )
