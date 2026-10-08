# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The INT broadcast uniform, eagle's Python half.

ENGINE_CLEANUP_ADDENDUM's int-role phase, consumers window. Window 7 landed the
C++ authority (``eagle::plugin::UniformInt`` = int64, ``bind_uniform_int`` beside
``bind_uniform`` in both registries, one name refused as two types in both
directions). This file covers the PYTHON half of the same contract:

* the decl-carrying, VERSIONED sidecar ``params`` block
  (:func:`eagle.sidecar.read_params`) — accepting the v1 bare name list AND the
  v2 ``{name, dtype}`` list, refusing a newer version LOUDLY rather than
  guessing a uniform's by-value type;
* the ONE params-shape normalizer every launch path dispatches on
  (:func:`eagle.roles.param_decls`);
* the int arm of :func:`eagle.marshal.coerce_uniforms` — ``np.int64``, never
  ``dt(...)``;
* :meth:`eagle.host_launch.HostPluginLibrary.bind_uniform_int` and its
  ``ctypes.c_longlong`` pack, end to end through a real dlopen'd plugin.

The sentinel is 2^32+1: it catches a 32-bit truncation (which lands on 1) AND a
by-value type mismatch (the kernel reading 8 float bytes as an integer, which
lands nowhere near it). It is exactly representable in a double, so where the
claim is specifically "no float64 is in this path" the value is 2^53+1, which a
double cannot carry.

Host-only: the packing tests dlopen a tiny g++-built plugin (the
``test_engine_hardening_small_caches`` fixture pattern — ABI-only);
the device-path assertions that need a driver are ``gpu``-marked.
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from eagle import interop, marshal
from eagle.host_launch import HostPluginLibrary
from eagle.roles import PARAMS_SCHEMA, param_decls, param_names
from eagle.sidecar import ParamSpec, read_params

#: See the module docstring.
K_SENTINEL = 2**32 + 1
BEYOND_DOUBLE = 2**53 + 1

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# 1 — read_params: both wire shapes, and a loud refusal for anything else
# --------------------------------------------------------------------------- #
def test_v1_bare_name_list_reads_as_float():
    """BACK-COMPAT read: every artifact written before this change carries no
    ``params_schema`` and a bare name list."""
    got = read_params({"params": ["mu", "k"]}, name="old.ptx")
    assert got == (ParamSpec("mu", "float"), ParamSpec("k", "float"))


def test_v1_explicit_version_reads_the_same():
    assert read_params({"params_schema": 1, "params": ["mu"]}, name="x") == (
        ParamSpec("mu", "float"),
    )


def test_v2_decl_list_carries_the_dtype():
    got = read_params(
        {
            "params_schema": 2,
            "params": [{"name": "mu"}, {"name": "k", "dtype": "int"}],
        },
        name="new.ptx",
    )
    assert got == (ParamSpec("mu", "float"), ParamSpec("k", "int"))


def test_a_newer_params_schema_is_refused_by_version():
    """The version leg: a block written by a FUTURE producer must be
    refused BY VERSION and name the upgrade — never bound by guessing at the
    entry shape, because guessing wrong makes an int uniform a double."""
    with pytest.raises(ValueError) as e:
        read_params(
            {"params_schema": PARAMS_SCHEMA + 1, "params": []}, name="future.ptx"
        )
    msg = str(e.value)
    assert f"'params_schema' is {PARAMS_SCHEMA + 1}" in msg
    assert "upgrade eagle" in msg


def test_a_shape_that_disagrees_with_its_declared_version_is_refused():
    """A decl-carrying entry tagged v1 would silently lose its dtype; a
    bare name tagged v2 is a half-migrated producer. Both are refused naming the
    version they declared."""
    with pytest.raises(ValueError, match="is not a bare name"):
        read_params({"params": [{"name": "k", "dtype": "int"}]}, name="x")
    with pytest.raises(ValueError, match="is not a .*name, dtype.* object"):
        read_params({"params_schema": 2, "params": ["k"]}, name="x")


def test_unknown_dtype_and_malformed_entries_are_refused():
    with pytest.raises(ValueError, match="unknown dtype"):
        read_params(
            {"params_schema": 2, "params": [{"name": "k", "dtype": "half"}]},
            name="x",
        )
    with pytest.raises(ValueError, match="has no 'name'"):
        read_params({"params_schema": 2, "params": [{"dtype": "int"}]}, name="x")
    with pytest.raises(ValueError, match="must be an integer"):
        read_params({"params_schema": "2", "params": []}, name="x")
    with pytest.raises(ValueError, match="first version is 1"):
        read_params({"params_schema": 0, "params": []}, name="x")


def test_required_flag_matches_the_two_calling_conventions():
    assert read_params({}, name="x", required=False) == ()
    with pytest.raises(KeyError):
        read_params({}, name="x")


@pytest.mark.parametrize("fixture", ["gravity.json", "bump.json", "drag_ps.json"])
def test_the_committed_v1_fixtures_still_read(fixture):
    """The committed sidecar fixtures are v1 artifacts and STAY v1 — they are the
    back-compat leg of this change, not something to re-mint. Every entry must
    read as a float param, exactly what it meant when it was written."""
    meta = json.loads((FIXTURES / fixture).read_text())
    assert "params_schema" not in meta
    for spec in read_params(meta, name=fixture):
        assert spec.dtype == "float"


# --------------------------------------------------------------------------- #
# 2 — param_decls: ONE normalizer, four legitimate producer shapes
# --------------------------------------------------------------------------- #
def test_param_decls_normalizes_every_producer_shape():
    # the duck type the code generator's ParamDecl / eagle's ParamSpec satisfy
    class _Decl:
        def __init__(self, name, dtype):
            self.name, self.dtype = name, dtype

    assert param_decls(["mu"]) == (("mu", "float"),)
    assert param_decls([("k", "int")]) == (("k", "int"),)
    assert param_decls([{"name": "k", "dtype": "int"}]) == (("k", "int"),)
    assert param_decls([{"name": "k"}]) == (("k", "float"),)
    assert param_decls([_Decl("k", "int")]) == (("k", "int"),)
    assert param_decls([ParamSpec("k", "int")]) == (("k", "int"),)
    assert param_names([ParamSpec("k", "int"), "mu"]) == ("k", "mu")


def test_param_decls_refuses_an_entry_it_cannot_type():
    """It CANNOT silently skip or default an unrecognised entry: a uniform whose
    type cannot be established is the silent-wrong-answer this normalization
    exists to prevent (applied to the params list)."""
    with pytest.raises(TypeError, match="neither a name nor a declared dtype"):
        param_decls([3.5])


# --------------------------------------------------------------------------- #
# 3 — coerce_uniforms: the int arm never goes through the Real dtype
# --------------------------------------------------------------------------- #
def test_coerce_uniforms_float_arm_is_unchanged():
    out = marshal.coerce_uniforms({"mu": 1.5}, ["mu"])
    assert out["mu"].dtype == np.float64 and out["mu"] == 1.5
    out32 = marshal.coerce_uniforms({"mu": 1.5}, ["mu"], dt=np.float32)
    assert out32["mu"].dtype == np.float32


def test_coerce_uniforms_int_arm_is_exact_beyond_double_precision():
    """The marshal site, stated as a differential: the int arm must carry
    2^53+1, and the float arm — the earlier path — provably cannot."""
    out = marshal.coerce_uniforms(
        {"k": BEYOND_DOUBLE}, [ParamSpec("k", "int")]
    )
    assert out["k"].dtype == np.int64
    assert int(out["k"]) == BEYOND_DOUBLE
    # the defeated path, for contrast: dt(...) is what every uniform used to take
    assert int(np.int64(np.float64(BEYOND_DOUBLE))) != BEYOND_DOUBLE


def test_coerce_uniforms_int_arm_ignores_the_real_dtype():
    """An integer slot is unrelated to the kernel's Real precision — the same
    rule ``coerce_mutable``'s int slot already follows."""
    out = marshal.coerce_uniforms(
        {"k": 7}, [ParamSpec("k", "int")], dt=np.float32
    )
    assert out["k"].dtype == np.int64


def test_coerce_int_uniform_refuses_what_it_cannot_carry_exactly():
    assert int(marshal.coerce_int_uniform("k", 2.0)) == 2  # exact, spelled float
    with pytest.raises(TypeError, match="non-integral"):
        marshal.coerce_int_uniform("k", 2.5)
    with pytest.raises(TypeError, match="bool"):
        marshal.coerce_int_uniform("k", True)
    with pytest.raises((TypeError, OverflowError, ValueError)):
        marshal.coerce_int_uniform("k", 2**63)


def test_coerce_uniforms_names_a_missing_param():
    with pytest.raises(TypeError, match="missing required parameter 'k'"):
        marshal.coerce_uniforms({}, [ParamSpec("k", "int")])


def test_coerce_uniforms_refuses_an_unknown_declared_dtype():
    with pytest.raises(ValueError, match="unknown dtype"):
        marshal.coerce_uniforms({"k": 1}, [("k", "half")])


# --------------------------------------------------------------------------- #
# 4 — the host plugin: bind_uniform_int -> c_longlong, end to end
# --------------------------------------------------------------------------- #
# out[i] = k (an INT uniform copied straight into an int64 per-sample slot).
# Depends only on the ABI POD, exactly like a deployed plugin — and reads the
# uniform slot as ``long long``, which is the declaration a generated
# kernel makes (``GRID_CONSTANT() Int p_k``, ``using Int = long long``).
_PLUGIN_SRC = r"""
#include <stdint.h>
struct ScalarHandle { void* ptr; };
extern "C" void a10_intecho_host(void* const* p, int n) {
    long long* out    = (long long*)((const ScalarHandle*)p[0])->ptr;
    const long long k = *(const long long*)p[1];
    const uint32_t ns = *(const uint32_t*)p[2];
    const int lim = (int)ns < n ? (int)ns : n;
    for (int i = 0; i < lim; ++i) out[i] = k;
}
"""

_V2_SIDECAR = {
    "kernel": "a10_intecho",
    "aether_abi": "aether-abi/1",
    "host_entry": "a10_intecho_host",
    "params_schema": 2,
    "params": [{"name": "k", "dtype": "int"}],
    "arg_spec": [
        ["mutable", "out"],
        ["uniform", "k"],
        ["nsamples", "n"],
    ],
    "mutables": [{"name": "out", "dtype": "int", "width": 1}],
}


def _gxx():
    if shutil.which(os.environ.get("CXX", "")):
        return os.environ.get("CXX")
    return shutil.which("g++") or (
        "/usr/bin/g++" if os.path.exists("/usr/bin/g++") else None
    )


@pytest.fixture(scope="module")
def int_plugin_so(tmp_path_factory):
    gxx = _gxx()
    if gxx is None:
        pytest.skip("no g++ available to build the host plugin fixture")
    tmp_path = tmp_path_factory.mktemp("a10_intecho")
    src = tmp_path / "a10_intecho_host.cpp"
    src.write_text(_PLUGIN_SRC)
    so = tmp_path / "a10_intecho_host.so"
    proc = subprocess.run(
        [gxx, "-O2", "-shared", "-fPIC", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"host plugin compile failed: {proc.stderr.strip()[:400]}")
    return str(so)


def test_host_int_uniform_arrives_exact(int_plugin_so):
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    n = 8
    out = np.zeros(n, dtype=np.int64)
    lib.bind_handle("out", interop.host_ptr(out))
    lib.bind_uniform_int("k", K_SENTINEL)
    assert lib.run(n) == n
    assert np.array_equal(out, np.full(n, K_SENTINEL, dtype=np.int64))


def test_host_int_uniform_carries_a_value_no_double_can(int_plugin_so):
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    n = 4
    out = np.zeros(n, dtype=np.int64)
    lib.bind_handle("out", interop.host_ptr(out))
    lib.bind_uniform_int("k", BEYOND_DOUBLE)
    lib.run(n)
    assert int(out[0]) == BEYOND_DOUBLE


def test_host_int_uniform_red_leg_float_pack_corrupts(int_plugin_so):
    """The pack site. Defeat the int arm — write the value into the
    FLOAT binding dict, so :meth:`run` packs a ``c_double`` — and the same
    plugin must NOT produce the sentinel: it reads the double's 8 bytes as a
    ``long long``.

    Without this leg the ``c_longlong`` branch could be dead and every assertion
    above would still pass on whatever path happened to run."""
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    n = 4
    out = np.zeros(n, dtype=np.int64)
    lib.bind_handle("out", interop.host_ptr(out))
    # the earlier line, reaching past the guards straight into the packer
    lib._uniform["k"] = float(K_SENTINEL)
    lib._pack_gen += 1
    lib.run(n)
    assert int(out[0]) != K_SENTINEL, (
        "the defeated float pack produced the right answer — the c_longlong "
        "branch is not what is carrying this value, so this gate proves nothing"
    )
    # ...and the real binder, same instance, same .so, is exact. The defeat left
    # a float64 binding under 'k' that the kind guard would (correctly) refuse to
    # sit beside — drop it first: this leg is about the PACK, not the guard.
    lib._uniform.pop("k")
    lib.bind_uniform_int("k", K_SENTINEL)
    lib.run(n)
    assert int(out[0]) == K_SENTINEL


def test_host_int_uniform_participates_in_the_arg_pack_cache(int_plugin_so):
    """The same cache rule still holds for the new binder: an idempotent rebind must not
    invalidate the cached pack, and a real change must."""
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    n = 4
    out = np.zeros(n, dtype=np.int64)
    for _ in range(3):
        lib.bind_handle("out", interop.host_ptr(out))
        lib.bind_uniform_int("k", K_SENTINEL)
        lib.run(n)
    assert lib._pack_stats() == {"hits": 2, "misses": 1}
    lib.bind_uniform_int("k", K_SENTINEL + 1)
    lib.run(n)
    assert lib._pack_stats()["misses"] == 2
    assert int(out[0]) == K_SENTINEL + 1


def test_binding_one_name_as_two_kinds_is_refused_both_directions(int_plugin_so):
    """The Python peer of ``eagle::plugin::check_uniform_kind_free`` — one name,
    one by-value slot, refused in BOTH directions."""
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    lib.bind_uniform_int("k", 3)
    with pytest.raises(ValueError, match="already bound as int64"):
        lib.bind_uniform("k", 3.0)

    float_sidecar = dict(_V2_SIDECAR)
    float_sidecar["params"] = ["k"]
    float_sidecar.pop("params_schema")
    lib2 = HostPluginLibrary(int_plugin_so, float_sidecar)
    lib2.bind_uniform("k", 3.0)
    with pytest.raises(ValueError, match="already bound as float64"):
        lib2.bind_uniform_int("k", 3)


def test_the_binder_is_cross_checked_against_the_sidecar_declaration(int_plugin_so):
    """The check the C++ registries structurally cannot make: they never parse
    ``params``, so they compare the caller only against ITSELF. This loader reads
    the declaration, so it compares the caller against the ARTIFACT — the
    "remaining seam" plugin/plugin_registry/uniform_binding.h names, closed on
    the side that has the data."""
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    with pytest.raises(ValueError, match="is DECLARED 'int'"):
        lib.bind_uniform("k", 3.0)

    float_sidecar = dict(_V2_SIDECAR)
    float_sidecar["params"] = [{"name": "k", "dtype": "float"}]
    lib2 = HostPluginLibrary(int_plugin_so, float_sidecar)
    with pytest.raises(ValueError, match="is DECLARED 'float'"):
        lib2.bind_uniform_int("k", 3)


def test_a_v1_sidecar_leaves_the_binder_uncross_checked(int_plugin_so):
    """Absence stays LENIENT, like every other earlier declaration in this
    loader: a v1 artifact declares no dtype, so the caller's choice is the only
    authority (exactly the C++ registries' situation)."""
    v1 = dict(_V2_SIDECAR)
    v1["params"] = ["k"]
    v1.pop("params_schema")
    lib = HostPluginLibrary(int_plugin_so, v1)
    lib.bind_uniform_int("k", K_SENTINEL)  # not refused
    n = 2
    out = np.zeros(n, dtype=np.int64)
    lib.bind_handle("out", interop.host_ptr(out))
    lib.run(n)
    assert int(out[0]) == K_SENTINEL


def test_int_binder_refuses_what_it_cannot_carry_exactly(int_plugin_so):
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    with pytest.raises(TypeError, match="non-integral"):
        lib.bind_uniform_int("k", 2.5)
    with pytest.raises(TypeError, match="bool"):
        lib.bind_uniform_int("k", True)
    with pytest.raises(ValueError, match="64-bit signed"):
        lib.bind_uniform_int("k", 2**63)
    lib.bind_uniform_int("k", 2.0)  # an exact integer spelled as a float is fine


def test_a_future_params_schema_is_refused_before_dlopen(int_plugin_so):
    """The version gate runs at CONSTRUCTION, so a params block this build cannot
    bind is refused before the library is even loaded — the validate-before-load
    discipline every other sidecar gate in this loader follows."""
    future = dict(_V2_SIDECAR)
    future["params_schema"] = PARAMS_SCHEMA + 1
    with pytest.raises(ValueError, match=f"'params_schema' is {PARAMS_SCHEMA + 1}"):
        HostPluginLibrary(int_plugin_so, future)


def test_the_uniform_box_is_a_longlong_not_a_double(int_plugin_so):
    """Structural, for the record: the packed POD for an int uniform is the
    8-byte SIGNED slot ``GRID_CONSTANT() Int p_<name>`` occupies."""
    lib = HostPluginLibrary(int_plugin_so, _V2_SIDECAR)
    n = 2
    out = np.zeros(n, dtype=np.int64)
    lib.bind_handle("out", interop.host_ptr(out))
    lib.bind_uniform_int("k", K_SENTINEL)
    lib.run(n)
    boxes = lib._pack_cache[2]
    (uniform_box,) = [b for b in boxes if isinstance(b, ctypes.c_longlong)]
    assert uniform_box.value == K_SENTINEL
    assert ctypes.sizeof(uniform_box) == 8


# --------------------------------------------------------------------------- #
# 6 — the DEVICE loader (gpu-marked: needs cupy + a driver to load the module)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_loaded_pure_reads_a_v2_params_block(tmp_path):
    """``LoadedKernel.params`` is decl-carrying, and ``param_names`` is the
    membership view its ``_allowed`` set is built from.

    Driven off a COPY of the committed ``bump`` fixture with a v2 sidecar
    written beside it — the committed fixtures themselves stay v1 on purpose
    (they are this change's back-compat corpus; see
    ``test_the_committed_v1_fixtures_still_read``)."""
    from eagle.loaded import LoadedPure

    meta = json.loads((FIXTURES / "bump.json").read_text())
    meta["params_schema"] = 2
    meta["params"] = [{"name": "rate", "dtype": "float"}]
    ptx = tmp_path / "bump.ptx"
    ptx.write_bytes((FIXTURES / "bump.ptx").read_bytes())
    ptx.with_suffix(".json").write_text(json.dumps(meta))

    loaded = LoadedPure(ptx)
    assert loaded.params == (ParamSpec("rate", "float"),)
    assert loaded.param_names == ("rate",)
    assert "rate" in loaded._allowed


@pytest.mark.gpu
def test_loaded_pure_still_reads_the_committed_v1_fixture(tmp_path):
    """The same loader, the same artifact, the v1 sidecar it actually ships
    with: unchanged behaviour, read as a float param."""
    from eagle.loaded import LoadedPure

    ptx = tmp_path / "bump.ptx"
    ptx.write_bytes((FIXTURES / "bump.ptx").read_bytes())
    ptx.with_suffix(".json").write_bytes((FIXTURES / "bump.json").read_bytes())

    loaded = LoadedPure(ptx)
    assert loaded.params == (ParamSpec("rate", "float"),)
    assert loaded.param_names == ("rate",)


# --------------------------------------------------------------------------- #
# 7 — what an OLD reader does with a NEW sidecar (stated, not assumed)
# --------------------------------------------------------------------------- #
def test_the_pre_a10_reader_body_refuses_a_v2_params_block():
    """The honest statement of this change's compatibility story.

    A reader that predates this change cannot be upgraded retroactively, so what it does
    with a v2 block is a fact to establish, not a design choice. Replay its exact
    body — ``tuple(meta["params"])`` then the ``set(...)`` every ``Loaded*``
    builds its binding surface from — and it RAISES (a dict is unhashable). It
    refuses; it does NOT misread, and it never binds a uniform of a type it did
    not understand.

    What it does NOT do is refuse *by version*, and that is deliberate rather
    than accidental: the plugin ``schema_version`` is a four-way pinned number
    whose C++ home (``plugin/roles.h``) is forward-strict, so bumping it would
    make EVERY artifact this producer writes unloadable by any C++ registry not
    rebuilt in the same change — and it would buy nothing, because the C++ side
    does not read the ``params`` block at all (verified: zero occurrences of
    ``params`` in ``plugin/sidecar.h``; it resolves uniforms by ``arg_spec``
    role). ``params_schema`` is therefore an ADDITIVE optional top-level key,
    which schema v1 explicitly admits and both languages ignore — the property
    conformance row 11 exists to hold — and it is forward-strict for every reader
    from this change on (``test_a_newer_params_schema_is_refused_by_version``)."""
    v2 = {"params": [{"name": "k", "dtype": "int"}]}

    def _pre_a10_reader(meta):
        params = tuple(meta["params"])  # eagle/loaded.py, earlier reader
        return set(params)  # LoadedVector/_allowed, earlier reader

    with pytest.raises(TypeError, match="unhashable"):
        _pre_a10_reader(v2)

    # ...and the SAME body is untouched by a v1 block, which is what keeps every
    # artifact written before this change loadable by every reader, old and new.
    assert _pre_a10_reader({"params": ["k"]}) == {"k"}


def test_the_cpp_sidecar_parser_has_no_params_reader():
    """The premise the whole versioning decision rests on, checked against the
    source rather than asserted: ``plugin/sidecar.h`` — the C++ parse/validate
    half — contains no ``params`` reader at all, so no C++ loader can misread
    the v2 shape. Skipped if the C++ tree is not beside this checkout."""
    header = pathlib.Path(__file__).resolve().parents[2] / "plugin" / "sidecar.h"
    if not header.exists():
        pytest.skip("eagle C++ tree not present beside this checkout")
    text = header.read_text()
    assert '"params"' not in text
    # ``params`` appears only as the launch-argument array's own name, never as a
    # sidecar KEY — pin the key form specifically.
    assert "sc.params" not in text and ".params " not in text
