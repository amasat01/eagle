# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``wide_in`` / ``wide_out`` role classification + marshalling
(the wide-buffer vocabulary role + its VJP scatter
gradient).

Gate-A style (no cupy import needed anywhere in this file — :func:`make_handle`
only reads ``arr.data.ptr``, so a lightweight fake with that one attribute
exercises :func:`~eagle.launch.assemble_args`'s construction without a real
device array): pure classification (:func:`~eagle.roles.classify_arg`) +
argument-tuple construction (:func:`~eagle.launch.assemble_args`) +ctypes
packing (:class:`~eagle.host_launch.HostPluginLibrary.run`, built directly via
``object.__new__`` with a stub ``_fn``, bypassing ``__init__``'s sidecar
validation).

**Scope note (updated: the vocabulary extension landed).** ``eagle.roles.ROLES``
and the C++ ``plugin/roles.h::kPluginArgRoles`` now BOTH admit ``wide_in``/
``wide_out`` (a later, separately-authorized round), so a sidecar naming them
passes ``eagle.sidecar.validate_sidecar`` in-process (a plain frozenset check,
no rebuild needed). What still blocks a full end-to-end load: the COMPILED
``eagle._core`` extension embeds the header's PRE-extension role list until it
is rebuilt against this source (out of THIS round's scope — no device
execution / binding rebuild here). So every test in this file still calls the
classification/construction/packing functions DIRECTLY rather than through
``HostPluginLibrary.__init__`` / ``LoadedKernel``'s full load (which would
need the rebuilt binary to prove anything new) — this file remains the
correct, honest test surface until that rebuild lands.
"""

from __future__ import annotations

import ctypes

import pytest

from eagle.host_launch import HostPluginLibrary, ScalarHandle
from eagle.launch import assemble_args
from eagle.roles import ARG_TAGS, classify_arg


def test_arg_tags_includes_wide_in_and_wide_out():
    assert {"WIDE_IN", "WIDE_OUT"} <= ARG_TAGS


def test_classify_arg_wide_in():
    assert classify_arg("wide_in", "theta") == "WIDE_IN"


def test_classify_arg_wide_out():
    assert classify_arg("wide_out", "theta_bar") == "WIDE_OUT"


def test_classify_arg_wide_in_out_do_not_collide_with_handle():
    # wide_in/wide_out are their OWN tags (not folded into "HANDLE") -- assert
    # the split is real, not just documented.
    assert classify_arg("wide_in", "x") != classify_arg("per_sample", "x")
    assert classify_arg("wide_out", "x") != classify_arg("mutable", "x")


class _FakeArr:
    """A stand-in for a cupy array exposing ONLY ``.data.ptr`` -- the one
    attribute :func:`eagle.abi.make_handle` reads. No cupy import needed."""

    class _Data:
        def __init__(self, ptr):
            self.ptr = ptr

    def __init__(self, ptr):
        self.data = self._Data(ptr)


def test_assemble_args_wide_in_pulls_from_its_own_dict():
    arg_spec = [("wide_in", "theta"), ("nsamples", "nsamples")]
    args = assemble_args(
        arg_spec, out=None, vec={}, per_sample={}, terminated=None,
        uniforms={}, n=8, wide_in={"theta": _FakeArr(0xABCD)},
    )
    assert args[0]["data"] == 0xABCD
    assert int(args[1]) == 8


def test_assemble_args_wide_out_pulls_from_its_own_dict():
    arg_spec = [("wide_out", "theta_bar")]
    args = assemble_args(
        arg_spec, out=None, vec={}, per_sample={}, terminated=None,
        uniforms={}, n=8, wide_out={"theta_bar": _FakeArr(0x1234)},
    )
    assert args[0]["data"] == 0x1234


def test_assemble_args_missing_wide_in_raises_keyerror():
    arg_spec = [("wide_in", "theta")]
    with pytest.raises(KeyError):
        assemble_args(
            arg_spec, out=None, vec={}, per_sample={}, terminated=None,
            uniforms={}, n=8,
        )  # wide_in= omitted entirely -> empty dict -> KeyError on 'theta'


def _bare_host_library(arg_spec) -> HostPluginLibrary:
    """A HostPluginLibrary built WITHOUT __init__ (bypasses validate_sidecar,
    the ROLES gate this file's module docstring explains) -- just enough
    state for .run()'s packing logic to exercise the new WIDE_IN/WIDE_OUT
    branches against a stub entry function."""
    lib = object.__new__(HostPluginLibrary)
    lib._arg_spec = arg_spec
    lib._vec_mut = set()
    lib._mat_mut = set()
    lib._vec = {}
    lib._mat = {}
    lib._handle = {}
    lib._uniform = {}
    lib._tables = {}
    lib._buffers = {}
    return lib


def test_host_plugin_library_run_packs_wide_in_and_wide_out():
    captured = {}

    def _stub_fn(params, n):
        # decode both ScalarHandle-shaped boxes back out to prove the RIGHT
        # pointer landed at the RIGHT arg-spec position.
        captured["theta_ptr"] = ctypes.cast(
            params[0], ctypes.POINTER(ScalarHandle)
        )[0].data
        captured["theta_bar_ptr"] = ctypes.cast(
            params[1], ctypes.POINTER(ScalarHandle)
        )[0].data
        captured["n"] = n.value if isinstance(n, ctypes.c_int32) else n

    lib = _bare_host_library([("wide_in", "theta"), ("wide_out", "theta_bar")])
    lib._fn = _stub_fn
    lib.bind_handle("theta", 0xABCD)
    lib.bind_handle("theta_bar", 0x1234)
    lib.run(8)

    assert captured["theta_ptr"] == 0xABCD
    assert captured["theta_bar_ptr"] == 0x1234
    assert captured["n"] == 8


def test_host_plugin_library_run_unbound_wide_raises_keyerror():
    lib = _bare_host_library([("wide_in", "theta")])
    lib._fn = lambda params, n: None
    with pytest.raises(KeyError):
        lib.run(8)  # 'theta' never bound via bind_handle


def test_validate_roles_now_accepts_wide_in_and_wide_out():
    """The one thing THIS round's C++/Python vocabulary extension actually
    unblocks: eagle.roles.validate_roles (the gate every loader's
    validate_sidecar runs before classify_arg ever sees a role) no longer
    rejects wide_in/wide_out -- a plain frozenset check, no rebuild needed.
    (The compiled eagle._core extension is the separate, NOT-done-here half;
    see this file's module docstring.)"""
    from eagle.roles import ROLES, validate_roles

    assert {"wide_in", "wide_out"} <= ROLES
    # no raise
    validate_roles([("wide_in", "theta"), ("wide_out", "theta_bar")], name="t")
