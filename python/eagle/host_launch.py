# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""CPU-plugin launcher -- the host twin of :mod:`eagle.registry`, and the
Python peer of the C++ ``eagle::cpu::PluginRegistry`` (``plugin/host_registry.h``).

Where :mod:`eagle.registry` launches CUDA kernels over cupy device arrays,
this module ``ctypes``-loads a host plugin ``.so`` and calls its
``<kernel>_host`` entry over **host** buffers -- numpy arrays or torch CPU
tensors, zero-copy via their raw data pointer (:func:`eagle.interop.host_ptr`).

It shares the SAME binary ABI as the device path: the entry is
``void <kernel>_host(void* const* params, int32_t n)``, each ``params[i]``
pointing to the same ``GRefMirror`` / ``ScalarHandle`` POD as
``plugin/gref_abi.h``, packed in ``arg_spec`` order -- the array the C++
registry would hand ``cuLaunchKernel``, here handed to a host function
running its own OpenMP loop. A CPU plugin is the same artifact contract as
a GPU plugin, minus the PTX.
"""

from __future__ import annotations

import ctypes
import math

from .abi import DEVICE_CPU, check_aether_abi
from .roles import check_launch_certified_pattern, classify_arg
from .sidecar import PARAMS_SCHEMA_KEY, read_params, validate_sidecar


class GRefMirror(ctypes.Structure):
    """ctypes mirror of ``plugin/gref_abi.h`` ``GRefMirror`` (a 40-byte
    width-independent POD view)."""

    _fields_ = [
        ("data_", ctypes.c_void_p),
        ("samples_", ctypes.c_uint64),
        ("compStride_", ctypes.c_uint64),
        ("sampleStride_", ctypes.c_uint64),
        ("deviceType_", ctypes.c_int32),
        ("deviceId_", ctypes.c_int32),
    ]


class ScalarHandle(ctypes.Structure):
    """ctypes mirror of ``plugin/gref_abi.h`` ``ScalarHandle`` (a 32-byte
    POD, GRefMirror's rank-1 sibling; carries both an extent and a device
    tag, unlike the earlier bare-pointer ``HandleT``)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("samples", ctypes.c_uint64),
        ("stride", ctypes.c_uint64),
        ("deviceType", ctypes.c_int32),
        ("deviceId", ctypes.c_int32),
    ]


# Guard the layouts at import -- the C++ side static_asserts the same sizes.
assert ctypes.sizeof(GRefMirror) == 40, (
    "GRefMirror must be 40 B (match aether View<T,extents<N,dyn>,layout_stride>)"
)
assert ctypes.sizeof(ScalarHandle) == 32, (
    "ScalarHandle must be 32 B (match aether View<T,extents<dyn>,layout_stride>)"
)


class HostPluginLibrary:
    """A dlopen'd CPU plugin, driven from Python over host buffers.

    Mirrors ``eagle::cpu::PluginRegistry``: bind the kernel's by-name
    buffers from host pointers the caller already owns (see
    :func:`eagle.interop.host_ptr`), then :meth:`run` packs the args in
    ``arg_spec`` order and calls the host entry.

    ``so_path`` is the plugin shared object; ``sidecar`` is the same dict
    the C++ registry and code generator speak: ``kernel``, ``aether_abi``
    (required), optional ``host_entry`` / ``scalar_type`` (must be
    absent/``"float64"``), ``arg_spec`` (``[role, name]`` pairs), and
    optional ``mutables`` / ``mat_shapes``.
    """

    # Class-level defaults so an instance built by ``object.__new__`` (the
    # classification/packing unit tests, bypassing sidecar validation) is
    # still correct; every write below shadows these per instance.
    _pack_gen = 0
    _pack_cache: tuple | None = None
    _pack_hits = 0
    _pack_misses = 0
    # ``None`` not ``{}``: a class-level mutable default would be shared
    # across instances the moment one bound an int uniform.
    _uniform_int: dict | None = None
    _param_dtype: dict | None = None

    def __init__(self, so_path, sidecar: dict):
        # Gate order: shared validate_sidecar -> launch certification ->
        # loader-specific gates. Family refusal precedes ABI-tag
        # complaints, since a descriptor at a host door is the wrong KIND.
        validate_sidecar(sidecar, name=f"host plugin '{sidecar.get('kernel', '?')}'")
        # The manifest-bypassing door (C++ ``add_plugin``'s mirror), so it
        # runs its own launch certification.
        check_launch_certified_pattern(
            sidecar.get("pattern"),
            subject=f"host plugin '{sidecar.get('kernel', '?')}':",
        )
        # Presence-required, like every other loader in both languages.
        check_aether_abi(sidecar, kind="host plugin", name=sidecar.get("kernel", "?"))
        # The CPU runner executes NATIVE float64 only (the same gate the
        # C++ registry enforces); checked before dlopen.
        scalar_type = sidecar.get("scalar_type", "")
        if scalar_type and scalar_type != "float64":
            raise ValueError(
                f"host plugin '{sidecar.get('kernel', '?')}': the host runner "
                f"executes float64 only (got '{scalar_type}')"
            )
        self._lib   = ctypes.CDLL(str(so_path))
        entry       = sidecar.get("host_entry") or (sidecar["kernel"] + "_host")
        self._fn    = getattr(self._lib, entry)
        self._fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32]
        self._fn.restype  = None

        self._arg_spec = [tuple(a) for a in sidecar["arg_spec"]]  # [(role, name), ...]
        self._vec_mut  = {
            m["name"]
            for m in sidecar.get("mutables", [])
            if m.get("dtype") == "vector"
        }
        # A matrix input/Mutable binds through the same GRefMirror as a
        # vector (width-(R*C)); its declared (R, C) is read here only to
        # let :meth:`bind_matrix` reject a shape mismatch.
        self._mat_mut = {
            m["name"]
            for m in sidecar.get("mutables", [])
            if m.get("dtype") == "matrix"
        }
        self._mat_shape: dict[str, tuple[int, int]] = {
            name: tuple(rc) for name, rc in sidecar.get("mat_shapes", {}).items()
        }
        self._mat_shape.update(
            {
                m["name"]: tuple(m["shape"])
                for m in sidecar.get("mutables", [])
                if m.get("dtype") == "matrix"
            }
        )
        self._vec: dict[str, tuple[int, int]] = {}
        self._mat: dict[str, tuple[int, int]] = {}
        self._handle: dict[str, int]          = {}
        self._uniform: dict[str, float]       = {}
        # A ``uniform`` occupies one by-value slot with two spellings
        # (``Real`` / ``Int``); two dicts so the float path is untouched.
        self._uniform_int: dict[str, int]     = {}
        # The declared dtype per uniform (off the sidecar's params block);
        # lets this loader cross-check the caller against the artifact,
        # which the C++ side (never parsing ``params``) cannot do.
        _param_specs = read_params(
            sidecar,
            name=f"host plugin '{sidecar.get('kernel', '?')}'",
            required=False,
        )
        # Absence stays lenient (a v1 params block says nothing about type).
        self._param_dtype: dict[str, str] = (
            {spec.name: spec.dtype for spec in _param_specs}
            if sidecar.get(PARAMS_SCHEMA_KEY, 1) >= 2
            else {}
        )
        self._tables: dict[str, tuple[int, int]] = {}  # name -> (ptr, flat count)
        self._buffers: dict[str, int] = {
            b["name"]: int(b.get("count", 0))
            for b in sidecar.get("buffers", [])
            if b.get("kind") == "lookup"
        }
        # The arg-pack cache: :meth:`run` used to rebuild the ctypes
        # ``params[]`` array and re-box every argument on every call
        # (measured at ~54% of a warm host call), though a warm loop
        # rebinds the same pointers each time. A generation counter moves
        # only when a bind actually changes a value; ``keep`` is cached
        # alongside ``params`` since the latter holds raw pointers into it.
        self._pack_gen = 0
        self._pack_cache = None
        self._pack_hits = 0
        self._pack_misses = 0

    def _pack_stats(self) -> dict:
        """This library's arg-pack cache hit/miss counters (record-only)."""
        return {"hits": self._pack_hits, "misses": self._pack_misses}

    def bind_vector(self, name: str, ptr: int, n: int) -> HostPluginLibrary:
        """Bind an ``out`` / ``vec_in`` / vector-``mutable`` SoA buffer by name."""
        value = (int(ptr), int(n))
        if self._vec.get(name) != value:
            self._vec[name] = value
            self._pack_gen += 1
        return self

    def bind_matrix(
        self, name: str, ptr: int, n: int, rows: int, cols: int
    ) -> HostPluginLibrary:
        """Bind a ``mat_in`` / matrix-``mutable`` flat ``(R*C, N)`` buffer.
        ``rows``/``cols`` are checked against the sidecar's declared shape;
        ``ptr`` addresses ``dim = r*C + c``, sample-fastest."""
        declared = self._mat_shape.get(name)
        if declared is None:
            raise ValueError(
                f"host plugin: {name!r} is not a declared matrix input / Mutable "
                f"(known matrix args: {sorted(self._mat_shape)})"
            )
        if (int(rows), int(cols)) != declared:
            raise ValueError(
                f"host plugin: matrix '{name}' bound as ({rows}, {cols}) but the "
                f"kernel was compiled for {declared}; rebuild or fix the buffer shape"
            )
        value = (int(ptr), int(n))
        if self._mat.get(name) != value:
            self._mat[name] = value
            self._pack_gen += 1
        return self

    def bind_handle(self, name: str, ptr: int) -> HostPluginLibrary:
        """Bind a ``per_sample`` / ``terminated`` / scalar-``mutable`` /
        wide flat buffer by name (all ride the same scalar-handle pointer).
        A ``lookup`` table uses :meth:`consolidate` instead."""
        value = int(ptr)
        if self._handle.get(name) != value:
            self._handle[name] = value
            self._pack_gen += 1
        return self

    def consolidate(self, name: str, ptr: int, count: int) -> HostPluginLibrary:
        """Consolidate a read-only ``lookup`` table -- the host twin of the
        device registry's ``consolidate`` (nothing to upload; just records
        pointer + count). ``count`` is checked against the sidecar's
        declared size; a declared table must be consolidated before
        :meth:`run`."""
        declared = self._buffers.get(name)
        if declared is not None and int(count) != declared:
            raise ValueError(
                f"host plugin: lookup table '{name}' consolidated with {int(count)} "
                f"elements but the kernel was compiled for {declared}; fix the buffer"
            )
        value = (int(ptr), int(count))
        if self._tables.get(name) != value:
            self._tables[name] = value
            self._pack_gen += 1
        return self

    def _check_uniform_kind(self, name: str, incoming: str) -> None:
        """Refuse a uniform binding that would give ONE name two by-value
        types -- the peer of ``check_uniform_kind_free``, plus the check
        the C++ side cannot make: the caller's binder against the
        artifact's own declared dtype."""
        bound_int = self._uniform_int or {}
        held = "int64" if name in bound_int else (
            "float64" if name in self._uniform else None
        )
        want = "int64" if incoming == "int" else "float64"
        if held is not None and held != want:
            raise ValueError(
                f"host plugin: uniform '{name}' is already bound as {held} and "
                f"cannot also be bound as {want}; a uniform occupies ONE by-value "
                f"kernel parameter slot, declared as exactly one of "
                f"'GRID_CONSTANT() Real p_{name}' (float64) or "
                f"'GRID_CONSTANT() Int p_{name}' (int64). Bind it once, through "
                "the binder that matches the artifact's declaration "
                "(bind_uniform for float64, bind_uniform_int for int64)."
            )
        declared = (self._param_dtype or {}).get(name)
        if declared is not None and declared != incoming:
            want_binder = "bind_uniform_int" if declared == "int" else "bind_uniform"
            raise ValueError(
                f"host plugin: uniform '{name}' is DECLARED {declared!r} by the "
                f"artifact's sidecar params block, but was bound through the "
                f"{'int' if incoming == 'int' else 'float'} binder; use "
                f"{want_binder}(...) — binding it the other way would pack the "
                "wrong 8 bytes into the kernel's parameter slot"
            )

    def bind_uniform(self, name: str, value: float) -> HostPluginLibrary:
        """Bind a ``uniform`` scalar by name (the float64 spelling). The
        change test is bit-exact where ``==`` is not: ``-0.0 == 0.0`` but
        the two are not interchangeable in a kernel, so a rebind between
        them must invalidate the cache; ``NaN != NaN`` already does."""
        self._check_uniform_kind(name, "float")
        value = float(value)
        prev = self._uniform.get(name)
        changed = (
            name not in self._uniform
            or prev != value
            or (value == 0.0 and math.copysign(1.0, prev) != math.copysign(1.0, value))
        )
        if changed:
            self._uniform[name] = value
            self._pack_gen += 1
        return self

    def bind_uniform_int(self, name: str, value: int) -> HostPluginLibrary:
        """Bind a ``uniform`` scalar by name as an exact 64-bit signed
        integer (``ctypes.c_longlong``, never ``c_double``: that widening
        is bit-exact only below 2^53). A non-integral value is refused, not
        truncated (``2.0`` is accepted as an integer spelling)."""
        self._check_uniform_kind(name, "int")
        if isinstance(value, bool):
            raise TypeError(
                f"host plugin: int uniform '{name}' was passed a bool ({value!r}); "
                "pass an integer"
            )
        if isinstance(value, float):
            if value != int(value):
                raise TypeError(
                    f"host plugin: int uniform '{name}' was passed a non-integral "
                    f"value ({value!r}); an int uniform binds an exact 8-byte "
                    "signed slot and is never rounded"
                )
        value = int(value)
        if not (-(2 ** 63) <= value < 2 ** 63):
            raise ValueError(
                f"host plugin: int uniform '{name}' value {value} does not fit a "
                "64-bit signed by-value slot"
            )
        if self._uniform_int is None:
            self._uniform_int = {}
        if name not in self._uniform_int or self._uniform_int[name] != value:
            self._uniform_int[name] = value
            self._pack_gen += 1
        return self

    def run(self, n: int) -> int:
        """Pack the args in ``arg_spec`` order and call the host entry over
        @p n samples, the same role -> params[] packing the C++ registry
        does. Returns the sample count run.

        The pack is CACHED: a hit replays the exact array a prior miss
        built from the same bindings, skipping the validation the miss path
        performs (it cannot newly fail, since nothing changed since it
        passed)."""
        cached = self._pack_cache
        if cached is not None and cached[0] == (self._pack_gen, n):
            self._pack_hits += 1
            self._fn(cached[1], ctypes.c_int32(n))
            return n
        self._pack_misses += 1
        keep: list = []  # keep the POD boxes alive through the call
        params = (ctypes.c_void_p * len(self._arg_spec))()
        for i, (role, name) in enumerate(self._arg_spec):
            tag = classify_arg(
                role, name, vec_mutables=self._vec_mut, mat_mutables=self._mat_mut
            )
            if tag == "GREF_MAT":
                # a matrix binds through the SAME GRefMirror as a vector.
                ptr, nn = self._mat[name]
                box = GRefMirror(ptr, nn, nn, 1, DEVICE_CPU, 0)
            elif tag == "GREF_VEC":
                ptr, nn = self._vec[name]
                box = GRefMirror(ptr, nn, nn, 1, DEVICE_CPU, 0)
            elif tag == "HANDLE":
                if role == "lookup":
                    # a consolidated read-only table, packed with the RUN's
                    # sample count (not the table's own element count).
                    if name not in self._tables:
                        raise ValueError(
                            f"host plugin: lookup table '{name}' was not "
                            f"supplied at consolidation (call "
                            f".consolidate('{name}', ptr, count))"
                        )
                    box = ScalarHandle(self._tables[name][0], n, 1, DEVICE_CPU, 0)
                else:  # per_sample, terminated, scalar/int mutable
                    box = ScalarHandle(self._handle[name], n, 1, DEVICE_CPU, 0)
            elif tag in ("WIDE_IN", "WIDE_OUT"):
                box = ScalarHandle(self._handle[name], n, 1, DEVICE_CPU, 0)
            elif tag == "NSAMPLES":
                box = ctypes.c_uint32(n)
            elif tag == "UNIFORM":
                # follows the binder the caller used (reconciled already by
                # _check_uniform_kind); c_longlong, not c_double.
                if name in (self._uniform_int or {}):
                    box = ctypes.c_longlong(self._uniform_int[name])
                else:
                    box = ctypes.c_double(self._uniform[name])
            else:
                # Defensive only: classify_arg's return is exactly ARG_TAGS.
                raise ValueError(
                    f"host plugin: unhandled ABI tag {tag!r} for role {role!r}"
                )
            keep.append(box)
            params[i] = ctypes.cast(ctypes.pointer(box), ctypes.c_void_p)
        # params holds raw pointers into keep's boxes, so keep must outlive it.
        self._pack_cache = ((self._pack_gen, n), params, keep)
        self._fn(params, ctypes.c_int32(n))
        return n
