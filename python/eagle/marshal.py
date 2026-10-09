# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Input-role marshalling for every kernel launcher.

The single home for coercing a launch call's inputs into the contiguous
device arrays / by-value scalars the aether GRef/HandleT ABI expects: one
implementation per input role (state vector, matrix, per-sample scalar,
lookup table, broadcast uniform, writable ``Mutable``, the ``terminated``
mask, wide per-sample buffer/gradient), shared verbatim by every launch
path so the input contract can never drift between kinds.

Each foreign tensor is imported to cupy through :mod:`eagle.interop`
(zero-copy DLPack for a CUDA-resident tensor, a host upload otherwise); the
cast/contiguate after that is a copy only when not already the target dtype
/ C-contiguous. The launchable's producer resolves ``scalar_type`` to the
Real dtype ``dt`` threaded through here (:func:`eagle.dtypes.np_dtype`).
"""

from __future__ import annotations

import threading

import numpy as np

from . import _layout
from .roles import param_decls


def _reconcile_n(n, count):
    """Fold one per-sample source's length into the batch size N (first
    source wins, every later one must agree)."""
    if n is None:
        return int(count)
    if int(count) != n:
        raise ValueError("inconsistent batch size N across per-sample inputs")
    return n


def require_n(n, *, hint="state-vector input"):
    """The canonical 'cannot size the batch' error, with a kind-appropriate hint."""
    if n is None:
        raise TypeError(f"cannot infer the sample count N; pass at least one {hint}")
    return n


# The conforming fast path: for the common case -- a cupy array already
# the target dtype, C-contiguous, and on the current device -- the
# coercion triple ``cp.ascontiguousarray(cp.asarray(interop.to_cupy(value),
# dtype=dt))`` is three no-ops, so returning the input object directly is
# bitwise identical. Anything else takes the general path unchanged.
# ``_fast_path_ok`` takes the cupy module as an argument so it is
# unit-testable with no device.

#: fast/slow counters over :func:`_as_dtype` (record-only; the gate reads them).
_coerce_stats = {"fast": 0, "slow": 0}

#: Forces :func:`_fast_path_ok` to always answer ``False`` (the
#: launch-parity corpus's forced-slow arm); never set in production.
_force_slow = False


def _coercion_stats() -> dict:
    """A copy of the fast/slow coercion counters (record-only)."""
    return dict(_coerce_stats)


def _reset_coercion_stats() -> None:
    """Zero the fast/slow coercion counters."""
    _coerce_stats["fast"] = 0
    _coerce_stats["slow"] = 0


def _force_slow_coercion(enabled: bool = True) -> None:
    """Force every :func:`_as_dtype` call down the general coercion path
    (the launch-parity corpus runs both arms and asserts bitwise-identical
    results). Never set in production."""
    global _force_slow
    _force_slow = bool(enabled)


def _fast_path_ok(value, dt, cp) -> bool:
    """True only for a cupy array already the exact target dtype,
    C-contiguous, and on the current device (a cross-device array takes
    the general path)."""
    if _force_slow:
        return False
    if not isinstance(value, cp.ndarray):
        return False
    if value.dtype != dt:
        return False
    if not value.flags.c_contiguous:
        return False
    try:
        return int(value.device.id) == int(cp.cuda.runtime.getDevice())
    except Exception:
        return False  # cannot report its device -> not safe to bind blind


def _as_dtype(value, dt=np.float64):
    """Import ``value`` to a contiguous ``dt`` cupy array (zero-copy DLPack
    where possible, else a cast/copy). ``dt`` is the resolved Real dtype
    (float64 default; float32 in float32 mode -- SoftDouble stays float64).
    An already-conforming cupy array is returned as-is (see the FAST PATH
    note above)."""
    import cupy as cp

    if _fast_path_ok(value, dt, cp):
        _coerce_stats["fast"] += 1
        return value

    _coerce_stats["slow"] += 1

    from . import interop

    return cp.ascontiguousarray(cp.asarray(interop.to_cupy(value), dtype=dt))


def coerce_vec_inputs(kw, names, n=None, widths=None, dt=np.float64):
    """State vectors -> contiguous ``(W, N)`` ``dt`` device arrays; returns
    ``(dict, n)`` with N reconciled. ``widths`` maps a name to its
    component count (default 3; a synthesized derivative seed may differ)."""
    out = {}
    for name in names:
        if name not in kw:
            raise TypeError(f"missing required input '{name}'")
        w = 3 if widths is None else widths[name]
        arr = _as_dtype(_adapted_input(name, kw[name], (w,)), dt)
        if arr.ndim != 2 or arr.shape[0] != w:
            raise ValueError(
                f"input '{name}' must have shape ({w}, N); got {tuple(arr.shape)}"
            )
        n = _reconcile_n(n, arr.shape[1])
        out[name] = arr
    return out, n


def coerce_wide_inputs(kw, names, n=None, dt=np.float64):
    """Wide per-sample flat inputs -> contiguous 2-D ``(rows, N)`` ``dt``
    device arrays; returns ``(dict, n)``. Unlike a fixed-width state
    vector, a wide buffer's true row count (``stride * n_in``) is a
    runtime quantity the sidecar never carries, so only the 2-D shape is
    checked -- the caller must match its own ``n_in``."""
    out = {}
    for name in names:
        if name not in kw:
            raise TypeError(f"missing required wide input '{name}'")
        arr = _as_dtype(kw[name], dt)
        if arr.ndim != 2:
            raise ValueError(
                f"wide input '{name}' must be a 2-D (rows, N) array; "
                f"got shape {tuple(arr.shape)}"
            )
        n = _reconcile_n(n, arr.shape[1])
        out[name] = arr
    return out, n


def coerce_wide_outputs(kw, names, n=None, dt=np.float64, exempt=()):
    """Wide gradient-scatter outputs (a VJP-derived kernel's disjoint
    scatter targets) -> contiguous 2-D ``(rows, N)`` ``dt`` device arrays,
    written in place; returns ``(dict, n)``. Never auto-allocated (the row
    count is a runtime quantity, as in :func:`coerce_wide_inputs`) -- the
    caller must pass a buffer. ``exempt`` names a declared row-indexed
    destination (e.g. an atomic ``Accum`` output's ``(rows, 1)`` buffer)
    that carries no batch axis, so it binds as-is and skips
    ``_reconcile_n``."""
    out = {}
    for name in names:
        if name not in kw:
            raise TypeError(
                f"missing required wide gradient output '{name}' — its true "
                "row count (stride * n_in) is a runtime quantity that cannot "
                "be inferred, so it is never auto-allocated; pass a "
                "pre-allocated (rows, N) array"
            )
        arr = _as_dtype(kw[name], dt)
        if arr.ndim != 2:
            raise ValueError(
                f"wide gradient output '{name}' must be a 2-D (rows, N) "
                f"array; got shape {tuple(arr.shape)}"
            )
        if name in exempt:
            out[name] = arr
            continue
        n = _reconcile_n(n, arr.shape[1])
        out[name] = arr
    return out, n


def coerce_mat_inputs(kw, shapes, n=None, dt=np.float64):
    """Per-sample matrix inputs -> contiguous flat ``(R*C, N)`` ``dt``
    device arrays; returns ``(dict, n)``. ``shapes`` maps each name to its
    declared ``(R, C)``; the caller passes either ``(R, C, N)`` or flat
    ``(R*C, N)``, both reshaping to the same row-major SoA buffer."""
    out = {}
    for name, (r, c) in shapes.items():
        if name in kw:
            kw = {**kw, name: _adapted_input(name, kw[name], (r, c), (r * c,))}
        if name not in kw:
            raise TypeError(f"missing required matrix input '{name}'")
        arr = _as_dtype(kw[name], dt)
        if arr.ndim == 3 and arr.shape[:2] == (r, c):
            arr = arr.reshape(r * c, arr.shape[2])
        elif not (arr.ndim == 2 and arr.shape[0] == r * c):
            raise ValueError(
                f"matrix input '{name}' must have shape ({r}, {c}, N) or "
                f"({r * c}, N); got {tuple(arr.shape)}"
            )
        n = _reconcile_n(n, arr.shape[1])
        out[name] = arr
    return out, n


def coerce_per_sample(kw, names, n=None, dt=np.float64):
    """Per-sample scalars -> contiguous ``(N,)`` ``dt`` device arrays
    (strict: a 2-D input is rejected, never silently flattened); returns
    ``(dict, n)``."""
    out = {}
    for name in names:
        if name not in kw:
            raise TypeError(f"missing required per-sample scalar '{name}'")
        a = _as_dtype(kw[name], dt)
        if a.ndim != 1:
            raise ValueError(
                f"per-sample scalar '{name}' must have shape (N,); "
                f"got {tuple(a.shape)}"
            )
        n = _reconcile_n(n, a.size)
        out[name] = a
    return out, n


def coerce_tables(kw, lookup_counts, dt=np.float64):
    """Lookup tables / shared constants -> flat ``(count,)`` ``dt``
    handles. The row-major flatten is the binding contract (the kernel
    reads ``handle[flat_index]``); the count is checked against the
    declaration."""
    out = {}
    for name, expect in lookup_counts.items():
        if name not in kw:
            raise TypeError(f"missing required lookup table '{name}'")
        t = _as_dtype(kw[name], dt).ravel()
        if t.size != int(expect):
            raise ValueError(
                f"lookup table '{name}' must have {int(expect)} elements; got {t.size}"
            )
        out[name] = t
    return out


def coerce_int_uniform(name, value):
    """One ``int`` broadcast param -> an exact ``np.int64``, never through
    ``dt(value)``: a float/double route is bit-exact only below 2^53. Not
    an exact integer (``2.5``) is refused rather than truncated; ``2.0`` is
    accepted, matching :func:`fill_mutable`."""
    if isinstance(value, bool):
        raise TypeError(
            f"int parameter '{name}' was passed a bool ({value!r}); pass an integer"
        )
    if isinstance(value, (float, np.floating)):
        if float(value) != int(value):
            raise TypeError(
                f"int parameter '{name}' was passed a non-integral value "
                f"({value!r}); an int uniform binds an exact 8-byte signed slot "
                "(GRID_CONSTANT() Int p_" + str(name) + ") and is never rounded"
            )
        value = int(value)
    out = np.int64(value)
    # Round-trip guard: np.int64() of a Python int larger than 2^63-1 raises,
    # but a numpy float128/Decimal-ish input could still land off by one.
    if int(out) != int(value):
        raise TypeError(
            f"int parameter '{name}' does not survive the int64 slot exactly "
            f"({value!r} -> {int(out)}); it is out of range for a 64-bit signed "
            "uniform"
        )
    return out


def coerce_uniforms(kw, params, dt=np.float64):
    """Broadcast ``Param`` constants -> their declared by-value type.
    ``params`` may be bare names (v1, all float) or decl-carrying entries
    (:func:`eagle.roles.param_decls` normalizes both); an ``int`` param
    binds ``np.int64`` exactly (:func:`coerce_int_uniform`), never ``dt``."""
    out = {}
    for name, dtype in param_decls(params):
        if name not in kw:
            raise TypeError(f"missing required parameter '{name}'")
        if dtype == "int":
            out[name] = coerce_int_uniform(name, kw[name])
        elif dtype == "float":
            out[name] = dt(kw[name])
        else:  # defensive: param_decls does not validate the vocabulary
            raise ValueError(
                f"parameter '{name}' declares unknown dtype {dtype!r} "
                "(expected 'float' or 'int')"
            )
    return out


def _is_array_value(value) -> bool:
    """Whether a passed Mutable value is a per-sample array (vs a scalar
    fill). Distinct from the codegen's ``_is_per_sample``, which asks
    whether a *kernel* does any per-sample work."""
    return not isinstance(value, (int, float, bool))


def _adapted_input(name, value, *heads):
    """A read-only input in component-major form (:mod:`eagle._layout`):
    as given, a zero-copy view, or a copy with a warning."""
    for head in heads:
        a = _layout.adapt(name, value, head, writes=False)
        if a is not None:
            return a.native
    return value


def _adapt_mutable(name, value, heads, adapted):
    for head in heads:
        a = _layout.adapt(name, value, head, writes=True)
        if a is not None:
            if adapted is not None:
                adapted[name] = a
            return a.native
    return value


def coerce_mutable(name, value, decl, dt=np.float64, adapted=None):
    """Coerce a provided per-sample ``Mutable`` array to its device buffer;
    returns ``(device_array, n)``. A ``float``/``int`` slot is contiguous
    ``(N,)`` (``int`` always ``int64``); a ``vector`` slot a ``(W, N)`` SoA
    array; a ``matrix`` slot bound flat ``(R*C, N)`` (a contiguous ``(R, C,
    N)`` input reshapes zero-copy, so the kernel's update lands in the
    caller's buffer).

    A sample-major ``vector``/``matrix`` slot is adapted first
    (:mod:`eagle._layout`); when ``adapted`` is a dict its record is
    stored there by name, so the caller can write the result back."""
    import cupy as cp

    if decl.dtype == "vector":
        value = _adapt_mutable(name, value, ((decl.width,),), adapted)
        arr = _as_dtype(value, dt)
        if arr.ndim != 2 or arr.shape[0] != decl.width:
            raise ValueError(
                f"Mutable vector '{name}' must be a ({decl.width}, N) SoA array; "
                f"got shape {tuple(arr.shape)}"
            )
        return arr, int(arr.shape[1])
    if decl.dtype == "matrix":
        r, c = decl.shape
        value = _adapt_mutable(name, value, ((r, c), (r * c,)), adapted)
        arr = _as_dtype(value, dt)
        if arr.ndim == 3 and arr.shape[:2] == (r, c):
            arr = arr.reshape(r * c, arr.shape[2])
        elif not (arr.ndim == 2 and arr.shape[0] == r * c):
            raise ValueError(
                f"Mutable matrix '{name}' must be a ({r}, {c}, N) or flat "
                f"({r * c}, N) array; got shape {tuple(arr.shape)}"
            )
        return arr, int(arr.shape[1])
    dtype = cp.int64 if decl.dtype == "int" else dt
    arr = _as_dtype(value, dtype)
    if arr.ndim != 1:
        raise ValueError(
            f"Mutable '{name}' must have shape (N,); got {tuple(arr.shape)}"
        )
    return arr, int(arr.size)


def fill_mutable(name, value, decl, n, dt=np.float64):
    """A broadcast / default ``Mutable`` buffer: a scalar ``value`` filled
    to ``(N,)`` as ``dt``/``int64``. A vector/matrix slot has no scalar
    fill -- it must be passed as an array."""
    import cupy as cp

    if decl.dtype == "vector":
        raise TypeError(
            f"Mutable vector '{name}' must be passed as a ({decl.width}, N) array "
            "(a scalar broadcast / default fill is not supported for a vector slot)"
        )
    if decl.dtype == "matrix":
        r, c = decl.shape
        raise TypeError(
            f"Mutable matrix '{name}' must be passed as a ({r}, {c}, N) array "
            "(a scalar broadcast / default fill is not supported for a matrix slot)"
        )
    if decl.dtype == "int":
        return cp.full(n, int(value), dtype=cp.int64)
    return cp.full(n, float(value), dtype=dt)


# The opt-in all-false ``terminated`` mask cache. An omitted mask gets a
# fresh ``cp.zeros(n, bool)`` per call -- identical every time, so
# cacheable, but ONLY for a kernel known never to write it, which this
# module (marshalling arbitrary plugins) cannot know on its own. So it is
# sidecar-declared opt-in (see :mod:`eagle.sidecar`'s
# ``terminated_readonly``); every undeclared plugin keeps its own private
# buffer. Helps the uncaptured path only -- a captured launch already
# requires a pre-allocated mask.

#: ``(device_id, n)`` -> the shared all-false mask for a DECLARING plugin.
_zero_mask_cache: dict = {}
_ZERO_MASK_COUNTS = {"hits": 0, "misses": 0, "bypassed": 0}

#: How many distinct ``(device, n)`` masks may be retained (each pins ``n``
#: bytes of device memory for the process's life). Past the cap the cache
#: bypasses -- a fresh private mask, same as an undeclared plugin gets.
ZERO_MASK_CACHE_CAP = 64
#: Serialises the cache, its cap and its tallies.
_ZERO_MASK_LOCK = threading.Lock()


def _zero_mask_stats() -> dict:
    """A copy of the opt-in zero-mask cache's counters (record-only).
    ``bypassed`` counts calls served a private mask past the cap."""
    with _ZERO_MASK_LOCK:
        return dict(_ZERO_MASK_COUNTS)


def _reset_zero_mask_cache() -> None:
    """Drop the cached masks and zero their counters (tests / teardown)."""
    with _ZERO_MASK_LOCK:
        _zero_mask_cache.clear()
        _ZERO_MASK_COUNTS["hits"] = 0
        _ZERO_MASK_COUNTS["misses"] = 0
        _ZERO_MASK_COUNTS["bypassed"] = 0


def _cached_zero_mask(n, cp):
    """The shared all-false ``(n,)`` bool mask for the current device,
    keyed on ``(device, n)``. Only reached for a plugin that declared the
    mask read-only."""
    key = (int(cp.cuda.runtime.getDevice()), int(n))
    with _ZERO_MASK_LOCK:
        mask = _zero_mask_cache.get(key)
        if mask is not None:
            _ZERO_MASK_COUNTS["hits"] += 1
            return mask
        if len(_zero_mask_cache) >= ZERO_MASK_CACHE_CAP:
            _ZERO_MASK_COUNTS["bypassed"] += 1
        else:
            _ZERO_MASK_COUNTS["misses"] += 1
            mask = cp.zeros(int(n), dtype=cp.bool_)
            _zero_mask_cache[key] = mask
            return mask
    return cp.zeros(int(n), dtype=cp.bool_)


def coerce_terminated(kw, n, *, readonly_mask=False):
    """The ``terminated`` mask -> a ``(N,)`` bool device array (all-false
    when omitted). ``readonly_mask`` (default ``False``) is the
    sidecar-declared opt-in: when ``True`` and the mask is omitted, it is
    served from the per-``(device, n)`` cache. A passed mask is never
    cached."""
    import cupy as cp

    term = kw.get("terminated")
    if term is None:
        if readonly_mask:
            return _cached_zero_mask(n, cp)
        return cp.zeros(n, dtype=cp.bool_)
    term = _as_dtype(term, cp.bool_)
    if term.shape != (n,):
        raise ValueError(f"'terminated' must have shape ({n},); got {term.shape}")
    return term
