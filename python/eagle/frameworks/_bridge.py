# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Returns through a framework bridge: ``Array.to(name)`` and the ``returns=`` of the doors.

``eagle.Array`` exposes eagle's memory and nothing else; a framework's array type is reached only through
``eagle.frameworks.<name>.from_eagle(a)``, looked up here when asked for, so ``import eagle`` imports no
framework. ``from_eagle`` is zero-copy where the framework can alias the memory (DLPack, with the stream handed
over), and an upload or a copy to the host otherwise."""

from __future__ import annotations

import importlib
import threading
import warnings

#: how to install each bridged framework
HINTS = {
    "numpy": "pip install numpy",
    "cupy": "pip install cupy-cuda12x  (or cupy-cuda13x for a CUDA 13 driver)",
    "torch": "pip install torch",
    "jax": "pip install jax  (jax[cuda12] for a GPU)",
    "warp": "pip install warp-lang",
}
_WARNED: set = set()
_LOCK = threading.Lock()


def _converter(name):
    """``eagle.frameworks.<name>.from_eagle``, or ``ImportError`` carrying the install hint."""
    if name not in HINTS:
        raise ValueError(f"eagle: unknown framework {name!r}; expected one of {sorted(HINTS)}")
    try:
        module = importlib.import_module(f"eagle.frameworks.{name}")
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] != name:
            raise
        raise ImportError(f"eagle: results as {name!r} need {name} installed ({HINTS[name]})") from exc
    return module.from_eagle


def stream_handle(stream) -> int:
    """The raw handle of a framework's stream object (cupy, torch, warp), by the bridge of the framework the
    object comes from."""
    name = type(stream).__module__.partition(".")[0]
    if name not in HINTS:
        raise TypeError(f"eagle: {type(stream).__name__} is not a stream object of a bridged framework; pass the raw handle")
    return int(importlib.import_module(f"eagle.frameworks.{name}").stream_handle(stream))


def to(a, name):
    """The :class:`eagle.Array` ``a`` as ``name``'s array type; raises ``ImportError`` (with the install hint)
    when the framework is missing."""
    return _converter(name)(a)


def route(result, returns):
    """``result`` (an array, or a dict / tuple / list of them) with every :class:`eagle.Array` converted to
    ``returns``'s type. ``None`` returns ``result`` as it is. A framework that is not installed warns ONCE and
    leaves the :class:`eagle.Array` results."""
    if returns is None:
        return result
    try:
        convert = _converter(returns)
    except ImportError as exc:
        with _LOCK:
            first = returns not in _WARNED
            _WARNED.add(returns)
        if first:
            warnings.warn(f"{exc}; returning eagle.Array", RuntimeWarning, stacklevel=3)
        return result
    return _convert(result, convert)


def _convert(x, convert):
    from ..arrays import Array

    if isinstance(x, Array):
        return convert(x)
    if isinstance(x, dict):
        return {k: _convert(v, convert) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_convert(v, convert) for v in x)
    return x
