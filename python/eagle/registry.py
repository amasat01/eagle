# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launchable **by name**, then call / stream / graph it.

A :class:`KernelRegistry` is a plain ``name -> launchable`` map, where a
*launchable* is anything EAGLE's launch protocol exposes: an in-process
compiled kernel or a deployed PTX one (:class:`~eagle.loaded.LoadedVector` /
``LoadedPure``). Every launchable shares the framework-polymorphic
``__call__`` (numpy blocks; cupy/torch defer to their current stream) and the
capturable ``launch`` primitive::

    reg = KernelRegistry([gravity, drag])
    acc = reg["gravity"](position=t, mu=MU)            # idiomatic
    reg["gravity"].launch(out=buf, position=p, mu=MU)  # into a captured graph

Or load a whole deployed manifest, name-keyed::

    reg = load_manifest("build/forces/manifest.json")
    acc = reg["gravity"](position=t, mu=MU)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Callable

#: The bounded per-process memo of loaded plugins, keyed by the CONTENT of the
#: artifact + sidecar bytes (never a path or mtime, since republishing
#: identical bytes elsewhere must not pay a fresh driver JIT). Every
#: :func:`load_manifest` call builds its own :class:`KernelRegistry` and
#: re-runs the schema/ABI checks; nothing else is memoised.
_PLUGIN_MEMO: dict = {}
#: Cleared wholesale when full rather than evicted one at a time -- see
#: :func:`_clear_plugin_memo`.
_PLUGIN_MEMO_LIMIT = 128

#: The registered-loader hook: ``pattern -> loader(manifest, path, directory)``.
#: The first-priority discovery leg, ahead of the lazy-default map and the
#: entry-point scan -- the one that fires in a sys.path/PYTHONPATH dev env,
#: where Python entry-points never do.
_PATTERN_LOADERS: dict[str, Callable] = {}

#: The lazy-default map. Empty today (``neural_block``'s provider moved out of
#: tree); kept so an in-tree-but-optional pattern can populate it again
#: without changing the resolution order (registered -> lazy-default ->
#: entry-points -> fail-loud).
_LAZY_PROVIDERS: dict = {}

#: The entry-point group, scanned exactly once (see ``_entry_points_scanned``).
_ENTRY_POINT_GROUP = "raptor.pattern_loaders"
_entry_points_scanned = False


def _clear_plugin_memo() -> None:
    """Drop every memoised plugin (:data:`_PLUGIN_MEMO`), for a caller that
    wants a genuinely fresh driver load."""
    _PLUGIN_MEMO.clear()


def _plugin_key(loader: Callable, entry: dict, artifact: pathlib.Path,
                sidecar: pathlib.Path) -> str:
    """The content identity of one loadable plugin (see :data:`_PLUGIN_MEMO`).

    Raises ``OSError`` if either file cannot be read; the caller falls back
    to loading it the long way."""
    h = hashlib.sha256()
    h.update(b"eagle.registry.plugin/1\n")
    h.update(getattr(loader, "__name__", repr(loader)).encode())
    h.update(b"\0")
    h.update(json.dumps(entry, sort_keys=True, default=str).encode())
    h.update(b"\0")
    h.update(artifact.read_bytes())
    h.update(b"\0")
    h.update(sidecar.read_bytes())
    return h.hexdigest()


def _load_plugin(loader: Callable, artifact: pathlib.Path, entry: dict):
    """Load one manifest entry, or return the plugin this process already
    loaded.

    The sidecar read here is ``<artifact>.json`` -- the path
    :class:`~eagle.loaded.LoadedKernel` itself reads -- so the digest covers
    the bytes the loader will actually parse."""
    sidecar = artifact.with_suffix(".json")
    try:
        key = _plugin_key(loader, entry, artifact, sidecar)
    except OSError:
        return loader(artifact)
    hit = _PLUGIN_MEMO.get(key)
    if hit is not None:
        return hit
    loaded = loader(artifact)
    if len(_PLUGIN_MEMO) >= _PLUGIN_MEMO_LIMIT:
        _PLUGIN_MEMO.clear()
    _PLUGIN_MEMO[key] = loaded
    return loaded


class UnknownPatternError(KeyError):
    """No loader is registered/discoverable for a manifest ``pattern``.

    Raised instead of silently skipping it."""


def register_pattern_loader(pattern: str, loader: Callable) -> None:
    """Register ``loader`` as the manifest loader for ``pattern``.

    ``loader(manifest, path, directory)`` returns whatever
    :func:`load_manifest` should hand back for a manifest of this pattern. A
    provider package calls this at import time -- the explicit discovery leg,
    alongside the entry-point scan (:func:`_discover_entry_points`)."""
    _PATTERN_LOADERS[pattern] = loader


def _discover_entry_points() -> None:
    """Scan :data:`_ENTRY_POINT_GROUP` once, caching every discovered loader
    into :data:`_PATTERN_LOADERS` (a name already registered is never
    overwritten)."""
    global _entry_points_scanned
    if _entry_points_scanned:
        return
    _entry_points_scanned = True

    import importlib.metadata as metadata

    try:
        eps = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # Python 3.9: entry_points() takes no group= kwarg
        eps = metadata.entry_points().get(_ENTRY_POINT_GROUP, ())
    for ep in eps:
        if ep.name not in _PATTERN_LOADERS:
            _PATTERN_LOADERS[ep.name] = ep.load()


def _resolve_pattern_loader(pattern: str) -> Callable:
    """Resolve ``pattern`` to a loader: registered -> the lazy-default map
    (import + retry) -> entry-points -> fail-loud."""
    if pattern in _PATTERN_LOADERS:
        return _PATTERN_LOADERS[pattern]

    module_name = _LAZY_PROVIDERS.get(pattern)
    if module_name is not None:
        import importlib

        importlib.import_module(module_name)
        if pattern in _PATTERN_LOADERS:
            return _PATTERN_LOADERS[pattern]

    _discover_entry_points()
    if pattern in _PATTERN_LOADERS:
        return _PATTERN_LOADERS[pattern]

    raise UnknownPatternError(
        f"no loader registered for pattern {pattern!r}; import or install the "
        "package that registers it"
    )


class KernelRegistry:
    """A ``name -> launchable`` map for resolving a launchable by name.

    Values are any launchable sharing the ``__call__`` + ``launch`` protocol.
    Dict-like: ``reg[name]``, ``name in reg``, ``len(reg)``, iteration over
    names, ``.get`` / ``.names``."""

    def __init__(self, kernels=None):
        self._by_name: dict = {}
        for k in kernels or []:
            self.register(k)

    def register(self, kernel, name: str | None = None) -> str:
        """Register ``kernel`` under ``name`` (default ``__name__``); return
        the key.

        Raises on a duplicate name -- pass an explicit ``name=`` to
        disambiguate."""
        base = name or getattr(kernel, "__name__", None)
        if not base:
            raise ValueError(
                "kernel has no usable __name__; pass an explicit name= to register it"
            )
        if base in self._by_name:
            raise ValueError(f"a kernel named {base!r} is already registered")
        self._by_name[base] = kernel
        return base

    def get(self, name: str, default=None):
        """Return the launchable registered under ``name``, or ``default``."""
        return self._by_name.get(name, default)

    def names(self) -> list:
        """The registered names, in insertion order."""
        return list(self._by_name)

    def __getitem__(self, name: str):
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"no kernel named {name!r}; registered: {sorted(self._by_name)}"
            ) from None

    def __contains__(self, name) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __repr__(self) -> str:
        return f"KernelRegistry({sorted(self._by_name)!r})"


def load_manifest(path):
    """Load a deployed ``manifest.json`` by its top-level ``pattern``.

    The Python analogue of the C++ ``PluginRegistry::from_manifest`` -- but
    where that injects the whole ordered set, this returns a **by-name**
    result:

    * ``"vector"`` / ``"pure"`` -- each enabled entry's artifact is
      path-loaded into ``LoadedVector`` / ``LoadedPure`` and keyed by its
      manifest ``id`` into a :class:`KernelRegistry`.
    * ``"neural_block"`` -- a neural deployment unit: the inline ``blocks[]``
      compose one ordered layer whose exec references resolve against the
      ``plugins[]`` kernels, returned as a frozen neural-layer wrapper.
      Recognized but not launch-certified, so it never enters the by-name map.

    An unknown or absent ``pattern`` is a hard error. Returns
    ``KernelRegistry | LoadedNeuralLayer``."""
    from .abi import check_aether_abi
    from .loaded import (
        LoadedPure,
        LoadedVector,
    )
    from .roles import RECOGNIZED_PATTERNS, check_execution_axis, check_schema_version

    path = pathlib.Path(path)
    manifest = json.loads(path.read_text())
    # Manifest-level guards, symmetric with the C++ ``from_manifest``, run
    # before loading any artifact; each entry's sidecar is re-checked at
    # ``Loaded*`` load. ``allow_legacy_version_key`` mirrors the legacy
    # producer, which stamps both ``version`` and ``schema_version``.
    schema_version = check_schema_version(
        manifest, name=path.name, allow_legacy_version_key=True
    )
    check_execution_axis(manifest, schema_version, name=path.name)
    check_aether_abi(manifest, kind="plugin manifest", name=path.name)
    directory = path.parent
    pattern = manifest.get("pattern")

    loaders = {"vector": LoadedVector, "pure": LoadedPure}
    if pattern not in loaders:
        # A recognized-but-not-launch-certified pattern (only "neural_block"
        # today) resolves through the registered-loader hook rather than a
        # hardcoded import, so eagle core carries zero neural imports.
        if pattern in RECOGNIZED_PATTERNS:
            return _resolve_pattern_loader(pattern)(manifest, path, directory)
        raise ValueError(
            f"{path.name}: manifest pattern {pattern!r} is not a supported plugin "
            f"family (expected one of {sorted(loaders)})"
        )
    loader = loaders[pattern]

    # An orphaned blocks[] under a kernel pattern would be silently ignored by
    # a kernel-manifest loader, so it must fail loudly instead.
    if manifest.get("blocks"):
        raise ValueError(
            f"{path.name}: a manifest carrying blocks[] must declare pattern "
            f"'neural_block'; got pattern {pattern!r} (block descriptors would be "
            "silently ignored by a kernel-manifest loader)"
        )

    _preload_plugins_pass(manifest, path)

    reg = KernelRegistry()
    for entry in manifest.get("plugins", []):
        if not entry.get("enabled", True):
            continue
        reg.register(_load_plugin(loader, directory / entry["artifact"], entry),
                     name=entry["id"])
    return reg


def _preload_plugins_pass(manifest, path) -> None:
    """The ``plugins[]`` pre-load pass, shared by the kernel and neural
    branches: rejects a duplicate plugin id and an unrecognized artifact
    ``format``, before any entry is loaded (so neither needs a GPU to catch)."""
    from .roles import MANIFEST_FORMATS

    seen_ids = set()
    for entry in manifest.get("plugins", []):
        pid = entry["id"]
        if pid in seen_ids:
            raise ValueError(
                f"{path.name}: duplicate plugin id {pid!r} in manifest; every "
                "entry in plugins[] must have a unique id"
            )
        seen_ids.add(pid)
        fmt = entry.get("format")
        if fmt not in MANIFEST_FORMATS:
            raise ValueError(
                f"{path.name}: manifest plugin {pid!r}: unknown artifact format "
                f"{fmt!r} (supported: {sorted(MANIFEST_FORMATS)}); upgrade eagle"
            )


#: The ``neural_block`` loader body lives outside this module, reached only
#: through :func:`_resolve_pattern_loader` -- which keeps this module's own
#: import surface neural-free.
