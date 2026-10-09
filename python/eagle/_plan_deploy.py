# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle.deploy: a hawk kernel built into plans that run where its data lives."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from . import _layout
from . import exec as eexec
from ._plan_binding import BoundPlan
from ._plan_planes import residency

if TYPE_CHECKING:
    from .plan import Plan


class AutoPlan:
    """A plugin with both targets, run where its data lives — the result of
    :func:`auto`.

    :attr:`host`/:attr:`device` are the :class:`Plan` for each target, built
    lazily; :meth:`select` picks one from the values a call binds (host- or
    device-resident; the sample count never enters). :meth:`run`/:meth:`bind`
    delegate to the selected plan. A default deploy with a usable GPU returns
    with the device side built and the host side still building."""

    __slots__ = ("plugin", "plan_kw", "_plans", "_host")

    def __init__(self, plugin, plan_kw: dict, host=None):
        self.plugin = plugin
        self.plan_kw = dict(plan_kw)
        self._plans = {}
        self._host = host  # (_HostSide, index) when the host side builds in the background

    def __repr__(self) -> str:
        targets = list(_exec_targets(self.plugin))
        if self._host is not None:
            targets = ["host", *targets]
        return f"AutoPlan(targets={targets})"

    def _side(self, name: str):
        """The plugin carrying ``name``'s entry (waiting for a host side still
        building)."""
        if name == "host" and self._host is not None:
            side, index = self._host
            return side.plugin(index)
        return self.plugin

    def _target(self, name: str) -> Plan:
        p = self._plans.get(name)
        if p is None:
            attr, structure = _AUTO_TARGETS[name]
            plugin = self._side(name)
            if getattr(plugin, attr, None) is None:
                raise ValueError(
                    f"eagle.plan.auto: the data is on the {name}, but this plugin "
                    f"carries no {attr} (its exec_targets: "
                    f"{list(_exec_targets(self.plugin))}); build it with "
                    "targets=(\"host\", \"cuda\") to run it on either side")
            from .plan import plan  # plan.py imports this module last

            p = plan(plugin, structure=structure, **self.plan_kw)
            self._plans[name] = p
        return p

    @property
    def host(self) -> Plan:
        """The :data:`~eagle.exec.HostTeam` plan (built on first use)."""
        return self._target("host")

    @property
    def device(self) -> Plan:
        """The :data:`~eagle.exec.DeviceKernel` plan (built on first use)."""
        return self._target("device")

    def select(self, planes: dict) -> Plan:
        """The plan ``planes`` selects (:func:`residency`): the device when
        any is device-resident, else the host; both sides refused."""
        arg_spec = getattr(self.plugin, "arg_spec", ()) or ()
        bound = {name: planes[name] for role, name in arg_spec
                 if role not in ("uniform", "nsamples") and name in planes}
        return self.device if residency(bound) == "device" else self.host

    @_layout.door
    def run(self, /, **kw):
        """:meth:`Plan.run` on the plan :meth:`select` picks for ``kw``."""
        return self.select(kw).run(**kw)

    @_layout.door
    def bind(self, /, **planes) -> BoundPlan:
        """:meth:`Plan.bind` on the plan :meth:`select` picks for ``planes``."""
        return self.select(planes).bind(**planes)


#: :func:`auto`'s two sides: ``(entry attribute, structure)``.
_AUTO_TARGETS = {"host": ("host_entry", eexec.HostTeam),
                 "device": ("device_function", eexec.DeviceKernel)}


def _exec_targets(plugin) -> tuple:
    """The targets ``plugin`` carries: its declared ``exec_targets``, else
    the entries it holds."""
    declared = getattr(plugin, "exec_targets", None)
    if declared:
        return tuple(declared)
    return tuple(t for t, (attr, _) in (("host", _AUTO_TARGETS["host"]),
                                        ("device", _AUTO_TARGETS["device"]))
                 if getattr(plugin, attr, None) is not None)


def auto(plugin, *, targets=None, cache_dir=None, scalar_type=None,
         **plan_kw) -> AutoPlan | tuple[AutoPlan, ...]:
    """An :class:`AutoPlan` for ``plugin``: run it where its data lives.
    ``eagle.deploy`` is this same function.

    ``plugin`` is a built plugin, a hawk kernel, or a list built into one
    bundle (returned as a tuple of plans in order). The build uses hawk's
    cache under ``cache_dir`` (``None``: hawk's default); a repeat call
    compiles nothing. ``targets`` (``None``: both sides when a GPU is
    usable -- the call returns with the device built, the host still
    building beside it -- else
    ``("host",)``), ``cache_dir`` and ``scalar_type`` are refused
    for an already-built plugin. ``scalar_type`` is the precision hawk builds
    in: ``"float64"`` (``None``, the default) or ``"float32"``; the planes
    passed at run time must match it.

    ``plan_kw`` are :func:`plan`'s keywords but ``structure``
    (``inner``/``gather`` do not apply); a side the plugin lacks is refused
    when first selected."""
    for key in ("structure", "inner", "gather"):
        if key in plan_kw:
            raise ValueError(
                f"eagle.plan.auto: {key}= is not an auto plan's to take: the data "
                "picks host_team or device_kernel per call; for an explicit "
                "structure use eagle.plan.plan(plugin, structure=...)")
    if scalar_type not in (None, *_BUILT_SCALAR_TYPES):
        raise ValueError(
            f"eagle.plan.auto: scalar_type must be one of {_BUILT_SCALAR_TYPES}, "
            f"got {scalar_type!r}")
    if isinstance(plugin, (list, tuple)):
        kernels = list(plugin)
        if not kernels:
            raise ValueError(
                "eagle.plan.auto: the list is empty; pass one hawk kernel or a "
                "list of them, built into one bundle")
        for i, k in enumerate(kernels):
            if not _is_hawk_kernel(k):
                raise ValueError(
                    f"eagle.plan.auto: a list is built into one bundle, so every "
                    f"item must be a hawk kernel, but item {i} is a "
                    f"{type(k).__name__}; pass a built plugin on its own: "
                    "eagle.plan.auto(plugin)")
        return tuple(AutoPlan(p, plan_kw, host)
                     for p, host in _deploy_hawk(kernels, targets, cache_dir, scalar_type))
    if _is_hawk_kernel(plugin):
        p, host = _deploy_hawk([plugin], targets, cache_dir, scalar_type)[0]
        return AutoPlan(p, plan_kw, host)
    for key, value in (("targets", targets), ("cache_dir", cache_dir),
                       ("scalar_type", scalar_type)):
        if value is not None:
            raise ValueError(
                f"eagle.plan.auto: {key}= builds a hawk kernel, but this plugin "
                "is already built: pass the kernel, or drop the keyword")
    return AutoPlan(plugin, plan_kw)


def _deploy_hawk(kernels, targets, cache_dir, scalar_type) -> list:
    """``[(plugin, background host side or None)]`` for ``kernels``, in order.

    With ``targets=None`` and a usable GPU the call returns once the device
    side is built, the host side building on beside it (:class:`_HostSide`):
    a GPU run never waits for the host compiler. Any build including ``cuda``
    compiles the loop's own small device kernels on a second thread
    meanwhile (:func:`eagle._until_done._warm_device_kernels`)."""
    _hawk_api()  # the refusal naming hawk comes first
    host = None
    if targets is None:
        targets = _eager_targets()
        if targets == ("cuda",):
            host = _HostSide(kernels, cache_dir, scalar_type)
    warm = _warm_in_background(tuple(targets))
    try:
        built = _hawk_plugins(kernels, targets, cache_dir, scalar_type)
    finally:
        if warm is not None:
            warm.join()
    return [(p, None if host is None else (host, i)) for i, p in enumerate(built)]


class _HostSide:
    """The host side of a default deploy, one bundle for all its kernels,
    built on its own thread from the deploy on: the deploy returns once the
    device side is built, and the first host plan waits for this build (and
    raises its error, if any). The thread is not a daemon, so the
    interpreter finishes the build before it exits. A forked child, which
    inherits no running thread, builds the host side itself."""

    __slots__ = ("_args", "_plugins", "_error", "_thread", "_pid")

    def __init__(self, kernels, cache_dir, scalar_type):
        self._args = (list(kernels), ("host",), cache_dir, scalar_type)
        self._plugins = self._error = None
        self._pid = os.getpid()
        self._thread = threading.Thread(target=self._build, name="eagle-host-build")
        self._thread.start()

    def _build(self):
        try:
            self._plugins = _hawk_plugins(*self._args)
        except BaseException as exc:  # handed to the first host plan
            self._error = exc

    def plugin(self, index: int):
        if os.getpid() != self._pid:  # forked while the parent's build ran
            self._pid = os.getpid()
            if self._plugins is None and self._error is None:
                self._build()
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._plugins[index]


#: Devices whose loop kernels this process already compiled.
_WARMED: set = set()


def _warm_in_background(targets):
    """Start :func:`eagle._until_done._warm_device_kernels` on its own thread
    for a build including ``cuda`` (once per device per process); the
    thread, or ``None``. A failure there only loses the head start."""
    if "cuda" not in targets:
        return None
    try:
        import cupy

        device = cupy.cuda.Device().id
    except Exception:
        return None
    if device in _WARMED:
        return None
    from ._until_done import _warm_device_kernels

    def warm():
        try:
            with cupy.cuda.Device(device):
                _warm_device_kernels()
            _WARMED.add(device)
        except Exception:
            pass

    thread = threading.Thread(target=warm, name="eagle-warm", daemon=True)
    thread.start()
    return thread


def _is_hawk_kernel(obj) -> bool:
    """Is ``obj`` a traced hawk kernel, read off its type (never by
    importing hawk)? A plugin or any other object answers ``False``."""
    return (type(obj).__module__.split(".", 1)[0] == "hawk"
            and hasattr(obj, "walk") and hasattr(obj, "sinks"))


def _wants_fast_entries(kernels, targets) -> bool:
    """Whether this bundle call should ask hawk for the fast-path entries
    (``<name>_range``, and an active-set kernel's ``<name>_persist``, behind
    ``HAWK_FAST_ENTRIES=1``) -- ANY kernel here automatic (``.one_step``
    set, :mod:`hawk.trace.steps`'s own ``auto`` marker), on a build that
    includes ``cuda`` (the entries are device-only). One flag for the
    whole bundle call, matching :func:`eagle._until_done._build_auto`'s
    own ONE-place eligibility: this is the single spot that decides eagle
    builds these kernels with the define by DEFAULT, so the ordinary
    deploy/simulate path gets the fast entries without the caller asking.
    A non-automatic kernel is unaffected either way -- the define gates
    code its own emission never reaches."""
    if "cuda" not in targets:
        return False
    return any(getattr(k, "one_step", None) is not None for k in kernels)


#: The precisions hawk builds (the third sidecar type, ``softdouble``, has
#: no hawk build yet).
_BUILT_SCALAR_TYPES = ("float64", "float32")


def _hawk_api():
    """The hawk entry points a build uses, or the refusal naming hawk."""
    try:
        from hawk.artifact import build_bundle, plugins
        from hawk.compile import default_cache_dir
        from hawk.emit import FAST_ENTRIES_MACRO
    except ImportError as exc:
        raise ImportError(
            "eagle.plan.auto: building a hawk kernel needs hawk, which does not "
            f"import here ({exc}); install hawk, or pass a built plugin") from exc
    return build_bundle, plugins, default_cache_dir, FAST_ENTRIES_MACRO


#: One lock per bundle folder this process builds (:func:`_hawk_plugins`).
_BUNDLE_LOCKS: dict = {}


def _after_fork_in_child():
    # a lock some other thread of the parent held would never be released here
    _BUNDLE_LOCKS.clear()


os.register_at_fork(after_in_child=_after_fork_in_child)


def _hawk_plugins(kernels, targets, cache_dir, scalar_type=None) -> tuple:
    """Build ``kernels`` into one bundle and return their plugins in order."""
    build_bundle, plugins, default_cache_dir, FAST_ENTRIES_MACRO = _hawk_api()
    from .registry import load_manifest

    targets = _default_targets() if targets is None else tuple(targets)
    defines = (f"{FAST_ENTRIES_MACRO}=1",) if _wants_fast_entries(kernels, targets) else ()
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    mode = scalar_type or "float64"
    where = root / "bundles" / _bundle_key(kernels, targets, defines, mode)
    # hawk writes a bundle folder in place: one build of it at a time here (a
    # background host build and a repeat deploy of the same kernels meet)
    with _BUNDLE_LOCKS.setdefault(str(where.resolve()), threading.Lock()):
        bundle = build_bundle(kernels, where, targets=targets, cache_dir=cache_dir,
                              defines=defines, mode=mode)
    by_name = plugins(bundle, device_loader=load_manifest)
    missing = [k.name for k in kernels if k.name not in by_name]
    if missing:
        raise ValueError(
            f"eagle.plan.auto: {missing[0]!r} was built as several units (a "
            f"segmented kernel: {sorted(by_name)}), which are planned one by one: "
            "build it with hawk.artifact.build_bundle and plan each plugin of "
            "hawk.artifact.plugins(bundle)")
    return tuple(by_name[k.name] for k in kernels)


def _default_targets() -> tuple:
    """``("host", "cuda")`` with a usable device and compiler, else ``("host",)``."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return ("host",)
    except Exception:
        return ("host",)
    from hawk.compile import device_compiler_kind, nvrtc

    if device_compiler_kind() == "nvcc":
        return ("host", "cuda")
    try:
        nvrtc.nvrtc_version()
    except Exception:
        return ("host",)
    return ("host", "cuda")


def _eager_targets() -> tuple:
    """What a default deploy waits for: the device alone when a GPU is usable
    (the host builds in the background, :class:`_HostSide`), else the host."""
    return ("cuda",) if "cuda" in _default_targets() else ("host",)


def _bundle_key(kernels, targets, defines=(), mode="float64") -> str:
    """The cache directory name of one :func:`auto` request; hawk's own
    stamp still decides whether the bytes there are the unit's.
    ``defines`` (e.g. the active-set fast-entries macro,
    :func:`_wants_fast_entries`) is eagle's own choice, not read off the
    kernel, so it is folded in here explicitly -- the same reason
    ``host_profile``/``opt_level``/the device arch are below: two requests
    that build different bytes never share one folder."""
    h = hashlib.sha256(b"eagle.plan.auto/2\n")
    for k in kernels:
        sinks = k.sinks
        tag = tuple(repr(getattr(sinks, a, None))
                    for a in ("primal", "kind", "wrt", "primal_unit"))
        h.update(json.dumps([k.name, str(k.walk.digest), repr(k.kind), tag]).encode())
    h.update(json.dumps(list(targets)).encode())
    h.update(json.dumps(list(defines)).encode())
    if mode != "float64":  # float64 keys stay as before, so existing caches still hit
        h.update(mode.encode())
    # One folder per host profile, so two profiles of one kernel never
    # rebuild over each other (hawk's stamp already keeps them apart).
    from hawk.compile.toolchain import host_profile, opt_level

    h.update(host_profile().encode())
    # Same reason, for the opt level: it changes the device (nvcc) build
    # too, so an "O0" request and an "O3" one of the same kernel must never
    # land in the same folder either.
    h.update(opt_level().encode())
    # And again for the resolved device arch (hawk.artifact.arch; this call
    # passes no device_arch= of its own, so it follows the running GPU) --
    # called only when "cuda" is actually a target, same gate as hawk's own
    # unit key, so a host-only request never pays for the device probe and
    # two different GPUs' default builds never share one folder.
    if "cuda" in targets:
        from hawk.artifact import arch as _hawk_arch

        h.update(_hawk_arch().encode())
    return h.hexdigest()[:32]
