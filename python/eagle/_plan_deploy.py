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

from . import _cell_rows, _lane_bind, _layout, _recording, _split
from . import exec as eexec
from ._plan_binding import BoundPlan
from ._plan_planes import _output_names, residency
from .roles import OUTPUT_ROLES

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

    __slots__ = ("plugin", "plan_kw", "kernel", "_plans", "_host", "_call", "_build",
                 "_variants", "_vlock")

    def __init__(self, plugin, plan_kw: dict, host=None, kernel=None, build=None):
        self.plugin = plugin
        self.plan_kw = dict(plan_kw)
        self.kernel = kernel  # the hawk kernel it was built from (None: a prebuilt plugin)
        self._build = build  # (targets, cache_dir, scalar_type) it was built with
        self._variants = {}  # {id(variant): (variant, its AutoPlan)}, see _routed
        self._vlock = threading.Lock()
        self._plans = {}
        self._host = host  # (_HostSide, index) when the host side builds in the background
        self._call = None  # the call's signature, built on the first call

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

    def _routed(self, args, kw):
        """``(plan, args, kw)`` for a call: a per-lane input of a ``Grid`` kernel given
        broadcast runs the kernel's variant of that pattern (hawk's rule,
        :mod:`hawk._broadcast`: built once per pattern, with this plan's build
        options), the caller's buffer bound compact. Anything else, and a plan of a
        prebuilt plugin or of a variant, is returned as it came."""
        kernel = self.kernel
        if kernel is None or getattr(kernel, "broadcast", None) or not hasattr(kernel, "lane_space"):
            return self, args, kw      # no kernel, a variant, or a stand-in that declares no lanes
        from hawk import _broadcast

        try:
            run, args, kw = _broadcast.route(kernel, args, kw)
        except _broadcast.HawkError as exc:     # eagle's own doors refuse with ValueError
            raise ValueError(str(exc)) from None
        if run is kernel:
            return self, args, kw
        with self._vlock:
            held = self._variants.get(id(run))
            if held is None:
                if getattr(run.sinks, "source", None) is not None:   # a derivative: hawk's own build
                    plan = run._plan or run._deploy()
                else:
                    targets, cache_dir, scalar_type = self._build
                    plan = auto(run, targets=targets, cache_dir=cache_dir,
                                scalar_type=scalar_type, **self.plan_kw)
                held = self._variants[id(run)] = (run, plan)
        return held[1], args, kw

    @_layout.door
    def run(self, /, **kw):
        """:meth:`Plan.run` on the plan :meth:`select` picks for ``kw``; a per-lane
        input given broadcast runs that pattern's variant (:meth:`_routed`)."""
        plan, _args, kw = self._routed((), kw)
        if plan is not self:
            return plan.run(**kw)
        return self.select(kw).run(**kw)

    @_layout.door
    def bind(self, /, **planes) -> BoundPlan:
        """:meth:`Plan.bind` on the plan :meth:`select` picks for ``planes``; a per-lane
        input given broadcast binds that pattern's variant (:meth:`_routed`), whose
        plane is the caller's buffer, compact."""
        plan, _args, planes = self._routed((), planes)
        if plan is not self:
            return plan.bind(**planes)
        return self.select(planes).bind(**planes)

    def _signature(self) -> tuple:
        """``(Signature, required names, prior reads, kernel name, all names,
        required set)``
        of a call, bound as :func:`eagle.simulate` binds a kernel's own
        arguments."""
        from ._simulate import _prior_reads, _signature
        from ._until_done import _RESERVED, _kernel_name

        arg_spec = _split.arg_spec(self.plugin)       # a split kernel: every stage's planes
        rows = _cell_rows.row_planes(self.plugin) | frozenset(_cell_rows.companions_of(self.plugin))
        decl = {nm: ("mutable" if role in OUTPUT_ROLES else role,)
                for role, nm in arg_spec
                if role not in ("nsamples", "terminated") and nm not in _RESERVED
                and nm not in rows}
        kernel = self.kernel
        declared = kernel.planes if kernel is not None and kernel.planes else decl   # a derivative declares none
        order = [nm for nm in declared if nm in decl]
        prior = _prior_reads([kernel])
        name = kernel.name if kernel is not None else _kernel_name(self.plugin)
        sig, required = _signature(decl, order, prior)
        return sig, tuple(required), prior, name, frozenset(order), frozenset(required)

    def bind_args(self, args, kw) -> dict:
        """``{plane: value}`` for a call ``k(*args, **kw)``: positional in the
        kernel's own order and/or by name, an un-prior-read Mutable left out
        when omitted. An omittable Mutable or a reduction's cell may also be
        passed by position, after the required arguments. A compensated sum's
        companion is eagle's and takes no position; a caller binds its own by
        name only."""
        call = self._call
        if call is None:
            call = self._call = self._signature()
        kernel = self.kernel
        if kernel is not None and (len(args) > len(call[1]) or kw.keys() - call[4]):
            comps = _cell_rows.companions_of(self.plugin)
            order = [nm for nm, decl in kernel.planes.items()   # the Grid declaration is no plane
                     if not getattr(decl, "grid", None) and nm not in comps]
            if len(args) > len(order):
                raise ValueError(f"{call[3]}() too many positional arguments")
            given = dict(zip(order, args))
            if given.keys() & kw.keys():
                raise ValueError(f"{call[3]}() got multiple values for argument {sorted(given.keys() & kw.keys())[0]!r}")
            given.update(kw)
            cells = {nm: given.pop(nm) for nm in list(given)
                     if nm in order and nm not in call[4] or nm in comps}
            return {**self._bind_args((), given, call), **cells}
        return self._bind_args(args, kw, call)

    def _bind_args(self, args, kw, call) -> dict:
        sig, required, prior, name, names, needed = call
        if self.kernel is not None or not args:     # a bad call takes the slow path
            if not kw and len(args) == len(required):
                return dict(zip(required, args))
            if len(args) <= len(required):
                bound = dict(zip(required, args))
                count = len(bound) + len(kw)
                bound.update(kw)
                if len(bound) == count and kw.keys() <= names and bound.keys() >= needed:
                    return bound
        if args and self.kernel is None:
            raise ValueError(f"{name}() was built from a prebuilt plugin, which has no call "
                             "signature of its own; pass its planes by name")
        from ._simulate import _bind

        try:
            return _bind(name, sig, required, prior, args, kw)
        except ValueError as exc:       # the call's own words, not the door's
            raise ValueError(str(exc).removeprefix("eagle.simulate: ")) from None

    def returned(self, planes: dict):
        """The kernel's ``returns`` tree filled from ``planes`` (``{slot name:
        value}``); ``planes`` itself when the kernel declares none."""
        tree = getattr(self.kernel, "returns", None)
        return planes if tree is None else _fill(tree, planes)

    def __call__(self, /, *args, **kw):
        """Run the kernel once, as a call to it: ``plan(x, a, b)`` binds
        like :func:`eagle.simulate` binds a kernel's arguments, runs one
        launch where the arrays live (:meth:`select`), and returns the
        kernel's ``returns`` tree, else what :meth:`Plan.run` returns. A
        Mutable argument is the caller's array, written in place. Under an
        active recorder (:mod:`eagle._recording`) the call is handed to it
        instead of launching."""
        plan, args, kw = self._routed(args, kw)
        if plan is not self:
            return plan(*args, **kw)
        recorder = _recording.current()
        if recorder is not None:
            return recorder.record_call(self, args, kw)
        kw, lifted = _lane_bind.take_extent(self.plugin, kw)
        bound = self.bind_args(args, kw)
        bound.update(lifted)
        result = self.select(bound).run(**bound)
        if getattr(self.kernel, "returns", None) is None:
            return result
        if not isinstance(result, dict):
            tree = self.kernel.returns
            (out,) = (tree,) if isinstance(tree, str) else _output_names(_split.arg_spec(self.plugin))
            result = {out: result}
        return self.returned(result)


def _fill(tree, planes):
    """``tree`` (nested tuples, lists and dicts of slot names) with each name
    replaced by its value in ``planes``."""
    if isinstance(tree, str):
        return planes[tree]
    if isinstance(tree, dict):
        return {k: _fill(v, planes) for k, v in tree.items()}
    return type(tree)(_fill(v, planes) for v in tree)


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
    building beside it -- else the host alone), ``cache_dir`` and ``scalar_type`` are refused
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
        return tuple(AutoPlan(p, plan_kw, host, k, (targets, cache_dir, scalar_type))
                     for k, (p, host)
                     in zip(kernels, _deploy_hawk(kernels, targets, cache_dir, scalar_type)))
    if _is_hawk_kernel(plugin):
        p, host = _deploy_hawk([plugin], targets, cache_dir, scalar_type)[0]
        return AutoPlan(p, plan_kw, host, plugin, (targets, cache_dir, scalar_type))
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
    a GPU run never waits for the host compiler. Any build including a device
    target compiles the loop's own small device kernels on a second thread
    meanwhile (:func:`eagle._until_done._warm_device_kernels`)."""
    _hawk_api()  # the refusal naming hawk comes first
    host = None
    if targets is None:
        targets = _eager_targets()
        if targets == _targets_of(_DEVICE):
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
        self._args = (list(kernels), _targets_of(_HOST), cache_dir, scalar_type)
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
    for a build including a device target (once per device per process); the
    thread, or ``None``. A failure there only loses the head start."""
    if not _has_entry(targets, _DEVICE):
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
    includes a device target (the entries are device-only). One flag for the
    whole bundle call, matching :func:`eagle._until_done._build_auto`'s
    own ONE-place eligibility: this is the single spot that decides eagle
    builds these kernels with the define by DEFAULT, so the ordinary
    deploy/simulate path gets the fast entries without the caller asking.
    A non-automatic kernel is unaffected either way -- the define gates
    code its own emission never reaches."""
    if not _has_entry(targets, _DEVICE):
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


def _build_by_name(kernels, targets, cache_dir, scalar_type=None) -> tuple:
    """Build ``kernels`` into one bundle; ``({plugin name: plugin}, bundle)``
    (a segmented kernel appears as its units)."""
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
    _split.link(by_name, bundle)
    return by_name, bundle


def _hawk_plugins(kernels, targets, cache_dir, scalar_type=None) -> tuple:
    """Build ``kernels`` into one bundle and return their plugins in order."""
    by_name, bundle = _build_by_name(kernels, targets, cache_dir, scalar_type)
    missing = [k.name for k in kernels if k.name not in by_name]
    if missing:
        raise ValueError(
            f"eagle.plan.auto: {missing[0]!r} was built as several units (a "
            f"segmented kernel: {sorted(by_name)}), which are planned one by one: "
            "build it with hawk.artifact.build_bundle and plan each plugin of "
            "hawk.artifact.plugins(bundle)")
    for k in kernels:  # what eagle.diff needs to build this kernel's vjp beside it
        by_name[k.name]._hawk_source = (k, bundle.digest, cache_dir)
    return tuple(by_name[k.name] for k in kernels)


#: The two ways eagle enters a built kernel: hawk's rows name theirs.
_HOST, _DEVICE = "host", "device"


def _targets_of(entry: str) -> tuple:
    """The ids of hawk's target rows eagle enters as ``entry``."""
    from hawk.targets import TARGETS

    return tuple(t for t, row in TARGETS.items() if row.entry == entry)


def _has_entry(targets, entry: str) -> bool:
    """Whether any of the target ids ``targets`` is entered as ``entry``."""
    return bool(set(targets) & set(_targets_of(entry)))


def _default_targets() -> tuple:
    """Every host row, plus every device row with a usable device and compiler."""
    host = _targets_of(_HOST)
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return host
    except Exception:
        return host
    from hawk.compile import device_compiler_kind, nvrtc

    if device_compiler_kind() != "nvcc":
        try:
            nvrtc.nvrtc_version()
        except Exception:
            return host
    return host + _targets_of(_DEVICE)


def _eager_targets() -> tuple:
    """What a default deploy waits for: the device alone when a GPU is usable
    (the host builds in the background, :class:`_HostSide`), else the host."""
    device = _targets_of(_DEVICE)
    return device if _has_entry(_default_targets(), _DEVICE) else _targets_of(_HOST)


def _bundle_key(kernels, targets, defines=(), mode="float64") -> str:
    """The cache directory name of one :func:`auto` request; hawk's own
    stamp still decides whether the bytes there are the unit's.
    ``defines`` (e.g. the active-set fast-entries macro,
    :func:`_wants_fast_entries`) is eagle's own choice, not read off the
    kernel, so it is folded in here explicitly -- the same reason
    ``host_profile``/``opt_level``/the device arch are below: two requests
    that build different bytes never share one folder."""
    from hawk.artifact.unit_cache import declaration_digest

    h = hashlib.sha256(b"eagle.plan.auto/3\n")
    for k in kernels:
        sinks = k.sinks
        tag = tuple(repr(getattr(sinks, a, None))
                    for a in ("primal", "kind", "wrt", "primal_unit"))
        # The declarations (plane sizes, Grid extents) never reach the walk, so two
        # sizes of one kernel would otherwise share, and race on, one folder.
        h.update(json.dumps([k.name, str(k.walk.digest), declaration_digest(k),
                             repr(k.kind), tag]).encode())
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
    # called only when a device row is actually a target, same gate as hawk's own
    # unit key, so a host-only request never pays for the device probe and
    # two different GPUs' default builds never share one folder.
    if _has_entry(targets, _DEVICE):
        from hawk.artifact import arch as _hawk_arch

        h.update(_hawk_arch().encode())
    return h.hexdigest()[:32]
