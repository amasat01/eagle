# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether-abi/2 ARTIFACT contract the rank bed is driven by — build one, and
load one back by what it DECLARES.

WHY THIS EXISTS. ``python/tests/mpi/test_rank_bed.py`` certifies
:data:`eagle.exec.RankPartition` against a deployed artifact rather than against a
hand-written fixture, because the bed is also pointed at a real deployed emission
(``$EAGLE_RANK_BED_ARTIFACT``). For that to be a swap and not a rewrite, NOTHING in
the bed may know a kernel's name: a row asks for "the unit declaring
``exec_access == 'mapreduce'``" and gets whatever the artifact carries under that
declaration. Everything a row needs — the entry symbols, the ``arg_spec``, the
access class, the reduction op, the scalar type — is read out of the manifest and
its sidecars here, and nothing is read out of a Python constant.

THE ARTIFACT SHAPE. A root directory of UNITS, one manifest per unit::

    <root>/<unit>/manifest.json     schema v2: aether_abi, exec_targets,
                                    exec_access, exec_op?, plugins[]
    <root>/<unit>/<id>.ptx          the device artifact (manifest `format`)
    <root>/<unit>/<id>.json         the sidecar: kernel, host_entry,
                                    host_artifact, arg_spec, scalar_type
    <root>/<unit>/<id>.so           the host artifact

ONE MANIFEST PER UNIT because the execution axis is a MANIFEST-level declaration
(``exec_access`` is one value, and ``exec_op`` is required iff that value is
``mapreduce``) — a single manifest covering four access classes could only do so by
misdeclaring three of them. The unit DIRECTORIES are named ``unit0…`` deliberately:
a name that spelled its access class would let a row find its unit by path instead
of by declaration, which is exactly the coupling the bed must not have.

THE HOST OBJECT RIDES THE SIDECAR, not the manifest's ``plugins[]`` entry, because
``MANIFEST_FORMATS`` is device-only today (``ptx``/``cubin``/``fatbin`` — the
manifest ``format`` discriminant selects how a loader reads DEVICE bytes). The
sidecar already carries ``host_entry`` (``eagle.host_launch`` reads it), so
``host_artifact`` sits beside it as the one additive key this contract adds.

VALIDATION IS EAGLE'S OWN. :func:`load` runs the SAME manifest gates
:func:`eagle.registry.load_manifest` runs — ``check_schema_version``,
``check_execution_axis``, ``check_aether_abi``, the ``format`` discriminant — plus
``validate_sidecar`` per plugin. What it does NOT do is build a ``LoadedPure``:
that loader reads the v1 launch vocabulary (``vector_inputs``/``params``/
``mutables``) and drives the v1 whole-view launch path, neither of which a v2
partition-triple body has or wants. So this returns the duck-typed plugin object
:mod:`eagle.plan` drives — the same shape ``test_exec_contract_rows.py`` and
``test_plan_packer_e2.py`` already hand it.
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import shutil
import subprocess

from _device_file import portable_arch

#: The repo root — it holds ``plugin/gref_abi.h`` (the header both fixture TUs
#: include) and ``tests/fixtures/`` (their sources).
EAGLE_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The DEFAULT artifact's bodies: the repo's own hand-written aether-abi/2
#: fixtures, the same two TUs the C++ exec-contract and MPI-bed suites compile.
#: Reusing them rather than writing a third copy is what keeps the Python bed and
#: the C++ bed certifying the same bodies.
HOST_SRC = EAGLE_ROOT / "tests" / "fixtures" / "host_plugin_execv2.cpp"
DEVICE_SRC = EAGLE_ROOT / "tests" / "fixtures" / "device_plugin_execv2.cu"

#: The units the DEFAULT artifact declares, as
#: ``(kernel, exec_access, exec_op, arg_spec)``. Written out here because this is
#: the PRODUCER — the emitter's job, handled elsewhere. Nothing in the
#: bed reads this table; the bed reads the manifests it produces.
_DEFAULT_UNITS = (
    (
        "exec_local",
        "sample_local",
        None,
        [["mutable", "y"], ["per_sample", "x"], ["uniform", "a"],
         ["uniform", "b"], ["nsamples", "n"]],
    ),
    (
        "exec_gather",
        "cross_sample_read",
        None,
        [["mutable", "y"], ["per_sample", "table"], ["nsamples", "n"]],
    ),
    (
        "exec_mapreduce",
        "mapreduce",
        "sum",
        [["accum_out", "partial"], ["per_sample", "x"], ["nsamples", "n"]],
    ),
    (
        "exec_scatter",
        "cross_sample_write",
        None,
        [["wide_out", "acc"], ["per_sample", "x"], ["nsamples", "n"]],
    ),
)

#: The access classes the rank bed certifies. An artifact missing one of these is
#: an INCOMPLETE input for the bed, and the bed says so in a row of its own rather
#: than quietly certifying the classes that happen to be present.
REQUIRED_ACCESS_CLASSES = (
    "sample_local", "cross_sample_read", "mapreduce", "cross_sample_write",
)


# --------------------------------------------------------------------------- #
# Producing an artifact.
# --------------------------------------------------------------------------- #
def _gxx() -> str:
    found = shutil.which(os.environ.get("CXX", "")) or shutil.which("g++")
    if found is None and os.path.exists("/usr/bin/g++"):
        found = "/usr/bin/g++"
    if found is None:
        raise RuntimeError(
            "building the default aether-abi/2 bed artifact needs g++ for the host "
            "object; none found on $CXX, $PATH or /usr/bin/g++"
        )
    return found


def _nvcc() -> str:
    found = shutil.which("nvcc")
    if found:
        return found
    cand = pathlib.Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "bin" / "nvcc"
    if cand.exists():
        return str(cand)
    raise RuntimeError(
        "building the default aether-abi/2 bed artifact needs nvcc for the device "
        "PTX; none found on $PATH or under $CUDA_PATH"
    )


def _run(cmd, what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{what} failed (rc {proc.returncode}):\n"
            f"  $ {' '.join(str(c) for c in cmd)}\n{proc.stderr}"
        )


def build(root) -> pathlib.Path:
    """Compile the repo's aether-abi/2 fixture TUs and lay out the DEFAULT bed
    artifact under ``root``; returns ``root``.

    A GATE BUILDS ITS OWN INPUTS: ``tests/mpi/check_rank_bed.sh`` calls this once,
    before it launches any rank, so the bed never runs against whatever object
    happened to be lying around from a previous build and the two ranks never race
    to compile the same file."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)

    stage = root / ".build"
    stage.mkdir(exist_ok=True)
    so = stage / "execv2.so"
    ptx = stage / "execv2.ptx"
    _run(
        [_gxx(), "-O2", "-std=c++17", "-shared", "-fPIC", f"-I{EAGLE_ROOT}",
         "-o", str(so), str(HOST_SRC)],
        "compiling the bed artifact's host object",
    )
    _run(
        [_nvcc(), "-ptx", "-std=c++17", f"-arch={portable_arch(_nvcc())}", f"-I{EAGLE_ROOT}",
         str(DEVICE_SRC), "-o", str(ptx)],
        "compiling the bed artifact's device PTX",
    )

    for i, (kernel, access, op, arg_spec) in enumerate(_DEFAULT_UNITS):
        # The unit directory carries NO information: a bed row must find its unit
        # by the manifest's declaration, never by a path it could have guessed.
        unit = root / f"unit{i}"
        unit.mkdir(exist_ok=True)
        plugin_id = f"p{i}"
        shutil.copyfile(ptx, unit / f"{plugin_id}.ptx")
        shutil.copyfile(so, unit / f"{plugin_id}.so")

        sidecar = {
            "kernel": kernel,
            "host_entry": f"{kernel}_host",
            "host_artifact": f"{plugin_id}.so",
            "aether_abi": "aether-abi/2",
            "schema_version": 2,
            "pattern": "pure",
            "scalar_type": "float64",
            "arg_spec": arg_spec,
        }
        (unit / f"{plugin_id}.json").write_text(json.dumps(sidecar, indent=2) + "\n")

        manifest = {
            "schema_version": 2,
            "pattern": "pure",
            "aether_abi": "aether-abi/2",
            "exec_targets": ["device", "host"],
            "exec_access": access,
            "plugins": [
                {"id": plugin_id, "order": 0, "enabled": True,
                 "artifact": f"{plugin_id}.ptx", "sidecar": f"{plugin_id}.json",
                 "format": "ptx"}
            ],
        }
        if op is not None:
            manifest["exec_op"] = op
        (unit / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return root


# --------------------------------------------------------------------------- #
# Loading one back.
# --------------------------------------------------------------------------- #
class LoadedV2Plugin:
    """One artifact plugin, in the shape :func:`eagle.plan.plan` drives.

    Every attribute is READ OFF THE ARTIFACT: ``arg_spec``/``scalar_type`` from
    the sidecar, ``exec_access``/``exec_op`` from the manifest that names it. The
    two entry points are resolved LAZILY and independently — a host-only row must
    not need cupy, and a device-only row must not need the ``.so`` to be loadable
    — so an artifact that ships only one target still serves the rows that use
    that one."""

    def __init__(self, directory, entry, sidecar, manifest):
        self._dir = pathlib.Path(directory)
        self._entry = dict(entry)
        self._sidecar = dict(sidecar)
        self.id = entry["id"]
        self.arg_spec = [tuple(a) for a in sidecar["arg_spec"]]
        self.scalar_type = sidecar.get("scalar_type", "float64")
        self.arg_widths = dict(sidecar.get("arg_widths", {}))
        self.exec_access = manifest["exec_access"]
        self.exec_op = manifest.get("exec_op")
        self.abi_tag = sidecar["aether_abi"]
        self._host_lib = None
        self._device_module = None
        self._device_kernel = None

    @property
    def host_entry(self) -> int:
        """The host entry's address, ``ctypes``-resolved from the sidecar's
        ``host_artifact`` + ``host_entry`` — the ``void* const*`` + int64 triple
        shape."""
        if self._host_lib is None:
            artifact = self._sidecar.get("host_artifact")
            if not artifact:
                raise ValueError(
                    f"artifact plugin {self.id!r}: its sidecar declares no "
                    "'host_artifact', so it carries no host target"
                )
            self._host_lib = ctypes.CDLL(str(self._dir / artifact))
        fn = getattr(self._host_lib, self._sidecar["host_entry"])
        return ctypes.cast(fn, ctypes.c_void_p).value

    @property
    def device_function(self) -> int:
        """The device entry's ``CUfunction`` handle, loaded from the manifest
        entry's ``artifact`` at its declared ``format``."""
        if self._device_kernel is None:
            import cupy as cp

            self._device_module = cp.RawModule(
                path=str(self._dir / self._entry["artifact"])
            )
            self._device_kernel = self._device_module.get_function(
                self._sidecar["kernel"]
            )
        return self._device_kernel.kernel.ptr


class Unit:
    """One manifest's worth of artifact: its declared execution axis plus the
    plugins it names."""

    def __init__(self, manifest_path, manifest, plugins):
        self.path = pathlib.Path(manifest_path)
        self.access = manifest["exec_access"]
        self.op = manifest.get("exec_op")
        self.targets = tuple(manifest["exec_targets"])
        self.plugins = tuple(plugins)

    @property
    def plugin(self):
        """The single plugin of a one-plugin unit; a unit naming several is
        refused rather than silently reduced to its first."""
        if len(self.plugins) != 1:
            raise ValueError(
                f"{self.path}: this unit names {len(self.plugins)} enabled "
                "plugins; the rank bed's rows drive exactly one body per unit"
            )
        return self.plugins[0]

    def __repr__(self) -> str:
        return f"Unit({self.path.parent.name!r}, access={self.access!r})"


def _load_unit(manifest_path) -> Unit:
    """Load one manifest through EAGLE's OWN gates, then bind its plugins."""
    from eagle.abi import check_aether_abi
    from eagle.roles import MANIFEST_FORMATS, check_execution_axis, check_schema_version
    from eagle.sidecar import validate_sidecar

    manifest_path = pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    name = f"{manifest_path.parent.name}/{manifest_path.name}"

    # The same three manifest-level gates, in the same ORDER, that
    # eagle.registry.load_manifest applies (see its ORDER comment: the execution
    # axis is judged before the ABI tag, so a malformed axis is refused naming
    # THAT rule).
    version = check_schema_version(manifest, name=name, allow_legacy_version_key=True)
    check_execution_axis(manifest, version, name=name)
    check_aether_abi(manifest, kind="plugin manifest", name=name)
    if version != 2:
        raise ValueError(
            f"{name}: the rank bed drives aether-abi/2 artifacts; this manifest "
            f"declares schema_version {version}"
        )

    plugins = []
    seen = set()
    for entry in manifest.get("plugins", []):
        if entry["id"] in seen:
            raise ValueError(f"{name}: duplicate plugin id {entry['id']!r}")
        seen.add(entry["id"])
        if entry.get("format") not in MANIFEST_FORMATS:
            raise ValueError(
                f"{name}: plugin {entry['id']!r}: unknown artifact format "
                f"{entry.get('format')!r} (supported: {sorted(MANIFEST_FORMATS)})"
            )
        if not entry.get("enabled", True):
            continue
        sidecar_path = manifest_path.parent / entry["sidecar"]
        sidecar = json.loads(sidecar_path.read_text())
        validate_sidecar(sidecar, name=sidecar_path.name)
        check_aether_abi(sidecar, kind="plugin sidecar", name=sidecar_path.name)
        plugins.append(
            LoadedV2Plugin(manifest_path.parent, entry, sidecar, manifest)
        )
    if not plugins:
        raise ValueError(f"{name}: names no ENABLED plugin")
    return Unit(manifest_path, manifest, plugins)


def load(root) -> tuple:
    """Every unit under ``root``, ordered by manifest path.

    Discovery is ``root/manifest.json`` plus ``root/*/manifest.json`` — a
    single-unit artifact and a multi-unit one look the same to a caller. An empty
    discovery RAISES: a bed that found no manifest would otherwise report a clean
    run over nothing."""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise ValueError(
            f"the bed artifact root {str(root)!r} is not a directory; point "
            "$EAGLE_RANK_BED_ARTIFACT at one, or let "
            "tests/mpi/check_rank_bed.sh build the default"
        )
    manifests = sorted(
        set(root.glob("manifest.json")) | set(root.glob("*/manifest.json"))
    )
    if not manifests:
        raise ValueError(
            f"no manifest.json under {str(root)!r} (looked at <root>/manifest.json "
            "and <root>/*/manifest.json) — an artifact with no manifest certifies "
            "nothing"
        )
    return tuple(_load_unit(m) for m in manifests)


def unit_for(units, access: str) -> Unit:
    """The unit DECLARING ``access``, named in the error when the artifact has
    none — the one lookup every bed row uses, so no row ever names a kernel."""
    matching = [u for u in units if u.access == access]
    if not matching:
        raise ValueError(
            f"this artifact declares no unit with exec_access={access!r}; it "
            f"declares {sorted({u.access for u in units})}. The rank bed certifies "
            f"{list(REQUIRED_ACCESS_CLASSES)} and cannot substitute one for another"
        )
    if len(matching) > 1:
        raise ValueError(
            f"this artifact declares {len(matching)} units with "
            f"exec_access={access!r}; a bed row must not have to choose between "
            "them ({}).".format(", ".join(str(u.path) for u in matching))
        )
    return matching[0]
