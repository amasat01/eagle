# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The backend wall checks: what the CUDA plugin exports; the core is CUDA-free.

Used by ``test_backend_wall.py`` and runnable on any artifact::

    python backend_wall.py symbols <libeagle_cuda.so>
    python backend_wall.py symbols <libeagle_cuda.so> --remint   # rewrite the manifests
    python backend_wall.py core <_core.so>
    python backend_wall.py plugin <libeagle_cuda.so> [--archs 61,70] [--ptx 90]

Each check returns a list of problems (empty = pass); the CLI prints them and exits
1 when there are any. The manifests are ``LC_ALL=C``-sorted name lists:
``expected_backend_symbols.txt`` (the locked ``eagle_backend_*`` seam) and
``expected_backend_x_symbols.txt`` (the experimental ``eagle_backend_x_*`` hooks).
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
LOCKED_MANIFEST = HERE / "expected_backend_symbols.txt"
X_MANIFEST = HERE / "expected_backend_x_symbols.txt"

#: What the plugin may need at load: glibc's pieces, the dynamic loader, OpenMP.
PLUGIN_ALLOWED_NEEDED = (
    "libc.so.",
    "libm.so.",
    "libpthread.so.",
    "libdl.so.",
    "librt.so.",
    "libgomp.so.",
    "ld-linux-x86-64.so.",
)
#: The ship build's device code: SASS for every architecture the toolkit
#: compiles from sm_60 up, which always includes at least these, plus PTX at
#: the newest SASS architecture (JIT-compiled on GPUs newer than the toolkit).
SHIP_SASS_FLOOR = ("61", "70", "80", "90")
SHIP_SASS_MIN = 60

_SEAM = re.compile(r"^eagle_backend_[a-z0-9_]+$")
_EXPERIMENTAL = re.compile(r"^eagle_backend_x_")
#: A CUDA runtime/driver API symbol (cudaMalloc, cuLaunchKernel, __cudaRegister...).
_CUDA_SYMBOL = re.compile(r"^(_+cuda|cuda[A-Z]|cu[A-Z][a-z])")


class ToolMissing(RuntimeError):
    """A binutils/CUDA tool the check needs is not on PATH."""


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolMissing(f"{name} is not on PATH")
    return path


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


#: auditwheel bundles a dependency into a wheel under a hashed name
#: (``libgomp-fb3a0f06.so.1.0.0``); the library it stands for is the unhashed one.
_VENDORED = re.compile(r"^(lib[A-Za-z0-9_+]+)-[0-9a-f]{8}(\.so(?:\.[0-9]+)*)$")


def unvendored(name: str) -> str:
    """``name`` with an auditwheel hash removed (``libgomp-fb3a0f06.so.1.0.0`` ->
    ``libgomp.so.1.0.0``); any other name unchanged."""
    match = _VENDORED.match(name)
    return match.group(1) + match.group(2) if match else name


def needed(path: str | os.PathLike) -> list[str]:
    """The DT_NEEDED entries of ``path``, auditwheel-bundled ones under the
    name of the library they stand for."""
    out = _run(_tool("readelf"), "-d", str(path))
    return [
        unvendored(ln.split("[", 1)[1].rstrip("]"))
        for ln in out.splitlines() if "(NEEDED)" in ln
    ]


def exported(path: str | os.PathLike) -> list[str]:
    """Every defined dynamic symbol of ``path`` (version nodes excluded), sorted."""
    out = _run(_tool("nm"), "-D", "--defined-only", str(path))
    names = set()
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) < 3 or parts[1] == "A":  # 'A' = a version-definition node
            continue
        names.add(parts[2].split("@", 1)[0])
    return sorted(names, key=lambda s: s.encode())


def undefined(path: str | os.PathLike) -> list[str]:
    """Every undefined dynamic symbol of ``path``."""
    out = _run(_tool("nm"), "-D", "--undefined-only", str(path))
    return [ln.split()[-1].split("@", 1)[0] for ln in out.splitlines() if ln.strip()]


def section_names(path: str | os.PathLike) -> list[str]:
    out = _run(_tool("readelf"), "-S", "-W", str(path))
    return re.findall(r"\]\s+(\S+)", out)


def read_manifest(path: pathlib.Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]


def _diff(label: str, have: list[str], want: list[str]) -> list[str]:
    problems = []
    for name in sorted(set(want) - set(have)):
        problems.append(f"{label}: MISSING {name} (in the manifest, not exported)")
    for name in sorted(set(have) - set(want)):
        problems.append(f"{label}: UNEXPECTED {name} (exported, not in the manifest)")
    return problems


def check_symbols(
    plugin: str | os.PathLike,
    locked: pathlib.Path = LOCKED_MANIFEST,
    experimental: pathlib.Path = X_MANIFEST,
) -> list[str]:
    """The plugin exports exactly the two manifests, nothing outside the seam."""
    problems = []
    names = exported(plugin)
    leaked = [n for n in names if not _SEAM.match(n)]
    problems += [
        f"wall: LEAKED export {n} (not an eagle_backend_* seam symbol)" for n in leaked
    ]
    seam = [n for n in names if _SEAM.match(n)]
    for path in (locked, experimental):
        rows = read_manifest(path)
        if rows != sorted(rows, key=lambda s: s.encode()):
            problems.append(f"{path.name}: not LC_ALL=C sorted")
    problems += _diff(
        "locked", [n for n in seam if not _EXPERIMENTAL.match(n)], read_manifest(locked)
    )
    problems += _diff(
        "experimental",
        [n for n in seam if _EXPERIMENTAL.match(n)],
        read_manifest(experimental),
    )
    return problems


def remint(plugin: str | os.PathLike) -> None:
    """Rewrite both manifests from ``plugin`` (refused under CI)."""
    if os.environ.get("CI"):
        raise SystemExit("refusing to re-mint the symbol manifests under $CI")
    seam = [n for n in exported(plugin) if _SEAM.match(n)]
    head = "# eagle-backend/1 {} symbols of libeagle_cuda.so (LC_ALL=C sorted).\n"
    LOCKED_MANIFEST.write_text(
        head.format("locked")
        + "".join(n + "\n" for n in seam if not _EXPERIMENTAL.match(n))
    )
    X_MANIFEST.write_text(
        head.format("experimental")
        + "".join(n + "\n" for n in seam if _EXPERIMENTAL.match(n))
    )


def check_core(core: str | os.PathLike) -> list[str]:
    """The core neither needs nor references the CUDA runtime/driver."""
    problems = []
    for lib in needed(core):
        if lib.startswith(("libcuda.", "libcudart.", "libnvrtc.", "libnvJitLink.")):
            problems.append(f"core NEEDS {lib}")
    for sym in undefined(core):
        if _CUDA_SYMBOL.match(sym) or sym.startswith("eagle_backend_"):
            problems.append(
                f"core references {sym} (undefined: a link-time CUDA/seam binding)"
            )
    sections = section_names(core)
    for sec in (".nv_fatbin", "__nv_relfatbin", ".nvFatBinSegment"):
        if sec in sections:
            problems.append(f"core carries a {sec} section (it was compiled by nvcc)")
    return problems


def check_plugin(
    plugin: str | os.PathLike,
    sass: tuple[str, ...] | None = None,
    ptx: tuple[str, ...] | None = None,
) -> list[str]:
    """The plugin is self-contained and carries the expected device code.

    With ``sass``/``ptx`` given (a dev build), the archs must match exactly.
    Without them (the ship build), the SASS must cover :data:`SHIP_SASS_FLOOR`,
    hold nothing below :data:`SHIP_SASS_MIN`, and the PTX must be exactly one
    copy, at the newest SASS architecture.
    """
    problems = []
    for lib in needed(plugin):
        if not lib.startswith(PLUGIN_ALLOWED_NEEDED):
            problems.append(f"plugin NEEDS {lib} (outside the self-contained set)")
    cuobjdump = _tool("cuobjdump")
    elf = _run(cuobjdump, "--list-elf", str(plugin))
    have_sass = sorted(set(re.findall(r"\.sm_(\d+)\.cubin", elf)))
    ptxs = _run(cuobjdump, "--list-ptx", str(plugin))
    have_ptx = sorted(set(re.findall(r"\.sm_(\d+)\.ptx", ptxs)))
    if sass is not None or ptx is not None:
        if sass is not None and have_sass != sorted(sass):
            problems.append(f"plugin SASS archs {have_sass}, expected {sorted(sass)}")
        if ptx is not None and have_ptx != sorted(ptx):
            problems.append(f"plugin PTX archs {have_ptx}, expected {sorted(ptx)}")
        return problems
    missing = sorted(set(SHIP_SASS_FLOOR) - set(have_sass), key=int)
    if missing:
        problems.append(f"plugin SASS archs {have_sass} miss {missing}")
    low = [a for a in have_sass if int(a) < SHIP_SASS_MIN]
    if low:
        problems.append(f"plugin SASS archs {low} are below sm_{SHIP_SASS_MIN}")
    newest = max(have_sass, key=int) if have_sass else None
    if have_ptx != ([newest] if newest else []):
        problems.append(f"plugin PTX archs {have_ptx}, expected [{newest}] (the newest SASS)")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    what, path = argv[0], argv[1]
    if what == "symbols":
        if "--remint" in argv:
            remint(path)
            print("re-minted", LOCKED_MANIFEST.name, X_MANIFEST.name)
            return 0
        problems = check_symbols(path)
    elif what == "core":
        problems = check_core(path)
    elif what == "plugin":
        sass = ptx = None
        if "--archs" in argv:
            sass = tuple(argv[argv.index("--archs") + 1].split(","))
        if "--ptx" in argv:
            ptx = tuple(a for a in argv[argv.index("--ptx") + 1].split(",") if a)
        problems = check_plugin(path, sass, ptx)
    else:
        print(__doc__)
        return 2
    for p in problems:
        print(p)
    print(f"{what}: {'FAIL' if problems else 'OK'} ({len(problems)} problem(s))")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
