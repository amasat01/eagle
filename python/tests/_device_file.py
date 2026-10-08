# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device artifact a HAWK-published unit holds for one kernel, whatever its format.

Since the default device target follows the installed driver vs toolkit (``cubin``
when the compiler is newer than the driver, ``ptx`` otherwise), a test consuming a
hawk-published unit must not hard-code the extension: it asks for the one file the
unit holds for that kernel. Mirrors hawk's own ``tests/_device_file.py``.

A test that compiles its OWN device fixture with ``nvcc`` follows the same rule
through :func:`device_target` / :func:`compile_device`, so it loads whichever
toolkit is first on ``PATH``: a PTX image from a toolkit newer than the driver
carries an ISA version the driver's JIT refuses
(``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``), a CUBIN for the device's own arch
does not.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_RELEASE = re.compile(r"release (\d+)\.(\d+)")


def device_artifact(directory: Path, stem: str) -> Path:
    """The single ``<stem>.ptx`` or ``<stem>.cubin`` in ``directory``."""
    found = [directory / f"{stem}{ext}" for ext in (".ptx", ".cubin")
             if (directory / f"{stem}{ext}").is_file()]
    assert len(found) == 1, (stem, sorted(p.name for p in directory.iterdir()))
    return found[0]


def nvcc_version(nvcc: str) -> tuple[int, int] | None:
    """``(major, minor)`` from ``nvcc --version``, or ``None`` if unreadable."""
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = _RELEASE.search(out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _driver_and_arch() -> tuple[tuple[int, int], str] | None:
    """The driver's CUDA version and the current device's ``sm_XX``, or
    ``None`` when no device is reachable."""
    try:
        import cupy as cp

        encoded = cp.cuda.runtime.driverGetVersion()
        cc = cp.cuda.Device().compute_capability
    except Exception:  # noqa: BLE001 -- no cupy, no driver or no device
        return None
    return (encoded // 1000, (encoded % 1000) // 10), f"sm_{cc}"


def portable_arch(nvcc: str) -> str:
    """The arch of a portable PTX fixture: ``sm_60``, or the oldest arch
    ``nvcc`` still targets when it dropped ``sm_60`` (CUDA 13 starts at
    ``sm_75``)."""
    try:
        out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return "sm_60"
    archs = [int(m) for m in re.findall(r"compute_(\d+)", out)]
    return f"sm_{max(60, min(archs))}" if archs else "sm_60"


def device_target(nvcc: str) -> tuple[str, str]:
    """``(format, arch)`` for a fixture compiled by ``nvcc``: ``("cubin",
    "sm_XX")`` for the current device when the toolkit is newer than the
    driver, else ``("ptx", portable_arch(nvcc))`` — the portable image these
    fixtures always used. With no device reachable the PTX answer stands
    (there is no arch to lower to, and nothing will load the image anyway)."""
    probed = _driver_and_arch()
    if probed is None:
        return "ptx", portable_arch(nvcc)
    driver, arch = probed
    toolkit = nvcc_version(nvcc)
    if toolkit is None or toolkit > driver:
        return "cubin", arch
    return "ptx", portable_arch(nvcc)


def compile_device(nvcc: str, source: Path, stem: Path, flags=()) -> Path:
    """Compile ``source`` to ``<stem>.ptx`` or ``<stem>.cubin`` by
    :func:`device_target` and return the path; a failed compile asserts with
    nvcc's stderr."""
    fmt, arch = device_target(nvcc)
    out = Path(f"{stem}.{fmt}")
    proc = subprocess.run(
        [nvcc, f"-{fmt}", *flags, f"-arch={arch}", str(source), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"device fixture compile failed:\n{proc.stderr}"
    return out
