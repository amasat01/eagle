#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Gate: EXECUTE every docs notebook, in memory (nothing is written back).

Every notebook under the given directories runs once on the CPU (the kernel is
started with ``CUDA_VISIBLE_DEVICES`` empty, so no GPU is visible) and, when a
CUDA GPU is present, once more on the GPU. Any cell error fails the gate; all
notebooks are still attempted so the report is complete.

A notebook declares what it needs in its metadata::

    "metadata": {"gate": {"gpu_only": "uses CuPy arrays throughout",
                          "needs": ["warp"]}}

* ``gpu_only`` (a reason string): skipped on the CPU leg with that reason.
* a CELL whose metadata carries ``"gate": {"gpu_only": "<reason>"}``: on the CPU leg the notebook
  runs up to (not including) that cell and stops there, so a CPU tutorial can end with a GPU-only section.
* ``needs`` (importable module names): the notebook is skipped, with the
  missing name in the reason, when one is not installed.

``make nbcheck`` checks that the committed notebooks are filled; this gate
checks that they still RUN (a committed output can outlive the code that
produced it). Uses nbclient, a myst_nb dependency: no extra tool.

    python _tools/nb_run_gate.py content            # CPU, plus GPU if present
    python _tools/nb_run_gate.py --cpu-only content
    python _tools/nb_run_gate.py --list content     # discovery dry-run: every notebook with its gate class

The discovery rule (``discover``) and the gate reading (``skip_reason``, ``truncate_at_gpu_cell``) are shared with
``nb_hosted.py`` / ``nb_hosted_run.py`` and Kaggle's ``kernel.py``: one rule, no hand lists.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback

TIMEOUT_S = 1800
SKIP_PARTS = frozenset({"_build", ".ipynb_checkpoints", "jupyter_execute", ".git"})


def discover(roots: dict[str, pathlib.Path]) -> dict[str, list[pathlib.Path]]:
    """Every notebook of each checkout, by glob: the site's ``notebooks/*.ipynb`` (repo name ``site``), any other repo's
    ``docs/content/**/*.ipynb``. Build output and checkpoint folders are skipped. A new tutorial is covered the day it
    is committed."""
    found = {}
    for repo, root in roots.items():
        base = root / "notebooks" if repo == "site" else root / "docs" / "content"
        pattern = "*.ipynb" if repo == "site" else "**/*.ipynb"
        found[repo] = sorted(p for p in base.glob(pattern) if not SKIP_PARTS & set(p.relative_to(base).parts))
    return found


def gate_class(path: pathlib.Path) -> str:
    """The gate a notebook declares, as one word: ``gpu_only``, ``needs:<mods>``, ``cell-stop`` (a GPU-only section
    at the end), joined with ``+``; ``plain`` when it declares none. Reads the JSON only (no nbformat needed)."""
    nb = json.loads(path.read_text())
    gate = nb.get("metadata", {}).get("gate", {})
    parts = (["gpu_only"] if gate.get("gpu_only") else []) + ([f"needs:{','.join(gate['needs'])}"] if gate.get("needs") else [])
    if any(c.get("metadata", {}).get("gate", {}).get("gpu_only") for c in nb.get("cells", [])):
        parts.append("cell-stop")
    return "+".join(parts) or "plain"


def list_notebooks(found: dict[str, list[pathlib.Path]]) -> int:
    """Print every notebook with its gate class and return how many there are."""
    total = 0
    for repo, nbs in found.items():
        for p in nbs:
            print(f"{repo:<8} {gate_class(p):<18} {p}")
        total += len(nbs)
    print(f"[nb_run_gate] {total} notebooks: " + ", ".join(f"{r} {len(n)}" for r, n in found.items()))
    return total


def truncate_at_gpu_cell(nb) -> bool:
    """CPU leg: cut the notebook before its first GPU-only cell (a CPU tutorial may end with a GPU-only section)."""
    for i, cell in enumerate(nb.cells):
        if cell.get("metadata", {}).get("gate", {}).get("gpu_only"):
            nb.cells = nb.cells[:i]
            return True
    return False


def gpu_present() -> bool:
    """True when a CUDA GPU is visible to a fresh process (not just a driver library)."""
    if os.environ.get("EAGLE_NB_GATE_NO_GPU"):
        return False
    probe = "import cupy; raise SystemExit(0 if cupy.cuda.runtime.getDeviceCount() > 0 else 1)"
    try:
        return subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def register_kernel(root: pathlib.Path) -> str:
    """A kernelspec that starts THIS interpreter, so a notebook runs against the package under test
    (a stray 'python3' kernelspec on the machine may point at another environment)."""
    spec = root / "kernels" / "eagle-nb-gate"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "eagle notebook gate", "language": "python"}))
    os.environ["JUPYTER_PATH"] = str(root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    return "eagle-nb-gate"


def skip_reason(nb, leg: str) -> str | None:
    gate = nb["metadata"].get("gate", {})
    if leg == "cpu" and gate.get("gpu_only"):
        return f"needs a CUDA GPU ({gate['gpu_only']})"
    for mod in gate.get("needs", []):
        if importlib.util.find_spec(mod) is None:
            return f"needs the '{mod}' package, which is not installed"
    return None


def run_leg(path: pathlib.Path, leg: str, kernel: str) -> str:
    """Return 'OK', 'SKIP: <reason>' or 'FAIL'."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    reason = skip_reason(nb, leg)
    if reason:
        return f"SKIP: {reason}"
    if leg == "cpu":
        truncate_at_gpu_cell(nb)
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    if leg == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        NotebookClient(
            nb,
            timeout=TIMEOUT_S,
            kernel_name=kernel,
            resources={"metadata": {"path": str(path.parent)}},
            allow_errors=False,
        ).execute()
    except Exception:  # noqa: BLE001 - report and continue with the rest
        traceback.print_exc(limit=1)
        return "FAIL"
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return "OK"


def main(argv: list[str]) -> int:
    args = argv[1:]
    cpu_only = "--cpu-only" in args
    roots = [pathlib.Path(a) for a in args if not a.startswith("--")] or [pathlib.Path(".")]
    notebooks = sorted(
        p for r in roots
        for p in ([r] if r.suffix == ".ipynb" else r.rglob("*.ipynb"))
        if not SKIP_PARTS & set(p.parts)
    )
    if not notebooks:
        print("nb_run_gate: no notebooks found under", [str(r) for r in roots])
        return 2
    if "--list" in args:
        list_notebooks({"paths": notebooks})
        return 0
    legs = ["cpu"] + ([] if cpu_only or not gpu_present() else ["gpu"])
    if "gpu" not in legs:
        print("[nb_run_gate] no CUDA GPU visible (or --cpu-only): the GPU leg is not run")
    failed, ran, skipped = [], 0, 0
    scratch = tempfile.TemporaryDirectory(prefix="eagle_nb_gate_")
    kernel = register_kernel(pathlib.Path(scratch.name))
    for leg in legs:
        for p in notebooks:
            result = run_leg(p, leg, kernel)
            print(f"[nb_run_gate] {leg} {result.split(':')[0]:<4} {p}" + (result[4:] if result.startswith("SKIP") else ""),
                  flush=True)
            if result == "FAIL":
                failed.append((leg, p))
            elif result == "OK":
                ran += 1
            else:
                skipped += 1
    print(f"[nb_run_gate] {ran} ran, {skipped} skipped, {len(failed)} failed ({'+'.join(legs)})")
    if failed:
        print("[nb_run_gate] FAILED:", *[f"{leg} {p}" for leg, p in failed], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
