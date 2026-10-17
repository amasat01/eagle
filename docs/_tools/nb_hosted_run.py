#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Runs inside the hosted venv (started by nb_hosted.py): execute each notebook from a COPY in an empty directory.

Prints one line per notebook: ``RESULT<TAB>PASS|FAIL|SKIP<TAB><path><TAB><first error line or skip reason>``.
Honours the same ``metadata.gate`` keys as nb_run_gate.py (``gpu_only``, ``needs``, per-cell stop) through its code;
``NB_HOSTED_GPU=1`` marks a GPU runtime. On a FAIL the whole error goes to stderr after a ``--- <path>`` line.
"""
import contextlib
import fcntl
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

import nb_run_gate  # the gate reading is shared with the in-place gate: one rule

TIMEOUT_S = int(os.environ.get("NB_HOSTED_CELL_TIMEOUT") or 1800)  # seconds per cell
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def first_error(exc: BaseException) -> str:
    lines = [ANSI.sub("", ln).strip() for ln in str(exc).splitlines() if ANSI.sub("", ln).strip()]
    # nbclient's message ends with the cell's traceback; its last line is "ErrorName: message"
    for ln in reversed(lines):
        if re.match(r"[A-Za-z_][\w.]*(Error|Exception|Exit|Warning|Interrupt)\b", ln):
            return ln[:300]
    return (lines[-1] if lines else type(exc).__name__)[:300]


@contextlib.contextmanager
def locked():
    """Hold NB_HOSTED_FLOCK (when set) while a notebook runs: another job shares the GPU."""
    name = os.environ.get("NB_HOSTED_FLOCK")
    if not name:
        yield
        return
    with open(name, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def run(path: pathlib.Path, kernel: str) -> tuple[str, str]:
    nb = nbformat.read(path, as_version=4)
    gpu = bool(os.environ.get("NB_HOSTED_GPU"))  # a GPU runtime: GPU-only notebooks run in full
    reason = nb_run_gate.skip_reason(nb, "gpu" if gpu else "cpu")
    if reason:
        return "SKIP", reason
    if not gpu:
        nb_run_gate.truncate_at_gpu_cell(nb)
    with tempfile.TemporaryDirectory(prefix="nb_hosted_") as tmp:
        tmp = pathlib.Path(tmp)
        work = tmp / "nb"
        work.mkdir()
        copy = work / path.name
        shutil.copyfile(path, copy)
        env_before = dict(os.environ)
        os.environ["RAPTOR_WORKDIR"] = str(tmp / "content")
        try:
            with locked():
                NotebookClient(nb, timeout=TIMEOUT_S, kernel_name=kernel, resources={"metadata": {"path": str(work)}},
                               allow_errors=False).execute()
        except BaseException as exc:  # noqa: BLE001
            print(f"--- {path}\n{ANSI.sub('', str(exc))}", file=sys.stderr, flush=True)
            return "FAIL", first_error(exc)
        finally:
            os.environ.clear()
            os.environ.update(env_before)
    # which branch the generated setup cell took: its own printed line
    for cell in nb.cells:
        if "colab-setup" in cell.get("metadata", {}).get("tags", []) and cell.cell_type == "code":
            text = "".join(o.get("text", "") for o in cell.get("outputs", []))
            line = next((ln for ln in text.splitlines() if ln.startswith(("GPU found", "Running on the CPU"))), "")
            return "PASS", line[:120]
    return "PASS", ""


def main(argv: list[str]) -> int:
    kernel_dir, notebooks = argv[1], argv[2:]
    spec = pathlib.Path(kernel_dir) / "kernels" / "hosted"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "hosted", "language": "python"}))
    os.environ["JUPYTER_PATH"] = kernel_dir
    for nbp in notebooks:
        status, msg = run(pathlib.Path(nbp), "hosted")
        print(f"RESULT\t{status}\t{nbp}\t{msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
