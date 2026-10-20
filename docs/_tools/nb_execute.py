#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Execute every notebook under the given directories IN PLACE (outputs kept).

`make nbexec` runs this on GPU 1 before a docs publish: the execution itself is
the gate (any cell error fails the run, all notebooks are still attempted so
the report is complete), and the filled notebooks are what gets committed —
`make nbcheck` then refuses to proceed if any committed notebook carries an
unexecuted cell, and CI's `conf.py` renders with `nb_execution_mode = "off"`
so the pages build never re-executes them. Uses nbclient (a myst_nb
dependency): no extra tool.
"""
import pathlib
import re
import sys
import traceback

import nbformat
from nbclient import NotebookClient

TIMEOUT_S = 3600


#: stderr text that is the machine's, not the page's: a local path or a library environment warning
_NOISE = re.compile(r"(?<![\w:/.])/(?:[\w.+-]+/)+[\w.+-]+|[A-Za-z]:\\|\w+Warning\b|warnings\.warn|Falling back to cpu")


def drop_machine_noise(nb) -> None:
    """Remove stderr stream outputs that name a local path or carry a library warning."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = [o for o in cell.outputs
                            if not (o.output_type == "stream" and o.name == "stderr"
                                    and _NOISE.search(o.text))]


def execute(path: pathlib.Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=TIMEOUT_S,
        kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except Exception:  # noqa: BLE001 - report and continue with the rest
        traceback.print_exc(limit=1)
        return False
    finally:
        drop_machine_noise(nb)
        nbformat.write(nb, path)  # keep whatever ran, even for a failed notebook
    return True


def main(argv: list[str]) -> int:
    roots = [pathlib.Path(a) for a in argv[1:]] or [pathlib.Path(".")]
    notebooks = sorted(
        p for r in roots
        for p in ([r] if r.suffix == ".ipynb" else r.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in p.parts
    )
    if not notebooks:
        print("nb_execute: no notebooks found under", [str(r) for r in roots])
        return 2
    failed = []
    for p in notebooks:
        ok = execute(p)
        print(f"[nb_execute] {'OK  ' if ok else 'FAIL'} {p}", flush=True)
        if not ok:
            failed.append(p)
    print(f"[nb_execute] {len(notebooks) - len(failed)}/{len(notebooks)} executed")
    if failed:
        print("[nb_execute] FAILED:", *[str(p) for p in failed], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
