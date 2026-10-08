#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Gate: every notebook under the given directories must be EXECUTED - each
code cell with non-blank source carries an execution count. The docs publish
runs this on the notebooks `make nbexec` filled, before rendering them with
Sphinx execution switched off, so an unfilled notebook can never reach the
published pages silently (nor trigger a second, hidden execution).

Also refuses a notebook whose OUTPUT (stream text, `text/plain`,
`text/html`, or an error traceback) contains an absolute local-machine
path. eagle's docs publish to a public site; a committed output with
`/local/...`, `/home/<user>/...` etc. leaks the machine that built
them. Only rendered OUTPUT is scanned, not cell source.
"""
import pathlib
import re
import sys

import nbformat

# Deliberately narrow: absolute local-filesystem roots a committed output
# should never contain, not a general "looks like a path" heuristic.
_LEAK_PATTERNS = [re.compile(p) for p in (r"/local/", r"/home/", r"/tmp/", r"/root/", r"/Users/", r"C:\\Users")]


def _text_of(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def _leaked_path(text: str) -> str | None:
    for pat in _LEAK_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


def _output_leaks(cell) -> list[str]:
    leaks = []
    for out in cell.get("outputs", []):
        texts = []
        if out.get("output_type") == "stream":
            texts.append(_text_of(out.get("text", "")))
        elif out.get("output_type") in ("display_data", "execute_result"):
            data = out.get("data", {})
            texts.extend(_text_of(data[mime]) for mime in ("text/plain", "text/html") if mime in data)
        elif out.get("output_type") == "error":
            texts.append(_text_of(out.get("traceback", [])))
        for text in texts:
            pattern = _leaked_path(text)
            if pattern:
                leaks.append(pattern)
    return leaks


def main(argv: list[str]) -> int:
    roots = [pathlib.Path(a) for a in argv[1:]] or [pathlib.Path(".")]
    notebooks = sorted(
        p for r in roots
        for p in ([r] if r.suffix == ".ipynb" else r.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in p.parts
    )
    if not notebooks:
        print("nb_check_filled: no notebooks found under", [str(r) for r in roots])
        return 2
    bad = []
    leaked = []
    for p in notebooks:
        nb = nbformat.read(p, as_version=4)
        unexecuted = [
            i for i, c in enumerate(nb.cells)
            if c.cell_type == "code" and c.source.strip() and c.get("execution_count") is None
        ]
        if unexecuted:
            bad.append((p, unexecuted))
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            for pattern in _output_leaks(c):
                leaked.append((p, i, pattern))
    for p, cells in bad:
        print(f"[nb_check_filled] UNEXECUTED {p}: code cells {cells[:6]}"
              f"{'...' if len(cells) > 6 else ''}")
    for p, i, pattern in leaked:
        print(f"[nb_check_filled] LEAKED PATH {p}: code cell {i} output matched {pattern!r}")
    print(f"[nb_check_filled] {len(notebooks) - len(bad)}/{len(notebooks)} notebooks "
          "fully executed")
    return 1 if (bad or leaked) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
