#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Fail if the publish or docs-deploy workflow builds, tests or installs source.

Those workflows may only fetch, verify and upload what release.yml built on
the pull request. Usage: ci_publish_subset.py [workflow.yml ...]; the default
is publish.yml and docs.yml next to this repo's tools/ directory.
"""
import re
import sys
from pathlib import Path

import yaml

FORBIDDEN = {
    "python -m build": re.compile(r"python\d*\s+-m\s+build\b"),
    "pip install of the source tree": re.compile(r"pip\s+install\b[^\n]*(\s-e\b|\s\.(\s|$|\[))"),
    "pytest": re.compile(r"\bpytest\b"),
    "make": re.compile(r"(^|[\s;&|])make\b"),
    "sphinx": re.compile(r"\bsphinx\b", re.IGNORECASE),
}


def violations(path):
    """Return 'file: job/step: reason' lines for every forbidden step."""
    out = []
    doc = yaml.safe_load(Path(path).read_text())
    for job, body in (doc.get("jobs") or {}).items():
        for i, step in enumerate(body.get("steps") or []):
            where = f"{Path(path).name}: {job}/{step.get('name') or step.get('id') or i}"
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                out.append(f"{where}: setup-python (nothing here needs a toolchain)")
            for label, rx in FORBIDDEN.items():
                if rx.search(step.get("run", "")):
                    out.append(f"{where}: {label}")
    return out


def main(argv):
    root = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    files = argv or [root / "publish.yml", root / "docs.yml"]
    bad = [v for f in files for v in violations(f)]
    for v in bad:
        print(f"FORBIDDEN in publish/deploy: {v}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
