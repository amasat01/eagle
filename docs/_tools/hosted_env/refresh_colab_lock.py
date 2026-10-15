#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Rewrite ``colab-cpu.txt`` (or, with ``--runtime gpu``, ``colab-gpu.txt``) from Google's published Colab package list.

The source is ``pip-freeze.txt`` / ``pip-freeze.gpu.txt`` (and ``os-info.txt`` for the Python version) in
https://github.com/googlecolab/backend-info, read at its latest commit. The lock keeps every pin as Colab has it,
filtered to what the docs notebooks and our packages touch (KEEP below). If the repository or the file cannot be
read, the script stops with an error and leaves the lock alone: it never invents a version.

    python refresh_colab_lock.py            # rewrite colab-cpu.txt next to this script
    python refresh_colab_lock.py --check    # exit 1 if the lock differs from the published list
    python refresh_colab_lock.py --runtime gpu   # the GPU runtime's lock, with CUDA 12 substitutions (below)

The GPU lock is for a host whose driver supports CUDA 12.x only: Colab's GPU runtime is CUDA 13, so torch becomes the
same version's cu126 build, cupy-cuda13x becomes cupy-cuda12x of the same version, and the CUDA 13 libraries the setup
cell's ``[cuda12]`` extras need become their CUDA 12.6 counterparts. Each substitution is listed in the lock header.
The CUDA 13 leg is covered by the Colab/Kaggle T4 release-candidate pass.
"""
import json
import pathlib
import re
import sys
import urllib.request

REPO = "googlecolab/backend-info"
RUNTIME = "gpu" if "--runtime" in sys.argv and "gpu" in sys.argv[sys.argv.index("--runtime") + 1:][:1] else "cpu"
LOCK = pathlib.Path(__file__).with_name(f"colab-{RUNTIME}.txt")
FREEZE = "pip-freeze.gpu.txt" if RUNTIME == "gpu" else "pip-freeze.txt"
#: GPU lock only: lines the CUDA 12 host needs in place of Colab's CUDA 13 ones (the extras the setup cell asks for)
NOT_IN_COLAB = ["warp-lang==1.16.0"]  # the newest CUDA 12 build with the notebooks' warp API (>=1.14); warp-lang 1.18 (latest) needs a CUDA 13 driver
CUDA12_EXTRA = [
    "cuda-bindings>=12.6,<13", "cuda-toolkit[cudart,nvrtc]==12.6.*", "nvidia-cuda-cccl-cu12>=12.6,<13",
    "nvidia-cuda-nvrtc-cu12==12.6.*",
]
#: distributions kept from the freeze (normalised names): the notebooks' imports, the notebook runner and their dependencies
KEEP = {
    "numpy", "torch", "matplotlib", "ipython", "ipykernel", "scipy", "jupyter-client", "jupyter-core", "nbclient",
    "nbformat", "pyzmq", "tornado", "traitlets", "pip", "setuptools", "wheel", "packaging", "contourpy", "cycler",
    "fonttools", "kiwisolver", "pillow", "pyparsing", "python-dateutil", "six", "filelock", "fsspec", "jinja2",
    "markupsafe", "networkx", "sympy", "mpmath", "typing-extensions", "jsonschema", "fastjsonschema", "platformdirs",
    "psutil", "nest-asyncio", "debugpy", "decorator", "jedi", "parso", "matplotlib-inline", "pexpect", "ptyprocess",
    "pickleshare", "prompt-toolkit", "pygments", "backcall", "wcwidth", "attrs", "referencing", "rpds-py",
    "jsonschema-specifications", "setuptools",
}
if RUNTIME == "gpu":
    KEEP |= {"cupy-cuda13x"}


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https hosts
        return r.read().decode("utf-8")


def build() -> str:
    try:
        commit = json.loads(get(f"https://api.github.com/repos/{REPO}/commits?per_page=1"))[0]
        sha, date = commit["sha"], commit["commit"]["committer"]["date"]
        freeze = get(f"https://raw.githubusercontent.com/{REPO}/{sha}/{FREEZE}")
        osinfo = get(f"https://raw.githubusercontent.com/{REPO}/{sha}/os-info.txt")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"refresh_colab_lock: cannot read {REPO}: {exc}") from exc
    py = re.search(r"^Python (\S+)", osinfo, re.M)
    if not py:
        raise SystemExit("refresh_colab_lock: no 'Python x.y.z' line in os-info.txt")
    lines, seen = [], set()
    subs = []
    for line in freeze.splitlines():
        m = re.match(r"([A-Za-z0-9_.-]+)\s*(==|@)", line)
        if m and re.sub(r"[-_.]+", "-", m.group(1)).lower() in KEEP:
            line = line.strip()
            if RUNTIME == "gpu" and line.startswith("torch"):
                ver = re.search(r"==\s*([0-9.]+)", line).group(1)
                new = (f"torch @ https://download.pytorch.org/whl/cu126/torch-{ver}%2Bcu126-cp313-cp313-"
                       "manylinux_2_28_x86_64.whl")
                subs.append(f"{line} -> the cu126 build of the same version")
                line = new
            elif line.startswith("cupy-cuda13x"):
                new = line.replace("cupy-cuda13x", "cupy-cuda12x")
                subs.append(f"{line} -> {new}")
                line = new
            lines.append(line)
            seen.add(re.sub(r"[-_.]+", "-", m.group(1)).lower())
    if RUNTIME == "gpu":
        lines += CUDA12_EXTRA + NOT_IN_COLAB
        subs.append("not in Colab's image, installed by the notebooks' setup cell from PyPI (the check runs offline; a CUDA 12 build, because the latest warp-lang needs a CUDA 13 driver): "
                    + ", ".join(NOT_IN_COLAB))
        subs.append("CUDA 13 cuda-bindings/cuda-toolkit/nvrtc/cccl -> " + ", ".join(CUDA12_EXTRA))
    for need in ("numpy", "torch", "matplotlib", "ipython", "ipykernel"):
        if need not in seen:
            raise SystemExit(f"refresh_colab_lock: {need} is not in Colab's pip-freeze.txt")
    head = [
        f"# The 'hosted' lock: Colab's {RUNTIME.upper()} runtime, for `make nbcheck-hosted{'-gpu' if RUNTIME == 'gpu' else ''}`. GENERATED by refresh_colab_lock.py: do not edit.",
        f"# Source: https://github.com/{REPO}/blob/{sha}/{FREEZE} (commit {sha[:12]}, {date}).",
        f"# Python: {py.group(1)} (os-info.txt at the same commit); run the driver with a Python of that minor.",
        "# Every pin is Colab's, filtered to what the docs notebooks and our packages touch. Refresh each release.",
    ]
    if subs:
        head += ["# CUDA 12 substitutions (this host's driver supports CUDA 12.6 at most):"] + [f"#   {x}" for x in subs]
        head += ["# CUDA 13 leg: covered by the Colab/Kaggle T4 release-candidate pass."]
    return "\n".join(head + sorted(lines, key=str.lower)) + "\n"


def main(argv: list[str]) -> int:
    text = build()
    if "--check" in argv:
        # the commit line moves with every upstream commit; compare the pins only
        pins = lambda t: [x for x in t.splitlines() if not x.startswith("#")]  # noqa: E731
        return 0 if LOCK.is_file() and pins(LOCK.read_text()) == pins(text) else 1
    LOCK.write_text(text)
    print(f"wrote {LOCK}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
