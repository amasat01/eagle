#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Write (or check) the Colab setup cell at the top of every docs notebook.

A docs notebook opened from its Colab launch button runs alone: its packages are
not installed and the helper files it imports (``_shared/``, sibling example
modules, fixtures) are not next to it. This script gives each notebook one
generated cell, derived from the notebook's own imports, that on Colab only

* ``pip install``s the PyPI packages the notebook imports (the ``[cuda12]`` or ``[cuda13]`` extra
  only when ``nvidia-smi`` shows a GPU, matching the runtime's CUDA major), and
* downloads the repo files it needs from raw.githubusercontent.com into
  ``/content/<repo> (Kaggle: /kaggle/working/<repo>)/<same repo path>`` and changes into the notebook's directory,
  so every relative path in the notebook resolves exactly as in a checkout.

Two optional environment variables test a release before it ships: ``RAPTOR_PRE=1`` adds ``--pre`` to the pip
install, and ``RAPTOR_REF=<git ref>`` downloads the helper files from that ref instead of ``main``. Two more
point the cell at a local mirror (``make nbcheck-hosted`` uses them): ``RAPTOR_RAW_BASE`` replaces
``https://raw.githubusercontent.com`` and ``RAPTOR_WORKDIR`` replaces ``/content`` (Kaggle: ``/kaggle/working``).

Elsewhere the cell installs nothing: it only checks the hard imports and raises an
ImportError that names the pip line. The cell is generated; do not edit it by hand.

    python _tools/nb_colab_setup.py            # refresh every docs notebook
    python _tools/nb_colab_setup.py --check    # exit 1 if any is stale (no writes)
"""
import ast
import json
import pathlib
import sys
import tomllib

# ---- per-repo configuration (the only lines that differ between the three repos) ----
REPO = "eagle"
GITHUB = "amasat01"
CONTENT = "docs/content"
#: repo-relative directories whose modules a notebook may import as helpers
HELPER_DIRS = ("docs/content/_shared", "examples/python")
#: third-party packages a helper needs beyond its module-level imports (it imports them inside functions)
HELPER_NEEDS = {"docs/content/_shared/nb_helpers.py": ("matplotlib",), "examples/python/zero_copy_interop.py": ("torch",)}
#: a notebook that mentions this directory name gets the whole directory downloaded
FIXTURE_DIRNAME = "_fixtures"
# ------------------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
#: import name -> PyPI distribution
PYPI = {
    "eagle": "raptor-eagle",
    "hawk": "raptor-hawk",
    "raptor": "raptor-core",
    "torch": "torch",
    "matplotlib": "matplotlib",
    "warp": "warp-lang",
    "cupy": "raptor-eagle",  # the [cuda12] / [cuda13] extra brings cupy
}


def _jax_floor() -> str:
    """The JAX release eagle needs, read from the ``jax`` extra in ``python/pyproject.toml``."""
    extra = tomllib.loads((ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]["jax"]
    return next(r.split(">=")[1] for r in extra if r.startswith("jax>="))


#: distributions that take the GPU extra when a GPU is visible
GPU_EXTRA = ("raptor-eagle", "raptor-hawk")
#: preinstalled on Colab and not listed
PREINSTALLED = frozenset({"numpy", "IPython", "jax"})
#: imports the cell checks off Colab when a notebook imports them at the top level of a cell
HARD = ("eagle", "hawk", "raptor", "torch", "matplotlib")
_FALLBACK_STDLIB = frozenset(
    "__future__ argparse collections ctypes importlib inspect io json math os pathlib re shutil statistics "
    "subprocess sys tempfile threading time types warnings".split()
)
STDLIB = getattr(sys, "stdlib_module_names", _FALLBACK_STDLIB)
TAG = "colab-setup"
NOTE_ID, CELL_ID = "colabnote", "colabsetup"
NOTE = (
    "**Colab or Kaggle:** run this first cell (it installs the packages and downloads the helper files this notebook "
    "uses). Elsewhere it only checks the imports."
)


def _clean(source: str) -> str:
    """Drop IPython magic lines so the cell parses as plain Python."""
    return "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in source.splitlines())


def _outside_try(node):
    """Every node under ``node`` that is not inside a try block (an optional import)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Try):
            continue
        yield child
        yield from _outside_try(child)


def _imports(source: str, mode: str) -> set[str]:
    """Imported top-level module names: ``top`` (cell level), ``all`` (anywhere), ``required`` (not under try)."""
    try:
        tree = ast.parse(_clean(source))
    except SyntaxError:
        return set()
    nodes = {"top": lambda: tree.body, "all": lambda: ast.walk(tree), "required": lambda: _outside_try(tree)}[mode]()
    found = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def code_sources(nb) -> list[str]:
    return [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and TAG not in c.get("metadata", {}).get("tags", [])
    ]


def helper_modules() -> dict[str, str]:
    """Importable helper module name -> repo-relative path."""
    found = {}
    for d in HELPER_DIRS:
        for p in sorted((ROOT / d).glob("*.py")):
            found[p.stem] = f"{d}/{p.name}"
    return found


def plan(nb, nb_path: pathlib.Path) -> dict:
    """Everything the generated cell must do for one notebook."""
    sources = code_sources(nb)
    anywhere = set().union(*(_imports(s, 'all') for s in sources)) if sources else set()
    top = set().union(*(_imports(s, 'top') for s in sources)) if sources else set()
    helpers = helper_modules()
    files = [helpers[m] for m in sorted(anywhere & set(helpers))]
    # a helper's own unconditional imports are the notebook's too
    for rel in list(files):
        anywhere |= _imports((ROOT / rel).read_text(encoding="utf-8"), "top") | set(HELPER_NEEDS.get(rel, ()))
    text = "\n".join(sources)
    if FIXTURE_DIRNAME in text:
        fixture_dirs = sorted(d for d in (ROOT / CONTENT).rglob(FIXTURE_DIRNAME) if d.is_dir())
        for d in fixture_dirs:
            files += sorted(p.relative_to(ROOT).as_posix() for p in d.iterdir() if p.is_file())
    third_party = sorted(m for m in anywhere if m not in STDLIB and m not in PREINSTALLED and m not in helpers)
    unknown = [m for m in third_party if m not in PYPI]
    if unknown:
        raise SystemExit(f"{nb_path}: imports {unknown} with no PyPI name in {pathlib.Path(__file__).name} PYPI")
    pkgs = sorted({PYPI[m] for m in third_party})
    return {
        "jax": _jax_floor() if "jax" in anywhere else None,
        "pkgs": pkgs,
        "files": files,
        "hard": [m for m in HARD if m in top],
        "nb_dir": nb_path.parent.relative_to(ROOT).as_posix(),
    }


def render(p: dict) -> str:
    def items(names):
        return ", ".join(f'f"{x}{{extra}}"' if x in GPU_EXTRA else f'"{x}"' for x in names)

    ours = [x for x in p["pkgs"] if x.startswith(("raptor-", "aether-"))]
    others = [x for x in p["pkgs"] if x not in ours]
    files = "".join(f'        "{f}",\n' for f in p["files"])
    hard = ", ".join(p["hard"])
    plain = " ".join(p["pkgs"])
    gpu_names = " and ".join(x for x in p["pkgs"] if x in GPU_EXTRA)
    gpu_hint = f" For a GPU, add [cuda12] (or [cuda13]) to {gpu_names}." if gpu_names else ""
    lines = [
        f"# Colab setup, generated by docs/{pathlib.Path(__file__).parent.name}/nb_colab_setup.py (do not edit).",
        "import os",
        "",
        "try:",
        "    import google.colab  # noqa: F401",
        "    HOSTED = True  # a Colab runtime",
        "except ImportError:",
        '    HOSTED = "KAGGLE_KERNEL_RUN_TYPE" in os.environ  # a Kaggle kernel',
        "",
        "if HOSTED:",
        "    import importlib.metadata, pathlib, re, shutil, subprocess, sys, urllib.request",
        "",
        '    gpu = bool(shutil.which("nvidia-smi")) and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0',
        '    extra = ""',
        "    if gpu:",
        "        # the CUDA major the runtime already has: a CuPy that is installed, else torch's, else the driver's",
        "        major = None",
        '        for m in ("13", "12"):',
        "            try:",
        '                importlib.metadata.version(f"cupy-cuda{m}x")',
        "                major = m",
        "                break",
        "            except importlib.metadata.PackageNotFoundError:",
        "                pass",
        "        if major is None:",
        "            try:",
        "                import torch",
        '                major = "13" if str(torch.version.cuda).startswith("13") else None',
        "            except ImportError:",
        "                pass",
        "        if major is None:",
        '            smi = re.search(r"CUDA Version: (\\d+)", subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)',
        '            major = "13" if smi and int(smi.group(1)) >= 13 else "12"',
        '        extra = f"[cuda{major}]"',
        '        print(f"GPU found: installing the {extra} extras to match this runtime.")',
        '    pre = ["--pre"] if os.environ.get("RAPTOR_PRE") else []  # optional: test a release candidate',
        '    ref = os.environ.get("RAPTOR_REF", "main")  # optional: fetch the helper files from this git ref',
        '    if pre or "RAPTOR_REF" in os.environ:',
        '        print(f"Pre-release test: installing with --pre, helper files from {ref}.")',
        "    # our packages with --upgrade, so a runtime that kept an older release gets the current one",
        f'    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pre, {items(ours)}])',
    ]
    if p["jax"]:
        # Colab keeps its own jax: raise it when it is below the floor eagle's JAX bridge needs
        lines += [
            "    from packaging.version import Version",
            "",
            "    try:",
            f'        old = Version(importlib.metadata.version("jax")) < Version("{p["jax"]}")',
            "    except importlib.metadata.PackageNotFoundError:",
            "        old = True",
            "    if old:",
            f'        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"jax{{extra}}>={p["jax"]}"])',
        ]
    if others:
        lines.append(f'    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", {items(others)}])')
    if p["files"]:
        lines += [
            '    base = os.environ.get("RAPTOR_RAW_BASE", "https://raw.githubusercontent.com")  # optional: a local mirror',
            '    top = os.environ.get("RAPTOR_WORKDIR") or ("/kaggle/working" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "/content")',
            f'    root = pathlib.Path(top) / "{REPO}"',
            "    for rel in (",
            files.rstrip("\n"),
            "    ):",
            "        (root / rel).parent.mkdir(parents=True, exist_ok=True)",
            f'        urllib.request.urlretrieve(f"{{base}}/{GITHUB}/{REPO}/{{ref}}/{{rel}}", root / rel)',
            f'    (root / "{p["nb_dir"]}").mkdir(parents=True, exist_ok=True)',
            f'    os.chdir(root / "{p["nb_dir"]}")',
        ]
    lines += [
        "    if not gpu:",
        '        print("Running on the CPU (no GPU in this runtime). For a GPU: " + ("Settings \u2192 Accelerator \u2192 GPU T4 x2." if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "Runtime \u2192 Change runtime type \u2192 T4 GPU."))',
    ]
    if p["hard"]:
        lines += [
            "",
            "try:",
            f"    import {hard}  # noqa: F401",
            "except ImportError as err:",
            "    raise ImportError(",
            f'        "{hard} not found. Install with `pip install {plain}`.{gpu_hint}"',
            '    ) from err',
        ]
    return "\n".join(lines)


def _cell(kind: str, source: str, cell_id: str) -> dict:
    tags = [TAG, "hide-input"] if kind == "code" else [TAG]
    cell = {"cell_type": kind, "id": cell_id, "metadata": {"tags": tags}}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cell["source"] = source.splitlines(keepends=True)
    return cell


def apply(nb, nb_path: pathlib.Path) -> bool:
    """Insert/refresh the two setup cells; True if the notebook changed."""
    p = plan(nb, nb_path)
    want_code = render(p)
    cells = nb["cells"]
    idx = [i for i, c in enumerate(cells) if TAG in c.get("metadata", {}).get("tags", [])]
    if idx:
        have = "".join(cells[idx[-1]]["source"])
        if have == want_code and len(idx) == 2:
            return False
        for i in reversed(idx):
            del cells[i]
    at = 1 if cells and cells[0]["cell_type"] == "markdown" else 0
    cells[at:at] = [_cell("markdown", NOTE, NOTE_ID), _cell("code", want_code, CELL_ID)]
    return True


def notebooks(roots) -> list[pathlib.Path]:
    out = []
    for r in roots:
        r = pathlib.Path(r)
        out += [r] if r.suffix == ".ipynb" else sorted(r.rglob("*.ipynb"))
    return [p for p in out if ".ipynb_checkpoints" not in p.parts]


def main(argv: list[str]) -> int:
    check = "--check" in argv
    roots = [a for a in argv[1:] if not a.startswith("--")] or [ROOT / CONTENT]
    stale = []
    for path in notebooks(roots):
        path = path.resolve()
        nb = json.loads(path.read_text(encoding="utf-8"))
        if apply(nb, path):
            stale.append(path)
            if not check:
                path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in stale:
        print(("STALE " if check else "updated ") + str(p.relative_to(ROOT)))
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
