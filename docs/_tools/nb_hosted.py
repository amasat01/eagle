#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Run every docs notebook the way a hosted (Colab) user does: built wheels, pinned environment, setup cell included.

``make nbcheck`` runs the notebooks in the developer's environment against source checkouts. This driver instead

1. builds CPU-only wheels of aether-dsc (sealed first), raptor-core, raptor-eagle and raptor-hawk from the local trees
   (from their sdists, as PyPI users get them) into a wheelhouse,
2. creates a fresh venv from a pinned lock (``hosted_env/colab-cpu.txt`` or ``floor-cpu.txt``) and installs the wheels
   (no editable installs); with ``--gpu`` (``make nbcheck-hosted-gpu``) eagle's CUDA plugin is built too,
3. executes each notebook from a copy in an empty directory, with a fake ``google.colab`` module so the generated
   setup cell takes its hosted branch. The cell's ``pip install`` finds the wheelhouse (PIP_FIND_LINKS), pinned to
   the installed family versions (PIP_CONSTRAINT), and the package index for anything else (jax). Its helper downloads come from a local HTTP mirror of the
   trees (RAPTOR_RAW_BASE, RAPTOR_REF, RAPTOR_WORKDIR).

Needs ``uv``, a Python interpreter of the lock's minor version, a C++23 compiler (set CXX) and network access to the
package indexes. The notebooks of ``--repos`` run (default ``all``: eagle, hawk, raptor docs/content/**, and the
site's notebooks/*.ipynb, found by glob in ``nb_run_gate.discover``: nothing is listed by hand); one PASS / FAIL / SKIP
line each, a per-notebook timeout (``--timeout``), and exit 1 on any FAIL.

    python nb_hosted.py --python /path/to/python3.13 --lock colab --eagle E --hawk H --raptor R --aether A --site S

``--source pypi [--version X] [--pre]`` skips the wheel build and installs the PUBLISHED wheels instead (the check
run after Publish). ``--list`` only prints the notebooks with their gate class, and needs no Python, uv or build.
"""
import argparse
import functools
import json
import http.server
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb_run_gate  # noqa: E402  (discovery and gate classes: one rule, shared)

REPOS = ("site", "eagle", "hawk", "raptor", "aether")
FAMILY_DISTS = ("raptor-eagle", "raptor-hawk", "raptor-core", "aether-dsc")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOCKS = {"colab": HERE / "hosted_env" / "colab-cpu.txt", "floor": HERE / "hosted_env" / "floor-cpu.txt",
         "colab-gpu": HERE / "hosted_env" / "colab-gpu.txt"}
IGNORE = shutil.ignore_patterns(".git", "build", "_build", "_doxybuild", "__pycache__", "*.so", "*.egg-info",
                                ".ipynb_checkpoints", "_sdist_include", "__install_p1", "_payload")
EMPTY_OPENSSL_CONF = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "nb_hosted_empty_openssl.cnf"
EMPTY_OPENSSL_CONF.touch()
FAKE_COLAB = {
    "google/__init__.py": "",
    "google/colab/__init__.py": "# stands in for the Colab runtime module: its presence selects the hosted branch\n",
    # jax's debugger imports this submodule at import time (and calls nothing on it unless a debugger runs)
    "google/colab/output.py": "",
    # a Colab CPU runtime has no nvidia-smi; this box may. Hide it from the setup cell's shutil.which probe.
    "sitecustomize.py": (
        "import shutil\n_which = shutil.which\n"
        "shutil.which = lambda cmd, *a, **k: None if cmd == 'nvidia-smi' else _which(cmd, *a, **k)\n"
    ),
}


def sh(cmd, env=None, cwd=None, log=None):
    """Run a command, failing loudly with the tail of its output."""
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, capture_output=True, text=True)
    if log:
        pathlib.Path(log).write_text(r.stdout + r.stderr)
    if r.returncode:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return r.stdout


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONNOUSERSITE"] = "1"
    # a standalone (uv-installed) Python bundles its own OpenSSL, which can reject the host's system-wide
    # OpenSSL policy file (ERROR_IN_SYSTEM_DEFAULT_CONFIG); an empty one makes pip's TLS context load
    env["OPENSSL_CONF"] = str(EMPTY_OPENSSL_CONF)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def build_wheels(py, uv, roots, work, rebuild, gpu=None):
    """``gpu``: None for CPU-only wheels, else (nvcc, host compiler, CUDA arch) to build eagle's plugin as well."""
    wh = work / ("wheelhouse_gpu" if gpu else "wheelhouse")
    stamp = wh / ".built"
    if stamp.is_file() and not rebuild:
        print(f"[nb_hosted] reusing wheelhouse {wh}", flush=True)
        return wh
    shutil.rmtree(wh, ignore_errors=True)
    wh.mkdir(parents=True)
    ws = work / "ws"
    shutil.rmtree(ws, ignore_errors=True)
    for name, root in roots.items():  # siblings named as the repos are: the seal and the sdist hooks look for them
        shutil.copytree(root, ws / name, ignore=IGNORE)
    builder = work / "builder"
    shutil.rmtree(builder, ignore_errors=True)
    sh([uv, "venv", "--seed", "--python", py, builder])
    bpy = builder / "bin" / "python"
    sh([uv, "pip", "install", "--python", bpy, "build", "scikit-build-core>=0.10", "nanobind>=3.1,<4", "ninja", "cmake",
        "setuptools>=61", "wheel", "cuda-bindings>=12.6,<13", "nvidia-cuda-nvrtc-cu12>=12.6,<13",
        "nvidia-cuda-cccl-cu12>=12.6,<13"])
    env = clean_env(PATH=f"{builder / 'bin'}:{os.environ['PATH']}", CMAKE_BUILD_PARALLEL_LEVEL=4)
    # aether-dsc: seal this checkout's headers (the NVRTC-clean gate runs; no GPU needed), then wheel it
    print("[nb_hosted] sealing and building aether-dsc", flush=True)
    sh([bpy, "-m", "aether_dsc.seal"], env={**env, "PYTHONPATH": str(ws / "aether" / "dsc")}, cwd=ws / "aether" / "dsc")
    wheel = [bpy, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", wh]
    sh(wheel + [ws / "aether" / "dsc"], env=env)
    print("[nb_hosted] building raptor-core", flush=True)
    sh(wheel + [ws / "raptor"], env=env)
    sdists = work / "sdists"
    shutil.rmtree(sdists, ignore_errors=True)
    print("[nb_hosted] building raptor-eagle from its sdist", flush=True)
    sh([bpy, "-m", "build", "--sdist", "--no-isolation", "-o", sdists, ws / "eagle" / "python"],
       env={**env, "EAGLE_SDIST_AETHER_INCLUDE": str(ws / "aether")})
    if gpu:
        nvcc, host_cxx, arch = gpu
        eagle_env = {**env, "PATH": f"{pathlib.Path(nvcc).parent}:{env['PATH']}", "CUDACXX": nvcc, "CUDAHOSTCXX": host_cxx,
                     "SKBUILD_CMAKE_ARGS": f"-DEAGLE_PYTHON_CUDA_PLUGIN=BUILD;-DCMAKE_CUDA_ARCHITECTURES={arch}"}
        print(f"[nb_hosted] eagle: CUDA plugin for sm_{arch} (nvcc {nvcc})", flush=True)
    else:
        eagle_env = {**env, "SKBUILD_CMAKE_ARGS": "-DEAGLE_PYTHON_CUDA_PLUGIN=OFF"}
    t_eagle = time.time()
    sh(wheel + [next(sdists.glob("raptor_eagle-*.tar.gz"))], env=eagle_env, log=work / "build_eagle.log")
    print(f"[nb_hosted] raptor-eagle wheel built in {time.time() - t_eagle:.0f} s", flush=True)
    print("[nb_hosted] building raptor-hawk (from its sdist)", flush=True)
    sh([bpy, "-m", "build", "--sdist", "--no-isolation", "-o", sdists, ws / "hawk"],
       env={**env, "HAWK_AETHER_INCLUDE": str(ws / "aether"), "HAWK_EAGLE_INCLUDE": str(ws / "eagle")})
    sh(wheel + [next(sdists.glob("raptor_hawk-*.tar.gz"))], env=env, log=work / "build_hawk.log")
    stamp.write_text("\n".join(p.name for p in sorted(wh.glob("*.whl"))) + "\n")
    shutil.rmtree(ws, ignore_errors=True)
    shutil.rmtree(builder, ignore_errors=True)
    return wh


class Mirror(http.server.SimpleHTTPRequestHandler):
    missing: list = []

    def log_message(self, fmt, *args):
        if args and str(args[1]) == "404":
            Mirror.missing.append(self.path)


def start_mirror(roots, work):
    """http://127.0.0.1:PORT/amasat01/<repo>/local/<path> serves the checkout of <repo>."""
    top = work / "mirror"
    shutil.rmtree(top, ignore_errors=True)
    for name, root in roots.items():
        (top / "amasat01" / name).mkdir(parents=True)
        (top / "amasat01" / name / "local").symlink_to(root)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Mirror, directory=str(top)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def error_tail(stderr, n=14):
    """The last ``n`` lines of the failing cell's report (from the runner's ``--- <path>`` line on)."""
    lines = [ln for ln in ANSI.sub("", stderr).splitlines() if ln.strip()]
    starts = [i for i, ln in enumerate(lines) if ln.startswith("--- ")]
    return lines[starts[-1]:][-n:] if starts else lines[-n:]


def run_notebook(vpy, kernels, nb, env, timeout):
    """Run one notebook in its own runner process (nb_hosted_run.py). Returns (status, message, error_lines, seconds):
    status PASS / FAIL / SKIP; on a FAIL, error_lines is the tail of the failing cell's traceback. A notebook that
    outlives ``timeout`` seconds FAILs."""
    t0 = time.time()
    try:
        out = subprocess.run([str(vpy), str(HERE / "nb_hosted_run.py"), str(kernels), str(nb)], env=env,
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "FAIL", f"timed out after {timeout} s", [], round(time.time() - t0)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT\t"):
            _, status, _, msg = (line.split("\t") + [""])[:4]
            return status, msg, error_tail(out.stderr) if status == "FAIL" else [], round(time.time() - t0)
    return "FAIL", f"the runner returned no result (rc {out.returncode})", error_tail(out.stderr), round(time.time() - t0)


def family_constraints(vpy, path):
    """Pin the installed family versions in a pip constraints file (PIP_CONSTRAINT): a notebook's setup cell runs
    ``pip install --upgrade raptor-eagle ...`` and must not replace the build or release under test."""
    code = ("import importlib.metadata as m\nfor n in %r:\n    try: print(f'{n}=={m.version(n)}')\n"
            "    except m.PackageNotFoundError: pass" % (FAMILY_DISTS,))
    path.write_text(sh([vpy, "-c", code]))
    return path


def write_summary(title, rows, json_path=None):
    """rows: dicts with repo, name, status, seconds, message, error. A markdown table to $GITHUB_STEP_SUMMARY."""
    if json_path:
        pathlib.Path(json_path).write_text(json.dumps(rows, indent=2) + "\n")
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    lines = [f"### {title}", "", "| notebook | result | seconds | note |", "|---|---|---|---|"]
    lines += [f"| {r['repo']}/{r['name']} | {r['status']} | {r['seconds']} | {r['message']} |" for r in rows]
    for r in rows:
        if r["status"] == "FAIL" and r["error"]:
            lines += ["", f"<details><summary>{r['repo']}/{r['name']}</summary>", "", "```", *r["error"], "```", "</details>"]
    with open(target, "a") as f:
        f.write("\n".join(lines) + "\n")


def execute(args, roots, work):
    lock = LOCKS.get(args.lock) or pathlib.Path(args.lock).resolve()
    if not lock.is_file():
        raise SystemExit(f"nb_hosted: --lock must be one of {sorted(LOCKS)} or an existing requirements file, got {args.lock}")
    lock_name = lock.stem if args.lock not in LOCKS else args.lock
    pypi = args.source == "pypi"
    wh = None if pypi else build_wheels(args.python, args.uv, roots, work, args.rebuild,
                                        (args.nvcc, args.cuda_host_cxx, args.cuda_arch) if args.gpu else None)

    venv = work / f"venv_{lock_name}"
    shutil.rmtree(venv, ignore_errors=True)
    sh([args.uv, "venv", "--seed", "--python", args.python, venv])
    vpy = venv / "bin" / "python"
    print(f"[nb_hosted] installing the {args.lock} lock", flush=True)
    sh([args.uv, "pip", "install", "--python", vpy, "--index-strategy", "unsafe-best-match", "-r", lock],
       env={**os.environ, "UV_CACHE_DIR": str(work / "uvcache")})
    # the runner needs nbclient/nbformat; the Colab lock carries them, the floor lock leaves them free
    sh([args.uv, "pip", "install", "--python", vpy, "nbclient", "nbformat", "ipykernel"],
       env={**os.environ, "UV_CACHE_DIR": str(work / "uvcache")})
    uv_env = {**os.environ, "UV_CACHE_DIR": str(work / "uvcache")}
    if pypi:  # the published wheels, as a user's pip gets them
        eagle_req = "raptor-eagle" + (f"=={args.version}" if args.version else "")
        sh([args.uv, "pip", "install", "--python", vpy, "--index-strategy", "unsafe-best-match",
            *(["--prerelease", "allow"] if args.pre else []), eagle_req, "raptor-hawk"], env=uv_env)
    else:
        sh([args.uv, "pip", "install", "--python", vpy, "--no-index", "--find-links", wh, "aether-dsc", "raptor-core",
            "raptor-eagle", "raptor-hawk"], env=uv_env)
    print(sh([vpy, "-c", "import numpy, torch, eagle, hawk, raptor, aether_dsc, sys; "
              "print('[nb_hosted] python', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__, "
              "'eagle', eagle.__version__, 'hawk', hawk.__version__)"], env=clean_env(PATH=f"{venv / 'bin'}:/usr/bin:/bin")).strip())

    fake = work / "fake"
    for rel, text in FAKE_COLAB.items():
        if args.gpu and rel == "sitecustomize.py":
            continue  # a GPU runtime has nvidia-smi: the setup cell must see it
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text(text)
    server = start_mirror(roots, work)
    # The index stays open, as on Colab: a setup cell may install a third-party package (jax) the lock does not carry.
    # The family pins keep the build (or release) under test from being replaced; built mode also looks in the wheelhouse.
    index = {"PIP_CONSTRAINT": family_constraints(vpy, work / "family_constraints.txt"),
             **({"RAPTOR_PRE": "1"} if args.pre else {}), **({} if pypi else {"PIP_FIND_LINKS": wh})}
    env = clean_env(PATH=f"{venv / 'bin'}:/usr/bin:/bin", PYTHONPATH=fake, **index,
                    PIP_DISABLE_PIP_VERSION_CHECK="1", NB_HOSTED_CELL_TIMEOUT=args.timeout, CUDA_VISIBLE_DEVICES=args.gpu_index if args.gpu else "",
                    CUDA_DEVICE_ORDER="PCI_BUS_ID", NB_HOSTED_GPU="1" if args.gpu else "", NB_HOSTED_FLOCK=args.gpu_lock or "",
                    RAPTOR_RAW_BASE=f"http://127.0.0.1:{server.server_address[1]}", RAPTOR_REF="local")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    rows = []
    t0 = time.time()
    found = nb_run_gate.discover({r: roots[r] for r in repos(args)})
    for repo, all_nbs in found.items():
        content = roots[repo] / ("notebooks" if repo == "site" else "docs/content")
        nbs = [p for p in all_nbs if args.only in p.as_posix()]
        if not nbs:
            continue
        print(f"[nb_hosted] {repo}: {len(nbs)} notebooks ({args.lock} lock, {args.source})", flush=True)
        for nb in nbs:
            status, msg, tail, secs = run_notebook(vpy, work / "kernels", nb, env, args.timeout)
            counts[status] += 1
            name = str(nb.relative_to(content))
            rows.append({"repo": repo, "name": name, "status": status, "seconds": secs, "message": msg, "error": tail})
            print(f"{status:<4} {repo}/{name}  [{secs} s]" + (f"  {msg}" if msg else ""), flush=True)
            if status == "FAIL":
                (work / f"fail_{repo}_{name.replace('/', '_')}.log").write_text("\n".join(tail))
    server.shutdown()
    if Mirror.missing:
        print("[nb_hosted] helper files the setup cells asked for and the mirror did not have:", *sorted(set(Mirror.missing)), sep="\n  ")
    print(f"[nb_hosted] {args.lock} lock: {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped "
          f"({time.time() - t0:.0f} s)")
    write_summary(f"Notebooks on the CPU ({args.lock} lock, {args.source})", rows, args.json)
    if not args.keep:
        shutil.rmtree(venv, ignore_errors=True)
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)
    return 1 if counts["FAIL"] else 0



def repos(args):
    """The repos whose notebooks run: ``--repos`` is ``all`` or a comma list of site, eagle, hawk, raptor, aether."""
    # aether's notebooks are C++ lessons with no hosted setup cell: aether's own docs CI compiles and checks them
    names = [n for n in REPOS if n != "aether"] if args.repos == "all" else args.repos.split(",")
    bad = [n for n in names if n not in REPOS]
    if bad:
        raise SystemExit(f"nb_hosted: --repos takes 'all' or a list of {list(REPOS)}, not {bad}")
    return names


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", help="interpreter of the lock's minor version (e.g. python3.13); not needed with --list")
    ap.add_argument("--lock", default="colab", help="colab, floor, or the path of a requirements file")
    ap.add_argument("--repos", default="all", help="'all' or a comma list of site,eagle,hawk,raptor,aether whose notebooks run")
    ap.add_argument("--source", default="built", choices=("built", "pypi"),
                    help="built: wheels built from the checkouts (default); pypi: install the published wheels")
    ap.add_argument("--version", default="", help="--source pypi: the exact raptor-eagle version (hawk takes its newest)")
    ap.add_argument("--pre", action="store_true", help="--source pypi: allow pre-releases (also RAPTOR_PRE=1 for the setup cells)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds one notebook may run before it FAILs (default 900)")
    ap.add_argument("--list", action="store_true", help="print every notebook with its gate class and exit (no build, no run)")
    ap.add_argument("--json", default=None, help="also write the per-notebook results to this file")
    for name in ("site", "aether", "eagle", "hawk", "raptor"):
        ap.add_argument(f"--{name}", default=os.environ.get(f"{name.upper()}_ROOT"), help=f"{name} checkout (or $ {name.upper()}_ROOT)")
    ap.add_argument("--uv", default=shutil.which("uv"), help="the uv binary (pip install uv into any venv)")
    ap.add_argument("--work", default=None, help="work directory (default: a fresh temp dir, removed at the end)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the wheelhouse even if --work holds one")
    ap.add_argument("--gpu", action="store_true", help="GPU mode: eagle's CUDA plugin is built (sm_<--cuda-arch>), nothing "
                    "is hidden from the setup cell, GPU-only notebooks run on --gpu-index under --gpu-lock (use the colab-gpu lock)")
    ap.add_argument("--nvcc", default="/usr/local/cuda-12.6/bin/nvcc", help="GPU mode: the nvcc of the CUDA toolkit")
    ap.add_argument("--cuda-host-cxx", default="/usr/bin/g++", help="GPU mode: the host compiler nvcc uses")
    ap.add_argument("--cuda-arch", default="61", help="GPU mode: CMAKE_CUDA_ARCHITECTURES of the plugin (this GPU only)")
    ap.add_argument("--gpu-index", default="1", help="GPU mode: the one GPU the notebooks may see (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--gpu-lock", default=None, help="GPU mode: a lock file flock()ed around each notebook's execution")
    ap.add_argument("--only", default="", help="run only notebooks whose path contains this text")
    ap.add_argument("--cleanup", action="store_true", help="always delete the whole --work directory at the end, also on failure")
    ap.add_argument("--keep", action="store_true", help="keep the venv and the work directory")
    args = ap.parse_args(argv)
    # a checkout not given is looked for next to this one (this script lives in <eagle>/docs/_tools)
    here_eagle = HERE.parents[1]
    for n in REPOS:
        if not getattr(args, n):
            guesses = [here_eagle] if n == "eagle" else [here_eagle.parent / n] + ([here_eagle.parent / "amasat01.github.io"] if n == "site" else [])
            setattr(args, n, next((str(g) for g in guesses if g.is_dir()), None))
    roots = {n: pathlib.Path(getattr(args, n)).resolve() for n in REPOS if getattr(args, n)}
    wanted = repos(args)
    needed = set(wanted) | (set() if args.source == "pypi" or args.list else {"aether", "eagle", "hawk", "raptor"})
    missing = [n for n in REPOS if n in needed and n not in roots]
    if missing:
        raise SystemExit(f"nb_hosted: give the checkouts of {missing} (--<name>, $<NAME>_ROOT, or a sibling directory)")
    if args.list:
        return 0 if nb_run_gate.list_notebooks(nb_run_gate.discover({r: roots[r] for r in wanted})) else 2
    if not args.python:
        raise SystemExit("nb_hosted: --python is required (unless --list)")
    if not args.uv:
        raise SystemExit("nb_hosted: uv not found; `python -m venv v && v/bin/pip install uv`, then --uv v/bin/uv")
    if args.version and args.source != "pypi":
        raise SystemExit("nb_hosted: --version needs --source pypi")
    work = pathlib.Path(args.work).resolve() if args.work else pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"nb_hosted_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        return execute(args, roots, work)
    finally:
        if args.cleanup:
            shutil.rmtree(work, ignore_errors=True)
            print(f"[nb_hosted] work directory {work} {'is gone' if not work.exists() else 'COULD NOT BE REMOVED'}", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
