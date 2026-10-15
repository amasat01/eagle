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
   setup cell takes its hosted branch. The cell's ``pip install`` resolves to the wheelhouse (PIP_NO_INDEX,
   PIP_FIND_LINKS) and its helper downloads come from a local HTTP mirror of the trees (RAPTOR_RAW_BASE,
   RAPTOR_REF, RAPTOR_WORKDIR).

Needs ``uv``, a Python interpreter of the lock's minor version, a C++23 compiler (set CXX) and network access to the
package indexes. The notebooks of ``--repos`` (default: all three) run; one PASS / FAIL / SKIP line each, and exit 1
on any FAIL.

    python nb_hosted.py --python /path/to/python3.13 --lock colab --eagle E --hawk H --raptor R --aether A
"""
import argparse
import functools
import http.server
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
LOCKS = {"colab": HERE / "hosted_env" / "colab-cpu.txt", "floor": HERE / "hosted_env" / "floor-cpu.txt",
         "colab-gpu": HERE / "hosted_env" / "colab-gpu.txt"}
IGNORE = shutil.ignore_patterns(".git", "build", "_build", "_doxybuild", "__pycache__", "*.so", "*.egg-info",
                                ".ipynb_checkpoints", "_sdist_include", "__install_p1", "_payload")
EMPTY_OPENSSL_CONF = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "nb_hosted_empty_openssl.cnf"
EMPTY_OPENSSL_CONF.touch()
FAKE_COLAB = {
    "google/__init__.py": "",
    "google/colab/__init__.py": "# stands in for the Colab runtime module: its presence selects the hosted branch\n",
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


def execute(args, roots, work):
    lock = LOCKS.get(args.lock) or pathlib.Path(args.lock).resolve()
    if not lock.is_file():
        raise SystemExit(f"nb_hosted: --lock must be one of {sorted(LOCKS)} or an existing requirements file, got {args.lock}")
    lock_name = lock.stem if args.lock not in LOCKS else args.lock
    wh = build_wheels(args.python, args.uv, roots, work, args.rebuild,
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
    sh([args.uv, "pip", "install", "--python", vpy, "--no-index", "--find-links", wh, "aether-dsc", "raptor-core",
        "raptor-eagle", "raptor-hawk"], env={**os.environ, "UV_CACHE_DIR": str(work / "uvcache")})
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
    env = clean_env(PATH=f"{venv / 'bin'}:/usr/bin:/bin", PYTHONPATH=fake, PIP_NO_INDEX="1", PIP_FIND_LINKS=wh,
                    PIP_DISABLE_PIP_VERSION_CHECK="1", CUDA_VISIBLE_DEVICES=args.gpu_index if args.gpu else "",
                    CUDA_DEVICE_ORDER="PCI_BUS_ID", NB_HOSTED_GPU="1" if args.gpu else "", NB_HOSTED_FLOCK=args.gpu_lock or "",
                    RAPTOR_RAW_BASE=f"http://127.0.0.1:{server.server_address[1]}", RAPTOR_REF="local")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    t0 = time.time()
    for repo in args.repos.split(","):
        content = roots[repo] / "docs" / "content"
        nbs = sorted(p for p in content.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts and "_build" not in p.parts
                     and args.only in p.as_posix())
        if not nbs:
            continue
        print(f"[nb_hosted] {repo}: {len(nbs)} notebooks ({args.lock} lock)", flush=True)
        out = subprocess.run([str(vpy), str(HERE / "nb_hosted_run.py"), str(work / "kernels"), *map(str, nbs)], env=env,
                             capture_output=True, text=True)
        (work / f"run_{repo}_{lock_name}.log").write_text(out.stdout + out.stderr)
        got = 0
        for line in out.stdout.splitlines():
            if line.startswith("RESULT\t"):
                _, status, path, msg = (line.split("\t") + [""])[:4]
                counts[status] += 1
                got += 1
                print(f"{status:<4} {repo}/{pathlib.Path(path).relative_to(content)}" + (f"  {msg}" if msg else ""), flush=True)
        if got != len(nbs):
            counts["FAIL"] += len(nbs) - got
            print(f"FAIL {repo}: the runner returned {got} of {len(nbs)} results (rc {out.returncode}); see "
                  f"{work / f'run_{repo}_{lock_name}.log'}\n" + "\n".join(out.stderr.splitlines()[-8:]))
    server.shutdown()
    if Mirror.missing:
        print("[nb_hosted] helper files the setup cells asked for and the mirror did not have:", *sorted(set(Mirror.missing)), sep="\n  ")
    print(f"[nb_hosted] {args.lock} lock: {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped "
          f"({time.time() - t0:.0f} s)")
    if not args.keep:
        shutil.rmtree(venv, ignore_errors=True)
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)
    return 1 if counts["FAIL"] else 0



def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="interpreter of the lock's minor version (e.g. python3.13)")
    ap.add_argument("--lock", default="colab", help="colab, floor, or the path of a requirements file")
    ap.add_argument("--repos", default="eagle,hawk,raptor", help="comma list of repos whose notebooks run")
    for name in ("aether", "eagle", "hawk", "raptor"):
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
    if not args.uv:
        raise SystemExit("nb_hosted: uv not found; `python -m venv v && v/bin/pip install uv`, then --uv v/bin/uv")
    # a checkout not given is looked for next to this one (this script lives in <eagle>/docs/_tools)
    here_eagle = HERE.parents[1]
    for n in ("aether", "eagle", "hawk", "raptor"):
        if not getattr(args, n):
            guess = here_eagle if n == "eagle" else here_eagle.parent / n
            setattr(args, n, str(guess) if guess.is_dir() else None)
    roots = {n: pathlib.Path(getattr(args, n)).resolve() for n in ("aether", "eagle", "hawk", "raptor") if getattr(args, n)}
    missing = [n for n in ("aether", "eagle", "hawk", "raptor") if n not in roots]
    if missing:
        raise SystemExit(f"nb_hosted: give the checkouts of {missing} (--<name>, $<NAME>_ROOT, or a sibling directory): the wheels are built from all four")
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
