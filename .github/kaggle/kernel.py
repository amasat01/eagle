# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# The script Kaggle runs on a T4: it installs the family, checks that JAX sees the GPU, runs the
# interop certification rows plus the JAX rows, executes two notebooks, and writes result.json.
# drive.py pushes it with the settings below filled in; `--local` runs the same steps in the
# current environment (no install, no Kaggle paths).
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

# drive.py replaces the next line with the run settings.
EMBEDDED_CONFIG = None

ORG = "https://github.com/amasat01"
SITE_REPO = "amasat01.github.io"
START_HERE = "notebooks/raptor_start_here.ipynb"
QUICKSTART = "docs/content/userguide/quickstart_python.ipynb"
FAMILY = ("raptor-eagle", "raptor-hawk", "jax", "jaxlib", "cupy-cuda12x", "cupy-cuda13x", "torch")


def header(text):
    print("\n" + "=" * 72 + "\n== " + text + "\n" + "=" * 72, flush=True)


def sh(*cmd, cwd=None, env=None, capture=False):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=capture)
    return r if capture else r.returncode


def settings():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", action="store_true", help="use the current environment; no install, no Kaggle paths")
    ap.add_argument("--cuda", default="12", help="CUDA major of the extras: 12 or 13")
    ap.add_argument("--source", default="pypi", choices=("pypi", "git"))
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--version", default="")
    for name in ("eagle", "hawk", "raptor", "aether"):
        ap.add_argument(f"--{name}-ref", default="main")
    ap.add_argument("--eagle-src", help="local eagle checkout (default: the one holding this script)")
    ap.add_argument("--raptor-src", help="local raptor checkout, sets RAPTOR_ROOT")
    ap.add_argument("--site-src", help="local checkout of the site repository (for the start-here notebook)")
    ap.add_argument("--out", help="directory for result.json (default: /kaggle/working, or ./out with --local)")
    args = ap.parse_args()
    cfg = dict(vars(args))
    if EMBEDDED_CONFIG and not args.local:
        cfg.update(EMBEDDED_CONFIG)
    cfg["local"] = args.local
    return argparse.Namespace(**cfg)


def driver_info():
    r = sh("nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader", capture=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    drv, name = [s.strip() for s in r.stdout.splitlines()[0].split(",", 1)]
    return drv, name


def make_env_python(cfg, work):
    """The interpreter the run uses. CUDA 12 runs in the image's own Python, as a Kaggle user does (the image is
    CUDA 12). CUDA 13 gets a clean venv: on top of the image's CUDA 12 packages it would mix two CUDA releases.
    Python older than 3.12 also gets a venv (JAX needs 3.12)."""
    clean = not cfg.local and cfg.cuda == "13"
    if cfg.local or (sys.version_info >= (3, 12) and not clean):
        return sys.executable, False
    sh(sys.executable, "-m", "pip", "install", "-q", "uv")
    want = sys.executable if sys.version_info >= (3, 12) else "3.12"
    venv = work / "venv"
    if sh(sys.executable, "-m", "uv", "venv", "--python", want, venv) != 0:
        raise SystemExit("could not create a clean Python environment with uv")
    return str(venv / "bin" / "python"), True


def pip(py, use_uv, *args):
    # uv takes a package from the first index that has it; best-match keeps PyPI's versions beside the torch index
    cmd = ([sys.executable, "-m", "uv", "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match"]
           if use_uv else [py, "-m", "pip", "install"])
    return sh(*cmd, *args)


def torch_index(cfg):
    """PyPI's torch wheels bundle CUDA 12; the CUDA 13 run takes torch built for CUDA 13."""
    return ["--extra-index-url", "https://download.pytorch.org/whl/cu130"] if cfg.cuda == "13" else []


def install(cfg, py, use_uv, work):
    cu = f"cuda{cfg.cuda}"
    pre = ["--pre"] if cfg.pre else []
    pin = f"=={cfg.version}" if cfg.version else ""
    others = ["jax[" + cu + "]", "torch", "warp-lang", "pytest", "nbclient", "ipykernel", "nbformat", "matplotlib"]
    if cfg.source == "pypi":
        rc = pip(py, use_uv, *pre, *torch_index(cfg), f"raptor-eagle[{cu}]{pin}", f"raptor-hawk[{cu}]", *others)
    else:
        src = work / "src"
        src.mkdir(exist_ok=True)
        for repo, ref in (("raptor", cfg.raptor_ref), ("aether", cfg.aether_ref),
                          ("eagle", cfg.eagle_ref), ("hawk", cfg.hawk_ref)):
            clone(repo, ref, src / repo)
        prefix = work / "install"
        arch = "-DCMAKE_CUDA_ARCHITECTURES=75"
        for lib in ("aether", "eagle"):
            if sh("cmake", "-S", src / lib, "-B", work / f"build_{lib}", arch,
                  f"-DCMAKE_INSTALL_PREFIX={prefix}") != 0 or \
               sh("cmake", "--install", work / f"build_{lib}") != 0:
                raise SystemExit(f"{lib} headers did not install")
        pip(py, use_uv, "scikit-build-core>=0.10", "nanobind>=3.1,<4", "cmake", "ninja")
        env = dict(os.environ, CMAKE_PREFIX_PATH=f"{prefix}:{os.environ.get('CMAKE_PREFIX_PATH', '')}",
                   SKBUILD_CMAKE_ARGS=f"{arch};-DCMAKE_CUDA_COMPILER={shutil.which('nvcc') or 'nvcc'}")
        os.environ.update(env)
        rc = pip(py, use_uv, src / "raptor")
        rc = rc or pip(py, use_uv, "--no-build-isolation", src / "eagle" / "python")
        rc = rc or pip(py, use_uv, "--no-build-isolation", src / "hawk")
        rc = rc or pip(py, use_uv, *pre, *torch_index(cfg), f"jax[{cu}]", "torch", "warp-lang", f"cupy-cuda{cfg.cuda}x",
                       "pytest", "nbclient", "ipykernel", "nbformat", "matplotlib")
    if rc != 0:
        raise SystemExit("install failed (see the pip output above)")


def clone(repo, ref, dest):
    if dest.exists():
        return
    if sh("git", "clone", "-q", "--depth", "1", "--branch", ref, f"{ORG}/{repo}.git", dest) != 0:
        raise SystemExit(f"could not clone {repo} at {ref}")
    print(repo, subprocess.run(["git", "-C", dest, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())


JAX_CHECK = 'import jax; d = jax.devices(); print("jax devices:", d); assert d[0].platform == "gpu"'
NV_DIRS = ("import nvidia, os; p = list(nvidia.__path__)[0]; print(os.pathsep.join(sorted("
           "os.path.join(p, d, 'lib') for d in os.listdir(p) if os.path.isdir(os.path.join(p, d, 'lib')))))")


def check_jax(py):
    if sh(py, "-c", JAX_CHECK) == 0:
        return
    # A system CUDA earlier on LD_LIBRARY_PATH can hide the env's own NVIDIA libraries.
    r = sh(py, "-c", NV_DIRS, capture=True)
    if r.returncode == 0 and r.stdout.strip():
        os.environ["LD_LIBRARY_PATH"] = r.stdout.strip() + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        if sh(py, "-c", JAX_CHECK) == 0:
            print("JAX sees the GPU after putting the env's NVIDIA libraries first")
            return
    print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
    sh("bash", "-c", "ls /usr/local/cuda*/lib64/libnvJitLink* /usr/local/cuda*/lib64/libcusparse* 2>&1 | head")
    raise SystemExit("JAX does not see the GPU, even with the env's NVIDIA libraries first")


def ensure_nvcc(py):
    if shutil.which("nvcc"):
        return
    code = "import nvidia, glob, os; p = list(nvidia.__path__)[0]; print(os.pathsep.join(glob.glob(p + '/*/bin')))"
    r = sh(py, "-c", code, capture=True)
    dirs = r.stdout.strip() if r.returncode == 0 else ""
    if dirs:
        os.environ["PATH"] = dirs + os.pathsep + os.environ["PATH"]
    print("nvcc:", shutil.which("nvcc") or "NOT FOUND (the spin kernel row will fail)")


def versions(py):
    code = ("import importlib.metadata as m, json\nout = {}\n"
            f"for n in {FAMILY!r}:\n"
            "    try: out[n] = m.version(n)\n"
            "    except m.PackageNotFoundError: pass\n"
            "print(json.dumps(out))")
    r = sh(py, "-c", code, capture=True)
    try:
        v = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}
    out = {"eagle": v.get("raptor-eagle"), "hawk": v.get("raptor-hawk"), "jax": v.get("jax"),
           "jaxlib": v.get("jaxlib"), "torch": v.get("torch"),
           "cupy": v.get("cupy-cuda13x") or v.get("cupy-cuda12x")}
    return out


def junit_summary(path):
    s = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "failed": []}
    if not path.exists():
        return None
    for suite in ET.parse(path).getroot().iter("testsuite"):
        for k in ("tests", "failures", "errors", "skipped"):
            s[k] += int(suite.get(k, 0))
        for case in suite.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                s["failed"].append(f"{case.get('classname')}::{case.get('name')}")
    return s


def run_tests(py, eagle_src, work):
    # a copy of the suite outside the checkout, as CI runs it: from inside, the source tree's eagle/ (no compiled
    # extension) would shadow the installed package
    pydir = work / "suite"
    shutil.rmtree(pydir, ignore_errors=True)
    pydir.mkdir()
    shutil.copytree(eagle_src / "python" / "tests", pydir / "tests")
    shutil.copy2(eagle_src / "python" / "conftest.py", pydir)
    runs = (("interop_matrix", ["tests", "-m", "interop_matrix and not repo_local"]),
            ("jax", ["tests/test_interop_jax.py", "-m", "not repo_local"]))
    total = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "failed": []}
    rcs = {}
    for name, sel in runs:
        header(f"pytest: {name}")
        xml = work / f"junit_{name}.xml"
        base = pathlib.Path(tempfile.mkdtemp(prefix=f"pytest_{name}_", dir=work))
        rcs[name] = sh(py, "-m", "pytest", *sel, "-q", "-rs", "-p", "no:cacheprovider", "--basetemp", base / "t",
                       "-W", "error::pytest.PytestUnknownMarkWarning", f"--junitxml={xml}", cwd=pydir)
        s = junit_summary(xml)
        if s is None:
            total["errors"] += 1
            total["failed"].append(f"{name}: no junit file (pytest exit {rcs[name]})")
            continue
        for k in ("tests", "failures", "errors", "skipped"):
            total[k] += s[k]
        total["failed"] += s["failed"]
    total["exit_codes"] = rcs
    return total


def register_kernel(py, work):
    """A Jupyter kernel that runs this environment's interpreter, so a notebook sees the installed packages."""
    prefix = work / "jupyter"
    if sh(py, "-m", "ipykernel", "install", "--prefix", prefix, "--name", "eagle_gpu") != 0:
        raise SystemExit("could not register a Jupyter kernel for the notebooks")
    os.environ["JUPYTER_PATH"] = f"{prefix}/share/jupyter" + os.pathsep + os.environ.get("JUPYTER_PATH", "")


def run_notebook(py, path, work, name):
    header(f"notebook: {name}")
    if path is None or not pathlib.Path(path).exists():
        print("notebook not found:", path)
        return {"name": name, "status": "missing", "seconds": 0}
    code = ("import sys, nbformat\nfrom nbclient import NotebookClient\n"
            "nb = nbformat.read(sys.argv[1], as_version=4)\n"
            "NotebookClient(nb, timeout=900, kernel_name='eagle_gpu', resources={'metadata': {'path': sys.argv[2]}}).execute()\n")
    cwd = pathlib.Path(path).parent  # the notebook's own folder, where its helper files are found
    t0 = time.time()
    rc = sh(py, "-c", code, path, cwd)
    return {"name": name, "status": "pass" if rc == 0 else "fail", "seconds": round(time.time() - t0)}


def write_result(out, result):
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("\nresult:", json.dumps({k: result[k] for k in ("status", "reason") if k in result}))


def main():
    cfg = settings()
    out = pathlib.Path(cfg.out or ("./out" if cfg.local else "/kaggle/working"))
    temp_root = "/kaggle/temp" if not cfg.local and pathlib.Path("/kaggle/temp").is_dir() else None
    work = pathlib.Path(tempfile.mkdtemp(prefix="eagle_gpu_", dir=temp_root))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    header("driver and GPU")
    sh("nvidia-smi")
    driver, gpu = driver_info()
    header("python")
    print(sys.version)
    result = {"status": "fail", "python": ".".join(map(str, sys.version_info[:3])), "driver": driver, "gpu": gpu,
              "cuda": None if cfg.local else cfg.cuda, "source": cfg.source, "versions": {}}
    if not cfg.local and cfg.cuda == "13" and driver and int(driver.split(".")[0]) < 580:
        result.update(status="skip", reason=f"CUDA 13 needs driver 580 or newer; this machine has {driver}")
        write_result(out, result)
        return 0
    if not cfg.local and not driver:
        result["reason"] = "no NVIDIA driver visible: the kernel did not get a GPU"
        write_result(out, result)
        return 1
    header("environment")
    py, use_uv = make_env_python(cfg, work)
    if not cfg.local:
        install(cfg, py, use_uv, work)
    ensure_nvcc(py)
    result["versions"] = versions(py)
    print(json.dumps(result["versions"], indent=2))
    header("JAX sees the GPU")
    try:
        check_jax(py)
    except SystemExit as e:
        result["reason"] = str(e)
        write_result(out, result)
        return 1
    header("test sources")
    if cfg.local:
        eagle_src = pathlib.Path(cfg.eagle_src or pathlib.Path(__file__).resolve().parents[2])
        raptor_src = pathlib.Path(cfg.raptor_src) if cfg.raptor_src else None
        site_src = pathlib.Path(cfg.site_src) if cfg.site_src else None
    else:
        src = work / "src"
        src.mkdir(exist_ok=True)
        clone("eagle", cfg.eagle_ref, src / "eagle")
        clone("raptor", cfg.raptor_ref, src / "raptor")
        clone(SITE_REPO, "main", src / SITE_REPO)
        eagle_src, raptor_src, site_src = src / "eagle", src / "raptor", src / SITE_REPO
    if raptor_src:
        os.environ["RAPTOR_ROOT"] = str(raptor_src)
    if cfg.pre:
        os.environ["RAPTOR_PRE"] = "1"
    os.environ["RAPTOR_REF"] = cfg.raptor_ref
    tests = run_tests(py, eagle_src, work)
    register_kernel(py, work)
    notebooks = [run_notebook(py, site_src / START_HERE if site_src else None, work, "raptor_start_here"),
                 run_notebook(py, eagle_src / QUICKSTART, work, "quickstart_python")]
    ok = tests["tests"] > 0 and not tests["failures"] and not tests["errors"] \
        and all(n["status"] == "pass" for n in notebooks)
    # In --local mode a missing site checkout is not a failure.
    if cfg.local:
        ok = tests["tests"] > 0 and not tests["failures"] and not tests["errors"] \
            and all(n["status"] in ("pass", "missing") for n in notebooks)
    result.update(status="pass" if ok else "fail", junit=tests, failed_tests=tests["failed"], notebooks=notebooks)
    write_result(out, result)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
