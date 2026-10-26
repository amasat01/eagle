# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# The script Kaggle runs on a T4: it installs the family, checks that JAX sees the GPU, then either
#   --mode tests      runs the interop certification rows plus the JAX rows, or
#   --mode notebooks  runs EVERY docs notebook (site, eagle, hawk; found by glob, see raptor_docs.nb.discover),
#                     each from a copy in an empty directory with its setup cell executing for real (the
#                     Kaggle branch), and records PASS / FAIL / SKIP + seconds per notebook,
# and writes result.json. drive.py pushes it with the settings below filled in, one kernel per mode and
# CUDA leg; `--local` runs the same steps in the current environment (no install, no Kaggle paths).
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

# drive.py replaces the next line with the run settings.
EMBEDDED_CONFIG = None

ORG = "https://github.com/amasat01"
SITE_REPO = "amasat01.github.io"
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
    ap.add_argument("--mode", default="tests", choices=("tests", "notebooks"), help="the interop test rows, or every docs notebook")
    ap.add_argument("--nb-timeout", type=int, default=900, help="notebooks mode: seconds one notebook may run before it FAILs")
    ap.add_argument("--only", default="", help="notebooks mode: run only notebooks whose path contains this text")
    ap.add_argument("--hawk-src", help="local hawk checkout (notebooks mode)")
    ap.add_argument("--aether-src", help="local aether checkout (notebooks mode; its helper files)")
    ap.add_argument("--local", action="store_true", help="use the current environment; no install, no Kaggle paths")
    ap.add_argument("--cuda", default="12", help="CUDA major of the extras: 12 or 13")
    ap.add_argument("--source", default="pypi", choices=("pypi", "git", "wheel"),
                    help="wheel: the raptor-eagle wheel shipped in a Kaggle dataset (the pull request's own build)")
    ap.add_argument("--wheel-sha256", default="", help="--source wheel: the sha256 of the shipped SHA256SUMS file")
    ap.add_argument("--wheel-dir", default="/kaggle/input", help="--source wheel: where the dataset is mounted")
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--version", default="")
    ap.add_argument("--sibling-pins", default="", help="source wheel: exact sibling pins, `pkg==V ...`")
    for name in ("eagle", "hawk", "raptor", "aether", "site"):
        ap.add_argument(f"--{name}-ref", default="main")
    ap.add_argument("--eagle-src", help="local eagle checkout (default: the one holding this script)")
    ap.add_argument("--raptor-src", help="local raptor checkout, sets RAPTOR_ROOT")
    ap.add_argument("--site-src", help="local checkout of the site repository (its notebooks run in notebooks mode)")
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
    if sh(sys.executable, "-m", "uv", "venv", "--seed", "--python", want, venv) != 0:  # --seed: pip, which the notebooks' setup cell calls
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


def shipped_wheel(cfg, py):
    """The mounted dataset's raptor-eagle wheel for ``py``'s CPython tag, after checking the shipped SHA256SUMS file
    against the sha256 in the config and the wheel against its line there (a dataset overwritten by another run fails
    here, not silently)."""
    tag = subprocess.run([str(py), "-c", "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"],
                         capture_output=True, text=True, check=True).stdout.strip()
    found = sorted(pathlib.Path(cfg.wheel_dir).rglob(f"raptor_eagle-*-{tag}-{tag}-*.whl"))
    if len(found) != 1:
        raise SystemExit(f"expected one {tag} raptor_eagle wheel under {cfg.wheel_dir}, found {len(found)}")
    wheel = found[0]
    sums = (wheel.parent / "SHA256SUMS").read_bytes()
    if hashlib.sha256(sums).hexdigest() != cfg.wheel_sha256:
        raise SystemExit(f"shipped SHA256SUMS does not match the expected {cfg.wheel_sha256}")
    got = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if f"{got}  {wheel.name}" not in sums.decode().splitlines():
        raise SystemExit(f"{wheel.name}: sha256 {got} is not the one in the shipped SHA256SUMS")
    listed = {ln.split()[-1]: ln.split()[0] for ln in sums.decode().splitlines()}
    # Kaggle drops "+" from dataset file names, so a sibling with a local version (0.7.1.dev1+g...) arrives
    # without it: each file is matched to its listed name, checked, and copied back under that name for pip.
    by_upload = {name.replace("+", ""): name for name in listed}
    staged = pathlib.Path(tempfile.mkdtemp(prefix="eagle-wheels-"))
    for f in sorted(p for p in wheel.parent.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        name = by_upload.get(f.name)
        if name is None or listed[name] != hashlib.sha256(f.read_bytes()).hexdigest():
            raise SystemExit(f"{f.name}: not in the shipped SHA256SUMS or its sha256 differs")
        shutil.copy2(f, staged / name)
    print("wheel", wheel.name, "sha256 ok")
    return staged / by_upload[wheel.name]


def install_args(cfg, py, cu, pre, pin, others):
    """pip arguments for source pypi or wheel. For wheel, the dataset folder is a find-links source and the siblings
    (raptor-hawk with its extra, raptor-core, aether-dsc) are exact pins: PyPI's when released, else built from the sdist
    or wheel the pull request's run shipped."""
    if cfg.source == "pypi":
        return [*pre, *torch_index(cfg), f"raptor-eagle[{cu}]{pin}", f"raptor-hawk[{cu}]", *others]
    wheel = shipped_wheel(cfg, py)
    pins = (getattr(cfg, "sibling_pins", "") or "").split()
    hawk = next((p.split("==")[1] for p in pins if p.startswith("raptor-hawk==")), None)
    rest = [p for p in pins if not p.startswith("raptor-hawk==")]
    return [*pre, *torch_index(cfg), "--find-links", str(wheel.parent), f"{wheel}[{cu}]",
            f"raptor-hawk[{cu}]" + (f"=={hawk}" if hawk else ""), *rest, *others]


def install(cfg, py, use_uv, work):
    cu = f"cuda{cfg.cuda}"
    pre = ["--pre"] if cfg.pre else []
    pin = f"=={cfg.version}" if cfg.version else ""
    others = ["jax[" + cu + "]", "torch", "warp-lang", "pytest", "nbclient", "ipykernel", "nbformat", "matplotlib"]
    if cfg.source in ("pypi", "wheel"):
        rc = pip(py, use_uv, *install_args(cfg, py, cu, pre, pin, others))
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


def pin_family(py, work):
    """A notebook's setup cell runs `pip install --upgrade raptor-eagle ...`: constrain the family to what is
    installed, so a git build or a pinned release is not replaced by PyPI's newest."""
    code = ("import importlib.metadata as m\nfor n in ('raptor-eagle', 'raptor-hawk', 'raptor-core', 'aether-dsc'):\n"
            "    try: print(f'{n}=={m.version(n)}')\n    except m.PackageNotFoundError: pass")
    path = work / "family_constraints.txt"
    path.write_text(sh(py, "-c", code, capture=True).stdout)
    os.environ["PIP_CONSTRAINT"] = str(path)


def table_of(rows):
    return {"PASS": sum(r["status"] == "PASS" for r in rows), "FAIL": sum(r["status"] == "FAIL" for r in rows),
            "SKIP": sum(r["status"] == "SKIP" for r in rows)}


def run_notebooks(cfg, py, work, roots, result, out):
    """Every notebook of the checkouts in ``roots`` (repo name -> path; 'site' = the site repository), one by one.
    result.json is rewritten after each one, so a kernel that dies still shows what ran."""
    sys.path.insert(0, str(roots["site"]))  # raptor_docs, the family's docs tooling, lives in the site checkout
    from raptor_docs.nb import discover, hosted as nb_hosted, run_gate as nb_run_gate
    found = discover({r: roots[r] for r in ("site", "eagle", "hawk") if r in roots})
    todo = [(repo, p, p.relative_to(roots[repo] / ("notebooks" if repo == "site" else "docs/content")))
            for repo, nbs in found.items() for p in nbs if cfg.only in p.as_posix()]
    header(f"{len(todo)} notebooks: " + ", ".join(f"{r} {len(n)}" for r, n in found.items()))
    nb_run_gate.list_notebooks({r: [p for p in n if cfg.only in p.as_posix()] for r, n in found.items()})
    pin_family(py, work)
    mirror_roots = {r: roots[r] for r in ("eagle", "hawk", "raptor", "aether") if r in roots}
    server = nb_hosted.start_mirror(mirror_roots, work)  # the setup cells' helper files come from these checkouts
    os.environ.update(RAPTOR_RAW_BASE=f"http://127.0.0.1:{server.server_address[1]}", RAPTOR_REF="local",
                      NB_HOSTED_GPU="1", NB_HOSTED_CELL_TIMEOUT=str(cfg.nb_timeout))
    if cfg.pre:
        os.environ["RAPTOR_PRE"] = "1"
    if cfg.local:  # a local run is not on Kaggle: select the setup cells' Kaggle branch, as the real kernel does
        os.environ.setdefault("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    rows = []
    result.update(status="incomplete", reason="notebooks still running", discovered=len(todo), notebooks=rows)
    for repo, path, rel in todo:
        header(f"notebook: {repo}/{rel}")
        status, msg, tail, secs = nb_hosted.run_notebook(py, work / "kernels", path, dict(os.environ), cfg.nb_timeout)
        print(f"{status} {repo}/{rel} [{secs} s] {msg}", flush=True)
        if tail:
            print("\n".join(tail), flush=True)
        rows.append({"name": f"{repo}/{rel}", "status": status, "seconds": secs, "reason": msg, "error": tail})
        write_result(out, result, quiet=True)
    server.shutdown()
    counts = table_of(rows)
    ok = counts["FAIL"] == 0 and counts["PASS"] > 0 and len(rows) == len(todo)
    result.update(status="pass" if ok else "fail", counts=counts)
    result.pop("reason", None)
    if not ok and not counts["FAIL"]:
        result["reason"] = "no notebook passed" if todo else "no notebooks found"
    return ok


def write_result(out, result, quiet=False):
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if quiet:
        return
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
    roots = {}
    if cfg.local:
        roots["eagle"] = pathlib.Path(cfg.eagle_src or pathlib.Path(__file__).resolve().parents[2])
        for name in ("raptor", "site", "hawk", "aether"):
            if getattr(cfg, f"{name}_src"):
                roots[name] = pathlib.Path(getattr(cfg, f"{name}_src"))
    else:
        src = work / "src"
        src.mkdir(exist_ok=True)
        clone("eagle", cfg.eagle_ref, src / "eagle")
        clone("raptor", cfg.raptor_ref, src / "raptor")
        roots.update(eagle=src / "eagle", raptor=src / "raptor")
        if cfg.mode == "notebooks":
            clone(SITE_REPO, cfg.site_ref, src / SITE_REPO)
            clone("hawk", cfg.hawk_ref, src / "hawk")
            clone("aether", cfg.aether_ref, src / "aether")
            roots.update(site=src / SITE_REPO, hawk=src / "hawk", aether=src / "aether")
    if "raptor" in roots:
        os.environ["RAPTOR_ROOT"] = str(roots["raptor"])
    if cfg.pre:
        os.environ["RAPTOR_PRE"] = "1"
    os.environ["RAPTOR_REF"] = cfg.raptor_ref
    result["mode"] = cfg.mode
    if cfg.mode == "notebooks":
        ok = run_notebooks(cfg, py, work, roots, result, out)
        write_result(out, result)
        return 0 if ok else 1
    tests = run_tests(py, roots["eagle"], work)
    ok = tests["tests"] > 0 and not tests["failures"] and not tests["errors"]
    result.update(status="pass" if ok else "fail", junit=tests, failed_tests=tests["failed"])
    write_result(out, result)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
