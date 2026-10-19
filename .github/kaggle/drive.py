# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Pushes kernel.py to Kaggle on a T4, polls it, fetches the output and turns result.json into a
# verdict. `--mode tests` (the interop rows) and `--mode notebooks` (every docs notebook) are separate
# kernels, so neither shares the other's time limit. Needs only the `kaggle` package and KAGGLE_API_TOKEN in the environment; the Kaggle
# username comes from the token. `--dry-run` prints the metadata and commands and touches nothing.
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
POLL_SECONDS = 30
LIMITS = {"tests": 75 * 60, "notebooks": 180 * 60}  # per kernel, by mode
# PEP 440 (the public version grammar of packaging's VERSION_PATTERN, lower case): 0.6.2, 0.6.2rc1, 1!2.0.post1.dev3
PEP440 = re.compile(r"^(?:\d+!)?\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")
GPU = "NvidiaTeslaT4"  # the machine_shape the push takes; the kernel records the GPU it got


def username():
    """The account behind the token, asked of the kaggle client (token introspection)."""
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        name = api.get_config_value(api.CONFIG_NAME_USER)
    except Exception as e:  # the message of a client error is not printed: it could quote the token
        raise SystemExit(f"could not derive the Kaggle username from the token: {type(e).__name__}")
    if not name:
        raise SystemExit("the Kaggle client returned no username for this token (expired or wrong token?)")
    return name


def build(args, user, folder):
    slug = f"eagle-{'nb' if args.mode == 'notebooks' else 'gpu'}-cuda{args.cuda}-{args.run_id}".lower()
    config = {k: getattr(args, k) for k in ("mode", "cuda", "source", "pre", "version", "nb_timeout", "wheel_sha256",
                                            "eagle_ref", "hawk_ref", "raptor_ref", "aether_ref")}
    text = (HERE / "kernel.py").read_text()
    if "EMBEDDED_CONFIG = None" not in text:
        raise SystemExit("kernel.py has no EMBEDDED_CONFIG line to fill")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(text.replace("EMBEDDED_CONFIG = None", f"EMBEDDED_CONFIG = {config!r}", 1))
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": GPU, "dataset_sources": [args.dataset] if args.source == "wheel" else [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def kaggle(*cmd, check=True):
    print("+ kaggle", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, "-m", "kaggle", *cmd], capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip(), file=sys.stderr)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {cmd[1] if len(cmd) > 1 else cmd[0]} failed (exit {r.returncode})")
    return r


def state(text):
    m = re.search(r"\b(QUEUED|RUNNING|COMPLETE|ERROR|CANCEL_REQUESTED|CANCEL_ACKNOWLEDGED|NEW_SCRIPT)\b", text)
    return m.group(1) if m else "UNKNOWN"


def summary(lines):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    text = "\n".join(lines) + "\n"
    print(text)
    if path:
        with open(path, "a") as f:
            f.write(text)


def title(args):
    return f"### Kaggle GPU, CUDA {args.cuda}, {'every docs notebook' if args.mode == 'notebooks' else 'interop tests'}"


def table(args, res):
    rows = [("status", res.get("status")), ("reason", res.get("reason")), ("python", res.get("python")),
            ("driver", res.get("driver")), ("gpu", res.get("gpu"))]
    rows += [(f"version {k}", v) for k, v in (res.get("versions") or {}).items()]
    j = res.get("junit")
    if j:
        rows += [("tests", j["tests"]), ("failures", j["failures"]), ("errors", j["errors"]), ("skipped", j["skipped"])]
    rows += [("failed test", t) for t in res.get("failed_tests", [])]
    out = [title(args), "", "| item | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows if v not in (None, "")]
    return out + notebook_table(res)


def notebook_table(res):
    """One row per notebook (PASS / FAIL / SKIP, seconds); a FAIL carries the failing cell's error lines."""
    nbs = res.get("notebooks") or []
    if not nbs:
        return []
    c = res.get("counts") or {}
    out = ["", f"Notebooks: {c.get('PASS', 0)} passed, {c.get('FAIL', 0)} failed, {c.get('SKIP', 0)} skipped"
           f" of {res.get('discovered', len(nbs))} found.", "", "| notebook | result | seconds | note |", "|---|---|---|---|"]
    out += [f"| {n['name']} | {n['status']} | {n['seconds']} | {n.get('reason') or ''} |" for n in nbs]
    for n in nbs:
        if n["status"] == "FAIL" and n.get("error"):
            out += ["", f"<details><summary>{n['name']}</summary>", "", "```", *n["error"], "```", "</details>"]
    return out


def commands(folder, outdir, limit):
    return [("kernels", "push", "-p", str(folder), "--timeout", str(limit), "--accelerator", GPU),
            ("kernels", "status", "USER/SLUG"), ("kernels", "output", "USER/SLUG", "-p", str(outdir))]


def push(cmd, wait=3600):
    """Push the kernel. Kaggle runs at most two GPU sessions per account and answers a third with an error but exit
    0, so wait for a free slot while that is the error and stop on any other one."""
    t0 = time.time()
    while True:
        r = kaggle(*cmd, check=False)
        out = r.stdout + r.stderr
        if "successfully pushed" in out:
            return
        if "GPU session count" not in out or time.time() - t0 > wait:
            raise SystemExit("kernel push failed: " + (out.strip().splitlines() or ["no output"])[-1])
        time.sleep(60)


def fetch_result(outdir):
    """result.json of the kernel (the notebooks kernel rewrites it after every notebook, so a kernel that ended as
    ERROR still shows what ran), or None."""
    path = outdir / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cuda", required=True, choices=("12", "13"))
    ap.add_argument("--mode", default="tests", choices=("tests", "notebooks"), help="the interop test rows, or every docs notebook")
    ap.add_argument("--nb-timeout", type=int, default=900, help="notebooks mode: seconds one notebook may run (default 900)")
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    ap.add_argument("--source", default="pypi", choices=("pypi", "git", "wheel"),
                    help="wheel: the pull request's own raptor-eagle wheel, from the private dataset made by wheel.py")
    ap.add_argument("--dataset", default="", help="--source wheel: USER/SLUG of that dataset")
    ap.add_argument("--wheel-sha256", default="", help="--source wheel: the sha256 of the dataset's SHA256SUMS file")
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--version", default="")
    for n in ("eagle", "hawk", "raptor", "aether"):
        ap.add_argument(f"--{n}-ref", default="main")
    ap.add_argument("--out", default="kaggle-out", help="where the kernel output and log are saved")
    ap.add_argument("--dry-run", action="store_true", help="print the metadata and commands; no network")
    args = ap.parse_args()
    if args.version and not PEP440.match(args.version):  # before any network call
        raise SystemExit(f"version must look like 0.6.2rc1 (a PEP 440 version), not {args.version!r}")
    if args.source == "wheel" and not (re.fullmatch(r"[\w.-]+/[\w.-]+", args.dataset) and re.fullmatch(r"[0-9a-f]{64}", args.wheel_sha256)):
        raise SystemExit("--source wheel needs --dataset USER/SLUG and --wheel-sha256 (64 hex digits)")
    limit = LIMITS[args.mode]
    outdir = pathlib.Path(args.out)
    folder = pathlib.Path(tempfile.mkdtemp(prefix="kaggle_kernel_"))
    user = "USER" if args.dry_run else username()
    meta = build(args, user, folder)
    if args.dry_run:
        print("kernel-metadata.json:\n" + json.dumps(meta, indent=2))
        for c in commands(folder, outdir, limit):
            print("+ kaggle", " ".join(c))
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    kid = meta["id"]
    push(commands(folder, outdir, limit)[0])
    t0 = time.time()
    st = "UNKNOWN"
    while time.time() - t0 < limit:
        time.sleep(POLL_SECONDS)
        st = state(kaggle("kernels", "status", kid, check=False).stdout)
        print(f"[{int(time.time() - t0)} s] {st}", flush=True)
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            break
    else:
        kaggle("kernels", "output", kid, "-p", str(outdir), check=False)
        res = fetch_result(outdir)
        summary([title(args), "", f"Timed out after {limit // 60} minutes."] + (notebook_table(res) if res else []))
        return 1
    kaggle("kernels", "output", kid, "-p", str(outdir), check=False)
    res = fetch_result(outdir)
    if res is None:
        summary([title(args), "", f"Kernel ended as {st} and left no result.json; see the log artifact."])
        return 1
    lines = table(args, res)
    if res.get("status") == "skip":
        print(f"::warning::Kaggle GPU CUDA {args.cuda} skipped: {res.get('reason')}")
        lines += ["", f"SKIPPED: {res.get('reason')}"]
    summary(lines)
    if res.get("status") == "skip":
        return 0
    return 0 if res.get("status") == "pass" and st == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
