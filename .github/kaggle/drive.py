# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Pushes kernel.py to Kaggle on a T4, polls it, fetches the output and turns result.json into a
# verdict. Needs only the `kaggle` package and KAGGLE_API_TOKEN in the environment; the Kaggle
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
LIMIT_SECONDS = 75 * 60
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
    slug = f"eagle-gpu-cuda{args.cuda}-{args.run_id}".lower()
    config = {k: getattr(args, k) for k in ("cuda", "source", "pre", "version",
                                            "eagle_ref", "hawk_ref", "raptor_ref", "aether_ref")}
    text = (HERE / "kernel.py").read_text()
    if "EMBEDDED_CONFIG = None" not in text:
        raise SystemExit("kernel.py has no EMBEDDED_CONFIG line to fill")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(text.replace("EMBEDDED_CONFIG = None", f"EMBEDDED_CONFIG = {config!r}", 1))
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": GPU, "dataset_sources": [],
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


def table(args, res):
    rows = [("status", res.get("status")), ("reason", res.get("reason")), ("python", res.get("python")),
            ("driver", res.get("driver")), ("gpu", res.get("gpu"))]
    rows += [(f"version {k}", v) for k, v in (res.get("versions") or {}).items()]
    j = res.get("junit")
    if j:
        rows += [("tests", j["tests"]), ("failures", j["failures"]), ("errors", j["errors"]), ("skipped", j["skipped"])]
    rows += [(f"notebook {n['name']}", f"{n['status']} ({n['seconds']} s)") for n in res.get("notebooks", [])]
    rows += [("failed test", t) for t in res.get("failed_tests", [])]
    out = [f"### Kaggle GPU, CUDA {args.cuda}", "", "| item | value |", "|---|---|"]
    return out + [f"| {k} | {v} |" for k, v in rows if v not in (None, "")]


def commands(folder, outdir):
    return [("kernels", "push", "-p", str(folder), "--timeout", str(LIMIT_SECONDS), "--accelerator", GPU),
            ("kernels", "status", "USER/SLUG"), ("kernels", "output", "USER/SLUG", "-p", str(outdir))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cuda", required=True, choices=("12", "13"))
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    ap.add_argument("--source", default="pypi", choices=("pypi", "git"))
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--version", default="")
    for n in ("eagle", "hawk", "raptor", "aether"):
        ap.add_argument(f"--{n}-ref", default="main")
    ap.add_argument("--out", default="kaggle-out", help="where the kernel output and log are saved")
    ap.add_argument("--dry-run", action="store_true", help="print the metadata and commands; no network")
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    folder = pathlib.Path(tempfile.mkdtemp(prefix="kaggle_kernel_"))
    user = "USER" if args.dry_run else username()
    meta = build(args, user, folder)
    if args.dry_run:
        print("kernel-metadata.json:\n" + json.dumps(meta, indent=2))
        for c in commands(folder, outdir):
            print("+ kaggle", " ".join(c))
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    kid = meta["id"]
    kaggle(*commands(folder, outdir)[0])
    t0 = time.time()
    st = "UNKNOWN"
    while time.time() - t0 < LIMIT_SECONDS:
        time.sleep(POLL_SECONDS)
        st = state(kaggle("kernels", "status", kid, check=False).stdout)
        print(f"[{int(time.time() - t0)} s] {st}", flush=True)
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            break
    else:
        kaggle("kernels", "output", kid, "-p", str(outdir), check=False)
        summary([f"### Kaggle GPU, CUDA {args.cuda}", "", f"Timed out after {LIMIT_SECONDS // 60} minutes."])
        return 1
    kaggle("kernels", "output", kid, "-p", str(outdir), check=False)
    result = outdir / "result.json"
    if st != "COMPLETE" or not result.exists():
        summary([f"### Kaggle GPU, CUDA {args.cuda}", "",
                 f"Kernel ended as {st} and left no result.json; see the log artifact."])
        return 1
    res = json.loads(result.read_text())
    lines = table(args, res)
    if res.get("status") == "skip":
        print(f"::warning::Kaggle GPU CUDA {args.cuda} skipped: {res.get('reason')}")
        lines += ["", f"SKIPPED: {res.get('reason')}"]
    summary(lines)
    return 1 if res.get("status") == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
