# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Puts the pull request's own raptor-eagle wheels (CPython 3.12 and newer, GIL builds, manylinux x86_64, from the build
# job's dist folder; the kernel installs the one matching Kaggle's Python) into the private Kaggle dataset
# USER/eagle-pr-wheel-<first 12 characters of the git tree hash>, with their SHA256SUMS lines beside them,
# creating the dataset the first time and a new version after that, and waits until the version is ready.
# Prints `dataset=USER/SLUG` and `sha256=...` (of the shipped SHA256SUMS file) (also appended to $GITHUB_OUTPUT), which drive.py --source wheel takes.
# Needs the `kaggle` package and KAGGLE_API_TOKEN. `--dry-run` stages the folder and touches nothing else.
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time

from drive import kaggle, username

SLUG = "eagle-pr-wheel"  # plus -<tree[:12]>: one dataset per tree, so runs of different pull requests never share one
WHEEL = re.compile(r"^raptor_eagle-[^-]+-(cp3(1[2-9]|[2-9]\d))-\1-manylinux[\w.]*x86_64\.whl$")
READY_SECONDS = 600


def stage(dist, folder):
    """Copy the matching wheels and their checksum lines into ``folder``; return (wheel names, sha256 of SHA256SUMS)."""
    wheels = sorted(p for p in pathlib.Path(dist).iterdir() if WHEEL.match(p.name))
    if not wheels:
        raise SystemExit(f"no CPython 3.12+ manylinux x86_64 raptor_eagle wheel in {dist}")
    names = {p.name for p in wheels}
    lines = [ln for ln in (pathlib.Path(dist) / "SHA256SUMS").read_text().splitlines() if ln.split()[-1] in names]
    if len(lines) != len(wheels):
        raise SystemExit(f"SHA256SUMS has {len(lines)} lines for {len(wheels)} wheels")
    folder.mkdir(parents=True, exist_ok=True)
    for p in wheels:
        shutil.copy2(p, folder)
    sums = "\n".join(lines) + "\n"
    (folder / "SHA256SUMS").write_text(sums)
    return sorted(names), hashlib.sha256(sums.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dist", help="folder holding the wheels and SHA256SUMS")
    ap.add_argument("tree", help="git tree hash the wheel was built from")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    folder = pathlib.Path(tempfile.mkdtemp(prefix="kaggle_wheel_"))
    names, sha = stage(args.dist, folder)
    user = "USER" if args.dry_run else username()
    slug = f"{SLUG}-{args.tree[:12]}"
    dataset = f"{user}/{slug}"
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": dataset, "licenses": [{"name": "other"}]}, indent=2) + "\n")
    if args.dry_run:
        print(f"staged {len(names)} wheels ({', '.join(names)}), SHA256SUMS {sha} in {folder} for {dataset}")
        return 0
    exists = kaggle("datasets", "status", dataset, check=False)
    if exists.returncode == 0 and exists.stdout.strip().lower() in ("ready", "pending", "error"):
        kaggle("datasets", "version", "-p", str(folder), "-m", f"{len(names)} wheels, SHA256SUMS {sha[:12]}")
    else:
        # private unless --public is given; Kaggle can answer a create that worked with a non-JSON error, so the
        # wait below, not this exit code, decides
        kaggle("datasets", "create", "-p", str(folder), check=False)
    t0 = time.time()
    while time.time() - t0 < READY_SECONDS:
        time.sleep(15)
        if kaggle("datasets", "status", dataset, check=False).stdout.strip().lower() == "ready":
            break
    else:
        raise SystemExit(f"dataset {dataset} was not ready after {READY_SECONDS // 60} minutes")
    out = f"dataset={dataset}\nsha256={sha}\n"
    print(out)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
