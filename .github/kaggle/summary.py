# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# The verdict of the Kaggle GPU legs from their state files (pass | fail | unrun; no file = unrun).
# usage: summary.py STATE_DIR RELEASE(0|1) [LEG ...]   exit 1 = fail; writes STATE_DIR/.label = add|remove (gpu-inconclusive)
#   a failed leg is always red; legs that did not run are red on a release PR (no merge without GPU evidence) and a visible
#   warning on any other PR.
import pathlib
import sys

LEGS = ("cuda12-tests", "cuda13-tests", "cuda12-notebooks", "cuda13-notebooks")
QUOTA = "Kaggle GPU quota exhausted: GPU evidence missing; re-run after reset or record local GPU evidence"


def verdict(states, release):
    """(exit code, messages, label) from {leg: state}."""
    unrun = [leg for leg, s in states.items() if s == "unrun"]
    failed = [leg for leg, s in states.items() if s == "fail"]
    out = [f"{leg}: {s}" for leg, s in states.items()]
    code = 0
    if failed:
        out.append("::error::GPU legs failed: " + " ".join(failed))
        code = 1
    if unrun and release:
        out.append(f"::error::release PR, GPU legs not run ({' '.join(unrun)}): {QUOTA}")
        code = 1
    elif unrun:
        out.append(f"::warning::GPU legs not run ({' '.join(unrun)}); not a release PR, so the check stays green: {QUOTA}")
    return code, out, "add" if unrun else "remove"


def main(argv):
    folder, release, legs = pathlib.Path(argv[1]), argv[2] == "1", argv[3:] or LEGS
    states = {}
    for leg in legs:
        f = folder / leg
        states[leg] = f.read_text().strip() if f.exists() else "unrun"
    code, out, label = verdict(states, release)
    print("\n".join(out))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".label").write_text(label)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
