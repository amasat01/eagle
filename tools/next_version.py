#!/usr/bin/env python3
"""Next version from the latest `vX.Y.Z[rcN]` tag reachable from a commit and a `release:*` label.

usage: next_version.py "<labels>" [<commit>]   labels: the PR's label list, one `release:patch|minor|major|rc|none` among them
(a bare `patch` works too), or an exact `release:X.Y.Z[rcN]` above the latest tag (to keep sibling packages in step);
commit defaults to HEAD. Prints the version (no `v`), nothing for `none`.
Stdlib only; runs on every Python the packages support (3.9+).
"""
import re
import subprocess
import sys

LABELS = ("patch", "minor", "major", "rc", "none")
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$")
EXACT = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$")


def order(g):
    """Sort key of (major, minor, patch, rc|None): 0.6.5rc1 < 0.6.5."""
    return (int(g[0]), int(g[1]), int(g[2]), int(g[3]) if g[3] else float("inf"))


def latest(commit):
    """(major, minor, patch, rc|None) of the highest release tag reachable from commit; (0, 0, 0, None) if none."""
    out = subprocess.run(["git", "tag", "--list", "v*", "--merged", commit], capture_output=True, text=True, check=True).stdout
    found = [m.groups() for m in map(TAG.match, out.split()) if m]
    best = max(found, key=order, default=None)
    return (int(best[0]), int(best[1]), int(best[2]), int(best[3]) if best[3] else None) if best else (0, 0, 0, None)


def bump(cur, label):
    major, minor, patch, rc = cur
    if label == "rc":
        return f"{major}.{minor}.{patch}rc{rc + 1}" if rc else f"{major}.{minor}.{patch + 1}rc1"
    if label == "patch":  # the rc's own release when the latest tag is an rc, else the next patch
        return f"{major}.{minor}.{patch}" if rc else f"{major}.{minor}.{patch + 1}"
    if label == "minor":
        return f"{major}.{minor + 1}.0"
    if label == "major":
        return f"{major + 1}.0.0"
    exact = EXACT.match(label)
    if exact:
        if order(exact.groups()) <= order(cur):
            sys.exit(f"next_version.py: release:{label} is not above the latest tag v{major}.{minor}.{patch}{f'rc{rc}' if rc else ''}")
        return label
    sys.exit(f"next_version.py: unknown label {label!r} (patch | minor | major | rc | X.Y.Z[rcN])")


def label_of(text):
    """The one release label in a whitespace-separated label list."""
    found = [t[len("release:"):] for t in text.split() if t.startswith("release:")] or ([text.strip()] if text.strip() in LABELS else [])
    if len(found) != 1:
        sys.exit(f"next_version.py: need exactly one release:* label, found {len(found)} in {text!r}")
    return found[0]


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    label = label_of(sys.argv[1])
    if label != "none":
        print(bump(latest(sys.argv[2] if len(sys.argv) == 3 else "HEAD"), label))
