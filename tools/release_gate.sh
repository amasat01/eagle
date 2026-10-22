#!/usr/bin/env bash
# Every merge to main is a release unless it only touches docs: docs/**, *.md, LICENSE, NOTICE.
# usage: release_gate.sh <base-sha> <head-sha>   (run from a clone that has both commits and an `origin`)
set -euo pipefail
base=$1 head=$2
files=$(git diff --name-only "$base...$head")
if [ -n "$files" ] && ! grep -vqE '^(docs/|LICENSE$|NOTICE$)|\.md$' <<< "$files"; then
  echo "docs-only change: no new version needed"
  exit 0
fi
tmp=$(mktemp -d)
trap 'git worktree remove --force "$tmp/head" 2>/dev/null; git worktree remove --force "$tmp/base" 2>/dev/null; rmdir "$tmp"' EXIT
version_at() {  # empty when that commit has no tools/version.py yet
  git worktree add -q --detach "$tmp/$1" "$2"
  [ -f "$tmp/$1/tools/version.py" ] && python "$tmp/$1/tools/version.py" get || true
}
v=$(version_at head "$head")
vb=$(version_at base "$base")
[ -n "$v" ] || { echo "::error::no tools/version.py at the head commit"; exit 1; }
if [ "$v" = "$vb" ]; then
  echo "::error::version $v is unchanged from main: run 'python tools/version.py bump X.Y.Z' (only docs-only changes skip it)"
  exit 1
fi
if [ -n "$(git ls-remote --tags origin "refs/tags/v$v")" ]; then
  echo "::error::tag v$v already exists: run 'python tools/version.py bump X.Y.Z' with a new version"
  exit 1
fi
echo "new version $v (main has ${vb:-none}): release"
