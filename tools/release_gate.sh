#!/usr/bin/env bash
# A pull request to main carries exactly one label release:patch|minor|major|rc|none. The release version is the top
# heading of CHANGELOG.md (`## 0.7.4 (date)`; `## Unreleased` is a dev build): the label is only checked against it.
# none = docs-only (docs/**, *.md, LICENSE, NOTICE) with no release heading; any other label needs the heading to be
# the version the label gives from the latest tag, a PR body (the release notes), and the tag must be free.
# Prints the version (nothing for none).
# usage: PR_LABELS="release:patch" PR_BODY="..." release_gate.sh <base-sha> <head-sha>   (a clone with tags, the PR commit checked out)
set -euo pipefail
base=$1 head=$2
err() { echo "::error::$*" >&2; exit 1; }
labels=()
for l in ${PR_LABELS:-}; do [[ $l == release:* ]] && labels+=("${l#release:}"); done
[ "${#labels[@]}" -eq 1 ] || err "need exactly one release:patch|minor|major|rc|none label, found ${#labels[@]}"
label=${labels[0]}
tool="$(dirname "$0")/next_version.py"
heading=$(python "$tool")
files=$(git diff --name-only "$base...$head")
docs_only=0
if [ -n "$files" ] && ! grep -vqE '^(docs/|LICENSE$|NOTICE$)|\.md$' <<< "$files"; then docs_only=1; fi
if [ "$label" = none ]; then
  [ "$docs_only" = 1 ] || err "release:none is for docs-only changes; this one touches code, tests, workflows or tools: label it patch, minor, major or rc"
  [ -z "$heading" ] || err "CHANGELOG.md is headed $heading, a release: label it release:patch|minor|major|rc, or head it Unreleased"
  echo "docs-only change: no release" >&2
  exit 0
fi
[ "$docs_only" = 0 ] || err "release:$label on a docs-only change: label it release:none"
[ -n "$(tr -d '[:space:]' <<< "${PR_BODY:-}")" ] || err "the PR body is the release notes: write it"
[ -n "$heading" ] || err "release:$label, but CHANGELOG.md's top heading is not a version above the latest tag: head it '## X.Y.Z (date)'"
v=$(python "$tool" "release:$label") || err "no version for label $label"
[ "$v" = "$heading" ] || err "release:$label gives $v from the latest tag, but CHANGELOG.md is headed $heading: relabel, or fix the heading"
if git rev-parse -q --verify "refs/tags/v$v" > /dev/null || [ -n "$(git ls-remote --tags origin "refs/tags/v$v" 2>/dev/null)" ]; then
  err "tag v$v already exists: rebase on main, or fix the heading"
fi
echo "release:$label -> $v" >&2
echo "$v"
