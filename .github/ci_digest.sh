#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Prints a short sha256 over what a CI job's output depends on. Cache keys are built from it.
# usage: ci_digest.sh [--tree DIR [--skip PREFIX]...]... [--file F]... [--version DIR] [--tag DIR] [--str S]...
#   --tree DIR     DIR's tracked files with their blob ids (`git ls-tree -r HEAD`); the --skip prefixes that follow
#                  name paths that do not enter the output (a prefix ending in / is a directory)
#   --file F       the file's content (the workflow that runs the job goes here)
#   --version DIR  the version the checkout builds as: its exact tag, else git describe (carries the commit)
#   --tag DIR      only its exact tag, else `untagged` (for jobs whose result does not depend on the commit id)
#   --str S        any other input: a tool pin, the platform, an env value
set -euo pipefail
# macOS has no sha256sum; shasum ships there. No hasher at all is an error: an empty digest would make every key equal.
if command -v sha256sum >/dev/null; then sha256() { sha256sum; }
elif command -v shasum >/dev/null; then sha256() { shasum -a 256; }
else echo "ci_digest.sh: neither sha256sum nor shasum found" >&2; exit 2; fi
skips=()
{
  while [ $# -gt 0 ]; do
    case $1 in
      --tree)
        shift; dir=$1; shift; skips=()
        while [ "${1:-}" = --skip ]; do skips+=("$2"); shift 2; done
        echo "tree $dir"
        git -C "$dir" ls-tree -r HEAD | awk -F'\t' -v skips="${skips[*]:-}" '
          BEGIN { n = split(skips, s, " ") }
          { for (i = 1; i <= n; i++) if (index($2, s[i]) == 1) next; print }' ;;
      --file) echo "file $2"; sha256 < "$2"; shift 2 ;;
      --version)
        echo "version $2: $(git -C "$2" describe --tags --match 'v*' --exact-match 2>/dev/null \
          || git -C "$2" describe --tags --long --match 'v*' --always)"; shift 2 ;;
      --tag) echo "tag $2: $(git -C "$2" describe --tags --match 'v*' --exact-match 2>/dev/null || echo untagged)"; shift 2 ;;
      --str) echo "str $2"; shift 2 ;;
      *) echo "ci_digest.sh: unknown argument $1" >&2; exit 2 ;;
    esac
  done
} | sha256 | cut -c1-16
