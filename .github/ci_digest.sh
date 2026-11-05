#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Prints a short sha256 over what a CI job's output depends on. Cache keys are built from it.
# usage: ci_digest.sh [--tree DIR [--skip PREFIX | --only PREFIX | --skip-ext SUFFIX]...]... [--file F]... [--version DIR] [--tag DIR] [--str S]...
#   --tree DIR     DIR's tracked files with their blob ids (`git ls-tree -r HEAD`); the --skip prefixes that follow
#                  name paths that do not enter the output (a prefix ending in / is a directory); the --only prefixes
#                  that follow name the only paths that do (none given: every path not skipped). A path is skipped
#                  first, then needs to match an --only prefix when there are any.
#                  --skip-ext SUFFIX skips every path that ends in SUFFIX (e.g. .ipynb: notebooks only some jobs run).
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
onlys=()
exts=()
{
  while [ $# -gt 0 ]; do
    case $1 in
      --tree)
        shift; dir=$1; shift; skips=(); onlys=(); exts=()
        while [ "${1:-}" = --skip ] || [ "${1:-}" = --only ] || [ "${1:-}" = --skip-ext ]; do
          case $1 in --skip) skips+=("$2") ;; --only) onlys+=("$2") ;; *) exts+=("$2") ;; esac
          shift 2
        done
        echo "tree $dir"
        git -C "$dir" ls-tree -r HEAD | awk -F'\t' -v skips="${skips[*]:-}" -v onlys="${onlys[*]:-}" -v exts="${exts[*]:-}" '
          BEGIN { n = split(skips, s, " "); m = split(onlys, o, " "); k = split(exts, x, " ") }
          { for (i = 1; i <= n; i++) if (index($2, s[i]) == 1) next
            for (i = 1; i <= k; i++) if (length($2) >= length(x[i]) && substr($2, length($2) - length(x[i]) + 1) == x[i]) next
            if (m > 0) { keep = 0; for (i = 1; i <= m; i++) if (index($2, o[i]) == 1) keep = 1; if (!keep) next }
            print }' ;;
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
