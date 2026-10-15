#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Freshness checks for the documented numbers and the family visuals; never writes.
# Used by ci.yml and publish.yml. Each check has its own exit code:
#   0 fresh, 1 stale, 3 a card the docs cite is missing, 4 tool or companion checkout missing.
# Usage: tools/docs_fresh.sh [site-checkout]      (default: site)
set -u
site="${1:-site}"
status=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# check <label> <regenerate command> <command...>
check() {
  local label="$1" regen="$2"
  shift 2
  "$@" >"$tmp/out" 2>"$tmp/err"
  local rc=$?
  cat "$tmp/out"
  cat "$tmp/err" >&2
  [ "$rc" -eq 0 ] && return 0
  local line
  line="$(tail -n 1 "$tmp/err")"
  [ -n "$line" ] || line="$(tail -n 1 "$tmp/out")"
  status=1
  local kind="$rc"
  # an uncaught exception also exits 1; it is a crash, not a stale page
  if [ "$rc" -eq 1 ] && grep -q '^Traceback' "$tmp/err"; then kind=crash; fi
  case "$kind" in
    1) echo "::error::$label out of date: regenerate with: $regen" ;;
    3) echo "::error::$label: a card the docs cite is missing ($line): land the card or drop the claim, regeneration will not help" ;;
    4) echo "::error::$label: generator or companion missing: $line: check that the paired companion branch has the tool" ;;
    *) echo "::error::$label: generator crashed (rc $rc): $line" ;;
  esac
  return 0
}

check "documented performance numbers" "python tools/sync_readme_cards.py, then commit the result" \
  python tools/sync_readme_cards.py --check

if [ -f "$site/tools/sync_family_visuals.py" ]; then
  check "family visuals" "python tools/sync_family_visuals.py --site-root <site checkout> --eagle-root <eagle checkout> --target eagle=<this repo>, then commit docs/_static/family_visuals" \
    python "$site/tools/sync_family_visuals.py" --check --site-root "$site" --eagle-root . --target eagle=.
else
  echo "family visuals: $site/tools/sync_family_visuals.py not found" >&2
  echo "::error::family visuals: generator or companion missing: $site/tools/sync_family_visuals.py not found: check out the companion site into $site" 
  status=1
fi
exit "$status"
