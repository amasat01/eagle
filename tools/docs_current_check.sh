#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Usage: docs_current_check.sh <repo> [<expected-version>]
# Read-only. Fetch the live documentation of <repo> (amasat01.github.io/<repo>/, or the user site for
# amasat01.github.io) and fail unless
#   - its _static/documentation_options.js VERSION is the expected one (default: the newest release tag of the
#     repo, of raptor for the user site), and is never a development string, and
#   - sitemap.xml is served and carries <lastmod>.
# RETRIES (default 1) and SLEEP seconds (default 15) wait for Pages to publish after a deploy.
set -euo pipefail
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { echo "usage: $0 <repo> [<expected-version>]" >&2; exit 2; }
repo=$1 want=${2:-}
retries=${RETRIES:-1} pause=${SLEEP:-15}
if [ "$repo" = amasat01.github.io ]; then url=https://amasat01.github.io/; tagrepo=raptor; else url=https://amasat01.github.io/$repo/; tagrepo=$repo; fi
if [ -z "$want" ]; then
  want=$(git ls-remote --tags --refs "https://github.com/amasat01/${tagrepo}" 'v*' | sed 's#.*refs/tags/v##' \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
fi
[ -n "$want" ] || { echo "::error::no release tag found for ${tagrepo}" >&2; exit 1; }
for attempt in $(seq 1 "$retries"); do
  got=$(curl -fsS "${url}_static/documentation_options.js" | sed -n "s/.*VERSION: *'\([^']*\)'.*/\1/p" | head -1 || true)
  ok=1
  [ "$got" = "$want" ] || ok=0
  case "$got" in *dev*|*+*) ok=0 ;; esac
  if [ "$ok" = 1 ] && curl -fsS "${url}sitemap.xml" | grep -q '<lastmod>'; then
    echo "${url} shows ${got}, the expected version, and serves a sitemap with lastmod"
    exit 0
  fi
  echo "attempt ${attempt}/${retries}: ${url} shows '${got}', expected '${want}'"
  [ "$attempt" = "$retries" ] || sleep "$pause"
done
echo "::error::the live documentation at ${url} is not current: shows '${got}', expected '${want}' (or sitemap.xml missing)"
exit 1
