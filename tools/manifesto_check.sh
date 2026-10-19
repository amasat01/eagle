#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Checks that each given file holds a verbatim copy of the family manifesto between
# <!-- manifesto:begin --> and <!-- manifesto:end -->. The source is the "## Manifesto" section of raptor's
# principles.md: fetched from raptor main, or read from the path in $MANIFESTO_SOURCE.
# Usage: tools/manifesto_check.sh FILE...   Exit: 0 same, 1 a file differs, 4 source unavailable.
set -u
url=https://raw.githubusercontent.com/amasat01/raptor/main/docs/content/principles.md
if [ -n "${MANIFESTO_SOURCE:-}" ]; then src="$(cat "$MANIFESTO_SOURCE")" || exit 4
else src="$(curl -fsSL "$url")" || { echo "::error::cannot fetch $url" >&2; exit 4; }; fi
want="$(printf '%s\n' "$src" | awk '/^## Manifesto/{on=1; next} on && NF{print; if (++n == 7) exit}')"
[ "$(printf '%s\n' "$want" | wc -l)" -eq 7 ] || { echo "::error::manifesto section not found in source" >&2; exit 4; }
status=0
for f in "$@"; do
  have="$(awk '/<!-- manifesto:end -->/{on=0} on{print} /<!-- manifesto:begin -->/{on=1}' "$f")"
  [ "$have" = "$want" ] || { echo "::error file=$f::manifesto block differs from raptor docs/content/principles.md" >&2; status=1; }
done
exit "$status"
