#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/check_gate.sh — eagle C++ suite integrity gate.
#
# Normative source: the test-count integrity design.
#
# WHY THIS EXISTS. gtest exits 0 having run NOTHING when a filter matches nothing or the
# source glob came up empty: `--gtest_filter=NoSuchTest` prints "[  PASSED  ] 0 tests." and
# returns 0. The exit code cannot tell "everything passed" from "there was nothing to run".
# Worse, an ambient GTEST_FILTER env var silences the run and the listing IDENTICALLY, so a
# self-referential "ran == listed" invariant agrees with itself while running a subset. The
# only non-self-referential reference is a committed name manifest — this script compares
# both the run and an ENV-SCRUBBED listing against git, never against each other.
#
# Usage:
#   tests/check_gate.sh <cuda|cpp> <path/to/eagle_tests>
#   tests/check_gate.sh <cuda|cpp> <path/to/eagle_tests> --remint [--allow-removals]
#
# There is deliberately NO count argument: the expectation is the manifest data file
# tests/expected_tests_<mode>.txt, never a literal on a command line or in CI YAML.
#
# Checks (C1-C6):
#   C1  a gtest summary line exists and reports ran >= 1
#   C2  ran == manifest line count
#   C3  env-scrubbed live listing set == manifest set, BOTH directions
#   C4  no summary line => RED (abort path)
#   C5  the binary's own rc != 0 => RED even when every count matches (the `| tee`
#       in the old template handed the pipeline status to tee; here rc is taken from
#       PIPESTATUS[0] and judged on its own)
#   C6  manifest hygiene: non-empty, newline-terminated, no blanks/comments, sorted, unique
#
set -u -o pipefail

PROG="tests/check_gate.sh"
BIN_LABEL="eagle_tests"

usage() {
    cat >&2 <<EOF
usage: $PROG <cuda|cpp> <path/to/$BIN_LABEL> [--remint [--allow-removals]]

  <cuda|cpp>          selects tests/expected_tests_<mode>.txt as the pinned manifest
  --remint            regenerate that manifest from this binary (local only; refused in CI)
  --allow-removals    required when a re-mint would DELETE manifest lines

No count is ever passed on the command line — the manifest file is the expectation.
EOF
    exit 2
}

MODE=""
BIN=""
REMINT=0
ALLOW_REMOVALS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --remint)         REMINT=1 ;;
        --allow-removals) ALLOW_REMOVALS=1 ;;
        -h|--help)        usage ;;
        -*)               echo "$PROG: unknown option '$1'" >&2; usage ;;
        *)
            if   [ -z "$MODE" ]; then MODE="$1"
            elif [ -z "$BIN"  ]; then BIN="$1"
            else echo "$PROG: unexpected argument '$1'" >&2; usage
            fi ;;
    esac
    shift
done
[ -n "$MODE" ] && [ -n "$BIN" ] || usage
case "$MODE" in cuda|cpp) ;; *) echo "$PROG: mode must be 'cuda' or 'cpp', got '$MODE'" >&2; usage ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/expected_tests_${MODE}.txt"

if [ ! -x "$BIN" ]; then
    echo "GATE RED: '$BIN' is not an executable file" >&2
    exit 1
fi
BIN_ABS="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
BIN_DIR="$(dirname "$BIN_ABS")"
BIN_EXE="./$(basename "$BIN_ABS")"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

REMINT_CMD="$PROG $MODE $BIN --remint"

# ---------------------------------------------------------------------------
# Listing normalisation. gtest prints
#     Suite.                       [# TypeParam = ...]
#       TestName                   [# GetParam() = ...]
# Comment tails are stripped (they carry TypeParam/GetParam text and would make the
# manifest churn on unrelated edits); the result is one full Suite.Test name per line.
# ---------------------------------------------------------------------------
normalize() {
    awk '
        /^[^ \t]/ {
            line = $0
            sub(/[ \t]*#.*$/, "", line); sub(/[ \t]+$/, "", line)
            if (line ~ /\.$/) { suite = line } else { suite = "" }
            next
        }
        /^[ \t]+[^ \t]/ {
            if (suite == "") next
            line = $0
            sub(/^[ \t]+/, "", line); sub(/[ \t]*#.*$/, "", line); sub(/[ \t]+$/, "", line)
            if (line != "") print suite line
        }
    '
}

# LOCK: the LISTING is env-scrubbed. An ambient GTEST_FILTER filters the listing too,
# which is exactly how a self-referential invariant is defeated.
# LOCK: only flags probed present in this binary's --help are used here.
list_tests_scrubbed() {
    ( cd "$BIN_DIR" && env -u GTEST_FILTER -u TESTBRIDGE_TEST_ONLY \
        "$BIN_EXE" --gtest_list_tests ) 2>"$TMP/list.err"
}

# ---------------------------------------------------------------------------
# --remint
# ---------------------------------------------------------------------------
if [ "$REMINT" -eq 1 ]; then
    # LOCK: CI compares, never regenerates. Regeneration inside the gate is the
    # self-fulfilling trap the whole mechanism exists to prevent.
    if [ -n "${CI:-}" ]; then
        echo "REFUSED: --remint is a LOCAL command; \$CI is set." >&2
        echo "         CI compares against the committed manifest and never regenerates it." >&2
        exit 2
    fi
    list_tests_scrubbed | normalize | LC_ALL=C sort -u > "$TMP/new.txt"
    LIST_RC=${PIPESTATUS[0]}
    if [ "$LIST_RC" -ne 0 ]; then
        echo "REFUSED: listing the binary failed (rc $LIST_RC); refusing to mint from it." >&2
        sed 's/^/  | /' "$TMP/list.err" >&2
        exit 1
    fi
    if [ ! -s "$TMP/new.txt" ]; then
        echo "REFUSED: the binary listed ZERO tests — refusing to mint an empty manifest." >&2
        exit 1
    fi
    if [ -f "$MANIFEST" ]; then
        LC_ALL=C comm -23 "$MANIFEST" "$TMP/new.txt" > "$TMP/removed.txt"
        LC_ALL=C comm -13 "$MANIFEST" "$TMP/new.txt" > "$TMP/added.txt"
        if [ -s "$TMP/removed.txt" ] && [ "$ALLOW_REMOVALS" -ne 1 ]; then
            echo "" >&2
            echo "  ####################################################################" >&2
            echo "  #  RE-MINT REFUSED: this would REMOVE $(wc -l < "$TMP/removed.txt") test(s) from the manifest." >&2
            echo "  #  Tests do not normally disappear. Read this list before you agree:" >&2
            echo "  ####################################################################" >&2
            sed 's/^/  - /' "$TMP/removed.txt" >&2
            echo "" >&2
            echo "  If every removal above is intended, re-run with:" >&2
            echo "      $REMINT_CMD --allow-removals" >&2
            echo "  and commit the manifest diff IN THE SAME COMMIT as the test change." >&2
            exit 1
        fi
        sed 's/^/  - /' "$TMP/removed.txt"
        sed 's/^/  + /' "$TMP/added.txt"
    fi
    cp "$TMP/new.txt" "$MANIFEST"
    echo "re-minted $MANIFEST ($(wc -l < "$MANIFEST") tests)"
    echo "Commit this manifest diff IN THE SAME COMMIT as the test change."
    exit 0
fi

# ---------------------------------------------------------------------------
# C6 — manifest hygiene. A malformed manifest is a broken instrument, not a pass.
# ---------------------------------------------------------------------------
RED=0
red() { echo "GATE RED: $*" >&2; RED=1; }

if [ ! -f "$MANIFEST" ]; then
    echo "GATE RED: no manifest at $MANIFEST — mint it with: $REMINT_CMD" >&2
    exit 1
fi
EXPECTED=$(awk 'END{print NR}' "$MANIFEST")
if [ "$EXPECTED" -lt 1 ]; then
    red "[C6] manifest $MANIFEST is EMPTY — an empty expectation certifies nothing"
fi
if [ -s "$MANIFEST" ] && [ "$(tail -c1 "$MANIFEST" | wc -l)" -ne 1 ]; then
    red "[C6] manifest $MANIFEST is not newline-terminated"
fi
# One 'Suite.Test' name per line and nothing else: this single shape test rejects blank
# lines, comments, stray whitespace and any hand-editing that smuggled a count in.
if grep -nvE '^[^[:space:]#]+\.[^[:space:]#]+$' "$MANIFEST" > "$TMP/shape.txt"; then
    red "[C6] manifest $MANIFEST has lines that are not a bare 'Suite.Test' name:"
    sed 's/^/         /' "$TMP/shape.txt" >&2
fi
if ! LC_ALL=C sort -c "$MANIFEST" 2>"$TMP/sortc.txt"; then
    red "[C6] manifest $MANIFEST is NOT sorted (LC_ALL=C): $(cat "$TMP/sortc.txt")"
fi
UNIQ=$(LC_ALL=C sort -u "$MANIFEST" | awk 'END{print NR}')
if [ "$UNIQ" -ne "$EXPECTED" ]; then
    red "[C6] manifest $MANIFEST has duplicate lines ($EXPECTED lines, $UNIQ unique):"
    LC_ALL=C sort "$MANIFEST" | uniq -d | sed 's/^/         /' >&2
fi

# ---------------------------------------------------------------------------
# C3 — env-scrubbed live listing vs the manifest, both directions.
# This simulates the "not built" class without a rebuild and, unlike ran-vs-listed,
# is not defeatable by an ambient filter.
# ---------------------------------------------------------------------------
list_tests_scrubbed | normalize | LC_ALL=C sort -u > "$TMP/listed.txt"
LIST_RC=${PIPESTATUS[0]}
LISTED=$(awk 'END{print NR}' "$TMP/listed.txt")
if [ "$LIST_RC" -ne 0 ]; then
    red "[C3] '--gtest_list_tests' failed with rc $LIST_RC"
    sed 's/^/         | /' "$TMP/list.err" >&2
fi
LC_ALL=C sort -u "$MANIFEST" > "$TMP/manifest.txt"
LC_ALL=C comm -23 "$TMP/manifest.txt" "$TMP/listed.txt" > "$TMP/missing.txt"
LC_ALL=C comm -13 "$TMP/manifest.txt" "$TMP/listed.txt" > "$TMP/extra.txt"
if [ -s "$TMP/missing.txt" ] || [ -s "$TMP/extra.txt" ]; then
    red "[C3] the built binary's test SET differs from $MANIFEST (listed $LISTED, pinned $EXPECTED)"
    if [ -s "$TMP/missing.txt" ]; then
        echo "         MISSING (pinned but NOT in the binary — deleted, renamed, or NOT BUILT):" >&2
        sed 's/^/         - /' "$TMP/missing.txt" >&2
    fi
    if [ -s "$TMP/extra.txt" ]; then
        echo "         UNEXPECTED (in the binary but NOT pinned — new tests, un-minted):" >&2
        sed 's/^/         + /' "$TMP/extra.txt" >&2
    fi
fi

# ---------------------------------------------------------------------------
# Run the suite. The RUN is deliberately NOT env-scrubbed (LOCK): an ambient
# GTEST_FILTER must make this gate RED, not be laundered into a clean run.
# ---------------------------------------------------------------------------
LOG="$BIN_DIR/gtest.log"
XML="$(basename "$BIN_ABS").xml"
echo "== $PROG: running $BIN_ABS (mode=$MODE, pinned $EXPECTED tests) =="
( cd "$BIN_DIR" && time "$BIN_EXE" --gtest_repeat=1 --gtest_break_on_failure \
    --gtest_shuffle --gtest_output="xml:$XML" ) 2>&1 | tee "$LOG"
BIN_RC=${PIPESTATUS[0]}

# C5 -- the binary's own rc, judged on its own. `| tee` above would otherwise
# hand the pipeline's status to tee and mask a failing suite entirely.
if [ "$BIN_RC" -ne 0 ]; then
    red "[C5] the test binary exited with rc $BIN_RC — the suite FAILED (counts are irrelevant)"
fi

# C1/C4 — the summary line must exist and report ran >= 1.
RAN=$(grep -oP '^\[==========\] \K[0-9]+(?= tests? from .* ran)' "$LOG" | tail -1)
if [ -z "$RAN" ]; then
    red "[C4] no gtest summary line in $LOG — the suite never reported a run (aborted, crashed, or not a gtest binary)"
elif [ "$RAN" -lt 1 ]; then
    red "[C1] the suite ran $RAN tests — gtest exits 0 having run NOTHING when a filter matches nothing"
elif [ "$RAN" -ne "$EXPECTED" ]; then
    # C2
    red "[C2] ran $RAN tests, manifest pins $EXPECTED"
    if [ "$RAN" -lt "$EXPECTED" ]; then
        echo "         A run smaller than the listing is the FILTER class: check GTEST_FILTER" >&2
        echo "         (GTEST_FILTER='${GTEST_FILTER:-<unset>}', TESTBRIDGE_TEST_ONLY='${TESTBRIDGE_TEST_ONLY:-<unset>}')" >&2
    fi
fi

echo "-----------------------------------------------------------------------"
echo "mode=$MODE  manifest=$MANIFEST  pinned=$EXPECTED  listed=$LISTED  ran=${RAN:-<none>}  binary_rc=$BIN_RC"
if [ "$RED" -ne 0 ]; then
    echo "" >&2
    echo "VERDICT: RED" >&2
    echo "If — and only if — the suite legitimately changed, re-mint DELIBERATELY with:" >&2
    echo "    $REMINT_CMD" >&2
    echo "and commit the manifest diff in the SAME commit as the test change. Never silence." >&2
    exit 1
fi
echo "VERDICT: GREEN"
exit 0
