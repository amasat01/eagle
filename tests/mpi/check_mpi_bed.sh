#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/mpi/check_mpi_bed.sh -- integrity gate for the 2-rank MPI bed.
#
# The MPI twin of tests/check_gate.sh, and deliberately its mirror image: the
# expectation is a COMMITTED NAME MANIFEST (tests/expected_tests_mpi_<mode>.txt),
# never a count on a command line; the LISTING is env-scrubbed while the RUN is
# not; and `--remint` is refused when $CI is set. Read that script first — the
# reasoning behind every one of those choices is written out there and not
# repeated here.
#
# WHAT IS DIFFERENT, AND WHY. A distributed suite has one failure mode a
# single-process suite does not: a run in which only ONE rank actually did
# anything. `mpirun` returns 0 when every rank returns 0, and a rank that never
# reached a row returns 0 too. So this gate does not judge the launcher's exit
# code alone — it reads BOTH ranks' XML reports and requires
#
#     pinned == listed == ran(rank 0)   AND   verdict(rank 1) == verdict(rank 0)
#
# where a verdict is the triple (tests, failures, errors). The reports are named
# per rank by the bed's own main() (see main_mpi_bed.cpp), and this script DELETES
# them before the run: a gate that reads whatever XML is lying around certifies
# the previous run.
#
# NON-VACUITY IS PROVEN, NOT ASSUMED. `--gtest_filter=NoSuchTest` makes the
# bed print "[  PASSED  ] 0 tests." and exit 0 on both ranks, and mpirun then
# reports success — the exact "green having run nothing" the manifest exists to
# catch. This script runs that probe through the SAME judgement it applies to the
# real run and REFUSES to report GREEN unless the probe comes out RED.
#
# Usage:
#   tests/mpi/check_mpi_bed.sh <cpp|cuda> <path/to/eagle_mpi_bed>
#   tests/mpi/check_mpi_bed.sh <cpp|cuda> <path/to/eagle_mpi_bed> --remint [--allow-removals]
#
# Environment:
#   MPIRUN                 the launcher (default: `mpirun` from $PATH)
#   EAGLE_MPI_BED_RANKS    the world size (default: 2 — the bed is a 2-rank bed)
#
set -u -o pipefail

PROG="tests/mpi/check_mpi_bed.sh"
BIN_LABEL="eagle_mpi_bed"
RANKS="${EAGLE_MPI_BED_RANKS:-2}"
MPIRUN="${MPIRUN:-mpirun}"

usage() {
    cat >&2 <<EOF
usage: $PROG <cpp|cuda> <path/to/$BIN_LABEL> [--remint [--allow-removals]]

  <cpp|cuda>          selects tests/expected_tests_mpi_<mode>.txt as the pinned manifest
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
MANIFEST="$SCRIPT_DIR/../expected_tests_mpi_${MODE}.txt"

if [ ! -x "$BIN" ]; then
    echo "GATE RED: '$BIN' is not an executable file" >&2
    exit 1
fi
if ! command -v "$MPIRUN" > /dev/null 2>&1; then
    echo "GATE RED: launcher '$MPIRUN' not found on \$PATH (set \$MPIRUN)" >&2
    exit 1
fi
BIN_ABS="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
BIN_DIR="$(dirname "$BIN_ABS")"
BIN_EXE="./$(basename "$BIN_ABS")"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

REMINT_CMD="$PROG $MODE $BIN --remint"

# ---------------------------------------------------------------------------
# Listing normalisation — byte-identical to tests/check_gate.sh's `normalize`,
# because the manifests it produces must be the same shape.
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

# The LISTING comes from a ONE-rank world: the test SET is a compile-time property,
# and two ranks listing into one merged stdout would interleave two copies of it.
# Env-scrubbed for the same reason check_gate.sh scrubs its own: an ambient
# GTEST_FILTER filters the listing exactly as it filters the run, so a listing that
# inherited it would agree with a filtered run about a suite neither of them ran.
list_tests_scrubbed() {
    ( cd "$BIN_DIR" && env -u GTEST_FILTER -u TESTBRIDGE_TEST_ONLY \
        "$MPIRUN" -np 1 --oversubscribe "$BIN_EXE" --gtest_list_tests ) 2>"$TMP/list.err"
}

# ---------------------------------------------------------------------------
# --remint
# ---------------------------------------------------------------------------
if [ "$REMINT" -eq 1 ]; then
    if [ -n "${CI:-}" ]; then
        echo "REFUSED: --remint is a LOCAL command; \$CI is set." >&2
        echo "         CI compares against the committed manifest and never regenerates it." >&2
        exit 2
    fi
    list_tests_scrubbed | normalize | LC_ALL=C sort -u > "$TMP/new.txt"
    LIST_RC=${PIPESTATUS[0]}
    if [ "$LIST_RC" -ne 0 ]; then
        echo "REFUSED: listing the bed failed (rc $LIST_RC); refusing to mint from it." >&2
        sed 's/^/  | /' "$TMP/list.err" >&2
        exit 1
    fi
    if [ ! -s "$TMP/new.txt" ]; then
        echo "REFUSED: the bed listed ZERO tests — refusing to mint an empty manifest." >&2
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

RED=0
red() { echo "GATE RED: $*" >&2; RED=1; }

# ---------------------------------------------------------------------------
# Manifest hygiene (verbatim in intent, matching check_gate.sh).
# ---------------------------------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "GATE RED: no manifest at $MANIFEST — mint it with: $REMINT_CMD" >&2
    exit 1
fi
EXPECTED=$(awk 'END{print NR}' "$MANIFEST")
if [ "$EXPECTED" -lt 1 ]; then
    red "[M1] manifest $MANIFEST is EMPTY — an empty expectation certifies nothing"
fi
if [ -s "$MANIFEST" ] && [ "$(tail -c1 "$MANIFEST" | wc -l)" -ne 1 ]; then
    red "[M1] manifest $MANIFEST is not newline-terminated"
fi
if grep -nvE '^[^[:space:]#]+\.[^[:space:]#]+$' "$MANIFEST" > "$TMP/shape.txt"; then
    red "[M1] manifest $MANIFEST has lines that are not a bare 'Suite.Test' name:"
    sed 's/^/         /' "$TMP/shape.txt" >&2
fi
if ! LC_ALL=C sort -c "$MANIFEST" 2>"$TMP/sortc.txt"; then
    red "[M1] manifest $MANIFEST is NOT sorted (LC_ALL=C): $(cat "$TMP/sortc.txt")"
fi
UNIQ=$(LC_ALL=C sort -u "$MANIFEST" | awk 'END{print NR}')
if [ "$UNIQ" -ne "$EXPECTED" ]; then
    red "[M1] manifest $MANIFEST has duplicate lines ($EXPECTED lines, $UNIQ unique):"
    LC_ALL=C sort "$MANIFEST" | uniq -d | sed 's/^/         /' >&2
fi

# ---------------------------------------------------------------------------
# Env-scrubbed listing vs the manifest, both directions.
# ---------------------------------------------------------------------------
list_tests_scrubbed | normalize | LC_ALL=C sort -u > "$TMP/listed.txt"
LIST_RC=${PIPESTATUS[0]}
LISTED=$(awk 'END{print NR}' "$TMP/listed.txt")
if [ "$LIST_RC" -ne 0 ]; then
    red "[M2] listing the bed under '$MPIRUN -np 1' failed with rc $LIST_RC"
    sed 's/^/         | /' "$TMP/list.err" >&2
fi
LC_ALL=C sort -u "$MANIFEST" > "$TMP/manifest.txt"
LC_ALL=C comm -23 "$TMP/manifest.txt" "$TMP/listed.txt" > "$TMP/missing.txt"
LC_ALL=C comm -13 "$TMP/manifest.txt" "$TMP/listed.txt" > "$TMP/extra.txt"
if [ -s "$TMP/missing.txt" ] || [ -s "$TMP/extra.txt" ]; then
    red "[M2] the built bed's test SET differs from $MANIFEST (listed $LISTED, pinned $EXPECTED)"
    if [ -s "$TMP/missing.txt" ]; then
        echo "         MISSING (pinned but NOT in the bed — deleted, renamed, or NOT BUILT):" >&2
        sed 's/^/         - /' "$TMP/missing.txt" >&2
    fi
    if [ -s "$TMP/extra.txt" ]; then
        echo "         UNEXPECTED (in the bed but NOT pinned — new rows, un-minted):" >&2
        sed 's/^/         + /' "$TMP/extra.txt" >&2
    fi
fi

# ---------------------------------------------------------------------------
# The run + per-rank verdicts.
#
# `run_bed <log> <extra gtest args...>` deletes the stale reports, runs the bed
# under `$MPIRUN -np $RANKS`, and leaves the launcher's rc in RUN_RC and each
# rank's "<tests> <failures> <errors>" in VERDICT_<r> (empty when that rank wrote
# no report at all — a crash, or a rank that never got as far as RUN_ALL_TESTS).
# ---------------------------------------------------------------------------
XML_BASE="$BIN_DIR/$(basename "$BIN_ABS")"
run_bed() {
    local log="$1"; shift
    local r
    for (( r = 0; r < RANKS; ++r )); do rm -f "${XML_BASE}_rank${r}.xml"; done
    ( cd "$BIN_DIR" && "$MPIRUN" -np "$RANKS" --oversubscribe \
        "$BIN_EXE" --gtest_output="xml:$(basename "$BIN_ABS").xml" "$@" ) > "$log" 2>&1
    RUN_RC=$?
    VERDICTS=()
    for (( r = 0; r < RANKS; ++r )); do
        local xml="${XML_BASE}_rank${r}.xml"
        if [ ! -s "$xml" ]; then VERDICTS+=(""); continue; fi
        VERDICTS+=("$(grep -m1 -oP '<testsuites [^>]*' "$xml" \
            | grep -oP '(tests|failures|errors)="\K[0-9]+' | tr '\n' ' ' | sed 's/ $//')")
    done
}

# `judge` applies the SAME rules to whatever run_bed last produced, appending its
# complaints to $1. It returns 0 for a clean run and 1 otherwise — which is what
# lets the non-vacuity probe below be checked with the real judgement rather than
# with a weaker copy of it.
judge() {
    local out="$1"
    local bad=0 r
    [ "$RUN_RC" -ne 0 ] && { echo "[M3] '$MPIRUN -np $RANKS' exited with rc $RUN_RC" >> "$out"; bad=1; }
    for (( r = 0; r < RANKS; ++r )); do
        if [ -z "${VERDICTS[$r]}" ]; then
            echo "[M4] rank $r wrote no XML report — it never reported a run (crashed, aborted, or deadlocked)" >> "$out"
            bad=1
        fi
    done
    [ "$bad" -eq 1 ] && return 1
    local ran fail err
    read -r ran fail err <<< "${VERDICTS[0]}"
    if [ "$ran" -lt 1 ]; then
        echo "[M5] rank 0 ran $ran tests — gtest exits 0 having run NOTHING when a filter matches nothing" >> "$out"
        bad=1
    elif [ "$ran" -ne "$EXPECTED" ]; then
        echo "[M6] rank 0 ran $ran tests, manifest pins $EXPECTED (GTEST_FILTER='${GTEST_FILTER:-<unset>}')" >> "$out"
        bad=1
    fi
    if [ "$fail" -ne 0 ] || [ "$err" -ne 0 ]; then
        echo "[M7] rank 0 reported $fail failure(s) and $err error(s)" >> "$out"
        bad=1
    fi
    for (( r = 1; r < RANKS; ++r )); do
        if [ "${VERDICTS[$r]}" != "${VERDICTS[0]}" ]; then
            echo "[M8] rank $r's verdict (tests failures errors) = '${VERDICTS[$r]}' differs from rank 0's '${VERDICTS[0]}'" >> "$out"
            bad=1
        fi
    done
    return $bad
}

# The REAL run. Deliberately NOT env-scrubbed: an ambient GTEST_FILTER must make
# this gate RED, not be laundered into a clean run.
echo "== $PROG: running $MPIRUN -np $RANKS $BIN_ABS (mode=$MODE, pinned $EXPECTED tests) =="
run_bed "$BIN_DIR/mpi_bed.log"
RUN_VERDICTS=("${VERDICTS[@]}")
RUN_LAUNCH_RC=$RUN_RC
: > "$TMP/complaints.txt"
if ! judge "$TMP/complaints.txt"; then
    while IFS= read -r line; do red "$line"; done < "$TMP/complaints.txt"
    sed -n '$p;/FAILED/p' "$BIN_DIR/mpi_bed.log" | sed 's/^/         | /' >&2
fi

# ---------------------------------------------------------------------------
# NON-VACUITY. The instrument must be able to fail: run the bed with a
# filter that matches nothing and require the judgement above to call it RED.
# ---------------------------------------------------------------------------
echo "== $PROG: non-vacuity probe (--gtest_filter=NoSuchTest must come out RED) =="
run_bed "$TMP/probe.log" --gtest_filter=NoSuchTest
PROBE_VERDICT="${VERDICTS[0]:-<no report>}"
: > "$TMP/probe_complaints.txt"
if judge "$TMP/probe_complaints.txt"; then
    red "[M9] the NoSuchTest probe came out GREEN (rank 0 verdict '$PROBE_VERDICT') — this gate cannot fail and certifies nothing"
else
    echo "   probe RED as required: rank 0 verdict (tests failures errors) = '$PROBE_VERDICT'"
    sed 's/^/     probe: /' "$TMP/probe_complaints.txt"
fi

echo "-----------------------------------------------------------------------"
echo "mode=$MODE  manifest=$MANIFEST  ranks=$RANKS  pinned=$EXPECTED  listed=$LISTED  launcher_rc=$RUN_LAUNCH_RC"
for (( i = 0; i < RANKS; ++i )); do
    echo "  rank $i verdict (tests failures errors) = '${RUN_VERDICTS[$i]:-<no report>}'"
done
if [ "$RED" -ne 0 ]; then
    echo "" >&2
    echo "VERDICT: RED" >&2
    echo "If — and only if — the bed's row set legitimately changed, re-mint DELIBERATELY with:" >&2
    echo "    $REMINT_CMD" >&2
    echo "and commit the manifest diff in the SAME commit as the row change. Never silence." >&2
    exit 1
fi
echo "VERDICT: GREEN"
exit 0
