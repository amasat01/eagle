#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# python/tests/mpi/check_rank_bed.sh — integrity gate for the Python rank bed.
#
# The PYTHON MIRROR of tests/mpi/check_mpi_bed.sh, which is itself the MPI mirror
# of tests/check_gate.sh. Read those two first: the reasoning behind a COMMITTED
# NAME MANIFEST rather than a count, behind reading BOTH ranks' reports rather
# than trusting the launcher's exit code, and behind refusing `--remint` when $CI
# is set, is written out there and is not repeated here.
#
# The judgement, unchanged in shape from the C++ bed's:
#
#     pinned == collected == ran(rank 0)   AND   verdict(rank 1) == verdict(rank 0)
#
# where a verdict is (tests, failures, errors, skipped) read out of that rank's own
# JUnit XML — SKIPPED included, because a skipped row is pinned but certifies
# nothing, which is the vacuity failure at the granularity of one row. `mpirun` returns 0 when every rank returns 0, and a rank that
# never reached a row returns 0 too, so the launcher's status alone certifies
# nothing.
#
# WHAT IS DIFFERENT FROM THE C++ BED, AND WHY.
#
#   * THE GATE BUILDS ITS OWN INPUTS. The bed drives a deployed aether-abi/2
#     ARTIFACT, not a linked fixture: this script compiles the default one (the
#     repo's own tests/fixtures/{host,device}_plugin_execv2 TUs, laid out as
#     manifests + sidecars by tests/v2_artifact.py) BEFORE it launches any rank,
#     and exports $EAGLE_RANK_BED_ARTIFACT. Building it once, outside the world,
#     is what stops two ranks racing to compile the same object; building it at
#     all is what stops the bed certifying whatever object was lying around.
#     $EAGLE_RANK_BED_ARTIFACT set by the CALLER is honoured as-is and nothing is
#     built — that is the door where the artifact is a real deployed emission.
#
#   * THE LISTING IS A COLLECTION. gtest lists tests; pytest collects them, and
#     `--collect-only -q` prints one node id per line. The manifest is therefore
#     a list of pytest node ids (`test_rank_bed.py::test_...`), pinned and
#     compared exactly as the gtest name manifests are.
#
#   * THE BED IS COLLECTED ONLY UNDER THIS GATE. python/conftest.py refuses to
#     RECURSE into tests/mpi unless $EAGLE_RANK_BED is set, so an ordinary
#     `pytest` stays mpirun-free and never initialises MPI in a session that is
#     also driving CUDA. This script sets it. (Naming the directory directly on a
#     pytest command line bypasses that hook — pytest's own rule for initial
#     arguments — and the bed then fails loudly at a world size of one rather
#     than hanging; conftest.py says so where the guard is written.)
#
# NON-VACUITY IS PROVEN, NOT ASSUMED. `-k NoSuchTest` makes pytest select
# nothing and exit 5 on every rank; the run below is put through the SAME
# judgement, and this gate REFUSES to report GREEN unless that probe comes out RED.
#
# Usage:
#   python/tests/mpi/check_rank_bed.sh
#   python/tests/mpi/check_rank_bed.sh --remint [--allow-removals]
#
# Environment:
#   PY                       the interpreter (default: `python3` from $PATH)
#   MPIRUN                   the launcher (default: `mpirun` from $PATH)
#   EAGLE_MPI_BED_RANKS      the world size (default: 2)
#   EAGLE_RANK_BED_ARTIFACT  an existing artifact root to drive (default: built here)
#
set -u -o pipefail

PROG="python/tests/mpi/check_rank_bed.sh"
RANKS="${EAGLE_MPI_BED_RANKS:-2}"
MPIRUN="${MPIRUN:-mpirun}"
PY="${PY:-python3}"

usage() {
    cat >&2 <<EOF
usage: $PROG [--remint [--allow-removals]]

  --remint            regenerate the pinned node-id manifest from THIS collection
                      (local only; refused in CI)
  --allow-removals    required when a re-mint would DELETE manifest lines

No count is ever passed on the command line — the manifest file is the expectation.
EOF
    exit 2
}

REMINT=0
ALLOW_REMOVALS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --remint)         REMINT=1 ;;
        --allow-removals) ALLOW_REMOVALS=1 ;;
        -h|--help)        usage ;;
        *)                echo "$PROG: unexpected argument '$1'" >&2; usage ;;
    esac
    shift
done

BED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # python/tests/mpi
PY_ROOT="$(cd "$BED_DIR/../.." && pwd)"                      # python
MANIFEST="$BED_DIR/expected_tests_rank_bed.txt"
REMINT_CMD="$PROG --remint"

for tool in "$MPIRUN" "$PY"; do
    if ! command -v "$tool" > /dev/null 2>&1 && [ ! -x "$tool" ]; then
        echo "GATE RED: '$tool' not found on \$PATH (set \$MPIRUN / \$PY)" >&2
        exit 1
    fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The bed is collected only under this gate (python/conftest.py).
export EAGLE_RANK_BED=1
export EAGLE_MPI_BED_RANKS="$RANKS"

# ---------------------------------------------------------------------------
# The gate builds its own inputs.
#
# Built ONCE, here, outside the world: two ranks compiling the same object into
# the same path would race, and a bed that reused a previous run's artifact would
# certify a body nobody in this run compiled. A caller-supplied artifact
# (a caller-supplied one) is used verbatim and nothing is built.
# ---------------------------------------------------------------------------
if [ -n "${EAGLE_RANK_BED_ARTIFACT:-}" ]; then
    echo "== $PROG: driving the CALLER's artifact at $EAGLE_RANK_BED_ARTIFACT =="
else
    ART="$TMP/artifact"
    echo "== $PROG: building the default aether-abi/2 bed artifact into $ART =="
    if ! ( cd "$PY_ROOT" && "$PY" -c "
import pathlib, sys
sys.path.insert(0, 'tests')
import v2_artifact
print(v2_artifact.build(pathlib.Path('$ART')))
" ) > "$TMP/artifact.log" 2>&1; then
        echo "GATE RED: building the bed artifact failed" >&2
        sed 's/^/         | /' "$TMP/artifact.log" >&2
        exit 1
    fi
    export EAGLE_RANK_BED_ARTIFACT="$ART"
fi

# ---------------------------------------------------------------------------
# The collection (a ONE-rank world: the test SET is a property of the source,
# and two ranks collecting into one merged stdout would interleave two copies of
# it). Env-scrubbed for the same reason check_gate.sh scrubs its listing: an
# ambient PYTEST_ADDOPTS filters the collection exactly as it filters the run, so
# a collection that inherited it would agree with a filtered run about a set
# neither of them ran.
# ---------------------------------------------------------------------------
collect_scrubbed() {
    ( cd "$PY_ROOT" && env -u PYTEST_ADDOPTS -u PYTEST_CURRENT_TEST \
        "$MPIRUN" -np 1 --oversubscribe \
        "$PY" -m pytest tests/mpi -p no:cacheprovider -q --collect-only ) \
        2>"$TMP/collect.err" | sed -n 's#^tests/mpi/##p'
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
    collect_scrubbed | LC_ALL=C sort -u > "$TMP/new.txt"
    COLLECT_RC=${PIPESTATUS[0]}
    if [ "$COLLECT_RC" -ne 0 ]; then
        echo "REFUSED: collecting the bed failed (rc $COLLECT_RC); refusing to mint from it." >&2
        sed 's/^/  | /' "$TMP/collect.err" >&2
        exit 1
    fi
    if [ ! -s "$TMP/new.txt" ]; then
        echo "REFUSED: the bed collected ZERO tests — refusing to mint an empty manifest." >&2
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
# Manifest hygiene, matching check_gate.sh's own manifest check in intent.
# ---------------------------------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "GATE RED: no manifest at $MANIFEST — mint it with: $REMINT_CMD" >&2
    exit 1
fi
EXPECTED=$(awk 'END{print NR}' "$MANIFEST")
if [ "$EXPECTED" -lt 1 ]; then
    red "[R1] manifest $MANIFEST is EMPTY — an empty expectation certifies nothing"
fi
if [ -s "$MANIFEST" ] && [ "$(tail -c1 "$MANIFEST" | wc -l)" -ne 1 ]; then
    red "[R1] manifest $MANIFEST is not newline-terminated"
fi
if grep -nvE '^[^[:space:]#]+\.py::[^[:space:]#]+$' "$MANIFEST" > "$TMP/shape.txt"; then
    red "[R1] manifest $MANIFEST has lines that are not a bare '<file>.py::<test>' node id:"
    sed 's/^/         /' "$TMP/shape.txt" >&2
fi
if ! LC_ALL=C sort -c "$MANIFEST" 2>"$TMP/sortc.txt"; then
    red "[R1] manifest $MANIFEST is NOT sorted (LC_ALL=C): $(cat "$TMP/sortc.txt")"
fi
UNIQ=$(LC_ALL=C sort -u "$MANIFEST" | awk 'END{print NR}')
if [ "$UNIQ" -ne "$EXPECTED" ]; then
    red "[R1] manifest $MANIFEST has duplicate lines ($EXPECTED lines, $UNIQ unique):"
    LC_ALL=C sort "$MANIFEST" | uniq -d | sed 's/^/         /' >&2
fi

# ---------------------------------------------------------------------------
# Env-scrubbed collection vs the manifest, both directions.
# ---------------------------------------------------------------------------
collect_scrubbed | LC_ALL=C sort -u > "$TMP/collected.txt"
COLLECT_RC=${PIPESTATUS[0]}
COLLECTED=$(awk 'END{print NR}' "$TMP/collected.txt")
if [ "$COLLECT_RC" -ne 0 ]; then
    red "[R2] collecting the bed under '$MPIRUN -np 1' failed with rc $COLLECT_RC"
    sed 's/^/         | /' "$TMP/collect.err" >&2
fi
LC_ALL=C sort -u "$MANIFEST" > "$TMP/manifest.txt"
LC_ALL=C comm -23 "$TMP/manifest.txt" "$TMP/collected.txt" > "$TMP/missing.txt"
LC_ALL=C comm -13 "$TMP/manifest.txt" "$TMP/collected.txt" > "$TMP/extra.txt"
if [ -s "$TMP/missing.txt" ] || [ -s "$TMP/extra.txt" ]; then
    red "[R2] the bed's test SET differs from $MANIFEST (collected $COLLECTED, pinned $EXPECTED)"
    if [ -s "$TMP/missing.txt" ]; then
        echo "         MISSING (pinned but NOT collected — deleted, renamed, or NOT COLLECTED):" >&2
        sed 's/^/         - /' "$TMP/missing.txt" >&2
    fi
    if [ -s "$TMP/extra.txt" ]; then
        echo "         UNEXPECTED (collected but NOT pinned — new rows, un-minted):" >&2
        sed 's/^/         + /' "$TMP/extra.txt" >&2
    fi
fi

# ---------------------------------------------------------------------------
# The run + per-rank verdicts.
#
# `run_bed <log> <extra pytest args...>` deletes the stale reports, runs the bed
# under `$MPIRUN -np $RANKS`, and leaves the launcher's rc in RUN_RC and each
# rank's "<tests> <failures> <errors>" in VERDICTS (empty when that rank wrote no
# report at all — a crash, or a rank that never got as far as running).
#
# THE XML IS NAMED PER RANK by $OMPI_COMM_WORLD_RANK. The C++ bed rewrites its
# own --gtest_output in main(); pytest's --junitxml is fixed at start-up, so the
# rank comes from the launcher's own environment instead. Both ranks share a
# working directory, so one un-suffixed path would have them racing to write it.
# ---------------------------------------------------------------------------
XML_BASE="$TMP/rank_bed"
export XML_BASE PY

# One rank's leg. It exists as a script because the report path has to carry the
# rank, and `mpirun` performs no shell expansion on the argv it is handed — the
# rank number is only in each rank's own ENVIRONMENT, so only a shell running
# INSIDE the rank can put it in the path. (gtest's bed does the equivalent by
# rewriting its own --gtest_output in main(); pytest fixes --junitxml at start-up.)
# Both ranks share a working directory, so one un-suffixed path would have them
# racing to write it, and the gate would then judge whichever won.
cat > "$TMP/rank_leg.sh" <<'RANKLEG'
#!/usr/bin/env bash
rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${OMPI_MCA_orte_ess_vpid:-0}}}"
exec "$PY" -m pytest tests/mpi -p no:cacheprovider \
     --junitxml="${XML_BASE}_rank${rank}.xml" "$@"
RANKLEG
chmod +x "$TMP/rank_leg.sh"

# One `<testsuite …>` attribute, by NAME rather than by position: pytest and gtest
# order those attributes differently, and reading them positionally is how a gate
# ends up comparing `errors` against an expected test count.
suite_attr() {
    grep -m1 -oP '<testsuite [^>]*' "$1" \
        | grep -m1 -oP "(?<![a-zA-Z])$2=\"\K[0-9]+"
}

run_bed() {
    local log="$1"; shift
    local r
    for (( r = 0; r < RANKS; ++r )); do rm -f "${XML_BASE}_rank${r}.xml"; done
    ( cd "$PY_ROOT" && "$MPIRUN" -np "$RANKS" --oversubscribe \
        "$TMP/rank_leg.sh" "$@" ) > "$log" 2>&1
    RUN_RC=$?
    VERDICTS=()
    for (( r = 0; r < RANKS; ++r )); do
        local xml="${XML_BASE}_rank${r}.xml"
        if [ ! -s "$xml" ]; then VERDICTS+=(""); continue; fi
        VERDICTS+=("$(suite_attr "$xml" tests) $(suite_attr "$xml" failures) \
$(suite_attr "$xml" errors) $(suite_attr "$xml" skipped)")
    done
}

# `judge` applies the SAME rules to whatever run_bed last produced, appending its
# complaints to $1. It returns 0 for a clean run and 1 otherwise — which is what
# lets the non-vacuity probe below be checked with the real judgement rather than
# with a weaker copy of it.
judge() {
    local out="$1"
    local bad=0 r
    [ "$RUN_RC" -ne 0 ] && { echo "[R3] '$MPIRUN -np $RANKS' exited with rc $RUN_RC" >> "$out"; bad=1; }
    for (( r = 0; r < RANKS; ++r )); do
        if [ -z "${VERDICTS[$r]}" ]; then
            echo "[R4] rank $r wrote no XML report — it never reported a run (crashed, aborted, or deadlocked)" >> "$out"
            bad=1
        fi
    done
    [ "$bad" -eq 1 ] && return 1
    local ran fail err skip
    read -r ran fail err skip <<< "${VERDICTS[0]}"
    if [ "$ran" -lt 1 ]; then
        echo "[R5] rank 0 ran $ran tests — a selection that matches nothing is not a pass" >> "$out"
        bad=1
    elif [ "$ran" -ne "$EXPECTED" ]; then
        echo "[R6] rank 0 ran $ran tests, manifest pins $EXPECTED (PYTEST_ADDOPTS='${PYTEST_ADDOPTS:-<unset>}')" >> "$out"
        bad=1
    fi
    if [ "$fail" -ne 0 ] || [ "$err" -ne 0 ]; then
        echo "[R7] rank 0 reported $fail failure(s) and $err error(s)" >> "$out"
        bad=1
    fi
    # A SKIPPED row in this bed is a row that certified nothing while counting
    # towards the pinned total — the "green having run nothing" failure at the
    # granularity of one row. The bed carries no skip machinery by construction;
    # this is what keeps it that way when someone adds a marker later.
    if [ "$skip" -ne 0 ]; then
        echo "[R7b] rank 0 SKIPPED $skip row(s); the rank bed must not skip — a skipped row is pinned but certifies nothing" >> "$out"
        bad=1
    fi
    for (( r = 1; r < RANKS; ++r )); do
        if [ "${VERDICTS[$r]}" != "${VERDICTS[0]}" ]; then
            echo "[R8] rank $r's verdict (tests failures errors skipped) = '${VERDICTS[$r]}' differs from rank 0's '${VERDICTS[0]}'" >> "$out"
            bad=1
        fi
    done
    return $bad
}

# The REAL run. Deliberately NOT env-scrubbed: an ambient PYTEST_ADDOPTS must make
# this gate RED, not be laundered into a clean run.
echo "== $PROG: running $MPIRUN -np $RANKS $PY -m pytest tests/mpi (pinned $EXPECTED tests) =="
run_bed "$TMP/rank_bed.log"
RUN_VERDICTS=("${VERDICTS[@]}")
RUN_LAUNCH_RC=$RUN_RC
: > "$TMP/complaints.txt"
if ! judge "$TMP/complaints.txt"; then
    while IFS= read -r line; do red "$line"; done < "$TMP/complaints.txt"
    grep -E "^(FAILED|ERROR)|^E " "$TMP/rank_bed.log" | head -40 | sed 's/^/         | /' >&2
    tail -5 "$TMP/rank_bed.log" | sed 's/^/         | /' >&2
fi

# ---------------------------------------------------------------------------
# NON-VACUITY. The instrument must be able to fail: run the bed with a
# selection that matches nothing and require the judgement above to call it RED.
# ---------------------------------------------------------------------------
echo "== $PROG: non-vacuity probe (-k NoSuchTest must come out RED) =="
run_bed "$TMP/probe.log" -k NoSuchTest
PROBE_VERDICT="${VERDICTS[0]:-<no report>}"
: > "$TMP/probe_complaints.txt"
if judge "$TMP/probe_complaints.txt"; then
    red "[R9] the NoSuchTest probe came out GREEN (rank 0 verdict '$PROBE_VERDICT') — this gate cannot fail and certifies nothing"
else
    echo "   probe RED as required: rank 0 verdict (tests failures errors skipped) = '$PROBE_VERDICT'"
    sed 's/^/     probe: /' "$TMP/probe_complaints.txt"
fi

echo "-----------------------------------------------------------------------"
echo "manifest=$MANIFEST  ranks=$RANKS  pinned=$EXPECTED  collected=$COLLECTED  launcher_rc=$RUN_LAUNCH_RC"
echo "artifact=$EAGLE_RANK_BED_ARTIFACT"
for (( i = 0; i < RANKS; ++i )); do
    echo "  rank $i verdict (tests failures errors skipped) = '${RUN_VERDICTS[$i]:-<no report>}'"
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
