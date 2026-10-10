#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/sanitize/sanitize_gate.sh — local pre-release sanitizer gate (CUDA mode, one GPU).
#
# Rows (see README.md): compute-sanitizer memcheck, racecheck, synccheck and valgrind each run
# THROUGH tests/check_gate.sh via a generated exec wrapper (listing + run + manifest + tool
# exit status in one command); initcheck is triage-only (reported, never RED by itself).
# Then the device canaries must trip: memcheck >= 1 error on dev-oob, racecheck >= 1
# hazard on dev-race. A tool that stays quiet on its canary is not instrumenting anything.
#
# Usage:   tests/sanitize/sanitize_gate.sh [build-dir]        (default: build)
# Env:     CUDA_VISIBLE_DEVICES  GPU to use (default 1)
#          SAN_ROWS              tools to run (default "memcheck racecheck synccheck valgrind initcheck")
#          SAN_LOG_DIR           log directory (default <build>/sanitize_gate_<UTC stamp>)
# Exit:    0 all green; 1 any RED. Record the printed log dir and suppressed count in the release ledger.
set -u -o pipefail

PROJ="eagle"
BUILD="${1:-build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
ROWS="${SAN_ROWS:-memcheck racecheck synccheck valgrind initcheck}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD="$(cd "$BUILD" && pwd)"
BIN="$BUILD/tests/${PROJ}_tests"
CANARY="$BUILD/tests/${PROJ}_canary_cu"
[ -x "$BIN" ] || { echo "sanitize_gate: $BIN missing (build the CUDA-mode tests first)" >&2; exit 1; }
[ -x "$CANARY" ] || { echo "sanitize_gate: $CANARY missing (CUDA-mode canary is built with the tests)" >&2; exit 1; }
LOG="${SAN_LOG_DIR:-$BUILD/sanitize_gate_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG"; LOG="$(cd "$LOG" && pwd)"
echo "== sanitize gate: $PROJ  GPU=$CUDA_VISIBLE_DEVICES  logs: $LOG"

RED=0
red() { echo "RED: $*" >&2; RED=1; }

for tool in $ROWS; do
    # wrapper dir stays inside the build tree: check_gate.sh reads the build config from the
    # CMakeCache.txt above the binary it is handed.
    WRAP="$BUILD/sanitize_wrap/$tool"
    "$HERE/make_wrapper.sh" "$tool" "$BIN" "$WRAP" "$LOG/$tool" > /dev/null || { red "$tool: wrapper generation"; continue; }
    echo "-- $tool: tests/check_gate.sh cuda $WRAP/${PROJ}_tests"
    "$ROOT/tests/check_gate.sh" cuda "$WRAP/${PROJ}_tests" > "$LOG/$tool.gate.txt" 2>&1
    rc=$?
    tail -n 4 "$LOG/$tool.gate.txt"
    if [ "$rc" -ne 0 ]; then
        if [ "$tool" = initcheck ]; then
            echo "TRIAGE: initcheck rc=$rc — confirm each reader consumes the uninitialised bytes (class C); not RED by itself"
        else
            red "$tool: check_gate rc=$rc (see $LOG/$tool.gate.txt)"
        fi
    fi
    # a compute-sanitizer log with no ERROR SUMMARY = the tool never saw CUDA = vacuous
    if [ "$tool" != valgrind ]; then
        n=$(grep -lE '(ERROR|RACECHECK) SUMMARY' "$LOG/$tool"/computesan.*.log 2>/dev/null | wc -l)
        [ "$n" -ge 1 ] || red "$tool: no log carries a tool summary"
    fi
done

# suppressed count (valgrind), for the ledger
if ls "$LOG"/valgrind/valgrind.*.log >/dev/null 2>&1; then
    echo "-- valgrind suppressed: $(grep -h 'ERROR SUMMARY' "$LOG"/valgrind/valgrind.*.log | sed -E 's/.*suppressed: ([0-9]+) from.*/\1/' | paste -sd' ')"
fi

# ---- canaries: the tools MUST go red here --------------------------------
errs_of() { # <log-glob-dir> <pattern>  -> max count among the summary lines
    grep -hoE "$2" "$1"/*.log 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1
}
canary() { # <tool> <mode> <summary-regex>
    local tool="$1" mode="$2" re="$3" d="$LOG/canary_$2"
    mkdir -p "$d"
    compute-sanitizer --tool="$tool" --error-exitcode=77 --log-file="$d/%p.log" "$CANARY" "$mode" > "$d/stdout.txt" 2>&1
    local n; n="$(errs_of "$d" "$re")"
    echo "-- canary $mode ($tool): ${n:-none} reported"
    if [ -z "$n" ] || [ "$n" -lt 1 ]; then red "canary $mode: $tool reported no error — the gate is blind"; fi
}
canary memcheck dev-oob  'ERROR SUMMARY: [0-9]+ error'
canary racecheck dev-race 'RACECHECK SUMMARY: [0-9]+ hazard|ERROR SUMMARY: [0-9]+ error'

if [ "$RED" -ne 0 ]; then echo "VERDICT: RED  (logs: $LOG)" >&2; exit 1; fi
echo "VERDICT: GREEN  (logs: $LOG)"
