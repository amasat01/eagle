#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Collect the calibration data of the dispatch cost model on this machine: a few
# minutes of isolated micro-timings written to one CSV. Nothing is fitted and nothing
# on the machine is changed (a private venv under --work, the CSV under --out).
#
#   calibrate.sh [options]
#     --profile P     gpu | cpu | auto   (default auto: gpu when a GPU is visible)
#     --work DIR      the venv and the kernel cache (default per target, as bootstrap.sh)
#     --out DIR|FILE  where the CSV goes (default WORK/calibration/<machine-slug>/)
#     --quick         smoke pass: small sizes, about a minute
#     --local         use the eagle and hawk already installed for `python` (no venv, no
#                     download): how to try this on a development machine
#     --python PATH   the interpreter for the venv or for --local (default python3)
#     --deadline-h H  stop the run after H hours (exit 3); the full run takes minutes
#     --dry-run       print every command, change nothing
#
# On a machine with a GPU the GPU profile times the first visible GPU; with several,
# set CUDA_VISIBLE_DEVICES to the one to measure. The CPU profile also runs on a GPU
# machine and measures the host (set OMP_PROC_BIND / OMP_NUM_THREADS to pin it).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
. "$HERE/lib.sh"

PROFILE=auto WORK="" OUT="" QUICK=0 LOCAL=0 PYTHON=python3 DEADLINE_H=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --profile) PROFILE=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --local) LOCAL=1; shift ;;
    --python) PYTHON=$2; shift 2 ;;
    --deadline-h) DEADLINE_H=$2; shift 2 ;;
    --dry-run) REPRO_DRY_RUN=1; shift ;;
    -h|--help) sed -n '5,22p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

TARGET=$(detect_target)
GPUS=$(detect_gpus)
if [[ $PROFILE == auto ]]; then PROFILE=$([[ -n $GPUS ]] && echo gpu || echo cpu); fi
[[ $PROFILE == gpu || $PROFILE == cpu ]] || die "unknown profile $PROFILE (gpu | cpu | auto)"
[[ $PROFILE == cpu || -n $GPUS ]] || die "profile gpu but no GPU is visible: use --profile cpu"
WORK=${WORK:-$(default_work "$TARGET")}
OUT=${OUT:-$WORK/calibration/$(machine_slug)}
log "target=$TARGET profile=$PROFILE work=$WORK out=$OUT$([[ $LOCAL == 1 ]] && echo ' (local install)')"

# hawk compiles each kernel the first time it runs: a C++ compiler is needed.
command -v "${CXX:-c++}" >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 \
  || die "no C++ compiler found: hawk compiles each kernel when it first runs (Debian/Ubuntu: sudo apt-get install build-essential)"

PY=$PYTHON
if [[ $LOCAL == 0 ]]; then
  command -v "$PYTHON" >/dev/null 2>&1 || die "no $PYTHON: install Python >= 3.10 or pass --python"
  "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "$PYTHON is older than 3.10"
  VENV=$WORK/envs/calibrate-$PROFILE
  PY=$VENV/bin/python
  LOCK=$HERE/locks/calibrate-$PROFILE.txt
  STAMP=$VENV/.repro-calibrate
  SUM=$(md5sum "$LOCK" | cut -c1-32)
  export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
  if [[ $REPRO_DRY_RUN == 0 && -f $STAMP && $(cat "$STAMP") == "$SUM" ]]; then
    log "python packages up to date"
  else
    run mkdir -p "$WORK/envs"
    [[ -d $VENV && $REPRO_DRY_RUN == 0 ]] && run rm -rf "$VENV"
    run "$PYTHON" -m venv "$VENV"
    run "$PY" -m pip install -q --no-cache-dir --pre -r "$LOCK"
    [[ $REPRO_DRY_RUN == 1 ]] || echo "$SUM" >"$STAMP"
  fi
fi

if [[ $REPRO_DRY_RUN == 0 ]]; then
  "$PY" -c 'import eagle, hawk' 2>/dev/null \
    || die "eagle and hawk are not importable by $PY: drop --local to install the pinned wheels"
fi

# The kernel cache lives under the work directory: a rerun reuses it, a new machine starts cold.
export HAWK_CACHE_DIR=${HAWK_CACHE_DIR:-$WORK/hawk-cache}
[[ $PROFILE == cpu ]] && export CUDA_VISIBLE_DEVICES=""
export CUDA_DEVICE_ORDER=PCI_BUS_ID

ARGS=(--target "$PROFILE" --out "$OUT")
[[ $QUICK == 1 ]] && ARGS+=(--quick)
run mkdir -p "$OUT"
CMD=("$PY" "$HERE/calibrate.py" "${ARGS[@]}")
if [[ -n $DEADLINE_H ]]; then
  SECS=$(awk -v h="$DEADLINE_H" 'BEGIN {printf "%d", h * 3600}')
  CMD=(timeout --signal=INT "$SECS" "${CMD[@]}")
fi
rc=0
run "${CMD[@]}" || rc=$?
if [[ $rc -eq 124 || $rc -eq 130 ]]; then log "deadline reached before the run finished: nothing written"; exit 3; fi
[[ $rc -eq 0 ]] || die "calibrate.py failed (exit $rc): see the output above"
if [[ $REPRO_DRY_RUN == 1 ]]; then log "dry run: nothing was run"; exit 0; fi
CSV=$(find "$OUT" -maxdepth 1 -name 'calibration-*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
log "done. Calibration data: ${CSV:-$OUT}"
