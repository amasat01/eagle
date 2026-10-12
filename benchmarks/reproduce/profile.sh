#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Profile eagle's launch decisions and kernels on this machine's GPU (Nsight Systems and
# Nsight Compute), unattended, in under 45 minutes, ending with one tarball to copy off.
#
#   profile.sh [options]
#     --env FILE        the GPU environment script bootstrap.sh wrote (default WORK/env-gpu.sh)
#     --out DIR         where results go (default WORK/profiles/<machine-slug>/)
#     --work DIR        bootstrap's work directory (default per target)
#     --quick           the probe and one point of every stage: under 8 minutes
#     --fresh           ignore the markers of an earlier run and start over
#     --deadline-min M  stop starting stages after M minutes (default 50)
#     --only A[,B...]   run just these stages (a name, or a prefix such as ncu), keeping the order
#     --dry-run         print the plan, run nothing
#
# Stages run independently, each under its own time cap; a failing stage never stops the
# next. Each finished stage leaves a marker, so re-running the same command after a
# preemption continues where it stopped. The tarball profile-<slug>.tgz (reports, raw
# captures under 200 MB, status.txt, facts.txt) is written on every exit, including
# errors, timeouts, Ctrl-C and SIGTERM. Nothing outside --out is changed or deleted.
#
# Test hooks (used by the test suite only): PROFILE_TEST_FAIL=<stage> forces that stage to
# fail; PROFILE_TEST_CAP_S=<seconds> replaces every stage's time cap;
# PROFILE_TOOLS_PATH_ONLY=1 looks for nsys and ncu on PATH alone; PROFILE_MIN_DISK_GB lowers
# the free-space check (default 5).
# shellcheck disable=SC1090,SC2317  # the env script is sourced by path; trap handlers run indirectly
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
. "$HERE/lib.sh"

ENV_FILE="" OUT="" WORK=${REPRO_WORK:-} QUICK=0 FRESH=0 DEADLINE_MIN=50 ONLY=
while [[ $# -gt 0 ]]; do
  case $1 in
    --env) ENV_FILE=${2:-}; shift 2 ;;
    --out) OUT=${2:-}; shift 2 ;;
    --work) WORK=${2:-}; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --deadline-min) DEADLINE_MIN=${2:-}; shift 2 ;;
    --only) ONLY=${2:-}; shift 2 ;;
    --dry-run) REPRO_DRY_RUN=1; shift ;;
    -h|--help) sed -n '5,28p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done
[[ $DEADLINE_MIN =~ ^[0-9]+$ ]] || die "--deadline-min takes whole minutes"

TARGET=$(detect_target)
WORK=${WORK:-$(default_work "$TARGET")}
ENV_FILE=${ENV_FILE:-$WORK/env-gpu.sh}
SLUG=$(machine_slug)
OUT=${OUT:-$WORK/profiles/$SLUG}
DATA=$OUT/data LOGS=$OUT/logs MARK=$OUT/.stages
EROOT=$(cd "$HERE/../.." && pwd)
TARBALL=$OUT/profile-$SLUG.tgz

# stage NAME:CAP_SECONDS, in run order
if [[ $QUICK == 1 ]]; then
  PLAN=(probe:180 nsys-quick:120 ncu-quick:180 compare-quick:300 crossover-quick:120 band-quick:150 cadence-sweep-quick:90
        ncu-ratio-quick:150 ncu-tsweep-quick:150 ncu-band-quick:180)
else
  PLAN=(probe:180 nsys-grid:540 nsys-width:150 ncu-spread-1e6:240 ncu-uniform-1e6:240 ncu-spread-1e4:240
        compare:1500 crossover:300 band:420 cadence-sweep:300 ncu-ratio:420 ncu-tsweep:750 ncu-band:300)
fi
if [[ -n $ONLY ]]; then
  CHOSEN=()
  IFS=, read -ra WANT <<<"$ONLY"
  for w in "${WANT[@]}"; do
    hit=0
    for s in "${PLAN[@]}"; do
      n=${s%%:*}
      if [[ $n == "$w" || $n == "$w-quick" || $n == "$w"-* ]]; then hit=1; fi
    done
    [[ $hit == 1 ]] || die "--only $w matches no stage; the stages are: $(printf '%s ' "${PLAN[@]%%:*}")"
  done
  for s in "${PLAN[@]}"; do
    n=${s%%:*}
    for w in "${WANT[@]}"; do
      if [[ $n == "$w" || $n == "$w-quick" || $n == "$w"-* ]]; then CHOSEN+=("$s"); break; fi
    done
  done
  PLAN=("${CHOSEN[@]}")
fi
if [[ $REPRO_DRY_RUN == 1 ]]; then
  log "env=$ENV_FILE out=$OUT deadline=${DEADLINE_MIN} min"
  for s in "${PLAN[@]}"; do log "stage ${s%%:*} (cap ${s##*:} s)"; done
  exit 0
fi

mkdir -p "$DATA" "$LOGS" "$MARK" || { echo "[repro] ERROR: cannot create $OUT" >&2; exit 2; }
[[ $FRESH == 1 ]] && rm -f "${MARK:?}"/*.ok

# --- status and the exit trap ------------------------------------------------------------
declare -A STATUS=()
PREFLIGHT_ERROR="" CURRENT="" CHILD="" NCU="" NSYS="" NCU_MODE="" T0=$SECONDS
EXCLUDED=""

write_status() {
  {
    echo "profile.sh status: $(date -u +%FT%TZ), elapsed $((SECONDS - T0)) s, deadline ${DEADLINE_MIN} min"
    [[ -n $PREFLIGHT_ERROR ]] && echo "PREFLIGHT FAILED: $PREFLIGHT_ERROR"
    local entry name
    for entry in "${PLAN[@]}"; do
      name=${entry%%:*}
      echo "$name: ${STATUS[$name]:-skipped: not reached}"
      if [[ ${STATUS[$name]:-} == failed* && -f $LOGS/$name.log ]]; then
        echo "  --- last 20 log lines ---"
        tail -n 20 "$LOGS/$name.log" | sed 's/^/  | /'
      fi
    done
    [[ -n $EXCLUDED ]] && printf 'raw captures over 200 MB left out of the tarball:\n%s' "$EXCLUDED"
  } >"$OUT/status.txt"
}

smi_clocks() {
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active \
    --format=csv 2>&1 || true
}

write_facts() {  # PHASE: before | after
  local f=$OUT/facts.txt
  if [[ $1 == before ]]; then
    {
      echo "date:      $(date -u +%FT%TZ)"
      echo "host:      $(uname -srm)"
      echo "cpu:       $(cpu_model) ($(nproc) logical)"
      echo "slug:      $SLUG"
      echo "eagle:     $(git -C "$EROOT" describe --tags --always --dirty 2>/dev/null || echo '?')"
      echo "nsys:      ${NSYS:-none} $([[ -n $NSYS ]] && "$NSYS" --version 2>/dev/null | head -1)"
      echo "ncu:       ${NCU:-none} $([[ -n $NCU ]] && "$NCU" --version 2>/dev/null | tail -1)"
      echo "-- nvidia-smi before"
      smi_clocks
      echo "-- packages"
      ( cd "$EROOT" && . "$ENV_FILE" 2>/dev/null
        python - <<'PY'
import importlib.metadata as md
for d in ("raptor-eagle", "raptor-hawk", "raptor-core", "cupy-cuda12x", "warp-lang", "numpy"):
    try:
        print(f"{d}: {md.version(d)}")
    except md.PackageNotFoundError:
        pass
PY
      ) 2>&1
    } >"$f"
  else
    { echo "-- nvidia-smi after"; smi_clocks; echo "-- stage walls"; cat "$OUT/walls.txt" 2>/dev/null; } >>"$f"
  fi
}

finalize() {
  trap - EXIT INT TERM
  if [[ -n $CHILD ]]; then
    kill -TERM -- "-$CHILD" 2>/dev/null; sleep 2; kill -KILL -- "-$CHILD" 2>/dev/null
    [[ -n $CURRENT && -z ${STATUS[$CURRENT]:-} ]] && STATUS[$CURRENT]="failed: interrupted ($1)"
  fi
  write_facts after
  EXCLUDED=$(find "$DATA" -type f -size +200M -printf '  %p (%s bytes)\n' 2>/dev/null)
  write_status
  local excl=() f
  while IFS= read -r f; do [[ -n $f ]] && excl+=(--exclude="${f#"$OUT"/}"); done \
    < <(find "$DATA" -type f -size +200M 2>/dev/null)
  if tar -C "$OUT" "${excl[@]}" -czf "$TARBALL.part" data logs status.txt facts.txt 2>/dev/null; then
    mv -f "$TARBALL.part" "$TARBALL"
    printf '[repro] results: %s (%s)\n' "$TARBALL" "$(stat -c %s "$TARBALL" | awk '{printf "%.1f MB", $1 / 1048576}')"
  else
    printf '[repro] ERROR: could not write %s; the files are in %s\n' "$TARBALL" "$OUT"
  fi
}

on_exit() { local rc=$?; finalize "exit $rc"; exit "$rc"; }
trap on_exit EXIT
trap 'finalize SIGINT; exit 130' INT
trap 'finalize SIGTERM; exit 143' TERM

fatal() { PREFLIGHT_ERROR=$1; printf '[repro] ERROR: %s\n' "$1" >&2; exit 2; }

# --- preflight ----------------------------------------------------------------------------
[[ -f $ENV_FILE ]] || fatal "env script $ENV_FILE not found: run bootstrap.sh --profile gpu first (or pass --env)"
if ! import_out=$( cd "$EROOT" && . "$ENV_FILE" && python -c 'import eagle, hawk, warp, cupy' 2>&1 ); then
  fatal "the environment $ENV_FILE cannot import eagle, hawk, warp and cupy: ${import_out##*$'\n'}"
fi
command -v nvidia-smi >/dev/null 2>&1 || fatal "nvidia-smi not found: this needs an NVIDIA driver"
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
SEL=${CUDA_VISIBLE_DEVICES:-}
[[ $NGPU -ge 1 ]] || fatal "nvidia-smi sees no GPU"
if [[ $NGPU -gt 1 && ! $SEL =~ ^[0-9]+$ ]]; then
  fatal "nvidia-smi sees $NGPU GPUs: set CUDA_VISIBLE_DEVICES to the one index to profile"
fi
if [[ $NGPU -gt 1 ]]; then
  # nothing of ours may run on the other GPUs: it would share the machine with the captures
  BUS_INDEX=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | tr -d ' ')
  while IFS=, read -r bus pid; do
    [[ -n $pid ]] || continue
    idx=$(awk -F, -v b="${bus// /}" '$2==b {print $1}' <<<"$BUS_INDEX")
    [[ $idx == "$SEL" ]] && continue
    if tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -Eq 'eagle|hawk|perf_card|gpu_profile\.py|calibrate'; then
      fatal "eagle is running on GPU $idx (process $pid): stop it first, the captures need a quiet machine"
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader 2>/dev/null)
fi
FREE_GB=$(df -B1G --output=avail "$OUT" 2>/dev/null | tail -1 | tr -d ' ')
MIN_GB=${PROFILE_MIN_DISK_GB:-5}
[[ ${FREE_GB:-0} -ge $MIN_GB ]] || fatal "only ${FREE_GB:-?} GB free in $OUT: at least $MIN_GB GB are needed (pass --out on a bigger disk)"

find_tool() {
  local t
  t=$(command -v "$1" 2>/dev/null) && { echo "$t"; return; }
  [[ ${PROFILE_TOOLS_PATH_ONLY:-0} == 1 ]] && return
  for t in /usr/local/cuda/bin/"$1" /usr/local/cuda-*/bin/"$1" /opt/nvidia/nsight-systems*/bin/"$1" \
           /opt/nvidia/nsight-systems-cli/*/bin/"$1" /opt/nvidia/nsight-compute/*/"$1"; do
    [[ -x $t ]] && { echo "$t"; return; }
  done
}
NSYS=$(find_tool nsys) NCU=$(find_tool ncu)
log "nsys: ${NSYS:-not found (nsys stages will be skipped)} $([[ -n $NSYS ]] && "$NSYS" --version 2>/dev/null | head -1)"
log "ncu:  ${NCU:-not found (ncu stages will be skipped)} $([[ -n $NCU ]] && "$NCU" --version 2>/dev/null | tail -1)"

NCU_SUDO=0
if [[ -z $NCU ]]; then
  NCU_MODE="skip:ncu not found on PATH or under /usr/local/cuda*/bin (install Nsight Compute)"
elif [[ $EUID -eq 0 ]] || ! grep -Eq 'RmProfilingAdminOnly: *1' /proc/driver/nvidia/params 2>/dev/null; then
  NCU_MODE=direct
elif sudo -n true 2>/dev/null; then
  NCU_MODE=sudo NCU_SUDO=1
  log "GPU performance counters are restricted to administrators: the ncu captures run under 'sudo -n' (passwordless); nothing system-wide is changed"
else
  NCU_MODE="skip:GPU counters are admin-only and 'sudo -n true' failed (no passwordless sudo); ncu not run"
fi
write_facts before

# --- stages -------------------------------------------------------------------------------
DEADLINE_S=$((DEADLINE_MIN * 60))
: >"$OUT/walls.txt"

run_stage() {  # NAME CAP_SECONDS
  local name=$1 cap=$2 rc t_start remaining reason
  local log_file=$LOGS/$name.log
  CURRENT=$name
  if [[ $FRESH == 0 && -f $MARK/$name.ok ]]; then STATUS[$name]="ok (finished in an earlier run)"; return; fi
  case $name in
    nsys*|cadence-sweep*) [[ -n $NSYS ]] || { STATUS[$name]="skipped: nsys not found (install Nsight Systems)"; return; } ;;
    ncu*)  if [[ $NCU_MODE == skip:* ]]; then STATUS[$name]="skipped: ${NCU_MODE#skip:}"; return; fi ;;
  esac
  remaining=$((DEADLINE_S - (SECONDS - T0)))
  if [[ $remaining -le 30 ]]; then STATUS[$name]="skipped: the ${DEADLINE_MIN}-minute deadline was reached"; return; fi
  [[ $cap -gt $remaining ]] && cap=$remaining
  [[ -n ${PROFILE_TEST_CAP_S:-} ]] && cap=$PROFILE_TEST_CAP_S
  log "stage $name (cap ${cap} s)"
  t_start=$SECONDS
  : >"$log_file"
  if [[ ${PROFILE_TEST_FAIL:-} == "$name" ]]; then
    echo "forced failure (PROFILE_TEST_FAIL)" >"$log_file"; rc=1
  else
    local args=(--stage "$name" --out "$DATA")
    # the comparison starts no cell it cannot finish inside its cap
    [[ $name == compare* || $name == crossover* || $name == band* ]] && args+=(--budget-s $((cap > 120 ? cap - 90 : cap / 2)))
    [[ -n $NSYS ]] && args+=(--nsys "$NSYS")
    [[ -n $NCU ]] && args+=(--ncu "$NCU")
    [[ $NCU_SUDO == 1 ]] && args+=(--ncu-sudo)
    # shellcheck disable=SC2016  # the inner script expands its own arguments
    CUDA_DEVICE_ORDER=PCI_BUS_ID setsid timeout --signal=TERM --kill-after=10 "$cap" \
      bash -c '. "$1" || exit 97; cd "$2" || exit 97; shift 2; exec "$@"' stage "$ENV_FILE" "$EROOT" \
      python "$HERE/gpu_profile.py" "${args[@]}" >"$log_file" 2>&1 &
    CHILD=$!
    wait "$CHILD"; rc=$?
    # whatever the stage left running (nsys, ncu, workers) goes with it
    kill -KILL -- "-$CHILD" 2>/dev/null
    CHILD=""
  fi
  printf '%s %s s\n' "$name" $((SECONDS - t_start)) >>"$OUT/walls.txt"
  case $rc in
    0) STATUS[$name]="ok ($((SECONDS - t_start)) s)"; touch "$MARK/$name.ok" ;;
    3) reason=$(sed -n 's/^SKIP: //p' "$log_file" | tail -1); STATUS[$name]="skipped: ${reason:-nothing to do here}" ;;
    124|137) STATUS[$name]="failed: exceeded its ${cap} s time cap (partial results kept)" ;;
    *) STATUS[$name]="failed: exit $rc" ;;
  esac
  log "stage $name: ${STATUS[$name]}"
  CURRENT=""
  write_status
}

for entry in "${PLAN[@]}"; do
  run_stage "${entry%%:*}" "${entry##*:}"
done
write_status
log "stages done in $((SECONDS - T0)) s; see $OUT/status.txt"
exit 0
