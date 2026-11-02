#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Run one device's session plan of eagle's cards in an environment made by bootstrap.sh.
#
#   run_cards.sh --plan NAME [options]
#     --plan NAME     a100 | l4 | t4 | p2000 | zen5 | c4a: the steps of that device, in session order
#                     (P0 sol_probe, P1 wide_sweep with its rows written, C1, C2, then the profile stages)
#     --upload URL    after each step, copy the results to URL/<slug> (gcloud storage rsync); a new machine given
#                     the same URL pulls what is there first and skips the steps and the C1/C2 parts already done
#     --work DIR      bootstrap's work directory (default: $REPRO_WORK, else per target)
#     --out DIR       results root (default WORK/cards); results go to OUT/<machine-slug>/
#     --slug S        machine slug (default: <gpu>--<cpu>, from the hardware)
#     --gpu-env FILE  env script of the gpu profile (default WORK/env-gpu.sh)
#     --cpu-env FILE  env script of the cpu profile (default WORK/env-cpu.sh)
#     --deadline-h H  start no step once H hours have passed (exit 3); run the same command to resume
#     --dry-run       print the plan with its minutes and every command, run nothing
#
# A step that fails or runs over its cap is reported and the next one runs (exit 1 at the end); a step that
# finished leaves a marker, so running the same command again continues where a preempted run stopped.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
. "$HERE/lib.sh"

PLAN="" UPLOAD="" WORK=${REPRO_WORK:-} OUT="" SLUG="" GPU_ENV="" CPU_ENV="" DEADLINE_H=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --plan) PLAN=${2:-}; shift 2 ;;
    --upload) UPLOAD=${2:-}; shift 2 ;;
    --work) WORK=${2:-}; shift 2 ;;
    --out) OUT=${2:-}; shift 2 ;;
    --slug) SLUG=${2:-}; shift 2 ;;
    --gpu-env) GPU_ENV=${2:-}; shift 2 ;;
    --cpu-env) CPU_ENV=${2:-}; shift 2 ;;
    --deadline-h) DEADLINE_H=${2:-}; shift 2 ;;
    --dry-run) REPRO_DRY_RUN=1; shift ;;
    -h|--help) sed -n '5,22p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

# The plans: NAME:EXPECTED_MIN:CAP_MIN per step, and the session cap in minutes. Expected minutes are derived, not guessed:
#   C1 = sum over (arm, cell) of 10 s process + (warm-up + 3 rounds) x the per-run wall, a warm-up only for an arm over the
#        120 s budget, no process at all for an arm predicted over budget at a larger N; + 20 s per (arm, N) of the memory pass.
#   Per-run walls at N=1e6 spread (uniform x 0.5, spread-short x 1/16, N=1e7 x 10, fp32 x 0.6):
#     P2000: eagle_graph 23.9 s, persist and simulate 13.6 s (measured by the rehearsal); warp 25.1 s and cupy_masked 286 s
#            scaled from the T4 ratios to persist (1.85, 21): estimates, to be replaced by the rehearsal's card.
#     T4:    graph 7.22, persist 4.28, simulate 4.28, cupy 90.6, torch_graphed 95.3, torch_compiled 23.1, jax 13.0, warp 7.9 s
#            (the Kaggle T4 card of the 11th run); + 40 s / 30 s compile per torch_compiled / jax process.
#     A100:  the T4 walls / 39 (FP64 9.7 / 0.25 TFLOP/s); L4: the T4 walls / 1.9 (FP64 0.47 / 0.25 TFLOP/s).
#   Estimates: P2000 24.8 min (timing N <= 1e6, memory to 1e7), L4 25.7, A100 27.0, T4 89.0 (the five cells, 1e7 included).
#   C2: the P2000 run measured 7.6 min without torch; T4, L4 and A100 add the torch arms and, on the T4 pair, mpi_partition.
#   P0, P4, P5, ncu: the stage's own cap in profile.sh (sol_probe 180 s, chunk_residency 900 s, launch_shape 600 s,
#   ncu_ensemble --quick 300 s) + 1 min; P1 (the full sweep): the quick sweep took 30 s for 18 cells, the full grid is 90.
#   A step cap is 1.3 x the estimate and never below its stage's own cap; p25 (stages 2-5) stops starting stages after 20 min.
# p25 = profile stages 2-5; ncu = the quick Nsight Compute smoke of the L4.
case $PLAN in
  a100)  STEPS=(p0:1:4 p1:4:6 c1:27:36 c2:14:18 p25:18:35)         SESSION_CAP=120; KIND=GPU ;;
  l4)    STEPS=(p0:1:4 p1:4:6 c1:26:34 c2:12:16 p4:8:16 ncu:2:6)   SESSION_CAP=90;  KIND=GPU ;;
  t4)    STEPS=(p0:1:4 p1:4:6 c1:89:116 c2:12:16 p4:8:16)          SESSION_CAP=180; KIND=GPU ;;
  p2000) STEPS=(p0:1:4 p1:4:6 c1:25:33 c2:8:11 p4:8:16 p5:3:11)    SESSION_CAP=90;  KIND=GPU ;;
  zen5|c4a) STEPS=(c1cpu:15:20 c2cpu:8:11)                         SESSION_CAP=45;  KIND=CPU ;;
  *) die "--plan takes a100, l4, t4, p2000, zen5 or c4a" ;;
esac

TARGET=$(detect_target)
WORK=${WORK:-$(default_work "$TARGET")}
GPU_ENV=${GPU_ENV:-$WORK/env-gpu.sh}
CPU_ENV=${CPU_ENV:-$WORK/env-cpu.sh}
ENVF=$GPU_ENV; [[ $KIND == CPU ]] && ENVF=$CPU_ENV
SLUG=${SLUG:-$(machine_slug)}
OUT=${OUT:-$WORK/cards}
RES=$OUT/$SLUG
START=$(date +%s)

# The eagle tree the cards run from: the env's source checkout, else this one.
src=""
[[ -f $ENVF ]] && src=$(sed -n 's/.*REPRO_SRC=\([^ ]*\).*/\1/p' "$ENVF" | head -1)
if [[ -n $src && -d $src/eagle ]]; then EROOT=$src/eagle; else EROOT=$(cd "$HERE/../.." && pwd); fi

# step_cmd NAME: the command of a step, one word per line, run from the eagle root in the env.
step_cmd() {
  case $1 in
    p0)    printf '%s\n' python benchmarks/reproduce/gpu_profile.py --stage sol_probe --out "$RES/profile" ;;
    p1)    printf '%s\n' python -m hawk.targets.cards.sweep --write --out "$RES/wide_sweep.json" --cache "$RES/sweep_cache" ;;
    c1)    if [[ $PLAN == l4 || $PLAN == p2000 ]]; then   # timing to 1e6, memory to 1e7
             printf '%s\n' python benchmarks/perf_card/perf_card.py --ns 1000000 --memory-ns 1000000 10000000 --parts "$RES/parts/c1" --out-dir "$RES/perf_card"
           else
             printf '%s\n' python benchmarks/perf_card/perf_card.py --parts "$RES/parts/c1" --out-dir "$RES/perf_card"
           fi ;;
    c1cpu) printf '%s\n' python benchmarks/perf_card/cpu_card.py --out-dir "$RES/perf_card" ;;
    c2|c2cpu) printf '%s\n' python benchmarks/perf_card/feature_card.py --parts "$RES/parts/c2" --out-dir "$RES/perf_card" ;;
    p25)   printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --deadline-min 20 --only ncu_corpus,ncu_ensemble,chunk_residency,launch_shape ;;
    p4)    printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only chunk_residency ;;
    p5)    printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only launch_shape ;;
    ncu)   printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only ncu_ensemble --quick ;;
  esac
}

expected=0 capsum=0
log "plan $PLAN on $SLUG (target $TARGET)"
for s in "${STEPS[@]}"; do
  IFS=: read -r name exp cap <<<"$s"
  printf '[repro]   %-7s expected %3s min, cap %3s min\n' "$name" "$exp" "$cap" >&2
  expected=$((expected + exp)); capsum=$((capsum + cap))
done
log "expected $expected min, step caps $capsum min, session cap $SESSION_CAP min (boot and upload not counted)"
(( capsum <= SESSION_CAP )) || die "plan $PLAN: step caps ($capsum min) exceed the session cap ($SESSION_CAP min)"
[[ $REPRO_DRY_RUN == 1 ]] && log "dry run: commands are printed, nothing is run"

MARKS=$RES/.steps
run mkdir -p "$MARKS"
if [[ -n $UPLOAD && $REPRO_DRY_RUN == 0 ]]; then gcloud storage rsync -r "$UPLOAD/$SLUG" "$RES" >/dev/null 2>&1 || true; fi
push() {
  [[ -n $UPLOAD ]] || return 0
  run gcloud storage rsync -r "$RES" "$UPLOAD/$SLUG" || log "upload failed: the results stay in $RES"
}

failed=0
for s in "${STEPS[@]}"; do
  IFS=: read -r name _ cap <<<"$s"
  if [[ $REPRO_DRY_RUN == 0 && -f $MARKS/$name.done ]]; then log "step $name: done earlier, skipped"; continue; fi
  if [[ -n $DEADLINE_H ]] && (( $(date +%s) - START > $(awk -v h="$DEADLINE_H" 'BEGIN {printf "%d", h * 3600}') )); then
    log "deadline of ${DEADLINE_H} h reached before step $name; run the same command to resume"
    exit 3
  fi
  mapfile -t cmd < <(step_cmd "$name")
  log "step $name (cap $cap min)"
  if [[ $REPRO_DRY_RUN == 1 ]]; then
    # shellcheck disable=SC2016  # $RAPTOR_HAWK_ROOT is printed, not expanded
    printf '+ (. %s; cd %s; HAWK_SRC=$RAPTOR_HAWK_ROOT EAGLE_SRC=%s timeout %sm %s)\n' "$ENVF" "$EROOT" "$EROOT" "$cap" "$(_quote "${cmd[@]}")"
    push
    continue
  fi
  [[ -f $ENVF ]] || die "env script $ENVF not found: run bootstrap.sh --profile ${KIND,,} first"
  mkdir -p "$RES"
  if ( # shellcheck disable=SC1090
       . "$ENVF"
       [[ $KIND == CPU ]] && export CUDA_VISIBLE_DEVICES=""
       export HAWK_SRC=${RAPTOR_HAWK_ROOT:-${HAWK_SRC:-}} EAGLE_SRC=$EROOT
       cd "$EROOT" && timeout "${cap}m" "${cmd[@]}" ); then
    touch "$MARKS/$name.done"
  else
    log "step $name FAILED or ran over its cap of $cap min; the next step runs"
    failed=1
  fi
  push
done

if [[ $REPRO_DRY_RUN == 0 ]]; then
  log "inventory of $RES"
  ( cd "$RES" && find . -type f ! -path './sweep_cache/*' ! -path './.steps/*' -printf '%8s  %p\n' | sort -k2 | head -200 ) >&2
  md5sum "$RES"/perf_card/*.json >&2 2>/dev/null || true
fi
(( failed == 0 )) || exit 1
log "results in $RES"
