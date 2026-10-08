#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Run eagle's performance cards in an environment made by bootstrap.sh.
#
#   run_cards.sh [options] [-- extra arguments passed to every card script]
#     --which W       gpu | cpu | all   (default: gpu when the GPU env exists, else cpu)
#     --preset P      smoke | short | full   (default full = the committed cards' matrix)
#                       smoke: N=1e3, 5 reps, timing only (minutes)
#                       short: N=1e3..1e5, every pass
#     --gpu-env FILE  env script of the gpu profile (default WORK/env-gpu.sh)
#     --cpu-env FILE  env script of the cpu profile (default WORK/env-cpu.sh)
#     --work DIR      bootstrap's work directory (default: $REPRO_WORK, else per target)
#     --out DIR       output root (default WORK/cards); cards go to OUT/<machine-slug>/
#     --slug S        machine slug (default: <gpu>--<cpu>, from the hardware)
#     --publish       write into eagle/benchmarks/{perf_card,rk78_card}/, replacing
#                     the committed card of the same device; never done otherwise
#     --split         run the GPU cards pass by pass and the CPU RK4 card config by
#                     config, resumable (default on Kaggle); re-run the same command
#                     to continue after a session ends
#     --deadline-h H  stop before starting a step once H hours have passed (exit 3)
#     --nsys M        auto | off   (auto: Nsight Systems kernel times when nsys is on PATH)
#     --dry-run       print every command, run nothing
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
. "$HERE/lib.sh"

WHICH="" PRESET=full GPU_ENV="" CPU_ENV="" WORK=${REPRO_WORK:-} OUT="" SLUG="" PUBLISH=0
SPLIT="" DEADLINE_H="" NSYS=auto EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --which) WHICH=$2; shift 2 ;;
    --preset) PRESET=$2; shift 2 ;;
    --gpu-env) GPU_ENV=$2; shift 2 ;;
    --cpu-env) CPU_ENV=$2; shift 2 ;;
    --env) case $2 in *cpu*) CPU_ENV=$2 ;; *) GPU_ENV=$2 ;; esac; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --slug) SLUG=$2; shift 2 ;;
    --publish) PUBLISH=1; shift ;;
    --split) SPLIT=1; shift ;;
    --no-split) SPLIT=0; shift ;;
    --deadline-h) DEADLINE_H=$2; shift 2 ;;
    --nsys) NSYS=$2; shift 2 ;;
    --dry-run) REPRO_DRY_RUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '5,28p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

TARGET=$(detect_target)
WORK=${WORK:-$(default_work "$TARGET")}
GPU_ENV=${GPU_ENV:-$WORK/env-gpu.sh}
CPU_ENV=${CPU_ENV:-$WORK/env-cpu.sh}
if [[ -z $WHICH ]]; then
  if [[ -f $GPU_ENV || ($REPRO_DRY_RUN == 1 && -n $(detect_gpus)) ]]; then WHICH=gpu; else WHICH=cpu; fi
fi
case $WHICH in gpu|cpu|all) ;; *) die "--which must be gpu, cpu or all" ;; esac
case $PRESET in smoke|short|full) ;; *) die "--preset must be smoke, short or full" ;; esac
[[ -z $SPLIT ]] && SPLIT=$([[ $TARGET == kaggle ]] && echo 1 || echo 0)
SLUG=${SLUG:-$(machine_slug)}
OUT=${OUT:-$WORK/cards}
START=$(date +%s)

# The eagle tree the cards run from: the env's source checkout, else this one.
eagle_root() {
  local f=$1 src=""
  [[ -f $f ]] && src=$(sed -n 's/.*REPRO_SRC=\([^ ]*\).*/\1/p' "$f" | head -1)
  if [[ -n $src && -d $src/eagle ]]; then echo "$src/eagle"; else (cd "$HERE/../.." && pwd); fi
}

# Presets: per-card arguments ("--no-*" passes are skipped in split mode too).
NS_SMOKE=(1000) NS_SHORT=(1000 10000 100000)
case $PRESET in
  smoke) NS=("${NS_SMOKE[@]}")
         A_PERF=(--reps 5 --no-sweep --no-nsys --no-memory --no-compile)
         A_RK78=(--reps 5 --no-nsys --no-memory --no-compile)
         A_CPU=(--reps 5 --configs spread:100 --allow-partial --no-memory --no-compile)
         A_CPU78=(--reps 5) ;;
  short) NS=("${NS_SHORT[@]}")
         A_PERF=(--sweep-ns 100000) A_RK78=() A_CPU=(--allow-partial) A_CPU78=() ;;
  full)  NS=() A_PERF=() A_RK78=() A_CPU=() A_CPU78=() ;;
esac
if [[ $NSYS == off ]] || ! command -v nsys >/dev/null 2>&1; then
  [[ $PRESET == smoke ]] || { A_PERF+=(--no-nsys); A_RK78+=(--no-nsys); }
fi
NSARG=(); [[ ${#NS[@]} -gt 0 ]] && NSARG=(--ns "${NS[@]}")

if [[ $PUBLISH == 1 ]]; then
  [[ $PRESET == full && ${#EXTRA[@]} -eq 0 ]] ||
    die "--publish writes the committed cards: only the full preset, without extra arguments"
  log "--publish: cards are written next to the committed ones in eagle/benchmarks/"
fi
log "target=$TARGET which=$WHICH preset=$PRESET split=$SPLIT slug=$SLUG"
[[ $REPRO_DRY_RUN == 1 ]] && log "dry run: commands are printed, nothing is run"

deadline_check() {
  [[ -z $DEADLINE_H ]] && return 0
  local now; now=$(date +%s)
  if (( now - START > $(awk -v h="$DEADLINE_H" 'BEGIN {printf "%d", h * 3600}') )); then
    log "deadline of ${DEADLINE_H} h reached before step '$1'; re-run the same command to resume"
    exit 3
  fi
}

# step NAME ENVFILE GPU|CPU EAGLE DIR CMD...: one resumable step, in its env.
STEPS=""
step() {
  local name=$1 envf=$2 kind=$3 eroot=$4; shift 4
  local mark=$STEPS/$name.done
  if [[ $REPRO_DRY_RUN == 0 && -f $mark ]]; then log "step $name: done earlier, skipped (delete $mark to re-run)"; return 0; fi
  deadline_check "$name"
  log "step $name"
  if [[ $REPRO_DRY_RUN == 1 ]]; then
    printf '+ (. %s; cd %s; %s%s)\n' "$envf" "$eroot" \
      "$([[ $kind == CPU ]] && echo 'CUDA_VISIBLE_DEVICES= ')" "$(_quote "$@")"
    return 0
  fi
  [[ -f $envf ]] || die "env script $envf not found: run bootstrap.sh --profile ${kind,,} first"
  ( # shellcheck disable=SC1090
    . "$envf"
    [[ $kind == CPU ]] && export CUDA_VISIBLE_DEVICES=""
    cd "$eroot" && "$@" )
  touch "$mark"
}

run_gpu_cards() {
  local eroot; eroot=$(eagle_root "$GPU_ENV")
  local out=$OUT/$SLUG; [[ $PUBLISH == 1 ]] && out=$eroot/benchmarks
  local parts=$OUT/$SLUG/parts
  STEPS=$OUT/$SLUG/.steps-$PRESET; run mkdir -p "$STEPS" "$out/perf_card" "$out/rk78_card"
  if [[ $SPLIT == 0 ]]; then
    step gpu-perf "$GPU_ENV" GPU "$eroot" python benchmarks/perf_card/perf_card.py \
      "${NSARG[@]}" "${A_PERF[@]}" --out-dir "$out/perf_card" "${EXTRA[@]}"
    step gpu-rk78 "$GPU_ENV" GPU "$eroot" python benchmarks/rk78_card/rk78_card.py \
      "${NSARG[@]}" "${A_RK78[@]}" --out-dir "$out/rk78_card" "${EXTRA[@]}"
    return
  fi
  local card args pass n i
  for card in perf rk78; do
    if [[ $card == perf ]]; then args=("${A_PERF[@]}"); else args=("${A_RK78[@]}"); fi
    for pass in nsys memory compile; do
      [[ " ${args[*]} " == *" --no-$pass "* ]] && continue
      if [[ $REPRO_DRY_RUN == 1 ]]; then n=1; else
        # shellcheck disable=SC1090
        n=$( . "$GPU_ENV"; cd "$eroot" && python "$HERE/passes.py" "$card" count "$pass" "${NSARG[@]}" | tail -1)
      fi
      for ((i = 0; i < n; i++)); do
        step "gpu-$card-$pass-$i" "$GPU_ENV" GPU "$eroot" python "$HERE/passes.py" "$card" "$pass" \
          --index "$i" --parts "$parts/$card" "${NSARG[@]}"
      done
    done
    # final: the timing pass + the card, from the saved passes; only the
    # arguments main() still needs (the passes come from --parts).
    local fin=()
    for a in "${args[@]}"; do [[ $a == --no-nsys || $a == --no-memory || $a == --no-compile ]] || fin+=("$a"); done
    step "gpu-$card-final" "$GPU_ENV" GPU "$eroot" python "$HERE/passes.py" "$card" final \
      --parts "$parts/$card" "${NSARG[@]}" "${fin[@]}" --out-dir "$out/${card}_card" "${EXTRA[@]}"
  done
}

run_cpu_cards() {
  local eroot; eroot=$(eagle_root "$CPU_ENV")
  local out=$OUT/$SLUG; [[ $PUBLISH == 1 ]] && out=$eroot/benchmarks
  local parts=$OUT/$SLUG/parts/cpu
  STEPS=$OUT/$SLUG/.steps-$PRESET; run mkdir -p "$STEPS" "$out/perf_card" "$out/rk78_card" "$parts"
  if [[ $SPLIT == 0 || $PRESET == smoke ]]; then
    step cpu-rk4 "$CPU_ENV" CPU "$eroot" python benchmarks/perf_card/cpu_card.py \
      "${NSARG[@]}" "${A_CPU[@]}" --parts-dir "$parts" --out-dir "$out/perf_card" "${EXTRA[@]}"
  else
    local cfg
    for cfg in spread:100 spread:1000 uniform:1000; do
      step "cpu-rk4-timing-${cfg/:/-}" "$CPU_ENV" CPU "$eroot" python benchmarks/perf_card/cpu_card.py \
        "${NSARG[@]}" "${A_CPU[@]}" --configs "$cfg" --no-memory --no-compile --allow-partial \
        --parts-dir "$parts" --out-dir "$parts/partial" "${EXTRA[@]}"
    done
    step cpu-rk4-final "$CPU_ENV" CPU "$eroot" python benchmarks/perf_card/cpu_card.py \
      "${NSARG[@]}" "${A_CPU[@]}" --no-timing --parts-dir "$parts" --out-dir "$out/perf_card" "${EXTRA[@]}"
  fi
  step cpu-rk78 "$CPU_ENV" CPU "$eroot" python benchmarks/rk78_card/cpu_rk78_card.py \
    "${NSARG[@]}" "${A_CPU78[@]}" --out-dir "$out/rk78_card" "${EXTRA[@]}"
}

[[ $WHICH == gpu || $WHICH == all ]] && run_gpu_cards
[[ $WHICH == cpu || $WHICH == all ]] && run_cpu_cards
if [[ $PUBLISH == 1 ]]; then
  log "cards written next to the committed ones; review with git diff before committing"
else
  log "cards in $OUT/$SLUG"
fi
