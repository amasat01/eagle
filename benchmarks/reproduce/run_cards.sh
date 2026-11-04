#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Run one device's session plan of eagle's cards in an environment made by bootstrap.sh.
#
#   run_cards.sh --plan NAME [options]
#     --plan NAME     a100 | l4 | l4-prof | t4 | p2000 | zen5 | c4a: the steps of that device, in session order
#                     (P0 sol_probe, P1 wide_sweep with its rows written, C1, C2, then the profile stages)
#     --upload URL    after each step, copy the results to URL/<slug> (gcloud storage rsync); a new machine given
#                     the same URL pulls what is there first and skips the steps and the C1/C2 parts already done
#     --work DIR      bootstrap's work directory (default: $REPRO_WORK, else per target)
#     --out DIR       results root (default WORK/cards); results go to OUT/<machine-slug>/
#     --slug S        machine slug (default: <gpu>--<cpu>, from the hardware)
#     --gpu-env FILE  env script of the gpu profile (default WORK/env-gpu.sh)
#     --cpu-env FILE  env script of the cpu profile (default WORK/env-cpu.sh)
#     --deadline-h H  the session deadline in hours (default: 1.5 x the plan's expected minutes)
#     --smoke         the same steps in the same order at their smallest sizes (one round, --quick stages), the
#                     plumbing check before a session: results and step markers go to <slug>-smoke (locally and in the
#                     bucket), so a full run afterwards on the same machine skips nothing and reuses the environment;
#                     the sweep's rows are not written into hawk
#     --ours-only     measure our arms only (eagle / hawk / aether): C1 and C2 time just those arms, the baseline libraries'
#                     rows come from --reference; the cmp (compile caches) and nvl (nvlibs counters) steps are dropped
#     --reference DIR the results directory (<slug>/) of the device's earlier run, required by --ours-only; its external
#                     rows are labelled in the card with their source; a different GPU is refused
#     --dry-run       print the plan with its minutes and every command, run nothing
#
# A step that fails or runs over its cap is reported and the next one runs (exit 1 at the end); a step that
# finished leaves a marker, so running the same command again continues where a preempted run stopped.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
. "$HERE/lib.sh"

PLAN="" UPLOAD="" WORK=${REPRO_WORK:-} OUT="" SLUG="" GPU_ENV="" CPU_ENV="" DEADLINE_H="" SMOKE=0 OURS_ONLY=0 REFERENCE=""
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
    --smoke) SMOKE=1; shift ;;
    --ours-only) OURS_ONLY=1; shift ;;
    --reference) REFERENCE=${2:-}; shift 2 ;;
    --dry-run) REPRO_DRY_RUN=1; shift ;;
    -h|--help) sed -n '5,25p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

# The plans: NAME:EXPECTED_MIN:WORST_MIN per step. Expected minutes (the cost estimate) are derived, not guessed:
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
# Cost is bounded by a session DEADLINE, not by step sums: 1.5 x the plan's expected minutes (boot and upload excluded), checked
#   before EVERY arm by the card scripts ($CARD_DEADLINE) and before every step here. When it passes, no new arm starts, the arms
#   not yet run read "not_run: session time limit" (the lowest-priority rows go first: see the order comment in each card
#   script), every card is assembled from its saved parts and the steps not started are listed. A step's timeout is a hang
#   guard only: the time left to the deadline + 12 min (one arm in flight: build 120 s + warm-up 120 s + rounds 300 s + a round
#   120 s); the stage steps keep their own caps (sol_probe 180 s, wide_sweep 600 s, chunk_residency 900 s, launch_shape 600 s,
#   ncu_ensemble --quick 300 s, + 1 min; the full profile stages' caps are profile.sh's, below). A step cap "D" means that hang guard.
# pcpu = the CPU profile step (ISA A/B, thread sweep, perf stat/record, small-N baselines, memory phases): 18 min, the local
#   sequence (about 35 min on a busy 4-core host) x 0.5 for the Zen5 session.
# ncu = the quick Nsight Compute smoke of the L4. The full profile steps (USER 11-03: every counter, every workload; the card
#   timings above stay unprofiled, these run after them in their own processes) are one profile.sh stage each, --set full:
#   corp = ncu_corpus (not in the a100 plan: the sweep answers team vs staged), ens = ncu_ensemble, feat = ncu_features, nvl = ncu_nvlibs, p4 = chunk_residency (timing grid + captures of its ties),
#   p5 = launch_shape (nsys). A full capture replays each kernel about 40 times: 30 s of process + (0.12 s + the kernel's own
#   time) x 40 per kernel, the kernels of one run capped at 32 (the caps below are profile.sh's, ~1.5 x the L4 walls).
#   Only our code is profiled (eagle / hawk kernels), never a baseline library: ens = the card's 5 eagle arms, feat = the 39
#   feature-card cells of ours (feature_card.OUR_ARMS), launch_shape drops Warp.
#   L4 (from the 073l4 card walls: persist/simulate 2.3 s spread, 2.0 uniform):
#     ens  = 10 captures (5 arms x spread, uniform; the attempt kernel, one launch) x 30 s + 40 x (persist and simulate kernel
#            times 17 s) + 6 short captures x 40 x 0.12 s = 10 min;  A100: kernel times / 20 (FP64) = 6 min, minus the 2 simulate
#            captures (a100 plan: --a100, same kernel as persist) x 30 s = 5.
#     feat = 43 cells (the 0.7.3 items among them: ordered sum and scatter, wide_forms team_k8, steps jvp_K8 and vjp_K8_auto, linalg_lane
#            n32 f32/f64, linalg_team n512, the sample_sum cell arms; their expectations ride in the row note) x (15 s + ~6 kernels x 5 s)
#            + the long kernels (segmented 3.7 s, ordered 0.8 s, x 40) = 39 cells 43 min + 4 new x 0.75 = 46 min; A100 36 + 3 = 39.
#     corp = <= 12 cells x (30 s + 40 x 0.2 s) x ~ 1.2 = 12 min; A100 10.
#     p4   = the chunk grid (64 cells, 17 min of runs + builds, 21 min) + <= 8 tie captures x 75 s = 32 min; A100: FP64 /20 -> 13.
#            L4 plans drop tps=2x at N=1e7 (16 cells = 10 x the 1e6 wall each: 7.7 min of runs + 1 min of builds): grid 12 + 10 = 22.
#            a100 (--a100): all 64 cells, no tie captures: 13 x 21 / 32 = 9.
#     p5   = nsys, no counters, 3 arms: 5 (L4), 4 (A100).
#     cmp  = l4-prof only (it runs no card, so the compile caches would stay empty): perf_card.py and feature_card.py --smoke into
#            $RES/compile_smoke (no card lands in perf_card/): every arm builds and compiles once. perf_card ~11 arms x 20 s +
#            torch_compiled's compile (limit 10 min) = 14 min; feature_card ~80 arms x 4 s (its full card ran 254 s) = 6 min; 20 min.
#     nvl  = ncu_nvlibs: 11 cells (closed NVIDIA library kernels of the rows eagle loses; counters only, 7 sections ~ 12 passes, never
#            --set full or source) x (15 s + ~10 kernels x 12 x 0.15 s) = 6 min, cuSOLVER's longer kernels + 4 = 10 (L4); 8 (A100).
#   NCU capture cap 600 s (persist's 40 passes x 2.3 s = 92 s + build; segmented's 150 s); stage caps ens 15 min, feat 65.
#   l4-prof = p0, p1, cmp, then the six profile steps (130 min; deadline 1.5 x); a100 = today's card steps, then the five without corp (111).
case $PLAN in
  a100)  STEPS=(p0:1:4 p1:4:11 c1:27:D c2:14:D ens:5:25 feat:39:70 nvl:8:25 p4:9:55 p5:4:16) ; KIND=GPU ;;
  l4-prof) STEPS=(p0:1:4 p1:4:11 cmp:20:40 corp:12:35 ens:10:25 feat:46:70 nvl:10:25 p4:22:55 p5:5:16) ; KIND=GPU ;;
  l4)    STEPS=(p0:1:4 p1:4:11 c1:26:D c2:12:D p4:8:16 ncu:2:6) ; KIND=GPU ;;
  t4)    STEPS=(p0:1:4 p1:4:11 c1:89:D c2:12:D p4:8:16) ;        KIND=GPU ;;
  p2000) STEPS=(p0:1:4 p1:4:11 c1:25:D c2:8:D p4:8:16 p5:3:11) ; KIND=GPU ;;
  zen5|c4a) STEPS=(c1cpu:15:D c2cpu:8:D pcpu:18:D) ;                       KIND=CPU ;;
  *) die "--plan takes a100, l4, l4-prof, t4, p2000, zen5 or c4a" ;;
esac
# ours-only: no baseline-library step (cmp, nvl); C1 and C2 time our arms only (the eagle arms are 5 of 11 C1 arms and ~6 % of the
# T4 per-run walls, the fixed per-cell process costs stay: C1 estimated at 1/3, C2 at 1/2 of the full minutes)
if [[ $OURS_ONLY == 1 ]]; then
  [[ -n $REFERENCE && -d $REFERENCE/parts ]] || die "--ours-only needs --reference DIR, the results directory of this device's earlier run (with parts/)"
  [[ $KIND == GPU ]] || die "--ours-only is for the GPU plans"
  KEPT=()
  for s in "${STEPS[@]}"; do
    IFS=: read -r name exp cap <<<"$s"
    case $name in cmp|nvl) continue ;; c1) exp=$(( (exp + 2) / 3 )) ;; c2) exp=$(( (exp + 1) / 2 )) ;; esac
    KEPT+=("$name:$exp:$cap")
  done
  STEPS=("${KEPT[@]}")
fi
# smoke: every step about 2 min (the card scripts' --smoke: N = 1e3, one round; the stages' --quick)
if [[ $SMOKE == 1 ]]; then for i in "${!STEPS[@]}"; do STEPS[i]=${STEPS[i]%%:*}:2:D; done; fi

TARGET=$(detect_target)
WORK=${WORK:-$(default_work "$TARGET")}
GPU_ENV=${GPU_ENV:-$WORK/env-gpu.sh}
CPU_ENV=${CPU_ENV:-$WORK/env-cpu.sh}
ENVF=$GPU_ENV; [[ $KIND == CPU ]] && ENVF=$CPU_ENV
SLUG=${SLUG:-$(machine_slug)}
# another GPU's results are refused here (the slug is <gpu>--<cpu>); the card scripts check the device itself
if [[ $OURS_ONLY == 1 && $(basename "$REFERENCE") != "${SLUG%%--*}"--* ]]; then
  die "--reference $REFERENCE is another device's run (this one: ${SLUG%%--*}): refused"
fi
[[ $SMOKE == 1 ]] && SLUG=$SLUG-smoke
OUT=${OUT:-$WORK/cards}
RES=$OUT/$SLUG
START=$(date +%s)

# The eagle tree the cards run from: the env's source checkout, else this one.
src=""
[[ -f $ENVF ]] && src=$(sed -n 's/.*REPRO_SRC=\([^ ]*\).*/\1/p' "$ENVF" | head -1)
if [[ -n $src && -d $src/eagle ]]; then EROOT=$src/eagle; else EROOT=$(cd "$HERE/../.." && pwd); fi

# step_cmd NAME: the command of a step, one word per line, run from the eagle root in the env (smoke: smoke_cmd's edit).
step_cmd() {
  if [[ $SMOKE == 1 ]]; then full_cmd "$1" | smoke_cmd "$1"; else full_cmd "$1"; fi
  if [[ $OURS_ONLY == 1 ]]; then case $1 in c1|c2) printf '%s\n' --reference "$REFERENCE" ;; esac; fi
}
# smoke_cmd NAME: a full command at its smallest: the card scripts' --smoke, the stages' --quick, the sweep --quick unwritten.
smoke_cmd() {
  local w
  while IFS= read -r w; do
    case $1:$w in p1:--write) continue ;; esac
    printf '%s\n' "$w"
  done
  case $1 in
    c1|c1cpu|c2|c2cpu|pcpu) printf '%s\n' --smoke ;;
    p1|corp|ens|feat|nvl|p4|p5) printf '%s\n' --quick ;;
  esac
}
full_cmd() {
  case $1 in
    p0)    printf '%s\n' python benchmarks/reproduce/gpu_profile.py --stage sol_probe --out "$RES/profile" ;;
    p1)    printf '%s\n' python -m hawk.targets.cards.sweep --write --out "$RES/profile/wide_sweep.json" --cache "$RES/profile/sweep_cache" ;;
    c1)    if [[ $PLAN == l4 || $PLAN == p2000 ]]; then   # timing to 1e6, memory to 1e7
             printf '%s\n' python benchmarks/perf_card/perf_card.py --ns 1000000 --memory-ns 1000000 10000000 --parts "$RES/parts/c1" --out-dir "$RES/perf_card"
           else
             printf '%s\n' python benchmarks/perf_card/perf_card.py --parts "$RES/parts/c1" --out-dir "$RES/perf_card"
           fi ;;
    c1cpu) printf '%s\n' python benchmarks/perf_card/cpu_card.py --parts "$RES/parts/c1cpu" --out-dir "$RES/perf_card" ;;
    pcpu)  printf '%s\n' python benchmarks/perf_card/pcpu.py --parts "$RES/parts/pcpu" --out-dir "$RES/pcpu" ;;
    c2|c2cpu) printf '%s\n' python benchmarks/perf_card/feature_card.py --parts "$RES/parts/c2" --out-dir "$RES/perf_card" ;;
    cmp)   printf '%s\n' bash -c 'rc=0; python benchmarks/perf_card/perf_card.py --smoke --out-dir "$1" || rc=1; python benchmarks/perf_card/feature_card.py --smoke --out-dir "$1" || rc=1; exit $rc' _ "$RES/compile_smoke" ;;
    corp)  printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only ncu_corpus ;;
    ens)   printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only ncu_ensemble
           [[ $PLAN != a100 ]] || printf '%s\n' --a100 ;;
    nvl)   printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only ncu_nvlibs ;;
    feat)  printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only ncu_features ;;
    p4)    # the card plans keep the 900 s chunk stage (--deadline-min 15 clips profile.sh's cap); the profile plans the full one
           printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only chunk_residency
           [[ $PLAN == l4-prof || $PLAN == a100 ]] || printf '%s\n' --deadline-min 15
           [[ $PLAN != a100 ]] || printf '%s\n' --a100 ;;
    p5)    printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only launch_shape ;;
    ncu)   printf '%s\n' bash benchmarks/reproduce/profile.sh --env "$ENVF" --out "$RES/profile" --only ncu_ensemble --quick ;;
  esac
}

expected=0
log "plan $PLAN on $SLUG (target $TARGET)"
for s in "${STEPS[@]}"; do
  IFS=: read -r name exp cap <<<"$s"
  printf '[repro]   %-7s expected %4s min\n' "$name" "$exp" >&2
  expected=$((expected + exp))
done
DEADLINE_MIN=$(( (expected * 3 + 1) / 2 ))
[[ -n $DEADLINE_H ]] && DEADLINE_MIN=$(awk -v h="$DEADLINE_H" 'BEGIN {printf "%d", h * 60}')
DEADLINE_AT=$(( START + DEADLINE_MIN * 60 ))
log "expected $expected min; session deadline $DEADLINE_MIN min (1.5 x); --max-run-duration = deadline + boot + upload + 15 = $((DEADLINE_MIN + 15)) min + boot"
[[ $KIND == GPU && $OURS_ONLY == 0 ]] && log "compile caches (warp, jax, torch, triton, cupy) under $RES/compiled/<lib>; card timings unchanged"
[[ $REPRO_DRY_RUN == 1 ]] && log "dry run: commands are printed, nothing is run"
export CARD_DEADLINE=$DEADLINE_AT

MARKS=$RES/.steps
run mkdir -p "$MARKS"
if [[ -n $UPLOAD && $REPRO_DRY_RUN == 0 ]]; then gcloud storage rsync -r "$UPLOAD/$SLUG" "$RES" >/dev/null 2>&1 || true; fi
push() {
  [[ -n $UPLOAD ]] || return 0
  run gcloud storage rsync -r "$RES" "$UPLOAD/$SLUG" || log "upload failed: the results stay in $RES"
}

# in_env CMD...: CMD in the plan's env from the eagle root (HAWK_SRC and EAGLE_SRC set; no GPU for a CPU plan).
# compile_caches: the open-source libraries' compile caches under $RES/compiled/<lib>, uploaded with the results: files only, no
# timed span changes (XLA's dump is written when a program compiles, which every card does before its timed rounds).
compile_caches() {
  printf '%s\n' "WARP_CACHE_PATH=$RES/compiled/warp" "XLA_FLAGS=${XLA_FLAGS:-} --xla_dump_to=$RES/compiled/jax" \
    "TORCHINDUCTOR_CACHE_DIR=$RES/compiled/torch" "TRITON_CACHE_DIR=$RES/compiled/triton" \
    "CUPY_CACHE_DIR=$RES/compiled/cupy" "CUPY_CACHE_SAVE_CUDA_SOURCE=1"
}
in_env() {
  ( # shellcheck disable=SC1090
    . "$ENVF"
    if [[ $KIND == GPU && $OURS_ONLY == 0 ]]; then while IFS= read -r kv; do export "$kv"; done < <(compile_caches); fi
    [[ $KIND == CPU ]] && export CUDA_VISIBLE_DEVICES=""
    export HAWK_SRC=${RAPTOR_HAWK_ROOT:-${HAWK_SRC:-}} EAGLE_SRC=$EROOT CARD_DEADLINE
    cd "$EROOT" && "$@" )
}

# assemble NAME: the card of a C1/C2 step written from its saved parts, a missing arm labelled not_run: step time limit.
assemble() {
  case $1 in c1|c1cpu|c2|c2cpu) ;; *) return 0 ;; esac
  mapfile -t acmd < <(step_cmd "$1")
  log "assembling the card of step $1 from its saved parts"
  if [[ $REPRO_DRY_RUN == 1 ]]; then printf '+ (... %s --assemble)\n' "$(_quote "${acmd[@]}")"; return 0; fi
  in_env timeout 10m "${acmd[@]}" --assemble || log "assembling step $1 failed"
}
finish_cards() {   # session end (or a deadline): every C1/C2 step that did not finish still leaves a complete labelled card
  local st
  for st in "${STEPS[@]}"; do
    [[ -f $MARKS/${st%%:*}.done ]] || assemble "${st%%:*}"
  done
  push
}

failed=0 skipped=()
for s in "${STEPS[@]}"; do
  IFS=: read -r name _ cap <<<"$s"
  if [[ $REPRO_DRY_RUN == 0 && -f $MARKS/$name.done ]]; then log "step $name: done earlier, skipped"; continue; fi
  if (( $(date +%s) > DEADLINE_AT )); then
    skipped+=("$name"); continue
  fi
  mapfile -t cmd < <(step_cmd "$name")
  [[ $cap == D ]] && cap=$(( (DEADLINE_AT - $(date +%s)) / 60 + 12 ))
  log "step $name (hang guard $cap min)"
  if [[ $REPRO_DRY_RUN == 1 ]]; then
    # shellcheck disable=SC2016  # $RAPTOR_HAWK_ROOT is printed, not expanded
    printf '+ (. %s; cd %s; HAWK_SRC=$RAPTOR_HAWK_ROOT EAGLE_SRC=%s timeout %sm %s)\n' "$ENVF" "$EROOT" "$EROOT" "$cap" "$(_quote "${cmd[@]}")"
    push
    continue
  fi
  [[ -f $ENVF ]] || die "env script $ENVF not found: run bootstrap.sh --profile ${KIND,,} first"
  mkdir -p "$RES"
  if in_env timeout "${cap}m" "${cmd[@]}"; then
    touch "$MARKS/$name.done"
  else
    log "step $name FAILED or ran over its cap of $cap min; the next step runs"
    failed=1
    assemble "$name"
  fi
  push
done

[[ ${#skipped[@]} -gt 0 ]] && log "session deadline passed: steps not started: ${skipped[*]} (their cards, if any, are assembled with the arms labelled not_run: session time limit)"
finish_cards
if [[ $REPRO_DRY_RUN == 0 ]]; then
  log "inventory of $RES"
  ( cd "$RES" && find . -type f ! -path './profile/sweep_cache/*' ! -path './.steps/*' -printf '%8s  %p\n' | sort -k2 | head -200 ) >&2
  md5sum "$RES"/perf_card/*.json >&2 2>/dev/null || true
fi
(( failed == 0 )) || exit 1
log "results in $RES"
