# shellcheck shell=bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for bootstrap.sh and run_cards.sh (sourced, never executed).
# Detection can be mocked for dry runs and tests:
#   REPRO_TARGET=kaggle|vastai|devbox|local   skip target detection
#   REPRO_GPUS="none" | "name,compute_cap,driver;name,..."   skip nvidia-smi

REPRO_DRY_RUN=${REPRO_DRY_RUN:-0}

log() { printf '[repro] %s\n' "$*" >&2; }
die() { printf '[repro] ERROR: %s\n' "$*" >&2; exit 1; }

# run CMD...: print the command; execute it unless this is a dry run.
run() {
  printf '+ %s\n' "$(_quote "$@")"
  [[ $REPRO_DRY_RUN == 1 ]] && return 0
  "$@"
}

# run_in DIR CMD...: run in a directory (printed as a cd).
run_in() {
  local dir=$1; shift
  printf '+ (cd %s && %s)\n' "$(_quote "$dir")" "$(_quote "$@")"
  [[ $REPRO_DRY_RUN == 1 ]] && return 0
  (cd "$dir" && "$@")
}

_quote() {
  local out="" a
  for a in "$@"; do
    if [[ $a =~ ^[A-Za-z0-9_./:=,+@%-]+$ ]]; then out+="$a "; else out+="$(printf '%q' "$a") "; fi
  done
  printf '%s' "${out% }"
}

# detect_target: kaggle | vastai | local | devbox
#   local  = this script sits in an eagle checkout with aether/hawk/raptor beside it
#   devbox = anything else (a fresh machine: repositories are cloned)
detect_target() {
  if [[ -n ${REPRO_TARGET:-} ]]; then echo "$REPRO_TARGET"; return; fi
  if [[ -n ${KAGGLE_KERNEL_RUN_TYPE:-} || -d /kaggle/working ]]; then echo kaggle; return; fi
  if [[ -n ${VAST_CONTAINERLABEL:-}${VAST_TCP_PORT_22:-}${CONTAINER_ID:-} ]]; then echo vastai; return; fi
  local root; root=$(checkout_root)
  if [[ -n $root ]]; then echo local; else echo devbox; fi
}

# checkout_root: the directory holding eagle/ and its siblings, when this
# script runs from inside such a family checkout; empty otherwise.
checkout_root() {
  local here; here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local root; root=$(cd "$here/../../.." && pwd)
  if [[ -d $root/eagle/python && -d $root/aether && -d $root/hawk && -d $root/raptor ]]; then
    echo "$root"
  fi
}

# detect_gpus: one "name,compute_cap,driver" line per GPU visible to CUDA
# (CUDA_VISIBLE_DEVICES honoured as PCI-order indices; empty = no GPU).
detect_gpus() {
  if [[ -n ${REPRO_GPUS:-} ]]; then
    [[ $REPRO_GPUS == none ]] || tr ';' '\n' <<<"$REPRO_GPUS"
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local sel=()
  if [[ -n ${CUDA_VISIBLE_DEVICES+x} ]]; then
    [[ -z $CUDA_VISIBLE_DEVICES || $CUDA_VISIBLE_DEVICES == -1 ]] && return 0
    [[ $CUDA_VISIBLE_DEVICES =~ ^[0-9,]+$ ]] && sel=(-i "$CUDA_VISIBLE_DEVICES")
  fi
  nvidia-smi "${sel[@]}" --query-gpu=name,compute_cap,driver_version \
    --format=csv,noheader 2>/dev/null | sed 's/, /,/g' || true
}

# gpu_archs: "60;75"-style CMake arch list of the visible GPUs (unique, sorted).
gpu_archs() {
  detect_gpus | awk -F, 'NF>=2 {gsub(/\./,"",$2); print $2}' | sort -un | paste -sd';' -
}

slugify() { tr '[:upper:]' '[:lower:]' <<<"$*" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'; }

cpu_model() {
  local m; m=$(awk -F: '/^model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)
  m=${m# }; echo "${m:-$(uname -m)}"
}

# machine_slug: "<gpu>--<cpu>" (or "cpu--<cpu>"), the output directory name.
machine_slug() {
  local gpu; gpu=$(detect_gpus | head -1 | cut -d, -f1)
  local cpu; cpu=$(slugify "$(cpu_model | sed -E 's/\((R|TM|tm|r)\)//g; s/ CPU//; s/@.*//')")
  echo "$(slugify "${gpu:-cpu}")--${cpu:0:40}"
}

# default_work TARGET: where envs, clones and build trees go.
default_work() {
  case $1 in
    kaggle) local d; for d in /kaggle/temp /kaggle/tmp; do [[ -d $d ]] && { echo "$d/raptor-repro"; return; }; done
            echo /tmp/raptor-repro ;;
    vastai) if [[ -d /workspace ]]; then echo /workspace/raptor-repro; else echo "$HOME/raptor-repro"; fi ;;
    *)      echo "${XDG_CACHE_HOME:-$HOME/.cache}/raptor-repro" ;;
  esac
}
