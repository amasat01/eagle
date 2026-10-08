#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Set up a machine to reproduce eagle's performance cards: a pinned toolchain +
# Python environment from locks/, the four RAPTOR repositories (aether, raptor,
# eagle, hawk) at their release tags or local checkouts, eagle's _core built for
# the GPUs present (CPU-only when there are none), hawk, an import smoke test and
# a machine fact sheet. Re-running it reuses the environment, the clones and
# the build trees (the package builds are incremental).
#
#   bootstrap.sh [options]
#     --target T      kaggle | vastai | devbox | local   (default: detected)
#     --profile P     gpu | cpu | auto   (default auto: gpu when a GPU is visible)
#     --work DIR      envs, clones, build trees, env script, fact sheet
#                     (default per target; see README.md)
#     --src DIR       directory with (or to receive) aether/ raptor/ eagle/ hawk/
#                     checkouts; existing checkouts are used as they are
#                     (default: the enclosing checkout for local, DIR/src otherwise)
#     --ref REPO=REF  ref to clone for REPO (repeatable; or AETHER_REF=... etc.)
#     --arch LIST     CUDA archs for eagle's plugin, e.g. "75" or "60;75"
#                     (default: the visible GPUs')
#     --env-prefix P  the environment prefix (default WORK/envs/cards-PROFILE)
#     --editable      editable installs (development); default: regular installs
#     --jobs N        parallel build jobs (default: all cores)
#     --no-smoke      skip the import smoke test
#     --dry-run       print every command, change nothing
#
# Afterwards: . WORK/env-PROFILE.sh  (or run_cards.sh --env WORK/env-PROFILE.sh)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/lib.sh
. "$HERE/lib.sh"

TARGET="" PROFILE=auto WORK="" SRC="" ARCH="" ENV_PREFIX="" EDITABLE=0 SMOKE=1
JOBS=$(nproc 2>/dev/null || echo 4)
declare -A REF=([aether]=${AETHER_REF:-v0.2.0} [raptor]=${RAPTOR_REF:-v0.2.0}
                [eagle]=${EAGLE_REF:-v0.4.0} [hawk]=${HAWK_REF:-v0.3.0})
GIT_BASE=${REPRO_GIT_BASE:-https://github.com/amasat01}
REPOS=(aether raptor eagle hawk)

while [[ $# -gt 0 ]]; do
  case $1 in
    --target) TARGET=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --src) SRC=$2; shift 2 ;;
    --ref) [[ $2 == *=* && -n ${REF[${2%%=*}]+x} ]] || die "--ref wants REPO=REF with REPO in ${REPOS[*]}"
           REF[${2%%=*}]=${2#*=}; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --env-prefix) ENV_PREFIX=$2; shift 2 ;;
    --editable) EDITABLE=1; shift ;;
    --jobs) JOBS=$2; shift 2 ;;
    --no-smoke) SMOKE=0; shift ;;
    --dry-run) REPRO_DRY_RUN=1; shift ;;
    -h|--help) sed -n '5,30p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

# --- facts --------------------------------------------------------------------
TARGET=${TARGET:-$(detect_target)}
case $TARGET in kaggle|vastai|devbox|local) ;; *) die "unknown target $TARGET" ;; esac
GPUS=$(detect_gpus)
if [[ $PROFILE == auto ]]; then PROFILE=$([[ -n $GPUS ]] && echo gpu || echo cpu); fi
[[ $PROFILE == gpu || $PROFILE == cpu ]] || die "unknown profile $PROFILE"
if [[ $PROFILE == gpu ]]; then
  ARCH=${ARCH:-$(gpu_archs)}
  [[ -n $ARCH ]] || die "profile gpu but no GPU visible; pass --arch (e.g. --arch 75) to build anyway"
fi
WORK=${WORK:-$(default_work "$TARGET")}
if [[ -z $SRC ]]; then
  if [[ $TARGET == local ]]; then SRC=$(checkout_root); fi
  SRC=${SRC:-$WORK/src}
fi
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/cards-$PROFILE}
LOCKS=$HERE/locks
ENV_FILE=$WORK/env-$PROFILE.sh
log "target=$TARGET profile=$PROFILE arch=${ARCH:-none} work=$WORK src=$SRC env=$ENV_PREFIX"
[[ $REPRO_DRY_RUN == 1 ]] && log "dry run: commands are printed, nothing is changed"

stamp_ok() { [[ $REPRO_DRY_RUN == 0 && -f $1 && $(cat "$1") == "$2" ]]; }
write_stamp() { [[ $REPRO_DRY_RUN == 1 ]] || echo "$2" >"$1"; }
md5_of() { md5sum "$@" | md5sum | cut -c1-32; }

run mkdir -p "$WORK/bin" "$WORK/build" "$SRC"

# --- 1. conda frontend (an existing micromamba/mamba/conda, else a pinned download)
# REPRO_MAMBA=/path/to/frontend picks one; REPRO_MAMBA=download forces the download.
MAMBA=${REPRO_MAMBA:-}
[[ $MAMBA == download ]] && MAMBA=$WORK/bin/micromamba
if [[ -z $MAMBA ]]; then
  for c in micromamba mamba conda; do command -v $c >/dev/null 2>&1 && { MAMBA=$(command -v $c); break; }; done
fi
if [[ -z $MAMBA || $MAMBA == "$WORK/bin/micromamba" ]]; then
  MAMBA=$WORK/bin/micromamba
  if [[ ! -x $MAMBA ]]; then
    MM_VER=${MICROMAMBA_VERSION:-2.9.0-0}
    run curl -fsSL -o "$MAMBA" \
      "https://github.com/mamba-org/micromamba-releases/releases/download/$MM_VER/micromamba-linux-64"
    run chmod +x "$MAMBA"
  fi
fi
log "conda frontend: $MAMBA"
case $(basename "$MAMBA") in
  micromamba) export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$WORK/mamba-root}
              MKENV=("$MAMBA" create -y -q -p "$ENV_PREFIX" -f "$LOCKS/toolchain-linux-64.lock") ;;
  *)          MKENV=("$MAMBA" create -y -q -p "$ENV_PREFIX" --file "$LOCKS/toolchain-linux-64.lock") ;;
esac

# --- 2. toolchain environment (explicit lock: exact URLs + md5) -----------------
TC_STAMP=$ENV_PREFIX/.repro-toolchain
TC_SUM=$(md5_of "$LOCKS/toolchain-linux-64.lock")
if stamp_ok "$TC_STAMP" "$TC_SUM"; then
  log "toolchain env up to date"
else
  [[ -d $ENV_PREFIX && $REPRO_DRY_RUN == 0 ]] && run rm -rf "$ENV_PREFIX"
  run "${MKENV[@]}"
  write_stamp "$TC_STAMP" "$TC_SUM"
fi
PY=$ENV_PREFIX/bin/python

# Build environment: the env's own GCC 14 and nvcc, nothing from the host.
unset CPATH CUDA_PATH CUDA_HOME NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS PYTHONPATH LD_PRELOAD
export PATH=$ENV_PREFIX/bin:$PATH CONDA_PREFIX=$ENV_PREFIX
export CC=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX CUDACXX=$ENV_PREFIX/bin/nvcc
export CMAKE_PREFIX_PATH=$ENV_PREFIX CMAKE_BUILD_PARALLEL_LEVEL=$JOBS CMAKE_GENERATOR=Ninja
export HAWK_AETHER_INCLUDE=$SRC/aether HAWK_EAGLE_INCLUDE=$SRC/eagle
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export TMPDIR=${TMPDIR:-$WORK/tmp}; run mkdir -p "$TMPDIR"

# --- 3. Python packages (hashed lock, no resolver) ------------------------------
PIP_STAMP=$ENV_PREFIX/.repro-pip-$PROFILE
PIP_SUM=$(md5_of "$LOCKS/cards-$PROFILE.txt")
if stamp_ok "$PIP_STAMP" "$PIP_SUM"; then
  log "python packages up to date"
else
  run "$PY" -m pip install -q --no-deps --require-hashes -r "$LOCKS/cards-$PROFILE.txt"
  write_stamp "$PIP_STAMP" "$PIP_SUM"
fi

# --- 4. sources -----------------------------------------------------------------
for r in "${REPOS[@]}"; do
  if [[ -d $SRC/$r ]]; then
    log "$r: using existing checkout $SRC/$r (left as it is)"
  else
    run git clone -q --depth 1 --branch "${REF[$r]}" "$GIT_BASE/$r.git" "$SRC/$r" ||
      die "clone of $r at ${REF[$r]} failed: pass --ref $r=<tag|branch> (or ${r^^}_REF)"
  fi
done

# --- 5. headers: aether + eagle (CUDA-mode packages; nvcc is in the env) --------
HDR_STAMP=$ENV_PREFIX/.repro-headers
HDR_SUM=""
if [[ $REPRO_DRY_RUN == 0 ]]; then
  HDR_SUM=$(for r in aether eagle; do git -C "$SRC/$r" rev-parse HEAD 2>/dev/null || echo "$r-notgit"
             git -C "$SRC/$r" status --porcelain -- . 2>/dev/null | md5sum; done | md5sum | cut -c1-32)
fi
if stamp_ok "$HDR_STAMP" "$HDR_SUM"; then
  log "headers up to date"
else
  for r in aether eagle; do
    run rm -rf "$WORK/build/hdr-$r"
    run cmake -S "$SRC/$r" -B "$WORK/build/hdr-$r" -DCMAKE_PREFIX_PATH="$ENV_PREFIX" \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-61}" -DCMAKE_CUDA_HOST_COMPILER="$CXX"
    run cmake --install "$WORK/build/hdr-$r" --prefix "$ENV_PREFIX"
  done
  write_stamp "$HDR_STAMP" "$HDR_SUM"
fi

# --- 6. raptor, aether-dsc, eagle (_core + plugin for ARCH), hawk ----------------
PIPI=("$PY" -m pip install -q --no-deps --no-build-isolation)
ED=(); [[ $EDITABLE == 1 ]] && ED=(-e)
if [[ $PROFILE == gpu ]]; then
  EAGLE_CFG=(-C "cmake.define.CMAKE_CUDA_ARCHITECTURES=$ARCH" -C "cmake.define.CMAKE_CUDA_HOST_COMPILER=$CXX")
else
  EAGLE_CFG=(-C "cmake.define.EAGLE_PYTHON_CUDA_PLUGIN=OFF")
fi
run "${PIPI[@]}" "${ED[@]}" "$SRC/raptor"
run "${PIPI[@]}" "${ED[@]}" "$SRC/aether/dsc"
run "${PIPI[@]}" -C "build-dir=$WORK/build/eagle-$PROFILE" "${EAGLE_CFG[@]}" "${ED[@]}" "$SRC/eagle/python"
run "${PIPI[@]}" -C "build-dir=$WORK/build/hawk-$PROFILE" "${ED[@]}" "$SRC/hawk"

# --- 7. env script --------------------------------------------------------------
if [[ $REPRO_DRY_RUN == 1 ]]; then
  log "would write $ENV_FILE"
else
  cat >"$ENV_FILE" <<EOF
# Generated by eagle/benchmarks/reproduce/bootstrap.sh ($PROFILE profile). Source it.
unset CPATH CUDA_PATH CUDA_HOME NVCC_PREPEND_FLAGS LD_PRELOAD PYTHONPATH
export REPRO_PROFILE=$PROFILE REPRO_TARGET=$TARGET REPRO_SRC=$SRC REPRO_WORK=$WORK
export CONDA_PREFIX=$ENV_PREFIX P=$ENV_PREFIX/bin/python
export PATH=$ENV_PREFIX/bin:\$PATH
export CC=$CC CXX=$CXX CUDAHOSTCXX=$CXX
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export HAWK_AETHER_INCLUDE=$SRC/aether HAWK_EAGLE_INCLUDE=$SRC/eagle HAWK_AETHER_SOURCE=$SRC/aether
export EAGLE_AETHER_INCLUDE=$SRC/aether RAPTOR_HAWK_ROOT=$SRC/hawk RAPTOR_EAGLE_ROOT=$SRC/eagle
export CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false
EOF
  [[ $PROFILE == cpu ]] && echo 'export CUDA_VISIBLE_DEVICES=""' >>"$ENV_FILE"
  log "wrote $ENV_FILE"
fi

# --- 8. smoke -------------------------------------------------------------------
if [[ $SMOKE == 1 ]]; then
  SMOKE_PY='import eagle, eagle._core, hawk, raptor, aether_dsc, numpy, torch, jax
print("import ok: eagle", eagle.__file__)
print("torch", torch.__version__, "jax", jax.__version__, jax.devices())'
  if [[ $PROFILE == gpu ]]; then
    SMOKE_PY+='
import cupy, warp, eagle.cuda as ec
ec.Stream()
print("eagle cuda backend", eagle._core.cuda_backend())
print("cupy devices", cupy.cuda.runtime.getDeviceCount(), "torch cuda", torch.cuda.is_available())'
  else
    SMOKE_PY+='
import numba
print("numba", numba.__version__)'
  fi
  run "$PY" -W ignore -c "$SMOKE_PY"
fi

# --- 9. machine fact sheet ------------------------------------------------------
facts() {
  echo "target:       $TARGET"
  echo "profile:      $PROFILE"
  echo "machine slug: $(machine_slug)"
  echo "host:         $(uname -srm)"
  echo "cpu:          $(cpu_model) ($(nproc) logical)"
  echo "memory:       $(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  echo "disk free:    $(df -h "$WORK" 2>/dev/null | awk 'NR==2 {print $4 " on " $6}')"
  if [[ -n $GPUS ]]; then
    while IFS= read -r g; do echo "gpu:          $g  (name, compute capability, driver)"; done <<<"$GPUS"
  else
    echo "gpu:          none visible"
  fi
  echo "eagle archs:  ${ARCH:-none (CPU-only build)}"
  echo "nvcc:         $("$ENV_PREFIX/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "g++:          $("$CXX" -dumpfullversion 2>/dev/null)"
  echo "nsys:         $(command -v nsys || echo 'not found (kernel-only times will be null)')"
  for r in "${REPOS[@]}"; do
    echo "$r:$(printf '%*s' $((12 - ${#r})) '')$(git -C "$SRC/$r" describe --tags --always --dirty 2>/dev/null || echo '?') ($SRC/$r)"
  done
  "$PY" - <<'EOF'
import importlib.metadata as md
for d in ("numpy", "torch", "jax", "jaxlib", "cupy-cuda12x", "warp-lang", "numba",
          "raptor-eagle", "raptor-hawk", "raptor-core", "aether-dsc"):
    try:
        print(f"{d + ':':13s} {md.version(d)}")
    except md.PackageNotFoundError:
        pass
EOF
}
if [[ $REPRO_DRY_RUN == 1 ]]; then
  log "would write $WORK/machine_facts.txt"
else
  facts | tee "$WORK/machine_facts.txt"
fi
log "done. Next: $HERE/run_cards.sh --env $ENV_FILE --preset smoke"
