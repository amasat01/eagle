#!/usr/bin/env bash
# Resolve sibling packages for a job: a version already on PyPI is only pinned, an unreleased one is
# built from its companion checkout into OUT (wheel for pure Python, sdist otherwise).
# usage: install_siblings.sh "pkg=path pkg=path ..." OUT [dry]     (path = package dir inside the checkout)
# Prints `<pkg> <V>: PyPI | built from <path>`; with $GITHUB_OUTPUT set, writes `dir`, `pins` and `built`
# (the packages built from source). `dry` only resolves: nothing is built.
set -euo pipefail
# this eagle checkout: its plugin/gref_layout.h goes into the sealed aether payload, wherever the job checked eagle out
eagle_root=$(cd "$(dirname "$0")/../../.." && pwd)
siblings=$1
out=$(mkdir -p "$2" && cd "$2" && pwd)
dry=${3:-}
pins=""
built=""
for pair in $siblings; do
  pkg=${pair%%=*}
  path=$(cd "${pair#*=}" && pwd)
  root=$path
  until [ -e "$root/.git" ]; do
    [ "$root" != / ] || { echo "no git checkout above $path" >&2; exit 1; }
    root=$(dirname "$root")
  done
  # the version the sibling is (or will be): its exact tag, else what setuptools-scm gives the checkout
  v=$(git -C "$root" describe --tags --match 'v*' --exact-match 2>/dev/null | sed 's/^v//' || true)
  if [ -z "$v" ]; then
    python -m pip install --quiet setuptools-scm
    v=$(python -m setuptools_scm -r "$root")
  fi
  pins="$pins $pkg==$v"
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$pkg/$v/json")" = 200 ]; then
    echo "$pkg $v: PyPI"
    continue
  fi
  echo "$pkg $v: built from $path"
  built="$built $pkg"
  [ -z "$dry" ] || continue
  python -m pip install --quiet --prefer-binary build
  case $pkg in
    aether-dsc)  # sealing runs the NVRTC-clean gate; the runner has no CUDA toolkit, so NVRTC and CCCL come from PyPI
      gate=--no-gate  # CPU-only macOS has no NVRTC: the same headers pass the gate in the Linux seal, and only the host route reads the blob there
      if [ "$(uname -s)" = Linux ]; then
        python -m pip install --quiet --prefer-binary cuda-bindings nvidia-cuda-nvrtc-cu12 nvidia-cuda-cccl-cu12
        gate=
      fi
      # shellcheck disable=SC2086  # $gate is one flag or nothing
      # SIBLING_SDISTS names packages to ship as an sdist instead (the Kaggle leg: built on Kaggle for its own
      # Python, with the CUDA plugin from its toolkit); the sealed payload is inside either way
      form=--wheel; case " ${SIBLING_SDISTS:-} " in *" $pkg "*) form=--sdist ;; esac
      (cd "$path" && python -m aether_dsc.seal $gate --layout-header "$eagle_root/plugin/gref_layout.h" && python -m build $form --outdir "$out") ;;
    raptor-core) (cd "$path" && python -m build --wheel --outdir "$out") ;;
    *) (cd "$path" && python -m build --sdist --outdir "$out") ;;
  esac
done
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "dir=$out" >> "$GITHUB_OUTPUT"
  echo "pins=${pins# }" >> "$GITHUB_OUTPUT"
  echo "built=${built# }" >> "$GITHUB_OUTPUT"
fi
