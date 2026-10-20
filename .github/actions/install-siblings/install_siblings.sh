#!/usr/bin/env bash
# Resolve sibling packages for a job: a version already on PyPI is only pinned, an unreleased one is
# built from its companion checkout into OUT (wheel for pure Python, sdist otherwise).
# usage: install_siblings.sh "pkg=path pkg=path ..." OUT [dry]     (path = package dir inside the checkout)
# Prints `<pkg> <V>: PyPI | built from <path>`; with $GITHUB_OUTPUT set, writes `dir`, `pins` and `built`
# (the packages built from source). `dry` only resolves: nothing is built.
set -euo pipefail
siblings=$1
out=$(mkdir -p "$2" && cd "$2" && pwd)
dry=${3:-}
pins=""
built=""
for pair in $siblings; do
  pkg=${pair%%=*}
  path=$(cd "${pair#*=}" && pwd)
  root=$path
  until [ -f "$root/tools/version.py" ]; do
    [ "$root" != / ] || { echo "no tools/version.py above $path" >&2; exit 1; }
    root=$(dirname "$root")
  done
  v=$(python "$root/tools/version.py" get)
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
      python -m pip install --quiet --prefer-binary cuda-bindings nvidia-cuda-nvrtc-cu12 nvidia-cuda-cccl-cu12
      (cd "$path" && python -m aether_dsc.seal && python -m build --wheel --outdir "$out") ;;
    raptor-core) (cd "$path" && python -m build --wheel --outdir "$out") ;;
    *) (cd "$path" && python -m build --sdist --outdir "$out") ;;
  esac
done
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "dir=$out" >> "$GITHUB_OUTPUT"
  echo "pins=${pins# }" >> "$GITHUB_OUTPUT"
  echo "built=${built# }" >> "$GITHUB_OUTPUT"
fi
