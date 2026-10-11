#!/bin/bash
# Parse every pure-C++ header on its own with one compiler: each header must be
# self-contained, and code only one compiler accepts (a declaration GCC reads as
# an object but clang as a function, say) fails here instead of in a user's build.
#
#   tests/check_headers.sh <c++ compiler> <aether include dir>
set -uo pipefail
cxx=${1:?usage: check_headers.sh <c++ compiler> <aether include dir>}
aether=${2:?usage: check_headers.sh <c++ compiler> <aether include dir>}
cd "$(dirname "$0")/.."

# Headers that need a toolkit this check does not install: CUDA (cuda.h) or MPI.
skip='^eagle/cuda/|^eagle/cuda\.h$|^eagle/exec/DeviceKernel\.h$|^eagle/exec/RankPartition\.h$'

probe=$(mktemp --suffix=.cpp)
trap 'rm -f "$probe"' EXIT
failed=0 checked=0
for h in $(find eagle -name '*.h' | LC_ALL=C sort | grep -Ev "$skip"); do
  printf '#include "%s"\n' "$h" > "$probe"
  checked=$((checked + 1))
  if ! out=$("$cxx" -std=c++23 -fsyntax-only -fopenmp -DEAGLE_CPU_ONLY=1 -DAETHER_CPP_MODE=1 \
               -I. -I"$aether" "$probe" 2>&1); then
    failed=$((failed + 1))
    echo "FAIL $h"
    echo "$out" | grep -E 'error' | head -5
  fi
done
echo "$cxx: $checked headers checked, $failed failed"
[ "$checked" -gt 0 ] && [ "$failed" -eq 0 ]
