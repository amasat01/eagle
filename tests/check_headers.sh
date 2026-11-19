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

# Headers that need a toolkit this check does not install: CUDA (cuda.h, nvcc-only) or MPI.
skip='^eagle/cuda/|^eagle/cuda\.h$|^eagle/array/launch\.h$|^eagle/exec/DeviceKernel\.h$|^eagle/exec/RankPartition\.h$'

probe_dir=$(mktemp -d)   # a folder: BSD mktemp takes no suffix, and the compiler needs the .cpp
probe="$probe_dir/probe.cpp"
trap 'rm -rf "$probe_dir"' EXIT
failed=0 checked=0
for h in $(find eagle -name '*.h' | LC_ALL=C sort | grep -Ev "$skip"); do
  printf '#include "%s"\n' "$h" > "$probe"
  checked=$((checked + 1))
  # -include raptor_facts.h: the installed aether target forces it on every consumer, so a header is checked as a user's build sees it.
  if ! out=$("$cxx" -std=c++23 -fsyntax-only -fopenmp -DEAGLE_CPU_ONLY=1 -DAETHER_CPP_MODE=1 \
               -I. -I"$aether" -include raptor_facts.h "$probe" 2>&1); then
    failed=$((failed + 1))
    echo "FAIL $h"
    echo "$out" | grep -E 'error' | head -5
  fi
done
echo "$cxx: $checked headers checked, $failed failed"
[ "$checked" -gt 0 ] && [ "$failed" -eq 0 ]
