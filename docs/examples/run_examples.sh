#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# Compile and run every EAGLE documentation example in its supported mode(s) and
# check the printed result against the committed expected output.
#
# This is the "tested cells" gate: the C++ shown in the tutorials/examples pages
# is literalinclude'd from these exact files, and this script proves they still
# compile and produce the documented output. CUDA-only examples (graph capture)
# are built with nvcc; dual-mode examples are built with BOTH nvcc (CUDA mode)
# and g++ (pure-C++/OpenMP mode).
#
#   CONDA_PREFIX=<env-with-eagle+aether>  ./run_examples.sh
#
# Optional: EAGLE_SM_ARCH (default sm_61) to target another GPU.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${CONDA_PREFIX:?set CONDA_PREFIX to the env with eagle + aether installed}"
INC="-I${PREFIX}/include"
# AETHER_HAS_CUDA is normally propagated transitively through the aether::aether
# CMake target (an INTERFACE compile definition the installed CMake config carries);
# this script bypasses CMake entirely (raw nvcc on the headers), so it must supply
# it itself or every aether CUDA-only symbol (Stream, copyAsync, Array::upload/
# download(stream)) is invisible to the TU.
CUDA_DEF="-DAETHER_HAS_CUDA=1 -DEAGLE_BLOCKSIZE=256 -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 -DNDEBUG"
CPP_DEF="-DEAGLE_CPU_ONLY=1 -DEAGLE_BLOCKSIZE=256 -DNDEBUG"
ARCH="${EAGLE_SM_ARCH:-sm_61}"
NVCC=(nvcc -std=c++20 -arch="${ARCH}" -Wno-deprecated-gpu-targets ${INC} ${CUDA_DEF} -Xcompiler -fPIE -Xcompiler -fopenmp)
GXX=(g++ -std=c++23 -fopenmp -O2 ${INC} ${CPP_DEF})

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
fail=0

run_and_check() { # <label> <exe> <expected-file>
    local label="$1" exe="$2" expf="$3" out rc
    out="$("$exe")"; rc=$?
    if [[ $rc -ne 0 ]]; then echo "FAIL  ${label} (exit ${rc})"; fail=1; return; fi
    if [[ -f "$expf" ]] && ! diff <(printf '%s\n' "$out") "$expf" >/dev/null; then
        echo "FAIL  ${label} (output mismatch)"; echo "    got: ${out}"; fail=1; return
    fi
    echo "ok    ${label} : ${out}"
}

if command -v nvcc >/dev/null 2>&1; then
    for src in "$DIR"/[0-9]*.cu; do
        name="$(basename "${src%.cu}")"
        if "${NVCC[@]}" "$src" -o "$WORK/$name" 2> "$WORK/$name.log"; then
            run_and_check "CUDA ${name}" "$WORK/$name" "$DIR/${name}.cuda.expected.txt"
        else echo "FAIL  CUDA ${name} (compile)"; sed 's/^/    /' "$WORK/$name.log"; fail=1; fi
    done
else
    echo "SKIP  CUDA mode (no nvcc on PATH)"
fi

for src in "$DIR"/[0-9]*.cpp; do
    name="$(basename "${src%.cpp}")"
    if "${GXX[@]}" "$src" -o "$WORK/${name}_cpp" 2> "$WORK/${name}_cpp.log"; then
        run_and_check "C++  ${name}" "$WORK/${name}_cpp" "$DIR/${name}.cpp.expected.txt"
    else echo "FAIL  C++  ${name} (compile)"; sed 's/^/    /' "$WORK/${name}_cpp.log"; fail=1; fi
done

if [[ $fail -eq 0 ]]; then echo "ALL EXAMPLES PASSED"; else echo "SOME EXAMPLES FAILED"; exit 1; fi
