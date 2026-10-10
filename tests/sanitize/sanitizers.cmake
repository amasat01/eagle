# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# EAGLE_SANITIZERS (cache string, default empty = no effect): a -fsanitize= list, e.g.
# `address,undefined` for the CPP_MODE sanitize job. Appends
# `-fsanitize=<value> -fno-omit-frame-pointer -g` to compile + link of the test
# targets, and builds the argv-selected canaries in the SAME configure:
#   eagle_canary     (host,  when EAGLE_SANITIZERS is set)    tests/sanitize/canary.cpp
#   eagle_canary_cu  (CUDA mode, with tests)                   tests/sanitize/canary.cu
# See tests/sanitize/README.md.

function(eagle_apply_sanitizers target)
  if(NOT EAGLE_SANITIZERS)
    return()
  endif()
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${EAGLE_SANITIZERS} -fno-omit-frame-pointer -g>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=${EAGLE_SANITIZERS},-fno-omit-frame-pointer>)
  target_link_options(${target} PRIVATE -fsanitize=${EAGLE_SANITIZERS})
endfunction()

if(EAGLE_SANITIZERS)
  add_executable(eagle_canary ${CMAKE_CURRENT_LIST_DIR}/canary.cpp)
  eagle_apply_sanitizers(eagle_canary)
endif()

if(NOT EAGLE_CPP_MODE)
  add_executable(eagle_canary_cu ${CMAKE_CURRENT_LIST_DIR}/canary.cu)
endif()
