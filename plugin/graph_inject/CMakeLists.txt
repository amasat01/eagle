# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Mid-pipeline JIT injection demo.
#
# Unlike host/ (a pure C++ Driver-API loader that needs no nvcc), this target has
# its own compiled-in CUDA kernels, so it IS an nvcc build. It builds a
# Runtime-API captured graph and injects a Driver-API-
# loaded PTX plugin into the opening between two compiled-in kernels. Links both
# the Runtime (cudart, auto) and the Driver (libcuda stub) libraries.
cmake_minimum_required(VERSION 3.18)
project(eagle_graph_inject CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)   # local Quadro P2000; override as needed
endif()

if(DEFINED ENV{CUDA_PATH})
  set(_CUDA_ROOT $ENV{CUDA_PATH})
else()
  set(_CUDA_ROOT /usr/local/cuda-12.6)
endif()
find_library(CUDA_DRIVER_LIB cuda
  HINTS ${_CUDA_ROOT}/targets/x86_64-linux/lib/stubs
        ${_CUDA_ROOT}/lib64/stubs
        ${_CUDA_ROOT}/lib/stubs)
if(NOT CUDA_DRIVER_LIB)
  message(FATAL_ERROR "Could not locate the libcuda stub under ${_CUDA_ROOT}")
endif()

add_executable(inject_demo inject_demo.cu)
# eagle/exec/*.h include each other repo-root-relative ("eagle/exec/Partition.h").
target_include_directories(inject_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(inject_demo PRIVATE ${CUDA_DRIVER_LIB})
