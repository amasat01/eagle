# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Pure-kernel injection demo.
#
# The pure analogue of graph_inject: it loads a runtime-compiled *pure* plugin
# set through the same independent PluginRegistry and injects it into a Runtime-API
# captured graph as a node whose writable `Mutable` output is read-modify-written in
# place. An nvcc build (uses the Runtime API for capture); links the Runtime (cudart,
# auto) and the Driver (libcuda stub) libraries.
cmake_minimum_required(VERSION 3.18)
project(eagle_pure_inject CXX CUDA)

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

add_executable(pure_inject pure_inject_demo.cu)
target_link_libraries(pure_inject PRIVATE ${CUDA_DRIVER_LIB})
