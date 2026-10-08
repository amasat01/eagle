# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Precompiled plugin host.
#
# Builds a standalone C++ executable that loads and runs a PTX plugin via
# the CUDA Driver API. It needs ONLY a C++17 compiler, cuda.h, and the CUDA
# driver library — NO nvcc, NO CuPy. (We link the driver *stub*
# libcuda.so for the build; its SONAME is libcuda.so.1, so at run time the real
# system driver is loaded.)
cmake_minimum_required(VERSION 3.18)
project(eagle_plugin_host CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA toolkit location — prefer CUDA_PATH env override; otherwise derive
# from nvcc on PATH (robust in any CUDA container); last resort /usr/local/cuda.
if(DEFINED ENV{CUDA_PATH})
  set(_CUDA_ROOT $ENV{CUDA_PATH})
else()
  find_program(_NVCC_EXE nvcc)
  if(_NVCC_EXE)
    get_filename_component(_NVCC_BIN "${_NVCC_EXE}" DIRECTORY)
    get_filename_component(_CUDA_ROOT "${_NVCC_BIN}" DIRECTORY)
  else()
    set(_CUDA_ROOT /usr/local/cuda)
  endif()
endif()

find_path(CUDA_DRIVER_INCLUDE cuda.h
  HINTS ${_CUDA_ROOT}/include)
find_library(CUDA_DRIVER_LIB cuda
  HINTS ${_CUDA_ROOT}/targets/x86_64-linux/lib/stubs
        ${_CUDA_ROOT}/lib64/stubs
        ${_CUDA_ROOT}/lib/stubs)

if(NOT CUDA_DRIVER_INCLUDE OR NOT CUDA_DRIVER_LIB)
  message(FATAL_ERROR "Could not locate cuda.h / libcuda stub under ${_CUDA_ROOT}")
endif()

add_executable(plugin_host plugin_host.cpp)
target_include_directories(plugin_host PRIVATE ${CUDA_DRIVER_INCLUDE})
target_link_libraries(plugin_host PRIVATE ${CUDA_DRIVER_LIB})
