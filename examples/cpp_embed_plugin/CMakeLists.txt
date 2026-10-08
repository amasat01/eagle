# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# C++ embedding reference example.
#
# A packaged, buildable demonstration of embedding a plugin deployment purely
# through eagle's C++ ABI surface (plugin/*.h) -- no Python, no code generator
# at build or run time. It exercises the two capabilities this example was
# written to close (see README.rst):
#   * a CUDA `mat_in` matrix input, bound through the device
#     `eagle::cuda::PluginRegistry::bind_matrix` (plugin/plugin_registry/registry.h);
#   * a derivative (VJP) artifact carrying the optional `derivative` sidecar block
#     (plugin/sidecar.h), launched and checked against a host reference.
#
# This does NOT replace or duplicate the three existing plugin/ demos
# (plugin_host.cpp, plugin/graph_inject/inject_demo.cu,
# plugin/pure_inject/pure_inject_demo.cu, all driven by
# eagle/python/tests/cpp_demo.py) -- see README.rst for how the four relate. Like
# pure_inject/graph_inject, this needs both the CUDA Driver API (cuLaunchKernel,
# via the independent PluginRegistry) and the Runtime API (stream/graph capture),
# so it is an nvcc build.
cmake_minimum_required(VERSION 3.18)
project(eagle_cpp_embed_plugin CXX CUDA)

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

# --- fixture PTX: raw PTX TEXT, loaded at run time via cuModuleLoadData (a
# driver-side JIT capped at the installed driver's max supported PTX ISA) -- so,
# like eagle/tests/CMakeLists.txt's device_plugin_softdouble fixture, this
# resolves nvcc PATH-first (matching the plugin toolchain's own contract) rather than
# trusting whichever nvcc CMake cached as CMAKE_CUDA_COMPILER for the
# embedded-SASS demo executable below -- a newer nvcc than the installed driver
# emits a newer PTX ISA that cuModuleLoadData then rejects
# (cudaErrorUnsupportedPtxVersion).
unset(_EMBED_FIXTURE_NVCC CACHE)
set(_EMBED_FIXTURE_NVCC "")
if(DEFINED ENV{PATH})
  string(REPLACE ":" ";" _embed_path_list "$ENV{PATH}")
  foreach(_embed_dir ${_embed_path_list})
    if(NOT _EMBED_FIXTURE_NVCC AND EXISTS "${_embed_dir}/nvcc")
      set(_EMBED_FIXTURE_NVCC "${_embed_dir}/nvcc")
    endif()
  endforeach()
endif()
if(NOT _EMBED_FIXTURE_NVCC AND EXISTS "${_CUDA_ROOT}/bin/nvcc")
  set(_EMBED_FIXTURE_NVCC "${_CUDA_ROOT}/bin/nvcc")
endif()
if(NOT _EMBED_FIXTURE_NVCC)
  message(FATAL_ERROR "Could not locate nvcc (checked $PATH and ${_CUDA_ROOT}/bin) "
    "to compile the example's fixture PTX")
endif()

set(_ART_DIR ${CMAKE_CURRENT_BINARY_DIR}/art)
file(MAKE_DIRECTORY ${_ART_DIR})

set(_EAGLE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(_FIXTURES mattrace toy_energy toy_energy_vjp)
set(_FIXTURE_PTX)
foreach(_fx ${_FIXTURES})
  set(_out ${_ART_DIR}/${_fx}.ptx)
  add_custom_command(
    OUTPUT ${_out}
    COMMAND ${_EMBED_FIXTURE_NVCC} --ptx
            -I${_EAGLE_ROOT}
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_fx}.cu
            -o ${_out}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_fx}.cu
    COMMENT "Compiling fixtures/${_fx}.cu -> PTX (${_EMBED_FIXTURE_NVCC})")
  list(APPEND _FIXTURE_PTX ${_out})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_fx}.json
                 ${_ART_DIR}/${_fx}.json COPYONLY)
endforeach()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixtures/manifest.json
               ${_ART_DIR}/manifest.json COPYONLY)
add_custom_target(embed_demo_fixtures DEPENDS ${_FIXTURE_PTX})

add_executable(embedding_demo embedding_demo.cu)
add_dependencies(embedding_demo embed_demo_fixtures)
target_include_directories(embedding_demo PRIVATE ${_EAGLE_ROOT})
target_link_libraries(embedding_demo PRIVATE ${CUDA_DRIVER_LIB})
target_compile_definitions(embedding_demo PRIVATE
  EMBED_DEMO_ART_DIR="${_ART_DIR}")
