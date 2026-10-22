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

# Two arms of the same demo: the CUDA arm (embedding_demo, nvcc) and the CPU arm
# (embedding_demo_cpu, no CUDA). -DEMBED_CPU_ONLY=ON builds only the CPU arm and
# needs no CUDA toolchain.
option(EMBED_CPU_ONLY "Build only the CPU arm (no CUDA toolchain needed)" OFF)
if(EMBED_CPU_ONLY)
  project(eagle_cpp_embed_plugin CXX)
else()
  project(eagle_cpp_embed_plugin CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(_ART_DIR ${CMAKE_CURRENT_BINARY_DIR}/art)
file(MAKE_DIRECTORY ${_ART_DIR})
set(_EAGLE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# --- CPU arm: the same three plugins as host code (fixtures/host_plugins.cpp, one
# shared object) loaded through eagle::cpu::PluginRegistry, plus the shared sidecars.
find_package(OpenMP)
add_library(host_plugins MODULE fixtures/host_plugins.cpp)
set_target_properties(host_plugins PROPERTIES POSITION_INDEPENDENT_CODE ON PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${_ART_DIR})
target_include_directories(host_plugins PRIVATE ${_EAGLE_ROOT})
if(OpenMP_CXX_FOUND)
  target_link_libraries(host_plugins PRIVATE OpenMP::OpenMP_CXX)
endif()
foreach(_fx mattrace toy_energy toy_energy_vjp)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_fx}.json
                 ${_ART_DIR}/${_fx}.json COPYONLY)
endforeach()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixtures/host_manifest.json
               ${_ART_DIR}/host_manifest.json COPYONLY)

# The float32 arm: one float32 plugin (loaded through eagle::cpu::PluginRegistry32).
add_library(host_plugin_f32 MODULE fixtures/host_plugin_f32.cpp)
set_target_properties(host_plugin_f32 PROPERTIES POSITION_INDEPENDENT_CODE ON PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${_ART_DIR})
target_include_directories(host_plugin_f32 PRIVATE ${_EAGLE_ROOT})
if(OpenMP_CXX_FOUND)
  target_link_libraries(host_plugin_f32 PRIVATE OpenMP::OpenMP_CXX)
endif()
foreach(_fx toy_energy32.json host_manifest_f32.json)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_fx} ${_ART_DIR}/${_fx} COPYONLY)
endforeach()

add_executable(embedding_demo_cpu embedding_demo_cpu.cpp)
add_dependencies(embedding_demo_cpu host_plugins host_plugin_f32)
target_include_directories(embedding_demo_cpu PRIVATE ${_EAGLE_ROOT})
target_link_libraries(embedding_demo_cpu PRIVATE ${CMAKE_DL_LIBS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(embedding_demo_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(embedding_demo_cpu PRIVATE EMBED_DEMO_ART_DIR="${_ART_DIR}")

if(EMBED_CPU_ONLY)
  return()
endif()

# --- CUDA arm ---------------------------------------------------------------
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
