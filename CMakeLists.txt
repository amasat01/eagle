# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

option(EAGLE_BUILD_TESTS "Whether to build tests" OFF)
option(EAGLE_DEBUG_MODE "Whether to compile with debug features" OFF)
option(EAGLE_WARN_AS_ERR "Whether to promote compilation warnings to errors" OFF)
option(EAGLE_CPP_MODE "Whether to install eagle as a pure C++ Package. Default is CUDA/C++" OFF)
option(EAGLE_BUILD_SANITIZE "Register the `sanitize` custom target (valgrind/compute-sanitizer wrapper)" OFF)
# The `eagle::exec::RankPartition`
# MPI bed. OFF by default and deliberately INERT when ON for every target but the
# bed itself -- `EAGLE_MPI` is defined PRIVATELY on `tests/mpi/eagle_mpi_bed`, and
# `find_package(MPI)` happens in tests/mpi/CMakeLists.txt, so flipping this option
# in an existing build directory recompiles the bed and NOTHING else (the abi build
# dirs stay incremental). eagle itself never links MPI: RankPartition.h is a
# header-only structure compiled only where the bed's private define reaches it.
option(EAGLE_MPI "Build the 2-rank MPI bed for eagle::exec::RankPartition (tests/mpi)" OFF)
set(EAGLE_SANITIZE_TOOL "both" CACHE STRING "Which sanitizer(s) the sanitize target runs: valgrind|computesan|both")
option(EAGLE_SANITIZE_MINIMAL "Run the sanitize target with EAGLE_TEST_MINIMAL=1" OFF)
option(EAGLE_SANITIZE_FAIL_ON_REACHABLE "Escalate still-reachable leaks to failures" OFF)

# CUDA architectures, when CMAKE_CUDA_ARCHITECTURES is not given: EAGLE_CUDA_ARCHS
# "native" (the default) builds for the GPUs in this machine only, the fastest
# build; "all" builds for every one of 61 70 75 80 86 89 90 100 120 that the nvcc
# in use still compiles (CUDA 13 drops the pre-Turing ones and adds Blackwell);
# one architecture or a list ("86", "75;86"; sm_86 is accepted) builds exactly those. With no GPU to query, "native"
# builds PTX for the oldest architecture nvcc compiles, which the driver
# compiles for any newer GPU at load. Runs BEFORE project() enables CUDA, so the
# compiler is located here (-DCMAKE_CUDA_COMPILER, then $CUDACXX, then PATH).
set(EAGLE_CUDA_ARCHS "native" CACHE STRING
    "CUDA architectures when CMAKE_CUDA_ARCHITECTURES is unset: native, all, one (86) or a list (75;86)")
if(NOT EAGLE_CPP_MODE AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(_eagle_nvcc "")
  if(CMAKE_CUDA_COMPILER)
    set(_eagle_nvcc "${CMAKE_CUDA_COMPILER}")
  elseif(NOT "$ENV{CUDACXX}" STREQUAL "")
    set(_eagle_nvcc "$ENV{CUDACXX}")
  else()
    find_program(_eagle_nvcc_found nvcc)
    set(_eagle_nvcc "${_eagle_nvcc_found}")
  endif()
  set(_eagle_supported "")
  if(_eagle_nvcc)
    execute_process(COMMAND "${_eagle_nvcc}" --list-gpu-arch
                    OUTPUT_VARIABLE _eagle_list RESULT_VARIABLE _eagle_rc ERROR_QUIET)
    if(_eagle_rc EQUAL 0)
      string(REGEX MATCHALL "compute_[0-9]+" _eagle_found "${_eagle_list}")
      foreach(_c IN LISTS _eagle_found)
        string(REPLACE "compute_" "" _c "${_c}")
        list(APPEND _eagle_supported ${_c})
      endforeach()
      list(SORT _eagle_supported COMPARE NATURAL)
    endif()
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "")
  if(EAGLE_CUDA_ARCHS STREQUAL "all")
    foreach(_a 61 70 75 80 86 89 90 100 120)
      if(NOT _eagle_supported OR _a IN_LIST _eagle_supported)
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${_a})
      endif()
    endforeach()
  elseif(EAGLE_CUDA_ARCHS STREQUAL "native")
    set(_eagle_caps "")
    find_program(_eagle_smi nvidia-smi)
    if(_eagle_smi)
      execute_process(COMMAND "${_eagle_smi}" --query-gpu=compute_cap --format=csv,noheader
                      OUTPUT_VARIABLE _eagle_smi_out RESULT_VARIABLE _eagle_smi_rc ERROR_QUIET)
      if(_eagle_smi_rc EQUAL 0)
        string(REGEX MATCHALL "[0-9]+\\.[0-9]+" _eagle_caps "${_eagle_smi_out}")
        list(TRANSFORM _eagle_caps REPLACE "\\." "")
        list(REMOVE_DUPLICATES _eagle_caps)
      endif()
    endif()
    if(_eagle_supported)
      list(GET _eagle_supported 0 _eagle_min)
      list(GET _eagle_supported -1 _eagle_max)
    endif()
    foreach(_c IN LISTS _eagle_caps)
      if(NOT _eagle_supported OR _c IN_LIST _eagle_supported)
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${_c})
      elseif(_c GREATER _eagle_max)
        # a GPU newer than this nvcc: PTX for its newest architecture
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${_eagle_max}-virtual)
      elseif(_c LESS _eagle_min)
        message(FATAL_ERROR "This machine's GPU (sm_${_c}) is older than the oldest architecture "
                            "${_eagle_nvcc} compiles (sm_${_eagle_min}): use an older CUDA toolkit, or "
                            "pass EAGLE_CUDA_ARCHS / CMAKE_CUDA_ARCHITECTURES for another target.")
      endif()
    endforeach()
    if(NOT CMAKE_CUDA_ARCHITECTURES AND _eagle_supported)
      # no GPU here: portable PTX for the oldest architecture nvcc compiles, and
      # never below sm_60 (the double atomicAdd the headers use needs it)
      foreach(_a IN LISTS _eagle_supported)
        if(NOT _a LESS 60)
          set(CMAKE_CUDA_ARCHITECTURES ${_a}-virtual)
          break()
        endif()
      endforeach()
    endif()
  else()
    # one architecture or several; "sm_86" and "compute_86" spell "86"
    string(REGEX REPLACE "(sm|compute)_" "" CMAKE_CUDA_ARCHITECTURES "${EAGLE_CUDA_ARCHS}")
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES)  # no nvcc to ask yet: the long-standing set
    set(CMAKE_CUDA_ARCHITECTURES 61 70 80 90)
  endif()
  message(STATUS "> Compiling for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if (EAGLE_CPP_MODE)
  message(STATUS "> Configuring eagle in C++ only mode")
  project(eagle VERSION 0.4.0 LANGUAGES CXX)
else()
  message(STATUS "> Configuring eagle in CUDA/C++ mode")
  project(eagle VERSION 0.4.0 LANGUAGES CUDA CXX)
endif()

###########################

# Supress a warning in newer CMake versions when using FetchContent
if (POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

###########################

# include OpenMP
find_package(OpenMP REQUIRED)
message(STATUS "> Found OpenMP")

# Dependency versions (also referenced by CI pipeline)
set(EAGLE_AETHER_VERSION "0.2")

# The VERSION argument is fixed
# on the MAJOR-matching CMake ConfigVersion rule: aether is 0.x, so an
# inherited ``1.14.2`` value would fail configure before any gate ran.
#
# CONVENTION: installed aether packages are MODE-SPECIFIC and a
# consumer selects the mode BY PREFIX. The CUDA-mode package sits at
# ``<prefix>``; its pure-C++ twin (exported defs carry ``AETHER_CPP_MODE=1``,
# never ``AETHER_HAS_CUDA``) sits at ``<prefix>/cpp``, mirroring the CUDA
# package's own layout. The two defines are mutually exclusive inside
# aether's ``macros.h``, so picking the wrong package is a hard compile error,
# not a silent mismatch.
if (EAGLE_CPP_MODE)
  set(EAGLE_AETHER_PREFIXES "")
  foreach(_p IN LISTS CMAKE_PREFIX_PATH)
    list(APPEND EAGLE_AETHER_PREFIXES "${_p}/cpp")
  endforeach()
  list(PREPEND CMAKE_PREFIX_PATH ${EAGLE_AETHER_PREFIXES})
endif()
find_package(aether ${EAGLE_AETHER_VERSION} REQUIRED CONFIG)
message(STATUS "> Found aether version ${aether_VERSION} (${aether_DIR})")

###########################

# This determines the size of thread blocks for GPU kernel calls.
# It should be an integer multiple of 32, and is constrained by
# the architecture of the GPU being used.
if (NOT DEFINED EAGLE_BLOCKSIZE)
  set(EAGLE_BLOCKSIZE 256)
endif()

# Compile flags for CUDA/C++
list(APPEND EAGLE_COMMON_FLAGS -Wall -Wextra -O3 -ftemplate-backtrace-limit=100)
# CUDA only flags
if(NOT EAGLE_CPP_MODE)
  list(APPEND EAGLE_CUDA_FLAGS ${EAGLE_COMMON_FLAGS} ${OpenMP_CXX_FLAGS} -Wno-deprecated-gpu-targets) # We are still using old GPUs
endif()

# C++ >= 23 is required
list(APPEND EAGLE_CXX_FLAGS ${EAGLE_COMMON_FLAGS} ${OpenMP_CXX_FLAGS} -Wpedantic -std=c++23)

# Treat warnings as errors - for CI pipeline
if (EAGLE_WARN_AS_ERR)
  list(APPEND ${EAGLE_CXX_FLAGS} -Werror)
  # CUDA only flag
  if (NOT EAGLE_CPP_MODE)
    list(APPEND ${EAGLE_CUDA_FLAGS} -Werror=all-warnings)
  endif()
endif()

# Preprocessor variable definitions
list(APPEND EAGLE_DEFINES EAGLE_BLOCKSIZE=${EAGLE_BLOCKSIZE} CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
if (EAGLE_DEBUG_MODE)
  list(APPEND EAGLE_DEFINES EAGLE_DEBUG_MODE)
else()
  list(APPEND EAGLE_DEFINES NDEBUG)
endif()

# Mark whether C++ only
if (EAGLE_CPP_MODE)
  list(APPEND EAGLE_DEFINES EAGLE_CPU_ONLY=1)
endif()

# GNUInstallDirs must be included BEFORE the interface target is defined: it is
# what defines CMAKE_INSTALL_INCLUDEDIR, and the $<INSTALL_INTERFACE:> genex
# below is evaluated with the value the variable holds at THIS point. Including
# it later (down in the installation section) silently expanded to
# $<INSTALL_INTERFACE:>, so the exported eagleTargets.cmake carried NO
# INTERFACE_INCLUDE_DIRECTORIES at all. Consumers still compiled only when the
# install prefix happened to be a directory the compiler already searches (a
# conda env), and broke for any other prefix.
include(GNUInstallDirs)

# Header-only libary
set(EAGLE_LIB eagle)
add_library(${EAGLE_LIB} INTERFACE)
target_include_directories(${EAGLE_LIB} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(${EAGLE_LIB} INTERFACE aether::aether ${OpenMP_CXX_LIBRARIES})

# Compile flags and options
# C++ >= 23 is required
target_compile_features(${EAGLE_LIB} INTERFACE cxx_std_23)
target_compile_definitions(${EAGLE_LIB} INTERFACE ${EAGLE_DEFINES})
target_compile_options(${EAGLE_LIB} INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:${EAGLE_CXX_FLAGS}>"
)
target_link_options(${EAGLE_LIB} INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:${EAGLE_CXX_FLAGS}>"
)
# CUDA-only options
if (NOT EAGLE_CPP_MODE)
  target_compile_options(${EAGLE_LIB} INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:${EAGLE_CUDA_FLAGS}>"
  )
  target_link_options(${EAGLE_LIB} INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:${EAGLE_CUDA_FLAGS}>"
  )
endif()

target_link_libraries(${EAGLE_LIB} INTERFACE stdc++fs)

###########################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build demo executables
if (EAGLE_CPP_MODE)
  add_executable(eagle_demo eagle_demo.cpp)
  target_compile_options(
    eagle_demo
    PRIVATE
    -Wall -Wextra -O3 -ftemplate-backtrace-limit=100
  )
else()
  add_executable(eagle_demo eagle_demo.cu)
  target_compile_options(
    eagle_demo
    PRIVATE
    "-Wall -Wextra -O3 -ftemplate-backtrace-limit=100"
  )
endif()

target_link_libraries(eagle_demo ${EAGLE_LIB})

# Position-independent code for the conda toolchain's PIE-default linker
# (CUDA TUs without -fPIC produce R_X86_64_32 relocations that the PIE
# linker rejects).
set_target_properties(eagle_demo PROPERTIES POSITION_INDEPENDENT_CODE ON)

###########################

# Tests
if (EAGLE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Sanitize custom target
if (EAGLE_BUILD_TESTS AND EAGLE_BUILD_SANITIZE)
  add_subdirectory(tests/sanitize)
endif()

###########################

# Installation voodoo
# This allows us to do `make install`, and downstream libraries can find eagle
# by doing `find_package(eagle CONFIG REQUIRED)`. This works as long as CMAKE_PREFIX_PATH
# includes the CMAKE_INSTALL_PREFIX used when configuring eagle.

# (GNUInstallDirs is included near the top, before the interface target — see
# the note there; the CMAKE_INSTALL_* variables it defines are already set.)

# Copy includes to install location
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/eagle" DESTINATION "include")
# Plugin-protocol headers (eagle::plugin PODs + the device/host registries) are
# installed under include/eagle/ so downstream consumers use the namespaced
# prefix: #include "eagle/plugin/sidecar.h". In-tree code keeps the repo-root
# "plugin/..." prefix; intra-tree includes are relative and relocate cleanly.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/plugin" DESTINATION "include/eagle")
# Configure targets to be exported (in this case just our lib)
install(TARGETS ${EAGLE_LIB}
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Create version config file eagleConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)

# Create package config file eagleConfig.cmake
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

# Create target config file eagleTargets.cmake at install location
install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

# Finally, copy cmake config files to install location
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

# make uninstall
add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)
