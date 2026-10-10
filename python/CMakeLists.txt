# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Builds the eagle Python package's compiled artifacts (scikit-build-core):
#
#   eagle/_core.*.so       the nanobind module, one per Python version
#   eagle/libeagle_cuda.so the CUDA backend plugin (eagle-backend/1 C seam),
#                          no Python dependency: built ONCE and copied into every
#                          per-Python wheel
#
# EAGLE_PYTHON_CUDA_PLUGIN selects where the plugin comes from:
#   BUILD   (default) build it here with nvcc
#   OFF     no plugin (the GPU-less leg); every CUDA entry point raises
#           eagle.BackendUnavailable
#   <path>  install that prebuilt libeagle_cuda.so (the per-Python phase)
# EAGLE_PYTHON_SKIP_CORE=ON builds the plugin alone (phase 1, no Python needed).
cmake_minimum_required(VERSION 3.20)
project(eagle_python LANGUAGES CXX)

set(EAGLE_PYTHON_CUDA_PLUGIN "BUILD" CACHE STRING
    "Where libeagle_cuda.so comes from: BUILD | OFF | <path to a prebuilt plugin>")
option(EAGLE_PYTHON_SKIP_CORE "Build only the CUDA plugin, not eagle._core" OFF)
option(EAGLE_BACKEND_INJECT_MISSING
    "Export one seam symbol under a wrong name (symbol-gate injection build only)" OFF)

string(TOUPPER "${EAGLE_PYTHON_CUDA_PLUGIN}" _eagle_plugin_mode)
if(NOT _eagle_plugin_mode STREQUAL "BUILD" AND NOT _eagle_plugin_mode STREQUAL "OFF")
  set(_eagle_plugin_mode "PATH")
  if(NOT EXISTS "${EAGLE_PYTHON_CUDA_PLUGIN}")
    message(FATAL_ERROR
      "EAGLE_PYTHON_CUDA_PLUGIN must be BUILD, OFF or the path of a prebuilt "
      "libeagle_cuda.so; '${EAGLE_PYTHON_CUDA_PLUGIN}' does not exist")
  endif()
endif()
if(EAGLE_PYTHON_SKIP_CORE AND NOT _eagle_plugin_mode STREQUAL "BUILD")
  message(FATAL_ERROR "EAGLE_PYTHON_SKIP_CORE=ON builds nothing unless EAGLE_PYTHON_CUDA_PLUGIN=BUILD")
endif()

# Bundle libstdc++/libgcc when the toolchain has their static archives (the
# wheel then loads in envs whose system libstdc++ predates the build
# toolchain's GLIBCXX); a toolchain without them (a distribution gcc without
# its libstdc++-static package) links them dynamically instead.
include(CheckLinkerFlag)
check_linker_flag(CXX "-static-libstdc++" EAGLE_STATIC_LIBSTDCXX)
if(EAGLE_STATIC_LIBSTDCXX)
  set(_eagle_static_runtime -static-libstdc++ -static-libgcc)
else()
  set(_eagle_static_runtime "")
  message(WARNING "eagle: no static libstdc++ in this toolchain; linking it dynamically")
endif()

# Only the plugin needs the CUDA toolkit; the core is plain C++.
if(_eagle_plugin_mode STREQUAL "BUILD")
  # SASS for every architecture this nvcc compiles from sm_60 up (aether's
  # compensated sums need FP64 atomicAdd, which sm_5x lacks), plus PTX at the
  # NEWEST, which GPUs released after the toolkit JIT once and cache. The list
  # is the toolkit's own (nvcc --list-gpu-arch), so a newer toolkit widens it.
  # Override for a faster local build with a QUOTED list, e.g.
  # -DCMAKE_CUDA_ARCHITECTURES=61.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # the compiler CMake will enable below: -DCMAKE_CUDA_COMPILER, then
    # $CUDACXX, then nvcc on PATH
    if(CMAKE_CUDA_COMPILER)
      set(_eagle_nvcc "${CMAKE_CUDA_COMPILER}")
    elseif(NOT "$ENV{CUDACXX}" STREQUAL "")
      set(_eagle_nvcc "$ENV{CUDACXX}")
    else()
      find_program(_eagle_nvcc_on_path nvcc)
      set(_eagle_nvcc "${_eagle_nvcc_on_path}")
    endif()
    set(_eagle_archs "")
    if(_eagle_nvcc)
      execute_process(COMMAND "${_eagle_nvcc}" --list-gpu-arch
                      OUTPUT_VARIABLE _eagle_listed ERROR_QUIET RESULT_VARIABLE _rc)
      if(_rc EQUAL 0)
        string(REGEX MATCHALL "compute_[0-9]+" _eagle_listed "${_eagle_listed}")
        foreach(_c IN LISTS _eagle_listed)
          string(REPLACE "compute_" "" _a "${_c}")
          if(_a GREATER_EQUAL 60)
            list(APPEND _eagle_archs ${_a})
          endif()
        endforeach()
        list(REMOVE_DUPLICATES _eagle_archs)
        list(SORT _eagle_archs COMPARE NATURAL)
      endif()
    endif()
    if(NOT _eagle_archs)  # no nvcc to ask here: CMake finds the compiler later
      set(_eagle_archs 61 70 80 90)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "")
    foreach(_a IN LISTS _eagle_archs)
      list(APPEND CMAKE_CUDA_ARCHITECTURES "${_a}-real")
    endforeach()
    list(GET _eagle_archs -1 _eagle_newest)
    list(APPEND CMAKE_CUDA_ARCHITECTURES "${_eagle_newest}-virtual")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# eagle C++ headers (INTERFACE) + its aether/OpenMP deps.
#
# This binding lives INSIDE the eagle repo, so it must compile the eagle headers
# that sit next to it. A bare find_package would resolve against whatever eagle
# is installed in the ambient environment prefix, which on a dev box is easily a
# a different (older) build — and the failure is SILENT: the wrong headers
# compile perfectly and produce a binding missing the very feature being added.
# Prefer a repo-local install of the tracked tree; the caller can still override
# by passing -Deagle_DIR explicitly.
if(NOT DEFINED eagle_DIR)
  set(_eagle_tracked "${CMAKE_CURRENT_SOURCE_DIR}/../__install_p1/share/eagle/cmake")
  if(EXISTS "${_eagle_tracked}/eagleConfig.cmake")
    set(eagle_DIR "${_eagle_tracked}")
  endif()
endif()
find_package(eagle CONFIG REQUIRED)

# Prove the resolved eagle is recent enough to contain what this TU binds,
# rather than discovering it at import time as a missing attribute. Keyed on the
# header the binding actually needs; a stale install fails the configure loudly.
get_target_property(_eagle_incs eagle::eagle INTERFACE_INCLUDE_DIRECTORIES)
set(_eagle_root "")
foreach(_dir IN LISTS _eagle_incs)
  if(EXISTS "${_dir}/eagle/cuda/CaptureFork.h")
    set(_eagle_root "${_dir}")
    break()
  endif()
endforeach()
if(NOT _eagle_root)
  message(FATAL_ERROR
    "The resolved eagle package does not provide eagle/cuda/CaptureFork.h.\n"
    "  eagle_DIR                     = ${eagle_DIR}\n"
    "  INTERFACE_INCLUDE_DIRECTORIES = ${_eagle_incs}\n"
    "This binding must compile the TRACKED eagle headers. Install the tracked "
    "tree to a prefix you own and pass -Deagle_DIR=<prefix>/share/eagle/cmake. "
    "Never install into another environment's prefix.")
endif()

# Compile the LIVE tracked headers, not the installed copy. The install supplies
# the package interface (compile definitions, flags, aether, and the eagle/plugin/
# prefix that only exists post-install), but its headers are a SNAPSHOT: they go
# stale the moment someone edits the tree, and the build then silently compiles
# yesterday's sources — the same silent-staleness failure this whole block
# exists to prevent, just relocated. Prepending the repo root makes
# eagle/{cuda,util,cpu,...} resolve live; eagle/plugin/ has no in-tree
# counterpart under that prefix, so it still falls through to the install.
get_filename_component(_eagle_src "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if(EXISTS "${_eagle_src}/eagle/cuda/CaptureFork.h")
  set_property(TARGET eagle::eagle PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${_eagle_src}" ${_eagle_incs})
  set(_eagle_root "${_eagle_src} (live tree, ahead of ${_eagle_root})")
endif()
message(STATUS "> eagle headers (verified to contain CaptureFork.h): ${_eagle_root}")

# Both artifacts carry eagle/aether inline and template instantiations; a
# STB_GNU_UNIQUE copy of one would bind across the two images at load time.
# GCC-only flag: clang never emits STB_GNU_UNIQUE and rejects it.
set(_eagle_no_gnu_unique
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-fno-gnu-unique>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-gnu-unique>)

# ---------------------------------------------------------------------------
# libeagle_cuda.so — the CUDA backend plugin.
#
# Driver API only: no CUDA runtime is linked (CUDA_RUNTIME_LIBRARY None). The
# runtime calls eagle's headers make are implemented over the driver by
# src/plugin/cuda_driver.cpp, which opens libcuda.so.1 at run time, so the only
# CUDA library the plugin ever loads is the driver's. The plugin's own kernels
# are compiled ahead of time into the embedded fat binary (the arch list above)
# and loaded with cuModuleLoadData. Hidden visibility and a version script
# export the eagle_backend_* C seam and nothing else; libstdc++/libgcc are
# bundled so the plugin loads next to any core. It is dlopen'd by eagle._core
# (src/seam/Loader.h), never linked, so nothing else may depend on it.
# --no-undefined turns a runtime call the translation layer does not implement
# into a link error instead of a load-time failure.
# ---------------------------------------------------------------------------
if(_eagle_plugin_mode STREQUAL "BUILD")
  add_library(eagle_cuda SHARED src/plugin/eagle_cuda.cu src/plugin/cuda_driver.cpp)
  target_compile_features(eagle_cuda PRIVATE cxx_std_20)
  target_include_directories(eagle_cuda PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  # Headers only (cuda.h, cudaTypedefs.h, cuda_runtime_api.h); nothing from the
  # toolkit is linked.
  target_include_directories(eagle_cuda SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(eagle_cuda PRIVATE eagle::eagle ${CMAKE_DL_LIBS})
  string(REPLACE ";" "," _eagle_plugin_archs "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(eagle_cuda PRIVATE
    EAGLE_PYTHON_VERSION="${eagle_VERSION}"
    EAGLE_CUDA_PLUGIN_ARCHS="${_eagle_plugin_archs}"
    $<$<BOOL:${EAGLE_BACKEND_INJECT_MISSING}>:EAGLE_BACKEND_INJECT_MISSING=1>)
  target_compile_options(eagle_cuda PRIVATE ${_eagle_no_gnu_unique})
  set(_eagle_map "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/eagle_cuda.map")
  target_link_options(eagle_cuda PRIVATE
    "LINKER:--version-script=${_eagle_map}" "LINKER:--no-undefined" ${_eagle_static_runtime})
  set_target_properties(eagle_cuda PROPERTIES
    OUTPUT_NAME eagle_cuda
    # Link through the C++ driver (the one carrying the static libstdc++), not
    # nvcc's host compiler.
    LINKER_LANGUAGE CXX
    LINK_DEPENDS "${_eagle_map}"
    CUDA_RUNTIME_LIBRARY None
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    POSITION_INDEPENDENT_CODE ON
    # One TU, whole-program device compilation: a relocatable (-rdc) build is
    # device-linked to SASS only, which would drop the 90-virtual PTX the arch
    # list asks for (JIT on newer GPUs). Nothing here needs device linking.
    CUDA_SEPARABLE_COMPILATION OFF)
  install(TARGETS eagle_cuda LIBRARY DESTINATION eagle)
  message(STATUS "> libeagle_cuda.so: built here (archs ${CMAKE_CUDA_ARCHITECTURES})")
elseif(_eagle_plugin_mode STREQUAL "PATH")
  install(FILES "${EAGLE_PYTHON_CUDA_PLUGIN}" DESTINATION eagle RENAME libeagle_cuda.so)
  message(STATUS "> libeagle_cuda.so: prebuilt ${EAGLE_PYTHON_CUDA_PLUGIN}")
else()
  message(STATUS "> libeagle_cuda.so: NOT included (EAGLE_PYTHON_CUDA_PLUGIN=OFF)")
endif()

if(EAGLE_PYTHON_SKIP_CORE)
  message(STATUS "> eagle._core: NOT built (EAGLE_PYTHON_SKIP_CORE=ON)")
  return()
endif()

# ---------------------------------------------------------------------------
# eagle._core — the nanobind module.
# ---------------------------------------------------------------------------

# Python + nanobind from the building interpreter (scikit-build-core provides it;
# the --cmake_dir fallback also covers a plain `cmake` invocation).
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_ROOT}" NO_DEFAULT_PATH)
endif()

# The core is plain C++ (EAGLE_CPU_ONLY + AETHER_CPP_MODE): it reaches CUDA only
# through the plugin, so it needs no CUDA toolkit to build and no driver to load.
# It takes the eagle/aether HEADERS only — not eagle::eagle / aether::aether, whose
# exported definitions describe the CUDA-mode package (AETHER_HAS_CUDA, ...).
find_package(OpenMP REQUIRED COMPONENTS CXX)
get_target_property(_eagle_core_incs eagle::eagle INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_aether_incs aether::aether INTERFACE_INCLUDE_DIRECTORIES)
nanobind_add_module(_core MODULE FREE_THREADED src/eagle_core.cpp)
target_compile_features(_core PRIVATE cxx_std_23)
# aether's own include directories first: an installed eagle's prefix may also
# hold an (older) aether copy, and the aether this configure resolved must win.
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${_aether_incs} ${_eagle_core_incs})
target_compile_definitions(_core PRIVATE EAGLE_CPU_ONLY=1 AETHER_CPP_MODE=1 EAGLE_BLOCKSIZE=256 NDEBUG)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
target_compile_options(_core PRIVATE ${_eagle_no_gnu_unique})
# libstdc++/libgcc bundled where the toolchain allows (see _eagle_static_runtime).
target_link_options(_core PRIVATE ${_eagle_static_runtime})
set_target_properties(_core PROPERTIES
  CXX_VISIBILITY_PRESET "hidden"
  POSITION_INDEPENDENT_CODE ON
  LINKER_LANGUAGE CXX)
# The wall, asserted where it is decided: no CUDA source reaches the core.
get_target_property(_core_sources _core SOURCES)
foreach(_src IN LISTS _core_sources)
  if(_src MATCHES "\\.cu$")
    message(FATAL_ERROR "eagle._core must stay CUDA-free; it lists the CUDA source ${_src}")
  endif()
endforeach()

# scikit-build-core maps this DESTINATION into the wheel's eagle package.
install(TARGETS _core LIBRARY DESTINATION eagle)

# ---------------------------------------------------------------------------
# eagle._mpi — the OPTIONAL second extension: eagle::exec::RankPartition.
#
# WHY IT IS A SECOND MODULE. `eagle/exec/RankPartition.h` refuses to compile
# without `EAGLE_MPI`, and that define exists so eagle never links libmpi: the C++
# side puts it PRIVATELY on one test target and nowhere else. Putting the rank
# surface into `_core` would drag libmpi into the extension every eagle user
# imports — including everyone with no MPI installed, for whom `import eagle`
# would simply stop working. `_core` stays MPI-free (an audit row in the Python
# suite reads its `ldd`), and the rank structure lives here, built only where an
# MPI is present.
#
# EAGLE_PYTHON_MPI is TRI-STATE, not a bool, because "off" and "not available"
# must stay distinguishable: ON demands an MPI and FAILS THE CONFIGURE without
# one (a silently skipped module is how a CI job certifies a surface it never
# built), OFF never looks, and AUTO builds it iff MPI is found. Either way this
# block prints what it decided — a build whose optional half quietly vanished is
# the thing the STATUS line exists to prevent.
# ---------------------------------------------------------------------------
set(EAGLE_PYTHON_MPI "AUTO" CACHE STRING
    "Build the optional eagle._mpi extension (eagle::exec::RankPartition): ON | OFF | AUTO")
set_property(CACHE EAGLE_PYTHON_MPI PROPERTY STRINGS ON OFF AUTO)
string(TOUPPER "${EAGLE_PYTHON_MPI}" _eagle_python_mpi)

if(_eagle_python_mpi STREQUAL "OFF")
  message(STATUS "> eagle._mpi: NOT built (EAGLE_PYTHON_MPI=OFF)")
elseif(_eagle_python_mpi STREQUAL "ON" OR _eagle_python_mpi STREQUAL "AUTO")
  if(_eagle_python_mpi STREQUAL "ON")
    find_package(MPI REQUIRED COMPONENTS CXX)
  else()
    find_package(MPI QUIET COMPONENTS CXX)
  endif()
  if(NOT MPI_CXX_FOUND)
    message(STATUS "> eagle._mpi: NOT built (EAGLE_PYTHON_MPI=AUTO and no MPI found); "
                   "eagle.exec.RankPartition will refuse, naming the rule")
  else()
    message(STATUS "> eagle._mpi: building against MPI ${MPI_CXX_VERSION} (${MPI_CXX_COMPILER})")

    # The library path the module re-opens RTLD_GLOBAL before MPI_Init, so Open
    # MPI's dlopen'ed MCA components can resolve against it (a Python extension is
    # loaded RTLD_LOCAL; see promoteMpiSymbols() in src/eagle_mpi.cpp). Resolved
    # HERE from what MPI::MPI_CXX actually carries — never a guessed SONAME. An
    # imported-target-only MPI leaves it empty and the module skips the promotion.
    set(_eagle_mpi_library "")
    foreach(_lib IN LISTS MPI_CXX_LIBRARIES)
      if(_lib MATCHES "libmpi\\.(so|dylib)")
        set(_eagle_mpi_library "${_lib}")
        break()
      endif()
    endforeach()
    message(STATUS "> eagle._mpi: RTLD_GLOBAL promotion target = "
                   "'${_eagle_mpi_library}'")

    # Plain C++ like _core (its device inner forwards to the CUDA backend).
    nanobind_add_module(_mpi MODULE FREE_THREADED src/eagle_mpi.cpp)
    target_compile_features(_mpi PRIVATE cxx_std_23)
    target_include_directories(_mpi PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/.." ${_aether_incs} ${_eagle_core_incs})
    # EAGLE_MPI is PRIVATE to this target, exactly as it is on the C++ bed: it is
    # what makes RankPartition.h compile at all, and it must not leak to anything
    # else in the build.
    target_compile_definitions(_mpi PRIVATE
      EAGLE_CPU_ONLY=1 AETHER_CPP_MODE=1 EAGLE_BLOCKSIZE=256 NDEBUG
      EAGLE_MPI=1 EAGLE_MPI_LIBRARY="${_eagle_mpi_library}")
    target_link_libraries(_mpi PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX ${CMAKE_DL_LIBS})
    target_compile_options(_mpi PRIVATE ${_eagle_no_gnu_unique})
    # The same wheel-portability link flags `_core` carries, for the same reason.
    target_link_options(_mpi PRIVATE ${_eagle_static_runtime})
    set_target_properties(_mpi PROPERTIES
      CXX_VISIBILITY_PRESET "hidden"
      POSITION_INDEPENDENT_CODE ON)
    install(TARGETS _mpi LIBRARY DESTINATION eagle)
  endif()
else()
  message(FATAL_ERROR
    "EAGLE_PYTHON_MPI must be ON, OFF or AUTO; got '${EAGLE_PYTHON_MPI}'")
endif()
