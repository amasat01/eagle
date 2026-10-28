# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Builds the eagle Python package's compiled artifacts (scikit-build-core):
#
#   eagle/_core.*.so       the nanobind module, one per Python version
#   eagle/libeagle_cuda.so the CUDA backend plugin (eagle-backend/1 C seam),
#                          no Python dependency: built ONCE and copied into every
#                          per-Python wheel
#   eagle/libeagle_xla.so  the XLA FFI handler behind eagle.frameworks.jax, plain
#                          C++, built on every leg (with or without CUDA)
#
# EAGLE_PYTHON_CUDA_PLUGIN selects where the plugin comes from:
#   BUILD   (default in a repository checkout) build it here with nvcc
#   AUTO    (default from a source distribution) BUILD when a CUDA compiler is
#           found, else OFF with a message saying so
#   OFF     no plugin (the GPU-less leg); every CUDA entry point raises
#           eagle.BackendUnavailable
#   <path>  install that prebuilt libeagle_cuda.so (the per-Python phase)
# EAGLE_PYTHON_SKIP_CORE=ON builds the plugin alone (phase 1, no Python needed).
# EAGLE_PYTHON_WARN_AS_ERR=ON turns compiler warnings in this package's own
# sources into errors.
#
# Built from a source distribution (pip install of the sdist), there is no
# repository above this directory: the eagle, plugin and aether headers come
# from the copy the sdist carries in _sdist_include/ (see _build_backend.py),
# and the plugin default is AUTO.
cmake_minimum_required(VERSION 3.20)
project(eagle_python LANGUAGES CXX)

get_filename_component(_eagle_src "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(_eagle_sdist_include "${CMAKE_CURRENT_SOURCE_DIR}/_sdist_include")
set(_eagle_from_sdist OFF)
# The eagle headers' root: the repo, else the sdist's copy (an sdist has no repo above it, even
# when an installed eagle is named by eagle_DIR; the installed package lacks plugin/).
set(_eagle_hdr_root "${_eagle_src}")
if(NOT EXISTS "${_eagle_src}/eagle/cuda/CaptureFork.h"
   AND EXISTS "${_eagle_sdist_include}/eagle/cuda/CaptureFork.h"
   AND NOT DEFINED eagle_DIR)
  set(_eagle_from_sdist ON)
endif()
if(NOT EXISTS "${_eagle_src}/eagle/cuda/CaptureFork.h" AND EXISTS "${_eagle_sdist_include}/eagle/cuda/CaptureFork.h")
  set(_eagle_hdr_root "${_eagle_sdist_include}")
endif()
if(_eagle_from_sdist)
  set(_eagle_plugin_default "AUTO")
else()
  set(_eagle_plugin_default "BUILD")
endif()

option(EAGLE_PYTHON_WARN_AS_ERR "Compiler warnings in this package's sources are errors" OFF)
set(EAGLE_PYTHON_CUDA_PLUGIN "${_eagle_plugin_default}" CACHE STRING
    "Where libeagle_cuda.so comes from: BUILD | AUTO | OFF | <path to a prebuilt plugin>")
option(EAGLE_PYTHON_SKIP_CORE "Build only the CUDA plugin, not eagle._core" OFF)
option(EAGLE_BACKEND_INJECT_MISSING
    "Export one seam symbol under a wrong name (symbol-gate injection build only)" OFF)

string(TOUPPER "${EAGLE_PYTHON_CUDA_PLUGIN}" _eagle_plugin_mode)
set(_eagle_plugin_skipped OFF)
if(_eagle_plugin_mode STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(_eagle_plugin_mode "BUILD")
  else()
    set(_eagle_plugin_mode "OFF")
    set(_eagle_plugin_skipped ON)
    message(STATUS "> CUDA toolkit not found: building eagle without the GPU plugin; "
                   "install the CUDA toolkit and reinstall for GPU support")
  endif()
endif()
if(NOT _eagle_plugin_mode STREQUAL "BUILD" AND NOT _eagle_plugin_mode STREQUAL "OFF")
  set(_eagle_plugin_mode "PATH")
  if(NOT EXISTS "${EAGLE_PYTHON_CUDA_PLUGIN}")
    message(FATAL_ERROR
      "EAGLE_PYTHON_CUDA_PLUGIN must be BUILD, AUTO, OFF or the path of a prebuilt "
      "libeagle_cuda.so; '${EAGLE_PYTHON_CUDA_PLUGIN}' does not exist")
  endif()
endif()
if(EAGLE_PYTHON_WARN_AS_ERR)
  set(_eagle_warn_as_err
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
else()
  set(_eagle_warn_as_err "")
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
if(EAGLE_STATIC_LIBSTDCXX AND NOT APPLE)
  set(_eagle_static_runtime -static-libstdc++ -static-libgcc)
elseif(APPLE)
  set(_eagle_static_runtime "")  # macOS links libc++ from the system
else()
  set(_eagle_static_runtime "")
  message(WARNING "eagle: no static libstdc++ in this toolchain; linking it dynamically")
endif()
# The extensions export only PyInit_*: keep the bundled runtime's symbols out of
# the dynamic table and drop unreferenced sections, as the plugins already do.
if(NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(_eagle_ext_compile -ffunction-sections -fdata-sections)
  set(_eagle_ext_link "LINKER:--exclude-libs,ALL" "LINKER:--gc-sections")
elseif(APPLE)
  # drop local symbols at link time: the install then needs no `strip`, which warns on a signed binary
  set(_eagle_ext_link "LINKER:-x")
endif()

# Only the plugin needs the CUDA toolkit; the core is plain C++.
if(_eagle_plugin_mode STREQUAL "BUILD")
  # SASS for every architecture this nvcc compiles from sm_60 up (aether's
  # compensated sums need FP64 atomicAdd, which sm_5x lacks), plus PTX at the
  # NEWEST (and at compute_100 when the toolkit has it), which GPUs released after the toolkit JIT once and cache. The list
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
    # Also PTX at compute_100 where nvcc has Blackwell: CUDA 13 renamed Jetson
    # Thor sm_101 to sm_110, so a CUDA 13 driver finds no SASS for it and cannot
    # JIT compute_121 PTX down to 11.0, but does JIT compute_100 PTX up to 11.0
    # (and to any later architecture).
    if(100 IN_LIST _eagle_archs AND NOT _eagle_newest EQUAL 100)
      list(APPEND CMAKE_CUDA_ARCHITECTURES "100-virtual")
    endif()
    list(APPEND CMAKE_CUDA_ARCHITECTURES "${_eagle_newest}-virtual")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# eagle C++ headers (INTERFACE) + its aether/OpenMP deps.
#
# From a source distribution: the header copy it carries, and nothing from the
# ambient environment (the copy is the exact tree this version was cut from).
# The compile definitions are the ones the installed CUDA-mode eagle and aether
# packages export; only the plugin uses them (the core takes include paths only).
if(_eagle_from_sdist)
  add_library(aether::aether INTERFACE IMPORTED)
  set_target_properties(aether::aether PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_eagle_sdist_include}"
    INTERFACE_COMPILE_DEFINITIONS "AETHER_WARP_SIZE=32;AETHER_HAS_CUDA=1")
  add_library(eagle::eagle INTERFACE IMPORTED)
  set_target_properties(eagle::eagle PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_eagle_sdist_include}"
    INTERFACE_COMPILE_DEFINITIONS
      "EAGLE_BLOCKSIZE=256;CUDA_API_PER_THREAD_DEFAULT_STREAM=1;NDEBUG"
    INTERFACE_LINK_LIBRARIES "aether::aether")
  if(_eagle_plugin_mode STREQUAL "BUILD")
    set_property(TARGET aether::aether APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES CUDA::toolkit)
  endif()
  # the version the plugin reports: this package's, which the C++ project shares
  set(eagle_VERSION "${SKBUILD_PROJECT_VERSION}")
  message(STATUS "> eagle headers (source distribution copy): ${_eagle_sdist_include}")
else()
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
if(EXISTS "${_eagle_src}/eagle/cuda/CaptureFork.h")
  set_property(TARGET eagle::eagle PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${_eagle_src}" ${_eagle_incs})
  set(_eagle_root "${_eagle_src} (live tree, ahead of ${_eagle_root})")
endif()
message(STATUS "> eagle headers (verified to contain CaptureFork.h): ${_eagle_root}")
endif()

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
# OpenMP: ONE runtime per process, owned by aether-dsc (aether/_omp/). Every eagle extension finds it at run time by a
# relative rpath into the installed aether package (the wheel repair is told to leave it, so nothing here is vendored).
# At build time it links aether's own file when aether_dsc.openmp() reports one, else the toolchain's OpenMP of the same
# soname (Linux) or, on macOS, the toolchain's libomp renamed to @rpath/libomp.dylib so it resolves to aether's copy.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import aether_dsc\no = aether_dsc.openmp()\nprint(';'.join([str(o.library), *o.compile_flags]) if o else '')"
  OUTPUT_VARIABLE _eagle_omp OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(_eagle_omp)
  list(POP_FRONT _eagle_omp _eagle_omp_library)
  set(_eagle_omp_compile ${_eagle_omp})
  add_library(eagle_openmp SHARED IMPORTED)
  set_target_properties(eagle_openmp PROPERTIES IMPORTED_LOCATION "${_eagle_omp_library}")
  message(STATUS "> OpenMP runtime (aether-dsc's): ${_eagle_omp_library}")
  # The compiler and the shipped runtime must be one family: clang's -fopenmp calls LLVM's __kmpc_*, which libgomp does
  # not export (the extension would build, then fail to import), and gcc's calls GOMP_*, which libomp does not export.
  file(STRINGS "${_eagle_omp_library}" _eagle_omp_kmpc REGEX "^__kmpc_reduce_nowait$" LIMIT_COUNT 1)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE AND NOT _eagle_omp_kmpc)
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} compiles OpenMP against libomp, but aether-dsc ships ${_eagle_omp_library}, a libgomp. "
                        "Build aether-dsc with the same compiler (it must ship libomp), or build eagle with GCC.")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND _eagle_omp_kmpc)
    message(FATAL_ERROR "GNU compiles OpenMP against libgomp, but aether-dsc ships ${_eagle_omp_library}, a libomp. "
                        "Build aether-dsc with the same compiler (it must ship libgomp), or build eagle with Clang.")
  endif()
else()
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  message(STATUS "> OpenMP runtime (the toolchain's; aether-dsc reports none here)")
endif()
function(eagle_link_openmp target)
  if(TARGET eagle_openmp)
    target_link_libraries(${target} PRIVATE eagle_openmp)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_eagle_omp_compile}>)
  else()
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    if(APPLE)
      foreach(_l IN LISTS OpenMP_CXX_LIBRARIES)
        if(_l MATCHES "libomp\\.dylib" AND EXISTS "${_l}")
          add_custom_command(TARGET ${target} POST_BUILD VERBATIM
            COMMAND sh -c "id=$(otool -D '${_l}' | tail -n 1); install_name_tool -change \"$id\" @rpath/libomp.dylib '$<TARGET_FILE:${target}>' && codesign --force --sign - '$<TARGET_FILE:${target}>'")
        endif()
      endforeach()
    endif()
  endif()
  if(APPLE)
    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "@loader_path/../aether/_omp")
  else()
    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../aether/_omp")
  endif()
  set_property(TARGET ${target} PROPERTY BUILD_WITH_INSTALL_RPATH ON)
endfunction()

if(_eagle_plugin_mode STREQUAL "BUILD")
  add_library(eagle_cuda SHARED src/plugin/eagle_cuda.cu src/plugin/cuda_driver.cpp)
  target_compile_features(eagle_cuda PRIVATE cxx_std_20)
  target_include_directories(eagle_cuda PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${_eagle_hdr_root}")
  # Headers only (cuda.h, cudaTypedefs.h, cuda_runtime_api.h); nothing from the
  # toolkit is linked.
  target_include_directories(eagle_cuda SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(eagle_cuda PRIVATE eagle::eagle ${CMAKE_DL_LIBS})
  # The installed aether/eagle packages name libgomp by path, from whichever
  # toolset built them; a second toolset's directories on the link line can hide
  # it. Drop those paths and let the compiler link OpenMP (-fopenmp) instead.
  foreach(_t aether::aether eagle::eagle)
    get_target_property(_libs ${_t} INTERFACE_LINK_LIBRARIES)
    if(_libs)
      list(FILTER _libs EXCLUDE REGEX "libgomp|libpthread|\\$<LINK_LANGUAGE")
      set_property(TARGET ${_t} PROPERTY INTERFACE_LINK_LIBRARIES "${_libs}")
    endif()
  endforeach()
  eagle_link_openmp(eagle_cuda)
  # Compact listing, e.g. "61,70,90/90": a long arch list must fit the
  # build_info archs field (checked at compile time in eagle_cuda.cu).
  set(_eagle_sass "")
  set(_eagle_ptx "")
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_a MATCHES "^([0-9]+)-virtual$")
      list(APPEND _eagle_ptx "${CMAKE_MATCH_1}")
    elseif(_a MATCHES "^([0-9]+)(-real)?$")
      list(APPEND _eagle_sass "${CMAKE_MATCH_1}")
      if(NOT CMAKE_MATCH_2)
        list(APPEND _eagle_ptx "${CMAKE_MATCH_1}")
      endif()
    else()
      list(APPEND _eagle_sass "${_a}")
    endif()
  endforeach()
  list(JOIN _eagle_sass "," _eagle_sass)
  list(JOIN _eagle_ptx "," _eagle_ptx)
  set(_eagle_plugin_archs "${_eagle_sass}/${_eagle_ptx}")
  # No version here: the plugin object depends only on its sources, headers, flags and architectures, so one build
  # serves every commit (eagle._core reports the package version, from src/eagle_version.cpp).
  target_compile_definitions(eagle_cuda PRIVATE
    EAGLE_CUDA_PLUGIN_ARCHS="${_eagle_plugin_archs}"
    $<$<BOOL:${EAGLE_BACKEND_INJECT_MISSING}>:EAGLE_BACKEND_INJECT_MISSING=1>)
  target_compile_options(eagle_cuda PRIVATE ${_eagle_no_gnu_unique} ${_eagle_warn_as_err})
  # Maxwell and Pascal (sm_5x/6x) are shipped on purpose; nvcc's notice that offline compilation for them will be
  # removed in a future release is the one warning we accept for that (the narrowest documented flag).
  target_compile_options(eagle_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
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
get_target_property(_eagle_core_incs eagle::eagle INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_aether_incs aether::aether INTERFACE_INCLUDE_DIRECTORIES)
nanobind_add_module(_core MODULE FREE_THREADED src/eagle_core.cpp src/eagle_version.cpp)
# the package version lives in this one small file: a new version recompiles it alone
set_source_files_properties(src/eagle_version.cpp PROPERTIES COMPILE_DEFINITIONS EAGLE_PYTHON_VERSION="${eagle_VERSION}")
target_compile_features(_core PRIVATE cxx_std_23)
# aether's own include directories first: an installed eagle's prefix may also
# hold an (older) aether copy, and the aether this configure resolved must win.
# the live eagle root ahead of aether's prefix, which may hold an installed (older) eagle copy
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${_eagle_hdr_root}" ${_aether_incs} ${_eagle_core_incs})
target_compile_definitions(_core PRIVATE EAGLE_CPU_ONLY=1 AETHER_CPP_MODE=1 EAGLE_BLOCKSIZE=256 NDEBUG
  $<$<BOOL:${_eagle_plugin_skipped}>:EAGLE_CORE_BUILT_WITHOUT_CUDA_PLUGIN=1>)
target_link_libraries(_core PRIVATE ${CMAKE_DL_LIBS})
eagle_link_openmp(_core)
target_compile_options(_core PRIVATE -ffp-contract=off ${_eagle_no_gnu_unique} ${_eagle_warn_as_err} ${_eagle_ext_compile})
# libstdc++/libgcc bundled where the toolchain allows (see _eagle_static_runtime).
target_link_options(_core PRIVATE ${_eagle_static_runtime} ${_eagle_ext_link})
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
# libeagle_xla.so — the generic XLA FFI handler behind eagle.frameworks.jax.
#
# Plain C++ on every leg (GPU-less included): no CUDA toolchain, no Python, no
# JAX at build time. The handler speaks XLA's stable C API; its one header is vendored
# (src/xla/vendor/xla/ffi/api/c_api.h, from jaxlib 0.11.2, FFI API 0.4; the JAX floor is that version) and the CUDA
# entry points it calls are addresses Python resolves from the loaded
# libeagle_cuda.so, so nothing CUDA is linked. A version script exports the
# three eagle_xla_* symbols and nothing else.
# ---------------------------------------------------------------------------
add_library(eagle_xla SHARED src/xla/eagle_xla.cpp)
target_compile_features(eagle_xla PRIVATE cxx_std_23)
# the host finish is eagle's canonical reduction (eagle/reduce/canonical.h), header only
target_include_directories(eagle_xla PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${_eagle_hdr_root}" ${_aether_incs} ${_eagle_core_incs})
target_compile_definitions(eagle_xla PRIVATE EAGLE_CPU_ONLY=1 AETHER_CPP_MODE=1 EAGLE_BLOCKSIZE=256 NDEBUG)
target_include_directories(eagle_xla SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/xla/vendor")
# one section per function/datum so the linker can drop the static runtime code the handler never calls
target_compile_options(eagle_xla PRIVATE ${_eagle_no_gnu_unique} ${_eagle_warn_as_err} -ffunction-sections -fdata-sections)
set(_eagle_xla_map "${CMAKE_CURRENT_SOURCE_DIR}/src/xla/eagle_xla.map")
# A sanitized build leaves the sanitizer runtime's symbols to the (sanitized)
# interpreter that loads the library, so --no-undefined would reject them.
if(APPLE)
  # Apple's ld: the same rules as the GNU ld flags below (export list, unreferenced code dropped; undefined symbols
  # are errors by default there, and spelling that out as -undefined,error is deprecated). The list is
  # eagle_xla.map's `global:` section.
  set(_eagle_xla_exports "${CMAKE_CURRENT_SOURCE_DIR}/src/xla/eagle_xla.exp")
  target_link_options(eagle_xla PRIVATE
    "LINKER:-exported_symbols_list,${_eagle_xla_exports}" "LINKER:-dead_strip")
  set_property(TARGET eagle_xla APPEND PROPERTY LINK_DEPENDS "${_eagle_xla_exports}")
  set_property(TARGET eagle_xla PROPERTY SUFFIX ".so")  # eagle.frameworks.jax opens libeagle_xla.so on every platform
else()
  if("${CMAKE_CXX_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}" MATCHES "-fsanitize")
    set(_eagle_xla_no_undefined "")
  else()
    set(_eagle_xla_no_undefined "LINKER:--no-undefined")
  endif()
  target_link_options(eagle_xla PRIVATE
    "LINKER:--version-script=${_eagle_xla_map}" ${_eagle_xla_no_undefined} "LINKER:--gc-sections" ${_eagle_static_runtime})
endif()
set_target_properties(eagle_xla PROPERTIES
  OUTPUT_NAME eagle_xla
  LINK_DEPENDS "${_eagle_xla_map}"
  CXX_VISIBILITY_PRESET "hidden"
  POSITION_INDEPENDENT_CODE ON
  LINKER_LANGUAGE CXX)
install(TARGETS eagle_xla LIBRARY DESTINATION eagle)
message(STATUS "> libeagle_xla.so: built here (XLA FFI C API header vendored from jaxlib 0.11.2)")

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
      "${_eagle_hdr_root}" ${_aether_incs} ${_eagle_core_incs})
    # EAGLE_MPI is PRIVATE to this target, exactly as it is on the C++ bed: it is
    # what makes RankPartition.h compile at all, and it must not leak to anything
    # else in the build.
    target_compile_definitions(_mpi PRIVATE
      EAGLE_CPU_ONLY=1 AETHER_CPP_MODE=1 EAGLE_BLOCKSIZE=256 NDEBUG
      EAGLE_MPI=1 EAGLE_MPI_LIBRARY="${_eagle_mpi_library}")
    target_link_libraries(_mpi PRIVATE MPI::MPI_CXX ${CMAKE_DL_LIBS})
    eagle_link_openmp(_mpi)
    target_compile_options(_mpi PRIVATE ${_eagle_no_gnu_unique} ${_eagle_warn_as_err} ${_eagle_ext_compile})
    # The same wheel-portability link flags `_core` carries, for the same reason.
    target_link_options(_mpi PRIVATE ${_eagle_static_runtime} ${_eagle_ext_link})
    set_target_properties(_mpi PROPERTIES
      CXX_VISIBILITY_PRESET "hidden"
      POSITION_INDEPENDENT_CODE ON)
    install(TARGETS _mpi LIBRARY DESTINATION eagle)
  endif()
else()
  message(FATAL_ERROR
    "EAGLE_PYTHON_MPI must be ON, OFF or AUTO; got '${EAGLE_PYTHON_MPI}'")
endif()

# The install step rewrites the installed extensions (rpaths, install names), which invalidates their code signatures,
# and Apple silicon kills a process that maps an invalid one: re-sign (ad hoc) and verify every installed extension.
if(APPLE)
  install(CODE [[
    file(GLOB _exts "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/eagle/*.so")
    foreach(_f IN LISTS _exts)
      execute_process(COMMAND codesign --force --sign - "${_f}" COMMAND_ERROR_IS_FATAL ANY)
      execute_process(COMMAND codesign --verify --strict "${_f}" COMMAND_ERROR_IS_FATAL ANY)
    endforeach()
  ]])
endif()
