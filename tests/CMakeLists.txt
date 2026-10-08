# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)
if (EAGLE_CPP_MODE)
  project(eagle_tests LANGUAGES CXX)
else()
  project(eagle_tests LANGUAGES CUDA CXX)
endif()

# Supress a warning in newer CMake versions when using FetchContent
if (POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

# Download GoogleTest
include(FetchContent)
message(STATUS "> Downloading dependency: GoogleTest")
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)
message(STATUS "> Downloading dependency: GoogleTest - done")

# NOTE: `include(GoogleTest)` (the CMake module providing gtest_discover_tests)
# was REMOVED with the ctest wiring below — see the comment at the former
# `gtest_discover_tests` site. FetchContent above is what supplies the googletest
# LIBRARY and is unaffected.

if (EAGLE_TEST_REPORT)
  list(APPEND EAGLE_GTEST_FLAGS "--gtest_output=xml:eagle_tests.xml")
endif()

if (EAGLE_CPP_MODE)
  # C++ TESTING - .cpp TEST FILES
  file(GLOB TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
  )
else()
  # CUDA TESTING - .cu TEST FILES
  file(GLOB TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu
  )
endif()

# The shared conformance corpus driver must be
# compiled and run by BOTH build modes, so it is added EXPLICITLY here rather than left
# to the mode-conditional glob above — which takes `.cpp` only under EAGLE_CPP_MODE and
# `.cu` only otherwise. That asymmetry is exactly how 105 `.cu` tests went unrun for
# four stages; a `.cpp` corpus driver would have been invisible to the CUDA gate in the
# same way. It contains no CUDA, so CXX compiles it in either mode. REMOVE_DUPLICATES
# keeps it single when the CPP_MODE glob has already matched it.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_ConformanceCorpus.cpp)
list(REMOVE_DUPLICATES TEST_FILES)

# test_DeviceProps.cpp exercises
# eagle/DeviceProps.h + eagle/cpu/DeviceProps.h, both CUDA-free by design
# (mirroring eagle/launch/Traits.h) -- same reasoning as the conformance
# corpus append above, it must compile+run in BOTH build modes. Unlike
# test_LaunchTraits.cpp/test_ConditionalGroupCpu.cpp/
# test_HostDispatchContract.cpp below, it needs NO forced
# EAGLE_CPU_ONLY define: it never reaches eagle/typedefs.h ->
# <aether/aether.h>, so there is no DEVICE()/HOST() macro to guard against. The
# CUDA-hardware half (eagle::cuda::deviceProps against the real device) is
# test_DeviceProps.cu, picked up by the CUDA-mode glob only.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_DeviceProps.cpp)
list(REMOVE_DUPLICATES TEST_FILES)

# eagle::launch::Traits unit test is CUDA-free,
# so -- same reasoning as the conformance corpus append above -- it must
# compile+run in BOTH build modes rather than being left to the
# mode-conditional glob. It is a plain .cpp reaching eagle/typedefs.h ->
# <aether/aether.h>; aether's AETHER_DEVICE()/AETHER_HOST() macros already
# expand to nothing for a host-compiler TU of a CUDA-mode build (the
# AETHER_DEVICE_COMPILER axis -- no per-source AETHER_CPP_MODE escape needed,
# unlike a per-source CPU-only escape hatch). EAGLE_CPU_ONLY is still forced on
# this ONE source to select eagle's own host-face code path so it compiles
# under CXX (g++) in the default CUDA-mode gate too, independent of the
# ambient mode -- Traits.h itself is unconditionally CUDA-free either way.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_LaunchTraits.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_LaunchTraits.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# Observer registration-consistency rows (the move-assign fix) are
# CUDA-free and must compile+run in BOTH build modes -- same reasoning and
# same recipe, verbatim, as test_LaunchTraits.cpp above: Observer.h reaches
# eagle/typedefs.h -> <aether/aether.h>, and this .cpp is compiled by CXX
# (g++) even under the CUDA-mode gate build.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_ObserverRegistration.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_ObserverRegistration.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# The ConditionalGroup CPU parity twin (G-CPU) must compile+run in
# BOTH build modes -- same reasoning as test_LaunchTraits.cpp above, and the
# same recipe, verbatim: eagle/conditional.h reaches eagle/cuda.h, whose
# __device__/__host__-qualified bodies only nvcc parses, but this .cpp is
# compiled by CXX (g++) even under the CUDA-mode gate build. Force
# EAGLE_CPU_ONLY on this ONE source so it compiles under g++ independent of
# the ambient EAGLE_CPP_MODE (aether's own macros need no such forcing --
# see the test_LaunchTraits.cpp comment above).
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_ConditionalGroupCpu.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_ConditionalGroupCpu.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# The GraphComposer CPU (reference) face parity twin
# must compile+run in BOTH build modes -- same reasoning and same recipe,
# verbatim, as test_ConditionalGroupCpu.cpp above: eagle/compose/
# GraphComposer.h reaches eagle/cuda.h, whose __device__/__host__-qualified
# bodies only nvcc parses, but this .cpp is compiled by CXX (g++) even under
# the CUDA-mode gate build. Force EAGLE_CPU_ONLY on this ONE source so it
# compiles under g++ independent of the ambient EAGLE_CPP_MODE. The CUDA
# face's own test (test_GraphComposer.cu) needs no such treatment -- it is
# already picked up by the CUDA-mode-only glob above.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_GraphComposerCpu.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_GraphComposerCpu.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# eagle::cpu::Host::launch's region-required
# work-sharing contract-guard test is CUDA-free, same reasoning as
# test_LaunchTraits.cpp / test_ConditionalGroupCpu.cpp above -- it must
# compile+run in BOTH build modes, and needs the same forced EAGLE_CPU_ONLY
# define since it reaches eagle/typedefs.h -> <aether/aether.h> and is
# always host-compiled (CXX, not nvcc) even under the default CUDA-mode gate.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_HostDispatchContract.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_HostDispatchContract.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# The packet arm (packetLoad/packetStore/loadMask/
# applyTail) RED-arm test for that fix is CUDA-free -- same reasoning as
# test_HostDispatchContract.cpp above (same header, eagle/cpu/Host.h) --
# so it must compile+run in BOTH build modes with the same forced defines,
# independent of the ambient EAGLE_CPP_MODE.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_HostPacketArm.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_HostPacketArm.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# The heterogeneous execution contract's host and
# CUDA-free rows -- the loader gates (both ABI tags, the schema ceiling, the flat
# execution axis, the layout self-check), the placement rules and
# eagle::exec::HostTeam. Same reasoning and the same recipe, verbatim, as
# test_HostDispatchContract.cpp above: it reaches eagle/typedefs.h ->
# <aether/aether.h> through eagle/exec/HostTeam.h and is always host-compiled
# (CXX, not nvcc) even under the default CUDA-mode gate, so it must compile+run in
# BOTH build modes with EAGLE_CPU_ONLY forced on this ONE source. The device face
# is test_ExecContractDevice.cu, picked up by the CUDA-mode glob.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_ExecContract.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_ExecContract.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# The host arm of the generic DLPack layer (plugin/interop.h) and the plugin
# bridge are CUDA-free -- same reasoning and same recipe as
# test_ExecContract.cpp above: compiled by CXX in BOTH build modes with
# EAGLE_CPU_ONLY forced on this ONE source. The device stream contract is
# test_DLPackStream.cu, picked up by the CUDA-mode glob.
list(APPEND TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_DLPackInterop.cpp)
list(REMOVE_DUPLICATES TEST_FILES)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_DLPackInterop.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

list(SORT TEST_FILES COMPARE NATURAL)

foreach(file ${TEST_FILES})
    get_filename_component(filename ${file} NAME_WE)
    message(STATUS "> Found test source " ${file})
endforeach()

list(LENGTH TEST_FILES TEST_FILES_LEN)
message(STATUS "> Found " ${TEST_FILES_LEN} " test sources")

add_executable(eagle_tests ${TEST_FILES})
target_link_libraries(eagle_tests eagle GTest::gtest_main)
target_include_directories(eagle_tests PRIVATE includes ../)

# The language-neutral conformance corpus both drivers read (the pytest driver
# resolves the same directory by relative path). Absolute, so the test binary is
# runnable from any working directory.
target_compile_definitions(eagle_tests PRIVATE
  EAGLE_CONFORMANCE_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/conformance")

# dlopen/dlsym for the host registry. Needed in BOTH modes since the conformance
# driver exercises `eagle::cpu::PluginRegistry::add_plugin` (the manifest-bypassing
# entry point) in each; the CPP-mode block below no longer owns this link.
target_link_libraries(eagle_tests ${CMAKE_DL_LIBS})

# ── ctest is deliberately NOT wired ───────────────────────────────────────────
# `gtest_discover_tests(eagle_tests)` and the matching subdirectory
# `enable_testing()` were REMOVED. They registered tests only in
# `<build>/tests/CTestTestfile.cmake`; CMake emits the ROOT CTestTestfile.cmake —
# the one carrying the `subdirs("tests")` link — only when testing is enabled at
# the TOP level, which it never was. So `ctest --test-dir build` reported
# "Total Tests: 0" and exited 0: a gate that passes having run nothing. Measured
# observed the same day: root 0, `<build>/tests` 195.
#
# Repairing the wiring was rejected in favour of deleting it. Nothing here ever
# used ctest — this repo's CI and every documented gate already invoke
# `build/tests/eagle_tests` directly, the suites are 100% gtest (so ctest's
# framework-neutrality buys nothing), and its one real advantage, `-j` parallelism,
# is FORBIDDEN for these GPU suites anyway. A broken-but-present ctest is the worst
# option: the vacuous command stays runnable and stays green.
#
# What ctest did provide and we consciously gave up: per-test process isolation
# (a segfaulting test no longer spares its siblings) and per-test timeouts. The
# replacement guarantee lives in CI, which pins the RAN COUNT — because the
# binary has the identical failure mode on its own: `--gtest_filter=NoSuchTest`
# prints "[  PASSED  ] 0 tests." and exits 0. The runner was never the defect;
# the missing count assertion was.

# Position-independent code is needed for PIE-default linkers (the
# conda-baked toolchain used downstream makes everything PIE by
# default, and CUDA TUs without -fPIC produce R_X86_64_32 relocations
# that fail to link). Set on the test target to match a downstream consumer's
# tests, which use -Xcompiler=-fPIE for the same reason.
set_target_properties(eagle_tests PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(EAGLE_CPP_MODE)
  target_compile_options(eagle_tests
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -ftemplate-backtrace-limit=100>)
else()
  target_compile_options(eagle_tests
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra -O3 -ftemplate-backtrace-limit=100>)

  # test_PluginRegistryDtype.cu exercises plugin/plugin_registry/registry.h
  # (namespace eagle::cuda), which is a CUDA Driver-API consumer — same
  # driver stub link recipe as plugin/graph_inject/CMakeLists.txt.
  if(DEFINED ENV{CUDA_PATH})
    set(_EAGLE_TESTS_CUDA_ROOT $ENV{CUDA_PATH})
  else()
    set(_EAGLE_TESTS_CUDA_ROOT /usr/local/cuda)
  endif()
  find_library(EAGLE_TESTS_CUDA_DRIVER_LIB cuda
    HINTS ${_EAGLE_TESTS_CUDA_ROOT}/targets/x86_64-linux/lib/stubs
          ${_EAGLE_TESTS_CUDA_ROOT}/lib64/stubs
          ${_EAGLE_TESTS_CUDA_ROOT}/lib/stubs)
  if(NOT EAGLE_TESTS_CUDA_DRIVER_LIB)
    message(FATAL_ERROR "Could not locate the libcuda stub under ${_EAGLE_TESTS_CUDA_ROOT}")
  endif()
  target_link_libraries(eagle_tests ${EAGLE_TESTS_CUDA_DRIVER_LIB})

  # Device-plugin fixture for test_PluginRegistryDtype.cu's
  # SoftdoubleLoadsAndLaunches: a hand-written, ABI-only .cu (no external
  # code generator -- see fixtures/device_plugin_softdouble.cu) nvcc'd to PTX once at
  # build time, mirroring the CPU host_plugin_* fixtures' role for the device
  # path. The test reads the compiled PTX from this absolute path (baked in
  # via a compile definition) and stages it into its own manifest.
  #
  # Deliberately NOT ${CMAKE_CUDA_COMPILER}: this step emits raw PTX TEXT that
  # PluginRegistry loads at runtime via cuModuleLoadData (a driver-side PTX
  # JIT capped at the installed driver's max supported PTX ISA), unlike the
  # rest of this target's .cu sources, which nvcc compiles straight to
  # embedded SASS for CMAKE_CUDA_ARCHITECTURES and so never hit that ceiling.
  # A newer nvcc than the driver supports emits a newer PTX ISA that
  # cuModuleLoadData then rejects — so this resolves nvcc the same way
  # the same toolchain resolution used elsewhere (PATH first, so the caller's env picks the toolchain;
  # CUDA_PATH as the fallback), independently of whichever nvcc CMake cached
  # as the project's CUDA_COMPILER.
  # A manual PATH walk, not find_program(): find_program also consults
  # CMAKE_PREFIX_PATH (populated here from a conda env baked into the shell
  # profile, ahead of a HINTS env-var lookup in practice) and can resolve to
  # a DIFFERENT nvcc than the one first on $PATH — silently defeating the
  # "PATH picks the toolchain" contract this fixture depends on. Mirrors
  # the same resolution convention (PATH first, $CUDA_PATH/bin fallback)
  # exactly, and re-resolves on every configure (a plain variable, not a
  # CACHE one) so a changed PATH is always honored.
  unset(_SD_FIXTURE_NVCC CACHE)
  set(_SD_FIXTURE_NVCC "")
  if(DEFINED ENV{PATH})
    string(REPLACE ":" ";" _sd_path_list "$ENV{PATH}")
    foreach(_sd_dir ${_sd_path_list})
      if(NOT _SD_FIXTURE_NVCC AND EXISTS "${_sd_dir}/nvcc")
        set(_SD_FIXTURE_NVCC "${_sd_dir}/nvcc")
      endif()
    endforeach()
  endif()
  if(NOT _SD_FIXTURE_NVCC AND DEFINED ENV{CUDA_PATH}
     AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc")
    set(_SD_FIXTURE_NVCC "$ENV{CUDA_PATH}/bin/nvcc")
  endif()
  if(NOT _SD_FIXTURE_NVCC)
    message(FATAL_ERROR "Could not locate nvcc (checked $PATH and $CUDA_PATH/bin) "
      "for the device_plugin_softdouble.cu PTX fixture")
  endif()
  set(_SD_FIXTURE_PTX ${CMAKE_CURRENT_BINARY_DIR}/device_plugin_softdouble.ptx)
  add_custom_command(
    OUTPUT ${_SD_FIXTURE_PTX}
    COMMAND ${_SD_FIXTURE_NVCC} --ptx
            -I${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/device_plugin_softdouble.cu
            -o ${_SD_FIXTURE_PTX}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/device_plugin_softdouble.cu
    COMMENT "Compiling device_plugin_softdouble.cu -> PTX (${_SD_FIXTURE_NVCC})")
  add_custom_target(device_plugin_softdouble_ptx DEPENDS ${_SD_FIXTURE_PTX})
  add_dependencies(eagle_tests device_plugin_softdouble_ptx)
  target_compile_definitions(eagle_tests PRIVATE
    DEVICE_PLUGIN_SOFTDOUBLE_PTX="${_SD_FIXTURE_PTX}")

  # Int-uniform device parity: the second PTX fixture, compiled by the SAME
  # resolved nvcc and for the same reason (raw PTX TEXT that cuModuleLoadData JITs, so
  # the driver's max supported PTX ISA is the ceiling — see the long note above).
  # fixtures/device_plugin_intuniform.cu declares `(ScalarHandle, double, long long,
  # uint32)`, the MIXED float64 + int64 uniform row set that
  # test_PluginRegistryUniformInt.cu launches.
  set(_IU_FIXTURE_PTX ${CMAKE_CURRENT_BINARY_DIR}/device_plugin_intuniform.ptx)
  add_custom_command(
    OUTPUT ${_IU_FIXTURE_PTX}
    COMMAND ${_SD_FIXTURE_NVCC} --ptx
            -I${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/device_plugin_intuniform.cu
            -o ${_IU_FIXTURE_PTX}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/device_plugin_intuniform.cu
    COMMENT "Compiling device_plugin_intuniform.cu -> PTX (${_SD_FIXTURE_NVCC})")
  add_custom_target(device_plugin_intuniform_ptx DEPENDS ${_IU_FIXTURE_PTX})
  add_dependencies(eagle_tests device_plugin_intuniform_ptx)
  target_compile_definitions(eagle_tests PRIVATE
    DEVICE_PLUGIN_INTUNIFORM_PTX="${_IU_FIXTURE_PTX}")

  # the aether-abi/2 DEVICE fixture PAIR --
  # sample_local / mapreduce / cross_sample_write bodies, the wide-role and triple
  # probes and a legacy v1 kernel, all in one module, plus the wrong-sizes arm of
  # the layout self-check. Compiled by the SAME resolved nvcc and for the same
  # reason as the two fixtures above (raw PTX TEXT that cuModuleLoadData JITs, so
  # the driver's max supported PTX ISA is the ceiling).
  #
  # -arch=sm_60 (or the oldest the toolkit offers) is REQUIRED here and not above: the cross_sample_write body uses
  # the float64 atomicAdd, which is a Pascal-and-later intrinsic. PTX emitted for
  # sm_60 JITs forward onto every architecture this project targets.
  # sm_60 where the toolkit still compiles it; a newer toolkit drops the oldest
  # targets (CUDA 13 starts at sm_75), so take the oldest it lists, floored at 60.
  set(_EC_FIXTURE_ARCH 60)
  execute_process(COMMAND ${_SD_FIXTURE_NVCC} --list-gpu-arch
                  OUTPUT_VARIABLE _ec_arch_list RESULT_VARIABLE _ec_arch_rc ERROR_QUIET)
  if(_ec_arch_rc EQUAL 0)
    string(REGEX MATCHALL "compute_[0-9]+" _ec_arch_found "${_ec_arch_list}")
    set(_ec_arch_min "")
    foreach(_c IN LISTS _ec_arch_found)
      string(REPLACE "compute_" "" _c "${_c}")
      if(_ec_arch_min STREQUAL "" OR _c LESS _ec_arch_min)
        set(_ec_arch_min ${_c})
      endif()
    endforeach()
    if(NOT _ec_arch_min STREQUAL "" AND _ec_arch_min GREATER 60)
      set(_EC_FIXTURE_ARCH ${_ec_arch_min})
    endif()
  endif()
  foreach(_ec_fixture device_plugin_execv2 device_plugin_badlayout)
    set(_EC_PTX ${CMAKE_CURRENT_BINARY_DIR}/${_ec_fixture}.ptx)
    add_custom_command(
      OUTPUT ${_EC_PTX}
      COMMAND ${_SD_FIXTURE_NVCC} --ptx -arch=sm_${_EC_FIXTURE_ARCH}
              -I${CMAKE_CURRENT_SOURCE_DIR}/..
              ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_ec_fixture}.cu
              -o ${_EC_PTX}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_ec_fixture}.cu
      COMMENT "Compiling ${_ec_fixture}.cu -> PTX (${_SD_FIXTURE_NVCC})")
    add_custom_target(${_ec_fixture}_ptx DEPENDS ${_EC_PTX})
    add_dependencies(eagle_tests ${_ec_fixture}_ptx)
  endforeach()
  # Named so tests/mpi/CMakeLists.txt (added at the END of this file, and so
  # inheriting this scope) can hand the SAME artifact to the MPI bed rather than
  # re-deriving a path relative to its own binary dir.
  set(EAGLE_DEVICE_PLUGIN_EXECV2_PTX
    "${CMAKE_CURRENT_BINARY_DIR}/device_plugin_execv2.ptx")
  target_compile_definitions(eagle_tests PRIVATE
    DEVICE_PLUGIN_EXECV2_PTX="${EAGLE_DEVICE_PLUGIN_EXECV2_PTX}"
    DEVICE_PLUGIN_BADLAYOUT_PTX="${CMAKE_CURRENT_BINARY_DIR}/device_plugin_badlayout.ptx")

  # A standalone helper process, NOT part of
  # eagle_tests' own sources (so it never inflates the gated test count by
  # itself) -- see helper_stream_nothrow_reset.cu for why the Stream teardown-
  # under-poison proof needs total process isolation (a cudaDeviceReset() that
  # would corrupt every other test sharing this binary's process). Its path is
  # injected into eagle_tests, which spawns it via popen() and checks its exit
  # status from DeviceErrorNothrowTest.StreamTeardownSurvivesPoisonedHandleIsolated.
  add_executable(eagle_probe_stream_nothrow
    ${CMAKE_CURRENT_SOURCE_DIR}/helper_stream_nothrow_reset.cu)
  target_link_libraries(eagle_probe_stream_nothrow eagle)
  target_include_directories(eagle_probe_stream_nothrow PRIVATE includes ../)
  set_target_properties(eagle_probe_stream_nothrow PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  add_dependencies(eagle_tests eagle_probe_stream_nothrow)
  target_compile_definitions(eagle_tests PRIVATE
    EAGLE_PROBE_STREAM_NOTHROW_BIN="$<TARGET_FILE:eagle_probe_stream_nothrow>")
endif()

# ---------------------------------------------------------------------------
# the aether-abi/2 HOST fixture trio.
#
# Built in BOTH build modes, unlike the v1 host fixtures below: the rows that
# dlopen them live in test_ExecContract.cpp (both gates) AND in
# test_ExecContractDevice.cu (the host/device twin needs both arms in one
# process), so a CPP_MODE-only guard here would leave the CUDA gate unable to
# link. They are ABI-only `.cpp` files with no eagle/aether/CUDA in them, so CXX
# builds them identically in either mode.
#
# host_plugin_execv2      the v2 bodies + the correct `eagle_layout_sizes`
# host_plugin_badlayout   a v2 plugin whose exported sizes DISAGREE (RED arm)
# host_plugin_nolayout    a v2 plugin exporting no layout symbol at all (RED arm)
# ---------------------------------------------------------------------------
find_package(OpenMP)
foreach(_ec_so host_plugin_execv2 host_plugin_badlayout host_plugin_nolayout)
  add_library(${_ec_so} MODULE ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_ec_so}.cpp)
  set_target_properties(${_ec_so} PROPERTIES
    POSITION_INDEPENDENT_CODE ON PREFIX "")
  target_include_directories(${_ec_so} PRIVATE ../)
  add_dependencies(eagle_tests ${_ec_so})
endforeach()
target_compile_definitions(eagle_tests PRIVATE
  HOST_PLUGIN_EXECV2_SO="$<TARGET_FILE:host_plugin_execv2>"
  HOST_PLUGIN_BADLAYOUT_SO="$<TARGET_FILE:host_plugin_badlayout>"
  HOST_PLUGIN_NOLAYOUT_SO="$<TARGET_FILE:host_plugin_nolayout>")

# ---------------------------------------------------------------------------
# A CPU-plugin fixture: a hand-written host plugin `.so` (ABI-only, no eagle /
# CUDA) that test_HostPlugin.cpp dlopen's. Built only in CPP mode (the
# host graph path); its absolute path is injected into the test binary. Mirrors
# a generated plugin: a shared object exporting `<kernel>_host`.
# ---------------------------------------------------------------------------
if(EAGLE_CPP_MODE)
  add_library(host_plugin_addvec MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/host_plugin_addvec.cpp)
  set_target_properties(host_plugin_addvec PROPERTIES
    POSITION_INDEPENDENT_CODE ON PREFIX "")
  target_include_directories(host_plugin_addvec PRIVATE ../)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(host_plugin_addvec PRIVATE OpenMP::OpenMP_CXX)
  endif()
  add_dependencies(eagle_tests host_plugin_addvec)
  target_compile_definitions(eagle_tests PRIVATE
    HOST_PLUGIN_ADDVEC_SO="$<TARGET_FILE:host_plugin_addvec>")

  # Matrix host parity: a hand-written matrix-input plugin `.so`
  # (ABI-only) that test_HostPlugin.cpp dlopen's to prove the registry packs a
  # `mat_in` GRef mirror. Same recipe as host_plugin_addvec.
  add_library(host_plugin_mattrace MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/host_plugin_mattrace.cpp)
  set_target_properties(host_plugin_mattrace PROPERTIES
    POSITION_INDEPENDENT_CODE ON PREFIX "")
  target_include_directories(host_plugin_mattrace PRIVATE ../)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(host_plugin_mattrace PRIVATE OpenMP::OpenMP_CXX)
  endif()
  add_dependencies(eagle_tests host_plugin_mattrace)
  target_compile_definitions(eagle_tests PRIVATE
    HOST_PLUGIN_MATTRACE_SO="$<TARGET_FILE:host_plugin_mattrace>")

  # Lookup-table host parity: a hand-written lookup plugin `.so` (ABI-only)
  # that test_HostPlugin.cpp dlopen's to prove the registry consolidates + packs a
  # `lookup` table by value. Same recipe as host_plugin_addvec.
  add_library(host_plugin_lookup MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/host_plugin_lookup.cpp)
  set_target_properties(host_plugin_lookup PROPERTIES
    POSITION_INDEPENDENT_CODE ON PREFIX "")
  target_include_directories(host_plugin_lookup PRIVATE ../)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(host_plugin_lookup PRIVATE OpenMP::OpenMP_CXX)
  endif()
  add_dependencies(eagle_tests host_plugin_lookup)
  target_compile_definitions(eagle_tests PRIVATE
    HOST_PLUGIN_LOOKUP_SO="$<TARGET_FILE:host_plugin_lookup>")

  # Int-uniform host parity: a hand-written plugin `.so` (ABI-only) exporting two
  # entries — an int64-uniform kernel and a MIXED (float64 + int64 uniform) one — that
  # test_HostPluginUniformInt.cpp dlopen's to prove the registry packs an integer
  # uniform as the 8 signed bytes a generated kernel's `GRID_CONSTANT() Int p_<name>`
  # slot expects. Same recipe as host_plugin_addvec.
  add_library(host_plugin_intuniform MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/host_plugin_intuniform.cpp)
  set_target_properties(host_plugin_intuniform PROPERTIES
    POSITION_INDEPENDENT_CODE ON PREFIX "")
  target_include_directories(host_plugin_intuniform PRIVATE ../)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(host_plugin_intuniform PRIVATE OpenMP::OpenMP_CXX)
  endif()
  add_dependencies(eagle_tests host_plugin_intuniform)
  target_compile_definitions(eagle_tests PRIVATE
    HOST_PLUGIN_INTUNIFORM_SO="$<TARGET_FILE:host_plugin_intuniform>")

  # A performance assessment -- a standalone bench (NOT part of eagle_tests, so
  # it never inflates the gated test count). Run on demand:
  #   ./build_cpp/tests/eagle_bench_host_graph
  add_executable(eagle_bench_host_graph
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_host_graph.cpp)
  target_link_libraries(eagle_bench_host_graph eagle ${CMAKE_DL_LIBS})
  target_include_directories(eagle_bench_host_graph PRIVATE includes ../)
  target_compile_options(eagle_bench_host_graph PRIVATE -O3)
  add_dependencies(eagle_bench_host_graph host_plugin_addvec)
  target_compile_definitions(eagle_bench_host_graph PRIVATE
    HOST_PLUGIN_ADDVEC_SO="$<TARGET_FILE:host_plugin_addvec>")
  if(OpenMP_CXX_FOUND)
    target_link_libraries(eagle_bench_host_graph OpenMP::OpenMP_CXX)
  endif()
endif()

# ---------------------------------------------------------------------------
# EAGLE_CPU_ONLY compile probe for eagle/launch/Traits.h.
# Proves the header is
# genuinely CUDA-free by INSTANTIATING the template and evaluating
# minBlocksPerSM() in a constexpr context under an EXPLICIT -DEAGLE_CPU_ONLY
# on this one TU, independent of whichever EAGLE_CPP_MODE this configure
# uses (the default gate build here is CUDA mode, so this is the only place
# that flag gets exercised). OBJECT library: compile-only, no main() needed
# -- part of the default `all` target, so a compile failure fails
# `cmake --build`. Traits.h reaches eagle/typedefs.h -> <aether/aether.h>,
# whose AETHER_DEVICE()/AETHER_HOST() macros already expand to nothing for a
# host-compiler TU of a CUDA-mode build (no per-source AETHER_CPP_MODE
# needed here, unlike a per-source CPU-only escape hatch); EAGLE_CPU_ONLY alone
# is enough to select eagle's own host-face path.
# ---------------------------------------------------------------------------
add_library(eagle_probe_launch_traits_cpu_only OBJECT
  ${CMAKE_CURRENT_SOURCE_DIR}/probe_LaunchTraitsCpuOnly.cpp)
target_link_libraries(eagle_probe_launch_traits_cpu_only PRIVATE eagle)
target_compile_definitions(eagle_probe_launch_traits_cpu_only PRIVATE
  EAGLE_CPU_ONLY=1)
set_target_properties(eagle_probe_launch_traits_cpu_only PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# (the subdirectory `enable_testing()` was removed here with the rest of the ctest
# wiring — see the note at the former `gtest_discover_tests` site above, #102.)

# ---------------------------------------------------------------------------
# The 2-rank MPI bed.
#
# LAST in this file on purpose: `add_subdirectory` gives the child a COPY of this
# scope as it stands at the call, so tests/mpi/CMakeLists.txt sees the googletest
# targets FetchContent made available above, the CUDA driver stub this file
# resolved, and EAGLE_DEVICE_PLUGIN_EXECV2_PTX. Guarded by EAGLE_MPI (default OFF),
# so nothing here is configured -- and `find_package(MPI)` is never called -- unless
# the bed is explicitly asked for.
# ---------------------------------------------------------------------------
if (EAGLE_MPI)
  add_subdirectory(mpi)
endif()
