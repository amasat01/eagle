# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# The 2-rank MPI bed for `eagle::exec::RankPartition` (MECHANISM only — no
# performance claims, no CUDA-aware MPI).
#
# WHY A SEPARATE BINARY. `eagle_tests` is run bare, once, by tests/check_gate.sh;
# every row here needs a 2-rank `mpirun` world around it, and a row that silently
# passes at world size 1 would certify nothing. So the bed is its own target with
# its own `main()` (MPI_Init before InitGoogleTest), its own name manifests
# (tests/expected_tests_mpi_{cpp,cuda}.txt) and its own gate
# (tests/mpi/check_mpi_bed.sh) — and the main suite's manifests stay untouched.
#
# EAGLE_MPI is defined PRIVATELY on this target only. Nothing else in the tree sees
# it, so `eagle/exec/RankPartition.h` (which `#error`s without it) is compiled here
# and nowhere else, and flipping -DEAGLE_MPI=ON in an existing build directory
# leaves every other object file untouched.

find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "> Found MPI ${MPI_CXX_VERSION} (${MPI_CXX_COMPILER})")

# The host rows compile and run in BOTH modes; the device rows are CUDA-mode only
# (their inner structure is eagle::exec::DeviceKernel, which is Driver-API code).
set(EAGLE_MPI_BED_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main_mpi_bed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_MpiBed.cpp)
if (NOT EAGLE_CPP_MODE)
  list(APPEND EAGLE_MPI_BED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_MpiBedDevice.cu)
endif()

add_executable(eagle_mpi_bed ${EAGLE_MPI_BED_SOURCES})
target_link_libraries(eagle_mpi_bed eagle GTest::gtest MPI::MPI_CXX ${CMAKE_DL_LIBS})
target_include_directories(eagle_mpi_bed PRIVATE ../includes ../../)
target_compile_definitions(eagle_mpi_bed PRIVATE EAGLE_MPI=1)
set_target_properties(eagle_mpi_bed PROPERTIES POSITION_INDEPENDENT_CODE ON)

# main_mpi_bed.cpp and test_MpiBed.cpp are always host-compiled (CXX, not nvcc),
# even under the CUDA-mode gate build, and test_MpiBed.cpp reaches
# eagle/typedefs.h -> <aether/aether.h> through eagle/exec/HostTeam.h. Same forced
# define, and for exactly the same reason, as tests/test_ExecContract.cpp.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/main_mpi_bed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_MpiBed.cpp
  PROPERTIES COMPILE_DEFINITIONS "EAGLE_CPU_ONLY=1")

# The aether-abi/2 HOST fixture pair (tests/CMakeLists.txt builds it in BOTH modes)
# — the SAME artifact the main suite's rows use, never a bed-private copy.
add_dependencies(eagle_mpi_bed host_plugin_execv2)
target_compile_definitions(eagle_mpi_bed PRIVATE
  HOST_PLUGIN_EXECV2_SO="$<TARGET_FILE:host_plugin_execv2>")

# The audit row `.DeviceInnerStagesThroughAether` scans the execution
# structures' own sources, so it needs their directory as an absolute path.
target_compile_definitions(eagle_mpi_bed PRIVATE
  EAGLE_EXEC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../eagle/exec")

if (NOT EAGLE_CPP_MODE)
  # Driver API (cuLaunchKernel / cuModuleLoadData) — the same stub the main test
  # target links, resolved once by tests/CMakeLists.txt.
  target_link_libraries(eagle_mpi_bed ${EAGLE_TESTS_CUDA_DRIVER_LIB})
  # The v2 DEVICE fixture, nvcc'd to raw PTX by tests/CMakeLists.txt.
  add_dependencies(eagle_mpi_bed device_plugin_execv2_ptx)
  target_compile_definitions(eagle_mpi_bed PRIVATE
    DEVICE_PLUGIN_EXECV2_PTX="${EAGLE_DEVICE_PLUGIN_EXECV2_PTX}")
endif()
