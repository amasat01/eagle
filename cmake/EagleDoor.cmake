# SPDX-License-Identifier: Apache-2.0
#
# libeagle_door.so: eagle's C door (eagle/door), exported as eagle::door.
#
#   EAGLE_BUILD_DOOR=ON                      build and install it
#   EAGLE_DOOR_WING_FACTS=<raptor_facts.h>   the cpu wing's generated facts header (aether's linalg
#                                            needs it first); required, no default
#   EAGLE_DOOR_CUDA_WING_FACTS=<raptor_facts.h>
#                                            the cuda wing's; required when eagle is in CUDA mode
#   EAGLE_DOOR_ROWS=<eagle_door_rows.h>      the generated row tables (weights, gemm routes), force-included into
#                                            Context.cpp only; required, no default: the host_device header for a
#                                            CUDA door, the host one for a host-only door (raptor's rs/wings/cards/door/)
#
# Host group always. In CUDA mode the device group is built in (device/Device.cpp without
# EAGLE_CPU_ONLY, device/Loop.cu, linked to the CUDA runtime); in C++ mode Device.cpp compiles
# with EAGLE_CPU_ONLY and refuses at run time. The library exports eg_* names only (door.map).

set(EAGLE_DOOR_WING_FACTS "" CACHE FILEPATH "The cpu wing's raptor_facts.h, included first by the door's host sources")
set(EAGLE_DOOR_CUDA_WING_FACTS "" CACHE FILEPATH "The cuda wing's raptor_facts.h, included first by the door's device loop (CUDA mode)")

set(EAGLE_DOOR_ROWS "" CACHE FILEPATH "The generated row header (EG_WEIGHT_ROWS, EG_GEMM_ROUTE_ROWS), included into Context.cpp")

if(NOT EAGLE_DOOR_ROWS OR NOT EXISTS "${EAGLE_DOOR_ROWS}")
  message(FATAL_ERROR "EAGLE_BUILD_DOOR: set EAGLE_DOOR_ROWS to the generated eagle_door_rows.h of this build (host_device for a CUDA door, host for a host-only door)")
endif()
if(NOT EAGLE_DOOR_WING_FACTS OR NOT EXISTS "${EAGLE_DOOR_WING_FACTS}")
  message(FATAL_ERROR "EAGLE_BUILD_DOOR: set EAGLE_DOOR_WING_FACTS to the cpu wing's raptor_facts.h")
endif()
if(NOT EAGLE_CPP_MODE AND (NOT EAGLE_DOOR_CUDA_WING_FACTS OR NOT EXISTS "${EAGLE_DOOR_CUDA_WING_FACTS}"))
  message(FATAL_ERROR "EAGLE_BUILD_DOOR in CUDA mode: set EAGLE_DOOR_CUDA_WING_FACTS to the cuda wing's raptor_facts.h")
endif()

set(_door "${CMAKE_CURRENT_SOURCE_DIR}/eagle/door")
# The sources are listed once, in sources.txt, which the Makefile reads too: "cuda <path>" lines only in CUDA mode.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_door}/sources.txt")
file(STRINGS "${_door}/sources.txt" _door_lines REGEX "^[^#]")
set(_door_src "")
set(_door_cuda_src "")
foreach(_line IN LISTS _door_lines)
  if(_line MATCHES "^cuda (.+)$")
    list(APPEND _door_cuda_src "${_door}/${CMAKE_MATCH_1}")
  else()
    list(APPEND _door_src "${_door}/${_line}")
  endif()
endforeach()

find_package(Threads REQUIRED)
add_library(eagle_door SHARED ${_door_src})
add_library(eagle::door ALIAS eagle_door)
set_target_properties(eagle_door PROPERTIES
  OUTPUT_NAME eagle_door EXPORT_NAME door
  CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties("${_door}/Context.cpp" PROPERTIES COMPILE_OPTIONS "-include;${EAGLE_DOOR_ROWS}" OBJECT_DEPENDS "${EAGLE_DOOR_ROWS}")
set(_door_warn -Wall -Wextra -pedantic -Werror)
# aether contributes its headers only: its package's mode defines are not the door's.
target_include_directories(eagle_door
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" $<TARGET_PROPERTY:aether::aether,INTERFACE_INCLUDE_DIRECTORIES>
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(eagle_door PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:AETHER_CPP_MODE=1>")
target_compile_options(eagle_door PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O2;-ffp-contract=off;${_door_warn};-fopenmp;-include;${EAGLE_DOOR_WING_FACTS}>")
target_link_options(eagle_door PRIVATE
  -Wl,--version-script=${_door}/door.map -Wl,--no-undefined -fopenmp)
target_link_libraries(eagle_door PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_property(TARGET eagle_door APPEND PROPERTY LINK_DEPENDS "${_door}/door.map")

# Every host source is EAGLE_CPU_ONLY; only the device group (Device.cpp) drops it in CUDA mode.
if(EAGLE_CPP_MODE)
  target_compile_definitions(eagle_door PRIVATE EAGLE_CPU_ONLY=1)
else()
  set(_cuda_cpp ${_door_cuda_src})
  list(FILTER _cuda_cpp INCLUDE REGEX "\\.cpp$")
  set(_host_only ${_door_src} ${_cuda_cpp})
  list(REMOVE_ITEM _host_only "${_door}/device/Device.cpp")
  target_sources(eagle_door PRIVATE ${_cuda_cpp})
  set_source_files_properties(${_host_only} PROPERTIES COMPILE_DEFINITIONS EAGLE_CPU_ONLY=1)
  find_package(CUDAToolkit REQUIRED)
  target_sources(eagle_door PRIVATE "${_door}/device/Loop.cu")
  target_link_libraries(eagle_door PRIVATE CUDA::cudart)
  set_source_files_properties("${_door}/device/Loop.cu" PROPERTIES
    COMPILE_OPTIONS "-Wno-deprecated-gpu-targets;-Werror=all-warnings;-diag-suppress=20050;-include;${EAGLE_DOOR_CUDA_WING_FACTS};-Xcompiler=-fPIC,-fvisibility=hidden,-Wall,-Wextra,-fopenmp"
    COMPILE_DEFINITIONS AETHER_HAS_CUDA)
  # The loop is the one CUDA unit: C++20 as the old path's device recipe.
  set_target_properties(eagle_door PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
endif()

install(TARGETS eagle_door EXPORT ${PROJECT_NAME}_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
