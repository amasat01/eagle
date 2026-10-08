# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Sanitize custom target: runs valgrind (+ compute-sanitizer when available)
# over the already-built `eagle_tests` binary via the python driver.
# Only registered when EAGLE_BUILD_SANITIZE=ON.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(sanitize
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_sanitize.py
            --binary $<TARGET_FILE:eagle_tests>
            --tool ${EAGLE_SANITIZE_TOOL}
            --suppressions ${CMAKE_CURRENT_SOURCE_DIR}/suppressions
            --log-dir ${CMAKE_BINARY_DIR}/sanitize
            --project-name EAGLE
            $<$<BOOL:${EAGLE_SANITIZE_MINIMAL}>:--minimal>
            $<$<BOOL:${EAGLE_SANITIZE_FAIL_ON_REACHABLE}>:--fail-on-reachable>
    DEPENDS eagle_tests
    USES_TERMINAL
    COMMENT "Running sanitize sweep on eagle_tests"
    VERBATIM
)
