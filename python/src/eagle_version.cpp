// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// The package version, in a translation unit of its own: a new version recompiles this file and nothing else.
// (EAGLE_PYTHON_VERSION is defined for this file only, by python/CMakeLists.txt.)
#ifndef EAGLE_PYTHON_VERSION
#define EAGLE_PYTHON_VERSION "unknown"
#endif

const char* eagle_python_version();

const char* eagle_python_version()
{
    return EAGLE_PYTHON_VERSION;
}
