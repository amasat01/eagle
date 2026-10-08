// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"
#include <deque>
#include <list>

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/** @brief Manage multiple vs single dependencies */
template<typename IdxT>
struct IsMultipleDependency {
    static constexpr bool value = false;
};
template<typename T>
struct IsMultipleDependency<std::vector<T>> {
    static constexpr bool value = true;
};
template<typename T>
struct IsMultipleDependency<std::initializer_list<T>> {
    static constexpr bool value = true;
};
template<typename T>
struct IsMultipleDependency<std::list<T>> {
    static constexpr bool value = true;
};
template<typename T>
struct IsMultipleDependency<std::deque<T>> {
    static constexpr bool value = true;
};

} // namespace cuda
} // namespace eagle

#endif
