// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Status.h
 * @brief ``eagle_backend_status`` -> Python exception, for the core only.
 *
 * | status            | raised                                   |
 * |-------------------|------------------------------------------|
 * | ERROR, BAD_HANDLE | ``RuntimeError`` (the backend's message)  |
 * | INVALID_ARGUMENT  | ``ValueError``                            |
 * | INDEX             | ``IndexError``                            |
 * | INTEROP           | ``eagle._core.InteropError``              |
 * | UNAVAILABLE       | ``eagle.BackendUnavailable``              |
 * | CALLBACK_FAILED   | the Python exception the callback raised  |
 * | anything else     | ``RuntimeError`` naming the value         |
 *
 * A Python callback the backend calls back into (a composer step, a pre-launch
 * hook) cannot raise THROUGH the backend: the core's trampoline stores the
 * exception here, returns nonzero, the backend unwinds its own stack and returns
 * CALLBACK_FAILED, and @ref check re-raises the stored exception.
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>

#include "Loader.h"
#include "eagle_backend.h"
#include "plugin/interop.h"

namespace eagle_seam {

namespace nb = nanobind;

/** @brief ``eagle.BackendUnavailable`` (defined in pure Python, ``eagle._backend``). */
inline nb::object unavailableType()
{
    return nb::module_::import_("eagle._backend").attr("BackendUnavailable");
}

/** @brief Raise ``eagle.BackendUnavailable(message)`` as a C++ exception. */
[[noreturn]] inline void raiseUnavailable(const std::string& message)
{
    nb::object type = unavailableType();
    PyErr_SetString(type.ptr(), message.c_str());
    throw nb::python_error();
}

/** @brief Translate @ref Unavailable to ``eagle.BackendUnavailable`` (call once, at module init). */
inline void registerTranslators()
{
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const Unavailable& e) {
            nb::object type = unavailableType();
            PyErr_SetString(type.ptr(), e.what());
        }
    });
}

namespace detail {

/** The exception a callback raised, held until the seam call returns. */
inline std::unique_ptr<nb::python_error>& pendingCallbackError()
{
    thread_local std::unique_ptr<nb::python_error> pending;
    return pending;
}

} // namespace detail

/** @brief Store the active Python exception for @ref check to re-raise. */
inline void storeCallbackError(nb::python_error&& e)
{
    detail::pendingCallbackError() = std::make_unique<nb::python_error>(std::move(e));
}

/**
 * @brief Raise the Python exception that @p status of a call into @p b means;
 *        return when it is ``EAGLE_BACKEND_OK``.
 */
inline void check(Backend& b, eagle_backend_status status)
{
    if (status == EAGLE_BACKEND_OK)
        return;
    if (status == EAGLE_BACKEND_CALLBACK_FAILED) {
        std::unique_ptr<nb::python_error> pending = std::move(detail::pendingCallbackError());
        if (pending)
            throw nb::python_error(std::move(*pending));
        throw std::runtime_error("eagle: a callback failed: " + b.lastError());
    }
    const std::string msg = b.lastError();
    switch (status) {
    case EAGLE_BACKEND_ERROR:
    case EAGLE_BACKEND_BAD_HANDLE:
        throw std::runtime_error(msg);
    case EAGLE_BACKEND_INVALID_ARGUMENT:
        throw std::invalid_argument(msg);
    case EAGLE_BACKEND_INDEX:
        throw nb::index_error(msg.c_str());
    case EAGLE_BACKEND_INTEROP:
        throw eagle::interop::InteropError(msg);
    case EAGLE_BACKEND_UNAVAILABLE:
        raiseUnavailable(msg);
    default:
        throw std::runtime_error("eagle: the " + b.label() + " returned status "
            + std::to_string(status) + ", which this eagle._core does not know: " + msg);
    }
}

} // namespace eagle_seam
