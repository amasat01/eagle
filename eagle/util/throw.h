// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>

/* aether has NO generic
 * thrower: ``aether::err::fail()`` (``aether/err/Error.h``) always throws
 * ``aether::Error``, whereas every eagle call site names its OWN exception type
 * (``std::invalid_argument`` in GraphComposer's argument validation,
 * ``std::runtime_error`` elsewhere) and the tests catch those types. Rather
 * than shim a thrower into aether, eagle owns the raise site outright here.
 *
 * The message text stays BYTE-IDENTICAL across releases, so no
 * caught-message expectation anywhere downstream moves. */

namespace eagle {
namespace err {
namespace detail {

/**
 * @brief Throw ``ErrorT`` with a message naming the raise site.
 *
 * @tparam ErrorT  Exception type to throw; defaults to ``std::runtime_error``.
 * @tparam Params  Extra constructor arguments forwarded to ``ErrorT``.
 * @param file     ``__FILE__`` at the raise site.
 * @param line     ``__LINE__`` at the raise site.
 * @param func     ``__func__`` at the raise site.
 * @param message  Caller-supplied description.
 * @param params   Extra ``ErrorT`` constructor arguments.
 */
template<typename ErrorT = std::runtime_error, typename... Params>
[[noreturn]] inline void throwAt(const std::string& file, int line,
    const std::string& func, const std::string& message, Params... params)
{
    std::string msg("\nException raised: ");
    msg += "\nWhat : ";
    msg += message;
    msg += "\nFunc : ";
    msg += func;
    msg += "\nSrc  : ";
    msg += file;
    msg += "\nLine : ";
    msg += std::to_string(line);
    msg += "\n";
    throw ErrorT(msg, params...);
}

} // namespace detail
} // namespace err
} // namespace eagle

/**
 * @brief Throw exception from host code.
 *
 * For device code, use EAGLE_GPU_THROW.
 */
#define EAGLE_THROW(EXCEPTION_TYPE, MESSAGE)                                   \
    ::eagle::err::detail::throwAt<EXCEPTION_TYPE>(                             \
        __FILE__, __LINE__, static_cast<const char*>(__func__), (MESSAGE));

/**
 * @brief Throw exception from host code if CONDITION is false.
 *
 * For device code, use EAGLE_GPU_ASSERT.
 */
#define EAGLE_ASSERT(CONDITION, MESSAGE)                                       \
    if (!(CONDITION)) {                                                        \
        ::eagle::err::detail::throwAt<std::runtime_error>(__FILE__, __LINE__,  \
            static_cast<const char*>(__func__),                                \
            "Condition check failed: " #CONDITION ". Help message: "           \
                + std::string(MESSAGE));                                       \
    }
