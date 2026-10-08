// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Loader.h
 * @brief Header-only loader of an ``eagle-backend/1`` plugin: discovery,
 *        ``dlopen``, the handshake and symbol binding, one backend per kind.
 *
 * The core never links a backend. It opens ``libeagle_<kind>.<ext>`` lazily, on
 * the first call that needs it, ``RTLD_NOW | RTLD_LOCAL``, and never closes it (a
 * device runtime keeps process-global state). ``import eagle._core`` touches
 * nothing.
 *
 * Discovery, first hit wins, and an explicit override never falls through:
 *   1. ``$EAGLE_BACKEND_<KIND>`` (e.g. ``EAGLE_BACKEND_CUDA``): that exact path;
 *   2. ``<directory of the core image>/libeagle_<kind>.<ext>`` (the wheel layout);
 *   3. the bare file name, through the dynamic loader's search path.
 *
 * Handshake: ``eagle_backend_version`` first, alone; its major must be exactly
 * this header's (the minor is informational). Then the rest of the mandatory
 * core, the capability bits and the device types. Every group whose bit is set
 * must export all of its 1.0 symbols, or the load is refused naming the missing
 * one. Every resolved address must lie outside the core image.
 *
 * At call time: a group whose bit is clear raises @ref Unavailable naming the
 * group; an experimental or post-1.0 symbol is bound on its own and raises naming
 * itself when absent. A failed load is cached: reported, never retried.
 */
#pragma once

#include <dlfcn.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "Groups.h"
#include "eagle_backend.h"

namespace eagle_seam {

/** @brief The backend (or the part of it asked for) cannot serve the call. */
class Unavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {

inline void coreAnchor() {}

inline std::string dlerrorText()
{
    const char* e = dlerror();
    return e ? std::string(e) : std::string("no dlerror message");
}

inline std::string upper(const std::string& s)
{
    std::string out = s;
    for (char& c : out)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

inline std::string versionText(std::uint32_t v)
{
    return std::to_string(v >> 16) + "." + std::to_string(v & 0xffffu);
}

} // namespace detail

/** @brief One loaded (or failed) backend of a given kind; see @ref backend. */
class Backend {
public:
    /** @brief The mandatory entries (valid when @ref loaded). */
    struct Meta {
        std::uint32_t (*version)(void) = nullptr;
        const char* (*last_error)(void) = nullptr;
        eagle_backend_status (*build_info)(struct eagle_backend_build_info*) = nullptr;
        eagle_backend_status (*probe)(void) = nullptr;
        void (*free)(void*) = nullptr;
        eagle_backend_status (*capabilities)(std::uint64_t*) = nullptr;
        eagle_backend_status (*device_types)(std::int32_t*, std::int32_t, std::int32_t*) = nullptr;
    };

    explicit Backend(std::string kind)
        : kind_(std::move(kind))
    {
        load_();
    }

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    const std::string& kind() const { return kind_; }
    bool loaded() const { return handle_ != nullptr; }
    /** The file that was loaded (empty when none was). */
    const std::string& path() const { return path_; }
    /** The whole failure, discovery trail included; empty once loaded. */
    const std::string& error() const { return error_; }
    std::uint32_t version() const { return version_; }
    std::uint64_t capabilities() const { return capabilities_; }
    const std::vector<std::int32_t>& deviceTypes() const { return deviceTypes_; }
    const struct eagle_backend_build_info& buildInfo() const { return info_; }
    const Meta& meta() const { return meta_; }

    /** @brief Throw @ref Unavailable unless the backend loaded. */
    void require() const
    {
        if (!loaded())
            throw Unavailable(error_);
    }

    /** @brief The backend's message for the calling thread's last failure (copied). */
    std::string lastError() const
    {
        if (meta_.last_error == nullptr)
            return error_;
        const char* m = meta_.last_error();
        return m ? std::string(m) : std::string();
    }

    /** @brief "cuda backend '<path>'" (or without the path when none loaded). */
    std::string label() const
    {
        return kind_ + " backend" + (path_.empty() ? std::string() : " '" + path_ + "'");
    }

    /** @brief Why @p g cannot be used ("" when it can); never throws. */
    std::string groupError(const GroupSpec& g) const
    {
        if (!loaded())
            return error_;
        if ((capabilities_ & g.bit) == 0)
            return "the " + label() + " does not provide " + g.name;
        return std::string();
    }

    /**
     * @brief The address of 1.0 symbol @p sym of group @p g.
     * @throws Unavailable when the backend did not load or does not provide @p g.
     */
    void* groupSymbol(const GroupSpec& g, const char* sym)
    {
        const std::string why = groupError(g);
        if (!why.empty())
            throw Unavailable(why);
        void* p = lookup_(sym);
        if (p == nullptr) // verified at load; reaching here is a core bug
            throw Unavailable("the " + label() + " lost '" + sym + "'");
        return p;
    }

    /** @brief An optional symbol (experimental, or added after 1.0), or nullptr. */
    void* optional(const char* sym)
    {
        if (!loaded())
            return nullptr;
        return lookup_(sym);
    }

    /**
     * @brief An optional symbol that must be present for this call.
     * @throws Unavailable naming @p sym when the backend lacks it.
     */
    void* requireOptional(const char* sym)
    {
        require();
        void* p = lookup_(sym);
        if (p == nullptr)
            throw Unavailable("the " + label() + " does not provide " + sym);
        return p;
    }

    /** @brief Whether the backend declared DLPack device type @p type. */
    bool servesDeviceType(std::int32_t type) const
    {
        for (std::int32_t t : deviceTypes_)
            if (t == type)
                return true;
        return false;
    }

    /** @brief The file a backend of @p kind is discovered under. */
    static std::string fileName(const std::string& kind)
    {
#if defined(__APPLE__)
        return "libeagle_" + kind + ".dylib";
#elif defined(_WIN32)
        return "libeagle_" + kind + ".dll";
#else
        return "libeagle_" + kind + ".so";
#endif
    }
    /** @brief The variable that overrides discovery for @p kind. */
    static std::string envName(const std::string& kind) { return "EAGLE_BACKEND_" + detail::upper(kind); }

private:
    bool resolve_(const char* name, void*& out, std::string& why) const
    {
        (void)dlerror();
        void* p = dlsym(handle_, name);
        if (p == nullptr) {
            why = detail::dlerrorText();
            return false;
        }
        Dl_info fi;
        if (dladdr(p, &fi) != 0 && fi.dli_fbase == coreBase_) {
            why = "it resolved into the eagle._core image itself, not the backend";
            return false;
        }
        out = p;
        return true;
    }

    void* lookup_(const char* name)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = symbols_.find(name);
        if (it == symbols_.end()) {
            void* p = nullptr;
            std::string why;
            resolve_(name, p, why);
            it = symbols_.emplace(name, p).first;
        }
        return it->second;
    }

    void fail_(const std::string& why)
    {
        error_ = "eagle: the " + kind_ + " backend is unavailable: " + why;
        handle_ = nullptr; // deliberately never dlclose'd (see the file doc)
        path_.clear();
        meta_ = Meta{};
        capabilities_ = 0;
        deviceTypes_.clear();
    }

    template <class Fn>
    bool bindMandatory_(const char* name, Fn& slot)
    {
        void* p = nullptr;
        std::string why;
        if (!resolve_(name, p, why)) {
            const std::string loadedPath = path_;
            fail_("'" + loadedPath + "' does not export the mandatory symbol '" + name + "' (" + why
                + "); it is not a complete eagle-backend/1 plugin.");
            return false;
        }
        std::memcpy(&slot, &p, sizeof p);
        return true;
    }

    void load_()
    {
        Dl_info self;
        std::string coreDir;
        if (dladdr(reinterpret_cast<const void*>(&detail::coreAnchor), &self) != 0) {
            coreBase_ = self.dli_fbase;
            if (self.dli_fname != nullptr) {
                const std::string p = self.dli_fname;
                const auto slash = p.find_last_of('/');
                if (slash != std::string::npos)
                    coreDir = p.substr(0, slash);
            }
        }

        const std::string file = fileName(kind_);
        const std::string env = envName(kind_);
        std::string trail;
        auto attempt = [&](const std::string& candidate) {
            (void)dlerror();
            void* h = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (h != nullptr) {
                handle_ = h;
                path_ = candidate;
                return true;
            }
            trail += "\n  tried " + candidate + ": " + detail::dlerrorText();
            return false;
        };

        if (const char* forced = std::getenv(env.c_str()); forced != nullptr && *forced != '\0') {
            if (!attempt(forced)) {
                fail_("$" + env + " names '" + std::string(forced)
                    + "', which could not be loaded (an explicit override never falls back to discovery)."
                    + trail);
                return;
            }
        } else {
            bool found = !coreDir.empty() && attempt(coreDir + "/" + file);
            if (!found)
                found = attempt(file);
            if (!found) {
                fail_("no " + file + " was found (set $" + env + " to its path, or install it next to eagle._core)."
                    + trail);
                return;
            }
        }

        // The version first, alone: a plugin of another major need not have the
        // rest of this major's mandatory core, and the version is the reason to give.
        if (!bindMandatory_("eagle_backend_version", meta_.version))
            return;
        version_ = meta_.version();
        const std::uint32_t major = version_ >> 16;
        if (major != EAGLE_BACKEND_VERSION_MAJOR) {
            const std::string loadedPath = path_;
            fail_("'" + loadedPath + "' implements eagle-backend " + detail::versionText(version_) + " (major "
                + std::to_string(major) + ") but this eagle._core needs major "
                + std::to_string(EAGLE_BACKEND_VERSION_MAJOR) + ".");
            return;
        }
        if (!bindMandatory_("eagle_backend_last_error", meta_.last_error)
            || !bindMandatory_("eagle_backend_build_info", meta_.build_info)
            || !bindMandatory_("eagle_backend_probe", meta_.probe)
            || !bindMandatory_("eagle_backend_free", meta_.free)
            || !bindMandatory_("eagle_backend_capabilities", meta_.capabilities)
            || !bindMandatory_("eagle_backend_device_types", meta_.device_types))
            return;

        std::uint64_t caps = 0;
        if (meta_.capabilities(&caps) != EAGLE_BACKEND_OK) {
            const std::string why = lastError();
            const std::string loadedPath = path_;
            fail_("'" + loadedPath + "' failed eagle_backend_capabilities: " + why);
            return;
        }
        capabilities_ = caps;

        std::int32_t count = 0;
        std::int32_t types[64] = {};
        if (meta_.device_types(types, 64, &count) != EAGLE_BACKEND_OK || count < 0) {
            const std::string why = lastError();
            const std::string loadedPath = path_;
            fail_("'" + loadedPath + "' failed eagle_backend_device_types: " + why);
            return;
        }
        deviceTypes_.assign(types, types + (count < 64 ? count : 64));

        std::memset(&info_, 0, sizeof info_);
        info_.struct_size = sizeof info_;
        if (meta_.build_info(&info_) != EAGLE_BACKEND_OK) {
            const std::string why = lastError();
            const std::string loadedPath = path_;
            fail_("'" + loadedPath + "' failed eagle_backend_build_info: " + why);
            return;
        }
        info_.eagle_version[sizeof info_.eagle_version - 1] = '\0';
        info_.archs[sizeof info_.archs - 1] = '\0';
        info_.backend_kind[sizeof info_.backend_kind - 1] = '\0';

        // A set bit promises the group's 1.0 symbols: one missing is a corrupt
        // plugin, refused at load naming the symbol (not "does not provide").
        for (const GroupSpec& g : groups()) {
            if ((capabilities_ & g.bit) == 0)
                continue;
            for (const char* s : g.symbols) {
                void* p = nullptr;
                std::string why;
                if (!resolve_(s, p, why)) {
                    const std::string loadedPath = path_;
                    fail_("'" + loadedPath + "' sets the " + g.name + " capability but does not export '" + s
                        + "' (" + why + ").");
                    return;
                }
                symbols_.emplace(s, p);
            }
        }
        error_.clear();
    }

    std::string kind_;
    void* handle_ = nullptr;
    const void* coreBase_ = nullptr;
    std::string path_;
    std::string error_;
    std::uint32_t version_ = 0;
    std::uint64_t capabilities_ = 0;
    std::vector<std::int32_t> deviceTypes_;
    struct eagle_backend_build_info info_ {};
    Meta meta_;
    std::mutex mutex_;
    std::map<std::string, void*> symbols_;
};

/**
 * @brief The process-wide backend of @p kind, loaded on first request.
 *
 * Never throws: a failed load is a @ref Backend whose ``loaded()`` is false and
 * whose ``error()`` carries the discovery trail. One per kind, leaked on purpose.
 */
inline Backend& backend(const std::string& kind)
{
    static std::mutex m;
    static auto* kinds = new std::map<std::string, Backend*>();
    std::lock_guard<std::mutex> lock(m);
    auto it = kinds->find(kind);
    if (it == kinds->end())
        it = kinds->emplace(kind, new Backend(kind)).first;
    return *it->second;
}

/** @brief Typed address of 1.0 symbol @p sym of group @p groupName in @p kind's backend. */
template <class Fn>
Fn groupFn(const char* kind, const char* groupName, const char* sym)
{
    void* p = backend(kind).groupSymbol(group(groupName), sym);
    Fn f;
    std::memcpy(&f, &p, sizeof f);
    return f;
}

/** @brief Typed address of optional symbol @p sym (raises naming it when absent). */
template <class Fn>
Fn optionalFn(const char* kind, const char* sym)
{
    void* p = backend(kind).requireOptional(sym);
    Fn f;
    std::memcpy(&f, &p, sizeof f);
    return f;
}

} // namespace eagle_seam

/** Bound address of a 1.0 group symbol of the cuda backend, cached per call site. */
#define EAGLE_SEAM_FN(groupName, sym)                                                                 \
    ([]() -> decltype(&::sym) {                                                                       \
        static const auto f = ::eagle_seam::groupFn<decltype(&::sym)>("cuda", groupName, #sym);       \
        return f;                                                                                     \
    }())

/** Bound address of an optional (``x_`` or post-1.0) symbol of the cuda backend. */
#define EAGLE_SEAM_XFN(sym)                                                                           \
    ([]() -> decltype(&::sym) {                                                                       \
        static const auto f = ::eagle_seam::optionalFn<decltype(&::sym)>("cuda", #sym);               \
        return f;                                                                                     \
    }())
