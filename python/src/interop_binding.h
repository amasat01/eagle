// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file interop_binding.h
 * @brief nanobind surface of the generic DLPack layer (``plugin/interop.h``
 *        over ``aether/interop/Buffer.h``), consumed by ``eagle.interop``.
 *
 * Bound surface:
 *   InteropBuffer      — one ``eagle::interop::OrderedBuffer``: built from a
 *                        DLPack capsule (either generation) or from raw
 *                        array-interface fields; reports access / owner /
 *                        producer / stream / pointer / shape / strides /
 *                        dtype / device; validates ``Requirements``; exports
 *                        versioned or legacy capsules under the stream
 *                        contract; re-fences.
 *   fence              — ``eagle::interop::fence`` on raw stream codes (in a pure
 *                        C++ core, through the CUDA fence provider).
 *   event_pool_created — how many events the fence pool has created (bound by
 *                        the core itself when the pool lives in a backend).
 *
 * Capsule protocol: an imported capsule is renamed ``used_dltensor[_versioned]``
 * the moment its tensor is taken, so its own destructor no longer calls the
 * deleter; an exported capsule calls the tensor's deleter from its destructor
 * only while it still carries the unconsumed name.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "plugin/interop.h"

namespace eagle_interop_binding {

namespace nb = nanobind;
using eagle::interop::OrderedBuffer;

inline constexpr const char* kVersionedName = "dltensor_versioned";
inline constexpr const char* kLegacyName = "dltensor";

inline void versioned_capsule_destructor(PyObject* cap)
{
    if (PyCapsule_IsValid(cap, kVersionedName)) {
        auto* t = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(cap, kVersionedName));
        if (t != nullptr && t->deleter != nullptr)
            t->deleter(t);
    }
}

inline void legacy_capsule_destructor(PyObject* cap)
{
    if (PyCapsule_IsValid(cap, kLegacyName)) {
        auto* t = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, kLegacyName));
        if (t != nullptr && t->deleter != nullptr)
            t->deleter(t);
    }
}

// The stream an imported CUDA buffer is ordered on: the consumer's stream
// argument, the legacy default stream when none was given; empty on the host.
inline std::optional<std::intptr_t> view_stream(const aether::interop::BufferView& b,
    std::optional<std::intptr_t> stream)
{
    if (b.view.device.type() == kDLCPU)
        return std::nullopt;
    return stream.value_or(eagle::interop::kStreamLegacy);
}

inline OrderedBuffer from_capsule(nb::object capsule, const std::string& producer, std::optional<std::intptr_t> stream)
{
    PyObject* cap = capsule.ptr();
    aether::interop::BufferView b;
    if (PyCapsule_IsValid(cap, kVersionedName)) {
        auto* t = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(cap, kVersionedName));
        PyCapsule_SetName(cap, "used_dltensor_versioned");
        b = aether::interop::importDLPack(t, producer);
    } else if (PyCapsule_IsValid(cap, kLegacyName)) {
        auto* t = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, kLegacyName));
        PyCapsule_SetName(cap, "used_dltensor");
        b = aether::interop::importDLPack(t, producer);
    } else {
        throw eagle::interop::InteropError(
            "dlpack: required an unconsumed 'dltensor_versioned' or 'dltensor' capsule");
    }
    if (b.view.device.type() == kDLCPU)
        eagle::interop::check_host_stream(stream);
    OrderedBuffer ob{ std::move(b), std::nullopt };
    ob.stream = view_stream(ob.buffer, stream);
    return ob;
}

inline OrderedBuffer from_interface(nb::object owner, std::uintptr_t ptr, bool read_only,
    const std::vector<std::int64_t>& shape, std::optional<std::vector<std::int64_t>> strides,
    const std::string& typestr, int device_type, int device_id, const std::string& producer,
    std::optional<std::intptr_t> stream)
{
    aether::interop::ArrayInterface ai;
    ai.data = reinterpret_cast<void*>(ptr);
    ai.readOnly = read_only;
    ai.shape = shape;
    if (strides)
        ai.strides = *strides;
    ai.typestr = typestr;
    ai.device = DLDevice{ static_cast<DLDeviceType>(device_type), device_id };
    ai.producer = producer;
    // The producer object itself is the keep-alive; dropping it needs the GIL.
    ai.keepAlive = std::shared_ptr<void>(new nb::object(std::move(owner)), [](void* p) {
        nb::gil_scoped_acquire gil;
        delete static_cast<nb::object*>(p);
    });
    if (ai.device.device_type == kDLCPU)
        eagle::interop::check_host_stream(stream);
    OrderedBuffer ob{ aether::interop::fromArrayInterface(ai), std::nullopt };
    ob.stream = view_stream(ob.buffer, stream);
    return ob;
}

inline std::vector<std::int64_t> shape_of(const OrderedBuffer& b)
{
    std::vector<std::int64_t> s(b.buffer.view.rank);
    for (std::size_t i = 0; i < s.size(); ++i)
        s[i] = static_cast<std::int64_t>(b.buffer.view.extents[i]);
    return s;
}

inline std::vector<std::int64_t> strides_of(const OrderedBuffer& b)
{
    std::vector<std::int64_t> s(b.buffer.view.rank);
    for (std::size_t i = 0; i < s.size(); ++i)
        s[i] = static_cast<std::int64_t>(b.buffer.view.strides[i]);
    return s;
}

inline void bind(nb::module_& m)
{
    nb::exception<eagle::interop::InteropError>(m, "InteropError", PyExc_ValueError);

    nb::class_<OrderedBuffer>(m, "InteropBuffer")
        .def_static("from_capsule", &from_capsule, nb::arg("capsule"), nb::arg("producer"),
            nb::arg("stream").none(),
            "Take the tensor out of a DLPack capsule (either generation).")
        .def_static("from_interface", &from_interface, nb::arg("owner"), nb::arg("ptr"), nb::arg("read_only"),
            nb::arg("shape"), nb::arg("strides").none(), nb::arg("typestr"), nb::arg("device_type"),
            nb::arg("device_id"), nb::arg("producer"), nb::arg("stream").none(),
            "Build the record from raw array-interface fields; ``owner`` is kept alive.")
        .def_prop_ro("ptr",
            [](const OrderedBuffer& b) { return reinterpret_cast<std::uintptr_t>(b.buffer.view.data); }, nb::lock_self())
        .def_prop_ro("shape", &shape_of, nb::lock_self())
        .def_prop_ro("strides", &strides_of, nb::lock_self())
        .def_prop_ro("dtype", [](const OrderedBuffer& b) { return aether::interop::dtypeName(b.buffer.view.dtype); }, nb::lock_self())
        .def_prop_ro("device_type", [](const OrderedBuffer& b) { return static_cast<int>(b.buffer.view.device.type()); }, nb::lock_self())
        .def_prop_ro("device_id", [](const OrderedBuffer& b) { return static_cast<int>(b.buffer.view.device.id()); }, nb::lock_self())
        .def_prop_ro("access", [](const OrderedBuffer& b) { return std::string(aether::interop::accessName(b.buffer.access)); }, nb::lock_self())
        .def_prop_ro("writable", [](const OrderedBuffer& b) { return b.buffer.writable(); }, nb::lock_self())
        .def_prop_ro("assumed_writable", [](const OrderedBuffer& b) { return b.buffer.assumedWritable; }, nb::lock_self())
        .def_prop_ro("owner", [](const OrderedBuffer& b) { return b.buffer.owner; }, nb::lock_self())
        .def_prop_ro("producer", [](const OrderedBuffer& b) { return b.buffer.producer; }, nb::lock_self())
        .def_prop_ro("stream", [](const OrderedBuffer& b) { return b.stream; }, nb::lock_self())
        .def("assume_writable", [](OrderedBuffer& b) { aether::interop::assumeWritable(b.buffer); }, nb::lock_self())
        .def(
            "refusals",
            [](const OrderedBuffer& b, std::optional<int> dtype_code, std::optional<int> dtype_bits,
                std::optional<std::int64_t> count, std::optional<std::vector<std::int64_t>> shape, bool contiguous,
                std::size_t alignment, std::optional<int> device_type, std::optional<int> device_id, bool writable) {
                aether::interop::Requirements r;
                if (dtype_code && dtype_bits) {
                    r.dtype = aether::DType(DLDataType{ static_cast<std::uint8_t>(*dtype_code),
                        static_cast<std::uint8_t>(*dtype_bits), 1 });
                }
                r.count = count;
                r.shape = shape;
                r.contiguous = contiguous;
                r.alignment = alignment;
                if (device_type)
                    r.deviceType = static_cast<DLDeviceType>(*device_type);
                if (device_id)
                    r.deviceId = static_cast<std::int32_t>(*device_id);
                r.writable = writable;
                return aether::interop::refusals(b.buffer, r);
            }, nb::lock_self(),
            nb::arg("dtype_code").none(), nb::arg("dtype_bits").none(), nb::arg("count").none(),
            nb::arg("shape").none(), nb::arg("contiguous"), nb::arg("alignment"), nb::arg("device_type").none(),
            nb::arg("device_id").none(), nb::arg("writable"),
            "Every refusal of the requirements, one message per failed check.")
        .def(
            "export_versioned",
            [](const OrderedBuffer& b, std::optional<std::intptr_t> stream) {
                DLManagedTensorVersioned* t = eagle::interop::export_versioned(b, stream);
                PyObject* cap = PyCapsule_New(t, kVersionedName, &versioned_capsule_destructor);
                if (cap == nullptr) {
                    t->deleter(t);
                    throw nb::python_error();
                }
                return nb::steal<nb::object>(cap);
            }, nb::lock_self(),
            nb::arg("stream").none(), "A versioned DLPack capsule, fenced onto ``stream``.")
        .def(
            "export_legacy",
            [](const OrderedBuffer& b, std::optional<std::intptr_t> stream) {
                DLManagedTensor* t = eagle::interop::export_legacy(b, stream);
                PyObject* cap = PyCapsule_New(t, kLegacyName, &legacy_capsule_destructor);
                if (cap == nullptr) {
                    t->deleter(t);
                    throw nb::python_error();
                }
                return nb::steal<nb::object>(cap);
            }, nb::lock_self(),
            nb::arg("stream").none(), "A legacy DLPack capsule, fenced onto ``stream``.")
        .def(
            "refence", [](const OrderedBuffer& b, std::optional<std::intptr_t> consumer) { eagle::interop::refence(b, consumer); }, nb::lock_self(),
            nb::arg("consumer").none(), "Order the buffer's stream before ``consumer`` again.");

    m.def(
        "fence",
        [](std::optional<std::intptr_t> producer, std::optional<std::intptr_t> consumer, int device) {
#ifndef EAGLE_CPU_ONLY
            eagle::interop::fence(producer, consumer, device);
#else
            // A pure C++ core reaches CUDA through the fence provider its device
            // backend installed for the CUDA device type.
            eagle::interop::FenceProvider provider = eagle::interop::fence_provider(kDLCUDA);
            if (provider == nullptr)
                throw eagle::interop::InteropError("device: a CUDA fence needs a CUDA build of eagle, this build is CPU-only");
            provider(kDLCUDA, device, producer, consumer);
#endif
        },
        nb::arg("producer").none(), nb::arg("consumer").none(), nb::arg("device") = 0,
        "Order ``producer`` before ``consumer`` (DLPack stream codes) without blocking the host.");
#ifndef EAGLE_CPU_ONLY
    m.def(
        "event_pool_created", [] { return eagle::interop::EventPool::instance().created(); },
        "How many events the fence pool has created.");
#endif
}

} // namespace eagle_interop_binding
