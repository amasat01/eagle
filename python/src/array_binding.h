// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file array_binding.h
 * @brief nanobind surface of ``eagle::array::Array`` (``_core.Array``, ``_core.DType``, the ``array_*`` functions).
 *
 * Included by ``eagle_core.cpp`` after ``ck`` and ``EBX``. This TU never includes a CUDA header: device
 * storage comes from the seam's ``mem_*``, device ops from ``cell_op``, host ops from ``eagle/array/host.h``,
 * and a host-only build never touches the backend (only device placements do).
 *
 * Every op ENQUEUES on the array's stream (or ``stream=``) and is capture-legal; ops that must read device
 * data on the host (``get``, ``sum``, ``item``, ``tolist``) synchronise, and so fail by name inside a capture.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "eagle/array/Array.h"
#include "eagle/array/host.h"
#include "plugin/dlpack.h"

namespace eagle_array_binding {

namespace nb = nanobind;
namespace ea = eagle::array;
using ea::Array;
using ea::DType;
using ea::Op;
using ea::Placement;

struct PyDType {
    DType d;
};

// ---- dtype and placement parsing -----------------------------------------------------------------

inline DType parse_dtype(nb::handle h)
{
    if (nb::isinstance<PyDType>(h)) return nb::cast<PyDType>(h).d;
    try {
        if (nb::isinstance<nb::str>(h)) return ea::from_name(nb::cast<std::string>(h));
        for (const char* attr : { "name", "__name__" })
            if (nb::hasattr(h, attr)) {
                nb::object v = h.attr(attr);
                if (nb::isinstance<nb::str>(v)) return ea::from_name(nb::cast<std::string>(v));
            }
    } catch (const std::invalid_argument& e) {
        throw nb::value_error(e.what());
    }
    throw nb::type_error("dtype: expected an eagle.DType, a name, or an object with .name");
}

inline Placement parse_place(const std::string& s)
{
    if (s == "cpu") return { Placement::Kind::Cpu, 0 };
    if (s == "pinned") return { Placement::Kind::Pinned, 0 };
    if (s == "cuda") return { Placement::Kind::Cuda, 0 };
    if (s.rfind("cuda:", 0) == 0) return { Placement::Kind::Cuda, std::stoi(s.substr(5)) };
    throw nb::value_error(("device '" + s + "' is not cpu, pinned, cuda or cuda:N").c_str());
}

inline std::string place_str(const Placement& p)
{
    return p.kind == Placement::Kind::Cpu ? "cpu" : p.kind == Placement::Kind::Pinned ? "pinned" : "cuda:" + std::to_string(p.device);
}

inline std::size_t item_of(const Array& a) { return ea::itemsize(a.dtype()); }

// ---- running an op ---------------------------------------------------------------------------------

struct Call {
    Op op;
    int dtype = 7;
    std::int64_t n = 0;
    void* p0 = nullptr;
    const void *p1 = nullptr, *p2 = nullptr, *p3 = nullptr;
    std::int64_t i0 = 0, i1 = 0;
    double f0 = 0.0, f1 = 0.0;
};

inline void run_call(const Placement& pl, std::uint64_t stream, const Call& c)
{
    if (c.n <= 0) return;
    if (pl.host()) {
        // a host op: pure arithmetic on raw pointers
        nb::gil_scoped_release release;
        ea::Args a;
        a.p0 = c.p0; a.p1 = c.p1; a.p2 = c.p2; a.p3 = c.p3;
        a.n = c.n; a.i0 = c.i0; a.i1 = c.i1; a.f0 = c.f0; a.f1 = c.f1; a.dtype = c.dtype;
        ea::host::run(c.op, a);
        return;
    }
    eagle_backend_cell_op_desc d {};
    d.struct_size = sizeof d;
    d.op = static_cast<std::uint32_t>(c.op);
    d.device = pl.device;
    d.stream = stream;
    d.n = c.n;
    d.p[0] = reinterpret_cast<std::uintptr_t>(c.p0);
    d.p[1] = reinterpret_cast<std::uintptr_t>(c.p1);
    d.p[2] = reinterpret_cast<std::uintptr_t>(c.p2);
    d.p[3] = reinterpret_cast<std::uintptr_t>(c.p3);
    d.i[0] = c.op == Op::Add ? c.dtype : c.i0;
    d.i[1] = c.i1;
    d.i[9] = c.dtype;
    d.f[0] = c.f0;
    d.f[1] = c.f1;
    ck(EBX(eagle_backend_cell_op)(&d));
}

/// The stream an op on ``a`` enqueues on: ``s`` if given, else the innermost open eagle stream scope for a
/// device array (so an op inside a capture is captured), else the array's own stream (no scope open).
inline std::uint64_t stream_of(const Array& a, const std::optional<std::uint64_t>& s)
{
    if (s) return *s;
    if (a.placement().host()) return a.stream();
    static const nb::object* const current = new nb::object(nb::module_::import_("eagle._stream").attr("scoped"));   // never destroyed: a static destructor would run after the interpreter is gone
    const auto cur = nb::cast<std::uint64_t>((*current)());
    return cur ? cur : a.stream();
}

inline void sync(const Array& a)
{
    if (!a.placement().host()) {
        eagle_backend_status status;
        {
            nb::gil_scoped_release release;
            status = EBX(eagle_backend_stream_sync)(a.placement().device, a.stream());   // this array's stream only
        }
        ck(status);                                                                       // raising needs the GIL back
    }
}

// ---- copies ---------------------------------------------------------------------------------------

inline void memcpy_kind(void* dst, const void* src, std::size_t bytes, bool dst_host, bool src_host, std::uint64_t stream)
{
    if (bytes == 0) return;
    if (dst_host && src_host) { std::memcpy(dst, src, bytes); return; }
    const int kind = dst_host ? EAGLE_BACKEND_COPY_D2H : src_host ? EAGLE_BACKEND_COPY_H2D : EAGLE_BACKEND_COPY_D2D;
    eagle_backend_status status;
    {
        nb::gil_scoped_release release;
        status = EBX(eagle_backend_memcpy)(reinterpret_cast<std::uintptr_t>(dst), reinterpret_cast<std::uintptr_t>(src), bytes, kind, stream);
    }
    ck(status);
}

/// Element-wise copy between host arrays of any strides (rank <= 6), parallel over the elements.
inline void host_strided_copy(const Array& dst, const Array& src)
{
    const std::int64_t n = dst.size();
    const std::size_t r = dst.rank(), it = item_of(dst);
    char* d = static_cast<char*>(dst.data());
    const char* s = static_cast<const char*>(src.data());
#pragma omp parallel for schedule(static)
    for (std::int64_t j = 0; j < n; ++j) {
        std::int64_t rem = j, doff = 0, soff = 0;
        for (std::size_t k = r; k-- > 0;) {
            const std::int64_t i = rem % dst.extent(k);
            rem /= dst.extent(k);
            doff += i * dst.stride(k);
            soff += i * src.stride(k);
        }
        std::memcpy(d + doff * static_cast<std::int64_t>(it), s + soff * static_cast<std::int64_t>(it), it);
    }
}

inline Array dense_host(const Array& src)
{
    Array t = Array::empty(src.shape(), src.dtype(), { Placement::Kind::Cpu, 0 });
    host_strided_copy(t, src);
    return t;
}

/// One strided-copy kernel (rank <= 4, no allocation): ``dst`` <- ``src``, same shape and dtype, every offset from the
/// logical index and each side's own strides.
inline void strided_op(const Array& dst, const Array& src, std::uint64_t stream)
{
    if (dst.rank() > 4) throw std::invalid_argument("eagle array: a strided device copy of rank > 4 cannot be done directly; copy it in pieces");
    eagle_backend_cell_op_desc d {};
    d.struct_size = sizeof d;
    d.op = EAGLE_BACKEND_CELL_STRIDED_COPY;
    d.device = dst.placement().device;
    d.stream = stream;
    d.n = dst.size();
    d.p[0] = reinterpret_cast<std::uintptr_t>(dst.data());
    d.p[1] = reinterpret_cast<std::uintptr_t>(src.data());
    d.i[0] = static_cast<std::int64_t>(item_of(dst));
    d.i[1] = static_cast<std::int64_t>(dst.rank());
    for (std::size_t k = 0; k < dst.rank(); ++k) {
        d.i[2 + k] = dst.extent(k);
        d.i[6 + k] = dst.stride(k) * d.i[0];
        d.j[k] = src.stride(k) * d.i[0];
    }
    ck(EBX(eagle_backend_cell_op)(&d));
}

/// A contiguous copy of a strided device array (``CONTIGUOUS_GATHER``, rank <= 4).
inline Array dense_device(const Array& src, std::uint64_t stream)
{
    Array t = Array::empty(src.shape(), src.dtype(), src.placement(), stream);
    if (src.size() == 0) return t;
    strided_op(t, src, stream);
    return t;
}

inline Array dense(const Array& a, std::uint64_t stream)
{
    if (a.contiguous()) return a;
    return a.placement().host() ? dense_host(a) : dense_device(a, stream);
}

/// ``dst`` and ``src`` as one pitched 2-D copy (``memcpy_strided``) when one side is contiguous and the other is a
/// pitched block: unit axes dropped, contiguous axes merged, at most two left, unit innermost stride, positive pitch.
inline bool pitched_copy(const Array& dst, const Array& src, int kind, std::uint64_t stream)
{
    const bool dc = dst.contiguous();
    if (dc == src.contiguous()) return false;
    const Array& v = dc ? src : dst;
    std::vector<std::int64_t> ext, str;
    for (std::size_t k = 0; k < v.rank(); ++k)
        if (v.extent(k) != 1) { ext.push_back(v.extent(k)); str.push_back(v.stride(k)); }
    for (std::size_t k = 0; k + 1 < ext.size();)
        if (str[k] == str[k + 1] * ext[k + 1]) { ext[k + 1] *= ext[k]; ext.erase(ext.begin() + static_cast<std::ptrdiff_t>(k)); str.erase(str.begin() + static_cast<std::ptrdiff_t>(k)); }
        else ++k;
    const std::int64_t it = static_cast<std::int64_t>(item_of(v));
    std::uint64_t width = static_cast<std::uint64_t>(it), height = 1, pitch = 0;
    if (ext.size() == 1 && str[0] > 0) { height = static_cast<std::uint64_t>(ext[0]); pitch = static_cast<std::uint64_t>(str[0] * it); }
    else if (ext.size() == 2 && str[1] == 1 && str[0] > 0) { width = static_cast<std::uint64_t>(ext[1] * it); height = static_cast<std::uint64_t>(ext[0]); pitch = static_cast<std::uint64_t>(str[0] * it); }
    else return false;
    ck(EBX(eagle_backend_memcpy_strided)(reinterpret_cast<std::uintptr_t>(dst.data()), dc ? width : pitch, reinterpret_cast<std::uintptr_t>(src.data()),
        dc ? pitch : width, width, height, kind, stream));
    return true;
}

inline bool capturing(std::uint64_t stream)
{
    std::int32_t c = 0;
    ck(EBX(eagle_backend_stream_is_capturing)(stream, &c));
    return c != 0;
}

/// ``dst`` <- ``src`` (same dtype and shape), any placements and strides.
///  - Both on one device (rank <= 4): one kernel, no allocation, capture-legal.
///  - One side on the host: a pitched 2-D copy when the strided side is a pitched block (capture-legal); else a staged
///    copy, which allocates, so a capturing stream refuses it by name.
inline void copy_into(const Array& dst, const Array& src, std::uint64_t stream)
{
    if (dst.dtype() != src.dtype()) throw std::invalid_argument("eagle copy: dtypes differ (astype first)");
    if (dst.shape() != src.shape()) throw std::invalid_argument("eagle copy: shapes differ");
    if (dst.size() == 0) return;
    const bool dh = dst.placement().host(), sh = src.placement().host();
    if (dh && sh) {
        if (dst.contiguous() && src.contiguous()) std::memcpy(dst.data(), src.data(), dst.nbytes());
        else host_strided_copy(dst, src);
        return;
    }
    const bool plain = dst.contiguous() && src.contiguous();
    if (!dh && !sh && dst.placement().device == src.placement().device && dst.rank() <= 4 && !plain) { strided_op(dst, src, stream); return; }
    if (plain) { memcpy_kind(dst.data(), src.data(), dst.nbytes(), dh, sh, stream); return; }
    if (dh != sh) {
        const int kind = dh ? EAGLE_BACKEND_COPY_D2H : EAGLE_BACKEND_COPY_H2D;
        if (pitched_copy(dst, src, kind, stream)) return;
        if (capturing(stream)) throw std::invalid_argument("eagle copy: a strided host operand that is not a pitched 2-D block needs a staging copy, which a capturing stream cannot do; copy through a contiguous array first");
    }
    const Array s = dense(src, stream);
    if (dst.contiguous()) { memcpy_kind(dst.data(), s.data(), dst.nbytes(), dh, sh, stream); return; }
    if (dh) {
        Array tmp = Array::empty(dst.shape(), dst.dtype(), { Placement::Kind::Cpu, 0 });
        memcpy_kind(tmp.data(), s.data(), tmp.nbytes(), true, false, stream);
        sync(src);
        host_strided_copy(dst, tmp);
        return;
    }
    Array staged = s;
    if (sh || s.placement().device != dst.placement().device) {
        staged = Array::empty(dst.shape(), dst.dtype(), dst.placement(), stream);
        memcpy_kind(staged.data(), s.data(), staged.nbytes(), false, sh, stream);
    }
    strided_op(dst, staged, stream);
}

inline Array copy_of(const Array& a, std::uint64_t stream)
{
    Array t = Array::empty(a.shape(), a.dtype(), a.placement(), stream);
    copy_into(t, a, stream);
    return t;
}

/// ``out`` <- ``a`` converted to ``out``'s dtype (``out`` contiguous, same shape): one kernel, no allocation for a contiguous ``a``.
inline void cast_into(const Array& out, const Array& a, std::uint64_t stream)
{
    const DType to = out.dtype();
    if (!ea::is_compute(to) || !ea::is_compute(a.dtype()))
        throw std::invalid_argument(std::string("eagle astype: ") + ea::name(a.dtype()) + " -> " + ea::name(to) + " is not supported (casts run on codes 0-9)");
    if (!out.contiguous() || out.size() != a.size()) throw std::invalid_argument("eagle astype: out must be contiguous with the input's size");
    const Array s = dense(a, stream);
    if (out.size() == 0) return;
    if (a.placement().host()) {
        ea::host::cast(out.data(), static_cast<int>(to), s.data(), static_cast<int>(a.dtype()), out.size());
        return;
    }
    eagle_backend_cell_op_desc d {};
    d.struct_size = sizeof d;
    d.op = EAGLE_BACKEND_CELL_CAST;
    d.device = a.placement().device;
    d.stream = stream;
    d.n = out.size();
    d.p[0] = reinterpret_cast<std::uintptr_t>(out.data());
    d.p[1] = reinterpret_cast<std::uintptr_t>(s.data());
    d.i[0] = static_cast<int>(to);
    d.i[1] = static_cast<int>(a.dtype());
    ck(EBX(eagle_backend_cell_op)(&d));
}

inline Array cast_to(const Array& a, DType to, std::uint64_t stream)
{
    if (to == a.dtype()) return copy_of(a, stream);
    Array t = Array::empty(a.shape(), to, a.placement(), stream);
    cast_into(t, a, stream);
    return t;
}

// ---- host reads --------------------------------------------------------------------------------------

inline nb::object read_item(const char* p, DType d)
{
    switch (d) {
    case DType::Bool: return nb::bool_(*reinterpret_cast<const std::uint8_t*>(p) != 0);
    case DType::U8: return nb::int_(*reinterpret_cast<const std::uint8_t*>(p));
    case DType::I8: return nb::int_(*reinterpret_cast<const std::int8_t*>(p));
    case DType::I16: return nb::int_(*reinterpret_cast<const std::int16_t*>(p));
    case DType::U16: return nb::int_(*reinterpret_cast<const std::uint16_t*>(p));
    case DType::I32: return nb::int_(*reinterpret_cast<const std::int32_t*>(p));
    case DType::U32: return nb::int_(*reinterpret_cast<const std::uint32_t*>(p));
    case DType::I64: return nb::int_(*reinterpret_cast<const std::int64_t*>(p));
    case DType::U64: return nb::int_(*reinterpret_cast<const std::uint64_t*>(p));
    case DType::F32: return nb::float_(*reinterpret_cast<const float*>(p));
    case DType::F64: return nb::float_(*reinterpret_cast<const double*>(p));
    case DType::F16: return nb::float_(static_cast<float>(*reinterpret_cast<const ea::Half*>(p)));
    default: return nb::float_(static_cast<float>(*reinterpret_cast<const ea::BFloat16*>(p)));
    }
}

/// A Python callable run before eagle waits on the device or reads device data back to the host (see
/// ``eagle._split``: a scatter launch checks its status plane at the next such point). Unset: nothing runs.
inline nb::object& sync_hook()
{
    static nb::object* hook = new nb::object();     // leaked on purpose: it may run at interpreter exit
    return *hook;
}

/// ``stream`` is the raw stream about to be waited on (0: the legacy default stream), or -1 for every stream.
inline void run_sync_hook(std::int64_t stream)
{
    if (sync_hook().is_valid() && !sync_hook().is_none()) sync_hook()(stream);
}

inline Array host_copy(const Array& a)
{
    if (a.placement().host()) return a;
    Array h = Array::empty(a.shape(), a.dtype(), { Placement::Kind::Cpu, 0 });
    copy_into(h, a, a.stream());
    sync(a);
    return h;
}

inline nb::object tolist_rec(const Array& a)
{
    if (a.rank() == 0) return read_item(static_cast<const char*>(a.data()), a.dtype());
    nb::list l;
    for (std::int64_t i = 0; i < a.extent(0); ++i) l.append(tolist_rec(a.index(i)));
    return std::move(l);
}

inline nb::object item_of_array(const Array& a)
{
    if (a.size() != 1) throw nb::value_error("the array has more than one element");
    const Array h = host_copy(a);
    return read_item(static_cast<const char*>(h.data()), h.dtype());
}

// ---- reductions and compaction: device ops with device results ---------------------------------------------------

inline int reduce_code(DType d) { return d == DType::F32 ? 0 : d == DType::F64 ? 1 : d == DType::I32 ? 2 : d == DType::I64 ? 3 : -1; }

template <class T>
inline T host_sum(const void* p, std::int64_t n)
{
    struct In {
        const T* p;
        T operator()(eagle::idx_t i) const { return p[i]; }
    };
    return eagle::reduce::canonical::reduce<T, aether::SumOp<T>>(In { static_cast<const T*>(p) }, static_cast<eagle::idx_t>(n));
}

inline std::int64_t reduce_work_elems(const Placement& pl, std::int64_t n, int code)
{
    if (pl.host() || n < 1) return 0;
    auto d = makeDesc<eagle_backend_reduce_desc>(pl.device);
    d.dtype = static_cast<std::uint32_t>(code);
    d.op = EAGLE_BACKEND_REDUCE_OP_SUM;
    d.n = n;
    d.columns = 1;
    ck(EBX(eagle_backend_x_reduce_canonical)(&d));
    return d.work_elems;
}

inline void zero_cell(const Array& cell, std::uint64_t stream)
{
    Call c { Op::Fill };
    c.dtype = static_cast<int>(cell.dtype());
    c.n = 1;
    c.p0 = cell.data();
    run_call(cell.placement(), stream, c);
}

/// ``out`` (one element, the dtype of ``a``: f32, f64, i32 or i64) <- the canonical sum of the contiguous ``a``.
/// Enqueue only on the device (the work buffers come from the caller); the host twin is the same reduction.
inline void sum_into(const Array& out, const Array& a, const Array* work_a, const Array* work_b, std::uint32_t policy, std::uint64_t stream)
{
    const int code = reduce_code(a.dtype());
    if (code < 0 || out.dtype() != a.dtype() || out.size() != 1) throw std::invalid_argument("eagle sum: f32, f64, i32 or i64 into one element of the same dtype");
    if (!a.contiguous()) throw std::invalid_argument("eagle sum: the input must be contiguous");
    if (a.size() == 0) { zero_cell(out, stream); return; }
    if (a.placement().host()) {
        nb::gil_scoped_release release;
        switch (a.dtype()) {
        case DType::F32: *static_cast<float*>(out.data()) = host_sum<float>(a.data(), a.size()); break;
        case DType::F64: *static_cast<double*>(out.data()) = host_sum<double>(a.data(), a.size()); break;
        case DType::I32: *static_cast<std::int32_t*>(out.data()) = host_sum<std::int32_t>(a.data(), a.size()); break;
        default: *static_cast<std::int64_t*>(out.data()) = host_sum<std::int64_t>(a.data(), a.size()); break;
        }
        return;
    }
    if (work_a == nullptr || work_b == nullptr || static_cast<std::int64_t>(work_a->size()) < reduce_work_elems(a.placement(), a.size(), code) || work_a->dtype() != a.dtype())
        throw std::invalid_argument("eagle sum: the device work buffers must be two arrays of the input dtype holding reduce_work_elems(n) each");
    auto d = makeDesc<eagle_backend_reduce_desc>(a.placement().device);
    d.dtype = static_cast<std::uint32_t>(code);
    d.op = EAGLE_BACKEND_REDUCE_OP_SUM;
    d.in = reinterpret_cast<std::uintptr_t>(a.data());
    d.out = reinterpret_cast<std::uintptr_t>(out.data());
    d.work_a = reinterpret_cast<std::uintptr_t>(work_a->data());
    d.work_b = reinterpret_cast<std::uintptr_t>(work_b->data());
    d.n = a.size();
    d.stream = stream;
    d.policy = policy;
    d.columns = 1;
    ck(EBX(eagle_backend_x_reduce_canonical)(&d));
}

/// The sum of ``a`` as a new 0-d device (or host) array: bool, u8, u32, u64 sum as int64, f16 and bf16 as float32.
inline Array sum_array(const Array& a, std::uint64_t stream)
{
    DType t = a.dtype();
    if (t == DType::Bool || t == DType::U8 || t == DType::U32 || t == DType::U64) t = DType::I64;
    else if (t == DType::F16 || t == DType::BF16) t = DType::F32;
    const Array src = t == a.dtype() ? dense(a, stream) : cast_to(a, t, stream);
    if (reduce_code(t) < 0) throw std::invalid_argument(std::string("eagle sum: dtype ") + ea::name(a.dtype()) + " is not summed");
    Array out = Array::empty({}, t, src.placement(), stream);
    const std::int64_t elems = reduce_work_elems(src.placement(), src.size(), reduce_code(t));
    Array wa, wb;
    if (elems > 0) { wa = Array::empty({ elems }, t, src.placement(), stream); wb = Array::empty({ elems }, t, src.placement(), stream); }
    sum_into(out, src, &wa, &wb, EAGLE_BACKEND_REDUCE_COMPENSATED, stream);
    return out;
}

// ---- python values -> arrays ---------------------------------------------------------------------------------

inline void store_item(char* p, DType d, nb::handle v)
{
    if (ea::is_float(d)) {
        const double x = nb::cast<double>(v);
        switch (d) {
        case DType::F32: *reinterpret_cast<float*>(p) = static_cast<float>(x); break;
        case DType::F64: *reinterpret_cast<double*>(p) = x; break;
        case DType::F16: *reinterpret_cast<ea::Half*>(p) = ea::Half(x); break;
        default: *reinterpret_cast<ea::BFloat16*>(p) = ea::BFloat16(x); break;
        }
        return;
    }
    nb::object as_int = nb::steal(PyNumber_Long(v.ptr()));
    if (!as_int.is_valid()) throw nb::python_error();
    const std::uint64_t bits = PyLong_AsUnsignedLongLongMask(as_int.ptr());
    switch (ea::itemsize(d)) {
    case 1: *reinterpret_cast<std::uint8_t*>(p) = d == DType::Bool ? (bits != 0) : static_cast<std::uint8_t>(bits); break;
    case 2: *reinterpret_cast<std::uint16_t*>(p) = static_cast<std::uint16_t>(bits); break;
    case 4: *reinterpret_cast<std::uint32_t*>(p) = static_cast<std::uint32_t>(bits); break;
    default: *reinterpret_cast<std::uint64_t*>(p) = bits; break;
    }
}

inline void shape_of_list(nb::handle h, std::vector<std::int64_t>& shape, std::size_t depth)
{
    if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        const std::int64_t n = static_cast<std::int64_t>(nb::len(h));
        if (depth == shape.size()) shape.push_back(n);
        else if (shape[depth] != n) throw nb::value_error("array: ragged nested sequence");
        if (n) shape_of_list(h[0], shape, depth + 1);
    }
}

/// ``a`` <- the Python scalar ``value``: one fill kernel when ``a`` is contiguous (nothing allocated), else a fill of
/// a temporary and a strided copy.
inline void fill_scalar(const Array& a, nb::handle value, std::uint64_t st)
{
    Call c { Op::Fill };
    c.dtype = static_cast<int>(a.dtype());
    if (ea::is_float(a.dtype())) c.f0 = nb::cast<double>(value);
    else { char buf[8]; store_item(buf, DType::I64, value); std::memcpy(&c.i0, buf, 8); if (a.dtype() == DType::Bool) c.i0 = c.i0 != 0; }
    if (a.contiguous()) { c.n = a.size(); c.p0 = a.data(); run_call(a.placement(), st, c); return; }
    Array tmp = Array::empty(a.shape(), a.dtype(), a.placement(), st);
    c.n = tmp.size(); c.p0 = tmp.data();
    run_call(a.placement(), st, c);
    copy_into(a, tmp, st);
}

inline void fill_from_list(char*& out, nb::handle h, DType d, std::size_t depth, std::size_t rank)
{
    if (depth == rank) { store_item(out, d, h); out += ea::itemsize(d); return; }
    for (nb::handle e : h) fill_from_list(out, e, d, depth + 1, rank);
}

inline DType infer_dtype(nb::handle h)
{
    if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        DType d = DType::Bool;
        bool any = false;
        for (nb::handle e : h) { const DType k = infer_dtype(e); d = any ? ea::promote(d, k) : k; any = true; }
        return any ? d : DType::F64;
    }
    if (PyBool_Check(h.ptr())) return DType::Bool;
    if (PyLong_Check(h.ptr())) return DType::I64;
    return DType::F64;
}

inline Array array_from_list(nb::handle values, std::optional<nb::handle> dtype, const Placement& pl)
{
    std::vector<std::int64_t> shape;
    shape_of_list(values, shape, 0);
    const DType d = dtype ? parse_dtype(*dtype) : infer_dtype(values);
    Array h = Array::empty(shape, d);
    char* out = static_cast<char*>(h.data());
    fill_from_list(out, values, d, 0, shape.size());
    if (pl.host() && pl.kind == Placement::Kind::Cpu) return h;
    Array t = Array::empty(shape, d, pl);
    copy_into(t, h, 0);
    sync(t);
    return t;
}

// ---- DLPack, CAI, buffer protocol -------------------------------------------------------------------------------

inline DLDataType to_dl(DType d)
{
    const int bits = static_cast<int>(ea::itemsize(d)) * 8;
    uint8_t code = kDLUInt;
    if (d == DType::Bool) code = kDLBool;
    else if (d == DType::BF16) code = kDLBfloat;
    else if (ea::is_float(d)) code = kDLFloat;
    else if (ea::is_signed_int(d)) code = kDLInt;
    return DLDataType { code, static_cast<uint8_t>(bits), 1 };
}

inline DType from_dl(DLDataType t)
{
    if (t.lanes != 1) throw nb::value_error("DLPack: vector lanes are not supported");
    for (int c = 0; c < ea::kDTypeCount; ++c) {
        const DLDataType k = to_dl(static_cast<DType>(c));
        if (k.code == t.code && k.bits == t.bits) return static_cast<DType>(c);
    }
    if (t.code == kDLComplex)
        throw nb::value_error(("complex" + std::to_string(t.bits) + " is not an eagle dtype; pass a float" + std::to_string(t.bits / 2) + " view of (re, im) pairs").c_str());
    throw nb::value_error("DLPack: unsupported dtype");
}

inline DLDevice dl_device(const Placement& p)
{
    return DLDevice { p.kind == Placement::Kind::Cpu ? kDLCPU : p.kind == Placement::Kind::Pinned ? kDLCUDAHost : kDLCUDA, p.device };
}

inline Placement from_dl_device(DLDevice d)
{
    if (d.device_type == kDLCPU) return { Placement::Kind::Cpu, 0 };
    if (d.device_type == kDLCUDAHost) return { Placement::Kind::Pinned, 0 };
    if (d.device_type == kDLCUDA) return { Placement::Kind::Cuda, d.device_id };
    throw nb::value_error("DLPack: only cpu, pinned and CUDA devices are supported");
}

/// What an exported tensor owns: a view (keeps the storage alive), its shape and element strides.
struct Exported {
    Array hold;
    std::vector<std::int64_t> shape, strides;
    DLManagedTensorVersioned v {};
    DLManagedTensor legacy {};
};

inline void fill_tensor(DLTensor& t, Exported& e)
{
    t.data = e.hold.data();
    t.device = dl_device(e.hold.placement());
    t.ndim = static_cast<int32_t>(e.hold.rank());
    t.dtype = to_dl(e.hold.dtype());
    t.shape = e.shape.data();
    t.strides = e.strides.data();
    t.byte_offset = 0;
}

inline nb::object export_dlpack(const Array& a, bool versioned, const nb::object& stream)
{
    if (!a.placement().host() && !stream.is_none()) {
        const auto s = nb::cast<std::intptr_t>(stream);
        if (s >= 0) eagle_interop_binding::fence_streams(a.stream() == 0 ? 1 : static_cast<std::intptr_t>(a.stream()), s, a.placement().device);
    }
    auto* e = new Exported { a, a.shape(), {}, {}, {} };
    for (std::size_t k = 0; k < a.rank(); ++k) e->strides.push_back(a.stride(k));
    if (versioned) {
        e->v.version = DLPackVersion { DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION };
        e->v.manager_ctx = e;
        e->v.deleter = [](DLManagedTensorVersioned* t) { delete static_cast<Exported*>(t->manager_ctx); };
        fill_tensor(e->v.dl_tensor, *e);
        return nb::steal(PyCapsule_New(&e->v, eagle_interop_binding::kVersionedName, eagle_interop_binding::versioned_capsule_destructor));
    }
    e->legacy.manager_ctx = e;
    e->legacy.deleter = [](DLManagedTensor* t) { delete static_cast<Exported*>(t->manager_ctx); };
    fill_tensor(e->legacy.dl_tensor, *e);
    return nb::steal(PyCapsule_New(&e->legacy, eagle_interop_binding::kLegacyName, eagle_interop_binding::legacy_capsule_destructor));
}

inline Array import_tensor(DLTensor& t, std::function<void()> release)
{
    std::vector<std::int64_t> shape(t.shape, t.shape + t.ndim), strides(static_cast<std::size_t>(t.ndim));
    std::int64_t acc = 1;
    for (int k = t.ndim; k-- > 0;) { strides[static_cast<std::size_t>(k)] = t.strides ? t.strides[k] : acc; acc *= shape[static_cast<std::size_t>(k)] > 0 ? shape[static_cast<std::size_t>(k)] : 1; }
    auto keep = std::shared_ptr<void>(nullptr, [release](void*) { nb::gil_scoped_acquire g; if (release) release(); });
    return Array::borrow(static_cast<char*>(t.data) + t.byte_offset, shape, strides, from_dl(t.dtype), from_dl_device(t.device), keep);
}

/// ``stream`` (a raw handle, or None) is passed to the producer's ``__dlpack__(stream=)``: the consumer stream of the DLPack contract.
inline Array array_from_dlpack(nb::object obj, nb::object stream = nb::none())
{
    nb::object cap;
    if (PyCapsule_CheckExact(obj.ptr())) cap = obj;
    else {
        nb::dict kw;
        if (!stream.is_none()) kw["stream"] = stream;
        nb::dict versioned(kw);
        versioned["max_version"] = nb::make_tuple(1, 0);
        try { cap = obj.attr("__dlpack__")(**versioned); }
        catch (nb::python_error& e) {
            if (!e.matches(PyExc_TypeError)) throw;
            cap = obj.attr("__dlpack__")(**kw);
        }
    }
    const char* nm = PyCapsule_GetName(cap.ptr());
    if (nm && std::strcmp(nm, eagle_interop_binding::kVersionedName) == 0) {
        auto* t = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(cap.ptr(), nm));
        if (t->version.major != DLPACK_MAJOR_VERSION) throw nb::value_error("DLPack: unsupported major version");
        PyCapsule_SetName(cap.ptr(), "used_dltensor_versioned");
        return import_tensor(t->dl_tensor, [t] { if (t->deleter) t->deleter(t); });
    }
    if (nm && std::strcmp(nm, eagle_interop_binding::kLegacyName) == 0) {
        auto* t = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), nm));
        PyCapsule_SetName(cap.ptr(), "used_dltensor");
        return import_tensor(t->dl_tensor, [t] { if (t->deleter) t->deleter(t); });
    }
    throw nb::value_error("DLPack: not a dltensor capsule");
}

inline std::shared_ptr<void> keep_object(nb::handle obj)
{
    return std::shared_ptr<void>(new nb::object(nb::borrow(obj)), [](void* p) { nb::gil_scoped_acquire g; delete static_cast<nb::object*>(p); });
}

inline Array array_from_cai(nb::object obj)
{
    nb::dict cai = nb::cast<nb::dict>(obj.attr("__cuda_array_interface__"));
    void* ptr = reinterpret_cast<void*>(nb::cast<std::uintptr_t>(nb::cast<nb::tuple>(cai["data"])[0]));
    const std::string ts = nb::cast<std::string>(cai["typestr"]);
    int dt = -1;
    for (int c = 0; c < ea::kDTypeCount; ++c) if (ts == ea::kTypestr[c]) { dt = c; break; }
    if (dt < 0) throw nb::value_error(("__cuda_array_interface__: typestr '" + ts + "' is not supported").c_str());
    const DType d = static_cast<DType>(dt);
    const auto shape = nb::cast<std::vector<std::int64_t>>(cai["shape"]);
    std::vector<std::int64_t> strides(shape.size());
    std::int64_t acc = 1;
    for (std::size_t k = shape.size(); k-- > 0;) { strides[k] = acc; acc *= shape[k] > 0 ? shape[k] : 1; }
    if (cai.contains("strides") && !cai["strides"].is_none()) {
        const auto s = nb::cast<std::vector<std::int64_t>>(cai["strides"]);
        for (std::size_t k = 0; k < s.size(); ++k) strides[k] = s[k] / static_cast<std::int64_t>(ea::itemsize(d));
    }
    std::int32_t dev = 0;
    ck(EBX(eagle_backend_pointer_device)(reinterpret_cast<std::uintptr_t>(ptr), &dev));
    return Array::borrow(ptr, shape, strides, d, { Placement::Kind::Cuda, dev < 0 ? 0 : dev }, keep_object(obj));
}

inline DType dtype_of_format(const char* f)
{
    while (*f == '@' || *f == '<' || *f == '=') ++f;
    if (*f == 'l') return DType::I64;
    if (*f == 'L') return DType::U64;
    if (*f == 'H') return DType::U16;
    if (*f && f[1] == '\0')
        for (int c = 0; c < ea::kDTypeCount; ++c) if (ea::kFormat[c] == *f) return static_cast<DType>(c);
    if (f[0] == 'Z' && (f[1] == 'f' || f[1] == 'd'))
        throw nb::value_error(f[1] == 'd' ? "complex128 is not an eagle dtype; pass a float64 view of (re, im) pairs"
                                          : "complex64 is not an eagle dtype; pass a float32 view of (re, im) pairs");
    throw nb::value_error(("buffer format '" + std::string(f) + "' is not supported").c_str());
}

inline Array array_from_buffer(nb::object obj)
{
    auto* view = new Py_buffer {};
    if (PyObject_GetBuffer(obj.ptr(), view, PyBUF_FULL) != 0) {
        PyErr_Clear();
        if (PyObject_GetBuffer(obj.ptr(), view, PyBUF_FULL_RO) != 0) { delete view; throw nb::python_error(); }
    }
    auto keep = std::shared_ptr<void>(view, [](void* p) { nb::gil_scoped_acquire g; PyBuffer_Release(static_cast<Py_buffer*>(p)); delete static_cast<Py_buffer*>(p); });
    const DType d = dtype_of_format(view->format ? view->format : "B");
    std::vector<std::int64_t> shape(view->shape, view->shape + view->ndim), strides;
    for (int k = 0; k < view->ndim; ++k) {
        if (view->strides[k] % view->itemsize) throw nb::value_error("buffer: strides must be a multiple of the itemsize");
        strides.push_back(view->strides[k] / view->itemsize);
    }
    if (static_cast<std::size_t>(view->itemsize) != ea::itemsize(d)) throw nb::value_error("buffer: itemsize does not match its format");
    return Array::borrow(view->buf, shape, strides, d, { Placement::Kind::Cpu, 0 }, keep);
}

inline constexpr const char* kFmtStr[ea::kDTypeCount] = { "?", "B", "i", "I", "q", "Q", "f", "d", "e", "H", "b", "h", "H" };

inline int getbuffer(PyObject* self, Py_buffer* buf, int)
{
    Array* a = nb::inst_ptr<Array>(self);
    if (!a->placement().host()) { PyErr_SetString(PyExc_BufferError, "device memory has no host buffer"); buf->obj = nullptr; return -1; }
    const std::size_t r = a->rank(), it = item_of(*a);
    auto* mem = new Py_ssize_t[2 * (r ? r : 1)];
    for (std::size_t i = 0; i < r; ++i) { mem[i] = a->extent(i); mem[r + i] = a->stride(i) * static_cast<Py_ssize_t>(it); }
    buf->buf = a->data();
    buf->obj = self;
    Py_INCREF(self);
    buf->len = static_cast<Py_ssize_t>(a->nbytes());
    buf->readonly = 0;
    buf->itemsize = static_cast<Py_ssize_t>(it);
    buf->format = const_cast<char*>(kFmtStr[static_cast<int>(a->dtype())]);
    buf->ndim = static_cast<int>(r);
    buf->shape = mem;
    buf->strides = mem + r;
    buf->suboffsets = nullptr;
    buf->internal = mem;
    return 0;
}

inline void releasebuffer(PyObject*, Py_buffer* buf) { delete[] static_cast<Py_ssize_t*>(buf->internal); }

inline PyType_Slot g_buffer_slots[] = { { Py_bf_getbuffer, reinterpret_cast<void*>(getbuffer) }, { Py_bf_releasebuffer, reinterpret_cast<void*>(releasebuffer) }, { 0, nullptr } };

// ---- indexing ------------------------------------------------------------------------------------------------------

inline Array getitem(const Array& a, nb::handle key)
{
    std::vector<nb::handle> keys;
    if (nb::isinstance<nb::tuple>(key)) for (nb::handle k : key) keys.push_back(k);
    else keys.push_back(key);
    std::size_t consumed = 0;
    for (auto k : keys) if (k.ptr() != Py_Ellipsis) ++consumed;
    if (consumed > a.rank()) throw nb::index_error("too many indices for the array");
    Array cur = a;
    std::size_t axis = 0;
    for (auto k : keys) {
        if (k.ptr() == Py_Ellipsis) { axis += a.rank() - consumed; continue; }
        if (nb::isinstance<nb::slice>(k)) {
            Py_ssize_t start = 0, stop = 0, step = 1;
            if (PySlice_Unpack(k.ptr(), &start, &stop, &step) != 0) throw nb::python_error();
            cur = cur.slice(axis, start, stop, step);   // Array::slice clamps with Python semantics
            ++axis;
        } else {
            try { cur = cur.select(axis, nb::cast<std::int64_t>(k)); }
            catch (const std::out_of_range& e) { throw nb::index_error(e.what()); }
        }
    }
    return cur;
}

// ---- the binding ------------------------------------------------------------------------------------------

inline void install_allocator()
{
    auto cpu = ea::allocator();
    ea::allocator() = [cpu](Placement p, std::size_t bytes, std::uint64_t stream) {
        if (p.kind == Placement::Kind::Cpu) return cpu(p, bytes, stream);
        auto s = std::make_shared<ea::Storage>();
        std::uint64_t chunk = 0, ptr = 0;
        ck(EBX(eagle_backend_mem_alloc)(p.kind == Placement::Kind::Cuda ? EAGLE_BACKEND_MEM_DEVICE : EAGLE_BACKEND_MEM_PINNED,
            p.kind == Placement::Kind::Cuda ? p.device : -1, bytes, stream, &chunk, &ptr));
        s->base = reinterpret_cast<void*>(ptr);
        s->bytes = bytes;
        s->place = p;
        s->release = [chunk] { EBX(eagle_backend_mem_release)(chunk); };
        return s;
    };
}

inline void bind(nb::module_& m)
{
    install_allocator();

    nb::class_<PyDType>(m, "DType")
        .def_prop_ro("name", [](const PyDType& d) { return std::string(ea::name(d.d)); })
        .def_prop_ro("itemsize", [](const PyDType& d) { return ea::itemsize(d.d); })
        .def_prop_ro("kind", [](const PyDType& d) { return std::string(1, ea::kind(d.d)); })
        .def_prop_ro("code", [](const PyDType& d) { return static_cast<int>(d.d); })
        .def_prop_ro("typestr", [](const PyDType& d) { return std::string(ea::kTypestr[static_cast<int>(d.d)]); })
        .def("__str__", [](const PyDType& d) { return std::string(ea::name(d.d)); })
        .def("__repr__", [](const PyDType& d) { return std::string("eagle.") + ea::name(d.d); })
        .def("__hash__", [](const PyDType& d) { return nb::hash(nb::str(ea::name(d.d))); })
        .def("__eq__", [](const PyDType& d, nb::handle o) {
            try { return parse_dtype(o) == d.d; } catch (...) { PyErr_Clear(); return false; } })
        .def("__ne__", [](const PyDType& d, nb::handle o) {
            try { return !(parse_dtype(o) == d.d); } catch (...) { PyErr_Clear(); return true; } })
        ;
    // keep ``obj`` (a framework stream object) alive until the storage is released: the stream-ordered free of the
    // bytes is enqueued first, then the object may go
    m.def("array_hold", [](const Array& a, nb::object obj) {
        auto* s = const_cast<ea::Storage*>(a.storage().get());
        if (s == nullptr) return;
        auto held = std::make_shared<nb::object>(std::move(obj));
        s->release = [old = std::move(s->release), held]() mutable {
            if (old) old();
            nb::gil_scoped_acquire g;
            held.reset();
        };
    }, nb::arg("array"), nb::arg("obj"));
    m.def("array_promote", [](nb::handle a, nb::handle b) { try { return PyDType { ea::promote(parse_dtype(a), parse_dtype(b)) }; } catch (const std::invalid_argument& e) { throw nb::type_error(e.what()); } });
    m.def("array_dtype", [](int code) { return PyDType { ea::from_code(code) }; }, nb::arg("code"));

    auto array_class = nb::class_<Array>(m, "Array", nb::type_slots(g_buffer_slots))
        .def_prop_ro("shape", [](const Array& a) { nb::list l; for (auto e : a.shape()) l.append(e); return nb::tuple(l); })
        .def_prop_ro("strides", [](const Array& a) { nb::list l; for (std::size_t i = 0; i < a.rank(); ++i) l.append(a.stride(i) * static_cast<std::int64_t>(item_of(a))); return nb::tuple(l); })
        .def_prop_ro("dtype", [](const Array& a) { return PyDType { a.dtype() }; })
        .def_prop_ro("ndim", &Array::rank)
        .def_prop_ro("size", &Array::size)
        .def_prop_ro("nbytes", &Array::nbytes)
        .def_prop_ro("itemsize", [](const Array& a) { return item_of(a); })
        .def_prop_ro("ptr", [](const Array& a) { return reinterpret_cast<std::uintptr_t>(a.data()); })
        .def_prop_ro("device", [](const Array& a) { return place_str(a.placement()); })
        .def_prop_ro("stream", &Array::stream)
        .def_prop_ro("is_contiguous", &Array::contiguous)
        .def_prop_ro("T", &Array::transpose)
        .def("transpose", &Array::transpose)
        .def("reshape", [](const Array& a, const std::vector<std::int64_t>& shape) {
            try { return a.reshape(shape); } catch (const std::invalid_argument& e) { throw nb::value_error(e.what()); } }, nb::arg("shape"))
        .def("ravel", [](const Array& a) { return a.contiguous() ? a.reshape({ a.size() }) : dense(a, a.stream()).reshape({ a.size() }); })
        .def("view", [](const Array& a, nb::handle dt) { return a.view_as(parse_dtype(dt)); }, nb::arg("dtype"))
        .def("astype", [](const Array& a, nb::handle dt, std::optional<std::uint64_t> stream) { return cast_to(a, parse_dtype(dt), stream_of(a, stream)); },
            nb::arg("dtype"), nb::arg("stream") = nb::none())
        .def("copy", [](const Array& a, std::optional<std::uint64_t> stream) { return copy_of(a, stream_of(a, stream)); }, nb::arg("stream") = nb::none())
        .def("get", [](const Array& a) { if (!a.placement().host()) run_sync_hook(static_cast<std::int64_t>(a.stream())); return host_copy(a); })
        .def("set", [](Array& a, nb::object src) {
            Array s = nb::isinstance<Array>(src) ? nb::cast<Array>(src) : array_from_buffer(src);
            copy_into(a, s, a.stream());
            if (!a.placement().host()) sync(a); })
        .def("synchronize", [](const Array& a) { sync(a); })
        .def("sum", [](const Array& a, std::optional<std::uint64_t> stream) { return sum_array(a, stream_of(a, stream)); }, nb::arg("stream") = nb::none())
        .def("item", &item_of_array)
        .def("__int__", [](const Array& a) { return nb::steal(PyNumber_Long(item_of_array(a).ptr())); })
        .def("__float__", [](const Array& a) { return nb::steal(PyNumber_Float(item_of_array(a).ptr())); })
        .def("__bool__", [](const Array& a) {
            if (a.size() != 1) throw nb::value_error("The truth value of an array with more than one element is ambiguous");
            return nb::bool_(PyObject_IsTrue(item_of_array(a).ptr()) == 1); })
        .def("__index__", [](const Array& a) {
            if (ea::is_float(a.dtype())) throw nb::type_error("only integer arrays can be an index");
            return nb::steal(PyNumber_Index(item_of_array(a).ptr())); })
        .def("tolist", [](const Array& a) { if (!a.placement().host()) run_sync_hook(static_cast<std::int64_t>(a.stream())); return tolist_rec(host_copy(a)); })
        .def("__eq__", [](nb::handle a, nb::handle o) { return nb::module_::import_("eagle.arrays").attr("_compare")(a, o, "eq"); }, nb::arg("o").none())
        .def("__ne__", [](nb::handle a, nb::handle o) { return nb::module_::import_("eagle.arrays").attr("_compare")(a, o, "ne"); }, nb::arg("o").none())
        .def("__lt__", [](nb::handle a, nb::handle o) { return nb::module_::import_("eagle.arrays").attr("_compare")(a, o, "lt"); }, nb::arg("o").none())
        .def("__le__", [](nb::handle a, nb::handle o) { return nb::module_::import_("eagle.arrays").attr("_compare")(a, o, "le"); }, nb::arg("o").none())
        .def("__gt__", [](nb::handle a, nb::handle o) { return nb::module_::import_("eagle.arrays").attr("_compare")(a, o, "gt"); }, nb::arg("o").none())
        .def("__ge__", [](nb::handle a, nb::handle o) { return nb::module_::import_("eagle.arrays").attr("_compare")(a, o, "ge"); }, nb::arg("o").none())
        .def("__len__", [](const Array& a) { if (a.rank() == 0) throw nb::type_error("len() of a 0-d array"); return a.extent(0); })
        .def("__repr__", [](const Array& a) {
            std::string s = "eagle.Array(shape=(";
            for (std::size_t i = 0; i < a.rank(); ++i) s += std::to_string(a.extent(i)) + (a.rank() == 1 ? "," : i + 1 < a.rank() ? ", " : "");
            return s + "), dtype=" + ea::name(a.dtype()) + ", device=" + place_str(a.placement()) + ")"; })
        .def("__getitem__", [](const Array& a, nb::handle key) { return getitem(a, key); })
        .def("__setitem__", [](Array& a, nb::handle key, nb::object value) {
            Array v = getitem(a, key);
            const std::uint64_t st = stream_of(a, std::nullopt);
            const bool foreign = !nb::isinstance<Array>(value) && !PyObject_CheckBuffer(value.ptr()) && (nb::hasattr(value, "__cuda_array_interface__") || nb::hasattr(value, "__dlpack__"));
            if (nb::isinstance<Array>(value) || foreign || PyObject_CheckBuffer(value.ptr())) {
                Array s = nb::isinstance<Array>(value) ? nb::cast<Array>(value)
                        : foreign ? nb::cast<Array>(nb::module_::import_("eagle.arrays").attr("asarray")(value)) : array_from_buffer(value);
                if (s.dtype() != v.dtype()) s = cast_to(s, v.dtype(), st);
                copy_into(v, s, st);
            } else if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
                copy_into(v, array_from_list(value, nb::cast(PyDType { v.dtype() }), v.placement()), st);
            } else {
                fill_scalar(v, value, st);
            } })
        .def("fill", [](Array& a, nb::object value, std::optional<std::uint64_t> stream) { fill_scalar(a, value, stream_of(a, stream)); },
            nb::arg("value"), nb::arg("stream") = nb::none())
        .def("__dlpack_device__", [](const Array& a) { const DLDevice d = dl_device(a.placement()); return nb::make_tuple(static_cast<int>(d.device_type), d.device_id); })
        .def("__dlpack__", [](const Array& a, nb::object stream, nb::object max_version, nb::object, nb::object) {
            bool versioned = false;
            if (!max_version.is_none()) versioned = nb::cast<int>(max_version[0]) >= 1;
            return export_dlpack(a, versioned, stream); },
            nb::arg("stream").none() = nb::none(), nb::arg("max_version").none() = nb::none(), nb::arg("dl_device").none() = nb::none(), nb::arg("copy").none() = nb::none())
        .def_prop_ro("__cuda_array_interface__", [](const Array& a) {
            if (a.placement().kind != Placement::Kind::Cuda) throw nb::attribute_error("only device arrays have __cuda_array_interface__");
            nb::dict d;
            nb::list sh, st;
            for (std::size_t i = 0; i < a.rank(); ++i) { sh.append(a.extent(i)); st.append(a.stride(i) * static_cast<std::int64_t>(item_of(a))); }
            d["shape"] = nb::tuple(sh);
            d["strides"] = a.contiguous() ? nb::object(nb::none()) : nb::object(nb::tuple(st));
            d["typestr"] = ea::kTypestr[static_cast<int>(a.dtype())];
            d["data"] = nb::make_tuple(reinterpret_cast<std::uintptr_t>(a.data()), false);
            d["version"] = 3;
            if (a.stream() != 0) d["stream"] = a.stream();
            return d; });
    array_class.attr("__hash__") = nb::none();

    m.def("array_empty", [](const std::vector<std::int64_t>& shape, nb::handle dtype, const std::string& device, std::uint64_t stream) {
        return Array::empty(shape, parse_dtype(dtype), parse_place(device), stream); },
        nb::arg("shape"), nb::arg("dtype") = "float64", nb::arg("device") = "cpu", nb::arg("stream") = 0);
    // a view of ``base``'s memory with another shape and element strides (a block's rows may alias); the caller keeps
    // every addressed element inside the storage
    m.def("array_view_as", [](nb::object base, const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& strides) {
        const Array& a = nb::cast<const Array&>(base);
        return Array::borrow(a.data(), shape, strides, a.dtype(), a.placement(), keep_object(base)); },
        nb::arg("base"), nb::arg("shape"), nb::arg("strides"));
    m.def("array_from_dlpack", &array_from_dlpack, nb::arg("obj"), nb::arg("stream") = nb::none());
    m.def("array_from_cai", &array_from_cai, nb::arg("obj"));
    m.def("array_from_buffer", &array_from_buffer, nb::arg("obj"));
    m.def("array_from_list", [](nb::handle values, nb::handle dtype, const std::string& device) {
        return array_from_list(values, dtype.is_none() ? std::nullopt : std::optional<nb::handle>(dtype), parse_place(device)); },
        nb::arg("values"), nb::arg("dtype").none() = nb::none(), nb::arg("device") = "cpu");
    m.def("array_copyto", [](Array& dst, const Array& src, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(dst, stream);
        if (src.dtype() == dst.dtype()) copy_into(dst, src, st);
        else if (dst.contiguous() && dst.shape() == src.shape() && dst.placement() == src.placement()) cast_into(dst, src, st);
        else copy_into(dst, cast_to(src, dst.dtype(), st), st); },
        nb::arg("dst"), nb::arg("src"), nb::arg("stream") = nb::none());
    m.def("array_deferred_frees", [] { std::int64_t left = 0; ck(EBX(eagle_backend_mem_drain)(&left)); return left; });

    // elementwise: out = op(a, b); the inputs may alias `out` (same-index reads); strided operands are gathered first
    const auto bin = [](std::uint32_t op, Array& out, const Array& a, const Array& b, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        if (a.size() != out.size() || b.size() != out.size()) throw nb::value_error("elementwise op: operand sizes differ");
        if (a.dtype() != b.dtype()) throw nb::value_error("elementwise op: operand dtypes differ (astype first)");
        const bool cmp = op == EAGLE_BACKEND_CELL_EQ || op == EAGLE_BACKEND_CELL_NE || op == EAGLE_BACKEND_CELL_GT;
        if (cmp ? (out.dtype() != DType::Bool && out.dtype() != DType::U8) : out.dtype() != a.dtype())
            throw nb::value_error("elementwise op: the output dtype must be the operand dtype (bool or uint8 for a comparison)");
        const Array da = dense(a, st), db = dense(b, st);
        Array o = out.contiguous() ? out : Array::empty(out.shape(), out.dtype(), out.placement(), st);
        Call c { static_cast<Op>(op) };
        c.dtype = static_cast<int>(a.dtype()); c.n = o.size(); c.p0 = o.data(); c.p1 = da.data(); c.p2 = db.data();
        run_call(o.placement(), st, c);
        if (!out.contiguous()) copy_into(out, o, st); };
    m.def("array_binary", bin, nb::arg("op"), nb::arg("out"), nb::arg("a"), nb::arg("b"), nb::arg("stream") = nb::none());
    // out = a (op) scalar: the scalar rides in the op's arguments, so no operand array is allocated (capture-legal)
    m.def("array_binary_scalar", [](std::uint32_t op, Array& out, const Array& a, nb::handle value, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        if (a.size() != out.size()) throw nb::value_error("elementwise op: operand sizes differ");
        const bool cmp = op == EAGLE_BACKEND_CELL_EQS || op == EAGLE_BACKEND_CELL_NES || op == EAGLE_BACKEND_CELL_GTS || op == EAGLE_BACKEND_CELL_LTS;
        if (cmp ? (out.dtype() != DType::Bool && out.dtype() != DType::U8) : out.dtype() != a.dtype())
            throw nb::value_error("elementwise op: the output dtype must be the operand dtype (bool or uint8 for a comparison)");
        const Array da = dense(a, st);
        Array o = out.contiguous() ? out : Array::empty(out.shape(), out.dtype(), out.placement(), st);
        Call c { static_cast<Op>(op) };
        c.dtype = static_cast<int>(a.dtype()); c.n = o.size(); c.p0 = o.data(); c.p1 = da.data();
        c.f0 = nb::cast<double>(value);
        c.i0 = nb::isinstance<nb::float_>(value) ? static_cast<std::int64_t>(c.f0) : nb::cast<std::int64_t>(value);
        run_call(o.placement(), st, c);
        if (!out.contiguous()) copy_into(out, o, st); },
        nb::arg("op"), nb::arg("out"), nb::arg("a"), nb::arg("value"), nb::arg("stream") = nb::none());
    m.def("array_where", [](Array& out, const Array& mask, const Array& a, const Array& b, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        const Array dm = dense(mask, st), da = dense(a, st), db = dense(b, st);
        if (!out.contiguous()) throw nb::value_error("where: the output must be contiguous");
        Call c { Op::Where };
        c.dtype = static_cast<int>(out.dtype()); c.n = out.size(); c.p0 = out.data(); c.p1 = da.data(); c.p2 = db.data(); c.p3 = dm.data();
        run_call(out.placement(), st, c); },
        nb::arg("out"), nb::arg("mask"), nb::arg("a"), nb::arg("b"), nb::arg("stream") = nb::none());
    m.def("array_clip", [](Array& out, const Array& a, double lo, double hi, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        const Array da = dense(a, st);
        if (!out.contiguous()) throw nb::value_error("clip: the output must be contiguous");
        Call c { Op::Clip };
        c.dtype = static_cast<int>(out.dtype()); c.n = out.size(); c.p0 = out.data(); c.p1 = da.data();
        c.f0 = lo; c.f1 = hi; c.i0 = static_cast<std::int64_t>(lo); c.i1 = static_cast<std::int64_t>(hi);
        run_call(out.placement(), st, c); },
        nb::arg("out"), nb::arg("a"), nb::arg("lo"), nb::arg("hi"), nb::arg("stream") = nb::none());
    m.def("array_take", [](Array& out, const Array& a, const Array& index, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        if (index.dtype() != DType::I32) throw nb::value_error("take: the index must be int32");
        const Array da = dense(a, st), di = dense(index, st);
        if (!out.contiguous() || out.size() != index.size() || out.dtype() != a.dtype()) throw nb::value_error("take: out must be contiguous, of the data dtype and the index size");
        Call c { Op::Take };
        c.dtype = static_cast<int>(out.dtype()); c.n = out.size(); c.p0 = out.data(); c.p1 = da.data(); c.p2 = di.data();
        run_call(out.placement(), st, c); },
        nb::arg("out"), nb::arg("a"), nb::arg("index"), nb::arg("stream") = nb::none());
    m.def("array_scatter", [](Array& out, const Array& index, const Array& src, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        if (index.dtype() != DType::I32) throw nb::value_error("scatter: the index must be int32");
        const Array ds = dense(src, st), di = dense(index, st);
        if (!out.contiguous() || src.dtype() != out.dtype() || src.size() != index.size()) throw nb::value_error("scatter: out must be contiguous, src of its dtype and the index size");
        Call c { Op::Scatter };
        c.dtype = static_cast<int>(out.dtype()); c.n = src.size(); c.p0 = out.data(); c.p1 = ds.data(); c.p2 = di.data();
        run_call(out.placement(), st, c); },
        nb::arg("out"), nb::arg("index"), nb::arg("src"), nb::arg("stream") = nb::none());
    m.def("array_arange", [](Array& out, std::int64_t start, std::int64_t step, std::optional<std::uint64_t> stream) {
        if (!out.contiguous()) throw nb::value_error("arange: the output must be contiguous");
        Call c { Op::Arange };
        c.dtype = static_cast<int>(out.dtype()); c.n = out.size(); c.p0 = out.data(); c.i0 = start; c.i1 = step;
        run_call(out.placement(), stream_of(out, stream), c); },
        nb::arg("out"), nb::arg("start"), nb::arg("step"), nb::arg("stream") = nb::none());
    m.def("array_linspace", [](Array& out, double start, double step, std::optional<std::uint64_t> stream) {
        if (!out.contiguous()) throw nb::value_error("linspace: the output must be contiguous");
        Call c { Op::Linspace };
        c.dtype = static_cast<int>(out.dtype()); c.n = out.size(); c.p0 = out.data(); c.f0 = start; c.f1 = step;
        run_call(out.placement(), stream_of(out, stream), c); },
        nb::arg("out"), nb::arg("start"), nb::arg("step"), nb::arg("stream") = nb::none());
    m.def("array_cumsum", [](Array& out, const Array& a, std::optional<Array> scratch, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        const Array da = dense(a, st);
        if (!out.contiguous() || out.dtype() != a.dtype() || out.size() != a.size()) throw nb::value_error("cumsum: out must be contiguous with the input's dtype and size");
        const Array sc = scratch ? *scratch : (out.placement().host() ? Array {} : Array::empty({ static_cast<std::int64_t>(ea::cumsum_scratch_bytes(a.size())) }, DType::U8, out.placement(), st));
        if (!out.placement().host() && sc.nbytes() < ea::cumsum_scratch_bytes(a.size())) throw nb::value_error("cumsum: the scratch is smaller than cumsum_scratch_bytes(n)");
        Call c { Op::Cumsum };
        c.dtype = static_cast<int>(out.dtype()); c.n = a.size(); c.p0 = out.data(); c.p1 = da.data(); c.p2 = sc.data();
        run_call(out.placement(), st, c); },
        nb::arg("out"), nb::arg("a"), nb::arg("scratch").none() = nb::none(), nb::arg("stream") = nb::none());
    m.def("array_cumsum_scratch_bytes", [](std::int64_t n) { return ea::cumsum_scratch_bytes(n); }, nb::arg("n"));
    m.def("array_count_nonzero", [](Array& out, const Array& a, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(out, stream);
        if (out.dtype() != DType::I64 || out.size() != 1) throw nb::value_error("count_nonzero: out is one int64 element");
        const Array da = dense(a, st);
        if (da.size() == 0) { zero_cell(out, st); return; }
        Call c { Op::CountNz };
        c.dtype = static_cast<int>(a.dtype()); c.n = da.size(); c.p0 = out.data(); c.p1 = da.data();
        run_call(out.placement(), st, c); },
        nb::arg("out"), nb::arg("a"), nb::arg("stream") = nb::none());
    m.def("array_flatnonzero", [](Array& index, Array& count, const Array& a, std::optional<Array> scratch, std::optional<std::uint64_t> stream) {
        const std::uint64_t st = stream_of(index, stream);
        if (index.dtype() != DType::I32 || index.size() < a.size() || !index.contiguous() || count.dtype() != DType::I64 || count.size() != 1)
            throw nb::value_error("flatnonzero: index is int32 with room for a.size() elements, count one int64");
        const Array da = dense(a, st);
        if (da.size() == 0) { zero_cell(count, st); return; }
        const Array sc = scratch ? *scratch : (index.placement().host() ? Array {} : Array::empty({ static_cast<std::int64_t>(ea::cumsum_scratch_bytes(da.size())) }, DType::U8, index.placement(), st));
        if (!index.placement().host() && sc.nbytes() < ea::cumsum_scratch_bytes(da.size())) throw nb::value_error("flatnonzero: the scratch is smaller than cumsum_scratch_bytes(n)");
        Call c { Op::FlatNz };
        c.dtype = static_cast<int>(a.dtype()); c.n = da.size(); c.p0 = index.data(); c.p1 = da.data(); c.p2 = sc.data(); c.p3 = count.data();
        run_call(index.placement(), st, c); },
        nb::arg("index"), nb::arg("count"), nb::arg("a"), nb::arg("scratch").none() = nb::none(), nb::arg("stream") = nb::none());
    m.def("array_sum", [](Array& out, const Array& a, std::optional<Array> work_a, std::optional<Array> work_b, std::uint32_t policy, std::optional<std::uint64_t> stream) {
        sum_into(out, a, work_a ? &*work_a : nullptr, work_b ? &*work_b : nullptr, policy, stream_of(out, stream)); },
        nb::arg("out"), nb::arg("a"), nb::arg("work_a").none() = nb::none(), nb::arg("work_b").none() = nb::none(),
        nb::arg("policy") = EAGLE_BACKEND_REDUCE_COMPENSATED, nb::arg("stream") = nb::none());
    m.def("array_reduce_work_elems", [](const Array& a) { return reduce_work_elems(a.placement(), a.size(), reduce_code(a.dtype())); }, nb::arg("a"));
}

} // namespace eagle_array_binding
