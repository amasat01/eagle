// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Array.h
 * @brief ``eagle::array::Array``: a runtime-typed strided view over shared storage.
 *
 * Rank <= 6, int64 shape and ELEMENT strides (negative allowed), a data pointer = storage base + byte
 * offset, and a ``shared_ptr<Storage>`` that every view shares: the last view releases the bytes. A
 * ``Storage`` is only "bytes plus a release function", so one type serves owned host memory, device
 * memory from the backend (``mem_alloc``/``mem_release``, installed by the binding as the allocator) and
 * ``borrow``ed memory (a DLPack owner, a Python object, a raw keep-alive).
 *
 * Storage paths: Python/_core = the backend seam's ``mem_*`` (installed as ``allocator()``); CUDA-mode C++
 * consumers may own an ``aether::Chunk`` and ``borrow`` it.
 *
 * Views never copy: ``slice``, ``index``, ``reshape`` (contiguous only, else refused by name),
 * ``transpose`` and ``view_as`` (byte reinterpretation of a contiguous array). Dependency-free: no aether,
 * no CUDA, no numpy.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "eagle/array/DType.h"

namespace eagle::array {

inline constexpr std::size_t kMaxRank = 6;

/// Where the bytes live. ``Cpu`` and ``Pinned`` are host-addressable; ``Cuda`` carries a device ordinal.
struct Placement {
    enum class Kind : std::uint8_t { Cpu = 0, Pinned = 1, Cuda = 2 } kind = Kind::Cpu;
    int device = 0;
    bool host() const { return kind != Kind::Cuda; }
    bool operator==(const Placement& o) const { return kind == o.kind && device == o.device; }
};

/// The bytes of one allocation and how to release them (``release`` runs once, when the last view goes).
struct Storage {
    void* base = nullptr;
    std::size_t bytes = 0;
    Placement place{};
    std::uint64_t stream = 0;   // allocation stream (device storage frees on it)
    std::function<void()> release;
    Storage() = default;
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    ~Storage() { if (release) release(); }
};

/// Allocates ``bytes`` of ``place`` memory on ``stream`` as a Storage; the binding installs the backend's.
using Allocator = std::function<std::shared_ptr<Storage>(Placement, std::size_t bytes, std::uint64_t stream)>;

inline Allocator& allocator()
{
    static Allocator a = [](Placement p, std::size_t bytes, std::uint64_t) {
        if (p.kind == Placement::Kind::Cuda)
            throw std::runtime_error("eagle::array: no device allocator is installed (the backend provides one)");
        auto s = std::make_shared<Storage>();
        const std::size_t padded = (bytes + 63) / 64 * 64;
        s->base = bytes ? std::aligned_alloc(64, padded) : nullptr;
        if (bytes && s->base == nullptr) throw std::bad_alloc();
        s->bytes = bytes;
        s->place = p;
        void* b = s->base;
        s->release = [b] { std::free(b); };
        return s;
    };
    return a;
}

class Array {
public:
    Array() = default;

    /// A new owned array (contiguous, row-major) of ``shape`` and ``dtype``.
    static Array empty(const std::vector<std::int64_t>& shape, DType dtype, Placement place = {}, std::uint64_t stream = 0)
    {
        Array a;
        a.dtype_ = dtype;
        a.set_shape_(shape);
        a.st_ = allocator()(place, a.nbytes(), stream);
        a.st_->stream = stream;
        a.data_ = static_cast<char*>(a.st_->base);
        return a;
    }

    /// Memory owned by someone else: ``keep`` (any object, may be empty) is held until the last view goes.
    static Array borrow(void* ptr, const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& strides,
        DType dtype, Placement place, std::shared_ptr<void> keep = {})
    {
        if (shape.size() != strides.size()) throw std::invalid_argument("eagle::array: shape and strides differ in rank");
        Array a;
        a.dtype_ = dtype;
        a.set_shape_(shape);
        for (std::size_t i = 0; i < shape.size(); ++i) a.strides_[i] = strides[i];
        auto s = std::make_shared<Storage>();
        s->base = ptr;
        s->place = place;
        s->release = [keep] {};          // the shared_ptr copy lives in the closure: dropped with the storage
        a.st_ = std::move(s);
        a.data_ = static_cast<char*>(ptr);
        return a;
    }

    // ---- queries
    void* data() const { return data_; }
    std::int64_t byte_offset() const { return st_ ? data_ - static_cast<char*>(st_->base) : 0; }
    DType dtype() const { return dtype_; }
    const Placement& placement() const { return st_->place; }
    std::uint64_t stream() const { return st_ ? st_->stream : 0; }
    std::size_t rank() const { return rank_; }
    std::int64_t extent(std::size_t i) const { return shape_[i]; }
    std::int64_t stride(std::size_t i) const { return strides_[i]; }
    std::vector<std::int64_t> shape() const { return { shape_.begin(), shape_.begin() + static_cast<std::ptrdiff_t>(rank_) }; }
    std::int64_t size() const { std::int64_t n = 1; for (std::size_t i = 0; i < rank_; ++i) n *= shape_[i]; return n; }
    std::size_t nbytes() const { return static_cast<std::size_t>(size()) * itemsize(dtype_); }
    bool contiguous() const
    {
        std::int64_t acc = 1;
        for (std::size_t i = rank_; i-- > 0;) {
            if (shape_[i] != 1 && strides_[i] != acc) return false;
            acc *= shape_[i];
        }
        return true;
    }
    const std::shared_ptr<Storage>& storage() const { return st_; }
    long use_count() const { return st_.use_count(); }

    // ---- views (share storage; never copy)
    Array slice(std::size_t axis, std::int64_t start, std::int64_t stop, std::int64_t step = 1) const
    {
        if (axis >= rank_) throw std::out_of_range("eagle::array slice: axis " + std::to_string(axis) + " of a rank-" + std::to_string(rank_) + " array");
        if (step == 0) throw std::invalid_argument("eagle::array slice: step must not be 0");
        const std::int64_t n = shape_[axis];
        // python slice semantics, clamped
        auto clamp = [&](std::int64_t v, std::int64_t lo, std::int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); };
        if (start < 0) start += n;
        if (stop < 0) stop += n;
        std::int64_t len;
        if (step > 0) { start = clamp(start, 0, n); stop = clamp(stop, 0, n); len = start < stop ? (stop - start + step - 1) / step : 0; }
        else { start = clamp(start, -1, n - 1); stop = clamp(stop, -1, n - 1); len = start > stop ? (start - stop - step - 1) / -step : 0; }
        Array r = *this;
        if (len > 0) r.data_ += start * strides_[axis] * static_cast<std::int64_t>(itemsize(dtype_));
        r.shape_[axis] = len;
        r.strides_[axis] = strides_[axis] * step;
        return r;
    }
    /// Element ``i`` of ``axis`` (negative counts from the end); the result has one axis less.
    Array select(std::size_t axis, std::int64_t i) const
    {
        if (axis >= rank_) throw std::out_of_range("eagle::array index: too many indices for a rank-" + std::to_string(rank_) + " array");
        if (i < -shape_[axis] || i >= shape_[axis]) throw std::out_of_range("eagle::array index " + std::to_string(i) + " is out of bounds for axis " + std::to_string(axis) + " with size " + std::to_string(shape_[axis]));
        if (i < 0) i += shape_[axis];
        Array r = *this;
        r.data_ += i * strides_[axis] * static_cast<std::int64_t>(itemsize(dtype_));
        for (std::size_t k = axis + 1; k < rank_; ++k) { r.shape_[k - 1] = shape_[k]; r.strides_[k - 1] = strides_[k]; }
        r.rank_ = rank_ - 1;
        return r;
    }
    /// Element ``i`` of axis 0.
    Array index(std::int64_t i) const { return select(0, i); }
    Array reshape(const std::vector<std::int64_t>& shape) const
    {
        std::int64_t n = 1;
        for (auto e : shape) n *= e;
        if (n != size()) throw std::invalid_argument("eagle::array reshape: size " + std::to_string(size()) + " into a shape of " + std::to_string(n));
        if (!contiguous()) throw std::invalid_argument("eagle::array reshape: a strided view cannot be reshaped in place; copy it first");
        Array r = *this;
        r.set_shape_(shape);
        return r;
    }
    Array transpose() const
    {
        Array r = *this;
        for (std::size_t i = 0; i < rank_; ++i) { r.shape_[i] = shape_[rank_ - 1 - i]; r.strides_[i] = strides_[rank_ - 1 - i]; }
        return r;
    }
    /// The same bytes as ``dtype`` (a contiguous array whose last extent scales by the size ratio).
    Array view_as(DType dtype) const
    {
        if (!contiguous()) throw std::invalid_argument("eagle::array view_as: needs a contiguous array");
        const std::size_t from = itemsize(dtype_), to = itemsize(dtype);
        if (nbytes() % to) throw std::invalid_argument("eagle::array view_as: the byte size is not a multiple of the new itemsize");
        std::vector<std::int64_t> s = shape();
        if (s.empty()) throw std::invalid_argument("eagle::array view_as: a 0-d array cannot change itemsize");
        if ((static_cast<std::size_t>(s.back()) * from) % to) throw std::invalid_argument("eagle::array view_as: the last axis does not divide into the new itemsize");
        s.back() = static_cast<std::int64_t>(static_cast<std::size_t>(s.back()) * from / to);
        Array r = *this;
        r.dtype_ = dtype;
        r.set_shape_(s);
        return r;
    }

private:
    void set_shape_(const std::vector<std::int64_t>& shape)
    {
        if (shape.size() > kMaxRank) throw std::invalid_argument("eagle::array: rank is capped at 6, got " + std::to_string(shape.size()));
        rank_ = shape.size();
        std::int64_t acc = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            if (shape[i] < 0) throw std::invalid_argument("eagle::array: negative extent");
            shape_[i] = shape[i];
            strides_[i] = acc;
            acc *= shape[i] > 0 ? shape[i] : 1;
        }
    }
    std::shared_ptr<Storage> st_;
    char* data_ = nullptr;
    std::array<std::int64_t, kMaxRank> shape_{}, strides_{};
    std::size_t rank_ = 0;
    DType dtype_ = DType::F64;
};

}  // namespace eagle::array
