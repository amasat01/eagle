// SPDX-License-Identifier: Apache-2.0

// Reduce and copy on host planes. Reduce is the canonical fixed-order tree
// (eagle/reduce/canonical.h) over each column of an (columns, n) samples-last
// plane; the host runs the same ordered tree for every policy, bitwise
// independent of the thread count. Copy is a byte copy of two planes of one
// shape and dtype.
#include "eagle/door/host/Ops.h"

#include "eagle/exec/HostTeam.h"
#include "eagle/reduce/canonical.h"


#include <limits>

namespace eagle {
namespace door {
namespace host {

namespace {

enum Code : std::int32_t { kF32, kF64, kI32, kI64, kU8, kNone };

Code code_of(const DLDataType& t) {
    if (t.lanes != 1) return kNone;
    if (t.code == kDLFloat) return t.bits == 32 ? kF32 : t.bits == 64 ? kF64 : kNone;
    if (t.code == kDLInt) return t.bits == 32 ? kI32 : t.bits == 64 ? kI64 : kNone;
    return t.code == kDLBool && t.bits == 8 ? kU8 : kNone;
}

bool same_dtype(const DLDataType& a, const DLDataType& b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

bool same_shape(const DLTensor& a, const DLTensor& b) {
    if (a.ndim != b.ndim) return false;
    for (int i = 0; i < a.ndim; ++i)
        if (a.shape[i] != b.shape[i]) return false;
    return true;
}

const char* bytes_of(const DLTensor& t) { return static_cast<const char*>(t.data) + t.byte_offset; }
std::size_t size_of(const DLTensor& t) { return static_cast<std::size_t>(elements(t)) * (t.dtype.bits / 8); }

}  // namespace

/** @brief Whether the memory of @p a and @p b overlaps. */
bool overlaps(const DLTensor& a, const DLTensor& b) {
    const char *pa = bytes_of(a), *pb = bytes_of(b);
    return pa < pb + size_of(b) && pb < pa + size_of(a);
}

/** @brief Refuse @p plane unless it is compact and (when @p write) writable. */
void check_access(const eg_plane_s& plane, bool write, const char* what) {
    if (!compact(plane.tensor->dl_tensor))
        refuse(EG_BAD_REQUEST, std::string(what) + ": the plane is not contiguous");
    if (write && (plane.tensor->flags & DLPACK_FLAG_BITMASK_READ_ONLY))
        refuse(EG_BAD_REQUEST, std::string(what) + ": a read-only plane is written");
}

namespace {

/** @brief One column of a samples-last plane as ``in(i)``. */
template <class T>
struct Column {
    const T* p;
    T operator()(idx_t i) const { return p[i]; }
};

template <class T>
const T* data_of(const eg_plane_s& p) {
    return reinterpret_cast<const T*>(bytes_of(p.tensor->dl_tensor));
}
template <class T>
T* mut_data_of(const eg_plane_s& p) {
    return const_cast<T*>(data_of<T>(p));
}

template <class T, class OP>
void columns(const ReduceOp& r, int threads) {
    namespace canon = eagle::reduce::canonical;
    const T* in = data_of<T>(*r.in) + r.first;
    T* out = mut_data_of<T>(*r.out);
    for (std::int64_t j = 0; j < r.cols; ++j) {
        const Column<T> x{ in + j * r.n };
        if constexpr (canon::compensated_v<T, OP>) {
            canon::Partial<T, OP> p;
            if (r.comp != nullptr && !r.pair) {
                const Column<T> c{ data_of<T>(*r.comp) + r.first + j * r.n };
                p = canon::partialRange<T, OP>(x, 0, r.n, c, threads);
            } else {
                p = canon::partialRange<T, OP>(x, 0, r.n, {}, threads);
            }
            if (r.pair) {
                out[j] = p.h;
                mut_data_of<T>(*r.comp)[j] = p.c;
            } else {
                out[j] = p.value();
            }
        } else {
            out[j] = canon::partialRange<T, OP>(x, 0, r.n, {}, threads).value();
        }
    }
}

template <class T>
void by_op(const ReduceOp& r, int threads) {
    if (r.op == EG_REDUCE_SUM) columns<T, aether::SumOp<T>>(r, threads);
    else if (r.op == EG_REDUCE_MAX) columns<T, aether::MaxOp<T>>(r, threads);
    else columns<T, eagle::reduce::canonical::MinOp<T>>(r, threads);
}

}  // namespace

/** @brief Check a reduce request; refuses as eg_reduce states. */
ReduceOp reduce_check(const eg_ctx& ctx, const eg_reduce_desc& desc) {
    if (desc.flags & ~EG_REDUCE_PAIR)
        refuse(EG_BAD_REQUEST, "eg_reduce_desc.flags: unknown bits " + std::to_string(desc.flags & ~EG_REDUCE_PAIR));
    if (desc.policy != EG_REDUCE_COMPENSATED && desc.policy != EG_REDUCE_FIXED)
        refuse(EG_BAD_REQUEST, "unknown reduce policy " + std::to_string(desc.policy));
    if (desc.op < EG_REDUCE_SUM || desc.op > EG_REDUCE_ALL)
        refuse(EG_BAD_REQUEST, "unknown reduce op " + std::to_string(desc.op));
    check_handle(ctx.planes, desc.in, "eg_reduce_desc.in");
    check_handle(ctx.planes, desc.out, "eg_reduce_desc.out");
    ReduceOp r;
    r.in = desc.in;
    r.out = desc.out;
    r.op = desc.op;
    r.pair = (desc.flags & EG_REDUCE_PAIR) != 0;
    const DLTensor& in = desc.in->tensor->dl_tensor;
    const DLTensor& out = desc.out->tensor->dl_tensor;
    if (in.ndim != 1 && in.ndim != 2)
        refuse(EG_BAD_REQUEST, "eg_reduce: in must be (n,) or (columns, n), got " + std::to_string(in.ndim) + " dimensions");
    r.cols = in.ndim == 2 ? in.shape[0] : 1;
    r.n = in.shape[in.ndim - 1];
    if (r.n < 1 || r.cols < 1) refuse(EG_BAD_REQUEST, "eg_reduce: in holds no samples");
    if (out.ndim != 1 || out.shape[0] != r.cols)
        refuse(EG_BAD_REQUEST, "eg_reduce: out must have shape (" + std::to_string(r.cols) + ",)");
    if (!same_dtype(in.dtype, out.dtype)) refuse(EG_BAD_REQUEST, "eg_reduce: out dtype differs from in");
    check_access(*desc.in, false, "eg_reduce in");
    check_access(*desc.out, true, "eg_reduce out");
    if (overlaps(in, out)) refuse(EG_BAD_REQUEST, "eg_reduce: out overlaps in");
    const Code code = code_of(in.dtype);
    r.code = code;
    if (code == kNone)
        refuse(EG_NOT_WRITTEN, "eg_reduce: dtype (code " + std::to_string(in.dtype.code) + ", bits " +
               std::to_string(in.dtype.bits) + ") is not one the host reduce runs");
    if ((r.op == EG_REDUCE_ALL) != (code == kU8))
        refuse(EG_NOT_WRITTEN, r.op == EG_REDUCE_ALL ? "eg_reduce: op ALL takes a bool plane"
                                                     : "eg_reduce: a bool plane takes op ALL only");
    const bool fsum = r.op == EG_REDUCE_SUM && (code == kF32 || code == kF64);
    if (desc.comp != nullptr) {
        check_handle(ctx.planes, desc.comp, "eg_reduce_desc.comp");
        if (!fsum) refuse(EG_NOT_WRITTEN, "eg_reduce: comp takes a float32 or float64 sum");
        const DLTensor& c = desc.comp->tensor->dl_tensor;
        if (!same_dtype(c.dtype, in.dtype)) refuse(EG_BAD_REQUEST, "eg_reduce: comp dtype differs from in");
        if (!same_shape(c, r.pair ? out : in))
            refuse(EG_BAD_REQUEST, std::string("eg_reduce: comp must have the shape of ") + (r.pair ? "out" : "in"));
        check_access(*desc.comp, r.pair, "eg_reduce comp");
        if (overlaps(c, out) || (r.pair && overlaps(c, in)))
            refuse(EG_BAD_REQUEST, "eg_reduce: comp overlaps another plane");
        r.comp = desc.comp;
    } else if (r.pair) {
        refuse(EG_BAD_REQUEST, "eg_reduce: EG_REDUCE_PAIR writes the compensation into comp, which is NULL");
    }
    if (r.pair && desc.policy == EG_REDUCE_FIXED)
        refuse(EG_NOT_WRITTEN, "eg_reduce: EG_REDUCE_PAIR with policy FIXED (a plain sum has no compensation)");
    return r;
}

/** @brief Run a checked reduce on at most @p threads threads of eagle's host pool: each column's canonical tree
 *  into out[column]. */
void reduce_exec(const ReduceOp& r, int threads) {
    switch (r.code) {
        case kF32: by_op<float>(r, threads); break;
        case kF64: by_op<double>(r, threads); break;
        case kI32: by_op<std::int32_t>(r, threads); break;
        case kI64: by_op<std::int64_t>(r, threads); break;
        default: columns<std::uint8_t, aether::LogicalAndOp>(r, threads);
    }
}

namespace {

/** @brief The identity of @p op over @p T: what a reduction over no sample is. */
template <class T>
T identity_of(std::int32_t op) {
    if (op == EG_REDUCE_SUM) return T(0);
    if (op == EG_REDUCE_ALL) return T(1);
    const bool low = op == EG_REDUCE_MAX;
    if constexpr (std::numeric_limits<T>::has_infinity)
        return low ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    else
        return low ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
}

template <class T>
void write_identity(const ReduceOp& r) {
    T* out = mut_data_of<T>(*r.out);
    for (std::int64_t j = 0; j < r.cols; ++j) out[j] = identity_of<T>(r.op);
}

}  // namespace

void fold_plan(const eg_plan_s& plan, int threads, std::int64_t first_row, std::int64_t end_row) {
    for (const CompanionFacts& c : plan.unit->companions) {
        const std::int64_t size = elements(plan.bindings[c.target].plane->tensor->dl_tensor);
        exec::HostTeam::run_serial(c.fold, plan.params.data(), exec::Partition{ 0, size, size });
    }
    for (std::size_t i = 0; i < plan.unit->cells.size(); ++i) {
        const CellFacts& c = plan.unit->cells[i];
        eg_reduce_desc d{};
        d.struct_size = sizeof(d);
        d.in = plan.rows[i];
        d.comp = plan.comps[i];
        d.out = plan.bindings[c.slot].plane;
        d.op = c.op;
        ReduceOp r = reduce_check(*plan.ctx, d);
        if (end_row > first_row) {
            r.first = first_row;
            r.n = end_row - first_row;
            reduce_exec(r, threads);
            continue;
        }
        switch (r.code) {
            case kF32: write_identity<float>(r); break;
            case kF64: write_identity<double>(r); break;
            case kI32: write_identity<std::int32_t>(r); break;
            case kI64: write_identity<std::int64_t>(r); break;
            default: write_identity<std::uint8_t>(r);
        }
    }
}

/** @brief Check a copy request; refuses as eg_copy states. */
void copy_check(const eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst) {
    check_handle(ctx.planes, src, "src");
    check_handle(ctx.planes, dst, "dst");
    const DLTensor& s = src->tensor->dl_tensor;
    const DLTensor& d = dst->tensor->dl_tensor;
    if (!same_shape(s, d) || !same_dtype(s.dtype, d.dtype))
        refuse(EG_BAD_REQUEST, "eg_copy: src and dst must have one shape and dtype");
    check_access(*src, false, "eg_copy src");
    check_access(*dst, true, "eg_copy dst");
}

/** @brief Run a checked copy (nothing when both are one range of memory). */
void copy_exec(const eg_plane_s& src, const eg_plane_s& dst) {
    const DLTensor& s = src.tensor->dl_tensor;
    const DLTensor& d = dst.tensor->dl_tensor;
    if (size_of(s) == 0 || bytes_of(s) == bytes_of(d)) return;
    std::memmove(const_cast<char*>(bytes_of(d)), bytes_of(s), size_of(s));
}

}  // namespace host
}  // namespace door
}  // namespace eagle
