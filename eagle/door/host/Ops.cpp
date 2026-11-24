// SPDX-License-Identifier: Apache-2.0

// Reduce and copy on host planes. Reduce is the canonical fixed-order tree
// (eagle/reduce/canonical.h) over each column of an (columns, n) samples-last
// plane; the host runs the same ordered tree for every policy, bitwise
// independent of the thread count. A minimum or maximum asked for its position
// folds pairs (value, lane) through the same tree, so the position costs no
// second pass over the plane. Copy is a byte copy of two planes of one
// shape and dtype.
#include "eagle/door/host/Ops.h"

#include "eagle/cpu/Scan.h"
#include "eagle/exec/HostTeam.h"
#include "eagle/filtering/Compact.h"
#include "eagle/filtering/KeyedSort.h"
#include "eagle/filtering/Reorder.h"
#include "eagle/reduce/canonical.h"


#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace eagle {
namespace door {
namespace host {

namespace {

enum Code : std::int32_t { kF32, kF64, kI32, kI64, kU8, kU32, kNone };

Code code_of(const DLDataType& t) {
    if (t.lanes != 1) return kNone;
    if (t.code == kDLFloat) return t.bits == 32 ? kF32 : t.bits == 64 ? kF64 : kNone;
    if (t.code == kDLInt) return t.bits == 32 ? kI32 : t.bits == 64 ? kI64 : kNone;
    if (t.code == kDLUInt) return t.bits == 32 ? kU32 : kNone;
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

/** @brief A sample with its lane: what the fold of a minimum or maximum with position carries. */
template <class T>
struct Ranked {
    T value;
    idx_t lane;
    constexpr Ranked(int = 0) : value(T(0)), lane(0) {}
    constexpr Ranked(T v, idx_t l) : value(v), lane(l) {}
};

/** @brief Column ``p`` as ``in(i)``, each sample ranked by its lane; a NaN gets lane ``n``, the lane no sample has. */
template <class T>
struct RankedColumn {
    const T* p;
    idx_t n;
    Ranked<T> operator()(idx_t i) const {
        const T v = p[i];
        if constexpr (std::is_floating_point_v<T>) return { v, std::isnan(v) ? n : i };
        else return { v, i };
    }
};

/** @brief The winner of two ranked samples: the larger value (@p Max) or the smaller; equal values go to the lower lane; a NaN
 *  (lane ``n``, value NaN) loses to every other sample, and two NaNs give the first. Associative, so the canonical tree's
 *  shape cannot change the winner. */
template <class T, bool Max>
struct Best {
    Ranked<T> operator()(const Ranked<T>& a, const Ranked<T>& b) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(b.value)) return a;
            if (std::isnan(a.value)) return b;
        }
        if (a.value == b.value) return a.lane <= b.lane ? a : b;
        return (Max ? a.value > b.value : a.value < b.value) ? a : b;
    }
};

/** @brief Each column's winner into out and its lane into index. */
template <class T, bool Max>
void ranked_columns(const ReduceOp& r, int threads) {
    namespace canon = eagle::reduce::canonical;
    const T* in = data_of<T>(*r.in) + r.first;
    T* out = mut_data_of<T>(*r.out);
    std::int32_t* index = mut_data_of<std::int32_t>(*r.index);
    for (std::int64_t j = 0; j < r.cols; ++j) {
        const RankedColumn<T> x{ in + j * r.n, static_cast<idx_t>(r.n) };
        const Ranked<T> best = canon::partialRange<Ranked<T>, Best<T, Max>>(x, 0, r.n, {}, threads).value();
        out[j] = best.value;
        index[j] = static_cast<std::int32_t>(best.lane);
    }
}

/** @brief Column ``p`` of the rows of a cell as ``in(i)``: each row's value ranked by the lane that won it; a NaN row (no winner) has lane ``none``. */
template <class T>
struct RankedRows {
    const T* p;
    const T* lanes;
    idx_t none;
    Ranked<T> operator()(idx_t i) const {
        const T v = p[i];
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(v)) return { v, none };
        }
        return { v, static_cast<idx_t>(lanes[i]) };
    }
};

/** @brief The cell's fold over its rows: the winner of (value, lane) pairs through the canonical tree, written as its value, or as its lane
 *  (the sample count when no row has a winner) when the cell takes the position. */
template <class T, bool Max>
void ranked_rows_columns(const ReduceOp& r, int threads) {
    namespace canon = eagle::reduce::canonical;
    const T* in = data_of<T>(*r.in) + r.first;
    const T* lanes = data_of<T>(*r.lanes) + r.first;
    T* out = mut_data_of<T>(*r.out);
    const idx_t none = static_cast<idx_t>(r.samples);
    for (std::int64_t j = 0; j < r.cols; ++j) {
        const RankedRows<T> x{ in + j * r.n, lanes + j * r.n, none };
        const Ranked<T> best = canon::partialRange<Ranked<T>, Best<T, Max>>(x, 0, r.n, {}, threads).value();
        out[j] = r.position ? static_cast<T>(best.lane) : best.value;
    }
}

template <class T>
void ranked_by_op(const ReduceOp& r, int threads) {
    if (r.lanes != nullptr) {
        if (r.op == EG_REDUCE_MAX) ranked_rows_columns<T, true>(r, threads);
        else ranked_rows_columns<T, false>(r, threads);
        return;
    }
    if (r.op == EG_REDUCE_MAX) ranked_columns<T, true>(r, threads);
    else ranked_columns<T, false>(r, threads);
}

/** @brief Each column's ALL or ANY into out: a bool byte, or a uint32 when out is one (a cell a conditional can read). */
template <class OP>
void mask_columns(const ReduceOp& r, int threads) {
    namespace canon = eagle::reduce::canonical;
    const std::uint8_t* in = data_of<std::uint8_t>(*r.in) + r.first;
    const bool wide = r.out->tensor->dl_tensor.dtype.code == kDLUInt;
    for (std::int64_t j = 0; j < r.cols; ++j) {
        const Column<std::uint8_t> x{ in + j * r.n };
        const std::uint8_t v = canon::partialRange<std::uint8_t, OP>(x, 0, r.n, {}, threads).value();
        if (wide) mut_data_of<std::uint32_t>(*r.out)[j] = v != 0 ? 1u : 0u;
        else mut_data_of<std::uint8_t>(*r.out)[j] = v;
    }
}

/** @brief The or of two mask bytes. */
struct LogicalOr {
    std::uint8_t operator()(std::uint8_t a, std::uint8_t b) const { return (a != 0 || b != 0) ? 1 : 0; }
};

template <class T>
void by_op(const ReduceOp& r, int threads) {
    if (r.index != nullptr || r.lanes != nullptr) return ranked_by_op<T>(r, threads);
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
    if (desc.op < EG_REDUCE_SUM || desc.op > EG_REDUCE_ANY)
        refuse(EG_BAD_REQUEST, "unknown reduce op " + std::to_string(desc.op));
    if (desc.index != nullptr && desc.op != EG_REDUCE_MIN && desc.op != EG_REDUCE_MAX)
        refuse(EG_BAD_REQUEST, "eg_reduce: index (a position) takes op MIN or MAX only; a sum, an all or an any has no single lane");
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
    // an ALL or ANY result may also be a uint32 plane: a cell a conditional reads (the mask itself is a bool plane)
    const bool wide_mask = (r.op == EG_REDUCE_ALL || r.op == EG_REDUCE_ANY) && out.dtype.code == kDLUInt && out.dtype.bits == 32 &&
                           out.dtype.lanes == 1;
    if (!wide_mask && !same_dtype(in.dtype, out.dtype)) refuse(EG_BAD_REQUEST, "eg_reduce: out dtype differs from in");
    check_access(*desc.in, false, "eg_reduce in");
    check_access(*desc.out, true, "eg_reduce out");
    if (overlaps(in, out)) refuse(EG_BAD_REQUEST, "eg_reduce: out overlaps in");
    const Code code = code_of(in.dtype);
    r.code = code;
    if (code == kNone)
        refuse(EG_NOT_WRITTEN, "eg_reduce: dtype (code " + std::to_string(in.dtype.code) + ", bits " +
               std::to_string(in.dtype.bits) + ") is not one the host reduce runs");
    const bool mask_op = r.op == EG_REDUCE_ALL || r.op == EG_REDUCE_ANY;
    if (mask_op != (code == kU8))
        refuse(EG_NOT_WRITTEN, mask_op ? std::string("eg_reduce: op ") + (r.op == EG_REDUCE_ALL ? "ALL" : "ANY") + " takes a bool plane"
                                       : "eg_reduce: a bool plane takes op ALL or ANY only");
    if (code == kU32 && r.op != EG_REDUCE_MIN && r.op != EG_REDUCE_MAX)
        refuse(EG_NOT_WRITTEN, "eg_reduce: dtype uint32 is served by ops MIN and MAX only");
    if (desc.index != nullptr) {
        check_handle(ctx.planes, desc.index, "eg_reduce_desc.index");
        const DLTensor& x = desc.index->tensor->dl_tensor;
        if (x.dtype.code != kDLInt || x.dtype.bits != 32 || x.dtype.lanes != 1)
            refuse(EG_BAD_REQUEST, "eg_reduce: index must be an int32 plane");
        if (x.ndim != 1 || x.shape[0] != r.cols)
            refuse(EG_BAD_REQUEST, "eg_reduce: index must have shape (" + std::to_string(r.cols) + ",)");
        if (r.n > std::numeric_limits<std::int32_t>::max())
            refuse(EG_BAD_REQUEST, "eg_reduce: index names a lane in 32 bits; more than 2^31 - 1 samples");
        check_access(*desc.index, true, "eg_reduce index");
        if (overlaps(x, in) || overlaps(x, out)) refuse(EG_BAD_REQUEST, "eg_reduce: index overlaps another plane");
        r.index = desc.index;
    }
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

ReduceOp cell_reduce_check(const eg_ctx& ctx, eg_reduce_desc desc, bool position, std::int64_t samples) {
    eg_plane* lanes = desc.op == EG_REDUCE_SUM ? nullptr : desc.comp;
    if (lanes != nullptr) desc.comp = nullptr;
    ReduceOp r = reduce_check(ctx, desc);
    if (lanes != nullptr) {    // the rows carry the lane that won each: fold pairs
        r.lanes = lanes;
        r.position = position;
        r.samples = samples;
    }
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
        case kU32: by_op<std::uint32_t>(r, threads); break;
        default:
            if (r.op == EG_REDUCE_ANY) mask_columns<LogicalOr>(r, threads);
            else mask_columns<aether::LogicalAndOp>(r, threads);
    }
}

namespace {

/** @brief The identity of @p op over @p T: what a reduction over no sample is. */
template <class T>
T identity_of(std::int32_t op) {
    if (op == EG_REDUCE_SUM) return T(0);
    if (op == EG_REDUCE_ALL) return T(1);
    if (op == EG_REDUCE_ANY) return T(0);
    const bool low = op == EG_REDUCE_MAX;
    if constexpr (std::numeric_limits<T>::has_infinity)
        return low ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    else
        return low ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
}

template <class T>
void write_identity(const ReduceOp& r) {
    T* out = mut_data_of<T>(*r.out);
    for (std::int64_t j = 0; j < r.cols; ++j) out[j] = r.position ? static_cast<T>(r.samples) : identity_of<T>(r.op);
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
        ReduceOp r = cell_reduce_check(*plan.ctx, d, c.position, plan.n_samples);
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
            case kU32: write_identity<std::uint32_t>(r); break;
            default: write_identity<std::uint8_t>(r);
        }
    }
}

namespace {

/** The element types of a scan, by the code ``ScanOp::code`` keeps. */
enum ScanCode : std::int32_t { kScanI32, kScanI64, kScanU32, kScanU64, kScanF32, kScanF64, kScanNone };

ScanCode scan_code_of(const DLDataType& t) {
    const bool real = t.code == kDLFloat, whole = t.code == kDLInt, natural = t.code == kDLUInt;
    if (real) return t.bits == 32 ? kScanF32 : t.bits == 64 ? kScanF64 : kScanNone;
    if (whole) return t.bits == 32 ? kScanI32 : t.bits == 64 ? kScanI64 : kScanNone;
    return natural && t.bits == 32 ? kScanU32 : natural && t.bits == 64 ? kScanU64 : kScanNone;
}

/** @brief The identity of a scan's op over @p T: 0 for a sum, else the largest (min) or smallest (max) value; a real type's is infinite. */
template <class T>
T scan_identity(eg_scan_op op) {
    if (op == EG_SCAN_SUM) return T(0);
    const bool low = op == EG_SCAN_MAX;
    if constexpr (std::numeric_limits<T>::has_infinity)
        return low ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    else
        return low ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
}

/** The folds of a scan. An integer sum wraps (it is added as an unsigned number, which C++ defines); a real minimum and
 *  maximum skip a NaN, as ``fmin`` and ``fmax`` do. */
template <class T>
struct ScanSum {
    T operator()(T a, T b) const {
        if constexpr (std::is_integral_v<T>) {
            using U = std::make_unsigned_t<T>;
            return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
        } else {
            return a + b;
        }
    }
};
template <class T>
struct ScanMin {
    T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) return std::fmin(a, b);
        else return b < a ? b : a;
    }
};
template <class T>
struct ScanMax {
    T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) return std::fmax(a, b);
        else return a < b ? b : a;
    }
};

/** @brief ``samples()`` and ``(i)`` over a pointer, what ``cpu::Scan`` reads and writes through. */
template <class T>
struct Samples {
    T* p;
    idx_t n;
    idx_t samples() const { return n; }
    T& operator()(idx_t i) const { return p[i]; }
};

/** The scan's block: the blocks, and so the grouping of a real sum, are a function of the extent alone. */
constexpr unsigned kScanBlock = 1024;

/** @brief The scan over @p n elements read by @p in and written through @p out,
 * of any element @p E and fold @p OP. */
template<class E, class OP, class InT, class OutT>
void scan_blocks(const InT& in, OutT&& out, idx_t n, const E& identity,
    bool exclusive, int threads)
{
    std::vector<E> totals(static_cast<std::size_t>(
        filtering::detail::ScanCore::countBlocks<kScanBlock>(
            n, std::min<idx_t>(kScanBlock, n))));
    const Samples<E> sums{ totals.data(), static_cast<idx_t>(totals.size()) };
    if (exclusive)
        cpu::Scan::exclusive<E, OP, kScanBlock>(
            in, out, sums, identity, threads);
    else
        cpu::Scan::inclusive<E, OP, kScanBlock>(
            in, out, sums, identity, threads);
}

template<class T, class OP>
void scan_form(
    const T* from, T* to, idx_t n, eg_scan_op op, bool exclusive, int threads)
{
    scan_blocks<T, OP>(Samples<const T>{ from, n }, Samples<T>{ to, n }, n,
        scan_identity<T>(op), exclusive, threads);
}

/** A value with the lane that holds it: what a minimum or maximum scan folds to
 * name its winner, as the positioned reduce folds pairs. */
template<class T>
struct Candidate {
    T value;
    std::int32_t lane;
};

/** The fold of candidates, the one definition of "winner" (the scan's, and the
 * reverse's through it): the smaller (@p Low) or larger value, the lower lane
 * among equal values, a NaN never. */
template<class T, bool Low>
struct ScanWinner {
    Candidate<T> operator()(Candidate<T> a, Candidate<T> b) const
    {
        if (std::isnan(b.value))
            return a;
        if (Low ? b.value < a.value : b.value > a.value)
            return b;
        if (Low ? a.value < b.value : a.value > b.value)
            return a;
        return b.lane < a.lane ? b : a;
    }
};

/** The input of a winner scan: each element with its own lane. */
template<class T>
struct CandidatesOf {
    const T* p;
    idx_t n;
    idx_t samples() const { return n; }
    Candidate<T> operator()(idx_t i) const
    {
        return { p[i], static_cast<std::int32_t>(i) };
    }
};

/** The output of a winner scan: a value plane and a lane plane, read and
 * written as one candidate per element. */
template<class T>
struct CandidatesTo {
    T* values;
    std::int32_t* lanes;
    idx_t n;
    struct Slot {
        T* value;
        std::int32_t* lane;
        operator Candidate<T>() const { return { *value, *lane }; }
        Slot& operator=(const Candidate<T>& c)
        {
            *value = c.value;
            *lane  = c.lane;
            return *this;
        }
    };
    idx_t samples() const { return n; }
    Slot operator()(idx_t i) const { return { values + i, lanes + i }; }
};

template<class T, bool Low>
void scan_winner_form(const T* from, T* to, std::int32_t* lanes, idx_t n,
    bool exclusive, int threads)
{
    const Candidate<T> identity{ scan_identity<T>(
                                     Low ? EG_SCAN_MIN : EG_SCAN_MAX),
        static_cast<std::int32_t>(n) };
    scan_blocks<Candidate<T>, ScanWinner<T, Low>>(CandidatesOf<T>{ from, n },
        CandidatesTo<T>{ to, lanes, n }, n, identity, exclusive, threads);
}

/** @brief The values and running winners of a minimum (@p op MIN) or maximum
 * scan of @p from. */
template<class T>
void scan_winner_by_op(const T* from, T* to, std::int32_t* lanes, idx_t n,
    eg_scan_op op, bool exclusive, int threads)
{
    if (op == EG_SCAN_MIN)
        scan_winner_form<T, true>(from, to, lanes, n, exclusive, threads);
    else
        scan_winner_form<T, false>(from, to, lanes, n, exclusive, threads);
}

template <class T>
void scan_by_op(const ScanOp& s, int threads) {
    const T* from = data_of<T>(*s.in);
    T* to         = mut_data_of<T>(*s.out);
    const idx_t n = static_cast<idx_t>(s.n);
    if (s.index != nullptr)
        scan_winner_by_op<T>(from, to, mut_data_of<std::int32_t>(*s.index), n,
            s.op, s.exclusive, threads);
    else if (s.op == EG_SCAN_SUM)
        scan_form<T, ScanSum<T>>(from, to, n, s.op, s.exclusive, threads);
    else if (s.op == EG_SCAN_MIN)
        scan_form<T, ScanMin<T>>(from, to, n, s.op, s.exclusive, threads);
    else
        scan_form<T, ScanMax<T>>(from, to, n, s.op, s.exclusive, threads);
}

/** The adjoint of a sum scan: out[k] folds in[0..k] (in[0..k-1] exclusive), so
 * in[i] reaches the outputs from i (from i + 1) on and its adjoint is the sum
 * of dy[i..] (dy[i + 1..]): the forward scan run over the adjoint read from the
 * last lane to the first. */
template<class T>
void scan_sum_pullback(const ScanOp& s, const T* dy, T* bar, int threads)
{
    const std::size_t n = static_cast<std::size_t>(s.n);
    std::vector<T> reversed(n), folded(n);
    for (std::size_t i = 0; i < n; ++i)
        reversed[i] = dy[n - 1 - i];
    scan_form<T, ScanSum<T>>(reversed.data(), folded.data(),
        static_cast<idx_t>(n), EG_SCAN_SUM, s.exclusive, threads);
    for (std::size_t i = 0; i < n; ++i)
        bar[i] = folded[n - 1 - i];
}

}  // namespace

/** @brief Check a scan request; refuses as eg_scan states. */
ScanOp scan_check(const eg_ctx& ctx, const eg_scan_desc& desc) {
    if (desc.flags & ~EG_SCAN_EXCLUSIVE)
        refuse(EG_BAD_REQUEST, "eg_scan_desc.flags: unknown bits " + std::to_string(desc.flags & ~EG_SCAN_EXCLUSIVE));
    if (desc.op < EG_SCAN_SUM || desc.op > EG_SCAN_MAX) refuse(EG_BAD_REQUEST, "unknown scan op " + std::to_string(desc.op));
    check_handle(ctx.planes, desc.in, "eg_scan_desc.in");
    check_handle(ctx.planes, desc.out, "eg_scan_desc.out");
    const DLTensor& in = desc.in->tensor->dl_tensor;
    const DLTensor& out = desc.out->tensor->dl_tensor;
    if (in.dtype.lanes != 1)
        refuse(EG_NOT_WRITTEN, "eg_scan: a value of rank above zero (a vector or matrix per sample) is not written; scan one component at a time");
    if (in.ndim != 1)
        refuse(EG_BAD_REQUEST, "eg_scan: in has " + std::to_string(in.ndim) +
               " run-time axes, a scan runs along exactly one; scan a group over that axis (g.scan) to scan along one of several");
    const ScanCode code = scan_code_of(in.dtype);
    if (code == kScanNone)
        refuse(EG_NOT_WRITTEN, "eg_scan: dtype (code " + std::to_string(in.dtype.code) + ", bits " + std::to_string(in.dtype.bits) +
               ") is not one of int32, int64, uint32, uint64, float32, float64");
    if (!same_dtype(in.dtype, out.dtype)) refuse(EG_BAD_REQUEST, "eg_scan: out dtype differs from in");
    if (!same_shape(in, out)) refuse(EG_BAD_REQUEST, "eg_scan: out extent differs from in");
    if (in.shape[0] > std::numeric_limits<std::int32_t>::max())
        refuse(EG_BAD_REQUEST, "eg_scan: " + std::to_string(in.shape[0]) + " elements is above the 2^31 - 1 a scan indexes");
    check_access(*desc.in, false, "eg_scan in");
    check_access(*desc.out, true, "eg_scan out");
    if (overlaps(in, out)) refuse(EG_BAD_REQUEST, "eg_scan: out overlaps in; a scan writes a plane distinct from the one it reads");
    if (desc.index != nullptr) {
        if (desc.op == EG_SCAN_SUM)
            refuse(EG_BAD_REQUEST,
                "eg_scan: index (a winner) takes op MIN or MAX only; a sum has "
                "no single lane");
        check_handle(ctx.planes, desc.index, "eg_scan_desc.index");
        const DLTensor& x = desc.index->tensor->dl_tensor;
        if (x.dtype.code != kDLInt || x.dtype.bits != 32 || x.dtype.lanes != 1)
            refuse(EG_BAD_REQUEST, "eg_scan: index must be an int32 plane");
        if (x.ndim != 1 || x.shape[0] != in.shape[0])
            refuse(EG_BAD_REQUEST,
                "eg_scan: index must have the extent of in, "
                    + std::to_string(in.shape[0]));
        check_access(*desc.index, true, "eg_scan index");
        if (overlaps(x, in) || overlaps(x, out))
            refuse(EG_BAD_REQUEST, "eg_scan: index overlaps another plane");
    }
    ScanOp s;
    s.in = desc.in;
    s.out = desc.out;
    s.index     = desc.index;
    s.op = desc.op;
    s.exclusive = (desc.flags & EG_SCAN_EXCLUSIVE) != 0;
    s.code = code;
    s.n = in.shape[0];
    return s;
}

/** @brief Run a checked scan: ``cpu::Scan`` over blocks of kScanBlock elements, each block scanned serially, the block totals
 *  prefixed in block order. */
void scan_exec(const ScanOp& s, int threads) {
    if (s.n == 0) return;
    switch (s.code) {
        case kScanI32: scan_by_op<std::int32_t>(s, threads); break;
        case kScanI64: scan_by_op<std::int64_t>(s, threads); break;
        case kScanU32: scan_by_op<std::uint32_t>(s, threads); break;
        case kScanU64: scan_by_op<std::uint64_t>(s, threads); break;
        case kScanF32: scan_by_op<float>(s, threads); break;
        default: scan_by_op<double>(s, threads);
    }
}

namespace {

/** The adjoint of a minimum or maximum scan: out[k] is the running winner of
 * in[0..k] (in[0..k-1] exclusive), the lowest lane among equals, and the
 * adjoint of out[k] goes to it. Winners never decrease, so each receives one
 * contiguous run of dy, added in lane order by this one pass: the same bits at
 * any thread count. The winners are @p winners, the plane the forward wrote, or
 * when it wrote none the forward's own winner scan over @p in. A lane with no
 * winner (the lane count: the exclusive scan's first output, or a prefix of
 * nothing but NaN) passes nothing. */
template<class T>
void scan_extremum_pullback(const ScanOp& s, const T* in, const T* dy, T* bar,
    const std::int32_t* winners, int threads)
{
    std::vector<std::int32_t> found;
    if (winners == nullptr) {
        std::vector<T> values(static_cast<std::size_t>(s.n));
        found.resize(static_cast<std::size_t>(s.n));
        scan_winner_by_op<T>(in, values.data(), found.data(),
            static_cast<idx_t>(s.n), s.op, s.exclusive, threads);
        winners = found.data();
    }
    std::fill(bar, bar + s.n, T(0));
    for (std::int64_t k = 0; k < s.n; ++k) {
        if (winners[k] < 0 || winners[k] > s.n)
            refuse(EG_BAD_REQUEST,
                "eg_graph_reverse: a winner of eg_scan is outside [0, "
                    + std::to_string(s.n)
                    + "]: the winner plane changed after the scan");
        if (winners[k] < s.n)
            bar[winners[k]] += dy[k];
    }
}

} // namespace

void scan_pullback(const ScanOp& s, const char* in, const char* dy, char* bar,
    int threads, const std::int32_t* winners)
{
    if (s.n == 0)
        return;
    const bool real32 = s.code == kScanF32;
    if (s.op == EG_SCAN_SUM) {
        if (real32)
            scan_sum_pullback(s, reinterpret_cast<const float*>(dy),
                reinterpret_cast<float*>(bar), threads);
        else
            scan_sum_pullback(s, reinterpret_cast<const double*>(dy),
                reinterpret_cast<double*>(bar), threads);
    } else if (real32) {
        scan_extremum_pullback(s, reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(dy), reinterpret_cast<float*>(bar),
            winners, threads);
    } else {
        scan_extremum_pullback(s, reinterpret_cast<const double*>(in),
            reinterpret_cast<const double*>(dy), reinterpret_cast<double*>(bar),
            winners, threads);
    }
}

namespace {
/** The fewest lanes for which the parallel predicate, scan and scatter beat one pass on one thread (measured on the host here:
 *  at 10,000 lanes the hand-off between threads costs several times the pass). */
constexpr std::int64_t kParallelCompactLanes = 1 << 16;
}  // namespace

/** @brief The compaction: the predicate, the scan and the scatter of ``filtering::compactHost`` over a scratch of its size. */
void compact_lanes(const std::uint8_t* mask, bool drop, std::int32_t* kept, std::uint32_t* count, std::int64_t n, int threads,
                   std::vector<std::uint64_t>& scratch) {
    // One thread, or too few lanes for the hand-off between threads to pay: one pass over the mask gives the same list the
    // predicate, scan and scatter give (ascending lanes), without the scan's two extra passes or any working memory.
    if (threads <= 1 || n < kParallelCompactLanes) {
        std::uint32_t live = 0;
        if (drop) {
            for (std::int64_t i = 0; i < n; ++i)
                if (mask[i] == 0) kept[live++] = static_cast<std::int32_t>(i);
        } else {
            for (std::int64_t i = 0; i < n; ++i)
                if (mask[i] != 0) kept[live++] = static_cast<std::int32_t>(i);
        }
        *count = live;
        return;
    }
    const std::size_t words = (filtering::compactScratchBytes(n) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t);
    if (scratch.size() < words) scratch.resize(words);
    filtering::compactHost(mask, drop ? filtering::kCompactMaskIsDrop : 0u, kept, count, scratch.data(), n, threads);
}

/** @brief Check a compaction request; refuses as eg_compact states. */
CompactOp compact_check(const eg_ctx& ctx, const eg_compact_desc& desc) {
    if (desc.flags != 0) refuse(EG_BAD_REQUEST, "eg_compact_desc.flags: unknown bits " + std::to_string(desc.flags));
    check_handle(ctx.planes, desc.keep, "eg_compact_desc.keep");
    check_handle(ctx.planes, desc.kept, "eg_compact_desc.kept");
    check_handle(ctx.planes, desc.count, "eg_compact_desc.count");
    const DLTensor& keep = desc.keep->tensor->dl_tensor;
    const DLTensor& kept = desc.kept->tensor->dl_tensor;
    const DLTensor& count = desc.count->tensor->dl_tensor;
    const auto axes = [](const DLTensor& t, const char* name) {
        if (t.ndim != 1)
            refuse(EG_BAD_REQUEST, std::string("eg_compact: ") + name + " has " + std::to_string(t.ndim) +
                   " run-time axes, a compaction runs along exactly one; compact a group over that axis to compact along one of several");
    };
    axes(keep, "keep");
    axes(kept, "kept");
    axes(count, "count");
    const bool mask = keep.dtype.lanes == 1 && keep.dtype.bits == 8 && keep.dtype.code == kDLBool;
    if (!mask) refuse(EG_BAD_REQUEST, "eg_compact: keep must be a bool plane (one byte per lane)");
    const bool index = kept.dtype.lanes == 1 && kept.dtype.bits == 32 && (kept.dtype.code == kDLInt || kept.dtype.code == kDLUInt);
    if (!index) refuse(EG_BAD_REQUEST, "eg_compact: kept must be an int32 or uint32 plane");
    if (kept.shape[0] != keep.shape[0]) refuse(EG_BAD_REQUEST, "eg_compact: kept extent differs from keep");
    if (count.dtype.code != kDLUInt || count.dtype.bits != 32 || count.dtype.lanes != 1 || count.shape[0] != 1)
        refuse(EG_BAD_REQUEST, "eg_compact: count must be a one-element uint32 plane");
    if (keep.shape[0] > std::numeric_limits<std::int32_t>::max())
        refuse(EG_BAD_REQUEST, "eg_compact: " + std::to_string(keep.shape[0]) + " elements is above the 2^31 - 1 a compaction indexes");
    check_access(*desc.keep, false, "eg_compact keep");
    check_access(*desc.kept, true, "eg_compact kept");
    check_access(*desc.count, true, "eg_compact count");
    if (overlaps(keep, kept) || overlaps(keep, count) || overlaps(kept, count))
        refuse(EG_BAD_REQUEST, "eg_compact: keep, kept and count must be three distinct planes");
    CompactOp c;
    c.keep = desc.keep;
    c.kept = desc.kept;
    c.count = desc.count;
    c.n = keep.shape[0];
    return c;
}

/** @brief Run a checked compaction. */
void compact_exec(const CompactOp& c, int threads) {
    std::vector<std::uint64_t> scratch;
    compact_lanes(data_of<std::uint8_t>(*c.keep), false, mut_data_of<std::int32_t>(*c.kept), mut_data_of<std::uint32_t>(*c.count), c.n, threads, scratch);
}

namespace {

/** @brief Whether @p t holds one 32-bit integer per element (int32 or uint32),
 * what an index plane is. */
bool is_index_plane(const DLTensor& t)
{
    return t.dtype.lanes == 1 && t.dtype.bits == 32
        && (t.dtype.code == kDLInt || t.dtype.code == kDLUInt);
}

template<class Key>
std::int64_t order_keys(const OrderOp& o, int threads)
{
    return filtering::orderByKeyHost(data_of<Key>(*o.keys), o.n, o.bound,
        mut_data_of<std::int32_t>(*o.order),
        mut_data_of<std::uint32_t>(*o.starts), threads);
}

} // namespace

/** @brief Check a sort request; refuses as eg_order_by_key states. */
OrderOp order_check(const eg_ctx& ctx, const eg_order_desc& desc)
{
    if (desc.flags != 0)
        refuse(EG_BAD_REQUEST,
            "eg_order_desc.flags: unknown bits " + std::to_string(desc.flags));
    check_handle(ctx.planes, desc.keys, "eg_order_desc.keys");
    check_handle(ctx.planes, desc.order, "eg_order_desc.order");
    check_handle(ctx.planes, desc.starts, "eg_order_desc.starts");
    const DLTensor& keys   = desc.keys->tensor->dl_tensor;
    const DLTensor& order  = desc.order->tensor->dl_tensor;
    const DLTensor& starts = desc.starts->tensor->dl_tensor;
    if (desc.bound < 1)
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: bound is " + std::to_string(desc.bound)
                + ", it must be 1 or more");
    if (desc.bound > std::numeric_limits<std::int32_t>::max())
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: bound " + std::to_string(desc.bound)
                + " is above the 2^31 - 1 a place index holds");
    const auto axes = [](const DLTensor& t, const char* name) {
        if (t.ndim != 1)
            refuse(EG_BAD_REQUEST,
                std::string("eg_order_by_key: ") + name + " has "
                    + std::to_string(t.ndim)
                    + " run-time axes, a sort runs along exactly one; sort a "
                      "group over that axis to sort along one of several");
    };
    axes(keys, "keys");
    axes(order, "order");
    axes(starts, "starts");
    if (keys.dtype.lanes == 1 && keys.dtype.code == kDLFloat)
        refuse(EG_NOT_WRITTEN,
            "eg_order_by_key: real keys are not written; keys must be an "
            "integer plane of 8 to 64 bits");
    const bool integer = keys.dtype.lanes == 1
        && (keys.dtype.code == kDLInt || keys.dtype.code == kDLUInt);
    if (!integer || keys.dtype.bits < 8 || keys.dtype.bits > 64)
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: keys must be an integer plane of 8 to 64 bits");
    if (!is_index_plane(order))
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: order must be an int32 or uint32 plane");
    if (order.shape[0] != keys.shape[0])
        refuse(
            EG_BAD_REQUEST, "eg_order_by_key: order extent differs from keys");
    if (!is_index_plane(starts))
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: starts must be an int32 or uint32 plane");
    if (starts.shape[0] != desc.bound + 1)
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: starts has " + std::to_string(starts.shape[0])
                + " elements, bound + 1 is " + std::to_string(desc.bound + 1));
    if (keys.shape[0] > std::numeric_limits<std::int32_t>::max())
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: " + std::to_string(keys.shape[0])
                + " elements is above the 2^31 - 1 a sort indexes");
    check_access(*desc.keys, false, "eg_order_by_key keys");
    check_access(*desc.order, true, "eg_order_by_key order");
    check_access(*desc.starts, true, "eg_order_by_key starts");
    if (overlaps(keys, order) || overlaps(keys, starts)
        || overlaps(order, starts))
        refuse(EG_BAD_REQUEST,
            "eg_order_by_key: keys, order and starts must be three distinct "
            "planes");
    OrderOp o;
    o.keys   = desc.keys;
    o.order  = desc.order;
    o.starts = desc.starts;
    o.bound  = desc.bound;
    o.n      = keys.shape[0];
    return o;
}

/** @brief Run a checked sort: ``filtering::orderByKeyHost`` over the key type
 * of the plane. */
void order_exec(const OrderOp& o, int threads)
{
    const DLDataType& type = o.keys->tensor->dl_tensor.dtype;
    const bool is_signed   = type.code == kDLInt;
    std::int64_t outside   = 0;
    switch (type.bits) {
    case 8:
        outside = is_signed ? order_keys<std::int8_t>(o, threads)
                            : order_keys<std::uint8_t>(o, threads);
        break;
    case 16:
        outside = is_signed ? order_keys<std::int16_t>(o, threads)
                            : order_keys<std::uint16_t>(o, threads);
        break;
    case 32:
        outside = is_signed ? order_keys<std::int32_t>(o, threads)
                            : order_keys<std::uint32_t>(o, threads);
        break;
    default:
        outside = is_signed ? order_keys<std::int64_t>(o, threads)
                            : order_keys<std::uint64_t>(o, threads);
    }
    if (outside != 0)
        refuse(EG_BAD_REQUEST,
            "sort: " + std::to_string(outside) + " keys are outside [0, "
                + std::to_string(o.bound) + ")");
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

/** @brief Check a reorder request; refuses as eg_reorder states. */
ReorderOp reorder_check(const eg_ctx& ctx, const eg_reorder_desc& desc)
{
    if (desc.flags != 0)
        refuse(EG_BAD_REQUEST,
            "eg_reorder_desc.flags: unknown bits "
                + std::to_string(desc.flags));
    if (desc.n_planes == 0 || desc.planes == nullptr)
        refuse(EG_BAD_REQUEST,
            "eg_reorder_desc.planes: a reorder moves one plane or more, "
            "n_planes is "
                + std::to_string(desc.n_planes));
    check_handle(ctx.planes, desc.order, "eg_reorder_desc.order");
    const DLTensor& order = desc.order->tensor->dl_tensor;
    if (order.ndim != 1)
        refuse(EG_BAD_REQUEST,
            "eg_reorder: order has " + std::to_string(order.ndim)
                + " run-time axes, an order is one plane of n indices");
    if (!is_index_plane(order))
        refuse(EG_BAD_REQUEST,
            "eg_reorder: order must be an int32 or uint32 plane");
    if (order.shape[0] > std::numeric_limits<std::int32_t>::max())
        refuse(EG_BAD_REQUEST,
            "eg_reorder: " + std::to_string(order.shape[0])
                + " elements is above the 2^31 - 1 a reorder indexes");
    check_access(*desc.order, false, "eg_reorder order");
    ReorderOp r;
    r.order = desc.order;
    r.n     = order.shape[0];
    for (std::uint32_t p = 0; p < desc.n_planes; ++p) {
        const std::string name = "plane " + std::to_string(p);
        check_handle(ctx.planes, desc.planes[p], "eg_reorder_desc.planes");
        const DLTensor& t = desc.planes[p]->tensor->dl_tensor;
        if (t.ndim < 1)
            refuse(EG_BAD_REQUEST,
                "eg_reorder: " + name
                    + " is a cell, a reorder runs along "
                      "the lane axis, its last");
        if (t.shape[t.ndim - 1] != r.n)
            refuse(EG_BAD_REQUEST,
                "eg_reorder: " + name + " has "
                    + std::to_string(t.shape[t.ndim - 1]) + " lanes, order has "
                    + std::to_string(r.n));
        check_access(*desc.planes[p], true, ("eg_reorder " + name).c_str());
        if (overlaps(t, order))
            refuse(EG_BAD_REQUEST, "eg_reorder: " + name + " overlaps order");
        for (std::uint32_t q = 0; q < p; ++q)
            if (overlaps(t, desc.planes[q]->tensor->dl_tensor))
                refuse(EG_BAD_REQUEST,
                    "eg_reorder: plane " + std::to_string(q) + " and " + name
                        + " overlap; the planes must be distinct");
        r.planes.push_back(desc.planes[p]);
    }
    return r;
}

/** @brief Run a checked reorder: count the entries of ``order`` outside
 * [0, n) in one pass, and only when there are none move the planes. */
void reorder_exec(const ReorderOp& r, int threads)
{
    const std::uint32_t* order = data_of<std::uint32_t>(*r.order);
    // Read as unsigned, a negative int32 entry is above n like any other.
    std::int64_t outside = 0;
    for (std::int64_t j = 0; j < r.n; ++j)
        outside += std::int64_t(order[j]) >= r.n;
    if (outside != 0)
        refuse(EG_BAD_REQUEST,
            "reorder: " + std::to_string(outside)
                + " entries of order are outside [0, " + std::to_string(r.n)
                + ")");
    // A plane of a vector value is (rows..., n), samples last: each row is a
    // plane of n lanes moved by the same order.
    std::vector<filtering::ReorderPlane> planes;
    std::uint32_t widest = 1;
    for (const eg_plane_s* p : r.planes) {
        const DLTensor& t       = p->tensor->dl_tensor;
        const auto bytes        = std::uint32_t(t.dtype.bits / 8);
        const std::int64_t rows = r.n == 0 ? 0 : elements(t) / r.n;
        char* first             = const_cast<char*>(bytes_of(t));
        for (std::int64_t row = 0; row < rows; ++row)
            planes.push_back({ first + row * r.n * bytes, bytes });
        widest = std::max(widest, bytes);
    }
    std::vector<std::uint64_t> stage(
        (filtering::permuteStageBytes(r.n, widest) + 7) / 8);
    filtering::permuteHost(planes.data(), std::uint32_t(planes.size()), order,
        stage.data(), r.n, threads);
}

}  // namespace host
}  // namespace door
}  // namespace eagle
