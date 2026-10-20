// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file eagle_xla.cpp
 * @brief ``libeagle_xla.so``: the generic XLA FFI handler behind
 *        ``eagle.frameworks.jax`` (see ``eagle_xla.h`` for the contract).
 */

#include "xla/eagle_xla.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "eagle/reduce/canonical.h"
#include "plugin/gref_layout.h"
#include "seam/eagle_backend.h"
#include "xla/ffi/api/c_api.h"

#define EAGLE_XLA_API extern "C" __attribute__((visibility("default")))

namespace {

/** The outcome of a step: ok, or an XLA error code with its message. */
struct Error {
    XLA_FFI_Error_Code code = XLA_FFI_Error_Code_OK;
    std::string text;
    bool failure() const { return code != XLA_FFI_Error_Code_OK; }
    static Error Success() { return {}; }
    static Error InvalidArgument(std::string t) { return {XLA_FFI_Error_Code_INVALID_ARGUMENT, std::move(t)}; }
    static Error Internal(std::string t) { return {XLA_FFI_Error_Code_INTERNAL, std::move(t)}; }
};

XLA_FFI_Error* toXla(const XLA_FFI_Api* api, const Error& e)
{
    if (!e.failure())
        return nullptr;
    XLA_FFI_Error_Create_Args a{};
    a.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
    a.message = e.text.c_str(); // XLA copies it
    a.errc = e.code;
    return api->XLA_FFI_Error_Create(&a);
}

/** Bytes per element of the XLA types a kernel can take; 0 for any other. */
std::size_t widthOf(std::int32_t dtype)
{
    switch (dtype) {
    case XLA_FFI_DataType_PRED: case XLA_FFI_DataType_S8: case XLA_FFI_DataType_U8: return 1;
    case XLA_FFI_DataType_S16: case XLA_FFI_DataType_U16: case XLA_FFI_DataType_F16: case XLA_FFI_DataType_BF16: return 2;
    case XLA_FFI_DataType_S32: case XLA_FFI_DataType_U32: case XLA_FFI_DataType_F32: return 4;
    case XLA_FFI_DataType_S64: case XLA_FFI_DataType_U64: case XLA_FFI_DataType_F64: case XLA_FFI_DataType_C64: return 8;
    case XLA_FFI_DataType_C128: return 16;
    default: return 0;
    }
}

using HostEntry = void (*)(void* const*, std::int64_t, std::int64_t, std::int64_t);
using RunDevice = eagle_backend_status (*)(const struct eagle_backend_launch_desc*, std::int32_t*);
using LastError = const char* (*)(void);
using MemsetDevice = eagle_backend_status (*)(std::uint64_t, std::uint64_t, std::uint64_t);
using MemcpyDevice = eagle_backend_status (*)(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
using ReduceDevice = eagle_backend_status (*)(struct eagle_backend_reduce_desc*);

/** A registered kernel: the C recipe with its arrays copied in. */
struct Recipe {
    std::uint32_t platform = EAGLE_XLA_PLATFORM_CPU;
    std::int32_t ordinal = 0;
    std::uint32_t block = 256;
    std::uint64_t entry = 0, lastError = 0, memset = 0, memcpy = 0, function = 0, reduce = 0;
    std::int32_t nargs = 0, nrets = 0, sampleBuffer = 0, indexBytes = 4;
    std::vector<std::int32_t> dtype;
    std::vector<std::uint8_t> zero;
    std::vector<std::int32_t> slotKind, slotBuffer;
    std::vector<std::uint8_t> slotCell;
    std::vector<std::int32_t> aliasOperand, aliasResult;
    std::int32_t cellRow = 1, policy = 0, workA = -1, workB = -1;
    std::vector<std::int32_t> cellRows, cellComp, cellOut, cellOp;
};

std::shared_mutex g_mutex;
std::unordered_map<std::string, std::shared_ptr<const Recipe>> g_table;

std::shared_ptr<const Recipe> find(std::string_view digest)
{
    std::shared_lock<std::shared_mutex> lock(g_mutex);
    const auto it = g_table.find(std::string(digest));
    return it == g_table.end() ? nullptr : it->second;
}

std::string quoted(std::string_view digest) { return "eagle: kernel '" + std::string(digest) + "'"; }

/** One XLA buffer, operand or result, seen the same way. */
struct View {
    void* data = nullptr;
    std::int32_t dtype = 0;
    std::int64_t last = 0;      // the trailing extent (1 for a rank-0 buffer)
    std::size_t bytes = 0;
};

View viewOf(const XLA_FFI_Buffer& b)
{
    View v;
    v.data = b.data;
    v.dtype = static_cast<std::int32_t>(b.dtype);
    v.last = b.rank == 0 ? 1 : b.dims[b.rank - 1];
    std::size_t elements = 1;
    for (std::int64_t i = 0; i < b.rank; ++i)
        elements *= static_cast<std::size_t>(b.dims[i]);
    v.bytes = elements * widthOf(v.dtype);
    return v;
}

/** Every buffer of one call, checked against the recipe; the error, if any. */
Error gather(const Recipe& r, std::string_view digest, const XLA_FFI_CallFrame& f, std::vector<View>& out)
{
    const XLA_FFI_Args& args = f.args;
    const XLA_FFI_Rets& rets = f.rets;
    if (args.size != r.nargs || rets.size != r.nrets)
        return Error::InvalidArgument(quoted(digest) + " takes " + std::to_string(r.nargs) + " operands and "
            + std::to_string(r.nrets) + " results, this call has " + std::to_string(args.size) + " and "
            + std::to_string(rets.size));
    out.clear();
    out.reserve(static_cast<std::size_t>(r.nargs + r.nrets));
    for (std::int64_t i = 0; i < args.size; ++i) {
        if (args.types[i] != XLA_FFI_ArgType_BUFFER)
            return Error::InvalidArgument(quoted(digest) + ": operand " + std::to_string(i) + " is not a buffer");
        out.push_back(viewOf(*static_cast<const XLA_FFI_Buffer*>(args.args[i])));
    }
    for (std::int64_t i = 0; i < rets.size; ++i) {
        if (rets.types[i] != XLA_FFI_RetType_BUFFER)
            return Error::InvalidArgument(quoted(digest) + ": result " + std::to_string(i) + " is not a buffer");
        out.push_back(viewOf(*static_cast<const XLA_FFI_Buffer*>(rets.rets[i])));
    }
    for (std::size_t i = 0; i < out.size(); ++i)
        if (out[i].dtype != r.dtype[i]) {
            const bool operand = static_cast<std::int32_t>(i) < r.nargs;
            const std::size_t at = operand ? i : i - static_cast<std::size_t>(r.nargs);
            return Error::InvalidArgument(quoted(digest) + ": " + (operand ? "operand " : "result ")
                + std::to_string(at) + " has XLA element type " + std::to_string(out[i].dtype) + ", this kernel build takes "
                + std::to_string(r.dtype[i]) + " (a float32 call runs the float32 build, a float64 call the float64 one)");
        }
    return Error::Success();
}

/** Honour the aliases, zero the flagged results, build the PODs and the params
 *  array; n out. */
template <class Copy, class Zero>
Error prepare(const Recipe& r, std::string_view digest, std::vector<View>& views, std::int32_t deviceType,
    std::int32_t deviceId, Copy&& copy, Zero&& zero, std::vector<std::array<std::uint64_t, 5>>& boxes, std::vector<void*>& params,
    std::int64_t& n)
{
    using eagle::plugin::GRefMirror;
    using eagle::plugin::ScalarHandle;
    static_assert(sizeof(GRefMirror) <= sizeof(std::array<std::uint64_t, 5>), "a box holds one GRefMirror");
    static_assert(sizeof(ScalarHandle) <= sizeof(std::array<std::uint64_t, 5>), "a box holds one ScalarHandle");
    n = views[static_cast<std::size_t>(r.sampleBuffer)].last;
    const std::size_t nslots = r.slotKind.size();
    for (std::size_t s = 0; s < nslots; ++s) {
        const std::int32_t b = r.slotBuffer[s];
        if (b < 0 || r.slotCell[s])
            continue;
        if (views[static_cast<std::size_t>(b)].last != n)
            return Error::InvalidArgument(quoted(digest) + ": slot " + std::to_string(s) + " holds "
                + std::to_string(views[static_cast<std::size_t>(b)].last) + " samples, the call has " + std::to_string(n));
    }
    // XLA may give an aliased result its own buffer (a fusion around the call does): the
    // carried state must still be what the kernel reads, so move it there first
    for (std::size_t a = 0; a < r.aliasOperand.size(); ++a) {
        const View& from = views[static_cast<std::size_t>(r.aliasOperand[a])];
        const View& to = views[static_cast<std::size_t>(r.nargs + r.aliasResult[a])];
        if (from.data == to.data || from.bytes == 0)
            continue;
        if (from.bytes != to.bytes)
            return Error::InvalidArgument(quoted(digest) + ": aliased operand " + std::to_string(r.aliasOperand[a])
                + " and result " + std::to_string(r.aliasResult[a]) + " differ in size");
        if (Error e = copy(to.data, from.data, from.bytes); e.failure())
            return e;
    }
    for (std::int32_t i = 0; i < r.nrets; ++i)
        if (r.zero[static_cast<std::size_t>(i)]) {
            View& v = views[static_cast<std::size_t>(r.nargs + i)];
            if (v.bytes == 0 && widthOf(v.dtype) == 0)
                return Error::InvalidArgument(quoted(digest) + ": result " + std::to_string(i) + " has an element type this handler cannot zero-fill");
            if (v.bytes > 0)
                if (Error e = zero(v.data, v.bytes); e.failure())
                    return e;
        }
    boxes.assign(nslots, std::array<std::uint64_t, 5>{});
    params.assign(nslots, nullptr);
    for (std::size_t s = 0; s < nslots; ++s) {
        void* box = boxes[s].data();
        params[s] = box;
        const std::int32_t kind = r.slotKind[s];
        if (kind == EAGLE_XLA_SLOT_NSAMPLES) {
            if (r.indexBytes == 8) {
                const std::uint64_t value = static_cast<std::uint64_t>(n);
                std::memcpy(box, &value, sizeof value);
            } else {
                const std::uint32_t value = static_cast<std::uint32_t>(n);
                std::memcpy(box, &value, sizeof value);
            }
            continue;
        }
        const View& v = views[static_cast<std::size_t>(r.slotBuffer[s])];
        if (kind == EAGLE_XLA_SLOT_GREF) {
            GRefMirror g;
            g.data_ = static_cast<double*>(v.data);
            g.samples_ = static_cast<std::uint64_t>(n);
            g.compStride_ = static_cast<std::uint64_t>(n);
            g.sampleStride_ = 1;
            g.deviceType_ = deviceType;
            g.deviceId_ = deviceId;
            std::memcpy(box, &g, sizeof g);
        } else {
            ScalarHandle h;
            h.data = v.data;
            h.samples = static_cast<std::uint64_t>(r.slotCell[s] ? v.last : n);
            h.stride = 1;
            h.deviceType = deviceType;
            h.deviceId = deviceId;
            std::memcpy(box, &h, sizeof h);
        }
    }
    return Error::Success();
}

std::shared_ptr<const Recipe> lookup(std::string_view digest, std::uint32_t platform, Error& err)
{
    auto r = find(digest);
    if (!r) {
        err = Error::InvalidArgument(quoted(digest)
            + " is not loaded in this process (an exported or cached module needs "
              "eagle.frameworks.jax.preload first)");
        return nullptr;
    }
    if (r->platform != platform) {
        err = Error::InvalidArgument(quoted(digest) + " was registered for another platform");
        return nullptr;
    }
    return r;
}

/** One cell on the host: the canonical reduction of its rows (+ companions). */
template <class T>
void finishCellHost(const Recipe& r, const std::vector<View>& views, std::size_t c, std::int64_t nrows)
{
    namespace canon = eagle::reduce::canonical;
    struct Ptr {
        const T* p;
        T operator()(eagle::idx_t i) const { return p[i]; }
    };
    const auto rows = static_cast<const T*>(views[static_cast<std::size_t>(r.cellRows[c])].data);
    const std::int32_t comp = r.cellComp[c];
    T* out = static_cast<T*>(views[static_cast<std::size_t>(r.cellOut[c])].data);
    const auto n = static_cast<eagle::idx_t>(nrows);
    if (r.cellOp[c] == EAGLE_XLA_CELL_SUM) {
        const canon::Column<Ptr> x{ { rows }, 1, 0 };
        if (comp < 0) {
            *out = canon::partialRange<T, aether::SumOp<T>>(x, 0, n).value();
        } else {
            const canon::Column<Ptr> k{ { static_cast<const T*>(views[static_cast<std::size_t>(comp)].data) }, 1, 0 };
            *out = canon::partialRange<T, aether::SumOp<T>>(x, 0, n, k).value();
        }
    } else {
        const canon::Column<Ptr> x{ { rows }, 1, 0 };
        *out = r.cellOp[c] == EAGLE_XLA_CELL_MAX ? canon::partialRange<T, aether::MaxOp<T>>(x, 0, n).value()
                                                 : canon::partialRange<T, canon::MinOp<T>>(x, 0, n).value();
    }
}

Error launchCpu(std::string_view digest, const XLA_FFI_CallFrame& frame)
{
    Error err = Error::Success();
    const auto r = lookup(digest, EAGLE_XLA_PLATFORM_CPU, err);
    if (!r)
        return err;
    std::vector<View> views;
    if (Error e = gather(*r, digest, frame, views); e.failure())
        return e;
    std::vector<std::array<std::uint64_t, 5>> boxes;
    std::vector<void*> params;
    std::int64_t n = 0;
    const auto zero = [](void* p, std::size_t bytes) {
        std::memset(p, 0, bytes);
        return Error::Success();
    };
    const auto copy = [](void* to, const void* from, std::size_t bytes) {
        std::memcpy(to, from, bytes);
        return Error::Success();
    };
    if (Error e = prepare(*r, digest, views, eagle::plugin::kEagleAbiDeviceCPU, 0, copy, zero, boxes, params, n);
        e.failure())
        return e;
    if (n > 0)
        reinterpret_cast<HostEntry>(r->entry)(params.data(), 0, n, n);
    const std::int64_t nrows = (n + r->cellRow - 1) / r->cellRow;
    for (std::size_t k = 0; k < r->cellOut.size(); ++k) {
        const std::int32_t dt = views[static_cast<std::size_t>(r->cellOut[k])].dtype;
        if (dt == XLA_FFI_DataType_F32)
            finishCellHost<float>(*r, views, k, nrows);
        else
            finishCellHost<double>(*r, views, k, nrows);
    }
    return Error::Success();
}

Error launchCuda(void* stream, std::int32_t ordinal, std::string_view digest, const XLA_FFI_CallFrame& frame)
{
    Error err = Error::Success();
    const auto r = lookup(digest, EAGLE_XLA_PLATFORM_CUDA, err);
    if (!r)
        return err;
    if (r->ordinal != ordinal)
        return Error::InvalidArgument(quoted(digest) + " was loaded for CUDA device " + std::to_string(r->ordinal)
            + ", this call runs on device " + std::to_string(ordinal) + " (one GPU per process until 0.7)");
    std::vector<View> views;
    if (Error e = gather(*r, digest, frame, views); e.failure())
        return e;
    const auto lastError = [&r]() {
        const char* text = r->lastError ? reinterpret_cast<LastError>(r->lastError)() : nullptr;
        return std::string(text ? text : "no message");
    };
    const auto zero = [&](void* p, std::size_t bytes) {
        const auto memsetDevice = reinterpret_cast<MemsetDevice>(r->memset);
        if (memsetDevice(reinterpret_cast<std::uint64_t>(p), static_cast<std::uint64_t>(bytes),
                reinterpret_cast<std::uint64_t>(stream))
            != EAGLE_BACKEND_OK)
            return Error::Internal(quoted(digest) + ": zero-filling a result failed: " + lastError());
        return Error::Success();
    };
    const auto copy = [&](void* to, const void* from, std::size_t bytes) {
        const auto memcpyDevice = reinterpret_cast<MemcpyDevice>(r->memcpy);
        if (memcpyDevice(reinterpret_cast<std::uint64_t>(to), reinterpret_cast<std::uint64_t>(from),
                static_cast<std::uint64_t>(bytes), reinterpret_cast<std::uint64_t>(stream))
            != EAGLE_BACKEND_OK)
            return Error::Internal(quoted(digest) + ": copying an aliased operand failed: " + lastError());
        return Error::Success();
    };
    std::vector<std::array<std::uint64_t, 5>> boxes;
    std::vector<void*> params;
    std::int64_t n = 0;
    if (Error e = prepare(*r, digest, views, eagle::plugin::kEagleAbiDeviceCUDA, ordinal, copy, zero, boxes, params, n);
        e.failure())
        return e;
    std::vector<std::uint64_t> addresses(params.size());
    for (std::size_t i = 0; i < params.size(); ++i)
        addresses[i] = reinterpret_cast<std::uint64_t>(params[i]);
    struct eagle_backend_launch_desc d {};
    d.struct_size = sizeof d;
    d.device = -1; // the device of XLA's stream
    d.block = r->block;
    d.function = r->function;
    d.params = addresses.data();
    d.param_sizes = nullptr;
    d.nparams = static_cast<std::int64_t>(addresses.size());
    d.base = 0;
    d.count = n;
    d.n_samples = n;
    d.stream = reinterpret_cast<std::uint64_t>(stream);
    std::int32_t launched = 0;
    if (reinterpret_cast<RunDevice>(r->entry)(&d, &launched) != EAGLE_BACKEND_OK)
        return Error::Internal(quoted(digest) + ": launch failed: " + lastError());
    // the cells, on the same stream: the seam's canonical reduce over the XLA-owned work results
    const std::int64_t nrows = (n + r->cellRow - 1) / r->cellRow;
    for (std::size_t k = 0; n > 0 && k < r->cellOut.size(); ++k) {
        const View& out = views[static_cast<std::size_t>(r->cellOut[k])];
        struct eagle_backend_reduce_desc rd {};
        rd.struct_size = sizeof rd;
        rd.dtype = out.dtype == XLA_FFI_DataType_F32 ? EAGLE_BACKEND_REDUCE_F32 : EAGLE_BACKEND_REDUCE_F64;
        rd.op = r->cellOp[k] == EAGLE_XLA_CELL_SUM   ? EAGLE_BACKEND_REDUCE_OP_SUM
              : r->cellOp[k] == EAGLE_XLA_CELL_MAX ? EAGLE_BACKEND_REDUCE_OP_MAX
                                                   : EAGLE_BACKEND_REDUCE_OP_MIN;
        rd.device = -1;
        rd.in = reinterpret_cast<std::uint64_t>(views[static_cast<std::size_t>(r->cellRows[k])].data);
        rd.out = reinterpret_cast<std::uint64_t>(out.data);
        rd.work_a = reinterpret_cast<std::uint64_t>(views[static_cast<std::size_t>(r->workA)].data);
        rd.work_b = reinterpret_cast<std::uint64_t>(views[static_cast<std::size_t>(r->workB)].data);
        rd.n = nrows;
        rd.stream = reinterpret_cast<std::uint64_t>(stream);
        rd.policy = static_cast<std::uint32_t>(r->policy);
        rd.columns = 1;
        rd.comp_in = r->cellComp[k] < 0 ? 0 : reinterpret_cast<std::uint64_t>(views[static_cast<std::size_t>(r->cellComp[k])].data);
        if (reinterpret_cast<ReduceDevice>(r->reduce)(&rd) != EAGLE_BACKEND_OK)
            return Error::Internal(quoted(digest) + ": finishing a cell failed: " + lastError());
    }
    return Error::Success();
}

bool validKind(std::int32_t k) { return k >= EAGLE_XLA_SLOT_GREF && k <= EAGLE_XLA_SLOT_INOUT; }

} // namespace

EAGLE_XLA_API std::int32_t eagle_xla_register(const char* digest, const struct eagle_xla_recipe* recipe)
{
    if (digest == nullptr || recipe == nullptr || recipe->struct_size != sizeof(struct eagle_xla_recipe))
        return EAGLE_XLA_BAD_RECIPE;
    const eagle_xla_recipe& c = *recipe;
    if ((c.platform != EAGLE_XLA_PLATFORM_CPU && c.platform != EAGLE_XLA_PLATFORM_CUDA) || c.entry == 0
        || (c.platform == EAGLE_XLA_PLATFORM_CUDA && (c.memset == 0 || c.memcpy == 0 || c.function == 0)))
        return EAGLE_XLA_BAD_RECIPE;
    if (c.nargs < 0 || c.nrets < 1 || c.nslots < 0 || c.dtype == nullptr || c.zero == nullptr
        || (c.nslots > 0 && (c.slot_kind == nullptr || c.slot_buffer == nullptr || c.slot_cell == nullptr))
        || c.nalias < 0 || (c.nalias > 0 && (c.alias_operand == nullptr || c.alias_result == nullptr))
        || c.sample_buffer < 0 || c.sample_buffer >= c.nargs + c.nrets || (c.index_bytes != 4 && c.index_bytes != 8))
        return EAGLE_XLA_BAD_SLOT;
    auto r = std::make_shared<Recipe>();
    r->platform = c.platform;
    r->ordinal = c.ordinal;
    r->block = c.block == 0 ? 256u : c.block;
    r->entry = c.entry;
    r->lastError = c.last_error;
    r->memset = c.memset;
    r->memcpy = c.memcpy;
    r->function = c.function;
    r->reduce = c.reduce;
    r->nargs = c.nargs;
    r->nrets = c.nrets;
    r->sampleBuffer = c.sample_buffer;
    r->indexBytes = c.index_bytes;
    r->dtype.assign(c.dtype, c.dtype + c.nargs + c.nrets);
    r->zero.assign(c.zero, c.zero + c.nrets);
    for (std::int32_t a = 0; a < c.nalias; ++a) {
        if (c.alias_operand[a] < 0 || c.alias_operand[a] >= c.nargs || c.alias_result[a] < 0 || c.alias_result[a] >= c.nrets
            || c.dtype[c.alias_operand[a]] != c.dtype[c.nargs + c.alias_result[a]])
            return EAGLE_XLA_BAD_SLOT;
        r->aliasOperand.push_back(c.alias_operand[a]);
        r->aliasResult.push_back(c.alias_result[a]);
    }
    for (std::int32_t s = 0; s < c.nslots; ++s) {
        const std::int32_t kind = c.slot_kind[s], buffer = c.slot_buffer[s];
        if (!validKind(kind))
            return EAGLE_XLA_BAD_SLOT;
        if (kind == EAGLE_XLA_SLOT_NSAMPLES ? buffer != -1 : (buffer < 0 || buffer >= c.nargs + c.nrets))
            return EAGLE_XLA_BAD_SLOT;
        if (kind == EAGLE_XLA_SLOT_INOUT && (buffer < c.nargs || c.zero[buffer - c.nargs] != 0))
            return EAGLE_XLA_BAD_SLOT;
        r->slotKind.push_back(kind);
        r->slotBuffer.push_back(buffer);
        r->slotCell.push_back(c.slot_cell[s] ? 1 : 0);
    }
    if (c.ncells < 0 || (c.ncells > 0 && (c.cell_rows == nullptr || c.cell_comp == nullptr || c.cell_out == nullptr
            || c.cell_op == nullptr || c.cell_row < 1))
        || (c.ncells > 0 && c.platform == EAGLE_XLA_PLATFORM_CUDA
            && (c.reduce == 0 || c.work_a < 0 || c.work_b < 0 || c.work_a >= c.nargs + c.nrets
                || c.work_b >= c.nargs + c.nrets)))
        return EAGLE_XLA_BAD_SLOT;
    const std::int32_t nbuf = c.nargs + c.nrets;
    for (std::int32_t k = 0; k < c.ncells; ++k) {
        const std::int32_t d = c.dtype[c.cell_out[k]];
        if (c.cell_rows[k] < 0 || c.cell_rows[k] >= nbuf || c.cell_comp[k] < -1 || c.cell_comp[k] >= nbuf
            || c.cell_out[k] < c.nargs || c.cell_out[k] >= nbuf || c.cell_op[k] < EAGLE_XLA_CELL_SUM
            || c.cell_op[k] > EAGLE_XLA_CELL_MIN || (d != XLA_FFI_DataType_F32 && d != XLA_FFI_DataType_F64))
            return EAGLE_XLA_BAD_SLOT;
        r->cellRows.push_back(c.cell_rows[k]);
        r->cellComp.push_back(c.cell_comp[k]);
        r->cellOut.push_back(c.cell_out[k]);
        r->cellOp.push_back(c.cell_op[k]);
    }
    r->cellRow = std::max<std::int32_t>(1, c.cell_row);
    r->policy = c.policy;
    r->workA = c.work_a;
    r->workB = c.work_b;
    std::unique_lock<std::shared_mutex> lock(g_mutex);
    g_table[digest] = std::move(r);
    return EAGLE_XLA_OK;
}

namespace {

/** The "digest" string attribute of a call frame. */
Error digestOf(const XLA_FFI_CallFrame& f, std::string_view& digest)
{
    for (std::int64_t i = 0; i < f.attrs.size; ++i) {
        const XLA_FFI_ByteSpan& name = *f.attrs.names[i];
        if (std::string_view(name.ptr, name.len) != "digest")
            continue;
        if (f.attrs.types[i] != XLA_FFI_AttrType_STRING)
            break;
        const XLA_FFI_ByteSpan& value = *static_cast<const XLA_FFI_ByteSpan*>(f.attrs.attrs[i]);
        digest = std::string_view(value.ptr, value.len);
        return Error::Success();
    }
    return Error::InvalidArgument("eagle: the call has no string attribute 'digest'");
}

/** XLA asks a handler for its API version and traits before using it. */
XLA_FFI_Error* reportMetadata(XLA_FFI_CallFrame* f)
{
    auto* ext = reinterpret_cast<XLA_FFI_Metadata_Extension*>(f->extension_start);
    if (ext->extension_base.struct_size < XLA_FFI_Metadata_Extension_STRUCT_SIZE
        || ext->metadata->struct_size < XLA_FFI_Metadata_STRUCT_SIZE)
        return toXla(f->api, Error::InvalidArgument("eagle: XLA's metadata structs are older than this handler's"));
    ext->metadata->api_version = XLA_FFI_Api_Version{XLA_FFI_Api_Version_STRUCT_SIZE, nullptr, XLA_FFI_API_MAJOR,
        XLA_FFI_API_MINOR};
    ext->metadata->traits = 0;
    ext->metadata->state_type_id = XLA_FFI_UNKNOWN_TYPE_ID;
    return nullptr;
}

/** The common entry: struct size, metadata, stage and digest checks; then `run`. */
template <class Run>
XLA_FFI_Error* handle(XLA_FFI_CallFrame* f, Run&& run)
{
    if (f->struct_size < XLA_FFI_CallFrame_STRUCT_SIZE)
        return toXla(f->api, Error::InvalidArgument("eagle: XLA's call frame is older than this handler's"));
    if (f->extension_start != nullptr && f->extension_start->type == XLA_FFI_Extension_Metadata)
        return reportMetadata(f);
    if (f->stage != XLA_FFI_ExecutionStage_EXECUTE)
        return toXla(f->api, Error::InvalidArgument("eagle: the handler runs in the execute stage only"));
    std::string_view digest;
    if (Error e = digestOf(*f, digest); e.failure())
        return toXla(f->api, e);
    return run(digest);
}

} // namespace

EAGLE_XLA_API XLA_FFI_Error* eagle_xla_launch_cpu(XLA_FFI_CallFrame* f)
{
    return handle(f, [f](std::string_view digest) { return toXla(f->api, launchCpu(digest, *f)); });
}

EAGLE_XLA_API XLA_FFI_Error* eagle_xla_launch_cuda(XLA_FFI_CallFrame* f)
{
    return handle(f, [f](std::string_view digest) -> XLA_FFI_Error* {
        XLA_FFI_Stream_Get_Args s{};
        s.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
        s.ctx = f->ctx;
        if (XLA_FFI_Error* e = f->api->XLA_FFI_Stream_Get(&s))
            return e;
        XLA_FFI_DeviceOrdinal_Get_Args d{};
        d.struct_size = XLA_FFI_DeviceOrdinal_Get_Args_STRUCT_SIZE;
        d.ctx = f->ctx;
        if (XLA_FFI_Error* e = f->api->XLA_FFI_DeviceOrdinal_Get(&d))
            return e;
        return toXla(f->api, launchCuda(s.stream, d.device_ordinal, digest, *f));
    });
}
