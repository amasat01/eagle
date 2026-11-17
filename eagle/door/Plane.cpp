// SPDX-License-Identifier: Apache-2.0

// Planes: DLPack tensors taken by eagle, reference-counted; the producer's
// deleter runs exactly once, when the last reference drops. A device context
// also takes kDLCUDA tensors of its own device.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"

#include <aether/aether.h>
#include <aether/banded/banded.h>

#include <cstring>
#include <limits>
#include <memory>
#include <thread>

namespace eagle {
namespace door {

namespace {

/** @brief The dtypes a host plane may carry: float64/32, int64/32/8, uint32, bool (int8: the status plane of eg_run_events). */
bool known_dtype(const DLDataType& t) {
    if (t.lanes != 1) return false;
    switch (t.code) {
        case kDLFloat: return t.bits == 64 || t.bits == 32;
        case kDLInt: return t.bits == 64 || t.bits == 32 || t.bits == 8;
        case kDLUInt: return t.bits == 32;
        case kDLBool: return t.bits == 8;
        default: return false;
    }
}

/** @brief The cell of @p x; a value outside the band is refused with aether's reason, naming @p what. */
std::uint64_t banded_cell(double x, const std::string& what) {
    try {
        return aether::banded::BandedReal::fromDouble(x).toBits();
    } catch (const aether::Error& e) {
        refuse(EG_BAD_REQUEST, what + ": " + e.what());
    }
}

}  // namespace

/** @brief eg_plane_wrap: check the tensor, then take ownership of it. */
void plane_wrap(eg_ctx& ctx, DLManagedTensorVersioned* tensor, eg_plane** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_plane_wrap: out is NULL");
    *out = nullptr;
    if (tensor == nullptr) refuse(EG_BAD_REQUEST, "eg_plane_wrap: tensor is NULL");
    if (tensor->version.major != DLPACK_MAJOR_VERSION)
        refuse(EG_NOT_WRITTEN, "eg_plane_wrap: DLPack major " + std::to_string(tensor->version.major) +
               " (this eagle reads major " + std::to_string(DLPACK_MAJOR_VERSION) + ")");
    const DLTensor& t = tensor->dl_tensor;
    if (t.device.device_type == kDLCUDA && ctx.dev.ordinal >= 0) {
        if (t.device.device_id != ctx.dev.ordinal)
            refuse(EG_BAD_REQUEST, "eg_plane_wrap: a kDLCUDA plane of device " + std::to_string(t.device.device_id) +
                   " on the context of device " + std::to_string(ctx.dev.ordinal));
    } else if (t.device.device_type != kDLCPU) {
        refuse(EG_NOT_WRITTEN, "eg_plane_wrap: device type " + std::to_string(t.device.device_type) +
               (ctx.dev.ordinal >= 0 ? " (planes are kDLCPU or kDLCUDA)" : " on a host-only context (host planes are kDLCPU)"));
    }
    if (!known_dtype(t.dtype))
        refuse(EG_NOT_WRITTEN, "eg_plane_wrap: dtype (code " + std::to_string(t.dtype.code) +
               ", bits " + std::to_string(t.dtype.bits) + ", lanes " +
               std::to_string(t.dtype.lanes) + ") is not one eagle runs");
    if (t.ndim < 0 || (t.ndim > 0 && t.shape == nullptr))
        refuse(EG_BAD_REQUEST, "eg_plane_wrap: ndim " + std::to_string(t.ndim) + " without a shape");
    std::int64_t size = 1;
    for (int i = 0; i < t.ndim; ++i) {
        if (t.shape[i] < 0)
            refuse(EG_BAD_REQUEST, "eg_plane_wrap: shape[" + std::to_string(i) + "] is negative");
        size *= t.shape[i];
    }
    if (t.data == nullptr && size != 0)
        refuse(EG_BAD_REQUEST, "eg_plane_wrap: null data with " + std::to_string(size) + " elements");
    auto* plane = new eg_plane_s;
    plane->ctx = &ctx;
    plane->tensor = tensor;
    try {
        ctx.planes.insert(plane);
    } catch (...) {
        delete plane;
        throw;
    }
    *out = plane;
}

namespace {
/** @brief The memory of a plane eagle made: the tensor, its shape and its bytes. */
struct Made {
    DLManagedTensorVersioned managed{};
    std::vector<std::int64_t> shape;
    std::unique_ptr<unsigned char[]> bytes;   ///< host planes
    eg_ctx* device = nullptr;                 ///< device and pinned planes: the context that allocated ``managed.dl_tensor.data``
    bool pinned = false;
    ~Made() {
        if (device == nullptr) return;
        if (pinned) pinned_free(*device, managed.dl_tensor.data);
        else device_free(*device, managed.dl_tensor.data);
    }
};
}  // namespace

/** @brief A plane of @p shape that eagle owns, zeroed when @p zero: in host memory, in page-locked host memory (@p pinned,
 *  which a plane reports as kDLCPU and an export as kDLCUDAHost) or (@p device) in the context's device memory
 *  (one reference, the caller's to release). */
eg_plane_s* make_plane(eg_ctx& ctx, const DLDataType& dtype, std::vector<std::int64_t> shape, bool device, bool pinned, bool zero) {
    auto made = std::make_unique<Made>();
    std::int64_t n = 1;
    for (std::int64_t e : shape) n *= e;
    made->shape = std::move(shape);
    const std::size_t size = static_cast<std::size_t>(n) * (dtype.bits / 8);
    DLManagedTensorVersioned& m = made->managed;
    if (device) {
        m.dl_tensor.data = size != 0 ? device_alloc(ctx, size, zero) : nullptr;
        made->device = &ctx;     // after the allocation: a failed one has nothing to free
        m.dl_tensor.device.device_type = kDLCUDA;
        m.dl_tensor.device.device_id = ctx.dev.ordinal;
    } else if (pinned) {
        m.dl_tensor.data = size != 0 ? pinned_alloc(ctx, size, zero) : nullptr;
        made->device = &ctx;
        made->pinned = true;
        m.dl_tensor.device.device_type = kDLCPU;
    } else {
        made->bytes = std::make_unique_for_overwrite<unsigned char[]>(size);
        if (zero) std::memset(made->bytes.get(), 0, size);
        m.dl_tensor.data = made->bytes.get();
        m.dl_tensor.device.device_type = kDLCPU;
    }
    m.version.major = DLPACK_MAJOR_VERSION;
    m.version.minor = DLPACK_MINOR_VERSION;
    m.manager_ctx = made.get();
    m.deleter = [](DLManagedTensorVersioned* t) { delete static_cast<Made*>(t->manager_ctx); };
    m.dl_tensor.ndim = static_cast<int>(made->shape.size());
    m.dl_tensor.dtype = dtype;
    m.dl_tensor.shape = made->shape.data();
    eg_plane* plane = nullptr;
    plane_wrap(ctx, &m, &plane);
    made.release();
    plane->pinned = pinned;
    return plane;
}

/** @brief eg_plane_new: check the request, then make the plane through make_plane (the allocator eagle's own planes use). */
void plane_new(eg_ctx& ctx, const eg_plane_desc* desc, eg_plane** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_plane_new: out is NULL");
    *out = nullptr;
    const eg_plane_desc d = read_input(desc, "eg_plane_desc");
    if ((d.flags & ~EG_PLANE_ZERO) != 0)
        refuse(EG_BAD_REQUEST, "eg_plane_new: unknown flag bits " + std::to_string(d.flags & ~EG_PLANE_ZERO));
    if (d.kind != EG_MEM_HOST && d.kind != EG_MEM_PINNED && d.kind != EG_MEM_DEVICE)
        refuse(EG_BAD_REQUEST, "eg_plane_new: unknown kind " + std::to_string(d.kind));
    if (d.kind != EG_MEM_HOST && ctx.dev.ordinal < 0)
        refuse(EG_NOT_WRITTEN, std::string("eg_plane_new: ") + (d.kind == EG_MEM_PINNED ? "PINNED" : "DEVICE") +
               " memory on a host-only context (make the context with EG_CTX_DEVICE)");
    const DLDataType dtype{ static_cast<std::uint8_t>(d.dtype_code), static_cast<std::uint8_t>(d.dtype_bits), 1 };
    if (d.dtype_code > 255 || d.dtype_bits > 255 || !known_dtype(dtype))
        refuse(EG_NOT_WRITTEN, "eg_plane_new: dtype (code " + std::to_string(d.dtype_code) + ", bits " + std::to_string(d.dtype_bits) +
               ") is not one eagle runs");
    if (d.ndim < 0 || (d.ndim > 0 && d.shape == nullptr))
        refuse(EG_BAD_REQUEST, "eg_plane_new: ndim " + std::to_string(d.ndim) + " without a shape");
    std::vector<std::int64_t> shape(d.shape, d.shape + d.ndim);
    std::uint64_t bytes = dtype.bits / 8;
    for (int i = 0; i < d.ndim; ++i) {
        if (shape[i] < 0) refuse(EG_BAD_REQUEST, "eg_plane_new: shape[" + std::to_string(i) + "] is negative");
        if (shape[i] != 0 && bytes > std::numeric_limits<std::int64_t>::max() / static_cast<std::uint64_t>(shape[i]))
            refuse(EG_BAD_REQUEST, "eg_plane_new: the size overflows");
        bytes *= static_cast<std::uint64_t>(shape[i]);
    }
    device_quiet(ctx, "eg_plane_new", EG_BAD_REQUEST, "a plane's memory is allocated outside the capture; make it first");
    *out = make_plane(ctx, dtype, std::move(shape), d.kind == EG_MEM_DEVICE, d.kind == EG_MEM_PINNED, (d.flags & EG_PLANE_ZERO) != 0);
}

namespace {
/** @brief An exported plane: the DLPack tensor handed to the consumer and the reference it keeps on the plane. */
struct Export {
    DLManagedTensorVersioned managed{};
    eg_ctx* ctx = nullptr;
    eg_plane_s* plane = nullptr;
};

/** @brief The consumer's release: wait for a call in progress on the context, drop the plane's reference, free the export.
 *  The last access to the context is the store that ends the wait for others, so a context freed right after is not touched. */
void export_release(DLManagedTensorVersioned* tensor) {
    auto* e = static_cast<Export*>(tensor->manager_ctx);
    eg_ctx& ctx = *e->ctx;
    bool idle = false;
    while (!ctx.busy.compare_exchange_weak(idle, true, std::memory_order_acquire)) {
        idle = false;
        std::this_thread::yield();
    }
    --ctx.exports;
    try {
        plane_release(ctx, e->plane);
    } catch (...) {
    }
    const bool last = ctx.closed && ctx.exports == 0;
    ctx.busy.store(false, std::memory_order_release);
    delete e;
    if (!last) return;
    try {
        ctx_free_all(ctx);
    } catch (...) {
    }
    delete &ctx;
}
}  // namespace

/** @brief eg_plane_dlpack: a DLPack tensor over the plane's memory that keeps the plane alive until the consumer releases it. */
void plane_dlpack(eg_ctx& ctx, eg_plane_s* plane, DLManagedTensorVersioned** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_plane_dlpack: out is NULL");
    *out = nullptr;
    check_handle(ctx.planes, plane, "plane");
    if (!plane->caller_ref) refuse(EG_BAD_REQUEST, "plane: already freed by the caller");
    auto e = std::make_unique<Export>();
    e->ctx = &ctx;
    e->plane = plane;
    e->managed = *plane->tensor;
    e->managed.manager_ctx = e.get();
    e->managed.deleter = export_release;
    if (plane->pinned) {
        e->managed.dl_tensor.device.device_type = kDLCUDAHost;
        e->managed.dl_tensor.device.device_id = 0;
    }
    ++plane->refs;
    ++ctx.exports;
    *out = &e.release()->managed;
}

/** @brief Whether @p op runs on device planes: the planes of @p ctx among @p planes are all device planes (true) or
 *  all host planes (false). A mix, or a device plane on a host-only context, is refused (handles not of @p ctx are
 *  left to @p op's own checks). */
bool device_planes(const eg_ctx& ctx, const char* op, std::initializer_list<const eg_plane_s*> planes) {
    int device = 0, host = 0;
    for (const eg_plane_s* p : planes)
        if (ctx.planes.count(const_cast<eg_plane_s*>(p)) != 0) (p->tensor->dl_tensor.device.device_type == kDLCPU ? host : device) += 1;
    if (device > 0 && host > 0) refuse(EG_BAD_REQUEST, std::string(op) + ": the planes are on both sides; one operation runs on one side");
    if (device > 0 && ctx.dev.ordinal < 0)
        refuse(EG_NOT_WRITTEN, std::string(op) + " of a device plane on a host-only context is not written in this build of the door");
    return device > 0;
}

/** @brief Drop one reference to @p plane; the last one runs the tensor's deleter. */
void plane_release(eg_ctx& ctx, eg_plane_s* plane) {
    if (--plane->refs > 0) return;
    ctx.planes.erase(plane);
    DLManagedTensorVersioned* tensor = plane->tensor;
    delete plane;
    if (tensor->deleter != nullptr) tensor->deleter(tensor);
}

/** @brief eg_banded_pack: stops at the first value outside the band; @p out is then undefined. */
void banded_pack(eg_ctx&, const double* in, std::uint64_t* out, std::uint64_t n) {
    if (n > 0 && (in == nullptr || out == nullptr))
        refuse(EG_BAD_REQUEST, std::string("eg_banded_pack: ") + (in == nullptr ? "in" : "out") + " is NULL with n > 0");
    for (std::uint64_t i = 0; i < n; ++i) out[i] = banded_cell(in[i], "eg_banded_pack: element " + std::to_string(i));
}

/** @brief eg_banded_unpack: every cell decodes, so only NULL is refused. */
void banded_unpack(eg_ctx&, const std::uint64_t* in, double* out, std::uint64_t n) {
    if (n > 0 && (in == nullptr || out == nullptr))
        refuse(EG_BAD_REQUEST, std::string("eg_banded_unpack: ") + (in == nullptr ? "in" : "out") + " is NULL with n > 0");
    for (std::uint64_t i = 0; i < n; ++i) out[i] = aether::banded::BandedReal::fromBits(in[i]).toDouble();
}

/** @brief eg_banded_pack_scalar: one value; @p out is untouched on a refusal. */
void banded_pack_scalar(eg_ctx&, double x, std::uint64_t* out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_banded_pack_scalar: out is NULL");
    *out = banded_cell(x, "eg_banded_pack_scalar");
}

}  // namespace door
}  // namespace eagle
