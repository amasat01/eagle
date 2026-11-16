// SPDX-License-Identifier: Apache-2.0

// Planes: DLPack tensors taken by eagle, reference-counted; the producer's
// deleter runs exactly once, when the last reference drops. A device context
// also takes kDLCUDA tensors of its own device.
#include "eagle/door/Context.h"

#include <memory>

namespace eagle {
namespace door {

namespace {

/** @brief The dtypes a host plane may carry: float64/32, int64/32, uint32, bool. */
bool known_dtype(const DLDataType& t) {
    if (t.lanes != 1) return false;
    switch (t.code) {
        case kDLFloat: return t.bits == 64 || t.bits == 32;
        case kDLInt: return t.bits == 64 || t.bits == 32;
        case kDLUInt: return t.bits == 32;
        case kDLBool: return t.bits == 8;
        default: return false;
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
/** @brief The memory of a plane eagle made: the tensor, its shape and its zeroed bytes. */
struct Made {
    DLManagedTensorVersioned managed{};
    std::vector<std::int64_t> shape;
    std::vector<unsigned char> bytes;
};
}  // namespace

/** @brief A zeroed host plane of @p shape that eagle owns (one reference, the caller's to release). */
eg_plane_s* make_plane(eg_ctx& ctx, const DLDataType& dtype, std::vector<std::int64_t> shape) {
    auto made = std::make_unique<Made>();
    std::int64_t n = 1;
    for (std::int64_t e : shape) n *= e;
    made->shape = std::move(shape);
    made->bytes.assign(static_cast<std::size_t>(n) * (dtype.bits / 8), 0);
    DLManagedTensorVersioned& m = made->managed;
    m.version.major = DLPACK_MAJOR_VERSION;
    m.version.minor = DLPACK_MINOR_VERSION;
    m.manager_ctx = made.get();
    m.deleter = [](DLManagedTensorVersioned* t) { delete static_cast<Made*>(t->manager_ctx); };
    m.dl_tensor.data = made->bytes.data();
    m.dl_tensor.device.device_type = kDLCPU;
    m.dl_tensor.ndim = static_cast<int>(made->shape.size());
    m.dl_tensor.dtype = dtype;
    m.dl_tensor.shape = made->shape.data();
    eg_plane* plane = nullptr;
    plane_wrap(ctx, &m, &plane);
    made.release();
    return plane;
}

/** @brief Refuse @p op over a device plane among @p planes (handles not of @p ctx are left to @p op's own checks). */
void host_planes_only(const eg_ctx& ctx, const char* op, std::initializer_list<const eg_plane_s*> planes) {
    for (const eg_plane_s* p : planes)
        if (ctx.planes.count(const_cast<eg_plane_s*>(p)) != 0 && p->tensor->dl_tensor.device.device_type != kDLCPU)
            refuse(EG_NOT_WRITTEN, std::string(op) + " of a device plane is not written in this build of the door");
}

/** @brief Drop one reference to @p plane; the last one runs the tensor's deleter. */
void plane_release(eg_ctx& ctx, eg_plane_s* plane) {
    if (--plane->refs > 0) return;
    ctx.planes.erase(plane);
    DLManagedTensorVersioned* tensor = plane->tensor;
    delete plane;
    if (tensor->deleter != nullptr) tensor->deleter(tensor);
}

}  // namespace door
}  // namespace eagle
