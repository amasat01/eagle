// SPDX-License-Identifier: Apache-2.0

// Host nodes (eg_host_node_new, eg_host_node_free, eg_host_call): one opaque callback the door runs where a launch would run. The door
// keeps what ordering needs (the declared planes and their access) and nothing of what the callback does. A node is reference-counted
// like a plan: the caller's reference, and one per recorded graph that runs it; the release function runs once, at the last drop.
// The callback runs with its context marked busy, so a door call on that context from inside it is refused by the node's name.
#include "eagle/door/Context.h"
#include "eagle/door/host/Graph.h"

#include <memory>

namespace eagle {
namespace door {

namespace {

/** @brief Marks @p ctx busy inside the callback of the host node called @p name for a scope. */
class Busy {
public:
    Busy(eg_ctx& ctx, const std::string& name) : ctx_(ctx) { ctx_.in_host_node = &name; }
    ~Busy() { ctx_.in_host_node = nullptr; }
    Busy(const Busy&) = delete;
    Busy& operator=(const Busy&) = delete;

private:
    eg_ctx& ctx_;
};

/** @brief Refuse EG_FAULT naming @p node and @p what when the callback returned @p code. */
void check_returned(const eg_host_node_s& node, const char* what, std::int32_t code) {
    if (code != 0) refuse(EG_FAULT, "host node '" + node.name + "'" + what + " failed (" + std::to_string(code) + ")");
}

}  // namespace

void host_node_new(eg_ctx& ctx, const eg_host_node_desc* desc, eg_host_node** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_host_node_new: out is NULL");
    *out = nullptr;
    const eg_host_node_desc d = read_input(desc, "eg_host_node_desc");
    if (d.flags != 0) refuse(EG_BAD_REQUEST, "eg_host_node_desc.flags: unknown bits " + std::to_string(d.flags));
    if (d.name == nullptr || d.name[0] == '\0') refuse(EG_BAD_REQUEST, "eg_host_node_desc.name is NULL or empty");
    if (d.fn == nullptr) refuse(EG_BAD_REQUEST, "eg_host_node_desc.fn is NULL");
    auto node = std::make_unique<eg_host_node_s>();
    node->ctx = &ctx;
    node->name = d.name;
    node->fn = d.fn;
    node->vjp = d.vjp;
    node->user = d.user;
    node->release = d.release;
    ctx.host_nodes.insert(node.get());
    *out = node.release();
}

void host_node_release(eg_ctx& ctx, eg_host_node_s* node) {
    if (--node->refs > 0) return;
    ctx.host_nodes.erase(node);
    const std::unique_ptr<eg_host_node_s> gone(node);
    if (gone->release != nullptr) gone->release(gone->user);
}

std::string host_node_label(const Node& n) { return "host node '" + n.host->name + "'"; }

std::vector<PlaneUse> host_node_uses(const Node& n) {
    std::vector<PlaneUse> uses;
    for (std::size_t i = 0; i < n.hplanes.size(); ++i) {
        const bool reads = (n.haccess[i] & EG_ACCESS_READ) != 0, writes = (n.haccess[i] & EG_ACCESS_WRITE) != 0;
        uses.push_back({ n.hplanes[i], reads && writes ? 2 : writes ? 1 : 0, -1, "planes[" + std::to_string(i) + "]" });
    }
    return uses;
}

void host_node_run(eg_ctx& ctx, const Node& n) {
    if (ctx.dev.ordinal >= 0) device_sync(ctx);    // every device plane is ready, and what the callback enqueues on the stream is ordered before the next segment
    std::vector<DLManagedTensorVersioned> views(n.hplanes.size());     // borrowed copies: no deleter, read-only unless declared written
    std::vector<DLManagedTensorVersioned*> pointers;
    for (std::size_t i = 0; i < views.size(); ++i) {
        views[i] = *n.hplanes[i]->tensor;
        views[i].manager_ctx = nullptr;
        views[i].deleter = nullptr;
        views[i].flags = (n.haccess[i] & EG_ACCESS_WRITE) != 0 ? views[i].flags & ~DLPACK_FLAG_BITMASK_READ_ONLY : views[i].flags | DLPACK_FLAG_BITMASK_READ_ONLY;
        pointers.push_back(&views[i]);
    }
    const Busy busy(ctx, n.host->name);
    check_returned(*n.host, "", n.host->fn(n.host->user, static_cast<std::int32_t>(views.size()), pointers.data()));
}

void host_vjp_run(eg_ctx& ctx, const eg_host_node_s& node, std::int32_t n, DLManagedTensorVersioned* const* primals, DLManagedTensorVersioned* const* adjoints_in, DLManagedTensorVersioned* const* adjoints_out) {
    const Busy busy(ctx, node.name);
    check_returned(node, " vjp", node.vjp(node.user, n, primals, adjoints_in, adjoints_out));
}

/** @brief Check a host call's request and make its node: the planes are declared once each, with a known access. */
Node host_node_check(eg_ctx& ctx, eg_host_node_s* node, std::int32_t n_planes, eg_plane_s* const* planes, const std::uint32_t* access) {
    check_handle(ctx.host_nodes, node, "host node");
    if (!node->caller_ref) refuse(EG_BAD_REQUEST, "host node '" + node->name + "': already freed by the caller");
    if (n_planes < 0 || (n_planes > 0 && (planes == nullptr || access == nullptr)))
        refuse(EG_BAD_REQUEST, "eg_host_call: n_planes " + std::to_string(n_planes) + " without planes and access");
    Node n;
    n.kind = Node::Host;
    n.host = node;
    for (std::int32_t i = 0; i < n_planes; ++i) {
        const std::string at = "eg_host_call: planes[" + std::to_string(i) + "]";
        check_handle(ctx.planes, planes[i], at.c_str());
        if (access[i] < EG_ACCESS_READ || access[i] > (EG_ACCESS_READ | EG_ACCESS_WRITE))
            refuse(EG_BAD_REQUEST, at + ": access " + std::to_string(access[i]) + " is not EG_ACCESS_READ (1), EG_ACCESS_WRITE (2) or both (3)");
        for (std::int32_t k = 0; k < i; ++k)
            if (planes[k] == planes[i]) refuse(EG_BAD_REQUEST, at + " is declared twice (with planes[" + std::to_string(k) + "])");
        if (planes[i]->tensor->dl_tensor.device.device_type != kDLCPU && ctx.dev.ordinal < 0)
            refuse(EG_BAD_REQUEST, at + ": a device plane on a host-only context");
        n.hplanes.push_back(planes[i]);
        n.haccess.push_back(access[i]);
    }
    return n;
}

}  // namespace door
}  // namespace eagle
