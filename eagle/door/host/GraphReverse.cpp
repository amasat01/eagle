// SPDX-License-Identifier: Apache-2.0

// The reverse of a recorded graph of plain launches (EG_RECORD_DIFF, eg_graph_reverse) on the host. At eg_record_end the
// door keeps, per launch, the planes it reads and writes and the links of its vjp, and plans the copies the reverse needs
// (a plane a later leaf overwrites is copied just before it, at each eg_graph_launch, host/Record.cpp). The leaves are the recording's
// launches in forward order with WHILE bodies unrolled (host/Shape.h ReversePlan); a leaf under a conditional runs in the reverse only
// if the tests the replay saved say so.
// eg_graph_reverse then runs the vjps in reverse node order over the planes each node read, summing the adjoints of a plane
// read by several nodes in node order, the seed first (the order of eagle::diff::transpose).
#include "eagle/door/Context.h"
#include "eagle/door/host/Graph.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Team.h"
#include "eagle/diff/Transpose.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <new>
#include <string>

namespace eagle {
namespace door {

namespace {

/** @brief One adjoint of a plane: the seed (node -1) or what the vjp of node ``node`` produced. */
struct Contribution {
    int node = -1;
    Bytes bytes;
};

const DLTensor& tensor(const eg_plane_s& p) { return p.tensor->dl_tensor; }
char* live(const eg_plane_s& p) { return static_cast<char*>(tensor(p).data) + tensor(p).byte_offset; }
std::size_t element_bytes(const eg_plane_s& p) { return static_cast<std::size_t>(tensor(p).dtype.bits / 8); }
std::size_t byte_count(const eg_plane_s& p) { return static_cast<std::size_t>(elements(tensor(p))) * element_bytes(p); }

/** @brief Refuse a unit the graph reverse does not run: a split, SEGMENTED, companion or cell unit, or an index or active set. */
void check_plain_unit(const eg_unit_s& u, const std::string& at) {
    if (!u.stages.empty() || u.seg.present || !u.dispatches.empty() || !u.companions.empty() || !u.cells.empty() ||
        u.index_count_slot >= 0 || u.active_count_slot >= 0)
        refuse(EG_NOT_WRITTEN, at + " (unit '" + u.name + "'): a split, SEGMENTED, companion, cell, index-set or active-set unit is not written in EG_RECORD_DIFF");
}

int find_plane(const GraphDiff& d, const eg_plane_s* p) {
    const auto it = std::find(d.planes.begin(), d.planes.end(), p);
    return it == d.planes.end() ? -1 : static_cast<int>(it - d.planes.begin());
}

/** @brief The links of @p vjp (the vjp of node @p at, a launch of @p plan) in terms of the launch, refusing what is not written. */
std::vector<VjpLink> link_vjp(const GraphDiff& d, const DiffNode& node, const eg_plan_s& plan, const eg_unit_s& vjp, const std::string& at) {
    const eg_unit_s& pu = *plan.unit;
    if (!vjp.derived.present) refuse(EG_BAD_REQUEST, at + ": unit '" + vjp.name + "' has no derived record");
    if (vjp.derived.of_unit != pu.index || vjp.library != pu.library)
        refuse(EG_BAD_REQUEST, at + ": unit '" + vjp.name + "' is not a derived unit of '" + pu.name + "'");
    check_plain_unit(vjp, at);
    std::vector<VjpLink> out;
    std::vector<int> linked(vjp.slots.size(), 0);
    for (std::size_t i = 0; i < vjp.derived.links.size(); ++i) {
        const DerivedFacts::Link& l = vjp.derived.links[i];
        const std::string where = at + ": unit '" + vjp.name + "' links[" + std::to_string(i) + "]";
        if (l.slot < 0 || l.slot >= static_cast<std::int32_t>(vjp.slots.size())) refuse(EG_BAD_REQUEST, where + ".slot is out of range");
        if (linked[static_cast<std::size_t>(l.slot)]++ != 0) refuse(EG_BAD_REQUEST, where + ": slot " + std::to_string(l.slot) + " has a second link");
        if (l.what != EG_WHAT_PRIMAL_SLOT)
            refuse(EG_BAD_REQUEST, where + ": a graph vjp has no tape slot, step plane or time start copy (what " + std::to_string(l.what) + ")");
        if (l.form != EG_FORM_VALUE && l.form != EG_FORM_ADJOINT_IN && l.form != EG_FORM_GRAD)
            refuse(EG_NOT_WRITTEN, where + ": form " + std::to_string(l.form) + " is not written in a graph vjp (VALUE, ADJOINT_IN and GRAD are)");
        if (l.of_slot < 0 || l.of_slot >= static_cast<std::int32_t>(pu.slots.size())) refuse(EG_BAD_REQUEST, where + ".of_slot is out of range");
        const SlotFacts& ps = pu.slots[static_cast<std::size_t>(l.of_slot)];
        VjpLink k;
        k.slot = l.slot;
        k.primal = l.of_slot;
        const auto is = [&](const std::vector<int>& slots_of_node) { return std::find(slots_of_node.begin(), slots_of_node.end(), k.plane) != slots_of_node.end(); };
        if (l.form == EG_FORM_GRAD && ps.role == EG_ROLE_UNIFORM)
            refuse(EG_NOT_WRITTEN, where + ": the gradient of the by-value uniform '" + ps.name + "' is not written; pass the parameter in a one-cell plane");
        if (l.form == EG_FORM_VALUE && ps.role == EG_ROLE_UNIFORM) {
            k.kind = VjpLink::VALUE_UNIFORM;
        } else {
            const eg_plane_s* bound = plan.bindings[static_cast<std::size_t>(l.of_slot)].plane;
            k.plane = bound == nullptr ? -1 : find_plane(d, bound);
            if (k.plane < 0) refuse(EG_BAD_REQUEST, where + ": '" + ps.name + "' of '" + pu.name + "' is no plane the graph tracks (a real read or written plane)");
            if (l.form == EG_FORM_VALUE) {
                k.kind = VjpLink::VALUE_PLANE;
            } else if (l.form == EG_FORM_ADJOINT_IN) {
                if (!is(node.writes)) refuse(EG_BAD_REQUEST, where + ": ADJOINT_IN of '" + ps.name + "', which the launch does not write");
                k.kind = VjpLink::ADJOINT_IN;
            } else {
                if (!is(node.reads)) refuse(EG_BAD_REQUEST, where + ": GRAD of '" + ps.name + "', which the launch does not read");
                k.kind = VjpLink::GRAD;
                k.reduced = l.reduced != 0;
                if (k.reduced && elements(tensor(*d.planes[static_cast<std::size_t>(k.plane)])) != 1)
                    refuse(EG_BAD_REQUEST, where + ": a reduced gradient is the sum over samples into a one-element plane; '" + ps.name + "' is not one");
            }
        }
        out.push_back(k);
    }
    for (std::size_t i = 0; i < vjp.slots.size(); ++i)
        if (linked[i] == 0 && vjp.slots[i].role != EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, at + ": unit '" + vjp.name + "' slot " + std::to_string(i) + " ('" + vjp.slots[i].name + "') has no link");
    return out;
}

/** @brief ((src[0] + src[1]) + ...) over @p count elements of @p es bytes, in the order of @p src, on the host or queued on @p dev. */
Bytes ordered_sum(eg_ctx* dev, std::vector<Contribution>& src, std::size_t count, std::size_t es) {
    if (dev != nullptr) {
        Bytes out(count * es, dev);
        copy_bytes(dev, out.data(), src.front().bytes.data(), count * es);
        for (std::size_t k = 1; k < src.size(); ++k)
            device_columns(*dev, 1, out.data(), static_cast<std::int64_t>(count), src[k].bytes.data(), static_cast<std::int64_t>(count), nullptr,
                           static_cast<std::int64_t>(count), 1, static_cast<unsigned>(es), 0);
        return out;
    }
    Bytes out(count * es);
    const auto go = [&](auto tag) {
        using T = decltype(tag);
        std::vector<const T*> in;
        for (const Contribution& c : src) in.push_back(reinterpret_cast<const T*>(c.bytes.data()));
        diff::orderedSum<T>(reinterpret_cast<T*>(out.data()), in, static_cast<std::ptrdiff_t>(count));
    };
    if (es == 8) go(double{});
    else go(float{});
    return out;
}

/** @brief The sum over the first @p count samples of @p x, a compensated pair in sample order, as a one-element plane. */
Bytes sum_over_samples(eg_ctx* dev, const Bytes& x, std::int64_t count, std::size_t es, int threads) {
    Bytes out(es, dev);
    if (dev != nullptr) {
        Bytes pair(2 * es, dev);
        device_pair_sum(*dev, es == 8, x.data(), count, pair.data(), pair.data() + es);
        double hd[2];
        float hf[2];
        device_read(*dev, pair.data(), es == 8 ? static_cast<void*>(hd) : static_cast<void*>(hf), 2 * es);
        const double h = es == 8 ? hd[0] : hf[0], c = es == 8 ? hd[1] : hf[1];
        const double v = std::isfinite(h) ? h + c : h;    // the value of the compensated pair
        const float vf = static_cast<float>(v);
        device_write(*dev, out.data(), es == 8 ? static_cast<const void*>(&v) : static_cast<const void*>(&vf), es);
    } else if (es == 8) {
        const double v = sample_sum(reinterpret_cast<const double*>(x.data()), count, threads);
        std::memcpy(out.data(), &v, es);
    } else {
        const float v = sample_sum(reinterpret_cast<const float*>(x.data()), count, threads);
        std::memcpy(out.data(), &v, es);
    }
    return out;
}

}  // namespace

bool leaf_ran(const GraphDiff& diff, std::size_t leaf) {
    for (const RGate& g : diff.plan.leaves[leaf].gates)
        if (!took(diff.plan.tests[static_cast<std::size_t>(g.test)], diff.saved[static_cast<std::size_t>(g.test)], g.arm)) return false;
    return true;
}

std::unique_ptr<GraphDiff> diff_plan(eg_ctx& ctx, const std::vector<eg_plan_s*>& launches, const ReversePlan& plan, const eg_record_end_desc& desc) {
    if (desc.n_vjps < 0 || (desc.n_vjps > 0 && desc.vjps == nullptr)) refuse(EG_BAD_REQUEST, "eg_record_end_desc.n_vjps without vjps");
    auto diff = std::make_unique<GraphDiff>();
    diff->nodes.resize(launches.size());
    diff->plan = plan;
    diff->saved.resize(plan.tests.size());
    const bool device = !launches.empty() && launches.front()->structure == EG_STRUCTURE_DEVICE;
    if (device) diff->dev = &ctx;
    try {
        std::vector<std::string> first_user;       // per plane: who named it first, for the overlap reason
        for (std::size_t i = 0; i < launches.size(); ++i) {
            const eg_plan_s& plan = *launches[i];
            const std::string at = "eg_record_end: launch " + std::to_string(i);
            diff->nodes[i].plan = launches[i];
            check_plain_unit(*plan.unit, at);
            if ((plan.structure == EG_STRUCTURE_DEVICE) != device)
                refuse(EG_NOT_WRITTEN, at + ": host and DEVICE plans in one EG_RECORD_DIFF graph are not written");
            std::map<int, std::string> named;       // plane -> the slot of this node that names it
            for (std::size_t s = 0; s < plan.bindings.size(); ++s) {
                const SlotFacts& sf = plan.unit->slots[s];
                eg_plane_s* p = plan.bindings[s].plane;
                const bool reads = reads_role(sf.role), writes = writes_role(sf.role);
                if (p == nullptr || sf.made || sf.dtype != EG_DTYPE_REAL || (!reads && !writes)) continue;
                int idx = find_plane(*diff, p);
                if (idx < 0) {
                    ++p->refs;
                    diff->planes.push_back(p);
                    first_user.push_back("launch " + std::to_string(i) + " slot '" + sf.name + "'");
                    idx = static_cast<int>(diff->planes.size()) - 1;
                    host::check_access(*p, writes, (at + " slot '" + sf.name + "'").c_str());
                }
                if (!named.emplace(idx, sf.name).second)
                    refuse(EG_BAD_REQUEST, at + ": slots '" + named[idx] + "' and '" + sf.name + "' name one plane");
                (writes ? diff->nodes[i].writes : diff->nodes[i].reads).push_back(idx);
            }
        }
        for (std::size_t a = 0; a < diff->planes.size(); ++a)
            for (std::size_t b = a + 1; b < diff->planes.size(); ++b)
                if (host::overlaps(tensor(*diff->planes[a]), tensor(*diff->planes[b])))
                    refuse(EG_BAD_REQUEST, "eg_record_end: the planes of " + first_user[a] + " and " + first_user[b] + " are views of overlapping memory");
        for (std::int32_t k = 0; k < desc.n_vjps; ++k) {
            const eg_node_vjp v = desc.vjps[k];
            const std::string at = "eg_record_end_desc.vjps[" + std::to_string(k) + "]";
            if (v.launch < 0 || static_cast<std::size_t>(v.launch) >= launches.size()) refuse(EG_BAD_REQUEST, at + ".launch " + std::to_string(v.launch) + " is out of range");
            DiffNode& node = diff->nodes[static_cast<std::size_t>(v.launch)];
            if (node.vjp != nullptr) refuse(EG_BAD_REQUEST, at + ".launch " + std::to_string(v.launch) + " is given twice");
            check_handle(ctx.units, v.vjp, (at + ".vjp").c_str());
            node.links = link_vjp(*diff, node, *launches[static_cast<std::size_t>(v.launch)], *v.vjp, at);
            node.vjp = v.vjp;
            ++v.vjp->refs;
        }
        // The copies the reverse needs: for each plane a leaf reads, the first leaf at or after it that writes the plane, and
        // the next ones while that writer may not run (it sits under a conditional).
        const auto writes = [&](const RLeaf& l, int p) {
            const std::vector<int>& w = diff->nodes[static_cast<std::size_t>(l.launch)].writes;
            return std::find(w.begin(), w.end(), p) != w.end();
        };
        for (std::size_t i = 0; i < plan.leaves.size(); ++i)
            for (int p : diff->nodes[static_cast<std::size_t>(plan.leaves[i].launch)].reads)
                for (std::size_t m = i; m < plan.leaves.size(); ++m) {
                    if (!writes(plan.leaves[m], p)) continue;
                    const auto same = [&](const Snapshot& s) { return s.node == m && s.plane == p; };
                    if (std::none_of(diff->snapshots.begin(), diff->snapshots.end(), same)) {
                        Snapshot snap;
                        snap.node = m;
                        snap.plane = p;
                        snap.size = byte_count(*diff->planes[static_cast<std::size_t>(p)]);
                        try {
                            snap.bytes = Bytes(snap.size, diff->dev);
                        } catch (const std::exception&) {
                            refuse(EG_FAULT, "eg_record_end: no memory for the copy of a plane before leaf " + std::to_string(m) + " of the tape");
                        }
                        diff->snapshots.push_back(std::move(snap));
                    }
                    if (plan.leaves[m].gates.empty()) break;
                }
        for (eg_plan_s* plan : launches) {
            const auto same = [&](const GraphDiff::PlanMark& m) { return m.plan == plan; };
            if (std::none_of(diff->marks.begin(), diff->marks.end(), same)) diff->marks.push_back({ plan, plan->binds, plan->launches });
        }
    } catch (...) {
        diff_free(ctx, *diff);
        throw;
    }
    return diff;
}

void diff_free(eg_ctx& ctx, GraphDiff& diff) {
    for (eg_plane_s* p : diff.planes) plane_release(ctx, p);
    for (const DiffNode& n : diff.nodes)
        if (n.vjp != nullptr) unit_release(ctx, n.vjp);
    diff.planes.clear();
    diff.nodes.clear();
}

void diff_check_launch(const GraphDiff& diff) {
    for (const GraphDiff::PlanMark& m : diff.marks)
        if (m.plan->binds != m.binds)
            refuse(EG_BAD_REQUEST, "eg_graph_launch: a plan of this EG_RECORD_DIFF graph was rebound (eg_bind, eg_clear) since eg_record_end; "
                   "its tape holds the planes it was recorded with");
}

void diff_launched(GraphDiff& diff) {
    for (GraphDiff::PlanMark& m : diff.marks) m.launches = m.plan->launches;
    diff.launched = true;
}

void graph_reverse(eg_ctx& ctx, eg_graph_s* graph, const eg_graph_reverse_desc* desc) {
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr) refuse(EG_BAD_REQUEST, "eg_graph_reverse while recording (eg_record_begin .. eg_record_end)");
    GraphDiff* tape = nullptr;
    if (device_graph(ctx, graph)) {
        tape = device_graph_diff(ctx, graph);
    } else {
        check_handle(ctx.graphs, graph, "graph");
        tape = graph->diff.get();
    }
    if (tape == nullptr) refuse(EG_BAD_REQUEST, "eg_graph_reverse: the graph was not ended with EG_RECORD_DIFF");
    GraphDiff& d = *tape;
    eg_ctx* const D = d.dev;       // a DEVICE graph: its buffers, vjp launches and sums are queued on the context's stream
    if (!d.launched) refuse(EG_BAD_REQUEST, "eg_graph_reverse: the graph was never launched (eg_graph_launch)");
    if (D != nullptr) device_diff_read_saved(*D, d);    // the tests the last replay saw: one read-back
    for (const GraphDiff::PlanMark& m : d.marks) {
        if (m.plan->binds != m.binds)
            refuse(EG_BAD_REQUEST, "eg_graph_reverse: a plan of the graph (unit '" + m.plan->unit->name + "') was rebound (eg_bind, eg_clear) after its last launch");
        if (m.plan->launches != m.launches)
            refuse(EG_BAD_REQUEST, "eg_graph_reverse: a plan of the graph (unit '" + m.plan->unit->name + "') was launched outside the graph after its last launch");
    }
    const eg_graph_reverse_desc r = read_input(desc, "eg_graph_reverse_desc");
    if (r.flags != 0) refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc.flags: unknown bits " + std::to_string(r.flags));
    if (r.n_seeds < 0 || (r.n_seeds > 0 && (r.outputs == nullptr || r.seeds == nullptr)) ||
        r.n_grads < 0 || (r.n_grads > 0 && (r.inputs == nullptr || r.grads == nullptr)))
        refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc: a count without its arrays");

    const std::size_t n_leaves = d.plan.leaves.size();
    const auto checked = [&](eg_plane_s* given, int k, const char* what, bool write) {
        const std::string at = std::string("eg_graph_reverse_desc.") + what + "[" + std::to_string(k) + "]";
        check_handle(ctx.planes, given, at.c_str());
        const int idx = find_plane(d, given);
        if (idx < 0) refuse(EG_BAD_REQUEST, at + " is no plane of the graph");
        host::check_access(*given, write, at.c_str());
        if (tensor(*given).device.device_type != (D != nullptr ? kDLCUDA : kDLCPU))
            refuse(EG_BAD_REQUEST, at + (D != nullptr ? ": a device graph takes device planes" : ": a host graph takes host planes"));
        return idx;
    };
    const auto same_shape = [&](const eg_plane_s& a, const eg_plane_s& b, const std::string& at) {
        if (tensor(a).dtype.code != tensor(b).dtype.code || tensor(a).dtype.bits != tensor(b).dtype.bits || elements(tensor(a)) != elements(tensor(b)))
            refuse(EG_BAD_REQUEST, at + ": the dtype or element count differs from the graph plane");
    };
    std::vector<std::vector<Contribution>> pending(d.planes.size());
    std::vector<char> written(d.planes.size(), 0);
    for (const DiffNode& n : d.nodes)
        for (int p : n.writes) written[static_cast<std::size_t>(p)] = 1;
    for (std::int32_t k = 0; k < r.n_seeds; ++k) {
        check_handle(ctx.planes, r.seeds[k], "eg_graph_reverse_desc.seeds");
        const int p = checked(r.outputs[k], k, "outputs", false);
        if (written[static_cast<std::size_t>(p)] == 0)
            refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc.outputs[" + std::to_string(k) + "] is not written by the graph");
        if (!pending[static_cast<std::size_t>(p)].empty()) refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc.outputs[" + std::to_string(k) + "] is given twice");
        same_shape(*d.planes[static_cast<std::size_t>(p)], *r.seeds[k], "eg_graph_reverse_desc.seeds[" + std::to_string(k) + "]");
        Contribution seed;
        seed.bytes = Bytes(byte_count(*r.seeds[k]), D);
        copy_bytes(D, seed.bytes.data(), live(*r.seeds[k]), byte_count(*r.seeds[k]));
        pending[static_cast<std::size_t>(p)].push_back(std::move(seed));
    }
    std::vector<char> input(d.planes.size(), 0), seen(d.planes.size(), 0);
    for (const DiffNode& n : d.nodes) {
        for (int p : n.reads) {
            if (seen[static_cast<std::size_t>(p)] == 0) input[static_cast<std::size_t>(p)] = 1;
            seen[static_cast<std::size_t>(p)] = 1;
        }
        for (int p : n.writes) seen[static_cast<std::size_t>(p)] = 1;
    }
    std::vector<int> wanted;
    for (std::int32_t k = 0; k < r.n_grads; ++k) {
        check_handle(ctx.planes, r.grads[k], "eg_graph_reverse_desc.grads");
        const int p = checked(r.inputs[k], k, "inputs", false);
        if (input[static_cast<std::size_t>(p)] == 0) refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc.inputs[" + std::to_string(k) + "] is not read before the graph writes it");
        host::check_access(*r.grads[k], true, ("eg_graph_reverse_desc.grads[" + std::to_string(k) + "]").c_str());
        same_shape(*d.planes[static_cast<std::size_t>(p)], *r.grads[k], "eg_graph_reverse_desc.grads[" + std::to_string(k) + "]");
        wanted.push_back(p);
    }

    // The adjoint a plane has accumulated: the seed first, then the nodes in node order; none: nullopt-like empty vector.
    const auto collect = [&](int p, bool& any) {
        std::vector<Contribution> src = std::move(pending[static_cast<std::size_t>(p)]);
        pending[static_cast<std::size_t>(p)].clear();
        const eg_plane_s& plane = *d.planes[static_cast<std::size_t>(p)];
        any = !src.empty();
        if (!any) return Bytes(byte_count(plane), D);
        std::stable_sort(src.begin(), src.end(), [](const Contribution& a, const Contribution& b) { return a.node < b.node; });
        if (src.size() == 1) return std::move(src.front().bytes);
        return ordered_sum(D, src, static_cast<std::size_t>(elements(tensor(plane))), element_bytes(plane));
    };
    const int threads = host::team_threads(ctx);
    for (std::size_t back = 0; back < n_leaves; ++back) {
        const std::size_t i = n_leaves - 1 - back;
        if (!leaf_ran(d, i)) continue;         // a leaf that did not run is the identity: what it would write keeps its adjoint
        const DiffNode& node = d.nodes[static_cast<std::size_t>(d.plan.leaves[i].launch)];
        const eg_plan_s& primal = *node.plan;
        std::vector<Bytes> adjoint;
        bool flows = false;
        for (int w : node.writes) {
            bool any = false;
            adjoint.push_back(collect(w, any));
            flows = flows || any;
        }
        if (node.vjp == nullptr) {
            if (flows) refuse(EG_BAD_REQUEST, "eg_graph_reverse: a gradient flows through launch " + std::to_string(d.plan.leaves[i].launch) + " (unit '" + primal.unit->name + "'), which has no vjp");
            continue;
        }
        const std::size_t n = static_cast<std::size_t>(primal.n_samples);
        eg_plan_s* vp = inner_plan(ctx, node.vjp, primal.structure, primal.n_samples);
        std::vector<Bytes> grads(node.links.size());
        try {
            for (std::size_t k = 0; k < node.links.size(); ++k) {
                const VjpLink& l = node.links[k];
                if (l.kind == VjpLink::VALUE_UNIFORM) {
                    bind_value(*vp, l.slot, primal.bindings[static_cast<std::size_t>(l.primal)].value, sizeof(primal.bindings[0].value));
                    continue;
                }
                const eg_plane_s& plane = *d.planes[static_cast<std::size_t>(l.plane)];
                char* data = nullptr;
                if (l.kind == VjpLink::VALUE_PLANE) {
                    data = live(plane);
                    for (std::size_t m = i; m < n_leaves && data == live(plane); ++m) {    // the first leaf at or after this one that ran and overwrites it
                        const std::vector<int>& w = d.nodes[static_cast<std::size_t>(d.plan.leaves[m].launch)].writes;
                        if (std::find(w.begin(), w.end(), l.plane) == w.end() || !leaf_ran(d, m)) continue;
                        for (Snapshot& s : d.snapshots)
                            if (s.node == m && s.plane == l.plane) data = reinterpret_cast<char*>(s.bytes.data());
                        break;
                    }
                } else if (l.kind == VjpLink::ADJOINT_IN) {
                    const std::vector<int>& w = node.writes;
                    data = adjoint[static_cast<std::size_t>(std::find(w.begin(), w.end(), l.plane) - w.begin())].data();
                } else {
                    grads[k] = Bytes(l.reduced ? n * element_bytes(plane) : byte_count(plane), D);
                    data = grads[k].data();
                }
                bind_raw(*vp, l.slot, data, l.kind == VjpLink::GRAD && l.reduced ? n : bound_count(*vp, l.slot, tensor(plane)));
            }
            if (D != nullptr) device_launch_prefix(*vp, primal.n_samples, primal.n_samples);
            else host::run_plan(*vp, threads);
        } catch (...) {
            inner_free(ctx, vp);
            throw;
        }
        inner_free(ctx, vp);
        for (std::size_t k = 0; k < node.links.size(); ++k) {
            const VjpLink& l = node.links[k];
            if (l.kind != VjpLink::GRAD) continue;
            const eg_plane_s& plane = *d.planes[static_cast<std::size_t>(l.plane)];
            Contribution c;
            c.node = static_cast<int>(i);
            c.bytes = l.reduced ? sum_over_samples(D, grads[k], primal.n_samples, element_bytes(plane), threads) : std::move(grads[k]);
            pending[static_cast<std::size_t>(l.plane)].push_back(std::move(c));
        }
    }
    for (std::int32_t k = 0; k < r.n_grads; ++k) {
        bool any = false;
        Bytes g = collect(wanted[static_cast<std::size_t>(k)], any);
        copy_bytes(D, live(*r.grads[k]), g.data(), byte_count(*r.grads[k]));
    }
    if (D != nullptr) device_sync(ctx);     // the gradients land before eg_graph_reverse returns
}

}  // namespace door
}  // namespace eagle
