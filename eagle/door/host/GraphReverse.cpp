// SPDX-License-Identifier: Apache-2.0

// The reverse of a recorded graph (EG_RECORD_DIFF, eg_graph_reverse), on the host and the device. At eg_record_end the door keeps, per
// leaf (a launch, copy, sum reduce, transform or product), the planes it reads and writes and what reverses it: a launch's vjp links, the
// adjoint transform of a transform, the two gradient products of a product. It plans the copies the reverse needs (a plane a later leaf
// overwrites is copied just before it, at each eg_graph_launch, host/Record.cpp). The leaves are the recording's in forward order with
// WHILE bodies unrolled (host/Shape.h ReversePlan); a leaf under a conditional runs in the reverse only if the tests the replay saved say so.
// eg_graph_reverse then reverses the leaves in reverse node order over the planes each node read, summing the adjoints of a plane
// read by several nodes in node order, the seed first (the order of eagle::diff::transpose).
#include "eagle/door/Context.h"
#include "eagle/door/host/Graph.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Team.h"
#include "eagle/diff/Transpose.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <new>
#include <string>
#include <tuple>

namespace eagle {
namespace door {

namespace {

/** @brief One adjoint of a plane: the seed (node -1) or what the vjp of node ``node`` produced. */
struct Contribution {
    int node = -1;
    int taint = -1;                             ///< the launch ordinal of the scatter-add vjp that fed it (or fed what it came from); -1: none
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
                k.scatter = vjp.slots[static_cast<std::size_t>(l.slot)].role == EG_ROLE_ACCUM_OUT;
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

/** @brief A plane over the memory @p data with the shape and dtype of @p like: for the calls that take planes, run over a buffer of the tape. */
class View {
public:
    View(const eg_plane_s& like, void* data) : managed_(*like.tensor) {
        managed_.dl_tensor.data = data;
        managed_.dl_tensor.byte_offset = 0;
        plane_.ctx = like.ctx;
        plane_.tensor = &managed_;
        plane_.caller_ref = false;
    }
    View(const View&) = delete;
    View& operator=(const View&) = delete;
    eg_plane_s* get() { return &plane_; }

private:
    DLManagedTensorVersioned managed_;
    eg_plane_s plane_;
};

/** @brief How a leaf that is not a launch registers its planes: ``(plane, writes, who, at)`` returns its index in GraphDiff::planes. */
using Track = std::function<int(eg_plane_s*, bool, const std::string&, const std::string&)>;

/** @brief The words that name the run, copy, reduce, transform or product leaf @p n (the @p ordinal-th of them) in a refusal. */
std::string leaf_name(const Node& n, std::size_t ordinal) {
    const char* const what = n.kind == Node::Run      ? "eg_run"
                             : n.kind == Node::Copy   ? "eg_copy"
                             : n.kind == Node::Reduce ? "eg_reduce"
                             : n.kind == Node::Fft    ? "eg_fft_run"
                                                      : "eg_gemm_run";
    return std::string("eg_record_end: ") + what + " (leaf " + std::to_string(ordinal) + ")";
}

/** @brief What the non-launch leaf @p n reads and writes, and what reverses it, into @p dn. @p ordinal is its place among those leaves.
 *  Only a float plane has an adjoint: a call with none is left with no accesses; a reduce that is not a 1-D sum has no gradient to pull back. */
void other_leaf(eg_ctx& ctx, const Node& n, DiffNode& dn, std::size_t ordinal, const Track& track) {
    const std::string at = leaf_name(n, ordinal);
    const auto floats = [](const eg_plane_s* p) { return tensor(*p).dtype.code == kDLFloat; };
    const auto read = [&](eg_plane_s* p, const char* name) { dn.reads.push_back(track(p, false, at + " " + name, at + " " + name)); };
    const auto write = [&](eg_plane_s* p, const char* name) { dn.writes.push_back(track(p, true, at + " " + name, at + " " + name)); };
    dn.kind = n.kind;
    switch (n.kind) {
        case Node::Run: {
            dn.runner = n.runner;
            const RunSlots run = recorded_run_slots(*n.runner);
            const eg_plan_s& primal = *n.runner->plans[0];
            const auto plane_at = [&](int slot) { return primal.bindings[static_cast<std::size_t>(slot)].plane; };
            for (int slot : run.states) {
                dn.slots.push_back(slot);
                read(plane_at(slot), primal.unit->slots[static_cast<std::size_t>(slot)].name.c_str());
            }
            for (int slot : run.states) write(plane_at(slot), primal.unit->slots[static_cast<std::size_t>(slot)].name.c_str());
            for (int slot : run.reads) {
                dn.slots.push_back(slot);
                read(plane_at(slot), primal.unit->slots[static_cast<std::size_t>(slot)].name.c_str());
            }
            return;
        }
        case Node::Copy:
            if (!floats(n.src)) return;
            read(n.src, "src");
            write(n.dst, "dst");
            return;
        case Node::Reduce:
            if (!floats(n.reduce.in)) return;
            dn.reduce = n.reduce;
            read(n.reduce.in, "in");
            write(n.reduce.out, "out");
            return;
        case Node::Fft:
            read(n.src, "in");
            write(n.dst, "out");
            dn.fft = n.fft;
            ++dn.fft->refs;
            dn.adjoint = fft_adjoint(ctx, *n.fft);
            return;
        default:
            read(n.src, "a");
            read(n.rhs, "b");
            dn.valued = dn.reads;
            if ((n.gemm->call.flags & EG_GEMM_ACCUMULATE) != 0) read(n.dst, "c");
            write(n.dst, "c");
            dn.gemm = *n.gemm;
            std::tie(dn.gemm_a, dn.gemm_b) = gemm_adjoints(dn.gemm);
            return;
    }
}

}  // namespace

void leaf_checks(const std::vector<const Node*>& others) {
    for (std::size_t j = 0; j < others.size(); ++j) {
        const Node& n = *others[j];
        const std::string at = leaf_name(n, j);
        if (n.kind == Node::Run) {
            if (!run_reversible(*n.runner))
                refuse(EG_NOT_WRITTEN, at + ": the run of runner " + std::to_string(j) + " in a DIFF recording needs EG_RUN_RECORD to be reversed");
            for (std::size_t i = 0; i < j; ++i)
                if (others[i]->kind == Node::Run && others[i]->runner == n.runner)
                    refuse(EG_BAD_REQUEST, at + ": its runner is run twice in the recording and its record holds one run: record the second run with a second runner");
        }
        if (n.kind != Node::Reduce) continue;
        if (n.reduce.op != EG_REDUCE_SUM) refuse(EG_BAD_REQUEST, at + " has no gradient eagle can pull back (a tie has no single answer); only op='sum' reverses");
        if (tensor(*n.reduce.in).dtype.code == kDLFloat && (n.reduce.cols != 1 || n.reduce.comp != nullptr))
            refuse(EG_NOT_WRITTEN, at + ": a 2-D or companion reduce: eagle does not pull back yet; only a 1-D op='sum' reduce reverses");
    }
}

bool leaf_ran(const GraphDiff& diff, std::size_t leaf) {
    for (const RGate& g : diff.plan.leaves[leaf].gates)
        if (!took(diff.plan.tests[static_cast<std::size_t>(g.test)], diff.saved[static_cast<std::size_t>(g.test)], g.arm)) return false;
    return true;
}

std::unique_ptr<GraphDiff> diff_plan(eg_ctx& ctx, const std::vector<eg_plan_s*>& launches, const std::vector<const Node*>& others, const ReversePlan& plan, const eg_record_end_desc& desc) {
    if (desc.n_vjps < 0 || (desc.n_vjps > 0 && desc.vjps == nullptr)) refuse(EG_BAD_REQUEST, "eg_record_end_desc.n_vjps without vjps");
    leaf_checks(others);
    auto diff = std::make_unique<GraphDiff>();
    diff->nodes.resize(launches.size() + others.size());
    diff->plan = plan;
    diff->saved.resize(plan.tests.size());
    const bool device = !launches.empty() ? launches.front()->structure == EG_STRUCTURE_DEVICE : !others.empty() && tensor(*(others.front()->kind == Node::Reduce ? others.front()->reduce.out : others.front()->dst)).device.device_type != kDLCPU;
    if (device) diff->dev = &ctx;
    try {
        std::vector<std::string> first_user;       // per plane: who named it first, for the overlap reason
        const Track track = [&](eg_plane_s* p, bool writes, const std::string& who, const std::string& at) {
            int idx = find_plane(*diff, p);
            if (idx < 0) {
                ++p->refs;
                diff->planes.push_back(p);
                first_user.push_back(who);
                idx = static_cast<int>(diff->planes.size()) - 1;
                host::check_access(*p, writes, at.c_str());
            }
            return idx;
        };
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
                const int idx = track(p, writes, "launch " + std::to_string(i) + " slot '" + sf.name + "'", at + " slot '" + sf.name + "'");
                if (!named.emplace(idx, sf.name).second)
                    refuse(EG_BAD_REQUEST, at + ": slots '" + named[idx] + "' and '" + sf.name + "' name one plane");
                (writes ? diff->nodes[i].writes : diff->nodes[i].reads).push_back(idx);
            }
            diff->nodes[i].valued = diff->nodes[i].reads;
        }
        for (std::size_t j = 0; j < others.size(); ++j) other_leaf(ctx, *others[j], diff->nodes[launches.size() + j], j, track);
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
        // The planes the body of each windowed WHILE writes, and the copies of them its forward keeps at the start of every w-th iteration.
        diff->loops.resize(plan.loops.size());
        for (std::size_t L = 0; L < plan.loops.size(); ++L) {
            const RLoop& loop = plan.loops[L];
            LoopTape& tape = diff->loops[L];
            for (int m = loop.first_leaf; m < loop.first_leaf + loop.cap * loop.per_iter; ++m) {
                const DiffNode& node = diff->nodes[static_cast<std::size_t>(plan.leaves[static_cast<std::size_t>(m)].node)];
                if (node.writes.empty()) continue;
                if (node.kind == Node::Launch) {
                    const eg_plan_s& primal = *node.plan;
                    for (std::size_t sl = 0; sl < primal.bindings.size(); ++sl)     // the rebuild re-runs this launch: every plane it binds must be tracked
                        if (primal.bindings[sl].plane != nullptr && find_plane(*diff, primal.bindings[sl].plane) < 0)
                            refuse(EG_NOT_WRITTEN, "eg_record_end: launch " + std::to_string(plan.leaves[static_cast<std::size_t>(m)].node) + " (unit '" + primal.unit->name +
                                   "') in a WHILE body binds the plane of slot '" + primal.unit->slots[sl].name + "', which is not a real plane the tape tracks; the loop's window cannot rebuild it");
                }
                for (int p : node.writes)
                    if (std::find(tape.planes.begin(), tape.planes.end(), p) == tape.planes.end()) tape.planes.push_back(p);
            }
            tape.ckpt.resize(static_cast<std::size_t>(loop.groups));
            for (std::vector<Bytes>& group : tape.ckpt)
                for (int p : tape.planes) {
                    try {
                        group.emplace_back(byte_count(*diff->planes[static_cast<std::size_t>(p)]), diff->dev);
                    } catch (const std::exception&) {
                        refuse(EG_FAULT, "eg_record_end: no memory for the window checkpoints of the WHILE of node " + std::to_string(loop.cond));
                    }
                }
        }
        // A run's reverse reads the planes it only reads as they are live (its record keeps its states, not those): no later leaf may overwrite one.
        for (std::size_t i = 0; i < plan.leaves.size(); ++i) {
            const DiffNode& run = diff->nodes[static_cast<std::size_t>(plan.leaves[i].node)];
            if (run.kind != Node::Run) continue;
            for (std::size_t k = run.writes.size(); k < run.reads.size(); ++k)
                for (std::size_t m = i + 1; m < plan.leaves.size(); ++m) {
                    const std::vector<int>& w = diff->nodes[static_cast<std::size_t>(plan.leaves[m].node)].writes;
                    if (std::find(w.begin(), w.end(), run.reads[k]) != w.end())
                        refuse(EG_NOT_WRITTEN, "eg_record_end: leaf " + std::to_string(plan.leaves[m].node) + " overwrites the plane of slot '" +
                               run.runner->plans[0]->unit->slots[static_cast<std::size_t>(run.slots[k])].name + "', which a run before it reads and its reverse needs as it was");
                }
        }
        // The copies the reverse needs: for each plane a leaf reads, the first leaf at or after it that writes the plane, and
        // the next ones while that writer may not run (it sits under a conditional). A leaf copy of a windowed WHILE shares the
        // buffer of its body leaf in iteration (k mod w), which the reverse refills window by window; a reader outside the loop
        // reads the loop's start-of-loop checkpoint instead.
        const auto writes = [&](const RLeaf& l, int p) {
            const std::vector<int>& w = diff->nodes[static_cast<std::size_t>(l.node)].writes;
            return std::find(w.begin(), w.end(), p) != w.end();
        };
        const std::function<int(std::size_t, int)> snapshot_of = [&](std::size_t m, int p) {
            for (std::size_t k = 0; k < diff->snapshots.size(); ++k)
                if (diff->snapshots[k].node == m && diff->snapshots[k].plane == p) return static_cast<int>(k);
            Snapshot snap;
            snap.node = m;
            snap.plane = p;
            snap.size = byte_count(*diff->planes[static_cast<std::size_t>(p)]);
            const RLeaf& lm = plan.leaves[m];
            if (lm.loop >= 0 && lm.iter >= plan.loops[static_cast<std::size_t>(lm.loop)].w) {
                const RLoop& loop = plan.loops[static_cast<std::size_t>(lm.loop)];
                const std::size_t local = (m - static_cast<std::size_t>(loop.first_leaf)) % static_cast<std::size_t>(loop.per_iter);
                snap.alias = snapshot_of(static_cast<std::size_t>(loop.first_leaf) + static_cast<std::size_t>(lm.iter % loop.w) * static_cast<std::size_t>(loop.per_iter) + local, p);
            } else {
                try {
                    snap.bytes = Bytes(snap.size, diff->dev);
                } catch (const std::exception&) {
                    refuse(EG_FAULT, "eg_record_end: no memory for the copy of a plane before leaf " + std::to_string(m) + " of the tape");
                }
            }
            diff->snapshots.push_back(std::move(snap));
            return static_cast<int>(diff->snapshots.size()) - 1;
        };
        for (std::size_t i = 0; i < plan.leaves.size(); ++i)
            for (int p : diff->nodes[static_cast<std::size_t>(plan.leaves[i].node)].valued)
                for (std::size_t m = i; m < plan.leaves.size(); ++m) {
                    if (!writes(plan.leaves[m], p)) continue;
                    if (plan.leaves[m].loop < 0 || plan.leaves[m].loop == plan.leaves[i].loop) snapshot_of(m, p);
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
    for (const DiffNode& n : diff.nodes) {
        if (n.vjp != nullptr) unit_release(ctx, n.vjp);
        if (n.fft != nullptr) fft_release(ctx, n.fft);
        if (n.adjoint != nullptr) fft_release(ctx, n.adjoint);
    }
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
    if ((r.flags & ~EG_REVERSE_STRICT) != 0) refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc.flags: unknown bits " + std::to_string(r.flags & ~EG_REVERSE_STRICT));
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
        check_handle(ctx.planes, r.inputs[k], "eg_graph_reverse_desc.inputs");
        if (tensor(*r.inputs[k]).dtype.code != kDLFloat)
            refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc.inputs[" + std::to_string(k) + "] is not a float plane: only a float input has a gradient");
        const int p = checked(r.inputs[k], k, "inputs", false);
        if (input[static_cast<std::size_t>(p)] == 0) refuse(EG_BAD_REQUEST, "eg_graph_reverse_desc.inputs[" + std::to_string(k) + "] is not read before the graph writes it");
        host::check_access(*r.grads[k], true, ("eg_graph_reverse_desc.grads[" + std::to_string(k) + "]").c_str());
        same_shape(*d.planes[static_cast<std::size_t>(p)], *r.grads[k], "eg_graph_reverse_desc.grads[" + std::to_string(k) + "]");
        wanted.push_back(p);
    }

    // The adjoint a plane has accumulated: the seed first, then the nodes in node order; none: nullopt-like empty vector.
    // @p origin gets the first scatter-add launch ordinal among the contributions, or -1.
    const auto collect = [&](int p, bool& any, int& origin) {
        std::vector<Contribution> src = std::move(pending[static_cast<std::size_t>(p)]);
        pending[static_cast<std::size_t>(p)].clear();
        const eg_plane_s& plane = *d.planes[static_cast<std::size_t>(p)];
        any = !src.empty();
        origin = -1;
        for (const Contribution& c : src)
            if (origin < 0) origin = c.taint;
        if (!any) return Bytes(byte_count(plane), D);
        std::stable_sort(src.begin(), src.end(), [](const Contribution& a, const Contribution& b) { return a.node < b.node; });
        if (src.size() == 1) return std::move(src.front().bytes);
        return ordered_sum(D, src, static_cast<std::size_t>(elements(tensor(plane))), element_bytes(plane));
    };
    const int threads = host::team_threads(ctx);
    // The buffer a snapshot entry reads and fills: a windowed WHILE's later iterations share the first window's.
    const auto store = [&](Snapshot& s) -> Bytes& { return s.alias >= 0 ? d.snapshots[static_cast<std::size_t>(s.alias)].bytes : s.bytes; };
    const auto position = [](const LoopTape& tape, int plane) { return static_cast<std::size_t>(std::find(tape.planes.begin(), tape.planes.end(), plane) - tape.planes.begin()); };
    // The value plane @p plane had when leaf @p i read it: before the first leaf at or after it that ran and overwrites it, else live.
    const auto value_of = [&](int plane, std::size_t i) {
        const eg_plane_s& p = *d.planes[static_cast<std::size_t>(plane)];
        char* data = live(p);
        for (std::size_t m = i; m < n_leaves; ++m) {
            const std::vector<int>& w = d.nodes[static_cast<std::size_t>(d.plan.leaves[m].node)].writes;
            if (std::find(w.begin(), w.end(), plane) == w.end() || !leaf_ran(d, m)) continue;
            const RLeaf &lm = d.plan.leaves[m], &li = d.plan.leaves[i];
            if (lm.loop >= 0 && !(li.loop == lm.loop && li.iter / d.plan.loops[static_cast<std::size_t>(lm.loop)].w == lm.iter / d.plan.loops[static_cast<std::size_t>(lm.loop)].w)) {
                // the writer is in a window the reader is not in: the plane is as the checkpoint after the reader's window (or at loop entry) kept it
                const LoopTape& tape = d.loops[static_cast<std::size_t>(lm.loop)];
                const std::size_t group = li.loop == lm.loop ? static_cast<std::size_t>(li.iter / d.plan.loops[static_cast<std::size_t>(lm.loop)].w) + 1 : 0;
                data = const_cast<char*>(tape.ckpt[group][position(tape, plane)].data());
            } else {
                for (Snapshot& s : d.snapshots)
                    if (s.node == m && s.plane == plane) data = store(s).data();
            }
            break;
        }
        return data;
    };
    const auto run_fft = [&](eg_fft_s& plan, View& in, View& out) {
        if (D != nullptr) device_fft_run(*D, plan, in.get(), out.get());
        else host::fft_exec(plan, *in.get(), *out.get(), threads);
    };
    const auto run_gemm = [&](const eg_gemm_s& product, View& a, View& b, View& c) {
        if (D != nullptr) device_gemm_run(*D, product, a.get(), b.get(), c.get());
        else host::gemm_exec(product, *a.get(), *b.get(), *c.get());
    };
    const auto plane_of = [&](int idx) -> const eg_plane_s& { return *d.planes[static_cast<std::size_t>(idx)]; };
    // Run the forward of the copy, reduce, transform or product @p node again over the buffers @p data_at names for its planes.
    const auto rerun = [&](const DiffNode& node, const auto& data_at) {
        const int in = node.reads.front(), out = node.writes.front();
        switch (node.kind) {
            case Node::Copy:
                if (data_at(in) != data_at(out)) copy_bytes(D, data_at(out), data_at(in), byte_count(plane_of(out)));
                break;
            case Node::Reduce: {
                View x(plane_of(in), data_at(in)), y(plane_of(out), data_at(out));
                host::ReduceOp r = node.reduce;
                r.in = x.get();
                r.out = y.get();
                if (D != nullptr) device_reduce_leaf(*D, r);
                else host::reduce_exec(r, threads);
                break;
            }
            case Node::Fft: {
                View x(plane_of(in), data_at(in)), y(plane_of(out), data_at(out));
                run_fft(*node.fft, x, y);
                break;
            }
            default: {
                View a(plane_of(node.reads[0]), data_at(node.reads[0])), b(plane_of(node.reads[1]), data_at(node.reads[1])), c(plane_of(out), data_at(out));
                run_gemm(node.gemm, a, b, c);
            }
        }
    };
    // Rebuild window @p group of loop @p index: re-launch its body's plans over private copies, from the window's checkpoint, saving
    // each leaf's pre-write copies where the snapshot entries share their buffers.
    const auto rebuild = [&](int index, int group) {
        LoopTape& tape = d.loops[static_cast<std::size_t>(index)];
        const RLoop& loop = d.plan.loops[static_cast<std::size_t>(index)];
        if (tape.built == group) return;
        std::vector<Bytes> state;
        for (std::size_t q = 0; q < tape.planes.size(); ++q) {
            const std::size_t size = byte_count(*d.planes[static_cast<std::size_t>(tape.planes[q])]);
            state.emplace_back(size, D);
            copy_bytes(D, state.back().data(), tape.ckpt[static_cast<std::size_t>(group)][q].data(), size);
        }
        // Where leaf plane @p idx lives while this window is rebuilt: its private copy when the loop writes it, else its value before the loop.
        const auto data_at = [&](int idx) {
            const std::size_t at = position(tape, idx);
            return at < tape.planes.size() ? state[at].data() : value_of(idx, static_cast<std::size_t>(loop.first_leaf));
        };
        for (int k = group * loop.w; k < std::min(loop.cap, (group + 1) * loop.w); ++k)
            for (int local = 0; local < loop.per_iter; ++local) {
                const std::size_t m = static_cast<std::size_t>(loop.first_leaf + k * loop.per_iter + local);
                if (!leaf_ran(d, m)) continue;
                const DiffNode& node = d.nodes[static_cast<std::size_t>(d.plan.leaves[m].node)];
                for (Snapshot& s : d.snapshots)
                    if (s.node == m) copy_bytes(D, store(s).data(), state[position(tape, s.plane)].data(), s.size);
                if (node.writes.empty()) continue;    // it writes nothing the tape tracks: nothing reads what it makes
                if (node.kind != Node::Launch) {
                    rerun(node, data_at);
                    continue;
                }
                const eg_plan_s& primal = *node.plan;
                eg_plan_s* rp = inner_plan(ctx, primal.unit, primal.structure, primal.n_samples);
                try {
                    for (std::size_t sl = 0; sl < primal.bindings.size(); ++sl) {
                        const eg_plan_s::Binding& b = primal.bindings[sl];
                        if (primal.unit->slots[sl].role == EG_ROLE_UNIFORM) {
                            bind_value(*rp, static_cast<int>(sl), b.value, sizeof(b.value));
                        } else if (b.plane != nullptr) {
                            bind_raw(*rp, static_cast<int>(sl), data_at(find_plane(d, b.plane)), bound_count(*rp, static_cast<int>(sl), tensor(*b.plane)));
                        }
                    }
                    if (D != nullptr) device_launch_prefix(*rp, primal.n_samples, primal.n_samples);
                    else host::run_plan(*rp, threads);
                } catch (...) {
                    inner_free(ctx, rp);
                    throw;
                }
                inner_free(ctx, rp);
            }
        tape.built = group;
    };
    // The adjoints @p adjoint of the planes the run, copy, sum reduce, transform or product @p node (leaf @p i) wrote, pulled back to what it read.
    const auto transpose = [&](const DiffNode& node, std::size_t i, std::vector<Bytes>& adjoint, int taint) {
        Bytes& dy = adjoint.front();
        const auto give = [&](int plane, Bytes bytes) {
            Contribution c;
            c.node = static_cast<int>(i);
            c.taint = taint;
            c.bytes = std::move(bytes);
            pending[static_cast<std::size_t>(plane)].push_back(std::move(c));
        };
        const int in = node.reads.front(), out = node.writes.front();
        switch (node.kind) {
            case Node::Copy: give(in, std::move(dy)); break;
            case Node::Reduce: {                // every sample of the sum gets the adjoint of the sum
                const std::size_t es = element_bytes(plane_of(in));
                const std::int64_t n = node.reduce.n;
                Bytes spread(byte_count(plane_of(in)), D);
                if (D != nullptr) {
                    Bytes first(static_cast<std::size_t>(n) * sizeof(std::int64_t), D);     // the sample 0 of every column
                    device_columns(*D, 0, spread.data(), n, dy.data(), 1, reinterpret_cast<const std::int64_t*>(first.data()), n, 1, static_cast<unsigned>(es), 0);
                } else {
                    for (std::int64_t k = 0; k < n; ++k) std::memcpy(spread.data() + static_cast<std::size_t>(k) * es, dy.data(), es);
                }
                give(in, std::move(spread));
                break;
            }
            case Node::Fft: {
                Bytes back(byte_count(plane_of(in)), D);
                View y(plane_of(out), dy.data()), x(plane_of(in), back.data());
                run_fft(*node.adjoint, y, x);
                give(in, std::move(back));
                break;
            }
            case Node::Run: {                   // eg_reverse of the runner, seeded with the adjoints of the states it overwrote
                const std::size_t n_states = node.writes.size();
                std::vector<std::unique_ptr<View>> views;
                std::vector<Bytes> grads;
                std::vector<eg_plane_s*> seed_planes, grad_planes;
                for (std::size_t k = 0; k < n_states; ++k) {
                    views.push_back(std::make_unique<View>(plane_of(node.writes[k]), adjoint[k].data()));
                    seed_planes.push_back(views.back()->get());
                }
                for (int plane : node.reads) {
                    grads.emplace_back(byte_count(plane_of(plane)), D);
                    views.push_back(std::make_unique<View>(plane_of(plane), grads.back().data()));
                    grad_planes.push_back(views.back()->get());
                }
                eg_bind_desc seeds{}, outs{};
                seeds.struct_size = outs.struct_size = sizeof(eg_bind_desc);
                seeds.n_planes = static_cast<std::int32_t>(n_states);
                seeds.plane_slots = node.slots.data();
                seeds.planes = seed_planes.data();
                outs.n_planes = static_cast<std::int32_t>(node.reads.size());
                outs.plane_slots = node.slots.data();
                outs.planes = grad_planes.data();
                reverse(ctx, node.runner, &seeds, &outs, false);
                for (std::size_t k = 0; k < node.reads.size(); ++k) give(node.reads[k], std::move(grads[k]));
                break;
            }
            default: {
                const int a = node.reads[0], b = node.reads[1];
                Bytes grad_a(byte_count(plane_of(a)), D), grad_b(byte_count(plane_of(b)), D);
                View dc(plane_of(out), dy.data()), ga(plane_of(a), grad_a.data()), gb(plane_of(b), grad_b.data());
                View va(plane_of(a), value_of(a, i)), vb(plane_of(b), value_of(b, i));
                run_gemm(node.gemm_a, dc, vb, ga);
                run_gemm(node.gemm_b, va, dc, gb);
                if ((node.gemm.call.flags & EG_GEMM_ACCUMULATE) != 0) {     // C was summed into: its earlier value passes the adjoint through
                    Bytes through(byte_count(plane_of(out)), D);
                    copy_bytes(D, through.data(), dy.data(), byte_count(plane_of(out)));
                    give(out, std::move(through));
                }
                give(a, std::move(grad_a));
                give(b, std::move(grad_b));
            }
        }
    };
    for (LoopTape& tape : d.loops) tape.built = -1;
    for (std::size_t back = 0; back < n_leaves; ++back) {
        const std::size_t i = n_leaves - 1 - back;
        if (!leaf_ran(d, i)) continue;         // a leaf that did not run is the identity: what it would write keeps its adjoint
        if (d.plan.leaves[i].loop >= 0) rebuild(d.plan.leaves[i].loop, d.plan.leaves[i].iter / d.plan.loops[static_cast<std::size_t>(d.plan.leaves[i].loop)].w);
        const DiffNode& node = d.nodes[static_cast<std::size_t>(d.plan.leaves[i].node)];
        std::vector<Bytes> adjoint;
        bool flows = false;
        int incoming = -1;                      // a scatter-add behind any adjoint this leaf reverses
        for (int w : node.writes) {
            bool any = false;
            int origin = -1;
            adjoint.push_back(collect(w, any, origin));
            flows = flows || any;
            if (incoming < 0) incoming = origin;
        }
        if (node.kind != Node::Launch) {
            if (flows) transpose(node, i, adjoint, incoming);
            continue;
        }
        const eg_plan_s& primal = *node.plan;
        if (node.vjp == nullptr) {
            if (flows) refuse(EG_BAD_REQUEST, "eg_graph_reverse: a gradient flows through launch " + std::to_string(d.plan.leaves[i].node) + " (unit '" + primal.unit->name + "'), which has no vjp");
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
                    data = value_of(l.plane, i);
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
            c.taint = l.scatter ? d.plan.leaves[i].node : incoming;
            c.bytes = l.reduced ? sum_over_samples(D, grads[k], primal.n_samples, element_bytes(plane), threads) : std::move(grads[k]);
            pending[static_cast<std::size_t>(l.plane)].push_back(std::move(c));
        }
    }
    std::vector<Bytes> result;                  // every gradient is known before any is written, so a refusal writes none
    std::vector<int> taints;
    for (std::int32_t k = 0; k < r.n_grads; ++k) {
        bool any = false;
        int origin = -1;
        result.push_back(collect(wanted[static_cast<std::size_t>(k)], any, origin));
        taints.push_back(origin);
        if (origin >= 0 && (r.flags & EG_REVERSE_STRICT) != 0)
            refuse(EG_BAD_REQUEST, "eg_graph_reverse: EG_REVERSE_STRICT, but the gradient of inputs[" + std::to_string(k) + "] comes through a cross-sample scatter-add "
                   "in the vjp of launch " + std::to_string(origin) + ", whose summing order is not fixed; drop EG_REVERSE_STRICT to accept it (bitwise[" + std::to_string(k) + "] is 0)");
    }
    for (std::int32_t k = 0; k < r.n_grads; ++k) {
        copy_bytes(D, live(*r.grads[k]), result[static_cast<std::size_t>(k)].data(), byte_count(*r.grads[k]));
        if (r.bitwise != nullptr) r.bitwise[k] = taints[static_cast<std::size_t>(k)] < 0;
    }
    if (r.passes != nullptr) *r.passes = 1;
    if (D != nullptr) device_sync(ctx);     // the gradients land before eg_graph_reverse returns
}

}  // namespace door
}  // namespace eagle
