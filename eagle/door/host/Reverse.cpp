// SPDX-License-Identifier: Apache-2.0
// A recorded run on the host (EG_RUN_RECORD) and its reverse (eg_reverse).
//
// The recording forward runs the one-step taping twin over the runner's own
// planes with its tape stores off, checkpointing the state (every MUTABLE and
// TERMINATED slot of the twin, its applied-steps plane included) at the start
// of each window. The reverse runs in passes of samples under the memory
// budget; per pass, window by window from the last, it gathers the window's
// checkpoint in stop order (longest first), recomputes the window with the
// tape on over each step's live prefix, then runs the step vjp backward with
// the state adjoints alternating between two buffers by step parity. Gradients
// of parameters summed over samples are gathered back to sample order, summed
// as a canonical compensated pair per pass and combined in pass order.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Team.h"
#include "eagle/diff/Record.h"
#include "eagle/diff/Reverse.h"
#include "eagle/diff/Window.h"
#include "eagle/reduce/canonical.h"
#include <algorithm>
#include <string>
#include <unistd.h>

namespace eagle {
namespace door {

/** @brief Bytes eagle owns, 8-byte aligned. */
struct Buf {
    std::vector<std::uint64_t> words;
    void resize(std::size_t bytes) { words.assign((bytes + 7) / 8, 0); }
    char* data() { return reinterpret_cast<char*>(words.data()); }
    void zero() { std::fill(words.begin(), words.end(), 0); }
};

/** @brief One plane the twin reads or carries: a state (checkpointed) or a read (gathered per pass). */
struct Carried {
    enum Kind { PLANE, STEPS, FILL, START } kind = PLANE;
    int twin_slot = -1;
    int primal_slot = -1;           ///< PLANE: the primal's slot; FILL: the primal UNIFORM it fills; STEPS, START: -1
    int of = -1;                    ///< START: the primal's time plane
    Buf own;                        ///< FILL, START: the n-wide plane eagle makes
    std::size_t bytes = 0;          ///< element bytes
    std::int64_t rows = 1;          ///< rows per sample
    char* src = nullptr;            ///< the n-wide plane the forward runs on
    Buf buf;                        ///< its m-wide copy in a reverse pass
};

/** @brief A taped state: its tape rows and its two adjoint buffers (m wide). */
struct TapedState {
    int primal_slot = -1;
    int state = -1;                 ///< index in RunRecord::state
    std::int64_t row0 = 0, rows = 1;
    Buf adj[2];
};

/** @brief A parameter's gradient: the step's share (written by the vjp) and its accumulation, summed over samples at the end. */
struct Grad {
    int primal_slot = -1;
    std::int64_t rows = 1;
    int fold = -1;                  ///< a per-sample gradient of the time start copy: the taped state it is added into
    Buf grad, acc;
};

/** @brief How a vjp slot is bound in a launch. */
struct VjpSlot {
    enum Kind { UNIFORM, TAPED, ADJ_IN, ADJ_OUT, GRAD, READ, STEP } kind = UNIFORM;
    int index = -1;                 ///< TAPED / ADJ_*: taped; GRAD: grads; READ: reads; UNIFORM: primal slot
};

struct RunRecord {
    eg_plan_s* fwd = nullptr;       ///< the twin over the runner's planes, tape off
    eg_plan_s* rec = nullptr;       ///< the twin over a pass's planes, tape on (made per reverse)
    eg_plan_s* vjp = nullptr;       ///< the step vjp over a pass's planes (made per reverse)
    eg_unit_s* twin_unit = nullptr;
    eg_unit_s* vjp_unit = nullptr;
    std::int64_t window = 1, width = 0, budget = 0;
    std::size_t scalar = 8;
    std::vector<Carried> state, reads;
    std::vector<TapedState> taped;
    std::vector<Grad> grads;
    std::vector<VjpSlot> vjp_slots;
    std::size_t step_bytes = 0;     ///< the width of the vjp's step plane (EG_SRC_STEPS), 0: none
    Buf steps;                      ///< applied steps, int64 per sample
    Buf dummy;                      ///< the forward's tape: one row of width * n, never written
    std::vector<std::vector<Buf>> pool;   ///< checkpoints: pool[k][state]
    std::int64_t taken = 0;
    bool live = false;              ///< a record not reversed yet
};

namespace {

char* plane_bytes(const eg_plane_s& p) {
    const DLTensor& t = p.tensor->dl_tensor;
    return static_cast<char*>(t.data) + t.byte_offset;
}
std::size_t elem_bytes(const DLTensor& t) { return static_cast<std::size_t>(t.dtype.bits / 8) * t.dtype.lanes; }

/** @brief A plan of a derived unit over eagle's memory: one whole partition of @p n samples. */
eg_plan_s* inner_plan(eg_ctx& ctx, eg_unit_s* unit, eg_structure structure, std::int64_t n) {
    auto* p = new eg_plan_s;
    p->ctx = &ctx;
    p->unit = unit;
    p->structure = structure;
    p->n_samples = n;
    p->parts = { exec::Partition::whole(n) };
    p->bindings.resize(unit->slots.size());
    p->params.resize(unit->slots.size());
    p->caller_ref = false;
    if (unit->finish.present) bind_raw(*p, unit->finish.counter_slot, &p->finished, 1);
    for (std::size_t i = 0; i < unit->slots.size(); ++i)
        if (unit->slots[i].role == EG_ROLE_NSAMPLES) {
            const std::uint32_t n32 = static_cast<std::uint32_t>(n);
            std::memcpy(p->bindings[i].value, &n32, sizeof(n32));
            p->bindings[i].bound = true;
            p->params[i] = p->bindings[i].value;
        }
    ++unit->refs;
    return p;
}
void inner_free(eg_ctx& ctx, eg_plan_s*& p) {
    if (p == nullptr) return;
    unit_release(ctx, p->unit);
    delete p;
    p = nullptr;
}
void bind_value(eg_plan_s& p, int slot, const void* value, std::size_t bytes) {
    eg_plan_s::Binding& b = p.bindings[static_cast<std::size_t>(slot)];
    std::memcpy(b.value, value, bytes);
    b.bound = true;
    p.params[static_cast<std::size_t>(slot)] = b.value;
}
void bind_int64(eg_plan_s& p, int slot, std::int64_t v) { bind_value(p, slot, &v, sizeof(v)); }

/** @brief The plane eagle made for twin slot @p s (a filled uniform or the start copy), or null. */
const Carried* made_plane(const RunRecord& r, int s) {
    for (const Carried& c : r.reads)
        if (c.twin_slot == s && (c.kind == Carried::FILL || c.kind == Carried::START)) return &c;
    return nullptr;
}

std::string field(const char* unit, const std::string& name) { return std::string(unit) + ".derived." + name; }

/** @brief The derived unit of @p role in the runner description; refuses what is missing or not written. */
eg_unit_s* derived_of(const eg_ctx& ctx, const eg_runner_desc& d, eg_derived_role role) {
    eg_unit_s* found = nullptr;
    for (std::int32_t i = 0; i < d.n_derived; ++i) {
        const std::string at = "derived_units[" + std::to_string(i) + "]";
        check_handle(ctx.units, d.derived_units[i], at.c_str());
        const eg_unit_s& u = *d.derived_units[i];
        const eg_derived_role r = d.derived_roles[i];
        if (r == EG_DERIVED_TAPE_FUSED || r == EG_DERIVED_FUSED_VJP)
            refuse(EG_NOT_WRITTEN, "derived_roles[" + std::to_string(i) + "]: the K-step twin and vjp (steps_reverse "
                   "fused) are not written in this build of the door");
        if (r >= EG_DERIVED_REFINE_ENTRY && r <= EG_DERIVED_STATE_GUARD_VJP)
            refuse(EG_NOT_WRITTEN, "derived_roles[" + std::to_string(i) + "]: the events refine roles are not written "
                   "in this build of the door");
        if (r != EG_DERIVED_TAPE && r != EG_DERIVED_STEP_VJP)
            refuse(EG_BAD_REQUEST, "derived_roles[" + std::to_string(i) + "]: unknown role " + std::to_string(r));
        if (!u.derived.present) refuse(EG_BAD_REQUEST, at + " (unit '" + u.name + "') has no derived record");
        if (u.derived.role != r)
            refuse(EG_BAD_REQUEST, at + " (unit '" + u.name + "'): derived.role " + std::to_string(u.derived.role) +
                   " is not derived_roles[" + std::to_string(i) + "] " + std::to_string(r));
        if (r != role) continue;
        if (found != nullptr) refuse(EG_BAD_REQUEST, "derived_roles: role " + std::to_string(r) + " given twice");
        found = d.derived_units[i];
    }
    if (found == nullptr)
        refuse(EG_BAD_REQUEST, std::string("EG_RUN_RECORD needs the ") +
               (role == EG_DERIVED_TAPE ? "one-step taping twin (EG_DERIVED_TAPE)" : "step vjp (EG_DERIVED_STEP_VJP)"));
    return found;
}

void check_slot(const eg_unit_s& u, std::int32_t slot, const std::string& name, std::int32_t role, std::int32_t dtype) {
    if (slot < 0 || slot >= static_cast<std::int32_t>(u.slots.size()))
        refuse(EG_BAD_REQUEST, field(u.name.c_str(), name) + " " + std::to_string(slot) + " is out of range");
    const SlotFacts& s = u.slots[static_cast<std::size_t>(slot)];
    if (s.role != role || (dtype >= 0 && s.dtype != dtype))
        refuse(EG_BAD_REQUEST, field(u.name.c_str(), name) + " " + std::to_string(slot) + " ('" + s.name +
               "') has the wrong role or dtype");
}

bool in_range(std::int32_t slot, std::size_t n) { return slot >= 0 && slot < static_cast<std::int32_t>(n); }

}  // namespace

/** @brief eg_runner_new with EG_RUN_RECORD: check the derived record against the primal and keep the twin and the vjp. */
void record_new(eg_ctx& ctx, eg_runner_s& runner, const eg_runner_desc& d) {
    if (runner.active && runner.act.reorder)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD with EG_RUN_REORDER (an active set that reorders the planes under record) is "
               "not written in this build of the door");
    if (runner.active && runner.act.group != 1)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD with a grouped active set is not written in this build of the door");
    if (d.n_plans != 1)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD of a several-plan step is not written in this build of the door");
    const eg_plan_s& primal = *d.plans[0];
    const eg_unit_s& pu = *primal.unit;
    eg_unit_s* twin = derived_of(ctx, d, EG_DERIVED_TAPE);
    eg_unit_s* vjp = derived_of(ctx, d, EG_DERIVED_STEP_VJP);
    if (d.n_derived != 2) refuse(EG_BAD_REQUEST, "n_derived " + std::to_string(d.n_derived) + ": a recorded run takes the twin and the vjp");
    const std::size_t np = pu.slots.size();
    if (runner.active != (twin->active_map_slot >= 0) || (twin->active_map_slot >= 0) != (twin->active_count_slot >= 0))
        refuse(EG_BAD_REQUEST, "unit '" + twin->name + "': the taping twin reads an active set (map and count slots) exactly "
               "when the runner's plans do");
    if (vjp->active_map_slot >= 0) refuse(EG_BAD_REQUEST, "unit '" + vjp->name + "': a step vjp reads no active set");
    for (const eg_unit_s* u : { twin, vjp })
        if (u->derived.of_unit != pu.index || u->library != pu.library)
            refuse(EG_BAD_REQUEST, field(u->name.c_str(), "of_unit") + " " + std::to_string(u->derived.of_unit) +
                   " is not the runner's primal (unit '" + pu.name + "', index " + std::to_string(pu.index) + ")");
    const DerivedFacts& t = twin->derived;
    const char* tn = twin->name.c_str();
    if (t.time0_slot != -1) check_slot(*twin, t.time0_slot, "time0_slot", EG_ROLE_PER_SAMPLE, -1);
    if (t.fused_steps_slot != -1 || t.tau_slot != -1 || t.n_refined != 0)
        refuse(EG_BAD_REQUEST, field(tn, "fused_steps_slot, tau_slot, refined") + ": not part of a one-step taping twin");
    if (twin->finish.present && twin->finish.steps_slot >= 0)
        refuse(EG_BAD_REQUEST, "unit '" + twin->name + "': a one-step taping twin takes no steps word");
    check_slot(*twin, t.tape_slot, "tape_slot", EG_ROLE_OUT, -1);
    check_slot(*twin, t.steps_slot, "steps_slot", EG_ROLE_MUTABLE, EG_DTYPE_INT64);
    check_slot(*twin, t.tape_on_slot, "tape_on_slot", EG_ROLE_UNIFORM, EG_DTYPE_INT64);
    const SlotFacts& tape = twin->slots[static_cast<std::size_t>(t.tape_slot)];
    if (t.tape_rows < 1) refuse(EG_BAD_REQUEST, field(tn, "tape_rows") + " " + std::to_string(t.tape_rows) + " < 1");
    if (tape.rank != 1 || tape.shape[0] < 1 || tape.shape[0] % t.tape_rows != 0)
        refuse(EG_BAD_REQUEST, field(tn, "tape_rows") + " " + std::to_string(t.tape_rows) + ": the tape slot's rows are "
               "not window * tape_rows");
    if (t.taped.empty()) refuse(EG_BAD_REQUEST, field(tn, "n_taped") + " is 0");
    std::int64_t rows = 0;
    for (std::size_t i = 0; i < t.taped.size(); ++i) {
        const DerivedFacts::Taped& x = t.taped[i];
        const std::string at = "taped[" + std::to_string(i) + "]";
        if (!in_range(x.slot, np) || pu.slots[static_cast<std::size_t>(x.slot)].role != EG_ROLE_MUTABLE)
            refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " " + std::to_string(x.slot) + " is not a MUTABLE slot of the primal");
        if (x.rows < 1 || x.row0 < 0 || x.row0 + x.rows > t.tape_rows)
            refuse(EG_BAD_REQUEST, field(tn, at + ".row0, rows") + " fall outside tape_rows");
        rows += x.rows;
    }
    if (rows != t.tape_rows)
        refuse(EG_BAD_REQUEST, field(tn, "tape_rows") + " " + std::to_string(t.tape_rows) + " is not the sum of the taped rows " +
               std::to_string(rows));
    const std::int64_t window = tape.shape[0] / t.tape_rows;
    const std::int64_t asked = d.window == -1 ? std::max<std::int64_t>(d.max_steps, 1) : d.window;
    if (asked != window)
        refuse(EG_BAD_REQUEST, "window " + std::to_string(d.window) + ": the taping twin '" + twin->name +
               "' records windows of " + std::to_string(window) + " steps (built in)");
    // The twin: one EG_SRC_PRIMAL link per ordinary slot, of the primal slot's role and dtype; none on a reserved slot.
    std::vector<int> twin_of(twin->slots.size(), -1);
    for (std::size_t i = 0; i < t.links.size(); ++i) {
        const DerivedFacts::Link& l = t.links[i];
        const std::string at = "links[" + std::to_string(i) + "]";
        if (!in_range(l.slot, twin->slots.size())) refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " is out of range");
        if (l.source == EG_SRC_TIME0 && l.slot != t.time0_slot)
            refuse(EG_BAD_REQUEST, field(tn, at) + ": EG_SRC_TIME0 is the link of time0_slot only (EG_SRC_TIME0 on slot " +
                   std::to_string(l.slot) + ")");
        if (l.source != EG_SRC_PRIMAL && l.source != EG_SRC_TIME0) refuse(EG_BAD_REQUEST, field(tn, at + ".source") + ": a taping twin links EG_SRC_PRIMAL");
        if (l.reduced != 0) refuse(EG_BAD_REQUEST, field(tn, at + ".reduced") + ": only a PARAM_GRAD link is reduced");
        if (l.slot == t.tape_slot || l.slot == t.steps_slot || l.slot == t.tape_on_slot || l.slot == twin->active_map_slot ||
            l.slot == twin->active_count_slot)
            refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " " + std::to_string(l.slot) + " is a reserved slot");
        if (twin_of[static_cast<std::size_t>(l.slot)] != -1)
            refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " " + std::to_string(l.slot) + " has a second link");
        if (!in_range(l.of_slot, np)) refuse(EG_BAD_REQUEST, field(tn, at + ".of_slot") + " is out of range");
        const SlotFacts& a = twin->slots[static_cast<std::size_t>(l.slot)];
        const SlotFacts& b = pu.slots[static_cast<std::size_t>(l.of_slot)];
        if (l.source == EG_SRC_TIME0) {   // the start copy of a time plane: a per-sample plane of the time plane's dtype
            if (b.role != EG_ROLE_MUTABLE || a.dtype != b.dtype || a.dtype != EG_DTYPE_REAL)
                refuse(EG_BAD_REQUEST, field(tn, at) + ": the time plane (primal slot '" + b.name + "') is not a MUTABLE real "
                       "plane of the start copy's dtype");
            twin_of[static_cast<std::size_t>(l.slot)] = l.of_slot;
            continue;
        }
        const bool filled = a.role == EG_ROLE_PER_SAMPLE && b.role == EG_ROLE_UNIFORM && a.dtype == b.dtype && a.dtype == EG_DTYPE_REAL;
        if (!filled && (a.role != b.role || a.dtype != b.dtype || a.rank != b.rank || a.mirror != b.mirror))
            refuse(EG_BAD_REQUEST, field(tn, at) + ": twin slot '" + a.name + "' and primal slot '" + b.name +
                   "' differ in role, dtype or shape");
        twin_of[static_cast<std::size_t>(l.slot)] = l.of_slot;
    }
    for (std::size_t i = 0; i < twin->slots.size(); ++i) {
        const int s = static_cast<int>(i);
        const std::int32_t role = twin->slots[i].role;
        if (s == t.tape_slot || s == t.steps_slot || s == t.tape_on_slot || role == EG_ROLE_NSAMPLES) continue;
        if (s == t.time0_slot) {
            if (twin_of[i] == -1) refuse(EG_BAD_REQUEST, field(tn, "links") + ": time0_slot " + std::to_string(s) + " has no EG_SRC_TIME0 link");
            continue;
        }
        if (s == twin->active_map_slot || s == twin->active_count_slot) continue;   // eagle's own: the runner's map
        if (twin->finish.present && s == twin->finish.counter_slot) continue;
        if (twin_of[i] == -1)
            refuse(EG_BAD_REQUEST, field(tn, "links") + ": twin slot " + std::to_string(i) + " ('" + twin->slots[i].name + "') has no link");
        if (role == EG_ROLE_OUT || role == EG_ROLE_ACCUM_OUT || role == EG_ROLE_LOOKUP)
            refuse(EG_NOT_WRITTEN, "unit '" + twin->name + "': twin slot '" + twin->slots[i].name +
                   "' (an output or lookup beyond the finish counter) under record is not written in this build of the door");
    }
    auto* r = new RunRecord;
    runner.record = r;
    r->twin_unit = twin;
    r->vjp_unit = vjp;
    ++twin->refs;
    ++vjp->refs;
    r->window = window;
    r->width = t.tape_rows;
    r->budget = static_cast<std::int64_t>(d.memory_budget);
    r->scalar = pu.scalar_type == EG_SCALAR_FLOAT32 ? 4 : 8;
    std::vector<int> taped_state_of_primal(np, -1);
    for (std::size_t i = 0; i < twin->slots.size(); ++i) {
        const int s = static_cast<int>(i);
        const std::int32_t role = twin->slots[i].role;
        if (s == t.tape_slot || s == t.tape_on_slot || role == EG_ROLE_NSAMPLES || role == EG_ROLE_UNIFORM) continue;
        if (s == twin->active_map_slot || s == twin->active_count_slot) continue;
        if (twin->finish.present && s == twin->finish.counter_slot) continue;
        Carried c;
        c.twin_slot = s;
        c.primal_slot = s == t.steps_slot ? -1 : twin_of[i];
        if (s == t.steps_slot) c.kind = Carried::STEPS;
        else if (s == t.time0_slot) {
            c.kind = Carried::START;
            c.of = twin_of[i];
            c.primal_slot = -1;
        } else if (role == EG_ROLE_PER_SAMPLE && pu.slots[static_cast<std::size_t>(twin_of[i])].role == EG_ROLE_UNIFORM)
            c.kind = Carried::FILL;
        if (role == EG_ROLE_MUTABLE || role == EG_ROLE_TERMINATED) r->state.push_back(std::move(c));
        else r->reads.push_back(std::move(c));
    }
    for (const DerivedFacts::Taped& x : t.taped) {
        TapedState ts;
        ts.primal_slot = x.slot;
        ts.row0 = x.row0;
        ts.rows = x.rows;
        for (std::size_t k = 0; k < r->state.size(); ++k)
            if (r->state[k].primal_slot == x.slot) ts.state = static_cast<int>(k);
        if (ts.state < 0)
            refuse(EG_BAD_REQUEST, field(tn, "taped") + ": primal slot " + std::to_string(x.slot) + " is not a state the twin carries");
        if (taped_state_of_primal[static_cast<std::size_t>(x.slot)] != -1)
            refuse(EG_BAD_REQUEST, field(tn, "taped") + ": primal slot " + std::to_string(x.slot) + " is taped twice");
        taped_state_of_primal[static_cast<std::size_t>(x.slot)] = static_cast<int>(r->taped.size());
        r->taped.push_back(std::move(ts));
    }
    // The vjp: exactly one link per slot.
    const DerivedFacts& v = vjp->derived;
    const char* vn = vjp->name.c_str();
    r->vjp_slots.assign(vjp->slots.size(), VjpSlot{});
    std::vector<int> linked(vjp->slots.size(), 0);
    for (std::size_t i = 0; i < v.links.size(); ++i) {
        const DerivedFacts::Link& l = v.links[i];
        const std::string at = "links[" + std::to_string(i) + "]";
        if (!in_range(l.slot, vjp->slots.size())) refuse(EG_BAD_REQUEST, field(vn, at + ".slot") + " is out of range");
        if (linked[static_cast<std::size_t>(l.slot)]++ != 0)
            refuse(EG_BAD_REQUEST, field(vn, at + ".slot") + " " + std::to_string(l.slot) + " has a second link");
        if (l.source == EG_SRC_EVENT)
            refuse(EG_NOT_WRITTEN, field(vn, at + ".source") + " " + std::to_string(l.source) +
                   " (an event multiplier) is not written in this build of the door");
        if (l.source == EG_SRC_STEPS) {   // the step index j, a plane eagle fills before each step's vjp
            const SlotFacts& vs_ = vjp->slots[static_cast<std::size_t>(l.slot)];
            if (l.of_slot != t.steps_slot && l.of_slot != -1)
                refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) +
                       " is neither the twin's steps_slot nor -1");
            if (l.reduced != 0) refuse(EG_BAD_REQUEST, field(vn, at + ".reduced") + ": only a PARAM_GRAD link is reduced");
            if (vs_.role != EG_ROLE_PER_SAMPLE || (vs_.dtype != EG_DTYPE_INT64 && vs_.dtype != EG_DTYPE_INT32))
                refuse(EG_BAD_REQUEST, field(vn, at) + ": a step plane is a per-sample int32 or int64 slot ('" + vs_.name + "')");
            if (r->step_bytes != 0) refuse(EG_BAD_REQUEST, field(vn, at) + ": a second step plane");
            r->step_bytes = vs_.dtype == EG_DTYPE_INT64 ? 8 : 4;
            r->vjp_slots[static_cast<std::size_t>(l.slot)] = { VjpSlot::STEP, -1 };
            continue;
        }
        if (l.source == EG_SRC_TIME0) {   // the gradient of the time plane's start copy, added into the time plane's
            const SlotFacts& vs_ = vjp->slots[static_cast<std::size_t>(l.slot)];
            if (t.time0_slot == -1)
                refuse(EG_BAD_REQUEST, field(vn, at) + ": EG_SRC_TIME0 but the taping twin holds no start copy (time0_slot -1)");
            if (!in_range(l.of_slot, np) || taped_state_of_primal[static_cast<std::size_t>(l.of_slot)] < 0)
                refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is not a taped state (the time plane)");
            if (l.reduced != 0) refuse(EG_BAD_REQUEST, field(vn, at + ".reduced") + ": only a PARAM_GRAD link is reduced");
            if (vs_.role != EG_ROLE_MUTABLE) refuse(EG_BAD_REQUEST, field(vn, at) + ": the start copy's gradient is a MUTABLE slot ('" + vs_.name + "')");
            for (const Grad& g : r->grads)
                if (g.fold >= 0) refuse(EG_BAD_REQUEST, field(vn, at) + ": a second start-copy gradient");
            Grad gr;
            gr.fold = taped_state_of_primal[static_cast<std::size_t>(l.of_slot)];
            r->vjp_slots[static_cast<std::size_t>(l.slot)] = { VjpSlot::GRAD, static_cast<int>(r->grads.size()) };
            r->grads.push_back(std::move(gr));
            continue;
        }
        if (l.source < EG_SRC_PRIMAL || l.source > EG_SRC_TAPED)
            refuse(EG_BAD_REQUEST, field(vn, at + ".source") + " " + std::to_string(l.source) + " is unknown");
        if (!in_range(l.of_slot, np)) refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is out of range");
        if (l.reduced != 0 && l.source != EG_SRC_PARAM_GRAD)
            refuse(EG_BAD_REQUEST, field(vn, at + ".reduced") + ": only a PARAM_GRAD link is reduced");
        const std::int32_t prole = pu.slots[static_cast<std::size_t>(l.of_slot)].role;
        const int tsi = taped_state_of_primal[static_cast<std::size_t>(l.of_slot)];
        int read = -1;
        for (std::size_t k = 0; k < r->reads.size(); ++k)
            if (r->reads[k].primal_slot == l.of_slot) read = static_cast<int>(k);
        VjpSlot& vs = r->vjp_slots[static_cast<std::size_t>(l.slot)];
        switch (l.source) {
        case EG_SRC_PRIMAL:
            if (prole == EG_ROLE_UNIFORM && vjp->slots[static_cast<std::size_t>(l.slot)].role == EG_ROLE_PER_SAMPLE) {
                if (read < 0 || r->reads[static_cast<std::size_t>(read)].kind != Carried::FILL)
                    refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) +
                           ": a plane filled with a parameter is one the taping twin reads too");
                vs = { VjpSlot::READ, read };
            } else if (prole == EG_ROLE_UNIFORM) vs = { VjpSlot::UNIFORM, l.of_slot };
            else if (read >= 0)
                refuse(EG_NOT_WRITTEN, field(vn, at) + ": a vjp read of a per-sample plane has no old-path oracle in this build "
                       "of the door");
            else refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) +
                        ": the recorded run holds no such read (a state is read from the tape)");
            break;
        case EG_SRC_TAPED:
        case EG_SRC_STATE_ADJOINT_IN:
        case EG_SRC_STATE_ADJOINT_OUT:
            if (tsi < 0) refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is not a taped state");
            vs = { l.source == EG_SRC_TAPED ? VjpSlot::TAPED : l.source == EG_SRC_STATE_ADJOINT_IN ? VjpSlot::ADJ_IN : VjpSlot::ADJ_OUT, tsi };
            break;
        default: {   // EG_SRC_PARAM_GRAD
            const bool uniform = prole == EG_ROLE_UNIFORM;
            if (!uniform && read < 0)
                refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is neither a parameter nor a read");
            if (uniform == (l.reduced == 0))
                refuse(EG_BAD_REQUEST, field(vn, at + ".reduced") + " " + std::to_string(l.reduced) +
                       (uniform ? ": a parameter's gradient is summed over samples (1)" : ": a per-sample read's gradient is not summed (0)"));
            if (!uniform)
                refuse(EG_NOT_WRITTEN, field(vn, at) + ": a per-sample read's gradient has no old-path oracle in this build "
                       "of the door");
            int g = -1;
            for (std::size_t k = 0; k < r->grads.size(); ++k)
                if (r->grads[k].primal_slot == l.of_slot) g = static_cast<int>(k);
            if (g >= 0) refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " has a second gradient");
            Grad gr;
            gr.primal_slot = l.of_slot;
            vs = { VjpSlot::GRAD, static_cast<int>(r->grads.size()) };
            r->grads.push_back(std::move(gr));
        }
        }
    }
    for (std::size_t i = 0; i < vjp->slots.size(); ++i)
        if (linked[i] == 0 && vjp->slots[i].role != EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, field(vn, "links") + ": vjp slot " + std::to_string(i) + " ('" + vjp->slots[i].name + "') has no link");
    if (vjp->finish.present) refuse(EG_BAD_REQUEST, "unit '" + vjp->name + "': a step vjp has no finish");
}

/** @brief Drop the runner's derived plans and units. */
void record_free(eg_ctx& ctx, eg_runner_s& runner) {
    RunRecord* r = runner.record;
    if (r == nullptr) return;
    inner_free(ctx, r->fwd);
    inner_free(ctx, r->rec);
    inner_free(ctx, r->vjp);
    if (r->twin_unit != nullptr) unit_release(ctx, r->twin_unit);
    if (r->vjp_unit != nullptr) unit_release(ctx, r->vjp_unit);
    delete r;
    runner.record = nullptr;
}

/** @brief eg_run of an EG_RUN_RECORD runner (``RunRecorder.run`` + ``diff::recordForward``): the twin, tape off, one
 *  step per launch, a checkpoint before the first step of each window, until done or max_steps. */
void record_forward(eg_ctx& ctx, eg_runner_s& runner, eg_run_report* report) {
    RunRecord& r = *runner.record;
    if (r.live)
        refuse(EG_BAD_REQUEST, "eg_run: the previous recorded run is not reversed yet; eg_reverse it before recording again");
    eg_plan_s& primal = *runner.plans[0];
    const std::int64_t n = primal.n_samples;
    const DerivedFacts& t = r.twin_unit->derived;
    if (r.fwd == nullptr) r.fwd = inner_plan(ctx, r.twin_unit, primal.structure, n);
    eg_plan_s& fwd = *r.fwd;
    r.steps.resize(static_cast<std::size_t>(n) * 8);
    r.dummy.resize(static_cast<std::size_t>(r.width * std::max<std::int64_t>(n, 1)) * r.scalar);
    const auto take = [&](Carried& c) {
        if (c.kind == Carried::STEPS) {
            c.src = r.steps.data();
            c.bytes = 8;
            c.rows = 1;
            return;
        }
        if (c.kind == Carried::FILL) {   // a parameter's value as a plane: eagle makes and fills it
            c.bytes = r.scalar;
            c.rows = 1;
            c.own.resize(static_cast<std::size_t>(n) * c.bytes);
            for (std::int64_t i = 0; i < n; ++i)
                std::memcpy(c.own.data() + static_cast<std::size_t>(i) * c.bytes, primal.bindings[static_cast<std::size_t>(c.primal_slot)].value, c.bytes);
            c.src = c.own.data();
            return;
        }
        if (c.kind == Carried::START) {   // the time plane as it is at run start
            const eg_plane_s& p = *primal.bindings[static_cast<std::size_t>(c.of)].plane;
            c.bytes = elem_bytes(p.tensor->dl_tensor);
            c.rows = 1;
            c.own.resize(static_cast<std::size_t>(n) * c.bytes);
            std::memcpy(c.own.data(), plane_bytes(p), static_cast<std::size_t>(n) * c.bytes);
            c.src = c.own.data();
            return;
        }
        const eg_plane_s& p = *primal.bindings[static_cast<std::size_t>(c.primal_slot)].plane;
        c.src = plane_bytes(p);
        c.bytes = elem_bytes(p.tensor->dl_tensor);
        c.rows = n > 0 ? elements(p.tensor->dl_tensor) / n : 1;
    };
    for (Carried& c : r.state) take(c);
    for (Carried& c : r.reads) take(c);
    for (const TapedState& x : r.taped)
        if (r.state[static_cast<std::size_t>(x.state)].rows != x.rows)
            refuse(EG_BAD_REQUEST, field(r.twin_unit->name.c_str(), "taped") + ": primal slot " + std::to_string(x.primal_slot) +
                   " is bound with " + std::to_string(r.state[static_cast<std::size_t>(x.state)].rows) + " rows, taped " +
                   std::to_string(x.rows));
    for (std::size_t i = 0; i < r.twin_unit->slots.size(); ++i) {   // the runner's planes, as bound now
        const int s = static_cast<int>(i);
        if (s == t.steps_slot) bind_raw(fwd, s, r.steps.data(), static_cast<std::uint64_t>(n));
        else if (s == t.tape_slot) bind_raw(fwd, s, r.dummy.data(), static_cast<std::uint64_t>(n));
        else if (s == t.tape_on_slot) bind_int64(fwd, s, 0);
        else if (s == r.twin_unit->active_map_slot) bind_raw(fwd, s, runner.act.map.data(), static_cast<std::uint64_t>(n));
        else if (s == r.twin_unit->active_count_slot) bind_raw(fwd, s, &runner.act.count, 1);
        else if (r.twin_unit->slots[i].role == EG_ROLE_NSAMPLES || (r.twin_unit->finish.present && s == r.twin_unit->finish.counter_slot)) continue;
        else if (const Carried* made = made_plane(r, s)) bind_raw(fwd, s, made->src, static_cast<std::uint64_t>(n));
        else {
            int of = -1;
            for (const DerivedFacts::Link& l : t.links)
                if (l.slot == s) of = l.of_slot;
            const eg_plan_s::Binding& b = primal.bindings[static_cast<std::size_t>(of)];
            if (r.twin_unit->slots[i].role == EG_ROLE_UNIFORM) bind_value(fwd, s, b.value, sizeof(b.value));
            else bind_raw(fwd, s, plane_bytes(*b.plane), static_cast<std::uint64_t>(n));
        }
    }
    // The finished count follows the mask (the plain loop's re-sync), counted by the twin from here.
    const FinishFacts& f = primal.unit->finish;
    const DLTensor& mt = primal.bindings[static_cast<std::size_t>(f.mask_slot)].plane->tensor->dl_tensor;
    const char* mask = static_cast<const char*>(mt.data) + mt.byte_offset;
    const std::size_t mb = elem_bytes(mt);
    std::uint32_t set = 0;
    for (std::int64_t i = 0; i < n; ++i)
        for (std::size_t b = 0; b < mb; ++b)
            if (mask[static_cast<std::size_t>(i) * mb + b] != 0) {
                ++set;
                break;
            }
    fwd.finished = set;
    ActiveState& act = runner.act;
    if (runner.active) {   // the map follows the mask from the first launch (the plain loop's compaction)
        active_compact(act, mask, mb);
        act.done_at_last = set;
    }
    r.steps.zero();
    r.taken = 0;
    const int threads = host::team_threads(ctx);
    const std::int64_t ran = diff::recordForward(
        runner.max_steps, r.window, 1, [&] { return static_cast<std::int64_t>(fwd.finished) >= n; },
        [&](std::int64_t k) {
            if (k == static_cast<std::int64_t>(r.pool.size())) {
                r.pool.emplace_back(r.state.size());
                for (std::size_t i = 0; i < r.state.size(); ++i)
                    r.pool.back()[i].resize(static_cast<std::size_t>(r.state[i].rows * n) * r.state[i].bytes);
            }
            for (std::size_t i = 0; i < r.state.size(); ++i)
                std::memcpy(r.pool[static_cast<std::size_t>(k)][i].data(), r.state[i].src,
                            static_cast<std::size_t>(r.state[i].rows * n) * r.state[i].bytes);
            r.taken = k + 1;
        },
        [&](std::int64_t j, std::int64_t k) {
            const std::int64_t live = runner.active ? static_cast<std::int64_t>(act.count) : n;
            if (live > 0) host::run_prefix(fwd, threads, live);
            if (runner.active && (j + k) % act.every_steps == 0) {
                active_compact(act, mask, mb);
                ++act.compactions;
                act.done_at_last = fwd.finished;
            }
        });
    primal.finished = fwd.finished;
    r.live = true;
    if (report == nullptr) return;
    eg_run_report out{};
    out.flags = (static_cast<std::int64_t>(fwd.finished) == n ? EG_REPORT_DONE : 0u) | EG_REPORT_EXACT_STEPS;
    out.mode = EG_MODE_HOST;
    out.n = n;
    out.finished = fwd.finished;
    out.launches = ran;
    out.steps = ran;
    out.mapped = ran * n;
    write_report(report, out);
}

namespace {

/** @brief One bound caller plane of a reverse, checked against what it binds. */
char* caller_plane(const eg_ctx& ctx, eg_plane_s* p, const DLTensor& like, std::int64_t elements_, const std::string& what) {
    check_handle(ctx.planes, p, what.c_str());
    const DLTensor& t = p->tensor->dl_tensor;
    if (t.dtype.code != like.dtype.code || t.dtype.bits != like.dtype.bits || t.dtype.lanes != like.dtype.lanes)
        refuse(EG_BAD_REQUEST, what + ": a dtype mismatch with the slot it binds");
    if (elements(t) != elements_ || !compact(t))
        refuse(EG_BAD_REQUEST, what + ": holds " + std::to_string(elements(t)) + " elements, compact " +
               std::to_string(elements_) + " expected");
    if (t.device.device_type != kDLCPU) refuse(EG_BAD_REQUEST, what + ": a host reverse takes host planes");
    return static_cast<char*>(t.data) + t.byte_offset;
}

/** @brief dst[r*n + cols[s]] = src[r*m + s] for s < count. */
void put_columns(char* dst, std::int64_t n, const char* src, std::int64_t m, const std::int64_t* cols, std::int64_t count,
                 std::int64_t rows, std::size_t bytes) {
    for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t s = 0; s < count; ++s)
            std::memcpy(dst + static_cast<std::size_t>(r * n + cols[s]) * bytes, src + static_cast<std::size_t>(r * m + s) * bytes, bytes);
}

template <class T>
void accumulate(Grad& g, std::int64_t m, std::int64_t count) {
    for (std::int64_t r = 0; r < g.rows; ++r)
        diff::accumulatePrefix(reinterpret_cast<T*>(g.acc.data()) + r * m, reinterpret_cast<const T*>(g.grad.data()) + r * m, count);
}

template <class T>
struct SampleColumn {
    const T* p;
    T operator()(std::int64_t i) const { return p[i]; }
};

/** @brief The canonical compensated pair of @p x[0, count) (``reduce_canonical_host`` with pair). */
template <class T>
reduce::canonical::Partial<T, aether::SumOp<T>> pair_sum(const T* x, std::int64_t count) {
    return reduce::canonical::partialRange<T, aether::SumOp<T>>(SampleColumn<T>{ x }, 0, count);
}

}  // namespace

/** @brief eg_reverse (``RunReverse.__call__`` / ``_run`` and ``diff_reverse_host``). */
void reverse(eg_ctx& ctx, eg_runner_s* runner, const eg_bind_desc* seeds_in, const eg_bind_desc* grads_in) {
    check_handle(ctx.runners, runner, "runner");
    if (runner->record == nullptr) refuse(EG_BAD_REQUEST, "eg_reverse: the runner is not recorded (no EG_RUN_RECORD)");
    if (ctx.rec != nullptr) refuse(EG_NOT_WRITTEN, "eg_reverse inside a recording is not written in this build of the door");
    RunRecord& r = *runner->record;
    if (!r.live) refuse(EG_BAD_REQUEST, "eg_reverse: no recorded run to reverse (eg_run first; a record is reversed once)");
    const eg_bind_desc seeds = read_input(seeds_in, "seeds");
    const eg_bind_desc grads = read_input(grads_in, "grads");
    eg_plan_s& primal = *runner->plans[0];
    const eg_unit_s& pu = *primal.unit;
    const std::int64_t n = primal.n_samples, W = r.window;
    const std::size_t np = pu.slots.size();
    const auto slot_ok = [&](std::int32_t s, const char* what, int i) {
        if (!in_range(s, np))
            refuse(EG_BAD_REQUEST, std::string(what) + ".plane_slots[" + std::to_string(i) + "] " + std::to_string(s) + " is an unknown slot");
    };
    if (seeds.n_uniforms != 0 || grads.n_uniforms != 0)
        refuse(EG_BAD_REQUEST, "eg_reverse: seeds and grads bind planes, not uniforms");
    if (seeds.n_planes < 0 || (seeds.n_planes > 0 && (seeds.plane_slots == nullptr || seeds.planes == nullptr)) ||
        grads.n_planes < 0 || (grads.n_planes > 0 && (grads.plane_slots == nullptr || grads.planes == nullptr)))
        refuse(EG_BAD_REQUEST, "eg_reverse: n_planes without plane_slots and planes");
    const auto taped_of = [&](std::int32_t s) {
        for (std::size_t k = 0; k < r.taped.size(); ++k)
            if (r.taped[k].primal_slot == s) return static_cast<int>(k);
        return -1;
    };
    std::vector<const char*> seed(r.taped.size(), nullptr);
    for (int i = 0; i < seeds.n_planes; ++i) {
        const std::int32_t s = seeds.plane_slots[i];
        slot_ok(s, "seeds", i);
        const int k = taped_of(s);
        if (k < 0)
            refuse(EG_BAD_REQUEST, "seeds.plane_slots[" + std::to_string(i) + "] " + std::to_string(s) + " ('" + pu.slots[static_cast<std::size_t>(s)].name +
                   "') is not a taped state of the recorded run: a slot with no link");
        const Carried& st = r.state[static_cast<std::size_t>(r.taped[static_cast<std::size_t>(k)].state)];
        seed[static_cast<std::size_t>(k)] = caller_plane(ctx, seeds.planes[i], primal.bindings[static_cast<std::size_t>(s)].plane->tensor->dl_tensor,
                                                         st.rows * n, "seeds.planes[" + std::to_string(i) + "]");
    }
    struct Out { char* data; int taped, grad; };
    std::vector<Out> outs;
    for (int i = 0; i < grads.n_planes; ++i) {
        const std::int32_t s = grads.plane_slots[i];
        slot_ok(s, "grads", i);
        const std::string what = "grads.planes[" + std::to_string(i) + "]";
        const int k = taped_of(s);
        int g = -1;
        for (std::size_t j = 0; j < r.grads.size(); ++j)
            if (r.grads[j].primal_slot == s) g = static_cast<int>(j);
        if (k < 0 && g < 0)
            refuse(EG_BAD_REQUEST, "grads.plane_slots[" + std::to_string(i) + "] " + std::to_string(s) + " ('" + pu.slots[static_cast<std::size_t>(s)].name +
                   "'): a slot with no link");
        if (k >= 0) {
            const Carried& st = r.state[static_cast<std::size_t>(r.taped[static_cast<std::size_t>(k)].state)];
            outs.push_back({ caller_plane(ctx, grads.planes[i], primal.bindings[static_cast<std::size_t>(s)].plane->tensor->dl_tensor, st.rows * n, what), k, -1 });
        } else {   // a parameter: the sum over samples, one value
            DLTensor like{};
            like.dtype = DLDataType{ kDLFloat, static_cast<std::uint8_t>(r.scalar * 8), 1 };
            outs.push_back({ caller_plane(ctx, grads.planes[i], like, 1, what), -1, g });
        }
    }
    // Samples per pass under the budget (each pass a whole number of samples; leaf 1 under the compensated sums).
    std::int64_t per = 0;
    for (const Carried& c : r.state) per += c.rows * static_cast<std::int64_t>(c.bytes);
    const std::int64_t windows = (runner->max_steps + W - 1) / W;
    per = windows * per + W * r.width * static_cast<std::int64_t>(r.scalar);
    std::int64_t budget = r.budget;
    if (budget == 0) budget = static_cast<std::int64_t>(::sysconf(_SC_AVPHYS_PAGES) * ::sysconf(_SC_PAGE_SIZE)) / 2;
    const std::int64_t m = n > 0 ? diff::chunkSamples(budget, per, n, 1) : 0;
    if (n > 0 && m == 0)
        refuse(EG_BAD_REQUEST, "memory_budget " + std::to_string(budget) + ": one sample alone needs " + std::to_string(per) +
               " bytes (its checkpoints and a " + std::to_string(W) + "-step tape)");
    for (Carried& c : r.state) c.buf.resize(static_cast<std::size_t>(c.rows * m) * c.bytes);
    for (Carried& c : r.reads) c.buf.resize(static_cast<std::size_t>(c.rows * m) * c.bytes);
    for (TapedState& x : r.taped)
        for (Buf& a : x.adj) a.resize(static_cast<std::size_t>(x.rows * m) * r.scalar);
    for (Grad& g : r.grads) {
        g.grad.resize(static_cast<std::size_t>(g.rows * m) * r.scalar);
        g.acc.resize(static_cast<std::size_t>(g.rows * m) * r.scalar);
    }
    Buf step_plane;
    step_plane.resize(static_cast<std::size_t>(m) * r.step_bytes);
    Buf tape, lookup;
    tape.resize(static_cast<std::size_t>(W * r.width * std::max<std::int64_t>(m, 1)) * r.scalar);
    // The recompute twin (tape on) and the vjp over the pass's planes.
    inner_free(ctx, r.rec);
    inner_free(ctx, r.vjp);
    r.rec = inner_plan(ctx, r.twin_unit, primal.structure, m);
    r.vjp = inner_plan(ctx, r.vjp_unit, primal.structure, m);
    eg_plan_s& rec = *r.rec;
    eg_plan_s& vjp = *r.vjp;
    const DerivedFacts& t = r.twin_unit->derived;
    for (Carried& c : r.state) bind_raw(rec, c.twin_slot, c.buf.data(), static_cast<std::uint64_t>(m));
    for (Carried& c : r.reads) bind_raw(rec, c.twin_slot, c.buf.data(), static_cast<std::uint64_t>(m));
    bind_raw(rec, t.tape_slot, tape.data(), static_cast<std::uint64_t>(m));
    bind_int64(rec, t.tape_on_slot, 1);
    // An active-set twin reads the identity map over its pass's slots (the pass's samples are in stop order), counted per launch.
    std::vector<std::int32_t> identity(static_cast<std::size_t>(std::max<std::int64_t>(m, 1)));
    for (std::size_t i = 0; i < identity.size(); ++i) identity[i] = static_cast<std::int32_t>(i);
    std::uint32_t rec_count = 0;
    if (r.twin_unit->active_map_slot >= 0) {
        bind_raw(rec, r.twin_unit->active_map_slot, identity.data(), static_cast<std::uint64_t>(m));
        bind_raw(rec, r.twin_unit->active_count_slot, &rec_count, 1);
    }
    for (const DerivedFacts::Link& l : t.links)
        if (r.twin_unit->slots[static_cast<std::size_t>(l.slot)].role == EG_ROLE_UNIFORM)
            bind_value(rec, l.slot, primal.bindings[static_cast<std::size_t>(l.of_slot)].value, sizeof(primal.bindings[0].value));
    for (std::size_t i = 0; i < r.vjp_slots.size(); ++i) {
        const VjpSlot& vs = r.vjp_slots[i];
        const int s = static_cast<int>(i);
        if (r.vjp_unit->slots[i].role == EG_ROLE_NSAMPLES) continue;
        if (vs.kind == VjpSlot::UNIFORM) bind_value(vjp, s, primal.bindings[static_cast<std::size_t>(vs.index)].value, sizeof(primal.bindings[0].value));
        else if (vs.kind == VjpSlot::GRAD) bind_raw(vjp, s, r.grads[static_cast<std::size_t>(vs.index)].grad.data(), static_cast<std::uint64_t>(m));
        else if (vs.kind == VjpSlot::READ) bind_raw(vjp, s, r.reads[static_cast<std::size_t>(vs.index)].buf.data(), static_cast<std::uint64_t>(m));
        else if (vs.kind == VjpSlot::STEP) bind_raw(vjp, s, step_plane.data(), static_cast<std::uint64_t>(m));
    }
    const auto bind_step = [&](std::int64_t row, std::int64_t parity) {
        for (std::size_t i = 0; i < r.vjp_slots.size(); ++i) {
            const VjpSlot& vs = r.vjp_slots[i];
            const int s = static_cast<int>(i);
            if (vs.kind == VjpSlot::TAPED) {
                const TapedState& x = r.taped[static_cast<std::size_t>(vs.index)];
                bind_raw(vjp, s, tape.data() + static_cast<std::size_t>((row * r.width + x.row0) * m) * r.scalar, static_cast<std::uint64_t>(m));
            } else if (vs.kind == VjpSlot::ADJ_IN) {
                bind_raw(vjp, s, r.taped[static_cast<std::size_t>(vs.index)].adj[parity].data(), static_cast<std::uint64_t>(m));
            } else if (vs.kind == VjpSlot::ADJ_OUT) {
                bind_raw(vjp, s, r.taped[static_cast<std::size_t>(vs.index)].adj[1 - parity].data(), static_cast<std::uint64_t>(m));
            }
        }
    };
    // Outputs in sample order: the taped states' adjoints, per-sample gradients and the passes' pairs.
    std::vector<Buf> fold(r.grads.size());   // a start copy's per-sample gradient, in sample order
    for (std::size_t g = 0; g < r.grads.size(); ++g)
        if (r.grads[g].fold >= 0) fold[g].resize(static_cast<std::size_t>(n) * r.scalar);
    std::vector<Buf> out_state(r.taped.size());
    for (std::size_t k = 0; k < r.taped.size(); ++k) out_state[k].resize(static_cast<std::size_t>(r.taped[k].rows * n) * r.scalar);
    std::vector<std::vector<std::pair<double, double>>> partials(r.grads.size());
    const std::int64_t* all_steps = reinterpret_cast<const std::int64_t*>(r.steps.data());
    const int threads = host::team_threads(ctx);
    for (std::int64_t s0 = 0; s0 < n; s0 += m) {
        const std::int64_t cnt = std::min(m, n - s0);
        const std::vector<diff::idx_t> order = diff::stopOrder(all_steps + s0, cnt);
        const std::vector<diff::idx_t> live = diff::liveCounts(all_steps + s0, cnt);
        std::vector<std::int64_t> cols(static_cast<std::size_t>(cnt));
        for (std::int64_t i = 0; i < cnt; ++i) cols[static_cast<std::size_t>(i)] = order[static_cast<std::size_t>(i)] + s0;
        for (Carried& c : r.reads) diff::gatherColumns(c.buf.data(), m, c.src, n, cols.data(), cnt, c.rows, c.bytes);
        for (std::size_t k = 0; k < r.taped.size(); ++k)
            for (Buf& a : r.taped[k].adj) {
                a.zero();
                if (seed[k] != nullptr) diff::gatherColumns(a.data(), m, seed[k], n, cols.data(), cnt, r.taped[k].rows, r.scalar);
            }
        for (Grad& g : r.grads) g.acc.zero();
        if (!live.empty()) {
            diff::AdjointChain chain(static_cast<std::int64_t>(live.size()), W, 0);
            diff::reverseWindows(
                W, live,
                [&](std::int64_t k, std::int64_t count) {
                    for (std::size_t i = 0; i < r.state.size(); ++i)
                        diff::gatherColumns(r.state[i].buf.data(), m, r.pool[static_cast<std::size_t>(k)][i].data(), n, cols.data(), count,
                                            r.state[i].rows, r.state[i].bytes);
                },
                [&](std::int64_t c, std::int64_t len) {
                    for (std::int64_t j = c; j < c + len; ++j) {
                        rec_count = static_cast<std::uint32_t>(live[static_cast<std::size_t>(j)]);
                        host::run_prefix(rec, threads, live[static_cast<std::size_t>(j)]);
                    }
                },
                [&](std::int64_t j, std::int64_t count) {
                    bind_step(j % W, chain.cur);
                    if (r.step_bytes == 8) diff::fillPrefix(reinterpret_cast<std::int64_t*>(step_plane.data()), j, count);
                    else if (r.step_bytes == 4) diff::fillPrefix(reinterpret_cast<std::int32_t*>(step_plane.data()), static_cast<std::int32_t>(j), count);
                    host::run_prefix(vjp, threads, count);
                    chain.flip();
                    for (Grad& g : r.grads) {
                        if (r.scalar == 8) accumulate<double>(g, m, count);
                        else accumulate<float>(g, m, count);
                    }
                });
        }
        for (std::size_t k = 0; k < r.taped.size(); ++k)
            put_columns(out_state[k].data(), n, r.taped[k].adj[1].data(), m, cols.data(), cnt, r.taped[k].rows, r.scalar);
        for (std::size_t g = 0; g < r.grads.size(); ++g) {
            Grad& gr = r.grads[g];
            if (gr.fold >= 0) {
                put_columns(fold[g].data(), n, gr.acc.data(), m, cols.data(), cnt, 1, r.scalar);
                continue;
            }
            Buf sample;
            sample.resize(static_cast<std::size_t>(cnt) * r.scalar);
            for (std::int64_t i = 0; i < cnt; ++i)
                std::memcpy(sample.data() + static_cast<std::size_t>(order[static_cast<std::size_t>(i)]) * r.scalar,
                            gr.acc.data() + static_cast<std::size_t>(i) * r.scalar, r.scalar);
            if (r.scalar == 8) {
                const auto p = pair_sum(reinterpret_cast<const double*>(sample.data()), cnt);
                partials[g].push_back({ p.h, p.c });
            } else {
                const auto p = pair_sum(reinterpret_cast<const float*>(sample.data()), cnt);
                partials[g].push_back({ static_cast<double>(p.h), static_cast<double>(p.c) });
            }
        }
    }
    for (std::size_t g = 0; g < r.grads.size(); ++g) {   // the start copy's gradient joins the time plane's
        if (r.grads[g].fold < 0) continue;
        Buf& dst = out_state[static_cast<std::size_t>(r.grads[g].fold)];
        if (r.scalar == 8)
            for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<double*>(dst.data())[i] += reinterpret_cast<const double*>(fold[g].data())[i];
        else
            for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<float*>(dst.data())[i] += reinterpret_cast<const float*>(fold[g].data())[i];
    }
    for (const Out& o : outs) {
        if (o.taped >= 0) {
            std::memcpy(o.data, out_state[static_cast<std::size_t>(o.taped)].data(),
                        static_cast<std::size_t>(r.taped[static_cast<std::size_t>(o.taped)].rows * n) * r.scalar);
        } else {   // diff_combine_chunks: the passes' pairs combined in pass order
            const auto& ps = partials[static_cast<std::size_t>(o.grad)];
            if (r.scalar == 8) {
                std::vector<reduce::canonical::Partial<double, aether::SumOp<double>>> p;
                for (const auto& [h, c] : ps) p.push_back({ h, c });
                const double v = ps.empty() ? 0.0 : reduce::canonical::combine(std::move(p)).value();
                std::memcpy(o.data, &v, sizeof(v));
            } else {
                std::vector<reduce::canonical::Partial<float, aether::SumOp<float>>> p;
                for (const auto& [h, c] : ps) p.push_back({ static_cast<float>(h), static_cast<float>(c) });
                const float v = ps.empty() ? 0.0f : reduce::canonical::combine(std::move(p)).value();
                std::memcpy(o.data, &v, sizeof(v));
            }
        }
    }
    r.live = false;
}

}  // namespace door
}  // namespace eagle
