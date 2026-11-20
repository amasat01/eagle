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
// as a canonical compensated pair per pass and combined in pass order. A DEVICE
// runner's record runs the same driver over device memory (the old device
// path's diff_record_device / diff_reverse_device): eagle's planes on the device,
// every copy, gather and launch queued on the context's stream, the finished
// counter read back once per window.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Team.h"
#include "eagle/diff/Record.h"
#include "eagle/diff/Reverse.h"
#include "eagle/diff/Window.h"
#include "eagle/events/Sensitivity.h"
#include "eagle/reduce/canonical.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace eagle {
namespace door {

/** @brief Bytes eagle owns, 8-byte aligned: on the host, or (resized with a device context) on that device. */
struct Buf {
    std::vector<std::uint64_t> words;
    std::shared_ptr<char> dev;      ///< a DEVICE record's bytes
    eg_ctx* ctx = nullptr;
    std::size_t bytes = 0;
    void resize(std::size_t n, eg_ctx* on = nullptr) {
        if (on == nullptr) {
            words.assign((n + 7) / 8, 0);
            return;
        }
        ctx = on;
        bytes = n;
        dev.reset(static_cast<char*>(device_alloc(*on, std::max<std::size_t>(n, 8))), [on](char* p) { device_free(*on, p); });
    }
    char* data() { return dev ? dev.get() : reinterpret_cast<char*>(words.data()); }
    void zero() {
        if (dev) device_zero(*ctx, dev.get(), bytes);
        else std::fill(words.begin(), words.end(), 0);
    }
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
    bool plane = false;             ///< the gradient of a per-sample read (reduced 0): one value per sample, in sample order, unsummed
    Buf grad, acc;
};

/** @brief How a vjp slot is bound in a launch. */
struct VjpSlot {
    enum Kind { UNIFORM, TAPED, ADJ_IN, ADJ_OUT, GRAD, READ, STEP, WSTATE, ZERO } kind = UNIFORM;   // WSTATE: a taped state at its window's start; ZERO: the window's entry flag
    int index = -1;                 ///< TAPED / ADJ_*: taped; GRAD: grads; READ: reads; UNIFORM: primal slot
};

/** @brief How a slot of a refine unit (REFINE_NEWTON, STATE_GUARD_VJP) is bound at a run. */
struct RSlot {
    enum Kind { NONE, UNIFORM, PLANE, TAPED, STEPS, TIME0, BUF } kind = NONE;
    int a = -1;                     ///< UNIFORM, PLANE: the primal slot; TAPED: the taped index; BUF: the plane eagle makes
};

/** @brief A plane eagle makes for a refine run: @c rows rows of n samples. */
struct RBuf {
    enum Fill { NONE, ONES, UNIFORM_OF } fill = NONE;
    std::int64_t rows = 1;
    int of = -1;                    ///< UNIFORM_OF: the primal uniform whose value fills it
    Buf buf;
};

/** @brief One refined state: the primal slot, its refined plane and derivative and the guard's gradient in it. */
struct RState {
    int primal = -1, refined = -1, dot = -1, gx = -1;
    int rbar = -1, share = -1;       ///< a recorded run: the refine vjp's end-state cotangent seed and its pre-stop cotangent
    std::int64_t rows = 1;
};

/** @brief events refine on the host: the Newton and the guard vjp over eagle-made planes. */
struct RefineRun {
    eg_unit_s* newton_unit = nullptr;
    eg_unit_s* guard_unit = nullptr;
    eg_plan_s* newton = nullptr;
    eg_plan_s* guard = nullptr;
    std::vector<RSlot> nslots, gslots;
    std::vector<RBuf> bufs;
    std::vector<int> g, dg, seed;   ///< per event: the guard, its derivative and the one-hot seed
    std::vector<RState> states;
    eg_unit_s* rvjp_unit = nullptr; ///< a recorded run: the vjp of the refine entry
    eg_plan_s* rvjp = nullptr;
    std::vector<RSlot> vslots;
    std::vector<int> gbar;          ///< per event: the refine vjp's guard seed
    std::vector<std::pair<int, int>> pgrads;   ///< (buffer, primal uniform): a per-sample share of a parameter's gradient
    std::vector<std::pair<int, int>> sgrads;   ///< (buffer, primal per-sample slot): the refine vjp's per-sample gradient of a read
    int t0grad = -1, t0of = -1;     ///< the start copy's per-sample gradient and the primal time plane it joins
    int tau = -1, h = -1;           ///< buffers: the candidate time; the step (-1: the primal plane h_primal)
    int h_primal = -1;
    Buf nsteps, status, which, lo, hi, tstop, zero;
    bool done = false;              ///< a refine run finished: status and t_stop are readable
    eg_no_event on_no_event = EG_NO_EVENT_REFUSE;
};

struct RunRecord {
    RefineRun* rf = nullptr;        ///< events refine (no EG_RUN_RECORD): the Newton and the guard vjp
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
    std::vector<VjpSlot> vjp_slots, fvjp_slots;
    eg_plan_s* fvjp = nullptr;      ///< the K-step vjp over a pass's planes (steps_reverse FUSED)
    eg_unit_s* fvjp_unit = nullptr;
    std::int64_t fused_k = 0;       ///< steps one fused vjp launch takes: the window, 0 for none
    bool fused = false;             ///< the twin is the K-step one (EG_DERIVED_TAPE_FUSED)
    std::int64_t max_k = 1;         ///< steps one launch fuses at most (the twin's finish)
    Buf fword;                      ///< the forward's steps word (fused_steps_slot)
    bool recorded = false;          ///< events refine with EG_RUN_RECORD: checkpoints kept, window 1, the refine vjp in rf
    std::size_t step_bytes = 0;     ///< the width of the vjp's step plane (EG_WHAT_STEPS), 0: none
    Buf steps;                      ///< applied steps, int64 per sample
    Buf dummy;                      ///< the forward's tape: one row of width * n, never written
    Buf tape;                       ///< a refine run's tape: one row of width * n, written (the state before each stop step)
    std::vector<std::vector<Buf>> pool;   ///< checkpoints: pool[k][state]
    std::vector<std::vector<std::int32_t>> epochs;   ///< a reordering run: epochs[k][sample] = its slot at checkpoint k
    std::int64_t taken = 0;
    bool live = false;              ///< a record not reversed yet
};

namespace {

char* plane_bytes(const eg_plane_s& p) {
    const DLTensor& t = p.tensor->dl_tensor;
    return static_cast<char*>(t.data) + t.byte_offset;
}
std::size_t elem_bytes(const DLTensor& t) { return static_cast<std::size_t>(t.dtype.bits / 8) * t.dtype.lanes; }

}  // namespace

/** @brief Copy @p bytes on one side: host memory, or device to device on @p dev. */
void copy_bytes(eg_ctx* dev, void* dst, const void* src, std::size_t bytes) {
    if (dev != nullptr) device_copy_bytes(*dev, dst, src, bytes);
    else std::memcpy(dst, src, bytes);
}

namespace {

/** @brief diff::gatherColumns on the host (on @p threads threads), or queued on @p dev's stream (@p index on the device). */
void gather(eg_ctx* dev, void* dst, std::int64_t dst_ld, const void* src, std::int64_t src_ld, const std::int64_t* index,
            std::int64_t count, std::int64_t rows, std::size_t bytes, int threads) {
    if (dev != nullptr) device_columns(*dev, 0, dst, dst_ld, src, src_ld, index, count, rows, static_cast<unsigned>(bytes), 0);
    else diff::gatherColumns(dst, dst_ld, src, src_ld, index, count, rows, bytes, threads);
}

/** @brief The int64 steps word of a K-step launch: written on the host, or filled on @p dev's stream. */
void set_word(eg_ctx* dev, Buf& word, std::int64_t k) {
    if (dev != nullptr) device_columns(*dev, 2, word.data(), 1, nullptr, 0, nullptr, 1, 1, 8, k);
    else std::memcpy(word.data(), &k, sizeof(k));
}

/** @brief Run @p plan over its first @p count samples: the host team, or one device launch with the block of @p of samples. */
void run_on(eg_plan_s& plan, int threads, std::int64_t count, std::int64_t of) {
    if (plan.structure == EG_STRUCTURE_DEVICE) device_launch_prefix(plan, count, of);
    else host::run_prefix(plan, threads, count);
}

}  // namespace

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
    if (unit->finish.present && structure == EG_STRUCTURE_DEVICE) {   // the finished counter: a device cell
        device_plan_cells(*p);
        bind_raw(*p, unit->finish.counter_slot, reinterpret_cast<void*>(p->device_counter), 1);
    } else if (unit->finish.present) {
        bind_raw(*p, unit->finish.counter_slot, &p->finished, 1);
    }
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
    device_plan_free(*p);
    delete p;
    p = nullptr;
}
void bind_value(eg_plan_s& p, int slot, const void* value, std::size_t bytes) {
    eg_plan_s::Binding& b = p.bindings[static_cast<std::size_t>(slot)];
    std::memcpy(b.value, value, bytes);
    b.bound = true;
    p.params[static_cast<std::size_t>(slot)] = b.value;
}
namespace {

void bind_int64(eg_plan_s& p, int slot, std::int64_t v) { bind_value(p, slot, &v, sizeof(v)); }

/** @brief The plane eagle made for twin slot @p s (a filled uniform or the start copy), or null. */
const Carried* made_plane(const RunRecord& r, int s) {
    for (const Carried& c : r.reads)
        if (c.twin_slot == s && (c.kind == Carried::FILL || c.kind == Carried::START)) return &c;
    return nullptr;
}

std::string field(const char* unit, const std::string& name) { return std::string(unit) + ".derived." + name; }

/** @brief The derived unit of @p role in the runner description; refuses what is missing or not written. */
eg_unit_s* derived_of(const eg_ctx& ctx, const eg_runner_desc& d, eg_derived_role role, bool refine = false) {
    eg_unit_s* found = nullptr;
    for (std::int32_t i = 0; i < d.n_derived; ++i) {
        const std::string at = "derived_units[" + std::to_string(i) + "]";
        check_handle(ctx.units, d.derived_units[i], at.c_str());
        const eg_unit_s& u = *d.derived_units[i];
        const eg_derived_role r = d.derived_roles[i];
        const bool rec = (d.flags & EG_RUN_RECORD) != 0;
        if (refine && r != EG_DERIVED_TAPE && r != EG_DERIVED_TAPE_FUSED && r != EG_DERIVED_REFINE_NEWTON && r != EG_DERIVED_STATE_GUARD_VJP &&
            !(rec && (r == EG_DERIVED_STEP_VJP || r == EG_DERIVED_REFINE_VJP)))
            refuse(EG_BAD_REQUEST, "derived_roles[" + std::to_string(i) + "]: a refine run takes the twin "
                   "(EG_DERIVED_TAPE), EG_DERIVED_REFINE_NEWTON and EG_DERIVED_STATE_GUARD_VJP (recorded: and EG_DERIVED_STEP_VJP, EG_DERIVED_REFINE_VJP)");
        if (r == EG_DERIVED_FUSED_VJP && (d.steps_reverse != EG_STEPS_REVERSE_FUSED || !rec || refine))
            refuse(EG_BAD_REQUEST, "derived_roles[" + std::to_string(i) + "]: the K-step vjp (EG_DERIVED_FUSED_VJP) belongs to a plain "
                   "recorded run with steps_reverse FUSED");
        if (r == EG_DERIVED_TAPE_FUSED && refine && !rec)
            refuse(EG_BAD_REQUEST, "derived_roles[" + std::to_string(i) + "]: a plain refine run takes the one-step twin");
        if (!refine && r >= EG_DERIVED_REFINE_ENTRY && r <= EG_DERIVED_STATE_GUARD_VJP)
            refuse(EG_NOT_WRITTEN, "derived_roles[" + std::to_string(i) + "]: the events refine roles are not written "
                   "in this build of the door");
        if (r != EG_DERIVED_TAPE && r != EG_DERIVED_TAPE_FUSED && r != EG_DERIVED_STEP_VJP && r != EG_DERIVED_FUSED_VJP && !refine)
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
        refuse(EG_BAD_REQUEST, std::string(refine ? "events refine needs the " : "EG_RUN_RECORD needs the ") +
               (role == EG_DERIVED_TAPE_FUSED ? "K-step taping twin (EG_DERIVED_TAPE_FUSED)"
                : role == EG_DERIVED_TAPE ? "one-step taping twin (EG_DERIVED_TAPE)"
                : role == EG_DERIVED_REFINE_NEWTON ? "Newton unit (EG_DERIVED_REFINE_NEWTON)"
                : role == EG_DERIVED_STATE_GUARD_VJP ? "state guard vjp (EG_DERIVED_STATE_GUARD_VJP)"
                : role == EG_DERIVED_REFINE_VJP ? "refine vjp (EG_DERIVED_REFINE_VJP)"
                : role == EG_DERIVED_FUSED_VJP ? "K-step vjp (EG_DERIVED_FUSED_VJP)" : "step vjp (EG_DERIVED_STEP_VJP)"));
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


/** @brief Bind a refine unit's slots: key (kind, a) planes dedupe across the Newton and the guard vjp. */
void refine_setup(RunRecord& r, const eg_unit_s& pu, eg_unit_s* newton, eg_unit_s* guard, eg_unit_s* rvjp,
                  const std::vector<int>& taped_of, const eg_runner_desc& d) {
    const DerivedFacts& t = r.twin_unit->derived;
    const char* tn = r.twin_unit->name.c_str();
    if (t.time0_slot == -1)
        refuse(EG_BAD_REQUEST, field(tn, "time0_slot") + " is -1: events refine reads the time plane's start copy");
    auto* f = new RefineRun;
    r.rf = f;
    f->newton_unit = newton;
    f->guard_unit = guard;
    ++newton->refs;
    ++guard->refs;
    f->on_no_event = d.on_no_event;
    const std::size_t np = pu.slots.size();
    int start_of = -1;   // the primal time plane the twin copies at run start
    for (const Carried& c : r.reads)
        if (c.kind == Carried::START) start_of = c.of;
    std::map<std::pair<int, int>, int> made;
    const auto buf = [&](int kind, int key, std::int64_t rows, RBuf::Fill fill, int of) {
        const auto it = made.find({ kind, key });
        if (it != made.end()) return it->second;
        RBuf b;
        b.rows = rows;
        b.fill = fill;
        b.of = of;
        f->bufs.push_back(std::move(b));
        made[{ kind, key }] = static_cast<int>(f->bufs.size()) - 1;
        return static_cast<int>(f->bufs.size()) - 1;
    };
    const auto rows_of = [&](int s) {
        const int ti = taped_of[static_cast<std::size_t>(s)];
        return ti >= 0 ? r.taped[static_cast<std::size_t>(ti)].rows : std::int64_t{ 1 };
    };
    const auto state_of = [&](int s) {
        for (std::size_t i = 0; i < f->states.size(); ++i)
            if (f->states[i].primal == s) return static_cast<int>(i);
        RState st;
        st.primal = s;
        st.rows = rows_of(s);
        f->states.push_back(st);
        return static_cast<int>(f->states.size()) - 1;
    };
    // The links the two units share: a primal read, a tape row, the start copy.
    const auto common = [&](const eg_unit_s& u, const DerivedFacts::Link& l, const std::string& at, RSlot& out) -> bool {
        const SlotFacts& sf = u.slots[static_cast<std::size_t>(l.slot)];
        const char* un = u.name.c_str();
        if (l.what == EG_WHAT_PRIMAL_SLOT && l.form == EG_FORM_VALUE) {
            if (!in_range(l.of_slot, np)) refuse(EG_BAD_REQUEST, field(un, at + ".of_slot") + " is out of range");
            const std::int32_t prole = pu.slots[static_cast<std::size_t>(l.of_slot)].role;
            if (sf.role == EG_ROLE_UNIFORM && prole == EG_ROLE_UNIFORM) out = { RSlot::UNIFORM, l.of_slot };
            else if (sf.role == EG_ROLE_PER_SAMPLE && prole == EG_ROLE_PER_SAMPLE) out = { RSlot::PLANE, l.of_slot };
            else if (sf.role == EG_ROLE_PER_SAMPLE && prole == EG_ROLE_UNIFORM)
                out = { RSlot::BUF, buf(11, l.of_slot, 1, RBuf::UNIFORM_OF, l.of_slot) };
            else refuse(EG_BAD_REQUEST, field(un, at) + ": slot '" + sf.name + "' and primal slot " + std::to_string(l.of_slot) +
                        " differ in role");
            return true;
        }
        if (l.what == EG_WHAT_TAPE_ROW) {
            const int ti = in_range(l.of_slot, np) ? taped_of[static_cast<std::size_t>(l.of_slot)] : -1;
            if (ti < 0) refuse(EG_BAD_REQUEST, field(un, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is not a taped state");
            if (sf.role != EG_ROLE_PER_SAMPLE) refuse(EG_BAD_REQUEST, field(un, at) + ": a tape row is a per-sample read");
            out = { RSlot::TAPED, ti };
            return true;
        }
        return false;
    };
    // ---- the Newton
    const DerivedFacts& nv = newton->derived;
    const char* nn = newton->name.c_str();
    if (newton->finish.present) refuse(EG_BAD_REQUEST, "unit '" + newton->name + "': a refine unit has no finish");
    f->tau = buf(5, 0, 1, RBuf::NONE, -1);
    f->nslots.assign(newton->slots.size(), RSlot{});
    std::vector<int> gb(events::kMaxGuards, -1), dgb(events::kMaxGuards, -1);
    int guards = 0, step_links = 0, tau_links = 0;
    for (std::size_t i = 0; i < nv.links.size(); ++i) {
        const DerivedFacts::Link& l = nv.links[i];
        const std::string at = "links[" + std::to_string(i) + "]";
        if (!in_range(l.slot, newton->slots.size())) refuse(EG_BAD_REQUEST, field(nn, at + ".slot") + " is out of range");
        if (f->nslots[static_cast<std::size_t>(l.slot)].kind != RSlot::NONE)
            refuse(EG_BAD_REQUEST, field(nn, at + ".slot") + " " + std::to_string(l.slot) + " has a second link");
        const SlotFacts& sf = newton->slots[static_cast<std::size_t>(l.slot)];
        RSlot& out = f->nslots[static_cast<std::size_t>(l.slot)];
        if (common(*newton, l, at, out)) continue;
        if (l.what == EG_WHAT_STEPS) {
            if (sf.role != EG_ROLE_PER_SAMPLE || sf.dtype != EG_DTYPE_INT64) refuse(EG_BAD_REQUEST, field(nn, at) + ": a step plane is a per-sample int64 slot");
            out = { RSlot::STEPS, -1 };
        } else if (l.what == EG_WHAT_TIME0) {
            if (sf.role != EG_ROLE_PER_SAMPLE || l.of_slot != start_of)
                refuse(EG_BAD_REQUEST, field(nn, at) + ": the start copy is a per-sample read of the twin's time plane");
            out = { RSlot::TIME0, -1 };
        } else if (l.what == EG_WHAT_GUARD) {
            const bool value = l.form == EG_FORM_VALUE;
            if (sf.role != EG_ROLE_MUTABLE || sf.dtype != EG_DTYPE_REAL || l.of_slot < 0 || l.of_slot >= events::kMaxGuards)
                refuse(EG_BAD_REQUEST, field(nn, at) + ": a guard value is a MUTABLE real slot of an event below " + std::to_string(events::kMaxGuards));
            (value ? gb : dgb)[static_cast<std::size_t>(l.of_slot)] = buf(value ? 1 : 2, l.of_slot, 1, RBuf::NONE, -1);
            out = { RSlot::BUF, (value ? gb : dgb)[static_cast<std::size_t>(l.of_slot)] };
            guards = std::max(guards, l.of_slot + 1);
        } else if (l.what == EG_WHAT_REFINED) {
            const bool value = l.form == EG_FORM_VALUE;
            if (sf.role != EG_ROLE_MUTABLE || sf.dtype != EG_DTYPE_REAL || !in_range(l.of_slot, np) ||
                pu.slots[static_cast<std::size_t>(l.of_slot)].role != EG_ROLE_MUTABLE)
                refuse(EG_BAD_REQUEST, field(nn, at) + ": a refined state is a MUTABLE real slot of a MUTABLE primal state");
            const int st = state_of(l.of_slot);
            RState& s = f->states[static_cast<std::size_t>(st)];
            int& which = value ? s.refined : s.dot;
            which = buf(value ? 3 : 4, l.of_slot, s.rows, RBuf::NONE, -1);
            out = { RSlot::BUF, which };
        } else if (l.what == EG_WHAT_TAU && l.form == EG_FORM_SEED) {
            if (sf.role != EG_ROLE_PER_SAMPLE || sf.dtype != EG_DTYPE_REAL) refuse(EG_BAD_REQUEST, field(nn, at) + ": the tau seed is a per-sample real slot");
            out = { RSlot::BUF, buf(6, 0, 1, RBuf::ONES, -1) };
        } else if (l.what == EG_WHAT_TAU) {   // (TAU, VALUE): the candidate time inside the stop step
            if (sf.role != EG_ROLE_PER_SAMPLE || sf.dtype != EG_DTYPE_REAL) refuse(EG_BAD_REQUEST, field(nn, at) + ": the tau is a per-sample real slot");
            if (++tau_links > 1) refuse(EG_BAD_REQUEST, field(nn, at) + ": a second (TAU, VALUE) link");
            out = { RSlot::BUF, f->tau };
        } else if (l.what == EG_WHAT_TIME_STEP) {
            if (sf.role != EG_ROLE_PER_SAMPLE || sf.dtype != EG_DTYPE_REAL || !in_range(l.of_slot, np))
                refuse(EG_BAD_REQUEST, field(nn, at) + ": the time step is a per-sample real slot of a primal slot");
            if (++step_links > 1) refuse(EG_BAD_REQUEST, field(nn, at) + ": a second (TIME_STEP, VALUE) link");
            const std::int32_t prole = pu.slots[static_cast<std::size_t>(l.of_slot)].role;
            if (prole == EG_ROLE_UNIFORM) {
                f->h = buf(7, 0, 1, RBuf::UNIFORM_OF, l.of_slot);
                out = { RSlot::BUF, f->h };
            } else if (prole == EG_ROLE_PER_SAMPLE) {
                f->h_primal = l.of_slot;
                out = { RSlot::PLANE, l.of_slot };
            } else refuse(EG_BAD_REQUEST, field(nn, at + ".of_slot") + " is neither a parameter nor a per-sample plane");
        } else {
            refuse(EG_NOT_WRITTEN, field(nn, at) + ": (what " + std::to_string(l.what) + ", form " + std::to_string(l.form) + ") on a Newton unit is not written in this build of the door");
        }
    }
    if (tau_links != 1) refuse(EG_BAD_REQUEST, field(nn, "links") + ": a Newton unit takes exactly one (TAU, VALUE) link");
    if (step_links != 1) refuse(EG_BAD_REQUEST, field(nn, "links") + ": a Newton unit takes exactly one (TIME_STEP, VALUE) link");
    if (guards < 1) refuse(EG_BAD_REQUEST, field(nn, "links") + ": a Newton unit takes at least one (GUARD, VALUE) link");
    if (guards != pu.finish.n_events)
        refuse(EG_BAD_REQUEST, field(nn, "links") + ": guards for events 0.." + std::to_string(guards - 1) +
               ", the primal's finish.n_events is " + std::to_string(pu.finish.n_events));
    for (int j = 0; j < guards; ++j) {
        if (gb[static_cast<std::size_t>(j)] < 0 || dgb[static_cast<std::size_t>(j)] < 0)
            refuse(EG_BAD_REQUEST, field(nn, "links") + ": event " + std::to_string(j) + " needs its (GUARD, VALUE) and (GUARD, DOT) links");
        f->g.push_back(gb[static_cast<std::size_t>(j)]);
        f->dg.push_back(dgb[static_cast<std::size_t>(j)]);
    }
    std::int64_t components = 0;
    for (const RState& s : f->states) {
        if (s.refined < 0 || s.dot < 0)
            refuse(EG_BAD_REQUEST, field(nn, "links") + ": primal state " + std::to_string(s.primal) + " needs its (REFINED, VALUE) and (REFINED, DOT) links");
        components += s.rows;
    }
    if (components > events::kMaxComponents) refuse(EG_NOT_WRITTEN, field(nn, "links") + ": more refined rows than " + std::to_string(events::kMaxComponents));
    for (std::size_t i = 0; i < newton->slots.size(); ++i)
        if (f->nslots[i].kind == RSlot::NONE && newton->slots[i].role != EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, field(nn, "links") + ": Newton slot " + std::to_string(i) + " ('" + newton->slots[i].name + "') has no link");
    // ---- the state guard vjp
    const DerivedFacts& gv = guard->derived;
    const char* gn = guard->name.c_str();
    if (guard->finish.present) refuse(EG_BAD_REQUEST, "unit '" + guard->name + "': a state guard vjp has no finish");
    f->gslots.assign(guard->slots.size(), RSlot{});
    f->seed.assign(static_cast<std::size_t>(guards), -1);
    for (std::size_t i = 0; i < gv.links.size(); ++i) {
        const DerivedFacts::Link& l = gv.links[i];
        const std::string at = "links[" + std::to_string(i) + "]";
        if (!in_range(l.slot, guard->slots.size())) refuse(EG_BAD_REQUEST, field(gn, at + ".slot") + " is out of range");
        if (f->gslots[static_cast<std::size_t>(l.slot)].kind != RSlot::NONE)
            refuse(EG_BAD_REQUEST, field(gn, at + ".slot") + " " + std::to_string(l.slot) + " has a second link");
        const SlotFacts& sf = guard->slots[static_cast<std::size_t>(l.slot)];
        RSlot& out = f->gslots[static_cast<std::size_t>(l.slot)];
        if (common(*guard, l, at, out)) continue;
        if (l.what == EG_WHAT_REFINED && l.form == EG_FORM_VALUE) {
            int found = -1;
            for (std::size_t k = 0; k < f->states.size(); ++k)
                if (f->states[k].primal == l.of_slot) found = static_cast<int>(k);
            if (found < 0 || sf.role != EG_ROLE_PER_SAMPLE)
                refuse(EG_BAD_REQUEST, field(gn, at) + ": the refined plane of primal state " + std::to_string(l.of_slot) + " is one the Newton writes, read per sample");
            out = { RSlot::BUF, f->states[static_cast<std::size_t>(found)].refined };
        } else if (l.what == EG_WHAT_GUARD && l.form == EG_FORM_ADJOINT_IN) {
            if (sf.role != EG_ROLE_PER_SAMPLE || l.of_slot < 0 || l.of_slot >= guards)
                refuse(EG_BAD_REQUEST, field(gn, at) + ": an event seed is a per-sample slot of an event of the Newton");
            f->seed[static_cast<std::size_t>(l.of_slot)] = buf(8, l.of_slot, 1, RBuf::NONE, -1);
            out = { RSlot::BUF, f->seed[static_cast<std::size_t>(l.of_slot)] };
        } else if (l.what == EG_WHAT_REFINED && l.form == EG_FORM_ADJOINT_OUT) {
            int found = -1;
            for (std::size_t k = 0; k < f->states.size(); ++k)
                if (f->states[k].primal == l.of_slot) found = static_cast<int>(k);
            if (found < 0 || sf.role != EG_ROLE_MUTABLE)
                refuse(EG_BAD_REQUEST, field(gn, at) + ": the guard's gradient in primal state " + std::to_string(l.of_slot) + " is a MUTABLE slot of a refined state");
            RState& s = f->states[static_cast<std::size_t>(found)];
            s.gx = buf(9, l.of_slot, s.rows, RBuf::NONE, -1);
            out = { RSlot::BUF, s.gx };
        } else if (l.what == EG_WHAT_PRIMAL_SLOT && (l.form == EG_FORM_GRAD || l.form == EG_FORM_DISCARD)) {   // written, never read: a scratch plane
            if (sf.role != EG_ROLE_MUTABLE) refuse(EG_BAD_REQUEST, field(gn, at) + ": a gradient or discarded slot is a MUTABLE slot");
            out = { RSlot::BUF, buf(10, l.slot, 1, RBuf::NONE, -1) };
        } else {
            refuse(EG_NOT_WRITTEN, field(gn, at) + ": (what " + std::to_string(l.what) + ", form " + std::to_string(l.form) + ") on a state guard vjp is not written in this build of the door");
        }
    }
    for (std::size_t i = 0; i < guard->slots.size(); ++i)
        if (f->gslots[i].kind == RSlot::NONE && guard->slots[i].role != EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, field(gn, "links") + ": guard slot " + std::to_string(i) + " ('" + guard->slots[i].name + "') has no link");
    for (int j = 0; j < guards; ++j)
        if (f->seed[static_cast<std::size_t>(j)] < 0) refuse(EG_BAD_REQUEST, field(gn, "links") + ": event " + std::to_string(j) + " has no (GUARD, ADJOINT_IN) seed");
    if (rvjp == nullptr) return;
    // ---- the refine vjp (a recorded run)
    const DerivedFacts& rv = rvjp->derived;
    const char* rn = rvjp->name.c_str();
    if (rvjp->finish.present) refuse(EG_BAD_REQUEST, "unit '" + rvjp->name + "': a refine vjp has no finish");
    ++rvjp->refs;
    f->rvjp_unit = rvjp;
    f->vslots.assign(rvjp->slots.size(), RSlot{});
    f->gbar.assign(static_cast<std::size_t>(guards), -1);
    const auto state_found = [&](std::int32_t primal_slot) {
        for (std::size_t k = 0; k < f->states.size(); ++k)
            if (f->states[k].primal == primal_slot) return static_cast<int>(k);
        return -1;
    };
    for (std::size_t i = 0; i < rv.links.size(); ++i) {
        const DerivedFacts::Link& l = rv.links[i];
        const std::string at = "links[" + std::to_string(i) + "]";
        if (!in_range(l.slot, rvjp->slots.size())) refuse(EG_BAD_REQUEST, field(rn, at + ".slot") + " is out of range");
        if (f->vslots[static_cast<std::size_t>(l.slot)].kind != RSlot::NONE)
            refuse(EG_BAD_REQUEST, field(rn, at + ".slot") + " " + std::to_string(l.slot) + " has a second link");
        const SlotFacts& sf = rvjp->slots[static_cast<std::size_t>(l.slot)];
        RSlot& out = f->vslots[static_cast<std::size_t>(l.slot)];
        if (common(*rvjp, l, at, out)) continue;
        const bool per_sample_real = sf.role == EG_ROLE_PER_SAMPLE && sf.dtype == EG_DTYPE_REAL;
        const bool mutable_real = sf.role == EG_ROLE_MUTABLE && sf.dtype == EG_DTYPE_REAL;
        if (l.what == EG_WHAT_STEPS) {
            if (sf.role != EG_ROLE_PER_SAMPLE || sf.dtype != EG_DTYPE_INT64) refuse(EG_BAD_REQUEST, field(rn, at) + ": a step plane is a per-sample int64 slot");
            out = { RSlot::STEPS, -1 };
        } else if (l.what == EG_WHAT_TIME0 && l.form == EG_FORM_VALUE) {
            if (sf.role != EG_ROLE_PER_SAMPLE || l.of_slot != start_of)
                refuse(EG_BAD_REQUEST, field(rn, at) + ": the start copy is a per-sample read of the twin's time plane");
            out = { RSlot::TIME0, -1 };
        } else if (l.what == EG_WHAT_TIME0) {   // (TIME0, GRAD): joins the time plane's gradient
            if (!mutable_real || l.of_slot != start_of || f->t0grad >= 0)
                refuse(EG_BAD_REQUEST, field(rn, at) + ": the start copy's gradient is one MUTABLE real slot of the twin's time plane");
            f->t0grad = buf(15, l.slot, 1, RBuf::NONE, -1);
            f->t0of = l.of_slot;
            out = { RSlot::BUF, f->t0grad };
        } else if (l.what == EG_WHAT_TIME_STEP && l.form == EG_FORM_VALUE) {
            if (!per_sample_real || !in_range(l.of_slot, np)) refuse(EG_BAD_REQUEST, field(rn, at) + ": the time step is a per-sample real slot of a primal slot");
            const std::int32_t prole = pu.slots[static_cast<std::size_t>(l.of_slot)].role;
            if (prole == EG_ROLE_UNIFORM) out = { RSlot::BUF, buf(7, 0, 1, RBuf::UNIFORM_OF, l.of_slot) };
            else if (prole == EG_ROLE_PER_SAMPLE) out = { RSlot::PLANE, l.of_slot };
            else refuse(EG_BAD_REQUEST, field(rn, at + ".of_slot") + " is neither a parameter nor a per-sample plane");
        } else if ((l.what == EG_WHAT_TIME_STEP || l.what == EG_WHAT_PRIMAL_SLOT) && l.form == EG_FORM_GRAD) {
            if (!mutable_real || !in_range(l.of_slot, np))
                refuse(EG_BAD_REQUEST, field(rn, at) + ": a gradient is a MUTABLE real slot of a primal slot");
            const std::int32_t grole = pu.slots[static_cast<std::size_t>(l.of_slot)].role;
            if (grole != EG_ROLE_UNIFORM && grole != EG_ROLE_PER_SAMPLE)
                refuse(EG_NOT_WRITTEN, field(rn, at) + ": the gradient of a slot that is neither a parameter nor a per-sample plane is not written");
            if ((grole == EG_ROLE_UNIFORM) != (l.reduced == 1))
                refuse(EG_BAD_REQUEST, field(rn, at + ".reduced") + " " + std::to_string(l.reduced) +
                       (grole == EG_ROLE_UNIFORM ? ": a parameter's gradient is summed over samples (1)" : ": a per-sample read's gradient is not summed (0)"));
            const int b = buf(15, l.slot, 1, RBuf::NONE, -1);
            (grole == EG_ROLE_UNIFORM ? f->pgrads : f->sgrads).push_back({ b, l.of_slot });
            out = { RSlot::BUF, b };
        } else if (l.what == EG_WHAT_TAU && l.form == EG_FORM_VALUE) {
            if (!per_sample_real) refuse(EG_BAD_REQUEST, field(rn, at) + ": the tau is a per-sample real slot");
            out = { RSlot::BUF, f->tau };
        } else if ((l.what == EG_WHAT_TAU || l.what == EG_WHAT_PRIMAL_SLOT) && l.form == EG_FORM_DISCARD) {
            if (!mutable_real) refuse(EG_BAD_REQUEST, field(rn, at) + ": a discarded slot is a MUTABLE real slot");
            out = { RSlot::BUF, buf(15, l.slot, 1, RBuf::NONE, -1) };
        } else if (l.what == EG_WHAT_GUARD && l.form == EG_FORM_ADJOINT_IN) {
            if (!per_sample_real || l.of_slot < 0 || l.of_slot >= guards)
                refuse(EG_BAD_REQUEST, field(rn, at) + ": a guard seed is a per-sample real slot of an event of the Newton");
            f->gbar[static_cast<std::size_t>(l.of_slot)] = buf(12, l.of_slot, 1, RBuf::NONE, -1);
            out = { RSlot::BUF, f->gbar[static_cast<std::size_t>(l.of_slot)] };
        } else if (l.what == EG_WHAT_REFINED && l.form == EG_FORM_ADJOINT_IN) {
            const int k = state_found(l.of_slot);
            if (k < 0 || !per_sample_real) refuse(EG_BAD_REQUEST, field(rn, at) + ": the end-state cotangent of primal state " + std::to_string(l.of_slot) + " is a per-sample real slot of a refined state");
            RState& st = f->states[static_cast<std::size_t>(k)];
            st.rbar = buf(13, l.of_slot, st.rows, RBuf::NONE, -1);
            out = { RSlot::BUF, st.rbar };
        } else if (l.what == EG_WHAT_PRIMAL_SLOT && l.form == EG_FORM_ADJOINT_OUT) {
            const int k = state_found(l.of_slot);
            if (k < 0 || !mutable_real) refuse(EG_BAD_REQUEST, field(rn, at) + ": the pre-stop cotangent of primal state " + std::to_string(l.of_slot) + " is a MUTABLE real slot of a refined state");
            RState& st = f->states[static_cast<std::size_t>(k)];
            st.share = buf(14, l.of_slot, st.rows, RBuf::NONE, -1);
            out = { RSlot::BUF, st.share };
        } else {
            refuse(EG_NOT_WRITTEN, field(rn, at) + ": (what " + std::to_string(l.what) + ", form " + std::to_string(l.form) + ") on a refine vjp is not written in this build of the door");
        }
    }
    for (int j = 0; j < guards; ++j)
        if (f->gbar[static_cast<std::size_t>(j)] < 0) refuse(EG_BAD_REQUEST, field(rn, "links") + ": event " + std::to_string(j) + " has no (GUARD, ADJOINT_IN) seed");
    for (const RState& st : f->states)
        if (st.rbar < 0) refuse(EG_BAD_REQUEST, field(rn, "links") + ": primal state " + std::to_string(st.primal) + " has no (REFINED, ADJOINT_IN) link");
    for (std::size_t i = 0; i < rvjp->slots.size(); ++i)
        if (f->vslots[i].kind == RSlot::NONE && rvjp->slots[i].role != EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, field(rn, "links") + ": refine vjp slot " + std::to_string(i) + " ('" + rvjp->slots[i].name + "') has no link");
}

}  // namespace

/** @brief eg_runner_new with EG_RUN_RECORD: check the derived record against the primal and keep the twin and the vjp. */
void record_new(eg_ctx& ctx, eg_runner_s& runner, const eg_runner_desc& d) {
    const bool ev = d.events == EG_EVENTS_REFINE, rec = (d.flags & EG_RUN_RECORD) != 0;
    const bool refine = ev && !rec;    // events refine without a record
    const bool rrefine = ev && rec;    // events refine, recorded: the reverse of a refined run
    if (ev && runner.active)
        refuse(EG_NOT_WRITTEN, "events REFINE with an active set: no old-path test in this build of the door");
    if (runner.active && runner.act.group != 1)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD with a grouped active set is not written in this build of the door");
    if (d.n_plans != 1)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD of a several-plan step is not written in this build of the door");
    const eg_plan_s& primal = *d.plans[0];
    const eg_unit_s& pu = *primal.unit;
    bool fused = false;
    for (std::int32_t i = 0; i < d.n_derived && d.derived_roles != nullptr; ++i) fused = fused || d.derived_roles[i] == EG_DERIVED_TAPE_FUSED;
    if (fused && ev) refuse(EG_NOT_WRITTEN, "events refine with the K-step twin is not written in this build of the door");
    if (runner.dev.ctx != nullptr) {   // a DEVICE record: the one-step run, its active set included; the rest by name
    }
    eg_unit_s* twin = derived_of(ctx, d, fused ? EG_DERIVED_TAPE_FUSED : EG_DERIVED_TAPE, ev);
    eg_unit_s* vjp = rec ? derived_of(ctx, d, EG_DERIVED_STEP_VJP, ev) : nullptr;
    eg_unit_s* newton = ev ? derived_of(ctx, d, EG_DERIVED_REFINE_NEWTON, true) : nullptr;
    eg_unit_s* guard = ev ? derived_of(ctx, d, EG_DERIVED_STATE_GUARD_VJP, true) : nullptr;
    eg_unit_s* rvjp = rrefine ? derived_of(ctx, d, EG_DERIVED_REFINE_VJP, true) : nullptr;
    const bool fusedrev = d.steps_reverse == EG_STEPS_REVERSE_FUSED;
    if (fusedrev && ev) refuse(EG_NOT_WRITTEN, "steps_reverse FUSED with events refine is not written in this build of the door");
    eg_unit_s* fvjp = fusedrev ? derived_of(ctx, d, EG_DERIVED_FUSED_VJP, ev) : nullptr;
    if (runner.dev.ctx != nullptr)
        for (const eg_unit_s* u : { twin, vjp, fvjp, newton, guard })
            if (u != nullptr && u->function == 0) refuse(EG_NOT_WRITTEN, "unit '" + u->name + "' on DEVICE: the unit has no device image");
    const std::int32_t want = refine ? 3 : rrefine ? 5 : fusedrev ? 3 : 2;
    if (d.n_derived != want)
        refuse(EG_BAD_REQUEST, "n_derived " + std::to_string(d.n_derived) + (refine ? ": a refine run takes the twin, the Newton and the state guard vjp"
                                                                                    : rrefine ? ": a recorded refine run takes the twin, the step vjp, the Newton, the state guard vjp and the refine vjp"
                                                                                    : fusedrev ? ": a recorded run with steps_reverse FUSED takes the twin, the step vjp and the K-step vjp"
                                                                                    : ": a recorded run takes the twin and the vjp"));
    const std::size_t np = pu.slots.size();
    if (runner.active != (twin->active_map_slot >= 0) || (twin->active_map_slot >= 0) != (twin->active_count_slot >= 0))
        refuse(EG_BAD_REQUEST, "unit '" + twin->name + "': the taping twin reads an active set (map and count slots) exactly "
               "when the runner's plans do");
    if (runner.active && runner.act.reorder && twin->derived.time0_slot != -1)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD with EG_RUN_REORDER with a start copy (time0_slot): no old-path oracle in this "
               "build of the door");
    if (vjp != nullptr && vjp->active_map_slot >= 0) refuse(EG_BAD_REQUEST, "unit '" + vjp->name + "': a step vjp reads no active set");
    for (const eg_unit_s* u : { twin, vjp, fvjp, newton, guard, rvjp })
        if (u != nullptr && (u->derived.of_unit != pu.index || u->library != pu.library))
            refuse(EG_BAD_REQUEST, field(u->name.c_str(), "of_unit") + " " + std::to_string(u->derived.of_unit) +
                   " is not the runner's primal (unit '" + pu.name + "', index " + std::to_string(pu.index) + ")");
    for (const eg_unit_s* u : { twin, vjp })
        if (u != nullptr && u->cell_row > 0 && u->active_count_slot < 0)
            refuse(EG_BAD_REQUEST, "unit '" + u->name + "': owns cell rows and is launched over the live prefix by the "
                   "reverse: it must be emitted in the active-set form");
    const DerivedFacts& t = twin->derived;
    const char* tn = twin->name.c_str();
    if (t.time0_slot != -1) check_slot(*twin, t.time0_slot, "time0_slot", EG_ROLE_PER_SAMPLE, -1);
    if (fused) {
        check_slot(*twin, t.fused_steps_slot, "fused_steps_slot", EG_ROLE_LOOKUP, EG_DTYPE_INT64);
        if (runner.active) refuse(EG_NOT_WRITTEN, "the K-step twin with an active set is not written in this build of the door");
        if (!twin->finish.present || (twin->finish.steps != EG_STEPS_AUTO && twin->finish.steps < 1))
            refuse(EG_BAD_REQUEST, "unit '" + twin->name + "': the K-step twin's finish names the steps one launch takes");
    } else if (t.fused_steps_slot != -1) {
        refuse(EG_BAD_REQUEST, field(tn, "fused_steps_slot") + ": not part of a one-step taping twin");
    }
    if (!fused && twin->finish.present && twin->finish.steps_slot >= 0)
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
    const std::int64_t asked = refine ? 1 : d.window == -1 ? std::max<std::int64_t>(d.max_steps, 1) : d.window;
    if (fusedrev && !(pu.finish.present && pu.finish.steps == asked))
        refuse(EG_BAD_REQUEST, "steps_reverse FUSED: the primal takes " + std::to_string(pu.finish.present ? pu.finish.steps : 0) +
               " steps per launch, the window is " + std::to_string(asked) + " (a K-step primal with window K)");
    if (rrefine && asked != 1)
        refuse(EG_BAD_REQUEST, "window " + std::to_string(d.window) + ": a recorded refine run takes window 1 only");
    if (asked != window)
        refuse(EG_BAD_REQUEST, "window " + std::to_string(d.window) + ": the taping twin '" + twin->name +
               "' records windows of " + std::to_string(window) + " steps (built in)");
    // The twin: one (PRIMAL_SLOT, VALUE) link per ordinary slot, of the primal slot's role and dtype; none on a reserved slot.
    std::vector<int> twin_of(twin->slots.size(), -1);
    for (std::size_t i = 0; i < t.links.size(); ++i) {
        const DerivedFacts::Link& l = t.links[i];
        const std::string at = "links[" + std::to_string(i) + "]";
        if (!in_range(l.slot, twin->slots.size())) refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " is out of range");
        if ((twin->finish.present && l.slot == twin->finish.counter_slot) || l.slot == twin->active_map_slot ||
            l.slot == twin->active_count_slot)
            refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " " + std::to_string(l.slot) + " is eagle's own (the finish counter, "
                   "the active map or count): it takes no link");
        if (l.what == EG_WHAT_TIME0 && l.slot != t.time0_slot)
            refuse(EG_BAD_REQUEST, field(tn, at) + ": (TIME0, VALUE) is the link of time0_slot only (a TIME0 link on slot " +
                   std::to_string(l.slot) + ")");
        if (l.slot == t.tape_slot || l.slot == t.steps_slot || l.slot == t.tape_on_slot || l.slot == t.fused_steps_slot || l.slot == twin->active_map_slot ||
            l.slot == twin->active_count_slot)
            refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " " + std::to_string(l.slot) + " is a reserved slot");
        if (twin_of[static_cast<std::size_t>(l.slot)] != -1)
            refuse(EG_BAD_REQUEST, field(tn, at + ".slot") + " " + std::to_string(l.slot) + " has a second link");
        if (!in_range(l.of_slot, np)) refuse(EG_BAD_REQUEST, field(tn, at + ".of_slot") + " is out of range");
        const SlotFacts& a = twin->slots[static_cast<std::size_t>(l.slot)];
        const SlotFacts& b = pu.slots[static_cast<std::size_t>(l.of_slot)];
        if (l.what == EG_WHAT_TIME0) {   // the start copy of a time plane: a per-sample plane of the time plane's dtype
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
        if (s == t.tape_slot || s == t.steps_slot || s == t.tape_on_slot || s == t.fused_steps_slot || role == EG_ROLE_NSAMPLES) continue;
        if (s == t.time0_slot) {
            if (twin_of[i] == -1) refuse(EG_BAD_REQUEST, field(tn, "links") + ": time0_slot " + std::to_string(s) + " has no (TIME0, VALUE) link");
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
    if (vjp != nullptr) ++vjp->refs;
    r->window = window;
    r->width = t.tape_rows;
    r->budget = static_cast<std::int64_t>(d.memory_budget);
    r->scalar = pu.scalar_type == EG_SCALAR_FLOAT32 ? 4 : 8;
    std::vector<int> taped_state_of_primal(np, -1);
    for (std::size_t i = 0; i < twin->slots.size(); ++i) {
        const int s = static_cast<int>(i);
        const std::int32_t role = twin->slots[i].role;
        if (s == t.tape_slot || s == t.tape_on_slot || s == t.fused_steps_slot || role == EG_ROLE_NSAMPLES || role == EG_ROLE_UNIFORM) continue;
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
    if (ev) refine_setup(*r, pu, newton, guard, rvjp, taped_state_of_primal, d);
    if (refine) return;
    r->recorded = rrefine;
    r->fused = fused;
    runner.steps_reverse = d.steps_reverse;
    if (fused) r->max_k = std::max<std::int64_t>(twin->finish.steps == EG_STEPS_AUTO ? twin->finish.steps_max : twin->finish.steps, 1);
    // The vjp (and the K-step vjp): exactly one link per slot.
    const auto link_vjp = [&](eg_unit_s* vjp, std::vector<VjpSlot>& out_slots, bool fz) {
    const DerivedFacts& v = vjp->derived;
    const char* vn = vjp->name.c_str();
    out_slots.assign(vjp->slots.size(), VjpSlot{});
    std::vector<int> linked(vjp->slots.size(), 0);
    for (std::size_t i = 0; i < v.links.size(); ++i) {
        const DerivedFacts::Link& l = v.links[i];
        const std::string at = "links[" + std::to_string(i) + "]";
        if (!in_range(l.slot, vjp->slots.size())) refuse(EG_BAD_REQUEST, field(vn, at + ".slot") + " is out of range");
        if (linked[static_cast<std::size_t>(l.slot)]++ != 0)
            refuse(EG_BAD_REQUEST, field(vn, at + ".slot") + " " + std::to_string(l.slot) + " has a second link");
        if (l.what == EG_WHAT_STEPS) {   // the step index j, a plane eagle fills before each step's vjp
            const SlotFacts& vs_ = vjp->slots[static_cast<std::size_t>(l.slot)];
            if (l.of_slot != t.steps_slot && l.of_slot != -1)
                refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) +
                       " is neither the twin's steps_slot nor -1");
            if (vs_.role != EG_ROLE_PER_SAMPLE || (vs_.dtype != EG_DTYPE_INT64 && vs_.dtype != EG_DTYPE_INT32))
                refuse(EG_BAD_REQUEST, field(vn, at) + ": a step plane is a per-sample int32 or int64 slot ('" + vs_.name + "')");
            if (fz) refuse(EG_NOT_WRITTEN, field(vn, at) + ": a step plane in a K-step vjp is not written in this build of the door");
            if (r->step_bytes != 0) refuse(EG_BAD_REQUEST, field(vn, at) + ": a second step plane");
            r->step_bytes = vs_.dtype == EG_DTYPE_INT64 ? 8 : 4;
            out_slots[static_cast<std::size_t>(l.slot)] = { VjpSlot::STEP, -1 };
            continue;
        }
        if (l.what == EG_WHAT_TIME0) {   // the gradient of the time plane's start copy, added into the time plane's
            const SlotFacts& vs_ = vjp->slots[static_cast<std::size_t>(l.slot)];
            if (t.time0_slot == -1)
                refuse(EG_BAD_REQUEST, field(vn, at) + ": (TIME0, GRAD) but the taping twin holds no start copy (time0_slot -1)");
            if (!in_range(l.of_slot, np) || taped_state_of_primal[static_cast<std::size_t>(l.of_slot)] < 0)
                refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is not a taped state (the time plane)");
            if (vs_.role != EG_ROLE_MUTABLE) refuse(EG_BAD_REQUEST, field(vn, at) + ": the start copy's gradient is a MUTABLE slot ('" + vs_.name + "')");
            for (const Grad& g : r->grads)
                if (g.fold >= 0) refuse(EG_BAD_REQUEST, field(vn, at) + ": a second start-copy gradient");
            Grad gr;
            gr.fold = taped_state_of_primal[static_cast<std::size_t>(l.of_slot)];
            out_slots[static_cast<std::size_t>(l.slot)] = { VjpSlot::GRAD, static_cast<int>(r->grads.size()) };
            r->grads.push_back(std::move(gr));
            continue;
        }
        if (!in_range(l.of_slot, np)) refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is out of range");
        const std::int32_t prole = pu.slots[static_cast<std::size_t>(l.of_slot)].role;
        const int tsi = taped_state_of_primal[static_cast<std::size_t>(l.of_slot)];
        int read = -1;
        for (std::size_t k = 0; k < r->reads.size(); ++k)
            if (r->reads[k].primal_slot == l.of_slot) read = static_cast<int>(k);
        VjpSlot& vs = out_slots[static_cast<std::size_t>(l.slot)];
        switch (l.what == EG_WHAT_TAPE_ROW || l.form == EG_FORM_ADJOINT_IN || l.form == EG_FORM_ADJOINT_OUT ? 1 : l.form == EG_FORM_VALUE ? 0 : 2) {
        case 0:   // (PRIMAL_SLOT, VALUE)
            if (prole == EG_ROLE_UNIFORM && vjp->slots[static_cast<std::size_t>(l.slot)].role == EG_ROLE_PER_SAMPLE) {
                if (read < 0 || r->reads[static_cast<std::size_t>(read)].kind != Carried::FILL)
                    refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) +
                           ": a plane filled with a parameter is one the taping twin reads too");
                vs = { VjpSlot::READ, read };
            } else if (prole == EG_ROLE_UNIFORM) vs = { VjpSlot::UNIFORM, l.of_slot };
            else if (fz && vjp->slots[static_cast<std::size_t>(l.slot)].role == EG_ROLE_TERMINATED)
                vs = { VjpSlot::ZERO, -1 };   // the window's entry flag: no sample has stopped yet
            else if (read >= 0) vs = { VjpSlot::READ, read };
            else refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) +
                        ": the recorded run holds no such read (a state is read from the tape)");
            break;
        case 1:   // (TAPE_ROW, VALUE), (PRIMAL_SLOT, ADJOINT_IN), (PRIMAL_SLOT, ADJOINT_OUT)
            if (tsi < 0) refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is not a taped state");
            vs = { l.what == EG_WHAT_TAPE_ROW ? (fz ? VjpSlot::WSTATE : VjpSlot::TAPED) : l.form == EG_FORM_ADJOINT_IN ? VjpSlot::ADJ_IN : VjpSlot::ADJ_OUT, tsi };
            break;
        default: {   // (PRIMAL_SLOT, GRAD)
            const bool uniform = prole == EG_ROLE_UNIFORM;
            if (!uniform && read < 0)
                refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " is neither a parameter nor a read");
            if (uniform == (l.reduced == 0))
                refuse(EG_BAD_REQUEST, field(vn, at + ".reduced") + " " + std::to_string(l.reduced) +
                       (uniform ? ": a parameter's gradient is summed over samples (1)" : ": a per-sample read's gradient is not summed (0)"));
            int g = -1;
            for (std::size_t k = 0; k < r->grads.size(); ++k)
                if (r->grads[k].primal_slot == l.of_slot) g = static_cast<int>(k);
            if (g >= 0 && !fz) refuse(EG_BAD_REQUEST, field(vn, at + ".of_slot") + " " + std::to_string(l.of_slot) + " has a second gradient");
            if (g >= 0) {   // the K-step vjp's share of a parameter the step vjp states too: the same plane and sum
                vs = { VjpSlot::GRAD, g };
                break;
            }
            Grad gr;
            gr.primal_slot = l.of_slot;
            gr.plane = !uniform;
            vs = { VjpSlot::GRAD, static_cast<int>(r->grads.size()) };
            r->grads.push_back(std::move(gr));
        }
        }
    }
    for (std::size_t i = 0; i < vjp->slots.size(); ++i)
        if (linked[i] == 0 && vjp->slots[i].role != EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, field(vn, "links") + ": vjp slot " + std::to_string(i) + " ('" + vjp->slots[i].name + "') has no link");
    if (vjp->finish.present) refuse(EG_BAD_REQUEST, "unit '" + vjp->name + "': a step vjp has no finish");
    };
    link_vjp(vjp, r->vjp_slots, false);
    if (fvjp != nullptr) {
        r->fvjp_unit = fvjp;
        ++fvjp->refs;
        r->fused_k = asked;
        link_vjp(fvjp, r->fvjp_slots, true);
    }
}

/** @brief Drop the runner's derived plans and units. */
void record_free(eg_ctx& ctx, eg_runner_s& runner) {
    RunRecord* r = runner.record;
    if (r == nullptr) return;
    inner_free(ctx, r->fwd);
    inner_free(ctx, r->rec);
    inner_free(ctx, r->vjp);
    inner_free(ctx, r->fvjp);
    if (r->fvjp_unit != nullptr) unit_release(ctx, r->fvjp_unit);
    if (r->rf != nullptr) {
        inner_free(ctx, r->rf->newton);
        inner_free(ctx, r->rf->guard);
        inner_free(ctx, r->rf->rvjp);
        if (r->rf->rvjp_unit != nullptr) unit_release(ctx, r->rf->rvjp_unit);
        if (r->rf->newton_unit != nullptr) unit_release(ctx, r->rf->newton_unit);
        if (r->rf->guard_unit != nullptr) unit_release(ctx, r->rf->guard_unit);
        delete r->rf;
    }
    if (r->twin_unit != nullptr) unit_release(ctx, r->twin_unit);
    if (r->vjp_unit != nullptr) unit_release(ctx, r->vjp_unit);
    delete r;
    runner.record = nullptr;
}


/** @brief Bind the slots of a refine unit over the runner's planes and eagle's: @p slots as refine_setup stated them. */
void bind_refine(eg_plan_s& plan, const std::vector<RSlot>& slots, const eg_plan_s& primal, RunRecord& r, RefineRun& f, std::int64_t n,
                 std::size_t es, const char* start) {
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const RSlot& s = slots[i];
        const int slot = static_cast<int>(i);
        switch (s.kind) {
        case RSlot::UNIFORM: bind_value(plan, slot, primal.bindings[static_cast<std::size_t>(s.a)].value, sizeof(primal.bindings[0].value)); break;
        case RSlot::PLANE:
            bind_raw(plan, slot, plane_bytes(*primal.bindings[static_cast<std::size_t>(s.a)].plane), static_cast<std::uint64_t>(n));
            break;
        case RSlot::TAPED:
            bind_raw(plan, slot, r.tape.data() + static_cast<std::size_t>(r.taped[static_cast<std::size_t>(s.a)].row0 * n) * es,
                     static_cast<std::uint64_t>(n));
            break;
        case RSlot::STEPS: bind_raw(plan, slot, f.nsteps.data(), static_cast<std::uint64_t>(n)); break;
        case RSlot::TIME0: bind_raw(plan, slot, const_cast<char*>(start), static_cast<std::uint64_t>(n)); break;
        case RSlot::BUF: bind_raw(plan, slot, f.bufs[static_cast<std::size_t>(s.a)].buf.data(), static_cast<std::uint64_t>(n)); break;
        case RSlot::NONE: break;
        }
    }
}

/** @brief One event operation over @p q: on the host (on @p threads threads), or queued on @p D's stream (the planes are device addresses). */
template <class T>
void apply_event(eg_ctx* D, const events::EventPlanes<T>& q, std::uint32_t op, int threads) {
    if (D != nullptr) device_event_op(*D, static_cast<unsigned>(sizeof(T)), op, &q);
    else events::applyHost(q, op, threads);
}

/** @brief events refine after the twin's forward (``RefinePass.run`` and ``events_refine_host``): locate each stop inside
 *  its stop step, run the guard vjp, test grazing, refuse by name, commit the refined states. */
template <class T>
void refine_pass(eg_ctx& ctx, eg_runner_s& runner, RunRecord& r, int threads) {
    RefineRun& f = *r.rf;
    eg_plan_s& primal = *runner.plans[0];
    const std::int64_t n = primal.n_samples;
    const std::size_t es = sizeof(T);
    eg_ctx* D = primal.structure == EG_STRUCTURE_DEVICE ? &ctx : nullptr;   // a DEVICE refine: eagle's planes and every round on the device
    /// Host bytes of a device buffer (rounded up to whole words), or the buffer itself on the host.
    const auto host_of = [&](Buf& b, std::size_t bytes, std::vector<char>& tmp) -> const char* {
        if (D == nullptr) return b.data();
        tmp.assign((bytes + 3) / 4 * 4, 0);
        device_read(*D, b.data(), tmp.data(), tmp.size());
        return tmp.data();
    };
    std::int64_t max_rows = 1;
    for (RBuf& b : f.bufs) {
        b.buf.resize(static_cast<std::size_t>(b.rows * n) * es, D);
        if (b.fill != RBuf::NONE) {
            std::vector<T> fill(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) {
                if (b.fill == RBuf::ONES) fill[static_cast<std::size_t>(i)] = T(1);
                else std::memcpy(&fill[static_cast<std::size_t>(i)], primal.bindings[static_cast<std::size_t>(b.of)].value, es);
            }
            if (D != nullptr) device_write(*D, b.buf.data(), fill.data(), fill.size() * es);
            else std::memcpy(b.buf.data(), fill.data(), fill.size() * es);
        }
        max_rows = std::max(max_rows, b.rows);
    }
    f.zero.resize(static_cast<std::size_t>(max_rows * n) * es, D);
    f.nsteps.resize(static_cast<std::size_t>(n) * 8, D);
    f.status.resize(static_cast<std::size_t>(n), D);
    f.which.resize(static_cast<std::size_t>(n), D);
    f.lo.resize(static_cast<std::size_t>(n) * es, D);
    f.hi.resize(static_cast<std::size_t>(n) * es, D);
    f.tstop.resize(static_cast<std::size_t>(n) * es, D);
    f.done = false;
    std::vector<char> tmp;
    const std::int64_t* applied = reinterpret_cast<const std::int64_t*>(r.steps.data());
    {   // the steps before each sample's stop step
        const std::int64_t* seen = reinterpret_cast<const std::int64_t*>(host_of(r.steps, static_cast<std::size_t>(n) * 8, tmp));
        std::vector<std::int64_t> before(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) before[static_cast<std::size_t>(i)] = std::max<std::int64_t>(seen[i] - 1, 0);
        if (D != nullptr) device_write(*D, f.nsteps.data(), before.data(), before.size() * 8);
        else std::memcpy(f.nsteps.data(), before.data(), before.size() * 8);
    }
    const T* h = f.h >= 0 ? reinterpret_cast<const T*>(f.bufs[static_cast<std::size_t>(f.h)].buf.data())
                          : reinterpret_cast<const T*>(plane_bytes(*primal.bindings[static_cast<std::size_t>(f.h_primal)].plane));
    T* tau = reinterpret_cast<T*>(f.bufs[static_cast<std::size_t>(f.tau)].buf.data());
    copy_bytes(D, tau, h, static_cast<std::size_t>(n) * es);   // the first evaluation is the step's end
    const char* start = nullptr;
    for (const Carried& c : r.reads)
        if (c.kind == Carried::START) start = c.src;
    if (f.newton == nullptr) f.newton = inner_plan(ctx, f.newton_unit, primal.structure, n);
    if (f.guard == nullptr) f.guard = inner_plan(ctx, f.guard_unit, primal.structure, n);
    bind_refine(*f.newton, f.nslots, primal, r, f, n, es, start);
    bind_refine(*f.guard, f.gslots, primal, r, f, n, es, start);
    events::EventPlanes<T> p;
    p.n = n;
    p.guards = static_cast<int>(f.g.size());
    for (std::size_t j = 0; j < f.g.size(); ++j) {
        p.g[j] = reinterpret_cast<const T*>(f.bufs[static_cast<std::size_t>(f.g[j])].buf.data());
        p.dg[j] = reinterpret_cast<const T*>(f.bufs[static_cast<std::size_t>(f.dg[j])].buf.data());
    }
    const eg_plane_s& mask = *primal.bindings[static_cast<std::size_t>(primal.unit->finish.mask_slot)].plane;
    p.terminated = reinterpret_cast<const std::uint8_t*>(plane_bytes(mask));
    p.applied = applied;
    p.t0 = reinterpret_cast<const T*>(start);
    p.h = h;
    p.status = reinterpret_cast<std::int8_t*>(f.status.data());
    p.which = reinterpret_cast<std::int8_t*>(f.which.data());
    p.tau = tau;
    p.lo = reinterpret_cast<T*>(f.lo.data());
    p.hi = reinterpret_cast<T*>(f.hi.data());
    p.tStop = reinterpret_cast<T*>(f.tstop.data());
    const auto apply = [&](const events::EventPlanes<T>& q, std::uint32_t op) { apply_event(D, q, op, threads); };
    const auto evaluate = [&] { run_on(*f.newton, threads, n, n); };
    evaluate();
    apply(p, events::kOpClassify);
    // A device run enqueues every round (no read-back: a located sample's update is a no-op), as the old path's.
    events::refineLoop(evaluate, [&](std::uint32_t op) { apply(p, op); }, [&] { return D != nullptr || events::anyPending(p); });
    if (!f.seed.empty()) {
        const char* which = host_of(f.which, static_cast<std::size_t>(n), tmp);
        for (std::size_t j = 0; j < f.seed.size(); ++j) {
            std::vector<T> seed(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) seed[static_cast<std::size_t>(i)] = which[i] == static_cast<char>(j) ? T(1) : T(0);
            Buf& dst = f.bufs[static_cast<std::size_t>(f.seed[j])].buf;
            if (D != nullptr) device_write(*D, dst.data(), seed.data(), seed.size() * es);
            else std::memcpy(dst.data(), seed.data(), seed.size() * es);
        }
    }
    run_on(*f.guard, threads, n, n);
    // The rows of every refined state, in the order of f.states.
    events::EventPlanes<T> q = p, c = p;
    int row = 0;
    for (const RState& s : f.states) {
        const eg_plane_s& sp = *primal.bindings[static_cast<std::size_t>(s.primal)].plane;
        if (elements(sp.tensor->dl_tensor) != s.rows * n)
            refuse(EG_BAD_REQUEST, "events refine: primal state " + std::to_string(s.primal) + " holds " +
                   std::to_string(elements(sp.tensor->dl_tensor)) + " elements, " + std::to_string(s.rows * n) + " expected");
        for (std::int64_t k = 0; k < s.rows; ++k, ++row) {
            const std::size_t off = static_cast<std::size_t>(k * n) * es;
            q.b[row] = reinterpret_cast<const T*>((s.gx >= 0 ? f.bufs[static_cast<std::size_t>(s.gx)].buf.data() : f.zero.data()) + off);
            q.c[row] = reinterpret_cast<const T*>(f.bufs[static_cast<std::size_t>(s.dot)].buf.data() + off);
            c.a[row] = reinterpret_cast<T*>(plane_bytes(sp) + off);
            c.b[row] = reinterpret_cast<const T*>(f.bufs[static_cast<std::size_t>(s.refined)].buf.data() + off);
        }
    }
    q.components = c.components = row;
    apply(q, events::kOpGraze);
    std::vector<char> status_tmp;
    const std::int8_t* status = reinterpret_cast<const std::int8_t*>(host_of(f.status, static_cast<std::size_t>(n), status_tmp));
    const struct { std::int8_t code; const char* text; } order[] = {
        { events::kTwoRules, "two stop rules cross in the same step" },
        { events::kGraze, "samples graze the surface (|dg/dtau| below the graze tolerance); the event time is not differentiable" },
        { events::kNoEvent, "samples have no crossing in their stop step; they hit max_steps or jumped the surface (EG_NO_EVENT_ZERO takes a zero event time for them)" } };
    for (const auto& o : order) {
        std::int64_t count = 0, first = -1;
        for (std::int64_t i = 0; i < n; ++i)
            if (status[i] == o.code) {
                if (first < 0) first = i;
                ++count;
            }
        if (count == 0 || (o.code == events::kNoEvent && f.on_no_event == EG_NO_EVENT_ZERO)) continue;
        refuse(EG_BAD_REQUEST, std::string("events refine: ") + (o.code == events::kTwoRules ? "" : std::to_string(count) + " ") + o.text +
               " (first: sample " + std::to_string(first) + ")");
    }
    apply(c, events::kOpCommit);
    if (D != nullptr) device_sync(*D);   // the committed states land before eg_run returns
    f.done = true;
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
    eg_ctx* D = primal.structure == EG_STRUCTURE_DEVICE ? &ctx : nullptr;   // a DEVICE record: eagle's planes on the device
    r.steps.resize(static_cast<std::size_t>(n) * 8, D);
    r.fword.resize(8, D);
    r.dummy.resize(static_cast<std::size_t>(r.width * std::max<std::int64_t>(n, 1)) * r.scalar, D);
    const bool refine = r.rf != nullptr;
    if (refine) r.tape.resize(static_cast<std::size_t>(r.width * std::max<std::int64_t>(n, 1)) * r.scalar, D);
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
            Buf fill;
            fill.resize(static_cast<std::size_t>(n) * c.bytes);
            for (std::int64_t i = 0; i < n; ++i)
                std::memcpy(fill.data() + static_cast<std::size_t>(i) * c.bytes, primal.bindings[static_cast<std::size_t>(c.primal_slot)].value, c.bytes);
            if (D != nullptr) {
                c.own.resize(static_cast<std::size_t>(n) * c.bytes, D);
                device_write(*D, c.own.data(), fill.data(), static_cast<std::size_t>(n) * c.bytes);
            } else {
                c.own = std::move(fill);
            }
            c.src = c.own.data();
            return;
        }
        if (c.kind == Carried::START) {   // the time plane as it is at run start
            const eg_plane_s& p = *primal.bindings[static_cast<std::size_t>(c.of)].plane;
            c.bytes = elem_bytes(p.tensor->dl_tensor);
            c.rows = 1;
            c.own.resize(static_cast<std::size_t>(n) * c.bytes, D);
            copy_bytes(D, c.own.data(), plane_bytes(p), static_cast<std::size_t>(n) * c.bytes);
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
        else if (s == t.tape_slot) bind_raw(fwd, s, refine ? r.tape.data() : r.dummy.data(), static_cast<std::uint64_t>(n));
        else if (s == t.tape_on_slot) bind_int64(fwd, s, refine ? 1 : 0);
        else if (r.fused && s == t.fused_steps_slot) bind_raw(fwd, s, r.fword.data(), 1);
        else if (s == r.twin_unit->active_map_slot)
            bind_raw(fwd, s, D != nullptr ? reinterpret_cast<void*>(runner.dev.map) : static_cast<void*>(runner.act.map.data()), static_cast<std::uint64_t>(n));
        else if (s == r.twin_unit->active_count_slot)
            bind_raw(fwd, s, D != nullptr ? reinterpret_cast<void*>(runner.dev.cells + 3 * sizeof(unsigned)) : static_cast<void*>(&runner.act.count), 1);
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
    // A reordering active set moves eagle's own per-slot planes (the steps, the start copy) with the runner's.
    std::vector<Owned> owned;
    const bool reordering = runner.active && runner.act.reorder;
    if (reordering) {
        owned = owned_planes(runner);
        owned.push_back({ r.steps.data(), 8 });
        for (Carried& c : r.reads)
            if (c.kind == Carried::START) owned.push_back({ c.own.data(), c.bytes });
    }
    r.epochs.clear();
    // The finished count follows the mask (the plain loop's re-sync), counted by the twin from here.
    const FinishFacts& f = primal.unit->finish;
    const DLTensor& mt = primal.bindings[static_cast<std::size_t>(f.mask_slot)].plane->tensor->dl_tensor;
    const char* mask = static_cast<const char*>(mt.data) + mt.byte_offset;
    const std::size_t mb = elem_bytes(mt);
    std::uint32_t set = 0;
    if (D != nullptr) device_mask_count(*D, mask, n, static_cast<unsigned>(mb), fwd.device_counter);
    for (std::int64_t i = 0; D == nullptr && i < n; ++i)
        for (std::size_t b = 0; b < mb; ++b)
            if (mask[static_cast<std::size_t>(i) * mb + b] != 0) {
                ++set;
                break;
            }
    fwd.finished = set;
    ActiveState& act = runner.act;
    const std::int64_t compactions0 = act.compactions, reorders0 = act.reorders;
    if (runner.active) {   // the map follows the mask from the first launch (the plain loop's compaction)
        if (D != nullptr) {
            device_active_compact(*D, runner, mask, owned, false);
        } else {
            active_compact(act, mask, mb);
            act.done_at_last = set;
        }
    }
    r.steps.zero();
    r.taken = 0;
    const int threads = host::team_threads(ctx);
    std::int64_t launches = 0, at = 0;
    const auto done = [&] {
        if (D == nullptr) return static_cast<std::int64_t>(fwd.finished) >= n;
        // The old device record reads the counter back once per window: a refine run's checkpoint window is 1, its read-back one the plain run's max_steps,
        // or the recorded run's default window.
        const std::int64_t every = refine ? (r.recorded ? diff::defaultWindow(runner.max_steps) : runner.max_steps) : r.window;
        if (at % every != 0) return false;
        std::uint32_t finished = 0;
        device_read(*D, reinterpret_cast<const void*>(fwd.device_counter), &finished, sizeof(finished));
        return static_cast<std::int64_t>(finished) >= n;
    };
    const std::int64_t ran = diff::recordForward(
        runner.max_steps, refine && !r.recorded ? runner.max_steps : r.window, r.max_k, done,
        [&](std::int64_t k) {
            if (refine && !r.recorded) return;   // a plain refine run keeps no checkpoints: the one-row tape holds each stop step's start
            if (k == static_cast<std::int64_t>(r.pool.size())) {
                r.pool.emplace_back(r.state.size());
                for (std::size_t i = 0; i < r.state.size(); ++i)
                    r.pool.back()[i].resize(static_cast<std::size_t>(r.state[i].rows * n) * r.state[i].bytes, D);
            }
            for (std::size_t i = 0; i < r.state.size(); ++i)
                copy_bytes(D, r.pool[static_cast<std::size_t>(k)][i].data(), r.state[i].src,
                           static_cast<std::size_t>(r.state[i].rows * n) * r.state[i].bytes);
            if (reordering) {
                if (k == static_cast<std::int64_t>(r.epochs.size())) r.epochs.emplace_back();
                std::vector<std::int32_t>& inv = r.epochs[static_cast<std::size_t>(k)];
                if (D != nullptr) {   // the device's sample -> slot map, read back at the checkpoint
                    inv.resize(static_cast<std::size_t>(n));
                    device_read(*D, reinterpret_cast<const void*>(runner.dev.inv), inv.data(), static_cast<std::size_t>(n) * sizeof(std::int32_t));
                } else {
                    inv = runner.act.inv;
                }
            }
            r.taken = k + 1;
        },
        [&](std::int64_t j, std::int64_t k) {
            const std::int64_t live = runner.active && D == nullptr ? static_cast<std::int64_t>(act.count) : n;   // a device launch covers all n: the kernel reads the count itself
            if (r.fused) set_word(D, r.fword, k);
            ++launches;
            if (live > 0) run_on(fwd, threads, live, n);
            at = j + k;
            if (runner.active && (j + k) % act.every_steps == 0) {
                ++act.compactions;
                if (D != nullptr) {
                    device_active_compact(*D, runner, mask, owned, reordering);
                } else {
                    active_compact(act, mask, mb);
                    act.done_at_last = fwd.finished;
                    if (reordering && act.fire) active_reorder(act, owned);
                }
            }
        });
    if (reordering && act.reorders != act.restored_at) {   // the planes land in sample order
        if (D != nullptr) device_active_restore(*D, runner, mask, owned);
        else active_restore(act, owned);
    }
    if (D != nullptr) {   // the twin counted on the device: the primal's counter follows
        device_read(*D, reinterpret_cast<const void*>(fwd.device_counter), &fwd.finished, sizeof(fwd.finished));
        copy_bytes(D, reinterpret_cast<void*>(primal.device_counter), reinterpret_cast<const void*>(fwd.device_counter), sizeof(fwd.finished));
    }
    primal.finished = fwd.finished;
    if (refine) {
        if (r.scalar == 8) refine_pass<double>(ctx, runner, r, threads);
        else refine_pass<float>(ctx, runner, r, threads);
    }
    if (!refine || r.recorded) r.live = true;
    if (report == nullptr) return;
    eg_run_report out{};
    out.flags = (static_cast<std::int64_t>(fwd.finished) == n ? EG_REPORT_DONE : 0u) | EG_REPORT_EXACT_STEPS;
    out.mode = D != nullptr ? EG_MODE_BAND : EG_MODE_HOST;
    out.n = n;
    out.finished = fwd.finished;
    out.launches = launches;
    out.steps = ran;
    out.mapped = ran * n;
    out.compactions = act.compactions - compactions0;
    out.reorders = act.reorders - reorders0;
    write_report(report, out);
}

namespace {

/** @brief One bound caller plane of a reverse, checked against what it binds. */
char* caller_plane(const eg_ctx& ctx, bool dev, eg_plane_s* p, const DLTensor& like, std::int64_t elements_, const std::string& what) {
    check_handle(ctx.planes, p, what.c_str());
    const DLTensor& t = p->tensor->dl_tensor;
    if (t.dtype.code != like.dtype.code || t.dtype.bits != like.dtype.bits || t.dtype.lanes != like.dtype.lanes)
        refuse(EG_BAD_REQUEST, what + ": a dtype mismatch with the slot it binds");
    if (elements(t) != elements_ || !compact(t))
        refuse(EG_BAD_REQUEST, what + ": holds " + std::to_string(elements(t)) + " elements, compact " +
               std::to_string(elements_) + " expected");
    if (t.device.device_type != (dev ? kDLCUDA : kDLCPU))
        refuse(EG_BAD_REQUEST, what + (dev ? ": a device reverse takes device planes" : ": a host reverse takes host planes"));
    return static_cast<char*>(t.data) + t.byte_offset;
}

/** @brief dst[r*n + cols[s]] = src[r*m + s] for s < count (put: on the host, or queued on @p dev with @p cols on the device). */
void put_columns(char* dst, std::int64_t n, const char* src, std::int64_t m, const std::int64_t* cols, std::int64_t count,
                 std::int64_t rows, std::size_t bytes) {
    for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t s = 0; s < count; ++s)
            std::memcpy(dst + static_cast<std::size_t>(r * n + cols[s]) * bytes, src + static_cast<std::size_t>(r * m + s) * bytes, bytes);
}

void put(eg_ctx* dev, char* dst, std::int64_t n, const char* src, std::int64_t m, const std::int64_t* cols, std::int64_t count,
         std::int64_t rows, std::size_t bytes) {
    if (dev != nullptr) device_columns(*dev, 3, dst, n, src, m, cols, count, rows, static_cast<unsigned>(bytes), 0);
    else put_columns(dst, n, src, m, cols, count, rows, bytes);
}

template <class T>
void accumulate(Grad& g, std::int64_t m, std::int64_t count, int threads) {
    if (g.acc.dev) {   // a DEVICE record: one queued accumulate over the rows
        device_columns(*g.acc.ctx, 1, g.acc.data(), m, g.grad.data(), m, nullptr, count, g.rows, sizeof(T), 0);
        return;
    }
    for (std::int64_t r = 0; r < g.rows; ++r)
        diff::accumulatePrefix(reinterpret_cast<T*>(g.acc.data()) + r * m, reinterpret_cast<const T*>(g.grad.data()) + r * m, count, threads);
}

template <class T>
struct SampleColumn {
    const T* p;
    T operator()(std::int64_t i) const { return p[i]; }
};

/** @brief The canonical compensated pair of @p x[0, count) (``reduce_canonical_host`` with pair), on @p threads threads. */
template <class T>
reduce::canonical::Partial<T, aether::SumOp<T>> pair_sum(const T* x, std::int64_t count, int threads) {
    return reduce::canonical::partialRange<T, aether::SumOp<T>>(SampleColumn<T>{ x }, 0, count, {}, threads);
}

/** @brief What a recorded refine run's reverse adds to the step-by-step one (``RefineReverse.seeds`` and ``_add_refine``). */
struct RefineExtra {
    std::vector<Buf> pre;                          ///< per taped state: the cotangent the step reverse starts from
    std::vector<std::int64_t> steps;               ///< the steps it runs per sample
    std::vector<std::pair<int, double>> params;    ///< (primal slot, the refine vjp's one canonical sum of that parameter's gradient)
    std::vector<std::pair<int, Buf>> planes;       ///< (primal per-sample slot, the refine vjp's per-sample gradient of that read, in sample order)
    Buf t0;                                        ///< the start copy's per-sample gradient
    bool has_t0 = false;
};

/** @brief The refine vjp of a recorded refine run: seed it with the end-state cotangent and the implicit-function guard seed,
 *  launch it once, and select the cotangent the step reverse starts from (events.py ``RefineReverse.seeds``). */
template <class T>
void refine_reverse(eg_ctx& ctx, eg_runner_s& runner, RunRecord& r, const std::vector<const char*>& seed, int threads, RefineExtra& out) {
    RefineRun& f = *r.rf;
    eg_plan_s& primal = *runner.plans[0];
    const std::int64_t n = primal.n_samples;
    const std::size_t es = sizeof(T);
    eg_ctx* D = primal.structure == EG_STRUCTURE_DEVICE ? &ctx : nullptr;   // a DEVICE record: the refine vjp and its planes on the device
    const auto buf_of = [&](int b) { return f.bufs[static_cast<std::size_t>(b)].buf.data(); };
    std::vector<Buf> xbar(r.taped.size());
    out.pre.resize(r.taped.size());
    for (std::size_t k = 0; k < r.taped.size(); ++k) {
        const std::size_t bytes = static_cast<std::size_t>(r.taped[k].rows * n) * es;
        xbar[k].resize(bytes, D);
        if (seed[k] != nullptr) copy_bytes(D, xbar[k].data(), seed[k], bytes);
        out.pre[k].resize(bytes, D);
        copy_bytes(D, out.pre[k].data(), xbar[k].data(), bytes);
    }
    events::EventPlanes<T> q;
    q.n = n;
    q.guards = static_cast<int>(f.g.size());
    for (std::size_t j = 0; j < f.g.size(); ++j) {
        q.g[j] = q.dg[j] = reinterpret_cast<const T*>(buf_of(f.dg[j]));
        q.gbar[j] = reinterpret_cast<T*>(buf_of(f.gbar[j]));
    }
    q.status = reinterpret_cast<std::int8_t*>(f.status.data());
    q.which = reinterpret_cast<std::int8_t*>(f.which.data());
    events::EventPlanes<T> sel = q;
    int row = 0;
    for (const RState& st : f.states) {
        int tk = -1;
        for (std::size_t k = 0; k < r.taped.size(); ++k)
            if (r.taped[k].primal_slot == st.primal) tk = static_cast<int>(k);
        if (tk < 0) refuse(EG_BAD_REQUEST, "eg_reverse: refined primal state " + std::to_string(st.primal) + " is not taped by the twin");
        for (std::int64_t k = 0; k < st.rows; ++k, ++row) {
            const std::size_t off = static_cast<std::size_t>(k * n) * es;
            q.a[row] = reinterpret_cast<T*>(buf_of(st.rbar) + off);
            q.b[row] = reinterpret_cast<const T*>(xbar[static_cast<std::size_t>(tk)].data() + off);
            q.c[row] = reinterpret_cast<const T*>(buf_of(st.dot) + off);
            sel.a[row] = reinterpret_cast<T*>(out.pre[static_cast<std::size_t>(tk)].data() + off);
            sel.b[row] = q.b[row];
            sel.c[row] = reinterpret_cast<const T*>((st.share >= 0 ? buf_of(st.share) : f.zero.data()) + off);
        }
    }
    q.components = sel.components = row;
    apply_event(D, q, events::kOpSeed, threads);
    if (f.rvjp == nullptr) f.rvjp = inner_plan(ctx, f.rvjp_unit, primal.structure, n);
    const char* start = nullptr;
    for (const Carried& c : r.reads)
        if (c.kind == Carried::START) start = c.src;
    bind_refine(*f.rvjp, f.vslots, primal, r, f, n, es, start);
    run_on(*f.rvjp, threads, n, n);
    apply_event(D, sel, events::kOpSelect, threads);
    std::vector<char> steps_host, status_host;   // a device run: the steps and statuses read back for the stop order
    const std::int64_t* steps_at = reinterpret_cast<const std::int64_t*>(r.steps.data());
    const std::int8_t* status_at = q.status;
    if (D != nullptr) {
        steps_host.resize(static_cast<std::size_t>(std::max<std::int64_t>(n, 1)) * 8);
        status_host.assign(static_cast<std::size_t>((n + 3) / 4 * 4), 0);
        device_read(*D, r.steps.data(), steps_host.data(), static_cast<std::size_t>(n) * 8);
        device_read(*D, f.status.data(), status_host.data(), status_host.size());
        steps_at = reinterpret_cast<const std::int64_t*>(steps_host.data());
        status_at = reinterpret_cast<const std::int8_t*>(status_host.data());
    }
    out.steps.assign(steps_at, steps_at + n);
    events::reverseSteps(out.steps.data(), status_at, n);
    for (const auto& [b, slot] : f.pgrads) {   // one canonical sum in sample order (pair 0)
        std::vector<reduce::canonical::Partial<T, aether::SumOp<T>>> one;
        if (D != nullptr) {
            Buf hc;
            hc.resize(2 * es, D);
            device_pair_sum(*D, es == 8, buf_of(b), n, hc.data(), hc.data() + es);
            T hv[2] = { T(0), T(0) };
            device_read(*D, hc.data(), hv, sizeof(hv));
            one.push_back({ hv[0], hv[1] });
        } else {
            one.push_back(pair_sum(reinterpret_cast<const T*>(buf_of(b)), n, threads));
        }
        out.params.push_back({ slot, static_cast<double>(reduce::canonical::combine(std::move(one), threads).value()) });
    }
    for (const auto& [b, slot] : f.sgrads) {   // a per-sample read's gradient: the refine vjp's plane, joined to the step reverse's
        Buf share;
        share.resize(static_cast<std::size_t>(n) * es, D);
        copy_bytes(D, share.data(), buf_of(b), static_cast<std::size_t>(n) * es);
        out.planes.push_back({ slot, std::move(share) });
    }
    if (f.t0grad >= 0) {
        out.t0.resize(static_cast<std::size_t>(n) * es, D);
        copy_bytes(D, out.t0.data(), buf_of(f.t0grad), static_cast<std::size_t>(n) * es);
        out.has_t0 = true;
    }
}


/** @brief The host memory a pass may use by default: what the system reports available (Linux: free pages; macOS: free and
 *  inactive pages); a system with no such query refuses by name, the caller states ``memory_budget``. */
std::int64_t host_available_bytes() {
#if defined(__linux__)
    return static_cast<std::int64_t>(::sysconf(_SC_AVPHYS_PAGES) * ::sysconf(_SC_PAGE_SIZE));
#elif defined(__APPLE__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS)
        refuse(EG_FAULT, "memory_budget 0: host_statistics64 failed; state memory_budget");
    return static_cast<std::int64_t>(vm.free_count + vm.inactive_count) * ::sysconf(_SC_PAGE_SIZE);
#else
    refuse(EG_BAD_REQUEST, "memory_budget 0: this system has no available-memory query; state memory_budget");
#endif
}
}  // namespace

double sample_sum(const double* x, std::int64_t count, int threads) { return pair_sum(x, count, threads).value(); }
float sample_sum(const float* x, std::int64_t count, int threads) { return pair_sum(x, count, threads).value(); }

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
    eg_ctx* D = primal.structure == EG_STRUCTURE_DEVICE ? &ctx : nullptr;   // a DEVICE record: its planes on the device
    const bool on_dev = D != nullptr;
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
        seed[static_cast<std::size_t>(k)] = caller_plane(ctx, on_dev, seeds.planes[i], primal.bindings[static_cast<std::size_t>(s)].plane->tensor->dl_tensor,
                                                         st.rows * n, "seeds.planes[" + std::to_string(i) + "]");
    }
    struct Out { char* data; int taped, grad, slot; bool plane; };
    const auto refine_param = [&](std::int32_t s) {
        if (r.rf != nullptr && r.recorded)
            for (const auto& pg : r.rf->pgrads)
                if (pg.second == s) return true;
        return false;
    };
    const auto refine_plane = [&](std::int32_t s) {
        if (r.rf != nullptr && r.recorded)
            for (const auto& pg : r.rf->sgrads)
                if (pg.second == s) return true;
        return false;
    };
    std::vector<Out> outs;
    for (int i = 0; i < grads.n_planes; ++i) {
        const std::int32_t s = grads.plane_slots[i];
        slot_ok(s, "grads", i);
        const std::string what = "grads.planes[" + std::to_string(i) + "]";
        const int k = taped_of(s);
        int g = -1;
        for (std::size_t j = 0; j < r.grads.size(); ++j)
            if (r.grads[j].primal_slot == s) g = static_cast<int>(j);
        const bool plane = (g >= 0 && r.grads[static_cast<std::size_t>(g)].plane) || (g < 0 && refine_plane(s));
        if (k < 0 && g < 0 && !refine_param(s) && !plane)
            refuse(EG_BAD_REQUEST, "grads.plane_slots[" + std::to_string(i) + "] " + std::to_string(s) + " ('" + pu.slots[static_cast<std::size_t>(s)].name +
                   "'): a slot with no link");
        if (k >= 0) {
            const Carried& st = r.state[static_cast<std::size_t>(r.taped[static_cast<std::size_t>(k)].state)];
            outs.push_back({ caller_plane(ctx, on_dev, grads.planes[i], primal.bindings[static_cast<std::size_t>(s)].plane->tensor->dl_tensor, st.rows * n, what), k, -1, s, false });
        } else if (plane) {   // a per-sample read: one gradient per sample, unsummed
            outs.push_back({ caller_plane(ctx, on_dev, grads.planes[i], primal.bindings[static_cast<std::size_t>(s)].plane->tensor->dl_tensor, n, what), -1, g, s, true });
        } else {   // a parameter: the sum over samples, one value
            DLTensor like{};
            like.dtype = DLDataType{ kDLFloat, static_cast<std::uint8_t>(r.scalar * 8), 1 };
            outs.push_back({ caller_plane(ctx, on_dev, grads.planes[i], like, 1, what), -1, g, s, false });
        }
    }
    RefineExtra extra;
    const int refine_threads = host::team_threads(ctx);
    if (r.rf != nullptr && r.recorded) {   // the refine vjp's seeds and share; the step reverse starts from the selected cotangent
        if (r.scalar == 8) refine_reverse<double>(ctx, *runner, r, seed, refine_threads, extra);
        else refine_reverse<float>(ctx, *runner, r, seed, refine_threads, extra);
        for (std::size_t k = 0; k < r.taped.size(); ++k) seed[k] = extra.pre[k].data();
    }
    // Samples per pass under the budget (each pass a whole number of samples; leaf 1 under the compensated sums).
    std::int64_t per = 0;
    for (const Carried& c : r.state) per += c.rows * static_cast<std::int64_t>(c.bytes);
    const std::int64_t windows = (runner->max_steps + W - 1) / W;
    per = windows * per + W * r.width * static_cast<std::int64_t>(r.scalar);
    std::int64_t budget = r.budget;
    if (budget == 0) budget = D != nullptr ? device_free_bytes(*D) / 2 : host_available_bytes() / 2;
    const std::int64_t m = n > 0 ? diff::chunkSamples(budget, per, n, 1) : 0;
    if (n > 0 && m == 0)
        refuse(EG_BAD_REQUEST, "memory_budget " + std::to_string(budget) + ": one sample alone needs " + std::to_string(per) +
               " bytes (its checkpoints and a " + std::to_string(W) + "-step tape)");
    for (Carried& c : r.state) c.buf.resize(static_cast<std::size_t>(c.rows * m) * c.bytes, D);
    for (Carried& c : r.reads) c.buf.resize(static_cast<std::size_t>(c.rows * m) * c.bytes, D);
    for (TapedState& x : r.taped)
        for (Buf& a : x.adj) a.resize(static_cast<std::size_t>(x.rows * m) * r.scalar, D);
    for (Grad& g : r.grads) {
        g.grad.resize(static_cast<std::size_t>(g.rows * m) * r.scalar, D);
        g.acc.resize(static_cast<std::size_t>(g.rows * m) * r.scalar, D);
    }
    Buf step_plane;
    step_plane.resize(static_cast<std::size_t>(m) * r.step_bytes, D);
    Buf tape, lookup;
    tape.resize(static_cast<std::size_t>(W * r.width * std::max<std::int64_t>(m, 1)) * r.scalar, D);
    // The recompute twin (tape on) and the vjp over the pass's planes.
    inner_free(ctx, r.rec);
    inner_free(ctx, r.vjp);
    inner_free(ctx, r.fvjp);
    r.rec = inner_plan(ctx, r.twin_unit, primal.structure, m);
    r.vjp = inner_plan(ctx, r.vjp_unit, primal.structure, m);
    if (r.fvjp_unit != nullptr) r.fvjp = inner_plan(ctx, r.fvjp_unit, primal.structure, m);
    eg_plan_s& rec = *r.rec;
    eg_plan_s& vjp = *r.vjp;
    const DerivedFacts& t = r.twin_unit->derived;
    for (Carried& c : r.state) bind_raw(rec, c.twin_slot, c.buf.data(), static_cast<std::uint64_t>(m));
    for (Carried& c : r.reads) bind_raw(rec, c.twin_slot, c.buf.data(), static_cast<std::uint64_t>(m));
    Buf rword;
    rword.resize(8, D);
    if (r.fused) bind_raw(rec, t.fused_steps_slot, rword.data(), 1);
    bind_raw(rec, t.tape_slot, tape.data(), static_cast<std::uint64_t>(m));
    bind_int64(rec, t.tape_on_slot, 1);
    // An active-set twin reads the identity map over its pass's slots (the pass's samples are in stop order), counted per launch.
    std::vector<std::int32_t> identity(static_cast<std::size_t>(std::max<std::int64_t>(m, 1)));
    for (std::size_t i = 0; i < identity.size(); ++i) identity[i] = static_cast<std::int32_t>(i);
    std::uint32_t rec_count = 0;
    Buf dmap, dcount;   // a DEVICE reverse's identity map and live-count cell
    if (r.twin_unit->active_map_slot >= 0) {
        if (D != nullptr) {
            dmap.resize(identity.size() * sizeof(std::int32_t), D);
            device_write(*D, dmap.data(), identity.data(), identity.size() * sizeof(std::int32_t));
            dcount.resize(sizeof(rec_count), D);
        }
        bind_raw(rec, r.twin_unit->active_map_slot, D != nullptr ? dmap.data() : reinterpret_cast<char*>(identity.data()), static_cast<std::uint64_t>(m));
        bind_raw(rec, r.twin_unit->active_count_slot, D != nullptr ? dcount.data() : reinterpret_cast<char*>(&rec_count), 1);
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
    Buf entry_flag;   // a K-step vjp's entry flag: no sample has stopped at its window's start
    entry_flag.resize(static_cast<std::size_t>(std::max<std::int64_t>(m, 1)) * 8, D);
    if (r.fvjp != nullptr)
        for (std::size_t i = 0; i < r.fvjp_slots.size(); ++i) {
            const VjpSlot& vs = r.fvjp_slots[i];
            const int s = static_cast<int>(i);
            if (r.fvjp_unit->slots[i].role == EG_ROLE_NSAMPLES) continue;
            const std::uint64_t cols = static_cast<std::uint64_t>(m);
            if (vs.kind == VjpSlot::UNIFORM) bind_value(*r.fvjp, s, primal.bindings[static_cast<std::size_t>(vs.index)].value, sizeof(primal.bindings[0].value));
            else if (vs.kind == VjpSlot::GRAD) bind_raw(*r.fvjp, s, r.grads[static_cast<std::size_t>(vs.index)].grad.data(), cols);
            else if (vs.kind == VjpSlot::READ) bind_raw(*r.fvjp, s, r.reads[static_cast<std::size_t>(vs.index)].buf.data(), cols);
            else if (vs.kind == VjpSlot::ZERO) bind_raw(*r.fvjp, s, entry_flag.data(), cols);
            else if (vs.kind == VjpSlot::WSTATE) bind_raw(*r.fvjp, s, r.state[static_cast<std::size_t>(r.taped[static_cast<std::size_t>(vs.index)].state)].buf.data(), cols);
        }
    const auto bind_fused = [&](std::int64_t parity) {
        for (std::size_t i = 0; i < r.fvjp_slots.size(); ++i) {
            const VjpSlot& vs = r.fvjp_slots[i];
            if (vs.kind == VjpSlot::ADJ_IN) bind_raw(*r.fvjp, static_cast<int>(i), r.taped[static_cast<std::size_t>(vs.index)].adj[parity].data(), static_cast<std::uint64_t>(m));
            else if (vs.kind == VjpSlot::ADJ_OUT) bind_raw(*r.fvjp, static_cast<int>(i), r.taped[static_cast<std::size_t>(vs.index)].adj[1 - parity].data(), static_cast<std::uint64_t>(m));
        }
    };
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
        if (r.grads[g].fold >= 0 || r.grads[g].plane) fold[g].resize(static_cast<std::size_t>(n) * r.scalar, D);
    std::vector<Buf> out_state(r.taped.size());
    for (std::size_t k = 0; k < r.taped.size(); ++k) out_state[k].resize(static_cast<std::size_t>(r.taped[k].rows * n) * r.scalar, D);
    std::vector<std::vector<std::pair<double, double>>> partials(r.grads.size());
    std::vector<std::int64_t> device_steps;   // a DEVICE record's applied steps, read back for the stop order
    if (D != nullptr && n > 0) {
        device_steps.resize(static_cast<std::size_t>(n));
        device_read(*D, r.steps.data(), device_steps.data(), static_cast<std::size_t>(n) * 8);
    }
    const std::int64_t* all_steps = !extra.steps.empty() ? extra.steps.data() : D != nullptr ? device_steps.data()
                                                                                            : reinterpret_cast<const std::int64_t*>(r.steps.data());
    Buf dcols, dorder, dslots;   // a DEVICE reverse's gather index, sample order and per-checkpoint slots
    const int threads = host::team_threads(ctx);
    for (std::int64_t s0 = 0; s0 < n; s0 += m) {
        const std::int64_t cnt = std::min(m, n - s0);
        const std::vector<diff::idx_t> order = diff::stopOrder(all_steps + s0, cnt);
        const std::vector<diff::idx_t> live = diff::liveCounts(all_steps + s0, cnt);
        std::vector<std::int64_t> cols(static_cast<std::size_t>(cnt));
        for (std::int64_t i = 0; i < cnt; ++i) cols[static_cast<std::size_t>(i)] = order[static_cast<std::size_t>(i)] + s0;
        const std::int64_t* at = cols.data();   // the gather index on the record's side
        if (D != nullptr) {
            dcols.resize(static_cast<std::size_t>(std::max<std::int64_t>(cnt, 1)) * 8, D);
            device_write(*D, dcols.data(), cols.data(), static_cast<std::size_t>(cnt) * 8);
            at = reinterpret_cast<const std::int64_t*>(dcols.data());
        }
        for (Carried& c : r.reads) gather(D, c.buf.data(), m, c.src, n, at, cnt, c.rows, c.bytes, threads);
        for (std::size_t k = 0; k < r.taped.size(); ++k)
            for (Buf& a : r.taped[k].adj) {
                a.zero();
                if (seed[k] != nullptr) gather(D, a.data(), m, seed[k], n, at, cnt, r.taped[k].rows, r.scalar, threads);
            }
        for (Grad& g : r.grads) g.acc.zero();
        if (!live.empty()) {
            diff::AdjointChain chain(static_cast<std::int64_t>(live.size()), W, r.fused_k);
            diff::reverseWindows(
                W, live,
                [&](std::int64_t k, std::int64_t count) {
                    const std::int64_t* from = at;   // the sample's column in this checkpoint: its slot then
                    std::vector<std::int64_t> slots;
                    if (!r.epochs.empty()) {
                        const std::vector<std::int32_t>& inv = r.epochs[static_cast<std::size_t>(k)];
                        slots.resize(static_cast<std::size_t>(count));
                        for (std::int64_t i = 0; i < count; ++i) slots[static_cast<std::size_t>(i)] = inv[static_cast<std::size_t>(cols[static_cast<std::size_t>(i)])];
                        from = slots.data();
                        if (D != nullptr) {   // the slots as a device gather index
                            dslots.resize(static_cast<std::size_t>(std::max<std::int64_t>(count, 1)) * 8, D);
                            device_write(*D, dslots.data(), slots.data(), static_cast<std::size_t>(count) * 8);
                            from = reinterpret_cast<const std::int64_t*>(dslots.data());
                        }
                    }
                    for (std::size_t i = 0; i < r.state.size(); ++i)
                        gather(D, r.state[i].buf.data(), m, r.pool[static_cast<std::size_t>(k)][i].data(), n, from, count,
                               r.state[i].rows, r.state[i].bytes, threads);
                },
                [&](std::int64_t c, std::int64_t len) {
                    if (chain.begin(c, len)) return;   // a full window of a FUSED reverse recomputes inside its vjp launch
                    for (std::int64_t j = c; j < c + len;) {
                        const std::int64_t k = r.fused ? std::min(r.max_k, c + len - j) : 1;   // the forward's launch shape, never across a window
                        if (r.fused) set_word(D, rword, k);
                        rec_count = static_cast<std::uint32_t>(live[static_cast<std::size_t>(j)]);
                        if (D != nullptr && r.twin_unit->active_map_slot >= 0) device_write(*D, dcount.data(), &rec_count, sizeof(rec_count));
                        run_on(rec, threads, live[static_cast<std::size_t>(j)], m);
                        j += k;
                    }
                },
                [&](std::int64_t j, std::int64_t count) {
                    if (chain.fused()) {   // the window's one launch, over everything live at its start
                        if (j != chain.top()) return;
                        const std::int64_t whole = live[static_cast<std::size_t>(chain.start)];
                        bind_fused(chain.cur);
                        run_on(*r.fvjp, threads, whole, m);
                        chain.flip();
                        for (Grad& g : r.grads) {
                            if (r.scalar == 8) accumulate<double>(g, m, whole, threads);
                            else accumulate<float>(g, m, whole, threads);
                        }
                        return;
                    }
                    bind_step(j % W, chain.cur);
                    if (D != nullptr && r.step_bytes != 0)
                        device_columns(*D, 2, step_plane.data(), m, nullptr, 0, nullptr, count, 1, static_cast<unsigned>(r.step_bytes), j);
                    else if (r.step_bytes == 8) diff::fillPrefix(reinterpret_cast<std::int64_t*>(step_plane.data()), j, count);
                    else if (r.step_bytes == 4) diff::fillPrefix(reinterpret_cast<std::int32_t*>(step_plane.data()), static_cast<std::int32_t>(j), count);
                    run_on(vjp, threads, count, m);
                    chain.flip();
                    for (Grad& g : r.grads) {
                        if (r.scalar == 8) accumulate<double>(g, m, count, threads);
                        else accumulate<float>(g, m, count, threads);
                    }
                });
        }
        for (std::size_t k = 0; k < r.taped.size(); ++k)
            put(D, out_state[k].data(), n, r.taped[k].adj[1].data(), m, at, cnt, r.taped[k].rows, r.scalar);
        for (std::size_t g = 0; g < r.grads.size(); ++g) {
            Grad& gr = r.grads[g];
            if (gr.fold >= 0 || gr.plane) {
                put(D, fold[g].data(), n, gr.acc.data(), m, at, cnt, 1, r.scalar);
                continue;
            }
            if (D != nullptr) {   // the old device reverse: scattered to sample order, the canonical pair on the device
                Buf sample, hc;
                sample.resize(static_cast<std::size_t>(std::max<std::int64_t>(cnt, 1)) * r.scalar, D);
                hc.resize(2 * r.scalar, D);
                dorder.resize(static_cast<std::size_t>(std::max<std::int64_t>(cnt, 1)) * 8, D);
                device_write(*D, dorder.data(), order.data(), static_cast<std::size_t>(cnt) * 8);
                put(D, sample.data(), cnt, gr.acc.data(), m, reinterpret_cast<const std::int64_t*>(dorder.data()), cnt, 1, r.scalar);
                device_pair_sum(*D, r.scalar == 8, sample.data(), cnt, hc.data(), hc.data() + r.scalar);
                double hd[2] = { 0.0, 0.0 };
                float hf[2] = { 0.0f, 0.0f };
                device_read(*D, hc.data(), r.scalar == 8 ? static_cast<void*>(hd) : static_cast<void*>(hf), 2 * r.scalar);
                partials[g].push_back(r.scalar == 8 ? std::pair<double, double>{ hd[0], hd[1] }
                                                    : std::pair<double, double>{ static_cast<double>(hf[0]), static_cast<double>(hf[1]) });
                continue;
            }
            Buf sample;
            sample.resize(static_cast<std::size_t>(cnt) * r.scalar);
            for (std::int64_t i = 0; i < cnt; ++i)
                std::memcpy(sample.data() + static_cast<std::size_t>(order[static_cast<std::size_t>(i)]) * r.scalar,
                            gr.acc.data() + static_cast<std::size_t>(i) * r.scalar, r.scalar);
            if (r.scalar == 8) {
                const auto p = pair_sum(reinterpret_cast<const double*>(sample.data()), cnt, threads);
                partials[g].push_back({ p.h, p.c });
            } else {
                const auto p = pair_sum(reinterpret_cast<const float*>(sample.data()), cnt, threads);
                partials[g].push_back({ static_cast<double>(p.h), static_cast<double>(p.c) });
            }
        }
    }
    for (std::size_t g = 0; g < r.grads.size(); ++g) {   // the start copy's gradient joins the time plane's
        if (r.grads[g].fold < 0) continue;
        Buf& dst = out_state[static_cast<std::size_t>(r.grads[g].fold)];
        if (D != nullptr) device_columns(*D, 1, dst.data(), n, fold[g].data(), n, nullptr, n, 1, static_cast<unsigned>(r.scalar), 0);
        else if (r.scalar == 8)
            for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<double*>(dst.data())[i] += reinterpret_cast<const double*>(fold[g].data())[i];
        else
            for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<float*>(dst.data())[i] += reinterpret_cast<const float*>(fold[g].data())[i];
    }
    if (extra.has_t0) {   // the refine vjp's start-copy gradient joins the time plane's
        int tk = -1;
        for (std::size_t k = 0; k < r.taped.size(); ++k)
            if (r.taped[k].primal_slot == r.rf->t0of) tk = static_cast<int>(k);
        if (tk >= 0 && D != nullptr)
            device_columns(*D, 1, out_state[static_cast<std::size_t>(tk)].data(), n, extra.t0.data(), n, nullptr, n, 1, static_cast<unsigned>(r.scalar), 0);
        else if (tk >= 0 && r.scalar == 8)
            for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<double*>(out_state[static_cast<std::size_t>(tk)].data())[i] += reinterpret_cast<const double*>(extra.t0.data())[i];
        else if (tk >= 0)
            for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<float*>(out_state[static_cast<std::size_t>(tk)].data())[i] += reinterpret_cast<const float*>(extra.t0.data())[i];
    }
    for (const Out& o : outs) {
        if (o.plane) {   // the step reverse's per-sample gradient (accumulated over steps), then the refine vjp's share added
            const std::size_t bytes = static_cast<std::size_t>(n) * r.scalar;
            Buf sum;
            sum.resize(bytes, D);
            if (o.grad >= 0) copy_bytes(D, sum.data(), fold[static_cast<std::size_t>(o.grad)].data(), bytes);
            else sum.zero();
            for (auto& [slot, share] : extra.planes) {
                if (slot != o.slot) continue;
                if (o.grad < 0) copy_bytes(D, sum.data(), share.data(), bytes);
                else if (D != nullptr) device_columns(*D, 1, sum.data(), n, share.data(), n, nullptr, n, 1, static_cast<unsigned>(r.scalar), 0);
                else if (r.scalar == 8)
                    for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<double*>(sum.data())[i] += reinterpret_cast<const double*>(share.data())[i];
                else
                    for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<float*>(sum.data())[i] += reinterpret_cast<const float*>(share.data())[i];
            }
            copy_bytes(D, o.data, sum.data(), bytes);
        } else if (o.taped >= 0) {
            copy_bytes(D, o.data, out_state[static_cast<std::size_t>(o.taped)].data(),
                       static_cast<std::size_t>(r.taped[static_cast<std::size_t>(o.taped)].rows * n) * r.scalar);
        } else {   // diff_combine_chunks: the passes' pairs combined in pass order
            const std::vector<std::pair<double, double>> none;
            const auto& ps = o.grad >= 0 ? partials[static_cast<std::size_t>(o.grad)] : none;
            if (r.scalar == 8) {
                std::vector<reduce::canonical::Partial<double, aether::SumOp<double>>> p;
                for (const auto& [h, c] : ps) p.push_back({ h, c });
                double v = ps.empty() ? 0.0 : reduce::canonical::combine(std::move(p), threads).value();
                for (const auto& [slot, x] : extra.params)
                    if (slot == o.slot) v += x;
                if (D != nullptr) device_write(*D, o.data, &v, sizeof(v));
                else std::memcpy(o.data, &v, sizeof(v));
            } else {
                std::vector<reduce::canonical::Partial<float, aether::SumOp<float>>> p;
                for (const auto& [h, c] : ps) p.push_back({ static_cast<float>(h), static_cast<float>(c) });
                float v = ps.empty() ? 0.0f : reduce::canonical::combine(std::move(p), threads).value();
                for (const auto& [slot, x] : extra.params)
                    if (slot == o.slot) v += static_cast<float>(x);
                if (D != nullptr) device_write(*D, o.data, &v, sizeof(v));
                else std::memcpy(o.data, &v, sizeof(v));
            }
        }
    }
    if (D != nullptr) device_sync(*D);   // the gradients land before eg_reverse returns
    r.live = false;
}

/** @brief Drop what the last run left to read: its record (eg_reverse then refuses) and its refine results. */
void record_reset(eg_runner_s& runner) {
    if (runner.record == nullptr) return;
    runner.record->live = false;
    if (runner.record->rf != nullptr) runner.record->rf->done = false;
}

/** @brief eg_run_events: the last refine run's stop times and statuses. */
void run_events(eg_ctx& ctx, eg_runner_s* runner, eg_plane_s* t_stop, eg_plane_s* status) {
    check_handle(ctx.runners, runner, "runner");
    if (runner->record == nullptr || runner->record->rf == nullptr || !runner->record->rf->done)
        refuse(EG_BAD_REQUEST, "eg_run_events: no refine run");
    RunRecord& r = *runner->record;
    RefineRun& f = *r.rf;
    const std::int64_t n = runner->plans[0]->n_samples;
    eg_ctx* D = runner->plans[0]->structure == EG_STRUCTURE_DEVICE ? &ctx : nullptr;   // a device refine: device planes out
    const auto check = [&](eg_plane_s* p, std::uint8_t code, std::uint8_t bits, const char* what) -> char* {
        if (p == nullptr) return nullptr;
        check_handle(ctx.planes, p, what);
        const DLTensor& t = p->tensor->dl_tensor;
        if (t.dtype.code != code || t.dtype.bits != bits || t.dtype.lanes != 1 || elements(t) != n || !compact(t) ||
            t.device.device_type != (D != nullptr ? kDLCUDA : kDLCPU))
            refuse(EG_BAD_REQUEST, std::string("eg_run_events: plane shape or dtype (") + what + ")");
        return static_cast<char*>(t.data) + t.byte_offset;
    };
    char* ts = check(t_stop, kDLFloat, static_cast<std::uint8_t>(r.scalar * 8), "t_stop");
    char* st = check(status, kDLInt, 8, "status");
    if (ts != nullptr) copy_bytes(D, ts, f.tstop.data(), static_cast<std::size_t>(n) * r.scalar);
    if (st != nullptr) copy_bytes(D, st, f.status.data(), static_cast<std::size_t>(n));
    if (D != nullptr) device_sync(*D);
}

}  // namespace door
}  // namespace eagle
