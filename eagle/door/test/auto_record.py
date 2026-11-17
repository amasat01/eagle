# SPDX-License-Identifier: Apache-2.0
"""Record what the old stack (eagle 0.7.5, run from its own environment) decides for an AUTO dispatch group.

    python test/auto_record.py <cost header> <door rows host> <door rows device> <gtest cases>   (the device props come from the GPU: run it through the pool)

Two outputs, neither tied to the machine that recorded them:
 * <gtest cases> (tests/dispatch_cases_075.inc): every look of every run below, with ALL the inputs of the old stack's decision (branch costs, counts, fused
   and remaining steps, the device properties and the weights it used) and its gain and policy; tests/test_DispatchPolicy.cpp feeds each to Dispatch.h
   and asserts the same gain and policy. That proves the one implementation equals 0.7.5's on any machine.
 * <door rows> (test/auto_decisions*.txt): the runs whose decisions are the SAME on every machine of a grid (host: 1 to 64 cores, 0 to 64 MB of cache;
   device: five GPUs from a P2000 to an H100): the door end-to-end tests assert exactly these, so the expectation needs no knowledge of the machine.

Each case is a run of the static two-branch group of the fixture libraries (test/fixture_unit.c, "sga_*"): N samples, n1 of them class 1
(limit L1, two steps a launch) and the rest class 0 (limits 2..5, one step a launch), looked at as 0.7.5's DispatchRunner looks: at the start, then every
16 steps while segmented. The rule is `predicted_gain` and `DispatchState` of the old stack, its weights replaced by the rows the door resolves on this
machine (read here from sysfs again, not from the door). A line is: unit n1 L1 max_steps seg_steps total_steps.
"""
import re
import sys

from eagle import _dispatch_policy as pol

N = 200


class Branch:
    def __init__(self, row, vectorised):
        self.row, self.vectorised = row, vectorised

    def row_tuple(self):
        return tuple(self.row)


class Cost:
    def __init__(self, fused_vectorised, branches):
        self.fused_vectorised, self.branches = fused_vectorised, branches


def read_costs(path):
    text = open(path).read()
    costs = {}
    for name in re.findall(r"#define AUTO_FLAGS_(\w+) ", text):
        flags = int(re.search(rf"AUTO_FLAGS_{name} (\d+)u", text).group(1))
        branches = []
        for j in (0, 1):
            body = re.search(rf"AUTO_COST_{name}_{j} (.*)", text).group(1)
            fields = [float(v.rstrip("u")) for v in body.split(",")]
            branches.append(Branch(fields[1:], bool(int(body.split(",")[0].rstrip("u")) & 1)))
        costs[name] = Cost(bool(flags & 1), branches)
    return costs


HOST_CORES = (1, 2, 4, 8, 16, 32, 64)
HOST_CACHE = (0.0, 2e6, 12976128.0, 64e6)
#: (name, peak_flops_sp, peak_bytes_per_s, fp64_ratio, L2 bytes): the GPUs the device rows must agree on
DEVICES = (("p2000", None, None, None, None), ("a100", 19.5e12, 1.555e12, 2.0, 40e6), ("l4", 30.3e12, 0.3e12, 64.0, 48e6),
           ("h100", 67e12, 3.35e12, 2.0, 50e6), ("rtx4090", 82.6e12, 1.008e12, 64.0, 72e6))


def host_weights(cores, cache):
    return {"arith32": 1.495, "special32": 25.2, "special64": 122.0, "elem_bytes": 8.0, "launch_seconds": 1.4e-6,
            "split_seconds": 0.35e-9, "sort_seconds": 13.2e-9, "lane_rate": 24.5e9 / 4 * cores, "memory_rate": 52.1e9,
            "cache_bytes": cache, "cache_ratio": 5.09, "chain_seconds": 0.23e-9 * 4 / cores, "vector_width": 8.0}


def device_machine(entry):
    """(props, weights) of a device of the grid; the first entry is the GPU of this machine, measured by the old stack."""
    from eagle import _core
    from eagle._device_props import device_props

    name, flops, bandwidth, ratio, l2 = entry
    if flops is None:
        p = device_props(0)
        props = {f: float(p[f]) if hasattr(p, "__getitem__") else float(getattr(p, f)) for f in ("peak_flops_sp", "peak_bytes_per_s", "fp64_ratio")}
        import importlib

        l2 = float(importlib.import_module("eagle.launch")._device_props().get("l2CacheSize", 0))
    else:
        props = {"peak_flops_sp": flops, "peak_bytes_per_s": bandwidth, "fp64_ratio": ratio}
    return props, dict(_core.default_dispatch_weights("device"), cache_bytes=l2)


def finish_step(limit, klass):
    return int(limit) if klass == 0 else -(-int(limit) // 2)


def run_case(cost, n1, l1, max_steps, target, props, w, looks=None):
    """The run as 0.7.5's DispatchRunner steps it: (segmented steps, total steps); every look is appended to @p looks."""
    finish = [finish_step(2 + i % 4, 0) for i in range(N - n1)] + [finish_step(l1, 1)] * n1
    classes = [0] * (N - n1) + [1] * n1

    def counts(step):
        return [sum(1 for c, f in zip(classes, finish) if f > step and c == j) for j in (0, 1)]

    state = pol.DispatchState("switch")

    def look(step):
        c, before = counts(step), state.policy
        gain = pol.predicted_gain(cost, c, target=target, props=props, fused_steps=1, remaining_steps=max_steps - step, weights=w)
        state.apply(gain)
        if looks is not None:
            looks.append((c, max_steps - step, before, gain, state.policy))

    look(0)
    seg = 0
    while state.policy == "segmented" and seg < max_steps and sum(counts(seg)) > 0:
        seg += 1
        if state.due() and sum(counts(seg)) > 0 and seg < max_steps:
            look(seg)
    return seg, min(max_steps, max(finish))


CASES = [(name, n1, l1, m) for name in ("rare", "even", "scalar", "heavy") for n1 in (2, 20, 100) for l1 in (20, 100) for m in (4, 64, 400)]


def write_door_rows(path, costs, machines, target):
    """The runs that every machine of @p machines decides the same way."""
    lines = [f"# {target}: runs decided the same on every machine of the grid (auto_record.py); unit n1 l1 max_steps seg_steps total_steps"]
    for name, n1, l1, m in CASES:
        runs = {run_case(costs[name], n1, l1, m, target, p, w) for p, w in machines}
        if len(runs) == 1:
            lines.append(f"{name} {n1} {l1} {m} {runs.pop()[0]} {run_case(costs[name], n1, l1, m, target, *machines[0])[1]}")
    open(path, "w").write("\n".join(lines) + "\n")


def write_cases(path, costs, machines):
    """Every look of every run on every machine, with the inputs and the old stack's answer, as a C++ initializer list."""
    keys = ("arith32", "special32", "special64", "elem_bytes", "launch_seconds", "split_seconds", "sort_seconds", "lane_rate", "memory_rate",
            "cache_bytes", "cache_ratio", "chain_seconds", "vector_width")
    out, seen = [], set()
    for target, grid in machines:
        for props, w in grid:
            for name, n1, l1, m in CASES:
                cost, looks = costs[name], []
                run_case(cost, n1, l1, m, target, props, w, looks)
                for c, remaining, before, gain, after in looks:
                    key = (target, name, tuple(sorted(props.items())) if props else None, tuple(sorted(w.items())), tuple(c), remaining, before)
                    if key in seen:
                        continue
                    seen.add(key)
                    p = props or pol._core.cpu_device_props()
                    rows = ", ".join("{ " + ", ".join(repr(float(v)) for v in b.row_tuple()) + f", {int(b.vectorised)} }}" for b in cost.branches)
                    out.append("    { %s, %d, { %s }, { %d, %d }, %d, %d, { %r, %r, %r }, { %s }, %r, %s, %s }," % (
                        '"host"' if target == "host" else '"device"', int(cost.fused_vectorised), rows, c[0], c[1], 1, remaining,
                        float(p["peak_flops_sp"]), float(p["peak_bytes_per_s"]), float(p["fp64_ratio"]),
                        ", ".join(repr(float(w[k])) for k in keys), float(gain), "true" if before == "segmented" else "false",
                        "true" if after == "segmented" else "false"))
    text = ("// SPDX-License-Identifier: Apache-2.0\n// Generated by eagle/door/test/auto_record.py from eagle 0.7.5's predicted_gain and DispatchState: {target, fusedVectorised, two branches\n"
            "// {arith32, arith64, special32, special64, loads, gathers, chain32, chain64, chainSpecial32, chainSpecial64, stores, vectorised}, counts, fused steps,\n"
            "// remaining steps, {peak_flops_sp, peak_bytes_per_s, fp64_ratio}, the 13 weights, gain, segmented before, segmented after}.\n" + "\n".join(out) + "\n")
    open(path, "w").write(text)


def main(header, host_rows, device_rows, cases):
    costs = read_costs(header)
    host = [({}, host_weights(c, k)) for c in HOST_CORES for k in HOST_CACHE]
    props_host = pol._core.cpu_device_props()
    host = [(props_host, w) for _, w in host]
    device = [device_machine(e) for e in DEVICES]
    write_door_rows(host_rows, costs, host, "host")
    write_door_rows(device_rows, costs, device, "device")
    write_cases(cases, costs, [("host", host), ("device", device)])


if __name__ == "__main__":
    main(*sys.argv[1:5])
