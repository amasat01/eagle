# SPDX-License-Identifier: Apache-2.0
"""Record what the old stack (eagle 0.7.5, run from its own environment) decides for an AUTO dispatch group, to test/auto_decisions.txt.

    python test/auto_record.py <cost header> <output file>

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


def machine():
    cores = {open(f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list").read().strip() for i in range(
        len(__import__("os").sched_getaffinity(0)) * 0 + __import__("os").cpu_count())}
    cores = len(cores)
    levels = []
    for i in range(16):
        base = f"/sys/devices/system/cpu/cpu0/cache/index{i}/"
        try:
            level, kind, size = (open(base + k).read().strip() for k in ("level", "type", "size"))
        except OSError:
            continue
        if kind != "Instruction":
            levels.append((int(level), int(size[:-1]) * (1024 if size[-1] == "K" else 1048576)))
    last = max(level for level, _ in levels)
    cache = sum(b for level, b in levels if level == last) + cores * sum(b for level, b in levels if level < last)
    return cores, float(cache)


def weights():
    cores, cache = machine()
    return {"arith32": 1.495, "special32": 25.2, "special64": 122.0, "elem_bytes": 8.0, "launch_seconds": 1.4e-6,
            "split_seconds": 0.35e-9, "sort_seconds": 13.2e-9, "lane_rate": 24.5e9 / 4 * cores, "memory_rate": 52.1e9,
            "cache_bytes": cache, "cache_ratio": 5.09, "chain_seconds": 0.23e-9 * 4 / cores, "vector_width": 8.0}


def finish_step(limit, klass):
    return int(limit) if klass == 0 else -(-int(limit) // 2)


def run_case(cost, n1, l1, max_steps, w):
    finish = [finish_step(2 + i % 4, 0) for i in range(N - n1)] + [finish_step(l1, 1)] * n1
    classes = [0] * (N - n1) + [1] * n1

    def counts(step):
        live = [f > step for f in finish]
        return [sum(1 for c, v in zip(classes, live) if v and c == j) for j in (0, 1)]

    state = pol.DispatchState("switch")
    c = counts(0)
    state.apply(pol.predicted_gain(cost, c, target="host", fused_steps=1, remaining_steps=max_steps, weights=w))
    seg = 0
    while state.policy == "segmented" and seg < max_steps and sum(counts(seg)) > 0:
        seg += 1
        if state.due() and sum(counts(seg)) > 0 and seg < max_steps:
            state.apply(pol.predicted_gain(cost, counts(seg), target="host", fused_steps=1, remaining_steps=max_steps - seg, weights=w))
    total = min(max_steps, max(finish))
    return seg, total


def main(header, out):
    costs, w = read_costs(header), weights()
    lines = [f"# machine: {machine()}"]
    for name in costs:
        for n1 in (2, 20, 100):
            for l1 in (20, 100):
                for max_steps in (4, 64, 400):
                    seg, total = run_case(costs[name], n1, l1, max_steps, w)
                    lines.append(f"{name} {n1} {l1} {max_steps} {seg} {total}")
    open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
