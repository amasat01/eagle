#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Card writer: zero-copy GPU interop vs the copying round-trip a caller
without the protocol would naturally write.

Backs the example notebook's claim that at 256 MB the copies cost over 40x
the alias path, with a committed, reproducible measurement.

Real run (GPU, done separately from this script's ``--dry`` path, never on this check's own machine time):
times the SAME two closures ``examples/python/zero_copy_interop.py``'s own
``run_copy_cost_demo`` builds (``aliased``: eagle's zero-copy ``to_cupy``
view both ways; ``naive``: ``.cpu().numpy()`` / ``cp.asarray`` / ``cp.asnumpy``
/ ``torch.from_numpy(...).cuda()`` round trip), but -- unlike that function,
which returns only the median -- keeps every repetition's raw wall time, and
writes ``zero_copy_interop_card.json``/``.md`` next to this script.

``--dry`` (CPU, tiny size, no GPU, no cupy, no torch required): the real
comparison is inherently a GPU/PCIe measurement (eagle's ``to_cupy`` view
vs. a host round trip), so there is no CPU twin of it to run honestly.
Instead this exercises the SAME card-writing mechanics with an explicit
HOST-ONLY analogue -- ``examples/python/zero_copy_interop.py``'s own ``run_host``
(eagle's native host format: no copy, no allocation) timed against the
same launch preceded by an explicit ``np.copy()`` round trip -- clearly
labelled as an analogue, not a stand-in for the real number: the card
carries ``"dry": true``, and is never written to the committed filename.

Usage::

    python zero_copy_interop_card.py --dry
    python zero_copy_interop_card.py --device              # needs a visible GPU
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (_HERE.parent, _REPO_ROOT / "examples" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _card_common as cc            # noqa: E402
import zero_copy_interop as example  # noqa: E402

CARD_NAME = "zero_copy_interop"
DRY_SIZE_MB = 1.0
DRY_REPS = 5


def _copy_cost_host(size_mb: float, k: float, reps: int, seed: int = 0) -> dict:
    """Host-only analogue of the GPU alias-vs-copy comparison: eagle's
    native host format (no copy at all, ``run_host``'s own launch) against
    the same launch preceded by an explicit ``np.copy()`` round trip --
    same kernel, same data, so the gap is exactly that one extra copy."""
    import numpy as np
    import eagle.exec as eexec
    from eagle import plan as eplan

    batch = max(1, int(size_mb * 1024 * 1024 / 12))
    x32 = np.ascontiguousarray(example.seed_positions(batch, seed), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = example.build_artifact(
            pathlib.Path(tmpdir) / "zero_copy_interop_card_dry", ("host",))
        plan = eplan.plan(example.host_plugin(bundle, example.KERNEL_NAME),
                          structure=eexec.HostTeam)

        def direct():
            y = np.zeros_like(x32)
            plan.bind(x=x32, k=k, y=y).launch()
            return y

        def copied():
            x_copy = x32.copy()
            y = np.zeros_like(x_copy)
            plan.bind(x=x_copy, k=k, y=y).launch()
            return y.copy()

        direct_result = direct()
        copied_result = copied()  # warm-up, excluded

        times = {"direct": [], "copied": []}
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            direct()
            times["direct"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            copied()
            times["copied"].append(time.perf_counter() - t0)

    agreement = float(abs(direct_result - copied_result).max() /
                      max(1e-30, abs(direct_result).max()))
    return {"size_mb": size_mb, "batch": batch, "agreement": agreement,
           "wall_s": {name: cc.median_iqr(v) for name, v in times.items()}}


def _code() -> dict:
    return {"kernel_source": inspect.getsource(example.build_artifact)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--device", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--size-mb", type=float, default=None)
    args = p.parse_args(argv)

    if args.dry == args.device:
        p.error("pass exactly one of --dry / --device")

    facts = cc.demo_facts(__file__, example, "examples/python/zero_copy_interop.py")
    if args.dry:
        reps = args.reps or DRY_REPS
        size_mb = args.size_mb or DRY_SIZE_MB
        results = _copy_cost_host(size_mb, k=2.0, reps=reps)
        card = {"schema": cc.DEMO_SCHEMA, "name": CARD_NAME, "dry": True,
               "facts": facts, "code": _code(), "results": results}
        out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(
            tempfile.mkdtemp(prefix="demo_card_dry_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{CARD_NAME}_card.DRY.json"
        md_path = out_dir / f"{CARD_NAME}_card.DRY.md"
        cc.write_card(card, json_path, cc.demo_md(card, CARD_NAME, "a host-only analogue; see this script's own docstring"), md_path)
        print(f"[dry] wrote {json_path} and {md_path} (not committed)")
        return 0

    # --device: the real run, GPU only. Mirrors
    # run_copy_cost_demo's own two closures, but keeps every repetition.
    import cupy as cp
    import eagle
    import eagle.exec as eexec
    import numpy as np
    import torch
    from eagle import plan as eplan

    size_mb = args.size_mb or 256.0
    reps = args.reps or 5
    k = 2.0
    batch = max(1, int(size_mb * 1024 * 1024 / 12))
    x32 = np.ascontiguousarray(example.seed_positions(batch, 0), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = example.build_artifact(
            pathlib.Path(tmpdir) / "zero_copy_interop_card", ("cuda",))
        plan = eplan.plan(example.device_plugin(bundle, example.KERNEL_NAME),
                          structure=eexec.DeviceKernel)
        t_in = torch.from_numpy(x32.copy()).cuda().contiguous()
        t_out = torch.empty_like(t_in)

        def aliased():
            cu_in, cu_out = eagle.to_cupy(t_in), eagle.to_cupy(t_out)
            plan.bind(x=cu_in, k=k, y=cu_out).launch()
            cp.cuda.runtime.deviceSynchronize()

        def naive():
            cu_in = cp.asarray(t_in.cpu().numpy())
            cu_out = cp.empty_like(cu_in)
            plan.bind(x=cu_in, k=k, y=cu_out).launch()
            cp.cuda.runtime.deviceSynchronize()
            torch.from_numpy(cp.asnumpy(cu_out)).cuda()

        aliased()
        naive()  # warm-up, excluded

        times = {"aliased": [], "naive": []}
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            aliased()
            times["aliased"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            naive()
            times["naive"].append(time.perf_counter() - t0)
        t_in = t_out = None
        cp.get_default_memory_pool().free_all_blocks()

    wall = {name: cc.median_iqr(v) for name, v in times.items()}
    ratio = (wall["naive"]["median"] / wall["aliased"]["median"]
            if wall["aliased"]["median"] else None)
    results = {"size_mb": size_mb, "batch": batch,
              "pcie_bytes_naive": 4 * x32.nbytes, "ratio": ratio, "wall_s": wall}
    card = {"schema": cc.DEMO_SCHEMA, "name": CARD_NAME, "dry": False,
           "facts": facts, "code": _code(), "results": results}
    out_dir = pathlib.Path(args.out) if args.out else _HERE
    json_path, md_path = out_dir / f"{CARD_NAME}_card.json", out_dir / f"{CARD_NAME}_card.md"
    cc.write_card(card, json_path, cc.demo_md(card, CARD_NAME, "a host-only analogue; see this script's own docstring"), md_path)
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
