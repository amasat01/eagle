# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Zero-copy interop demo: one hawk kernel fed from, and handed back into,
NumPy, CuPy and PyTorch arrays with no copy where the family's own
certification matrix (``raptor.conformance.interop``) says there is none.

The kernel itself is intentionally trivial (``y = k * x``, a per-sample
vector scale) -- this demo is about the POINTERS a crossing does or does not
copy, not the physics. The device arm feeds it a CUDA-resident torch tensor
(a zero-copy view into eagle, row ``T-IN-CUDA-ALIAS``), writes the result
straight into a second CUDA torch buffer through the same zero-copy view
(row ``T-OUT-CUDA-ALIAS``), and separately feeds it a CPU-resident torch
tensor, which the identical adapter uploads with an honest copy (row
``T-IN-CPU-COPY``) -- three lines, three measured pointers, three declared
rows, checked against each other so a mismatch fails loudly rather than
silently. The host arm runs the same kernel on CPU threads with plain numpy
in/out; no row in the matrix governs that crossing (numpy is eagle's own
host currency, the same way cupy is its own device currency), so this demo
makes no ALIAS/COPY claim about it.

``--device`` also runs a copy-cost demo: the same kernel through the alias
path above versus the copying round-trip a caller without this protocol
would naturally write (CPU tensor out, upload, launch, download, CPU tensor
back), timed at a size where the PCIe cost is easy to see.

Run it with ``python examples/python/zero_copy_interop.py --host`` (CPU, needs no
GPU or torch) or ``--device`` (GPU, needs torch).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import statistics
import tempfile
import time
from types import SimpleNamespace

import numpy as np

#: The kernel's name, and so the name of every file it is published under.
KERNEL_NAME = "zc_scale"
#: Acceptance: the kernel's result must agree with the plain numpy reference
#: to this relative tolerance (float32 arithmetic, so not bit-exact).
REL_TOL = 1e-5


# --------------------------------------------------------------------------- #
# The kernel: ONE traced hawk kernel = one per-sample vector scale.
# --------------------------------------------------------------------------- #
def step_kernel():
    """Trace ``y = k * x``: a ``Vector[3]`` input, a uniform, a mutable output."""
    import hawk

    @hawk.kernel
    def zc_scale(x: hawk.Vector[3], k: hawk.Param, y: hawk.Mutable[hawk.Vector[3]]):
        y = k * x

    return zc_scale


def build_artifact(directory, targets) -> object:
    """Emit, compile and publish the step kernel into ``directory``."""
    import hawk

    return hawk.build([step_kernel()], pathlib.Path(directory),
                        mode="float32", targets=tuple(targets))


def check_manifest(bundle) -> dict:
    """Read the published manifest back and validate its shape through raptor."""
    from raptor.schema import validate_manifest

    doc = json.loads(pathlib.Path(bundle.manifest_path).read_text())
    validate_manifest(doc)
    return doc


# --------------------------------------------------------------------------- #
# The consumer side: load the artifact through eagle and run it.
# --------------------------------------------------------------------------- #
def _sidecar(bundle, name: str) -> dict:
    return next(a.sidecar for a in bundle.artifacts if a.name == name)


def host_plugin(bundle, name: str):
    """The plan-able view of the bundle's HOST entry (mirrors two_body_hybrid.py)."""
    from hawk.artifact import plan_view

    library = ctypes.CDLL(str(pathlib.Path(bundle.directory) / f"{name}.so"))
    view = plan_view(_sidecar(bundle, name))
    entry = getattr(library, view.pop("host_entry"))
    return SimpleNamespace(
        host_entry=ctypes.cast(entry, ctypes.c_void_p).value,
        _keepalive=library, **view)


def device_plugin(bundle, name: str):
    """The plan-able view of the bundle's DEVICE entry (mirrors two_body_hybrid.py)."""
    from eagle.registry import load_manifest
    from hawk.artifact import plan_view

    registry = load_manifest(pathlib.Path(bundle.manifest_path))
    loaded = registry[name]
    view = plan_view(_sidecar(bundle, name))
    view.pop("host_entry", None)
    return SimpleNamespace(
        device_function=loaded.fn.kernel.ptr,
        _keepalive=(registry, loaded), **view)


def seed_positions(batch: int, seed: int) -> np.ndarray:
    """A ``(3, batch)`` float64 plane of arbitrary vectors.

    This demo is about the pointers a crossing does or does not copy, not
    the physics, so any well-conditioned values work.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, size=(3, batch))


def _max_rel(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(a - ref) / np.maximum(np.abs(ref), 1e-6)))


def _check_row(row_id: str, measured_alias: bool) -> str:
    """Look ``row_id`` up in raptor's own declaration and assert the MEASURED
    pointer behavior agrees with what it certifies.

    This is the mechanical form of "never claim an alias a row does not
    certify": if a row says ``ALIAS`` and this crossing copied (or the
    reverse), this raises loudly instead of the example printing a claim
    nothing backs.
    """
    from raptor.conformance.interop import ROWS

    kind = ROWS[row_id].kind
    if kind == "ALIAS":
        assert measured_alias, (
            f"{row_id} is declared ALIAS but this crossing measured a copy")
    elif kind == "COPY":
        assert not measured_alias, (
            f"{row_id} is declared COPY but this crossing measured an alias")
    else:
        raise AssertionError(f"{row_id} is kind={kind!r}, not ALIAS or COPY")
    return kind


# --------------------------------------------------------------------------- #
# The device arm: torch tensors crossing into eagle, CUDA and CPU alike.
# --------------------------------------------------------------------------- #
def run_device(bundle, positions: np.ndarray, k: float) -> dict:
    """Needs torch and a GPU.

    Three legs, three certified rows: a CUDA torch tensor read by the kernel
    (``T-IN-CUDA-ALIAS``), a CUDA torch buffer the kernel writes into in
    place (``T-OUT-CUDA-ALIAS``), and a CPU torch tensor read by the SAME
    kernel through the SAME adapter, which uploads it (``T-IN-CPU-COPY``).
    All three go through :func:`eagle.to_cupy`, the one public entry point
    that imports a foreign tensor to a cupy view -- zero-copy when the
    tensor is already device-resident, a host upload otherwise.

    The CUDA input leg also gets a BEHAVIOURAL check: a matching pointer
    proves the two arrays share an address, not that the kernel actually
    reads through it. Mutating ``t_in`` in place and relaunching the SAME
    bound plan (no rebind) is what actually exercises that memory.
    """
    import cupy as cp
    import eagle
    import eagle.exec as eexec
    import torch
    from eagle import plan as eplan

    x32 = np.ascontiguousarray(positions, dtype=np.float32)
    plan = eplan.plan(device_plugin(bundle, KERNEL_NAME), structure=eexec.DeviceKernel)

    # ---- T-IN-CUDA-ALIAS: a CUDA torch tensor, viewed as cupy, no copy. ----
    t_in = torch.from_numpy(x32.copy()).cuda().contiguous()
    cu_in = eagle.to_cupy(t_in)
    in_cuda_alias = cu_in.data.ptr == t_in.data_ptr()
    _check_row("T-IN-CUDA-ALIAS", in_cuda_alias)

    # ---- T-OUT-CUDA-ALIAS: a CUDA torch buffer, filled in place through its
    # own cupy view -- "viewed as CuPy", written, then read back. ----
    t_out = torch.empty_like(t_in)
    cu_out = eagle.to_cupy(t_out)
    out_cuda_alias = cu_out.data.ptr == t_out.data_ptr()
    _check_row("T-OUT-CUDA-ALIAS", out_cuda_alias)

    bound = plan.bind(x=cu_in, k=k, y=cu_out)
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    # the pointer never moved: the kernel wrote INTO t_out's own storage.
    out_cuda_alias_after = cu_out.data.ptr == t_out.data_ptr()
    result_cuda = t_out.cpu().numpy()  # ordinary device -> host copy; not a row

    # ---- T-IN-CUDA-ALIAS, BEHAVIOURAL: the pointer check above says the two
    # addresses match; this proves the kernel actually reads THAT memory,
    # not a snapshot taken when cu_in was built. Mutate t_in IN PLACE,
    # relaunch the SAME bound plan with no rebind, and the output must track
    # the mutation. ----
    t_in.mul_(3)
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    mutated_result = t_out.cpu().numpy()
    mutated_ref = k * t_in.cpu().numpy()
    in_cuda_alias_behavioural = bool(
        np.allclose(mutated_result, mutated_ref, rtol=REL_TOL, atol=1e-6))

    # ---- T-IN-CPU-COPY: a CPU torch tensor, read by the same kernel. The
    # SAME eagle.to_cupy call this time uploads (no device memory to alias
    # from a host tensor), so the row declares COPY, not ALIAS. ----
    t_in_cpu = torch.from_numpy(x32.copy()).contiguous()
    cu_in_cpu = eagle.to_cupy(t_in_cpu)
    in_cpu_alias = cu_in_cpu.data.ptr == t_in_cpu.data_ptr()
    _check_row("T-IN-CPU-COPY", in_cpu_alias)

    cu_out_cpu = cp.empty_like(cu_in_cpu)
    bound_cpu = plan.bind(x=cu_in_cpu, k=k, y=cu_out_cpu)
    bound_cpu.launch()
    cp.cuda.runtime.deviceSynchronize()
    result_cpu = cp.asnumpy(cu_out_cpu)

    ref = k * x32
    return {
        "in_cuda_alias": bool(in_cuda_alias),
        "out_cuda_alias": bool(out_cuda_alias and out_cuda_alias_after),
        "in_cuda_alias_behavioural": in_cuda_alias_behavioural,
        "in_cpu_alias": bool(in_cpu_alias),
        "max_rel_cuda": _max_rel(result_cuda, ref),
        "max_rel_cpu": _max_rel(result_cpu, ref),
    }


# --------------------------------------------------------------------------- #
# The host arm: plain numpy in/out, eagle's own currency, no row involved.
# --------------------------------------------------------------------------- #
def run_host(bundle, positions: np.ndarray, k: float) -> dict:
    """Needs no GPU, no cupy, no torch.

    ``eagle.exec.HostTeam`` reads and writes the caller's own numpy arrays
    directly (``plan.bind`` allocates, coerces and copies nothing) -- this
    is eagle's native host format, not a framework crossing, so no row in
    the matrix applies to it and none is claimed here.
    """
    import eagle.exec as eexec
    from eagle import plan as eplan

    x = np.ascontiguousarray(positions, dtype=np.float32)
    y = np.zeros_like(x)
    plan = eplan.plan(host_plugin(bundle, KERNEL_NAME), structure=eexec.HostTeam)
    bound = plan.bind(x=x, k=k, y=y)

    ptr_before = x.ctypes.data
    bound.launch()
    ptr_after = x.ctypes.data
    ptr_stable = ptr_before == ptr_after  # never reallocated -- launch wrote INTO y

    ref = k * x
    return {"ptr_stable": bool(ptr_stable), "max_rel": _max_rel(y, ref)}


# --------------------------------------------------------------------------- #
# The copy-cost demo: the alias path against the round-trip a caller
# without this protocol would naturally write.
# --------------------------------------------------------------------------- #
def run_copy_cost_demo(size_mb: float = 256.0, k: float = 2.0, reps: int = 5,
                       build_dir=None, verbose: bool = True) -> dict:
    """Time the zero-copy alias path against the copying round-trip: torch
    CUDA -> ``.cpu().numpy()`` -> ``cp.asarray`` -> kernel -> ``cp.asnumpy``
    -> ``torch.from_numpy(...).cuda()`` -- the code a caller who never heard
    of ``to_cupy`` would naturally write. Same kernel, same data, so the gap
    is exactly the copies. ``batch`` is derived from ``size_mb`` at 12 bytes
    per sample (one float32 ``Vector[3]``). Demo measurement, not a
    benchmark; frees its own buffers before returning -- the P2000 this demo
    targets has 5 GB, shared with everything else on the box.
    """
    import cupy as cp
    import eagle
    import eagle.exec as eexec
    import torch
    from eagle import plan as eplan

    batch = max(1, int(size_mb * 1024 * 1024 / 12))
    x32 = np.ascontiguousarray(seed_positions(batch, 0), dtype=np.float32)
    nbytes = x32.nbytes

    with tempfile.TemporaryDirectory() as tmpdir:
        where = pathlib.Path(build_dir or tmpdir) / "zero_copy_interop_copycost"
        bundle = build_artifact(where, ("cuda",))
        plan = eplan.plan(
            device_plugin(bundle, KERNEL_NAME), structure=eexec.DeviceKernel
        )

        t_in = torch.from_numpy(x32.copy()).cuda().contiguous()
        t_out = torch.empty_like(t_in)  # pre-allocated ONCE: the alias path
        # never needs to reallocate between calls -- that is the point.

        def aliased():
            cu_in = eagle.to_cupy(t_in)
            cu_out = eagle.to_cupy(t_out)
            bound = plan.bind(x=cu_in, k=k, y=cu_out)
            bound.launch()
            cp.cuda.runtime.deviceSynchronize()

        def naive():
            host_in = t_in.cpu().numpy()       # device -> host copy
            cu_in = cp.asarray(host_in)         # host -> device copy
            cu_out = cp.empty_like(cu_in)
            bound = plan.bind(x=cu_in, k=k, y=cu_out)
            bound.launch()
            cp.cuda.runtime.deviceSynchronize()
            host_out = cp.asnumpy(cu_out)       # device -> host copy
            torch.from_numpy(host_out).cuda()   # host -> device copy

        aliased()
        naive()  # warm-up (build/first-launch/JIT cost), excluded

        times = {"aliased": [], "naive": []}
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            aliased()
            times["aliased"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            naive()
            times["naive"].append(time.perf_counter() - t0)

        # correctness (untimed): read both paths' results back and compare.
        aliased_result = t_out.cpu().numpy().copy()
        host_in = t_in.cpu().numpy()
        cu_in = cp.asarray(host_in)
        cu_out = cp.empty_like(cu_in)
        bound = plan.bind(x=cu_in, k=k, y=cu_out)
        bound.launch()
        cp.cuda.runtime.deviceSynchronize()
        naive_result = cp.asnumpy(cu_out)

        t_in = t_out = cu_in = cu_out = None
        cp.get_default_memory_pool().free_all_blocks()

    ms = {name: statistics.median(v) * 1e3 for name, v in times.items()}
    pcie_bytes = 4 * nbytes  # naive: input down+up, output down+up
    ratio = ms["naive"] / ms["aliased"] if ms["aliased"] > 0 else float("nan")
    agreement = _max_rel(aliased_result, naive_result)

    results = {
        "size_mb": size_mb, "batch": batch,
        "ms_aliased": ms["aliased"], "ms_naive": ms["naive"],
        "pcie_bytes_naive": pcie_bytes, "ratio": ratio, "agreement": agreement,
    }
    if verbose:
        report_copy_cost(results)
    assert agreement <= REL_TOL, (
        f"aliased and naive paths disagree: max relative diff = {agreement:.3e}")
    return results


def report_copy_cost(results: dict) -> None:
    """Print the copy-cost demo's numbers."""
    print(f"copy-cost demo (demo measurement, not a benchmark): "
          f"size={results['size_mb']:.0f} MB (batch={results['batch']})")
    print(f"  aliased (to_cupy views + launch + sync)   : "
          f"{results['ms_aliased']:.2f} ms")
    print(f"  naive (cpu/numpy round-trip both ways)    : "
          f"{results['ms_naive']:.2f} ms")
    print(f"  bytes moved across PCIe by the naive path : "
          f"{results['pcie_bytes_naive'] / (1024 * 1024):.1f} MB "
          f"(down+up for input, down+up for output)")
    print(f"  ratio (naive / aliased)                   : {results['ratio']:.2f}x")
    print(f"  agreement (aliased vs naive, max rel diff): "
          f"{results['agreement']:.3e} (tolerance {REL_TOL:.0e})")


# --------------------------------------------------------------------------- #
# The demo itself.
# --------------------------------------------------------------------------- #
def run_demo(batch: int = 64, k: float = 2.0, seed: int = 0,
             target: str = "host", build_dir=None, verbose: bool = True) -> dict:
    """Build, run and check. Returns every measured number."""
    positions = seed_positions(batch, seed)

    targets = ("host",) if target == "host" else ("cuda",)
    with tempfile.TemporaryDirectory() as tmpdir:
        where = pathlib.Path(build_dir or tmpdir) / "zero_copy_interop"
        bundle = build_artifact(where, targets)
        manifest = check_manifest(bundle)
        if target == "host":
            arm = run_host(bundle, positions, k)
        else:
            arm = run_device(bundle, positions, k)

    results = {
        "target": target, "batch": batch, "k": k,
        "exec_targets": manifest["exec_targets"],
        **arm,
    }
    if verbose:
        report(results)

    if target == "host":
        assert results["ptr_stable"], (
            "the host input array was reallocated -- expected an in-place launch")
        assert results["max_rel"] <= REL_TOL, (
            f"host result differs from the numpy reference by "
            f"{results['max_rel']:.3e}, tolerance {REL_TOL:.0e}")
    else:
        assert results["in_cuda_alias"], (
            "T-IN-CUDA-ALIAS: expected an alias, measured a copy")
        assert results["in_cuda_alias_behavioural"], (
            "T-IN-CUDA-ALIAS: the pointer matches but a relaunch after mutating "
            "the input in place did not see the mutation")
        assert results["out_cuda_alias"], (
            "T-OUT-CUDA-ALIAS: expected an alias, measured a copy")
        assert not results["in_cpu_alias"], (
            "T-IN-CPU-COPY: expected a copy, measured an alias")
        assert results["max_rel_cuda"] <= REL_TOL, (
            f"CUDA-leg result differs from the numpy reference by "
            f"{results['max_rel_cuda']:.3e}, tolerance {REL_TOL:.0e}")
        assert results["max_rel_cpu"] <= REL_TOL, (
            f"CPU-leg result differs from the numpy reference by "
            f"{results['max_rel_cpu']:.3e}, tolerance {REL_TOL:.0e}")
    return results


def report(results: dict) -> None:
    """Print every crossing's pointer identity and the row it exercises."""
    print(f"target            : {results['target']} "
          f"(manifest exec_targets {results['exec_targets']})")
    print(f"batch x k         : {results['batch']} x {results['k']}")
    if results["target"] == "device":
        print(f"T-IN-CUDA-ALIAS  (ALIAS): torch cuda tensor -> cupy view, "
              f"data_ptr equal = {results['in_cuda_alias']}")
        print(f"T-IN-CUDA-ALIAS  (behavioural): t_in.mul_(3) in place + relaunch "
              f"(no rebind), output tracks the mutation = "
              f"{results['in_cuda_alias_behavioural']}")
        print(f"T-OUT-CUDA-ALIAS (ALIAS): torch cuda buffer <- cupy view, "
              f"data_ptr equal = {results['out_cuda_alias']}")
        print(f"T-IN-CPU-COPY    (COPY) : torch cpu tensor -> cupy upload, "
              f"data_ptr equal = {results['in_cpu_alias']} (False is correct)")
        print("read back to host (torch.cpu().numpy() / cupy.asnumpy()): an "
              "ordinary device -> host copy, no row -- always a copy, leaving "
              "the device is not a crossing this matrix certifies")
        print(f"result vs numpy reference: CUDA leg max|rel diff| = "
              f"{results['max_rel_cuda']:.3e}, CPU leg = {results['max_rel_cpu']:.3e} "
              f"(tolerance {REL_TOL:.0e})")
    else:
        print("host arm: plain numpy in/out through eagle.exec.HostTeam -- no row "
              "in the matrix governs this crossing (eagle's own host currency)")
        print(f"input array pointer stable across launch = {results['ptr_stable']}")
        print(f"result vs numpy reference: max|rel diff| = {results['max_rel']:.3e} "
              f"(tolerance {REL_TOL:.0e})")


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=64,
                        help="independent vectors (default: 64)")
    parser.add_argument("--k", type=float, default=2.0,
                        help="the kernel's scale factor (default: 2.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the input vectors (default: 0)")
    parser.add_argument("--build-dir", default=None,
                        help="where to publish the artifact (default: a "
                             "temporary directory)")
    parser.add_argument("--size-mb", type=float, default=256.0,
                        help="copy-cost demo size in MB, --device only "
                             "(default: 256; batch = size / 12 bytes)")
    parser.add_argument("--copy-cost-reps", type=int, default=5,
                        help="interleaved timing reps for the copy-cost "
                             "demo, median reported (default: 5)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--host", dest="target", action="store_const",
                       const="host", help="run on CPU threads, numpy only "
                       "(the default; needs no GPU or torch)")
    where.add_argument("--device", dest="target", action="store_const",
                       const="device", help="run on the GPU (needs torch)")
    parser.set_defaults(target="host")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point."""
    args = parse_args(argv)
    run_demo(batch=args.batch, k=args.k, seed=args.seed, target=args.target,
             build_dir=args.build_dir)
    if args.target == "device":
        run_copy_cost_demo(size_mb=args.size_mb, k=args.k,
                           reps=args.copy_cost_reps, build_dir=args.build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
