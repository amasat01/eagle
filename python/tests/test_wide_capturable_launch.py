# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wide roles on the CAPTURABLE launch
path -- ``eagle.launch._launch_pure``'s ``wide_inputs``/``wide_outputs``/
``wide_out_exempt`` name tuples and the ``launch()`` dispatcher's forwarding of
all three out of ``**extra``.

The allocating twin (``pure_prepare``) has carried these keywords since
the start, and is covered by
``test_wide_device_coercions.py``; this file covers the OTHER path, and the
distinction is the whole point. ``pure_prepare`` COERCES -- a non-conforming
buffer becomes a fresh device array, which is a ``cudaMalloc`` and therefore
illegal inside a stream capture (``eagle/python/eagle/pipeline.py``'s own rule:
"Steps must not allocate device memory or synchronize"). ``_launch_pure``
instead REFUSES anything that is not already a pre-allocated cupy array, so the
refusal tests below are not error-message trivia: they ARE the capture-safety
contract, and each one is a call that would have silently allocated on the other
path.

Two surfaces, deliberately both:

* a ``_StubFn`` stand-in for the compiled kernel (this suite's convention for
  launch-layer coverage one level down the call stack -- see
  ``test_wide_device_coercions.py``'s own ``_StubFn`` note), which is what lets
  a test read back the exact assembled argument TUPLE;
* one REAL ``cupy.RawKernel`` bound through the full ``launch("pure", ...)``
  door, because a stub cannot prove the wide handles land at the right ABI
  positions -- only a kernel that reads through them can. Its parameters mirror
  ``eagle.abi.HANDLE_DTYPE`` (a wide buffer binds as ``HandleT``, i.e. a bare
  device pointer, exactly like a lookup table -- see ``assemble_args``'s
  docstring), so no GRef-shaped operand is needed and the kernel stays small.

The refusal tests run WITHOUT a GPU: ``require_cupy`` only imports cupy and
runs an ``isinstance`` check, and every refusal below fires before the first
line of ``_launch_pure`` that would touch a device pointer.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

from eagle.launch import _launch_pure, launch

_MDecl = namedtuple("_MDecl", ("name", "dtype", "width", "shape"), defaults=(None,))


class _StubFn:
    """Records the exact ``(grid, block, args)`` a launch call receives -- no
    real compiled kernel needed; ``_launch_pure`` only requires ``fn`` be
    callable with those three positionals (``_kernel_attrs`` degrades to ``{}``
    on a plain object with no ``.attributes``)."""

    def __init__(self):
        self.calls = []

    def __call__(self, grid, block, args):
        self.calls.append((grid, block, args))


# ============================================================================
# HOST: the refusals. Every one is a call that ``pure_prepare`` would have
# ACCEPTED by allocating -- which is precisely what may not happen here.
# ============================================================================
@pytest.mark.gpu
def test_launch_pure_refuses_a_numpy_wide_input():
    with pytest.raises(TypeError, match="pre-allocated cupy .*wide input"):
        _launch_pure(
            _StubFn(),
            [("wide_in", "x")],
            (),
            (),
            (),
            [],
            kw={"x": np.zeros((6, 4))},
            grid=None,
            block=None,
            wide_inputs=("x",),
        )


@pytest.mark.gpu
def test_launch_pure_refuses_a_missing_wide_input():
    with pytest.raises(TypeError, match="pre-allocated cupy .*wide input"):
        _launch_pure(
            _StubFn(),
            [("wide_in", "x")],
            (),
            (),
            (),
            [],
            kw={},
            grid=None,
            block=None,
            wide_inputs=("x",),
        )


@pytest.mark.gpu
def test_launch_pure_refuses_a_numpy_wide_output():
    with pytest.raises(TypeError, match="pre-allocated cupy .*wide gradient output"):
        _launch_pure(
            _StubFn(),
            [("wide_out", "x_bar")],
            (),
            (),
            (),
            [],
            kw={"x_bar": np.zeros((6, 4))},
            grid=None,
            block=None,
            wide_outputs=("x_bar",),
        )


@pytest.mark.gpu
def test_launch_pure_refuses_a_missing_wide_output():
    with pytest.raises(TypeError, match="pre-allocated cupy .*wide gradient output"):
        _launch_pure(
            _StubFn(),
            [("wide_out", "x_bar")],
            (),
            (),
            (),
            [],
            kw={},
            grid=None,
            block=None,
            wide_outputs=("x_bar",),
        )


@pytest.mark.gpu
def test_launch_pure_refuses_an_exempt_wide_output_it_was_not_handed():
    """An exempt name is exempt from N-RECONCILIATION only -- never from the
    pre-allocation contract itself."""
    with pytest.raises(TypeError, match="pre-allocated cupy .*wide gradient output"):
        _launch_pure(
            _StubFn(),
            [("wide_out", "dest")],
            (),
            (),
            (),
            [],
            kw={"dest": np.zeros((4, 1))},
            grid=None,
            block=None,
            wide_outputs=("dest",),
            wide_out_exempt=("dest",),
        )


# ============================================================================
# HOST: dispatcher forwarding. ``launch()`` takes ``**extra``, so an
# unforwarded keyword is SILENTLY DROPPED rather than raising -- the failure
# mode this pair is written against. The two errors are distinguishable by
# construction: forwarded, the wide loop refuses the numpy buffer; dropped,
# nothing sizes the batch and ``require_n`` refuses instead (proved by the
# third test, which is this pair's RED twin).
# ============================================================================
def _pure_launch(**extra):
    return launch(
        "pure",
        _StubFn(),
        extra.pop("arg_spec"),
        (),
        (),
        (),
        kw=extra.pop("kw"),
        grid=None,
        block=None,
        mutables_decl=[],
        **extra,
    )


@pytest.mark.gpu
def test_launch_dispatcher_forwards_wide_inputs_to_the_pure_path():
    with pytest.raises(TypeError, match="pre-allocated cupy .*wide input"):
        _pure_launch(
            arg_spec=[("wide_in", "x")],
            kw={"x": np.zeros((6, 4))},
            wide_inputs=("x",),
        )


@pytest.mark.gpu
def test_launch_dispatcher_forwards_wide_outputs_to_the_pure_path():
    with pytest.raises(TypeError, match="pre-allocated cupy .*wide gradient output"):
        _pure_launch(
            arg_spec=[("wide_out", "x_bar")],
            kw={"x_bar": np.zeros((6, 4))},
            wide_outputs=("x_bar",),
        )


def test_a_dropped_wide_name_fails_with_a_different_error():
    """The RED twin of the two forwarding tests: with the wide names absent,
    the SAME call reaches ``require_n`` instead (nothing sizes the batch), so
    neither test above can pass for an unrelated reason."""
    with pytest.raises(TypeError, match="cannot infer the sample count N"):
        _pure_launch(arg_spec=[("wide_in", "x")], kw={"x": np.zeros((6, 4))})


# ============================================================================
# GPU: binding, the N reconciliation, and its exemption.
# ============================================================================
@pytest.mark.gpu
def test_launch_pure_binds_wide_in_and_out_and_merges_the_handoff():
    import cupy as cp

    fn = _StubFn()
    kw = {
        "x": cp.arange(24, dtype=cp.float64).reshape(6, 4),
        "x_bar": cp.zeros((6, 4), dtype=cp.float64),
        "y": cp.zeros(4, dtype=cp.float64),
        "terminated": cp.zeros(4, dtype=cp.bool_),
    }
    result = _launch_pure(
        fn,
        [("wide_in", "x"), ("wide_out", "x_bar"), ("mutable", "y")],
        (),
        (),
        (),
        [_MDecl("y", "float", 1)],
        kw=kw,
        grid=None,
        block=None,
        wide_inputs=("x",),
        wide_outputs=("x_bar",),
    )
    assert len(fn.calls) == 1, "the kernel must be launched exactly once"
    _grid, _block, args = fn.calls[0]
    assert len(args) == 3  # one per arg_spec entry
    assert args[0]["data"] == kw["x"].data.ptr
    assert args[1]["data"] == kw["x_bar"].data.ptr
    # the wide gradient output rides the SAME returned dict as the Mutable
    # handoff, exactly as pure_prepare returns it.
    assert result["x_bar"] is kw["x_bar"]
    assert result["y"] is kw["y"]


@pytest.mark.gpu
def test_launch_pure_wide_free_call_returns_only_the_mutable_handoff():
    """The additive-only contract: a wide-free kernel's return value is what it
    always was."""
    import cupy as cp

    fn = _StubFn()
    kw = {"y": cp.zeros(4, dtype=cp.float64), "terminated": cp.zeros(4, dtype=cp.bool_)}
    result = _launch_pure(
        fn,
        [("mutable", "y")],
        (),
        (),
        (),
        [_MDecl("y", "float", 1)],
        kw=kw,
        grid=None,
        block=None,
    )
    assert len(fn.calls) == 1
    assert set(result) == {"y"}


@pytest.mark.gpu
def test_launch_pure_rejects_a_mismatched_wide_input_plane():
    import cupy as cp

    kw = {
        "x": cp.zeros((6, 5), dtype=cp.float64),  # N=5
        "y": cp.zeros(4, dtype=cp.float64),  # N=4
        "terminated": cp.zeros(4, dtype=cp.bool_),
    }
    with pytest.raises(ValueError, match="inconsistent batch size N"):
        _launch_pure(
            _StubFn(),
            [("wide_in", "x"), ("mutable", "y")],
            (),
            (),
            (),
            [_MDecl("y", "float", 1)],
            kw=kw,
            grid=None,
            block=None,
            wide_inputs=("x",),
        )


@pytest.mark.gpu
def test_launch_pure_rejects_a_mismatched_unexempt_wide_output_plane():
    import cupy as cp

    kw = {
        "x_bar": cp.zeros((6, 5), dtype=cp.float64),  # N=5
        "y": cp.zeros(4, dtype=cp.float64),  # N=4
        "terminated": cp.zeros(4, dtype=cp.bool_),
    }
    with pytest.raises(ValueError, match="inconsistent batch size N"):
        _launch_pure(
            _StubFn(),
            [("wide_out", "x_bar"), ("mutable", "y")],
            (),
            (),
            (),
            [_MDecl("y", "float", 1)],
            kw=kw,
            grid=None,
            block=None,
            wide_outputs=("x_bar",),
        )


@pytest.mark.gpu
def test_launch_pure_exempts_a_row_indexed_wide_out_from_n_reconciliation():
    """The atomic ``Accum`` arm: a ``(rows, 1)`` destination
    beside a genuine N=4 batch launches, and its width never becomes N."""
    import cupy as cp

    fn = _StubFn()
    kw = {
        "dest": cp.zeros((6, 1), dtype=cp.float64),
        "y": cp.zeros(4, dtype=cp.float64),
        "terminated": cp.zeros(4, dtype=cp.bool_),
    }
    result = _launch_pure(
        fn,
        [("wide_out", "dest"), ("mutable", "y"), ("nsamples", "nsamples")],
        (),
        (),
        (),
        [_MDecl("y", "float", 1)],
        kw=kw,
        grid=None,
        block=None,
        wide_outputs=("dest",),
        wide_out_exempt=("dest",),
    )
    _grid, _block, args = fn.calls[0]
    assert int(args[2]) == 4, "N must come from the Mutable, never the (rows, 1) dest"
    assert result["dest"] is kw["dest"]


@pytest.mark.gpu
def test_launch_pure_still_reconciles_a_wide_out_that_is_not_exempt():
    """The exemption is OPT-IN PER NAME: the same ``(6, 1)`` buffer that the
    test above accepts is refused the moment its name is left off
    ``wide_out_exempt`` -- so that test proves the exemption, not a blanket
    relaxation."""
    import cupy as cp

    kw = {
        "dest": cp.zeros((6, 1), dtype=cp.float64),
        "y": cp.zeros(4, dtype=cp.float64),
        "terminated": cp.zeros(4, dtype=cp.bool_),
    }
    with pytest.raises(ValueError, match="inconsistent batch size N"):
        _launch_pure(
            _StubFn(),
            [("wide_out", "dest"), ("mutable", "y")],
            (),
            (),
            (),
            [_MDecl("y", "float", 1)],
            kw=kw,
            grid=None,
            block=None,
            wide_outputs=("dest",),
        )


# ============================================================================
# GPU: one real kernel through the full launch() door -- the proof that the
# wide handles land at the ABI positions the arg_spec names, which no stub can
# give. ``H`` mirrors eagle.abi.HANDLE_DTYPE (a single device pointer, the wire
# shape every wide buffer takes -- assemble_args' docstring).
# ============================================================================
_WIDE_PROBE_SRC = r"""
struct H { double* p; };

extern "C" __global__
void cb1a_wide_probe(H x, H x_bar, H dest, H y, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = x.p[i];          // wide input, row 0
    double b = x.p[n + i];      // wide input, row 1
    y.p[i] = a + b;             // the Mutable handoff
    x_bar.p[i] = 2.0 * a;       // wide output plane, row 0
    x_bar.p[n + i] = 3.0 * b;   // wide output plane, row 1
    atomicAdd(dest.p + (i & 3u), 1.0);  // the atomic (rows, 1) destination
}
"""


@pytest.mark.gpu
def test_a_real_pure_kernel_reads_wide_planes_and_an_atomic_dest():
    import cupy as cp

    n = 8
    x = cp.arange(2 * n, dtype=cp.float64).reshape(2, n)
    x_bar = cp.zeros((2, n), dtype=cp.float64)
    dest = cp.zeros((4, 1), dtype=cp.float64)
    y = cp.zeros(n, dtype=cp.float64)
    result = launch(
        "pure",
        cp.RawKernel(_WIDE_PROBE_SRC, "cb1a_wide_probe"),
        [
            ("wide_in", "x"),
            ("wide_out", "x_bar"),
            ("wide_out", "dest"),
            ("mutable", "y"),
            ("nsamples", "nsamples"),
        ],
        (),
        (),
        (),
        kw={
            "x": x,
            "x_bar": x_bar,
            "dest": dest,
            "y": y,
            "terminated": cp.zeros(n, dtype=cp.bool_),
        },
        grid=None,
        block=None,
        mutables_decl=[_MDecl("y", "float", 1)],
        wide_inputs=("x",),
        wide_outputs=("x_bar", "dest"),
        wide_out_exempt=("dest",),
    )
    cp.cuda.runtime.deviceSynchronize()
    expect = np.arange(2 * n, dtype=np.float64).reshape(2, n)
    np.testing.assert_array_equal(y.get(), expect[0] + expect[1])
    np.testing.assert_array_equal(x_bar.get()[0], 2.0 * expect[0])
    np.testing.assert_array_equal(x_bar.get()[1], 3.0 * expect[1])
    # n=8 threads, each adding 1.0 into row (i & 3) -> 2 per row.
    np.testing.assert_array_equal(dest.get(), np.full((4, 1), 2.0))
    # every RMW'd destination is readable off ONE returned dict.
    assert set(result) == {"x_bar", "dest", "y"}


@pytest.mark.gpu
def test_the_real_kernel_launch_allocates_nothing():
    """The capture-safety property itself, measured rather than asserted: the
    whole ``launch()`` call must not grow cupy's default pool (a pool growth is
    the ``cudaMalloc`` that would fail a stream capture)."""
    import cupy as cp

    n = 8
    kernel = cp.RawKernel(_WIDE_PROBE_SRC, "cb1a_wide_probe")
    kw = {
        "x": cp.arange(2 * n, dtype=cp.float64).reshape(2, n),
        "x_bar": cp.zeros((2, n), dtype=cp.float64),
        "dest": cp.zeros((4, 1), dtype=cp.float64),
        "y": cp.zeros(n, dtype=cp.float64),
        "terminated": cp.zeros(n, dtype=cp.bool_),
    }
    arg_spec = [
        ("wide_in", "x"),
        ("wide_out", "x_bar"),
        ("wide_out", "dest"),
        ("mutable", "y"),
        ("nsamples", "nsamples"),
    ]
    common = dict(
        kw=kw,
        grid=None,
        block=None,
        mutables_decl=[_MDecl("y", "float", 1)],
        wide_inputs=("x",),
        wide_outputs=("x_bar", "dest"),
        wide_out_exempt=("dest",),
    )
    # warm first: the kernel's own NVRTC compile + eagle's device-property and
    # kernel-attribute caches are one-time costs, not per-launch ones.
    launch("pure", kernel, arg_spec, (), (), (), **common)
    cp.cuda.runtime.deviceSynchronize()
    pool = cp.get_default_memory_pool()
    before = pool.used_bytes()
    launch("pure", kernel, arg_spec, (), (), (), **common)
    cp.cuda.runtime.deviceSynchronize()
    assert pool.used_bytes() == before
