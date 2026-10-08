# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.gemm`` — the moved GEMM/cuBLAS capture runtime.

**Provenance.** This file ports the code generator's own test corpus for the
moved surface (host-side ``sgemm_params``/``CublasCaptureHandle``/
library-resolution corpus, ported near-verbatim: same cases, same assertions,
imports repointed to ``eagle.gemm``) and its plan/capture seam test (the
``lower_dense_gemm``/``captured_gemm_calls`` seam, adapted: the code
generator's version drives ``lower_dense_gemm`` off a record obtained by
tracing a real ``@kernel``-decorated function through its contraction
recognizer; this file has no tracer to call (see ``eagle.gemm.contraction``'s
docstring for why ``recognize`` did not move), so it drives the SAME lowering
off a HAND-BUILT :class:`~eagle.gemm.contraction.RecognizedContraction` shaped
exactly like the record the code generator's own recognizer would return for
the same two kernels — same operand layout, same
``operand_class``/``term_form``, same axes. ``lower_dense_gemm`` consumes a
record by duck-typed attribute access (its own docstring says so), so this
substitution changes nothing about what is under test.

This suite was collection-time verified against a stubbed ``eagle.gemm`` (a
bare ``raise NotImplementedError`` in ``capture``/``plan``/``contraction`` at
module scope): ``pytest --collect-only`` on this file raised that error for
every test in it — collection-time red, the whole file, because
``eagle/gemm/__init__.py``'s own ``from.capture import (...)`` / ``from.plan import
(...)`` / ``from.contraction import (...)`` re-exports fail before a single test body
runs. Restoring the real module bodies turned every
row green.

Sections:

1. The ``sgemm_params`` mapping corpus (the code generator's own, host-only,
   GPU-free by construction — the whole point of proving a
   strides-to-cuBLAS-args mapping with numpy rather than a device).
2. The plan/capture seam: ``lower_dense_gemm`` + ``captured_gemm_calls``
   against hand-built records.
3. GPU row: one Sgemm issued through :class:`~eagle.gemm.capture.CublasCaptureHandle`
   inside a REAL CUDA stream capture, replayed, compared to the uncaptured cupy
   ``matmul`` of the same operands within the K*eps float32 reassociation band the code
   generator's own test uses for exactly this class of comparison (``_band``, copied
   here verbatim rather than invented) — see :func:`_band` below.

A stream-binding refusal is NOT present: ``CublasCaptureHandle`` (see
:mod:`eagle.gemm.capture`) sets the stream once at construction via
``cublasSetStream_v2`` and never checks what stream a later ``sgemm()`` call
happens to run on — cuBLAS itself always issues on the HANDLE's bound stream
regardless of any notion of a Python "current stream", so there is no
wrong-stream call for the handle to refuse. Confirmed by reading the ported
module in full and by grep across the code generator's own test suite:
nothing checks for such a case. Recorded here rather than silently omitted.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from eagle.gemm import capture as gemm_capture
from eagle.gemm.capture import (
    CUBLAS_OP,
    CublasResolutionError,
    GemmMappingUnavailable,
    check_plan_capturable,
    mapped_library_paths,
    require_float32,
    resolve_cublas_path,
    sgemm_params,
    sgemm_params_for_arrays,
)
from eagle.gemm.contraction import ContractionOperand, RecognizedContraction
from eagle.gemm.plan import (
    LoweredPlan,
    MatmulSpec,
    PlanStep,
    WorkspaceSpec,
    assembly_step,
    captured_gemm_calls,
    lower_dense_gemm,
    matmul_step,
)

M, K, N = 3, 4, 5

#: All eight flag combinations ``matmul_step`` can be constructed with.
#: Enumerated, not sampled -- a corpus that samples a flag space certifies the
#: points it happened to draw.
FLAG_TRIPLES = tuple(itertools.product((False, True), repeat=3))

#: The three triples the funded dense cells actually construct, read
#: off ``_gemm_plan``'s own step factories -- ``{transpose_a}`` for the forward
#: contraction, ``{transpose_b, transpose_out}`` and
#: ``{transpose_a, transpose_b, transpose_out}`` for the two adjoint legs.
PRODUCTION_TRIPLES = {
    "forward_contraction": (True, False, False),
    "a_bar_adjoint": (False, True, True),
    "b_bar_adjoint": (True, True, True),
}

LAYOUTS = ("contiguous", "row_slice", "col_slice")


def flag_id(flags):
    ta, tb, tout = flags
    names = [n for n, f in (("ta", ta), ("tb", tb), ("tout", tout)) if f]
    return "+".join(names) if names else "none"


def make_operand(shape, layout, rng):
    """A float32 buffer of ``shape``, laid out per ``layout``.

    Returns ``(base, view, slicer)``: the owning allocation, the operand itself,
    and the index tuple that carves one out of the other."""
    rows, cols = shape
    if layout == "contiguous":
        base = rng.standard_normal((rows, cols)).astype(np.float32)
        sl = (slice(None), slice(None))
    elif layout == "row_slice":
        base = rng.standard_normal((rows + 3, cols)).astype(np.float32)
        sl = (slice(0, rows), slice(None))
    elif layout == "col_slice":
        base = rng.standard_normal((rows, cols + 4)).astype(np.float32)
        sl = (slice(None), slice(1, 1 + cols))
    else:  # pragma: no cover - the parametrization is closed
        raise AssertionError(layout)
    return base, base[sl], sl


def cm_read(x, ld):
    """The column-major matrix a raw Sgemm sees at ``x``'s first element, read
    with leading dimension ``ld``."""
    r, c = x.shape
    need = (r - 1) * ld + c
    flat = np.lib.stride_tricks.as_strided(x, shape=(need,), strides=(x.itemsize,))
    out = np.empty((c, r), dtype=np.float64)
    for j in range(r):
        out[:, j] = flat[j * ld : j * ld + c]
    return out


def cm_write(x, ld, values):
    """Write a column-major ``(m, n)`` result into ``x``'s raw memory at leading
    dimension ``ld`` -- the destination half of :func:`cm_read`."""
    m, n = values.shape
    need = (n - 1) * ld + m
    flat = np.lib.stride_tricks.as_strided(x, shape=(need,), strides=(x.itemsize,))
    for j in range(n):
        flat[j * ld : j * ld + m] = values[:, j]


def simulate_sgemm(params, arrays, *, lda=None, ldb=None, ldc=None):
    """Run the column-major GEMM ``params`` describes, in numpy."""
    first, second = params.operands
    left = cm_read(arrays[first], params.lda if lda is None else lda)
    right = cm_read(arrays[second], params.ldb if ldb is None else ldb)
    left = left.T if params.op_a == "T" else left
    right = right.T if params.op_b == "T" else right
    assert left.shape == (params.m, params.k), (left.shape, params)
    assert right.shape == (params.k, params.n), (right.shape, params)
    cm_write(arrays["out"], params.ldc if ldc is None else ldc, left @ right)


def reference_matmul(a, b, out_shape, flags):
    """``matmul_step``'s OWN result for these operands."""
    ta, tb, tout = flags
    step = matmul_step(
        "out", a="a", b="b", out="out", transpose_a=ta, transpose_b=tb,
        transpose_out=tout,
    )
    buffers = {
        "a": np.ascontiguousarray(a),
        "b": np.ascontiguousarray(b),
        "out": np.zeros(out_shape, dtype=np.float32),
    }
    step.run(buffers, "host")
    return buffers["out"]


def build_case(flags, layout, seed):
    ta, tb, tout = flags
    rng = np.random.default_rng(seed)
    a_shape = (K, M) if ta else (M, K)
    b_shape = (N, K) if tb else (K, N)
    out_shape = (N, M) if tout else (M, N)
    a_base, a, _ = make_operand(a_shape, layout, rng)
    b_base, b, _ = make_operand(b_shape, layout, rng)
    out_base, out, out_sl = make_operand(out_shape, layout, rng)
    out[...] = 0.0
    return a, b, out, out_base, out_sl, out_shape


# ============================================================================
# 1. The corpus: 8 flag triples x 3 layouts, every point checked against
# matmul_step's own answer. (ported verbatim from the code generator's own file)
# ============================================================================
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("flags", FLAG_TRIPLES, ids=[flag_id(f) for f in FLAG_TRIPLES])
def test_sgemm_mapping_reproduces_matmul_step(flags, layout):
    ta, tb, tout = flags
    seed = 1000 + FLAG_TRIPLES.index(flags) * 10 + LAYOUTS.index(layout)
    a, b, out, out_base, out_sl, out_shape = build_case(flags, layout, seed)

    expected = reference_matmul(a, b, out_shape, flags)
    params = sgemm_params_for_arrays(
        a, b, out, transpose_a=ta, transpose_b=tb, transpose_out=tout
    )
    simulate_sgemm(params, {"a": a, "b": b, "out": out})
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("flags", FLAG_TRIPLES, ids=[flag_id(f) for f in FLAG_TRIPLES])
def test_the_mapping_writes_nothing_outside_the_destination_view(flags, layout):
    ta, tb, tout = flags
    seed = 2000 + FLAG_TRIPLES.index(flags) * 10 + LAYOUTS.index(layout)
    a, b, out, out_base, out_sl, out_shape = build_case(flags, layout, seed)
    marker = np.zeros(out_base.shape, dtype=bool)
    marker[out_sl] = True
    before = out_base.copy()

    params = sgemm_params_for_arrays(
        a, b, out, transpose_a=ta, transpose_b=tb, transpose_out=tout
    )
    simulate_sgemm(params, {"a": a, "b": b, "out": out})
    np.testing.assert_array_equal(out_base[~marker], before[~marker])


@pytest.mark.parametrize("name", sorted(PRODUCTION_TRIPLES))
def test_the_production_triples_are_covered_by_name(name):
    flags = PRODUCTION_TRIPLES[name]
    assert flags in FLAG_TRIPLES
    ta, tb, tout = flags
    a, b, out, _base, _sl, out_shape = build_case(flags, "col_slice", 3000)
    expected = reference_matmul(a, b, out_shape, flags)
    params = sgemm_params_for_arrays(
        a, b, out, transpose_a=ta, transpose_b=tb, transpose_out=tout
    )
    simulate_sgemm(params, {"a": a, "b": b, "out": out})
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_the_operand_order_swaps_for_every_arm_but_transpose_out():
    for flags in FLAG_TRIPLES:
        ta, tb, tout = flags
        a, b, out, *_ = build_case(flags, "contiguous", 4000)
        params = sgemm_params_for_arrays(
            a, b, out, transpose_a=ta, transpose_b=tb, transpose_out=tout
        )
        assert params.operands == (("a", "b") if tout else ("b", "a")), flags
        assert params.op_a in CUBLAS_OP and params.op_b in CUBLAS_OP


def test_leading_dimensions_come_from_strides_not_shape():
    a, b, out, *_ = build_case((False, False, True), "col_slice", 5000)
    params = sgemm_params_for_arrays(
        a, b, out, transpose_a=False, transpose_b=False, transpose_out=True
    )
    assert params.operands == ("a", "b")
    assert a.shape[1] == K
    assert params.lda == K + 4  # the PARENT's width -- a.strides[0] // itemsize


def test_a_shape_derived_leading_dimension_computes_the_wrong_product():
    """RED leg: the corpus's greens are attributable to the strides."""
    flags = (False, False, True)
    a, b, out, _base, _sl, out_shape = build_case(flags, "col_slice", 5001)
    expected = reference_matmul(a, b, out_shape, flags)
    params = sgemm_params_for_arrays(
        a, b, out, transpose_a=False, transpose_b=False, transpose_out=True
    )
    simulate_sgemm(params, {"a": a, "b": b, "out": out}, lda=a.shape[1])
    assert not np.allclose(out, expected, rtol=1e-5, atol=1e-5)


def _params(a_shape, a_strides, b_shape, b_strides, o_shape, o_strides, **flags):
    kw = {"transpose_a": False, "transpose_b": False, "transpose_out": False}
    kw.update(flags)
    return sgemm_params(
        a_shape, a_strides, b_shape, b_strides, o_shape, o_strides, 4, **kw
    )


def test_a_non_2d_operand_is_refused():
    with pytest.raises(GemmMappingUnavailable, match="must be 2-D"):
        _params((M * K,), (4,), (K, N), (4 * N, 4), (M, N), (4 * N, 4))


def test_an_f_ordered_operand_is_refused():
    a = np.asfortranarray(np.zeros((M, K), dtype=np.float32))
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    with pytest.raises(GemmMappingUnavailable, match="unit-stride along its last axis"):
        sgemm_params_for_arrays(a, b, out)


def test_a_step_sliced_operand_is_refused():
    a = np.zeros((M, 2 * K), dtype=np.float32)[:, ::2]
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    assert a.shape == (M, K)
    with pytest.raises(GemmMappingUnavailable, match="unit-stride along its last axis"):
        sgemm_params_for_arrays(a, b, out)


def test_a_broadcast_operand_has_no_expressible_leading_dimension():
    a = np.broadcast_to(np.zeros(K, dtype=np.float32), (M, K))
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    with pytest.raises(GemmMappingUnavailable, match="smaller than its own width"):
        sgemm_params_for_arrays(a, b, out)


def test_disagreeing_contracted_extents_are_refused():
    with pytest.raises(GemmMappingUnavailable, match="contracted extents disagree"):
        _params((M, K), (4 * K, 4), (K + 1, N), (4 * N, 4), (M, N), (4 * N, 4))


def test_a_misshapen_destination_is_refused():
    with pytest.raises(GemmMappingUnavailable, match=r"'out' must be \(3, 5\)"):
        _params((M, K), (4 * K, 4), (K, N), (4 * N, 4), (N, M), (4 * M, 4))


def test_a_transposed_destination_wants_the_other_shape():
    with pytest.raises(GemmMappingUnavailable, match=r"'out' must be \(5, 3\)"):
        _params(
            (M, K), (4 * K, 4), (K, N), (4 * N, 4), (M, N), (4 * N, 4),
            transpose_out=True,
        )


def test_a_non_float32_itemsize_is_refused():
    with pytest.raises(GemmMappingUnavailable, match="float32-only"):
        sgemm_params(
            (M, K), (8 * K, 8), (K, N), (8 * N, 8), (M, N), (8 * N, 8), 8,
            transpose_a=False, transpose_b=False, transpose_out=False,
        )


def test_float64_arrays_are_refused_by_name():
    a = np.zeros((M, K))
    b = np.zeros((K, N), dtype=np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    with pytest.raises(GemmMappingUnavailable, match="operand 'a' is float64"):
        sgemm_params_for_arrays(a, b, out)


def test_require_float32_names_every_offending_spec():
    specs = (
        WorkspaceSpec("x", (2, 2), "float32"),
        WorkspaceSpec("theta", (2, 2), "float64"),
        WorkspaceSpec("y", (2, 2), "float64"),
    )
    with pytest.raises(GemmMappingUnavailable, match=r"\['theta', 'y'\]"):
        require_float32(specs)


def test_require_float32_passes_an_all_float32_plan():
    require_float32((WorkspaceSpec("x", (2, 2), "float32"),))


def _plain_plan(dtype):
    return LoweredPlan(
        steps=(matmul_step("c", a="a", b="b", out="c"),),
        mechanism="dense_gemm",
        determinism="test",
        inputs=(
            WorkspaceSpec("a", (M, K), dtype),
            WorkspaceSpec("b", (K, N), dtype),
        ),
        outputs=(WorkspaceSpec("c", (M, N), dtype),),
    )


def test_a_gemv_plan_refuses_capture():
    with pytest.raises(GemmMappingUnavailable, match="'gemv' operand class"):
        check_plan_capturable(_plain_plan("float32"), operand_class="gemv")


def test_a_float64_gemm_plan_refuses_capture():
    with pytest.raises(GemmMappingUnavailable, match="float32-only"):
        check_plan_capturable(_plain_plan("float64"), operand_class="gemm")


def test_a_float32_gemm_plan_is_accepted():
    check_plan_capturable(_plain_plan("float32"), operand_class="gemm")


# ============================================================================
# Library resolution: the refusals, and the laziness that keeps this file
# host-only.
# ============================================================================
_MAPS = """\
55a1b2c00000-55a1b2c21000 r--p 00000000 08:02 100 /usr/bin/python3.12
7f2a00000000-7f2a01000000 r--p 00000000 08:02 200 {p}
7f2a01000000-7f2a05000000 r-xp 01000000 08:02 200 {p}
7f2a05000000-7f2a05100000 rw-p 00000000 00:00 0
7f2a06000000-7f2a06100000 r--p 00000000 08:02 300 /lib/libcublasLt.so.12
"""


def test_many_mappings_of_one_library_are_one_path():
    text = _MAPS.format(p="/opt/cuda/lib/libcublas.so.12")
    assert mapped_library_paths(text) == ("/opt/cuda/lib/libcublas.so.12",)


def test_a_process_without_the_library_yields_nothing():
    assert mapped_library_paths(_MAPS.format(p="/lib/libfoo.so.1")) == ()


def test_two_different_files_are_two_paths():
    text = (
        _MAPS.format(p="/opt/conda/lib/libcublas.so.12")
        + "7f2a07000000-7f2a08000000 r-xp 0 08:02 400 /opt/pip/libcublas.so.12\n"
    )
    assert mapped_library_paths(text) == (
        "/opt/conda/lib/libcublas.so.12",
        "/opt/pip/libcublas.so.12",
    )


def test_a_similarly_named_library_is_not_a_match():
    assert mapped_library_paths(_MAPS.format(p="/lib/libcublasLt.so.12")) == ()


def test_the_pip_wheel_s_exact_soname_matches():
    """The bare ``libcublas.so.12`` name a pip wheel maps (the equality arm
    the match already had, kept working by the fix)."""
    text = _MAPS.format(p="/opt/pip/lib/libcublas.so.12")
    assert mapped_library_paths(text) == ("/opt/pip/lib/libcublas.so.12",)


def test_a_conda_style_versioned_file_matches_the_soname():
    """The real gap this row closes: conda maps the ACTUAL file behind the
    soname, further versioned (``libcublas.so.12.9.1.4``), which a bare
    equality check never matched -- every captured GEMM raised
    ``CublasResolutionError`` in a conda env before this fix."""
    text = _MAPS.format(p="/opt/conda/lib/libcublas.so.12.9.1.4")
    assert mapped_library_paths(text) == ("/opt/conda/lib/libcublas.so.12.9.1.4",)


def test_a_bare_numeric_extension_of_the_soname_is_not_a_match():
    """The trap the fix must NOT fall into: ``startswith(soname)`` alone
    (no separating dot) would also accept ``libcublas.so.120`` -- a
    DIFFERENT, unversioned-looking name that merely happens to start with
    the same characters. Only ``soname`` extended by a real ``.`` suffix
    counts."""
    text = _MAPS.format(p="/lib/libcublas.so.120")
    assert mapped_library_paths(text) == ()


def test_a_versioned_cublasLt_is_still_not_a_match():
    assert mapped_library_paths(_MAPS.format(p="/lib/libcublasLt.so.12.9.1.4")) == ()


@pytest.fixture
def stub_maps(monkeypatch):
    def install(paths):
        monkeypatch.setattr(gemm_capture, "_warm_cupy_cublas", lambda: None)
        monkeypatch.setattr(gemm_capture, "mapped_library_paths", lambda text: paths)

    return install


def test_resolution_refuses_when_nothing_is_mapped(stub_maps):
    stub_maps(())
    with pytest.raises(CublasResolutionError, match="no libcublas.so.12 or libcublas.so.13 is mapped"):
        resolve_cublas_path()


def test_resolution_refuses_an_ambiguous_process(stub_maps):
    stub_maps(("/opt/a/libcublas.so.12", "/opt/b/libcublas.so.12"))
    with pytest.raises(CublasResolutionError, match="2 different"):
        resolve_cublas_path()


def test_resolution_takes_the_cuda_13_cublas(stub_maps):
    stub_maps(("/opt/a/libcublas.so.13.0.0.19",))
    assert resolve_cublas_path() == "/opt/a/libcublas.so.13.0.0.19"


def test_resolution_returns_the_sole_match(stub_maps):
    stub_maps(("/opt/a/libcublas.so.12",))
    assert resolve_cublas_path() == "/opt/a/libcublas.so.12"


def test_importing_the_module_touches_no_device():
    """Structural: no module-level statement calls the resolving entry points,
    and cupy is imported only inside a function."""
    source = Path(gemm_capture.__file__).read_text()
    tree = ast.parse(source)
    forbidden = {"_warm_cupy_cublas", "resolve_cublas_path", "load_cublas"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                name = getattr(call.func, "id", None) or getattr(
                    call.func, "attr", None
                )
                assert name not in forbidden, f"module-level {name}()"
    for line in source.splitlines():
        if "import cupy" in line:
            assert line.startswith(" "), f"module-level cupy import: {line!r}"


# ============================================================================
# 2. The plan/capture seam -- adapted from the code generator's
# plan-capture-seam test: HAND-BUILT records replace recognize(), since
# eagle has no tracer to derive one from (see eagle.gemm.contraction's
# docstring). lower_dense_gemm consumes a record by duck-typed attribute
# access, so the substitution changes nothing about what is exercised.
# ============================================================================
#: The f32 reassociation band -- a GEMM sums ``K`` products, so two summation
#: orders differ by at most ``K`` roundings of the result's own magnitude.
#: Copied from the code generator's own plan-capture-seam test's ``_band`` -- the
#: settled band for exactly this comparison, not invented here.
_EPS32 = float(np.finfo(np.float32).eps)


def _band(reference):
    envelope = float(np.max(np.abs(reference))) or 1.0
    return K * _EPS32 * envelope


def _named_operand(name, dims, reduce_dim):
    """A dense :class:`ContractionOperand` over two DECLARED axes, one of
    which is the reduce axis -- the shape ``recognize(_staged_pair)`` /
    ``recognize(_row_sum)`` would themselves have returned for a plain
    product-of-reads / bare-read addend (the code generator's own record for these two
    kernels, transcribed by hand)."""
    bound = 1
    for _label, size, _stride in dims:
        bound *= size
    return ContractionOperand(
        name=name,
        role="named",
        dims=dims,
        bound=bound,
        reads=1,
        reduce_stride=next(s for lbl, _sz, s in dims if lbl == reduce_dim),
        reduce_dim=reduce_dim,
        stride_labels=(f"{name}_{reduce_dim}_stride",),
        dense=True,
        refusals=(),
    )


def _staged_pair_record():
    """Mirrors ``recognize(_staged_pair)`` in the code generator's seam test: two Table
    operands declared ``(k, free)``, BOTH contracted over the FIRST axis --
    the staged-buffer orientation whose adjoints construct ``transpose_out``
    legs."""
    a = _named_operand("a", (("k", K, 1), ("row", M, K)), reduce_dim="k")
    b = _named_operand("b", (("k", K, 1), ("col", N, K)), reduce_dim="k")
    return RecognizedContraction(
        carry="_pf_acc",
        reduce_var="e",
        reduce_extent=K,
        reduce_start=0,
        reduce_step=1,
        operands=(a, b),
        operand_class="gemm",
        term_form="product_of_reads",
        outputs=(("y", "float32"),),
        dense=True,
        refusals=(),
        staged_fanin=(),
        fanin_source="none",
        adjoint_accumulates=(),
    )


def _row_sum_record():
    """Mirrors ``recognize(_row_sum)``: a single operand, bare-read, gemv
    class -- the row-sum shape the fences record but do not fund a captured
    form for."""
    x = _named_operand("x", (("sample", M, K), ("col", K, 1)), reduce_dim="col")
    return RecognizedContraction(
        carry="_pf_acc",
        reduce_var="j",
        reduce_extent=K,
        reduce_start=0,
        reduce_step=1,
        operands=(x,),
        operand_class="gemv",
        term_form="bare_read",
        outputs=(("y", "float32"),),
        dense=True,
        refusals=(),
        staged_fanin=(),
        fanin_source="none",
        adjoint_accumulates=(),
    )


SEED = 20260821


def _seam_plan(direction, dtype="float32"):
    return lower_dense_gemm(_staged_pair_record(), direction=direction, dtype=dtype)


def _seam_buffers(plan, *, seed=SEED):
    rng = np.random.default_rng(seed)
    written = {name for step in plan.steps for name in step.writes}
    out = {}
    for spec in plan.required:
        if spec.name in written:
            out[spec.name] = np.zeros(spec.shape, dtype=spec.dtype)
        else:
            out[spec.name] = rng.standard_normal(spec.shape).astype(spec.dtype)
    return out


@pytest.mark.parametrize("direction", ["forward", "adjoint"])
def test_every_matmul_step_publishes_a_spec_that_rebuilds_it(direction):
    import dataclasses

    plan = _seam_plan(direction)
    assert plan.kinds == ("matmul",) * len(plan.steps)

    for step in plan.steps:
        spec = step.matmul
        assert isinstance(spec, MatmulSpec), f"{step.name!r} published no MatmulSpec"
        assert spec.out in step.writes
        assert set((spec.a, spec.b)) == set(step.reads)

        original = _seam_buffers(plan)
        rebuilt = {name: buf.copy() for name, buf in original.items()}
        step.run(original, "host")
        matmul_step(step.name, **dataclasses.asdict(spec)).run(rebuilt, "host")
        np.testing.assert_array_equal(
            rebuilt[spec.out],
            original[spec.out],
            err_msg=f"{step.name!r}'s published spec rebuilds a DIFFERENT step",
        )


def test_the_published_flags_are_the_ones_the_orientation_implies():
    forward = _seam_plan("forward").steps[0].matmul
    a_bar, b_bar = (step.matmul for step in _seam_plan("adjoint").steps)

    assert (forward.transpose_a, forward.transpose_b, forward.transpose_out) == (
        True, False, False,
    )
    assert (a_bar.transpose_a, a_bar.transpose_b, a_bar.transpose_out) == (
        False, True, True,
    )
    assert (b_bar.transpose_a, b_bar.transpose_b, b_bar.transpose_out) == (
        True, True, True,
    )


def test_the_new_field_does_not_change_what_a_step_compares_equal_to():
    common = dict(kind="matmul", reads=("a", "b"), writes=("y",))
    plain = PlanStep(name="y", **common, device_fn=None, host_fn=None)
    described = PlanStep(
        name="y",
        **common,
        device_fn=None,
        host_fn=None,
        matmul=MatmulSpec(a="a", b="b", out="y"),
    )
    assert plain == described
    assert hash(plain) == hash(described)


@pytest.mark.parametrize("direction", ["forward", "adjoint"])
def test_the_captured_calls_reproduce_the_plans_own_answer(direction):
    plan = _seam_plan(direction)
    expected = _seam_buffers(plan)
    plan("host", expected)

    actual = _seam_buffers(plan)
    calls = captured_gemm_calls(plan, actual, operand_class="gemm")
    assert tuple(call.name for call in calls) == tuple(s.name for s in plan.steps)

    for call in calls:
        simulate_sgemm(call.params, dict(call.operands))

    for spec in plan.outputs:
        np.testing.assert_allclose(
            actual[spec.name],
            expected[spec.name],
            rtol=1e-5,
            atol=_band(expected[spec.name]),
            err_msg=f"the captured mapping disagrees with the plan on {spec.name!r}",
        )


def test_the_calls_bind_the_callers_own_buffers_and_allocate_nothing():
    plan = _seam_plan("adjoint")
    buffers = _seam_buffers(plan)
    for call in captured_gemm_calls(plan, buffers, operand_class="gemm"):
        spec = next(s.matmul for s in plan.steps if s.name == call.name)
        assert call.operands["a"] is buffers[spec.a]
        assert call.operands["b"] is buffers[spec.b]
        assert call.operands["out"] is buffers[spec.out]


def test_a_sliced_operand_is_mapped_off_its_strides_not_its_shape():
    plan = _seam_plan("forward")
    contiguous = _seam_buffers(plan)
    expected = {name: buf.copy() for name, buf in contiguous.items()}
    plan("host", expected)

    rng = np.random.default_rng(SEED + 7)
    sliced = {}
    for spec in plan.required:
        parent = rng.standard_normal(
            (spec.shape[0], spec.shape[1] + 4)
        ).astype(spec.dtype)
        view = parent[:, 1 : 1 + spec.shape[1]]
        view[...] = contiguous[spec.name]
        sliced[spec.name] = view
        assert view.strides[0] != view.shape[1] * view.itemsize

    for call in captured_gemm_calls(plan, sliced, operand_class="gemm"):
        simulate_sgemm(call.params, dict(call.operands))

    for spec in plan.outputs:
        np.testing.assert_allclose(
            sliced[spec.name], expected[spec.name],
            rtol=1e-5, atol=_band(expected[spec.name]),
        )


def test_a_gemv_plan_refuses():
    plan = lower_dense_gemm(_row_sum_record(), direction="forward", dtype="float32")
    with pytest.raises(GemmMappingUnavailable, match="gemv"):
        captured_gemm_calls(plan, _seam_buffers(plan), operand_class="gemv")


def test_a_float64_plan_refuses_even_though_that_is_the_lowering_default():
    plan = lower_dense_gemm(_staged_pair_record(), direction="forward")
    assert all(spec.dtype == "float64" for spec in plan.required)
    with pytest.raises(GemmMappingUnavailable, match="float32"):
        captured_gemm_calls(plan, _seam_buffers(plan), operand_class="gemm")


def test_a_plan_carrying_a_step_that_is_not_a_matmul_refuses():
    spec_a = WorkspaceSpec("a", (K, M), "float32")
    spec_y = WorkspaceSpec("y", (M, M), "float32")
    plan = LoweredPlan(
        steps=(
            matmul_step("y", a="a", b="a", out="y", transpose_a=True),
            assembly_step(
                "scale", reads=("y",), writes=("y",),
                fn=lambda buffers, xp: xp.multiply(buffers["y"], 2.0, out=buffers["y"]),
            ),
        ),
        mechanism="dense_gemm",
        determinism="test",
        inputs=(spec_a,),
        outputs=(spec_y,),
    )
    with pytest.raises(GemmMappingUnavailable, match="assembly"):
        captured_gemm_calls(
            plan,
            {"a": np.zeros((K, M), np.float32), "y": np.zeros((M, M), np.float32)},
            operand_class="gemm",
        )


def test_an_unbound_buffer_is_refused_before_any_mapping_happens():
    plan = _seam_plan("forward")
    buffers = _seam_buffers(plan)
    missing = plan.steps[0].matmul.b
    del buffers[missing]
    with pytest.raises(GemmMappingUnavailable, match=missing):
        captured_gemm_calls(plan, buffers, operand_class="gemm")


# ============================================================================
# 3. GPU row: one Sgemm through a REAL CUDA stream capture, replayed,
# vs. the uncaptured cupy matmul of the same operands.
# ============================================================================
@pytest.mark.gpu
def test_a_captured_sgemm_replays_within_the_reassociation_band_of_cupy_matmul():
    import cupy as cp

    from eagle.gemm.capture import CublasCaptureHandle

    rng = np.random.default_rng(90210)
    a_h = rng.standard_normal((M, K)).astype(np.float32)
    b_h = rng.standard_normal((K, N)).astype(np.float32)
    a = cp.asarray(a_h)
    b = cp.asarray(b_h)
    out = cp.zeros((M, N), dtype=cp.float32)

    expected = cp.matmul(a, b)
    cp.cuda.runtime.deviceSynchronize()

    stream = cp.cuda.Stream(non_blocking=True)
    handle = CublasCaptureHandle(stream.ptr)
    params = sgemm_params_for_arrays(a, b, out)
    try:
        with stream:
            handle.warm(params, {"a": a, "b": b, "out": out})
            out[...] = 0.0  # warm() wrote the destination; re-clear before capture
            stream.begin_capture()
            handle.sgemm(params, {"a": a, "b": b, "out": out})
            graph = stream.end_capture()
            graph.launch(stream=stream)
        stream.synchronize()

        np.testing.assert_allclose(
            cp.asnumpy(out),
            cp.asnumpy(expected),
            rtol=1e-5,
            atol=_band(cp.asnumpy(expected)),
            err_msg=(
                "a captured Sgemm and cupy's own uncaptured matmul disagree "
                "beyond the K*eps float32 reassociation band"
            ),
        )

        # A second replay must reproduce the FIRST replay's own answer -- the
        # graph's determinism claim checked directly,
        # not merely re-derived by comparing a buffer to itself.
        first_replay = cp.asnumpy(out).copy()
        out[...] = 0.0
        graph.launch(stream=stream)
        stream.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(out), first_replay)
    finally:
        handle.destroy()
