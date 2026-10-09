# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The deployed (PTX-loaded) kernel launchers -- no nvcc, no aether headers
at runtime.

:class:`LoadedKernel` factors out the sidecar read + binary ABI guard +
driver-module load that :class:`LoadedVector` and :class:`LoadedPure`
share, and inherits the framework-polymorphic launch skeleton from
:class:`~eagle.launch.LaunchMixin`; each subclass supplies only its
kind-specific sidecar fields, ``_allowed`` set, and output handling. The
coercion + launch bodies come from :mod:`eagle.marshal` / :mod:`eagle.launch`,
shared with every launcher family so they can never drift.
"""

from __future__ import annotations

import json
import pathlib
from collections import namedtuple

from . import _layout
from .abi import check_aether_abi
from .launch import (
    LaunchMixin,
    launch,
    pure_origin,
    pure_prepare,
)
from .sidecar import read_params, read_terminated_readonly, validate_sidecar

#: The sidecar's minimal Mutable descriptor: name / dtype / width / shape,
#: enough to size N and pick the ABI (vector/matrix -> GRef, scalar/int ->
#: flat handle). ``shape`` is ``(rows, cols)`` for a matrix slot, else ``None``.
_MutableSpec = namedtuple(
    "_MutableSpec", ("name", "dtype", "width", "shape"), defaults=(None,)
)


class LoadedKernel(LaunchMixin):
    """Common scaffolding for a PTX-loaded launcher (no nvcc at runtime):
    reads the shared sidecar fields, rejects a binary-ABI mismatch, and
    driver-loads the module. Subclasses read their own fields, build
    ``self._allowed``, and define ``launch`` / ``__call__`` over the
    inherited :meth:`~eagle.launch.LaunchMixin._dispatch`. The parsed
    sidecar is kept on ``self._meta``."""

    def __init__(self, ptx_path, *, abi_kind):
        ptx_path = pathlib.Path(ptx_path)
        # The artifact's own id; the stem is the id by construction (the
        # code generator names every artifact file f"{id}{ext}").
        self.id = ptx_path.stem
        self._meta = json.loads(ptx_path.with_suffix(".json").read_text())
        # Reject a binary-ABI mismatch before binding any by-value struct.
        check_aether_abi(self._meta, kind=abi_kind, name=ptx_path.name)
        #: The artifact's own ABI generation, read by :meth:`_refuse_v2_door`
        #: and by ``eagle.plan.plan``'s legacy-partition bridge.
        self.abi_tag = self._meta.get("aether_abi")
        # Run BEFORE the driver module load below, so a rejected artifact
        # never reaches cupy.
        validate_sidecar(self._meta, name=ptx_path.name)
        self.derivative = self._meta.get("derivative")  # a VJP/JVP artifact, or None
        # "float64" when absent -- pre-dtype sidecars.
        self.scalar_type = self._meta.get("scalar_type", "float64")
        self.kernel_name = self._meta["kernel"]
        self.vector_inputs = tuple(self._meta["vector_inputs"])
        self.params = read_params(self._meta, name=ptx_path.name)
        #: The binding names of :attr:`params`.
        self.param_names = tuple(p.name for p in self.params)
        self.per_sample = tuple(self._meta.get("per_sample", ()))
        self.arg_spec = [tuple(x) for x in self._meta["arg_spec"]]
        # Opt-in: lets an omitted ``terminated`` mask be served from a shared
        # buffer (:func:`eagle.marshal.coerce_terminated`). Absent => False.
        self.terminated_readonly = read_terminated_readonly(
            self._meta, name=ptx_path.name
        )

        import cupy as cp

        self.module = cp.RawModule(path=str(ptx_path))  # driver load (JITs PTX)
        self.fn = self.module.get_function(self.kernel_name)

    def _refuse_v2_door(self, door: str) -> None:
        """Refuse an ``aether-abi/2`` artifact at this v1 launcher, loudly.

        A v2 entry's parameters are the by-value mirrors followed by the
        partition triple (``base``, ``count``, ``nSamples``), which this
        launcher does not pack; handing it a v1 block is a wrong-length
        argument array (observed on real hardware as a SIGSEGV). The v2
        door is ``eagle.plan.plan(...).bind(...)``."""
        from . import exec as eexec

        if self.abi_tag != eexec.ABI_TAG_V2:
            return
        raise ValueError(
            f"{type(self).__name__}.{door}: this artifact declares "
            f"{self.abi_tag!r}, whose entry takes the by-value mirrors followed "
            "by the partition triple {base, count, nSamples}; this is the "
            f"{eexec.ABI_TAG_V1!r} launcher and it packs neither the triple nor "
            "the v2 role vocabulary. Drive it through the v2 door instead: "
            "eagle.plan.plan(plugin, structure=eagle.exec.DeviceKernel)"
            ".bind(**planes).launch()"
        )

    def _require_pattern(self, ptx_path, expected: str, what: str, *, default=None):
        """Reject an artifact of the wrong kind by its sidecar ``pattern`` tag."""
        pat = self._meta.get("pattern", default)
        if pat != expected:
            raise ValueError(
                f"{pathlib.Path(ptx_path).name!r} is not {what} (pattern {pat!r})"
            )


class LoadedVector(LoadedKernel):
    """A precompiled vector kernel loaded from a PTX (+ sidecar) artifact --
    no nvcc.

    Drop-in equivalent to the in-process compiled vector-kernel launcher."""

    def __init__(self, ptx_path):
        super().__init__(ptx_path, abi_kind="vector plugin")
        meta = self._meta
        # An untagged sidecar (an older artifact) is a vector kernel by definition.
        self._require_pattern(
            ptx_path, "vector", "a vector plugin", default="vector"
        )
        self.accumulate = bool(meta.get("accumulate", True))
        self._table_counts = {
            b["name"]: int(b["count"])
            for b in meta.get("buffers", [])
            if b["kind"] == "lookup"
        }
        self.lookup_tables = tuple(self._table_counts)
        self._allowed = (
            set(self.vector_inputs)
            | set(self.param_names)
            | set(self.per_sample)
            | set(self.lookup_tables)
            | {"terminated"}
        )

    def launch(self, *, out, grid=None, block=None, **kw):
        """Capturable single launch on the current stream (cupy arrays
        only). Inputs, ``out``, and any lookup tables must be pre-allocated
        and passed by name. ``block=None`` defers to eagle's launch-policy
        resolver."""
        self._refuse_v2_door("launch")
        self._reject_unknown(kw, self._allowed)
        return launch(
            "vector",
            self.fn,
            self.arg_spec,
            self.vector_inputs,
            self.params,
            self.per_sample,
            out=out,
            kw=kw,
            grid=grid,
            block=block,
        )

    @_layout.door
    def __call__(self, *, out=None, **kw):
        """Allocate, launch, and return the contribution, in the caller's
        tensor framework (numpy in -> numpy out, cupy/torch -> same via
        DLPack): blocking for numpy, non-blocking on the framework's stream
        otherwise. Pass ``out=`` a pre-allocated device buffer to fill it
        in place -- see :func:`~eagle.interop.as_out_buffer`."""
        import cupy as cp

        from . import interop
        from .marshal import (
            coerce_per_sample,
            coerce_tables,
            coerce_terminated,
            coerce_uniforms,
            coerce_vec_inputs,
            require_n,
        )

        self._refuse_v2_door("__call__")
        self._reject_unknown(kw, self._allowed)
        origin = self._origin_from(kw, self.vector_inputs)

        def prepare():
            from .dtypes import np_dtype

            dt = np_dtype(self.scalar_type)
            vec, n = coerce_vec_inputs(kw, self.vector_inputs, dt=dt)
            n = require_n(n)
            ps, n = coerce_per_sample(kw, self.per_sample, n, dt=dt)
            uniforms = coerce_uniforms(kw, self.params, dt=dt)
            tables = coerce_tables(kw, self._table_counts, dt=dt)
            term = coerce_terminated(
                kw, n, readonly_mask=self.terminated_readonly
            )
            # Fill a provided device buffer in place (zeroed first), else allocate.
            if out is None:
                out_cu = cp.zeros((3, n), dtype=dt)
            else:
                out_cu = interop.as_out_buffer(out, (3, n), dt)
                out_cu.fill(0)
            self.launch(
                out=out_cu, **vec, **ps, **uniforms, **tables, terminated=term
            )
            return out_cu

        def export(out_cu):
            return out if out is not None else origin.from_cupy(out_cu)

        return self._dispatch(origin=origin, prepare=prepare, export=export)


class LoadedPure(LoadedKernel):
    """A precompiled pure kernel loaded from a PTX (+ sidecar) artifact --
    no nvcc. Drop-in equivalent to the in-process compiled pure launcher via
    :mod:`eagle.launch`. The ``Mutable`` buffers are the handoff
    (read-modify-written in place), so there is no separate ``out=``."""

    def __init__(self, ptx_path):
        super().__init__(ptx_path, abi_kind="pure plugin")
        meta = self._meta
        self._require_pattern(ptx_path, "pure", "a pure plugin")
        self._mutables_decl = [
            _MutableSpec(
                m["name"],
                m["dtype"],
                int(m.get("width", 1) or 1),
                tuple(m["shape"]) if m.get("shape") is not None else None,
            )
            for m in meta.get("mutables", [])
        ]
        self._mutables = tuple(m.name for m in self._mutables_decl)
        self._mutable_defaults = {
            m["name"]: m.get("default") for m in meta.get("mutables", [])
        }
        self._table_counts = {
            b["name"]: int(b["count"])
            for b in meta.get("buffers", [])
            if b["kind"] == "lookup"
        }
        self.lookup_tables = tuple(self._table_counts)
        # matrix content (empty when the kernel has none): ``mat_in`` order
        # + each one's ``(R, C)``, and the width-honest vector-input widths
        # (a synthesized VJP/JVP seed may not be a 3-vector).
        self.matrix_inputs = tuple(meta.get("matrix_inputs", ()))
        self._mat_shapes = {
            k: tuple(v) for k, v in meta.get("mat_shapes", {}).items()
        }
        self._vec_widths = {k: int(v) for k, v in meta.get("vec_widths", {}).items()}
        # Wide content is additive-only, defaulting to empty: ``wide_inputs``
        # is a per-sample flat buffer; ``wide_outputs`` a VJP-derived
        # kernel's gradient-scatter target.
        self.wide_inputs = tuple(meta.get("wide_inputs", ()))
        self.wide_outputs = tuple(meta.get("wide_outputs", ()))
        # a matrix Mutable's buffer is bound flat (R*C, N); reshaped to
        # (R, C, N) on export.
        self._mut_mat_shapes = {
            m.name: m.shape for m in self._mutables_decl if m.dtype == "matrix"
        }
        self._allowed = (
            set(self.vector_inputs)
            | set(self.matrix_inputs)
            | set(self.param_names)
            | set(self.per_sample)
            | set(self._mutables)
            | set(self.lookup_tables)
            | set(self.wide_inputs)
            | set(self.wide_outputs)
            | {"terminated"}
        )

    def launch(self, *, grid=None, block=None, **kw):
        """Capturable single pure launch on the current stream (cupy arrays
        only). The writable ``Mutable`` buffers, any inputs, lookup tables,
        and the ``terminated`` mask must be pre-allocated;
        read-modify-written in place. ``block=None`` defers to eagle's
        launch-policy resolver."""
        self._refuse_v2_door("launch")
        self._reject_unknown(kw, self._allowed)
        return launch(
            "pure",
            self.fn,
            self.arg_spec,
            self.vector_inputs,
            self.params,
            self.per_sample,
            mutables_decl=self._mutables_decl,
            kw=kw,
            grid=grid,
            block=block,
            matrix_inputs=self.matrix_inputs,
        )

    @_layout.door
    def __call__(self, **kw):
        """Launch the pure kernel and return the updated ``Mutable`` buffers
        as a dict, in the caller's framework."""
        self._refuse_v2_door("__call__")
        self._reject_unknown(kw, self._allowed)
        if not self._mutables:
            raise TypeError("a pure kernel without a Mutable cannot be launched")
        origin = pure_origin(
            kw,
            vector_inputs=(*self.vector_inputs, *self.matrix_inputs),
            mutable_names=self._mutables,
            per_sample=self.per_sample,
        )

        def prepare():
            from .dtypes import np_dtype

            dt = np_dtype(self.scalar_type)
            return pure_prepare(
                self.fn,
                self.arg_spec,
                vector_inputs=self.vector_inputs,
                per_sample=self.per_sample,
                params=self.params,
                mutables_decl=self._mutables_decl,
                mutable_defaults=self._mutable_defaults,
                lookup_counts=self._table_counts,
                kw=kw,
                mat_shapes=self._mat_shapes,
                vec_widths=self._vec_widths,
                wide_inputs=self.wide_inputs,
                wide_outputs=self.wide_outputs,
                dt=dt,
                readonly_mask=self.terminated_readonly,
            )

        def export(mut):
            # a matrix slot is bound flat (R*C, N); hand it back as a
            # zero-copy (R, C, N) reshape view. A no-op for a wide name
            # (absent from ``_mut_mat_shapes``).
            def _shaped(name):
                shape = self._mut_mat_shapes.get(name)
                if shape is None:
                    return mut[name]
                return mut[name].reshape(*shape, mut[name].shape[-1])

            return {
                name: origin.from_cupy(_shaped(name))
                for name in (*self._mutables, *self.wide_outputs)
            }

        return self._dispatch(origin=origin, prepare=prepare, export=export)
