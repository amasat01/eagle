# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Sample-major planes: the one place eagle adapts them to its native layout.

A per-sample plane of width ``w >= 2`` is COMPONENT-MAJOR, ``(w, N)``: one
contiguous row per component, which is what makes a per-component access
coalesced. Most NumPy/PyTorch/JAX code stores per-sample vectors SAMPLE-MAJOR,
``(N, w)``. Every eagle door that binds a per-sample plane calls :func:`adapt`
on it before reading the sample count, so the rules below hold on host and
device alike:

* A plane is classified against its declared HEAD (``(w,)``, or ``(R, C)`` for a
  matrix): ``head + (N,)`` is native; ``(N,) + head`` is sample-major. A shape
  that matches BOTH (``N == w``; ``N == R == C`` for a square matrix) is
  AMBIGUOUS and is refused, never guessed: it names the argument, its shape,
  both readings and the two zero-copy fixes below. Scalar planes, buffers whose
  length is a runtime quantity (lookup, wide, accumulate) and ``Param``
  uniforms are never adapted.
* The caller says which axis holds the samples in two ways. PER CALL:
  ``layout="samples_first"`` (``(N, w)``) or ``layout="samples_last"``
  (``(w, N)``, native) on the door resolves every ambiguous plane of that call;
  planes that are not ambiguous behave as before. PER ARRAY:
  ``eagle.samples_first(x)`` / ``eagle.samples_last(x)`` wrap one argument and
  are honoured for ANY shape; a marker that contradicts the shape is refused
  naming the argument, and a marker beats the per-call ``layout``. Neither
  copies. The marker protocol is shared with HAWK (which imports no eagle): any
  object carrying ``__raptor_samples_axis__`` (``"first"`` or ``"last"``) and
  the wrapped array as ``.array`` is a marker, so each package accepts the
  other's. A marker on one-sample values is ignored.
* A sample-major plane whose native permutation (``x.T``, or the leading axis
  moved last) is C-contiguous already holds the native bytes: it is bound as
  that view, zero-copy, with no warning, input or output.
* Any other sample-major plane is copied into a native buffer and a
  :class:`LayoutWarning` names the argument, both shapes, that zero-copy is
  lost for it, and how to avoid it. A written plane is copied in before the
  launch and copied back into the caller's array after it; a read-only input is
  copied in only. The warning is issued once per call site and argument, before
  any copy, so ``warnings.filterwarnings("error", category=eagle.LayoutWarning)``
  turns every copy into an error with nothing half-done.
* ONE SAMPLE is a value whose shape IS the plane's per-sample head: ``()`` (a
  Python number or a 0-d array) for a scalar plane, ``(w,)`` for a vector, a
  matrix's ``(R, C)`` or flat ``(R*C,)``. It is classified ``"single"`` and runs
  as a batch of one through its zero-copy view ``value.reshape(head + (1,))``
  (a number, input only, becomes a one-cell array). A shape that is BOTH a
  batch and a single sample is a batch: ``(1,)`` on a scalar plane, ``(w, 1)``,
  ``(1, w)`` and ``(w, w)`` are never one sample. A door opts in to single
  samples (``adapt(..., single=True)``); the others bind such a value as given,
  so their own shape check still names it.

HAWK's host runtime restates the zero-copy half of these rules
(``hawk.runtime.plane_layout``; hawk imports no eagle), the ambiguity refusal and
the markers, and refuses the copy half; ``tests/test_layout.py`` asserts the two
classify identically.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import sys
import warnings

__all__ = ["LayoutWarning", "LayoutError", "classify", "is_single", "adapt",
           "Adapted", "samples_first", "samples_last", "split_marks",
           "check_layout", "scope", "door", "rewrap", "sample_extent"]

#: The marker protocol (shared with hawk, which imports no eagle): an object
#: with this attribute set to ``"first"`` or ``"last"`` and the wrapped array
#: as ``.array`` declares which axis of that array holds the samples.
MARK_ATTR = "__raptor_samples_axis__"
LAYOUTS = ("samples_first", "samples_last")


class LayoutWarning(UserWarning):
    """A sample-major plane was copied into eagle's component-major layout.

    Filterable like any warning; make it an error with
    ``warnings.filterwarnings("error", category=eagle.LayoutWarning)``."""


class LayoutError(ValueError):
    """A plane's layout is ambiguous, or a marker contradicts its shape."""


class _Marked:
    """One array with the axis that holds its samples declared. Built by
    :func:`samples_first` / :func:`samples_last`; nothing else touches it."""

    __slots__ = ("array", "__raptor_samples_axis__")

    def __init__(self, array, axis):
        self.array = array
        self.__raptor_samples_axis__ = axis

    def __repr__(self):
        return f"eagle.samples_{self.__raptor_samples_axis__}({self.array!r})"


def samples_first(x):
    """Mark ``x`` as sample-major: its FIRST axis holds the samples, ``(N, w)``
    (a matrix plane ``(N, R, C)``). Zero-copy, honoured for any shape; passes
    through any eagle or hawk door that binds per-sample planes, and beats the
    door's ``layout=``. A shape that contradicts it is refused."""
    return _Marked(x, "first")


def samples_last(x):
    """Mark ``x`` as component-major (native): its LAST axis holds the samples,
    ``(w, N)``. Zero-copy, honoured for any shape; passes through any eagle or
    hawk door and beats the door's ``layout=``. A shape that contradicts it is
    refused."""
    return _Marked(x, "last")


def _mark_of(value):
    """``"first"``/``"last"`` when ``value`` is a marker of either package."""
    axis = getattr(value, MARK_ATTR, None)
    if axis in ("first", "last") and hasattr(value, "array"):
        return axis
    return None


def check_layout(layout) -> None:
    """Refuse a per-call ``layout`` that is neither ``None`` nor one of
    ``"samples_first"``, ``"samples_last"``."""
    if layout is not None and layout not in LAYOUTS:
        raise LayoutError(
            f"eagle: layout={layout!r} is not one of 'samples_first' (planes "
            f"are (N, w)) or 'samples_last' (planes are (w, N)), or None")


def split_marks(kw, layout=None):
    """``(plain, axes)``: ``kw`` with every marker unwrapped to its array, and
    ``{name: "first"|"last"}`` for the marked names. ``layout`` is validated.
    ``kw`` itself is returned when nothing is marked."""
    check_layout(layout)
    axes = {}
    for name, value in kw.items():
        axis = _mark_of(value)
        if axis is not None:
            axes[name] = axis
    if not axes:
        return kw, axes
    return {n: (v.array if n in axes else v) for n, v in kw.items()}, axes


#: The active ``(layout, {name: axis})`` of the door being served, per thread
#: and task (a :class:`contextvars.ContextVar`, so concurrent calls never see
#: each other's choice). :func:`adapt` falls back to it.
_scope = contextvars.ContextVar("eagle_layout_scope", default=(None, {}))


@contextlib.contextmanager
def scope(layout, axes):
    """Make ``layout`` (the call's) and ``axes`` (``{name: "first"|"last"}``,
    the per-array markers) the choice every :func:`adapt` inside the block
    falls back to. Nests: an inner scope adds to the outer one."""
    check_layout(layout)
    outer_layout, outer_axes = _scope.get()
    token = _scope.set((layout or outer_layout, {**outer_axes, **axes}))
    try:
        yield
    finally:
        _scope.reset(token)


def door(fn):
    """Give a public door that binds per-sample planes a ``layout=`` keyword and
    marker support: markers among its keyword arguments are unwrapped, and the
    call runs inside :func:`scope`."""

    @functools.wraps(fn)
    def wrapper(*args, layout=None, **kw):
        plain, axes = split_marks(kw, layout)
        if layout is None and not axes:
            return fn(*args, **plain)
        with scope(layout, axes):
            return fn(*args, **plain)

    return wrapper


def rewrap(planes, axes):
    """:func:`split_marks`'s inverse: ``planes`` with each name in ``axes``
    wrapped in its marker again."""
    if not axes:
        return planes
    return {n: (_Marked(v, axes[n]) if n in axes else v)
            for n, v in planes.items()}


def sample_extent(name, value, head, axis=None, layout=None) -> int:
    """The sample count N of a per-sample plane whose per-sample head is
    ``head``, once its orientation is settled (marker, call layout, or the
    unambiguous shape). Raises :class:`LayoutError` for an ambiguous plane or a
    contradicted marker, naming ``name``."""
    shape = tuple(int(e) for e in value.shape)
    try:
        strides = _element_strides(value)
    except (AttributeError, TypeError):
        strides = (1,) * len(shape)
    layout_scope, axes = _scope.get()
    kind = classify(shape, strides, head, axis or axes.get(name),
                    layout or layout_scope)
    if kind == "ambiguous":
        raise LayoutError(_ambiguous_message(name, shape, head))
    if kind == "conflict":
        raise LayoutError(_conflict_message(name, shape, head, axis or axes.get(name)))
    if kind in ("view", "copy"):
        return shape[0]
    return shape[-1] if shape else 1


def _c_contiguous(shape, strides) -> bool:
    """Whether ``strides`` (in ELEMENTS) are the C-contiguous ones for ``shape``.
    A unit-extent axis's stride is irrelevant, as numpy and torch both treat it."""
    expect = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent != 1 and stride != expect:
            return False
        expect *= extent
    return True


def _width(head) -> int:
    w = 1
    for h in head:
        w *= h
    return w


def is_single(shape, head) -> bool:
    """Whether ``shape`` is ONE sample of a plane whose declared per-sample
    head is ``head`` (``()``/``(1,)`` scalar, ``(w,)``, ``(R, C)``). A shape
    that is also a batch is a batch: ``(1,)``, ``(w, 1)``, ``(1, w)`` and
    ``(w, w)`` never are one sample."""
    width = 1
    for h in head:
        width *= int(h)
    k = len(shape)
    if width <= 1:
        return k == 0
    if k == 1:
        return int(shape[0]) == width
    if k == 2:
        r, c = int(shape[0]), int(shape[1])
        return r * c == width and r != 1 and r != width
    return False


def classify(shape, strides, head, axis=None, layout=None) -> str:
    """``"native"``, ``"view"``, ``"copy"``, ``"single"``, ``"ambiguous"`` or
    ``"conflict"`` for a plane of ``shape`` and element ``strides`` whose
    declared per-sample head is ``head``.

    ``"single"`` is one sample (:func:`is_single`), decided first. ``"native"``
    covers every other shape that is not sample-major, including the ones the
    door's own shape check refuses, so that check still names them.
    ``"ambiguous"`` is a shape that is both ``head + (N,)`` and ``(N,) + head``
    with nothing to say which; ``axis`` (``"first"``/``"last"``, the array's own
    marker) and ``layout`` (``"samples_first"``/``"samples_last"``, the call's)
    resolve it, the marker winning. ``"conflict"`` is a marker the shape
    contradicts. Neither is ever taken as native or guessed."""
    if is_single(shape, head):
        return "single"
    return _classify_batch(shape, strides, head, axis, layout)


def _classify_batch(shape, strides, head, axis=None, layout=None) -> str:
    """:func:`classify` for a shape already known not to be one sample."""
    shape, head = tuple(int(s) for s in shape), tuple(int(h) for h in head)
    k = len(head)
    if not head or (k == 1 and head[0] <= 1) or len(shape) != k + 1:
        return "native"
    is_native, is_sample_major = shape[:k] == head, shape[1:] == head
    if axis is None and is_native and is_sample_major:
        # the per-call layout resolves only what is ambiguous
        axis = {"samples_first": "first", "samples_last": "last"}.get(layout)
        if axis is None:
            return "ambiguous"
    if axis == "first" and not is_sample_major:
        return "conflict"
    if axis == "last" and not is_native:
        return "conflict"
    if axis == "last" or (axis is None and (is_native or not is_sample_major)):
        return "native"
    perm_shape = shape[1:] + shape[:1]
    perm_strides = tuple(strides[1:]) + tuple(strides[:1])
    return "view" if _c_contiguous(perm_shape, perm_strides) else "copy"


def _element_strides(value):
    """``value``'s strides in elements, for numpy, cupy and torch alike."""
    stride = getattr(value, "stride", None)
    if callable(stride) and not hasattr(value, "strides"):     # torch
        return tuple(stride())
    return tuple(s // value.itemsize for s in value.strides)


def _native_view(value, ndim: int):
    """The native permutation of a sample-major plane, as a VIEW."""
    if ndim == 2:
        return value.T
    order = tuple(range(1, ndim)) + (0,)
    permute = getattr(value, "permute", None)
    return permute(*order) if permute is not None else value.transpose(order)


def _user_frame():
    """The first frame outside eagle and the frameworks that call it — the call
    site a warning names and is de-duplicated by."""
    frame = sys._getframe(1)
    depth = 1
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if not module.startswith(("eagle.", "torch.", "hawk.", "cupy.")) \
                and module not in ("eagle", "torch", "hawk", "cupy"):
            return frame, depth
        frame = frame.f_back
        depth += 1
    return None, 1


_seen: set = set()


def _reset_seen() -> None:
    """Forget every call site already warned for (tests only)."""
    _seen.clear()


def _warn(name, shape, head, *, writes: bool) -> None:
    native = "(" + ", ".join(str(h) for h in head) + ", N)"
    back = (" and copied back into it after the launch" if writes
            else " (read-only: no copy back)")
    message = (
        f"eagle: argument {name!r} was given sample-major, shape {tuple(shape)}, "
        f"but the kernel's plane is component-major {native}. It was copied "
        f"into a component-major buffer{back}, so zero-copy is lost for this "
        f"argument. To avoid the copy, allocate it as {native}, or pass the "
        f"transposed view of a contiguous {native} array."
    )
    frame, depth = _user_frame()
    key = (None if frame is None else (frame.f_code.co_filename, frame.f_lineno),
           name)
    if key in _seen:
        return
    # `depth` counts _warn's own frame as 1, exactly as stacklevel does
    warnings.warn(message, LayoutWarning, stacklevel=depth)
    _seen.add(key)       # only once the warning did not raise (strict mode)


def _shape_text(shape) -> str:
    return "(" + ", ".join(str(int(e)) for e in shape) + ("," if len(shape) == 1 else "") + ")"


def _ambiguous_message(name, shape, head) -> str:
    native = "(" + ", ".join(str(h) for h in head) + ", N)"
    first = "(N, " + ", ".join(str(h) for h in head) + ")"
    n = int(shape[-1])
    return (
        f"eagle: argument {name!r} has shape {_shape_text(shape)}, which reads "
        f"both as component-major {native} with N = {n} (samples last) and as "
        f"sample-major {first} with N = {n} (samples first); eagle will not "
        f"guess. Say which axis holds the samples, zero-copy: pass "
        f"layout=\"samples_first\" or layout=\"samples_last\" to the call "
        f"(resolves every ambiguous plane), or mark this array: "
        f"eagle.samples_first({name}) / eagle.samples_last({name})."
    )


def _conflict_message(name, shape, head, axis) -> str:
    native = "(" + ", ".join(str(h) for h in head) + ", N)"
    first = "(N, " + ", ".join(str(h) for h in head) + ")"
    want, other = (first, native) if axis == "first" else (native, first)
    return (
        f"eagle: argument {name!r} is marked samples_{axis} (shape {want}) but "
        f"its shape is {_shape_text(shape)}, which is not {want}"
        + (f"; it looks like {other}, so use eagle.samples_"
           f"{'last' if axis == 'first' else 'first'}({name}) or drop the marker"
           if _shape_fits(shape, head, "last" if axis == "first" else "first")
           else "")
        + "."
    )


def _shape_fits(shape, head, axis) -> bool:
    k = len(head)
    shape = tuple(int(e) for e in shape)
    return (shape[:k] == tuple(head)) if axis == "last" else (shape[1:] == tuple(head))


class Adapted:
    """One adapted plane: ``native`` is what gets bound; ``caller`` the array
    the caller passed; ``scratch`` is ``native`` when it is a copy (else
    ``None``); ``writes`` whether it is copied back."""

    __slots__ = ("name", "caller", "native", "scratch", "writes", "layout")

    def __init__(self, name, caller, native, scratch, writes, layout):
        self.name, self.caller, self.native = name, caller, native
        self.scratch, self.writes, self.layout = scratch, writes, layout

    def caller_view(self):
        """The caller's array seen in its native layout (a view, never a copy)."""
        return _native_view(self.caller, self.caller.ndim)

    def copy_in(self) -> None:
        """Refresh the scratch from the caller's array (a written plane is
        read-modify-write, and a sample a launch skips must keep its value)."""
        _copyto(self.scratch, self.caller_view())

    def copy_back(self) -> None:
        """Write the scratch back into the caller's array."""
        _copyto(self.caller_view(), self.scratch)

    def write_back_from(self, result) -> None:
        """Write ``result`` — the plane the kernel actually wrote, in native
        layout, possibly in another framework or flattened (a matrix bound
        ``(R*C, N)``) — back into the caller's array."""
        dst = self.caller_view()
        src = result.reshape(tuple(dst.shape))
        dst_mod, src_mod = _module(dst), _module(src)
        if dst_mod == src_mod:
            _copyto(dst, src)
        elif dst_mod == "torch":
            import torch

            dst.copy_(torch.from_dlpack(src))
        elif dst_mod == "cupy":
            import cupy

            cupy.copyto(dst, cupy.asarray(src))
        else:
            import numpy

            numpy.copyto(dst, src.get() if src_mod == "cupy" else src)


def _module(value) -> str:
    return type(value).__module__.split(".")[0]


def _copyto(dst, src) -> None:
    copy_ = getattr(dst, "copy_", None)
    if copy_ is not None:            # torch
        copy_(src)
        return
    module = type(dst).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        cupy.copyto(dst, src)
    else:
        import numpy

        numpy.copyto(dst, src)


def _contiguous_copy(view):
    contiguous = getattr(view, "contiguous", None)
    if contiguous is not None and not hasattr(view, "flags"):   # torch
        return contiguous()
    module = type(view).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        return cupy.ascontiguousarray(view)
    import numpy

    return numpy.ascontiguousarray(view)


def is_number(value) -> bool:
    """A Python (or numpy) scalar number: one sample's value, never a plane."""
    if isinstance(value, (int, float, complex)):
        return True
    module = type(value).__module__
    if module != "numpy":
        return False
    import numpy

    return isinstance(value, numpy.generic)


def adapt(name, value, head, *, writes: bool, single: bool = False, dtype=None,
          axis=None, layout=None):
    """Adapt one plane. Returns ``None`` when ``value`` is neither sample-major
    nor (with ``single=True``) one sample — bind it as given — else an
    :class:`Adapted` whose ``native`` is bound.

    ``head`` is the declared per-sample shape (``(w,)``, ``(R, C)``, or a flat
    ``(R*C,)``; ``()`` or ``(1,)`` for a scalar plane); ``writes`` whether the
    kernel writes the plane. With ``single=True`` one sample is adapted too:
    ``native`` is the zero-copy ``(w, 1)``/``(1,)`` view of the caller's array
    (layout ``"single"``), and a number on a read-only plane becomes a one-cell
    array of ``dtype`` (the door refuses a number on a written plane first).

    ``axis`` is this array's own marker (``"first"``/``"last"``, from
    :func:`split_marks`), ``layout`` the call's; an ambiguous plane that
    neither resolves, and a marker its shape contradicts, raise
    :class:`LayoutError` naming ``name``."""
    shape = getattr(value, "shape", None)
    if (shape is None or shape == ()) and single and not writes \
            and is_number(value):
        import numpy

        cell = numpy.asarray([value], dtype=dtype)
        return Adapted(name, value, cell, None, writes, "single")
    if shape is None:
        return None
    if is_single(shape, head):
        if not single:
            return None
        width = _width(tuple(int(h) for h in head))
        flat = () if width <= 1 else (width,)
        return Adapted(name, value, value.reshape(flat + (1,)), None, writes,
                       "single")
    try:
        strides = _element_strides(value)
    except (AttributeError, TypeError):
        return None
    if axis is None or layout is None:
        layout_scope, axes = _scope.get()
        axis = axis or axes.get(name)
        layout = layout or layout_scope
    kind = _classify_batch(shape, strides, head, axis, layout)
    if kind == "ambiguous":
        raise LayoutError(_ambiguous_message(name, shape, head))
    if kind == "conflict":
        raise LayoutError(_conflict_message(name, shape, head, axis))
    if kind == "native":
        return None
    layout = kind
    view = _native_view(value, len(shape))
    if layout == "view":
        return Adapted(name, value, view, None, writes, layout)
    _warn(name, shape, head, writes=writes)
    scratch = _contiguous_copy(view)
    return Adapted(name, value, scratch, scratch, writes, layout)
