# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Gradient keys of a builder graph: the names that enter and leave it.

Seeds are keyed ``"result"`` (``"result.<path>"`` into a tree), a
``Mutable``-written argument's name, or a ``buffer=`` label; ``wrt`` names
builder arguments or ``buffer=`` labels (the cotangent at that buffer). Anything else is a temporary and is
refused. Gradients come back under the builder's names."""

from __future__ import annotations

_TEMPORARY = ("{k!r} is not a builder argument, 'result' or a buffer= label: gradients flow through the arrays that "
              "enter and leave the graph. For a temporary, give it a dedicated buffer: `{k} = kernel(..., buffer=True)` "
              "(or pass an array), then use that name")


def _end_planes(graph):
    """``{id(array): (output key, input key)}``: the last writer's and the first reader's plane;
    ``where`` has the last writer's ``(leaf, plane)``."""
    out, inp, where = {}, {}, {}
    for i, a in enumerate(graph._leaves()[0]):
        for name, arr, _role in a.writes:
            out[id(arr)] = f"{a.label}.{name}"
            where[id(arr)] = (i, name)
        for name, arr, _role in a.reads:
            inp.setdefault(id(arr), f"{a.label}.{name}")
    return out, inp, where


def translate(graph, seeds, wrt):
    """``(seeds, wrt, keys)`` in the recorded graph's plane keys; ``keys`` maps each builder name to its input
    plane, or to ``("at", (leaf, plane))`` for a ``buffer=`` label (the cotangent arriving at that buffer)."""
    from .._builder import tree_leaves

    st = graph._bld
    out, inp, where = _end_planes(graph)
    held = lambda a: st.views[id(a)][1] if id(a) in st.views else a      # the flat plane behind a lane-shaped view
    named = {n: held(a) for n, a in st.args.items() if hasattr(a, "shape")}
    leaves = {f"result{p}": held(a) for p, a in tree_leaves(st.result)}
    labelled = {k: held(a) for k, a in st.buffers.items()}

    def refuse_deep(k):
        if k in st.deep:
            raise KeyError(f"eagle.diff.reverse: {k!r} is a buffer inside a device loop or branch, which is not a gradient "
                           "key in this release; move it out of the body")

    def seed_key(k):
        refuse_deep(k)
        arr = leaves.get(k) if k.startswith("result") else named.get(k, labelled.get(k))
        if arr is None:
            if k.startswith("result"):
                raise KeyError(f"eagle.diff.reverse: {k!r} is not in the result tree")
            raise KeyError("eagle.diff.reverse: " + _TEMPORARY.format(k=k))
        if id(arr) not in out:
            raise KeyError(f"eagle.diff.reverse: {k!r} is an input the graph never writes: seed an output "
                           "(a Mutable-written argument or 'result')")
        return out[id(arr)]

    def wrt_key(k):
        refuse_deep(k)
        arr = named.get(k, labelled.get(k))
        if arr is None or id(arr) not in inp:
            raise KeyError("eagle.diff.reverse: " + _TEMPORARY.format(k=k))
        return ("at", where[id(arr)]) if id(arr) in out and k not in named else inp[id(arr)]

    if wrt is None:
        floats = [n for n, a in named.items() if str(a.dtype).startswith("float")]
        raise ValueError("eagle.diff.reverse: a builder graph has no default gradients: pass wrt=(...) naming "
                         f"the arguments it could take: {', '.join(floats)}")
    wrt = (wrt,) if isinstance(wrt, str) else tuple(wrt)
    keys = {k: wrt_key(k) for k in wrt}
    inputs = tuple(v for v in keys.values() if isinstance(v, str))
    def flat(k, v):
        """A seed given in the lane shape of the view the caller holds, as the flat plane."""
        arr = leaves.get(k) if k.startswith("result") else named.get(k, labelled.get(k))
        return v.reshape(arr.shape) if hasattr(v, "shape") and v.shape != arr.shape and v.size == arr.size else v

    return {seed_key(k): flat(k, v) for k, v in seeds.items()}, inputs, keys


def own_shape(graph, name, grad):
    """``grad`` in the shape of the builder argument ``name`` it is the gradient of."""
    arg = graph._bld.args.get(name)
    shape = getattr(arg, "shape", None)
    return grad.reshape(shape) if shape is not None and grad.shape != tuple(shape) and grad.size == arg.size else grad
