# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the builder tests: the doc's return-form kernels written in
today's Mutable form, and the runner for the worked example of the graph page."""

import ast
import inspect
import json
import linecache
import os
import pathlib
import re
import sys
import tempfile
import textwrap
import types

import pytest

import eagle

hawk = pytest.importorskip("hawk")

CACHE = os.path.join(os.environ.get("STREAM_SHM", tempfile.gettempdir()), "builder_hawk_cache")
RETURNS = hasattr(hawk.Kernel, "returns")
#: The graph page whose cells tagged ``worked`` are the blocks the tests run; ``EAGLE_GRAPH_API_DOC``
#: names another page (a notebook, or a Markdown file with ```python fences).
DOC = os.environ.get("EAGLE_GRAPH_API_DOC",
                     str(pathlib.Path(__file__).resolve().parents[2] / "docs/content/userguide/graph_forms.ipynb"))


def kernel(fn, targets=("host",)):
    """``@kernel`` over ``def k(x: Scalar) -> Scalar: return e``: until hawk
    declares ``returns``, the returned slot is an omitted ``out: Mutable``."""
    if RETURNS:
        return eagle.deploy(hawk.kernel(fn), targets=targets, cache_dir=CACHE)
    src = textwrap.dedent(inspect.getsource(fn))
    fd = ast.parse(src).body[0]
    fd.decorator_list = []
    if fd.returns is not None:
        fd.args.args.append(ast.arg("out", ast.Subscript(ast.Name("Mutable", ast.Load()), fd.returns, ast.Load())))
        fd.returns = None
        fd.body[-1] = ast.Assign([ast.Name("out", ast.Store())], fd.body[-1].value)
    ast.fix_missing_locations(fd)
    text = ast.unparse(fd) + "\n"
    name = f"<builder kernel {fn.__name__}>"
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    ns = dict(fn.__globals__)
    exec(compile(text, name, "exec"), ns)
    return eagle.deploy(hawk.kernel(ns[fn.__name__]), targets=targets, cache_dir=CACHE)


class HawkShim(types.ModuleType):
    """``import hawk`` for the doc's snippets: ``hawk.kernel`` is :func:`kernel`."""

    def __init__(self, targets=("host",)):
        super().__init__("hawk")
        self.kernel = lambda fn: kernel(fn, targets)

    def __getattr__(self, n):
        if n == "math":
            import importlib
            return importlib.import_module("hawk.math")
        return getattr(hawk, n)


def doc_blocks(n=8):
    with open(DOC) as f:
        if DOC.endswith(".ipynb"):
            cells = json.load(f)["cells"]
            return ["".join(c["source"]) for c in cells
                    if c["cell_type"] == "code" and "worked" in c["metadata"].get("tags", ())][:n]
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)[:n]


def run_doc(n=8, tolerate=(), errors=None, gpu=False):
    ns = {}
    for i, src in enumerate(doc_blocks(n), 1):
        if gpu:
            src = src.replace("import numpy as np", "import cupy as np")
        name = f"<doc snippet {i}>"
        linecache.cache[name] = (len(src), None, src.splitlines(True), name)
        saved = sys.modules.get("hawk")
        if i == 1 and not RETURNS:
            sys.modules["hawk"] = HawkShim(("host", "cuda") if gpu else ("host",))
        try:
            exec(compile(src, name, "exec"), ns)
        except Exception as e:
            if i not in tolerate: raise
            if errors is not None: errors[i] = e
        finally:
            if saved is not None: sys.modules["hawk"] = saved
    return ns
