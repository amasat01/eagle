# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle — the Python face of the EAGLE launch engine.

Which call do I use?
    one kernel (or a step's worth), run until every sample finishes -- :func:`simulate`
    a plan already paced by its own finish kernel, one call         -- :func:`until_done`
    build a kernel into a plan that runs where its data lives       -- :func:`deploy` / :mod:`plan`
    resolve a launchable BY NAME, then call/stream/graph it         -- :class:`KernelRegistry`

EAGLE owns everything about *executing* compiled kernels: the
framework-polymorphic launch skeleton (:class:`~eagle.launch.LaunchMixin`), the
capturable kind-dispatched :func:`~eagle.launch.launch` primitive,
input-role marshalling (:mod:`eagle.marshal`), the by-value launch ABI
(:mod:`eagle.abi`), DLPack
framework adapters (:mod:`eagle.interop`), CUDA-graph capture/replay
(:class:`~eagle.pipeline.GraphPipeline`), composing many already-built
launchables into one launch list (:class:`~eagle.compose.GraphComposer`),
and the plugin protocol consumers —
:class:`~eagle.registry.KernelRegistry` / :func:`~eagle.registry.load_manifest`
on the Python side, mirroring the C++ ``host/`` registries.

Producers of launchable artifacts (compiled CUDA sources, PTX bundles with
sidecar/manifest metadata) live upstream — the code generator is the first — and depend
on this package; the eagle core never imports a producer. The optional framework bridges
(:mod:`eagle.frameworks`) are the one exception: they import hawk lazily, on first use,
and so does :func:`eagle.deploy` (the same function as :func:`eagle.plan.auto`), which
builds a hawk kernel into a plan that runs where its data lives.
:func:`eagle.simulate` runs a model (one hawk kernel, or the kernels of one step)
on every sample until each one finishes, called exactly like the kernel itself.
"""

from ._active_set import ActiveSet, compaction_body
from ._backend import BackendUnavailable
from ._conditional import RepeatWhile, SkipGuard, Skippable, repeat_while, skippable
from ._device_props import DeviceProps, device_props
from ._layout import LayoutWarning, samples_first, samples_last
from ._simulate import SimResult, Simulation, simulate, simulation
from ._until_done import (
    FINISHED_PLANE,
    FUSED_STEPS_PLANE,
    Runner,
    RunReport,
    run_until_done,
    until_done,
)
from .abi import (
    ABI_VERSION,
    check_aether_abi,
    make_gref,
    make_handle,
)
from .compose import GraphComposer
from .dtypes import SCALAR_TYPES, np_dtype
from .interop import detect, origin_adapter, to_cupy
from .launch import KERNEL_NAME, assemble_args, launch, pure_prepare
from .loaded import LoadedKernel, LoadedPure, LoadedVector
from .pipeline import GraphPipeline
from .registry import KernelRegistry, load_manifest

__all__ = [
    "ABI_VERSION",
    "ActiveSet",
    "BackendUnavailable",
    "DeviceProps",
    "FINISHED_PLANE",
    "FUSED_STEPS_PLANE",
    "GraphComposer",
    "GraphPipeline",
    "KERNEL_NAME",
    "KernelRegistry",
    "LayoutWarning",
    "samples_first",
    "samples_last",
    "LoadedKernel",
    "LoadedPure",
    "LoadedVector",
    "RepeatWhile",
    "RunReport",
    "Runner",
    "SCALAR_TYPES",
    "SimResult",
    "Simulation",
    "SkipGuard",
    "Skippable",
    "assemble_args",
    "check_aether_abi",
    "compaction_body",
    "deploy",
    "detect",
    "device_props",
    "launch",
    "load_manifest",
    "make_gref",
    "make_handle",
    "np_dtype",
    "origin_adapter",
    "pure_prepare",
    "repeat_while",
    "run_until_done",
    "simulate",
    "simulation",
    "skippable",
    "to_cupy",
    "until_done",
]


def __getattr__(name):
    """``eagle.deploy`` IS :func:`eagle.plan.auto`, bound on first use so that
    ``import eagle`` imports neither :mod:`eagle.plan` nor hawk."""
    if name == "deploy":
        from .plan import auto

        globals()["deploy"] = auto
        return auto
    raise AttributeError(f"module 'eagle' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
