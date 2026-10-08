Python API reference
=====================

Complete reference for the ``eagle`` Python package, produced by Sphinx
``autodoc``/``autosummary`` from the live package — the signatures shown here
are the ones actually exported by the build you have installed.

``eagle`` owns everything about *executing* compiled kernels: the
framework-polymorphic launch skeleton, the capturable kind-dispatched
``launch`` primitive, input-role marshalling, the by-value launch ABI, DLPack
framework adapters, CUDA-graph capture/replay, composing several already-built
launchables into one launch list, and the plugin protocol consumers
(:class:`~eagle.registry.KernelRegistry`, :func:`~eagle.registry.load_manifest`).

The top-level namespace
------------------------

``eagle/__init__.py`` re-exports its whole intended public surface — every
launcher, every DLPack adapter, the by-value ABI helpers, the plugin
registry — directly as ``eagle.<name>``, so this one page is a complete
single-page reference for everything a typical caller needs (see
:doc:`../userguide/quickstart_python`, :doc:`../interop`).

.. autosummary::
   :toctree: generated
   :template: module.rst

   eagle

``eagle.deploy`` is bound on first use (so ``import eagle`` imports no hawk)
and is the same function as :func:`eagle.plan.auto`:

.. autofunction:: eagle.deploy
   :no-index:

Submodules not re-exported at the top level
----------------------------------------------

A handful of names are reached by their submodule path rather than directly
off ``eagle`` — mostly the newer ``aether-abi/2`` bind-by-name door
(:mod:`eagle.plan`, :mod:`eagle.exec`), the shared input-role coercion layer,
and the GPU/CPU device-dispatch internals.

.. automodule:: eagle.plan
   :members:

.. automodule:: eagle.exec
   :members:

.. automodule:: eagle.marshal
   :members:

.. automodule:: eagle.sidecar
   :members:

.. automodule:: eagle.roles
   :members:

.. automodule:: eagle.cuda
   :members:

The low-level launch stack
--------------------------

:func:`eagle.launch`, :class:`eagle.LoadedKernel` and friends
(:mod:`eagle.launch`, :mod:`eagle.loaded`, :mod:`eagle.marshal`) and the ctypes
host launcher :mod:`eagle.host_launch` are the LOW-LEVEL launch stack: one
compiled ``aether-abi/1`` kernel launched by hand, with its arguments marshalled
per call. New code reaches kernels through :func:`eagle.plan.plan`,
:func:`eagle.deploy`, :func:`eagle.until_done` and :func:`eagle.simulate`; the
low-level stack stays for neural-block manifests and for callers that need
exactly one launch and nothing else.

.. autoclass:: eagle.launch.LaunchMixin
   :members:

.. autoclass:: eagle.launch.LaunchPlan

.. autofunction:: eagle.launch.launch_plan

.. autofunction:: eagle.launch.pure_origin

.. autodata:: eagle.launch.DEFAULT_BLOCK

.. automodule:: eagle.host_launch
   :members:

Framework bridges
-----------------

:mod:`eagle.frameworks.torch` turns a hawk per-sample kernel into a
``torch.autograd.Function`` (see :doc:`../userguide/tutorials/03_torch_training`).
It is the one ``eagle`` module that imports torch, and only when it is imported
itself.

.. automodule:: eagle.frameworks.torch
   :no-members:

.. autosummary::
   :toctree: generated

   function
   KernelFunction

GEMM helpers
--------------

.. automodule:: eagle.gemm.capture
   :members:

.. automodule:: eagle.gemm.contraction
   :members:

.. automodule:: eagle.gemm.plan
   :members:

.. seealso::

   :doc:`api_reference` — the C++ API (breathe/Doxygen).
   :doc:`../interop` — numpy, cupy and torch interoperability, worked.
