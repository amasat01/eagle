# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device backend's typed refusal.

``eagle._core`` reaches the GPU through a separately loaded backend plugin
(``eagle/libeagle_cuda.so`` for kind ``"cuda"``). When that plugin is absent,
cannot be loaded, implements an incompatible seam version, does not provide the
capability asked for, or finds no usable device, the call raises
:class:`BackendUnavailable` with the reason (and, for a missing plugin, every
path that was tried). Host execution never needs the backend.

Pure Python on purpose: ``import eagle`` must not load the compiled extension,
and ``except eagle.BackendUnavailable`` must work on a machine where it never
loads.
"""


class BackendUnavailable(RuntimeError):
    """The device backend cannot serve this call (see the message for why).

    A :class:`RuntimeError`, so code that already guards a device call with
    ``except RuntimeError`` keeps working.
    """


__all__ = ["BackendUnavailable"]
