# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Framework bridges: a compiled per-sample kernel as a native op of an ML framework.

Each bridge lives in its own submodule, named after the framework it serves
(:mod:`eagle.frameworks.torch`). Importing this package imports none of them, and
no framework is imported until its submodule is: plain ``import eagle`` stays
framework-free.
"""
