Using eagle from several threads
================================

The rule is the same for every object that owns a run: one object, one caller
at a time. Separate objects run in parallel without any care.

:class:`eagle.Graph`, :class:`eagle.Simulation`, :class:`eagle.Runner` (from
:func:`eagle.until_done`) and the runner behind a ``dispatch=`` simulation
each hold a non-blocking guard. A second thread that enters while one call is
in flight gets :class:`eagle.Busy`; it is never queued behind the first and
never handed a half-built state. The guard is a C++ atomic, so this holds on
free-threaded Python too.

.. code-block:: python

   import threading, eagle

   # one runner per thread: runs side by side
   def work(plan, planes):
       runner = eagle.until_done(plan, max_steps=10_000, **planes)
       runner.run()

   # one runner shared by two threads: the second caller gets eagle.Busy
   # unless it waits for the first to return

Inside the package, module-level state is limited to read-only tables, locked
tallies and idempotent memos (computing the same value twice is harmless); a
test enforces it, so importing eagle from many threads is safe.

Free-threaded CPython (3.13t and 3.14t) is supported and tested in CI. See
:doc:`graph_model` for the graph guard in context and :doc:`installation` for
the free-threaded install notes.
