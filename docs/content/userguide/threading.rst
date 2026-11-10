Threading
=========

The family-wide model, and how it is checked, is in `raptor's threading page <https://amasat01.github.io/raptor/content/threading.html>`__; this page covers eagle's objects only.

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

A graph's runs stay ordered across threads: the first stage of each run waits
on the device for the run before it, whichever thread launched it. One
exception couples threads: a host-synchronized node (``opaque(fn,
eager=True)``) syncs the whole device, so it also waits for every other
thread's GPU work.

Inside the package, module-level state is limited to read-only tables, locked
tallies and idempotent memos (computing the same value twice is harmless); a
test enforces it, so importing eagle from many threads is safe.

Host threads
------------

Host stages run on eagle's own thread pool, created at the first launch that
needs more than one thread. Its threads wait for work by spinning briefly and
then sleeping, so on a busy machine an idle eagle thread gives its core back.

How many threads a stage uses is measured, not configured. The first launch of a
built stage runs on the calling thread and is timed; later launches use one
thread for every 40 µs of that serial time, up to the budget. A stage cheaper
than that (a one-sample fork in a per-step hook, say) keeps running on the
calling thread, with no thread woken. ``g.host_lanes`` reports the threads per
stage (``1`` = the calling thread); rebinding a graph measures again.

The budget is one number for the whole process: ``eagle.exec.set_host_threads(N)``,
else ``OMP_NUM_THREADS``, else the physical cores this process may run on
(capped by the container's CPU quota). ``eagle.exec.host_threads()`` reads it.
A launch from inside another host launch, or while another thread's launch
holds the pool, runs on its own thread, so launches never nest and never wait
for each other. Inside your own OpenMP parallel region, launch from one thread
(or hand the graph over as a host table, see :doc:`graph_model`); every thread
that launches there would otherwise compete for the pool.

``os.fork`` is supported: the child builds its own pool at its first launch, and
eagle's launches make no OpenMP call. Forking while another thread makes its
first call into an OpenMP runtime (an OpenMP-backed op, or your own parallel
region) can hang the child under libomp: that is a limit of the runtime, not of eagle.

Free-threaded CPython (3.14t and 3.15t) is supported and tested in CI. See
:doc:`graph_model` for the graph guard in context and :doc:`installation` for
the free-threaded install notes.
