Gradients through a graph: ``eagle.diff``
=========================================

.. contents:: On this page
   :local:
   :depth: 2

You built a step as an :class:`eagle.Graph` (see :doc:`graph_model`), ran it,
and now you want to know how a result depends on what went in: how the final
state moves with the initial one, how a cost moves with a parameter, how a loss
moves with the weights. :mod:`eagle.diff` answers that for the whole graph, for
every sample at once, in one backward sweep, on the CPU or the GPU.

.. code-block:: python

   g.build(record=True); g.launch()
   grads = eagle.diff.reverse(g.record, seeds={"s": dJ_ds})   # dJ_ds: ∂J/∂s per sample
   dJ_dx, dJ_dk = grads["x"], grads["k"]

Those three lines are the whole idea. The rest of this page explains each
word in them, and what eagle promises about the numbers that come back.

.. note::

   New to it? :doc:`tutorials/10_gradients_through_a_graph` builds this up
   step by step and checks every gradient against finite differences.

Words used on this page
-----------------------

Several communities differentiate programs, and each has its own words for the
same few things. These are the ones eagle uses; the synonyms are there so you
can recognise what you already know.

A short glossary of these words (reverse mode, seed, adjoint, vjp and the rest)
lives in the :doc:`../glossary`; each term below links there.

Seeds, and what comes back
--------------------------

``seeds`` maps output planes to their seeds. An output you do not name is
seeded with zero: it does not enter your objective. The names are the plane
names you bound, as the node that writes the plane calls it:

.. code-block:: python

   y = np.zeros(n); z = np.zeros(n)
   g = (eagle.Graph()
        .add(square.bind(x=x, a=0.5, y=y), name="square")
        .add(wave.bind(x=y, y=z), name="wave")
        .build(record=True))
   g.launch()
   grads = eagle.diff.reverse(g.record, {"wave.y": np.ones(n)})

Here both nodes call their output ``y``, so the plain name is ambiguous and
the output is named ``"node.plane"``. A name that is unique needs no prefix.
The result, :class:`~eagle.diff.Gradients`, is a read-only mapping keyed the
same way:

- ``grads["x"]``: the gradient with respect to the per-sample input ``x``,
  one row per sample, in sample order;
- ``grads["a"]``: the gradient with respect to the uniform ``a``, which every
  sample shares, so its per-sample contributions are summed into one value;
- ``grads.report``: the :term:`generation` of the Record, and per plane
  whether the gradient is :term:`bitwise`.

The two mistakes everyone makes once both get a message that says what to do
instead: seeding an *input* (``seeds={"x": ...}`` when you want ``∂J/∂x``:
that is ``grads["x"]``, not a seed), and passing a seed that is not one value
per sample.

Where the derivative kernels come from
---------------------------------------

Each launch node is reversed by its kernel's :term:`vjp`. For a plan made by
:func:`eagle.plan.auto` (or :func:`eagle.deploy`) from a hawk kernel, eagle
builds the vjp itself the first time it is needed, beside the kernel, and
keeps it. You write nothing.

For a plan built some other way (from a plugin you built with hawk by hand,
say), pass the vjp plan yourself:

.. code-block:: python

   rev = eagle.diff.Reverse(g, vjps={"square": square_vjp_plan})

eagle checks that each one really is the vjp of that node's kernel and refuses
a mismatch, naming both kernels. A vjp is a kernel *of* the reverse, never a
step of a forward graph: ``eagle.Graph.add`` and ``eagle.plan.auto`` refuse
one.

.. _autodiff-record:

The Record
----------

``build(record=True)`` makes the graph keep what its reverse needs. The vjps
recompute from the values each node *read*, and most of those are still sitting
in your planes when the reverse runs, so they cost nothing. The exception is a
plane a later node overwrites (or a node overwrites in place): eagle copies it
once, just before the overwrite. A graph that overwrites nothing copies nothing.

Because a Record points at the graph's own planes, the rules are about not
mixing up launches:

- A ``record=True`` graph records one run per launch: ``launch()`` (not
  ``launch(n)``, not ``enqueue``).
- While a Record is live (neither reversed nor kept), the next recording
  launch is refused: reverse it first, or keep a copy with
  ``record.keep()``.
- Reversing a Record whose launch is no longer the newest is refused: the
  message says which :term:`generation` you have and which is current.
- ``record.keep()`` copies everything the reverse reads (a cost in memory
  and one copy) and stays valid however often the graph runs again.
- A launch while a reverse is reading the Record raises :class:`eagle.Busy`.

You will never get a gradient computed from a mix of two launches: either the
reverse is refused, or it is the gradient of the launch you recorded. Keep the
graph's input planes unchanged between the launch and its reverse (or
``keep()`` first); eagle reverses the values the planes hold.

Built once, run many times
--------------------------

:func:`eagle.diff.reverse` is the eager door: it builds a reverse per call. In
a loop (a shooting iteration, an optimiser step) build it once:

.. code-block:: python

   rev = eagle.diff.Reverse(g)            # captured once on the GPU
   for it in range(n_iter):
       g.launch()
       grads = rev(g.record, seeds={"xT": dphi_dxT})

On the GPU the reverse is captured into one CUDA graph and replayed.

To change a parameter between launches, use ``g.update``:

.. code-block:: python

   for it in range(n_iter):
       g.update(k=k)                      # every node's k; {"join.k": k} for one node
       g.launch()
       grads = rev(g.record, seeds={"xT": dphi_dxT})
       k -= lr * grads["k"]

A recorded graph keeps each parameter (a kernel ``uniform``) in a one-cell
plane of its own, which the forward kernels and their vjps both read, so
``update`` is a copy into that cell: neither the forward nor the reverse is
rebuilt or recaptured. Like a recording launch, ``update`` is refused until the
last Record is reversed or kept, so a reverse always sees the parameters
its launch used. (The plans you bound keep their own values: a recorded graph
runs its own copies of them, bound to the cells.)

A plan made by :func:`eagle.plan.auto` gets its parameter cells
automatically. A plugin you built by hand must already read its parameters
from planes (``hawk.diff.param_planes(kernel)``); one with by-value
parameters is refused at ``build(record=True)``.

.. _autodiff-bits:

Same bits, every time
---------------------

Floating-point addition is not associative, so a gradient that is a sum can
change in its last bits with the order of the additions (see
:doc:`determinism`). eagle fixes the order everywhere it sums:

- Where several nodes read one plane, their adjoints are added in a fixed
  order: the seed first, then the readers in the order you added them. No
  atomics.
- A parameter's gradient is summed over the samples with the canonical tree,
  in **sample** order. If an :class:`eagle.ActiveSet` has physically reordered
  the samples, the sum still runs in sample order (eagle gathers through the
  set's permutation first), so a reorder never changes a bit.
- Per-sample gradients come back in sample order, paired with their samples
  through the same permutation.

So a gradient is the same bits for every OpenMP thread count and every run.
The one exception: a kernel that reads a table by index (``table.at(i)``) has a
vjp that scatter-adds into the table's gradient from many samples at once, in
whatever order they arrive. That gradient is still correct, but not
:term:`bitwise`; ``grads.report.bitwise["table"]`` says ``False``, and
``eagle.diff.reverse(..., strict=True)`` refuses such a reverse instead.

.. _autodiff-conditionals:

Gradients through conditionals
------------------------------

A step inside :func:`eagle.skippable` runs only while its guard cell is
non-zero; one inside :func:`eagle.repeat_while` repeats while its guard holds,
up to a cap. Both reverse. The rule is the same for both: **the reverse
differentiates the launch that happened, and never evaluates a guard.**

- **Recorded.** ``build(record=True)`` saves the guard cell a ``skippable``
  read (a one-value ``uint32`` copy), and a ``repeat_while`` already writes
  its iteration count into a plane. These are all the reverse needs.
- **Replayed.** The reverse turns those saved values into flags it owns and
  runs the member's derivative only when the member ran: once for a taken
  ``skippable``, once per iteration, newest first, for a ``repeat_while``.
  The state each iteration started with is rebuilt from checkpoints kept every
  few iterations, on private copies, so your planes are never written.
  Flipping the guard between launches therefore needs no rebuild, and a
  guard that counts its own reads is read zero times by the reverse.
- **Passed through.** A skipped member leaves its planes as they were, so the
  adjoint of a plane it would have updated in place goes to the plane's
  earlier writer, and every other adjoint of the member is zero.

The sum over a ``repeat_while``'s iterations runs oldest first, the order of
the same chain written out as separate nodes, so the two give the same bits.

.. code-block:: python

   flag = np.array([1], dtype="uint32")             # 1: run `maybe`
   maybe = eagle.skippable(wave_plan.bind(x=x, y=y).launch,
                           eagle.SkipGuard.nonzero(flag))
   g = (eagle.Graph()
        .add(square_plan.bind(x=x, a=0.8, y=y), name="first")
        .add_node(maybe, name="maybe")
        .add(wave_plan.bind(x=y, y=z), name="last")
        .build(record=True))
   rev = eagle.diff.Reverse(g)
   for on in (1, 0):
       flag[0] = on                                  # no rebuild, no recapture
       g.launch()
       grads = rev(g.record, {"last.y": np.ones(n)})  # on=1: grads["a"] is 0

The member of a ``repeat_while`` must be a plan launch; its cap is the most
iterations a reverse unrolls. The full, checked version of this example is in
:doc:`tutorials/11_conditionals_budgets_events`.

.. _autodiff-budget:

Memory budget
-------------

Reversing a recorded :func:`eagle.until_done` or :func:`eagle.simulation`
(``record=True``) keeps the state at the start of every ``window`` steps and,
working one window at a time from the last, recomputes the window's steps and
steps the adjoint back through them. One sample therefore holds about
``ceil(max_steps / window) + window`` states: the default window,
``clamp(ceil(sqrt(max_steps)), 16, 1024)``, balances the two terms, and
``window="full"`` trades memory for no recompute.

``memory_budget`` (bytes; ``None``, the default, is half the free device
memory, or half the available host RAM) caps what the reverse holds. When the
whole batch does not fit, the reverse runs it in **passes** of ``256 * 2^q``
samples, the largest that fits, and combines the passes' parameter sums with
the canonical tree. Those sums are exactly the subtrees of the one-pass tree,
so **the gradients are the same bits at any budget** (see
:ref:`autodiff-bits`). What a smaller budget costs is time: each pass launches
its own reverse over fewer samples.

.. code-block:: python

   sim = eagle.simulation(pendulum, ..., max_steps=400, record=True,
                          window=20, memory_budget=1_000_000)
   sim.run()
   grads = eagle.diff.reverse(sim.record, {"x": np.ones(n)})
   grads.report.chunks                  # how many passes it took

A budget that cannot hold one sample, or fewer than the 256 a pass needs to keep
parameter gradients bitwise, is refused with the bytes and the fix: raise
``memory_budget``, lower ``max_steps``, or set ``window``. ``window=`` and
``memory_budget=`` without ``record=True`` are refused too.

.. _autodiff-events:

Event refinement
----------------

A run-until-done stops a sample at the first *step* whose rule fires. Its stop
time is then a multiple of the step, and nudging an input moves it by nothing,
so the gradient of the stop time is zero. That is the right answer for the
stepped model, and the wrong one if the model is a continuous trajectory that
you sampled. ``events="refine"`` finds the crossing *inside* the stop step.

The kernel declares its time plane and step, ``@hawk.kernel(time="t",
time_step="dt")``, and its stop rule is read as a smooth guard ``g`` that
changes sign (``terminated = y < 0``; a count or a flag has no crossing and is
refused). The run then:

1. steps as usual, remembering the state before each sample's last step;
2. solves ``g = 0`` for the partial step ``tau`` in ``[0, dt]`` by Newton's
   method with a bisection fallback (at most 8 rounds);
3. writes the state *at* the crossing over the stepped one, and the event time
   and how it was found into ``runner.t_stop`` and ``runner.event_status``.

The reverse (:func:`eagle.diff.reverse` of a ``record=True`` run, no extra
flag) adds the **event sensitivity**: how the stop time moves with the
initial state and the parameters, from the implicit function theorem on
``g = 0``, together with its effect on the end state. Nothing else changes.

.. code-block:: python

   sim = eagle.simulation(fall, dt=0.1, g=9.81, y=y0, v=v0, t=t0,
                          max_steps=200, events="refine", record=True)
   sim.run()
   sim.runner.t_stop                                    # the event time
   grads = eagle.diff.reverse(sim.record, {"t": np.ones(n)})
   grads["y"]                                           # dt*/dy0, here 1/(g t*)

A sample that never crosses inside its stop step (it hit ``max_steps``) is
refused, unless you pass ``on_no_event="zero"``: its stop stays the stepped
one, with zero event sensitivity. A grazing crossing, and two stop rules
crossing in one step, are always refused, because the sensitivity is not
defined there.

**Phases.** A hybrid trajectory (a ball that bounces, a pendulum that hits a
wall) is a *phase* graph: each continuous piece is a refined run, and
:func:`eagle.diff.phase` makes it a node of an ordinary :class:`eagle.Graph`,
with a reset kernel between pieces.

.. code-block:: python

   graph = (eagle.Graph()
            .add_node(eagle.diff.phase(down), name="fall")
            .add(bounce_plan.bind(e=e, v_in=v_down, v=v_up), name="bounce")
            .add_node(eagle.diff.phase(up), name="rise")
            .build(record=True))
   graph.launch()                       # restarts each phase, runs it to its event
   grads = eagle.diff.reverse(graph.record, {"y": np.ones(n)})

Each launch of the graph restarts every phase and runs it to its event. The
reverse chains the phases through the reset kernels' vjps, which is the chain
rule across each jump (the saltation of the jump, in the hybrid-systems
literature). The number of phases is the graph's, so a phase inside
:func:`eagle.repeat_while` is refused. On the GPU a phase is an eager node (it
reads its own progress back between launches), and the graph is captured in
segments around it. The checked version of both examples is in
:doc:`tutorials/11_conditionals_budgets_events`. A discontinuity in the model itself, a branch on a learnable
threshold, is trained through in :doc:`tutorials/19_train_through_a_branch`.

.. _autodiff-run-vjps:

Your model inside a recorded loop
---------------------------------

A step list ``[node, plan]`` runs your model before the kernel on every step,
and a recorded run of it reverses with your vjp:

.. code-block:: python

   node = eagle.graph.opaque(model_step, reads={"x": x, "v": v}, writes={"corr": corr},
                             name="model")
   runner = eagle.until_done([node, plan], max_steps=1000, record=True, x=x, v=v, corr=corr)
   runner.run()
   grads = eagle.diff.reverse(runner.record, {"x": dJ_dx}, vjps={"model": vjp})

The node reads the plan's state and per-sample planes and writes planes the
plan reads (``eagle.simulate([node, kernel], ...)`` takes the same list). The
reverse re-runs it once per step while it recomputes a window, then calls
``vjp(primals, cotangents)`` once per step, host-synchronized, on the samples
still running at that step; your model's parameter gradients accumulate as
you write them. **Cost**: a hooked reverse costs what your model costs per
step; eagle's share is the un-hooked reverse (0.12 ms/step at 4k rows on a
P2000). **Memory**: one more window of each plane the node writes per sample.

The node must be pure, because the reverse re-runs it: the same inputs give
the same outputs (``model.eval()``, no random draw, no counter), row ``s``
of a write depends only on row ``s`` of the reads, it writes only its declared
planes, and the weights stay as they were until the reverse returns
(``run()``, ``reverse()``, then ``optimizer.step()``). eagle cannot see an
impure node; the test that catches it is yours: reverse twice, compare. The
vjp gets views of eagle's buffers, valid during the call: copy what you keep.

**HAZARD 3**: during the reverse the node's planes are eagle's scratch, and
they come back bitwise at the end. Do not read them while a reverse of that run
is in flight; a second reverse of it raises :class:`eagle.Busy`.

A node runs before the plan in a step: write the step as ``[node, plan]``. The run calls the
node on the initial state before step 0, and applying it once more after the last step is
yours; your framework's autograd of that call gives the seeds.

What a reverse refuses
----------------------

Each refusal names the node and what to do instead:

- an opaque node (``eagle.graph.opaque``): eagle cannot know its derivative;
- a kernel that reads a plane before assigning it (a value carried from one
  launch to the next): its derivative needs the overwritten value;
- a ``"max"`` or ``"all"`` :func:`eagle.reduce`: a tie has no single
  derivative (``"sum"`` reverses);
- a reverse of a reverse (second order);
- a launch node with no vjp and none passed;
- a ``repeat_while`` whose body is not a plan launch;
- a ``record=True`` run of a step made of several plans, or of a kernel taking
  several steps per launch (record one kernel per step);
- a recorded step with a node after the plan, a node reading another node's
  plane or a ``Param``, ``events="refine"``, or a ``Node`` as the vjp;
- the ``events="refine"`` cases above: no smooth crossing, no crossing in the
  stop step, a grazing crossing, two rules crossing at once.

Threads
-------

A :class:`~eagle.diff.Reverse` owns its adjoint planes. Distinct ``Reverse``
objects run at the same time, also on the same Record (a reverse never writes
the Record), and give the same bits. One ``Reverse`` shared by two threads
raises :class:`eagle.Busy` in the second; give each thread its own.

For C++ users
-------------

The plan of a reverse is plain C++: ``eagle::diff::transpose`` takes a graph
``Description`` (:doc:`graph_view`) and, per node, the planes its vjp reads and
writes, and returns the order of the derivative steps, the summing joins and
the copies; ``eagle::diff::orderedSum`` is the fixed-order sum. See
:doc:`../api/api_diff`.
