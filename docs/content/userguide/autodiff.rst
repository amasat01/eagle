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

   New to it? :doc:`tutorials/12_gradients_through_a_graph` builds this up
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
- While a Record is live (neither reversed nor detached), the next recording
  launch is refused: reverse it first, or keep a copy with
  ``record.detach()``.
- Reversing a Record whose launch is no longer the newest is refused: the
  message says which :term:`generation` you have and which is current.
- ``record.detach()`` copies everything the reverse reads (a cost in memory
  and one copy) and stays valid however often the graph runs again.
- A launch while a reverse is reading the Record raises :class:`eagle.Busy`.

You will never get a gradient computed from a mix of two launches: either the
reverse is refused, or it is the gradient of the launch you recorded. Keep the
graph's input planes unchanged between the launch and its reverse (or
``detach()`` first); eagle reverses the values the planes hold.

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
last Record is reversed or detached, so a reverse always sees the parameters
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

What a reverse refuses
----------------------

Each refusal names the node and what to do instead:

- an opaque node (``eagle.graph.opaque``): eagle cannot know its derivative;
- a node inside ``eagle.skippable`` / ``eagle.repeat_while``: not reversible
  yet;
- a kernel that reads a plane before assigning it (a value carried from one
  launch to the next): its derivative needs the overwritten value;
- a ``"max"`` or ``"all"`` :func:`eagle.reduce`: a tie has no single
  derivative (``"sum"`` reverses);
- a reverse of a reverse (second order);
- a launch node with no vjp and none passed.

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
