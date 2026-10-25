What happens under the hood
===========================

This page follows a graph builder from the decorator to a replay, so you can see
where each piece of eagle lives. None of it is needed to use the builder; it is
here for the curious and for anyone who wants to predict what a function will cost.

.. code-block:: text

   decorate        call the builder            build                   run
   rewrite once -> record kernel calls    ->   CUDA: capture once,  -> g(): replay
                   as nodes (nothing runs)       instantiate            bind: patch pointers
                                                 CPU: a host schedule

Decorating: one rewrite
-----------------------

``@eagle.Graph.builder`` reads the function's source and rewrites it once (the result
is kept, so a second call does not parse again). The rewrite turns ``if``, ``while``,
``for`` and ``match`` into calls on a recorder; everything else is left alone. The
function is still ordinary Python: you can print inside it, and a ``for`` over a Python
list works.

Calling the builder: recording
------------------------------

Calling the builder runs the rewritten function once, with the recorder active. A kernel
call does not launch; it binds the arguments and appends a **node** to the graph, and
allocates one array for each output the kernel returns (``buffer=`` supplies or keeps
that array). The value you get back is that array, so the next call reads it, and
that is how eagle finds the edges: a node that reads an array comes after the node that
wrote it, and two nodes that could race are refused.

A condition decides what is recorded:

- A Python value (``if debug:``, ``for k in kernels:``) is decided **once**, here, and
  only the chosen path is recorded.
- A one-cell ``uint32`` array is a **device condition**. Both arms are recorded, under
  a conditional node, and which one runs is decided on the device at every replay.
  ``while`` and the capped ``for _ in range(N): if not cell: break`` become a loop node;
  a ``break`` or ``continue`` in the body guards the rest of it; ``match`` is a switch.

**Buffers.** What a graph allocates is what the hand-built version would: one array per
kernel output that is not written in place. A name that is rebound inside a device
construct and read after it is merged **in place** into the array it held before, when
the kernel is per-sample (it reads and writes only its own sample) and reduces nothing
across samples. If that array is one of your arguments, the graph keeps its own copy and
copies your array into it at the start of every launch, so your array is never
changed. A temporary made inside a loop body is one array, reused each turn.

Forks and ``wait``
------------------

``g.parallel()`` records a fork with one child graph per branch; leaving the ``with``
is the join, and what follows is ordered after all of them. On the GPU each branch is
captured on its own stream. ``b.wait(x)`` becomes one ordering edge: an event recorded on the
stream of the node that wrote ``x``, and a wait on that event in the branch that reads it. In
the captured graph that is a plain edge, so it costs nothing per replay. On the CPU the
branches are tasks and a ``wait`` is an event the consumer's task waits on.

Build
-----

The recorded nodes go into an ordinary :class:`eagle.Graph` and ``build`` runs, as it
does for a hand-made one (the first graph of :doc:`graph_model`).

- **On the GPU** the graph is *captured* once into a CUDA graph and *instantiated*. A
  conditional is a CUDA conditional node (IF, IF/ELSE, SWITCH, WHILE); an ``elif`` chain
  decides its conditions once at the head, into a cell. A driver without IF/ELSE or SWITCH
  nodes gets the same arms from plain IF nodes, with one extra node for the decision.
  Nested conditionals are captured on the stream of the body they sit in.
- **On the CPU** the nodes become a host schedule: each stage runs its nodes one after
  another over the thread team, or, for a fork of small kernels, runs the branches as tasks.

Which side is chosen by where the arrays live; ``g.backend`` and ``g.schedule`` say what was chosen.

Running, ``bind`` and what they cost
------------------------------------

``g()`` is ``launch(1)`` plus returning ``result``: on the GPU one ``cudaGraphLaunch`` and a
wait, on the CPU the closures in order. ``g(n)`` is *n* of those and one wait. Nothing is
parsed, recorded or compiled again.

``g.bind(x=new)`` swaps the arrays by the builder's parameter names. On the GPU it patches the
pointers in the instantiated graph, one call per node that held the array (microseconds each),
with no capture; if a node cannot be patched (your own code over the array, say), eagle
rebuilds and warns once. On the CPU it re-packs the arguments. The shape, dtype and device
must be the same.

Gradients
---------

``record=True`` makes the graph keep what a reverse pass needs, and keeps a map from names to
arrays: each builder argument to the array bound to it, ``result`` to the arrays of the
result tree, and each ``buffer=`` array to its label. :func:`eagle.diff.reverse` reads the map
to turn the names in ``seeds`` and ``wrt`` into arrays, then walks the nodes last to first
running each kernel's derivative (a fork's branches reversed together, a loop's turns
last to first). A name that is not in the map, a plain temporary, has no array that
survives the run, so it is refused with the way out: give it a ``buffer=``.

See also :doc:`graph_forms`, every form of the builder in one executed page.
