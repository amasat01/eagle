Same bits, every time: reductions
=================================

.. contents:: On this page
   :local:
   :depth: 2

Adding up a million floating-point numbers sounds like it has one right
answer. It does not: floating-point addition is not associative, so
``(a + b) + c`` and ``a + (b + c)`` can differ in the last bits, and a parallel
sum is free to group the additions however its threads happen to split the
work. The usual symptom is a total that changes when you change the number of
OpenMP threads, or move from one machine to another, even though nothing about
your data changed.

eagle's answer is to fix the grouping. Its sum is defined as one specific
tree of additions — the **canonical tree** — and every implementation computes
exactly that tree. This page explains the tree, what it promises, and what it
deliberately does not.

The canonical tree
------------------

Cut the samples, in sample order, into **leaf blocks** of 256 consecutive
samples (the last block may be shorter).

1. **Leaf.** Inside a block, slot ``t`` is combined with slot ``t + 128``, then
   ``t + 64``, and so on down to ``t + 1``. This is the shape a 256-thread CUDA
   block naturally uses in shared memory; the host runs the same steps in the
   same order. A slot past the end of the data is simply absent (it is not
   padded with zeros, which would flip a ``-0.0`` result to ``+0.0``).
2. **Pairwise levels.** The leaf results, in block order, are combined in
   pairs: ``(0, 1)``, ``(2, 3)``, ... If a level has an odd number of nodes,
   the last one is carried up unchanged. Repeat until one value is left.

That value is the answer. Because it is a function of the values in sample
order and nothing else, it does not depend on how many OpenMP threads ran, on
how the host scheduled the leaves, or on the CUDA grid.

.. code-block:: cpp

   #include <eagle/eagle.h>

   // Host: identical bits for OMP_NUM_THREADS = 1, 2, 4, 8, ...
   double total = eagle::cpu::Reduction<double, aether::SumOp<double>>::reduce(
       hostRef, /* init= */ 0.0);

   // Or call the tree directly, on any callable in(i):
   double same = eagle::reduce::canonical::reduce<double, aether::SumOp<double>>(
       hostRef, hostRef.samples());

``cpu::Reduction`` uses the tree for floating-point sums and for any operator
you supply yourself. Integer sums, ``MaxOp`` and ``LogicalAndOp`` are exactly
associative, so they keep their native OpenMP clauses: any schedule gives the
same value.

On the GPU
----------

``eagle::reduce::canonical::enqueue`` computes the same tree on a stream: one
256-thread block per leaf, then one small kernel per pairwise level. For a sum
whose operands and additions are the same, the device result matches the host
result bit for bit.

.. code-block:: cpp

   const idx_t nl = eagle::reduce::canonical::deviceWorkElems(n);
   auto a = eagle::makeArray<double>(nl);
   auto b = eagle::makeArray<double>(nl);
   a.upload(); b.upload();

   double* result = eagle::reduce::canonical::enqueue<double, aether::SumOp<double>>(
       arr.deviceView().as_const(), n, a.deviceView().data(),
       b.deviceView().data(), stream);
   // after the stream has run, element 0 behind `result` holds the sum

Splitting the work into chunks
------------------------------

When the data arrives in pieces you can reduce each piece and combine the
results, and still get the same bits as one big reduction. The rule is that
every chunk holds ``256 * 2^q`` samples (the last may be shorter) and the chunk
results are combined with the **same pairwise function** — each chunk is then
exactly one subtree of the big tree. ``reduceChunked`` does this and refuses a
chunk size that breaks the rule.

Two things look equivalent and are not: folding the chunk results from left to
right, and choosing a chunk of, say, three leaves. Both change the grouping and
therefore the bits. eagle's own tests plant exactly these wrong orders and
require that they *do* differ from the canonical result, so the checks cannot
pass by accident.

In a graph: ``eagle.reduce``
----------------------------

From Python the canonical tree is a node of :class:`eagle.Graph`.
``eagle.reduce(x, out, op="sum")`` reads the 1-D plane ``x`` and writes the
one-element plane ``out``; ``op`` is ``"sum"``, ``"max"``, ``"min"`` or ``"all"``
(numpy's names). Because the node declares what it reads and writes, eagle derives its
place in the graph for you: after the nodes that produce ``x``, before the
nodes that read ``out``, including after a fork/join.

.. code-block:: python

   import numpy as np
   import eagle

   total = np.zeros(1)
   g = (eagle.Graph()
        .add_group([[step_a], [step_b]])        # fork, then join
        .add_node(eagle.reduce(energy, total), name="total")
        .build())
   g.launch()          # the same bits with 1 thread or 64

On numpy arrays the node runs the host tree; on cupy arrays it enqueues the
device tree on the stream being captured, with two small work buffers owned by
the built graph. Launch it as many times as you like: each launch gives the
same bits for the same values, and a numpy ``x`` and its cupy copy give the
same bits too. A floating-point ``"sum"`` is where the order matters. For
integers, ``"max"`` and ``"all"`` every order gives the same value, and the
node simply uses the same tree.

Policies
~~~~~~~~

A floating-point sum takes ``policy=``. Every policy runs the same leaf blocks; they
differ in how the leaf results are combined.

* ``"compensated"`` (the default) is an accurate sum. On the GPU it commits the leaves with
  two-sum atomics, so the bits can differ from run to run; the CPU runs ``"ordered"``.
* ``"ordered"`` combines the leaves pairwise, level by level: the same bits on every run,
  every thread count, CPU and GPU.
* ``"auto"`` is ``"compensated"`` in this release.

Integer sums, ``"max"``, ``"min"`` and ``"all"`` are exact and ignore the policy.

A float64 sum on a GPU with slow FP64 takes a different path. Each term is added exactly
into a wide integer, using integer instructions only, and the leaves are combined in the
fixed pairwise levels under both policies. The result is the same bits on every run, and
the host runs the same integer arithmetic, so host and device agree too; it is also about
2.5 times faster on those GPUs. float32 sums, and float64 on GPUs with fast FP64, use the
compensated sums above.

.. code-block:: python

   total = np.zeros(1)
   node = eagle.reduce(energy, total, policy="ordered")

.. _determinism-cells:

Cells from a kernel
-------------------

A hawk kernel that stores ``hawk.all_samples(x).sum()`` (or ``.mean()``, ``.max()``,
``.min()``) to a ``Mutable`` output makes a **cell**: one value for the whole batch. The
kernel does not write that value itself. It writes **rows**, one partial per 32 samples,
and a ``<name>_comp`` row of rounding corrections for a sum. A vector or matrix cell is one
such pair per entry (``total_0``, ``total_1``, ...).

eagle turns rows into the cell with ``eagle.reduce``, taking the rows as ``x``, the
corrections as ``comp=`` and, for ``"max"``/``"min"``, the operation as ``op=``. Every
door that launches such a kernel does this for you: ``eagle.deploy(...).run`` and
``eagle.simulate`` return the cell finished, as a one-element array (a vector cell is one
element per column). The scratch for the rows is made once per kernel and sample count, never
per launch, so it is safe inside a captured graph. Calling the compiled kernel with
``hawk.run`` gives the same cell, added on the host, bitwise equal to ``policy="ordered"``.

Inside ``jax.jit`` the JAX bridge finishes the cells in the same XLA call; see
:doc:`../interop/jax`. It uses the default policy, since the bridge has no ``policy=``
argument yet. Where the bits must be fixed, finish the values yourself with
``eagle.reduce(..., policy="ordered")`` or take the cell from ``hawk.run``.

Reading a total later in the same kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A kernel that reads a reduced value after computing it, such as
``z = x / hawk.all_samples(x).sum()``, is built as stages: one launch, the finish of its
cells, then the next launch, with the sidecar listing them under ``stages``.
``Plan.run`` and ``Plan.bind`` run the stages in sequence, and so does every door built on
them: a ``Graph``, the reverse, and the torch, JAX and Warp bridges. ``eagle.until_done``
and ``eagle.simulate`` launch one kernel per step and cannot run the finish between stages,
so they refuse a split kernel by name (``'<kernel>' runs in N stages; <door> runs single-stage kernels in this release``); use ``eagle.plan.auto(kernel).run`` or ``.bind``.

Limits
~~~~~~

* Folding a kernel's cells across MPI ranks (an ``eagle.exec.RankPartition`` run) is refused in this
  release (our scope; it returns in the next): run such a kernel on one rank.

What is promised, and what is not
---------------------------------

* The canonical tree is **fixed-order and atomic-free**: run-to-run identical,
  and on the CPU identical for every thread count.
* ``eagle::cuda::Reduction`` (the block-iterated GPU reducer) keeps serving C++
  users under a weaker promise: fixed order *for a given launch geometry*. It
  is not the canonical tree.
* A scatter-add that accumulates from many threads with atomics is outside this
  promise: the order in which atomics land is not fixed.

.. seealso::

   :doc:`../modules/module_reduce` — the reduction module, and
   :doc:`../api/api_reduce` — the API reference.
