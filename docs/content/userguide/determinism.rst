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

EAGLE's answer is to fix the grouping. Its sum is defined as one specific
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
therefore the bits. EAGLE's own tests plant exactly these wrong orders and
require that they *do* differ from the canonical result, so the checks cannot
pass by accident.

In a graph: ``eagle.reduce``
----------------------------

From Python the canonical tree is a node of :class:`eagle.Graph`.
``eagle.reduce(x, out, op="sum")`` reads the 1-D plane ``x`` and writes the
one-element plane ``out``; ``op`` is ``"sum"``, ``"max"`` or ``"all"`` (numpy's
names). Because the node declares what it reads and writes, eagle derives its
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
