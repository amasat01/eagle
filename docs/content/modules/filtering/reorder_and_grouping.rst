Reorder and grouping
====================

The occasional physical reorder
-------------------------------

A map keeps the live samples' warps full, but as the batch thins out its
gathers touch memory further and further apart. The reorder restores
contiguity without changing any address: ``reorderDevice`` /
``reorderHost`` (``eagle/filtering/Reorder.h``) permute every plane the caller
registers **in place** over ``[0, span)``, through a scratch staging area that
holds a group of planes at a time (one stage kernel and one place kernel per
group of up to 64 planes that fit the staging area, plus a final words kernel;
``reorderScratchBytes`` takes an optional ``stageBytesPerSample`` to make the
groups larger), so the kept samples fill ``[0, count)`` in ascending sample order and the
dropped ones follow them. Pointers baked into a captured graph stay valid.

- ``perm`` (slot → sample id) is permuted with the planes and ``inv``
  (sample id → slot) rebuilt from it, so sample ``i`` is always at
  ``plane[inv[i]]``. The mask is one of the planes.
- The index map keeps holding **physical** slots: right after a reorder it is
  the identity over ``[0, count)``, and the next compactions thin it as usual.
- ``span`` shrinks to ``count`` at every reorder; samples past it are all
  finished and never move again, so a run moves at most
  ``N / (1 - theta)`` samples per plane in total.
- ``restoreDevice`` / ``restoreHost`` put every plane back in sample order and
  reset ``perm`` / ``inv`` to the identity (a full gather per plane, once, at
  the end of a run).

When to reorder is decided by the compaction itself. With a
``ReorderTrigger``, the predicate counts ``live32`` — the aligned 32-sample
groups holding at least one kept sample (one warp ballot each) — and the lane
that writes the count writes ``fire = count < theta * 32 * live32 && count <
span && span >= 4096``: the live samples fill less than ``theta`` of the warps
they occupy, a reorder would shrink the span, and the span is large enough
for the moves to pay. A clumped finishing order (the live samples already
contiguous) never fires; a random one fires about once per halving of the
live set. With a ``fire`` word, every reorder kernel exits while it is 0, and
the reorder clears it.

.. code-block:: cpp

   eagle::filtering::ReorderTrigger trig{ live32, span, fire, 0.5f };
   eagle::filtering::compactDevice(terminated, eagle::filtering::kCompactMaskIsDrop,
                                   map, count, scratch, n, stream, &trig);
   // planes: {data, elemBytes} for every per-sample plane, terminated included;
   // scratch2: reorderScratchBytes(n, widest element) bytes
   eagle::filtering::reorderDevice(planes, nPlanes, terminated,
                                   eagle::filtering::kCompactMaskIsDrop, perm, inv,
                                   map, count, span, fire, scratch2, n, stream);

``eagle::filtering::resetDevice(perm, inv, indexMap, count, span, n, stream)``
(host twin ``resetHost``) puts a set back to the identity in one kernel: ``perm``,
``inv`` and, when non-null, ``indexMap`` become ``0..n-1`` and ``*count = *span = n``.
It is capture-legal and reads no ``fire`` word. The seam symbol is
``eagle_backend_reset_device`` (seam minor 1) and ``eagle._core.reset_device``
its binding; ``ActiveSet.reset`` uses it on a reordering device set.

**Planes in pinned host memory.** A plane may live in mapped pinned host memory
(``cudaHostAlloc`` with the mapped flag, or ``cudaHostRegister`` of a buffer with it):
the kernels read and write it over the bus through its device address, and the
result is byte-identical to the device-memory path. The contract: planes are
structure-of-arrays and at most 16 bytes per element (a 4-double quaternion is
four planes); the host must not touch a plane between the enqueue and the
stream's synchronisation; under graph capture the buffer must be allocated or
registered before the capture and outlive the graph. Pageable memory is not
addressable by the device: ``eagle::filtering::checkPlanesResident(planes, nPlanes,
device)`` throws naming the plane index and its class, ``classifyPointer`` gives the
class (host, device, managed, mapped pinned host, pageable, another device), and both
are setup-time checks, run on every reorder call only under ``EAGLE_DEBUG_MODE``.
The seam symbol is ``eagle_backend_pointer_residency`` (seam minor 1) with the
binding ``eagle._core.pointer_residency``. From Python, ``ActiveSet.own`` on a
device set accepts a numpy plane when it is mapped pinned host memory (for example
``cupyx.empty_pinned``); a pageable numpy plane raises ``TypeError``. The check runs
once, in ``own``.

**When it pays.** The reorder is an explicit opt-in. On the performance
card's oscillator (a memory-bound RK4 step, stop steps spread over two
decades, Quadro P2000) it pays on the spread-1000 distribution from about 1e5 samples: at N = 1e5 it cuts the
spread-1000 wall from |rg_c100k| (compaction alone) to |rg_r100k|, at N = 1e6 from |rg_c1m|
to |rg_r1m|, while a uniform batch (nothing to reorder) costs a little more, and a spread-100 batch (few samples left to recover) can lose at N = 1e5.
Below that it does not pay: at N = 1e3 and 1e4 the reorder arm is about |rg_g1k| and |rg_g10k|
slower than compaction alone (the extra conditional node per loop iteration
and the map-routed stop rule cost more than the coalescing recovers). Use
compaction alone for small batches, and for compute-bound steps, where the
map is already close to the dense ideal.

.. c: wall arm=eagle_graph_compact n=100000
.. |rg_c100k| replace:: 41.4 ms
.. c: wall arm=eagle_graph_reorder n=100000
.. |rg_r100k| replace:: 32.2 ms
.. c: wall arm=eagle_graph_compact n=1000000
.. |rg_c1m| replace:: 374 ms
.. c: wall arm=eagle_graph_reorder n=1000000
.. |rg_r1m| replace:: 253 ms
.. c: pctgap arm=eagle_graph_reorder den_arm=eagle_graph_compact n=1000
.. |rg_g1k| replace:: 4.4%
.. c: pctgap arm=eagle_graph_reorder den_arm=eagle_graph_compact n=10000
.. |rg_g10k| replace:: 7%

From Python, ``eagle.ActiveSet(mask, reorder=0.5)`` (or ``reorder=True`` for
the 0.5 default) turns it on; ``ActiveSet(mask)`` never reorders (``theta`` is
accepted in ``[0.25, 0.75]``; above it a reorder fires at nearly every
compaction of a short run, below it the batch stays scattered long after a
reorder pays). The set owns the mask; :meth:`eagle.ActiveSet.own` hands it the
other per-sample planes to move (the same arrays the plan binds; a
``(D, n)`` C-contiguous array is ``D`` planes), and
:meth:`eagle.ActiveSet.indirect` declares bound planes the caller keeps in
sample order and reads through ``perm``. ``plan.bind`` refuses a bound
per-sample plane that is neither. :func:`eagle.compaction_body` then appends
:meth:`eagle.ActiveSet.reorder_if_degraded` — the reorder under one
conditional node guarded by ``fire`` — after the compaction, so a captured
loop reorders only when it pays (``eagle_backend_reorder_device``).

At every exit the caller sees sample ``i`` at position ``i``:

- **End of run:** :meth:`eagle.ActiveSet.restore` (eager; refused under
  capture).
- **Hooks:** read ``plane[inv[i]]`` (``perm`` and ``inv`` are plain arrays,
  device or host).
- **Exports:** eagle's export doors (``import_buffer`` / DLPack,
  ``from_dlpack``, the framework bridges) refuse an owned plane while it is
  permuted, naming ``restore()``; :meth:`eagle.ActiveSet.in_sample_order`
  returns a gathered copy in sample order instead.

The reorder pays on memory-bound kernels, where gathers through a thin map
dominate; on a compute-bound kernel the map alone is already close to the
dense ideal and the reorder adds little.

Grouping samples by branch: K classes
--------------------------------------

Liveness is a two-way split, live or finished. A kernel whose samples fall
into several *branches* (say, three force models chosen per sample) has a
second divergence problem: when the classes are interleaved, every warp runs
every branch. ``ActiveSet(mask, reorder=theta, classes=K, class_plane=cls)``
generalises the reorder to ``K`` classes: one stable counting sort of the
slots by ``(live, class)``, so the live samples sit in ``K`` contiguous runs,
one per class, followed by the finished ones. ``cls`` is a ``uint8`` plane
holding each sample's class in ``[0, K)`` (``K`` is at most 255); the set owns
it, so it moves with every other plane.

.. code-block:: python

   import numpy as np
   import eagle

   n, K = 8192, 3
   rng = np.random.default_rng(0)
   done = (rng.random(n) < 0.3).astype(np.uint8)     # terminated mask
   cls = rng.integers(0, K, n).astype(np.uint8)       # branch per sample
   x = np.arange(n, dtype=np.float64)                 # a plane to carry along

   active = eagle.ActiveSet(done, reorder=0.5, classes=K, class_plane=cls)
   active.own(x)

   active.compact()          # also computes mixed32 and class_fire
   active.class_sort()       # one stable permutation by (live, class)

   off = active.offsets      # class c occupies slots [off[c], off[c + 1])
   run1 = x[off[1]:off[2]]   # the live samples of class 1, contiguous
   assert off[K] == active.live
   assert (active.in_sample_order(x) == np.arange(n)).all()

   active.restore()          # every plane back in sample order

**One permutation, one pairing.** With ``classes=K`` the liveness reorder *is*
the class sort: there is never a second permutation on top of the first, which
would scatter the class runs. ``perm`` and ``inv`` stay the only planes that
carry sample identity, exactly as for the plain reorder: ``plane[inv[i]]`` is
sample ``i``, ``restore()`` and ``in_sample_order()`` put the caller's order
back, and a sample's value in every owned plane stays together through any
number of sorts. The sort is *stable*: inside a class the samples keep their
relative order, so a sorted run is a pure function of the class plane and the
mask, and the host and device agree to the bit.

**Within the live prefix.** ``class_sort(within_live=True)`` permutes only the
slots ``[0, count)`` among themselves, by class, and leaves ``count`` and
``span`` alone. A pass that grows its live set one prefix at a time (a reverse
sweep) uses it to re-group without ever moving a slot out of its prefix.

**When it fires.** Every compaction of a K-class set also counts ``mixed32``,
the aligned 32-sample groups that hold two or more live classes, and sets
``class_fire``:

.. code-block:: text

   class_fire =  (mixed32 > 0 and mixed32 >= class_theta * live32 and span >= 4096)
              or  stale_count > STALE_MAX * count
              or  fire                       # the liveness trigger

The first term is the divergence proxy: a mixed warp runs both branches, so
once the live samples are mixed across ``class_theta`` of the warps they
occupy (default :data:`eagle._active_set.CLASS_THETA`, 0.5) and the span is
large enough for the moves to pay, a sort pays. The second is for a kernel
whose class is decided by evolving state: it writes ``stale_count``, and the
set re-sorts when more than :data:`eagle._active_set.STALE_MAX` (5 %) of the
live samples changed class. The third keeps a thinning batch from being lost
when the classes happen to be grouped already. All three are integer counts,
so the decision is exact and deterministic; there is no hysteresis at the
trigger, because a sort resets ``mixed32`` and the stale count to zero.
A batch whose classes never change fires only when liveness moves samples.
:meth:`eagle.ActiveSet.reorder` and :meth:`eagle.ActiveSet.reorder_if_degraded`
run the sort exactly when ``class_fire`` is set (the latter under one
conditional node, so it is capturable); ``class_sort()`` itself is
unconditional unless ``if_fired=True``. Each sort bumps ``class_sorts``, never
``reorders``.

``ActiveSet(..., epochs=True)`` additionally keeps the ``perm`` snapshot of every
sort in :attr:`eagle.ActiveSet.epochs`, so a later pass can map a plane saved
before a sort through ``inv`` of that epoch.

**Backends.** The device face is two seam symbols, bound one by one like
``eagle_backend_reorder_device``: ``eagle_backend_compact_classes_device`` and
``eagle_backend_reorder_classes_device``. Their presence is the capability
(``eagle._core.backend_has(symbol)`` reports it); a backend without either
refuses ``classes=K`` naming the backend and the symbol. The C++ faces are
``compactDevice`` plus ``classTriggerDevice`` and ``reorderClassesDevice`` in
``eagle/filtering/ReorderClasses.h``, with serial host twins. The device sort
costs one inclusive scan per class (plus one for the finished samples), so it
suits a handful of classes; with 255 classes expect it to dominate a short
step.
