Module: filtering
=================

**Stream compaction** shrinks a per-sample array down to just the samples
that are still active, so later kernels can launch over a smaller ``N``
(see :doc:`../userguide/foundations` for the full walkthrough). The
``filtering`` module is the machinery that implements it: a parallel
prefix-scan turns each active sample's flag into a **scatter offset** — its
new, packed position in the output array — and a scatter step writes each
active sample there.

Worked example, 8 samples: flags ``[1,0,1,1,0,0,1,0]`` (1 = keep, at indices
0, 2, 3, 6). The *exclusive* prefix scan over those flags — each position's
running count of ``1``\ s *before* it — is ``[0,1,1,2,3,3,3,4]``. Reading
that scan only at the kept positions gives each kept sample's new, packed
index: index 0 → new index 0, index 2 → new index 1, index 3 → new index 2,
index 6 → new index 3. The scatter step then writes each kept sample to its
new index, producing a packed 4-element output: the original samples 0, 2,
3, and 6, in order, with everything else dropped.

Headers
-------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Header
     - Contents
   * - ``eagle/filtering/Scanner.h``
     - ``Scanner`` prefix-scan container plus the non-owning ``RefScanner``.
   * - ``eagle/cuda/Scan.h`` / ``eagle/cpu/Scan.h``
     - The ``cuda::Scan`` (GPU) and ``cpu::Scan`` (CPU) scan engines.
   * - ``eagle/filtering/Slice.h``
     - ``FilteringSlice`` — compaction helper built on top of ``Scanner``,
       plus the non-owning ``RefSlice``.
   * - ``eagle/filtering/Compact.h``
     - Active-set compaction over raw buffers: ``compactDevice`` /
       ``compactHost``, the ascending index map of the kept samples and
       their count, and optionally the reorder trigger (``ReorderTrigger``).
   * - ``eagle/filtering/Reorder.h``
     - The occasional physical reorder behind the map: ``reorderDevice`` /
       ``reorderHost`` permute caller-owned planes in place and keep the
       ``perm``/``inv`` indices; ``restoreDevice`` / ``restoreHost`` undo it.
   * - ``eagle/filtering/scan_detail.h``
     - Low-level prefix-scan and scatter kernels (internal).

``Scanner`` — prefix scan
-------------------------

``Scanner`` computes an inclusive or exclusive prefix scan (prefix sum) in
parallel.  It is the foundation for array compaction: scanning a 0/1 selection
mask yields the scatter offsets of the selected elements.

.. code-block:: cpp

   #include <eagle/eagle.h>

   // Prefix scan over an int array of size N
   eagle::filtering::Scanner<int, aether::SumOp<int>, /*Inclusive=*/true> scanner(N);

   // Fill the input array on the host
   auto ref = scanner.hostRef();
   // ... write the 0/1 selection flags into ref ...

   // CPU scan
   scanner.hostRun();

   // GPU scan
   scanner.upload(stream);
   scanner.deviceRun(/*includeRetrieval=*/true);

   // Number of selected elements after the scan
   idx_t count = scanner.hostRef().numElements();

``FilteringSlice`` — compaction
-------------------------------

``FilteringSlice<DataT>`` wraps a source array of type ``DataT`` together
with its own internal scanner, so compaction is one object instead of two.
Construct it directly from the array to compact; fill its scanner's
selection flags (one ``0``/``1`` per element) through ``slice.scanner()``,
then run it — ``hostUpdate()`` on CPU, or ``upload()`` /
``deviceUpdate()`` / ``download()`` on GPU:

.. code-block:: cpp

   #include <eagle/eagle.h>

   // "observable" here just means an array wrapper that supports
   // FilteringSlice's mutation-notification hooks — any aether array type
   // FilteringSlice supports works the same way.
   using ArrT = eagle::util::observable::Array<idx_t>;
   ArrT source(N);

   eagle::filtering::FilteringSlice<ArrT> slice(source);

   // Fill the 0/1 selection flags on the host
   auto& flags = slice.scanner().inputs();
   for (idx_t i = 0; i < source.size(); ++i)
       flags[i] = keepThisSample(i) ? 1 : 0;

   // CPU: scan + scatter in one call
   slice.hostUpdate();

   // GPU: upload flags, scan + scatter, download the compacted result
   slice.upload();
   slice.deviceUpdate(/*includeRetrieval=*/true);

   // slice.hostRef() now exposes only the selected elements, packed

Active-set compaction — an index map, nothing moved
----------------------------------------------------

``compactDevice`` / ``compactHost`` (``eagle/filtering/Compact.h``) run the
same predicate → scan → scatter over raw caller buffers and produce an
**index map** instead of a packed copy: ``map[0, count)`` holds the indices of
the kept samples in ascending order, and ``count`` their number. A per-sample
kernel that reads the map (thread ``t`` → sample ``map[t]``, exiting at
``count``) then packs the live samples into full warps while every sample's
data stays in its own slot. The device face only enqueues kernels on its
stream — no allocation, no synchronisation, no host read — so it can be
recorded into a CUDA graph and replayed against a mask that changes between
replays.

.. code-block:: cpp

   #include <eagle/eagle.h>

   // terminated: n bytes on the device (set = finished)
   // map: n int32, count: one uint32, scratch: compactScratchBytes(n) bytes
   eagle::filtering::compactDevice(terminated, eagle::filtering::kCompactMaskIsDrop,
                                   map, count, scratch, n, stream);

From Python the same step is :class:`eagle.ActiveSet` (through the CUDA
backend's ``filtering`` capability group, ``eagle_backend_compact_device``),
and :func:`eagle.compaction_body` builds the loop body that compacts every
few steps inside :func:`eagle.repeat_while`.

The occasional physical reorder
-------------------------------

A map keeps the live samples' warps full, but as the batch thins out its
gathers touch memory further and further apart. The reorder restores
contiguity without changing any address: ``reorderDevice`` /
``reorderHost`` (``eagle/filtering/Reorder.h``) permute every plane the caller
registers **in place** over ``[0, span)``, through one scratch staging plane,
so the kept samples fill ``[0, count)`` in ascending sample order and the
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

**When it pays.** The reorder is an explicit opt-in. On the performance
card's oscillator (a memory-bound RK4 step, stop steps spread over two
decades, Quadro P2000) it pays from about 1e5 samples: at N = 1e5 it cuts the
spread-1000 wall from 45.6 ms (compaction alone) to 34.5 ms, at N = 1e6 from 416 ms
to 280 ms, while a uniform batch (nothing to reorder) costs at most 4 % more.
Below that it does not pay: at N = 1e3 and 1e4 the reorder arm is about 6 %
slower than compaction alone (the extra conditional node per loop iteration
and the map-routed stop rule cost more than the coalescing recovers). Use
compaction alone for small batches, and for compute-bound steps, where the
map is already close to the dense ideal.

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

Running until every sample is done, in one call
------------------------------------------------

A step kernel can finish its own samples: it assigns its ``terminated`` mask
(``terminated = t >= t_final``) and counts each newly finished sample into
one counter plane, ``finished_count`` (:data:`eagle.FINISHED_PLANE`). Its
sidecar then carries a ``finish`` declaration
(:func:`eagle.sidecar.read_finish`), and :func:`eagle.run_until_done` runs the
whole batch:

.. code-block:: python

   report = eagle.run_until_done(plan, max_steps=10_000,
                                 x=x, v=v, k=k, omega=omega, dt=1e-3)
   assert report.done            # report.launches, .compactions, .reorders, ...

The call binds the plan once (the caller passes every plane but the counter,
the active-set planes and, optionally, the mask, which the runner allocates)
and builds the loop above: a :func:`eagle.repeat_while` guarded by
``finished_count != n``, and, for an artifact built with
``Guard(active_set=True)``, the :func:`eagle.compaction_body` cadence over an
:class:`eagle.ActiveSet`, with ``reorder=theta`` for the physical reorder.
Whether it compacts is read off the artifact, never a flag. On the device the
loop is one CUDA graph (one WHILE node that reads the counter on the device),
captured on the first run and replayed after; a reordering run ends with the
planes back in sample order.

:func:`eagle.until_done` returns the :class:`eagle.Runner` instead:
``runner.run()`` runs (a second run continues from the current mask, so it
launches nothing once everything is done), ``runner.reset()`` clears the mask
and the counter, and ``runner.loop`` is the :class:`eagle.RepeatWhile` to
compose into a larger :class:`eagle.GraphPipeline`. The explicit path stays
public; the runner is written in it.

Letting the run pick the steps per launch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A kernel built with ``@hawk.kernel(steps="auto")`` (or
``hawk.steps(kernel, "auto")``) takes up to 64 steps per launch and reads how
many from a one-cell device word, :data:`eagle.FUSED_STEPS_PLANE`, that the
runner owns. The same call runs it; nothing else changes for the caller:

.. code-block:: python

   report = eagle.run_until_done(plan, max_steps=10_000, x=x, v=v, dt=1e-3)
   report.launches_by_k          # e.g. {16: 1, 32: 1, 64: 15}
   assert report.steps <= 10_000 # the steps the loop ran, never above max_steps

After each launch one single-thread policy kernel picks the next launch's
steps: it doubles them when fewer than 1/16 of the live samples finished
during the launch, halves them when more than 1/4 did, and keeps them between
8 and 64 (starting at 16). On a batch whose samples all run about as long it
climbs to 64 steps per launch, the same fusion a hand-written multi-step
kernel gets; on a batch whose stop steps are spread out it stays shorter, and
an artifact built with ``Guard(active_set=True)`` compacts after every launch
in which a sample finished, so the live samples keep sharing warps. The last
launch takes exactly the steps left, so no sample takes more than
``max_steps`` steps. :attr:`eagle.RunReport.steps` is the steps the loop ran:
a sample still running at the end took exactly that many, and the
longest-running sample of a finished batch may have stopped inside the last
launch (``report.exact_steps`` is ``True`` when ``steps`` is exactly what that
sample attempted: the cap stopped the run, or the last launch took one step).
A launch whose word is 1 runs the kernel's straight-line single step, so it
costs what the one-step kernel costs.

The cadence is the policy's, so ``every=`` is refused for such a kernel;
``reorder=theta`` still opts into the physical reorder (it is off by default,
because a fused launch reads each plane once per launch and gains less from
it). On the host a one-step launch runs the single step vectorised across
samples while a fused launch runs each sample's scalar loop, so the host
measures instead of following the band (``runner.host_policy ==
"measured"``, the default): it runs one step per launch and, each time the
live samples have halved, times one fused launch of 8 steps; once that beats
the one-step launch per step it continues under the band. A batch whose
samples all run to the end never probes and runs exactly the one-step loop.
``runner.host_policy = "band"`` runs the device's band on the host (the same
launches as the device). The results are those of one step per launch:
bit-equal on the host; on the device the fused loop may round a contracted
multiply-add differently from the single step, never from its own body run one
step at a time.

Termination is discrete: a ``vjp``/``jvp`` derived from a finishing kernel
does not finish samples (it zeroes terminated samples, as before). Launch the
derived kernels before the primal in a step, so they read the mask as it was
when the step began.

.. seealso::

   :doc:`../api/api_filtering` — full ``Scanner`` / ``FilteringSlice`` API reference.
