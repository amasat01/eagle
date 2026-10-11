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

The rest of the module, one page each:

.. toctree::
   :maxdepth: 1
   :hidden:

   filtering/scan_and_compaction
   filtering/reorder_and_grouping
   filtering/run_until_done

- :doc:`filtering/scan_and_compaction` -- the prefix scan, ``FilteringSlice``, and the active-set index map.
- :doc:`filtering/reorder_and_grouping` -- the occasional physical reorder, and grouping samples by branch.
- :doc:`filtering/run_until_done` -- running every sample to completion in one call.

.. seealso::

   :doc:`../api/api_filtering` — full ``Scanner`` / ``FilteringSlice`` API reference.
