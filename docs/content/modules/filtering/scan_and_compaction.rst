Scan and compaction
===================

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

Compacting by groups
~~~~~~~~~~~~~~~~~~~~

Some kernels need whole groups of lanes together: a sum over a lane space's fastest axis
adds the lanes of one group, so a group with one live lane must run in full. A hawk kernel
declares this with ``@hawk.kernel(group=32)``, and ``ActiveSet(mask, group=32)`` then keeps a
group while any of its lanes is live. ``map`` holds group indices and ``count`` is the number
of lanes launched, the live groups times the group size.

.. code-block:: python

   import numpy as np
   import eagle

   done = np.zeros(100, dtype=bool)
   done[:70] = True                       # lanes 0 to 69 have finished ...
   done[75] = True                        # ... and lane 75

   by_lane = eagle.ActiveSet(done)
   by_lane.compact()
   by_group = eagle.ActiveSet(done, group=32)
   by_group.compact()
   print(int(by_lane.count[0]), int(by_group.count[0]), by_group.map[:2])

.. code-block:: text

   29 64 [2 3]

Groups 0 and 1 are finished, groups 2 and 3 (the last one partial) are not. ``reorder=`` does
not combine with ``group=``. At bind, a kernel that declares a group in its sidecar's
``launch.lane_space`` is checked against the set: a set that compacts at another group is
refused, naming both groups.
