Running until every sample is done
==================================


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
-----------------------------------------

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
samples while a fused launch runs each sample's scalar loop, so the host does not simply follow the band. ``runner.host_policy`` defaults
to ``"auto"``: ``"tiled"`` (``steps_max`` steps per launch) when the entry's
fused side is tiled, else ``"measured"``, which runs one step per launch and,
each time the live samples have halved, times one fused launch of 8 steps;
once that beats the one-step launch per step it continues under the band. A
batch whose samples all run to the end never probes and runs exactly the
one-step loop. ``runner.host_policy = "band"`` runs the device's band on the
host (the same launches as the device). The results are those of one step per launch:
bit-equal on the host; on the device the fused loop may round a contracted
multiply-add differently from the single step, never from its own body run one
step at a time.

Termination is discrete: a ``vjp``/``jvp`` derived from a finishing kernel
does not finish samples (it zeroes terminated samples, as before). Launch the
derived kernels before the primal in a step, so they read the mask as it was
when the step began.

Fused or persistent, picked for you
-----------------------------------

On the device a kernel can have two entries: the fused one (a launch per
loop turn) and a persistent one (one launch whose lanes claim samples until
none are left). The runner chooses between them, and the choice needs no
flag:

- run 1 takes the model's pick of the two, sized without any probing;
- run 2 runs the other entry, so both have a measured wall;
- run 3 times the pick again after the sizing was refitted to what the first
  runs read back;
- later runs are settled: they carry no timing events.

The pick changes only when the other entry wins by more than 2 %
(``eagle._until_done.PICK_HYSTERESIS`` is 0.98), so two near-equal entries
do not trade places. A settled runner looks again when a run reads back
something different from before, and every 64 runs.
:attr:`eagle.RunReport.timed` says whether a run was timed, and
``report.probe`` whether it ran the entry the batch was not using, so a
caller that wants a clean measurement warms until a run is neither.

Persistent launches take ``chunk=``, the steps a lane runs between two looks
for new samples (default 32, or whatever the launch policy chose for the
batch's mean steps). :attr:`eagle.Runner.launch_policy` reports what was
chosen: ``block``, ``grid``, ``chunk``, ``fill``, ``rate`` and ``k``.
:attr:`eagle.RunReport.slot_fill` is the steps run over the persistent
entry's chunk slots, in ``(0, 1]``; it is set only for a persistent run that
counted it, ``None`` otherwise. Pair the policy with
:meth:`eagle.plan.BoundPlan.entry_reports`, whose entries carry registers,
local memory, ``text_bytes`` (code size) and ``demand`` (threads per SM the
launch asks for).
