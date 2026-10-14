# Where the performance comes from

This page explains the shape of the numbers on the cards: where eagle is ahead, where it ties, and why. Each section states a mechanism and then shows the card figures that bear on it. The numbers come from the committed [performance cards](reading.md) and from the large-N [compare evidence](https://github.com/amasat01/eagle/tree/main/benchmarks/compare_card); none is typed by hand. Unless a sentence names another device it is about the NVIDIA H100 card, RK4 oscillators, N = 10⁶ samples, up to 1000 steps each.

<!--ctx card=rk4 dev=h100 n=1e6 dist=spread steps=1000-->

## Summary

eagle is ahead where samples finish at different times, because it does not spend GPU lanes on samples that are already finished. It ties where every sample runs the same number of steps, because both eagle and NVIDIA Warp's per-thread kernel then sit at the GPU's FP64 limit and there is nothing left to recover. The sections below take the mechanisms one at a time.

## Lanes stay on live samples

In a per-thread kernel each lane owns one sample for the whole run. When that sample stops, its lane keeps its place in the warp until the slowest sample in the same warp stops, so a thinning batch costs nearly as much as a full one. eagle's persistent path launches once, sized to what the device can keep resident, and each lane takes the next unfinished sample from a shared counter when its own is done. The measure of the difference is useful work per lane: more of the lanes executing each instruction are on live samples. It is not a claim about fewer stalls; issue slots are about equally busy in both kernels.

The cards show the effect as the gap between the two distributions. Going from the uniform batch to the spread batch (samples stopping at their own step counts, so far fewer sample-steps in total), `warp_kernel` takes <!--c x arm=warp_kernel den_arm=warp_kernel den_dist=uniform-->0.90×<!--/c--> of its uniform time, while `eagle.simulate` takes <!--c x arm=eagle_simulate den_arm=eagle_simulate den_dist=uniform-->0.34×<!--/c-->. The per-thread kernel does almost the same amount of lane-time on either shape; eagle's time follows the work that is left. On the spread batch `warp_kernel` takes <!--c wall arm=warp_kernel-->1.54 ms<!--/c--> against `eagle.simulate`'s <!--c wall arm=eagle_simulate-->613 µs<!--/c-->.

## No per-step launch cost

Launching one kernel per step costs microseconds multiplied by the number of steps, and it is paid whether the batch holds a thousand samples or a million. The automatic policy avoids it by fusing several steps into one launch or by keeping one kernel resident until the batch is done. The card's own per-step loop, which launches once per step from the host, shows the cost: at N = 10⁶ on spread-1000 it takes <!--c wall arm=eager_loop-->32.3 ms<!--/c--> on the H100 against <!--c wall arm=eagle_simulate-->613 µs<!--/c--> for `eagle.simulate`, and <!--c wall dev=p2000 arm=eager_loop-->723 ms<!--/c--> against <!--c wall dev=p2000 arm=eagle_simulate-->150 ms<!--/c--> on the P2000.

The same cost matters more the faster the GPU, because the compute it sits next to shrinks while the launch cost does not: the per-step loop is <!--c x dev=p2000 arm=eager_loop den_arm=eagle_simulate-->4.8×<!--/c--> `eagle.simulate` on the P2000 and <!--c x arm=eager_loop den_arm=eagle_simulate-->52×<!--/c--> on the H100.

## When compaction and reorder pay off

Compaction packs the live samples into the first warps, and the reorder sorts them so that samples with similar remaining work share a warp. Both cost extra launches and passes over the batch. On the smaller cards the saved lane time is worth more than that cost: on the P2000 at N = 10⁶ on spread-1000 the plain graph takes <!--c wall dev=p2000 arm=eagle_graph-->586 ms<!--/c--> and the compaction-plus-reorder arm <!--c wall dev=p2000 arm=eagle_graph_reorder-->253 ms<!--/c-->; on the T4 the figures are <!--c wall dev=t4 arm=eagle_graph-->173 ms<!--/c--> and <!--c wall dev=t4 arm=eagle_graph_reorder-->113 ms<!--/c-->. On the A100 and the H100 the same arm loses: <!--c wall dev=a100 arm=eagle_graph-->5.33 ms<!--/c--> against <!--c wall dev=a100 arm=eagle_graph_reorder-->17.2 ms<!--/c--> on the A100 and <!--c wall arm=eagle_graph-->2.13 ms<!--/c--> against <!--c wall arm=eagle_graph_reorder-->10.7 ms<!--/c--> on the H100, because those devices finish the batch before the bookkeeping has anything to recover.

The automatic policy chooses between these arms for the device and the batch, so a caller does not have to know which side of this line the GPU falls on.

## Scaling with N

At small N every tool sits on a floor set by the longest sample's chain of steps. It is not a launch cost: on the H100 at N = 10³ the automatic arm takes <!--c wall arm=eagle_graph_auto n=1000 steps=100-->42.4 µs<!--/c--> when samples run up to 100 steps and <!--c wall arm=eagle_graph_auto n=1000-->94.9 µs<!--/c--> when they run up to 1000, so the floor grows with the step count. At N = 10⁴ the two tools are level (`warp_kernel` takes <!--c x arm=warp_kernel den_arm=eagle_simulate n=10000-->0.98×<!--/c--> of `eagle.simulate`'s time).

Beyond that the gap opens as there is more work per lane to recover. On the H100, `warp_kernel` takes <!--c x arm=warp_kernel den_arm=eagle_simulate-->2.5×<!--/c--> `eagle.simulate`'s time at N = 10⁶ on the card, <!--c cmp dev=h100 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->3.1×<!--/c--> at N = 10⁷ and <!--c cmp dev=h100 n=1e8 dist=spread steps=1000 arm=warp den_arm=auto-->3.2×<!--/c--> at N = 10⁸ (compare evidence: one session per device, no repetitions rule, so a measurement at load rather than a certified figure). A million samples is a light load for this device, and the gap keeps growing as the batch grows.

## Across GPUs, against NVIDIA Warp

At N = 10⁷ on spread-1000, with walls normalised to a common SM clock, `warp_kernel` takes <!--c cmp dev=p2000 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->3.7×<!--/c--> the time of the automatic arm on the P2000, <!--c cmp dev=t4 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->3.9×<!--/c--> on the T4, <!--c cmp dev=a100 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->3.5×<!--/c--> on the A100 and <!--c cmp dev=h100 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->3.1×<!--/c--> on the H100. The gap is larger on the smaller GPUs. The per-thread kernel pays about the same relative dead-lane cost on every device: its spread time is <!--c x dev=p2000 arm=warp_kernel den_arm=warp_kernel den_dist=uniform-->0.88×<!--/c--> of its uniform time on the P2000 and <!--c x arm=warp_kernel den_arm=warp_kernel den_dist=uniform-->0.90×<!--/c--> on the H100. What varies from card to card is how much of eagle's own throughput survives a spread batch.

This trend holds against Warp only. Against array-style tools (JAX, CuPy) the gap grows on bigger GPUs, for a different reason: they make whole-array passes over finished samples, so their cost follows the batch size rather than the work that is left.

## Where eagle does not win

On the dense uniform batch, where every sample runs every step, there is nothing to reclaim and the two tools tie. On the H100 at N = 10⁶ `warp_kernel` takes <!--c wall arm=warp_kernel dist=uniform-->1.71 ms<!--/c--> and `eagle.simulate` <!--c wall arm=eagle_simulate dist=uniform-->1.75 ms<!--/c-->; at N = 10⁷ the compare evidence has <!--c cmp dev=h100 n=1e7 dist=uniform steps=1000 arm=warp-->15.9 ms<!--/c--> for Warp and <!--c cmp dev=h100 n=1e7 dist=uniform steps=1000 arm=auto-->16.8 ms<!--/c--> for the automatic arm. Warp is the better fit for that shape.

Cold compile is the other place: JAX's first call takes <!--c comp key=jax:1000 which=cold-->242 ms<!--/c--> against `eagle.simulate`'s <!--c comp key=eagle:eagle_simulate which=cold-->875 ms<!--/c-->, and <!--c comp key=jax:1000 which=warm-->40 ms<!--/c--> against <!--c comp key=eagle:eagle_simulate which=warm-->70 ms<!--/c--> once both are warm.

## What is not shown yet

The H100 gap at large N stays smaller than the smaller cards' gap. There is a hypothesis for it, but no profiler evidence yet, so it is not stated here as a result; it is the subject of a study in a future release.
