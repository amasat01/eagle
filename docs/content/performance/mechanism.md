# Where the performance comes from

This page explains the shape of the numbers on the ensemble card: where eagle is ahead, where it ties, and why. Each section
states a mechanism, then shows the card figures that bear on it.

<!--ctx card=ensemble dev=h100 n=1e6 dist=spread-->

## Summary

eagle is ahead where samples finish at different times, because it does not spend GPU lanes on samples that are already
finished. It ties where every sample runs the same number of steps, because both approaches then keep every lane busy.

## Lanes stay on live samples

A persistent launch starts a fixed set of workers that pick up the next sample as soon as they free up, so a lane never waits
for the slowest sample of its block. The persistent-launch arm takes <!--c wall arm=eagle_graph_persist-->49 ms<!--/c--> against
<!--c wall arm=eagle_graph-->105 ms<!--/c--> for the graph loop that launches one step per sample slot. `eagle.simulate` picks
between the forms from the batch size and takes <!--c wall arm=eagle_simulate-->48.9 ms<!--/c-->.

## One launch, many steps

A CUDA graph records the whole step loop once and replays it, so the host does no per-step work. The same graph arm on a
uniform batch takes <!--c wall arm=eagle_graph dist=uniform-->51.3 ms<!--/c-->, which is what the device needs when no lane
idles.

## Where compaction and reorder fit

Compaction packs the live samples and reorder re-sorts them when the batch has thinned. They pay on small cards, where idle
lanes cost the most; the committed card carries no cell for them, and the profile kit measures them on its own stage. The
[coverage table](coverage.md) lists them.

## Memory

Every arm keeps one state per sample and no mask planes. At N = 1,000,000 `eagle.simulate` holds
<!--c mib arm=eagle_simulate-->102 MiB<!--/c--> on the device, <!--c xmin arm=eagle_simulate-->1.4×<!--/c--> the bare minimum state.
