# Reading the numbers

What the ensemble card shows, with its own figures. Each statement can be checked against the tables on the [results page](rk78.md).
For the mechanisms behind these numbers see [Where the performance comes from](mechanism.md); for what else is measured, the
[coverage table](coverage.md).

## The workload

An adaptive RKF7(8) integration of two-body Kepler orbits in float64: each sample takes as many steps as its own orbit needs, so
a spread batch (eccentricity from 0 to 0.9) has samples that finish early, and a uniform batch (every eccentricity 0.45) does not.
Every arm runs in its own process on a clean device; its warm-up is also its check against the exact answer. An arm that runs out
of memory is shown as "out of memory", one whose warm-up takes more than 120 s as "> 120 s per run".

<!--ctx card=ensemble dev=h100 n=1e6 dist=spread-->

## Where eagle is ahead

On a spread batch at N = 1,000,000 `eagle.simulate` takes <!--c wall arm=eagle_simulate-->48.9 ms<!--/c-->; the CUDA-graph
device loop takes <!--c wall arm=eagle_graph-->105 ms<!--/c--> and its persistent-launch form
<!--c wall arm=eagle_graph_persist-->49 ms<!--/c-->. NVIDIA Warp's per-thread kernel takes <!--c wall arm=warp_kernel-->113 ms<!--/c-->
(<!--c x arm=warp_kernel den_arm=eagle_simulate-->2.3×<!--/c--> `eagle.simulate`), JAX's `vmap(while_loop)`
<!--c wall arm=jax_vmap-->1.24 s<!--/c--> and the CuPy masked loop <!--c wall arm=cupy_masked-->9.34 s<!--/c-->.

On a uniform batch nothing finishes early, so the lead shrinks: `eagle.simulate` takes
<!--c wall arm=eagle_simulate dist=uniform-->40.8 ms<!--/c--> against Warp's <!--c wall arm=warp_kernel dist=uniform-->55.1 ms<!--/c-->.
Warp's per-thread kernel is the right choice when every sample runs the same number of steps.

## How close to the device's limit

The card's efficiency column is the attempts per second times the instructions of one attempt, over the device's measured FMA
rate: 100 % would be every FP64 unit issuing the step's instructions back to back. On the spread batch the best eagle arm
reaches <!--c eff tool=eagle dist=spread-->82%<!--/c-->, the best Warp arm <!--c eff tool=warp dist=spread-->35%<!--/c--> and the
best CuPy arm <!--c eff tool=cupy dist=spread--><1%<!--/c-->. A row above 100 % is never published: the operation count would be
wrong.

## Memory

The memory pass runs first, one arm per process. `eagle.simulate` needs <!--c mib arm=eagle_simulate-->102 MiB<!--/c--> of device
memory at N = 1,000,000, <!--c xmin arm=eagle_simulate-->1.4×<!--/c--> the bare minimum state; the CuPy masked loop needs
<!--c mib arm=cupy_masked-->936 MiB<!--/c-->.

## On a CPU

<!--ctx card=ensemblecpu dev=amd-epyc-9b45 n=1e6 dist=spread-->
The same workload on CPU threads: the eagle CPU path takes <!--c wall arm=eagle_cpu8-->2.49 s<!--/c--> on eight threads and
<!--c wall arm=eagle_cpu1-->19.9 s<!--/c--> on one; Numba's `prange` takes <!--c wall arm=numba_prange-->19.6 s<!--/c--> and JAX
<!--c wall arm=jax_vmap-->60.5 s<!--/c-->.

## Other devices

The numbers above are one device. The results page has a tab for each device measured; the order of the arms and the size of
the lead change from device to device, which is why the cards are re-run on each.
