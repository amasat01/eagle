# Run one kernel across several GPUs

`eagle.exec.RankPartition` splits a kernel's samples across MPI ranks. Every rank ends with the whole output, bit for bit the same as one rank would have computed. Give each rank its own GPU and the samples are spread over the cards.

You need the optional `eagle._mpi` extension, built with `-DEAGLE_PYTHON_MPI=ON` (the default `AUTO` builds it when an MPI is found). Without it every call raises `RuntimeError` naming the switch. `eagle.exec.RankPartition.available` says which you have, and `RankPartition.world()` gives `(rank, size)`, `(0, 1)` without `mpirun`.

## Write it once

The plan is the same at any world size. Pass `RankPartition` as the structure and name the structure each rank's share runs through: `eagle.exec.DeviceKernel` for a GPU, `HostTeam` (the default) for CPU threads.

```python
import eagle.exec as eexec
from eagle import plan as eplan

p = eplan.plan(plugin, structure=eexec.RankPartition, inner=eexec.DeviceKernel)
rank, size = eexec.RankPartition.world()
out = p.run(device=rank, **planes)          # every rank returns the whole plane
```

`plugin` is a deployed kernel (see [Launch a kernel and use the registry](tutorials/01_launch_and_registry)) and `planes` are its inputs by name. `device=rank` uploads this rank's share to GPU number `rank`; eagle does not choose the card for you. Set `CUDA_VISIBLE_DEVICES` per rank in your launcher instead if you prefer.

## Launch it

```bash
mpirun -np 2 python my_run.py
```

The same script run alone (`python my_run.py`) is one rank and gives the same answer.

## What is split, and what is not

- **Split:** the samples. Rank `r` of `R` takes one contiguous block; when the count does not divide evenly, the first `n % R` ranks take one extra.
- **Replicated:** the inputs. Every rank holds all of them, so a kernel that reads other samples' inputs (`cross_sample_read`) is legal.
- **Gathered:** the outputs, through the host. They must be `float64` and C-contiguous; another dtype is refused by name before anything launches. Device outputs are copied home, so this is not a way to keep a plane resident on one card.
- **Reductions:** a `mapreduce` combines the per-rank partials in a fixed rank order, never with atomics, so every rank gets the same value. Kernels that write cells are cut at whole rows of 32 samples, and each rank finishes the rows exactly as a single rank would. See [Determinism](determinism).
- **Refused:** a kernel that writes other samples (`cross_sample_write`) cannot be rank-partitioned, at any world size, and the plan says so.

`eagle.exec.RankPartition.local(partition)` gives this rank's share of a partition, and `allgather_plane` and `allgather_fold` are the two collectives the run uses.
