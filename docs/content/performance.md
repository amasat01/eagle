# Performance

RAPTOR lets you focus on the application while still making efficient use of
GPUs. This section measures that on two workloads, a fixed-step one and an
adaptive one, against other common ways of writing the same computation, on
both a GPU and a CPU, and publishes every number the run produced, including
the regimes where another approach is faster.

Wall time to integrate a million samples, each stopping at its own step: one
bar per tool, log scale. Pick a device with the tabs; the pick carries over to
every page in this section.

## RK4 oscillators: GPU

```{include} /content/_generated/perf_card_gpu_overview.md
```

## RK4 oscillators: CPU

```{include} /content/_generated/perf_card_cpu_overview.md
```

## Adaptive RK7(8): GPU

```{include} /content/_generated/rk78_card_gpu_overview.md
```

## Adaptive RK7(8): CPU

```{include} /content/_generated/rk78_card_cpu_overview.md
```

## Where to go next

- [RK4 oscillators: results](performance/rk4.md): every workload, every arm and
  every N, per device.
- [Adaptive RK7(8): results](performance/rk78.md): the same for an adaptive solver.
- [Reading the numbers](performance/reading.md): where each approach wins, and why.
- [The workload and the ways of running it](performance/workload.md): what is
  measured and what each arm does.
- [Method](performance/method.md): how a number is produced.
- [Reproduce](performance/reproduce.md): the cards on your machine or a cloud VM.
- [Hardware caveat](performance/hardware.md): what transfers to other devices.

```{toctree}
:hidden:
:maxdepth: 1

performance/rk4
performance/rk78
performance/reading
performance/workload
performance/method
performance/reproduce
performance/hardware
```
