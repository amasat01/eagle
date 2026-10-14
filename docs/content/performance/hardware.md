# Hardware caveat

Every card records the device it was measured on: its name, compute capability or CPU model, and, for a GPU,
the FP64 throughput and memory bandwidth the card's ratios are taken against. The devices with a committed
card are listed below; the lists are generated from the cards, so a device added next to them appears here
without an edit.

## GPUs

```{include} /content/_generated/perf_card_gpu_devices.md
```

## CPUs

```{include} /content/_generated/perf_card_cpu_devices.md
```

The method transfers to other hardware unchanged; the absolute numbers, and the crossover points between arms,
differ from device to device, because FP64 throughput, memory bandwidth, launch overhead and core counts differ.
[Reading the numbers](reading.md) describes the NVIDIA H100 card and the AMD EPYC 9B45 card and names the
device wherever it quotes another; each other device has its own tab on the results pages. A card from a new
device is added beside these, produced by the same scripts.

One device is worth a note. The Tesla T4 (Kaggle, power-capped at 70 W) showed
significant SM clock fluctuations between and within arm blocks, so its cells can move more from run to run; each card records the SM clock of every timed run.
