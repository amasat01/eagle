# Hardware caveat

The GPU cards come from a Quadro P2000, a development, consumer-class
card whose FP64 throughput (1:32 of its FP32 rate) and memory bandwidth are
recorded in each card, and from a Kaggle Tesla T4; the CPU cards come from an Intel Xeon W-2125
workstation chip (4 cores, 8 logical CPUs). The method transfers to other
hardware unchanged; the absolute numbers, and possibly the crossover points
between arms, will differ on data-centre GPUs or larger CPUs, whose FP64
throughput, bandwidth and core counts differ from these cards'. A card from
such a device would be added beside these, produced by the same scripts.

The GPU card is measured on two devices: the Quadro P2000, the reference card for every ratio quoted in Reading the numbers, and a Tesla T4 (Kaggle, power-capped at 70 W, CUDA 12.6). On the T4 we observed significant SM clock fluctuations between and within arm blocks, so its cells move more from run to run than the P2000's; the card records the SM clock of every timed run.
