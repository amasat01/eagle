# Reproduce

From the eagle repository root, in an environment with eagle, hawk and CuPy, an NVIDIA
driver for the GPU cards (no CUDA toolkit or `nvcc`: hawk compiles kernels with NVRTC) and
(optionally) Nsight Systems:

```bash
python benchmarks/perf_card/perf_card.py       # RK4 oscillators, GPU
python benchmarks/perf_card/cpu_card.py        # RK4 oscillators, CPU
python benchmarks/rk78_card/rk78_card.py       # RK7(8) Kepler orbits, GPU
python benchmarks/rk78_card/cpu_rk78_card.py   # RK7(8) Kepler orbits, CPU
```

Each run builds the hawk kernel(s), verifies the arms against each other,
times them, and writes its own `card_<device>.json` together with the
`card_<device>.md` table the results pages show. The JSON carries the md5 of the script
that produced it; a test re-renders the table from the JSON so the two
cannot drift. `--quick` runs a reduced matrix as a smoke test.

To reproduce the cards on another machine (Kaggle, a rented GPU or your own box) with the same pinned software, see [`benchmarks/reproduce/`](https://github.com/amasat01/eagle/tree/main/benchmarks/reproduce).
The P2000 card ran locally, the T4 on Kaggle, and the other cards on cloud VMs through `benchmarks/reproduce/run_cards.sh`.
On Kaggle, choose the GPU T4 x2 accelerator, turn Internet on and run the notebook:
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/amasat01/eagle/blob/main/benchmarks/reproduce/kaggle.ipynb)

**Adding a device.** Drop the new `card_<slug>.json`/`.md` (GPU) or
`cpu_card_<slug>.json`/`.md` (CPU) next to the existing ones under
`benchmarks/perf_card/` or `benchmarks/rk78_card/` -- `benchmarks/reproduce/run_cards.sh
--publish` writes them there directly. Rebuilding the docs (`make html` or `make
strict`) then regenerates the overview and results pages' tabs and tables
on its own; re-running a card for a device already listed replaces
that device's rows the same way. No page needs editing.
