# early_termination card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-17T07:48:44Z

## Facts

```json
{
  "device": {
    "clock_rate_hz": 1480500000.0,
    "compute_capability": "6.1",
    "cuda_driver_api": 12060,
    "cuda_runtime": 12090,
    "driver_version": "560.35.03",
    "memory_bus_width_bits": 160,
    "memory_clock_hz": 3504000000.0,
    "name": "Quadro P2000",
    "nvidia_smi": {
      "clocks_max_mem": "3504 MHz",
      "clocks_max_sm": "1721 MHz",
      "driver_version": "560.35.03",
      "memory_total": "5120 MiB",
      "name": "Quadro P2000",
      "power_limit": "75.00 W"
    },
    "pci_bus_id": "00000000:21:00.0",
    "sm_count": 8,
    "total_memory_bytes": 5283250176
  },
  "example_script": "examples/python/early_termination.py",
  "example_script_md5": "729fb616a0517c03fe7caf02ec3cf39f",
  "generated_utc": "2026-10-17T07:48:44Z",
  "peak": {
    "datasheet_cross_check": "Quadro P2000 data sheet: 3.0 TFLOPS peak FP32 (FP64 is 1/32 of FP32 on compute capability 6.1), 140 GB/s memory bandwidth",
    "dram_bytes_per_s": 140160000000.0,
    "dram_source": "2 x memory clock (cudaDeviceProp::memoryClockRate) x bus width (cudaDeviceProp::memoryBusWidth) / 8",
    "fp64_flops": 94752000000.0,
    "fp64_source": "SM count x FP64 FMA per clock per SM (CUDA C++ Programming Guide, Arithmetic Instructions throughput table, compute capability 6.1: 4) x 2 FLOP per FMA x the device's reported maximum clock (cudaDeviceProp::clockRate)"
  },
  "script_md5": "c2ff1e53ab1b87199ab86c98356f2ef6",
  "software": {
    "cpu": "Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz",
    "cupy": "14.2.0",
    "eagle_commit": "3f9a26785f55bcd5caac37c950d9263451469a27",
    "hawk": "0.5.1",
    "hawk_commit": "127857f8a327e96c9becf32d1cb7e85df9f5e198",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "numpy": "2.5.3",
    "omp_num_threads": null,
    "platform": "Linux-6.12.0-100.28.2.el9uek.x86_64-x86_64-with-glibc2.34",
    "python": "3.12.14",
    "torch": "2.12.1",
    "warp": "1.17.0"
  }
}
```

## Results

```json
{
  "batch": 1000000,
  "last_termination": 285,
  "ratio_eager_over_guarded": 1.0177214041156901,
  "ratio_unguarded_over_guarded": 13.760553759039608,
  "steps": 4000,
  "wall_s_per_step": {
    "eager": {
      "iqr": 5.914221983402889e-07,
      "median": 0.00021437406726181507,
      "q1": 0.0002138554737903178,
      "q3": 0.00021444689598865808,
      "samples": [
        0.00021333688031882048,
        0.00021437406726181507,
        0.00021451972471550108
      ]
    },
    "guarded": {
      "iqr": 1.514662289990305e-08,
      "median": 0.00021064120926894247,
      "q1": 0.00021063810586929323,
      "q3": 0.00021065325249219313,
      "samples": [
        0.0002106652957154438,
        0.00021063500246964396,
        0.00021064120926894247
      ]
    },
    "unguarded": {
      "iqr": 4.5168801443649237e-07,
      "median": 0.002898539684014395,
      "q1": 0.0028982803910039363,
      "q3": 0.002898732079018373,
      "samples": [
        0.002898924474022351,
        0.002898539684014395,
        0.002898021097993478
      ]
    }
  }
}
```
