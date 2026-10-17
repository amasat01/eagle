# zero_copy_interop card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-17T07:50:00Z

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
  "example_script": "examples/python/zero_copy_interop.py",
  "example_script_md5": "9138329a9a098d5c167d6f1a9cc1dc73",
  "generated_utc": "2026-10-17T07:50:00Z",
  "peak": {
    "datasheet_cross_check": "Quadro P2000 data sheet: 3.0 TFLOPS peak FP32 (FP64 is 1/32 of FP32 on compute capability 6.1), 140 GB/s memory bandwidth",
    "dram_bytes_per_s": 140160000000.0,
    "dram_source": "2 x memory clock (cudaDeviceProp::memoryClockRate) x bus width (cudaDeviceProp::memoryBusWidth) / 8",
    "fp64_flops": 94752000000.0,
    "fp64_source": "SM count x FP64 FMA per clock per SM (CUDA C++ Programming Guide, Arithmetic Instructions throughput table, compute capability 6.1: 4) x 2 FLOP per FMA x the device's reported maximum clock (cudaDeviceProp::clockRate)"
  },
  "script_md5": "46b87c934d14527449704b9d35c4971e",
  "software": {
    "cpu": "Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz",
    "cupy": "14.2.0",
    "eagle_commit": "3f9a26785f55bcd5caac37c950d9263451469a27-dirty",
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
  "batch": 22369621,
  "pcie_bytes_naive": 1073741808,
  "ratio": 172.75593165206732,
  "size_mb": 256.0,
  "wall_s": {
    "aliased": {
      "iqr": 4.7759152948856354e-05,
      "median": 0.004767125938087702,
      "q1": 0.004754059948027134,
      "q3": 0.00480181910097599,
      "samples": [
        0.004767125938087702,
        0.004754059948027134,
        0.00474407896399498,
        0.004871696699410677,
        0.00480181910097599
      ]
    },
    "naive": {
      "iqr": 0.05543810501694679,
      "median": 0.8235492827370763,
      "q1": 0.782105172984302,
      "q3": 0.8375432780012488,
      "samples": [
        0.782105172984302,
        0.8375432780012488,
        0.850585927721113,
        0.749931228812784,
        0.8235492827370763
      ]
    }
  }
}
```
