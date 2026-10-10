# eagle sanitize

Wraps `eagle_tests` under valgrind and NVIDIA compute-sanitizer, and gates the suite under
ASan/UBSan. Three entry points, one finding policy.

## Commands

```bash
# 1. Report driver (valgrind + compute-sanitizer, human-readable); build with -DEAGLE_BUILD_SANITIZE=ON
cmake --build build --target sanitize
#    exit 0 clean | 1 findings | 2 RED: no valgrind XML, no compute-sanitizer "ERROR SUMMARY",
#    target exit status not 0/77, or gtest count != tests/expected_tests_<mode>.txt

# 2. Local pre-release gate: CUDA mode, GPU via CUDA_VISIBLE_DEVICES (default 1)
tests/sanitize/sanitize_gate.sh build
#    memcheck / racecheck / synccheck / valgrind each THROUGH tests/check_gate.sh
#    (tests/sanitize/make_wrapper.sh writes the exec wrapper); initcheck is triage-only;
#    then the canaries must go red. Prints its log dir: record it + the suppressed count.

# 3. Host sanitizer build (the CI `sanitize-cpp` job)
cmake -B build -DEAGLE_BUILD_TESTS=ON -DEAGLE_CPP_MODE=ON -DEAGLE_SANITIZERS=address,undefined .
cmake --build build -j
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  LSAN_OPTIONS=suppressions=$PWD/tests/sanitize/lsan.supp \
  tests/check_gate.sh cpp build/tests/eagle_tests
```

## Finding policy

- RED: any valgrind non-leak error with a frame in our code; any definite or indirect leak not
  matched by an owned entry; any compute-sanitizer memcheck / racecheck / synccheck error.
- Triage (class C): initcheck errors; uninitialised values whose origin is a device scratch buffer.
- Noise: still-reachable (warning); possibly-lost whose whole stack is third-party.
- Suppressions (`suppressions/*.supp`): one scenario per entry, header
  `# owner:<repo> · why: · evidence: · added:`, a `fun:`/`obj:` anchor, never a
  `Memcheck:Leak` entry covering definite/indirect. The report prints the suppressed count;
  growth without a new owned entry is a finding.

## Canaries (non-vacuity)

| Binary | Mode | Must produce |
|---|---|---|
| `eagle_canary` (`canary.cpp`, built when `EAGLE_SANITIZERS` is set) | `heap-oob` | ASan heap-buffer-overflow |
| | `leak` | LeakSanitizer report (`detect_leaks=1`) |
| | `uninit` | UBSan invalid `bool` load (ASan fill pattern; true uninit = valgrind) |
| | `shift-ub` | UBSan shift exponent (`halt_on_error=1` for non-zero exit) |
| `eagle_canary_cu` (`canary.cu`, CUDA mode) | `dev-oob` | memcheck >= 1 error (word-widened atomic on a 1-byte allocation) |
| | `dev-race` | racecheck >= 1 hazard (shared write-write, no `__syncthreads`) |
