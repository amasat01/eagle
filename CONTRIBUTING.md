# Contributing

Read [How we build](https://amasat01.github.io/raptor/content/principles.html) first: the twelve rules every pull request in the
family is read against, plus the style, release and license terms.

**This repo:** the execution layer. Kernel launch, graph capture, the plugin registry and the batch runner, with a header-only C++
core and a Python package. The same code runs on the host and the GPU.

## Build and run the gate

C++ (CPU build shown, as in `.github/workflows/ci.yml`; drop `-DEAGLE_CPP_MODE=ON` for CUDA). aether is a sibling checkout:

```bash
cmake -DAETHER_CPP_MODE=ON -DCMAKE_INSTALL_PREFIX=$PWD/__install -B ../aether/build -S ../aether
cmake --install ../aether/build
cmake -B build -DCMAKE_PREFIX_PATH=$PWD/__install -DEAGLE_BUILD_TESTS=ON -DEAGLE_DEBUG_MODE=ON -DEAGLE_WARN_AS_ERR=ON -DEAGLE_CPP_MODE=ON .
cmake --build build -j"$(nproc)"
tests/check_gate.sh cpp build/tests/eagle_tests     # or: cuda
```

Never run the test binary or `ctest` bare. eagle has no `ctest` wiring, and an empty run exits 0. `tests/check_gate.sh` checks the run
and the listing against `tests/expected_tests_<mode>.txt`; commit the manifest diff with your tests.

Python: `pip install -e ./python`, then

```bash
pytest python/tests -q -m "not gpu and not interop_matrix"     # drop the -m filter on a GPU machine
pytest python/tests -m ft -W error::RuntimeWarning              # free-threading rows, on a free-threaded Python
```

Docs: `cd docs && make strict` (zero warnings), then `make nbexec` (needs a GPU) and `make nbcheck` for notebooks, `make linkcheck`
for links. `make nbcheck-hosted` runs the notebooks the way a Colab user does; the comments above the target in `docs/Makefile`
say what it needs. Toolchain and docstring style: [Documentation guide](https://amasat01.github.io/eagle/content/devguide/docs_guide.html).

## What a PR here must show

- **C++ first, then bind.** The behaviour exists and is tested in C++; the Python binding only exposes it.
- **The seam stays additive.** `python/src/seam/eagle_backend.h` is versioned: a new backend symbol bumps
  `EAGLE_BACKEND_VERSION_MINOR`, is bound by symbol, and gets a line in `CHANGELOG.md`. Nothing is removed or reordered within major 1.
- **Host and device agree.** Guard CUDA-only code with `#ifndef EAGLE_CPU_ONLY` and give it an OpenMP counterpart; update
  [Host/plugin parity](https://amasat01.github.io/eagle/content/devguide/host-plugin-parity.html).
- **Free-threading.** A new Python feature has an `ft` test.
- **Limits.** A limit a page states is an entry in `docs/content/_shared/limits.json`; `python/tests/test_doc_limits.py` checks it.
- The family checklist from [How we build](https://amasat01.github.io/raptor/content/principles.html), nothing more.

## Style specifics

- Classes `PascalCase`, functions `camelCase`, private data `member_`, template parameters `DataT`, `KernelFunc`.
- Run the repo's `.clang-format`: `find eagle -name '*.h' -print0 | xargs -0 clang-format -i`.
- `DEVICEHOST()` marks host-and-device functions, `KERNEL()` marks `__global__` kernels. Internal helpers go in a `detail`
  namespace wrapped in `/// @cond INTERNAL` ... `/// @endcond`.
- A new error code: add an enumerator (a power of two) to `eagle::err::EagleErrorTypes` and its message to
  `eagle::err::errorMessages` in `eagle/util/DeviceError.h`; raise it with `EAGLE_GPU_THROW` or `EAGLE_GPU_ASSERT`.

## Release

Bump `version` in `python/pyproject.toml` and update `CHANGELOG.md` in a pull request. Its green run builds and tests the
wheels and the sdist; merge it up to date, then tag `vX.Y.Z` on that commit. `.github/workflows/publish.yml` uploads the files
the pull request built (`workflow_dispatch` dry-runs it). `raptor-core` publishes first (see the family guide).
