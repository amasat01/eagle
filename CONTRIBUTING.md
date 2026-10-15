# Contributing

Contributions — issues and pull requests — are welcome on GitHub, under the
terms below.

## Terms

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (`git commit -s`); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

## Using the code

The code is available under the Apache License 2.0 (see `LICENSE`).
Contributions are accepted under the same terms (inbound = outbound).

## Checking the notebooks as a hosted user runs them

`make nbcheck` runs the docs notebooks in your own environment against source checkouts. `make nbcheck-hosted` (in `docs/`) runs
them the way a Colab user does: it builds CPU-only wheels of `aether-dsc`, `raptor-core`, `raptor-eagle` and `raptor-hawk` from the
four checkouts (side by side, or given as `AETHER_ROOT`, `EAGLE_ROOT`, `HAWK_ROOT`, `RAPTOR_ROOT`), installs them into a fresh venv
pinned to Colab's CPU runtime (`docs/_tools/hosted_env/colab-cpu.txt`, generated from Google's published package list by
`refresh_colab_lock.py`; refresh it each release), and executes every notebook of this repository from a copy in an empty directory
with a stand-in `google.colab` module, so the generated setup cell takes its hosted branch (pip from the local wheels, helper files
from a local mirror). `make nbcheck-floor` does the same with the oldest numpy and torch that have Python 3.13 wheels
(`floor-cpu.txt`). Needs `uv`, a Python 3.13 (`NBH_PYTHON=...`), a C++23 compiler (`CXX=...`) and network access to the package indexes; one
PASS, FAIL or SKIP line per notebook, exit 1 on a FAIL. `NBH_ARGS="--work DIR --keep"` keeps the wheelhouse between runs.
`make nbcheck-hosted-gpu` runs the same on a GPU: Colab's GPU-runtime lock with CUDA 12 builds (`colab-gpu.txt`), eagle's CUDA plugin built for
`NBH_GPU_ARCH`, and the GPU-only notebooks run too, on GPU `NBH_GPU_INDEX` (under the `flock` file `NBH_GPU_LOCK`, if set). It needs about 5 GB of disk.

## Releasing

`raptor-eagle` (the Python package; the C++ library itself is not
published to PyPI) publishes through `.github/workflows/publish.yml`,
triggered by pushing a tag `v<version>` that matches
`python/pyproject.toml`'s `project.version`. `workflow_dispatch` runs the
same build, check and test-wheel steps without publishing.

1. Bump `version` in `python/pyproject.toml`, update `CHANGELOG.md`.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The workflow builds manylinux wheels for CPython 3.9-3.14 (including free-threaded 3.13t and 3.14t) with
   `cibuildwheel`: `eagle._core` is plain C++ and needs no CUDA toolkit,
   but the `libeagle_cuda.so` plugin it dlopen's is compiled with `nvcc`
   (headers + compiler only — no driver, nothing executed) inside the
   manylinux container. The published wheel needs only the NVIDIA driver
   at run time (`libcuda.so.1`, resolved by `dlopen`), never a CUDA
   runtime — `auditwheel repair` additionally excludes `libcuda.so.1` as a
   defensive guard. The workflow runs `twine check --strict`, tests each
   wheel (together with a freshly built `raptor-core` sibling) on a
   driver-less runner on every supported CPython — importing, running a
   host kernel, and getting the typed `eagle.BackendUnavailable` refusal
   for device work — then publishes via a PyPI Trusted Publisher
   (environment `pypi`; no token in this repository).

**Publish order across the family:** `raptor-eagle` depends on
`raptor-core` (in the `raptor` repo), which has no family dependencies and
publishes first. Release `raptor-core` before `raptor-eagle`.

## How work flows

- Changes go to the long-lived `dev` branch, either directly or through a
  short-lived feature branch that is merged into `dev`.
- Work that spans repositories uses the SAME branch name in each repository.
  CI checks out the companion repositories at that branch if it exists there,
  and at `main` otherwise; the log line "companion <repo> @ <branch>" shows
  which one was used.
- CI runs on every push, so a red result is seen before anything reaches `main`.
- `main` is updated only by merging `dev` at a clean state (a release or
  milestone point) with green CI. Nothing is pushed to `main` directly.
