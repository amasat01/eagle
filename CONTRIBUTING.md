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
