# Contributing

Contributions — issues and pull requests — are welcome on GitHub, under the
terms below. See the top-level `CONTRIBUTING.md` for the full statement.

## Terms

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (`git commit -s`); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

## Tests

C++ (GoogleTest): build with `EAGLE_BUILD_TESTS=ON`, then run the gate, not
`ctest` directly — eagle has no `ctest` wiring, and `ctest --test-dir build`
reports zero tests without failing.

```bash
tests/check_gate.sh cuda build/tests/eagle_tests   # or: cpp, for a CPP_MODE build
```

The gate checks both the run and the test listing against the committed name
manifest `tests/expected_tests_<mode>.txt`, so a filtered or empty run cannot
pass silently.

Python (pytest):

```bash
pytest python/tests -m "not gpu and not interop_matrix"   # drop the -m filter on a GPU machine
```

See the [Developer guide](devguide/dev_guide) for coding conventions, the
contribution workflow, and how to add a new error code.

## Building the docs

```bash
cd docs/
make strict     # 0 warnings required
make nbexec     # execute every notebook in place (needs a GPU)
make nbcheck    # refuse an unfilled notebook
make linkcheck  # crawl every link
```

See the [Documentation guide](devguide/docs_guide) for the full toolchain,
dependencies, and docstring style.
