# Contributing to eagle

eagle is the execution layer: kernel launch, graph capture, the plugin registry and the batch runner, with a header-only C++ core and a Python package. Read [How we build](https://amasat01.github.io/raptor/content/principles.html) first: the rules every pull request in the family is read against. The terms are in the repository's `CONTRIBUTING.md`.

## Build and run the gate

Python, from a clone:

```bash
pip install -e ./python
pytest python/tests -q -m "not gpu and not interop_matrix"     # drop the -m filter on a GPU machine
```

C++, through the gate and never the bare binary (an empty run exits 0). aether is a sibling checkout; the full configure lines are in `CONTRIBUTING.md`:

```bash
tests/check_gate.sh cpp build/tests/eagle_tests     # or: cuda
```

Docs: `cd docs && make strict`, then `make nbcheck` for notebooks and `make linkcheck` for links.

## Where things live

- `eagle/`: the header-only C++ core, one folder per idea.
- `python/eagle/`: the Python package; `python/src/` is the binding.
- `plugin/`: the C++ plugin registry. `tests/` and `python/tests/`: the tests.
- `docs/content/`: the pages; `docs/examples/`: examples the docs gate compiles.

## Next

```{toctree}
:maxdepth: 1

devguide/dev_guide
devguide/graph_capture_replay
devguide/docs_guide
```

Adding a thing: the [Developer guide](devguide/dev_guide). Writing a notebook or a docstring: the [Documentation guide](devguide/docs_guide). A page about how capture and replay work: [Capture and replay](devguide/graph_capture_replay).
