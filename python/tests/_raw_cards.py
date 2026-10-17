# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The full performance card behind a committed one, for the tests that read raw rows.

Git holds the slim cards (see ``benchmarks/slim_card.py``); the full ones are release
assets. Set ``EAGLE_CARD_RAW_DIR`` to a directory laid out like ``benchmarks/`` (e.g.
``<dir>/perf_card/card_<slug>.json``). Without it the calling test is skipped; with it, a
missing or altered file fails the test (CI downloads the assets and sets it).
"""

import importlib.util
import pathlib

import pytest

_SLIM = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "slim_card.py"


def raw_card(card_path):
    spec = importlib.util.spec_from_file_location("eagle_slim_card", _SLIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    card = module.raw_card(pathlib.Path(card_path))
    if card is None:
        pytest.skip(f"needs the full card {pathlib.Path(card_path).name}: set EAGLE_CARD_RAW_DIR "
                    "to the card release assets (benchmarks/slim_card.py)")
    return card
