# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The shipped host weights against the committed calibration sample: the
isolated groups (class slopes, launch and state traffic, split overhead and
class sort) must be predicted within a relative RMS of 0.20, the bar under
which ``benchmarks/reproduce/fit_profile.py`` accepts a fit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = Path(__file__).parent / "data" / "calibration_cpu_sample.csv"


@pytest.fixture(scope="module")
def fit_profile():
    path = ROOT / "benchmarks" / "reproduce" / "fit_profile.py"
    if not path.exists():
        pytest.skip("the benchmark kit is not part of this checkout")
    spec = importlib.util.spec_from_file_location("fit_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_host_weights_predict_the_calibration_within_twenty_percent(fit_profile):
    header, cases = fit_profile.read(SAMPLE)
    props = fit_profile.props_of(header)
    rms = fit_profile.rel_rmse(cases, "host", props, None, "ABC")
    assert rms < 0.20, rms


def test_a_refit_is_accepted_on_the_sample(fit_profile):
    header, cases = fit_profile.read(SAMPLE)
    weights, rms = fit_profile.fit(cases, header, groups="ABC", cache_bytes=12.7e6)
    assert rms < 0.20 and weights["lane_rate"] > 0


def test_a_kernel_whose_fused_host_loop_runs_scalar_follows_the_mix_weighted_sum(fit_profile):
    """Group E (paired fused/split, heavy branch at 1-50%): the fused loop of
    the case kernel costs what its own branches cost, not the sum of both. With
    the scalar form the shipped weights predict those rows within 0.20; the
    vectorised form (hawk reports that fact for this kernel) overpredicts them
    by orders of magnitude."""
    header, cases = fit_profile.read(SAMPLE)
    props = fit_profile.props_of(header)
    for row, _ in cases.values():
        row["fused_vectorised"] = "0"
    assert fit_profile.rel_rmse(cases, "host", props, None, "E") < 0.20
