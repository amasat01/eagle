# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``plane_meta``: opaque notes on planes, read by name and validated, never interpreted."""

import pytest

from eagle.sidecar import read_plane_meta, validate_sidecar

SPEC = [["per_sample", "x"], ["mutable", "y"]]


def test_notes_are_read_back_by_plane_name():
    meta = {"arg_spec": SPEC, "plane_meta": {"x": {"unit": "km", "n": [1, 2]}}}
    assert read_plane_meta(meta, name="k") == {"x": {"unit": "km", "n": [1, 2]}}
    assert read_plane_meta({"arg_spec": SPEC}, name="k") == {}


@pytest.mark.parametrize("bad", [{"z": {"a": 1}}, {"x": "km"}, ["x"]], ids=["unknown plane", "not a dict", "not a map"])
def test_a_malformed_block_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="k: sidecar plane_meta must map each plane of the arg_spec"):
        read_plane_meta({"arg_spec": SPEC, "plane_meta": bad}, name="k")


def test_the_shared_validator_checks_it():
    with pytest.raises(ValueError, match="plane_meta"):
        validate_sidecar({"arg_spec": SPEC, "plane_meta": {"nope": {}}}, name="k")
