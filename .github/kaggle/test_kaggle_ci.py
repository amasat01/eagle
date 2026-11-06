# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# Offline checks of summary.py and quota.py (the quota answer is the documented `kaggle quota --csv` shape, not a recorded call).
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import quota  # noqa: E402
import summary  # noqa: E402

PASS = dict.fromkeys(summary.LEGS, "pass")
ONE_FAIL = {**PASS, "cuda13-tests": "fail"}
QUOTA_OUT = {**PASS, "cuda12-notebooks": "unrun", "cuda13-notebooks": "unrun"}


@pytest.mark.parametrize("states,release,code,label", [
    (PASS, True, 0, "remove"), (PASS, False, 0, "remove"),
    (ONE_FAIL, True, 1, "remove"), (ONE_FAIL, False, 1, "remove"),
    (QUOTA_OUT, True, 1, "add"), (QUOTA_OUT, False, 0, "add"),
    ({**ONE_FAIL, "cuda12-tests": "unrun"}, False, 1, "add"),
])
def test_verdict(states, release, code, label):
    c, out, lab = summary.verdict(states, release)
    assert (c, lab) == (code, label)
    if QUOTA_OUT == states:
        assert any("quota exhausted" in line for line in out)
        assert any(line.startswith("::error::") for line in out) == release
        assert any(line.startswith("::warning::") for line in out) != release


def test_missing_state_file_is_unrun(tmp_path):
    (tmp_path / "cuda12-tests").write_text("pass\n")
    assert summary.main(["s", str(tmp_path), "1", "cuda12-tests", "cuda13-tests"]) == 1
    assert (tmp_path / ".label").read_text() == "add"
    assert summary.main(["s", str(tmp_path), "0", "cuda12-tests", "cuda13-tests"]) == 0


CSV = "resource,used,remaining,total,refreshAt\nGPU,28.50h,1.50h,30.00h,2026-11-09T00:00:00+00:00\nTPU,0.00h,20.00h,20.00h,2026-11-09T00:00:00+00:00\n"


def fake(out, rc=0):
    return lambda *a, **k: subprocess.CompletedProcess(a, rc, out, "")


def test_parse():
    assert quota.parse(CSV) == {"remaining": 1.5, "total": 30.0, "refresh": "2026-11-09T00:00:00+00:00"}
    assert quota.parse("No quota information available") is None
    assert quota.parse("resource,used\nGPU,x\n") is None


def test_check():
    ok, msg = quota.check(3.0, fake(CSV))
    assert not ok and "quota exhausted" in msg and "1.5 h left" in msg
    assert quota.check(1.0, fake(CSV))[0]
    assert quota.check(3.0, fake("", 2))[0]  # old client: no pre-check, launch
    assert quota.check(3.0, fake("garbage"))[0]
