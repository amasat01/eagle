"""Release wheel jobs build for the portable ISA (a source build is -march=native)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest


def test_release_wheel_jobs_build_for_the_portable_isa():
    """A source build targets this machine (-march=native); every wheel job must say x86-64-v3 instead."""
    release = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "release.yml"
    if not release.is_file():
        pytest.skip("release.yml is not next to the tests (installed-wheel run)")
    jobs = re.split(r"(?m)^  (?=[A-Za-z0-9_-]+:\s*$)", release.read_text())
    wheel_jobs = [j for j in jobs if "cibuildwheel" in j and "runs-on" in j]
    assert wheel_jobs, "no cibuildwheel job found: the pattern needs updating"
    for job in wheel_jobs:
        envs = len(re.findall(r"CIBW_ENVIRONMENT(?:_LINUX|_MACOS)?:", job))
        assert envs and job.count("AETHER_MARCH=x86-64-v3") >= envs, f"wheel job {job.split(':', 1)[0]}: every CIBW_ENVIRONMENT needs AETHER_MARCH=x86-64-v3"
