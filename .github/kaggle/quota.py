# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# The remaining weekly Kaggle GPU quota, from `kaggle quota --csv` (kaggle >= 2.x; the 1.x client has no such command):
#   resource,used,remaining,total,refreshAt
#   GPU,12.50h,17.50h,30.00h,2026-11-09T00:00:00+00:00
# Prints nothing about the account or the token. `check(need_hours)` is for drive.py: it returns (enough, message);
# when the client or the answer is unusable it says enough (a missing quota API must not block a run Kaggle would accept).
import csv
import io
import subprocess
import sys


def parse(text):
    """{'remaining': hours, 'total': hours, 'refresh': iso text} of the GPU row, or None when there is none."""
    for row in csv.DictReader(io.StringIO(text.strip())):
        if (row.get("resource") or "").strip().upper() == "GPU":
            try:
                hours = lambda key: float(row[key].strip().rstrip("h"))
                return {"remaining": hours("remaining"), "total": hours("total"), "refresh": (row.get("refreshAt") or "").strip()}
            except (KeyError, ValueError, AttributeError):
                return None
    return None


def check(need_hours, run=subprocess.run):
    r = run([sys.executable, "-m", "kaggle", "quota", "--csv"], capture_output=True, text=True)
    q = parse(r.stdout) if r.returncode == 0 else None
    if q is None:
        return True, "Kaggle quota not readable (old client or no answer): launching without the pre-check"
    if q["remaining"] < need_hours:
        return False, (f"Kaggle GPU quota exhausted: {q['remaining']:.1f} h left of {q['total']:.1f} h, this leg may need up to "
                       f"{need_hours:.1f} h (resets {q['refresh'] or 'weekly'}): GPU evidence missing; re-run after reset or record local GPU evidence")
    return True, f"Kaggle GPU quota: {q['remaining']:.1f} h left of {q['total']:.1f} h"
