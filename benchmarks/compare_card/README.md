# Large-N compare evidence

What this is: the walls of eagle's arms and NVIDIA Warp at N = 1e7 (and 1e8 on the H100), one session per device,
kept so that prose can quote the gap at load as generated numbers.

The producer is the `profile-kit` branch's `gpu_profile.py` at the commit recorded in each sidecar (`kit_commit`, with its md5).

Command: `python benchmarks/reproduce/gpu_profile.py --stage compare` on the device. Each `<slug>.csv` is the
stage's `compare.csv`, copied byte for byte. `<slug>` is the device slug of the perf cards (`card_<slug>.json`).

Columns: `kernel`, `n`, `steps`, `dist`, `arm`, `status`, `median_ms`, `best_ms`, the launch mode and fields, the
bitwise comparison against `auto`, `sm_mhz` and `clock_normalised_ms`. `clock_normalised_ms` is the wall rescaled
to the device's reference SM clock; the generated numbers read it when a row has it and `median_ms` otherwise. Only
rows with `status` `ok` can be quoted; a skipped arm is refused by name.

`<slug>.json` records the device, the session, the eagle and hawk commits (from the session log, else the card the
session produced), the md5 of the CSV, the profile-kit commit and the md5 of `gpu_profile.py` at it and the time of the run.

This is not a card: each wall is a single session with no repetitions rule, so read it as a measurement at load, not
as a certified figure. The perf cards stay the reference for N up to 1e6.

Quoting it: `<!--c cmp dev=h100 n=1e7 dist=spread steps=1000 arm=warp den_arm=auto-->3.2×<!--/c-->` is a ratio of
two arms' walls (without `den_arm`, one arm's wall); `kernel=` defaults to `osc`. `tools/sync_readme_cards.py`
regenerates and checks it like every other claim span.

Adding a device: `python tools/ingest_compare.py <session-dir> --slug <slug> --kit-commit <sha>` (dry run), then with `--apply`. It
copies the CSV, writes the sidecar and refuses a CSV with another header or a session without commits. No code
change.
