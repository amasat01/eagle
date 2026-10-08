#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""run_sanitize.py — drive valgrind / NVIDIA compute-sanitizer over a gtest binary.

Parse tool output and emit a human-readable report. The same script is reused
across the sibling header-only libraries; the only per-project knob is the
--project-name flag, which selects the env var consumed by the C++-side
`Test::isMinimalMode()` helper.

Exit codes:
  0  clean (no errors, no definite/indirect/possible leaks)
  1  real errors or leaks detected (still-reachable is a warning by default)
  2  driver infrastructure error (missing tool, bad config, ...)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class C:
    _on = _use_color()
    RESET = "\033[0m"  if _on else ""
    BOLD  = "\033[1m"  if _on else ""
    RED   = "\033[31m" if _on else ""
    GRN   = "\033[32m" if _on else ""
    YLW   = "\033[33m" if _on else ""
    CYA   = "\033[36m" if _on else ""
    GRY   = "\033[90m" if _on else ""


LEAK_KINDS = (
    "Leak_DefinitelyLost",
    "Leak_IndirectlyLost",
    "Leak_PossiblyLost",
    "Leak_StillReachable",
)


@dataclass
class ValgrindLeak:
    kind: str
    bytes: int
    blocks: int
    top_frame: str


@dataclass
class ValgrindError:
    kind: str
    what: str
    top_frames: list = field(default_factory=list)


@dataclass
class ValgrindReport:
    errors: list = field(default_factory=list)
    leaks: list = field(default_factory=list)
    wall: float = 0.0
    suppressed: int = 0
    xml_missing: bool = False


@dataclass
class CsanResult:
    tool: str
    errors: int = 0
    wall: float = 0.0
    sample: str = ""
    log_path: str = ""


CSAN_TOOLS = ("memcheck", "initcheck", "racecheck", "synccheck")


def _parse_frame(frame: ET.Element) -> str:
    fn = (frame.findtext("fn") or "??").strip()
    f  = (frame.findtext("file") or "").strip()
    ln = (frame.findtext("line") or "").strip()
    loc = f"{f}:{ln}" if f and ln else (frame.findtext("obj") or "").strip()
    return f"{fn}  ({loc})" if loc else fn


def parse_valgrind_xml(path: Path) -> ValgrindReport:
    rep = ValgrindReport()
    if not path.exists():
        rep.xml_missing = True
        return rep
    try:
        tree = ET.parse(str(path))
    except ET.ParseError as e:
        rep.errors.append(ValgrindError(kind="XmlParseError",
                                        what=f"{e}"))
        return rep

    root = tree.getroot()
    for err in root.findall("error"):
        kind = (err.findtext("kind") or "Unknown").strip()
        what = (err.findtext("xwhat/text")
                or err.findtext("what") or "").strip()
        frames = [_parse_frame(f) for f in err.findall(".//stack/frame")][:4]
        if kind in LEAK_KINDS:
            leaked_bytes  = int(err.findtext("xwhat/leakedbytes") or "0")
            leaked_blocks = int(err.findtext("xwhat/leakedblocks") or "0")
            rep.leaks.append(ValgrindLeak(
                kind=kind, bytes=leaked_bytes, blocks=leaked_blocks,
                top_frame=frames[0] if frames else ""))
        else:
            rep.errors.append(ValgrindError(
                kind=kind, what=what, top_frames=frames))

    for pair in root.findall("suppcounts/pair"):
        rep.suppressed += int(pair.findtext("count") or "0")
    return rep


def run_valgrind(args, log_dir: Path) -> "ValgrindReport | None":
    if shutil.which("valgrind") is None:
        print(f"{C.YLW}valgrind not found on PATH — skipping{C.RESET}")
        return None

    xml_path = log_dir / "valgrind.xml"
    supp_flags = []
    if args.suppressions:
        supp_dir = Path(args.suppressions)
        if supp_dir.is_dir():
            for supp in sorted(supp_dir.glob("*.supp")):
                supp_flags.append(f"--suppressions={supp}")

    cmd = [
        "valgrind",
        "--tool=memcheck",
        "--leak-check=full",
        "--show-leak-kinds=all",
        "--errors-for-leak-kinds=definite,indirect,possible",
        "--error-exitcode=77",
        "--track-origins=yes",
        "--num-callers=25",
        "--child-silent-after-fork=yes",
        "--xml=yes",
        f"--xml-file={xml_path}",
        *supp_flags,
        args.binary,
    ]
    if args.gtest_filter:
        cmd.append(f"--gtest_filter={args.gtest_filter}")
    if args.gtest_repeat and args.gtest_repeat > 1:
        cmd.append(f"--gtest_repeat={args.gtest_repeat}")

    env = os.environ.copy()
    if args.minimal:
        env[f"{args.project_name.upper()}_TEST_MINIMAL"] = "1"

    print(f"{C.CYA}> valgrind memcheck{C.RESET}  {args.binary}")
    start = time.monotonic()
    with open(log_dir / "valgrind.stdout.log", "wb") as out, \
         open(log_dir / "valgrind.stderr.log", "wb") as err:
        subprocess.run(cmd, env=env, stdout=out, stderr=err, check=False)
    wall = time.monotonic() - start

    rep = parse_valgrind_xml(xml_path)
    rep.wall = wall
    return rep


def run_compute_sanitizer(args, log_dir: Path) -> list:
    if shutil.which("compute-sanitizer") is None:
        print(f"{C.GRY}compute-sanitizer not on PATH — skipping CUDA checks{C.RESET}")
        return []

    results = []
    env = os.environ.copy()
    if args.minimal:
        env[f"{args.project_name.upper()}_TEST_MINIMAL"] = "1"

    for tool in CSAN_TOOLS:
        log_path = log_dir / f"computesan.{tool}.log"
        cmd = [
            "compute-sanitizer",
            f"--tool={tool}",
            "--leak-check=full",
            "--padding=256",
            "--launch-timeout=60",
            "--error-exitcode=1",
            f"--log-file={log_path}",
            args.binary,
        ]
        if args.gtest_filter:
            cmd.append(f"--gtest_filter={args.gtest_filter}")

        print(f"{C.CYA}> compute-sanitizer {tool}{C.RESET}  {args.binary}")
        start = time.monotonic()
        subprocess.run(cmd, env=env, check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        wall = time.monotonic() - start

        errors = 0
        sample = ""
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            m = re.search(r"ERROR SUMMARY:\s*(\d+)\s*error", text)
            if m:
                errors = int(m.group(1))
            # grab the first ~12 lines of the first error block, if any
            first = re.search(r"=========\s+(?:Program|Invalid|Uninitialized|Race|Memory|Host|Barrier)[^\n]*\n(?:=========[^\n]*\n){1,15}",
                              text)
            if first:
                sample = first.group(0)
        results.append(CsanResult(tool=tool, errors=errors, wall=wall,
                                  sample=sample, log_path=str(log_path)))
    return results


def render_valgrind(rep: ValgrindReport, fail_on_reachable: bool):
    lines = []
    n_err = len(rep.errors)
    totals = {k: [0, 0] for k in LEAK_KINDS}
    for lk in rep.leaks:
        totals.setdefault(lk.kind, [0, 0])
        totals[lk.kind][0] += lk.bytes
        totals[lk.kind][1] += lk.blocks

    def_b = totals["Leak_DefinitelyLost"][0]
    ind_b = totals["Leak_IndirectlyLost"][0]
    pos_b = totals["Leak_PossiblyLost"][0]
    rch_b = totals["Leak_StillReachable"][0]

    has_hard = (n_err > 0) or (def_b + ind_b + pos_b > 0)
    has_rch  = rch_b > 0
    real_fail = has_hard or (fail_on_reachable and has_rch)

    if real_fail:
        head_color, mark = C.RED, "X"
    elif has_rch:
        head_color, mark = C.YLW, "!"
    else:
        head_color, mark = C.GRN, "v"

    lines.append(f"{head_color}{mark}  valgrind memcheck{C.RESET}  "
                 f"({rep.wall:.1f}s, {rep.suppressed} suppressed)")
    if rep.xml_missing:
        lines.append(f"    {C.YLW}(no XML produced — check stdout/stderr logs){C.RESET}")

    def row(label, nbytes, nblocks, color):
        if nbytes > 0:
            lines.append(f"  {label:<22} {color}{nbytes:>10} bytes / "
                         f"{nblocks} blocks{C.RESET}")
        else:
            lines.append(f"  {label:<22} {nbytes:>10} bytes / {nblocks} blocks")

    lines.append(f"  errors:                {C.RED if n_err else ''}{n_err}{C.RESET}")
    row("leak definitely:",     def_b, totals['Leak_DefinitelyLost'][1],    C.RED)
    row("leak indirectly:",     ind_b, totals['Leak_IndirectlyLost'][1],    C.RED)
    row("leak possibly:",       pos_b, totals['Leak_PossiblyLost'][1],      C.RED)
    row("leak still-reachable:",rch_b, totals['Leak_StillReachable'][1],    C.YLW)
    if has_rch:
        lines.append(f"    {C.YLW}(still-reachable is a warning unless --fail-on-reachable){C.RESET}")

    if rep.errors:
        lines.append("")
        lines.append(f"{C.BOLD}  top errors:{C.RESET}")
        for e in rep.errors[:8]:
            lines.append(f"    - {C.RED}{e.kind}{C.RESET}  {e.what}")
            for fr in e.top_frames[:3]:
                lines.append(f"        {fr}")
        if len(rep.errors) > 8:
            lines.append(f"    ... ({len(rep.errors) - 8} more in XML)")

    leak_show = [lk for lk in rep.leaks
                 if lk.kind != "Leak_StillReachable" or fail_on_reachable]
    if leak_show or has_rch:
        lines.append("")
        lines.append(f"{C.BOLD}  top leak sites:{C.RESET}")
        by_site = {}
        for lk in rep.leaks:
            by_site.setdefault((lk.kind, lk.top_frame), [0, 0])
            by_site[(lk.kind, lk.top_frame)][0] += lk.bytes
            by_site[(lk.kind, lk.top_frame)][1] += 1
        sortable = [(agg[0], kind, frame, agg[1])
                    for (kind, frame), agg in by_site.items()]
        sortable.sort(reverse=True)
        for total, kind, frame, count in sortable[:10]:
            if kind == "Leak_StillReachable":
                tag, color = "!", C.YLW
            else:
                tag, color = "X", C.RED
            lines.append(f"    {color}{tag}{C.RESET} {kind:<22} "
                         f"{total:>10} bytes in {count} site(s)")
            lines.append(f"        {frame}")

    return real_fail, "\n".join(lines)


def render_csan(results: list):
    if not results:
        return False, f"{C.GRY}compute-sanitizer: not run{C.RESET}"
    lines = []
    any_fail = False
    for r in results:
        if r.errors > 0:
            any_fail = True
            lines.append(f"{C.RED}X  compute-sanitizer {r.tool}{C.RESET}  "
                         f"({r.wall:.1f}s)  errors: {r.errors}  log: {r.log_path}")
            if r.sample:
                for sl in r.sample.splitlines()[:12]:
                    lines.append(f"    {C.GRY}{sl}{C.RESET}")
        else:
            lines.append(f"{C.GRN}v  compute-sanitizer {r.tool}{C.RESET}  "
                         f"({r.wall:.1f}s)  clean")
    return any_fail, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--tool", choices=("valgrind", "computesan", "both"),
                    default="both")
    ap.add_argument("--suppressions", default="")
    ap.add_argument("--log-dir", default="sanitize_logs")
    ap.add_argument("--gtest-filter", default="")
    ap.add_argument("--gtest-repeat", type=int, default=1)
    ap.add_argument("--minimal", action="store_true")
    ap.add_argument("--fail-on-reachable", action="store_true")
    ap.add_argument("--project-name", default="EAGLE")
    args = ap.parse_args(argv)

    binary_path = Path(args.binary)
    if not binary_path.is_file():
        print(f"{C.RED}binary not found: {args.binary}{C.RESET}", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{C.BOLD}Sanitize sweep — project={args.project_name}  "
          f"tool={args.tool}  minimal={args.minimal}  "
          f"binary={binary_path.name}{C.RESET}")
    print(f"{C.GRY}logs: {log_dir.resolve()}{C.RESET}")
    print()

    vg_rep = None
    csan_results = []
    if args.tool in ("valgrind", "both"):
        vg_rep = run_valgrind(args, log_dir)
    if args.tool in ("computesan", "both"):
        csan_results = run_compute_sanitizer(args, log_dir)

    print()
    print(f"{C.BOLD}-- Sanitize Report -- {binary_path.name} "
          f"{'(minimal)' if args.minimal else ''}{C.RESET}")
    exit_code = 0

    if vg_rep is not None:
        fail, text = render_valgrind(vg_rep, args.fail_on_reachable)
        print(text)
        if fail:
            exit_code = 1
        print()

    if csan_results:
        fail, text = render_csan(csan_results)
        print(text)
        if fail:
            exit_code = 1
        print()

    if exit_code == 0:
        print(f"{C.GRN}{C.BOLD}v sanitize clean{C.RESET}  "
              f"logs at {log_dir}")
    else:
        print(f"{C.RED}{C.BOLD}X sanitize FAILED{C.RESET}  "
              f"see logs at {log_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
