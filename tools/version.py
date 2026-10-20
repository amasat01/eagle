#!/usr/bin/env python3
"""Single version source for this repo: `get`, `check`, `bump X.Y.Z [--date D]`, `notes [X.Y.Z]`.

The version strings live in the files listed in tools/version_sites.toml (file + regex with one
capture group around the version); the first entry is the primary. `bump` rewrites all of them,
CITATION.cff and the top `## Unreleased` CHANGELOG heading. Stdlib only; needs Python >= 3.11 (tomllib).
Run from anywhere; paths are relative to the repo root.
"""
import datetime
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CITE_V = r'^version: "([^"]*)"'
CITE_D = r'^date-released: "([^"]*)"'
HEAD = re.compile(r"^## ", re.M)


def sites():
    """(file, regex) per site, then CITATION.cff's version."""
    tab = tomllib.loads((ROOT / "tools/version_sites.toml").read_text())
    return [(s["file"], s["pattern"]) for s in tab["site"]] + [("CITATION.cff", CITE_V)]


def matches(file, pat):
    text = (ROOT / file).read_text()
    found = list(re.finditer(pat, text, re.M))
    if not found:
        sys.exit(f"{file}: pattern {pat!r} matches nothing")
    return text, found


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def get():
    file, pat = sites()[0]
    return matches(file, pat)[1][0].group(1)


def top_heading():
    text = (ROOT / "CHANGELOG.md").read_text()
    m = HEAD.search(text)
    return text, m


def check():
    want, bad = get(), []
    for file, pat in sites():
        text, found = matches(file, pat)
        bad += [(f"{file}:{line_of(text, m.start(1))}", m.group(1)) for m in found if m.group(1) != want]
    text, m = top_heading()
    if m:
        line = text[m.start():text.index("\n", m.start())]
        if line != "## Unreleased" and not re.fullmatch(rf"## {re.escape(want)} \(\d{{4}}-\d{{2}}-\d{{2}}\)", line):
            bad.append((f"CHANGELOG.md:{line_of(text, m.start())}", f"heading {line!r} (want '## Unreleased' or '## {want} (date)')"))
    for where, found in bad:
        print(f"{where}: found {found}, expected {want}")
    return 1 if bad else 0


def bump(new, date):
    for file, pat in sites():
        text, found = matches(file, pat)
        for m in reversed(found):
            text = text[:m.start(1)] + new + text[m.end(1):]
        if file == "CITATION.cff":
            text = re.sub(CITE_D, f'date-released: "{date}"', text, flags=re.M)
        (ROOT / file).write_text(text)
    text, m = top_heading()
    new_head = f"## {new} ({date})"
    if m and text[m.start():].startswith("## Unreleased"):
        text = text[:m.start()] + new_head + text[m.start() + len("## Unreleased"):]
    else:
        print("warning: no '## Unreleased' heading; inserting one for the release", file=sys.stderr)
        at = m.start() if m else len(text)
        text = text[:at] + new_head + "\n\n" + text[at:]
    (ROOT / "CHANGELOG.md").write_text(text)
    return check()


def notes(ver):
    text = (ROOT / "CHANGELOG.md").read_text()
    m = re.search(rf"^## {re.escape(ver)}(?: .*)?$", text, re.M)
    if not m:
        sys.exit(f"CHANGELOG.md: no section for {ver}")
    nxt = HEAD.search(text, m.end())
    print(text[m.end():nxt.start() if nxt else len(text)].strip())
    return 0


def main(argv):
    cmd, args = (argv[0] if argv else ""), argv[1:]
    if cmd == "get" and not args:
        print(get())
        return 0
    if cmd == "check" and not args:
        return check()
    if cmd == "notes" and len(args) <= 1:
        return notes(args[0] if args else get())
    if cmd == "bump" and args and re.fullmatch(r"\d+\.\d+\.\d+", args[0]):
        date = datetime.date.today().isoformat()
        if args[1:2] == ["--date"] and len(args) == 3:
            date = args[2]
        elif len(args) != 1:
            sys.exit("usage: version.py bump X.Y.Z [--date YYYY-MM-DD]")
        return bump(args[0], date)
    sys.exit("usage: version.py get | check | bump X.Y.Z [--date YYYY-MM-DD] | notes [X.Y.Z]")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
