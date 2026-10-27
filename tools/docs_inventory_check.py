# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Fail when a public name has no page in the built documentation.

    python tools/docs_inventory_check.py <objects.inv> <package> [<package> ...]

For every package given, each name in its `__all__` (and each public submodule's, when the package lists its
submodules with `--submodules`) must appear in the Python domain of the inventory: either as `package.name`
or under a submodule of the package. A glossary term of the same name does not count. The same file is
shared by hawk, eagle, raptor and aether; `--allow name,name` lists names that are documented elsewhere on
purpose.
"""
import argparse
import importlib
import sys
from pathlib import Path


def inventory_names(path):
    from sphinx.util.inventory import InventoryFile
    data = InventoryFile.loads(Path(path).read_bytes(), uri="")
    data = getattr(data, "data", data)
    return {name for domain, entries in data.items() if domain.startswith("py:") for name in entries}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inventory")
    ap.add_argument("packages", nargs="+")
    ap.add_argument("--allow", default="", help="comma separated names to skip")
    args = ap.parse_args(argv)
    skip = {n for n in args.allow.split(",") if n}
    names = inventory_names(args.inventory)
    missing = []
    for pkg in args.packages:
        mod = importlib.import_module(pkg)
        public = list(getattr(mod, "__all__", ()))
        if not public:
            print(f"::error::{pkg} has no __all__")
            return 2
        for name in public:
            if name in skip or name.startswith("_"):
                continue
            if f"{pkg}.{name}" in names or any(n.startswith(pkg + ".") and n.rsplit(".", 1)[-1] == name for n in names):
                continue
            missing.append(f"{pkg}.{name}")
    if missing:
        print(f"{len(missing)} public name(s) have no page in {args.inventory}:")
        for n in missing:
            print("  ", n)
        return 1
    print(f"every public name of {', '.join(args.packages)} is in {args.inventory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
