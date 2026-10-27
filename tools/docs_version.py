# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The version a documentation build shows: the release it is built for, never a development string.

`release_version(root)` is for a project's docs/conf.py. The pull-request build tags its checkout with the
version the PR's release label gives (the same step the wheels use), so setuptools-scm reads that tag. With no
label the checkout is past the last tag, and the newest release tag is shown instead of `X.Y.Z.devN+g...`.
`family_version()` is for the landing site: the newest release of raptor, the family's spine.
`DOCS_VERSION` in the environment overrides both. Stdlib plus setuptools-scm; one file, copied unchanged into
every repository that builds docs.
"""
import os
import re
import subprocess
from pathlib import Path

_TAG = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(rc\d+)?$")


def _key(tag):
    m = _TAG.match(tag)
    return (int(m[1]), int(m[2]), int(m[3]), 1 if not m[4] else 0, m[4] or "")


def _is_release(version):
    return ".dev" not in version and "+" not in version


def release_version(root):
    env = os.environ.get("DOCS_VERSION", "").lstrip("v")
    if env:
        return env
    scm = "0.0.0"
    try:
        import setuptools_scm
        scm = setuptools_scm.get_version(root=str(root), fallback_version="0.0.0")
    except Exception:
        pass
    if _is_release(scm) and scm != "0.0.0":
        return scm
    try:
        tags = subprocess.run(["git", "tag", "--list", "v*", "--merged", "HEAD"], cwd=str(root), capture_output=True,
                              text=True, check=True).stdout.split()
        tags = [t for t in tags if _TAG.match(t)]
        if tags:
            return max(tags, key=_key).lstrip("v")
    except Exception:
        pass
    return scm.split(".dev")[0].split("+")[0]


def family_version(repo="raptor"):
    env = os.environ.get("DOCS_VERSION", "").lstrip("v")
    if env:
        return env
    try:
        out = subprocess.run(["git", "ls-remote", "--tags", "--refs", f"https://github.com/amasat01/{repo}", "v*"],
                             capture_output=True, text=True, check=True, timeout=60).stdout
        tags = [line.rsplit("/", 1)[-1] for line in out.splitlines()]
        tags = [t for t in tags if _TAG.match(t)]
        if tags:
            return max(tags, key=_key).lstrip("v")
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("raptor-core")
    except Exception:
        return "unversioned"


if __name__ == "__main__":
    import sys
    print(family_version() if sys.argv[1:] == ["family"] else release_version(Path(sys.argv[1] if sys.argv[1:] else ".")))


def sitemap_lastmod(app, exception):
    """`build-finished` handler: add <lastmod> (the date of the last commit that touched the page's source) to sitemap.xml.

    Generated pages have no source of their own and take the date of HEAD. Needs git history; without it the date of the build."""
    if exception is not None:
        return
    import datetime
    import xml.etree.ElementTree as ET
    path = Path(app.outdir) / "sitemap.xml"
    base = app.config.html_baseurl
    if not path.exists() or not base:
        return
    cwd = str(Path(app.srcdir))

    def git_date(*args):
        try:
            out = subprocess.run(["git", "log", "-1", "--format=%cs", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
            return out or None
        except Exception:
            return None

    head = git_date() or datetime.date.today().isoformat()
    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", ns)
    tree = ET.parse(path)
    docs = set(app.env.found_docs)
    for url in tree.getroot().findall(f"{{{ns}}}url"):
        loc = url.find(f"{{{ns}}}loc").text or ""
        name = loc[len(base):] if loc.startswith(base) else loc
        name = name[:-5] if name.endswith(".html") else name
        when = None
        if name in docs:
            when = git_date("--", str(app.env.doc2path(name)))
        ET.SubElement(url, f"{{{ns}}}lastmod").text = when or head
    tree.write(path, encoding="utf-8", xml_declaration=True)
