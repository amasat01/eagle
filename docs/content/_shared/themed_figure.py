# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A notebook chart in a light and a dark version, shown as ONE output.

``show()`` renders the current matplotlib figure twice, recolouring its neutral parts (text, axes, ticks, grid, neutral lines)
with the light palette and then the dark one, and displays both as a single HTML output: the light image (class ``only-light``)
and the dark image (class ``only-dark``, ``display:none`` inline). The docs theme CSS (``raptor-theme.css``) shows the one that
matches the page theme; a plain Jupyter or Colab page shows the light one. Data colours (bars, markers, fills) are left alone: they
are mid-tones chosen for both backgrounds. The figure background is transparent.

This file is stdlib plus matplotlib and IPython only, because it is downloaded next to a notebook on Colab. The one source is
``raptor_docs/themed_figure.py``; a repo's copy in its helper directory is written and checked by

    python -m raptor_docs.themed_figure --sync PATH [--check]
"""
import base64
import html
import io
import pathlib
import sys

#: Semantic colours of a figure, one dict per mode (``raptor_docs.themed_svg`` draws its SVGs from the same dicts). Text colours
#: reach 4.5:1 on ``bg``; ``rule`` is the grid line colour.
LIGHT = {"bg": "#fcfcfb", "edge": "#e1e0d9", "ink": "#0b0b0b", "sub": "#52514e", "bar": "#9a9992", "us": "#7d53dc",
         "on": "#6d4aff", "off": "#7b869c", "grid": "#e9edf5", "rule": "#d4d3cb"}
DARK = {"bg": "#1a1a19", "edge": "#2c2c2a", "ink": "#f2f2ee", "sub": "#c3c2b7", "bar": "#7f7e78", "us": "#9c7df0",
        "on": "#a99bff", "off": "#8793ab", "grid": "#2a3347", "rule": "#454541"}
PALETTES = {"light": LIGHT, "dark": DARK}

DPI = 200

#: A figure at least this tall for its width (height / width) is shown at most ``NARROW_CAP`` wide, so a square chart does not
#: fill the column.
NARROW_ASPECT = 0.8
NARROW_CAP = "60%"


def _rgb(color):
    from matplotlib.colors import to_rgb
    return to_rgb(color)


def _neutral_ink(color) -> bool:
    """A grey of dark or middle lightness (black, ``gray``, ``#898781``, ``#555``): ink that has to follow the page."""
    try:
        r, g, b = _rgb(color)
    except (ValueError, TypeError):
        return False
    return max(r, g, b) - min(r, g, b) <= 0.06 and 0.2126 * r + 0.7152 * g + 0.0722 * b <= 0.62


def _paper(color) -> bool:
    """Near white: the separating edge of a cell or bar, which should be the page colour."""
    try:
        r, g, b = _rgb(color)
    except (ValueError, TypeError):
        return False
    return min(r, g, b) >= 0.93


def _recolour(fig, pal) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    for ax in fig.axes:
        ax.tick_params(colors=pal["sub"], labelcolor=pal["ink"])
        for spine in ax.spines.values():
            spine.set_edgecolor(pal["sub"])
        for axis in (ax.xaxis, ax.yaxis):
            for line in axis.get_gridlines():
                line.set_color(pal["rule"])
        leg = ax.get_legend()
        if leg is not None and leg.get_frame_on():
            leg.get_frame().set_facecolor(pal["bg"])
            leg.get_frame().set_edgecolor(pal["rule"])
    grid = {id(line) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for line in axis.get_gridlines()}
    for art in fig.findobj():
        if isinstance(art, Text):
            if _neutral_ink(art.get_color()):
                art.set_color(pal["ink"])
            box = art.get_bbox_patch()
            if box is not None and _neutral_ink(box.get_edgecolor()):
                box.set_edgecolor(pal["sub"])
        elif isinstance(art, Line2D) and id(art) not in grid:
            if _neutral_ink(art.get_color()):
                art.set_color(pal["sub"])
        elif isinstance(art, Patch) and art.get_edgecolor() is not None:
            edge = art.get_edgecolor()
            if len(edge) == 4 and edge[3] > 0:
                if _paper(edge):
                    art.set_edgecolor(pal["bg"])
                elif _neutral_ink(edge):
                    art.set_edgecolor(pal["sub"])


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", transparent=True)
    return buf.getvalue()


def _alt(fig) -> str:
    texts = [fig._suptitle.get_text()] if getattr(fig, "_suptitle", None) else []
    texts += [ax.get_title() for ax in fig.axes if ax.get_title()]
    return "Chart: " + "; ".join(texts) if texts else "Chart"


def pair_html(fig, alt=None, narrow=None) -> str:
    """The HTML of the two-image output for ``fig`` (recolours ``fig`` in place)."""
    alt = html.escape(alt if alt is not None else _alt(fig), quote=True)
    imgs = []
    for mode in ("light", "dark"):
        _recolour(fig, PALETTES[mode])
        data = base64.b64encode(_png(fig)).decode("ascii")
        extra = ' style="display:none"' if mode == "dark" else ""
        imgs.append(f'<img class="only-{mode}" alt="{alt if mode == "light" else ""}"{extra} src="data:image/png;base64,{data}">')
    if narrow is None:
        w, h = fig.get_size_inches()
        narrow = h / w >= NARROW_ASPECT
    style = f' style="max-width:{NARROW_CAP};margin:0 auto"' if narrow else ""
    return f'<div class="themed-figure"{style}>' + "".join(imgs) + "</div>"


def show(fig=None, alt=None) -> None:
    """Display ``fig`` (default: the current figure) as a light/dark pair and close it. Use in place of ``plt.show()``."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    fig = fig if fig is not None else plt.gcf()
    display(HTML(pair_html(fig, alt)))
    plt.close(fig)


def sync(path, check=False) -> int:
    """Write (or check) ``path`` as a copy of this file. 0 fresh, 1 stale."""
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    target = pathlib.Path(path)
    if check:
        if not target.is_file() or target.read_text(encoding="utf-8") != src:
            print(f"{path}: not a copy of raptor_docs/themed_figure.py: run python -m raptor_docs.themed_figure --sync {path}",
                  file=sys.stderr)
            return 1
        return 0
    target.write_text(src, encoding="utf-8")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) in (2, 3) and args[0] == "--sync" and args[2:] in ([], ["--check"]):
        sys.exit(sync(args[1], check=args[2:] == ["--check"]))
    sys.exit(__doc__)
