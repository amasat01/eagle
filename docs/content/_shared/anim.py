# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Write a notebook animation to the docs static folder: a GIF always, an MP4 when ``imageio-ffmpeg`` is installed.

Pillow comes with matplotlib. Frames are PIL images; the GIF uses a small palette to stay within the media budget
(1.5 MB per GIF, 3 MB per MP4).

Embed a GIF with ``{image}`` and ``:class: only-light`` / ``only-dark``. If the figure is square or tall (height / width >= 0.8)
add ``narrow`` (``:class: only-light narrow``): the page then shows it at 60% of the column instead of full width.
"""
import os
import pathlib


def static_anim_dir() -> pathlib.Path:
    """``docs/_static/anim``, created. A notebook runs from its own directory, three levels below ``docs``."""
    # a gate run sets DOCS_ANIM_DIR to a folder of its own: running the docs never rewrites the committed animations
    out = pathlib.Path(os.environ.get("DOCS_ANIM_DIR") or pathlib.Path.cwd().parents[2] / "_static" / "anim")
    out.mkdir(parents=True, exist_ok=True)
    return out


def frame_of(fig, facecolor):
    """The matplotlib figure as an RGB PIL image (the figure is closed)."""
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=facecolor, dpi=150)
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def write_anim(frames, stem: str, fps: int = 14, colors: int = 48) -> dict:
    """Write ``frames`` (RGB PIL images) as ``<stem>.gif`` and, if possible, ``<stem>.mp4``; return ``{path: bytes}``."""
    import numpy as np
    from PIL import Image

    out = static_anim_dir()
    gif = out / f"{stem}.gif"
    small = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in frames]
    small[0].save(gif, save_all=True, append_images=small[1:], duration=round(1000 / fps), loop=0, optimize=True)
    sizes = {gif.name: gif.stat().st_size}
    try:
        import imageio_ffmpeg
    except ImportError:
        return sizes
    w, h = frames[0].width // 2 * 2, frames[0].height // 2 * 2
    mp4 = out / f"{stem}.mp4"
    writer = imageio_ffmpeg.write_frames(str(mp4), (w, h), fps=fps, quality=6)
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(np.asarray(f.convert("RGB"))[:h, :w]))
    writer.close()
    sizes[mp4.name] = mp4.stat().st_size
    return sizes
