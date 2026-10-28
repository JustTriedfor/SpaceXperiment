"""The hull pictures for the landing cams: every transparent PNG in src/app/cam_hulls/ is one overlay (hull, legs and fins as the camera sees them).

The page picks one per episode for the landing cam and a different one for the partner cam, so adding more pictures there is all it takes to get more variety.
A picture is any transparent PNG of about 16:9 with the hull along the bottom and the legs meeting at its top. `apex` is where (as a fraction of the picture's
height) the hull's top is; the page shifts the camera's view so that the engine's flame appears exactly there, whichever picture is shown.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

HULL_DIR = Path(__file__).with_name("cam_hulls")
NAME_RE = re.compile(r"[A-Za-z0-9_\-]{1,60}\.png")
_CACHE: dict = {}


def apex_of(path: Path) -> float:
    """Height (0 = top, 1 = bottom) at which the hull's top fills the middle of the picture: the first row where more than 60% of the columns
    between 42% and 54% of the width are opaque (the thin legs above it do not count)."""
    from PIL import Image
    al = np.asarray(Image.open(path).convert("RGBA"))[..., 3] / 255.0
    h, w = al.shape
    cov = (al[:, int(0.42 * w):int(0.54 * w)] > 0.5).mean(axis=1)
    hit = np.nonzero(cov > 0.6)[0]
    return float(hit[0]) / h if len(hit) else 0.42


def hull_list() -> list:
    out = []
    if HULL_DIR.is_dir():
        for p in sorted(HULL_DIR.glob("*.png")):
            if not NAME_RE.fullmatch(p.name):
                continue
            key = (p.name, p.stat().st_mtime_ns)
            if key not in _CACHE:
                try:
                    _CACHE[key] = round(apex_of(p), 4)
                except Exception:                                   # a damaged picture must not hide the others
                    continue
            out.append(dict(name=p.name, url="/cam_hulls/" + p.name, apex=_CACHE[key]))
    return out


def hull_file(name: str) -> Path | None:
    """The file for a name from the listing, or None (only plain names inside the folder are ever served)."""
    if not NAME_RE.fullmatch(name or ""):
        return None
    p = HULL_DIR / name
    return p if p.is_file() else None


MODEL_DIR = Path(__file__).with_name("models")


def model_file(name: str) -> Path | None:
    """A picture from src/app/models/ (plain names only), e.g. the stage in the airflow panel."""
    if not NAME_RE.fullmatch(name or ""):
        return None
    p = MODEL_DIR / name
    return p if p.is_file() else None
