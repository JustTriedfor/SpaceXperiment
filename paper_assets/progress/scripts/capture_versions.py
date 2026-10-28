"""Screenshots of the two versions of the viewer that can still be run, at 1280x720, headless Chromium (Playwright).

    prototype  = the original Rocketeer (Box2D LunarLander) viewer, C:\\Users\\firdo\\rocketeer\\src (files last modified 18-19 Oct 2026), served from a temp copy
                 on port 8793 with one episode its own training run recorded (custom_s0_staged8, episode 4428, env 5; the file carries no seed)
    current    = SpaceXperiment viewer, port 8791, episode = sx_imitation_quiet, level 1, seed 122 (make_live_episode.py)

    python paper_assets/progress/scripts/capture_versions.py proto  <out.png>
    python paper_assets/progress/scripts/capture_versions.py current <out.png>

Both pages are left as they are (default panels, default layout); the only things done are: pick the lander tile, let the recorded episode play in real
time up to FRACTION of its frames (the feet are about to touch down), and pause.
"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

which, out = sys.argv[1], sys.argv[2]
URL = {"proto": "http://localhost:8793/", "current": "http://localhost:8791/"}[which]
SLOT = {"proto": 5, "current": 0}[which]
FRACTION = {"proto": 0.80, "current": 0.86}[which]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 720})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL)
    pg.wait_for_function(f"slots[{SLOT}] && slots[{SLOT}].ep && slots[{SLOT}].last >= 0", timeout=60000)
    if SLOT:
        pg.evaluate(f"selectSlot({SLOT})")                                       # the viewer's own function behind the numbered lander buttons
    n = pg.evaluate(f"slots[{SLOT}].ep.frames.length")
    target = int(n * FRACTION)
    pg.wait_for_function(f"slots[{SLOT}].last >= {target}", timeout=120000)
    pg.evaluate("paused = true")
    pg.wait_for_timeout(500)
    print(which, "frame", pg.evaluate(f"slots[{SLOT}].last"), "of", n, "| page errors:", errs[:3])
    pg.screenshot(path=out)
    b.close()
