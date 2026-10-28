"""The UI figures: captures the real viewer with Playwright (headless Chromium, 1920x1080, device scale 2) while it plays ONE recorded landing of the trained controller.

    python paper_assets/scripts/make_live_episode.py --level 1          # records the episode (seed printed) into logs/zz_paper_view/live/
    python paper_assets/scripts/capture_ui.py [--port 8791]             # needs the viewer: python src/app/server.py --port 8791 --run zz_paper_view

Outputs (paper_assets/): fig3_1_interface_full.png, fig3_2_recording_panels.png, fig4_1_landing_scatter.png, fig4_2_descent_phase_portrait.png, fig4_2b_decision_boundary.png
The viewer itself is not changed: panels are only moved and switched on through the page's own settings, the landing scatter is fed with the episodes of episodes.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_assets"
LEVEL_COLOURS = ["#6ee7ff", "#ffd166", "#ff9d4d", "#ff6b9a"]

VIS_ON = ["net", "term", "tele", "cam", "phase", "gFuel", "gRec", "gLand", "gBnd", "gPfd", "gAtt", "flames", "smoke", "trails", "grid", "tags", "rings"]
PANEL_IDS = ["mini", "term", "tele", "camBox", "cam2Box", "phaseBox", "gp_fuel", "gp_rec", "gp_land", "gp_bnd", "gp_pfd", "gp_att"]

# stage pixels (1920 x 1080): id -> (left, top, width)
FULL = {"mini": (14, 80, 330), "term": (14, 310, 330), "phaseBox": (14, 610, 360), "gp_fuel": (400, 80, 330), "gp_rec": (400, 360, 330), "gp_land": (400, 660, 330),
        "tele": (1640, 80, 270), "camBox": (1320, 700, 470), "gp_pfd": (1300, 80, 320), "gp_bnd": (1300, 440, 320), "gp_att": (1000, 760, 280)}
RECORDING = {"camBox": (30, 100, 640), "gp_pfd": (700, 100, 380), "tele": (1110, 100, 330), "gp_fuel": (1470, 100, 420)}      # cam, flight display, telemetry (incl. its forces), fuel profile


def js_layout(spec: dict) -> str:
    return """(spec) => {
      const stage = document.querySelector('.stage'), sr = stage.getBoundingClientRect(), z = parseFloat(stage.style.getPropertyValue('--zf')) || 1;
      const ids = %s; ids.forEach(id => { const el = document.getElementById(id); if (!el) return; if (!spec[id]) { el.style.display = 'none'; return; }
        const [x, y, w] = spec[id]; el.style.display = ''; el.style.right = el.style.bottom = 'auto'; el.style.left = (x / z) + 'px'; el.style.top = (y / z) + 'px'; el.style.width = (w / z) + 'px'; });
      return z; }""" % json.dumps(PANEL_IDS)


def rect(pg, i):
    return pg.evaluate("(id) => { const r = document.getElementById(id).getBoundingClientRect(); return {x: r.left, y: r.top, width: r.width, height: r.height}; }", i)


def shot(pg, i, path, pad=6, cut_bottom=0):
    """cut_bottom: CSS px dropped from the bottom (the decision boundary's two footnotes overlap each other in the viewer; the strip is left out)."""
    r = rect(pg, i)
    pg.screenshot(path=str(path), clip=dict(x=max(0, r["x"] - pad), y=max(0, r["y"] - pad), width=r["width"] + 2 * pad, height=r["height"] + 2 * pad - cut_bottom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--keep-run", action="store_true", help="do not delete logs/zz_paper_view afterwards")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    base = f"http://localhost:{a.port}/"
    ep = json.loads((ROOT / "logs" / "zz_paper_view" / "live" / "ep_00000001_e0.json").read_text())
    frames = ep["frames"]
    land_i = next(i for i, f in enumerate(frames) if f["y"] - 15.0 <= 28.0)                 # about 28 m above the pad: the engine is firing, the plume is in the air
    scatter = [r for r in csv.DictReader(open(OUT / "episodes.csv")) if r["controller"] == "sx_imitation_quiet" and r["touchdown_speed_mps"] != ""]
    scatter.sort(key=lambda r: (int(r["seed"]), int(r["level"])))
    land = [dict(x=float(r["final_x_m"]), v=float(r["touchdown_speed_mps"]), col=LEVEL_COLOURS[int(r["level"])], ok=r["end"] == "rest", soft=r["success"] == "1", lv=int(r["level"])) for r in scatter]
    vis = {k: (k in VIS_ON) for k in ("net", "term", "tele", "cam", "cam2", "phase", "gFuel", "gRec", "gLand", "gBnd", "gPfd", "gAtt", "gHeat", "gCrit", "gPca", "gGhost", "gRew", "gEn",
                                      "airPts", "ignPts", "landPts", "endNotes", "rings", "trails", "flames", "smoke", "tags", "grid")}
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        ctx.add_init_script("localStorage.setItem('spacexperiment.vis', %s); localStorage.setItem('spacexperiment.mode', 'advanced'); localStorage.setItem('spacexperiment.layout', 'landscape');" % json.dumps(json.dumps(vis)))
        pg = ctx.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(base)
        pg.wait_for_function("slots[0] && slots[0].ep && slots[0].last >= 0", timeout=60000)
        pg.evaluate("document.getElementById('followBtn').click()")                      # follow cam: the stage fills the picture
        pg.evaluate("(L) => { LANDINGS.length = 0; L.forEach(q => LANDINGS.push(q)); }", land)
        pg.evaluate(js_layout(FULL), FULL)
        pg.wait_for_function(f"slots[0].last >= {land_i}", timeout=90000)
        pg.evaluate("paused = true")                                                     # freeze the landing about 28 m up
        pg.wait_for_timeout(400)
        print("frozen at frame", pg.evaluate("slots[0].last"), "of", len(frames), "target", land_i, "| altitude", round(frames[pg.evaluate("slots[0].last")]["y"] - 15.0, 1), "m")
        pg.screenshot(path=str(OUT / "fig3_1_interface_full.png"))
        # ---- the recording panels, cropped together
        pg.evaluate("() => { const d = document.createElement('div'); d.style.cssText = 'position:absolute;inset:0;z-index:4;background:#070a14'; document.querySelector('.stage').appendChild(d); %s.forEach(id => { const e = document.getElementById(id); if (e) e.style.zIndex = 6; }); }" % json.dumps(PANEL_IDS))     # a plain dark ground behind the cropped panels
        pg.evaluate(js_layout(RECORDING), RECORDING)
        pg.wait_for_timeout(600)
        boxes = [rect(pg, i) for i in RECORDING]
        x0, y0 = min(b_["x"] for b_ in boxes) - 10, min(b_["y"] for b_ in boxes) - 10
        x1, y1 = max(b_["x"] + b_["width"] for b_ in boxes) + 10, max(b_["y"] + b_["height"] for b_ in boxes) + 10
        pg.screenshot(path=str(OUT / "fig3_2_recording_panels.png"), clip=dict(x=max(0, x0), y=max(0, y0), width=min(1920, x1) - max(0, x0), height=min(1080, y1) - max(0, y0)))
        # ---- the end of the landing: the whole path in the phase portrait and the decision boundary; the scatter from episodes.csv
        pg.evaluate("(n) => { const s = slots[0], f = s.ep.frames; for (let i = s.last + 1; i < f.length; i++) onFrame(s, f[i], i); s.fi = f.length - 1; s.last = f.length - 1; }", len(frames))
        pg.evaluate("(L) => { LANDINGS.length = 0; L.forEach(q => LANDINGS.push(q)); }", land)
        single = {"phaseBox": (30, 100, 520), "gp_bnd": (600, 100, 600), "gp_land": (1240, 100, 560)}
        pg.evaluate(js_layout(single), single)
        pg.wait_for_timeout(1200)
        shot(pg, "phaseBox", OUT / "fig4_2_descent_phase_portrait.png")
        shot(pg, "gp_bnd", OUT / "fig4_2b_decision_boundary.png", cut_bottom=26)
        shot(pg, "gp_land", OUT / "fig4_1_landing_scatter.png")
        print("page errors:", errors[:5])
        b.close()
    if not a.keep_run:
        shutil.rmtree(ROOT / "logs" / "zz_paper_view", ignore_errors=True)


if __name__ == "__main__":
    main()
