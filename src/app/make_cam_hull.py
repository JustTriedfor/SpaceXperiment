"""Draws src/app/cam_hull_generated.png (the app itself shows cam_hull.png, your own overlay; copy this one over it to use the generated hull): the stage's hull as the landing cam sees it, computed (not drawn by hand) by ray-tracing the real stage geometry from the camera's real mounting.

    python src/app/make_cam_hull.py            (rewrites cam_hull_generated.png)

What it shows: the camera is bolted to the lower part of the stage, 1.5 m off the skin, looking down the length of the 3.66 m cylinder. So the hull is a long column that
narrows toward a vanishing point (the engines, 14 m away): its sides converge, the stringers run toward the vanishing point, the tank joint and the soot-black engine section
are bands of constant depth that curve across it, and the column ends in the small circle of the engine skirt. The numbers below must match CAMV in index.html.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H, SS = 1280, 720, 2                 # the picture, and the supersampling used for smooth edges
FOV_DEG, SHIFT = 110.0, 0.15           # horizontal field of view; how far up the optical axis sits (fraction of the height)
OFF = 3.3                               # metres from the camera to the stage's axis (the hull's radius is 1.83, so the lens is 1.5 m off the skin)
R = 3.66 / 2                            # hull radius
HC = 30.0                               # metres from the camera down to the engine bells (the camera is on the upper part of the stage: the hull is visible along that stretch)
LEG_TOP, LEG_FOOT = 11.5, 10.5           # the legs hinge this high above the bells and their feet are this far from the axis (deployed)
rng = np.random.default_rng(5)


def noise(shape, scale, seed):
    r = np.random.default_rng(seed).random((shape[0] // scale + 3, shape[1] // scale + 3))
    img = Image.fromarray((r * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    w, h = W * SS, H * SS
    fl = (w / 2) / np.tan(np.radians(FOV_DEG / 2))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a, b = xx + 0.5 - w / 2, h * (0.5 - SHIFT) - yy - 0.5                  # a: right, b: up, in pixels from the optical axis
    A = a * a + b * b
    disc = (OFF * b) ** 2 - A * (OFF * OFF - R * R)                        # the ray against the cylinder whose axis runs along the view direction, OFF below the lens
    hit = disc > 0
    t = np.where(hit, (-OFF * b - np.sqrt(np.where(hit, disc, 1.0))) / np.maximum(A, 1e-9), 0.0)
    depth = t * fl                                                          # metres along the axis to where the ray meets the skin
    vis = hit & (t > 0) & (depth < HC)
    height = np.where(vis, HC - depth, 0.0)                                  # metres above the engine bells
    hr, hu = t * a, t * b + OFF                                              # the hit point relative to the axis (in the cross-section)
    nr, nu = hr / R, hu / R                                                  # surface normal
    phi = np.arctan2(hr, hu)                                                 # angle around the stage

    # ---- paint: a sooted white tank over a black engine section, with a tank joint and stringers
    n1 = noise((h, w), 40 * SS, 1)
    streak = noise((h, w), 3 * SS, 2)
    gr_h = np.clip(height, 0, None)
    soot = np.clip(np.exp(-(gr_h - 2.4) / 11.0) * 0.95 + 0.30 * (n1 - 0.5) + 0.22 * (streak - 0.5), 0.0, 1.0)
    paint = 0.70 * (1 - soot) + 0.05 * soot                                  # white paint ... soot black
    paint = np.where(gr_h < 2.4, 0.05 + 0.03 * n1, paint)                    # the engine section is black
    joint = sum(np.exp(-((gr_h - hj) / 0.08) ** 2) * 0.45 for hj in (8.5, 20.5, 27.0)) + np.exp(-((gr_h - 2.4) / 0.05) ** 2) * 0.35
    stringer = (np.abs(np.sin(phi * 12.0)) < 0.045) * 0.10
    albedo = np.clip(paint - joint - stringer, 0.01, 1.0)
    # ---- light: from the upper left, plus the sky; the underside of the column is dark
    light = np.clip(0.30 + 0.85 * np.clip(-0.55 * nr + 0.80 * nu, -0.4, 1.0), 0.12, 1.1)
    lum = albedo * light
    lum = np.clip(lum + 0.015 * (streak - 0.5), 0.0, 1.0) ** 0.95
    rgb = np.stack([lum * 1.00, lum * 0.97, lum * 0.93], axis=-1)

    # ---- edges: soft at the skirt end and where the ray just grazes the skin
    a_end = np.clip((HC - depth) / 0.15, 0.0, 1.0)
    a_gr = np.clip(np.sqrt(np.clip(disc, 0, None)) / (0.02 * np.sqrt(np.maximum(A, 1.0))), 0.0, 1.0)
    alpha = np.where(vis, a_end * a_gr, 0.0)
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = np.clip(rgb, 0, 1)
    out[..., 3] = alpha
    out[..., :3] *= out[..., 3:4]                                           # premultiplied, so the down-sizing does not fringe
    img = Image.fromarray((out * 255).astype(np.uint8), "RGBA").resize((W, H), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    al = np.maximum(arr[..., 3:4], 1e-4)
    arr[..., :3] = np.clip(arr[..., :3] / al, 0, 1)
    # ---- the landing legs, fixed to the hull. A Falcon 9 has four, 90 degrees apart; the camera sits between two of them, so those two show as a V that points up and out
    #      from the hull, each a tapered blade from its hinge (LEG_TOP above the bells) to its foot, in perspective. They are part of the picture, so they never move.
    legs = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dr = ImageDraw.Draw(legs)
    fls = (W * SS / 2) / np.tan(np.radians(FOV_DEG / 2))
    pt = lambda r_, u_, z_: np.array([W * SS / 2 + fls * r_ / z_, H * SS * (0.5 - SHIFT) - fls * u_ / z_])
    s45 = np.sqrt(0.5)
    for sd in (-1, 1):
        zt, zf = HC - LEG_TOP, HC + 1.0
        T = pt(sd * R * s45, -OFF + R * s45, zt)                                # the hinge, on the hull 45 degrees to either side of the camera's side
        F = pt(sd * LEG_FOOT * s45, -OFF + LEG_FOOT * s45, zf)                  # the foot pad, far out and below the engines
        d = (F - T) / np.linalg.norm(F - T)
        n = np.array([-d[1], d[0]])
        wa, wb = 1.15 * fls / zt, 0.22 * fls / zf                          # half-widths in pixels, a little exaggerated so the blades read at this size
        lit = [tuple(T - n * wa), tuple(F - n * wb), tuple(F), tuple(T)]
        drk = [tuple(T), tuple(F), tuple(F + n * wb), tuple(T + n * wa)]
        if sd > 0:
            lit, drk = drk, lit
        dr.polygon(lit, fill=(238, 242, 246, 255)); dr.polygon(drk, fill=(46, 52, 60, 255))
        dr.line([tuple(T - n * wa), tuple(F - n * wb), tuple(F + n * wb), tuple(T + n * wa), tuple(T - n * wa)], fill=(22, 25, 29, 255), width=2 * SS)
    legs = np.asarray(legs.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
    la = legs[..., 3:4]
    ha = arr[..., 3:4]
    comp = la + ha * (1 - la)                                                    # the legs are in front of the hull
    rgbc = (legs[..., :3] * la + arr[..., :3] * ha * (1 - la)) / np.maximum(comp, 1e-4)
    arr = np.concatenate([rgbc, comp], axis=-1)
    path = Path(__file__).with_name("cam_hull_generated.png")
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA").save(path, optimize=True)
    print("wrote", path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
