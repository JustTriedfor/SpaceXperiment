"""Shared helpers for the ablation jobs (Wilson interval, paths)."""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "ablation_2026-10"
Z = 1.959963984540054
CHECKPOINT = "rt1_final"          # the existing trained checkpoint (SHA-256 d8bc33ec...e86); the paper calls its results "sx_retrained"


def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
