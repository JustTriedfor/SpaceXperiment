"""Runs src.training.finetune.finetune in its own process (so the GPU context and its memory are released when it ends).   python fit_worker.py '<json kwargs>'"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.training.finetune import finetune  # noqa: E402

k = json.loads(sys.argv[1])
finetune(k.pop("checkpoint"), k.pop("sources"), k.pop("out_name"), log=lambda s: print(s, flush=True), **k)
