"""Quick test: how often is the player found?
    python scripts/check_detection.py --sport cricket            (normal vs tracked)
    python scripts/check_detection.py --sport cricket --compare  (tries 4 tracking settings)
Run this BEFORE re-extracting the whole dataset."""
import os
os.environ["GLOG_minloglevel"] = "2"          # hide the long MediaPipe/TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse, random, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from src import config
from src.pose import extract_pose

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True, choices=config.SPORTS)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--compare", action="store_true", help="compare several tracking settings")
args = ap.parse_args()

videos = [p for p in (config.RAW_DIR / args.sport).rglob("*") if p.suffix.lower() in config.VIDEO_TYPES]
random.Random(1).shuffle(videos)
videos = videos[:args.n]
rate = lambda kp: float((kp[..., 3].max(axis=1) > 0.05).mean())

if args.compare:
    variants = {
        "A  no tracking":              dict(use_tracker=False),
        "B  tracking (first version)":  dict(use_tracker=True),
        "C  + fresh detection per crop": dict(use_tracker=True, static=True, min_det=0.3),
        "D  + high resolution":         dict(use_tracker=True, max_width=1280, imgsz=960),
        "E  + bigger crop margin":      dict(use_tracker=True, margin=0.30),
    }
    res = {k: [] for k in variants}; secs = {k: 0.0 for k in variants}
    for v in videos:
        for name, kw in variants.items():
            t = time.time(); res[name].append(rate(extract_pose(v, **kw)[0])); secs[name] += time.time() - t
        print(f"done {v.name}", flush=True)
    print(f"\n{'setting':34s} {'avg detection':>14s} {'seconds/clip':>13s}")
    for name in variants:
        print(f"{name:34s} {np.mean(res[name]):>14.2f} {secs[name] / len(videos):>13.1f}")
    print("\nPer clip:", {n: [round(x, 2) for x in r] for n, r in res.items()})
else:
    print(f"{'clip':40s} {'normal':>7s} {'tracked':>8s}")
    a_all, b_all = [], []
    for v in videos:
        a = rate(extract_pose(v)[0]); b = rate(extract_pose(v, use_tracker=True)[0])
        a_all.append(a); b_all.append(b)
        print(f"{v.parent.name + '/' + v.name:40s} {a:>7.2f} {b:>8.2f}", flush=True)
    print(f"\nAverage detection: normal {np.mean(a_all):.2f}, tracked {np.mean(b_all):.2f}")
