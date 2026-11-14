"""Step 1: videos -> keypoints.   python scripts/extract_dataset.py --sport cricket [--tracker]
Expects videos in data/raw_videos/<sport>/<action>/*.mp4"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pose import extract_pose, save_pose

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True, choices=config.SPORTS)
ap.add_argument("--tracker", action="store_true", help="use YOLO player tracking (needs ultralytics)")
args = ap.parse_args()

for action in config.SPORTS[args.sport]["actions"]:
    folder = config.RAW_DIR / args.sport / action
    folder.mkdir(parents=True, exist_ok=True)
    for video in sorted(folder.iterdir()):
        if video.suffix.lower() not in config.VIDEO_TYPES:
            continue
        out = config.KP_DIR / args.sport / action / f"{video.stem}.npz"
        if out.exists():
            continue
        try:
            kp, motion, meta = extract_pose(video, use_tracker=args.tracker)
            save_pose(out, kp, motion, meta)
            print(f"{action}/{video.name}: {kp.shape[0]} frames")
        except Exception as e:
            print(f"SKIPPED {video.name}: {e}")
