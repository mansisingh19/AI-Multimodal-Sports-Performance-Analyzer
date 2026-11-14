"""Command-line test.   python scripts/analyze_video.py --sport cricket --video my_clip.mp4"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pipeline import SportsAnalyzer

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True, choices=config.SPORTS)
ap.add_argument("--video", required=True)
args = ap.parse_args()
r = SportsAnalyzer(args.sport).analyze(args.video)
print(f"Action: {r['action']} ({r['confidence']:.0%})")
for p, s in r["scores"].items():
    print(f"  {config.pretty(p)}: {s:.0f}/100")
print(f"Overall: {r['overall']:.0f}/100\n{r['feedback']['summary']}")
for t in r["feedback"]["recommendations"]:
    print(" -", t)
