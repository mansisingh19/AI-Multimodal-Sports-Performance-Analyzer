"""Step 2: create the CSV in which a coach/player rates each clip (0-100 per parameter).
python scripts/make_label_template.py --sport cricket
Fill the empty columns. Optional column player_id lets training keep each player in one fold."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True, choices=config.SPORTS)
args = ap.parse_args()

ids = [f"{a}/{p.stem}" for a in config.SPORTS[args.sport]["actions"]
       for p in sorted((config.KP_DIR / args.sport / a).glob("*.npz"))]
df = pd.DataFrame({"video_id": ids, "player_id": ""})
for p in config.SPORTS[args.sport]["parameters"]:
    df[p] = ""
out = config.LABEL_DIR / f"{args.sport}_scores.csv"
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    sys.exit(f"{out} already exists - not overwriting (your ratings are safe).")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows. Fill in the scores (0-100).")
