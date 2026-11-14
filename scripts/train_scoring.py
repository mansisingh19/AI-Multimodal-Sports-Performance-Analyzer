"""Step 4: train the performance scoring model (needs the labels CSV).
python scripts/train_scoring.py --sport cricket"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.scoring import train_scoring_model

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True, choices=config.SPORTS)
args = ap.parse_args()
s = train_scoring_model(args.sport)
print(pd.DataFrame(s["comparison"])[["model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))
print("Best model:", s["best_model"])
print("Top features:", [t["feature"] for t in s["top_features"][:8]])
