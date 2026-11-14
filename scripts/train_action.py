"""Step 3: train the action recognition model.   python scripts/train_action.py --sport cricket"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.action_model import train_action_model

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True, choices=config.SPORTS)
ap.add_argument("--epochs", type=int, default=150)
args = ap.parse_args()
m = train_action_model(args.sport, epochs=args.epochs)
print(json.dumps({k: v for k, v in m.items() if k != "confusion_matrix"}, indent=2))
print("Confusion matrix:", m["confusion_matrix"])
