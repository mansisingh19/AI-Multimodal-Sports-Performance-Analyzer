"""Finds out WHY the scoring model is weak.   python scripts/diagnose_scoring.py --sport cricket
Prints: baseline, simpler models, label quality, best features, pose-detection rate."""
import argparse, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import extract_features, feature_names
from src.pose import load_pose
from src.scoring import load_labels

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True, choices=config.SPORTS)
args = ap.parse_args()
sport = args.sport
df, params = load_labels(sport)
actions = config.SPORTS[sport]["actions"]

feats, onehot, det = [], [], []
for vid in df["video_id"]:
    kp, motion = load_pose(config.KP_DIR / sport / f"{vid}.npz")
    det.append(float((kp[..., 3].max(axis=1) > 0.05).mean()))
    feats.append(extract_features(kp, motion))
    onehot.append([1.0 if vid.startswith(a + "/") else 0.0 for a in actions])
F = pd.DataFrame(feats)[feature_names()]
X = np.hstack([F.values, np.array(onehot)])
Y = df[params].values.astype(float)
overall = Y.mean(axis=1)

print(f"\nClips used: {len(df)}")
print(f"\n1) Pose detection (fraction of frames with a person found): mean {np.mean(det):.2f}, "
      f"worst {np.min(det):.2f}, clips below 0.8: {int(np.sum(np.array(det) < 0.8))}")

C = np.corrcoef(Y.T)
off = C[~np.eye(len(params), dtype=bool)]
print(f"\n2) Labels: average correlation between the {len(params)} parameters = {off.mean():.2f}")
print("   (above 0.8 means the rater gave almost the same score for everything)")
print("   Score spread (std) per parameter:", {p: round(float(v), 1) for p, v in zip(params, Y.std(axis=0))})

print("\n3) Features most related to the overall score (Spearman rho):")
rho = {c: spearmanr(F[c], overall)[0] for c in F.columns}
for c, r in sorted(rho.items(), key=lambda x: -abs(x[1]))[:6]:
    print(f"   {c:22s} {r:+.2f}")

models = {
    "Predict-the-mean (baseline)": DummyRegressor(),
    "Ridge (alpha=30)": make_pipeline(StandardScaler(), Ridge(alpha=30)),
    "Ridge (alpha=300)": make_pipeline(StandardScaler(), Ridge(alpha=300)),
    "RandomForest (small, regularised)": RandomForestRegressor(n_estimators=300, min_samples_leaf=6,
                                                               max_features=0.3, random_state=0, n_jobs=-1),
}


def cv_scores(model, X, y, seeds=(0, 1, 2)):
    mae, r2 = [], []
    for s in seeds:
        p = np.clip(cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=s)), 0, 100)
        mae.append(mean_absolute_error(y, p)); r2.append(r2_score(y, p))
    return np.mean(mae), np.mean(r2)


print("\n4) Models without the GRU embedding (5-fold CV, repeated 3 times). Target = overall score")
rows = []
for name, m in models.items():
    mae, r2 = cv_scores(m, X, overall)
    rows.append((name, mae, r2))
print(pd.DataFrame(rows, columns=["model", "MAE", "R2"]).round(3).to_string(index=False))

best = max(rows[1:], key=lambda r: r[2])
print(f"\n5) Best simple model: {best[0]}. R2 per parameter:")
m = models[best[0]]
print("   " + ", ".join(f"{p}: {cv_scores(m, X, Y[:, i])[1]:.2f}" for i, p in enumerate(params)))

print("\nHow to read this: R2 below 0 = worse than guessing the average. About 0.1-0.3 = weak but real signal."
      " Above 0.3 = good for a small, subjective dataset.")
