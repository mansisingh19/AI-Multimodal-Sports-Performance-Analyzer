"""Performance assessment: fused multimodal features -> scores out of 100 (Machine Learning module).

Fusion (multimodal) vector =
    [ pose-derived biomechanical features        (numerical modality)
    + video motion-energy features               (video modality, inside the feature dict)
    + action probabilities                       (what the action model saw)
    + PCA-reduced GRU clip embedding             (learned pose-sequence modality) ]
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor

from . import config
from .features import extract_features, feature_names
from .pose import detection_rate, load_pose

EMB_COMPONENTS = 8


def fuse(features, action_probs, embedding, pca=None):
    parts = [np.array([features[k] for k in feature_names()]),
             np.array(action_probs, dtype=float)]
    if embedding is not None and pca is not None:
        parts.append(pca.transform(np.asarray(embedding).reshape(1, -1))[0])
    return np.concatenate(parts)


def candidate_models(seed=42):
    models = {
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
        "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=seed)),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = MultiOutputRegressor(XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                              subsample=0.8, random_state=seed))
    except ImportError:
        pass
    return models


def load_labels(sport):
    path = config.LABEL_DIR / f"{sport}_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}. Run scripts/make_label_template.py and fill in the scores.")
    df = pd.read_csv(path)
    params = config.SPORTS[sport]["parameters"]
    missing = [p for p in params if p not in df.columns]
    if missing:
        raise ValueError(f"Label file is missing columns: {missing}")
    df = df.dropna(subset=params).reset_index(drop=True)
    good = [detection_rate(load_pose(config.KP_DIR / sport / f"{v}.npz")[0]) >= config.MIN_DETECTION_SCORING
            for v in df["video_id"]]
    dropped = len(df) - sum(good)
    df = df[good].reset_index(drop=True)
    print(f"Labelled clips: {len(df) + dropped}; used: {len(df)}; left out because the player was found in "
          f"fewer than {config.MIN_DETECTION_SCORING:.0%} of frames: {dropped}")
    if len(df) < 30:
        raise RuntimeError(f"Only {len(df)} labelled clips. Label at least 100-300 clips for a trustworthy model.")
    return df, params


def build_training_matrix(sport, df, recognizer=None):
    actions = config.SPORTS[sport]["actions"]
    rows, embs = [], []
    for vid in df["video_id"]:
        kp, motion = load_pose(config.KP_DIR / sport / f"{vid}.npz")
        feats = extract_features(kp, motion)
        action = vid.split("/")[0]
        onehot = [1.0 if a == action else 0.0 for a in actions]   # true action during training
        emb = recognizer.predict(kp)[3] if recognizer is not None else None
        rows.append((feats, onehot)); embs.append(emb)
    pca = None
    if recognizer is not None:
        E = np.vstack(embs)
        pca = PCA(n_components=min(EMB_COMPONENTS, E.shape[0] - 1, E.shape[1])).fit(E)
    X = np.vstack([fuse(f, o, e, pca) for (f, o), e in zip(rows, embs)])
    return X, pca


def train_scoring_model(sport, out_dir=None, seed=42):
    df, params = load_labels(sport)
    try:
        from .action_model import ActionRecognizer
        recognizer = ActionRecognizer(sport)
    except (FileNotFoundError, ImportError):
        recognizer = None          # still works without the embedding block (train action model first for full fusion)
    X, pca = build_training_matrix(sport, df, recognizer)
    Y = df[params].values.astype(float)

    n_splits = min(5, len(df))
    cv = GroupKFold(n_splits=n_splits) if "player_id" in df.columns and df["player_id"].notna().all() \
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df["player_id"] if isinstance(cv, GroupKFold) else None

    table, best_name, best_mae = [], None, 1e9
    for name, model in candidate_models(seed).items():
        pred = np.clip(cross_val_predict(model, X, Y, cv=cv, groups=groups), 0, 100)
        mae = mean_absolute_error(Y, pred)
        table.append({"model": name, "MAE": mae, "RMSE": mean_squared_error(Y, pred) ** 0.5,
                      "R2": r2_score(Y, pred),
                      **{f"MAE_{p}": mean_absolute_error(Y[:, i], pred[:, i]) for i, p in enumerate(params)}})
        if mae < best_mae:
            best_name, best_mae = name, mae

    best = candidate_models(seed)[best_name].fit(X, Y)
    names = feature_names() + [f"p_{a}" for a in config.SPORTS[sport]["actions"]] + \
            ([f"emb_{i}" for i in range(pca.n_components_)] if pca is not None else [])
    importance = np.mean([est.feature_importances_ for est in best.estimators_], axis=0)
    top = sorted(zip(names, importance), key=lambda x: -x[1])[:15]

    out = Path(out_dir or config.MODEL_DIR / sport); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "pca": pca, "params": params, "model_name": best_name}, out / "scoring_model.joblib")
    pd.DataFrame(table).to_csv(out / "scoring_comparison.csv", index=False)
    summary = {"sport": sport, "n_clips": len(df), "best_model": best_name, "cv_folds": n_splits,
               "comparison": table, "top_features": [{"feature": n, "importance": float(v)} for n, v in top]}
    (out / "scoring_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


class PerformanceScorer:
    def __init__(self, sport, model_dir=None):
        path = Path(model_dir or config.MODEL_DIR / sport) / "scoring_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No trained scoring model for '{sport}'. Run: python scripts/train_scoring.py --sport {sport}")
        b = joblib.load(path)
        self.model, self.pca, self.params, self.name = b["model"], b["pca"], b["params"], b["model_name"]

    def score(self, features, action_probs, embedding):
        x = fuse(features, action_probs, embedding, self.pca).reshape(1, -1)
        vals = np.clip(self.model.predict(x)[0], 0, 100)
        scores = {p: float(v) for p, v in zip(self.params, vals)}
        return scores, float(np.mean(vals))
