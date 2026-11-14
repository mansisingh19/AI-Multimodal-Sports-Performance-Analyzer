"""Checks that the whole pipeline runs end-to-end on FAKE data.
    python tests/smoke_test.py
The numbers it prints mean nothing; it only proves the code works."""
import os, sys, tempfile
from pathlib import Path

tmp = tempfile.mkdtemp()
os.environ["SPORTS_AI_ROOT"] = tmp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from synthetic import make_clip
from src import config, db
from src.features import extract_features, build_sequence, feature_names
from src.feedback import generate_feedback
from src.pose import save_pose
from src.scoring import train_scoring_model, PerformanceScorer

rng = np.random.default_rng(0)
sport, actions, params = "cricket", config.SPORTS["cricket"]["actions"], config.SPORTS["cricket"]["parameters"]

rows = []
for ai, action in enumerate(actions):
    for n in range(14):
        q = rng.uniform(0.2, 1.0)
        kp, motion = make_clip(ai, q, rng=rng)
        save_pose(config.KP_DIR / sport / action / f"clip{n}.npz", kp, motion, {"fps": 30, "width": 640, "height": 360})
        score = lambda: float(np.clip(40 + 55 * q + rng.normal(0, 4), 0, 100))
        rows.append({"video_id": f"{action}/clip{n}", "player_id": f"p{n % 5}", **{p: score() for p in params}})
config.LABEL_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(config.LABEL_DIR / f"{sport}_scores.csv", index=False)
print("1. synthetic dataset created")

kp, motion = make_clip(0, 0.8, rng=rng)
f = extract_features(kp, motion)
assert len(f) == len(feature_names()) and all(np.isfinite(list(f.values())))
seq = build_sequence(kp, config.SEQ_LEN)
assert seq.shape == (config.SEQ_LEN, 52), seq.shape
print(f"2. features OK ({len(f)} features), sequence shape {seq.shape}")

summary = train_scoring_model(sport)
print("3. scoring trained:", summary["best_model"], {c["model"]: round(c["MAE"], 2) for c in summary["comparison"]})

scorer = PerformanceScorer(sport)
scores, overall = scorer.score(f, [1.0, 0.0, 0.0], None)
print("4. scoring inference OK, overall =", round(overall, 1))

fb = generate_feedback({**scores, "balance": 50.0}, overall)
assert fb["recommendations"]
print("5. feedback OK:", fb["summary"])

db.save_result(sport, "x.mp4", {"action": "cover_drive", "confidence": 0.9, "overall": overall, "scores": scores})
assert len(db.load_history()) == 1
print("6. database OK")

try:
    import torch  # noqa
    from src.action_model import train_action_model, ActionRecognizer
    m = train_action_model(sport, epochs=15)
    print("7. action model trained, test accuracy =", round(m["test_accuracy"], 2))
    a, c, probs, emb = ActionRecognizer(sport).predict(kp)
    summary = train_scoring_model(sport)          # now with the GRU embedding fused in
    s2, o2 = PerformanceScorer(sport).score(f, list(probs.values()), emb)
    print("8. full multimodal fusion OK, overall =", round(o2, 1))
except ImportError:
    print("7-8. torch not installed here - action model part skipped (install torch to test it)")
print("ALL CHECKS PASSED")
