"""Helper functions for the labelling tool (kept separate from Streamlit so they can be tested)."""
import io
import random
from itertools import zip_longest

import cv2
import pandas as pd
from PIL import Image

from . import config

# What the rater should look at, in simple words
RUBRIC = {
    "balance": "Body stays steady over the feet. No stumbling, falling away or leaning too far.",
    "footwork": "Feet move to the right place at the right time. Good, stable base at the end.",
    "posture": "Back is straight, knees are bent, body shape looks strong and controlled.",
    "head_stability": "Head stays still and eyes stay level, especially at the moment of contact.",
    "follow_through": "Swing finishes smoothly and completely, hands finish high. No sudden stop.",
    "action_execution": "Overall quality of the shot. Would a coach call it a good shot?",
    "body_rotation": "Hips and shoulders turn well to create power.",
    "wrist_action": "Wrist snaps through the contact smoothly and on time.",
}


def label_path(sport):
    return config.LABEL_DIR / f"{sport}_scores.csv"


def ensure_label_file(sport):
    """Create the CSV (one row per clip, empty scores) if it does not exist. Returns the DataFrame."""
    path, params = label_path(sport), config.SPORTS[sport]["parameters"]
    if path.exists():
        df = pd.read_csv(path)
    else:
        ids = [f"{a}/{p.stem}" for a in config.SPORTS[sport]["actions"]
               for p in sorted((config.KP_DIR / sport / a).glob("*.npz"))]
        if not ids:
            raise FileNotFoundError("No keypoints found. Run scripts/extract_dataset.py first.")
        df = pd.DataFrame({"video_id": ids, "player_id": ""})
    for p in params:
        if p not in df.columns:
            df[p] = float("nan")
    return df


def save_labels(sport, df):
    path = label_path(sport)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def is_labelled(df, sport):
    return df[config.SPORTS[sport]["parameters"]].notna().all(axis=1)


def review_order(df, sport, seed=0):
    """Unlabelled clip ids in random order, mixed across actions (so every shot gets labelled evenly)."""
    rng = random.Random(seed)
    todo = df.loc[~is_labelled(df, sport), "video_id"].tolist()
    groups = {}
    for vid in todo:
        groups.setdefault(vid.split("/")[0], []).append(vid)
    for g in groups.values():
        rng.shuffle(g)
    mixed = [v for row in zip_longest(*groups.values()) for v in row if v is not None]
    return mixed


def find_video(sport, video_id):
    action, stem = video_id.split("/", 1)
    folder = config.RAW_DIR / sport / action
    for ext in config.VIDEO_TYPES:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Video for {video_id} not found in {folder}")


def clip_gif(video_path, width=420, step=2, ms=70):
    """Small looping GIF bytes so the clip can be shown in any browser (AVI does not play in browsers)."""
    cap = cv2.VideoCapture(str(video_path))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, int(h * width / w)))
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"Could not read {video_path}")
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=ms, loop=0)
    return buf.getvalue()
