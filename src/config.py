"""Central settings. To add a new sport, add one entry to SPORTS (nothing else changes)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SPORTS_AI_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
RAW_DIR = DATA / "raw_videos"        # data/raw_videos/<sport>/<action>/clip.mp4
KP_DIR = DATA / "keypoints"          # data/keypoints/<sport>/<action>/clip.npz
LABEL_DIR = DATA / "labels"          # data/labels/<sport>_scores.csv
MODEL_DIR = ROOT / "models"          # models/<sport>/...
DB_PATH = ROOT / "results.db"

SEQ_LEN = 32          # every clip is resampled to this many frames for the action model
VIDEO_TYPES = (".mp4", ".avi", ".mov", ".mkv")

# Clips where the player was found in too few frames are left out of training (their keypoints are unreliable)
MIN_DETECTION_ACTION = 0.4    # action model
MIN_DETECTION_SCORING = 0.6   # scoring model (needs cleaner keypoints)

SPORTS = {
    "cricket": {
        "title": "Cricket (Batting)",
        "actions": ["cover_drive", "pull_shot", "defensive_shot"],
        "parameters": ["balance", "footwork", "posture", "head_stability",
                       "follow_through", "action_execution"],
    },
    "badminton": {
        "title": "Badminton",
        "actions": ["smash", "serve", "clear"],
        "parameters": ["balance", "footwork", "posture", "body_rotation",
                       "wrist_action", "action_execution"],
    },
}


def pretty(name: str) -> str:
    return name.replace("_", " ").title()
