"""End-to-end analysis: video file -> action, scores, feedback."""
from .action_model import ActionRecognizer
from .config import SPORTS
from .feedback import generate_feedback
from .features import extract_features
from .pose import detection_rate, extract_pose, render_pose_frames
from .scoring import PerformanceScorer


class SportsAnalyzer:
    def __init__(self, sport):
        if sport not in SPORTS:
            raise ValueError(f"Unknown sport: {sport}")
        self.sport = sport
        self.recognizer = ActionRecognizer(sport)
        self.scorer = PerformanceScorer(sport)

    def analyze(self, video_path, use_tracker=False, n_frames=6):
        kp, motion, meta = extract_pose(video_path, use_tracker=use_tracker)
        detection = detection_rate(kp)
        action, conf, probs, emb = self.recognizer.predict(kp)
        feats = extract_features(kp, motion)
        scores, overall = self.scorer.score(feats, [probs[c] for c in self.recognizer.classes], emb)
        result = {"sport": self.sport, "action": action, "confidence": conf, "action_probs": probs,
                  "scores": scores, "overall": overall, "pose_detection": detection, "features": feats, "meta": meta,
                  "feedback": generate_feedback(scores, overall)}
        result["pose_frames"] = render_pose_frames(video_path, kp, n=n_frames)
        return result
