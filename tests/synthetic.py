"""Fake stick-figure keypoints, ONLY for testing that the code runs. Not real sports data."""
import numpy as np

BASE = {0: (0.50, 0.20), 11: (0.46, 0.30), 12: (0.54, 0.30), 13: (0.44, 0.40), 14: (0.56, 0.40),
        15: (0.43, 0.48), 16: (0.57, 0.48), 23: (0.47, 0.55), 24: (0.53, 0.55), 25: (0.47, 0.72),
        26: (0.53, 0.72), 27: (0.47, 0.90), 28: (0.53, 0.90)}


def make_clip(action_idx, quality, T=60, rng=None):
    """quality 0..1: low quality = more head/hip wobble and sloppier footwork."""
    rng = rng or np.random.default_rng()
    kp = np.zeros((T, 33, 4), dtype=np.float32)
    kp[..., 3] = 1.0
    t = np.linspace(0, 1, T)
    for j in range(33):
        x, y = BASE.get(j, (0.5, 0.5))
        kp[:, j, 0], kp[:, j, 1] = x, y
    swing = np.sin(np.pi * t)
    for w in (15, 16):
        kp[:, w, 0] += (0.25 + 0.1 * action_idx) * swing * (1 if action_idx != 1 else -1)
        kp[:, w, 1] -= (0.10 + 0.15 * action_idx) * swing
    wobble = (1 - quality) * 0.04
    for j in (0, 23, 24):
        kp[:, j, 0] += rng.normal(0, wobble, T)
    for j in (27, 28):
        kp[:, j, 0] += (1 - quality) * 0.05 * np.sin(6 * np.pi * t) + rng.normal(0, 0.003, T)
    kp[..., :2] += rng.normal(0, 0.002, kp[..., :2].shape).astype(np.float32)
    motion = (0.02 + 0.03 * swing + rng.normal(0, 0.002, T)).astype(np.float32)
    return kp, motion
