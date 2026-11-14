"""Pose keypoints -> numbers (Feature Extraction layer).

Two kinds of output:
  * extract_features()  : one dictionary of biomechanical features per clip (for the ML scorer)
  * build_sequence()    : a fixed-length joint sequence (for the deep-learning action model)

All distances are divided by the torso length, so they do not depend on how far
the player is from the camera.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

# MediaPipe landmark ids
NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28

SELECTED = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
MIRROR_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]   # swap left <-> right joints
EPS = 1e-6


# ---------- cleaning ----------
def clean_keypoints(kp):
    """Fill frames where no person was found, then smooth. Returns xy (T,33,2), visibility (T,33)."""
    xy = kp[..., :2].astype(np.float64).copy()
    vis = kp[..., 3]
    good = vis.max(axis=1) > 0.05
    T = len(xy)
    if good.sum() == 0:
        raise ValueError("No person was detected in this video.")
    if good.sum() < T:
        t = np.arange(T)
        for j in range(33):
            for c in range(2):
                xy[:, j, c] = np.interp(t, t[good], xy[good, j, c])
    if T >= 5:
        xy = uniform_filter1d(xy, size=5, axis=0, mode="nearest")
    return xy, vis


def torso_length(xy):
    sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hp = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    return max(float(np.mean(np.linalg.norm(sh - hp, axis=1))), 1e-3)


def angle(a, b, c):
    """Angle at point b (degrees) for arrays of shape (T,2)."""
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + EPS)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def line_angle(p, q):
    """Orientation (degrees, unwrapped) of the line from p to q."""
    d = q - p
    return np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))


def path_length(p):
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


# ---------- biomechanical features ----------
def extract_features(kp, motion=None):
    xy, _ = clean_keypoints(kp)
    T = len(xy)
    L = torso_length(xy)
    g = lambda j: xy[:, j]

    sh_mid, hip_mid = (g(L_SH) + g(R_SH)) / 2, (g(L_HIP) + g(R_HIP)) / 2
    an_mid, nose = (g(L_AN) + g(R_AN)) / 2, g(NOSE)
    f = {}

    # knees
    lk, rk = angle(g(L_HIP), g(L_KN), g(L_AN)), angle(g(R_HIP), g(R_KN), g(R_AN))
    kavg = (lk + rk) / 2
    f.update(knee_mean=kavg.mean(), knee_min=min(lk.min(), rk.min()),
             knee_range=np.ptp(kavg), knee_asym=np.mean(np.abs(lk - rk)))

    # elbows (the "active" arm is the one that moves more)
    le, re = angle(g(L_SH), g(L_EL), g(L_WR)), angle(g(R_SH), g(R_EL), g(R_WR))
    act = le if le.std() > re.std() else re
    f.update(elbow_mean=((le + re) / 2).mean(), elbow_active_min=act.min(),
             elbow_active_max=act.max(), elbow_active_range=np.ptp(act))

    # rotation of shoulders and hips
    sh_ang, hp_ang = line_angle(g(L_SH), g(R_SH)), line_angle(g(L_HIP), g(R_HIP))
    f.update(shoulder_rot_range=np.ptp(sh_ang), shoulder_rot_std=sh_ang.std(),
             hip_rot_range=np.ptp(hp_ang), hip_rot_std=hp_ang.std(),
             separation_mean=np.mean(np.abs(sh_ang - hp_ang)),
             separation_max=np.max(np.abs(sh_ang - hp_ang)))

    # spine lean (angle of the torso from vertical)
    d = sh_mid - hip_mid
    lean = np.degrees(np.arctan2(d[:, 0], -d[:, 1]))
    f.update(lean_mean_abs=np.mean(np.abs(lean)), lean_std=lean.std())

    # head stability
    f.update(head_motion=np.mean(nose.std(axis=0)) / L,
             head_rel_motion=np.mean((nose - sh_mid).std(axis=0)) / L,
             head_over_base=np.mean(np.abs(nose[:, 0] - an_mid[:, 0])) / L)

    # balance
    base = np.abs(g(L_AN)[:, 0] - g(R_AN)[:, 0])
    f.update(com_offset=np.mean(np.abs(hip_mid[:, 0] - an_mid[:, 0])) / L,
             com_offset_std=np.std(hip_mid[:, 0] - an_mid[:, 0]) / L,
             hip_sway=hip_mid[:, 0].std() / L, base_width=base.mean() / L)

    # footwork
    la, ra = g(L_AN), g(R_AN)
    tail = slice(int(T * 0.8), T)
    f.update(ankle_path=(path_length(la) + path_length(ra)) / L,
             step_width_range=np.ptp(base) / L,
             foot_shift=max(np.linalg.norm(la[-1] - la[0]), np.linalg.norm(ra[-1] - ra[0])) / L,
             foot_stability_end=(la[tail].std(axis=0).mean() + ra[tail].std(axis=0).mean()) / (2 * L),
             jump_height=max(0.0, np.median(an_mid[:, 1]) - an_mid[:, 1].min()) / L)

    # wrist / follow-through (active wrist = the one that travels more)
    lw, rw = g(L_WR), g(R_WR)
    wr = lw if path_length(lw) > path_length(rw) else rw
    speed = np.linalg.norm(np.diff(wr, axis=0), axis=1) / L if T > 1 else np.zeros(1)
    f.update(wrist_travel=path_length(wr) / L, wrist_range_x=np.ptp(wr[:, 0]) / L,
             wrist_range_y=np.ptp(wr[:, 1]) / L,
             wrist_height_end=(sh_mid[-1, 1] - wr[-1, 1]) / L,
             wrist_height_max=(sh_mid[:, 1] - wr[:, 1]).max() / L,
             wrist_speed_peak=speed.max(),
             wrist_speed_ratio=speed.max() / (speed.mean() + EPS),
             wrist_jerk=np.std(np.diff(speed)) / (speed.mean() + EPS) if len(speed) > 2 else 0.0)

    # video modality: how much the picture changes (motion energy)
    m = np.asarray(motion if motion is not None else [0.0], dtype=np.float64)
    f.update(motion_mean=m.mean(), motion_max=m.max(), motion_std=m.std())

    return {k: float(v) for k, v in f.items()}


FEATURE_NAMES = None


def feature_names():
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        t = np.linspace(0, 1, 40)
        dummy = np.zeros((40, 33, 4)); dummy[..., 3] = 1
        rng = np.random.default_rng(0)
        dummy[..., :2] = rng.random((40, 33, 2)) + t[:, None, None] * 0.1
        FEATURE_NAMES = list(extract_features(dummy).keys())
    return FEATURE_NAMES


# ---------- sequence for the deep-learning model ----------
def normalized_joints(kp):
    """(T,13,2): joints centred on the first-frame hip centre and scaled by torso length."""
    xy, _ = clean_keypoints(kp)
    L = torso_length(xy)
    origin = ((xy[0, L_HIP] + xy[0, R_HIP]) / 2)
    return ((xy[:, SELECTED] - origin) / L).astype(np.float32)


def resample(seq, length):
    """Linearly resample (T,J,2) to (length,J,2)."""
    T = len(seq)
    src, dst = np.linspace(0, T - 1, T), np.linspace(0, T - 1, length)
    flat = seq.reshape(T, -1)
    out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape(length, *seq.shape[1:]).astype(np.float32)


def mirror(seq):
    """Flip left <-> right (makes a right-handed batter look left-handed). Used for augmentation."""
    out = seq[:, MIRROR_PERM].copy()
    out[..., 0] *= -1
    return out


def to_model_input(seq):
    """(L,13,2) -> (L,52): positions + velocities."""
    vel = np.vstack([np.zeros((1, *seq.shape[1:]), dtype=seq.dtype), np.diff(seq, axis=0)])
    return np.concatenate([seq.reshape(len(seq), -1), vel.reshape(len(seq), -1)], axis=1)


def build_sequence(kp, length):
    return to_model_input(resample(normalized_joints(kp), length))
