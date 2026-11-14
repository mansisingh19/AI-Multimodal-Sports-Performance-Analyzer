"""Video -> pose keypoints (Computer Vision module).

Output of extract_pose():
    keypoints : (T, 33, 4)  x, y, z, visibility  (x and y are divided by the frame HEIGHT,
                            so angles are not distorted by the video aspect ratio)
    motion    : (T,)        mean frame-difference energy (the "video" modality)
    meta      : dict        fps, width, height
"""
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    _mp_pose = mp.solutions.pose
    POSE_CONNECTIONS = _mp_pose.POSE_CONNECTIONS
except Exception:  # lets the rest of the project be imported without mediapipe
    mp, _mp_pose, POSE_CONNECTIONS = None, None, []


def _resize(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))


class PlayerTracker:
    """Optional YOLO person detector. Follows the same athlete across frames by
    picking the box that overlaps most with the previous box (needs `ultralytics`)."""

    def __init__(self, weights="yolov8n.pt", imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.prev = None

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
        return inter / (area(a) + area(b) - inter + 1e-9)

    @classmethod
    def _pick(cls, boxes, prev, width):
        """Choose which detected person is the athlete."""
        if prev is None:   # first frame: big person, close to the centre of the picture
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            return int(np.argmax(area * (1 - 0.5 * np.abs(cx - width / 2) / (width / 2))))
        ious = [cls._iou(prev, b) for b in boxes]
        if max(ious) > 0.05:
            return int(np.argmax(ious))
        pc = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        return int(np.argmin(np.linalg.norm(centres - pc, axis=1)))

    def box(self, frame):
        res = self.model(frame, classes=[0], imgsz=self.imgsz, verbose=False)[0]
        if len(res.boxes) == 0:
            return self.prev
        boxes = res.boxes.xyxy.cpu().numpy()
        new = boxes[self._pick(boxes, self.prev, frame.shape[1])]
        self.prev = new if self.prev is None else 0.7 * self.prev + 0.3 * new  # smooth the box
        return self.prev


def extract_pose(video_path, use_tracker=False, max_width=640, max_frames=600, detect_every=1,
                 margin=0.30, static=False, imgsz=640, min_det=0.5):
    """Run pose estimation on every frame (or every k-th frame for very long videos).

    Options that only matter when use_tracker=True (YOLO finds the player and the frame is cropped):
      detect_every : run YOLO every k-th frame (reuse the last box in between)
      margin       : extra space around the player box (0.30 = 30 %)
      static       : treat every crop as a fresh image for MediaPipe (instead of frame-to-frame tracking)
      imgsz        : YOLO input size
    """
    if _mp_pose is None:
        raise ImportError("This project needs the classic MediaPipe Pose API. Run: pip install mediapipe==0.10.14 "
                          "(newer versions removed mp.solutions)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    stride = max(1, int(np.ceil(total / max_frames)))
    tracker = PlayerTracker(imgsz=imgsz) if use_tracker else None

    kps, motion, prev_gray, idx, n_done, H, W = [], [], None, -1, 0, None, None
    with _mp_pose.Pose(static_image_mode=static, model_complexity=1,
                       min_detection_confidence=min_det, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            frame = _resize(frame, max_width)
            H, W = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion.append(0.0 if prev_gray is None
                          else float(np.mean(cv2.absdiff(gray, prev_gray))) / 255.0)
            prev_gray = gray

            x0, y0, x1, y1 = 0, 0, W, H
            if tracker is not None:
                b = tracker.box(frame) if n_done % detect_every == 0 else tracker.prev
                n_done += 1
                if b is not None:
                    mx, my = margin * (b[2] - b[0]), margin * (b[3] - b[1])
                    x0, y0 = int(max(0, b[0] - mx)), int(max(0, b[1] - my))
                    x1, y1 = int(min(W, b[2] + mx)), int(min(H, b[3] + my))
            crop = frame[y0:y1, x0:x1]
            result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks:
                cw, ch = x1 - x0, y1 - y0
                pts = [[(x0 + lm.x * cw) / H, (y0 + lm.y * ch) / H, lm.z, lm.visibility]
                       for lm in result.pose_landmarks.landmark]
            else:
                pts = [[0, 0, 0, 0]] * 33
            kps.append(pts)
    cap.release()

    if not kps:
        raise ValueError("No frames could be read from the video.")
    meta = {"fps": float(fps / stride), "width": W, "height": H}
    return np.array(kps, dtype=np.float32), np.array(motion, dtype=np.float32), meta


def detection_rate(kp):
    """Fraction of frames in which a person was found."""
    return float((np.asarray(kp)[..., 3].max(axis=1) > 0.05).mean())


def save_pose(path, keypoints, motion, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keypoints=keypoints, motion=motion,
                        fps=meta["fps"], width=meta["width"], height=meta["height"])


def load_pose(path):
    d = np.load(path)
    return d["keypoints"], d["motion"]


def render_pose_frames(video_path, keypoints, n=6, max_width=640):
    """Return n RGB frames (evenly spaced) with the skeleton drawn on them."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(np.ceil(total / 600)))          # same stride as extract_pose
    picks = np.linspace(0, len(keypoints) - 1, n).astype(int)
    out = []
    for k in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k) * stride)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _resize(frame, max_width)
        H, W = frame.shape[:2]
        pts = [(int(p[0] * H), int(p[1] * H), p[3]) for p in keypoints[k]]
        for a, b in POSE_CONNECTIONS:
            if pts[a][2] > 0.3 and pts[b][2] > 0.3:
                cv2.line(frame, pts[a][:2], pts[b][:2], (0, 255, 0), 2)
        for x, y, v in pts:
            if v > 0.3:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out
