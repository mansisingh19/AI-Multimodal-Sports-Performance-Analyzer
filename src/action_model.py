"""Action recognition: pose-keypoint sequence -> shot/stroke type (Deep Learning module).

Model: bidirectional GRU over (time x [positions + velocities]) of 13 body joints.
It uses both spatial information (where the joints are) and temporal information (how they move).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from . import config
from .features import (build_sequence, mirror, normalized_joints, resample, to_model_input)
from .pose import detection_rate, load_pose


class PoseGRU(nn.Module):
    def __init__(self, in_dim=52, hidden=64, n_classes=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Linear(hidden * 2, 64)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        o, _ = self.gru(x)
        e = torch.relu(self.embed(self.drop(o.mean(dim=1))))   # clip embedding (pose modality)
        return self.out(self.drop(e)), e


# ---------- data ----------
def list_clips(sport):
    """[(path, class_index)] for data/keypoints/<sport>/<action>/*.npz"""
    actions = config.SPORTS[sport]["actions"]
    items = []
    for ci, action in enumerate(actions):
        for p in sorted((config.KP_DIR / sport / action).glob("*.npz")):
            if detection_rate(load_pose(p)[0]) >= config.MIN_DETECTION_ACTION:
                items.append((p, ci))
    return items


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, items, augment=False):
        self.augment = augment
        self.labels = [c for _, c in items]
        self.seqs = [normalized_joints(load_pose(p)[0]) for p, _ in items]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        if self.augment:
            T = len(s)
            keep = max(8, int(T * np.random.uniform(0.8, 1.0)))       # random crop in time
            start = np.random.randint(0, T - keep + 1)
            s = s[start:start + keep]
            if np.random.rand() < 0.5:                                  # left/right handed
                s = mirror(s)
            s = s + np.random.normal(0, 0.02, s.shape).astype(np.float32)
        x = to_model_input(resample(s, config.SEQ_LEN))
        return torch.tensor(x, dtype=torch.float32), self.labels[i]


# ---------- training ----------
def train_action_model(sport, epochs=150, patience=25, batch_size=16, lr=1e-3, seed=42, out_dir=None):
    torch.manual_seed(seed); np.random.seed(seed)
    actions = config.SPORTS[sport]["actions"]
    items = list_clips(sport)
    if len(items) < 3 * len(actions):
        raise RuntimeError(f"Need more clips: found {len(items)} in data/keypoints/{sport}. "
                           "Run scripts/extract_dataset.py first (aim for 50+ clips per action).")
    labels = [c for _, c in items]
    tr, rest = train_test_split(items, test_size=0.30, stratify=labels, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=[c for _, c in rest], random_state=seed)

    loaders = {
        "train": torch.utils.data.DataLoader(ClipDataset(tr, True), batch_size=batch_size, shuffle=True),
        "val": torch.utils.data.DataLoader(ClipDataset(va), batch_size=batch_size),
        "test": torch.utils.data.DataLoader(ClipDataset(te), batch_size=batch_size),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PoseGRU(n_classes=len(actions)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def run(loader):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                ps += model(x.to(device))[0].argmax(1).cpu().tolist(); ys += y.tolist()
        return ys, ps

    best, best_state, wait, history = -1, None, 0, []
    for ep in range(epochs):
        model.train(); total = 0
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device))[0], y.to(device))
            loss.backward(); opt.step(); total += loss.item()
        val_acc = accuracy_score(*run(loaders["val"]))
        history.append({"epoch": ep + 1, "train_loss": total / len(loaders["train"]), "val_acc": val_acc})
        if val_acc > best:
            best, wait = val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)

    ys, ps = run(loaders["test"])
    p, r, f1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    metrics = {"sport": sport, "classes": actions, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "best_val_accuracy": best, "test_accuracy": accuracy_score(ys, ps),
               "test_precision_macro": p, "test_recall_macro": r, "test_f1_macro": f1,
               "confusion_matrix": confusion_matrix(ys, ps, labels=list(range(len(actions)))).tolist(),
               "epochs_run": len(history)}

    out = Path(out_dir or config.MODEL_DIR / sport); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "classes": actions, "seq_len": config.SEQ_LEN}, out / "action_model.pt")
    (out / "action_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "action_history.json").write_text(json.dumps(history, indent=2))
    _plot_confusion(metrics, out / "confusion_matrix.png")
    return metrics


def _plot_confusion(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm, names = np.array(metrics["confusion_matrix"]), metrics["classes"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------- inference ----------
class ActionRecognizer:
    def __init__(self, sport, model_dir=None):
        path = Path(model_dir or config.MODEL_DIR / sport) / "action_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"No trained action model for '{sport}'. Run: python scripts/train_action.py --sport {sport}")
        ck = torch.load(path, map_location="cpu")
        self.classes, self.seq_len = ck["classes"], ck["seq_len"]
        self.model = PoseGRU(n_classes=len(self.classes))
        self.model.load_state_dict(ck["state"]); self.model.eval()

    def predict(self, kp):
        """Returns (label, confidence 0-1, {class: prob}, embedding vector)."""
        x = torch.tensor(build_sequence(kp, self.seq_len), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, emb = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        i = int(probs.argmax())
        return self.classes[i], float(probs[i]), dict(zip(self.classes, map(float, probs))), emb[0].numpy()
