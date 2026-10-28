"""SORT-style multi-object tracker: constant-velocity Kalman filter + Hungarian IoU matching (NumPy/SciPy)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in x1,y1,x2,y2 format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def box_to_z(box):
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def x_to_box(x):
    s, r = max(float(x[2, 0]), 1e-6), max(float(x[3, 0]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0, 0] - w / 2, x[1, 0] - h / 2, x[0, 0] + w / 2, x[1, 0] + h / 2])


class KalmanBoxTrack:
    """State: [cx, cy, area, aspect, vx, vy, v_area]. Measurement: [cx, cy, area, aspect]."""
    _next_id = 1

    def __init__(self, box, cls_id: int, score: float):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.diag([1.0, 1.0, 10.0, 0.01])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 1e-4])
        self.x = np.zeros((7, 1))
        self.x[:4] = box_to_z(box)
        self.id = KalmanBoxTrack._next_id
        KalmanBoxTrack._next_id += 1
        self.cls_id, self.score = int(cls_id), float(score)
        self.hits, self.age, self.time_since_update = 1, 0, 0

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:  # keep area positive
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box, cls_id, score):
        z = box_to_z(box)
        y = z - self.H @ self.x                        # innovation
        S = self.H @ self.P @ self.H.T + self.R        # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.cls_id, self.score = int(cls_id), float(score)
        self.hits += 1
        self.time_since_update = 0

    @property
    def box(self):
        return x_to_box(self.x)


class SORTTracker:
    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.tracks: list[KalmanBoxTrack] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """dets: (N,6) x1,y1,x2,y2,score,cls. Returns (M,7): x1,y1,x2,y2,track_id,cls,score."""
        self.frame_count += 1
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        valid = np.isfinite(preds).all(axis=1)
        self.tracks = [t for t, v in zip(self.tracks, valid) if v]
        preds = preds[valid]

        iou = iou_matrix(preds, dets[:, :4]) if len(dets) else np.zeros((len(preds), 0))
        # only allow matches between the same class
        for i, t in enumerate(self.tracks):
            for j in range(len(dets)):
                if t.cls_id != int(dets[j, 5]):
                    iou[i, j] = 0.0

        matched_t, matched_d = set(), set()
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    self.tracks[r].update(dets[c, :4], dets[c, 5], dets[c, 4])
                    matched_t.add(r)
                    matched_d.add(c)

        for j in range(len(dets)):
            if j not in matched_d:
                self.tracks.append(KalmanBoxTrack(dets[j, :4], dets[j, 5], dets[j, 4]))

        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                b = t.box
                out.append([*b, t.id, t.cls_id, t.score])
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(out, dtype=np.float32).reshape(-1, 7)

    @staticmethod
    def reset_ids():
        KalmanBoxTrack._next_id = 1
