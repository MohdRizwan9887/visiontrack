"""YOLOv8 object detector running on ONNX Runtime with NumPy/OpenCV pre- and post-processing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Timings:
    pre_ms: list = field(default_factory=list)
    infer_ms: list = field(default_factory=list)
    post_ms: list = field(default_factory=list)


def letterbox(img: np.ndarray, size: int = 640, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to a square. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


class ONNXDetector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45,
                 providers: list[str] | None = None, intra_threads: int = 0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            so.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(
            model_path, so, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 640
        self.conf, self.iou = conf, iou
        self.timings = Timings()

    def preprocess(self, frame: np.ndarray):
        img, r, pad = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True)
        return blob.astype(np.float32), r, pad

    def postprocess(self, out: np.ndarray, r: float, pad, orig_shape):
        # YOLOv8 output: (1, 4 + num_classes, num_anchors) -> (num_anchors, 4 + nc)
        preds = out[0].T
        boxes_cxcywh = preds[:, :4]
        scores_all = preds[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(cls_ids)), cls_ids]
        keep = scores >= self.conf
        boxes_cxcywh, scores, cls_ids = boxes_cxcywh[keep], scores[keep], cls_ids[keep]
        if len(scores) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        x, y, w, h = boxes_cxcywh.T
        x1 = (x - w / 2 - pad[0]) / r
        y1 = (y - h / 2 - pad[1]) / r
        x2 = (x + w / 2 - pad[0]) / r
        y2 = (y + h / 2 - pad[1]) / r
        H, W = orig_shape[:2]
        x1, x2 = np.clip(x1, 0, W - 1), np.clip(x2, 0, W - 1)
        y1, y2 = np.clip(y1, 0, H - 1), np.clip(y2, 0, H - 1)

        # class-aware NMS via offset trick
        offsets = cls_ids * 4096.0
        nms_boxes = np.stack([x1 + offsets, y1 + offsets, x2 - x1, y2 - y1], axis=1)
        idx = cv2.dnn.NMSBoxes(nms_boxes.tolist(), scores.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        dets = np.stack([x1, y1, x2, y2, scores, cls_ids], axis=1)[idx]
        return dets.astype(np.float32)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Returns detections as an (N, 6) array: x1, y1, x2, y2, score, class_id."""
        t0 = time.perf_counter()
        blob, r, pad = self.preprocess(frame)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]
        t2 = time.perf_counter()
        dets = self.postprocess(out, r, pad, frame.shape)
        t3 = time.perf_counter()
        self.timings.pre_ms.append((t1 - t0) * 1e3)
        self.timings.infer_ms.append((t2 - t1) * 1e3)
        self.timings.post_ms.append((t3 - t2) * 1e3)
        return dets
