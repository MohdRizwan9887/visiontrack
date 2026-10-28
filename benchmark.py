"""Benchmark PyTorch vs ONNX Runtime FP32 vs INT8 (latency/FPS) and check detection agreement.

Usage:
    python benchmark.py --images data/calib --pt models/yolov8n.pt \
        --onnx models/yolov8n.onnx --int8 models/yolov8n_int8.onnx
"""
import argparse
import glob
import time

import cv2
import numpy as np

from visiontrack.detector import ONNXDetector
from visiontrack.tracker import iou_matrix


def stats(ms):
    ms = np.array(ms)
    return f"mean={ms.mean():6.1f} ms  p50={np.percentile(ms, 50):6.1f}  p95={np.percentile(ms, 95):6.1f}  FPS={1000/ms.mean():5.1f}"


def agreement(ref, other, thr=0.5):
    """Fraction of reference detections matched (IoU>thr, same class) by another model."""
    if len(ref) == 0:
        return float("nan")
    iou = iou_matrix(ref[:, :4], other[:, :4]) if len(other) else np.zeros((len(ref), 0))
    hit = 0
    for i in range(len(ref)):
        if iou.shape[1] and any(iou[i, j] > thr and ref[i, 5] == other[j, 5] for j in range(iou.shape[1])):
            hit += 1
    return hit / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/calib")
    ap.add_argument("--pt", default="models/yolov8n.pt")
    ap.add_argument("--onnx", default="models/yolov8n.onnx")
    ap.add_argument("--int8", default="models/yolov8n_int8.onnx")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images + "/*.jpg") + glob.glob(args.images + "/*.png"))
    frames = [cv2.imread(p) for p in paths]
    assert frames, "no images found"
    frames = [frames[i % len(frames)] for i in range(args.runs)]

    results = {}
    # --- ONNX models ---
    detectors = {"ONNX FP32": ONNXDetector(args.onnx)}
    try:
        detectors["ONNX INT8"] = ONNXDetector(args.int8)
    except Exception as e:  # noqa
        print("INT8 model unavailable:", e)
    for name, det in detectors.items():
        for f in frames[:args.warmup]:
            det(f)
        det.timings.__init__()
        outs = [det(f) for f in frames]
        total = [a + b + c for a, b, c in zip(det.timings.pre_ms, det.timings.infer_ms, det.timings.post_ms)]
        print(f"{name:10s} end-to-end: {stats(total)}   (inference only: {np.mean(det.timings.infer_ms):.1f} ms)")
        results[name] = outs

    # --- PyTorch baseline ---
    from ultralytics import YOLO
    pt = YOLO(args.pt)
    for f in frames[:args.warmup]:
        pt.predict(f, imgsz=640, conf=0.25, verbose=False, device="cpu")
    ts, outs = [], []
    for f in frames:
        t0 = time.perf_counter()
        r = pt.predict(f, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
        ts.append((time.perf_counter() - t0) * 1e3)
        b = r.boxes
        outs.append(np.concatenate([b.xyxy.numpy(), b.conf.numpy()[:, None], b.cls.numpy()[:, None]], 1)
                    if len(b) else np.zeros((0, 6)))
    print(f"{'PyTorch':10s} end-to-end: {stats(ts)}")
    results["PyTorch"] = outs

    ref = results["PyTorch"]
    for name in ("ONNX FP32", "ONNX INT8"):
        if name in results:
            ag = np.nanmean([agreement(r, o) for r, o in zip(ref, results[name])])
            print(f"{name:10s} detection agreement with PyTorch (IoU>0.5): {ag*100:.1f}%")


if __name__ == "__main__":
    main()
