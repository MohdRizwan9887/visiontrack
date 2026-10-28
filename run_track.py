"""Run detection + multi-object tracking on a video file or webcam and write an annotated video.

Usage:
    python run_track.py --model models/yolov8n.onnx --source input.mp4 --output out.mp4
"""
import argparse
import time

import cv2
import numpy as np

from visiontrack import COCO_NAMES, ONNXDetector, SORTTracker


def color_for(i: int):
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n.onnx")
    ap.add_argument("--source", required=True, help="video path or webcam index")
    ap.add_argument("--output", default="out.mp4")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="COCO class ids to keep, e.g. 0 2")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    det = ONNXDetector(args.model, conf=args.conf)
    trk = SORTTracker()
    seen_ids, n, t_start = set(), 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        dets = det(frame)
        if args.classes is not None and len(dets):
            dets = dets[np.isin(dets[:, 5].astype(int), args.classes)]
        tracks = trk.update(dets)
        for x1, y1, x2, y2, tid, cid, sc in tracks:
            seen_ids.add(int(tid))
            col = color_for(int(tid))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(frame, f"#{int(tid)} {COCO_NAMES[int(cid)]} {sc:.2f}", (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    dt = time.perf_counter() - t_start
    print(f"frames={n} unique_track_ids={len(seen_ids)} end_to_end_fps={n/dt:.1f}")
    for name, v in (("pre", det.timings.pre_ms), ("infer", det.timings.infer_ms), ("post", det.timings.post_ms)):
        if v:
            print(f"  {name:5s} mean={np.mean(v):.2f} ms  p95={np.percentile(v, 95):.2f} ms")


if __name__ == "__main__":
    main()
