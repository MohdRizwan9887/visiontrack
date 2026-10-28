"""Export YOLOv8 to ONNX and produce a statically quantized INT8 (QDQ) model.

Usage:
    python export_onnx.py --weights models/yolov8n.pt --calib-dir data/calib --imgsz 640
"""
import argparse
import glob
import os

import cv2
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from visiontrack.detector import letterbox


class FrameCalibrationReader(CalibrationDataReader):
    def __init__(self, paths, input_name, size):
        self.paths, self.input_name, self.size, self.i = paths, input_name, size, 0

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        img = cv2.imread(self.paths[self.i])
        self.i += 1
        img, _, _ = letterbox(img, self.size)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True)
        return {self.input_name: blob.astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-dir", default="data/calib", help="folder of representative frames (jpg/png)")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    onnx_path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=13)
    print("FP32 ONNX:", onnx_path)

    if args.skip_int8:
        return
    paths = sorted(glob.glob(os.path.join(args.calib_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.calib_dir, "*.png")))
    if not paths:
        raise SystemExit(f"No calibration images in {args.calib_dir}; add frames or use --skip-int8")

    prep = onnx_path.replace(".onnx", "_prep.onnx")
    quant_pre_process(onnx_path, prep)
    int8_path = onnx_path.replace(".onnx", "_int8.onnx")
    reader = FrameCalibrationReader(paths, "images", args.imgsz)
    # Keep the detection head's decode stage (DFL, box decoding, sigmoid, final concat) in FP32:
    # box coordinates (0-640) and class scores (0-1) share one output tensor, so INT8 there destroys scores.
    import onnx
    g = onnx.load(prep).graph
    head = max((n.name.split("/")[1] for n in g.node if n.name.startswith("/model.")),
               key=lambda m: int(m.split(".")[1]))
    exclude = [n.name for n in g.node
               if n.name.startswith(f"/{head}/") and not (n.op_type == "Conv" and "/dfl/" not in n.name)]
    print(f"Excluding {len(exclude)} detection-head decode nodes from quantization")
    quantize_static(prep, int8_path, reader, quant_format=QuantFormat.QDQ, nodes_to_exclude=exclude,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax)
    os.remove(prep)
    print("INT8 ONNX:", int8_path, f"({os.path.getsize(int8_path)/1e6:.1f} MB vs "
          f"{os.path.getsize(onnx_path)/1e6:.1f} MB FP32)")


if __name__ == "__main__":
    main()
