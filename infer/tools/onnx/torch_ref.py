#!/usr/bin/env python3
"""Torch CPU reference for the YOLO verification.

    python torch_ref.py 160

Runs the official yolov8n.pt through torch's eager mode on the same fixed
input as make_refs.py and writes /tmp/yolo<N>_torch.npy.  The exported ONNX
graph turns the model's SiLU into Sigmoid+Mul, so this is a genuinely
independent path (torch kernels vs the numpy evaluator vs the QPUs).
"""
import os
import sys

import numpy as np


def main(argv):
    size = int(argv[0]) if argv else 160
    x = np.load("/tmp/yolo%d_in.npy" % size)
    import torch
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultra")
    from ultralytics import YOLO

    m = YOLO("/home/raspi/work/onnx/models/yolov8n.pt").model.float().eval()
    with torch.no_grad():
        y = m(torch.from_numpy(x))
    if isinstance(y, (list, tuple)):
        y = y[0]
    y = y.numpy()
    np.save("/tmp/yolo%d_torch.npy" % size, np.asarray(y, np.float32))
    print("torch %s -> %s  min %.4g max %.4g"
          % (x.shape, y.shape, y.min(), y.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
