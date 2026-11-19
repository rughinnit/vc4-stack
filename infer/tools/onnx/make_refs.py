#!/usr/bin/env python3
"""Make the fixed input + the two CPU references for the YOLO benchmarks.

The input is the same deterministic tensor `onnxrun.py` uses when no --input
is given (default_rng(0) standard normal), so the numbers are reproducible
after a reboot:

    python make_refs.py 160
    python make_refs.py 320
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref_onnx                                          # noqa: E402

MODELS = os.environ.get("YOLO_MODELS", "/home/raspi/work/onnx/models")


def main(argv):
    size = int(argv[0]) if argv else 160
    model = os.path.join(MODELS, "yolov8n_%d.onnx" % size)
    x = np.random.default_rng(0).standard_normal((1, 3, size, size))
    x = x.astype(np.float32)
    np.save("/tmp/yolo%d_in.npy" % size, x)
    out, = ref_onnx.run_model(model, {"images": x})
    out = np.asarray(out, np.float32)
    np.save("/tmp/yolo%d_ref.npy" % size, out)
    print("%s: in %s -> ref %s  min %.4g max %.4g"
          % (model, x.shape, out.shape, out.min(), out.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
