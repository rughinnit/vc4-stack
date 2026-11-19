#!/usr/bin/env python3
"""Export a real pretrained YOLOv8n to ONNX with ultralytics.

    YOLO_IMGSZ=640 python export_yolo.py

The weights (yolov8n.pt, ~6 MB) are the official Ultralytics COCO checkpoint;
ultralytics downloads them once.  Nothing here fabricates weights or results.

This board's torchvision 0.29.2 wheel registers no ops against the venv's
custom torch 2.15.0+cpu build, so importing it fails.  A plain detection ONNX
export uses no torchvision code, so when the real import fails we swap in the
minimal name-only shim in tvshim/.  It fails loudly if a torchvision transform
or NMS is ever actually called.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


def ensure_torchvision():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        torchvision.ops.nms  # force the C++ registration
        return "real torchvision %s" % torchvision.__version__
    except Exception as e:
        print("torchvision unusable (%s: %s)" % (type(e).__name__, e))
        for k in list(sys.modules):
            if k == "torchvision" or k.startswith("torchvision."):
                del sys.modules[k]
        sys.path.insert(0, os.path.join(HERE, "tvshim"))
        print("using the pure-python torchvision shim (ONNX export uses no "
              "torchvision code)")
        return "tvshim"


def main():
    tv = ensure_torchvision()
    from ultralytics import YOLO

    imgsz = int(os.environ.get("YOLO_IMGSZ", "640"))
    os.makedirs(MODELS, exist_ok=True)
    os.chdir(MODELS)
    model = YOLO(os.environ.get("YOLO_WEIGHTS", "yolov8n.pt"))
    path = model.export(format="onnx", imgsz=imgsz, opset=13,
                        simplify=False, dynamic=False, device="cpu")
    dst = os.path.join(MODELS, "yolov8n_%d.onnx" % imgsz)
    if os.path.abspath(path) != os.path.abspath(dst):
        shutil.move(path, dst)
    print("torchvision: %s" % tv)
    print("exported %s (%.1f KiB)" % (dst, os.path.getsize(dst) / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
