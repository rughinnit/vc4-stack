#!/usr/bin/env python3
"""YOLO through the thin adapter: numpy boxes, replaceable post-processing.

    python3 infer/examples/run_yolo.py --model yolov8n.onnx --image bus.jpg

The adapter only loads/letterboxes the image and calls
``postprocess.decode_yolo``; everything is numpy and can be replaced.
"""
import argparse
import os
import sys

import numpy as np

try:
    import vc4infer
except ImportError:                     # in-tree checkout: package in infer/python
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    import vc4infer

from vc4infer.postprocess import Detections, decode_yolo


def only_cars(raw, imgsz, conf, iou, max_det=300, agnostic=False):
    """A bring-your-own post-process: decode, then keep one class."""
    d = decode_yolo(raw, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                    agnostic=agnostic)
    keep = d.classes == 5                       # COCO 'bus'
    return Detections(d.boxes[keep], d.scores[keep], d.classes[keep],
                      {**d.meta, "filter": "class == bus"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolov8n.onnx")
    ap.add_argument("--image", default="bus.jpg")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default="bus_vc4infer.jpg")
    a = ap.parse_args()

    print("vc4infer %s, devices %r" % (vc4infer.__version__,
                                       vc4infer.devices()))
    with vc4infer.Engine(a.model) as eng:
        det = vc4infer.yolo.Detector(eng)
        r = det.predict(a.image, imgsz=a.imgsz, conf=a.conf, iou=a.iou)
        print(r, "orig_shape", r.orig_shape)
        print("speed ms:", {k: round(v, 1) for k, v in r.speed_ms.items()})
        for d in r.to_dict(det.names)["detections"]:
            print("  %-12s %.3f  %s" % (d["class_name"], d["confidence"],
                                        [round(v, 1) for v in d["box"]]))

        # 1) thresholds are arguments
        strict = det.predict(a.image, conf=0.60, iou=0.30)
        print("conf 0.60 / iou 0.30 -> %d detections %s"
              % (len(strict), [round(float(c), 3) for c in strict.scores]))

        # 2) replace the post-processing entirely
        cars = det.predict(a.image, postprocess=only_cars)
        print("custom postprocess (bus only) -> %d %s"
              % (len(cars), np.asarray(cars.scores).round(3).tolist()))

        # 3) bypass the adapter: raw engine output + your own numpy
        raw = eng.run({"images": np.zeros((1, 3, a.imgsz, a.imgsz),
                                          np.float32)})["output0"]
        print("raw head:", raw.shape)

        r.plot().save(a.out, quality=92)
        print("saved", a.out)


if __name__ == "__main__":
    main()
