#!/usr/bin/env python3
"""Run the vc4det student on the board and decode its 8-channel head.

    vc4infer build vc4det-n3_96.onnx -o vc4det-n3_96.vc4x
    python3 infer/tools/det/student_detect.py vc4det-n3_96.vc4x \
        --input models/BloodImage_00007.jpg --classes models/bccd_names.txt

`vc4infer run` decodes a YOLO head of the ``(1,4+nc,A)`` shape and refuses
anything else, so this head -- ``(1,8,12,12)``, one objectness plus four box
terms plus three class scores per cell -- gets its own decode, from
:mod:`det_post`, which needs numpy only.  No torch on this board is a feature.

Prints one JSON document per image: detections in original image pixels,
scores, class names, and the engine's own wall time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import det_post as P  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help=".vc4x or .onnx")
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--classes", default=None, help="one class name per line")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--out", default=None, help="annotated image for --input")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from vc4infer import Engine
    from vc4infer.postprocess import Letterbox, draw, Detections

    names = ["class%d" % i for i in range(args.nc)]
    if args.classes:
        with open(args.classes) as f:
            names = [ln.strip() for ln in f if ln.strip()]

    docs = []
    with Engine(args.model) as eng:
        spec = eng.inputs[0]
        imgsz = args.imgsz or int(spec.shape[2])
        for path in args.input:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            x, meta = Letterbox(imgsz)(img)
            t0 = time.perf_counter()
            out = eng.run({spec.name: x})
            wall = (time.perf_counter() - t0) * 1e3
            raw = out[eng.outputs[0].name]
            boxes, scores, cls = P.decode(raw, nc=args.nc, imgsz=imgsz)
            boxes, scores, cls = P.postprocess(boxes[0], scores[0], cls[0],
                                               conf=args.conf, iou=args.iou,
                                               max_det=args.max_det, nc=args.nc)
            boxes_o = Letterbox.to_original(boxes, meta)
            det = [{"class": names[int(c)] if int(c) < len(names) else int(c),
                    "conf": round(float(s), 4),
                    "box": [round(float(v), 1) for v in b]}
                   for b, s, c in zip(boxes_o, scores, cls)]
            counts = {}
            for d in det:
                counts[d["class"]] = counts.get(d["class"], 0) + 1
            doc = {"input": os.path.abspath(path), "model": os.path.abspath(
                args.model), "imgsz": imgsz, "wall_ms": round(wall, 2),
                "launches": int(eng.stats.get("launches", -1)),
                "gpu_ms": round(float(eng.stats.get("gpu_ms", -1)), 2),
                "n": len(det), "counts": counts, "detections": det}
            docs.append(doc)
            if args.out:
                try:
                    draw(img, Detections(boxes_o, scores, cls), names).save(
                        args.out, quality=92)
                except Exception as exc:            # drawing is optional
                    doc["draw_error"] = str(exc)
            if not args.json:
                print("%s  %d detections  %.1f ms  %s" % (
                    os.path.basename(path), len(det), wall, counts))
                for d in det:
                    print("  %-10s %.3f  %s" % (d["class"], d["conf"], d["box"]))
    print(json.dumps(docs, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
