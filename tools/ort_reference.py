#!/usr/bin/env python3
"""ort_reference.py - the same graph through ONNX Runtime on the CPU.

    ~/scratch/ortvenv/bin/python tools/ort_reference.py MODEL.onnx IMAGE
        [--imgsz 160] [--conf 0.25] [--iou 0.45] [--classes FILE] [--json]

This is the honest CPU baseline: ncnn in the project's comparison runs a
*different* export of one model, while this runs the exact graph the QPU runs -
the ONNX an artifact was built from, or the graph embedded inside the artifact -
with the same letterbox and the same post-processing.  If the two disagree, the
engine disagrees with itself, not with another engine.

CPU threads default to 4: the board has four A53s and ONNX Runtime scales with
them, unlike the generic ncnn wheel.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "infer", "python"))

from vc4infer import postprocess as pp  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--classes")
    ap.add_argument("--names", help="comma-separated class names, as a fallback "
                                    "when the model has no names file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import onnxruntime as rt

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    sess = rt.InferenceSession(args.model, sess_options=opts,
                               providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    img = Image.open(args.image).convert("RGB")
    x, meta = pp.Letterbox(args.imgsz)(img)
    feed = np.asarray(x, dtype=np.float32)

    import time
    t0 = time.perf_counter()
    raw = sess.run([out_name], {in_name: feed})[0]
    ms = (time.perf_counter() - t0) * 1e3

    d = pp.decode_yolo(raw, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                       max_det=300)
    boxes = pp.Letterbox.to_original(d.boxes, meta)

    names = None
    if getattr(args, "names", None):
        names = [s.strip() for s in args.names.split(",") if s.strip()]
    elif args.classes:
        with open(args.classes, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
    else:
        names = list(pp.COCO80)

    dets = []
    for b, s, c in zip(boxes, d.scores, d.classes):
        c = int(c)
        dets.append({"x1": round(float(b[0]), 1), "y1": round(float(b[1]), 1),
                     "x2": round(float(b[2]), 1), "y2": round(float(b[3]), 1),
                     "score": round(float(s), 3), "class": c,
                     "name": names[c] if c < len(names) else "class %d" % c})

    if args.json:
        print(json.dumps({"backend": "cpu (onnxruntime)", "threads": args.threads,
                          "imgsz": args.imgsz, "ms": round(ms, 1),
                          "count": len(dets), "detections": dets}))
    else:
        print("%s  ONNX Runtime, %d threads, %d px: %.0f ms, %d detections"
              % (os.path.basename(args.image), args.threads, args.imgsz, ms, len(dets)))
        for x in dets[:10]:
            print("  %-12s %.2f  [%.0f, %.0f, %.0f, %.0f]"
                  % (x["name"], x["score"], x["x1"], x["y1"], x["x2"], x["y2"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
