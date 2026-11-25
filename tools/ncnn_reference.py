#!/usr/bin/env python3
"""ncnn_reference.py - the same image through ncnn on the CPU, as a baseline.

    ~/venvs/ncnn/bin/python tools/ncnn_reference.py IMAGE NCNN_DIR
        [--imgsz 160] [--threads 2] [--conf 0.25] [--iou 0.45]
        [--classes FILE] [--json]

NCNN_DIR is an Ultralytics `format="ncnn"` export: `model.ncnn.param`,
`model.ncnn.bin`, and the `in0`/`out0` blob names that export uses.  There is no
`.pt` involved: the CPU runs its own graph, so this is an independent reference
for the boxes the QPUs return.

The Python wheel is generic, not the tuned build the project measured earlier:
threads must be tuned by hand.  Measured here at 160 px on this board, one
client quiet: **135 ms at 1 thread, 106 ms at 2, 414 ms at 4** - four threads
oversubscribes the four A53s and is slower than two.  Default is 2.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "infer", "python"))

import ncnn  # noqa: E402
from vc4infer import postprocess as pp  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("ncnn_dir")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--classes")
    ap.add_argument("--names", help="comma-separated class names, as a fallback "
                                    "when the model has no names file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    param = os.path.join(args.ncnn_dir, "model.ncnn.param")
    model = os.path.join(args.ncnn_dir, "model.ncnn.bin")
    for p in (param, model):
        if not os.path.isfile(p):
            sys.exit("no such file: %s" % p)

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = args.threads
    net.load_param(param)
    net.load_model(model)

    img = Image.open(args.image).convert("RGB")
    x, meta = pp.Letterbox(args.imgsz)(img)
    plane = np.ascontiguousarray(np.asarray(x)[0])

    t0 = time.perf_counter()
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(plane).clone())
    _, out = ex.extract("out0")
    ms = (time.perf_counter() - t0) * 1e3
    raw = np.array(out)
    raw = raw.reshape(1, raw.shape[0], raw.shape[1])

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

    out_dets = []
    for b, s, c in zip(boxes, d.scores, d.classes):
        c = int(c)
        out_dets.append({"x1": round(float(b[0]), 1), "y1": round(float(b[1]), 1),
                         "x2": round(float(b[2]), 1), "y2": round(float(b[3]), 1),
                         "score": round(float(s), 3),
                         "class": c, "name": names[c] if c < len(names) else "class %d" % c})

    if args.json:
        print(json.dumps({"backend": "ncnn", "threads": args.threads,
                          "imgsz": args.imgsz, "ms": round(ms, 1),
                          "count": len(out_dets), "detections": out_dets}))
    else:
        print("%s  ncnn, %d thread(s), %d px: %.0f ms, %d detections"
              % (os.path.basename(args.image), args.threads, args.imgsz,
                 ms, len(out_dets)))
        for x in out_dets[:10]:
            print("  %-12s %.2f  [%.0f, %.0f, %.0f, %.0f]"
                  % (x["name"], x["score"], x["x1"], x["y1"], x["x2"], x["y2"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
