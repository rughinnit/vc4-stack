#!/usr/bin/env python3
"""tile_detect.py - count cells in a large micrograph by tiling the frame.

    tools/tile_detect.py MODEL IMAGE [--tile 640] [--overlap 0.25]
                         [--conf 0.25] [--iou 0.45] [--classes FILE] [--json]

A 1600-pixel field shrunk into one 320 input makes a red cell 5x smaller than the
detector was trained on, and it misses.  Tiling keeps the magnification: each
`--tile` window of the ORIGINAL image is resized to the model's input, detected,
and its boxes are mapped back, so a smaller `--tile` means more magnification for
the model.

      tile = 2 x model input   cells twice as large as the whole-frame run
      tile = model input       cells at their original size, no shrink at all

Smaller tiles are not free: a window that holds one cell gives the model no
context, and cost grows as (image / tile)^2.  Overlap stops a cell on a boundary
from being cut in half in every window that contains it.

Tiling only helps when the cells are TOO SMALL after the whole-frame resize.  On
a 1600x1272 smear where a whole-frame 320 run already detects 33 red cells at
0.65, tiling made it worse rather than better - 640-pixel windows gave 31 false
WBC and 18 RBC, 320-pixel windows gave 37 false WBC and no RBC - because the
window enlarges the cells past the size the model learned.  Measure the
whole-frame run first; tile when it finds nothing and the cells are tiny in the
frame, not when it already works.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

from vc4infer.yolo import Detector


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes, scores, classes, iou_thr):
    keep = []
    for c in np.unique(classes):
        order = np.where(classes == c)[0]
        order = order[np.argsort(-scores[order])]
        while len(order):
            keep.append(order[0])
            order = np.array([j for j in order[1:]
                              if iou(boxes[order[0]], boxes[j]) <= iou_thr],
                             dtype=int)
    return np.array(sorted(keep), dtype=int)


def windows(w, h, tile, overlap):
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    return [(x, y) for y in ys for x in xs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("--tile", type=int, default=0,
                    help="window size in original pixels (default 2x the model input)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--classes")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    names = None
    if args.classes:
        with open(args.classes, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
    det = Detector(args.model, names=names)
    tile = args.tile or det.imgsz * 2

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    boxes, scores, classes = [], [], []
    wins = windows(W, H, tile, args.overlap)
    for (x, y) in wins:
        crop = img.crop((x, y, min(x + tile, W), min(y + tile, H)))
        scale_x = crop.size[0] / float(det.imgsz)
        scale_y = crop.size[1] / float(det.imgsz)
        small = crop.resize((det.imgsz, det.imgsz), Image.BILINEAR)
        d = det.predict(small, conf=args.conf, iou=args.iou)
        for b, s, c in zip(d.boxes, d.scores, d.classes):
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            boxes.append([b[0] * scale_x + x, b[1] * scale_y + y,
                          b[2] * scale_x + x, b[3] * scale_y + y])
            scores.append(float(s))
            classes.append(int(c))

    if boxes:
        boxes = np.asarray(boxes, dtype=float)
        scores = np.asarray(scores, dtype=float)
        classes = np.asarray(classes, dtype=int)
        keep = nms(boxes, scores, classes, args.iou)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    else:
        boxes, scores = np.zeros((0, 4)), np.zeros(0)
        classes = np.zeros(0, dtype=int)

    counts = {}
    for c in classes:
        counts[det.names[c]] = counts.get(det.names[c], 0) + 1

    if args.json:
        print(json.dumps({
            "image": os.path.basename(args.image), "size": [W, H],
            "tile": tile, "model_input": det.imgsz, "tiles": len(wins),
            "overlap": args.overlap, "count": int(len(boxes)),
            "counts": counts,
            "detections": [
                {"x1": round(float(b[0]), 1), "y1": round(float(b[1]), 1),
                 "x2": round(float(b[2]), 1), "y2": round(float(b[3]), 1),
                 "score": round(float(s), 3), "name": det.names[int(c)]}
                for b, s, c in zip(boxes, scores, classes)]}, indent=1))
    else:
        print("%s  %dx%d  %d tile(s) of %d px, resized to %d"
              % (os.path.basename(args.image), W, H, len(wins), tile, det.imgsz))
        print("  %d detections  %s" % (len(boxes), counts or ""))
        for b, s, c in list(zip(boxes, scores, classes))[:10]:
            print("  %-10s %.2f  [%.0f, %.0f, %.0f, %.0f]"
                  % (det.names[int(c)], s, b[0], b[1], b[2], b[3]))
    det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
