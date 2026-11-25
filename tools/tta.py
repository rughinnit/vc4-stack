#!/usr/bin/env python3
"""tta.py - does test-time augmentation help on these images, and by how much?

Runs one image through a detector at the model's own size several times - plain,
horizontally flipped, and shifted by half a tile - then merges every box with one
global NMS.  The point is to measure the gain before promising it; TTA costs one
inference per variant.

    tools/tta.py MODEL IMAGE [--conf 0.25] [--iou 0.45] [--classes FILE]

Measured on two real smears with the 320 artifact at conf 0.25: a 1600x1272
field went from 33 boxes plain to 40 merged (39 RBC, 1 WBC), and a 588x366 field
from 0 to 1.  The variants disagree - 33 plain against 26 flipped on the same
image - which says the model is unstable on that domain rather than that the
merged count is right.  Use this to find out whether augmentation helps a given
image; do not expect it to replace training on the domain.
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


def nms(boxes, scores, classes, thr):
    keep = []
    for c in np.unique(classes):
        order = np.where(classes == c)[0]
        order = order[np.argsort(-scores[order])]
        while len(order):
            keep.append(order[0])
            order = np.array([j for j in order[1:]
                              if iou(boxes[order[0]], boxes[j]) <= thr], dtype=int)
    return np.array(sorted(keep), dtype=int)


def run(det, img, conf, iou, flip, dx, dy):
    """One pass. dx/dy shift the frame by a fraction of the image."""
    W, H = img.size
    src = img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(src, (int(dx * W), int(dy * H)))
    d = det.predict(canvas, conf=conf, iou=iou)
    boxes = np.asarray(d.boxes, dtype=float).copy()
    if len(boxes):
        boxes[:, [0, 2]] -= dx * W
        boxes[:, [1, 3]] -= dy * H
        if flip:
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
    return boxes, np.asarray(d.scores, dtype=float), np.asarray(d.classes, dtype=int)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--classes")
    args = ap.parse_args(argv)

    names = None
    if args.classes:
        with open(args.classes, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
    det = Detector(args.model, names=names)
    img = Image.open(args.image).convert("RGB")

    variants = [("plain", 0, 0.0, 0.0), ("flip", 1, 0.0, 0.0),
                ("shift", 0, 0.05, 0.05), ("flip+shift", 1, 0.05, 0.05)]
    allb, alls, allc = [], [], []
    per = {}
    for name, flip, dx, dy in variants:
        b, s, c = run(det, img, args.conf, args.iou, flip, dx, dy)
        per[name] = len(b)
        allb.append(b); alls.append(s); allc.append(c)
    boxes = np.concatenate(allb) if allb else np.zeros((0, 4))
    scores = np.concatenate(alls) if alls else np.zeros(0)
    classes = np.concatenate(allc) if allc else np.zeros(0, dtype=int)
    merged = 0
    if len(boxes):
        keep = nms(boxes, scores, classes, args.iou)
        merged = len(keep)
        counts = {}
        for c in classes[keep]:
            counts[det.names[int(c)]] = counts.get(det.names[int(c)], 0) + 1
    else:
        counts = {}
    print(json.dumps({"image": os.path.basename(args.image),
                      "per_variant": per, "merged": merged, "counts": counts,
                      "model": os.path.basename(args.model)}))
    det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
