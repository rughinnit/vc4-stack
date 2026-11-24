#!/usr/bin/env python3
"""Teacher detections -> pseudo-labels for the vc4det student.

    /mnt/Ext/venv-tsl2/bin/python infer/tools/det/pseudo.py \
        --teacher ~/bccd/runs/detect/runs/bccd-v11n-320/weights/best.pt \
        --images ~/bccd/datasets/merged/images/train \
        --labels ~/bccd/datasets/merged/labels/train \
        --out /mnt/Ext/bccd-student/pseudo/train --imgsz 320

`train_student.py` merges this directory with the ground truth at load time;
nothing here edits the dataset.  One line per detection, normalised to the
*original* image, so the trainer can letterbox it to the student's canvas:

    cls cx cy w h conf p0 p1 p2

Columns 1-5 are YOLO's; ``conf`` is the teacher's score and ``p0..p2`` its
per-class scores (the argmax class carries ``conf``, the others 0, because
ultralytics' Boxes expose only the winning class; the trainer uses the vector
as a soft class target, so a hard vector still down-weights an unsure
detection relative to a 1.0 target).

Ground-truth boxes are not copied into this directory.  A detection that
overlaps a ground-truth box of the same image by ``--dedup-iou`` or more is
dropped: the human label wins, and the teacher's contribution is the objects
the label set does not have.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def iou_matrix(a, b):
    import numpy as np
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def read_gt(path, w, h):
    """YOLO txt -> (boxes xyxy pixels, classes).  Empty file -> empty arrays."""
    import numpy as np
    boxes, cls = [], []
    if not os.path.isfile(path):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            cls.append(c)
    return (np.asarray(boxes, np.float32).reshape(-1, 4),
            np.asarray(cls, np.int64))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="ultralytics .pt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, help="ground truth txt dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=320,
                    help="teacher input size (default: the size it was "
                         "fine-tuned at)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6, help="teacher NMS IoU")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--dedup-iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N only")
    ap.add_argument("--report", default=None, help="write a JSON summary here")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import det_post as P

    label_index = P.index_labels(args.labels)
    unresolved = 0

    paths = sorted(glob.glob(os.path.join(args.images, "*")))
    if args.limit:
        paths = paths[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    model = YOLO(args.teacher)
    stats = {"teacher": os.path.abspath(args.teacher), "imgsz": args.imgsz,
             "conf": args.conf, "iou": args.iou, "images": 0, "gt": 0,
             "kept": 0, "dup": 0, "unlabelled_images": 0,
             "per_class_kept": {}, "per_class_gt": {}}
    results = model.predict(source=paths, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, max_det=args.max_det,
                            device=args.device, stream=True, verbose=False,
                            half=False)
    for path, r in zip(paths, results):
        h, w = r.orig_shape
        stem = os.path.splitext(os.path.basename(path))[0]
        lab = label_index.get(stem) or label_index.get(P.norm_stem(stem))
        if lab is None:
            unresolved += 1
        gt_b, gt_c = read_gt(lab, w, h)
        det = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros(
            (0, 4), np.float32)
        conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        cls = r.boxes.cls.cpu().numpy().astype(np.int64) if r.boxes is not None \
            else np.zeros(0, np.int64)
        lines = []
        for i in range(len(det)):
            if len(gt_b) and iou_matrix(det[i:i + 1], gt_b).max() >= args.dedup_iou:
                stats["dup"] += 1
                continue
            b = det[i]
            cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
            bw, bh = (b[2] - b[0]) / w, (b[3] - b[1]) / h
            if bw <= 0 or bh <= 0:
                continue
            probs = [0.0, 0.0, 0.0]
            probs[int(cls[i])] = float(conf[i])
            lines.append("%d %.6f %.6f %.6f %.6f %.4f %s" % (
                int(cls[i]), cx, cy, bw, bh, float(conf[i]),
                " ".join("%.4f" % v for v in probs)))
            stats["per_class_kept"][str(int(cls[i]))] = \
                stats["per_class_kept"].get(str(int(cls[i])), 0) + 1
        with open(os.path.join(args.out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["kept"] += len(lines)
        stats["gt"] += len(gt_b)
        for c in gt_c:
            stats["per_class_gt"][str(int(c))] = \
                stats["per_class_gt"].get(str(int(c)), 0) + 1
        if stats["images"] % 100 == 0:
            print("  %d/%d images, %d kept" % (stats["images"], len(paths),
                                               stats["kept"]), flush=True)

    stats["unlabelled_images"] = unresolved
    print(json.dumps(stats, indent=1))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(stats, f, indent=1)
    return stats


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
