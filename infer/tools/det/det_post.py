#!/usr/bin/env python3
"""Head geometry of the vc4det student, numpy only.

The student's head is 6 logits per cell in the shipping YAML (objectness, cx,
cy, w, h, one class) and 8 in `vc4det-n3.yaml` (objectness, cx, cy, w, h, three
class scores).  This module decodes either: it has no torch and no ultralytics
dependency, so the board can use it (`student_detect.py` runs there, where
there is no torch) and the training script can use it for its AP metric.

Decode convention, identical in training, validation and on the board:

    p    = sigmoid(raw)                     raw: (N, 5+nc, 12, 12), stride 8
    obj  = p[0]
    cx   = (col + p[1]) * 8                 pixels in the 96x96 canvas
    cy   = (row + p[2]) * 8
    w    = p[3] * 96
    h    = p[4] * 96
    cls  = p[5:5+nc]                        independent sigmoids
    score= obj * max(cls)

Boxes are xyxy in the letterboxed canvas pixels -- the same frame ultralytics'
predictions live in after its own letterbox, so the two can be compared
directly.
"""
from __future__ import annotations

import os
import re

import numpy as np

IMGSZ = 96
STRIDE = 8
PAD_VALUE = 114
EPS = 1e-9


def norm_stem(stem):
    """Label stem for an image stem that the dataset renamed on one side only.

    ``scripts/make_merged.py`` on the training workstation copies the Kaggle BCD
    half as ``images/train/bcd_<name>.jpg`` but writes its labels as
    ``labels/train/<name>.txt``, so 866 of 1071 train images have no label file
    whose stem matches theirs.  Ultralytics resolves labels by path substitution
    (``img2label_paths``) and therefore trains those images as backgrounds.
    Stripping a leading lowercase dataset tag restores the pairing.
    """
    m = re.match(r"^([a-z]{2,5})_(?=[A-Z])", stem)
    return stem[m.end():] if m else stem


def index_labels(directory):
    """stem -> path for a YOLO label directory, exact stems first.

    A second key per file with :func:`norm_stem` applied is added only where it
    does not collide with a real stem.
    """
    if not directory:
        return {}
    out = {}
    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".txt":
            out[stem] = os.path.join(directory, name)
    for stem, path in list(out.items()):
        out.setdefault(norm_stem(stem), path)
    return out


def decode(raw, nc=3, imgsz=IMGSZ, stride=STRIDE):
    """(N,5+nc,H,W) logits -> boxes (N,A,4) xyxy pixels, score, class ids."""
    a = np.asarray(raw, np.float32)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4:
        raise ValueError("expected (N,5+nc,H,W), got %r" % (a.shape,))
    if a.shape[1] < 5 + nc:
        raise ValueError("head has %d channels, need %d for %d classes"
                         % (a.shape[1], 5 + nc, nc))
    p = 1.0 / (1.0 + np.exp(-a))
    n, c, h, w = p.shape
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (p[:, 1] + gx[None]) * stride
    cy = (p[:, 2] + gy[None]) * stride
    bw = p[:, 3] * imgsz
    bh = p[:, 4] * imgsz
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                     axis=-1).reshape(n, h * w, 4)
    obj = p[:, 0].reshape(n, -1)
    cls_p = p[:, 5:5 + nc].reshape(n, nc, -1).transpose(0, 2, 1)
    score = cls_p.max(axis=-1)
    cls = cls_p.argmax(axis=-1)
    return boxes, (obj * score).astype(np.float32), cls.astype(np.int64)


def iou_matrix(a, b):
    """Pairwise IoU, (N,4) x (M,4) xyxy."""
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
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, EPS)


def nms(boxes, scores, iou_thr):
    """Greedy NMS for one class.  boxes (N,4) xyxy, scores (N,)."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, EPS), 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, np.int64)


def postprocess(boxes, scores, cls, conf=0.25, iou=0.6, max_det=100, nc=3):
    """Per-class NMS on one image's cells; boxes in canvas pixels."""
    keep_all = []
    for c in range(nc):
        m = (cls == c) & (scores >= conf)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        keep_all.append(idx[nms(boxes[idx], scores[idx], iou)])
    if not keep_all:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64))
    keep = np.concatenate(keep_all)
    keep = keep[np.argsort(-scores[keep])][:max_det]
    return boxes[keep], scores[keep], cls[keep]


def map_metric(preds, gts, nc=3, iou_ths=None, max_det=100):
    """COCO-style mAP50 and mAP50-95 from per-image predictions and truth.

    ``preds``: list of (boxes (n,4), scores (n,), cls (n,))
    ``gts``:   list of (boxes (m,4), cls (m,))
    """
    if iou_ths is None:
        iou_ths = np.arange(0.5, 1.0, 0.05)
    n_gt = np.zeros(nc)
    for _, cls in gts:
        for c in np.asarray(cls).reshape(-1):
            n_gt[int(c)] += 1
    ap = np.zeros((len(iou_ths), nc))
    for ti, thr in enumerate(iou_ths):
        for c in range(nc):
            if n_gt[c] == 0:
                ap[ti, c] = np.nan
                continue
            rows = []
            for i, (pb, ps, pc) in enumerate(preds):
                m = np.asarray(pc) == c
                if not m.any():
                    continue
                bb = np.asarray(pb)[m]
                ss = np.asarray(ps)[m]
                order = np.argsort(-ss)[:max_det]
                for k in order:
                    rows.append((float(ss[k]), i, bb[k]))
            rows.sort(key=lambda r: -r[0])
            gt_hit = [np.zeros(len(np.asarray(b, np.float32).reshape(-1, 4)), bool)
                      for b, _ in gts]
            tps = []
            for _, i, box in rows:
                gb, gc = gts[i]
                gb = np.asarray(gb, np.float32).reshape(-1, 4)
                gm = np.asarray(gc) == c
                if not gm.any():
                    tps.append(0)
                    continue
                ious = iou_matrix(box[None], gb[gm])[0]
                j = int(np.argmax(ious))
                gj = np.flatnonzero(gm)[j]
                if ious[j] >= thr and not gt_hit[i][gj]:
                    tps.append(1)
                    gt_hit[i][gj] = True
                else:
                    tps.append(0)
            if not tps:
                ap[ti, c] = 0.0
                continue
            tp = np.cumsum(np.asarray(tps, np.float64))
            fp = np.cumsum(1 - np.asarray(tps, np.float64))
            rec = tp / n_gt[c]
            prec = tp / np.maximum(tp + fp, EPS)
            mrec = np.linspace(0, 1, 101)
            p_interp = np.zeros(101)
            for j, r in enumerate(mrec):
                sel = rec >= r
                p_interp[j] = prec[sel].max() if sel.any() else 0.0
            ap[ti, c] = float(p_interp.mean())
    per_class50 = ap[0]
    return {"map50": float(np.nanmean(per_class50)),
            "map": float(np.nanmean(ap)),
            "ap50_per_class": [float(v) for v in per_class50],
            "n_gt": [int(v) for v in n_gt],
            "iou_ths": [float(t) for t in iou_ths]}
