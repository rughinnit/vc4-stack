#!/usr/bin/env python3
"""The latency/accuracy curve for one detector at several input sizes.

For each resolution this builds the input from one real image with the
package's own ``Letterbox``, takes an ONNX Runtime CPU reference on that exact
array, runs the graph on the VC4 engine with the optimiser passes, and decodes
detections on both sides.  Boxes are mapped back to original-image pixels, so
the resolutions are comparable.

    # CPU side only (no device; CPUExecutionProvider)
    python3 accfps_curve.py --models models --out out --image bus.jpg refs

    # engine side (one client, run under tools/thermal-guard.sh)
    VC4_OPT=all python3 accfps_curve.py --models models --out out points

    # join the two, no device
    python3 accfps_curve.py --models models --out out report

``--model`` measures a single ONNX file at ``--tag`` instead of the curve; the
``--imgsz`` list is used otherwise.  The reference is always recomputed by
ONNX Runtime for the model actually measured, so a fine-tuned checkpoint is
compared against its own CPU output, not against the base model.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

REPEAT = 5
WARMUP = 3


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("refs", "points", "report", "all"))
    ap.add_argument("--models", default="models",
                    help="directory holding yolov8n_<imgsz>.onnx")
    ap.add_argument("--out", default="out")
    ap.add_argument("--image", default="bus.jpg")
    ap.add_argument("--imgsz", default="96,128,160")
    ap.add_argument("--model", help="one ONNX file instead of the curve")
    ap.add_argument("--tag", help="output tag for --model (default: imgsz)")
    ap.add_argument("--repeat", type=int, default=REPEAT)
    ap.add_argument("--conf", type=float, default=0.25)
    return ap.parse_args(argv)


def points_of(args):
    if args.model:
        tag = args.tag or os.path.splitext(os.path.basename(args.model))[0]
        imgsz = int(tag.split("_")[-1]) if tag.split("_")[-1].isdigit() else None
        if imgsz is None:
            raise SystemExit("--tag must end in the input size, e.g. 96ft")
        return [(imgsz, args.model, tag)]
    if not tag_digits(args.imgsz):
        raise SystemExit("--imgsz must be a comma-separated list of sizes")
    return [(s, os.path.join(args.models, "yolov8n_%d.onnx" % s), str(s))
            for s in tag_digits(args.imgsz)]


def tag_digits(text):
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            return []
        vals.append(int(part))
    return vals


def letterbox(image, imgsz):
    from vc4infer.postprocess import Letterbox
    x, meta = Letterbox(imgsz)(Image.open(image).convert("RGB"))
    return x, meta


def detections(tensor, imgsz, meta, conf):
    from vc4infer.postprocess import Letterbox, decode_yolo
    d = decode_yolo(tensor, imgsz=imgsz, conf=conf)
    boxes = d.boxes.astype(np.float64)
    if len(boxes):
        boxes = Letterbox.to_original(boxes, meta)
    return {"count": int(len(d)), "classes": [int(c) for c in d.classes],
            "scores": [float(s) for s in d.scores],
            "boxes": boxes.tolist()}


def cpu_side(args, imgsz, model, tag, meta):
    import onnxruntime as ort
    x, _ = letterbox(args.image, imgsz)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x})[0]
    np.save(os.path.join(args.out, "in_%s.npy" % tag), x)
    np.save(os.path.join(args.out, "cpu_%s.npy" % tag), out.astype(np.float32))
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1e3)
    res = {"tag": tag, "imgsz": imgsz, "shape": list(out.shape),
           "median_ms": float(np.median(times)),
           "runs_ms": [float(t) for t in times]}
    with open(os.path.join(args.out, "cpu_%s.json" % tag), "w") as f:
        json.dump(res, f, indent=1)
    return res


def engine_side(args, imgsz, model, tag, meta):
    import vc4infer
    x = np.load(os.path.join(args.out, "in_%s.npy" % tag))
    ref = np.load(os.path.join(args.out, "cpu_%s.npy" % tag))
    eng = vc4infer.Engine(model, passes=os.environ.get("VC4_OPT") or "all")
    name = eng.inputs[0].name
    try:
        for _ in range(WARMUP):
            out = eng.run({name: x})
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            out = eng.run({name: x})
            times.append((time.perf_counter() - t0) * 1e3)
        got = np.asarray(out["output0"], np.float64)
        eng.run({name: x}, profile=True)      # counts for the profiled run
        exp = ref.astype(np.float64)
        d = np.abs(got - exp)
        tol5 = 5e-4 + 5e-4 * np.maximum(np.abs(exp), 1.0)
        tol1 = 1e-4 + 1e-4 * np.maximum(np.abs(exp), 1.0)
        prof = eng.last_profile
        mem = eng.memory()
        res = {
            "tag": tag, "imgsz": imgsz, "model": model,
            "runs_ms": [float(t) for t in times],
            "median_ms": float(np.median(times)),
            "best_ms": float(min(times)),
            "fps_median": 1000.0 / float(np.median(times)),
            "fps_best": 1000.0 / float(min(times)),
            "launches": int(sum(p["launches"] for p in prof)),
            "gpu_ms": float(sum(p["gpu_ms"] for p in prof)),
            "wall_sum_ms": float(sum(p["wall_ms"] for p in prof)),
            "nodes": len(prof),
            "arena_bytes": mem["arena_bytes"],
            "pool_bytes": mem["pool_bytes"],
            "peak_bytes": mem["peak_bytes"],
            "accuracy": {
                "max_abs": float(d.max()),
                "max_rel_scaled": float((d / np.maximum(np.abs(exp), 1.0)).max()),
                "elements": int(d.size),
                "over_5e-4": int((d > tol5).sum()),
                "over_1e-4": int((d > tol1).sum()),
            },
            "detections_vc4": detections(got, imgsz, meta, args.conf),
            "detections_cpu": detections(exp, imgsz, meta, args.conf),
        }
    finally:
        eng.close()
    with open(os.path.join(args.out, "point_%s.json" % tag), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    return res


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(ref, got, iou_min=0.5):
    """Greedy class-matched IoU pairing; returns (i, j, iou) triples."""
    pairs = sorted(((iou(a, b), i, j)
                    for i, a in enumerate(ref["boxes"])
                    for j, b in enumerate(got["boxes"])
                    if ref["classes"][i] == got["classes"][j]),
                   reverse=True)
    used_r, used_g, keep = set(), set(), []
    for iou_ij, i, j in pairs:
        if iou_ij < iou_min or i in used_r or j in used_g:
            continue
        used_r.add(i)
        used_g.add(j)
        keep.append((i, j, iou_ij))
    return keep


def report(args, pts):
    base = [p for p in pts if p["imgsz"] == max(q["imgsz"] for q in pts)][0]
    ref = base["detections_cpu"]
    rows = []
    for p in pts:
        m = match(ref, p["detections_vc4"])
        m_cpu = match(ref, p["detections_cpu"])
        deltas = [abs(a - b) for i, j, _ in m
                  for a, b in zip(ref["boxes"][i],
                                  p["detections_vc4"]["boxes"][j])]
        rows.append({
            "tag": p.get("tag", str(p["imgsz"])), "imgsz": p["imgsz"],
            "median_ms": p["median_ms"], "fps": p["fps_median"],
            "det_vc4": p["detections_vc4"]["count"],
            "det_cpu": p["detections_cpu"]["count"],
            "matched_base": len(m), "matched_base_cpu": len(m_cpu),
            "max_delta_base_px": max(deltas) if deltas else None,
        })
    print(json.dumps(rows, indent=1))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def main(argv):
    args = parse(argv)
    os.makedirs(args.out, exist_ok=True)
    pts = []
    for imgsz, model, tag in points_of(args):
        if args.mode in ("refs", "all"):
            print("CPU %d: %s" % (imgsz, json.dumps(
                cpu_side(args, imgsz, model, tag, None))), flush=True)
        if args.mode in ("points", "all"):
            _, meta = letterbox(args.image, imgsz)
            pts.append(engine_side(args, imgsz, model, tag, meta))
    if args.mode == "report":
        for imgsz, model, tag in points_of(args):
            with open(os.path.join(args.out, "point_%s.json" % tag)) as f:
                pts.append(json.load(f))
        report(args, pts)
    elif args.mode == "all":
        report(args, pts)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
