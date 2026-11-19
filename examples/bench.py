#!/usr/bin/env python3
"""Time a model from Python; the numbers match ``vc4infer bench``.

    python3 examples/bench.py --model yolov8n.onnx --reps 3
    python3 examples/bench.py --model yolov8n.onnx --input bus.jpg
"""
import argparse
import os

import numpy as np

import vc4infer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolov8n.onnx")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--input", default=None, help="optional .npy or image")
    a = ap.parse_args()
    if not os.path.isfile(a.model):
        raise SystemExit("no such model: %s" % a.model)

    with vc4infer.Engine(a.model) as eng:
        feeds = None
        if a.input and a.input.lower().endswith(".npy"):
            feeds = {eng.inputs[0].name: np.load(a.input).astype(np.float32)}
        elif a.input:
            from PIL import Image
            from vc4infer.postprocess import Letterbox
            size = eng.inputs[0].shape[2]
            x, _ = Letterbox(size)(Image.open(a.input).convert("RGB"))
            feeds = {eng.inputs[0].name: x}
        r = eng.profile(reps=a.reps, feeds=feeds)
        ins = eng.inputs[0]
    print("model:  %s" % r["model"])
    print("input:  %s" % "x".join(str(d) for d in ins.shape))
    print("warm-up (JIT): %.1f ms" % r["warmup_ms"])
    print("runs:   %s ms" % " / ".join("%.1f" % t for t in r["runs_ms"]))
    print("%.1f ms/image (mean), %.1f ms (median), best %.1f ms"
          % (r["mean_ms"], r["median_ms"], r["min_ms"]))
    print("%.2f FPS (mean), %.2f FPS (best)"
          % (r["fps_mean"], r["fps_best"]))
    print("%d launches/image, GPU %.1f ms (%.0f%% of wall), node wall sum %.1f "
          "ms" % (r["launches"], r["gpu_ms"],
                  100.0 * r["gpu_ms"] / r["mean_ms"], r["wall_sum_ms"]))
    print("arena %.1f MiB, graph pool %.1f MiB, peak %.1f MiB"
          % (r["arena_mb"], r["pool_mb"], r["peak_mb"]))
    top = sorted(r["nodes"], key=lambda n: -n["wall_ms"])[:5]
    print("top nodes:", ", ".join("%s %.1f ms" % (n["name"].split("/")[-1],
                                                  n["wall_ms"]) for n in top))


if __name__ == "__main__":
    main()
