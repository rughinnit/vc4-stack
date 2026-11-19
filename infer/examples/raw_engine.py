#!/usr/bin/env python3
"""Raw engine use: inputs/outputs, plan, profile, memory. No YOLO.

    python3 infer/examples/raw_engine.py --model modelC.onnx
    python3 infer/examples/raw_engine.py --model yolov8n.onnx   # raw head, no decode
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="modelC.onnx")
    ap.add_argument("--input", default=None, help="optional .npy feed")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if not os.path.isfile(a.model):
        raise SystemExit("no such model: %s" % a.model)

    print("devices:", vc4infer.devices())
    print(vc4infer.Device("vc4").describe())
    with vc4infer.Engine(a.model) as eng:
        print(eng)
        for spec in eng.inputs:
            print("  in  ", spec)
        for spec in eng.outputs:
            print("  out ", spec)

        plan = eng.plan()
        print("plan: %d nodes, %d dispatches, %d unverified"
              % (len(plan.nodes), plan.dispatches, len(plan.unverified)))
        print("unsupported:", eng.unsupported())

        feeds = None
        if a.input:
            feeds = {eng.inputs[0].name: np.load(a.input)}
        out = eng.run(feeds if feeds is not None else
                      {eng.inputs[0].name: np.zeros(eng.inputs[0].shape,
                                                    np.float32)})
        for name, arr in out.items():
            print("  %-20s %-16s min %.4g max %.4g mean %.4g"
                  % (name, "x".join(str(v) for v in arr.shape),
                     arr.min(), arr.max(), arr.mean()))

        prof = eng.profile(reps=a.reps, feeds=feeds)
        print("%.1f ms/image (mean), %.2f FPS, %d launches, GPU %.1f ms "
              "(%.0f%% of wall)"
              % (prof["mean_ms"], prof["fps_mean"], prof["launches"],
                 prof["gpu_ms"], 100.0 * prof["gpu_ms"] / prof["mean_ms"]))
        print("top nodes:", ", ".join("%s %.2f ms" % (n["name"].split("/")[-1],
                                                      n["wall_ms"])
                                      for n in sorted(
                                          prof["nodes"],
                                          key=lambda n: -n["wall_ms"])[:5]))
        mem = eng.memory()
        print("memory: arena %.1f MiB, pool %.1f MiB, peak %.1f MiB"
              % (mem["arena_mb"], mem["pool_mb"], mem["peak_mb"]))


if __name__ == "__main__":
    main()
