#!/usr/bin/env python3
"""Benchmark + verify a fixed-shape ONNX model on the VC4 engine.

    python yolo_bench.py models/yolov8n_160.onnx --input in.npy --ref ref.npy

Runs the graph once as a warm-up (kernel JIT is not counted), then N timed
runs, and compares the raw graph output with a CPU reference using

    |got - ref| <= atol + rtol * max(|ref|, 1)

The reference is the numpy evaluator in ref_onnx.py (checked against the
torch CPU model that produced the ONNX); see docs/ONNX.md.

The engine must be serialized: run under the shared GPU lock, e.g.
    flock /tmp/vc4-gpu.lock python3 yolo_bench.py ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from onnxrun import Runner, ENGINE, vc4infer       # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--input", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--json")
    ap.add_argument("--cache", default=os.path.join(ENGINE, "kcache"))
    args = ap.parse_args(argv)

    x = np.load(args.input).astype(np.float32)
    ref = np.load(args.ref).astype(np.float32)

    eng = vc4infer._binding.Engine(cache_dir=args.cache)
    try:
        r = Runner(args.model, eng)
        feed = {r.float_inputs[0][0]: x}
        t0 = time.perf_counter()
        out = r.run(feed, profile=True)     # warm-up (kernel JIT)
        jit_ms = (time.perf_counter() - t0) * 1e3
        prof = r.profile
        s0 = eng.stats
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            out = r.run(feed, profile=True)
            times.append((time.perf_counter() - t0) * 1e3)
        s1 = eng.stats
        prof = r.profile                   # profile of the last run
        got = np.asarray(out[0], np.float64)
        exp = ref.astype(np.float64)
        d = np.abs(got - exp)
        tol = args.atol + args.rtol * np.maximum(np.abs(exp), 1.0)
        over = int((d > tol).sum())
        res = {
            "model": os.path.basename(args.model),
            "input": list(x.shape),
            "output": list(got.shape),
            "warmup_ms": jit_ms,
            "runs_ms": times,
            "best_ms": min(times),
            "mean_ms": sum(times) / len(times),
            "fps_mean": 1000.0 / (sum(times) / len(times)),
            "fps_best": 1000.0 / min(times),
            "nodes": len(prof),
            "launches": int(sum(p["launches"] for p in prof)),
            "gpu_ms": float(sum(p["gpu_ms"] for p in prof)),
            "wall_sum_ms": float(sum(p["wall_ms"] for p in prof)),
            "arena_bytes": eng.arena_bytes,
            "arena_used": int(eng.lib.vc4_arena_used(eng.ctx)),
            "pool_bytes": r.pool.size,
            "pool_peak": r.pool.used_bytes,
            "accuracy": {
                "max_abs": float(d.max()),
                "max_rel_scaled": float((d / np.maximum(np.abs(exp), 1.0)).max()),
                "elements": int(d.size),
                "over_tolerance": over,
                "atol": args.atol,
                "rtol": args.rtol,
            },
            "profile": prof,
        }
        print("%s: %s -> %s" % (res["model"], res["input"], res["output"]))
        print("  warm-up (JIT): %.1f ms" % res["warmup_ms"])
        print("  runs: %s" % ", ".join("%.1f" % t for t in times))
        print("  %.1f ms/image (mean), %.2f FPS mean, %.2f FPS best"
              % (res["mean_ms"], res["fps_mean"], res["fps_best"]))
        print("  %d launches, GPU %.1f ms, node wall sum %.1f ms"
              % (res["launches"], res["gpu_ms"], res["wall_sum_ms"]))
        print("  arena %.1f MiB mapped, graph pool %.1f MiB, peak %.1f MiB"
              % (res["arena_bytes"] / 2 ** 20, res["pool_bytes"] / 2 ** 20,
                 res["pool_peak"] / 2 ** 20))
        print("  accuracy: max abs %.4g, max rel %.4g, %d/%d over tol"
              % (d.max(), res["accuracy"]["max_rel_scaled"], over, d.size))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(res, f, indent=1)
            print("  wrote %s" % args.json)
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
