#!/usr/bin/env python3
"""Benchmark YOLOv8n-160 three ways, each in its own process.

    PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
        infer/onnxruntime_ep/bench_yolo160_ort.py --all --reps 5

Backends:
  vc4     vc4infer.ort.InferenceSession(providers=["vc4"])  (the provider)
  native  vc4infer.Engine                                   (the engine alone)
  ort     onnxruntime CPU                                   (the baseline)

Separate processes matter: vc4d serves one 64 MiB arena, and two VC4 clients in
one process would fight over it.  Outputs are checked against the stored CPU
reference in every process, so a fast wrong answer cannot win.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS = os.path.join(HERE, "..", "packaging", "vc4infer", "data",
                              "models")


def _paths(models):
    return (os.path.join(models, "yolov8n_160.onnx"),
            os.path.join(models, "yolov8n_160.in.npy"),
            os.path.join(models, "yolov8n_160.ref.npy"))


def run_one(backend, models, reps):
    path, inp, refp = _paths(models)
    x = np.load(inp).astype(np.float32)
    ref = np.load(refp).astype(np.float32)
    feed_name = "images"
    info = {"backend": backend, "reps": reps}

    if backend == "vc4":
        from vc4infer.ort import InferenceSession
        sess = InferenceSession(path, providers=["vc4"])
        feed = {feed_name: x}
        sess.run(None, feed)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = sess.run(None, feed)[0]
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        s = sess.report(per_node=False)
        info.update(nodes=s["counts"], pct_macs=s["pct_macs"],
                    parts=[p.backend for p in sess.partitions],
                    timing=sess.last_timing)
        sess.close()
    elif backend == "native":
        import vc4infer
        eng = vc4infer.Engine(path)
        eng.run({feed_name: x})
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = eng.run({feed_name: x})["output0"]
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
        eng.close()
    elif backend == "ort":
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feed = {feed_name: x}
        sess.run(None, feed)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = sess.run(None, feed)[0]
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = statistics.median(times)
    else:
        raise SystemExit("unknown backend %r" % backend)

    tol = 5e-4 + 5e-4 * np.maximum(np.abs(ref), 1.0)
    d = np.abs(np.asarray(out, dtype=np.float64) - ref)
    info.update(ms=ms, fps=1000.0 / ms,
                ms_min=min(times), ms_max=max(times), ms_mean=statistics.mean(times),
                over_5e4=int((d > tol).sum()), elements=int(d.size),
                max_abs=float(d.max()))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vc4", "native", "ort"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json")
    args = ap.parse_args()
    models = os.path.abspath(args.models)

    if args.all:
        rows = []
        for b in ("vc4", "native", "ort"):
            fd, tmp = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--backend", b, "--models", models,
                            "--reps", str(args.reps), "--json", tmp],
                           check=True, cwd=os.getcwd())
            with open(tmp) as f:
                rows.append(json.load(f))
            os.unlink(tmp)
        for r in rows:
            print("%-7s median %7.2f ms (min %7.2f, max %7.2f)  %6.2f FPS  "
                  "%d/%d over 5e-4  max abs %.4g"
                  % (r["backend"], r["ms"], r["ms_min"], r["ms_max"],
                     r["fps"], r["over_5e4"], r["elements"], r["max_abs"]))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=2)
        return 0

    if not args.backend:
        raise SystemExit("use --backend X or --all")
    info = run_one(args.backend, models, args.reps)
    print(json.dumps(info, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(info, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
