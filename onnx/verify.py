#!/usr/bin/env python3
"""Run the exported models on the VC4 engine and compare with torch CPU.

    python verify.py [modelA modelB modelC ...]

A warm-up run compiles the QPU kernels (JIT) and is not counted; then three
runs are measured.  Per-node times are the median of the three runs.  The
tolerance is the fp32 one used by the engine's own tests:
    |got - ref| <= atol + rtol * max(|ref|, 1)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from onnxrun import Runner, ENGINE  # noqa: E402
import vc4infer  # noqa: E402

ATOL = 1e-4
RTOL = 1e-4
RUNS = 3
NAMES = ("modelA", "modelB", "modelC")


def compare(got, ref):
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    den = np.maximum(np.abs(ref.astype(np.float64)), 1.0)
    tol = ATOL + RTOL * den
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / np.maximum(np.abs(ref), 1e-12)).max()),
        "max_rel_scaled": float((d / den).max()),
        "mismatches": int((d > tol).sum()),
        "elements": int(d.size),
    }


def median(xs):
    return float(np.median(xs))


def run_model(name):
    onnx_path = os.path.join(HERE, "models", name + ".onnx")
    x = np.load(os.path.join(HERE, "models", name + ".in.npy")).astype(np.float32)
    ref = np.load(os.path.join(HERE, "models", name + ".out.npy")).astype(np.float32)

    eng = vc4infer.Engine(cache_dir=os.environ.get(
        "VC4INFER_CACHE", os.path.join(ENGINE, "kcache")))
    try:
        r = Runner(onnx_path, eng)
        t0 = time.perf_counter()
        r.run({r.float_inputs[0][0]: x}, profile=False)
        warm = (time.perf_counter() - t0) * 1e3

        walls, procs, outs, cmp = [], [], [], None
        for _ in range(RUNS):
            t0 = time.perf_counter()
            outs = r.run({r.float_inputs[0][0]: x}, profile=True)
            walls.append((time.perf_counter() - t0) * 1e3)
            procs.append(list(r.profile))
            cmp = compare(outs[0], ref)
            if cmp["mismatches"]:
                break

        nodes = {}
        for p in procs[-1]:
            key = (p["name"], p["op"])
            nodes.setdefault(key, []).append(p)
        rows = []
        for key, ps in nodes.items():
            rows.append({
                "node": key[0], "op": key[1],
                "wall_ms": median([p["wall_ms"] for p in ps]),
                "gpu_ms": median([p["gpu_ms"] for p in ps]),
                "launches": int(median([p["launches"] for p in ps])),
                "flops": int(ps[-1]["flops"]),
            })
        total_launches = sum(p["launches"] for p in procs[-1])
        total_gpu = sum(p["gpu_ms"] for p in procs[-1])
        return {
            "model": name,
            "input": list(x.shape),
            "output": list(ref.shape),
            "onnx_kib": os.path.getsize(onnx_path) / 1024.0,
            "warmup_ms": warm,
            "runs_ms": walls,
            "nodes": rows,
            "total_launches": total_launches,
            "total_gpu_ms": total_gpu,
            "accuracy": cmp,
            "pass": cmp["mismatches"] == 0,
        }
    finally:
        eng.close()


def main(argv):
    names = argv or list(NAMES)
    results = []
    for n in names:
        res = run_model(n)
        results.append(res)
        print("== %s: %s  (%s -> %s, %d nodes)" %
              (n, "PASS" if res["pass"] else "FAIL",
               res["input"], res["output"], len(res["nodes"])))
        print("   warmup (JIT) %.1f ms; runs %s ms" %
              (res["warmup_ms"], ["%.1f" % v for v in res["runs_ms"]]))
        print("   max abs %.3g   max rel %.3g (den=max(|ref|,1))   "
              "mismatches %d/%d   launches %d   gpu %.3f ms" %
              (res["accuracy"]["max_abs"], res["accuracy"]["max_rel_scaled"],
               res["accuracy"]["mismatches"], res["accuracy"]["elements"],
               res["total_launches"], res["total_gpu_ms"]))
        print("   %-28s %-16s %8s %8s %6s" %
              ("node", "op", "wall ms", "gpu ms", "launch"))
        for row in res["nodes"]:
            print("   %-28s %-16s %8.3f %8.3f %6d" %
                  (row["node"][:28], row["op"], row["wall_ms"],
                   row["gpu_ms"], row["launches"]))
        print()

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
