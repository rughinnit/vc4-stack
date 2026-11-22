#!/usr/bin/env python3
"""Per-node profile of one vc4det export, written as JSON.

    VC4_OPT=all PYTHONPATH=infer/python VC4INFER_LIB=infer/build/libvc4infer.so.1 \
        python3 infer/tools/det/profile.py MODEL OUT.json

One warm-up, one profiled run.  The rows are the engine's own counters, so the
wall/GPU split per node is what the runner measured, not a model.
"""
import json
import sys

import numpy as np
import vc4infer


def main(model, out):
    eng = vc4infer.Engine(model)
    name = eng.inputs[0].name
    x = np.random.default_rng(0).standard_normal(
        eng.inputs[0].shape).astype(np.float32)
    eng.run({name: x})
    eng.run({name: x}, profile=True)
    rows = [{"name": p["name"], "op": p["op"],
             "launches": int(p["launches"]),
             "wall_ms": float(p.get("wall_ms", 0.0)),
             "gpu_ms": float(p.get("gpu_ms", 0.0))}
            for p in eng.last_profile]
    total = {"launches": sum(r["launches"] for r in rows),
             "wall_ms": sum(r["wall_ms"] for r in rows),
             "gpu_ms": sum(r["gpu_ms"] for r in rows)}
    json.dump({"model": model, "rows": rows, "total": total},
              open(out, "w"), indent=1)
    print("launches %d  wall %.1f ms  gpu %.1f ms"
          % (total["launches"], total["wall_ms"], total["gpu_ms"]))
    for r in rows:
        print("  %-28s %-8s x%d  wall %6.2f  gpu %6.2f"
              % (r["name"], r["op"], r["launches"], r["wall_ms"],
                 r["gpu_ms"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
