#!/usr/bin/env python3
"""Per-node profile of one vc4det export, written as JSON.

    VC4_OPT=all PYTHONPATH=infer/python VC4INFER_LIB=infer/build/libvc4infer.so.1 \
        python3 infer/tools/det/profile.py MODEL OUT.json

One warm-up, one profiled run.  The rows are the engine's own counters, so the
wall/GPU split per node is what the runner measured, not a model.  The tool
refuses an engine outside this checkout; ``--allow-installed`` tests an
installed one on purpose.
"""
import json
import sys

import numpy as np

import os as _os
import sys as _sys
_ROOT = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.join(_ROOT, "tools"))
from checkout_guard import require_checkout                # noqa: E402
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
    _argv = [a for a in sys.argv[1:] if a != "--allow-installed"]
    if "--allow-installed" in sys.argv[1:]:
        require_checkout(_os.path.abspath(__file__), allow_installed=True)
    else:
        require_checkout(_os.path.abspath(__file__))
    if len(_argv) != 2:
        print("usage: node_profile.py MODEL OUT.json [--allow-installed]",
              file=sys.stderr)
        sys.exit(2)
    main(_argv[0], _argv[1])
