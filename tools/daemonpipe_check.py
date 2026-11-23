#!/usr/bin/env python3
"""daemonpipe_check.py - hash a model's output through the daemon.

The daemon's pipelined path prepares request N+1 while request N runs, and must
not change a single byte of what a client gets back.  This prints one SHA-256
per output tensor (plus one over all of them) for a model run through
libvc4infer, so the same command against a daemon started with --no-prefetch
and one started without it can be compared with diff(1).

usage:
  VC4INFER_DAEMON=1 VC4INFER_PROF=1 tools/daemonpipe_check.py MODEL.onnx [--frames N]
  VC4INFER_DAEMON=1 VC4INFER_PROF=1 tools/daemonpipe_check.py --suite
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "infer", "python"))

from vc4infer.engine import Engine                          # noqa: E402

MODELS = os.environ.get("VC4INFER_MODELS", os.path.expanduser("~/work/onnx/models"))


def digest(outs):
    h = hashlib.sha256()
    for k in sorted(outs):
        a = np.ascontiguousarray(outs[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def one(path, frames, pipeline):
    eng = Engine(path)
    try:
        spec = eng.inputs[0]
        n = int(np.prod(spec.shape))
        rng = np.random.default_rng(1234)
        x = {spec.name: rng.standard_normal(n).astype(np.float32)
             .reshape(spec.shape)}
        run = eng.run_pipelined if pipeline else eng.run
        run(x)                                  # warm-up, also builds the plan
        for _ in range(frames):
            run(x)
        outs = run(x)
        per = {k: hashlib.sha256(np.ascontiguousarray(outs[k]).tobytes())
               .hexdigest()[:16] for k in sorted(outs)}
        return outs, per
    finally:
        eng.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--suite", action="store_true",
                    help="modelA, modelB, modelC and yolov8n at 96/128/160")
    ap.add_argument("--frames", type=int, default=2)
    args = ap.parse_args()

    if args.suite:
        paths = [os.path.join(MODELS, "%s.onnx" % m)
                 for m in ("modelA", "modelB", "modelC")]
        paths += [os.path.join(MODELS, "yolov8n_%d.onnx" % s)
                  for s in (96, 128, 160)]
    else:
        paths = args.models
    if not paths:
        ap.error("name a model, or --suite")

    for path in paths:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print("%-22s missing" % name)
            continue
        for pipeline in (False, True):
            outs, per = one(path, args.frames, pipeline)
            print("%-22s %-9s %s" % (name,
                                     "pipelined" if pipeline else "sync",
                                     digest(outs)))
            for k in sorted(per):
                print("    %-28s %s" % (k, per[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
