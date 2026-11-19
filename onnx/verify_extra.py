#!/usr/bin/env python3
"""Verify the native conv variants / batch against the numpy ONNX oracle.

Runs each graph 5 times, compares every output element, and prints the MACs
against the path the runner used before (per-node table in docs/ONNX.md).

    VC4INFER_DAEMON=1 python3 verify_extra.py [--repeat 5]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref_onnx                                          # noqa: E402
from onnxrun import Runner, ENGINE, vc4infer             # noqa: E402

MODELS = os.path.join(HERE, "models")

# model -> (description, previous path, MACs now, MACs before)
INFO = {
    "batched": ("2 images, Conv16->16 3x3 + Relu + 1x1",
                "batch != 1 (raised)", 25092000, 25092000),
    "grouped": ("Conv 64->64 g=4 3x3", "groups != 1/depthwise (raised)",
                64 * 16 * 16 * (16 * 9), 4 * 64 * 16 * 16 * (16 * 9)),
    "dilated": ("Conv 16->16 3x3 d=2 p=2", "dilations != 1 (raised)",
                16 * 24 * 24 * 16 * 9, 4 * 16 * 24 * 24 * 16 * 9),
    # native per-channel depthwise is not shipped (it wedges the QPU); the
    # runner uses the verified block-diagonal dense expansion, so MACs are
    # unchanged.
    "depthwise": ("Conv 32->32 g=32 3x3", "dense block-diagonal",
                  32 * 16 * 16 * 9 * 32, 32 * 16 * 16 * 9 * 32),
    "stride2": ("Conv 32->32 3x3 s=2 p=1", "4 phase images + stride-1 conv",
                32 * 16 * 16 * 32 * 9, 4 * 32 * 16 * 16 * 32 * 9),
    "bilinear": ("Conv 8->16 3x3 + Resize 2x linear", "mode=linear (raised)",
                 16 * 12 * 12 * 8 * 9 + 7 * 16 * 24 * 24, 0),
}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args(argv)

    eng = vc4infer.Engine(cache_dir=os.path.join(ENGINE, "kcache"))
    rows = []
    try:
        for name, (desc, prev, mnow, mbefore) in INFO.items():
            model = os.path.join(MODELS, name + ".onnx")
            x = np.load(os.path.join(MODELS, name + ".in.npy"))
            ref, = ref_onnx.run_model(model, {"x": x})
            r = Runner(model, eng)
            feed = {r.float_inputs[0][0]: x}
            t0 = time.perf_counter()
            out = r.run(feed, profile=True)
            jit = (time.perf_counter() - t0) * 1e3
            times, prof = [], None
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out = r.run(feed, profile=True)
                times.append((time.perf_counter() - t0) * 1e3)
                prof = r.profile
            got = np.asarray(out[0], np.float64)
            exp = np.asarray(ref, np.float64)
            d = np.abs(got - exp)
            tol = args.atol + args.rtol * np.maximum(np.abs(exp), 1.0)
            over = int((d > tol).sum())
            launches = int(sum(p["launches"] for p in prof))
            gpu = float(sum(p["gpu_ms"] for p in prof))
            wall = float(sum(p["wall_ms"] for p in prof))
            rows.append((name, desc, str(got.shape), over, d.size, d.max(),
                         launches, gpu, sum(times) / len(times)))
            print("%-10s %-38s out %-14s mism %d/%d maxabs %.3g"
                  % (name, desc, got.shape, over, d.size, d.max()))
            print("           runs %s  mean %.1f ms  launches %d  gpu %.1f ms"
                  % (", ".join("%.1f" % t for t in times), sum(times) / len(times),
                     launches, gpu))
            ratio = ("x%.2f" % (mbefore / max(mnow, 1))) if mbefore else "n/a"
            print("           MACs now %d vs %-10d before (%s): %s"
                  % (mnow, mbefore, prev, ratio))
            assert over == 0, "%s: %d over tolerance" % (name, over)
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
