#!/usr/bin/env python3
"""Minimal reproduction for SqueezeNet conv10: a 1x1 conv to 1000 channels.

SqueezeNet plans 66/66 and then fails on its last conv, 1x1 512 -> 1000 at
13x13, with ``copy2d_x: invalid argument``.  The conv pads Cout to 1008 for
the GEMM and slices the real 1000 channels back; the slice was the failing
call.  This script builds that one node, and the same node with Cout=1008,
so the two differ only in the channel alignment:

    TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 \
        python3 benchmodels/conv_probe.py --cout 1000
    ... --cout 1008
    ... --cout 1000 --hw 8        # the spatial extent is not the trigger

It compares the engine with onnx.reference on the same input and prints the
launch count, wall time and the largest elementwise difference.
"""
import argparse
import os
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "infer", "python"))
import vc4infer                                                   # noqa: E402


def make(cout, cin, hw, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, cin, hw, hw)).astype(np.float32)
    w = (rng.standard_normal((cout, cin, 1, 1)) * 0.02).astype(np.float32)
    b = np.zeros(cout, np.float32)
    node = helper.make_node("Conv", ["X", "W", "B"], ["Y"],
                            kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                            strides=[1, 1], dilations=[1, 1], group=1)
    g = helper.make_graph(
        [node], "conv10", [helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, [1, cin, hw, hw])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                       [1, cout, hw, hw])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    return m, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cout", type=int, default=1000)
    ap.add_argument("--cin", type=int, default=512)
    ap.add_argument("--hw", type=int, default=13)
    ap.add_argument("--arena-mb", type=int, default=96)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    m, x = make(a.cout, a.cin, a.hw)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                        "conv1x1_%d.onnx" % a.cout)
    onnx.save(m, path)
    want = np.asarray(ReferenceEvaluator(m).run(None, {"X": x})[0])

    eng = vc4infer.Engine(path, arena_mb=a.arena_mb)
    eng.run({"X": x})                     # warm-up
    t0 = time.perf_counter()
    got = np.asarray(eng.run({"X": x})["Y"])
    ms = (time.perf_counter() - t0) * 1e3
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    scaled = (d / (1.0 + np.abs(want))).max() if d.size else 0.0
    echo = {"model": os.path.basename(path), "cin": a.cin, "cout": a.cout,
            "hw": a.hw, "launches": int(eng.stats["launches"]),
            "wall_ms": round(ms, 2), "max_abs": float(d.max()) if d.size else 0.0,
            "max_scaled": float(scaled), "shape": list(got.shape)}
    print(" ".join("%s=%s" % (k, v) for k, v in echo.items()))
    eng.close()
    if a.out:
        import json
        with open(a.out, "w") as f:
            json.dump(echo, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
