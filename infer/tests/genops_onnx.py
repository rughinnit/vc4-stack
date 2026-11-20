#!/usr/bin/env python3
"""Graph-level check for the verified generic-unary ONNX ops.

Builds one tiny ONNX graph per op (three shapes each + an edge case), runs it
through vc4infer's executor on the QPUs, and compares with
onnx.reference.ReferenceEvaluator (float32, the same reference the rest of the
tree uses).  Tolerance 1e-4 + 1e-4*max(|ref|,1); every element is checked.

    VC4INFER_DAEMON=1 VC4INFER_CACHE=/tmp/kc python3 infer/tests/genops_onnx.py
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))
import vc4infer                                          # noqa: E402


def _model(op, shape, attrs=None, limits=None, opset=13):
    """One op, one float input, optional constant limits for Clip."""
    attrs = attrs or {}
    inits = []
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    names = ["x"]
    if limits is not None:
        for i, v in enumerate(limits):
            if v is None:
                names.append("")
                continue
            n = "lim%d" % i
            inits.append(numpy_helper.from_array(np.array([v], np.float32), n))
            names.append(n)
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    node = helper.make_node(op, names, ["y"], name="n_" + op, **attrs)
    g = helper.make_graph([node], "g_" + op, inputs, [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 8
    return m


CASES = [
    # (op, attrs, limits, [shapes], opset)
    ("Abs", {}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 1, 1, 1)], 13),
    ("Neg", {}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 16, 4, 4)], 13),
    ("Sign", {}, None, [(1, 3, 8, 8), (3, 1, 5, 5), (1, 1, 1, 1)], 13),
    ("Clip", {}, (-1.5, 2.5), [(1, 3, 8, 8), (2, 5, 7, 3), (1, 1, 1, 1)], 13),
    ("Clip", {}, (0.0, None), [(1, 3, 8, 8), (2, 16, 3, 3)], 13),
    ("Clip", {"min": -0.5, "max": 0.5}, None, [(1, 3, 8, 8)], 6),
    ("LeakyRelu", {"alpha": 0.1}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 1, 1, 1)], 13),
    ("LeakyRelu", {"alpha": 0.02}, None, [(1, 4, 6, 6)], 13),
    ("HardSigmoid", {"alpha": 0.2, "beta": 0.5}, None,
     [(1, 3, 8, 8), (2, 5, 7, 3), (1, 16, 4, 4)], 13),
    ("HardSwish", {}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 1, 1, 1)], 14),
    ("Tanh", {}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 16, 4, 4)], 13),
    ("Floor", {}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 1, 1, 1)], 13),
    ("Ceil", {}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 1, 1, 1)], 13),
    ("Softsign", {}, None, [(1, 3, 8, 8), (2, 5, 7, 3), (1, 16, 4, 4)], 13),
]

EDGE = [
    np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-30, 1e30, 3.0, -3.0,
              1e-8, -1e-8], np.float32),
    np.array([[[[2.5, -2.5, 0.0, 7.0]]]], np.float32),
    np.array([[[[1.0]]]], np.float32),
]


def main():
    eng = None
    rows = []
    fails = 0
    for op, attrs, limits, shapes, opset in CASES:
        for si, shape in enumerate(shapes):
            m = _model(op, shape, attrs, limits, opset)
            path = "/tmp/genops_%s_%d.onnx" % (op, si)
            onnx.save(m, path)
            rng = np.random.default_rng(11 + si)
            x = rng.uniform(-6.0, 6.0, shape).astype(np.float32)
            if si == len(shapes) - 1:
                x = np.resize(EDGE[si % len(EDGE)], shape).astype(np.float32)
            ref = ReferenceEvaluator(m).run(None, {"x": x})[0]
            if eng is None:
                eng = vc4infer.Engine(path)
            else:
                # one arena for the whole run: swap the graph, keep the device
                from vc4infer import _runtime
                eng._runner = _runtime.Runner(path, eng._eng, mode="strict",
                                              host_policy="device")
            out = eng._runner.run({"x": x})[0]
            err = np.abs(out.astype(np.float64) - ref.astype(np.float64))
            den = np.maximum(np.abs(ref.astype(np.float64)), 1.0)
            bad = int(np.sum(err > 1e-4 * den))
            ma, mr = float(err.max()), float((err / den).max())
            fails += bad
            rows.append((op, attrs, str(shape), x.size, ma, mr, bad, si))
            os.unlink(path)
    print("%-12s %-18s %-14s %6s %10s %10s %6s %3s" %
          ("op", "attrs", "shape", "n", "max_abs", "max_rel", "bad", "s"))
    for r in rows:
        print("%-12s %-18s %-14s %6d %10.3g %10.3g %6d %3d" % r)
    print("\n%d mismatches over %d elements, %d graphs" %
          (fails, sum(r[3] for r in rows), len(rows)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
