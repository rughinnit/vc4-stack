#!/usr/bin/env python3
"""check_unary_domains - the executor must REFUSE what a kernel cannot answer.

Every unary kernel in the inverse/hyperbolic family has a documented domain
(docs/OPS-UNARY.md).  This script feeds one out-of-domain value per op through
the real executor and requires a named refusal - not a wrong number, not a
crash.  It is the mirror image of check_onnx_unary.py: that one proves the
inside of the domain, this one proves the boundary is a wall.

    VC4INFER_DAEMON=1 ... python3 infer/tests/check_unary_domains.py
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))
import vc4infer                                          # noqa: E402

# op -> (bad value, must appear in the refusal text)
CASES = {
    "Asin": (1.5, "domain"),
    "Acos": (-1.0000001, "domain"),
    "Atan": (float("inf"), "domain"),
    "Atanh": (1.0, "domain"),
    "Asinh": (float("-inf"), "domain"),
    "Acosh": (0.5, "domain"),
    "Sinh": (90.0, "domain"),
    "Cosh": (-90.0, "domain"),
    "Tan": (1.5707964, "domain"),          # |cos| = 5e-8: a pole
}

# the values that must still RUN (the inside of each boundary)
INSIDE = {
    "Asin": [1.0, -1.0], "Acos": [1.0, -1.0], "Atanh": [0.99999994],
    "Acosh": [1.0], "Tan": [1.50],
}


def _model(op, shape=(1, 4, 4, 1)):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    node = helper.make_node(op, ["x"], ["y"], name="n_" + op)
    g = helper.make_graph([node], "g_" + op, [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def run(eng, m, path, x):
    from vc4infer import _runtime
    eng._runner = _runtime.Runner(path, eng._eng, mode="strict",
                                  host_policy="device")
    return eng._runner.run({"x": x})[0]


def main():
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "unary_domains")
    os.makedirs(d, exist_ok=True)
    eng = None
    fails = 0
    print("%-8s %-12s %s" % ("op", "value", "result"))
    for op, (bad, want) in CASES.items():
        path = os.path.join(d, "dom_%s.onnx" % op)
        onnx.save(_model(op), path)
        x = np.full((1, 4, 4, 1), bad, np.float32)
        if eng is None:
            eng = vc4infer.Engine(path)
        try:
            run(eng, None, path, x)
        except Exception as e:                            # noqa: BLE001
            msg = str(e)
            ok = want in msg and op in msg
            print("%-8s %-12g %s: %s" % (op, bad, "refused" if ok else
                                         "refused (text?)", msg.split("\n")[0]))
            if not ok:
                fails += 1
        else:
            print("%-8s %-12g ANSWERED - must refuse" % (op, bad))
            fails += 1
        # and the inside of the boundary must still run and agree
        for v in INSIDE.get(op, []):
            x = np.full((1, 4, 4, 1), v, np.float32)
            ref = ReferenceEvaluator(_model(op)).run(None, {"x": x})[0]
            got = run(eng, None, path, x)
            err = np.max(np.abs(got.astype(np.float64) -
                                ref.astype(np.float64)))
            tol = 1e-4 + 1e-4 * max(float(np.max(np.abs(ref))), 1.0)
            if not err <= tol:
                print("%-8s %-12g inside value FAILED err=%.3g" % (op, v, err))
                fails += 1
        os.unlink(path)
    print("\n%d failures" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
