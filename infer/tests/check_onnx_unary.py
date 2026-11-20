#!/usr/bin/env python3
"""check_onnx_unary - graph-level verification of the whole unary family.

One ONNX graph per op, three shapes each (`1x3x8x8`, `2x5x7x3`, `1x1x1x1`),
random data inside the op's verified domain plus a hand-picked boundary
vector, compared with onnx.reference.ReferenceEvaluator at

    |got - ref| <= 1e-4 + 1e-4*max(|ref|, 1)

composed ops (Tanh is native now; Gelu-erf and Mish are composed from the
verified Erf/Softplus/Tanh kernels) go through the executor's own path.

    sudo flock /tmp/vc4-gpu.lock env ... python3 infer/tests/check_onnx_unary.py
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


# op: (attrs, limits, opset, random range, edge vector)
OPS = {
    "Identity": ({}, None, 13, (-10, 10), None),
    "Abs": ({}, None, 13, (-10, 10), None),
    "Neg": ({}, None, 13, (-10, 10), None),
    "Sign": ({}, None, 13, (-10, 10), [0.0, -0.0, 1.0, -1.0, 1e-45, -1e-45,
                                       1e30, -1e30, 0.5, -0.5, 2.0, -2.0,
                                       3.0, -3.0, 4.0, -4.0]),
    "Floor": ({}, None, 13, (-10, 10), [-3.5, -0.5, 0.0, -0.0, 0.5, 3.5,
                                       1e7, -1e7, 1e10, -1e10, 2.9, -2.9,
                                       0.999, -0.999, 3.0, -3.0]),
    "Ceil": ({}, None, 13, (-10, 10), [-3.5, -0.5, 0.0, -0.0, 0.5, 3.5,
                                      1e7, -1e7, 1e10, -1e10, 2.9, -2.9,
                                      0.999, -0.999, 3.0, -3.0]),
    "Round": ({}, None, 13, (-10, 10), [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5,
                                       2.5, 3.5, 2.4, 2.6, -2.4, -2.6,
                                       0.0, -0.0, 1e7, -1e7]),
    "Clip": ({}, (-1.5, 2.5), 13, (-10, 10), None),
    "LeakyRelu": ({"alpha": 0.1}, None, 13, (-10, 10), None),
    "HardSigmoid": ({"alpha": 0.2, "beta": 0.5}, None, 13, (-10, 10), None),
    "HardSwish": ({}, None, 14, (-10, 10), None),
    "Exp": ({}, None, 13, (-20, 20), [-87.0, -20.0, -1.0, -1e-8, 0.0, 1e-8,
                                     0.5, 1.0, 2.0, 10.0, 80.0, 88.0,
                                     -0.5, 3.0, 4.0, 20.0]),
    "Log": ({}, None, 13, (1e-30, 1e38), [2 ** -126, 1e-30, 1e-8, 0.5, 1.0,
                                          2.0, 3.0, 1e10, 1e38, 3.4e38,
                                          0.25, 4.0, 16.0, 100.0, 7.0, 9.0]),
    "Sqrt": ({}, None, 13, (0.0, 1e20), [0.0, 1e-45, 1e-30, 1e-8, 0.25, 1.0,
                                         2.0, 4.0, 1e10, 1e20, 16.0, 100.0,
                                         3.0, 5.0, 7.0, 9.0]),
    "Reciprocal": ({}, None, 13, (0.1, 10.0), [1e-30, -1e-30, 1e-10, -1e-10,
                                               0.5, -0.5, 1.0, -1.0, 2.0,
                                               -2.0, 1e10, -1e10, 1e30,
                                               -1e30, 3.0, -3.0]),
    "Erf": ({}, None, 13, (-6, 6), [-1e30, -6.0, -3.0, -1.0, -1e-8, 0.0,
                                   1e-8, 1.0, 3.0, 6.0, 1e30, 0.5, -0.5,
                                   2.0, -2.0, 0.25]),
    "Sin": ({}, None, 13, (-100.0, 100.0), [-1e5, -1e3, -np.pi, -1.5, -0.5,
                                            0.0, 0.5, 1.5, np.pi, 1e3, 1e5,
                                            0.1, 0.2, 0.3, 0.4, 0.6]),
    "Cos": ({}, None, 13, (-100.0, 100.0), [-1e5, -1e3, -np.pi, -1.5, -0.5,
                                            0.0, 0.5, 1.5, np.pi, 1e3, 1e5,
                                            0.1, 0.2, 0.3, 0.4, 0.6]),
    "Tanh": ({}, None, 13, (-10, 10), [-1e30, -30.0, -10.0, -1.0, -1e-8, 0.0,
                                      1e-8, 1.0, 10.0, 30.0, 1e30, 0.5, -0.5,
                                      2.0, -2.0, 0.25]),
    "Softplus": ({}, None, 13, (-10, 10), [-1e30, -30.0, -10.0, -1.0, -1e-8,
                                           0.0, 1e-8, 1.0, 10.0, 30.0, 1e30,
                                           0.5, -0.5, 2.0, -2.0, 0.25]),
    "Softsign": ({}, None, 13, (-10, 10), None),
    "Elu": ({"alpha": 1.0}, None, 13, (-10, 10), None),
    "Selu": ({}, None, 13, (-10, 10), None),
    "Gelu": ({}, None, 20, (-6, 6), [-1e10, -30.0, -10.0, -1.0, -1e-8, 0.0,
                                     1e-8, 1.0, 10.0, 30.0, 1e10, 0.5, -0.5,
                                     2.0, -2.0, 0.25]),
    "Gelu#tanh": ({"approximate": "tanh"}, None, 20, (-6, 6),
                  [-1e10, -30.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 10.0,
                   30.0, 1e10, 0.5, -0.5, 2.0, -2.0, 0.25]),
    "Mish": ({}, None, 18, (-10, 10), [-1e30, -30.0, -10.0, -1.0, -1e-8, 0.0,
                                      1e-8, 1.0, 10.0, 30.0, 1e30, 0.5, -0.5,
                                      2.0, -2.0, 0.25]),
}

SHAPES = [(1, 3, 8, 8), (2, 5, 7, 3), (1, 1, 1, 1)]


def check(got, ref):
    ref = np.asarray(ref, np.float64)
    got = np.asarray(got, np.float64)
    nan_ok = np.isnan(ref) & np.isnan(got)
    inf_ok = np.isinf(ref) & np.isinf(got) & (np.sign(got) == np.sign(ref))
    err = np.abs(got - ref)
    good = nan_ok | inf_ok | (err <= 1e-4 + 1e-4 * np.maximum(np.abs(ref), 1.0))
    return int(np.sum(~good)), float(np.nanmax(np.where(good, 0.0, err))), \
        float(np.nanmax(np.where(good, 0.0, err / np.maximum(np.abs(ref), 1.0))))


def main():
    eng = None
    rows = []
    fails = 0
    rng = np.random.default_rng(17)
    for name, (attrs, limits, opset, rng_range, edge) in OPS.items():
        op = name.split("#")[0]
        for si, shape in enumerate(SHAPES):
            m = _model(op, shape, attrs, limits, opset)
            d = os.environ.get("TMPDIR", "/tmp") + "/unary_onnx"
            os.makedirs(d, exist_ok=True)
            path = d + "/unary_%s_%d.onnx" % (name.replace("#", "_"), si)
            onnx.save(m, path)
            if si == len(SHAPES) - 1 and edge is not None:
                x = np.resize(np.asarray(edge, np.float32), shape).astype(np.float32)
            else:
                lo, hi = rng_range
                x = rng.uniform(lo, hi, shape).astype(np.float32)
                if edge is not None and si == 0:
                    x.ravel()[:16] = np.asarray(edge, np.float32)[:16]
            ref = ReferenceEvaluator(m).run(None, {"x": x})[0]
            if eng is None:
                eng = vc4infer.Engine(path)
            else:
                from vc4infer import _runtime
                eng._runner = _runtime.Runner(path, eng._eng, mode="strict",
                                              host_policy="device")
            out = eng._runner.run({"x": x})[0]
            bad, ma, mr = check(out, ref)
            fails += bad
            rows.append((name, str(shape), x.size, ma, mr, bad))
            os.unlink(path)
    print("%-14s %-14s %7s %10s %10s %6s" %
          ("op", "shape", "n", "max_abs", "max_rel", "bad"))
    for r in rows:
        print("%-14s %-14s %7d %10.3g %10.3g %6d" % r)
    print("\n%d mismatches over %d elements, %d graphs" %
          (fails, sum(r[2] for r in rows), len(rows)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
