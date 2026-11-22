#!/usr/bin/env python3
"""check_uops_gpu - element-verify every generic unary op on the QPUs.

For each op the script runs four shapes through `vc4_unary` (a single vector,
a 17-element tail, an exact 4-vector group and a multi-group 4096-vector) and
compares every element with a float64 reference at the executor tolerance

    |got - ref| <= 1e-4 + 1e-4*max(|ref|, 1)

Shapes are chosen to exercise the 16-vector and 4-vector group paths, the
padded tail, and the launch batching (4096 > 12 groups for the long bodies).
`1x1x1x1` is the n=1 case.

Run it under the interop lock (direct /dev/mem) or through vc4d:

    sudo flock /tmp/vc4-gpu.lock env VC4INFER_LIB=... \
        python3 infer/tests/check_uops_gpu.py [op ...]
"""

# A GPU client: refuse unless VC4_REQUIRE_GPU=1 (tests/gpu_gate.py).
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from gpu_gate import require_gpu as _require_gpu
_require_gpu(__file__)
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))
from vc4infer import _binding as B          # noqa: E402

TOL = lambda ref: 1e-4 + 1e-4 * np.maximum(np.abs(ref), 1.0)
SHAPES = (1, 16, 17, 256, 4096)


def erf(x):
    import math
    return np.vectorize(math.erf)(x)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


OPS = {
    "identity": (B.UOP_IDENTITY, lambda x: x, (), (-10, 10)),
    "abs": (B.UOP_ABS, np.abs, (), (-10, 10)),
    "neg": (B.UOP_NEG, np.negative, (), (-10, 10)),
    "sign": (B.UOP_SIGN, np.sign, (), (-10, 10)),
    "floor": (B.UOP_FLOOR, np.floor, (), (-10, 10)),
    "ceil": (B.UOP_CEIL, np.ceil, (), (-10, 10)),
    "round": (B.UOP_ROUND, np.round, (), (-10, 10)),
    "clip": (B.UOP_CLIP, lambda x: np.clip(x, -1.5, 2.5), (-1.5, 2.5), (-10, 10)),
    "leakyrelu": (B.UOP_LEAKYRELU,
                  lambda x: np.where(x > 0, x, 0.1 * x), (0.1,), (-10, 10)),
    "hardsigmoid": (B.UOP_HARDSIGMOID,
                    lambda x: np.clip(x / 6 + 0.5, 0, 1), (1 / 6, 0.5), (-10, 10)),
    "hardswish": (B.UOP_HARDSWISH,
                  lambda x: x * np.clip(x / 6 + 0.5, 0, 1), (1 / 6, 0.5), (-10, 10)),
    "exp": (B.UOP_EXP, np.exp, (), (-20, 20)),
    "log": (B.UOP_LOG, np.log, (), (0.001, 100.0)),
    "sqrt": (B.UOP_SQRT, np.sqrt, (), (0.0, 100.0)),
    "reciprocal": (B.UOP_RECIPROCAL, lambda x: 1.0 / x, (), (-10, 10)),
    "tanh": (B.UOP_TANH, np.tanh, (), (-10, 10)),
    "erf": (B.UOP_ERF, erf, (), (-6, 6)),
    "softplus": (B.UOP_SOFTPLUS, softplus, (), (-10, 10)),
    "softsign": (B.UOP_SOFTSIGN, lambda x: x / (1 + np.abs(x)), (), (-10, 10)),
    "elu": (B.UOP_ELU,
            lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1),
            (1.0,), (-10, 10)),
    "selu": (B.UOP_SELU,
             lambda x: 1.0507 * np.where(
                 x > 0, x, 1.6732632 * (np.exp(np.minimum(x, 0)) - 1)),
             (1.6732632, 1.0507), (-10, 10)),
    "gelu": (B.UOP_GELU,
             lambda x: 0.5 * x * (1 + np.tanh(
                 0.7978845608028654 * (x + 0.044715 * x ** 3))), (), (-6, 6)),
    "sin": (B.UOP_SIN, np.sin, (), (-6.0, 6.0)),
    "cos": (B.UOP_COS, np.cos, (), (-6.0, 6.0)),
    # inverse / hyperbolic (docs/OPS-UNARY.md).  The range is the kernel's
    # verified domain, not the whole float line: each op refuses outside it.
    "asin": (B.UOP_ASIN, np.arcsin, (), (-1.0, 1.0)),
    "acos": (B.UOP_ACOS, np.arccos, (), (-1.0, 1.0)),
    "atan": (B.UOP_ATAN, np.arctan, (), (-1e5, 1e5)),
    "atanh": (B.UOP_ATANH, np.arctanh, (), (-0.9999, 0.9999)),
    "asinh": (B.UOP_ASINH, np.arcsinh, (), (-1e5, 1e5)),
    "acosh": (B.UOP_ACOSH, np.arccosh, (), (1.0, 1e5)),
    "sinh": (B.UOP_SINH, np.sinh, (), (-88.0, 88.0)),
    "cosh": (B.UOP_COSH, np.cosh, (), (-88.0, 88.0)),
    "tan": (B.UOP_TAN, np.tan, (), (-3.0, 3.0)),
}

# edge vectors: 16 hand-picked boundary values, resized to the shape
EDGE = {
    "exp": [0.0, -0.0, 1e-8, -1e-8, 0.5, -0.5, 1.0, -1.0, 10.0, -10.0,
            80.0, -80.0, 88.0, -88.0, 1.0, -1.0],
    "log": [2 ** -126, 1e-30, 1e-8, 0.5, 1.0, 2.0, 1e10,
            1e38, 3.4e38, 0.25, 4.0, 16.0, 100.0, 3.0, 5.0, 7.0],
    "sqrt": [0.0, -0.0, 1e-45, 1e-30, 1e-8, 0.25, 1.0, 2.0, 4.0,
             1e10, 1e20, 16.0, 100.0, 3.0, 5.0, 7.0],
    "reciprocal": [0.0, -0.0, 1e-30, -1e-30, 1e-10, -1e-10, 0.5, -0.5,
                   1.0, -1.0, 2.0, -2.0, 1e10, -1e10, 1e30, -1e30],
    "sin": [-1e5, -1e3, -np.pi, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, np.pi,
            1e3, 1e5, 0.1, 0.2, 0.3, 0.4],
    "cos": [-1e5, -1e3, -np.pi, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, np.pi,
            1e3, 1e5, 0.1, 0.2, 0.3, 0.4],
    "floor": [-3.5, -2.5, -0.5, -0.0, 0.0, 1e-8, -1e-8, 1e7, -1e7, 1e10,
              -1e10, 2.9, -2.9, 0.999, -0.999, 3.0],
    "ceil": [-3.5, -2.5, -0.5, -0.0, 0.0, 1e-8, -1e-8, 1e7, -1e7, 1e10,
             -1e10, 2.9, -2.9, 0.999, -0.999, 3.0],
    "round": [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -0.0, 0.0,
              1e7, -1e7, 2.4, 2.6, -2.4, -2.6],
    "tanh": [-1e30, -30.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 10.0, 30.0,
             1e30, 0.5, -0.5, 2.0, -2.0, 0.25],
    "erf": [-1e30, -6.0, -3.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 3.0, 6.0,
            1e30, 0.5, -0.5, 2.0, -2.0, 0.25],
    "mish": [-1e30, -30.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 10.0, 30.0,
             1e30, 0.5, -0.5, 2.0, -2.0, 0.25],
    "gelu": [-1e10, -30.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 10.0, 30.0,
             1e10, 0.5, -0.5, 2.0, -2.0, 0.25],
    "gelu_erf": [-1e10, -30.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 10.0,
                 30.0, 1e10, 0.5, -0.5, 2.0, -2.0, 0.25],
    # the inverse family's own edges: the branch points (|x| = 0.5, 1), the
    # smallest fp32 step below 1, and the domains' limits
    "asin": [-1.0, -0.99999994, -0.75, -0.5, -0.50000006, -1e-8, 0.0, 1e-8,
             0.49999994, 0.5, 0.50000006, 0.75, 0.99999994, 1.0, -0.25, 0.25],
    "acos": [-1.0, -0.99999994, -0.75, -0.5, -0.50000006, -1e-8, 0.0, 1e-8,
             0.49999994, 0.5, 0.50000006, 0.75, 0.99999994, 1.0, -0.25, 0.25],
    "atan": [-1e5, -1e10, -1e3, -2.5, -1.0000001, -1.0, -0.5, -1e-8, 0.0,
             1e-8, 0.5, 1.0, 1.0000001, 2.5, 1e3, 1e5],
    "atanh": [-0.99999994, -0.99, -0.9, -0.5, -1e-8, 0.0, 1e-8, 0.5, 0.9,
              0.99, 0.99999994, -0.25, 0.25, -0.75, 0.75, -0.1],
    "asinh": [-1e5, -1e3, -100.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 10.0,
              100.0, 1e3, 1e5, -0.5, 0.5, 2.0],
    "acosh": [1.0, 1.0000001, 1.000001, 1.00000012, 1.001, 1.25, 1.5, 2.0,
              3.0, 7.0, 10.0, 100.0, 1e3, 1e5, 1.0000002, 1.1],
    "sinh": [-88.0, -80.0, -44.0, -20.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0,
             10.0, 20.0, 44.0, 80.0, 88.0, 0.5],
    "cosh": [-88.0, -80.0, -44.0, -20.0, -10.0, -1.0, -1e-8, 0.0, 1e-8, 1.0,
             10.0, 20.0, 44.0, 80.0, 88.0, 0.5],
    # tan: the poles are at +-pi/2 (~1.5708) and +-3pi/2; these sit just far
    # enough away that |cos| >= 0.05, plus the reduction's own limit
    "tan": [-1e5, -3.0, -1.52, -1.5, -1.0, -0.5, -1e-8, 0.0, 1e-8, 0.5, 1.0,
            1.5, 1.52, 3.0, 1e5, 0.7],
}
EDGE_DEFAULT = [0.0, -0.0, 1e-8, -1e-8, -10.0, 10.0, 0.5, -0.5, 1.0, -1.0,
                2.0, -2.0, 3.5, -3.5, 1e5, -1e5]


# tan's domain is not an interval: the executor refuses |cos(x)| < 0.05, so
# the random part of the vector is resampled until it is inside that domain
# (the leading edge vector is already chosen inside it).
def off_poles(x, first, lo, hi, rng):
    for _ in range(64):
        bad = np.zeros(len(x), bool)
        bad[first:] = np.abs(np.cos(x[first:].astype(np.float64))) < 0.05
        if not bad.any():
            return x
        x[bad] = rng.uniform(lo, hi, int(bad.sum())).astype(np.float32)
    return x


def check(got, ref):
    ref = np.asarray(ref, np.float64)
    got = np.asarray(got, np.float64)
    nan_ok = np.isnan(ref)
    inf_ok = np.isinf(ref) & (np.sign(got) == np.sign(ref))
    ok = nan_ok & np.isnan(got) | inf_ok
    err = np.abs(got - ref)
    good = ok | (err <= TOL(ref))
    nbad = int(np.sum(~good))
    ma = float(np.nanmax(np.where(good, 0.0, err))) if err.size else 0.0
    den = np.maximum(np.abs(ref), 1.0)
    mr = float(np.nanmax(np.where(good, 0.0, err / den))) if err.size else 0.0
    return nbad, ma, mr


def main(argv):
    names = argv[1:] or list(OPS)
    eng = B.Engine()
    rng = np.random.default_rng(11)
    print("%-12s %6s %10s %10s %6s" %
          ("op", "n", "max_abs", "max_rel", "bad"))
    total_bad = 0
    rows = []
    for name in names:
        if name not in OPS:
            continue
        uop, fn, params, (lo, hi) = OPS[name]
        edge = np.asarray(EDGE.get(name, EDGE_DEFAULT), np.float32)
        for n in SHAPES:
            x = rng.uniform(lo, hi, n).astype(np.float32)
            if n <= 16:
                x = np.resize(edge, n).astype(np.float32)
            elif n == 17:
                x[:16] = edge
            else:
                x[:16] = edge
            if name == "tan":
                x = off_poles(x, 16 if n > 16 else n, lo, hi, rng)
            a = eng.upload(x)
            o = eng.alloc(n * 4 + 64)
            p = list(params) + [0.0] * (3 - len(params))
            eng.unary(uop, a, o, n, *p)
            got = eng.download(o, n * 4)
            bad, ma, mr = check(got, fn(x.astype(np.float64)))
            total_bad += bad
            rows.append((name, n, ma, mr, bad))
            print("%-12s %6d %10.3g %10.3g %6d" % (name, n, ma, mr, bad))
    print("\n%d mismatches over %d elements, %d runs" %
          (total_bad, sum(r[1] for r in rows), len(rows)))
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
