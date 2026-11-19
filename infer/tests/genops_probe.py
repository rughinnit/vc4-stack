#!/usr/bin/env python3
"""genops_probe - kernel-level accuracy + cost check for the generic QPU ops.

    VC4INFER_DAEMON=1 VC4INFER_CACHE=/tmp/kc VC4INFER_KGEN=$PWD/infer/python/vc4infer/kgen/kgen.py \
        python3 infer/tests/genops_probe.py [op ...]

Every op is compared with a numpy float64 reference at the executor tolerance
1e-4 + 1e-4*max(|ref|,1).  Prints max abs, max rel, mismatch count, launches
and GPU milliseconds from the engine's own statistics.  No host arithmetic is
involved: the reference is verification only.

Two shapes are always used (a 4096-element random vector and a hand-picked
edge vector), plus a third for the periodic ops.
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))
from vc4infer import _binding as B          # noqa: E402

TOL = lambda ref: 1e-4 + 1e-4 * np.maximum(np.abs(ref), 1.0)


def stats(eng):
    import ctypes
    s = B.Stats()
    eng.lib.vc4_get_stats(eng.ctx, ctypes.byref(s))
    return s


def run(eng, op, x, p0=0.0, p1=0.0, p2=0.0):
    a = eng.upload(x.astype(np.float32))
    n = x.size
    npad = ((n + 255) // 256) * 256
    o = eng.alloc(npad * 4 + 64)
    s0 = stats(eng)
    eng.unary(op, a, o, n, p0, p1, p2)
    s1 = stats(eng)
    y = eng.download(o, npad * 4)[:n]
    return y, s1.launches - s0.launches, (s1.kernel_ns - s0.kernel_ns) / 1e6


def check(got, ref):
    ref64 = np.asarray(ref, np.float64)
    err = np.abs(np.asarray(got, np.float64) - ref64)
    den = np.maximum(np.abs(ref64), 1.0)
    return int(np.sum(err > TOL(ref64))), float(err.max()), float((err / den).max())


def erf(x):
    import math
    return np.vectorize(math.erf)(x)


OPS = {
    'abs': (B.UOP_ABS, np.abs, (), 'random'),
    'neg': (B.UOP_NEG, np.negative, (), 'random'),
    'sign': (B.UOP_SIGN, np.sign, (), 'signed'),
    'clip': (B.UOP_CLIP, lambda x: np.clip(x, -1.5, 2.5), (-1.5, 2.5), 'random'),
    'leakyrelu': (B.UOP_LEAKYRELU, lambda x: np.where(x > 0, x, 0.1 * x), (0.1,), 'random'),
    'hardsigmoid': (B.UOP_HARDSIGMOID, lambda x: np.clip(x / 6 + 0.5, 0, 1),
                    (1 / 6, 0.5), 'random'),
    'hardswish': (B.UOP_HARDSWISH, lambda x: x * np.clip(x / 6 + 0.5, 0, 1),
                  (1 / 6, 0.5), 'random'),
    'exp': (B.UOP_EXP, np.exp, (), 'positive'),
    'log': (B.UOP_LOG, np.log, (), 'positive'),
    'sqrt': (B.UOP_SQRT, np.sqrt, (), 'positive'),
    'reciprocal': (B.UOP_RECIPROCAL, lambda x: 1.0 / x, (), 'positive'),
    'tanh': (B.UOP_TANH, np.tanh, (), 'random'),
    'erf': (B.UOP_ERF, erf, (), 'random'),
    'sin': (B.UOP_SIN, np.sin, (), 'random'),
    'cos': (B.UOP_COS, np.cos, (), 'random'),
    'floor': (B.UOP_FLOOR, np.floor, (), 'random'),
    'ceil': (B.UOP_CEIL, np.ceil, (), 'random'),
    'round': (B.UOP_ROUND, np.round, (), 'signed'),
    'softplus': (B.UOP_SOFTPLUS,
                 lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0), (), 'random'),
    'softsign': (B.UOP_SOFTSIGN, lambda x: x / (1 + np.abs(x)), (), 'random'),
    'elu': (B.UOP_ELU, lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1), (1.0,), 'random'),
    'selu': (B.UOP_SELU,
             lambda x: 1.0507 * np.where(x > 0, x, 1.6732632 * (np.exp(np.minimum(x, 0)) - 1)),
             (1.6732632, 1.0507), 'random'),
    'gelu': (B.UOP_GELU,
             lambda x: 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3))),
             (), 'random'),
}


def main(argv):
    eng = B.Engine()
    which = argv[1:] or sorted(OPS)
    rng = np.random.default_rng(7)
    print("%-12s %7s %10s %10s %6s %7s %8s" %
          ("op", "n", "max_abs", "max_rel", "bad", "launch", "gpu_ms"))
    for name in which:
        if name not in OPS:
            continue
        op, fn, params, kind = OPS[name]
        if kind == 'positive':
            shapes = [rng.uniform(0.01, 8.0, 4096),
                      np.array([1e-6, 0.5, 1.0, 2.0, 8.0, 100.0])]
        elif kind == 'signed':
            shapes = [rng.uniform(-5, 5, 4096),
                      np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.0, -0.0,
                                -0.4, 3.0, -3.0, 1e-7])]
        else:
            shapes = [rng.uniform(-8, 8, 4096),
                      np.array([0.0, 1.0, -1.0, 6.0, -6.0])]
        for k, x in enumerate(shapes):
            p = list(params) + [0.0] * (3 - len(params))
            got, nl, ms = run(eng, op, x, *p)
            bad, ma, mr = check(got, fn(x.astype(np.float32)))
            print("%-12s %7d %10.3g %10.3g %6d %7d %8.3f  [shape %d]" %
                  (name, x.size, ma, mr, bad, nl, ms, k))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
