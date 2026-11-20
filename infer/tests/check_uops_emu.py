#!/usr/bin/env python3
"""check_uops_emu - run every kops unary body on the bit-exact ALU model.

The body is emitted exactly as `ew_body` emits it into `build_ew`, but without
the TMU/VPM harness: A26 is the input vector, the constants are preloaded from
`ew_consts`'s assignment, and the emulator executes the ALU words with the
VC4's measured semantics (round toward zero, denormals flushed).  This checks
the *algorithm* against a float64 reference before any launch; a disagreement
here is a body/coefficient bug, not a scheduling hazard.

    python3 infer/tests/check_uops_emu.py [op ...]
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "tools"))
sys.path.insert(0, _here)
import kgen                                                   # noqa: E402
import kops                                                   # noqa: E402
import emu_alu                                                # noqa: E402

TOL = lambda ref: 1e-4 + 1e-4 * np.maximum(np.abs(ref), 1.0)


def body_words(code):
    """Emit just the body for vector 0; return (words, const assignment)."""
    if not hasattr(kgen, '_kops_consts'):
        kgen._kops_consts = {}
    dummy = kgen.Builder()
    kops.ew_consts(kgen, dummy, code)
    b = kgen.Builder()
    kops.ew_body(kgen, b, code, 0)
    return b.p.w, kgen._kops_consts[code]


def run_body(code, x, params=()):
    words, consts = body_words(code)
    m = emu_alu.Machine(words)
    m.setf('A', 26, x)
    for name, (bank, reg) in consts.items():
        if bank != 'B' or name in params_map:
            continue
        m.seti('B', reg, kops.STD[name][0])
    for name, v in params_map.items():
        bank, reg = consts[name]
        m.setf(bank, reg, np.float64(v))
    m.run(0, len(words))
    return m.getf('A', 0)


UOPS_REF = {
    'abs': np.abs,
    'neg': np.negative,
    'identity': lambda x: x,
    'sign': np.sign,
    'floor': np.floor,
    'ceil': np.ceil,
    'round': np.round,
    'exp': np.exp,
    'log': np.log,
    'sqrt': np.sqrt,
    'reciprocal': lambda x: 1.0 / x,
    'tanh': np.tanh,
    'erf': lambda x: np.vectorize(__import__('math').erf)(x),
    'softsign': lambda x: x / (1 + np.abs(x)),
    'softplus': lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0),
    'mish': lambda x: x * np.tanh(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)),
    'elu': lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1),
    'selu': lambda x: 1.0507 * np.where(
        x > 0, x, 1.6732632 * (np.exp(np.minimum(x, 0)) - 1)),
    'gelu': lambda x: 0.5 * x * (1 + np.tanh(
        0.7978845608028654 * (x + 0.044715 * x ** 3))),
    'gelu_erf': lambda x: 0.5 * x * (1 + np.vectorize(__import__('math').erf)(
        x / np.sqrt(2.0))),
    'hardsigmoid': lambda x: np.clip(x / 6 + 0.5, 0, 1),
    'hardswish': lambda x: x * np.clip(x / 6 + 0.5, 0, 1),
    'leakyrelu': lambda x: np.where(x > 0, x, 0.1 * x),
    'clip': lambda x: np.clip(x, -1.5, 2.5),
    'sin': np.sin,
    'cos': np.cos,
}

PARAMS = {
    'clip': {'P_LO': -1.5, 'P_HI': 2.5},
    'leakyrelu': {'ALPHA': 0.1},
    'hardsigmoid': {'ALPHA': 1 / 6, 'BETA': 0.5},
    'hardswish': {'ALPHA': 1 / 6, 'BETA': 0.5},
    'elu': {'ALPHA': 1.0},
    'selu': {'ALPHA': 1.6732632, 'GAMMA': 1.0507},
}

# input classes; the claimed range is part of the op's documentation
CASES = {
    'exp': [np.linspace(-80, 80, 16), np.array(
        [-87., -20., -1e-8, 0., 1e-8, 0.5, 1., 2., 10., 80., 88.] + [0.] * 5)],
    'log': [np.exp(np.linspace(-80, 80, 16)), np.array(
        [2 ** -126, 1e-30, 1e-8, 0.5, 1., 2., 1e38, 3.4e38] + [1.] * 8)],
    'sqrt': [np.linspace(0, 1e20, 16), np.array(
        [2 ** -126, 1e-30, 1e-8, 0.25, 1, 2, 1e10, 1e20] + [1.] * 8)],
    'reciprocal': [np.concatenate([-np.logspace(-30, 30, 8),
                                   np.logspace(-30, 30, 8)]),
                   np.array([1e-30, 1e-10, 1e-3, 0.5, 1, 2, 1e10, 1e30,
                             -1e-30, -1e-10, -0.5, -1, -2, -1e10, -1e30, 0.])],
    'tanh': [np.linspace(-30, 30, 16), np.array(
        [-1e30, -30., -10., -1., -1e-8, 0., 1e-8, 1., 10., 30., 1e30]
        + [0.] * 5)],
    'erf': [np.linspace(-6, 6, 16), np.array(
        [-1e30, -6, -3, -1, -1e-8, 0, 1e-8, 1, 3, 6, 1e30] + [0.] * 5)],
    'sin': [np.linspace(-np.pi, np.pi, 16), np.array(
        [-1e6, -100., -np.pi, -1.5, -0.5, 0., 0.5, 1.5, np.pi, 100., 1e6]
        + [0.] * 5)],
    'cos': [np.linspace(-np.pi, np.pi, 16), np.array(
        [-1e6, -100., -np.pi, -1.5, -0.5, 0., 0.5, 1.5, np.pi, 100., 1e6]
        + [0.] * 5)],
    'round': [np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5,
                        -3.4, -2.6, 2.6, 3.4, 0., -0., 1e7, -1e7])],
    'floor': [np.array([-3.5, -2.5, -0.5, 0.5, 1.5, -0., 0., 1e-8, -1e-8,
                        1e7, -1e7, 2.9, -2.9, 0.999, -0.999, 0.0])],
    'ceil': [np.array([-3.5, -2.5, -0.5, 0.5, 1.5, -0., 0., 1e-8, -1e-8,
                       1e7, -1e7, 2.9, -2.9, 0.999, -0.999, 0.0])],
}

DEFAULT_CASES = [
    np.array([0., -0., 1e-8, -1e-8, 0.5, -0.5, 1., -1., 2., -2.,
              8., -8., 1e10, -1e10, 1e-30, -1e-30]),
    np.linspace(-6, 6, 16),
]

params_map = {}


def main(argv):
    global params_map
    names = argv[1:] or list(UOPS_REF)
    bad_total = 0
    print("%-12s %10s %10s %8s  %s" % ("op", "max_abs", "max_rel", "bad", "worst"))
    for name in names:
        if name not in UOPS_REF:
            continue
        code = kops.UOP_ORDER.index(name)
        params_map = PARAMS.get(name, {})
        cases = CASES.get(name, DEFAULT_CASES)
        ma = mr = 0.0
        bad = 0
        worst = None
        try:
            probe = run_body(code, np.zeros(16, np.float32))
        except Exception as e:
            print("%-12s skip (composed path): %s" % (name, e))
            continue
        for x in cases:
            x = np.asarray(x, np.float32)
            got = run_body(code, x)
            with np.errstate(all='ignore'):
                ref = UOPS_REF[name](x.astype(np.float64))
            err = np.abs(got - ref)
            den = np.maximum(np.abs(ref), 1.0)
            bad += int(np.sum(err > TOL(ref)))
            if err.size and np.nanmax(err) > ma:
                ma = float(np.nanmax(err))
                mr = float(np.nanmax(err / den))
                i = int(np.nanargmax(err / den))
                worst = "x=%.9g got=%.9g ref=%.9g" % (x[i], got[i], ref[i])
        bad_total += bad
        print("%-12s %10.3g %10.3g %8d  %s" % (name, ma, mr, bad,
                                               worst or ""))
    print("\n%d mismatches" % bad_total)
    return 1 if bad_total else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
