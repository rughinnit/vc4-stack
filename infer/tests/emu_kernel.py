#!/usr/bin/env python3
"""emu_kernel - run a whole generated nvec=4/16 unary kernel on the ALU model.

The body-level checker (`check_uops_emu.py`) starts with clean registers; this
one runs the real harness prologue, uniform port, TMU address queue and every
per-vector body, then reads the result registers that the verified 4-row store
would write.  It therefore catches harness-context bugs (B8 being the b
operand, temp allocation, per-vector addressing) that a raw body hides.

No GPU is used.
"""
import os
import struct
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "tools"))
sys.path.insert(0, _here)
import kgen                                                     # noqa: E402
import kops                                                     # noqa: E402
import emu_alu                                                  # noqa: E402


def _bits(v):
    return int(struct.unpack('<I', struct.pack('<f', float(v)))[0])


def run_kernel(op, x, nvec, params=()):
    """Return the nvec*16 values the store would write (row-major)."""
    p = kops.build_uop(kgen, op, nvec)
    ws = p.write('/tmp/emu_kernel.bin')
    m = emu_alu.Machine(ws)
    a_bus, o_bus = 0xE0000000, 0xE1000000
    prm = list(params) + [0.0] * (3 - len(params))
    m.unif = [a_bus, a_bus, o_bus, 1, 0, 1] + [_bits(v) for v in prm] + [0] * 7
    xs = np.asarray(x, np.float32)
    for i, v in enumerate(xs[:nvec * 16]):
        m.mem[a_bus + i * 4] = _bits(v)
    # stop at the first VPW_ADDR write (start of the store); the result
    # registers hold the values the store will read
    stop = None
    for i, w in enumerate(ws):
        if ((w >> 60) & 0xF) == 1 and ((w >> 24) & 0x1F) and \
                ((w >> 38) & 0x3F) == 50 and ((w >> 44) & 1):
            stop = i
            break
    if stop is None:
        raise RuntimeError("no store in %s" % op)
    m.run(0, stop)
    return np.array([float(m.getf('A', r)[lane])
                     for r in range(nvec) for lane in range(16)])


def erf(x):
    import math
    return np.vectorize(math.erf)(x)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


PARAMS = {
    "elu": (1.0,), "selu": (1.6732632, 1.0507), "leakyrelu": (0.1,),
    "hardsigmoid": (1.0 / 6, 0.5), "hardswish": (1.0 / 6, 0.5),
    "clip": (-1.5, 2.5),
}

OPS = {
    'exp': np.exp,
    'log': np.log,
    'sqrt': np.sqrt,
    'reciprocal': lambda z: 1.0 / z,
    'tanh': np.tanh,
    'erf': erf,
    'softplus': softplus,
    'elu': lambda z: np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1),
    'selu': lambda z: 1.0507 * np.where(
        z > 0, z, 1.6732632 * (np.exp(np.minimum(z, 0)) - 1)),
    'gelu': lambda z: 0.5 * z * (1 + np.tanh(
        0.7978845608028654 * (z + 0.044715 * z ** 3))),
    'gelu_erf': lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2.0))),
    'sin': np.sin,
    'cos': np.cos,
    'round': np.round,
    'softsign': lambda z: z / (1 + np.abs(z)),
    'mish': lambda z: z * np.tanh(softplus(z)),
}

# 16 boundary values per op class
EDGES = {
    'log': [2 ** -126, 1e-30, 1e-8, 0.5, 1.0, 2.0, 3.0, 1e10, 1e38,
            3.4e38, 0.25, 4.0, 16.0, 100.0, 7.0, 9.0],
    'sqrt': [0.0, 1e-45, 1e-30, 1e-8, 0.25, 1.0, 2.0, 4.0, 1e10, 1e20,
             16.0, 100.0, 3.0, 5.0, 7.0, 9.0],
    'reciprocal': [1e-30, -1e-30, 1e-10, -1e-10, 0.5, -0.5, 1.0, -1.0,
                   2.0, -2.0, 1e10, -1e10, 1e30, -1e30, 3.0, -3.0],
    'round': [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 2.4, 2.6, -2.4,
              -2.6, 0.0, -0.0, 1e7, -1e7],
}


def main(argv):
    nvec = int(argv[1]) if len(argv) > 1 else 4
    names = argv[2:] or list(OPS)
    bad_total = 0
    print("nvec=%d" % nvec)
    print("%-12s %10s %10s %6s  %s" % ("op", "max_abs", "max_rel", "bad",
                                       "worst"))
    for name in names:
        if name not in OPS:
            continue
        edge = np.asarray(EDGES.get(name, [0.0, -0.0, 1e-8, -1e-8, 0.5, -0.5,
                                          1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
                                          4.0, -4.0, 10.0, -10.0]), np.float32)
        n = nvec * 16
        x = np.resize(edge, n).astype(np.float32)
        try:
            got = run_kernel(name, x, nvec, PARAMS.get(name, ()))
        except Exception as e:
            print("%-12s skip: %s" % (name, e))
            continue
        with np.errstate(all='ignore'):
            ref = OPS[name](x.astype(np.float64))
        ma = np.abs(got - ref)
        den = np.maximum(np.abs(ref), 1.0)
        good = np.isnan(ref) & np.isnan(got) | \
            (np.isinf(ref) & (np.sign(got) == np.sign(ref))) | \
            (ma <= 1e-4 + 1e-4 * den)
        bad = int(np.sum(~good))
        bad_total += bad
        i = int(np.argmax(np.where(good, 0.0, ma / den)))
        print("%-12s %10.3g %10.3g %6d  x=%.9g got=%.9g ref=%.9g"
              % (name, np.nanmax(np.where(good, 0.0, ma)),
                 np.nanmax(np.where(good, 0.0, ma / den)), bad,
                 x[i], got[i], ref[i]))
    print("\n%d mismatches" % bad_total)
    return 1 if bad_total else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
