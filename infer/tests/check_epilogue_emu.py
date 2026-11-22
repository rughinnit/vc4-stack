#!/usr/bin/env python3
"""check_epilogue_emu - run the GEMM activation epilogue bodies on the ALU model.

`kops.act_apply` is what `kgen.build_gemm` emits between the last k iteration
and the VPM store, so a wrong body here is a wrong conv, not a wrong number in
an elementwise op.  This executes those exact words -- constants included --
for every activation the generator offers, on the bit-exact ALU model
(round toward zero, denormals flushed), and compares against a float64
reference.

No GPU is used.

    python3 infer/tests/check_epilogue_emu.py [act ...]
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "tools"))
sys.path.insert(0, _here)
import kgen                                                     # noqa: E402
import kops                                                     # noqa: E402
import emu_alu                                                  # noqa: E402

NREG = 16              # the asm3 GEMM's C tile: 16 rows x 16 columns
TEMPS = (26, 27)       # A_AADR / A_BADR, dead between the k loop and the store
SUPPORTED = (kops.ACT_RELU,)

REF = {
    kops.ACT_RELU: lambda x: np.maximum(x, 0.0),
}


def body_words(act):
    """The activation's constants and body words, exactly as build_gemm emits."""
    load = kgen.Builder()
    A = kops.act_setup(kgen, load, act, list(range(0, 19)), TEMPS)
    b = kgen.Builder()
    kops.act_apply(b, A, list(range(NREG)))
    return A, b.p.w


def run(act, x):
    A, words = body_words(act)
    m = emu_alu.Machine(words)
    xs = np.asarray(x, np.float32).reshape(NREG, 16)
    for r in range(NREG):
        m.setf('A', r, xs[r])
    for name, reg in A.consts.items():
        m.seti('B', reg - 200, kops.STD[name][0])
    m.run(0, len(words))
    return np.stack([m.getf('A', r) for r in range(NREG)])


def main(argv):
    want = argv[1:]
    acts = [a for a in SUPPORTED if not want or kops.ACT_NAMES[a] in want]
    if not acts:
        print("nothing to check for %s" % (want,))
        return 2
    rng = np.random.default_rng(0)
    # body-sized chunks: random values, then the edges the bodies special-case
    # (both signed zeros, denormals, a large negative and positive, +/-inf)
    chunks = [rng.standard_normal(NREG * 16).astype(np.float32),
              np.tile(np.array([0.0, -0.0, 1e-40, -1e-40, -1e30, 1e30, -3.5,
                                3.5], np.float32), NREG * 2)]
    bad = 0
    for act in acts:
        for chunk in chunks:
            got = run(act, chunk)
            ref = REF[act](chunk).reshape(NREG, 16)
            d = np.abs(got - ref)
            tol = 1e-4 + 1e-4 * np.maximum(np.abs(ref), 1.0)
            n = int((d > tol).sum())
            bad += n
            print("%-5s %3d vectors  max|err| %.3g  over_tol %d"
                  % (kops.ACT_NAMES[act], NREG, float(d.max()), n))
    if bad:
        print("FAIL: %d element(s) outside tolerance" % bad)
        return 1
    print("PASS epilogue bodies on the ALU model")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
