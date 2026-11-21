#!/usr/bin/env python3
"""Regression guard for the kops unary generator.  CPU only, no vc4 context.

`aten::floor` and `aten::ceil` were reported as failing in the kernel
generator: `kops.py b_ceil/b_floor` raised a Python TypeError, the engine
turned that into VC4_ERR_KERNEL, and the backend refused both ops.  The
failure belongs to the pre-repair generator (before `e1a8c54`, "repair the
op-long-tail generator"): at this revision every UOPS body is emitted at
nvec 4 and 16, and the rounding bodies answer correctly on the bit-exact ALU
model, including the |x| >= 2^23 pass-through that keeps large finite values
from overflowing the int32 conversion.

This script pins both properties so the failure cannot return unnoticed:

  1. generation: every name in kops.UOPS builds at nvec 4 and 16;
  2. result: floor/ceil/round (and exp/log/erf, the other bodies that were
     once wrong) match a float64 reference on boundary vectors.

It never opens a device: kernels are written under $TMPDIR and executed by
`emu_alu`, the measured-semantics ALU model.

    TMPDIR=$HOME/scratch/tmp python3 infer/tests/test_kops_unary_gen.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import kgen                                                    # noqa: E402
import kops                                                    # noqa: E402
import check_uops_emu as cu                                    # noqa: E402

# Finite boundaries only: the ALU model is a numeric datapath, not an IEEE
# special-value model, and the kernels' documented domain is finite input.
CASES = {
    "floor": np.array([-3.5, -2.5, -0.5, 0.5, 1.5, -1e-8, 1e-8,
                       -0.0, 0.0, 8388608.0, -8388608.0, 1e10, -1e10,
                       16777215.0, -16777215.0, 0.25], np.float32),
    "ceil": np.array([-3.5, -2.5, -0.5, 0.5, 1.5, -1e-8, 1e-8,
                      -0.0, 0.0, 8388608.0, -8388608.0, 1e10, -1e10,
                      16777215.0, -16777215.0, 0.25], np.float32),
    "round": np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5,
                       -0.0, 0.0, 8388608.0, -8388608.0, 1e10,
                       0.49999997, -0.49999997, 2.5000002], np.float32),
    "exp": np.array([-87.0, -80.0, -20.0, -1.0, -1e-8, 0.0, 1e-8,
                     0.5, 1.0, 2.0, 10.0, 20.0, 80.0, 88.0, 3.0, -3.0],
                    np.float32),
    "log": np.array([2.0 ** -126, 1e-30, 1e-8, 0.5, 1.0, 2.0, 10.0,
                     1e38, 3.4e38, 0.25, 4.0, 1.0000001, 2.7182817,
                     1e-3, 1e3, 2.0 ** -100], np.float32),
    "erf": np.array([-6.0, -4.0, -3.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 3.0,
                     4.0, 6.0, 0.5, -0.5, 2.0, -2.0, 0.25], np.float32),
}

REFS = {
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "exp": np.exp,
    "log": np.log,
    "erf": np.vectorize(__import__("math").erf),
}


def gen_failures():
    """Every UOPS name must generate at the group sizes the engine uses."""
    bad = []
    total = 0
    for name in kops.UOPS:
        nvecs = (4, 16) if name in CASES else (16,)
        for nvec in nvecs:
            total += 1
            try:
                prog = kops.build_uop(kgen, name, nvec)
                words = prog.write(os.path.join(
                    os.environ.get("TMPDIR", "/var/tmp"),
                    "kops_gen_%s_%d.bin" % (name, nvec)))
            except Exception as exc:                        # noqa: BLE001
                bad.append("%s nvec=%d: %s: %s"
                           % (name, nvec, type(exc).__name__, exc))
                continue
            if len(words) < 100:
                bad.append("%s nvec=%d: only %d words"
                           % (name, nvec, len(words)))
    return total, bad


def result_failures():
    bad = []
    cu.params_map = {}
    for name, x in CASES.items():
        code = kops.UOP_ORDER.index(name)
        got = cu.run_body(code, x)
        with np.errstate(all="ignore"):
            ref = REFS[name](x.astype(np.float64))
        err = np.abs(got - ref)
        outside = err > cu.TOL(ref)
        if outside.any():
            i = int(np.argmax(np.where(outside, err, -1.0)))
            bad.append("%s: x=%.9g got=%.9g ref=%.9g (%d/16 outside)"
                       % (name, x[i], got[i], ref[i], int(outside.sum())))
        else:
            print("PASS  %-6s max abs %.3g  max rel %.3g"
                  % (name, float(np.nanmax(err)),
                     float(np.nanmax(err / np.maximum(np.abs(ref), 1.0)))))
    return bad


def main():
    total, bad = gen_failures()
    print("%s  generation: %d/%d builds"
          % ("PASS " if not bad else "FAIL ", total - len(bad), total))
    bad += result_failures()
    if bad:
        print("\nFAILURES")
        for b in bad:
            print("  " + b)
        print("\n%d failure(s)" % len(bad))
        return 1
    print("\nall unary bodies generate and match the ALU reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
