#!/usr/bin/env python3
"""The autotuner's search space, checked without a board.

CPU only: no vc4 context is opened, no blob is launched.  Three properties:

  1. the space the Python mirror (vc4infer/kgen/autotune.py) enumerates is the
     space the C library enumerates - compared line for line against
     `build/test_autotune --space` and `--refine` when that binary is built;
  2. every candidate is a launch the generator can actually build: each one is
     emitted through kgen and its instruction count must fit the 2048-word
     code slot.  This is the instruction-slot check; the 12-QPU ceiling and the
     row-block partition are the legality filter's job;
  3. the legality filter refuses what the kernels cannot run: more QPUs than
     the hardware has, more than there are row blocks, an auxiliary split that
     does not divide the work, and the other op's field set.

    TMPDIR=$HOME/scratch/tmp python3 infer/tests/test_autotune_space.py
"""
import ctypes
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KGEN = os.path.join(HERE, "..", "python", "vc4infer", "kgen")
# the tree's package first: a wheel installed in site-packages may predate
# TuneShape, and the mirror must be the one this checkout builds
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, KGEN)

import autotune as at                                          # noqa: E402
import kgen                                                    # noqa: E402

CODE_SLOT_WORDS = 2048          # VC4_CODE_SLOT in examine/src/vc4_int.h
BINARY = os.path.join(HERE, "..", "build", "test_autotune")


def words(builder, *args, **kw):
    p = builder(*args, **kw)
    # Prog.finish() resolves branches; G10Asm.words() encodes its dict form
    if hasattr(p, "finish"):
        return len(p.finish())
    return len(p.words())


def gen_ok(op, shape, params):
    """Generate the candidate's blob; return its instruction count."""
    if op == at.MATMUL:
        return words(kgen.build_gemm10, shape["m"], shape["n"], shape["k"],
                     nqpu=params["gemm_q"])
    return words(kgen.build_gemm, shape["m"], shape["n"], shape["k"],
                 nqpu=params["gemm_q"], bmode=1)


def check(cond, what):
    print("%-4s %s" % ("PASS" if cond else "FAIL", what))
    return 0 if cond else 1


def field_args(shape):
    return [str(shape[k]) for k in at.SHAPE_KEYS]


def c_space(op, shape):
    out = subprocess.run([BINARY, "--space", op] + field_args(shape),
                         check=True, capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        p, g, a = (int(x) for x in line.split())
        rows.append({"pack_q": p, "gemm_q": g, "im2col_q": a})
    return rows


def c_refine(op, shape, best):
    out = subprocess.run([BINARY, "--refine", op] + field_args(shape) +
                         [str(best["pack_q"]), str(best["gemm_q"]),
                          str(best["im2col_q"])],
                         check=True, capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        p, g, a = (int(x) for x in line.split())
        rows.append({"pack_q": p, "gemm_q": g, "im2col_q": a})
    return rows


def main():
    bad = 0
    cases = [
        (at.MATMUL, at.matmul_shape(128, 128, 128)),
        (at.MATMUL, at.matmul_shape(256, 256, 256)),
        (at.MATMUL, at.matmul_shape(512, 512, 512)),
        (at.MATMUL, at.matmul_shape(64, 64, 16)),
        (at.MATMUL, at.matmul_shape(16, 64, 16)),
        (at.CONV2D, at.conv_shape(16, 32, 32, 32, 3, 3, 1)),
        (at.CONV2D, at.conv_shape(16, 16, 6, 14, 3, 3, 1)),
        (at.CONV2D, at.conv_shape(32, 16, 16, 16, 1, 1, 0)),
    ]

    for op, shape in cases:
        space = at.space(op, shape)
        bad += check(len(space) > 0 and space[0] == at.defaults(op, shape),
                     "%s: the default launch is the first candidate"
                     % at.describe(op, shape))

        # every candidate is legal, unique, and generates within the slot
        legal = all(at.legal(op, shape, c) for c in space)
        unique = len({tuple(sorted(c.items())) for c in space}) == len(space)
        bad += check(legal, "%s: every candidate is a legal launch"
                     % at.describe(op, shape))
        bad += check(unique, "%s: no duplicate candidate"
                     % at.describe(op, shape))
        counts = []
        for c in space:
            try:
                n = gen_ok(op, shape, c)
            except AssertionError as e:
                counts.append((c, "refused: %s" % e))
                continue
            counts.append((c, n))
        over = [t for t in counts if isinstance(t[1], str)]
        too_big = [t for t in counts
                   if isinstance(t[1], int) and t[1] > CODE_SLOT_WORDS]
        bad += check(not over, "%s: the generator builds every candidate"
                     % at.describe(op, shape))
        for c, e in over:
            print("     %s" % e)
        bad += check(not too_big, "%s: every blob fits %d words"
                     % (at.describe(op, shape), CODE_SLOT_WORDS))

        # the fine pass stays near the winner and adds nothing measured
        seen = space
        for best in (space[0], space[len(space) // 2]):
            fine = at.refine(op, shape, best, seen)
            ok = all(at.legal(op, shape, c) and c not in seen for c in fine)
            bad += check(ok, "%s: refine around gemm_q=%d is legal and new"
                         % (at.describe(op, shape), best["gemm_q"]))

    # ---- the C library enumerates the same space --------------------------
    if os.path.exists(BINARY):
        for op, shape in cases:
            py = at.space(op, shape)
            c = c_space(op, shape)
            bad += check(py == c, "%s: C and Python coarse spaces agree"
                         % at.describe(op, shape))
            for best in (py[0], py[-1]):
                pr = at.refine(op, shape, best, py)
                cr = c_refine(op, shape, best)
                bad += check(pr == cr, "%s: C and Python refine agree at "
                             "gemm_q=%d" % (at.describe(op, shape),
                                            best["gemm_q"]))
    else:
        print("SKIP %s is not built; the C/Python space cross-check did not run"
              % os.path.relpath(BINARY))

    # ---- the filter refuses what the kernels cannot run -------------------
    s = at.matmul_shape(128, 128, 128)
    bad += check(not at.legal(at.MATMUL, s,
                              {"pack_q": 4, "gemm_q": 13, "im2col_q": 0}),
                 "13 QPUs is refused (the board has 12)")
    bad += check(not at.legal(at.MATMUL, s,
                              {"pack_q": 3, "gemm_q": 12, "im2col_q": 0}),
                 "a packer split that does not divide 16 row blocks is refused")
    bad += check(not at.legal(at.MATMUL, s,
                              {"pack_q": 4, "gemm_q": 12, "im2col_q": 4}),
                 "the other op's field set is refused")
    s2 = at.matmul_shape(16, 64, 16)
    bad += check(not at.legal(at.MATMUL, s2,
                              {"pack_q": 2, "gemm_q": 3, "im2col_q": 0}),
                 "more QPUs than row blocks is refused")
    s3 = at.conv_shape(16, 16, 6, 14, 3, 3, 1)
    bad += check(not at.legal(at.CONV2D, s3,
                              {"pack_q": 0, "gemm_q": 6, "im2col_q": 4}),
                 "an im2col split that does not divide 6 row blocks is refused")
    bad += check(at.legal(at.CONV2D, s3,
                          {"pack_q": 0, "gemm_q": 6, "im2col_q": 3}),
                 "im2col_q=3 on 6 row blocks is a launch (3 divides 6)")

    # ---- the C structs and the ctypes mirror have one layout --------------
    layout = (48, 12, 136, 52)
    if os.path.exists(BINARY):
        got = tuple(int(x) for x in subprocess.run(
            [BINARY, "--layout"], check=True, capture_output=True,
            text=True).stdout.split())
        bad += check(got == layout,
                     "C layout (shape, params, result, kernel@) is %r" %
                     (got,))
    try:
        from vc4infer import _binding
        mirror = (ctypes.sizeof(_binding.TuneShape),
                  ctypes.sizeof(_binding.TuneParams),
                  ctypes.sizeof(_binding.TuneResult),
                  _binding.TuneResult.kernel.offset)
        bad += check(mirror == layout,
                     "the ctypes mirror matches the C layout %r" % (mirror,))
    except Exception as e:                      # library not built
        print("SKIP the ctypes layout check did not run: %s" % e)

    print("test_autotune_space: %s" % ("PASS" if bad == 0 else
                                       "%d FAILURE(S)" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
