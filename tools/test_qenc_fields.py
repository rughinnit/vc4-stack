#!/usr/bin/env python3
"""Unit test: qenc pack / unpack / PM field placement.

Reference bit map (vc4-gpu-tools/include/vc4_qpu_defines.h):

    QPU_UNPACK_SHIFT 57, QPU_UNPACK_MASK 0x7   -> 3 bits, 57..59
    QPU_PM_SHIFT     56, QPU_PM_MASK     0x1   -> 1 bit,  56
    QPU_PACK_SHIFT   52, QPU_PACK_MASK   0xf   -> 4 bits, 52..55

The encoder used to place unpack as 4 bits at 56 (so unpack bit 3 landed in
PM) and pack as 5 bits at 52..56 (so pack bit 4 landed in PM and overlapped
unpack).  This test sets non-zero values and checks every bit against the
header positions, and checks that the neighbours (sig at 60..63, cond_add at
49..51, cond_mul at 46..48) are not disturbed.

    python3 test_qenc_fields.py [--qenc-dir DIR]

Exit status 0 on success.
"""
import argparse
import os
import sys

# header truth, not qenc's own constants
H_UNPACK_SHIFT, H_UNPACK_MASK = 57, 0x7
H_PM_SHIFT, H_PM_MASK = 56, 0x1
H_PACK_SHIFT, H_PACK_MASK = 52, 0xF


def field(w, shift, mask):
    return (w >> shift) & mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qenc-dir", default=os.environ.get("ASM10", "/home/raspi/work/asm10"))
    a = ap.parse_args()

    sys.path.insert(0, a.qenc_dir)
    import qenc

    # the encoder's own constants must agree with the header
    for name, want in (("UNPACK_SHIFT", H_UNPACK_SHIFT), ("UNPACK_MASK", H_UNPACK_MASK),
                       ("PM_SHIFT", H_PM_SHIFT), ("PM_MASK", H_PM_MASK),
                       ("PACK_SHIFT", H_PACK_SHIFT), ("PACK_MASK", H_PACK_MASK)):
        got = getattr(qenc, name, None)
        assert got == want, "%s: qenc has %r, header says %r" % (name, got, want)
    print("OK    qenc field constants match vc4_qpu_defines.h")

    fails = 0
    cases = 0
    for unpack in range(8):
        for pack in range(16):
            for pm in (0, 1):
                w = qenc.alu(add_op=qenc.ADD_OR, mul_op=qenc.MUL_NOP,
                             cond_add=qenc.COND_ALWAYS, cond_mul=qenc.COND_NEVER,
                             sig=qenc.SIG_NONE, unpack=unpack, pm=pm, pack=pack)
                cases += 1
                errs = []
                if field(w, H_UNPACK_SHIFT, H_UNPACK_MASK) != unpack:
                    errs.append("unpack %d -> %d"
                                % (unpack, field(w, H_UNPACK_SHIFT, H_UNPACK_MASK)))
                if field(w, H_PM_SHIFT, H_PM_MASK) != pm:
                    errs.append("pm %d -> %d"
                                % (pm, field(w, H_PM_SHIFT, H_PM_MASK)))
                if field(w, H_PACK_SHIFT, H_PACK_MASK) != pack:
                    errs.append("pack %d -> %d"
                                % (pack, field(w, H_PACK_SHIFT, H_PACK_MASK)))
                # no bleed into the neighbouring fields
                if field(w, 60, 0xF) != qenc.SIG_NONE:
                    errs.append("sig corrupted")
                if field(w, 49, 0x7) != qenc.COND_ALWAYS:
                    errs.append("cond_add corrupted")
                if field(w, 46, 0x7) != qenc.COND_NEVER:
                    errs.append("cond_mul corrupted")
                if field(w, 30, 0x1) != 0:
                    errs.append("mul opcode corrupted")
                if errs:
                    fails += 1
                    if fails <= 5:
                        print("FAIL  unpack=%d pm=%d pack=%d: %s"
                              % (unpack, pm, pack, ", ".join(errs)))

    print("%s  %d/%d pack/unpack/PM combinations"
          % ("OK   " if not fails else "FAIL ", cases - fails, cases))

    # a few documented values, spelled out
    w = qenc.alu(add_op=qenc.ADD_OR, unpack=5, pm=1, pack=9,
                 cond_add=qenc.COND_ALWAYS, cond_mul=qenc.COND_NEVER)
    assert (w >> H_UNPACK_SHIFT) & 7 == 5
    assert (w >> H_PM_SHIFT) & 1 == 1
    assert (w >> H_PACK_SHIFT) & 0xF == 9
    print("OK    unpack=5 pm=1 pack=9 -> high word 0x%08x" % (w >> 32))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
