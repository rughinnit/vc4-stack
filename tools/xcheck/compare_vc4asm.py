#!/usr/bin/env python3
"""Reverse cross-check: assemble a fixed instruction list with maazl/vc4asm and
compare the bytes with what asm10/qenc.py emits for the same operation.

    python3 compare_vc4asm.py --qenc-dir /home/raspi/work/asm10 \
        --vc4asm /home/raspi/work/xcheck/vc4asm/build/vc4asm

Requires vc4asm (https://github.com/maazl/vc4asm).  The list covers the shapes
our kernels actually emit: register-file A/B reads, special reads/writes, small
immediates, LDI, signals, end, dual add+mul, r5rep/r5quad, and the rotator.
Exit status 0 if every case is bit-exact, 1 otherwise.
"""
import argparse
import os
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qenc-dir", default=os.environ.get("ASM10", "/home/raspi/work/asm10"))
    ap.add_argument("--vc4asm", default="/home/raspi/work/xcheck/vc4asm/build/vc4asm")
    ap.add_argument("--keep", action="store_true", help="keep the generated .qasm/.hex")
    a = ap.parse_args()

    sys.path.insert(0, a.qenc_dir)
    from qenc import (ACC0, ADD_ADD, ADD_NOP, ADD_OR, ADD_SHL, BR_ALL_C_SET,
                      COND_ALWAYS, COND_NEVER, NOP_ADDR, REGB, SIG_ALU_IMMEDIATE,
                      SIG_LOAD_TMU1, W_NOP, acc, alu, areg, branch, breg,
                      end_program, fmul_fadd, imm, ldi, nop, rot_inst)
    from qenc import _ops

    cases = [
        ("or ra1, unif, unif",
         _ops(add_op=ADD_OR, add_out=1, add_a=areg(32), add_b=areg(32)),
         "\tor ra1, unif, unif", 1),
        ("or ra0, qpu_num, qpu_num",
         _ops(add_op=ADD_OR, add_out=0, add_a=breg(38), add_b=breg(38)),
         "\tor ra0, qpu_num, qpu_num", 1),
        ("or vpm, rb0, rb0",
         _ops(add_op=ADD_OR, add_out=48, add_a=breg(0), add_b=breg(0)),
         "\tor vpm, rb0, rb0", 1),
        ("add vw_setup, r5, ra0",
         _ops(add_op=ADD_ADD, add_out=49, ws=1, add_a=acc(5), add_b=areg(0)),
         "\tadd vw_setup, r5, ra0", 1),
        ("add vw_addr, ra4, rb1",
         _ops(add_op=ADD_ADD, add_out=50, ws=1, add_a=areg(4), add_b=breg(1)),
         "\tadd vw_addr, ra4, rb1", 1),
        ("shl r0, ra0, 2",
         _ops(add_op=ADD_SHL, add_out=32, add_a=areg(0), add_b=imm(2)),
         "\tshl r0, ra0, 2", 1),
        ("add r0, unif, r0",
         _ops(add_op=ADD_ADD, add_out=32, add_a=areg(32), add_b=acc(0)),
         "\tadd r0, unif, r0", 1),
        ("add rb0, r4, 1",
         _ops(add_op=ADD_ADD, add_out=0, ws=1, add_a=acc(4), add_b=imm(1)),
         "\tadd rb0, r4, 1", 1),
        ("ldi vw_setup, 0xc0010000", ldi(0xC0010000, 49, ws=1),
         "\tldi vw_setup, 0xc0010000", 1),
        ("nop", nop(), "\tnop", 1),
        ("nop.load_tmu1",
         alu(sig=SIG_LOAD_TMU1, cond_add=COND_NEVER, cond_mul=COND_NEVER),
         "\tldtmu1", 1),
        ("nop.end", end_program(), "\tthrend", 1),
        ("fadd ra0,ra0,r2; fmul r2,r5,rb16",
         fmul_fadd(mul_out=34, mul_a=acc(5), mul_b=breg(16), add_out=0,
                   add_a=areg(0), add_b=acc(2)),
         "\tfadd ra0, ra0, r2; fmul r2, r5, rb16", 1),
        ("rep ws=1 (r5rep)",
         _ops(add_op=ADD_OR, add_out=37, ws=1, add_a=acc(1), add_b=acc(1)),
         "\tmov r5rep, r1", 1),
        ("rep ws=0 (r5quad)",
         _ops(add_op=ADD_OR, add_out=37, ws=0, add_a=acc(1), add_b=acc(1)),
         "\tmov r5quad, r1", 1),
        ("mov t0s, r3",
         _ops(add_op=ADD_OR, add_out=56, add_a=acc(3), add_b=acc(3)),
         "\tmov t0s, r3", 1),
        ("branch +2 words",
         branch(16, cond=BR_ALL_C_SET),
         "\tbrr.allc -, r:tgt\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\ntgt:\tnop", 7),
    ]
    # every rotate form gemm6 emits: 0 is the plain move, 1..15 are the
    # rotator small immediates 49..63 (48 = rotate by r5 is not used).
    # The rotate shares its word with the r5rep broadcast (that is how the
    # kernel uses it), so compare the real two-slot word, not a standalone
    # rotate whose add-slot condition would be synthetic.
    rep = dict(add_op=ADD_OR, add_out=37, ws=1, add_a=acc(1), add_b=acc(1))
    cases += [
        ("r5rep + rot%d" % k,
         _ops(**{**rot_inst(k, src=0, out=1), **rep}),
         ("\tmov r5rep, r1; v8min r1, r0, r0" if k == 0
          else "\tmov r5rep, r1; mov r1, r0>>%d" % k), 1)
        for k in range(16)
    ]

    d = os.path.dirname(os.path.abspath(__file__))
    qasm = os.path.join(d, "compare_vc4asm.qasm")
    hexf = os.path.join(d, "compare_vc4asm.hex")
    with open(qasm, "w") as f:
        f.write("\n".join(c[2] for c in cases) + "\n")
    r = subprocess.run([a.vc4asm, "-C", hexf, qasm],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr, sep="\n")
        return 2

    hexes = []
    for line in open(hexf):
        v = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", line)]
        if len(v) >= 2:
            hexes.append((v[0], v[1]))
    if len(hexes) != sum(c[3] for c in cases):
        print("vc4asm produced %d words, expected %d"
              % (len(hexes), sum(c[3] for c in cases)))
        return 2

    bad = 0
    off = 0
    for name, qw, _, nw in cases:
        lo, hi = hexes[off]
        off += nw
        qlo, qhi = qw & 0xFFFFFFFF, (qw >> 32) & 0xFFFFFFFF
        if (qlo, qhi) == (lo, hi):
            print("OK    %s" % name)
        else:
            bad += 1
            print("DIFF  %s" % name)
            print("        qenc   %08x %08x" % (qlo, qhi))
            print("        vc4asm %08x %08x" % (lo, hi))
            print("        xor    %08x %08x" % (qlo ^ lo, qhi ^ hi))
    print("bit-exact %d/%d cases" % (len(cases) - bad, len(cases)))
    if not a.keep:
        for p in (qasm, hexf):
            try:
                os.unlink(p)
            except OSError:
                pass
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
