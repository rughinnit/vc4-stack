#!/usr/bin/env python3
"""unif8: dump the 8 uniforms the kernel reads, through a correct store.

Expect C rows 0..3 (cols 0..15 then 16..31) = u0..u7.
"""
import sys
sys.path.insert(0, "/home/raspi/work/engine-fast2/tools")
import kgen
from kgen import (G10Asm, g10_or, g10_addi, g10_store, acc, areg, breg, imm,
                  ADD_ADD, SIG_END_PROGRAM, R_UNIF, W_NOP, R_VPW_WAIT, REGB)

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
a = G10Asm()
a.word(add=g10_or(36, imm(1), imm(1)), note="tmu_noswap")
# u0..u7 -> A r0..r7
for i in range(8):
    a.word(add=g10_or(i, areg(R_UNIF), areg(R_UNIF)), note="u%d -> ra%d" % (i, i))
# CADR = u1, CADR2 = C + 4*N*4  (u1 is in ra1)
a.word(add=g10_or(21, areg(1), areg(1)), note="CADR = u1")
a.ldi(4 * N * 4, 31, ws=1, note="B r31 = 4 C rows")
a.nop()
a.word(add=g10_addi(ADD_ADD, 22, areg(21), (REGB, 31)), note="CADR2")
a.ldi(64, 24, ws=1, note="B r24 = 64 (C column advance)")
# the B-file half of the accumulator rows: distinctive constants
for i in range(16):
    a.ldi(2000 + i, i, ws=1)
a.nop(), a.nop()
g10_store(a, (N - 16) * 4)
a.label("tdone")
a.word(sig=SIG_END_PROGRAM)
a.nop(), a.nop(), a.nop()
a.check()
w = a.write(OUT)
print("unif8: %d instructions -> %s" % (len(w), OUT))
