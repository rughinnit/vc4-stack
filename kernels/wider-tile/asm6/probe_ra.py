#!/usr/bin/env python3
"""asm6 probe: can the VPM write port read A-file r20..r23 individually, and can
an A-file value reach vpw_addr at all?

 p1: ldi distinct constants into ra20..ra23 -> VPM rows 0..3 -> DMA to u1+0x1000
 p2: `or ra21, ra20, ra20` then re-dump                        -> u1+0x2000
 p3: ldi 0x43210000 into ra21 then dump                       -> u1+0x3000
 p4: DMA to rb31 (u1 itself, offset 0)                        -> C rows 0..3
Values: 1.0000001 / 2.0000001 / 3.0000001 / 4.0000001 / 5.0x.. so rows are
unambiguous.
"""
import os
import sys

ASM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ASM)
from qenc import *          # noqa: F401,F403
from gemm4 import Asm, or_inst, addi

N = int(os.environ.get('N', '64'))
stride = (N - 16) * 4
DMA = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)

RA_U0, RA_U2, RA_U3, RA_U4, RA_U5, RA_U6 = 24, 25, 26, 28, 29, 30
RA_D0, RA_D1, RA_D2 = 31, 23, 22      # destinations (A file)
RA_O1, RA_O2, RA_O3 = 16, 17, 18      # offset constants (A file)
RA_VPMQ = 27


def chunk(a, dst, vbase, reg, label, rows=4, same=True):
    # dst is a register descriptor (areg/breg)
    a.ldi(0x1A00, R5, ws=1)
    a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1))
    for r in range(rows):
        if not same:
            a.ldi(vbase + r, reg, ws=0)
            a.nop()
        a.word(add=or_inst(W_VPM, areg(reg), areg(reg)), note='%s r%d' % (label, r))
        if same:
            pass
    a.ldi(DMA, R5, ws=1)
    a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(3), ws=1))
    a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1)
    a.word(add=or_inst(W_VPW_ADDR, dst, dst, ws=1))
    a.word(add=or_inst(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
    a.nop(), a.nop(), a.nop(), a.nop()


def chunk_regs(a, dst, regs, label):
    a.ldi(0x1A00, R5, ws=1)
    a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1))
    for r, reg in enumerate(regs):
        a.word(add=or_inst(W_VPM, reg, reg), note='%s r%d' % (label, r))
    a.ldi(DMA, R5, ws=1)
    a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(3), ws=1))
    a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1)
    a.word(add=or_inst(W_VPW_ADDR, dst, dst, ws=1))
    a.word(add=or_inst(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
    a.nop(), a.nop(), a.nop(), a.nop()


def build():
    a = Asm()
    a.word(add=or_inst(36, imm(1), imm(1)))
    a.word(add=or_inst(RA_U0, areg(R_UNIF), areg(R_UNIF)))            # u0
    a.word(add=or_inst(15, areg(R_UNIF), areg(R_UNIF), ws=1))         # u1 -> rb15
    a.word(add=or_inst(RA_U2, areg(R_UNIF), areg(R_UNIF)))            # u2
    a.word(add=or_inst(RA_U3, areg(R_UNIF), areg(R_UNIF)))            # u3
    a.word(add=or_inst(RA_U4, areg(R_UNIF), areg(R_UNIF)))            # u4
    a.word(add=or_inst(RA_U5, areg(R_UNIF), areg(R_UNIF)))            # u5
    a.word(add=or_inst(RA_U6, areg(R_UNIF), areg(R_UNIF)))            # u6
    a.word(add=or_inst(RA_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)))
    a.nop()
    a.word(add=addi(ADD_SHL, RA_VPMQ, areg(RA_VPMQ), imm(2)))         # 4*qpu
    a.word(add=addi(ADD_SHL, 32 + 3, areg(R_QPU_NUM), imm(2)))        # r3 = qpu*4
    a.ldi(0x1000, RA_O1, ws=0)                                        # ra16 = 0x1000
    a.ldi(0x2000, RA_O2, ws=0)
    a.ldi(0x3000, RA_O3, ws=0)
    a.nop()
    a.word(add=addi(ADD_ADD, RA_D0, breg(15), areg(RA_O1)))           # +0x1000
    a.word(add=addi(ADD_ADD, RA_D1, breg(15), areg(RA_O2)))           # +0x2000
    a.word(add=addi(ADD_ADD, RA_D2, breg(15), areg(RA_O3)))           # +0x3000
    a.nop(), a.nop()

    # p1: distinct ldis into ra20..ra23
    a.ldi(0x3F800001, 20, ws=0)
    a.ldi(0x40000001, 21, ws=0)
    a.ldi(0x40800001, 22, ws=0)
    a.ldi(0x41000001, 23, ws=0)
    a.nop()
    chunk_regs(a, areg(RA_D0), tuple(areg(r) for r in (20, 21, 22, 23)), 'p1')

    # p2: ra21 = ra20
    a.word(add=or_inst(21, areg(20), areg(20)), note='ra21 = ra20')
    a.nop(), a.nop()
    chunk_regs(a, areg(RA_D1), tuple(areg(r) for r in (20, 21, 22, 23)), 'p2')

    # p3: ra21 = a fresh ldi
    a.ldi(0x43210001, 21, ws=0)
    a.nop()
    chunk(a, areg(RA_D2), 0x43210001, 21, 'p3', rows=4, same=True)

    # p4: destination = u1 itself (B-file reg 15)
    chunk(a, breg(15), 0x41800001, 20, 'p4', rows=4, same=True)
    a.word(sig=SIG_END_PROGRAM)
    a.nop(), a.nop(), a.nop()
    return a


if __name__ == '__main__':
    a = build()
    a.check()
    ws = a.write(os.path.join(ASM, 'probe_ra.bin'))
    print('probe_ra.bin %d instructions' % len(ws))
    if os.environ.get('LIST'):
        print(a.listing())
