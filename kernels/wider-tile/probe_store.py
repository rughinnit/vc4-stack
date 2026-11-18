#!/usr/bin/env python3
"""asm5 probe: fully controlled VPM->RAM store, no k loop, no MACs.

Value written for chunk c, row r, lane L is  1000*c + 10*r + L  (float).
C destination for chunk c is (u1) + c*16*N bytes, i.e. C rows 4c..4c+3,
columns 0..15 (a chunk stores 4 rows x 16 columns).

Knobs (env):
  N        gemm N (sets the row stride (N-16)*4), default 128
  NCH      number of chunks, default 8
  VROW     'fixed'  -> VPM row base 4*qpu for every chunk  (asm3/gemm4 style)
           'walk'   -> VPM row base 4*qpu + 4*c
  GAP      extra nops between the 4 vpm writes and the DMA trigger
  DPRD     extra nops after the destination address write (before the wait)
  BASESRC  'a'  -> 'or vpw_addr, ra21, ra21'  (as gemm4)
           'ldi'-> destination loaded with ldi (hardcoded 0xde610000)
  DSTEP    'acc'-> add vpw_setup, r5, r3   (as gemm4)
           'b'  -> add vpw_setup, r5, rb23 (as asm3)
"""
import os
import sys

ASM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ASM)
from qenc import *          # noqa: F401,F403
from gemm4 import Asm, or_inst, addi, RA_VPMQ, RA_CADR, RB_UC, ACC_BADR

N = int(os.environ.get('N', '128'))
NCH = int(os.environ.get('NCH', '8'))
VROW = os.environ.get('VROW', 'fixed')
GAP = int(os.environ.get('GAP', '0'))
DPRD = int(os.environ.get('DPRD', '0'))
BASESRC = os.environ.get('BASESRC', 'a')
DSTEP = os.environ.get('DSTEP', 'acc')

RA_U0, RA_U2, RA_U3, RA_U4, RA_U5, RA_U6 = 24, 25, 26, 27, 28, 29
RA_L, RA_V = 19, 20
RB_K = 30          # scratch (A file 16..23 free in probe)
RB_DSTSTEP = 23


def build():
    a = Asm()
    stride = (N - 16) * 4
    dma = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)
    # seven uniform reads, exactly the gemm4 order (u0..u6)
    a.word(add=or_inst(36, imm(1), imm(1)), note='tmu_noswap')
    a.word(add=or_inst(RA_U0, areg(R_UNIF), areg(R_UNIF)), note='u0 N')
    a.word(add=or_inst(RB_UC, areg(R_UNIF), areg(R_UNIF), ws=1), note='u1 C')
    a.word(add=or_inst(RA_U2, areg(R_UNIF), areg(R_UNIF)), note='u2 A')
    a.word(add=or_inst(RA_U3, areg(R_UNIF), areg(R_UNIF)), note='u3 B')
    a.word(add=or_inst(RA_U4, areg(R_UNIF), areg(R_UNIF)), note='u4 panel')
    a.word(add=or_inst(RA_U5, areg(R_UNIF), areg(R_UNIF)), note='u5 tile0')
    a.word(add=or_inst(RA_U6, areg(R_UNIF), areg(R_UNIF)), note='u6 nqpu')
    a.word(add=or_inst(RA_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)), note='qpu')
    a.nop()
    a.word(add=addi(ADD_SHL, RA_VPMQ, areg(RA_VPMQ), imm(2)), note='4*qpu')
    a.word(add=or_inst(RA_L, areg(R_ELEM_NUM), areg(R_ELEM_NUM)), note='lane')
    a.word(add=or_inst(RA_CADR, breg(RB_UC), breg(RB_UC)), note='CADR = u1')
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_VPMQ), imm(7)), note='r3 = 4q*128')
    a.ldi(16 * N, RB_DSTSTEP, ws=1, note='dest step 4 rows of C')
    a.nop(), a.nop()

    for c in range(NCH):
        a.ldi(0x1A00, R5, ws=1, note='vpw generic')
        if VROW == 'fixed':
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1),
                   note='vpw row base 4q')
        else:
            a.word(add=addi(ADD_ADD, 32 + ACC_BADR, areg(RA_VPMQ), imm(4)),
                   note='row base 4q+4')
            a.nop()
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(ACC_BADR), ws=1),
                   note='vpw row base 4q+4c')
        for r in range(4):
            a.ldi(int(1000 * c + 10 * r), RB_K, ws=1,
                  note='base %d' % (1000 * c + 10 * r))
            a.nop()
            a.word(add=addi(ADD_ADD, RA_V, areg(RA_L), breg(RB_K)), note='+ lane')
            a.word(add=dict(add_op=ADD_ITOF, add_out=RA_V, add_a=areg(RA_V),
                            add_b=areg(RA_V)), note='itof')
            a.word(add=or_inst(W_VPM, areg(RA_V), areg(RA_V)), note='vpm row')
        for _ in range(GAP):
            a.nop()
        a.ldi(dma, R5, ws=1)
        if DSTEP == 'acc':
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(ACC_BADR), ws=1))
        else:
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), breg(RB_DSTSTEP), ws=1))
        a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1)
        if BASESRC == 'a':
            a.word(add=or_inst(W_VPW_ADDR, areg(RA_CADR), areg(RA_CADR), ws=1))
        else:
            a.ldi(0xde610000, 21, ws=0, note='hardcoded C')
            a.word(add=or_inst(W_VPW_ADDR, areg(21), areg(21), ws=1))
        for _ in range(DPRD):
            a.nop()
        a.word(add=or_inst(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
        a.nop(), a.nop(), a.nop(), a.nop()
        a.word(add=addi(ADD_ADD, RA_CADR, areg(RA_CADR), (REGB, RB_DSTSTEP)),
               note='dest += 4 C rows')
    a.word(sig=SIG_END_PROGRAM)
    a.nop(), a.nop(), a.nop()
    return a


if __name__ == '__main__':
    a = build()
    a.check()
    ws = a.write(os.path.join(ASM, 'probe_store.bin'))
    print('probe_store.bin %d instructions  N=%d nch=%d vrow=%s gap=%d dprd=%d base=%s dstep=%s'
          % (len(ws), N, NCH, VROW, GAP, DPRD, BASESRC, DSTEP))
    if os.environ.get('LIST'):
        print(a.listing())
