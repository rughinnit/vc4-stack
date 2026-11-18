#!/usr/bin/env python3
"""Store-path test for the asm4 tile geometry: preload the 32 accumulators with
identifiable values and run the exact 8-chunk VPM->RAM store block, no k loop.

  acc(row, j)[lane] = 1000*row + 10*j + lane      (row 0..7, j 0..3)

Expected C[row][j*16+lane] = that value at the C base, i.e.
  C[row][col] with row 0..7, col 0..63.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qenc import *          # noqa: F401,F403
from gemm4 import (Asm, acc_addr, ref, or_inst, addi, NROWS, NCOLS,
                   RA_VPMQ, RA_CADR, RA_CADR2, RB_UC, RB_UP, ACC_BADR)
import gemm4
import struct


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]


def build(N):
    a = Asm()
    stride = (N - 16) * 4
    a.word(add=or_inst(36, imm(1), imm(1)), note='tmu_noswap')
    a.word(add=or_inst(RA_CADR, areg(R_UNIF), areg(R_UNIF)), note='u0')
    a.word(add=or_inst(RB_UC, areg(R_UNIF), areg(R_UNIF), ws=1), note='u1 C')
    a.word(add=or_inst(RA_CADR + 100, areg(R_UNIF), areg(R_UNIF)), note='u2')
    a.word(add=or_inst(RA_CADR + 101, areg(R_UNIF), areg(R_UNIF)), note='u3')
    a.word(add=or_inst(RB_UP, areg(R_UNIF), areg(R_UNIF), ws=1), note='u4 panel')
    a.word(add=or_inst(RA_CADR + 102, areg(R_UNIF), areg(R_UNIF)), note='u5')
    a.word(add=or_inst(RA_CADR + 103, areg(R_UNIF), areg(R_UNIF)), note='u6')
    a.word(add=or_inst(RA_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)), note='qpu')
    a.nop()
    a.word(add=addi(ADD_SHL, RA_VPMQ, areg(RA_VPMQ), imm(2)), note='4*qpu')
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_VPMQ), imm(7)), note='r3 = 4q<<7')
    a.ldi(0x42C80000, 22, ws=1, note='1000.0f scratch')
    a.word(add=or_inst(25, areg(R_ELEM_NUM), areg(R_ELEM_NUM)), note='S1 = lane (int)')
    a.nop(), a.nop()
    a.word(add=or_inst(RA_CADR2, (REGB, RB_UC), (REGB, RB_UC)), note='CADR2 = C')
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_VPMQ), imm(7)), note='r3 = 4q<<7')
    a.nop()
    # preload: acc(row,j) = 1000*row + 10*j + lane
    for r in range(NROWS):
        for j in range(4):
            f, addr = acc_addr(r, j)
            val = 1000.0 * r + 10.0 * j
            a.ldi(int(val), 23, ws=1, note='scratch %d' % int(val)) if val else None
            a.word(add=dict(add_op=ADD_ITOF, add_out=addr, ws=f,
                            add_a=areg(25), add_b=areg(25)),
                   note='itof acc(%d,%d)' % (r, j))
            if val:
                a.word(add=dict(add_op=ADD_FADD, add_out=addr, ws=f,
                                add_a=ref(f, addr), add_b=(REGB, 30)),
                       note='+ %d' % int(val))
    # 30 is RB_ASTEP in gemm4; use a dedicated scratch instead
    return a


def build2(N):
    """same but with a clean scratch register in the B file (rb30 unused here)"""
    a = Asm()
    stride = (N - 16) * 4
    a.word(add=or_inst(36, imm(1), imm(1)), note='tmu_noswap')
    a.word(add=or_inst(24, areg(R_UNIF), areg(R_UNIF)), note='u0 (unused)')
    a.word(add=or_inst(RB_UC, areg(R_UNIF), areg(R_UNIF), ws=1), note='u1 C')
    for k in (2, 3):
        a.word(add=or_inst(24, areg(R_UNIF), areg(R_UNIF)), note='u%d' % k)
    a.word(add=or_inst(RB_UP, areg(R_UNIF), areg(R_UNIF), ws=1), note='u4 panel')
    for k in (5, 6):
        a.word(add=or_inst(24, areg(R_UNIF), areg(R_UNIF)), note='u%d' % k)
    a.word(add=or_inst(RA_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)), note='qpu')
    a.nop()
    a.word(add=addi(ADD_SHL, RA_VPMQ, areg(RA_VPMQ), imm(2)), note='4*qpu')
    a.word(add=or_inst(25, areg(R_ELEM_NUM), areg(R_ELEM_NUM)), note='S1 = lane')
    a.word(add=or_inst(RA_CADR, (REGB, RB_UC), (REGB, RB_UC)), note='CADR = C')
    a.word(add=or_inst(RA_CADR2, (REGB, RB_UC), (REGB, RB_UC)), note='CADR2 = C')
    a.nop()
    for r in range(NROWS):
        for j in range(4):
            f, addr = acc_addr(r, j)
            val = 1000 * r + 10 * j
            a.word(add=dict(add_op=ADD_ITOF, add_out=addr, ws=f,
                            add_a=areg(25), add_b=areg(25)),
                   note='itof acc(%d,%d)' % (r, j))
            if val:
                if f == 0:
                    a.ldi(fbits(val), 30, ws=1, note='scratch %d' % val)
                    src = (REGB, 30)
                else:
                    a.ldi(fbits(val), 24, ws=0, note='scratch %d (A file)' % val)
                    src = areg(24)
                a.word(add=dict(add_op=ADD_FADD, add_out=addr, ws=f,
                                add_a=ref(f, addr), add_b=src),
                       note='+ %d' % val)
    # the store block, identical to gemm4's
    dma = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_VPMQ), imm(7)), note='r3 = 4q<<7')
    a.nop()
    NCH = int(os.environ.get('CHUNKS', '8'))
    done = 0
    for j in range(4):
        for c in range(2):
            if done >= NCH:
                break
            done += 1
            a.ldi(0x1A00, R5, ws=1, note='vpw generic row 4q')
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1))
            for r in range(4):
                f, addr = acc_addr(4 * c + r, j)
                a.word(add=or_inst(W_VPM, ref(f, addr), ref(f, addr)))
            a.ldi(dma, R5, ws=1, note='vdw 4 rows x 16 words')
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(ACC_BADR), ws=1))
            a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1, note='stride')
            base = RA_CADR if c == 0 else RA_CADR2
            a.word(add=or_inst(W_VPW_ADDR, areg(base), areg(base), ws=1))
            a.word(add=or_inst(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
            a.nop(), a.nop(), a.nop(), a.nop()
        if j != 3 and done < NCH:
            a.word(add=addi(ADD_ADD, RA_CADR, areg(RA_CADR), (REGB, 24)),
                   note='next 16 columns') if False else None
            a.ldi(64, 24, ws=1)
            a.word(add=addi(ADD_ADD, RA_CADR, areg(RA_CADR), (REGB, 24)))
            a.word(add=addi(ADD_ADD, RA_CADR2, areg(RA_CADR2), (REGB, 24)))
    a.word(sig=SIG_END_PROGRAM)
    a.nop(), a.nop(), a.nop()
    return a


if __name__ == '__main__':
    ASM = os.path.dirname(os.path.abspath(__file__))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    a = build2(N)
    a.check()
    ws = a.write(os.path.join(ASM, 'test_store4_%d.bin' % N))
    print('test_store4_%d.bin %d instructions' % (N, len(ws)))
    if os.environ.get('LIST'):
        print(a.listing())
