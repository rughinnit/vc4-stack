#!/usr/bin/env python3
"""8-row packing kernel: transpose A (row-major NxN) into the k-major panel

    PANEL8[(rb*N + k)*8 + r] = A[rb*8 + r][k]

i.e. the asm3 packer (pack.py) with the 16-row tile replaced by an 8-row tile, so
one 64 B line of the panel holds A[8 rows][k] ++ A[8 rows][k+1] (two k columns).

Same two VPM engines, no ALU data movement:
  1. memory -> VPM : VDR (ID=1), RowLength=16 words, NumberRows=8,
                     memory pitch = N*4 bytes => VPM rows 16q..16q+7 hold 8 rows
                     of an 8x16 strip of A.
  2. VPM -> memory : VDW (ID=2) vertical, Depth=8 Units=8.  Measured on
                     silicon (probe_vdw8c): with Depth=Units=8 the hardware
                     writes 8 consecutive 32 B vectors, vector j = column j of
                     the 8 loaded rows = PANEL8[rb][k0+j][0..7].  (Depth and
                     Units do *not* behave as the spec table suggests here:
                     with non-equal values the vector width follows Depth and
                     the count follows Units, and the vector step is
                     max(64, ...) bytes, which pads; Depth=Units=8 is exact.)

One VPM tile (16 rows) per QPU => at most 4 QPUs.

Uniforms (unchanged protocol, harness supplies 8-row values):
  u0 = N, u1 = this QPU's first-row-block byte offset (rb0*32*N), u2 = A bus addr,
  u3 = 32*N (row-block byte stride), u4 = panel bus addr, u5 = row blocks per QPU,
  u6 = QPU launch index (VPM row base = 16*u6)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qenc import *          # noqa: F401,F403
from qenc import _ops
from gen import Prog
import vpmwords as V

ROWS = 8
VECB = ROWS * 4                 # bytes per output vector (one k of one row block)
STRIPB = 8 * VECB               # bytes per 8-k strip (the VDW output block)
KSTRIP = 8                      # k's per strip

# A file
RA_N, RA_A, RA_P = 16, 17, 18
RA_SRCP, RA_DSTP = 19, 20
RA_SRC, RA_DST = 21, 22
RA_KB, RA_CNT = 23, 24
RA_VDRW, RA_VDWW = 25, 26
RA_VDRS, RA_VDWS, RA_TMP = 28, 29, 27
RA_NCOL = 30
# B file
RB_OFF, RB_ST = 16, 17
RB_64, RB_VECB, RB_NEG1 = 18, 19, 20
RB_T = 21


def add_or(out, a, b, ws=0):
    return dict(add_op=ADD_OR, add_out=out, ws=ws, add_a=a, add_b=b)


def add_(op, out, a, b, ws=0, sf=0):
    return dict(add_op=op, add_out=out, ws=ws, add_a=a, add_b=b, sf=sf)


def build(N, vpitch=1):
    ncols = N // 16
    assert ncols & (ncols - 1) == 0
    p = Prog()

    def e(*ws):
        for w in ws:
            p.emit(_ops(**w) if isinstance(w, dict) else w)

    # ---------------- prologue
    e(add_or(36, imm(1), imm(1)))                       # tmu_noswap (unused)
    e(add_or(RA_N, areg(R_UNIF), areg(R_UNIF)))         # u0 = N
    e(add_or(RB_OFF, ws=1, a=areg(R_UNIF), b=areg(R_UNIF)))   # u1 = rb0*32N
    e(add_or(RA_A, areg(R_UNIF), areg(R_UNIF)))         # u2 = A base
    e(add_or(RB_ST, ws=1, a=areg(R_UNIF), b=areg(R_UNIF)))    # u3 = 32N
    e(add_or(RA_P, areg(R_UNIF), areg(R_UNIF)))         # u4 = panel base
    e(add_or(RA_CNT, areg(R_UNIF), areg(R_UNIF)))       # u5 = row blocks for this QPU
    e(add_or(RA_TMP, areg(R_UNIF), areg(R_UNIF)))       # u6 = launch index q
    e(add_(ADD_SHR, RA_NCOL, areg(RA_N), imm(3)))       # strips = N/8
    e(add_(ADD_SHL, RB_T, ws=1, a=areg(RA_N), b=imm(2)))  # N*4 = VDR memory pitch
    e(nop(), nop())

    # VDR basic: ID=1, VPM row 16*q, vpitch, 8 rows, 16 words/row
    e(add_(ADD_SHL, RB_NEG1, ws=1, a=areg(RA_TMP), b=imm(8)))     # q<<8
    e(ldi(V.vdr_basic(0, vpitch, numrows=ROWS, rowlen=KSTRIP), RA_VDRW))
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VDRW, areg(RA_VDRW), (REGB, RB_NEG1)))
    # VDW vertical: ID=2, row 16*q, depth 8 (VPM rows = words out), units 16
    e(add_(ADD_SHL, RB_NEG1, ws=1, a=areg(RA_TMP), b=imm(11)))    # q<<11
    e(ldi(V.vdw_basic(0, KSTRIP, ROWS, horizontal=0), RA_VDWW))
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VDWW, areg(RA_VDWW), (REGB, RB_NEG1)))
    # stride words (rewritten before every DMA, like the proven store path)
    e(ldi(V.vdr_stride(0) & ~0xFFF, RA_TMP))           # ID nibble only
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VDRS, areg(RA_TMP), (REGB, RB_T)))   # ID=9 | pitch=N*4
    e(ldi(V.vdw_stride(0), RA_VDWS))                   # ID=3, block mode, stride 0
    e(nop(), nop())
    e(ldi(KSTRIP * 4, RB_64, ws=1))            # source k-step (8 words)
    e(ldi(STRIPB, RB_VECB, ws=1))
    e(ldi(0xFFFFFFFF, RB_NEG1, ws=1))
    e(add_(ADD_ADD, RA_SRCP, areg(RA_A), (REGB, RB_OFF)))
    e(add_(ADD_ADD, RA_DSTP, areg(RA_P), (REGB, RB_OFF)))
    e(nop(), nop())

    # ---------------- row-block loop
    p.label('rb')
    e(add_or(RA_KB, areg(RA_NCOL), areg(RA_NCOL)))      # k-tile counter
    e(add_or(RA_SRC, areg(RA_SRCP), areg(RA_SRCP)))
    e(add_or(RA_DST, areg(RA_DSTP), areg(RA_DSTP)))
    e(nop())

    # ---------------- 8x16 strip loop
    p.label('kb')
    e(add_or(W_VPR_SETUP, areg(RA_VDRW), areg(RA_VDRW), ws=0))
    e(add_or(W_VPR_SETUP, areg(RA_VDRS), areg(RA_VDRS), ws=0))
    e(add_or(W_VPR_ADDR, areg(RA_SRC), areg(RA_SRC), ws=0))
    e(add_or(W_NOP, areg(50), areg(50)))               # vpr_wait (A file 50)
    e(add_or(W_VPW_SETUP, areg(RA_VDWW), areg(RA_VDWW), ws=1))
    e(add_or(W_VPW_SETUP, areg(RA_VDWS), areg(RA_VDWS), ws=1))
    e(add_or(W_VPW_ADDR, areg(RA_DST), areg(RA_DST), ws=1))
    e(add_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))   # vpw_wait
    e(add_(ADD_ADD, RA_KB, (REGB, RB_NEG1), areg(RA_KB), sf=1))
    p.branch_to('kb', cond=BR_ALL_Z_CLEAR)
    e(add_(ADD_ADD, RA_SRC, areg(RA_SRC), (REGB, RB_64)))      # delay 1
    e(add_(ADD_ADD, RA_DST, areg(RA_DST), (REGB, RB_VECB)))    # delay 2
    e(nop())                                                   # delay 3

    # ---------------- next row block
    e(add_(ADD_ADD, RA_SRCP, areg(RA_SRCP), (REGB, RB_ST)))
    e(add_(ADD_ADD, RA_DSTP, areg(RA_DSTP), (REGB, RB_ST)))
    e(add_(ADD_ADD, RA_CNT, (REGB, RB_NEG1), areg(RA_CNT), sf=1))
    p.branch_to('rb', cond=BR_ALL_Z_CLEAR)
    e(nop(), nop(), nop())

    e(end_program())
    e(nop(), nop(), nop())
    return p


if __name__ == '__main__':
    ASM = os.path.dirname(os.path.abspath(__file__))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    VP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    p = build(N, vpitch=VP)
    w = p.write(os.path.join(ASM, f'pack8_{N}_vp{VP}.bin'))
    print(f'pack8_{N}_vp{VP}.bin {len(w)} instructions')
    if os.environ.get('LIST'):
        print(listing(w))
