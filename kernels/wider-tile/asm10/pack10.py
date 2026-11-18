#!/usr/bin/env python3
"""asm10 packer: same panel, fewer and longer VPM DMA commands.

Panel (unchanged, bit-exact):
    PANEL[(rb*N + k)*8 + r] = A[rb*8 + r][k]

Changes vs pack8.py (asm9), all inside the packer:

  * RowLength = KL (8 or 16 words) instead of a hard 8.  KL=16 loads a full
    VPM row (64 B) per A row per command, so every 64 B line of A is read
    exactly once (pack8 read it as two 32 B halves = 2x the read commands)
    and one command covers two k strips.
  * mode='pair' (needs 16 VPM rows/QPU): one VDR loads 16 consecutive A rows
    (= two 8-row panel blocks) x KL words -> one read command per 2 KB of A.
  * VPM row base = VPMROWS*u6, VPMROWS = nrow (8 single / 16 pair), so
    single mode fits 8 QPUs and pair mode 4 QPUs (64 VPM rows).
  * the write side stays on the ONE proven-exact shape (Depth=Units=8, a
    contiguous 32 B x 8 vector = 256 B): the probe showed any other
    Depth/Units pair steps the destination by 64 B and pads.  A KL=16 read
    therefore feeds 2 (single) or 4 (pair) VDWs.

Uniform protocol unchanged (qrun_gemm.c needs no edit): QRUN_PACK_ROWS=8 for
mode single, 16 for mode pair.

  u0 = N, u1 = this QPU's first source byte offset, u2 = A bus addr,
  u3 = one 8-row block byte stride (32*N), u4 = panel bus addr,
  u5 = work units this QPU, u6 = QPU launch index.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qenc import *          # noqa: F401,F403
from qenc import _ops
from gen import Prog
import vpmwords as V

ROWS = 8                        # A rows per 32 B panel vector
# A file
RA_D2, RA_A, RA_P = 16, 17, 18
RA_VD1, RA_VD2, RA_SRC, RA_DST = 19, 20, 21, 22
RA_KB, RA_CNT, RA_VDRW = 23, 24, 25
RA_VD0, RA_D3, RA_VDRS, RA_VDWS = 26, 27, 28, 29
RA_VD3, RA_D4 = 30, 31
# B file
RB_OFF, RB_ST, RB_KL = 16, 17, 18
RB_VECB, RB_NEG1, RB_UNIT = 19, 20, 22
RB_BLKB, RB_RO, RB_C = 23, 24, 26
#   RB_C = small constant scratch


def add_or(out, a, b, ws=0):
    return dict(add_op=ADD_OR, add_out=out, ws=ws, add_a=a, add_b=b)


def add_(op, out, a, b, ws=0, sf=0):
    return dict(add_op=op, add_out=out, ws=ws, add_a=a, add_b=b, sf=sf)


def build(N, kl=16, mode='single', vpitch=1, wwait=True,
          ronly=False, wonly=False, cpitch=False, ustep=False):
    assert N % ROWS == 0 and kl in (8, 16) and N % kl == 0
    pair = (mode == 'pair')
    nrow = 2 * ROWS if pair else ROWS        # A rows per VDR command
    kiter = N // kl
    le1 = nrow.bit_length() - 1              # log2(nrow)
    srcstep = kl * 4                         # bytes per k iteration (one A row)
    dststep = kl * ROWS * 4                  # bytes per k iteration in the panel
    unitb = nrow * N * 4                     # source == dest bytes per work unit
    vdws = [(g, c) for g in range(nrow // ROWS) for c in range(kl // 8)]
    vdwreg = [RA_VD0, RA_VD1, RA_VD2, RA_VD3][:len(vdws)]
    dstreg = [RA_DST, RA_D2, RA_D3, RA_D4][:len(vdws)]
    assert len(vdws) <= 4
    p = Prog()

    def e(*ws):
        for w in ws:
            p.emit(_ops(**w) if isinstance(w, dict) else w)

    # ---------------- prologue -------------------------------------------
    e(add_or(36, imm(1), imm(1)))                                # tmu_noswap
    e(add_(ADD_SHL, RB_ST, ws=1, a=areg(R_UNIF), b=imm(2)))      # u0 -> N*4 pitch
    e(add_or(RB_OFF, ws=1, a=areg(R_UNIF), b=areg(R_UNIF)))      # u1 = byte offset
    e(add_or(RA_A, areg(R_UNIF), areg(R_UNIF)))                  # u2 = A base
    e(add_or(RB_KL, ws=1, a=areg(R_UNIF), b=areg(R_UNIF)))       # u3 (unused)
    e(add_or(RA_P, areg(R_UNIF), areg(R_UNIF)))                  # u4 = panel base
    e(add_or(RA_CNT, areg(R_UNIF), areg(R_UNIF)))                # u5 = units
    e(add_or(RA_D3, areg(R_UNIF), areg(R_UNIF)))                 # u6 = QPU index
    e(add_or(RB_UNIT, ws=1, a=areg(R_UNIF), b=areg(R_UNIF)))     # u7 = unit step

    # VDR basic: ID=1, row base nrow*q (row field at bit 4), nrow rows, KL words
    e(add_(ADD_SHL, RB_RO, ws=1, a=areg(RA_D3), b=imm(le1 + 4)))
    e(ldi(V.vdr_basic(0, vpitch, numrows=nrow, rowlen=kl), RA_VDRW))
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VDRW, areg(RA_VDRW), (REGB, RB_RO)))
    # VDW (one proven shape: depth 8 words = 32 B, units 8 = 256 B contiguous)
    e(add_(ADD_SHL, RB_RO, ws=1, a=areg(RA_D3), b=imm(le1 + 7)))
    e(ldi(V.vdw_basic(0, ROWS, 8, horizontal=0, col=0), RA_VD0))
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VD0, areg(RA_VD0), (REGB, RB_RO)))
    for i, (g, c) in enumerate(vdws):
        if i == 0:
            continue                     # RA_VD0 is the (0,0) setup already
        e(ldi(g * 1024 + c * 64, RB_C, ws=1))    # +8 VPM rows / +8 words (col)
        e(nop(), nop())
        e(add_(ADD_ADD, vdwreg[i], areg(RA_VD0), (REGB, RB_C)))
    # strides
    if cpitch:
        e(ldi(kl * 4, RB_ST, ws=1))                         # contiguous probe
    e(ldi(V.vdr_stride(0) & ~0xFFF, RA_KB))                 # ID nibble only
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VDRS, areg(RA_KB), (REGB, RB_ST)))   # ID=9 | pitch
    e(ldi(V.vdw_stride(0), RA_VDWS))                        # ID=3, block mode
    e(ldi(srcstep, RB_KL, ws=1))
    e(ldi(dststep, RB_VECB, ws=1))
    if not ustep:
        e(ldi(unitb, RB_UNIT, ws=1))
    e(ldi(N * ROWS * 4, RB_BLKB, ws=1))                     # one panel block
    e(ldi(0xFFFFFFFF, RB_NEG1, ws=1))                       # -1
    e(ldi(256, RB_C, ws=1))                                 # 8 k columns of panel
    e(nop(), nop())

    # ---------------- work-unit loop (an 8-row block, or a pair of them) ---
    p.label('rb')
    e(ldi(kiter, RA_KB))                                    # N/KL k-strips
    e(add_(ADD_ADD, RA_SRC, areg(RA_A), (REGB, RB_OFF)))
    e(add_(ADD_ADD, RA_DST, areg(RA_P), (REGB, RB_OFF)))
    e(nop(), nop())

    # ---------------- KL-word strip loop -----------------------------------
    p.label('kb')
    if not wonly:
        e(add_or(W_VPR_SETUP, areg(RA_VDRW), areg(RA_VDRW), ws=0))
        e(add_or(W_VPR_SETUP, areg(RA_VDRS), areg(RA_VDRS), ws=0))
        e(add_or(W_VPR_ADDR, areg(RA_SRC), areg(RA_SRC), ws=0))
        e(add_or(W_NOP, areg(50), areg(50)))                    # vpr_wait
    # per-iteration panel destinations (each VDW writes 256 B)
    if len(vdws) > 1:
        e(add_(ADD_ADD, RA_D2, areg(RA_DST), (REGB, RB_C)))     # +8 k
        if len(vdws) > 2:
            e(add_(ADD_ADD, RA_D3, areg(RA_DST), (REGB, RB_BLKB)))   # + one block
            e(nop())
            e(add_(ADD_ADD, RA_D4, areg(RA_D3), (REGB, RB_C)))
    for i in range(0 if ronly else len(vdws)):
        e(add_or(W_VPW_SETUP, areg(vdwreg[i]), areg(vdwreg[i]), ws=1))
        e(add_or(W_VPW_SETUP, areg(RA_VDWS), areg(RA_VDWS), ws=1))
        e(add_or(W_VPW_ADDR, areg(dstreg[i]), areg(dstreg[i]), ws=1))
        if wwait:
            e(add_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))  # vpw_wait
    if not wwait and not ronly:
        e(add_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
    e(add_(ADD_ADD, RA_KB, (REGB, RB_NEG1), areg(RA_KB), sf=1))  # KB -= 1
    p.branch_to('kb', cond=BR_ALL_Z_CLEAR)
    e(add_(ADD_ADD, RA_SRC, areg(RA_SRC), (REGB, RB_KL)))       # delay 1
    e(add_(ADD_ADD, RA_DST, areg(RA_DST), (REGB, RB_VECB)))     # delay 2
    e(nop())                                                    # delay 3

    # ---------------- next work unit ---------------------------------------
    e(add_(ADD_ADD, RA_CNT, (REGB, RB_NEG1), areg(RA_CNT), sf=1))   # CNT -= 1
    p.branch_to('rb', cond=BR_ALL_Z_CLEAR)
    e(add_(ADD_ADD, RA_A, areg(RA_A), (REGB, RB_UNIT)))          # delay 1
    e(add_(ADD_ADD, RA_P, areg(RA_P), (REGB, RB_UNIT)))          # delay 2
    e(nop())                                                     # delay 3

    e(end_program())
    e(nop(), nop(), nop())
    return p, nrow, len(vdws)


if __name__ == '__main__':
    ASM = os.path.dirname(os.path.abspath(__file__))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    kl = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    mode = sys.argv[3] if len(sys.argv) > 3 else 'single'
    VP = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ww = os.environ.get('NWW') is None
    ro = os.environ.get('RONLY') is not None
    wo = os.environ.get('WONLY') is not None
    cp = os.environ.get('CPITCH') is not None
    us = os.environ.get('USTRIDE') is not None
    p, nrow, nvdw = build(N, kl, mode, vpitch=VP, wwait=ww, ronly=ro, wonly=wo,
                          cpitch=cp, ustep=us)
    flat = ("" if ww else "_now") + ("_ro" if ro else "") + ("_wo" if wo else "") + ("_cp" if cp else "") + ("_us" if us else "")
    name = "pack10_%d_k%d_%s_vp%d%s.bin" % (N, kl, mode, VP, flat)
    w = p.write(os.path.join(ASM, name))
    print('%s %d instructions (%d A rows/VDR, %d VDW per VDR)'
          % (name, len(w), nrow, nvdw))
    if os.environ.get('LIST'):
        print(listing(w))
