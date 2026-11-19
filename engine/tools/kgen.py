#!/usr/bin/env python3
"""kgen - QPU kernel generator for libvc4infer.

The GEMM body is the hand-written, element-verified asm3 kernel (16x16 C tile,
packed fmul+fadd, hardware rotate+replicate broadcast, both TMUs 2-deep,
39-word k loop, VPM->DMA store).  This file reuses the asm3 encoder
(qenc.py) and the proven VPM control words (vpmwords.py) unchanged, and only
generalises the *addressing* from square N x N to arbitrary M x N x K, plus two
new kernels: im2col-panel and elementwise.

No instruction encoding is invented here; every sequence is either taken
verbatim from asm3 (gemm2.py / pack.py / gen.py store path) or is a mechanical
register/stride substitution on one of those sequences.

KERNEL ABI (uniforms read in order u0..; 64 B block per QPU = 16 words)

  gemm   u0 panel bus
         u1 C bus
         u2 B bus
         u3 A row-block byte offset   (rb0 * 64*K)
         u4 A row-block byte stride   (nqpu * 64*K)
         u5 C row-block byte offset   (rb0 * 16*N*4)
         u6 C row-block byte stride   (nqpu * 16*N*4)
         u7 row blocks for this QPU
         compile-time: M/16, N/16, K, nqpu, B mode
                       (0 = row-major KxN, 1 = k-major B panel)

  pack   u0 source row length (floats)   u1 source row-block byte offset
         u2 source base bus             u3 source row-block byte stride
         u4 destination base bus        u5 row blocks for this QPU
         u6 QPU index (VPM row 16*q)    u7 destination k-block byte stride
         u8 destination row-block stride u9 k blocks (K/16)

  im2col u0 padded-plane byte pitch      u1 source row-block byte offset
         u2 padded input base bus        u3 row-block byte stride
         u4 panel base bus               u5 row blocks for this QPU
         u6 QPU index                    u7 destination k-block stride (1024)
         u8 destination row-block stride u9 ci blocks (Cin/16)
         u9  Wp*4
         compile-time: Cin, K, KH, KW (taps unrolled)

  ew     u0 a bus, u1 b bus, u2 out bus, u3 groups for this QPU,
         u4 QPU index, u5 nqpu
         a group is 16 vectors = 256 floats = one verified VPM->DMA store
"""
import os
import struct
import sys

ASM3 = os.environ.get("ASM3_DIR", "/home/raspi/work/asm3")
sys.path.insert(0, ASM3)

from qenc import *                                   # noqa: F401,F403
from qenc import _ops, imm as _imm, W_VPM, W_VPW_SETUP, W_VPW_ADDR
from gen import Prog
import vpmwords as V

MUL_V8MIN = 4

# Prog.emit takes either encoded words or the dict form used everywhere below
_orig_emit = Prog.emit


def _emit(self, *ws):
    for w in ws:
        _orig_emit(self, _ops(**w) if isinstance(w, dict) else w)


Prog.emit = _emit


def imm(v):
    """qenc small immediate.  Values 48..63 are the mul-ALU vector-rotate
    encodings and are legal here, so the range is not asserted."""
    return _imm(v)


def imm_small(v):
    """Small immediate for the plain ALU fields: -16..15 only."""
    assert -16 <= v <= 15, "small immediate out of range: %d" % v
    return _imm(v)


def add_or(out, a, b, ws=0):
    return dict(add_op=ADD_OR, add_out=out, ws=ws, add_a=a, add_b=b)


def add_(op, out, a, b, ws=0, sf=0):
    return dict(add_op=op, add_out=out, ws=ws, add_a=a, add_b=b, sf=sf)


def mul_fmul(out_acc, a, b):
    return dict(mul_op=MUL_FMUL, mul_out=32 + out_acc, ws=0, mul_a=a, mul_b=b)


def mul_rot(out_acc, src_acc, n):
    return dict(mul_op=MUL_V8MIN, mul_out=32 + out_acc, ws=0,
                mul_a=acc(src_acc), mul_b=imm(48 + n))


def rep_all(src_acc):
    return add_or(37, acc(src_acc), acc(src_acc), ws=1)


def emit_slot(p, m, a, sig=SIG_NONE):
    if sig != SIG_NONE:
        for d, k in ((m, 'mul_b'), (a, 'add_b')):
            if d and d.get(k) is not None and d[k][0] == 8:
                raise AssertionError('TMU signal cannot share a word with a small immediate')
    kw = dict(sig=sig)
    if m:
        kw.update(m)
    else:
        kw.update(mul_op=MUL_NOP, mul_out=W_NOP)
    if a:
        kw.update(a)
    else:
        kw.update(add_op=ADD_NOP, add_out=W_NOP, cond_add=COND_NEVER)
    if not m:
        kw.setdefault('cond_mul', COND_NEVER)
    p.emit(_ops(**kw))


class Builder:
    """Emits words and checks the QPU's no-forwarding rule.

    The QPU does not forward: a read of a register written in one of the two
    preceding words still sees the old value.  asm3's verified prologues keep
    three words between a write and its first dependent read; this checker
    enforces that on every kernel here except the verified k loop, where the
    intra-word read of the previous product is intentional.
    """

    def __init__(self):
        self.p = Prog()
        self.check = True
        self.last_write = {}
        self.violations = []
        self.idx = 0
        self.nops = 0
        self.last_branch = -99

    @staticmethod
    def _rfile(desc):
        if not desc:
            return None
        mux, addr = desc
        if mux == 8:            # small immediate
            return None
        if mux == REGA:
            return ('A', addr)
        if mux == REGB:
            return ('B', addr)
        return ('ACC', addr)    # 0..5 = accumulators

    def _writes(self, w):
        out = []
        sig = (w >> 60) & 0xF
        if sig == SIG_LOAD_IMMEDIATE:
            reg = (w >> 38) & 0x3F
            ws = (w >> 44) & 1
            if reg < 32:
                out.append(('A' if ws == 0 else 'B', reg))
            elif reg < 38:
                out.append(('ACC', reg - 32))
            return out
        if sig in (SIG_ALU_IMMEDIATE, SIG_NONE, SIG_LOAD_TMU0, SIG_LOAD_TMU1,
                   SIG_LOAD_COLOR, SIG_LOAD_COVERAGE, SIG_WAIT_SCORE):
            ws = (w >> 44) & 1
            aw = (w >> 38) & 0x3F
            mw = (w >> 32) & 0x3F
            if (w >> 24) & 0x1F:                 # non-nop add
                if aw < 32:
                    out.append(('A' if ws == 0 else 'B', aw))
                elif aw < 38:
                    out.append(('ACC', aw - 32))
            if (w >> 29) & 7:                    # non-nop mul
                if mw < 32:
                    out.append(('B' if ws == 0 else 'A', mw))
                elif mw < 38:
                    out.append(('ACC', mw - 32))
        return out

    def _reads(self, w):
        """Registers read by this word.

        The mux, not the operand slot, selects the file: qenc puts any A-file
        operand in raddr_a and any B-file operand in raddr_b, so a B-slot (the
        mul_b field) can address the A file and vice versa.  A word with a
        small immediate has that immediate in raddr_b, not a B-file read.
        """
        out = []
        if (w >> 24) & 0x1F or (w >> 29) & 7:
            ra = (w >> 18) & 0x3F
            rb = (w >> 12) & 0x3F
            is_imm = ((w >> 60) & 0xF) == SIG_ALU_IMMEDIATE
            for mux in ((w >> 9) & 7, (w >> 6) & 7, (w >> 3) & 7, w & 7):
                if mux == REGA:
                    if ra != NOP_ADDR:
                        out.append(('A', ra))
                elif mux == REGB:
                    if rb != NOP_ADDR and not is_imm:
                        out.append(('B', rb))
                elif mux < 6:
                    out.append(('ACC', mux))
        return out

    def _bad(self, word):
        """Return the first (reg, distance) violation for this word, or None.

        QPU general registers have no forwarding: a read needs three words
        after the write.  Accumulators have a forwarding path and may be read
        in the following word (asm3's prog_alu relies on that).  A read in the
        same word that also writes the register still sees the *previous*
        value, so a write from an earlier word must still be checked.
        """
        for r in self._reads(word):
            lw = self.last_write.get(r)
            if lw is None or lw >= self.idx:
                continue
            need = 1 if r[0] == 'ACC' else 3
            if self.idx - lw < need:
                return (r, self.idx - lw)
        return None

    def e(self, *ws):
        for w in ws:
            word = _ops(**w) if isinstance(w, dict) else w
            if self.check:
                # pad with nops so a dependent read is never closer than three
                # words (the QPU has no forwarding).  Never inside a branch's
                # delay slots.
                if self.idx - self.last_branch > 3:
                    for _ in range(4):
                        if self._bad(word) is None:
                            break
                        self.p.emit(nop())
                        self.idx += 1
                        self.nops += 1
                bad = self._bad(word)
                if bad is not None:
                    self.violations.append((self.idx, bad[0], bad[1]))
            self.p.emit(word)
            if self.check:
                for r in self._writes(word):
                    self.last_write[r] = self.idx
            if ((word >> 60) & 0xF) == SIG_BRANCH:
                self.last_branch = self.idx
            self.idx += 1

    def report(self, name):
        """Residual violations after padding (should always be 0)."""
        for idx, r, d in self.violations:
            print("  UNPADDED %s word %d: read %s%d, %d word(s) after the write"
                  % (name, idx, r[0], r[1], d), file=sys.stderr)
        return len(self.violations)

    def read_uni(self, dst, ws=0, nops=2):
        """Read the next uniform word into dst; the uniform port auto-increments."""
        self.e(dict(add_op=ADD_OR, add_out=dst, ws=ws,
                    add_a=areg(R_UNIF), add_b=areg(R_UNIF)))
        for _ in range(nops):
            self.e(nop())


def add_rr(b, op, dst, a_reg, b_reg):
    """dst = a_reg op b_reg when both live in the A file (the ALU can address
    only one A-file and one B-file register per word).  acc3 is scratch: it is
    always written before it is read in every kernel here.

    The QPU has no forwarding: a read of a register written in the previous
    word still sees the old value, so two nops are inserted (asm3's prologues
    do the same)."""
    b.e(add_or(32 + 3, areg(a_reg), areg(a_reg)))
    b.e(nop(), nop())
    b.e(add_(op, dst, acc(3), areg(b_reg)))


# =============================================================== GEMM
A_PN, A_CB, A_BB = 16, 17, 18
A_TMP = 19
A_LANE = 20
A_ARB, A_RBCNT, A_CBCNT = 21, 22, 23
A_COLC = 24
A_VPMQ = 25
A_AADR, A_BADR, A_CADR = 26, 27, 28
A_CRB, A_BOFF, A_KCNT = 29, 30, 31
A_RBIDX = 16      # current row block index (u7; A_PN is dead after ARB)

B_BST, B_NEG1, B_ASTEP, B_VPMQ7 = 20, 21, 22, 23
B_CSTEP, B_CBSTEP, B_RBST_A, B_BLANE = 24, 25, 26, 27
B_ALANE, B_RBST_C, B_BOFFSTEP = 28, 29, 30
B_NEGTR, B_NQPU = 31, 19


def emit_kloop(p, depth=3):
    """The verified 39-word k loop (asm3 gemm2.py, variant 1 head).

    Every TMU pop keeps its own instruction word, never the word that writes a
    TMU address register: that shared word is the 12-QPU corruption
    root-caused in asm3/NOTES.md.  A[r][k] is broadcast with a mul-ALU vector
    rotate plus the hardware element-0 replicate port; the MAC is one packed
    fmul+fadd word.
    """
    head = [
        (None, None, SIG_LOAD_TMU0),
        (None, add_or(32, acc(4), acc(4)), SIG_NONE),
        (None, None, SIG_LOAD_TMU1),
        (None, add_or(W_TMU0_S, areg(A_AADR), areg(A_AADR), ws=1), SIG_NONE),
        (None, add_or(W_TMU1_S, areg(A_BADR), areg(A_BADR), ws=1), SIG_NONE),
    ]
    core = [(mul_rot(1, 0, 15), rep_all(0), SIG_NONE)]
    for m in range(16):
        nxt = rep_all(1) if m < 15 else None
        core.append((mul_fmul(2, acc(5), acc(4)), nxt, SIG_NONE))
        rot = mul_rot(1, 0, 14 - m) if m < 14 else None
        core.append((rot, add_(ADD_FADD, m, areg(m), acc(2)), SIG_NONE))
    core[31] = (core[31][0], add_(ADD_ADD, A_KCNT, (REGB, B_NEG1), areg(A_KCNT), sf=1),
                SIG_NONE)

    p.label('kloop')
    for m, a, s in head:
        emit_slot(p, m, a, s)
    for m, a, s in core[:31]:
        emit_slot(p, m, a, s)
    emit_slot(p, *core[31][:2])
    p.branch_to('kloop', cond=BR_ALL_Z_CLEAR)
    emit_slot(p, *core[32][:2])                                   # delay 1
    p.emit(add_(ADD_ADD, A_AADR, areg(A_AADR), (REGB, B_ASTEP)))  # delay 2
    p.emit(add_(ADD_ADD, A_BADR, areg(A_BADR), (REGB, B_BST)))    # delay 3
    for _ in range(depth):
        p.emit(alu(sig=SIG_LOAD_TMU0, cond_add=COND_NEVER, cond_mul=COND_NEVER))
    for _ in range(depth):
        p.emit(alu(sig=SIG_LOAD_TMU1, cond_add=COND_NEVER, cond_mul=COND_NEVER))
    p.emit(nop())


def emit_store16(p, stride_bytes, cstep_reg=B_CSTEP):
    """Verified VPM->DMA store: 4 chunks of 4 rows of C (asm3 gemm2.py).

    stride_bytes is the memory stride between consecutive 16-word rows; with
    0 the four rows of a chunk are contiguous (used by the elementwise kernel,
    where a group is 256 contiguous floats).
    """
    dma = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)
    for c in range(4):
        p.emit(ldi(0x1A00, R5, ws=1))   # R5 is out-reg 37 (acc5); a literal 5
                                         # would write B-file r5 and the ALU
                                         # read of acc(5) below would be junk
        p.emit(_ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1,
                    add_a=acc(5), add_b=areg(A_VPMQ)))
        for r in range(4):
            p.emit(add_or(W_VPM, areg(4 * c + r), areg(4 * c + r)))
        p.emit(ldi(dma, R5, ws=1))
        p.emit(_ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1,
                    add_a=acc(5), add_b=(REGB, B_VPMQ7)))
        p.emit(ldi(0xC0010000 | (stride_bytes & 0xFFFF), W_VPW_SETUP, ws=1))
        p.emit(add_or(W_VPW_ADDR, areg(A_CADR), areg(A_CADR), ws=1))
        p.emit(_ops(add_op=ADD_OR, add_out=W_NOP,
                    add_a=(REGB, R_VPW_WAIT), add_b=(REGB, R_VPW_WAIT)))
        p.emit(nop(), nop(), nop(), nop())
        p.emit(add_(ADD_ADD, A_CADR, areg(A_CADR), (REGB, cstep_reg)))


def build_gemm(M, N, K, nqpu, bmode=0, depth=3):
    TR, TC = M // 16, N // 16
    assert M % 16 == 0 and N % 16 == 0 and K % 16 == 0, "shapes must be 16-aligned"
    assert TR >= 1 and TC >= 1 and 1 <= nqpu <= 12
    stride = (N - 16) * 4
    tr_neg = (0 - TR) & 0xFFFFFFFF
    b = Builder()
    p = b.p

    b.e(add_or(36, imm(1), imm(1)))                        # tmu_noswap
    b.read_uni(A_PN)                                       # u0 panel
    b.read_uni(A_CB)                                       # u1 C
    b.read_uni(A_BB)                                       # u2 B
    b.read_uni(A_TMP)                                      # u3 A rb offset
    add_rr(b, ADD_ADD, A_ARB, A_PN, A_TMP)
    b.e(nop(), nop())
    b.read_uni(A_TMP)                                      # u4 A rb stride
    b.e(add_or(B_RBST_A, ws=1, a=areg(A_TMP), b=areg(A_TMP)), nop())
    b.read_uni(A_TMP)                                      # u5 C rb offset
    add_rr(b, ADD_ADD, A_CRB, A_CB, A_TMP)
    b.e(nop(), nop())
    b.read_uni(A_TMP)                                      # u6 C rb stride
    b.e(add_or(B_RBST_C, ws=1, a=areg(A_TMP), b=areg(A_TMP)), nop())
    b.read_uni(A_RBIDX)                                    # u7 first row block

    b.e(add_or(A_COLC, areg(A_CRB), areg(A_CRB)), nop(), nop())
    b.e(add_or(A_LANE, areg(R_ELEM_NUM), areg(R_ELEM_NUM)), nop(), nop())
    b.e(add_or(A_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)), nop(), nop())
    b.e(add_(ADD_SHL, A_VPMQ, areg(A_VPMQ), imm(2)), nop(), nop())
    b.e(add_(ADD_SHL, B_ALANE, ws=1, a=areg(A_LANE), b=imm(2)))
    b.e(add_(ADD_SHL, B_BLANE, ws=1, a=areg(A_LANE), b=imm(2)))
    b.e(ldi(N * 4 if bmode == 0 else 64, B_BST, ws=1))
    b.e(ldi(4 * N * 4, B_CSTEP, ws=1))
    b.e(ldi(64, B_CBSTEP, ws=1))
    b.e(ldi(64, B_ASTEP, ws=1))
    # column-block stride in B: 16 floats for row-major B, a whole k-major
    # panel block (K*16 floats) when B is the packed panel (bmode=1)
    b.e(ldi(64 if bmode == 0 else 64 * K, B_BOFFSTEP, ws=1))
    b.e(ldi(0xFFFFFFFF, B_NEG1, ws=1))
    p.emit(ldi(tr_neg, B_NEGTR, ws=1))
    b.e(ldi(nqpu, B_NQPU, ws=1))
    b.e(add_(ADD_SHL, B_VPMQ7, ws=1, a=areg(A_VPMQ), b=imm(7)))
    b.e(add_or(A_BOFF, imm(0), imm(0)))
    b.e(nop(), nop())

    p._builder = b
    # upper bound on the row blocks this QPU can ever own (compile-time)
    p.emit(ldi((TR + nqpu - 1) // nqpu, A_RBCNT, ws=0))
    b.e(nop(), nop())
    p.label('rb_top')
    b.e(add_(ADD_ADD, A_TMP, (REGB, B_NEGTR), areg(A_RBIDX), sf=1), nop(), nop())
    p.branch_to('rb_body', cond=BR_ALL_N_SET)      # idx < TR  <=>  idx-TR < 0
    b.e(nop(), nop(), nop())
    p.branch_to('tdone', cond=BR_ALWAYS)
    b.e(nop(), nop(), nop())
    p.label('rb_body')
    b.e(add_or(A_COLC, areg(A_CRB), areg(A_CRB)), nop())
    # B's column byte offset is advanced through the column sweep; reset it for
    # every row block (it used to carry over from the previous row block, which
    # made every row block after the first read B at the wrong columns).
    b.e(add_or(A_BOFF, imm(0), imm(0)), nop())
    p.emit(ldi(TC, A_CBCNT, ws=0))
    b.e(nop())

    p.label('cb_top')
    b.e(add_(ADD_ADD, A_AADR, areg(A_ARB), (REGB, B_ALANE)), nop())
    b.e(add_(ADD_ADD, A_BADR, areg(A_BB), (REGB, B_BLANE)))
    add_rr(b, ADD_ADD, A_BADR, A_BADR, A_BOFF)
    b.e(nop())
    b.e(add_or(A_CADR, areg(A_COLC), areg(A_COLC)))
    for r in range(16):
        b.e(add_or(r, imm(0), imm(0)))
    b.e(nop(), nop())
    for _ in range(depth):                                  # prime both TMUs
        b.e(add_or(W_TMU0_S, areg(A_AADR), areg(A_AADR), ws=1))
        b.e(add_or(W_TMU1_S, areg(A_BADR), areg(A_BADR), ws=1))
        b.e(nop(), nop())
        b.e(add_(ADD_ADD, A_AADR, areg(A_AADR), (REGB, B_ASTEP)))
        b.e(add_(ADD_ADD, A_BADR, areg(A_BADR), (REGB, B_BST)))
        b.e(nop(), nop())
    p.emit(ldi(K, A_KCNT, ws=0))
    b.e(nop(), nop())
    b.check = False          # verified asm3 k loop: intra-word product read
    emit_kloop(p, depth)
    b.check = False
    emit_store16(p, stride)
    b.check = True           # (the store's ldi->acc5 pattern is asm3's)
    b.e(add_(ADD_ADD, A_COLC, areg(A_COLC), (REGB, B_CBSTEP)))
    b.e(add_(ADD_ADD, A_BOFF, areg(A_BOFF), (REGB, B_BOFFSTEP)))
    b.e(add_(ADD_ADD, A_CBCNT, (REGB, B_NEG1), areg(A_CBCNT), sf=1))
    p.branch_to('cb_top', cond=BR_ALL_Z_CLEAR)
    b.e(nop(), nop(), nop())
    b.e(add_(ADD_ADD, A_ARB, areg(A_ARB), (REGB, B_RBST_A)))
    b.e(add_(ADD_ADD, A_CRB, areg(A_CRB), (REGB, B_RBST_C)))
    b.e(add_(ADD_ADD, A_RBIDX, areg(A_RBIDX), (REGB, B_NQPU)))
    b.e(add_(ADD_ADD, A_RBCNT, (REGB, B_NEG1), areg(A_RBCNT), sf=1))
    p.branch_to('rb_top', cond=BR_ALL_Z_CLEAR)
    b.e(nop(), nop(), nop())
    p.label('tdone')
    b.e(end_program())
    b.e(nop(), nop(), nop())
    return p



















# =============================================================== GEMM (asm10)
#
# The asm10 kernel (asm10/gemm6.py): 8 rows x 64 columns of C per tile, three B
# vector sets, a 6-phase (3 k pair) k loop, 12 QPUs.  Its record on this board
# is 13.745 GFLOP/s GEMM-only / 0 mismatches / 19-of-19 clean launches, and the
# instruction schedule below is *verbatim* from it (same SETV sets, same
# FORDER/BORDER issue rows, same POP/COPY words, same prime, same store).
# Only the addressing was generalised so the engine can use it for any M,N,K:
#
#   * a tile is one 8-row block x one 64-column block, iterated as a nested
#     (row block, column block) loop with a compile-time iteration bound plus
#     an in-loop `rb < TR` guard -- the same pattern as the old asm3 kernel, so
#     a corrupted register still cannot turn the tile loop into an unbounded
#     DMA march;
#   * the A and C row-block offsets and strides come from uniforms u3..u6,
#     so no geometry has to be a power of two (asm10 shifted a tile index);
#   * K/2 % 3 == 0 (K a multiple of 48 -- the conv shapes 144/288/576, and
#     K=48) gets a *drain* group: the last 3 k pairs are emitted after the loop
#     with the last two phases' B issues and the final A pop suppressed, so
#     every TMU pop has a matching issue and nothing is left in flight.
#     asm10 asserted K/2 % 3 != 0 and had no path for it;
#   * B is row-major (bmode=1, the im2col weight panel, is gone: the conv path
#     now transposes its weights on the ARM -- a byte copy, no arithmetic --
#     and every GEMM reads B row-major).
#
# KERNEL ABI (unchanged from the asm3 kernel)
#   u0 panel bus          u1 C bus            u2 B bus
#   u3 A row-block byte offset  u4 A row-block byte stride   (re-derived
#   u5 C row-block byte offset  u6 C row-block byte stride    from M/N/K)
#   u7 first row block this QPU owns
#
# A panel (pack10 layout): PANEL[(rb*K + k)*8 + r] = A[rb*8 + r][k]
# B row-major with row stride N*4; C row-major with row stride N*4.

G10_MUL_V8MIN = 4
G10_NROWS = 8
G10_NCOLS = 64

G10_RA_AADR, G10_RA_CADR, G10_RA_CADR2, G10_RA_KCNT = 20, 21, 22, 23
G10_RA_ARB, G10_RA_RBIDX, G10_RA_TMP, G10_RA_CBOFF = 24, 25, 26, 27
G10_RA_CTMP, G10_RA_CBCNT = 30, 31
G10_RB_BST, G10_RB_NEGTR, G10_RB_RBCNT, G10_RB_CARB = 20, 21, 22, 23
G10_RB_O64, G10_RB_O128, G10_RB_O192, G10_RB_BLANE = 24, 25, 26, 27
G10_RB_TMP, G10_RB_O256 = 30, 31

# Three B vector sets.  C(j0),C(j1) live in the A file, C(j2),C(j3) in the B
# file, so v(j2),v(j3) must be B-file and v(j1) A-file; v(j0) must be A-file
# because the B issues ride on the w1 word whose fmul reads v(j0) with raddr_a.
G10_SETV = (
    {0: (0, 16), 1: (0, 17), 2: (1, 16), 3: (1, 17)},
    {0: (0, 18), 1: (0, 19), 2: (1, 18), 3: (1, 19)},
    {0: (0, 28), 1: (0, 29), 2: (1, 28), 3: (1, 29)},
)   # (file, address); file 0 = A file, 1 = B file

G10_ACC_AVEC, G10_ACC_ROT, G10_ACC_TMP, G10_ACC_BADR = 0, 1, 2, 3
G10_ACC_TMU, G10_ACC_BCAST = 4, 5
G10_ACC_TMP2 = G10_ACC_TMP
G10_ACC_SCR = 3             # same register as BADR (bumped after its last use)

G10_FORDER = (0, 2, 1, 3)      # v(j0) is A-file: w1 can host a raddr_b add op
G10_BORDER = (2, 0, 3, 1)      # v(j2) is B-file: w1 can host a raddr_a add op
G10_ISSUE_ROW = {0: 0, 2: 1, 1: 2, 3: 3}     # w1 of which row carries the issue
G10_TMU_OF = {0: 0, 2: 1, 1: 0, 3: 1}
G10_POP_RW = {0: (1, 4), 2: (3, 4), 1: (5, 4), 3: (6, 4)}
G10_COPY_RW = {0: (1, 5, 'mul'), 2: (4, 1, 'add'), 1: (5, 5, 'mul'),
               3: (7, 1, 'add')}
G10_BUMP_RW = (5, 1)
G10_APOP_RW = (7, 4)


class G10Asm:
    """Word-list assembler with an explicit-schedule hazard checker.

    This is asm10/gemm6.py's Asm class: emit_phase/emit_boundary below put two
    instructions in one word and rely on a schedule that a generic nop-padder
    would destroy, so the schedule is written out and then *verified* (check()
    for per-word legality, verify() for the RAW/TMU rules) before the blob is
    written.
    """

    def __init__(self):
        self.w = []
        self.labels = {}

    def word(self, mul=None, add=None, sig=SIG_NONE, ws=0, note=''):
        self.w.append(dict(mul=mul, add=add, sig=sig, ws=ws, note=note))
        return len(self.w) - 1

    def nop(self, note=''):
        return self.word(note=note)

    def ldi(self, value, out_reg, ws=0, note=''):
        self.w.append(dict(ldi=(value, out_reg, ws), note=note))
        return len(self.w) - 1

    def label(self, name):
        self.labels[name] = len(self.w)

    def branch_to(self, name, cond=BR_ALWAYS, note=''):
        idx = len(self.w)
        self.w.append(dict(branch=cond, target=name, note=note))
        return idx

    def put(self, idx, slot, op, ws=None, sig=None):
        d = self.w[idx]
        if d.get(slot) is not None:
            raise AssertionError('word %d (%s): %s slot already used'
                                 % (idx, d['note'], slot))
        d[slot] = op
        if ws is not None:
            if d['ws'] != ws:
                raise AssertionError('word %d (%s): ws %d != %d'
                                     % (idx, d['note'], d['ws'], ws))
            d['ws'] = ws
        if sig is not None:
            d['sig'] = sig

    def pop(self, idx, tmu, note):
        d = self.w[idx]
        if d['sig'] != SIG_NONE:
            raise AssertionError('two pops in word %d' % idx)
        if d.get('add') is not None and d['add'].get('add_b') is not None \
                and d['add']['add_b'][0] == IMM:
            raise AssertionError('pop on small-immediate word %d' % idx)
        d['sig'] = SIG_LOAD_TMU1 if tmu else SIG_LOAD_TMU0
        d['note'] += ' POP ' + note

    def words(self):
        idx = {id(d): i for i, d in enumerate(self.w)}
        out = []
        for d in self.w:
            if 'ldi' in d:
                v, o, ws = d['ldi']
                out.append(ldi(v, o, ws=ws))
                continue
            if 'branch' in d:
                o = (self.labels[d['target']] - (idx[id(d)] + 4)) * 8
                out.append(branch(o, cond=d['branch']))
                continue
            kw = dict(sig=d['sig'], ws=d['ws'])
            if d['mul']:
                kw.update(d['mul'])
            else:
                kw.update(mul_op=MUL_NOP, mul_out=W_NOP, cond_mul=COND_NEVER)
            if d['add']:
                kw.update(d['add'])
            else:
                kw.update(add_op=ADD_NOP, add_out=W_NOP, cond_add=COND_NEVER)
            try:
                out.append(_ops(**kw))
            except ValueError as ex:
                raise ValueError('word %d (%s): %s | mul=%s add=%s'
                                 % (len(out), d['note'], ex, d['mul'], d['add']))
        return out

    # ---- static checks ----------------------------------------------------
    def check(self):
        for i, d in enumerate(self.w):
            if 'branch' in d or 'ldi' in d:
                continue
            ops = [o for o in (d['mul'], d['add']) if o]
            if d['sig'] in (SIG_LOAD_TMU0, SIG_LOAD_TMU1):
                for o in ops:
                    if o.get('add_out') in (W_TMU0_S, W_TMU1_S) or \
                       o.get('mul_out') in (W_TMU0_S, W_TMU1_S):
                        raise AssertionError('word %d (%s): issue+pop' % (i, d['note']))
            if sum(1 for o in ops if o.get('add_out') in (W_TMU0_S, W_TMU1_S)) > 1:
                raise AssertionError('word %d: two issues' % i)
            if d['sig'] != SIG_NONE:
                for o in ops:
                    for k in ('add_b', 'mul_b'):
                        if o.get(k) is not None and o[k][0] == IMM:
                            raise AssertionError('word %d (%s): pop+imm' % (i, d['note']))
            wss = [o['ws'] for o in ops if 'ws' in o]
            if len(set(wss)) > 1:
                raise AssertionError('word %d (%s): ws conflict %s'
                                     % (i, d['note'], wss))
        return True

    # ---- schedule simulator ----------------------------------------------
    def verify(self):
        """Walk the program and assert every hazard rule on the schedule."""
        GP_A, GP_B, ACC = 0, 1, 2
        W = []
        for i, d in enumerate(self.w):
            rec = dict(wr=[], rd=set(), mul_wr=[], iss=None, pop=None,
                       note=d.get('note', ''))
            if 'branch' in d:
                W.append(rec)
                continue
            if 'ldi' in d:
                v, o, wsb = d['ldi']
                if 32 <= o <= 37:
                    rec['wr'].append((ACC, o - 32, 'ldi'))
                elif o < 32:
                    rec['wr'].append((GP_B if wsb else GP_A, o, 'ldi'))
                W.append(rec)
                continue
            ops = []
            if d['mul']:
                ops.append(('mul', d['mul']))
            if d['add']:
                ops.append(('add', d['add']))
            add_ws = d['add'].get('ws') if d['add'] else None
            mul_ws = d['mul'].get('ws') if d['mul'] else None
            eff = add_ws if add_ws is not None else (mul_ws if mul_ws is not None
                                                     else d['ws'])
            ports = {}
            for kind, o in ops:
                for slot in ('add_a', 'add_b', 'mul_a', 'mul_b'):
                    if o.get(slot) is not None:
                        mux, addr = o[slot]
                        if mux == IMM:
                            p, key, rd = 'rb', ('imm', addr), None
                        elif mux == REGA:
                            p, key, rd = 'ra', addr, (GP_A, addr)
                        elif mux == REGB:
                            p, key, rd = 'rb', addr, (GP_B, addr)
                        else:
                            p, key, rd = None, ('acc', mux), (ACC, mux)
                        if rd:
                            rec['rd'].add(rd)
                        if p:
                            if p in ports and ports[p] != key:
                                rec.setdefault('perr', []).append(
                                    '%s: %s vs %s' % (p, ports[p], key))
                            ports[p] = key
                out = o.get('add_out') if kind == 'add' else o.get('mul_out')
                if out is None or out == W_NOP:
                    continue
                if out == W_TMU0_S:
                    rec['iss'] = 0
                    continue
                if out == W_TMU1_S:
                    rec['iss'] = 1
                    continue
                if out >= 38:
                    continue
                if out <= 37:
                    ent = (ACC, out - 32, kind)
                elif kind == 'add':
                    ent = (GP_A if eff == 0 else GP_B, out, kind)
                else:
                    ent = (GP_B if eff == 0 else GP_A, out, kind)
                rec['wr'].append(ent)
                if kind == 'mul':
                    rec['mul_wr'].append(ent)
            if d['sig'] in (SIG_LOAD_TMU0, SIG_LOAD_TMU1):
                rec['pop'] = 0 if d['sig'] == SIG_LOAD_TMU0 else 1
            W.append(rec)

        n = len(W)
        errs = []

        # -- 0. one A-side and one B-side register port per word
        for i, rec in enumerate(W):
            if rec.get('perr'):
                errs.append('word %d (%s): raddr port conflict %s'
                            % (i, rec['note'], rec['perr']))

        # -- 1. accumulator temps: fmul write -> fadd read, 1..3 words later
        for i, rec in enumerate(W):
            for (f, a, kind) in rec['mul_wr']:
                if f != ACC or a not in (G10_ACC_TMP, G10_ACC_TMP2):
                    continue
                found = None
                for k in range(i + 1, min(i + 4, n)):
                    if (ACC, a) in W[k]['rd']:
                        found = k
                        break
                    if any(w[0] == ACC and w[1] == a for w in W[k]['wr']):
                        break
                if found is None:
                    errs.append('word %d (%s): temp r%d never consumed'
                                % (i, rec['note'], a))

        # -- 2. broadcast: rep (acc r5) write -> fmul read >= 1
        last_rep = None
        for i, rec in enumerate(W):
            if (ACC, G10_ACC_BCAST) in [w[:2] for w in rec['wr']]:
                last_rep = i
            if (ACC, G10_ACC_BCAST) in rec['rd']:
                if last_rep is None:
                    errs.append('word %d (%s): broadcast read before rep'
                                % (i, rec['note']))
                elif i - last_rep < 1:
                    errs.append('word %d (%s): broadcast read %d words after rep'
                                % (i, rec['note'], i - last_rep))

        # -- 3. rot write (acc r1) -> rep read >= 2
        for i, rec in enumerate(W):
            if (ACC, G10_ACC_ROT) in [w[:2] for w in rec['wr']]:
                for k in range(i + 1, min(i + 8, n)):
                    if (ACC, G10_ACC_ROT) in W[k]['rd']:
                        if k - i < 2:
                            errs.append('word %d: rot->rep only %d words'
                                        % (i, k - i))
                        break
                    if any(w[0] == ACC and w[1] == G10_ACC_ROT for w in W[k]['wr']):
                        break

        # -- 3b. A vector (acc r0) write -> v8min read >= 2
        for i, rec in enumerate(W):
            if (ACC, G10_ACC_AVEC) in [w[:2] for w in rec['wr']]:
                for k in range(i + 1, min(i + 8, n)):
                    if (ACC, G10_ACC_AVEC) in W[k]['rd']:
                        if k - i < 2:
                            errs.append('word %d: A copy->rot only %d words'
                                        % (i, k - i))
                        break
                    if any(w[0] == ACC and w[1] == G10_ACC_AVEC for w in W[k]['wr']):
                        break

        # -- 4. GP RAW gaps (A and B file) >= 2 words
        last = {}
        for i, rec in enumerate(W):
            for (f, a) in rec['rd']:
                if f == ACC:
                    continue
                if (f, a) in last and i - last[(f, a)] < 2:
                    errs.append('word %d (%s): GP %s%d RAW only %d words'
                                % (i, rec['note'], 'A' if f == GP_A else 'B', a,
                                   i - last[(f, a)]))
            for (f, a, kind) in rec['wr']:
                if f == ACC:
                    continue
                last[(f, a)] = i

        # -- 5. TMU: per TMU FIFO order, issue->pop >= 8, depth <= 4
        for t in (0, 1):
            q = []
            for i, rec in enumerate(W):
                if rec['iss'] == t:
                    q.append(i)
                    if len(q) > 4:
                        errs.append('word %d: %d in flight on TMU%d'
                                    % (i, len(q), t))
                if rec['pop'] == t:
                    if not q:
                        errs.append('word %d (%s): pop with empty TMU%d FIFO'
                                    % (i, rec['note'], t))
                        continue
                    a = q.pop(0)
                    if i - a < 8:
                        errs.append('word %d (%s): issue->pop only %d words'
                                    % (i, rec['note'], i - a))
            if q:
                errs.append('TMU%d left %d in flight' % (t, len(q)))

        # -- 6. every pop lands in r4: each pop's value must be copied to a GP
        #       register before the next pop on either TMU
        pops = [i for i, rec in enumerate(W) if rec['pop'] is not None]
        for k, i in enumerate(pops):
            nxt = pops[k + 1] if k + 1 < len(pops) else n
            ok = False
            for j in range(i + 1, nxt):
                if (ACC, G10_ACC_TMU) in W[j]['rd']:
                    ok = True
                    break
            if not ok:
                errs.append('word %d (%s): pop not copied before the next pop '
                            '(r4 aliasing)' % (i, W[i]['note']))

        if errs:
            raise AssertionError('schedule verify failed:\n  ' + '\n  '.join(errs))
        return True

    def write(self, path):
        ws = self.words()
        with open(path, 'wb') as f:
            for w in ws:
                f.write(struct.pack('<Q', w))
        return ws

    def listing(self):
        return '\n'.join('%3d: %s' % (i, disasm(w)) for i, w in enumerate(self.words()))


def g10_acc_addr(row, j):
    """(file, address) of C accumulator (row, j); file 0 = A file, 1 = B file."""
    return (0, 2 * row + j) if j < 2 else (1, 2 * row + (j - 2))


def g10_ref(f, addr):
    return areg(addr) if f == 0 else breg(addr)


def g10_or(out, x, y, ws=0):
    return dict(add_op=ADD_OR, add_out=out, ws=ws, add_a=x, add_b=y)


def g10_addi(op, out, x, y, ws=0, sf=0):
    return dict(add_op=op, add_out=out, ws=ws, add_a=x, add_b=y, sf=sf)


def g10_fmul(j, vec, t):
    f, addr = vec
    return dict(mul_op=MUL_FMUL, mul_out=32 + t, mul_a=acc(G10_ACC_BCAST),
                mul_b=g10_ref(f, addr))


def g10_rot(n):
    if n == 0:
        # small immediate 48 is 1.0f, not "rotate by 0" (only 49..63 are the
        # rotate amounts), so row 0 of the k-even half needs an exact copy.
        return dict(mul_op=MUL_V8MIN, mul_out=32 + G10_ACC_ROT,
                    mul_a=acc(G10_ACC_AVEC), mul_b=acc(G10_ACC_AVEC))
    return dict(mul_op=MUL_V8MIN, mul_out=32 + G10_ACC_ROT,
                mul_a=acc(G10_ACC_AVEC), mul_b=imm(48 + n))


def g10_fadd(r, j, t):
    f, addr = g10_acc_addr(r, j)
    return dict(add_op=ADD_FADD, add_out=addr, ws=f, add_a=g10_ref(f, addr),
                add_b=acc(t))


def g10_rep(ws=1):
    return dict(add_op=ADD_OR, add_out=37, ws=ws, add_a=acc(G10_ACC_ROT),
                add_b=acc(G10_ACC_ROT))


def g10_copy_add(f, addr):
    return dict(add_op=ADD_OR, add_out=addr, ws=f, add_a=acc(G10_ACC_TMU),
                add_b=acc(G10_ACC_TMU))


def g10_copy_mul(f, addr):
    """copy ACC_TMU -> GP register addr in file f, on the mul ALU (ws is
    inverted for the mul ALU)."""
    return dict(mul_op=MUL_V8MIN, mul_out=addr, ws=1 - f,
                mul_a=acc(G10_ACC_TMU), mul_b=acc(G10_ACC_TMU))


def g10_issue_b(j, tmu, ws=0):
    """issue the B request for column block j at BADR + 64j on TMU0/TMU1."""
    reg = W_TMU1_S if tmu else W_TMU0_S
    if j == 0:
        return g10_or(reg, acc(G10_ACC_BADR), acc(G10_ACC_BADR), ws=ws)
    return g10_addi(ADD_ADD, reg, acc(G10_ACC_BADR),
                    (REGB, [None, G10_RB_O64, G10_RB_O128, G10_RB_O192][j]),
                    ws=ws)


def g10_issue_a(ws=0):
    return g10_or(W_TMU0_S, areg(G10_RA_AADR), areg(G10_RA_AADR), ws=ws)


def g10_bump_badr(ws=0):
    return g10_addi(ADD_ADD, 32 + G10_ACC_BADR, acc(G10_ACC_BADR),
                    (REGB, G10_RB_BST), ws=ws)


def g10_rot_for(h, r):
    return (16 - (r + (8 if h else 0))) & 15


def g10_prime_issue(a, tag):
    for j in G10_FORDER:
        a.word(add=g10_issue_b(j, G10_TMU_OF[j]), note='%s issue j%d' % (tag, j))
    a.word(add=g10_bump_badr(), note='%s BADR bump' % tag)


def g10_prime_pop(a, s, tag):
    for j in G10_FORDER:
        a.word(sig=SIG_LOAD_TMU1 if G10_TMU_OF[j] else SIG_LOAD_TMU0,
               note='%s POP j%d' % (tag, j))
        f, addr = s[j]
        a.word(add=g10_copy_add(f, addr), note='%s copy j%d' % (tag, j))


def g10_phase(a, p, h, tag, do_pop=True, do_issue=True, a_pop=False,
              rot_next=False, a_issue=False):
    """Emit one 48-word half: 8 rows x 6 words, set (p mod 3) in use."""
    su = G10_SETV[p % 3]
    sp = G10_SETV[(p + 1) % 3]
    si = G10_SETV[(p + 2) % 3]
    idx = {}
    orders = [G10_FORDER] * G10_NROWS
    if a_issue:
        # row 6 carries the A issue: its w1 mul must read a B-file vector so
        # that raddr_a stays free for the issue
        orders[6] = G10_BORDER
    for r in range(G10_NROWS):
        o = orders[r]
        idx[(r, 0)] = a.word(add=g10_rep(ws=1), ws=1,
                             note='%s r%d w0 rep' % (tag, r))
        idx[(r, 1)] = a.word(mul=g10_fmul(o[0], su[o[0]], G10_ACC_TMP),
                             note='%s r%d w1 fmul j%d' % (tag, r, o[0]))
        idx[(r, 2)] = a.word(mul=g10_fmul(o[1], su[o[1]], G10_ACC_TMP2),
                             add=g10_fadd(r, o[0], G10_ACC_TMP),
                             note='%s r%d w2 fmul j%d|fadd j%d'
                                  % (tag, r, o[1], o[0]))
        idx[(r, 3)] = a.word(mul=g10_fmul(o[2], su[o[2]], G10_ACC_TMP),
                             add=g10_fadd(r, o[1], G10_ACC_TMP2),
                             note='%s r%d w3 fmul j%d|fadd j%d'
                                  % (tag, r, o[2], o[1]))
        idx[(r, 4)] = a.word(mul=g10_fmul(o[3], su[o[3]], G10_ACC_TMP2),
                             add=g10_fadd(r, o[2], G10_ACC_TMP),
                             note='%s r%d w4 fmul j%d|fadd j%d'
                                  % (tag, r, o[3], o[2]))
        idx[(r, 5)] = a.word(add=g10_fadd(r, o[3], G10_ACC_TMP2), ws=1,
                             note='%s r%d w5 fadd j%d' % (tag, r, o[3]))
    # rot(row i) -> mul slot of w0 of row i-1 (same word as that row's rep)
    for i in range(1, G10_NROWS):
        a.put(idx[(i - 1, 0)], 'mul', g10_rot(g10_rot_for(h, i)))
    if rot_next:
        # A vector does not change at this phase boundary
        a.put(idx[(G10_NROWS - 1, 0)], 'mul', g10_rot(g10_rot_for(1 - h, 0)))
    if do_issue:
        for j in G10_FORDER:
            a.put(idx[(G10_ISSUE_ROW[j], 1)], 'add', g10_issue_b(j, G10_TMU_OF[j]))
        a.put(idx[G10_BUMP_RW], 'add', g10_bump_badr())
    if a_issue:
        a.put(idx[(6, 1)], 'add', g10_issue_a())
    if do_pop:
        # every pop lands in r4: pop and copy in an order where each copy
        # precedes the next pop (j0 w4r1, j2 w4r3, j1 w4r5, j3 w4r6)
        for j in G10_FORDER:
            r, w = G10_POP_RW[j]
            a.pop(idx[(r, w)], G10_TMU_OF[j], '%s j%d' % (tag, j))
        for j in G10_FORDER:
            r, w, slot = G10_COPY_RW[j]
            f, addr = sp[j]
            op = g10_copy_add(f, addr) if slot == 'add' else g10_copy_mul(f, addr)
            a.put(idx[(r, w)], slot, op)
    if a_pop:
        a.pop(idx[G10_APOP_RW], 0, '%s A' % tag)
    return idx


def g10_boundary(a, p, counter=False):
    """A-vector change before phase p+1: pop, copy, rot, issue for k pair +2."""
    a.word(sig=SIG_LOAD_TMU0,
           add=g10_addi(ADD_ADD, G10_RA_AADR, areg(G10_RA_AADR),
                        (REGB, G10_RB_O64)),
           note='B%d: A pop + AADR += 64' % p)
    a.word(mul=g10_copy_mul(0, 32 + G10_ACC_AVEC),
           note='B%d: r0 = A vector' % p)
    a.nop(note='B%d: copy->rot gap' % p)
    a.word(mul=g10_rot(g10_rot_for(0, 0)), note='B%d: rot(p%d row 0)' % (p, p + 1))
    if counter:
        a.word(add=g10_addi(ADD_SUB, G10_RA_KCNT, areg(G10_RA_KCNT), imm(1), sf=1),
               note='k counter')
    else:
        a.nop(note='B%d: rot->rep gap' % p)


def g10_kloop_body(a, label):
    """One 6-phase group: 3 k pairs, 303 words."""
    a.label(label)
    a.nop(note='prologue (keeps rot->rep >= 2 across the back edge)')
    for p in range(6):
        g10_phase(a, p, p & 1, 'p%d' % p, do_pop=True, do_issue=True,
                  a_pop=(p == 5), rot_next=(p in (0, 2, 4)),
                  a_issue=(p in (0, 2, 4)))
        if p in (1, 3):
            g10_boundary(a, p, counter=(p == 3))


def g10_store(a, stride):
    """VPM->DMA store of the whole 8x64 C tile: 4 column chunks x 2 row halves.

    The VPM row base is 4*qpu, the VDW row field is that * 128 and the memory
    stride of a 16-word C row is (N-16)*4.
    """
    dma = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)
    # r3 = 4*qpu*128 for the VDW row field, RA_TMP = 4*qpu for the vpw row
    a.word(add=g10_or(G10_RA_TMP, breg(R_QPU_NUM), breg(R_QPU_NUM)),
           note='qpu')
    a.nop(), a.nop()
    a.word(add=g10_addi(ADD_SHL, G10_RA_TMP, areg(G10_RA_TMP), imm(2)),
           note='4*qpu')
    a.nop(), a.nop()
    a.word(add=g10_addi(ADD_SHL, 32 + G10_ACC_SCR, areg(G10_RA_TMP), imm(7)),
           note='r3 = 4*qpu*128')
    a.nop()
    for j in range(4):
        for c in range(2):
            a.ldi(0x1A00, R5, ws=1, note='vpw generic row 4q')
            a.word(add=g10_addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(G10_RA_TMP),
                                ws=1))
            for r in range(4):
                f, addr = g10_acc_addr(4 * c + r, j)
                a.word(add=g10_or(W_VPM, g10_ref(f, addr), g10_ref(f, addr)))
            a.ldi(dma, R5, ws=1, note='vdw 4 rows x 16 words')
            a.word(add=g10_addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(G10_ACC_SCR),
                                ws=1))
            a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1, note='stride (N-16)*4')
            base = G10_RA_CADR if c == 0 else G10_RA_CADR2
            a.word(add=g10_or(W_VPW_ADDR, areg(base), areg(base), ws=1))
            a.word(add=g10_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
            a.nop(), a.nop(), a.nop(), a.nop()
        if j != 3:
            a.word(add=g10_addi(ADD_ADD, G10_RA_CADR, areg(G10_RA_CADR),
                                (REGB, G10_RB_O64)), note='next 16 columns')
            a.word(add=g10_addi(ADD_ADD, G10_RA_CADR2, areg(G10_RA_CADR2),
                                (REGB, G10_RB_O64)), note='next 16 columns')


def g10_dump_avec(a, base_reg):
    """Store the 16 lanes of acc0 (the A vector) to memory at base_reg.

    Uses the same VPM->DMA path as the real store: 1 chunk of 4 identical rows
    of 16 words, stride 0, VPM row 0 (single-QPU probe only).
    """
    dma = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)
    a.ldi(0x1A00, R5, ws=1, note='probe vpw')
    a.word(add=g10_addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(5), ws=1))
    for _ in range(4):
        a.word(add=g10_or(W_VPM, acc(G10_ACC_AVEC), acc(G10_ACC_AVEC)))
    a.ldi(dma, R5, ws=1, note='probe vdw 4 rows x 16 words')
    a.word(add=g10_addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(5), ws=1))
    a.ldi(0xC0010000, W_VPW_SETUP, ws=1)
    a.word(add=g10_or(W_VPW_ADDR, areg(base_reg), areg(base_reg), ws=1))
    a.word(add=g10_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
    a.nop(), a.nop(), a.nop(), a.nop()


def build_gemm10(M, N, K, nqpu=12, probe=None, single=False):
    """The asm10 GEMM for arbitrary M (8-aligned), N (64-aligned), K (16-aligned)."""
    assert M % G10_NROWS == 0 and N % G10_NCOLS == 0 and K % 16 == 0, \
        "gemm10 needs M%%8==0, N%%64==0, K%%16==0 (got %d,%d,%d)" % (M, N, K)
    TR, NCB = M // G10_NROWS, N // G10_NCOLS
    NQ = min(nqpu, TR)
    KPAIRS = K // 2
    NBODY, REM = divmod(KPAIRS, 3)
    drain = (REM == 0)
    nloop = NBODY - 1 if drain else NBODY
    assert nloop >= 1, "gemm10 needs K >= 16"
    assert REM in (0, 1, 2)
    RBCNT = (TR + NQ - 1) // NQ
    stride = (N - 16) * 4
    astride = NQ * G10_NROWS * K * 4
    cstride = NQ * G10_NROWS * N * 4

    a = G10Asm()

    if probe == 'lane':
        # debug build: dump acc3 = elem_num<<2 through the (proven) store path
        for i in range(8):
            a.word(add=g10_or(i, areg(R_UNIF), areg(R_UNIF), ws=0),
                   note='u%d' % i)
        a.word(add=g10_addi(ADD_SHL, 32 + G10_ACC_SCR, areg(R_ELEM_NUM), imm(2)),
               note='r3 = elem_num*4')
        a.nop(), a.nop()
        a.word(add=g10_or(0, acc(G10_ACC_SCR), acc(G10_ACC_SCR), ws=0),
               note='A0 = lane*4')
        a.word(add=g10_or(G10_RA_CADR, areg(1), areg(1)), note='C base = u1')
        a.nop(), a.nop()
        g10_store(a, stride)
        a.label('tdone')
        a.word(sig=SIG_END_PROGRAM)
        a.nop(), a.nop(), a.nop()
        a.check()
        a.verify()
        return a
    if probe == 'unif':
        # debug build: dump u0..u7 into C through the (proven) store path
        for i in range(8):
            a.word(add=g10_or(i, areg(R_UNIF), areg(R_UNIF), ws=0),
                   note='dump u%d' % i)
        a.word(add=g10_or(G10_RA_CADR, areg(1), areg(1)), note='C base = u1')
        a.nop(), a.nop()
        g10_store(a, stride)
        a.label('tdone')
        a.word(sig=SIG_END_PROGRAM)
        a.nop(), a.nop(), a.nop()
        a.check()
        a.verify()
        return a

    # ---------------------------------------------------------------- setup
    # The ALU reads one A-file and one B-file register per word, so the live
    # values are split between the files to keep every address computation a
    # single word: A file = A/C tile addresses + the column-block offset, B
    # file = the C row-block base, the B base, the constants and the stride
    # temporaries.
    # Read all eight uniforms first and only then combine them.  A GP register
    # written by a uniform read is NOT visible to the next instruction on this
    # silicon (measured: u3 landed in B r30 and the very next word read B r30
    # one instruction too early, so the A row-block base came out 255 bytes
    # high and every A k-pair was read 8 k out of place).  asm10 kept >= 6
    # words between each uniform read and its first use; do the same.
    a.word(add=g10_or(36, imm(1), imm(1)), note='tmu_noswap')
    a.word(add=g10_or(G10_RA_ARB, areg(R_UNIF), areg(R_UNIF)), note='u0 panel')
    a.word(add=g10_or(G10_RB_CARB, areg(R_UNIF), areg(R_UNIF), ws=1), note='u1 C')
    a.word(add=g10_or(G10_RB_BLANE, areg(R_UNIF), areg(R_UNIF), ws=1), note='u2 B')
    a.word(add=g10_or(G10_RB_TMP, areg(R_UNIF), areg(R_UNIF), ws=1),
           note='u3 A rb offset')
    a.word(add=g10_or(G10_RB_O64, areg(R_UNIF), areg(R_UNIF), ws=1),
           note='u4 A rb stride (re-derived, sink only)')
    a.word(add=g10_or(G10_RA_CTMP, areg(R_UNIF), areg(R_UNIF)), note='u5 C rb offset')
    a.word(add=g10_or(G10_RB_BST, areg(R_UNIF), areg(R_UNIF), ws=1),
           note='u6 C rb stride (re-derived, sink only)')
    a.word(add=g10_or(G10_RA_RBIDX, areg(R_UNIF), areg(R_UNIF)), note='u7 first rb')
    a.nop(), a.nop(), a.nop()
    a.word(add=g10_addi(ADD_ADD, G10_RA_ARB, areg(G10_RA_ARB), (REGB, G10_RB_TMP)),
           note='A rb base')
    a.word(add=g10_addi(ADD_ADD, G10_RB_CARB, breg(G10_RB_CARB), areg(G10_RA_CTMP),
                        ws=1), note='C rb base')
    a.nop(), a.nop()
    a.ldi(N * 4, G10_RB_BST, ws=1, note='B row stride')
    a.ldi(64, G10_RB_O64, ws=1)
    a.ldi(128, G10_RB_O128, ws=1)
    a.ldi(192, G10_RB_O192, ws=1)
    a.ldi(256, G10_RB_O256, ws=1)
    a.ldi((0 - TR) & 0xFFFFFFFF, G10_RB_NEGTR, ws=1, note='-TR')
    a.ldi(RBCNT, G10_RB_RBCNT, ws=1, note='row-block bound')
    a.ldi(0, G10_RA_CBOFF, note='column-block offset')
    # asm10's exact spacing: the accumulator read must be 2 words after the
    # write (measured: a 3-word gap reads a different rotation slot and puts
    # the A base 255 bytes off).
    a.word(add=g10_addi(ADD_SHL, 32 + G10_ACC_SCR, areg(R_ELEM_NUM), imm(2)),
           note='r3 = lane*4')
    a.nop()
    a.word(add=g10_addi(ADD_ADD, G10_RA_ARB, areg(G10_RA_ARB),
                        acc(G10_ACC_SCR)), note='A + lane*4')
    a.word(add=g10_addi(ADD_ADD, G10_RB_BLANE, breg(G10_RB_BLANE),
                        acc(G10_ACC_SCR), ws=1), note='B + lane*4')
    a.nop(), a.nop()

    # ------------------------------------------------------- row block loop
    a.label('rb_top')
    a.word(add=g10_addi(ADD_ADD, 32 + G10_ACC_SCR, areg(G10_RA_RBIDX),
                        (REGB, G10_RB_NEGTR), sf=1), note='rb - TR')
    a.nop(), a.nop()
    a.branch_to('rb_body', cond=BR_ALL_N_SET)
    a.nop(), a.nop(), a.nop()
    a.branch_to('tdone', cond=BR_ALWAYS)
    a.nop(), a.nop(), a.nop()
    a.label('rb_body')
    a.ldi(NCB, G10_RA_CBCNT, note='column blocks')
    a.word(add=g10_or(G10_RA_CBOFF, imm(0), imm(0)), note='cb offset = 0')
    a.nop(), a.nop()

    # ------------------------------------------------------ column block loop
    a.label('cb_top')
    a.word(add=g10_or(G10_RA_AADR, areg(G10_RA_ARB), areg(G10_RA_ARB)),
           note='A tile base')
    a.word(add=g10_addi(ADD_ADD, G10_RA_CADR, areg(G10_RA_CBOFF),
                        (REGB, G10_RB_CARB)), note='C tile base')
    a.word(add=g10_addi(ADD_ADD, 32 + G10_ACC_BADR, areg(G10_RA_CBOFF),
                        (REGB, G10_RB_BLANE)), note='B tile base')
    a.ldi(4 * N * 4, G10_RB_TMP, ws=1, note='4 C rows')
    a.nop(), a.nop()
    a.word(add=g10_addi(ADD_ADD, G10_RA_CADR2, areg(G10_RA_CADR),
                        (REGB, G10_RB_TMP)), note='+ 4 C rows')
    a.nop(), a.nop()
    if probe == 'addrs':
        # dump AADR, CADR, CADR2, CBOFF, BLANE, C_ARB, A_ARB, RBIDX through the
        # (proven) store, to the dump address in u8 (so the store does not
        # depend on the very addresses being measured)
        a.word(add=g10_or(G10_RA_CADR2, areg(R_UNIF), areg(R_UNIF)),
               note='u8 dump address')
        a.nop(), a.nop()
        a.word(add=g10_or(G10_RA_CADR, areg(G10_RA_CADR2), areg(G10_RA_CADR2)),
               note='CADR = u8')
        a.nop(), a.nop()
        for src, dst in ((G10_RA_AADR, 0), (G10_RA_AADR, 2), (G10_RA_CADR2, 4),
                         (G10_RA_CBOFF, 6), (G10_RB_BLANE, 1), (G10_RB_CARB, 3),
                         (G10_RA_ARB, 5), (G10_RA_RBIDX, 7)):
            if src in (G10_RB_BLANE, G10_RB_CARB):
                a.word(add=g10_or(dst, breg(src), breg(src), ws=0))
            else:
                a.word(add=g10_or(dst, areg(src), areg(src), ws=0))
            a.nop(), a.nop()
        a.branch_to('store_x', note='probe: dump addresses')
        a.nop(), a.nop(), a.nop()
    for r in range(G10_NROWS):
        for j in range(4):
            f, addr = g10_acc_addr(r, j)
            a.word(add=g10_or(addr, imm(0), imm(0), ws=f))
    a.nop()
    a.word(add=g10_or(32 + G10_ACC_TMP, imm(0), imm(0)), note='r2 = 0 (mul temp)')
    a.nop()
    a.ldi(nloop, G10_RA_KCNT, note='k-loop iterations (3 k pairs each)')

    # ---------------------------------------------------------- tile prime
    a.word(add=g10_or(W_TMU0_S, areg(G10_RA_AADR), areg(G10_RA_AADR)),
           note='prime issue A kp0')
    a.word(add=g10_addi(ADD_ADD, G10_RA_AADR, areg(G10_RA_AADR),
                        (REGB, G10_RB_O64)), note='AADR += 64')
    g10_prime_issue(a, 'prime s0')
    for _ in range(8):
        a.nop()
    a.word(sig=SIG_LOAD_TMU0, note='prime POP A')
    a.word(add=g10_copy_add(0, 32 + G10_ACC_AVEC), note='prime r0 = A')
    g10_prime_pop(a, G10_SETV[0], 'prime s0')
    g10_prime_issue(a, 'prime s1 (left in flight)')
    a.nop()
    a.word(mul=g10_rot(g10_rot_for(0, 0)), note='prime rot(p0 row 0)')

    if probe == 'const':
        # debug build: dump a known constant to validate the dump path itself
        a.word(add=g10_or(32 + G10_ACC_AVEC, imm(21), imm(21)))
        a.nop(), a.nop()
        g10_dump_avec(a, G10_RA_CADR)
        for _ in range(2):
            for t in (0, 1):
                a.word(sig=SIG_LOAD_TMU1 if t else SIG_LOAD_TMU0, note='probe drain')
                a.word(add=g10_copy_add(0, 32 + G10_ACC_AVEC))
        a.label('tdone')
        a.word(sig=SIG_END_PROGRAM)
        a.nop(), a.nop(), a.nop()
        a.check()
        a.verify()
        return a
    if probe == 'prime_a':
        # debug build: copy the A vector into the store's row-0 inputs and use
        # the (proven) C store to report it
        a.word(add=g10_or(0, acc(G10_ACC_AVEC), acc(G10_ACC_AVEC), ws=0),
               note='A0 = A vector')
        a.nop(), a.nop()
        g10_store(a, stride)
        # drain the four requests of set 1 so nothing is left in flight
        for _ in range(2):
            for t in (0, 1):
                a.word(sig=SIG_LOAD_TMU1 if t else SIG_LOAD_TMU0, note='probe drain')
                a.word(add=g10_copy_add(0, 32 + G10_ACC_AVEC))
        a.label('tdone')
        a.word(sig=SIG_END_PROGRAM)
        a.nop(), a.nop(), a.nop()
        a.check()
        a.verify()
        return a

    # -------------------------------------------------------------- k loop
    g10_kloop_body(a, 'kloop')
    a.branch_to('kloop', cond=BR_ALL_Z_CLEAR, note='k loop')
    a.word(mul=dict(mul_op=MUL_V8MIN, mul_out=32 + G10_ACC_AVEC, ws=1,
                    mul_a=acc(G10_ACC_TMU), mul_b=acc(G10_ACC_TMU)),
           note='delay1: r0 = A vector of the next k pair')
    a.word(add=g10_addi(ADD_ADD, G10_RA_AADR, areg(G10_RA_AADR),
                        (REGB, G10_RB_O64)), note='delay2: AADR += 64')
    a.word(mul=g10_rot(g10_rot_for(0, 0)),
           note='delay3: rot(row0) of the next k pair')

    # ---------------------------------------------------- k-loop tail
    # The last main-loop iteration prefetched set 0 (next k even) and set 1
    # (next k odd); REM = 1 uses them as they are, REM = 2 also fills set 2.
    a.nop(note='tail entry pad (rot->rep >= 2 on the loop-exit path)')
    if drain:
        # KPAIRS % 3 == 0: the last group is emitted here with the issues of
        # its last two phases suppressed, so every pop has an issue and the
        # TMUs are empty at the store.
        for p in range(6):
            g10_phase(a, p, p & 1, 'd%d' % p, do_pop=(p < 5), do_issue=(p < 4),
                      rot_next=(p in (0, 2, 4)), a_issue=(p in (0, 2)))
            if p in (1, 3):
                g10_boundary(a, p, counter=False)
    elif REM == 1:
        g10_phase(a, 6, 0, 't0', do_pop=True, do_issue=False, rot_next=True)
        g10_phase(a, 7, 1, 't1', do_pop=False, do_issue=False, rot_next=False)
    else:
        g10_phase(a, 6, 0, 't0', do_pop=True, do_issue=True, rot_next=True,
                  a_issue=True)
        g10_phase(a, 7, 1, 't1', do_pop=True, do_issue=True, rot_next=False)
        g10_boundary(a, 7, counter=False)
        g10_phase(a, 8, 0, 't2', do_pop=True, do_issue=False, rot_next=True)
        g10_phase(a, 9, 1, 't3', do_pop=False, do_issue=False, rot_next=False)
    a.nop()

    # -------------------------------------------------------------- store
    a.label('store_x')
    if probe == 'regs':
        # overwrite the accumulator inputs of the (proven) store with address
        # registers so the store itself reports them
        for src, dst in ((G10_RA_AADR, 0), (G10_RA_CADR, 2), (G10_RA_CADR2, 4),
                         (G10_RA_CBOFF, 6), (G10_RA_ARB, 1), (G10_RA_RBIDX, 3),
                         (G10_RB_BLANE, 5), (G10_RB_CARB, 7)):
            if src in (G10_RB_BLANE, G10_RB_CARB):
                a.word(add=g10_or(dst, breg(src), breg(src), ws=0))
            else:
                a.word(add=g10_or(dst, areg(src), areg(src), ws=0))
            a.nop(), a.nop()
    g10_store(a, stride)

    if single:
        # debug build: exactly one tile, no tile loop
        a.branch_to('tdone', note='single tile')
        a.nop(), a.nop(), a.nop()
        a.label('tdone')
        a.word(sig=SIG_END_PROGRAM)
        a.nop(), a.nop(), a.nop()
        a.check()
        a.verify()
        return a

    # ------------------------------------------------- next column / row block
    a.word(add=g10_addi(ADD_ADD, G10_RA_CBOFF, areg(G10_RA_CBOFF),
                        (REGB, G10_RB_O256)), note='next column block')
    a.word(add=g10_addi(ADD_SUB, G10_RA_CBCNT, areg(G10_RA_CBCNT), imm(1), sf=1),
           note='cb counter')
    a.branch_to('cb_top', cond=BR_ALL_Z_CLEAR)
    a.nop(), a.nop(), a.nop()
    a.ldi(astride, G10_RB_TMP, ws=1, note='A rb stride')
    a.nop(), a.nop()
    a.word(add=g10_addi(ADD_ADD, G10_RA_ARB, areg(G10_RA_ARB),
                        (REGB, G10_RB_TMP)))
    a.nop(), a.nop()
    a.ldi(cstride, G10_RA_CTMP, ws=0, note='C rb stride')
    a.nop(), a.nop()
    a.word(add=g10_addi(ADD_ADD, G10_RB_CARB, breg(G10_RB_CARB),
                        areg(G10_RA_CTMP), ws=1))
    a.nop()
    a.word(add=g10_addi(ADD_ADD, G10_RA_RBIDX, areg(G10_RA_RBIDX), imm(NQ)),
           note='next row block')
    # the B file cannot hold a register and a small immediate in one word, so
    # the counter decrement borrows the A-file stride temporary
    a.ldi(1, G10_RA_CTMP, note='1')
    a.nop(), a.nop()
    a.word(add=g10_addi(ADD_SUB, G10_RB_RBCNT, breg(G10_RB_RBCNT),
                        areg(G10_RA_CTMP), sf=1, ws=1))
    a.branch_to('rb_top', cond=BR_ALL_Z_CLEAR)
    a.nop(), a.nop(), a.nop()

    a.label('tdone')
    a.word(sig=SIG_END_PROGRAM)
    a.nop(), a.nop(), a.nop()

    a.check()
    a.verify()
    return a


# =============================================================== pack/transpose
def build_pack(kblocks, vpitch=1, dstride=0):
    """K-major panel builder: PANEL[(rb*K + k)*16 + r] = SRC[rb*16 + r][k].

    The asm3 packer (VDR + vertical VDW, no ALU data movement) with the source
    and destination strides as uniforms, so the same blob builds the A panel
    and a k-major B panel from a [Cout][K] weight matrix.

    dstride != 0 turns the 1 KB transposed block into a strided store: each of
    the 16 rows of the transposed block is written dstride bytes further into
    memory than the previous one (the field is *additional* to the 64 B a row
    writes; emit_store16 uses the same field).  With dstride = dst_row_pitch-64
    and the per-kblock/per-rowblock strides set by the caller, the same kernel
    is a dense 16x16 block transpose (vc4__tpose_run).
    """
    A = dict(RL=17, SRCB=18, PB=19, SRCP=20, DSTP=21, SRC=22, DST=23,
             CNT=24, KB=25, VDRW=26, VDWW=27, TMP=28, VDRS=29, VDWS=30, PITCH=31)
    B = dict(OFF=16, ST=17, ST64=18, NEG1=19, DKB=20, DRB=21, Q=22, RBT=23,
             DOFF=24)

    b = Builder()
    p = b.p
    b.e(add_or(36, imm(1), imm(1)))
    b.read_uni(A['RL'])                                      # u0 row length (floats)
    b.read_uni(B['OFF'], ws=1)                               # u1 src rb offset
    b.read_uni(A['SRCB'])                                    # u2 src base
    b.read_uni(B['ST'], ws=1)                                # u3 src rb stride
    b.read_uni(A['PB'])                                      # u4 dst base
    b.read_uni(A['CNT'])                                     # u5 row blocks
    b.read_uni(16)                                           # u6 qpu index (A16)
    b.read_uni(B['DKB'], ws=1)                               # u7 dst k-block stride
    b.read_uni(B['DRB'], ws=1)                               # u8 dst rb stride
    b.read_uni(B['DOFF'], ws=1)                              # u9 dst rb offset

    b.e(add_(ADD_SHL, A['PITCH'], ws=1, a=areg(A['RL']), b=imm(2)), nop())  # *4
    b.e(add_(ADD_SHL, B['RBT'], ws=1, a=areg(16), b=imm(8)))          # q<<8
    b.e(ldi(V.vdr_basic(0, vpitch), A['VDRW']), nop())
    b.e(add_(ADD_ADD, A['VDRW'], areg(A['VDRW']), (REGB, B['RBT'])), nop())
    b.e(add_(ADD_SHL, B['RBT'], ws=1, a=areg(16), b=imm(11)))         # q<<11
    b.e(ldi(V.vdw_basic(0, 16, 16, horizontal=0), A['VDWW']), nop())
    b.e(add_(ADD_ADD, A['VDWW'], areg(A['VDWW']), (REGB, B['RBT'])), nop(), nop())
    b.e(ldi(V.vdr_stride(0) & ~0xFFF, A['TMP']), nop(), nop())
    b.e(add_(ADD_ADD, A['VDRS'], areg(A['TMP']), (REGB, A['PITCH'])), nop())
    b.e(ldi(V.vdw_stride(dstride), A['VDWS']), nop())
    b.e(add_(ADD_ADD, A['SRCP'], areg(A['SRCB']), (REGB, B['OFF'])), nop())
    # destination row blocks can have a different stride than the source
    # (the conv weight panel includes the folded bias k-block)
    b.e(add_(ADD_ADD, A['DSTP'], areg(A['PB']), (REGB, B['DOFF'])), nop())
    b.e(ldi(64, B['ST64'], ws=1))
    b.e(ldi(0xFFFFFFFF, B['NEG1'], ws=1))
    b.e(nop(), nop())

    p._builder = b
    p.label('rb')
    b.e(add_or(A['SRC'], areg(A['SRCP']), areg(A['SRCP'])))
    b.e(add_or(A['DST'], areg(A['DSTP']), areg(A['DSTP'])))
    p.emit(ldi(kblocks, A['TMP'], ws=0))                     # compile-time k blocks
    b.e(nop(), nop())
    p.label('kb')
    b.e(add_or(W_VPR_SETUP, areg(A['VDRW']), areg(A['VDRW'])))
    b.e(add_or(W_VPR_SETUP, areg(A['VDRS']), areg(A['VDRS'])))
    b.e(add_or(W_VPR_ADDR, areg(A['SRC']), areg(A['SRC'])))
    b.e(add_or(W_NOP, areg(50), areg(50)))                       # vpr_wait
    b.e(add_or(W_VPW_SETUP, areg(A['VDWW']), areg(A['VDWW']), ws=1))
    b.e(add_or(W_VPW_SETUP, areg(A['VDWS']), areg(A['VDWS']), ws=1))
    b.e(add_or(W_VPW_ADDR, areg(A['DST']), areg(A['DST']), ws=1))
    b.e(add_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))   # vpw_wait
    b.e(add_(ADD_ADD, A['TMP'], (REGB, B['NEG1']), areg(A['TMP']), sf=1))
    p.branch_to('kb', cond=BR_ALL_Z_CLEAR)
    b.e(add_(ADD_ADD, A['SRC'], areg(A['SRC']), (REGB, B['ST64'])))
    b.e(add_(ADD_ADD, A['DST'], areg(A['DST']), (REGB, B['DKB'])))
    b.e(nop())
    b.e(add_(ADD_ADD, A['SRCP'], areg(A['SRCP']), (REGB, B['ST'])))
    b.e(add_(ADD_ADD, A['DSTP'], areg(A['DSTP']), (REGB, B['DRB'])))
    b.e(add_(ADD_ADD, A['CNT'], (REGB, B['NEG1']), areg(A['CNT']), sf=1))
    p.branch_to('rb', cond=BR_ALL_Z_CLEAR)
    b.e(nop(), nop(), nop())
    b.e(end_program())
    b.e(nop(), nop(), nop())
    return p


# =============================================================== pack (asm10)
def build_pack10(K, vpitch=1):
    """K-major panel builder: asm10's layout and work assignment.

        PANEL[(rb*K + k)*8 + r] = SRC[rb*8 + r][k]

    This is asm10/pack10.py in its adopted configuration (single mode: 8 A
    rows and KL=16 words per VDR command, the one proven 8x8-word VDW shape per
    256 B half, grid-stride work assignment) and it is *instruction-identical*
    to asm10's `pack10_<K>_k16_single_vp1_us.bin`: at N=256 that kernel builds
    the panel in 382 us instead of 458 us for the same bytes, and the panel is
    bit-exact (asm10/RESULTS10.md).

    KERNEL ABI (uniforms u0..u7, 64 B block per QPU)
      u0 source row length in floats (K)
      u1 this QPU's first work-unit byte offset (source and panel)
      u2 source base bus
      u3 one work-unit byte stride (32*K; the kernel re-derives it)
      u4 panel base bus
      u5 work units for this QPU
      u6 QPU launch index (VPM row base = 8*q)
      u7 work-unit byte step (grid stride = pnq * 32*K)

    A work unit is 8 A rows x K: 32*K bytes of source and of panel, so the same
    byte offset applies to both when the source is the padded A copy.
    """
    assert K % 16 == 0, "pack10 needs K % 16 == 0"
    kl = 16
    nrow = 8
    kiter = K // kl
    srcstep = kl * 4                # bytes per k strip (one A row)
    dststep = kl * nrow * 4         # bytes per k strip in the panel
    le1 = nrow.bit_length() - 1
    vdws = [(g, c) for g in range(nrow // 8) for c in range(kl // 8)]

    RA_D2, RA_A, RA_P = 16, 17, 18
    RA_VD1, RA_VD2, RA_SRC, RA_DST = 19, 20, 21, 22
    RA_KB, RA_CNT, RA_VDRW = 23, 24, 25
    RA_VD0, RA_D3, RA_VDRS, RA_VDWS = 26, 27, 28, 29
    RA_VD3, RA_D4 = 30, 31
    RB_OFF, RB_ST, RB_KL = 16, 17, 18
    RB_VECB, RB_NEG1, RB_UNIT = 19, 20, 22
    RB_BLKB, RB_RO, RB_C = 23, 24, 26

    vdwreg = [RA_VD0, RA_VD1, RA_VD2, RA_VD3][:len(vdws)]
    dstreg = [RA_DST, RA_D2, RA_D3, RA_D4][:len(vdws)]
    assert len(vdws) <= 4

    p = Prog()

    def e(*ws):
        for w in ws:
            p.emit(_ops(**w) if isinstance(w, dict) else w)

    # ---------------- prologue -------------------------------------------
    e(add_or(36, imm(1), imm(1)))                                # tmu_noswap
    e(add_(ADD_SHL, RB_ST, ws=1, a=areg(R_UNIF), b=imm(2)))      # u0 -> K*4 pitch
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
    e(ldi(V.vdw_basic(0, 8, 8, horizontal=0, col=0), RA_VD0))
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VD0, areg(RA_VD0), (REGB, RB_RO)))
    for i, (g, c) in enumerate(vdws):
        if i == 0:
            continue                     # RA_VD0 is the (0,0) setup already
        e(ldi(g * 1024 + c * 64, RB_C, ws=1))    # +8 VPM rows / +8 words (col)
        e(nop(), nop())
        e(add_(ADD_ADD, vdwreg[i], areg(RA_VD0), (REGB, RB_C)))
    # strides
    e(ldi(V.vdr_stride(0) & ~0xFFF, RA_KB))                 # ID nibble only
    e(nop(), nop())
    e(add_(ADD_ADD, RA_VDRS, areg(RA_KB), (REGB, RB_ST)))   # ID=9 | pitch
    e(ldi(V.vdw_stride(0), RA_VDWS))                        # ID=3, block mode
    e(ldi(srcstep, RB_KL, ws=1))
    e(ldi(dststep, RB_VECB, ws=1))
    e(ldi(K * nrow * 4, RB_BLKB, ws=1))                     # one work unit
    e(ldi(0xFFFFFFFF, RB_NEG1, ws=1))                       # -1
    e(ldi(256, RB_C, ws=1))                                 # 8 k columns of panel
    e(nop(), nop())

    # ---------------- work-unit loop (an 8-row block) ---------------------
    p.label('rb')
    e(ldi(kiter, RA_KB))                                    # K/KL k strips
    e(add_(ADD_ADD, RA_SRC, areg(RA_A), (REGB, RB_OFF)))
    e(add_(ADD_ADD, RA_DST, areg(RA_P), (REGB, RB_OFF)))
    e(nop(), nop())

    # ---------------- KL-word strip loop ---------------------------------
    p.label('kb')
    e(add_or(W_VPR_SETUP, areg(RA_VDRW), areg(RA_VDRW), ws=0))
    e(add_or(W_VPR_SETUP, areg(RA_VDRS), areg(RA_VDRS), ws=0))
    e(add_or(W_VPR_ADDR, areg(RA_SRC), areg(RA_SRC), ws=0))
    e(add_or(W_NOP, areg(50), areg(50)))                    # vpr_wait
    # per-strip panel destinations (each VDW writes 256 B)
    if len(vdws) > 1:
        e(add_(ADD_ADD, RA_D2, areg(RA_DST), (REGB, RB_C)))     # +8 k
        if len(vdws) > 2:
            e(add_(ADD_ADD, RA_D3, areg(RA_DST), (REGB, RB_BLKB)))   # + one block
            e(nop())
            e(add_(ADD_ADD, RA_D4, areg(RA_D3), (REGB, RB_C)))
    for i in range(len(vdws)):
        e(add_or(W_VPW_SETUP, areg(vdwreg[i]), areg(vdwreg[i]), ws=1))
        e(add_or(W_VPW_SETUP, areg(RA_VDWS), areg(RA_VDWS), ws=1))
        e(add_or(W_VPW_ADDR, areg(dstreg[i]), areg(dstreg[i]), ws=1))
        e(add_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))   # vpw_wait
    e(add_(ADD_ADD, RA_KB, (REGB, RB_NEG1), areg(RA_KB), sf=1))  # KB -= 1
    p.branch_to('kb', cond=BR_ALL_Z_CLEAR)
    e(add_(ADD_ADD, RA_SRC, areg(RA_SRC), (REGB, RB_KL)))       # delay 1
    e(add_(ADD_ADD, RA_DST, areg(RA_DST), (REGB, RB_VECB)))     # delay 2
    e(nop())                                                    # delay 3

    # ---------------- next work unit -------------------------------------
    e(add_(ADD_ADD, RA_CNT, (REGB, RB_NEG1), areg(RA_CNT), sf=1))   # CNT -= 1
    p.branch_to('rb', cond=BR_ALL_Z_CLEAR)
    e(add_(ADD_ADD, RA_A, areg(RA_A), (REGB, RB_UNIT)))          # delay 1
    e(add_(ADD_ADD, RA_P, areg(RA_P), (REGB, RB_UNIT)))          # delay 2
    e(nop())                                                     # delay 3

    e(end_program())
    e(nop(), nop(), nop())
    return p


# =============================================================== im2col
def build_im2col(Cin, K, KH, KW, vpitch=1):
    """Build the packed A panel straight from a padded NCHW image.

    One row block = 16 output pixels that are contiguous in the padded image.
    For one filter tap, a single VDR loads 16 channels x 16 pixels (pitch = the
    padded plane size) and the vertical VDW writes the transposed 1 KB block
    PANEL[rb][tap*Cin + c*16 .. +15][0..15].  No im2col matrix is ever
    materialised and no ARM arithmetic is involved.

    Requires Cin % 16 == 0 and H*Wp*4 <= 8191 (13-bit VDR pitch field).
    """
    assert Cin % 16 == 0 and K == Cin * KH * KW, "im2col expects K = Cin*KH*KW"
    nseg = KH * KW

    A = dict(SRCB=16, PB=17, SRC=18, DST=19, SEGSRC=20, CNT=21, KB=22,
             PITCH=23, VDRW=24, VDWW=25, TMP=26, VDRS=27, VDWS=28, DSTP=30)
    B = dict(OFF=16, ST=17, DKB=18, DRB=19, NEG1=20, C4=21, P16=22,
             Q=23, T=24, WP4=25, PIT=26, DOFF=27)

    b = Builder()
    p = b.p
    b.e(add_or(36, imm(1), imm(1)))
    b.read_uni(A['PITCH'])                                   # u0 plane byte pitch
    b.read_uni(B['OFF'], ws=1)                               # u1 src rb offset
    b.read_uni(A['SRCB'])                                    # u2 padded base
    b.read_uni(B['ST'], ws=1)                                # u3 rb stride
    b.read_uni(A['PB'])                                      # u4 panel base
    b.read_uni(A['CNT'])                                     # u5 row blocks
    b.read_uni(29)                                           # u6 qpu index (A29)
    b.read_uni(B['DKB'], ws=1)                               # u7 1024
    b.read_uni(B['DRB'], ws=1)                               # u8 64*K
    b.read_uni(B['WP4'], ws=1)                               # u9  Wp*4
    b.read_uni(B['DOFF'], ws=1)                              # u10 dst rb offset

    b.e(add_(ADD_SHL, B['P16'], ws=1, a=areg(A['PITCH']), b=imm(4)))   # 16*pitch
    b.e(add_or(B['PIT'], ws=1, a=areg(A['PITCH']), b=areg(A['PITCH'])), nop())
    b.e(ldi(4, B['C4'], ws=1))
    b.e(ldi(0xFFFFFFFF, B['NEG1'], ws=1), nop())
    b.e(add_(ADD_SHL, B['T'], ws=1, a=areg(29), b=imm(8)))             # q<<8
    b.e(ldi(V.vdr_basic(0, vpitch), A['VDRW']), nop())
    b.e(add_(ADD_ADD, A['VDRW'], areg(A['VDRW']), (REGB, B['T'])), nop())
    b.e(add_(ADD_SHL, B['T'], ws=1, a=areg(29), b=imm(11)))            # q<<11
    # horizontal: the VDR rows are the channels (the panel's k) and the columns
    # are the 16 pixels (the panel's r), so the panel must not be transposed.
    # (A vertical VDW would store it as [pixel][channel].)
    b.e(ldi(V.vdw_basic(0, 16, 16, horizontal=1), A['VDWW']), nop())
    b.e(add_(ADD_ADD, A['VDWW'], areg(A['VDWW']), (REGB, B['T'])), nop(), nop())
    b.e(ldi(V.vdr_stride(0) & ~0xFFF, A['TMP']), nop(), nop())
    b.e(add_(ADD_ADD, A['VDRS'], areg(A['TMP']), (REGB, B['PIT'])), nop())
    b.e(ldi(V.vdw_stride(0), A['VDWS']), nop())
    b.e(add_(ADD_ADD, A['SRC'], areg(A['SRCB']), (REGB, B['OFF'])), nop())
    # the destination row block is a full k-major panel (64*nk bytes), so its
    # offset is independent of the 64-byte source row block
    b.e(add_(ADD_ADD, A['DSTP'], areg(A['PB']), (REGB, B['DOFF'])), nop(), nop())

    p._builder = b
    p.label('rb')
    b.e(add_or(A['DST'], areg(A['DSTP']), areg(A['DSTP'])), nop())
    for s in range(nseg):
        ky, kx = divmod(s, KW)
        b.e(add_or(A['SEGSRC'], areg(A['SRC']), areg(A['SRC'])))
        for _ in range(ky):
            b.e(add_(ADD_ADD, A['SEGSRC'], areg(A['SEGSRC']), (REGB, B['WP4'])))
        for _ in range(kx):
            b.e(add_(ADD_ADD, A['SEGSRC'], areg(A['SEGSRC']), (REGB, B['C4'])))
        b.e(add_or(A['TMP'], areg(A['SEGSRC']), areg(A['SEGSRC'])))
        p.emit(ldi(Cin // 16, A['KB'], ws=0))          # compile-time ci blocks
        b.e(nop(), nop())
        p.label('cib%d' % s)
        b.e(add_or(W_VPR_SETUP, areg(A['VDRW']), areg(A['VDRW'])))
        b.e(add_or(W_VPR_SETUP, areg(A['VDRS']), areg(A['VDRS'])))
        b.e(add_or(W_VPR_ADDR, areg(A['TMP']), areg(A['TMP'])))
        b.e(add_or(W_NOP, areg(50), areg(50)))                   # vpr_wait
        b.e(add_or(W_VPW_SETUP, areg(A['VDWW']), areg(A['VDWW']), ws=1))
        b.e(add_or(W_VPW_SETUP, areg(A['VDWS']), areg(A['VDWS']), ws=1))
        b.e(add_or(W_VPW_ADDR, areg(A['DST']), areg(A['DST']), ws=1))
        b.e(add_or(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
        b.e(add_(ADD_ADD, A['TMP'], areg(A['TMP']), (REGB, B['P16'])))
        b.e(add_(ADD_ADD, A['DST'], areg(A['DST']), (REGB, B['DKB'])))
        b.e(add_(ADD_ADD, A['KB'], (REGB, B['NEG1']), areg(A['KB']), sf=1))
        p.branch_to('cib%d' % s, cond=BR_ALL_Z_CLEAR)
        b.e(nop(), nop(), nop())
    # next row block
    b.e(add_(ADD_ADD, A['SRC'], areg(A['SRC']), (REGB, B['ST'])))
    b.e(add_(ADD_ADD, A['DSTP'], areg(A['DSTP']), (REGB, B['DRB'])))
    b.e(add_(ADD_ADD, A['CNT'], (REGB, B['NEG1']), areg(A['CNT']), sf=1))
    p.branch_to('rb', cond=BR_ALL_Z_CLEAR)
    b.e(nop(), nop(), nop())
    b.e(end_program())
    b.e(nop(), nop(), nop())
    return p


# =============================================================== elementwise
OP_ADD, OP_MUL, OP_RELU, OP_SIGMOID, OP_SILU, OP_RELU_BWD, OP_RSQRT = 0, 1, 2, 3, 4, 5, 6

E_AB, E_BB, E_OB = 16, 17, 18
E_CNT, E_LANE = 19, 20
E_AA, E_AB2, E_AO = 21, 22, 23
E_TMP, E_VPMQ, E_AV, E_T8 = 24, 25, 26, 27
E_CADR = 28
E_T9, E_T10, E_T11 = 29, 30, 31

B_VB = 8
B_K = dict(C1=9, C2=10, C3=11, C4=12, C5=13, NLOG2E=14, MAGIC=15,
           CLAMP_LO=16, CLAMP_HI=17, ONE=18, TWO=19, RCPMAGIC=20,
           C4INT=21, STEP=22, VPMQ7=23, CSTEP=24, C16=25, ACOPY=26,
           QOFF=27, LOFF=28, GAP=29, HALF=27)


def build_ew(op, depth=2):
    """out[i] = f(a[i], b[i]); one group = 16 vectors = 256 floats per QPU
    iteration, stored with the verified 4-chunk VPM->DMA sequence (stride 0,
    step 256 bytes so the 16 rows land contiguously)."""
    b = Builder()
    p = b.p
    b.e(add_or(36, imm(1), imm(1)))
    b.read_uni(E_AB)                                     # u0 a
    b.read_uni(E_BB)                                     # u1 b
    b.read_uni(E_OB)                                     # u2 out
    b.read_uni(E_CNT)                                    # u3 groups
    b.read_uni(E_TMP)                                    # u4 qpu index
    b.read_uni(E_T9)                                     # u5 nqpu
    b.e(add_(ADD_SHL, B_K['STEP'], ws=1, a=areg(E_T9), b=imm(10)))  # nqpu*1024
    # per-iteration gap: after the 16 vectors of a group have advanced the TMU
    # address by 1024, QPU i's next group is i+nqpu, so add (nqpu-1)*1024.
    b.e(add_(ADD_ADD, E_T8, areg(E_T9), (REGB, B_K['C4INT'])))
    b.e(add_(ADD_SHL, B_K['GAP'], ws=1, a=areg(E_T8), b=imm(10)))
    b.e(add_or(E_LANE, areg(R_ELEM_NUM), areg(R_ELEM_NUM)), nop())
    b.e(add_or(E_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)), nop())
    b.e(add_(ADD_SHL, E_VPMQ, areg(E_VPMQ), imm(2)), nop())            # 4*qpu
    b.e(add_(ADD_SHL, B_K['VPMQ7'], ws=1, a=areg(E_VPMQ), b=imm(7)))
    b.e(add_(ADD_SHL, B_K['QOFF'], ws=1, a=areg(E_TMP), b=imm(10)))   # q*1024
    b.e(add_(ADD_SHL, B_K['LOFF'], ws=1, a=areg(E_LANE), b=imm(2)))   # lane*4
    b.e(nop(), nop())
    b.e(add_(ADD_ADD, E_AA, areg(E_AB), (REGB, B_K['QOFF'])))
    b.e(add_(ADD_ADD, E_AB2, areg(E_BB), (REGB, B_K['QOFF'])))
    b.e(add_(ADD_ADD, E_AO, areg(E_OB), (REGB, B_K['QOFF'])), nop())
    b.e(add_(ADD_ADD, E_AA, areg(E_AA), (REGB, B_K['LOFF'])))
    b.e(add_(ADD_ADD, E_AB2, areg(E_AB2), (REGB, B_K['LOFF'])))
    b.e(nop(), nop())

    b.e(ldi(0xFFFFFFFF, B_K['C4INT'], ws=1))
    b.e(ldi(256, B_K['CSTEP'], ws=1))
    b.e(ldi(64, B_K['C16'], ws=1))     # one group vector = 16 floats = 64 B
    if op == OP_MUL:
        # constants for the correctly-rounded product (see _ew_compute):
        # 12-bit mantissa mask, abs mask, 24<<23 (half-ulp exponent step),
        # exponent mask
        for name, val in (('M', 0xFFFFF000), ('A', 0x7FFFFFFF),
                          ('H', 0x0C000000), ('X', 0x7F800000)):
            b.e(ldi(val, {'M': 9, 'A': 10, 'H': 11, 'X': 15}[name], ws=1), nop())
    if op == OP_RSQRT:
        b.e(ldi(0x5F3759DF, 0, ws=1))      # B0: rsqrt magic seed
        b.e(ldi(0x3FC00000, 1, ws=1))      # B1: 1.5
        b.e(ldi(0x3F000000, 2, ws=1))      # B2: 0.5
        b.e(nop(), nop())
    if op in (OP_SIGMOID, OP_SILU):
        for name, val in (('NLOG2E', 0xBFB8AA3B), ('MAGIC', 0x4B400000),
                          ('CLAMP_LO', 0xC2B40000), ('CLAMP_HI', 0x42B40000),
                          ('ONE', 0x3F800000), ('TWO', 0x40000000),
                          ('RCPMAGIC', 0x7EF311C3),
                          ('C1', 0x3F317218), ('C2', 0x3E75FDF0),
                          ('C3', 0x3D635847), ('C4', 0x3C1D955B),
                          ('C5', 0x3AAEC3FF), ('HALF', 0x3F000000)):
            b.e(ldi(val, B_K[name], ws=1), nop())
    b.e(nop(), nop())

    p._builder = b
    p.label('grp_top')
    b.e(add_or(E_CADR, areg(E_AO), areg(E_AO)), nop())
    # prime the TMU pipeline (issue/pop never share a word: the asm3 rule)
    for _ in range(depth):
        b.e(add_or(W_TMU0_S, areg(E_AA), areg(E_AA), ws=1))
        b.e(add_or(W_TMU1_S, areg(E_AB2), areg(E_AB2), ws=1))
        b.e(nop(), nop())
        b.e(add_(ADD_ADD, E_AA, areg(E_AA), (REGB, B_K['C16'])))
        b.e(add_(ADD_ADD, E_AB2, areg(E_AB2), (REGB, B_K['C16'])))
        b.e(nop(), nop())
    for j in range(16):
        if j + depth < 16:
            b.e(add_or(W_TMU0_S, areg(E_AA), areg(E_AA), ws=1))
            b.e(add_or(W_TMU1_S, areg(E_AB2), areg(E_AB2), ws=1))
            b.e(nop(), nop())
            b.e(add_(ADD_ADD, E_AA, areg(E_AA), (REGB, B_K['C16'])))
            b.e(add_(ADD_ADD, E_AB2, areg(E_AB2), (REGB, B_K['C16'])))
            b.e(nop(), nop())
        b.e(alu(sig=SIG_LOAD_TMU0, cond_add=COND_NEVER, cond_mul=COND_NEVER), nop())
        b.e(add_or(E_AV, acc(4), acc(4)), nop())
        b.e(alu(sig=SIG_LOAD_TMU1, cond_add=COND_NEVER, cond_mul=COND_NEVER), nop())
        b.e(add_or(B_VB, ws=1, a=acc(4), b=acc(4)), nop())
        _ew_compute(b, op, j)
    b.check = False
    emit_store16(p, 0)
    b.check = True
    b.e(add_(ADD_ADD, E_AO, areg(E_AO), (REGB, B_K['STEP'])))
    b.e(add_(ADD_ADD, E_AA, areg(E_AA), (REGB, B_K['GAP'])))
    b.e(add_(ADD_ADD, E_AB2, areg(E_AB2), (REGB, B_K['GAP'])))
    b.e(add_(ADD_ADD, E_CNT, (REGB, B_K['C4INT']), areg(E_CNT), sf=1))
    p.branch_to('grp_top', cond=BR_ALL_Z_CLEAR)
    b.e(nop(), nop(), nop())
    b.e(end_program())
    b.e(nop(), nop(), nop())
    return p


def _ew_compute(b, op, j):
    """Vector j -> r(j).  Operands: r26 (a) and b8 (b)."""
    aa = areg(E_AV)
    bb = breg(B_VB)
    K = B_K
    if op == OP_ADD:
        b.e(_ops(add_op=ADD_FADD, add_out=j, add_a=aa, add_b=bb))
    elif op == OP_MUL:
        # The QPU FPU is round-toward-zero for *both* fmul and fadd (measured:
        # 1.0 + 0.75 ulp stays 1.0).  A single fmul therefore is not IEEE
        # mul.  Build the correctly-rounded product:
        #   split a,b into 12-bit halves (exact), p = trunc(a*b),
        #   e = (ah*bh - p) + ah*bl + al*bh + al*bl   (= a*b - p exactly)
        #   inc = 1 iff |e| > half-ulp(p)  (|e| == half: tie -> away from zero)
        #   result = bits(p) + inc, i.e. round away from zero by one ulp when
        #   the discarded part is over half.  All of it is integer/bit work,
        #   so it does not depend on the FPU rounding mode.
        MASK, ABSM, HALFSUB, EXPM = 9, 10, 11, 15   # B file, free for OP_MUL
        AH, BA, AL = 16, 17, 24                     # A file, dead after prologue
        # NB: A19 is the group counter, 20..23/25/26/28/29 are loop state;
        # only 27/30/31 are dead after the prologue.
        T_ABS, T_HALF, T_C = 18, 27, 30
        T_INC, T_NZ, T_LSB = 31, 18, 17
        T_M = 20                                      # A file (E_LANE, dead)
        T_MB, T_MNOT, T_LSB2 = 16, 17, 19             # B file (free for OP_MUL;
                                                      # 15 holds EXPM every vector)
        BH, BL, ACOPY = 13, 14, 12                  # B file
        b.e(_ops(add_op=ADD_AND, add_out=AH, add_a=areg(E_AV),
                 add_b=breg(MASK)))                 # ah
        b.e(add_or(ACOPY, ws=1, a=areg(E_AV), b=areg(E_AV)))
        b.e(_ops(add_op=ADD_FSUB, add_out=AL, add_a=breg(ACOPY),
                 add_b=areg(AH)))                   # al
        b.e(add_or(BA, a=breg(B_VB), b=breg(B_VB)))
        b.e(_ops(add_op=ADD_AND, add_out=BH, ws=1, add_a=areg(BA),
                 add_b=breg(MASK)))                 # bh
        b.e(_ops(add_op=ADD_FSUB, add_out=BL, ws=1, add_a=areg(BA),
                 add_b=breg(BH)))                   # bl
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 0, ws=0, mul_a=areg(E_AV),
                 mul_b=breg(B_VB)))                 # p = trunc(a*b)
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 1, ws=0, mul_a=areg(AH),
                 mul_b=breg(BH)))                   # ah*bh (exact)
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 2, add_a=acc(1),
                 add_b=acc(0)))                     # e = ah*bh - p
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 3, ws=0, mul_a=areg(AH),
                 mul_b=breg(BL)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(2), add_b=acc(3)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 3, ws=0, mul_a=areg(AL),
                 mul_b=breg(BH)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(2), add_b=acc(3)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 3, ws=0, mul_a=areg(AL),
                 mul_b=breg(BL)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(2), add_b=acc(3)))
        b.e(_ops(add_op=ADD_AND, add_out=T_ABS, ws=1, add_a=acc(2),
                 add_b=breg(ABSM)))                 # |e| (B file)
        b.e(_ops(add_op=ADD_AND, add_out=T_HALF, add_a=acc(0),
                 add_b=breg(EXPM)))                 # exponent(p)
        b.e(_ops(add_op=ADD_SUB, add_out=T_HALF, add_a=areg(T_HALF),
                 add_b=breg(HALFSUB)))              # 2^(exp-24) = half ulp
        b.e(_ops(add_op=ADD_MAX, add_out=T_HALF, add_a=areg(T_HALF),
                 add_b=imm(0)))                     # clamp (p == 0 case)
        b.e(_ops(add_op=ADD_SUB, add_out=T_C, add_a=breg(T_ABS),
                 add_b=areg(T_HALF)))               # c = |e| - half
        # inc = 1 iff |e| > half; on |e| == half use round-to-even (p's LSB).
        b.e(_ops(add_op=ADD_NOT, add_out=T_INC, add_a=areg(T_C),
                 add_b=areg(T_C)))
        b.e(_ops(add_op=ADD_SHR, add_out=T_INC, add_a=areg(T_INC),
                 add_b=imm(31)))                    # >? -> 0/1 (tie counts)
        b.e(_ops(add_op=ADD_AND, add_out=T_NZ, add_a=areg(T_C),
                 add_b=breg(ABSM)))                 # nz = c & 0x7fffffff
        b.e(_ops(add_op=ADD_SUB, add_out=T_M, add_a=areg(T_NZ),
                 add_b=imm(1)))
        b.e(_ops(add_op=ADD_ASR, add_out=T_M, add_a=areg(T_M),
                 add_b=imm(31)))                    # m = (nz==0) ? -1 : 0
        b.e(_ops(add_op=ADD_AND, add_out=T_LSB, add_a=acc(0),
                 add_b=imm(1)))                     # p & 1
        b.e(add_or(T_MB, ws=1, a=areg(T_M), b=areg(T_M)))    # m in B
        b.e(_ops(add_op=ADD_NOT, add_out=T_MNOT, ws=1, add_a=areg(T_M),
                 add_b=areg(T_M)))                  # ~m in B
        b.e(_ops(add_op=ADD_AND, add_out=T_LSB2, ws=1, add_a=areg(T_LSB),
                 add_b=breg(T_MB)))                 # (p&1) & m, in B
        b.e(_ops(add_op=ADD_AND, add_out=T_INC, add_a=areg(T_INC),
                 add_b=breg(T_MNOT)))               # inc &= ~m
        b.e(_ops(add_op=ADD_OR, add_out=T_INC, add_a=areg(T_INC),
                 add_b=breg(T_LSB2)))               # inc | lsb
        b.e(_ops(add_op=ADD_ADD, add_out=j, add_a=acc(0),
                 add_b=areg(T_INC)))                # bits(p) + inc
    elif op == OP_RELU:
        b.e(_ops(add_op=ADD_FMAX, add_out=j, add_a=aa, add_b=imm(0)))
    elif op == OP_RSQRT:
        # y = 1/sqrt(a) for a > 0: bit-trick seed, then y = y*(1.5 - 0.5*a*y*y)
        b.e(_ops(add_op=ADD_SHR, add_out=32 + 1, add_a=aa, add_b=imm(1)))
        b.e(_ops(add_op=ADD_SUB, add_out=32 + 1, add_a=breg(0), add_b=acc(1)))
        for _ in range(4):
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=aa, mul_b=acc(1)))
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(2), mul_b=acc(1)))
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(2), mul_b=breg(2)))
            b.e(_ops(add_op=ADD_FSUB, add_out=32 + 2, add_a=breg(1), add_b=acc(2)))
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 1, ws=0, mul_a=acc(1), mul_b=acc(2)))
        b.e(add_or(j, acc(1), acc(1)))
    elif op == OP_RELU_BWD:
        # out = b if a > 0 else 0.  Integer mask on the bits of relu(a):
        # +0 -> 0, any positive float has a positive int pattern, a negative
        # float (sign bit set) is a negative int -> (bits-1)>>31 == -1.  The
        # mask is then all-ones or 0, so the product is a bit copy (exact).
        b.e(_ops(add_op=ADD_FMAX, add_out=32 + 0, add_a=aa, add_b=imm(0)))
        b.e(_ops(add_op=ADD_SUB, add_out=32 + 0, add_a=acc(0), add_b=imm(1)))
        b.e(_ops(add_op=ADD_ASR, add_out=32 + 0, add_a=acc(0), add_b=imm(31)))
        b.e(_ops(add_op=ADD_NOT, add_out=32 + 1, add_a=acc(0), add_b=acc(0)))
        b.e(_ops(add_op=ADD_AND, add_out=j, add_a=acc(1), add_b=bb))
    elif op in (OP_SIGMOID, OP_SILU):
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 0, ws=0, mul_a=aa,
                 mul_b=breg(K['NLOG2E'])))
        b.e(_ops(add_op=ADD_FMAX, add_out=32 + 0, add_a=acc(0),
                 add_b=breg(K['CLAMP_LO'])))
        b.e(_ops(add_op=ADD_FMIN, add_out=32 + 0, add_a=acc(0),
                 add_b=breg(K['CLAMP_HI'])))
        # The QPU fadd truncates, so t + 2^23 floors instead of rounding.  Add
        # 0.5 first so n = round(t) and f = t - n stays in [-0.5,0.5], the
        # interval the minimax polynomial was fitted on.
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 1, add_a=acc(0),
                 add_b=breg(K['HALF'])))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 3, add_a=acc(1),
                 add_b=breg(K['MAGIC'])))
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 2, add_a=acc(3),
                 add_b=breg(K['MAGIC'])))          # n
        # convert r1 (still in acc3) to the 2^n scale before the polynomial
        # overwrites acc3 with p
        b.e(_ops(add_op=ADD_FTOI, add_out=E_T10, add_a=acc(3), add_b=acc(3)))
        b.e(_ops(add_op=ADD_SHL, add_out=E_T10, add_a=areg(E_T10), add_b=imm(8)))
        b.e(_ops(add_op=ADD_SHL, add_out=E_T10, add_a=areg(E_T10), add_b=imm(15)))
        b.e(_ops(add_op=ADD_ADD, add_out=E_T10, add_a=areg(E_T10),
                 add_b=breg(K['ONE'])))
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 1, add_a=acc(0),
                 add_b=acc(2)))                    # f
        b.e(add_or(32 + 3, breg(K['C5']), breg(K['C5'])))
        for c in ('C4', 'C3', 'C2', 'C1'):
            # Horner: p = p*f + c  (multiply first; the old emit did
            # (p + c)*f, which is a different polynomial)
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 3, ws=0, mul_a=acc(3),
                     mul_b=acc(1)))
            b.e(_ops(add_op=ADD_FADD, add_out=32 + 3, add_a=acc(3),
                     add_b=breg(K[c])))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 3, ws=0, mul_a=acc(3),
                 mul_b=acc(1)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 3, add_a=acc(3),
                 add_b=breg(K['ONE'])))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 3, ws=0, mul_a=acc(3),
                 mul_b=areg(E_T10)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 0, add_a=acc(3),
                 add_b=breg(K['ONE'])))
        b.e(_ops(add_op=ADD_SUB, add_out=E_T11, add_a=breg(K['RCPMAGIC']),
                 add_b=acc(0)))
        for _ in range(3):
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=areg(E_T11),
                     mul_b=acc(0)))
            b.e(_ops(add_op=ADD_FSUB, add_out=32 + 2, add_a=breg(K['TWO']),
                     add_b=acc(2)))
            b.e(_ops(mul_op=MUL_FMUL, mul_out=E_T11, ws=1, mul_a=areg(E_T11),
                     mul_b=acc(2)))
        if op == OP_SILU:
            b.e(add_or(B_K['ACOPY'], ws=1, a=areg(E_AV), b=areg(E_AV)))
            b.e(_ops(mul_op=MUL_FMUL, mul_out=j, ws=1, mul_a=areg(E_T11),
                     mul_b=breg(B_K['ACOPY'])))
        else:
            b.e(add_or(j, areg(E_T11), areg(E_T11)))
    else:
        raise ValueError("bad ew op %r" % (op,))


# =============================================================== softmax+CE
def build_softmax(depth=2, probe=None):
    """Fused row softmax + cross-entropy backward, ONE 8-row group per launch.

    Each QPU owns exactly one group of 8 rows, so the kernel has no loop and
    no branch: the group count is the launch's QPU count (the host launches
    ceil(rows/8/12) times).  A row is one 16-lane vector (the model output is
    padded to 16 columns with a very negative bias, which contributes exp=0).

      max  = horizontal max over 16 lanes (rotate + fmax, 4 steps)
      e    = 2^((z-max)*log2e)  (degree-5 2^f polynomial + magic exponent,
                                 the constants proven in build_ew)
      S    = horizontal sum (rotate + fadd, 4 steps)
      p    = e * (1/S)          (bit-trick reciprocal + 3 Newton steps)
      dl   = (p - onehot(label)) * (1/B)

    plus, per row, loss = (max - z_y) + ln2*log2(S) and
    correct = (z_y == max).  Each QPU writes its totals (lane 0 loss, lane 1
    correct rows) as one 64 B slot; the host reduces the slots with a GPU
    GEMM (ones^T * partial), never on the CPU.

    KERNEL ABI (uniforms u0..u5, 64 B block per QPU)
      u0 logits group bus (8 rows x 16 floats = 512 B)
      u1 dlogits group bus
      u2 labels group bus (8 x int32 = 32 B)
      u3 partial slot bus (64 B)
      u4 QPU index (VPM row base = 4*q)
      u5 1/B as float bits
    """
    # A file
    A_LOG, A_DL, A_LB, A_PART = 16, 17, 18, 19
    A_INV, A_LN2, A_L2E, A_QID = 20, 21, 22, 23
    A_LANE, A_SADDR, A_M0, A_M1 = 24, 25, 26, 27
    A_EXP = 28
    # B file: constants and running totals
    B_HALF, B_MAGIC, B_RCPM, B_ONE, B_TWO, B_MANT = 0, 1, 2, 3, 4, 5
    B_C64, B_C256, B_ABS, B_CLAMP = 6, 7, 8, 9
    B_LOSS, B_CORR = 10, 11
    B_L0, B_L1, B_L2, B_L3, B_L4, B_L5, B_L6 = 12, 13, 14, 15, 16, 17, 18
    EXPC = (19, 20, 21, 22, 23)
    B_T0, B_T1, B_T2, B_T3 = 24, 25, 26, 27
    B_Q4, B_Q9 = 28, 29
    # exp polynomial (same constants as build_ew: 2^f on [-0.5, 0.5])
    EXP = (0x3AAEC3FF, 0x3C1D955B, 0x3D635847, 0x3E75FDF0, 0x3F317218)
    # log2(1+f) on [0,1], degree 6 (max |err| = 2.1e-6, fitted offline)
    LOG2 = (0x360E8598, 0x3FB8A308, 0xBF37B1DF, 0x3EE93AD9,
            0xBE8C9B6B, 0x3DF452A2, 0xBCCDCF2D)
    RPG = 8

    b = Builder()
    p = b.p
    b.e(add_or(36, imm(1), imm(1)))                    # tmu_noswap
    b.read_uni(A_LOG)                                  # u0
    b.read_uni(A_DL)                                   # u1
    b.read_uni(A_LB)                                   # u2
    b.read_uni(A_PART)                                 # u3
    b.read_uni(A_QID)                                  # u4
    b.read_uni(A_INV)                                  # u5
    b.e(ldi(0x3F000000, B_HALF, ws=1))
    b.e(ldi(0x4B400000, B_MAGIC, ws=1))
    b.e(ldi(0x7EF311C3, B_RCPM, ws=1))
    b.e(ldi(0x3F800000, B_ONE, ws=1))
    b.e(ldi(0x40000000, B_TWO, ws=1))
    b.e(ldi(0x007FFFFF, B_MANT, ws=1))
    b.e(ldi(64, B_C64, ws=1))
    b.e(ldi(0x3727C5AC, B_C256, ws=1))          # 1e-5 (argmax tolerance)
    b.e(ldi(0x7FFFFFFF, B_ABS, ws=1))
    b.e(ldi(0xC2AE0000, B_CLAMP, ws=1))                # -87.0
    b.e(ldi(0x3F317218, A_LN2))                        # ln(2)
    b.e(ldi(0x3FB8AA3B, A_L2E))                        # log2(e)
    b.e(ldi(0x00000000, B_LOSS, ws=1))
    b.e(ldi(0x00000000, B_CORR, ws=1))
    for i, v in enumerate(EXP):
        b.e(ldi(v, EXPC[i], ws=1))
    for i, v in enumerate(LOG2):
        b.e(ldi(v, (B_L0, B_L1, B_L2, B_L3, B_L4, B_L5, B_L6)[i], ws=1))
    b.e(add_(ADD_SHL, B_Q4, ws=1, a=areg(A_QID), b=imm(2)))     # 4q
    b.e(add_(ADD_SHL, B_Q9, ws=1, a=areg(A_QID), b=imm(9)))     # q<<9
    b.e(ldi(127, A_QID))                               # 127 does not fit the
    b.e(nop(), nop())                                  # small-immediate field
    b.e(add_or(A_LANE, areg(R_ELEM_NUM), areg(R_ELEM_NUM)))
    b.e(_ops(add_op=ADD_SUB, add_out=32 + 0, add_a=areg(A_LANE), add_b=imm(1)))
    b.e(_ops(add_op=ADD_ASR, add_out=32 + 0, add_a=acc(0), add_b=imm(31)))
    b.e(_ops(add_op=ADD_AND, add_out=A_M0, add_a=acc(0), add_b=breg(B_ONE)))
    b.e(_ops(add_op=ADD_XOR, add_out=32 + 1, add_a=areg(A_LANE), add_b=imm(1)))
    b.e(_ops(add_op=ADD_SUB, add_out=32 + 1, add_a=acc(1), add_b=imm(1)))
    b.e(_ops(add_op=ADD_ASR, add_out=32 + 1, add_a=acc(1), add_b=imm(31)))
    b.e(_ops(add_op=ADD_AND, add_out=A_M1, add_a=acc(1), add_b=breg(B_ONE)))
    b.e(nop(), nop())

    p._builder = b
    p.label('grp')

    for j in range(RPG):
        # label_j: one 64 B line at the group's label base, lane 0 is row j
        b.e(add_or(W_TMU0_S, areg(A_LB), areg(A_LB), ws=1))
        b.e(nop(), nop())
        b.e(alu(sig=SIG_LOAD_TMU0, cond_add=COND_NEVER, cond_mul=COND_NEVER), nop())
        b.e(add_or(37, acc(4), acc(4), ws=1))          # rep_all(label lane 0)
        b.e(add_or(B_T0, ws=1, a=acc(5), b=acc(5)))
        # z_y = logits[row][label]; all lanes read the same word (only lane 0
        # matters), so the read is harmless even near the row end (the buffer
        # carries 64 B of slack).
        b.e(add_(ADD_SHL, 32 + 2, a=acc(5), b=imm(2)))
        b.e(_ops(add_op=ADD_ADD, add_out=32 + 2, add_a=acc(2), add_b=areg(A_LOG)))
        b.e(add_or(A_SADDR, acc(2), acc(2)))
        b.e(nop(), nop())
        b.e(add_or(W_TMU0_S, areg(A_SADDR), areg(A_SADDR), ws=1))
        b.e(nop(), nop())
        b.e(alu(sig=SIG_LOAD_TMU0, cond_add=COND_NEVER, cond_mul=COND_NEVER), nop())
        b.e(add_or(37, acc(4), acc(4), ws=1))          # rep_all(z_y)
        b.e(add_or(B_T1, ws=1, a=acc(5), b=acc(5)))
        if probe == 'zy':
            b.e(add_or(j, breg(B_T1), breg(B_T1)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LOG, add_a=areg(A_LOG), add_b=breg(B_C64)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LB, add_a=areg(A_LB), add_b=imm(4)))
            b.e(nop())
            continue
        # x: the row's 16 logits, one per lane
        b.e(_ops(add_op=ADD_SHL, add_out=32 + 2, add_a=areg(A_LANE), add_b=imm(2)))
        b.e(_ops(add_op=ADD_ADD, add_out=32 + 2, add_a=acc(2), add_b=areg(A_LOG)))
        b.e(add_or(A_SADDR, acc(2), acc(2)))
        b.e(nop(), nop())
        b.e(add_or(W_TMU0_S, areg(A_SADDR), areg(A_SADDR), ws=1))
        b.e(nop(), nop())
        b.e(alu(sig=SIG_LOAD_TMU0, cond_add=COND_NEVER, cond_mul=COND_NEVER), nop())
        b.e(add_or(32 + 0, acc(4), acc(4)))            # x
        b.e(add_or(B_T3, ws=1, a=acc(0), b=acc(0)))    # keep x
        if probe and probe.startswith('rot'):
            n = int(probe[3:])
            if n:
                b.e(mul_rot(1, 0, n))
                b.e(add_or(j, acc(1), acc(1)))
            else:
                b.e(add_or(j, acc(0), acc(0)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LOG, add_a=areg(A_LOG), add_b=breg(B_C64)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LB, add_a=areg(A_LB), add_b=imm(4)))
            b.e(nop())
            continue
        if probe == 'raw':
            b.e(add_or(j, acc(0), acc(0)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LOG, add_a=areg(A_LOG), add_b=breg(B_C64)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LB, add_a=areg(A_LB), add_b=imm(4)))
            b.e(nop())
            continue
        # max over lanes.  mul_rot(v,n) rotates right by n (asm3 convention:
        # rep(rot(v,n)) == v[16-n]), so {8,12,14,15} covers all 16 offsets.
        # An ADD-ALU accumulator result is not visible to the MUL ALU in the
        # very next word on this board (measured), hence the 2-word gaps.
        for s in (8, 4, 2, 1):
            b.e(mul_rot(1, 0, (16 - s) & 15))
            b.e(nop(), nop())
            b.e(_ops(add_op=ADD_FMAX, add_out=32 + 0, add_a=acc(0), add_b=acc(1)))
            b.e(nop(), nop())
        b.e(add_or(B_T2, ws=1, a=acc(0), b=acc(0)))    # max
        # t = (x - max) * log2e, clamped to [-87, 0]
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 1, add_a=breg(B_T3), add_b=acc(0)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 1, ws=0, mul_a=acc(1), mul_b=areg(A_L2E)))
        b.e(_ops(add_op=ADD_FMAX, add_out=32 + 1, add_a=acc(1), add_b=breg(B_CLAMP)))
        b.e(_ops(add_op=ADD_FMIN, add_out=32 + 1, add_a=acc(1), add_b=imm(0)))
        # exp(t) = 2^n * poly(f)
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(1), add_b=breg(B_HALF)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 3, add_a=acc(2), add_b=breg(B_MAGIC)))
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 2, add_a=acc(3), add_b=breg(B_MAGIC)))
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 1, add_a=acc(1), add_b=acc(2)))   # f
        b.e(_ops(add_op=ADD_FTOI, add_out=32 + 3, add_a=acc(3), add_b=acc(3)))
        b.e(_ops(add_op=ADD_SHL, add_out=32 + 3, add_a=acc(3), add_b=imm(8)))
        b.e(_ops(add_op=ADD_SHL, add_out=32 + 3, add_a=acc(3), add_b=imm(15)))
        b.e(_ops(add_op=ADD_ADD, add_out=32 + 3, add_a=acc(3), add_b=breg(B_ONE)))
        b.e(add_or(32 + 2, breg(EXPC[0]), breg(EXPC[0])))
        for c in (EXPC[1], EXPC[2], EXPC[3], EXPC[4]):
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(2), mul_b=acc(1)))
            b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(2), add_b=breg(c)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(2), mul_b=acc(1)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(2), add_b=breg(B_ONE)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(2), mul_b=acc(3)))
        b.e(add_or(A_EXP, acc(2), acc(2)))             # keep exp
        # sum over lanes
        b.e(add_or(32 + 0, areg(A_EXP), areg(A_EXP)))
        b.e(nop(), nop(), nop())
        for s in (8, 4, 2, 1):
            b.e(mul_rot(1, 0, (16 - s) & 15))
            b.e(nop(), nop())
            b.e(_ops(add_op=ADD_FADD, add_out=32 + 0, add_a=acc(0), add_b=acc(1)))
            b.e(nop(), nop())
        # S = acc0, exp = A_EXP; reciprocal
        b.e(_ops(add_op=ADD_SUB, add_out=32 + 1, add_a=breg(B_RCPM), add_b=acc(0)))
        for _ in range(3):
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(1), mul_b=acc(0)))
            b.e(_ops(add_op=ADD_FSUB, add_out=32 + 2, add_a=breg(B_TWO), add_b=acc(2)))
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 1, ws=0, mul_a=acc(1), mul_b=acc(2)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=areg(A_EXP), mul_b=acc(1)))
        if probe and probe.startswith('red'):
            k = int(probe[3:])
            b.e(add_or(32 + 0, areg(A_EXP), areg(A_EXP)))
            b.e(nop(), nop(), nop())
            for s in (8, 4, 2, 1)[:k]:
                b.e(mul_rot(1, 0, (16 - s) & 15))
                b.e(nop(), nop())
                b.e(_ops(add_op=ADD_FADD, add_out=32 + 0, add_a=acc(0), add_b=acc(1)))
                b.e(nop(), nop())
            b.e(add_or(j, acc(0), acc(0)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LOG, add_a=areg(A_LOG), add_b=breg(B_C64)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LB, add_a=areg(A_LB), add_b=imm(4)))
            b.e(nop())
            continue
        if probe in ('p', 'sum', 'max', 'exp'):
            if probe == 'p':
                b.e(add_or(j, acc(2), acc(2)))
            elif probe == 'sum':
                b.e(add_or(j, acc(0), acc(0)))
            elif probe == 'max':
                b.e(add_or(j, breg(B_T2), breg(B_T2)))
            else:
                b.e(add_or(j, areg(A_EXP), areg(A_EXP)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LOG, add_a=areg(A_LOG), add_b=breg(B_C64)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LB, add_a=areg(A_LB), add_b=imm(4)))
            b.e(nop())
            continue
        # onehot(label) via integer lane equality
        b.e(_ops(add_op=ADD_XOR, add_out=32 + 3, add_a=areg(A_LANE), add_b=breg(B_T0)))
        b.e(_ops(add_op=ADD_SUB, add_out=32 + 3, add_a=acc(3), add_b=imm(1)))
        b.e(_ops(add_op=ADD_ASR, add_out=32 + 3, add_a=acc(3), add_b=imm(31)))
        b.e(_ops(add_op=ADD_AND, add_out=32 + 3, add_a=acc(3), add_b=breg(B_ONE)))
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 2, add_a=acc(2), add_b=acc(3)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(2), mul_b=areg(A_INV)))
        b.e(add_or(j, acc(2), acc(2)))                 # dlogits row j -> A[j]
        # log2(S): e2 = exponent-127, f = mantissa-1
        b.e(_ops(add_op=ADD_SHR, add_out=32 + 3, add_a=acc(0), add_b=imm(23)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_SUB, add_out=32 + 3, add_a=acc(3), add_b=areg(A_QID)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_ITOF, add_out=32 + 3, add_a=acc(3), add_b=acc(3)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_AND, add_out=32 + 1, add_a=acc(0), add_b=breg(B_MANT)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_OR, add_out=32 + 1, add_a=acc(1), add_b=breg(B_ONE)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 1, add_a=acc(1), add_b=breg(B_ONE)))
        b.e(nop(), nop())
        b.e(add_or(32 + 2, breg(B_L6), breg(B_L6)))
        b.e(nop(), nop())
        for c in (B_L5, B_L4, B_L3, B_L2, B_L1, B_L0):
            b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 2, ws=0, mul_a=acc(2), mul_b=acc(1)))
            b.e(nop())
            b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(2), add_b=breg(c)))
            b.e(nop(), nop())
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 2, add_a=acc(2), add_b=acc(3)))
        b.e(nop(), nop())
        # lse = max + ln2*log2(S); loss = lse - z_y
        b.e(add_or(32 + 0, breg(B_T2), breg(B_T2)))
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 1, add_a=acc(0), add_b=breg(B_T1)))
        b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 3, ws=0, mul_a=acc(2), mul_b=areg(A_LN2)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 1, add_a=acc(1), add_b=acc(3)))
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 1, add_a=acc(1), add_b=breg(B_LOSS)))
        b.e(add_or(B_LOSS, ws=1, a=acc(1), b=acc(1)))
        # correct += (z_y ~= max).  The hardware vector rotate used by the
        # reduction is 1-ulp lossy on this board (measured), so the argmax is
        # an absolute-tolerance test on the logit scale: |z_y - max| <= 1e-5.
        # d >= 0, so the bit pattern of d is monotonic in d; compare
        # bits(d) with bits(1e-5) as integers.
        b.e(add_or(32 + 0, breg(B_T1), breg(B_T1)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_FSUB, add_out=32 + 1, add_a=acc(0), add_b=breg(B_T2)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_AND, add_out=32 + 1, add_a=acc(1), add_b=breg(B_ABS)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_SUB, add_out=32 + 1, add_a=acc(1), add_b=breg(B_C256)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_ASR, add_out=32 + 1, add_a=acc(1), add_b=imm(31)))
        b.e(nop(), nop())
        b.e(_ops(add_op=ADD_AND, add_out=32 + 1, add_a=acc(1), add_b=breg(B_ONE)))
        b.e(nop(), nop())
        if probe == 'cor':
            b.e(add_or(j, acc(1), acc(1)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LOG, add_a=areg(A_LOG), add_b=breg(B_C64)))
            b.e(_ops(add_op=ADD_ADD, add_out=A_LB, add_a=areg(A_LB), add_b=imm(4)))
            b.e(nop())
            continue
        b.e(_ops(add_op=ADD_FADD, add_out=32 + 1, add_a=acc(1), add_b=breg(B_CORR)))
        b.e(nop(), nop())
        b.e(add_or(B_CORR, ws=1, a=acc(1), b=acc(1)))
        b.e(_ops(add_op=ADD_ADD, add_out=A_LOG, add_a=areg(A_LOG), add_b=breg(B_C64)))
        b.e(_ops(add_op=ADD_ADD, add_out=A_LB, add_a=areg(A_LB), add_b=imm(4)))
        b.e(nop())

    # ---- store the group's 8 dlogits rows (verified 2-chunk store) --------
    b.e(ldi(256, B_C256, ws=1))                    # restore the store step
    b.e(nop(), nop())
    b.e(add_or(A_SADDR, areg(A_DL), areg(A_DL)))
    for c in range(RPG // 4):
        b.e(ldi(0x1A00, R5, ws=1))
        b.e(_ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1,
                 add_a=acc(5), add_b=breg(B_Q4)))
        for r in range(4):
            b.e(add_or(W_VPM, areg(4 * c + r), areg(4 * c + r)))
        b.e(ldi(0x82104000, R5, ws=1))
        b.e(_ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1,
                 add_a=acc(5), add_b=breg(B_Q9)))
        b.e(ldi(0xC0010000, W_VPW_SETUP, ws=1))
        b.e(add_or(W_VPW_ADDR, areg(A_SADDR), areg(A_SADDR), ws=1))
        b.e(_ops(add_op=ADD_OR, add_out=W_NOP,
                 add_a=(REGB, R_VPW_WAIT), add_b=(REGB, R_VPW_WAIT)))
        b.e(nop(), nop(), nop(), nop())
        b.e(_ops(add_op=ADD_ADD, add_out=A_SADDR, add_a=areg(A_SADDR), add_b=breg(B_C256)))

    # ---- this QPU's totals: lane0 = loss, lane1 = correct rows ------------
    b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 0, ws=0, mul_a=breg(B_LOSS), mul_b=areg(A_M0)))
    b.e(_ops(mul_op=MUL_FMUL, mul_out=32 + 1, ws=0, mul_a=breg(B_CORR), mul_b=areg(A_M1)))
    b.e(_ops(add_op=ADD_FADD, add_out=32 + 0, add_a=acc(0), add_b=acc(1)))
    b.e(add_or(0, acc(0), acc(0)))
    b.e(ldi(0x1A00, R5, ws=1))
    b.e(_ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1,
             add_a=acc(5), add_b=breg(B_Q4)))
    b.e(add_or(W_VPM, areg(0), areg(0)))
    b.e(ldi(0x80904000, R5, ws=1))
    b.e(_ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1,
             add_a=acc(5), add_b=breg(B_Q9)))
    b.e(ldi(0xC0010000, W_VPW_SETUP, ws=1))
    b.e(add_or(W_VPW_ADDR, areg(A_PART), areg(A_PART), ws=1))
    b.e(_ops(add_op=ADD_OR, add_out=W_NOP,
             add_a=(REGB, R_VPW_WAIT), add_b=(REGB, R_VPW_WAIT)))
    b.e(nop(), nop(), nop(), nop())

    b.e(end_program())
    b.e(nop(), nop(), nop())
    return p


# =============================================================== CLI
def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    kind, out = argv[1], argv[2]
    opts = {}
    i = 3
    while i < len(argv):
        opts[argv[i].lstrip('-')] = argv[i + 1]
        i += 2

    def g(k, d=None):
        return int(opts[k]) if k in opts else d

    if kind == 'gemm':
        # the asm10 kernel: 8x64 tiles, 3-phase k loop (build_gemm above is the
        # asm3 16x16 generator, kept as 'gemm3' for A/B and reference)
        p = build_gemm10(g('M'), g('N'), g('K'), g('nqpu', 12),
                         opts.get('probe'), bool(g('single', 0)))
    elif kind == 'gemm3':
        p = build_gemm(g('M'), g('N'), g('K'), g('nqpu', 12), g('bmode', 0),
                       g('depth', 3))
    elif kind == 'pack':
        p = build_pack10(g('k'))
    elif kind == 'pack3':
        p = build_pack(g('kblocks'), g('vpitch', 1), g('dstride', 0))
    elif kind == 'im2col':
        p = build_im2col(g('cin'), g('k'), g('kh'), g('kw'), g('vpitch', 1))
    elif kind == 'ew':
        p = build_ew(g('op'))
    elif kind == 'softmax':
        pr = opts.get('probe')
        p = build_softmax(probe=None if pr in (None, 'full') else pr)
    else:
        print("unknown kernel kind %r" % kind, file=sys.stderr)
        return 2
    if hasattr(p, '_builder'):
        n = p._builder.report(out)
        if n and os.environ.get('KGEN_STRICT'):
            print("kgen: %d read-after-write violations" % n, file=sys.stderr)
            return 3
    w = p.write(out)
    print("%s: %d instructions -> %s" % (kind, len(w), out))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
