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


# =============================================================== pack/transpose
def build_pack(kblocks, vpitch=1):
    """K-major panel builder: PANEL[(rb*K + k)*16 + r] = SRC[rb*16 + r][k].

    The asm3 packer (VDR + vertical VDW, no ALU data movement) with the source
    and destination strides as uniforms, so the same blob builds the A panel
    and a k-major B panel from a [Cout][K] weight matrix.
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
    b.e(ldi(V.vdw_stride(0), A['VDWS']), nop())
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
OP_ADD, OP_MUL, OP_RELU, OP_SIGMOID, OP_SILU = 0, 1, 2, 3, 4

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
        p = build_gemm(g('M'), g('N'), g('K'), g('nqpu', 12), g('bmode', 0),
                       g('depth', 3))
    elif kind == 'pack':
        p = build_pack(g('kblocks'), g('vpitch', 1))
    elif kind == 'im2col':
        p = build_im2col(g('cin'), g('k'), g('kh'), g('kw'), g('vpitch', 1))
    elif kind == 'ew':
        p = build_ew(g('op'))
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
