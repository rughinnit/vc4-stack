#!/usr/bin/env python3
"""asm4: hand-written VideoCore IV QPU GEMM, 8x64 C tile, 4 columns per lane.

asm3's 16x16 tile pays a rotate + a replicate per C row per k to make A[r][k]
lane-uniform, and each broadcast feeds exactly ONE MAC (2 broadcast ops per 4
MAC ops).  Here each lane owns 4 columns (j = 0..3, one 16-column block each),
so one broadcast feeds 4 MACs (2 broadcast ops per 16 MAC ops).

Tile 8 rows x 64 columns:
  * 16 lanes = the 16 columns of a 16-wide block; 4 accumulators per lane =>
    32 accumulators: j=0,1 in the A file (addr 2r+j), j=2,3 in the B file.
    One word can read one A-file and one B-file register, so a MAC word keeps
    its accumulator in one file and its B vector in the other; this forces the
    fmul/fadd pairing used below.
  * A: 8-row k-major packed panel (pack8.py)
        PANEL8[(rb*N + k)*8 + r] = A[rb*8 + r][k]
    one 64 B TMU request covers TWO k columns (lane r = k, lane 8+r = k+1), so
    one iteration = one k pair with a single A request; the halves differ only
    in compile-time rotation amounts -> one code path.
  * B: row major, 4 requests per k (one per 16-column block), lane L =
    B[k][j*16+L]; register sets P (k even) and Q (k odd) so that pops for the
    next k never clobber a live vector.

7 words per row (the rotate -> replicate -> product chain is serial, and the
single mul temp r2 is consumed exactly one word after it is written):
   w0: <B copy> | rep(row r)         r1 lane 0 -> r5 on all 16 lanes
   w1: fmul(j1)                      every MAC of row r reads the same r5
   w2: fmul(j0) | fadd(j1)
   w3: <B-file copy> | fadd(j0)
   w4: fmul(j3) | <issue / copy>
   w5: fmul(j2) | fadd(j3)           TMU1 pops ride here
   w6: <free> | fadd(j2)

Hazards kept from asm3 (all empirically required on this silicon):
  * never TMU issue + TMU pop in one word;
  * no pop on a word with a small immediate (the rotate words);
  * one pop per word, <= 4 requests in flight per TMU (FIFO depth 4);
  * a copy of r4 never shares a word with a pop;
  * TMU issue -> pop distance >= 8 words;
  * VPM DMA store DEPTH=16 words, UNITS=4 rows, stride (N-16)*4, VPM row base
    4*qpu (12 QPUs x 4 rows), 16 encoded as 0.
"""
import os
import struct
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qenc import *          # noqa: F401,F403
from qenc import _ops, branch, disasm

MUL_V8MIN = 4
NROWS = 8
NCOLS = 64
POP_ORDER = (0, 1, 2, 3)

# ---- A file ----------------------------------------------------------------
RA_B2, RA_B2Q, RA_B3, RA_B3Q = 16, 17, 18, 19
RA_AADR, RA_CADR, RA_CADR2, RA_KCNT = 20, 21, 22, 23
RA_T, RA_S1, RA_S2, RA_VPMQ, RA_COFF, RA_UP, RA_UB, RA_S3 = 24, 25, 26, 27, 28, 29, 30, 31
# ---- B file ----------------------------------------------------------------
RB_B0, RB_B0Q, RB_B1, RB_B1Q = 16, 17, 18, 19
RB_BST, RB_NEG1, RB_NEGNT, RB_UP = 20, 21, 22, 23
RB_O64, RB_O128, RB_O192, RB_CSTEP = 24, 25, 26, 27
RB_ONE, RB_TSTR, RB_UB, RB_UC = 28, 29, 30, 31
# ---- accumulator bank ------------------------------------------------------
ACC_AVEC, ACC_ROT, ACC_TMP, ACC_BADR, ACC_TMU, ACC_BCAST = 0, 1, 2, 3, 4, 5


def acc_addr(row, j):
    """(file, address) of accumulator (row, j); file 0 = A file, 1 = B file."""
    return (0, 2 * row + j) if j < 2 else (1, 2 * row + (j - 2))


def ref(f, addr):
    return areg(addr) if f == 0 else breg(addr)


def bset(j, h):
    return {0: (RB_B0, RB_B0Q), 1: (RB_B1, RB_B1Q),
            2: (RA_B2, RA_B2Q), 3: (RA_B3, RA_B3Q)}[j][h]


def fmul_j(j, h):
    f = 0 if j >= 2 else 1                    # file of B vector
    return dict(mul_op=MUL_FMUL, mul_out=32 + ACC_TMP, mul_a=acc(ACC_BCAST),
                mul_b=ref(f, bset(j, h)))


def rot_inst(n):
    if n == 0:
        # small immediate 48 is rotate-by-r5, not "rotate by 0"; 49..63 are
        # the rotate amounts 1..15 that asm3 uses (1.0f is code 32).  There is
        # no rotate-by-0 code, so row 0 ("no rotation") needs an exact copy of
        # the A vector instead.  This is the same v8min self-copy idiom the
        # B-file copies use.
        return dict(mul_op=MUL_V8MIN, mul_out=32 + ACC_ROT,
                    mul_a=acc(ACC_AVEC), mul_b=acc(ACC_AVEC))
    return dict(mul_op=MUL_V8MIN, mul_out=32 + ACC_ROT, mul_a=acc(ACC_AVEC),
                mul_b=imm(48 + n))


def fadd_inst(f, addr):
    return dict(add_op=ADD_FADD, add_out=addr, ws=f, add_a=ref(f, addr),
                add_b=acc(ACC_TMP))


def or_inst(out, x, y, ws=0):
    return dict(add_op=ADD_OR, add_out=out, ws=ws, add_a=x, add_b=y)


def addi(op, out, x, y, ws=0, sf=0):
    return dict(add_op=op, add_out=out, ws=ws, add_a=x, add_b=y, sf=sf)


def rep_inst(ws=1):
    # r5 is the broadcast port; the source is the rotate result ACC_ROT.
    return dict(add_op=ADD_OR, add_out=37, ws=ws, add_a=acc(ACC_ROT),
                add_b=acc(ACC_ROT))


def copy_add(f, addr):
    return dict(add_op=ADD_OR, add_out=addr, ws=f, add_a=acc(ACC_TMU),
                add_b=acc(ACC_TMU))


class Asm:
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
        return True

    def write(self, path):
        ws = self.words()
        with open(path, 'wb') as f:
            for w in ws:
                f.write(struct.pack('<Q', w))
        return ws

    def audit(self):
        rows = []
        for i, d in enumerate(self.w):
            if 'branch' in d:
                rows.append('%3d: br->%s  %s' % (i, d['target'], d['note']))
            elif 'ldi' in d:
                rows.append('%3d: ldi  %s' % (i, d['note']))
            else:
                rows.append('%3d: %-30s | %-30s | sig=%d  %s'
                            % (i, _ops_(d['mul']), _ops_(d['add']), d['sig'], d['note']))
        return '\n'.join(rows)

    def listing(self):
        return '\n'.join('%3d: %s' % (i, disasm(w)) for i, w in enumerate(self.words()))


def _ops_(o):
    if not o:
        return '-'
    if 'mul_op' in o:
        return '%s %s,%s,%s' % ({0: 'nop', 1: 'fmul', 4: 'v8min'}.get(o['mul_op'], o['mul_op']),
                               o['mul_out'], o['mul_a'], o['mul_b'])
    return '%s %s%s a=%s b=%s' % (
        {1: 'fadd', 12: 'add', 13: 'sub', 17: 'shl', 21: 'or'}.get(o['add_op'], o['add_op']),
        o['add_out'], '.setf' if o.get('sf') else '', o['add_a'], o['add_b'])


def build(N, debug_store=False):
    assert N % NCOLS == 0 and N % NROWS == 0
    ncb = N // NCOLS
    ntiles = ncb * (N // NROWS)
    assert ncb & (ncb - 1) == 0
    lg_ncb = ncb.bit_length() - 1
    SROWB = (32 * N).bit_length() - 1
    KPAIRS = N // 2
    stride = (N - 16) * 4

    a = Asm()

    def rot_for(h, r):
        return (16 - (r + (8 if h else 0))) & 15

    # ---------------------------------------------------------------- setup
    a.word(add=or_inst(36, imm(1), imm(1)), note='tmu_noswap')
    a.word(add=or_inst(RA_S1, areg(R_UNIF), areg(R_UNIF)), note='u0 N (unused)')
    a.word(add=or_inst(RB_UC, areg(R_UNIF), areg(R_UNIF), ws=1), note='u1 C')
    a.word(add=or_inst(RA_S2, areg(R_UNIF), areg(R_UNIF)), note='u2 A (unused)')
    a.word(add=or_inst(RB_UB, areg(R_UNIF), areg(R_UNIF), ws=1), note='u3 B')
    a.word(add=or_inst(RB_UP, areg(R_UNIF), areg(R_UNIF), ws=1), note='u4 panel')
    a.word(add=or_inst(RA_T, areg(R_UNIF), areg(R_UNIF)), note='u5 tile0')
    a.word(add=or_inst(RB_TSTR, areg(R_UNIF), areg(R_UNIF), ws=1), note='u6 tile stride')
    if debug_store:
        a.word(add=or_inst(RA_COFF, areg(R_UNIF), areg(R_UNIF)),
               note='u7 C bus address (harness, u7 = gpu_bus + out_off)')
    a.word(add=or_inst(RA_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)), note='qpu')
    a.nop()
    a.word(add=addi(ADD_SHL, RA_VPMQ, areg(RA_VPMQ), imm(2)), note='4*qpu')
    a.ldi(N * 4, RB_BST, ws=1, note='N*4')
    a.ldi(16 * N, RB_CSTEP, ws=1, note='4 C rows')
    a.ldi(64, RB_O64, ws=1)
    a.ldi(128, RB_O128, ws=1)
    a.ldi(192, RB_O192, ws=1)
    a.ldi(0x3F800000, RB_ONE, ws=1, note='1.0f')
    a.ldi(0xFFFFFFFF, RB_NEG1, ws=1, note='-1')
    a.ldi((1 << 32) - ntiles, RB_NEGNT, ws=1, note='-ntiles')
    a.nop()
    a.nop()

    # ------------------------------------------------------------ tile loop
    a.label('tile')
    a.word(add=addi(ADD_ADD, RA_S1, areg(RA_T), (REGB, RB_NEGNT), sf=1), note='t-ntiles')
    a.branch_to('tbody', cond=BR_ALL_N_SET)
    a.nop(), a.nop(), a.nop()
    a.branch_to('tdone')
    a.nop(), a.nop(), a.nop()
    a.label('tbody')
    a.word(add=addi(ADD_AND, RA_S1, areg(RA_T), imm(ncb - 1)), note='colblk')
    a.word(add=addi(ADD_SHR, RA_S2, areg(RA_T), imm(lg_ncb)), note='rowblk')
    a.nop(), a.nop()
    a.word(add=addi(ADD_SHL, RA_S2, areg(RA_S2), imm(SROWB)), note='rowoff = rowblk*32N')
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_S1), imm(8)), note='r3 = colblk*256')
    a.word(add=addi(ADD_ADD, RA_AADR, (REGB, RB_UP), areg(RA_S2)), note='panel + rowoff')
    a.word(add=addi(ADD_ADD, RA_CADR, (REGB, RB_UC), areg(RA_S2)), note='C + rowoff')
    # an A-file write is not visible to the next instruction (asm3 has two nops
    # here); without this gap CADR keeps a stale value and the tile store lands
    # in the packed A panel instead of C
    a.nop()
    a.word(add=addi(ADD_ADD, RA_CADR, areg(RA_CADR), acc(ACC_BADR)), note='+ colblk*256')
    # 64 columns per column block = 256 bytes (not 64: that is the 16-column
    # tile of asm3)
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_S1), imm(8)), note='r3 = colblk*256')
    a.word(add=addi(ADD_ADD, RA_S3, (REGB, RB_UB), acc(ACC_BADR)), note='S3 = B + colblk*256')
    a.word(add=addi(ADD_ADD, RA_CADR2, areg(RA_CADR), (REGB, RB_CSTEP)), note='+ 4 C rows')
    a.ldi(KPAIRS, RA_KCNT, note='k pairs')
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(R_ELEM_NUM), imm(2)), note='r3 = lane*4')
    a.nop()
    a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), acc(ACC_BADR)), note='+ lane*4')
    a.word(add=addi(ADD_ADD, RA_S3, areg(RA_S3), acc(ACC_BADR)), note='+ lane*4')
    a.nop()
    a.word(add=or_inst(32 + ACC_BADR, areg(RA_S3), areg(RA_S3)), note='r3 = BADR (lane-offset)')
    for r in range(NROWS):
        for j in range(4):
            f, addr = acc_addr(r, j)
            a.word(add=or_inst(addr, imm(0), imm(0), ws=f))
    a.nop()
    a.word(add=or_inst(32 + ACC_TMP, imm(0), imm(0)), note='r2 = 0 (mul temp)')
    a.nop()
    # prime: A(kp0) and B(k0) -> set P
    a.word(add=or_inst(W_TMU0_S, areg(RA_AADR), areg(RA_AADR), ws=1), note='issue A')
    a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), (REGB, RB_O64)), note='AADR += 64')
    for j in POP_ORDER:
        if j == 0:
            a.word(add=or_inst(W_TMU1_S, acc(ACC_BADR), acc(ACC_BADR), ws=1),
                   note='issue B j0')
        else:
            a.word(add=addi(ADD_ADD, W_TMU1_S, acc(ACC_BADR),
                            (REGB, [None, RB_O64, RB_O128, RB_O192][j]), ws=1),
                   note='issue B j%d' % j)
    a.word(add=addi(ADD_ADD, 32 + ACC_BADR, acc(ACC_BADR), (REGB, RB_BST)), note='BADR -> k1')
    a.nop(), a.nop(), a.nop(), a.nop()
    for j in POP_ORDER:
        tgt = bset(j, 0)
        a.word(sig=SIG_LOAD_TMU1, note='pop B k0 j%d' % j)
        a.word(add=copy_add(1 if j < 2 else 0, tgt), note='copy B k0 j%d' % j)
    a.word(sig=SIG_LOAD_TMU0, note='pop A')
    a.word(add=copy_add(0, 32 + ACC_AVEC), note='r0 = A vector')
    a.nop()
    a.word(mul=rot_inst(rot_for(0, 0)), note='rot for row0')
    a.nop()

    skip = os.environ.get('SKIP', '')
    if skip in ('all', 'prime'):
        a.branch_to('store', note='SKIP prime+loop')
        a.nop(), a.nop(), a.nop()
    if debug_store == 'first':
        a.word(add=or_inst(RA_UP, areg(RA_COFF), areg(RA_COFF)),
               note='dbg dest = u7 (C bus), does not touch RA_CADR')
        a.word(add=or_inst(RA_VPMQ, breg(R_QPU_NUM), breg(R_QPU_NUM)))
        a.nop(), a.nop()
        a.ldi(0x1A00, R5, ws=1)
        a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1))
        for v in (1, 2, 3, 4):
            a.ldi(0x3F800000 + v, 22, ws=0)
            a.word(add=or_inst(W_VPM, areg(22), areg(22)))
        a.ldi((2 << 30) | (4 << 23) | (16 << 16) | (1 << 14), R5, ws=1)
        a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1),
               note='dma row base = 4*qpu')
        a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1)
        a.word(add=or_inst(W_VPW_ADDR, areg(RA_UP), areg(RA_UP), ws=1))
        a.word(add=or_inst(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
        a.nop(), a.nop(), a.nop(), a.nop()
    if debug_store:
        # dump UC, UP, UB, T and both VPM write setups to C (row r = value)
        a.word(add=or_inst(RA_UP, areg(RA_COFF), areg(RA_COFF)),
               note='dbg dest = u7 (C bus), does not touch RA_CADR')
        a.nop(), a.nop()
        a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_VPMQ), imm(7)), note='dbg r3 = 4q<<7')
        a.nop()
        a.ldi(0x1A00, R5, ws=1)
        a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1))
        a.word(add=or_inst(W_VPM, (REGB, RB_UC), (REGB, RB_UC)))
        a.word(add=or_inst(W_VPM, (REGB, RB_UP), (REGB, RB_UP)))
        a.word(add=or_inst(W_VPM, (REGB, RB_UB), (REGB, RB_UB)))
        a.word(add=or_inst(W_VPM, areg(RA_T), areg(RA_T)))
        a.ldi((2 << 30) | (4 << 23) | (16 << 16) | (1 << 14), R5, ws=1)
        a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1),
               note='dma row base = 4*qpu')
        a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1)
        a.word(add=or_inst(W_VPW_ADDR, areg(RA_CADR), areg(RA_CADR), ws=1))
        a.word(add=or_inst(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
        a.nop(), a.nop(), a.nop(), a.nop()

    if skip == 'loop':
        a.branch_to('store', note='SKIP kloop')
        a.nop(), a.nop(), a.nop()

    # ----------------------------------------------------------------- body
    a.label('kloop')
    # gap so that row 0's rotate (in the 3rd branch delay slot) is not read by
    # the replicate in the very next word
    a.nop(note='row0 rot -> rep gap')
    idx = {}
    # One k-pair iteration = 16 rows (both halves, rows 0..7).  The broadcast
    # of row i is repped at w0 of row i and must not be written again until the
    # last MAC of that row (w5), so the rotate that feeds row i+1 lives in w3
    # of row i, where the fadd j0 occupancy leaves raddr_b free for the small
    # immediate and no TMU signal is present.
    rows = [(h, r) for h in range(2) for r in range(NROWS)]
    for h in range(2):
        for r in range(NROWS):
            d = {}
            d['w0'] = a.word(add=rep_inst(ws=1), ws=1,
                             note='h%d r%d w0 rep(row %d)' % (h, r, r))
            d['w1'] = a.word(mul=fmul_j(1, h), ws=0,
                             note='h%d r%d fmul j1' % (h, r))
            d['w2'] = a.word(mul=fmul_j(0, h), add=fadd_inst(*acc_addr(r, 1)), ws=0,
                             note='h%d r%d fmul j0|fadd j1' % (h, r))
            d['w3'] = a.word(add=fadd_inst(*acc_addr(r, 0)), ws=0,
                             note='h%d r%d fadd j0' % (h, r))
            d['w4'] = a.word(mul=fmul_j(3, h), ws=0,
                             note='h%d r%d fmul j3' % (h, r))
            d['w5'] = a.word(mul=fmul_j(2, h), add=fadd_inst(*acc_addr(r, 3)), ws=1,
                             note='h%d r%d fmul j2|fadd j3' % (h, r))
            if r in (4, 5):
                # B-file copy of this row's j=0/j=1 pop: one word
                # after the pop (r4 is reused by the next pop)
                d['wc'] = a.word(note='h%d r%d B copy' % (h, r))
            d['w6'] = a.word(add=fadd_inst(*acc_addr(r, 2)), ws=1,
                             note='h%d r%d fadd j2' % (h, r))
            idx[(h, r)] = d

    def slot(h, r, name):
        return idx[(h, r)][name]

    # rot(row i) -> mul slot of w3 of row i-1 (any distance before the rep is
    # fine: ACC_ROT has exactly one writer).  Row (0,0) is rotated by the 3rd
    # branch delay slot on the loop path and by the prologue on first entry.
    for i in range(1, len(rows)):
        h, r = rows[i]
        ph, pr = rows[i - 1]
        a.put(slot(ph, pr, 'w3'), 'mul', rot_inst(rot_for(h, r)), ws=0)

    def issue(word, j):
        if j == 0:
            a.put(word, 'add', or_inst(W_TMU1_S, acc(ACC_BADR), acc(ACC_BADR), ws=0),
                  ws=0)
        else:
            a.put(word, 'add', addi(ADD_ADD, W_TMU1_S, acc(ACC_BADR),
                                    (REGB, [None, RB_O64, RB_O128, RB_O192][j]), ws=0),
                  ws=0)

    # A issue rides on w0 (raddr_a only); B issues/bumps on w4 (raddr_b only,
    # BADR lives in accumulator r3 so no A-file address is needed).
    a.put(slot(0, 0, 'w1'), 'add',
          or_inst(W_TMU0_S, areg(RA_AADR), areg(RA_AADR), ws=0), ws=0)
    for r in range(4):
        issue(slot(0, r, 'w4'), POP_ORDER[r])
    a.put(slot(0, 5, 'w4'), 'add',
          addi(ADD_ADD, 32 + ACC_BADR, acc(ACC_BADR), (REGB, RB_BST)), ws=0)
    for r in range(4):
        issue(slot(1, r, 'w4'), POP_ORDER[r])
    a.put(slot(1, 5, 'w4'), 'add',
          addi(ADD_ADD, 32 + ACC_BADR, acc(ACC_BADR), (REGB, RB_BST)), ws=0)

    # TMU1 pops: rows 4-6 on w5, row 7 on w2 so its copy still has a slot after
    for h in range(2):
        for n, r in enumerate((4, 5, 6, 7)):
            j = POP_ORDER[n]
            w = slot(h, r, 'w5') if r != 7 else slot(h, r, 'w2')
            if a.w[w]['sig'] != SIG_NONE:
                raise AssertionError('two pops in one word')
            a.w[w]['sig'] = SIG_LOAD_TMU1
            a.w[w]['note'] += ' POP B j%d' % j
            tgt = bset(j, 1 - h)
            if j < 2:                             # B file copy in a w0 mul slot
                a.put(slot(h, r, 'wc'), 'mul',
                      dict(mul_op=MUL_V8MIN, mul_out=tgt, ws=0,
                           mul_a=acc(ACC_TMU), mul_b=acc(ACC_TMU)), ws=0)
            elif j == 2:                          # A file copy, next row w1
                a.put(slot(h, r + 1, 'w1'), 'add', copy_add(0, tgt), ws=0)
            else:                                 # A file copy, same row w4
                a.put(slot(h, r, 'w4'), 'add', copy_add(0, tgt), ws=0)
    a.w[slot(1, 7, 'w6')]['sig'] = SIG_LOAD_TMU0
    a.w[slot(1, 7, 'w6')]['note'] += ' POP A'

    # two utility words: the counter (flags must be set before the branch) and
    # the A address bump (needs both raddr fields)
    a.word(add=addi(ADD_ADD, RA_KCNT, (REGB, RB_NEG1), areg(RA_KCNT), sf=1),
           note='k counter')
    a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), (REGB, RB_O64)),
           note='AADR += 64')

    a.branch_to('kloop', cond=BR_ALL_Z_CLEAR, note='k loop')
    a.word(mul=dict(mul_op=MUL_V8MIN, mul_out=32 + ACC_AVEC, ws=1,
                    mul_a=acc(ACC_TMU), mul_b=acc(ACC_TMU)),
           note='delay1: r0 = A vector')
    a.nop()
    a.word(mul=rot_inst(rot_for(0, 0)),
           note='delay3: rot(row0) of the next iteration')

    # ----------------------------------------------------------------- drain
    # Every request issued inside a k iteration is popped inside the same
    # iteration, and the prologue balances too, so nothing is left in flight.
    a.nop()

    # ----------------------------------------------------------------- store
    a.label('store')
    dma = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)
    a.word(add=addi(ADD_SHL, 32 + ACC_BADR, areg(RA_VPMQ), imm(7)), note='r3 = 4*qpu*128')
    a.nop()
    for j in range(4):
        for c in range(2):
            a.ldi(0x1A00, R5, ws=1, note='vpw generic row 4q')
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1))
            for r in range(4):
                f, addr = acc_addr(4 * c + r, j)
                a.word(add=or_inst(W_VPM, ref(f, addr), ref(f, addr)))
            a.ldi(dma, R5, ws=1, note='vdw 4 rows x 16 words')
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(ACC_BADR), ws=1))
            a.ldi(0xC0010000 | stride, W_VPW_SETUP, ws=1, note='stride (N-16)*4')
            base = RA_CADR if c == 0 else RA_CADR2
            a.word(add=or_inst(W_VPW_ADDR, areg(base), areg(base), ws=1))
            a.word(add=or_inst(W_NOP, (REGB, R_VPW_WAIT), (REGB, R_VPW_WAIT)))
            a.nop(), a.nop(), a.nop(), a.nop()
        if j != 3:
            a.word(add=addi(ADD_ADD, RA_CADR, areg(RA_CADR), (REGB, RB_O64)),
                   note='next 16 columns')
            a.word(add=addi(ADD_ADD, RA_CADR2, areg(RA_CADR2), (REGB, RB_O64)),
                   note='next 16 columns')
    a.branch_to('tile')
    a.word(add=addi(ADD_ADD, RA_T, areg(RA_T), (REGB, RB_TSTR)), note='delay1: next tile')
    a.nop(), a.nop()

    a.label('tdone')
    a.word(sig=SIG_END_PROGRAM)
    a.nop(), a.nop(), a.nop()
    return a


if __name__ == '__main__':
    ASM = os.path.dirname(os.path.abspath(__file__))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    a = build(N, debug_store=('first' if os.environ.get('DBGSTORE') == 'first' else bool(os.environ.get('DBGSTORE'))))
    a.check()
    path = os.path.join(ASM, 'gemm4_%d%s.bin' % (N, '_dbg' + (os.environ.get('DBGSTORE') or '') if os.environ.get('DBGSTORE') else ''))
    ws = a.write(path)
    print('%s %d instructions' % (os.path.basename(path), len(ws)))
    if os.environ.get('AUDIT'):
        print(a.audit())
    if os.environ.get('LIST'):
        print(a.listing())
