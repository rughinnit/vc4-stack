#!/usr/bin/env python3
"""asm8: VideoCore IV QPU GEMM, 8x64 C tile, 6-word k-loop rows (port-aware).

Same 8x64 wider tile as asm4/asm7 (16 lanes x 4 columns per lane, 8 rows,
8-row k-major packed panel).  The k loop is re-scheduled from 7 to 6 words per
row: 104 words per k-pair iteration instead of 118, i.e. 410 instead of 484
execution cycles per k-pair (0.40 instead of 0.47 cycles/MAC at 12 QPUs).

Per row (h = k-parity half, r = row), one temp, cross-group pairing:

   w0: rot(row+1)      | rep(row r)      ACC_ROT -> r5 broadcast (ws=1)
   w1: fmul(j0) ra     | <issue / BADR bump / copy>
   w2: fmul(j2) rb     | fadd(j0) ra
   w3: fmul(j1) ra     | fadd(j2) rb
   w4: fmul(j3) rb     | fadd(j1) ra
   w5: <A-file copy>   | fadd(j3) rb

Why the row fits in 6 words
---------------------------
The add ALU has exactly one A-side (raddr_a) and one B-side (raddr_b) port,
so a word may hold one fmul and one fadd only if their operands come from
OPPOSITE register files (the fmul reads its B vector, the fadd reads its C
accumulator).  asm7 keeps vector_file(j) = opposite(accum_file(j)), which
allows only same-group pairing (j0/j1 fmul with j0/j1 fadd) and forces three
words per group - fmul_a, fmul_b|fadd_a, fadd_b - hence 7 words.  Here the
vector registers are swapped (j0/j1 vectors live in the A file, j2/j3 in the
B file, i.e. vector_file(j) == accum_file(j)), which makes CROSS-group pairing
legal, and then 4 fmul + 4 fadd + rep + rot fit in 6 words.  6 is the floor:
rep -> 4 fmul -> the last fadd is 6 words, and the rep needs a word of its own
because the fmuls read r5 one word later.

The rot for row i sits in w0 of row i-1 - the same word as that row's rep.  The
mul ALU writes ACC_ROT while the add ALU reads it and the add reads the OLD
value: the same same-word WAR idiom asm7 relies on for its single mul temp.

Why there is still only ONE mul temp (the r3 plan does not work)
---------------------------------------------------------------
NOTES7 proposed moving BADR to a general register to free accumulator r3 as a
second temp.  Accumulators are r0..r5 and every one of them is pinned:
  r0 A vector  - the v8min rotate only works on an accumulator operand (with a
                 general-register source the rotate is applied per 4-lane
                 group, verified: rows then repeat with period 4),
  r1 rot       - the rep reads it on the same word, which is only port-free
                 for an accumulator,
  r3 BADR      - as an accumulator the offset B issues use raddr_b only and
                 can share a word with an fmul; in a general register they
                 would need BOTH ports and 10 dedicated words per iteration,
  r4 TMU read  - fixed by the ISA (every TMU pop lands in r4),
  r5 broadcast - fixed by the ISA (the rep writes it).
So a second temp cannot be had, and it is not needed: the 6-word row above uses
one temp with alternating same-word WAR, exactly like asm7's fastest row.

Invariants kept from asm3/asm4/asm7 (all empirically required on this Si):
  * never TMU issue + TMU pop in one word;
  * no pop on a word with a small immediate (the rotate words are w0);
  * one pop per word, <= 4 requests in flight per TMU (FIFO depth 4);
  * every pop lands in accumulator r4, so each copy must precede the NEXT pop
    and the A pop must follow the last B copy (two bugs found this way);
  * TMU issue -> pop distance >= 8 words, pops in issue (FIFO) order;
  * A-file and B-file RAW gaps >= 2 words (accumulator gaps are 1);
  * ws bit: add writes A file (ws=0) / B file (ws=1) and the mul ALU writes
    the opposite file, so a word's fadd file and its mul copy must agree;
  * ws=1 rep, v8min small immediate 49..63 only (48 is 1.0f, so "rotate by
    0" is an exact self-copy);
  * B column block 64*K, VPM DMA store DEPTH=16 words, UNITS=4 rows, stride
    (N-16)*4, VPM row base 4*qpu (12 QPUs x 4 rows), 16 encoded as 0.

Build a size with:  python3 gemm5.py <N>      (LAYOUT=1 AUDIT=1 LIST=1)
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
# TMU1 issue order inside a half (also the pop order: the FIFO is in order).
# j0/j1 are copied last (into the A-file vectors at w5, the only free mul
# slot) and are read first in the next half, so they go last.
ISSUE_ORDER = (2, 3, 0, 1)

# ---- A file ----------------------------------------------------------------
# j0/j1 B vectors live here (same file as their C accumulators, which is what
# makes the cross-group fmul/fadd pairing legal)
RA_V0, RA_V1, RA_V0Q, RA_V1Q = 16, 17, 18, 19
RA_AADR, RA_CADR, RA_CADR2, RA_KCNT = 20, 21, 22, 23
RA_T, RA_S1, RA_S2, RA_VPMQ, RA_COFF, RA_UP = 24, 25, 26, 27, 28, 29
RA_UB = 30                  # unused (kept for the debug paths)
RA_S3 = 31
# ---- B file ----------------------------------------------------------------
# j2/j3 B vectors live here
RB_V2, RB_V3, RB_V2Q, RB_V3Q = 16, 17, 18, 19
RB_BST, RB_NEG1, RB_NEGNT, RB_UP = 20, 21, 22, 23
RB_O64, RB_O128, RB_O192, RB_CSTEP = 24, 25, 26, 27
RB_NCB1, RB_TSTR, RB_UB, RB_UC = 28, 29, 30, 31
# ---- accumulator bank ------------------------------------------------------
# Six accumulators, seven roles (A vector, rot, mul temp, BADR, TMU read,
# broadcast) is the reason a second mul temp cannot be had: every role is
# pinned.  The A vector must stay an accumulator (the v8min rotate only works
# on an accumulator operand), the rot must write one (the rep reads it port
# free on the same word) and BADR must be one (so that the offset B issues use
# raddr_b only).  One temp is enough for 6 words/row - see the header.
ACC_AVEC, ACC_ROT, ACC_TMP, ACC_BADR, ACC_TMU, ACC_BCAST = 0, 1, 2, 3, 4, 5
ACC_TMP2 = ACC_TMP
ACC_SCR = 3                 # same register as BADR (bumped after its last use)


def acc_addr(row, j):
    """(file, address) of C accumulator (row, j); file 0 = A file, 1 = B file."""
    return (0, 2 * row + j) if j < 2 else (1, 2 * row + (j - 2))


def ref(f, addr):
    return areg(addr) if f == 0 else breg(addr)


def bset(j, h):
    return {0: (RA_V0, RA_V0Q), 1: (RA_V1, RA_V1Q),
            2: (RB_V2, RB_V2Q), 3: (RB_V3, RB_V3Q)}[j][h]


def fmul_j(j, h, t):
    f = 0 if j < 2 else 1                     # vector file == accumulator file
    return dict(mul_op=MUL_FMUL, mul_out=32 + t, mul_a=acc(ACC_BCAST),
                mul_b=ref(f, bset(j, h)))


def rot_inst(n):
    if n == 0:
        # v8min's small immediate 48 is the float 1.0, not "rotate by 0"
        # (only 49..63 are the rotate amounts 1..15), so row 0 ("no
        # rotation") needs an exact copy of the A vector instead.
        return dict(mul_op=MUL_V8MIN, mul_out=32 + ACC_ROT,
                    mul_a=acc(ACC_AVEC), mul_b=acc(ACC_AVEC))
    return dict(mul_op=MUL_V8MIN, mul_out=32 + ACC_ROT, mul_a=acc(ACC_AVEC),
                mul_b=imm(48 + n))


def fadd_j(r, j, t):
    f, addr = acc_addr(r, j)
    return dict(add_op=ADD_FADD, add_out=addr, ws=f, add_a=ref(f, addr),
                add_b=acc(t))


def or_inst(out, x, y, ws=0):
    return dict(add_op=ADD_OR, add_out=out, ws=ws, add_a=x, add_b=y)


def addi(op, out, x, y, ws=0, sf=0):
    return dict(add_op=op, add_out=out, ws=ws, add_a=x, add_b=y, sf=sf)


def rep_inst(ws=1):
    # r5 is the broadcast port; the source is the rotate result ACC_ROT.
    return dict(add_op=ADD_OR, add_out=37, ws=ws, add_a=acc(ACC_ROT),
                add_b=acc(ACC_ROT))


def copy_add(f, addr):
    """copy ACC_TMU -> GP register addr in file f, on the add ALU."""
    return dict(add_op=ADD_OR, add_out=addr, ws=f, add_a=acc(ACC_TMU),
                add_b=acc(ACC_TMU))


def copy_mul(f, addr):
    """copy ACC_TMU -> GP register addr in file f, on the mul ALU.

    The mul ALU writes the file opposite to ws, so file f needs ws=1-f.
    """
    return dict(mul_op=MUL_V8MIN, mul_out=addr, ws=1 - f,
                mul_a=acc(ACC_TMU), mul_b=acc(ACC_TMU))


def issue_b(j, tmu=1, ws=0):
    """issue the B request for column block j at BADR + 64 j on TMU0/TMU1.

    BADR is an accumulator, so these use raddr_b only (or nothing) and can
    share a word whose mul slot reads an A-file vector.  The four B requests
    of a half are split 2/2 over the two TMUs so that each TMU FIFO holds at
    most two/three entries instead of four.
    """
    reg = W_TMU1_S if tmu else W_TMU0_S
    if j == 0:
        return or_inst(reg, acc(ACC_BADR), acc(ACC_BADR), ws=ws)
    return addi(ADD_ADD, reg, acc(ACC_BADR),
                (REGB, [None, RB_O64, RB_O128, RB_O192][j]), ws=ws)


def bump_badr(ws=0):
    return addi(ADD_ADD, 32 + ACC_BADR, acc(ACC_BADR), (REGB, RB_BST), ws=ws)


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

    # ---- static checks ----------------------------------------------------
    def check(self):
        """Structural checks (encoder level)."""
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
                            # one port can serve several operands, but only if
                            # they are the same register
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
                    continue                      # special registers
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
                if f != ACC or a not in (ACC_TMP, ACC_TMP2):
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
            if (ACC, ACC_BCAST) in [w[:2] for w in rec['wr']]:
                last_rep = i
            if (ACC, ACC_BCAST) in rec['rd']:
                if last_rep is None:
                    errs.append('word %d (%s): broadcast read before rep'
                                % (i, rec['note']))
                elif i - last_rep < 1:
                    errs.append('word %d (%s): broadcast read %d words after rep'
                                % (i, rec['note'], i - last_rep))

        # -- 3. rot write (acc r1) -> rep read >= 2
        for i, rec in enumerate(W):
            if (ACC, ACC_ROT) in [w[:2] for w in rec['wr']]:
                for k in range(i + 1, min(i + 8, n)):
                    if (ACC, ACC_ROT) in W[k]['rd']:
                        if k - i < 2:
                            errs.append('word %d: rot->rep only %d words'
                                        % (i, k - i))
                        break
                    if any(w[0] == ACC and w[1] == ACC_ROT for w in W[k]['wr']):
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

        if errs:
            raise AssertionError('schedule verify failed:\n  ' + '\n  '.join(errs))
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
    nrb = N // NROWS
    ntiles = ncb * nrb
    assert ncb & (ncb - 1) == 0 and nrb & (nrb - 1) == 0
    lg_ncb = ncb.bit_length() - 1
    lg_nrb = nrb.bit_length() - 1
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
    a.ldi(0xFFFFFFFF, RB_NEG1, ws=1, note='-1')
    a.ldi((1 << 32) - ntiles, RB_NEGNT, ws=1, note='-ntiles')
    a.ldi(ncb - 1, RB_NCB1, ws=1, note='column blocks - 1')
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
    a.word(add=addi(ADD_AND, RA_S1, areg(RA_T), (REGB, RB_NCB1)), note='colblk')
    a.word(add=addi(ADD_SHR, RA_S2, areg(RA_T), imm(lg_ncb)), note='rowblk')
    a.nop(), a.nop()
    a.word(add=addi(ADD_SHL, RA_S2, areg(RA_S2), imm(SROWB)), note='rowoff = rowblk*32N')
    a.word(add=addi(ADD_SHL, 32 + ACC_SCR, areg(RA_S1), imm(8)), note='r3 = colblk*256')
    a.word(add=addi(ADD_ADD, RA_AADR, (REGB, RB_UP), areg(RA_S2)), note='panel + rowoff')
    a.word(add=addi(ADD_ADD, RA_CADR, (REGB, RB_UC), areg(RA_S2)), note='C + rowoff')
    # an A-file write is not visible to the next instruction (asm3 has two nops
    # here); without this gap CADR keeps a stale value and the tile store lands
    # in the packed A panel instead of C
    a.nop()
    a.word(add=addi(ADD_ADD, RA_CADR, areg(RA_CADR), acc(ACC_SCR)), note='+ colblk*256')
    # 64 columns per column block = 256 bytes (not 64: that is the 16-column
    # tile of asm3)
    a.word(add=addi(ADD_SHL, 32 + ACC_SCR, areg(RA_S1), imm(8)), note='r3 = colblk*256')
    a.word(add=addi(ADD_ADD, RA_S3, (REGB, RB_UB), acc(ACC_SCR)), note='S3 = B + colblk*256')
    a.word(add=addi(ADD_ADD, RA_CADR2, areg(RA_CADR), (REGB, RB_CSTEP)), note='+ 4 C rows')
    a.ldi(KPAIRS, RA_KCNT, note='k pairs')
    a.word(add=addi(ADD_SHL, 32 + ACC_SCR, areg(R_ELEM_NUM), imm(2)), note='r3 = lane*4')
    a.nop()
    a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), acc(ACC_SCR)), note='+ lane*4')
    a.word(add=addi(ADD_ADD, RA_S3, areg(RA_S3), acc(ACC_SCR)), note='+ lane*4')
    a.nop()
    a.word(add=or_inst(32 + ACC_BADR, areg(RA_S3), areg(RA_S3)), note='r3 = BADR (lane-offset)')
    for r in range(NROWS):
        for j in range(4):
            f, addr = acc_addr(r, j)
            a.word(add=or_inst(addr, imm(0), imm(0), ws=f))
    a.nop()
    a.word(add=or_inst(32 + ACC_TMP, imm(0), imm(0)), note='r2 = 0 (mul temp 1)')
    a.word(add=or_inst(32 + ACC_TMP2, imm(0), imm(0)), note='r0 = 0 (mul temp 2)')
    a.nop()
    # prime: A(kp0) and B(k0) -> set P
    a.word(add=or_inst(W_TMU0_S, areg(RA_AADR), areg(RA_AADR), ws=1), note='issue A')
    a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), (REGB, RB_O64)), note='AADR += 64')
    for j in ISSUE_ORDER:
        a.word(add=issue_b(j, ws=1), note='issue B j%d' % j)
    a.word(add=bump_badr(), note='BADR -> k1')
    a.nop(), a.nop(), a.nop(), a.nop()
    for j in ISSUE_ORDER:
        tgt = bset(j, 0)
        a.word(sig=SIG_LOAD_TMU1, note='pop B k0 j%d' % j)
        a.word(add=copy_add(1 if j >= 2 else 0, tgt), note='copy B k0 j%d' % j)
    a.word(sig=SIG_LOAD_TMU0, note='pop A')
    a.word(add=copy_add(0, 32 + ACC_AVEC), note='r0 = A vector')
    a.nop()
    a.word(mul=rot_inst(rot_for(0, 0)), note='rot for row0')
    a.nop(note='rot -> rep gap')

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
        a.word(add=addi(ADD_SHL, 32 + ACC_SCR, areg(RA_VPMQ), imm(7)), note='dbg r3 = 4q<<7')
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
    # 6 words per row, two mul temps, vectors in the same file as their C
    # accumulators so that an fmul and an fadd from different groups share a
    # word (one raddr_a operand, one raddr_b operand).
    a.label('kloop')
    a.word(add=or_inst(W_TMU0_S, areg(RA_AADR), areg(RA_AADR)),
           note='body w0: A issue (rot->rep gap)')
    rows = [(h, r) for h in range(2) for r in range(NROWS)]
    idx = {}
    # body word 1 carries half 0's first B issue (an ALU op on TMU1_S never
    # touches ACC_ROT), so the rot(delay3) -> rep distance stays >= 2.
    w_gap = a.word(add=issue_b(ISSUE_ORDER[0]), note='h0 w1: B issue j%d' % ISSUE_ORDER[0])
    for h in range(2):
        for r in range(NROWS):
            d = {}
            d[0] = a.word(add=rep_inst(ws=1), ws=1,
                          note='h%d r%d w0 rep' % (h, r))
            d[1] = a.word(mul=fmul_j(0, h, ACC_TMP),
                          note='h%d r%d w1 fmul j0' % (h, r))
            d[2] = a.word(mul=fmul_j(2, h, ACC_TMP2),
                          add=fadd_j(r, 0, ACC_TMP),
                          note='h%d r%d w2 fmul j2|fadd j0' % (h, r))
            d[3] = a.word(mul=fmul_j(1, h, ACC_TMP),
                          add=fadd_j(r, 2, ACC_TMP2),
                          note='h%d r%d w3 fmul j1|fadd j2' % (h, r))
            d[4] = a.word(mul=fmul_j(3, h, ACC_TMP2),
                          add=fadd_j(r, 1, ACC_TMP),
                          note='h%d r%d w4 fmul j3|fadd j1' % (h, r))
            d[5] = a.word(add=fadd_j(r, 3, ACC_TMP2), ws=1,
                          note='h%d r%d w5 fadd j3' % (h, r))
            idx[(h, r)] = d

    def W(h, r, n):
        return idx[(h, r)][n]

    # rot(row i) -> mul slot of w0 of row i-1 (same word as the rep of row
    # i-1: the add ALU reads the old ACC_ROT).  row 0 of a half is rotated by
    # the previous half's last w0; row (0,0) by the branch delay slot.
    for i in range(1, len(rows)):
        h, r = rows[i]
        ph, pr = rows[i - 1]
        a.put(W(ph, pr, 0), 'mul', rot_inst(rot_for(h, r)))

    def pop(word, tmu, note):
        w = a.w[word]
        if w['sig'] != SIG_NONE:
            raise AssertionError('two pops in word %d' % word)
        w['sig'] = SIG_LOAD_TMU1 if tmu else SIG_LOAD_TMU0
        w['note'] += ' POP ' + note

    # ---------------- half 0: TMU1 issues for set Q (used in half 1).
    # The copy of each pop must sit before the NEXT pop: every TMU pop, on
    # either TMU, lands in accumulator r4, so a later pop destroys an uncopied
    # value (that mistake cost column block j=3 exactly once).
    a.put(W(0, 0, 1), 'add', issue_b(ISSUE_ORDER[1]), ws=0)     # 3
    a.put(W(0, 1, 1), 'add', issue_b(ISSUE_ORDER[2]), ws=0)     # 9
    a.put(W(0, 2, 1), 'add', issue_b(ISSUE_ORDER[3]), ws=0)     # 15
    a.put(W(0, 3, 1), 'add', bump_badr(), ws=0)                 # 21
    pop(W(0, 4, 4), 1, 'B j%d' % ISSUE_ORDER[0])                # 30
    a.put(W(0, 5, 1), 'add', copy_add(1, bset(ISSUE_ORDER[0], 1)))   # 33
    pop(W(0, 5, 4), 1, 'B j%d' % ISSUE_ORDER[1])                # 36
    a.put(W(0, 6, 1), 'add', copy_add(1, bset(ISSUE_ORDER[1], 1)))   # 39
    pop(W(0, 6, 4), 1, 'B j%d' % ISSUE_ORDER[2])                # 42
    a.put(W(0, 6, 5), 'mul', copy_mul(0, bset(ISSUE_ORDER[2], 1)))   # 43
    pop(W(0, 7, 4), 1, 'B j%d' % ISSUE_ORDER[3])                # 48
    a.put(W(0, 7, 5), 'mul', copy_mul(0, bset(ISSUE_ORDER[3], 1)))   # 49

    # ---------------- half 1: TMU1 issues for set P (next k pair)
    a.put(W(1, 0, 1), 'add', issue_b(ISSUE_ORDER[0]), ws=0)     # 51
    a.put(W(1, 1, 1), 'add', issue_b(ISSUE_ORDER[1]), ws=0)     # 57
    a.put(W(1, 2, 1), 'add', issue_b(ISSUE_ORDER[2]), ws=0)     # 63
    a.put(W(1, 3, 1), 'add', issue_b(ISSUE_ORDER[3]), ws=0)     # 69
    a.put(W(1, 4, 1), 'add', bump_badr(), ws=0)                 # 75
    pop(W(1, 4, 4), 1, 'B j%d' % ISSUE_ORDER[0])                # 78
    a.put(W(1, 5, 1), 'add', copy_add(1, bset(ISSUE_ORDER[0], 0)))   # 81
    pop(W(1, 5, 4), 1, 'B j%d' % ISSUE_ORDER[1])                # 84
    a.put(W(1, 6, 1), 'add', copy_add(1, bset(ISSUE_ORDER[1], 0)))   # 87
    pop(W(1, 6, 4), 1, 'B j%d' % ISSUE_ORDER[2])                # 90
    a.put(W(1, 6, 5), 'mul', copy_mul(0, bset(ISSUE_ORDER[2], 0)))   # 91
    pop(W(1, 7, 4), 1, 'B j%d' % ISSUE_ORDER[3])                # 96
    a.put(W(1, 7, 5), 'mul', copy_mul(0, bset(ISSUE_ORDER[3], 0)))   # 97

    # two utility words: both need raddr_a and raddr_b, so neither can share a
    # word with an fmul (whose vector always takes one of the two ports)
    a.word(add=addi(ADD_ADD, RA_KCNT, (REGB, RB_NEG1), areg(RA_KCNT), sf=1),
           note='k counter')
    w_aadr = a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), (REGB, RB_O64)),
                    note='AADR += 64')
    # the A pop MUST come after the last B copy of the iteration: every TMU pop
    # (either TMU) lands in accumulator r4, so a B pop between the A pop and the
    # A copy would hand the A vector the B data
    pop(w_aadr, 0, 'A')

    a.branch_to('kloop', cond=BR_ALL_Z_CLEAR, note='k loop')
    br = len(a.w) - 1
    a.word(mul=dict(mul_op=MUL_V8MIN, mul_out=32 + ACC_AVEC, ws=1,
                    mul_a=acc(ACC_TMU), mul_b=acc(ACC_TMU)),
           note='delay1: r0 = A vector of k+1')
    a.nop()
    a.word(mul=rot_inst(rot_for(0, 0)),
           note='delay3: rot(row0) of the next iteration')
    # loop-carried distances (the linear verifier cannot see across the back
    # edge): rot(delay3) -> rep, A copy(delay1) -> first rot
    d_rot_rep = (br + 4 + 2) - (br + 3)
    d_copy_rot = (br + 4 + 2) - (br + 1)
    assert d_rot_rep >= 2, d_rot_rep
    assert d_copy_rot >= 2, d_copy_rot
    if os.environ.get('LAYOUT'):
        print('loop: rot->rep %d, A-copy->rot %d, iteration %d words'
              % (d_rot_rep, d_copy_rot, br + 4 - a.labels['kloop']))

    # ----------------------------------------------------------------- drain
    # Every request issued inside a k iteration is popped inside the same
    # iteration, and the prologue balances too, so nothing is left in flight.
    a.nop()

    # ----------------------------------------------------------------- store
    a.label('store')
    dma = (2 << 30) | (4 << 23) | (16 << 16) | (1 << 14)
    a.word(add=addi(ADD_SHL, 32 + ACC_SCR, areg(RA_VPMQ), imm(7)), note='r3 = 4*qpu*128')
    a.nop()
    for j in range(4):
        for c in range(2):
            a.ldi(0x1A00, R5, ws=1, note='vpw generic row 4q')
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), areg(RA_VPMQ), ws=1))
            for r in range(4):
                f, addr = acc_addr(4 * c + r, j)
                a.word(add=or_inst(W_VPM, ref(f, addr), ref(f, addr)))
            a.ldi(dma, R5, ws=1, note='vdw 4 rows x 16 words')
            a.word(add=addi(ADD_ADD, W_VPW_SETUP, acc(5), acc(ACC_SCR), ws=1))
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
    a.verify()
    path = os.path.join(ASM, 'gemm5_%d%s.bin' % (N, '_dbg' + (os.environ.get('DBGSTORE') or '') if os.environ.get('DBGSTORE') else ''))
    ws = a.write(path)
    print('%s %d instructions' % (os.path.basename(path), len(ws)))
    if os.environ.get('AUDIT'):
        print(a.audit())
    if os.environ.get('LIST'):
        print(a.listing())
