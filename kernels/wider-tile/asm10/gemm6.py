#!/usr/bin/env python3
"""asm9: 3-phase k loop -- three B vector sets, one k-pair of request lookahead.

Same 8x64 C tile, same 6-word port-aware row as asm8 (gemm5.py), same k-major
packed 8-row A panel, same VPM store.  The only change is the *pipelining* of
the B requests: instead of two B vector sets filled one half (52 words, ~130
cycles) before use, there are now **three** sets and a k loop unrolled over
three k pairs (six halves), so every request is issued a full k pair before it
is popped (issue -> pop distance 57..69 words = ~240..285 cycles, and
issue -> use well over a whole k pair).

Why three sets and six halves
-----------------------------
The A vector (accumulator r0) covers 2 k values (16 lanes = 8 rows x 2 k), and
the v8min rotate can only read an accumulator, so the A vector can never be
double buffered -- the A vector therefore has to change once per k pair and,
crucially, at a *fixed* point in the loop body.  That forces the body to hold
a whole number of k pairs (2 phases each).  A rotating fill distance of two
phases with S sets needs S >= 3 (at any instant one set is in use, one is
ready, one is being filled), so the set index is (phase mod 3) and the body
period must be a multiple of 3.  lcm(2, 3) = 6 halves = 3 k pairs.

  phase p (48 words):  use set u(p) = p mod 3
                       pop+copy set u(p+1)      (issued at phase p-1)
                       issue set u(p+2)         (popped at phase p+1,
                                                 used at phase p+2)

Every phase therefore issues 4 B requests, pops the 4 issued one phase earlier,
and uses the 4 popped one phase before.  Per TMU FIFO (depth 4) the in-flight
count is exactly 4 on TMU0 (2 carried + 2 new, and TMU0 also carries the one
in-flight A request) and 4 on TMU1 at the peak, i.e. exactly at the limit and
never above.

Register budget (this is what made the third set possible)
----------------------------------------------------------
Freeing four registers, none of them on a hot port path:
  * RB_NCB1 (B28)   -> the column-block mask is a small immediate imm(ncb-1)
                       (ncb-1 = 0/1/3), so the register is gone;
  * RB_UB  (B30)    -> the B base moves to an A-file register (RA_UB, A30).
                       Its only uses are raddr_a-only adds (`S3 = UB + r3`),
                       so the port assignment is unchanged;
  * RA_COFF/RA_UP/RA_UB debug-only A registers are reused as vectors (the
    DBGSTORE build is not carried over from asm8).
That leaves 6 A-file + 6 B-file vector registers = three sets of four, each set
2 A-file + 2 B-file (forced by the pairing rule: C(j0),C(j1) live in the A
file, C(j2),C(j3) in the B file, so v(j2),v(j3) must be B-file and v(j1)
A-file; v(j0) must be A-file because the B issues ride on the w1 word whose
fmul reads v(j0) with raddr_a).

The k-pair boundary (A vector change), 5 words
----------------------------------------------
  B0: A pop (sig)          | AADR += 64          (raddr_b = RB_O64)
  B1: r0 <- r4 (mul slot)  | A issue for kp+2    (raddr_a = AADR)
  B2: pad                  | (nothing)
  B3: rot(next phase row 0) (mul_b = small imm)
  B4: k counter (sub imm 1, sf=1)               [between phases 3 and 4]
The back-edge boundary rides in the branch and its three delay slots:
  [p5 w4 r7: A pop] [branch] [delay1: r0 <- r4] [delay2: AADR += 64]
  [delay3: rot(p0 row 0) + A issue for the next k pair]
so it costs no extra words.  One wasted word per iteration (the prologue word
that keeps rot -> rep >= 2 across the back edge).

Per half: 48 words = 8 rows x 6 words (identical row to asm8) + 1 (p0..p5) ;
per iteration 303 words for 3 k pairs = 101 words/k pair (asm8: 104).

Schedule inside a phase (48 words; row r = words 6r..6r+5)
---------------------------------------------------------
  w0 rep(row r)            | rot(row r+1)          [rot(row 0) at the boundary]
  w1 fmul(j0)              | B issue / copy / BADR bump
  w2 fmul(j2)              | fadd(j0)
  w3 fmul(j1)              | fadd(j2)
  w4 fmul(j3)              | fadd(j1)  + TMU pop
  w5 fadd(j3)              | copy from r4
  issues  j0,j2,j1,j3 at w1 of rows 0..3   (TMU0, TMU1, TMU0, TMU1)
  BADR bump at w1 of row 5
  pops    j0 w4r1, j2 w4r3, j1 w4r5, j3 w4r6
  copies  j0 w5r1, j2 w1r4, j1 w5r5, j3 w1r7   (each after its pop, before the
                                                next pop -- every pop lands in
                                                r4)
  rot(next phase row 0) at w0 of row 7 when the A vector does not change

Invariants kept from asm3/4/7/8 (all empirically required on this Si):
  * never TMU issue + TMU pop in one word; no pop on a word with a small
    immediate; one pop per word; <= 4 in flight per TMU (FIFO depth 4);
  * every pop lands in accumulator r4 -> each copy must precede the next pop,
    and the A pop must follow the last B copy of the k pair;
  * TMU issue -> pop >= 8 words, pops in issue (FIFO) order per TMU;
  * A-file and B-file RAW gaps >= 2 words; accumulator gaps >= 2 for the
    A vector (copy -> rot);
  * ws=1 rep; v8min small immediates 49..63 are rotate 1..15 and 48 is
    rotate-by-r5 (dynamic), so "rotate by 0" is the plain accumulator move;
    rot -> rep >= 2;
  * B column block 64*K bytes, VPM DMA store DEPTH=16, UNITS=4 rows, stride
    (N-16)*4, VPM row base 4*qpu, 16 encoded as 0.

Build:  python3 gemm6.py <N>        (LAYOUT=1 AUDIT=1 LIST=1)
"""
import os
import struct
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qenc
from qenc import *          # noqa: F401,F403
from qenc import _ops, branch, disasm

MUL_V8MIN = 4
NROWS = 8
NCOLS = 64

# ---- A file (16..31) -------------------------------------------------------
RA_AADR, RA_CADR, RA_CADR2, RA_KCNT = 20, 21, 22, 23
RA_T, RA_S1, RA_S2, RA_VPMQ = 24, 25, 26, 27
RA_UB, RA_S3 = 30, 31          # u3 (B base), colblk/lane scratch (becomes BADR)
# ---- B file (16..31) -------------------------------------------------------
RB_BST, RB_NEG1, RB_NEGNT, RB_UP = 20, 21, 22, 23
RB_O64, RB_O128, RB_O192, RB_CSTEP = 24, 25, 26, 27
RB_TSTR, RB_UC = 30, 31

# Three B vector sets.  C(j0),C(j1) -> A file, C(j2),C(j3) -> B file, so the
# fmul vector of j2/j3 must be B-file, j1 A-file and j0 A-file (the B issues
# ride on the w1 word, whose fmul is j0 and must use raddr_a).
SETV = (
    {0: (0, 16), 1: (0, 17), 2: (1, 16), 3: (1, 17)},
    {0: (0, 18), 1: (0, 19), 2: (1, 18), 3: (1, 19)},
    {0: (0, 28), 1: (0, 29), 2: (1, 28), 3: (1, 29)},
)   # (file, address); file 0 = A file, 1 = B file

ACC_AVEC, ACC_ROT, ACC_TMP, ACC_BADR, ACC_TMU, ACC_BCAST = 0, 1, 2, 3, 4, 5
ACC_TMP2 = ACC_TMP
ACC_SCR = 3                 # same register as BADR (bumped after its last use)

# fmul order inside a row; the fadds trail by one word in the same order
FORDER = (0, 2, 1, 3)      # v(j0) is A-file: w1 can host a raddr_b add op
BORDER = (2, 0, 3, 1)      # v(j2) is B-file: w1 can host a raddr_a add op
ISSUE_ROW = {0: 0, 2: 1, 1: 2, 3: 3}     # w1 of which row carries the issue
TMU_OF = {0: 0, 2: 1, 1: 0, 3: 1}
POP_RW = {0: (1, 4), 2: (3, 4), 1: (5, 4), 3: (6, 4)}
COPY_RW = {0: (1, 5, 'mul'), 2: (4, 1, 'add'), 1: (5, 5, 'mul'), 3: (7, 1, 'add')}
BUMP_RW = (5, 1)
APOP_RW = (7, 4)


def acc_addr(row, j):
    """(file, address) of C accumulator (row, j); file 0 = A file, 1 = B file."""
    return (0, 2 * row + j) if j < 2 else (1, 2 * row + (j - 2))


def ref(f, addr):
    return areg(addr) if f == 0 else breg(addr)


def fmul_j(j, vec, t):
    f, addr = vec
    return dict(mul_op=MUL_FMUL, mul_out=32 + t, mul_a=acc(ACC_BCAST),
                mul_b=ref(f, addr))


def rot_inst(n):
    """v8min rotate of the A vector by n lanes (qenc.rot_inst).

    Small immediates: 48 = rotate by r5 (dynamic), 49..63 = rotate right by
    1..15 lanes, so rotate by 0 has no code -- it is the plain accumulator
    move (v8min acc, acc).  1.0f is small immediate 32, not 48.  The codes
    48..63 also alias the constants -16..-1 when read as a constant.

    The B MUX must be REGB (the small-immediate mux).  vc4asm encodes this
    operand as ACC0; measured on this board that does not rotate (65533/65536
    wrong GEMM elements), while REGB is element-exact.
    """
    return qenc.rot_inst(n, src=ACC_AVEC, out=ACC_ROT)


def fadd_j(r, j, t):
    f, addr = acc_addr(r, j)
    return dict(add_op=ADD_FADD, add_out=addr, ws=f, add_a=ref(f, addr),
                add_b=acc(t))


def or_inst(out, x, y, ws=0):
    return dict(add_op=ADD_OR, add_out=out, ws=ws, add_a=x, add_b=y)


def addi(op, out, x, y, ws=0, sf=0):
    return dict(add_op=op, add_out=out, ws=ws, add_a=x, add_b=y, sf=sf)


def rep_inst(ws=1):
    return dict(add_op=ADD_OR, add_out=37, ws=ws, add_a=acc(ACC_ROT),
                add_b=acc(ACC_ROT))


def copy_add(f, addr):
    """copy ACC_TMU -> GP register addr in file f, on the add ALU."""
    return dict(add_op=ADD_OR, add_out=addr, ws=f, add_a=acc(ACC_TMU),
                add_b=acc(ACC_TMU))


def copy_mul(f, addr):
    """copy ACC_TMU -> GP register addr in file f, on the mul ALU (ws is
    inverted for the mul ALU)."""
    return dict(mul_op=MUL_V8MIN, mul_out=addr, ws=1 - f,
                mul_a=acc(ACC_TMU), mul_b=acc(ACC_TMU))


def issue_b(j, tmu, ws=0):
    """issue the B request for column block j at BADR + 64j on TMU0/TMU1.

    BADR is an accumulator, so these use raddr_b only and can share a word
    whose mul slot reads an A-file vector.
    """
    reg = W_TMU1_S if tmu else W_TMU0_S
    if j == 0:
        return or_inst(reg, acc(ACC_BADR), acc(ACC_BADR), ws=ws)
    return addi(ADD_ADD, reg, acc(ACC_BADR),
                (REGB, [None, RB_O64, RB_O128, RB_O192][j]), ws=ws)


def issue_a(ws=0):
    """issue the A request at AADR on TMU0 (raddr_a only)."""
    return or_inst(W_TMU0_S, areg(RA_AADR), areg(RA_AADR), ws=ws)


def bump_badr(ws=0):
    return addi(ADD_ADD, 32 + ACC_BADR, acc(ACC_BADR), (REGB, RB_BST), ws=ws)


def rot_for(h, r):
    return (16 - (r + (8 if h else 0))) & 15


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

        # -- 3b. A vector (acc r0) write -> v8min read >= 2
        for i, rec in enumerate(W):
            if (ACC, ACC_AVEC) in [w[:2] for w in rec['wr']]:
                for k in range(i + 1, min(i + 8, n)):
                    if (ACC, ACC_AVEC) in W[k]['rd']:
                        if k - i < 2:
                            errs.append('word %d: A copy->rot only %d words'
                                        % (i, k - i))
                        break
                    if any(w[0] == ACC and w[1] == ACC_AVEC for w in W[k]['wr']):
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
                if (ACC, ACC_TMU) in W[j]['rd']:
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
    if debug_store:
        raise NotImplementedError('DBGSTORE builds are not carried over to asm9 '
                                  '(its registers are the third B set); use '
                                  'asm8/gemm5.py for that tooling')
    assert N % NCOLS == 0 and N % NROWS == 0
    ncb = N // NCOLS
    nrb = N // NROWS
    ntiles = ncb * nrb
    KPAIRS = N // 2
    NBODY = KPAIRS // 3
    REM = KPAIRS % 3
    assert NBODY >= 1 and REM in (1, 2)
    lg_ncb = ncb.bit_length() - 1
    SROWB = (32 * N).bit_length() - 1
    stride = (N - 16) * 4

    a = Asm()

    # ---------------------------------------------------------------- setup
    a.word(add=or_inst(36, imm(1), imm(1)), note='tmu_noswap')
    a.word(add=or_inst(RA_S1, areg(R_UNIF), areg(R_UNIF)), note='u0 N (unused)')
    a.word(add=or_inst(RB_UC, areg(R_UNIF), areg(R_UNIF), ws=1), note='u1 C')
    a.word(add=or_inst(RA_S2, areg(R_UNIF), areg(R_UNIF)), note='u2 A (unused)')
    a.word(add=or_inst(RA_UB, areg(R_UNIF), areg(R_UNIF)), note='u3 B (A file)')
    a.word(add=or_inst(RB_UP, areg(R_UNIF), areg(R_UNIF), ws=1), note='u4 panel')
    a.word(add=or_inst(RA_T, areg(R_UNIF), areg(R_UNIF)), note='u5 tile0')
    a.word(add=or_inst(RB_TSTR, areg(R_UNIF), areg(R_UNIF), ws=1), note='u6 tile stride')
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
    a.word(add=addi(ADD_SHL, 32 + ACC_SCR, areg(RA_S1), imm(8)), note='r3 = colblk*256')
    a.word(add=addi(ADD_ADD, RA_AADR, (REGB, RB_UP), areg(RA_S2)), note='panel + rowoff')
    a.word(add=addi(ADD_ADD, RA_CADR, (REGB, RB_UC), areg(RA_S2)), note='C + rowoff')
    a.nop()
    a.word(add=addi(ADD_ADD, RA_CADR, areg(RA_CADR), acc(ACC_SCR)), note='+ colblk*256')
    a.word(add=addi(ADD_SHL, 32 + ACC_SCR, areg(RA_S1), imm(8)), note='r3 = colblk*256')
    a.word(add=addi(ADD_ADD, RA_S3, areg(RA_UB), acc(ACC_SCR)), note='S3 = B + colblk*256')
    a.word(add=addi(ADD_ADD, RA_CADR2, areg(RA_CADR), (REGB, RB_CSTEP)), note='+ 4 C rows')
    a.ldi(NBODY, RA_KCNT, note='k-loop iterations (3 k pairs each)')
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
    a.word(add=or_inst(32 + ACC_TMP, imm(0), imm(0)), note='r2 = 0 (mul temp)')
    a.nop()

    skip = os.environ.get('SKIP', '')

    # ----------------------------------------------------- prime (per tile)
    # A(kp0) and set 0 are filled and popped; set 1 is issued and left in
    # flight (popped by phase 0 of the first iteration); A(kp1) is issued and
    # left in flight (popped at the first k-pair boundary).
    a.word(add=or_inst(W_TMU0_S, areg(RA_AADR), areg(RA_AADR)), note='prime issue A kp0')
    a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), (REGB, RB_O64)), note='AADR += 64')
    prime_issue(a, 'prime s0')
    for _ in range(8):
        a.nop()
    a.word(sig=SIG_LOAD_TMU0, note='prime POP A')
    a.word(add=copy_add(0, 32 + ACC_AVEC), note='prime r0 = A')
    prime_pop(a, SETV[0], 'prime s0')
    prime_issue(a, 'prime s1 (left in flight)')
    a.nop()
    a.word(mul=rot_inst(rot_for(0, 0)), note='prime rot(p0 row 0)')

    if skip in ('all', 'prime'):
        a.branch_to('store', note='SKIP prime+loop')
        a.nop(), a.nop(), a.nop()
    if skip == 'loop':
        a.branch_to('store', note='SKIP kloop')
        a.nop(), a.nop(), a.nop()

    # ------------------------------------------------------------ k loop
    a.label('kloop')
    a.nop(note='prologue (keeps rot->rep >= 2 across the back edge)')
    for p in range(6):
        emit_phase(a, p, p & 1, 'p%d' % p, do_pop=True, do_issue=True,
                   a_pop=(p == 5), rot_next=(p in (0, 2, 4)),
                   a_issue=(p in (0, 2, 4)))
        if p in (1, 3):
            emit_boundary(a, p, counter=(p == 3))
    a.branch_to('kloop', cond=BR_ALL_Z_CLEAR, note='k loop')
    a.word(mul=dict(mul_op=MUL_V8MIN, mul_out=32 + ACC_AVEC, ws=1,
                    mul_a=acc(ACC_TMU), mul_b=acc(ACC_TMU)),
           note='delay1: r0 = A vector of the next k pair')
    a.word(add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), (REGB, RB_O64)),
           note='delay2: AADR += 64')
    a.word(mul=rot_inst(rot_for(0, 0)),
           note='delay3: rot(row0) of the next k pair')
    body = len(a.w) - a.labels['kloop']
    d_rot_rep = (a.labels['kloop'] + 2) - (a.labels['kloop'] - 1)
    assert d_rot_rep >= 2, d_rot_rep

    # ------------------------------------------------------------ tail
    # The last k-pair of the main loop prefetched set 0 (next k even) and set 1
    # (next k odd); REM = 1 uses them as they are, REM = 2 also fills set 2.
    a.nop(note='tail entry pad (rot->rep >= 2 on the loop-exit path)')
    if REM == 1:
        emit_phase(a, 6, 0, 't0', do_pop=True, do_issue=False, rot_next=True)
        emit_phase(a, 7, 1, 't1', do_pop=False, do_issue=False, rot_next=False)
    else:
        emit_phase(a, 6, 0, 't0', do_pop=True, do_issue=True, rot_next=True,
                   a_issue=True)
        emit_phase(a, 7, 1, 't1', do_pop=True, do_issue=True, rot_next=False)
        emit_boundary(a, 7, counter=False)
        emit_phase(a, 8, 0, 't2', do_pop=True, do_issue=False, rot_next=True)
        emit_phase(a, 9, 1, 't3', do_pop=False, do_issue=False, rot_next=False)
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

    if os.environ.get('LAYOUT'):
        print('N=%d: NBODY=%d REM=%d  loop body %d words (%.1f per k pair), '
              'rot->rep %d' % (N, NBODY, REM, body, body / 3.0, d_rot_rep))
    return a


def prime_issue(a, tag):
    for j in FORDER:
        a.word(add=issue_b(j, TMU_OF[j]), note='%s issue j%d' % (tag, j))
    a.word(add=bump_badr(), note='%s BADR bump' % tag)


def prime_pop(a, s, tag):
    for j in FORDER:
        a.word(sig=SIG_LOAD_TMU1 if TMU_OF[j] else SIG_LOAD_TMU0,
               note='%s POP j%d' % (tag, j))
        f, addr = s[j]
        a.word(add=copy_add(f, addr), note='%s copy j%d' % (tag, j))


def emit_phase(a, p, h, tag, do_pop=True, do_issue=True, a_pop=False,
               rot_next=False, a_issue=False):
    """Emit one 48-word half: 8 rows x 6 words, set (p mod 3) in use.

    Uses SETV[p%3], pops/copies SETV[(p+1)%3] (issued at phase p-1) and issues
    SETV[(p+2)%3] (popped at phase p+1, used at phase p+2).
    """
    su = SETV[p % 3]
    sp = SETV[(p + 1) % 3]
    si = SETV[(p + 2) % 3]
    idx = {}
    orders = [FORDER] * NROWS
    if a_issue:
        # row 6 carries the A issue: its w1 mul must read a B-file vector so
        # that raddr_a stays free for the issue (same 6-word row, valid
        # alternating order, temp chain unchanged)
        orders[6] = BORDER
    for r in range(NROWS):
        o = orders[r]
        idx[(r, 0)] = a.word(add=rep_inst(ws=1), ws=1,
                             note='%s r%d w0 rep' % (tag, r))
        idx[(r, 1)] = a.word(mul=fmul_j(o[0], su[o[0]], ACC_TMP),
                             note='%s r%d w1 fmul j%d' % (tag, r, o[0]))
        idx[(r, 2)] = a.word(mul=fmul_j(o[1], su[o[1]], ACC_TMP2),
                             add=fadd_j(r, o[0], ACC_TMP),
                             note='%s r%d w2 fmul j%d|fadd j%d'
                                  % (tag, r, o[1], o[0]))
        idx[(r, 3)] = a.word(mul=fmul_j(o[2], su[o[2]], ACC_TMP),
                             add=fadd_j(r, o[1], ACC_TMP2),
                             note='%s r%d w3 fmul j%d|fadd j%d'
                                  % (tag, r, o[2], o[1]))
        idx[(r, 4)] = a.word(mul=fmul_j(o[3], su[o[3]], ACC_TMP2),
                             add=fadd_j(r, o[2], ACC_TMP),
                             note='%s r%d w4 fmul j%d|fadd j%d'
                                  % (tag, r, o[3], o[2]))
        idx[(r, 5)] = a.word(add=fadd_j(r, o[3], ACC_TMP2), ws=1,
                             note='%s r%d w5 fadd j%d' % (tag, r, o[3]))
    # rot(row i) -> mul slot of w0 of row i-1 (same word as that row's rep)
    for i in range(1, NROWS):
        a.put(idx[(i - 1, 0)], 'mul', rot_inst(rot_for(h, i)))
    if rot_next:
        # A vector does not change at this phase boundary
        a.put(idx[(NROWS - 1, 0)], 'mul', rot_inst(rot_for(1 - h, 0)))
    if do_issue:
        for j in FORDER:
            a.put(idx[(ISSUE_ROW[j], 1)], 'add', issue_b(j, TMU_OF[j]))
        a.put(idx[BUMP_RW], 'add', bump_badr())
    if a_issue:
        # the A request for the NEXT k pair: issued here, popped at the k-pair
        # boundary at the end of the following phase (one k pair of lookahead,
        # ~60 words).  It must sit in the TMU0 FIFO after this k pair's B
        # issues and before the next phase's, which is exactly the pop order.
        a.put(idx[(6, 1)], 'add', issue_a())
    if do_pop:
        # every pop lands in r4: pop and copy in an order where each copy
        # precedes the next pop (j0 w4r1, j2 w4r3, j1 w4r5, j3 w4r6)
        for j in FORDER:
            r, w = POP_RW[j]
            a.pop(idx[(r, w)], TMU_OF[j], '%s j%d' % (tag, j))
        for j in FORDER:
            r, w, slot = COPY_RW[j]
            f, addr = sp[j]
            op = copy_add(f, addr) if slot == 'add' else copy_mul(f, addr)
            a.put(idx[(r, w)], slot, op)
    if a_pop:
        a.pop(idx[APOP_RW], 0, '%s A' % tag)
    return idx


def emit_boundary(a, p, counter=False):
    """A-vector change before phase p+1: pop, copy, rot, issue for k pair +2."""
    a.word(sig=SIG_LOAD_TMU0,
           add=addi(ADD_ADD, RA_AADR, areg(RA_AADR), (REGB, RB_O64)),
           note='B%d: A pop + AADR += 64' % p)
    a.word(mul=copy_mul(0, 32 + ACC_AVEC),
           note='B%d: r0 = A vector' % p)
    a.nop(note='B%d: copy->rot gap' % p)
    a.word(mul=rot_inst(rot_for(0, 0)), note='B%d: rot(p%d row 0)' % (p, p + 1))
    if counter:
        a.word(add=addi(ADD_SUB, RA_KCNT, areg(RA_KCNT), imm(1), sf=1),
               note='k counter')
    else:
        a.nop(note='B%d: rot->rep gap' % p)


if __name__ == '__main__':
    ASM = os.path.dirname(os.path.abspath(__file__))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    a = build(N)
    a.check()
    a.verify()
    path = os.path.join(ASM, 'gemm6_%d.bin' % N)
    ws = a.write(path)
    print('%s %d instructions' % (os.path.basename(path), len(ws)))
    if os.environ.get('AUDIT'):
        print(a.audit())
    if os.environ.get('LIST'):
        print(a.listing())
