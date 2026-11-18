#!/usr/bin/env python3
"""sim9 -- functional simulation of the asm9 request pipeline (no GPU).

Walks one tile of the emitted program (tile-body -> prime -> k loop -> tail),
modelling

  * the per-TMU request FIFO (the tag is decoded from the address the issue
    actually computes: B -> (k, column block j), A -> k pair),
  * every pop landing in accumulator r4 and being copied out,
  * the register file, and
  * the broadcast/rotate chain (the tag of r5 must be the A tag of the k pair
    the row is multiplying),

and then checks the whole tile:

  * every phase uses all four column blocks of exactly one k,
  * the k values, in program order, are exactly 0, 1, ..., N-1,
  * every row's broadcast and A vector carry the A tag of k//2,
  * nothing is left in flight and no pop hits an empty FIFO.

Usage: python3 sim9.py [N ...]
"""
import sys
import gemm6
from gemm6 import ACC_AVEC, ACC_TMU, ACC_BCAST, ACC_BADR, MUL_V8MIN
from qenc import (ADD_OR, ADD_ADD, ADD_SUB, ADD_AND, ADD_SHL, ADD_SHR,
                  MUL_FMUL, W_NOP, W_TMU0_S, W_TMU1_S, IMM, REGA,
                  REGB, BR_ALL_Z_CLEAR, BR_ALL_N_SET, R_UNIF, R_ELEM_NUM,
                  R_QPU_NUM)

NQPU = 12


class Bug(Exception):
    pass


def run(N):
    a = gemm6.build(N)
    W = a.w
    lab = a.labels
    gp = {}
    acc = {}
    fifo = {0: [], 1: []}
    uni = [N, 0, 0, 0, 0, 0, NQPU]      # u0..u6
    flag_z = False
    flag_neg = False
    fmuls = []
    ucache = {'pc': -1, 'v': 0}

    def val(desc):
        if desc is None:
            return 0
        mux, addr = desc
        if mux == IMM:
            return addr
        if mux == REGA:
            if addr == R_UNIF:
                if ucache['pc'] != pc:      # one uniform FIFO pop per word
                    ucache['v'] = uni.pop(0)
                    ucache['pc'] = pc
                return ucache['v']
            if addr == R_ELEM_NUM:
                return 0
            return gp.get((0, addr), 0)
        if mux == REGB:
            if addr == R_QPU_NUM:
                return 0
            return gp.get((1, addr), 0)
        return acc.get(mux, 0)

    def operator(op, va, vb):
        if op == ADD_OR:
            if isinstance(va, tuple) and isinstance(vb, tuple):
                if va != vb:
                    raise Bug('or of two different tags')
                return va
            if isinstance(va, tuple):
                if vb != 0:
                    raise Bug('or(tag, %r)' % (vb,))
                return va
            if isinstance(vb, tuple):
                if va != 0:
                    raise Bug('or(%r, tag)' % (va,))
                return vb
            return va | vb
        if op in (ADD_ADD, ADD_SUB, ADD_AND, ADD_SHL, ADD_SHR):
            if not (isinstance(va, int) and isinstance(vb, int)):
                raise Bug('int op %d on %r, %r' % (op, va, vb))
            if op == ADD_ADD:
                return va + vb
            if op == ADD_SUB:
                return va - vb
            if op == ADD_AND:
                return va & vb
            if op == ADD_SHL:
                return va << vb
            return va >> vb
        return 0

    stride = 4 * N

    def bt(addr):
        k, rem = divmod(addr, stride)
        if addr < 0 or rem % 64:
            raise Bug('bad B address %d' % addr)
        return (k, rem // 64)

    pc = 0
    pending = None
    steps = 0
    while True:
        steps += 1
        if steps > 8000000:
            raise Bug('runaway at word %d' % pc)
        if pc >= lab['store']:
            break
        d = W[pc]
        if 'ldi' in d:
            v, o, ws = d['ldi']
            if o < 32:
                gp[(0 if ws == 0 else 1, o)] = v
            pc += 1
            continue
        if 'branch' in d:
            cond = d['branch']
            take = (cond == BR_ALL_Z_CLEAR and not flag_z) or \
                   (cond == BR_ALL_N_SET and flag_neg) or \
                   (cond == 15)
            if take:
                pending = [3, lab[d['target']]]     # 3 delay slots, then jump
            pc += 1
            continue
        mul, add = d['mul'], d['add']
        res_add = res_mul = None
        out_add = add.get('add_out') if add else None
        out_mul = mul.get('mul_out') if mul else None
        if add:
            va, vb = val(add.get('add_a')), val(add.get('add_b'))
            res_add = operator(add['add_op'], va, vb)
            if add.get('sf'):
                r32 = (res_add & 0xFFFFFFFF) if isinstance(res_add, int) else res_add
                flag_z = (r32 == 0)
                flag_neg = isinstance(r32, int) and bool(r32 & 0x80000000)
            if out_add in (W_TMU0_S, W_TMU1_S):
                t = 0 if out_add == W_TMU0_S else 1
                if add['add_a'][0] == ACC_BADR:      # B column-block request
                    tag = bt(operator(add['add_op'], va, vb))
                else:                                # A k-pair request
                    tag = ('A', va // 64)
                fifo[t].append(tag)
                if len(fifo[t]) > 4:
                    raise Bug('word %d: TMU%d %d in flight' % (pc, t, len(fifo[t])))
        if mul:
            va = val(mul.get('mul_a'))
            if mul['mul_op'] == MUL_V8MIN:
                ma = mul.get('mul_a')
                if ma[0] == ACC_AVEC or ma[0] == ACC_TMU:
                    res_mul = va
                else:
                    raise Bug('word %d: v8min on mux %r' % (pc, ma))
            else:
                res_mul = ('fmul', va, val(mul.get('mul_b')))
        # ---- writes
        if add and out_add is not None and out_add != W_NOP:
            if 32 <= out_add <= 37:
                acc[out_add - 32] = res_add
            elif out_add < 32:
                gp[(0 if add.get('ws', d['ws']) == 0 else 1, out_add)] = res_add
        if mul and out_mul is not None and out_mul != W_NOP:
            if 32 <= out_mul <= 37:
                acc[out_mul - 32] = res_mul
            elif out_mul < 32:
                gp[(1 if mul.get('ws', d['ws']) == 0 else 0, out_mul)] = res_mul
        # ---- record vector use
        if mul and mul['mul_op'] == MUL_FMUL:
            bv = val(mul['mul_b'])
            if not (isinstance(bv, tuple) and len(bv) == 2
                    and isinstance(bv[0], int)):
                raise Bug('word %d: fmul on a non-B tag %r' % (pc, bv))
            fmuls.append((bv[0], bv[1], acc.get(ACC_BCAST), acc.get(ACC_AVEC)))
        # ---- pop last
        if d['sig'] in (10, 11):
            t = 0 if d['sig'] == 10 else 1
            if not fifo[t]:
                raise Bug('word %d (%s): pop on empty TMU%d' % (pc, d['note'], t))
            acc[ACC_TMU] = fifo[t].pop(0)
        if pending is not None:
            pending[0] -= 1
            if pending[0] == 0:
                pc = pending[1]
                pending = None
                continue
        pc += 1

    if fifo[0] or fifo[1]:
        raise Bug('%d/%d requests left in flight' % (len(fifo[0]), len(fifo[1])))
    if len(fmuls) % 32:
        raise Bug('fmul count %d not a multiple of 32' % len(fmuls))
    order = []
    for ph in range(len(fmuls) // 32):
        grp = fmuls[32 * ph:32 * ph + 32]
        ks = set(x[0] for x in grp)
        if len(ks) != 1:
            raise Bug('phase %d mixes k %s' % (ph, sorted(ks)))
        k = ks.pop()
        if sorted(x[1] for x in grp[:4]) != [0, 1, 2, 3]:
            raise Bug('phase %d block set %s' % (ph, sorted(x[1] for x in grp[:4])))
        for i, (kg, jg, bc, av) in enumerate(grp):
            if bc != ('A', k // 2) or av != ('A', k // 2):
                raise Bug('phase %d (k=%d) row %d: broadcast %r A %r, want %r'
                          % (ph, k, i // 4, bc, av, ('A', k // 2)))
        order.append(k)
    if order != list(range(N)):
        raise Bug('k order %r..., want 0..%d' % (order[:8], N - 1))
    return len(W), len(fmuls) // 32, order


if __name__ == '__main__':
    for N in ([int(x) for x in sys.argv[1:]] or [64, 128, 256]):
        words, phases, order = run(N)
        print('sim9 N=%d: %d words, %d phases, k order 0..%d OK, FIFO empty'
              % (N, words, phases, order[-1]))
