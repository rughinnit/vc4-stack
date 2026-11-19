#!/usr/bin/env python3
"""Build hand-written VideoCore IV QPU programs for qrun."""
import struct
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qenc import *
from qenc import _ops


class Prog:
    def __init__(self):
        self.w = []
        self.labels = {}
        self.fixups = []

    def emit(self, *words):
        for w in words:
            self.w.append(w)
        return len(self.w)

    def label(self, name):
        self.labels[name] = len(self.w)

    def branch_to(self, name, cond=BR_ALL_Z_CLEAR):
        idx = len(self.w)
        self.w.append(0)
        self.fixups.append((idx, name, cond))
        return idx

    def finish(self):
        for idx, name, cond in self.fixups:
            target = self.labels[name]
            # relative branches are relative to the instruction after the three
            # delay slots, i.e. pc + 4 instructions (verified against vc4c output)
            off = (target - (idx + 4)) * 8
            self.w[idx] = branch(off, cond=cond)
        return self.w

    def write(self, path):
        words = self.finish()
        with open(path, 'wb') as f:
            for w in words:
                f.write(struct.pack('<Q', w))
        return words


def store_vector(value_acc, base_areg, qpu_areg=5, scratch0=2, scratch1=3):
    """Store one 16-word vector (from accumulator value_acc) to [base + qpu*64]."""
    return [
        ldi(0x1A34, R5, ws=1),                                # generic VPM write setup: row 52, matching the DMA base
        _ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=areg(qpu_areg)),
        _ops(add_op=ADD_OR, add_out=W_VPM, add_a=acc(value_acc), add_b=acc(value_acc)),
        ldi(0x80815A00, R5, ws=1),                            # DMA write setup base (depth 16, 1 unit)
        _ops(add_op=ADD_SHL, add_out=scratch1, add_a=areg(qpu_areg), add_b=imm(7)),
        _ops(add_op=ADD_ADD, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=acc(scratch1)),
        ldi(0xC0010000, W_VPW_SETUP, ws=1),                   # memory stride 0
        _ops(add_op=ADD_SHL, add_out=scratch0, add_a=areg(qpu_areg), add_b=imm(6)),
        _ops(add_op=ADD_ADD, add_out=W_VPW_ADDR, ws=1, add_a=acc(scratch0), add_b=areg(base_areg)),
        _ops(add_op=ADD_OR, add_out=W_NOP, add_a=(REGB, R_VPW_WAIT), add_b=(REGB, R_VPW_WAIT)),
    ]


def prog_store_test():
    """out[qpu][lane] = lane + 100*qpu"""
    p = Prog()
    p.emit(_ops(add_op=ADD_OR, add_out=1, add_a=areg(R_UNIF), add_b=areg(R_UNIF)))
    p.emit(_ops(add_op=ADD_OR, add_out=6, add_a=areg(R_UNIF), add_b=areg(R_UNIF)))
    p.emit(_ops(add_op=ADD_OR, add_out=5, add_a=breg(R_QPU_NUM), add_b=breg(R_QPU_NUM)))
    p.emit(_ops(add_op=ADD_OR, add_out=2, add_a=areg(R_ELEM_NUM), add_b=areg(R_ELEM_NUM)))
    p.emit(_ops(add_op=ADD_ITOF, add_out=R0, add_a=areg(2), add_b=areg(2)))
    p.emit(ldi(0x42C80000, 20, ws=1))                  # 100.0f in rb20
    p.emit(_ops(add_op=ADD_ITOF, add_out=R1, add_a=areg(5), add_b=areg(5)))
    p.emit(_ops(mul_op=MUL_FMUL, add_out=W_NOP, mul_out=R1, mul_a=acc(1), mul_b=breg(20)))
    p.emit(_ops(add_op=ADD_FADD, add_out=R2, add_a=acc(0), add_b=acc(1)))
    p.emit(*store_vector(0, 6))
    p.emit(end_program())
    p.emit(nop(), nop(), nop())
    return p


def prog_alu(nbody=16):
    """ALU-only MAC loop.

    Per iteration: nbody MAC instructions (each 1 fmul + 1 fadd on the two ALUs),
    one counter decrement and one branch => nbody+2 issue slots for 2*nbody flops/lane.

    r0 *= 1.0, r1 += 1.0, r2 *= 1.0, r3 += 1.0 rotating, so every accumulator read
    is at least one instruction after its write (accumulator rule allows distance 1).
    r1 and r3 end as exactly 2*N (short-run oracle).
    """
    assert nbody >= 8
    p = Prog()
    p.emit(_ops(add_op=ADD_OR, add_out=1, add_a=areg(R_UNIF), add_b=areg(R_UNIF)))   # N
    p.emit(_ops(add_op=ADD_OR, add_out=6, add_a=areg(R_UNIF), add_b=areg(R_UNIF)))   # out base
    p.emit(_ops(add_op=ADD_OR, add_out=5, add_a=breg(R_QPU_NUM), add_b=breg(R_QPU_NUM)))
    p.emit(ldi(0x3F800000, 20, ws=1))     # 1.0f in rb20
    p.emit(ldi(0xFFFFFFFF, 21, ws=1))     # -1 in rb21
    for r in range(4):
        p.emit(_ops(add_op=ADD_OR, add_out=32 + r, add_a=imm(0), add_b=imm(0)))

    p.label('loop')
    body = []
    pat = [(0, 1), (2, 3)]
    for i in range(nbody):
        m, a = pat[i % 2]
        body.append(_ops(add_op=ADD_FADD, mul_op=MUL_FMUL, add_out=32 + a, mul_out=32 + m,
                         add_a=acc(a), add_b=breg(20), mul_a=acc(m), mul_b=breg(20)))
    # counter: ra1 = -1 + ra1, sets Z when it hits zero
    for w in body[:nbody - 3]:
        p.emit(w)
    p.emit(_ops(add_op=ADD_ADD, add_out=1, add_a=(REGB, 21), add_b=areg(1), sf=1))
    p.branch_to('loop', cond=BR_ALL_Z_CLEAR)
    # the three MACs after the branch are its delay slots
    for w in body[nbody - 3:]:
        p.emit(w)
    p.emit(*store_vector(1, 6))
    p.emit(end_program())
    p.emit(nop(), nop(), nop())
    return p


if __name__ == '__main__':
    outdir = os.path.dirname(os.path.abspath(__file__))
    names = sys.argv[1:] or ['store', 'alu']
    if 'store' in names:
        w = prog_store_test().write(os.path.join(outdir, 'store.bin'))
        print('store.bin', len(w), 'instructions')
    if 'alu' in names:
        for n in (8, 16, 32):
            w = prog_alu(n).write(os.path.join(outdir, f'alu{n}.bin'))
            print(f'alu{n}.bin', len(w), 'instructions')


def prog_store_hard(patch_idx_out=7):
    """Minimal store test: value = 1.0 in every lane, no uniforms, address patched by qrun."""
    p = Prog()
    p.emit(ldi(0x3F800000, R0))            # r0 = 1.0
    p.emit(ldi(0x1A34, R5, ws=1))          # VPM generic write setup (row 52), QPU 0
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=acc(5)))
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPM, add_a=acc(0), add_b=acc(0)))
    p.emit(ldi(0x80815A00, R5, ws=1))      # DMA write setup
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=acc(5)))
    p.emit(ldi(0xC0010000, W_VPW_SETUP, ws=1))
    assert len(p.w) == patch_idx_out
    p.emit(ldi(0x00000000, R2))            # patched to the output bus address
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPW_ADDR, ws=1, add_a=acc(2), add_b=acc(2)))
    p.emit(_ops(add_op=ADD_OR, add_out=W_NOP, add_a=(REGB, R_VPW_WAIT), add_b=(REGB, R_VPW_WAIT)))
    p.emit(end_program())
    p.emit(nop(), nop(), nop())
    return p


def prog_alu_hard(nbody=8, count=1000, patch_idx_out=None):
    """ALU loop with a hardcoded iteration count and a patched store address (no uniforms)."""
    assert nbody >= 8
    p = Prog()
    p.emit(ldi(count, 1))                 # ra1 = count
    p.emit(ldi(0x3F800000, 20, ws=1))     # 1.0f
    p.emit(ldi(0xFFFFFFFF, 21, ws=1))     # -1
    for r in range(4):
        p.emit(_ops(add_op=ADD_OR, add_out=32 + r, add_a=imm(0), add_b=imm(0)))
    p.label('loop')
    body = []
    pat = [(0, 1), (2, 3)]
    for i in range(nbody):
        m, a = pat[i % 2]
        body.append(_ops(add_op=ADD_FADD, mul_op=MUL_FMUL, add_out=32 + a, mul_out=32 + m,
                         add_a=acc(a), add_b=breg(20), mul_a=acc(m), mul_b=breg(20)))
    for w in body[:nbody - 3]:
        p.emit(w)
    p.emit(_ops(add_op=ADD_ADD, add_out=1, add_a=(REGB, 21), add_b=areg(1), sf=1))
    p.branch_to('loop', cond=BR_ALL_Z_CLEAR)
    for w in body[nbody - 3:]:
        p.emit(w)
    # store r1 (== 2*count) to the patched address
    p.emit(ldi(0x1A34, R5, ws=1))
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=acc(5)))
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPM, add_a=acc(1), add_b=acc(1)))
    p.emit(ldi(0x80815A00, R5, ws=1))
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPW_SETUP, ws=1, add_a=acc(5), add_b=acc(5)))
    p.emit(ldi(0xC0010000, W_VPW_SETUP, ws=1))
    if patch_idx_out is not None:
        assert len(p.w) == patch_idx_out
    else:
        patch_idx_out = len(p.w)
    p.emit(ldi(0x00000000, R2))
    p.emit(_ops(add_op=ADD_OR, add_out=W_VPW_ADDR, ws=1, add_a=acc(2), add_b=acc(2)))
    p.emit(_ops(add_op=ADD_OR, add_out=W_NOP, add_a=(REGB, R_VPW_WAIT), add_b=(REGB, R_VPW_WAIT)))
    p.emit(end_program())
    p.emit(nop(), nop(), nop())
    return p, patch_idx_out


def prog_alu_timed(nbody, count):
    """Pure ALU loop, iteration count compiled in, no memory access at all."""
    p = Prog()
    p.emit(ldi(count, 1))
    p.emit(ldi(0x3F800000, 20, ws=1))
    p.emit(ldi(0xFFFFFFFF, 21, ws=1))
    for r in range(4):
        p.emit(_ops(add_op=ADD_OR, add_out=32 + r, add_a=imm(0), add_b=imm(0)))
    p.label('loop')
    body = []
    pat = [(0, 1), (2, 3)]
    for i in range(nbody):
        m, a = pat[i % 2]
        body.append(_ops(add_op=ADD_FADD, mul_op=MUL_FMUL, add_out=32 + a, mul_out=32 + m,
                         add_a=acc(a), add_b=breg(20), mul_a=acc(m), mul_b=breg(20)))
    for w in body[:nbody - 3]:
        p.emit(w)
    p.emit(_ops(add_op=ADD_ADD, add_out=1, add_a=(REGB, 21), add_b=areg(1), sf=1))
    p.branch_to('loop', cond=BR_ALL_Z_CLEAR)
    for w in body[nbody - 3:]:
        p.emit(w)
    p.emit(end_program())
    p.emit(nop(), nop(), nop())
    return p


def prog_mem(count, with_load=True, patch_idx=1):
    """Streaming TMU-fed MAC loop, software-pipelined two iterations deep.

    Per iteration: one TMU request (64 B, one word per lane), one address bump,
    a pop of the previous request's result attached to a MAC, and six MAC
    instructions.  With with_load=False the same loop runs without any TMU
    traffic, which isolates the cost of the load path.
    """
    p = Prog()
    p.emit(_ops(add_op=ADD_OR, add_out=36, add_a=imm(1), add_b=imm(1)))  # tmu_noswap
    assert len(p.w) == patch_idx
    p.emit(ldi(0x00000000, 8))            # patched: input base bus address
    p.emit(_ops(add_op=ADD_OR, add_out=9, add_a=areg(R_ELEM_NUM), add_b=areg(R_ELEM_NUM)))
    p.emit(_ops(add_op=ADD_SHL, add_out=R0, add_a=areg(9), add_b=imm(2)))
    p.emit(_ops(add_op=ADD_ADD, add_out=8, add_a=areg(8), add_b=acc(0)))
    p.emit(ldi(64, 22, ws=1))             # address bump
    p.emit(ldi(0x3F800000, 35))           # 1.0f in accumulator r3
    p.emit(ldi(0, 32))                    # r0 = 0
    p.emit(ldi(count, 1))
    p.emit(ldi(0xFFFFFFFF, 21, ws=1))
    # prime the TMU pipeline with two requests
    p.emit(_ops(add_op=ADD_OR, add_out=W_TMU0_S, ws=1, add_a=areg(8), add_b=areg(8)))
    p.emit(_ops(add_op=ADD_ADD, add_out=8, add_a=areg(8), add_b=(REGB, 22)))
    p.emit(_ops(add_op=ADD_OR, add_out=W_TMU0_S, ws=1, add_a=areg(8), add_b=areg(8)))
    p.emit(_ops(add_op=ADD_ADD, add_out=8, add_a=areg(8), add_b=(REGB, 22)))

    def mac(add_out=R1, sig=SIG_NONE, use_r4=True, mul_out=R2):
        return _ops(add_op=ADD_FADD, mul_op=MUL_FMUL, add_out=add_out, mul_out=mul_out,
                    add_a=acc(1), add_b=acc(3),
                    mul_a=acc(4 if use_r4 else 2), mul_b=acc(3), sig=sig)

    p.label('loop')
    # request (+ a mul on the live r4)
    if with_load:
        p.emit(_ops(add_op=ADD_OR, mul_op=MUL_FMUL, add_out=W_TMU0_S, mul_out=R2, ws=1,
                    add_a=areg(8), add_b=areg(8), mul_a=acc(4), mul_b=acc(3)))
    else:
        p.emit(_ops(mul_op=MUL_FMUL, add_out=W_NOP, mul_out=R2,
                    mul_a=acc(4), mul_b=acc(3)))
    # bump + pop (+ a mul on the previous r4)
    p.emit(_ops(add_op=ADD_ADD, mul_op=MUL_FMUL, add_out=8, mul_out=R0, ws=0,
                add_a=areg(8), add_b=(REGB, 22), mul_a=acc(4), mul_b=acc(3),
                sig=SIG_LOAD_TMU0 if with_load else SIG_NONE))
    p.emit(mac(use_r4=True))              # uses the freshly popped value
    # counter
    p.emit(_ops(add_op=ADD_ADD, add_out=1, add_a=(REGB, 21), add_b=areg(1), sf=1))
    p.branch_to('loop', cond=BR_ALL_Z_CLEAR)
    # three delay slots, all doing MAC work
    p.emit(mac(), mac(), mac())
    p.emit(end_program())
    p.emit(nop(), nop(), nop())
    return p, patch_idx
