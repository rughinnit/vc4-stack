#!/usr/bin/env python3
"""g10sim - scalar control-flow/address simulator for the asm10 GEMM kernel.

Runs the *dict* form of the generated program (G10Asm.w) on lane 0 and checks
the things a hang or a wild DMA comes from:

  * the program terminates within a step budget (no unbounded loop),
  * every TMU address read (A panel, B matrix) is inside its buffer,
  * every VPM->memory store address is inside the C buffer,
  * the uniform port is read exactly as many times as the launcher writes.

No floating-point semantics are modelled: the only values that drive control
flow or addressing are integers, and lane 0 sees the same flags as every other
lane because every compare uses a lane-invariant value.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
import kgen
from kgen import (ADD_ADD, ADD_SUB, ADD_SHL, ADD_SHR, ADD_AND, ADD_OR,
                  ADD_XOR, ADD_ROR, ADD_NOT, ADD_FADD, ADD_FSUB,
                  ADD_FMIN, ADD_FMAX, ADD_FTOI, ADD_ITOF, ADD_MIN, ADD_MAX,
                  IMM, REGA, REGB, W_TMU0_S, W_TMU1_S, W_VPM, W_VPW_SETUP,
                  W_VPW_ADDR, W_VPR_SETUP, W_VPR_ADDR, W_NOP,
                  BR_ALL_Z_CLEAR, BR_ALWAYS, BR_ALL_N_SET)

M32 = 0xFFFFFFFF


def sx(v):
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


class Sim:
    def __init__(self, proga, unis, base_a, size_a, base_b, size_b,
                 base_c, size_c, lane=0, qpu=0):
        self.a = proga
        self.unis = list(unis)
        self.ui = 0
        self.uhit = False
        self.uval = 0
        self.A = [0] * 32
        self.B = [0] * 32
        self.acc = [0] * 6
        self.lane, self.qpu = lane, qpu
        self.pc = 0
        self.z = self.n = 0
        self.limits = dict(a=(base_a, base_a + size_a), b=(base_b, base_b + size_b),
                           c=(base_c, base_c + size_c))
        self.reads = []
        self.writes = []
        self.steps = 0

    def unif(self):
        if not self.uhit:
            self.uval = self.unis[self.ui] if self.ui < len(self.unis) else 0
            self.ui += 1
            self.uhit = True
        return self.uval

    def rd(self, mux, addr):
        if mux == IMM:
            return addr
        if mux == REGA:
            if addr == 32:
                return self.unif()
            if addr == 38:
                return self.lane
            if addr < 32:
                return self.A[addr]
            return 0
        if mux == REGB:
            if addr == 32:
                return self.unif()
            if addr == 38:
                return self.qpu
            if addr < 32:
                return self.B[addr]
            return 0
        if mux < 6:
            return self.acc[mux]
        raise AssertionError("bad mux %r/%r in word %d (%s)" %
                             (mux, addr, self.pc // 4,
                              self.a.w[self.pc // 4].get('note', '')))

    def wr(self, out, ws, val):
        if out == W_TMU0_S or out == W_TMU1_S:
            self.reads.append((val & M32, self.pc // 4,
                               self.a.w[self.pc // 4].get('note', '')))
            return
        if out in (W_VPW_ADDR,):
            self.writes.append(val & M32)
            return
        if out == W_VPR_ADDR:
            self.reads.append((val & M32, self.pc // 4,
                               self.a.w[self.pc // 4].get('note', '')))
            return
        if out >= 38:
            return
        if out >= 32:
            self.acc[out - 32] = val & M32
        elif ws == 0:
            self.A[out] = val & M32
        else:
            self.B[out] = val & M32

    def alu(self, op, a, b):
        if op in (ADD_FADD, ADD_FSUB, ADD_FMIN, ADD_FMAX, ADD_FTOI,
                  ADD_ITOF):
            return 0                      # FP results never drive addressing
        a &= M32
        b &= M32
        if op == ADD_ADD:
            return (a + b) & M32
        if op == ADD_SUB:
            return (a - b) & M32
        if op == ADD_SHL:
            return (a << (b & 31)) & M32
        if op == ADD_SHR:
            return (a >> (b & 31)) & M32
        if op == ADD_AND:
            return a & b
        if op == ADD_OR:
            return a | b
        if op == ADD_XOR:
            return a ^ b
        return 0

    def run(self, maxsteps=400000):
        w = self.a.w
        labels = self.a.labels
        while True:
            self.steps += 1
            if self.steps > maxsteps:
                raise AssertionError("step budget exhausted (no termination)")
            d = w[self.pc // 4]
            self.uhit = False
            if 'branch' in d:
                tgt = labels[d['target']]
                cond = d['branch']
                take = (cond == BR_ALWAYS or
                        (cond == BR_ALL_Z_CLEAR and not self.z) or
                        (cond == BR_ALL_N_SET and self.n))
                self.pc = tgt * 4 if take else self.pc + 4
                continue
            if 'ldi' in d:
                v, o, ws = d['ldi']
                if o < 32 or 32 <= o <= 37:
                    self.wr(o, ws, v)
                self.pc += 4
                continue
            add = d.get('add')
            mul = d.get('mul')
            av = bv = 0
            if add:
                av = self.rd(add['add_a'][0], add['add_a'][1])
                bv = self.rd(add['add_b'][0], add['add_b'][1])
                res = self.alu(add['add_op'], av, bv)
                if add.get('sf'):
                    self.z = (res & M32) == 0
                    self.n = bool(res & 0x80000000)
                self.wr(add['add_out'], add.get('ws', d['ws']), res)
            if mul:
                mv = self.rd(mul['mul_a'][0], mul['mul_a'][1])
                nv = self.rd(mul['mul_b'][0], mul['mul_b'][1])
                # only the address-free v8min/fmul paths write accumulators
                self.wr(mul['mul_out'], 1 - d['ws'], 0)
            if d.get('sig') == 3:         # SIG_END_PROGRAM
                break
            self.pc += 4
        return self

    def check(self, tag):
        (alo, ahi), (blo, bhi), (clo, chi) = (self.limits['a'],
                                              self.limits['b'],
                                              self.limits['c'])
        bad = 0
        shown = 0
        for (r, pc, note) in self.reads:
            if not (alo <= r < ahi) and not (blo <= r < bhi):
                if shown < 4:
                    print("  %s: BAD TMU read 0x%08x at word %d (%s)"
                          % (tag, r, pc, note))
                    shown += 1
                bad += 1
            if not (r + 64 <= ahi) and not (r + 64 <= bhi):
                if (alo <= r < ahi) or (blo <= r < bhi):
                    if shown < 4:
                        print("  %s: TMU read 0x%08x runs past its buffer at "
                              "word %d (%s)" % (tag, r, pc, note))
                        shown += 1
                    bad += 1
        for r in self.writes:
            if not (clo <= r < chi):
                if shown < 6:
                    print("  %s: BAD store address 0x%08x (C %08x..%08x)"
                          % (tag, r, clo, chi))
                    shown += 1
                bad += 1
            elif not (r + 64 <= chi):
                if shown < 6:
                    print("  %s: store 0x%08x runs past C" % (tag, r))
                    shown += 1
                bad += 1
        return bad


def run_shape(M, N, K, nqpu, verbose=False):
    a = kgen.build_gemm10(M, N, K, nqpu)
    NCB, TR = N // 64, M // 8
    nq = min(nqpu, TR)
    A_BASE, B_BASE, C_BASE = 0x00100000, 0x00400000, 0x00800000
    pa, pb, pc = M * K * 4, K * N * 4, M * N * 4
    bad = 0
    for q in range(nq):
        u = [A_BASE, C_BASE, B_BASE, q * 32 * K, nq * 32 * K,
             q * 8 * N * 4, nq * 8 * N * 4, q]
        s = Sim(a, u, A_BASE, pa, B_BASE, pb, C_BASE, pc, lane=0, qpu=q)
        s.run()
        bad += s.check("M=%d N=%d K=%d q=%d" % (M, N, K, q))
        if s.ui != 8:
            print("  M=%d N=%d K=%d q=%d: read %d uniforms, expected 8"
                  % (M, N, K, q, s.ui))
            bad += 1
        if verbose:
            print("  q=%d steps=%d reads=%d writes=%d"
                  % (q, s.steps, len(s.reads), len(s.writes)))
    ntiles = (TR + nq - 1) // nq * nq * NCB // nq if nq else 0
    print("gemm sim: M=%d N=%d K=%d nq=%d -> %d steps/QPU, %s"
          % (M, N, K, nq, s.steps, "PASS" if bad == 0 else "FAIL"))
    return bad == 0


if __name__ == "__main__":
    ok = True
    for (M, N, K, q) in ((16, 64, 16, 2), (24, 64, 48, 3), (8, 64, 16, 1),
                         (128, 128, 128, 12), (256, 256, 256, 12),
                         (512, 64, 288, 12), (16, 128, 48, 2),
                         (64, 64, 32, 4), (40, 192, 80, 5)):
        ok &= run_shape(M, N, K, q)
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
